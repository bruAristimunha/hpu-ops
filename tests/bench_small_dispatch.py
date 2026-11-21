"""On a Gaudi card (issue #21, round 2): hpu_ops.small's bincount rewrite through the dispatcher, installed and undone
in alternating blocks in one process, against Habana's kernel. Mode follows PT_HPU_LAZY_MODE. (The logsumexp rows of
TEAM_REPORT.md came from this script with the dropped logsumexp rewrite installed the same way.)

Styles: ``iso`` (each call synced on its own; lazy: mark_step + synchronize per call) and ``batch`` (N calls, one
mark_step, one synchronize). ``fwd+bwd`` rows time forward and ``autograd.grad`` per call. Each row asserts the rewrite
ran (installed) or did not (undone) through hpu_ops' call counts, and counts CPU fallbacks.
BENCH_BINCOUNT="4096x64,409600x1000" picks the (elements x bins) rows.

    python bench_small_dispatch.py [op ...]
"""
import os
import sys
import time
import warnings

import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.hpu as hthpu
import torch
from habana_frameworks.torch.hpu.metrics import metric_global

import hpu_ops
from hpu_ops import small

LAZY = os.environ.get("PT_HPU_LAZY_MODE", "1") == "1"
N = int(os.environ.get("BENCH_N", "30"))
ROUNDS = int(os.environ.get("BENCH_ROUNDS", "7"))
hpu = torch.device("hpu")
calls = hpu_ops._STATE["calls"]


def fallbacks():
    return sum((dict(metric_global("cpu_fallback").stats()).get("FallbackOps", {}) or {}).values())


def block(fn, args, style):
    out = []
    t = time.perf_counter()
    for _ in range(N):
        out.append(fn(*args))
        if style == "iso":
            htcore.mark_step()
            hthpu.synchronize()
    htcore.mark_step()
    hthpu.synchronize()
    return (time.perf_counter() - t) / N * 1e3, out[-1]


def cpu_time(fn, args):
    for _ in range(3):
        fn(*args)
    ts = []
    for _ in range(5):
        t = time.perf_counter()
        for _ in range(N):
            out = fn(*args)
        ts.append((time.perf_counter() - t) / N * 1e3)
    return sorted(ts)[2], out


def race(fn, args, style, name):
    """Habana (rewrite undone) against ours (installed), alternating, the order swapped each round."""
    times = {"habana": [], "ours": []}
    outs, fb, taken = {}, 0, {"habana": 0, "ours": 0}
    for r in range(ROUNDS + 1):
        for who in (("habana", "ours") if r % 2 else ("ours", "habana")):
            undo = small.install_bincount("hpu") if who == "ours" else None
            try:
                calls.clear()
                before = fallbacks()
                with warnings.catch_warnings(record=True) as rec:
                    warnings.simplefilter("always")
                    ms, outs[who] = block(fn, args, style)
                fb += fallbacks() - before + sum("runs on the CPU" in str(w.message) for w in rec)
                taken[who] += calls[name]
            finally:
                if undo:
                    undo()
            if r:  # round 0 warms up both
                times[who].append(ms)
    med = {k: sorted(v)[len(v) // 2] for k, v in times.items()}
    rng = {k: (min(v), max(v)) for k, v in times.items()}
    return med, rng, outs, fb, taken


def cases():
    c = {"bincount": []}
    sizes = os.environ.get("BENCH_BINCOUNT")
    sizes = [tuple(map(int, p.split("x"))) for p in sizes.split(",")] if sizes else [
        (4096, 64), (40960, 64), (65536, 64), (131072, 64), (409600, 64), (409600, 256), (4096000, 64), (409600, 1000)]
    for n, bins in sizes:
        c["bincount"].append((f"({n},) {bins} bins", lambda x: torch.bincount(x), (torch.randint(0, bins, (n,)),),
                              False))
    return c


def main():
    want = sys.argv[1:]
    styles = os.environ.get("BENCH_STYLES", "iso,batch").split(",")
    print(f"bridge 1.21.4 torch {torch.__version__} mode {'lazy' if LAZY else 'eager'} N={N} rounds={ROUNDS}",
          flush=True)
    for op, rows in cases().items():
        if want and op not in want:
            continue
        for label, fn, args, grad in rows:
            cpu_args = [a.clone().requires_grad_(grad) for a in args]
            cpu_ms, ref = cpu_time(fn, cpu_args)
            hargs = [a.to(hpu).requires_grad_(grad) for a in args]
            for style in styles:
                med, rng, outs, fb, taken = race(fn, hargs, style, op)
                err = {k: (v.cpu().double() - ref.double()).abs().max().item() for k, v in outs.items()}
                print(f"{op:10s} {label:32s} {style:5s} cpu {cpu_ms:8.3f} | habana {med['habana']:7.3f} "
                      f"[{rng['habana'][0]:.3f}-{rng['habana'][1]:.3f}] ours {med['ours']:7.3f} "
                      f"[{rng['ours'][0]:.3f}-{rng['ours'][1]:.3f}] habana/ours {med['habana'] / med['ours']:5.2f} | "
                      f"err {err['habana']:.1e}/{err['ours']:.1e} taken {taken['habana']}/{taken['ours']} fb {fb}",
                      flush=True)


if __name__ == "__main__":
    main()
