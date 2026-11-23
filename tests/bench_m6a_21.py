"""M6A (#21) interleaved Gaudi2 before/after benchmark across Eager, Lazy, and torch.compile(backend='hpu_backend').

Run in Eager (`PT_HPU_LAZY_MODE=0`), Lazy (`PT_HPU_LAZY_MODE=1`), or Compile (`BENCH_COMPILE=1 PT_HPU_LAZY_MODE=0`):
    python3 tests/bench_m6a_21.py
BENCH_OPS=index_add (comma separated) keeps those ops' rows.
"""
import os
import time
import warnings

import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.hpu as hthpu
import torch
from habana_frameworks.torch.hpu.metrics import metric_global

import hpu_ops
from hpu_ops import small

LAZY = os.environ.get("PT_HPU_LAZY_MODE", "0") == "1"
COMPILE = os.environ.get("BENCH_COMPILE", "0") == "1"
N = int(os.environ.get("BENCH_N", "20"))
ROUNDS = int(os.environ.get("BENCH_ROUNDS", "5"))
HPU = torch.device("hpu")
CALLS = hpu_ops._STATE["calls"]


def fallbacks():
    return sum((dict(metric_global("cpu_fallback").stats()).get("FallbackOps", {}) or {}).values())


def cpu_time(fn, args):
    torch.set_num_threads(1)
    for _ in range(3):
        fn(*args)
    ts = []
    for _ in range(5):
        t = time.perf_counter()
        for _ in range(N):
            out = fn(*args)
        ts.append((time.perf_counter() - t) / N * 1e3)
    return sorted(ts)[2], out


def block(fn, args, style):
    outs = []
    t = time.perf_counter()
    for _ in range(N):
        outs.append(fn(*args))
        if style == "iso":
            if LAZY:
                htcore.mark_step()
            hthpu.synchronize()
    if LAZY:
        htcore.mark_step()
    hthpu.synchronize()
    return (time.perf_counter() - t) / N * 1e3, outs[-1]


def install_small(device_type):
    return hpu_ops._dispatch.chain([
        small.install_bincount(device_type),
        small.install_histc(device_type),
        small.install_logsumexp(device_type),
        small.install_index_add(device_type),
    ])


def race(fn, args, style, call_key, compile_mode=False):
    times = {"habana": [], "ours": []}
    outs, fb, taken = {}, {"habana": 0, "ours": 0}, {"habana": 0, "ours": 0}
    for r in range(ROUNDS + 1):
        for who in (("habana", "ours") if r % 2 else ("ours", "habana")):
            undo = install_small("hpu") if who == "ours" else None
            try:
                CALLS.clear()
                before = fallbacks()
                run_fn = fn
                if compile_mode:
                    torch._dynamo.reset()
                    run_fn = torch.compile(fn, backend="hpu_backend")
                    for _ in range(2):
                        run_fn(*args)
                    hthpu.synchronize()
                    CALLS.clear()
                    before = fallbacks()
                with warnings.catch_warnings(record=True) as rec:
                    warnings.simplefilter("always")
                    ms, outs[who] = block(run_fn, args, style)
                fb[who] += fallbacks() - before + sum("runs on the CPU" in str(w.message) for w in rec)
                taken[who] += CALLS[call_key]
            finally:
                if undo:
                    undo()
            if r:
                times[who].append(ms)
    med = {k: sorted(v)[len(v) // 2] for k, v in times.items()}
    return med, outs, fb, taken


def bench_native_only(label, fn, args, style):
    cpu_ms, ref = cpu_time(fn, [a.clone() for a in args])
    hargs = [a.to(HPU) for a in args]
    run_fn = fn
    if COMPILE:
        torch._dynamo.reset()
        run_fn = torch.compile(fn, backend="hpu_backend")
    for _ in range(3):
        run_fn(*hargs)
        if LAZY:
            htcore.mark_step()
    hthpu.synchronize()
    fb0 = fallbacks()
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        hpu_ms, out = block(run_fn, hargs, style)
    fb = fallbacks() - fb0 + sum("runs on the CPU" in str(w.message) for w in rec)
    err = (out.cpu().double() - ref.double()).abs().max().item()
    print(f"native   {label:38s} {style:5s} cpu {cpu_ms:7.3f} | hpu {hpu_ms:7.3f} | hpu/cpu {hpu_ms/cpu_ms:6.2f}x | err {err:.1e} fb {fb}", flush=True)


def run_ab():
    mode = "compile" if COMPILE else ("lazy" if LAZY else "eager")
    print(f"=== M6A #21 A/B benchmark ({mode}, N={N}, rounds={ROUNDS}) ===", flush=True)
    ab_cases = []
    if COMPILE:
        for n, ml in [(4096, 256), (40960, 256), (40960, 1000), (409600, 1000)]:
            ab_cases.append(("bincount", f"uint8 ({n},) ml={ml}", lambda x, m=ml: torch.bincount(x, minlength=m),
                             (torch.randint(0, 256, (n,), dtype=torch.uint8),), "iso"))
    else:
        for n, bins, ml in [(4096, 64, 64), (4096, 256, 256), (40960, 128, 128), (40960, 256, 256),
                            (40960, 512, 512), (409600, 1000, 1000)]:
            for st in ("iso", "batch"):
                ab_cases.append(("bincount", f"int64 ({n},) {bins}b ml={ml}",
                                 lambda x, m=ml: torch.bincount(x, minlength=m),
                                 (torch.randint(0, bins, (n,)),), st))
        if LAZY:
            for shape, dim in [((4096,), 0), ((409600,), 0), ((64, 22, 22), -1), ((640, 22, 22), -1), ((6400, 22, 22), -1)]:
                for st in ("iso", "batch"):
                    ab_cases.append(("logsumexp", f"fwd {shape}", lambda x, d=dim: torch.logsumexp(x, d),
                                     (torch.randn(*shape),), st))
                    ab_cases.append(("logsumexp", f"fwdbwd {shape}",
                                     lambda x, d=dim: torch.autograd.grad(torch.logsumexp(x, d).sum(), x)[0],
                                     (torch.randn(*shape, requires_grad=True),), st))
            # taken (_index_add_wins), then declined: 640 source rows per narrow row, and past the 256-row cap
            for rows, n, cols in [(16, 40960, 22), (64, 81920, 22), (128, 163840, 22), (64, 409600, 22),
                                  (256, 409600, 22), (256, 40960, 128), (64, 40960, 22), (1024, 40960, 22),
                                  (1024, 40960, 128)]:
                for st in ("iso", "batch"):
                    ab_cases.append(("index_add", f"({rows},{cols})<-({n},{cols})",
                                     lambda d, i, s: d.index_add(0, i, s),
                                     (torch.randn(rows, cols), torch.randint(0, rows, (n,)), torch.randn(n, cols)), st))

    if os.environ.get("BENCH_OPS"):
        ab_cases = [c for c in ab_cases if c[0] in os.environ["BENCH_OPS"].split(",")]
    for op, label, fn, args, style in ab_cases:
        if "fwdbwd" in label:
            cpu_ms, ref = cpu_time(fn, [a.detach().clone().requires_grad_() for a in args])
            hargs = [a.detach().to(HPU).requires_grad_() for a in args]
        else:
            cpu_ms, ref = cpu_time(fn, [a.clone() for a in args])
            hargs = [a.to(HPU) for a in args]
        with (torch.no_grad() if op == "index_add" else torch.enable_grad()):
            med, outs, fb, taken = race(fn, hargs, style, op, compile_mode=COMPILE)
        err = {k: (v.cpu().double() - ref.double()).abs().max().item() for k, v in outs.items()}
        print(f"{op:10s} {label:28s} {style:5s} cpu {cpu_ms:7.3f} | habana {med['habana']:7.3f} "
              f"ours {med['ours']:7.3f} | speedup {med['habana']/med['ours']:5.2f}x | "
              f"err {err['habana']:.1e}/{err['ours']:.1e} taken {taken['habana']}/{taken['ours']} "
              f"fb {fb['habana']}/{fb['ours']}", flush=True)


def run_non_winners():
    styles = ("iso",) if COMPILE else ("iso", "batch")
    print(f"=== M6A #21 Native-Kept Ops ({'compile' if COMPILE else ('lazy' if LAZY else 'eager')}) ===", flush=True)
    hpu_ops.enable()
    cases = [
        ("trace (1024,1024)", lambda x: torch.trace(x), (torch.randn(1024, 1024),)),
        ("scatter_add (64,22,22) d-1", lambda d, i, s: d.scatter_add(-1, i, s),
         (torch.zeros(64, 22, 22), torch.randint(0, 22, (64, 22, 22)), torch.randn(64, 22, 22))),
        ("where (640,22,22)", lambda c, x, y: torch.where(c, x, y),
         (torch.randn(640, 22, 22) > 0, torch.randn(640, 22, 22), torch.randn(640, 22, 22))),
        ("lerp (640,22,22)", lambda a, b: torch.lerp(a, b, 0.5),
         (torch.randn(640, 22, 22), torch.randn(640, 22, 22))),
        ("cumsum (64,22,22) d-1", lambda x: torch.cumsum(x, -1), (torch.randn(64, 22, 22),)),
        ("cumprod (64,22,22) d-1", lambda x: torch.cumprod(x, -1), (torch.randn(64, 22, 22),)),
        ("matmul (640,22,22)", lambda a, b: a @ b, (torch.randn(640, 22, 22), torch.randn(640, 22, 22))),
        ("topk (640,22,22) k=5", lambda x: torch.topk(x, 5, dim=-1).values, (torch.randn(640, 22, 22),)),
        ("median (640,22,22) d-1", lambda x: torch.median(x, dim=-1).values, (torch.randn(640, 22, 22),)),
    ]
    for label, fn, args in cases:
        for st in styles:
            bench_native_only(label, fn, args, st)


def main():
    run_ab()
    if os.environ.get("BENCH_ALL", "1") == "1":
        run_non_winners()


if __name__ == "__main__":
    main()
