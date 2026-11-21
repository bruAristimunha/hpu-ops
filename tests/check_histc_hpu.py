"""Gaudi check of hpu-ops' histc route (small.install_histc) against ATen's CPU kernel on the values the card holds,
plus its per-call cost. PT_HPU_LAZY_MODE=0|1 python tests/check_histc_hpu.py [check|compile|perf]"""
import math
import os
import sys
import time

import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.hpu as hthpu
import torch

import hpu_ops
from hpu_ops import small

LAZY = os.environ.get("PT_HPU_LAZY_MODE", "1") != "0"
hpu = torch.device("hpu")


def sync():
    if LAZY:
        htcore.mark_step()
    hthpu.synchronize()


def fallbacks():
    from habana_frameworks.torch.hpu.metrics import metric_global

    return sum((dict(metric_global("cpu_fallback").stats()).get("FallbackOps", {}) or {}).values())


def data(n, lo, hi, dtype, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(n, generator=g, dtype=torch.float64) * (hi - lo) * 1.2 + lo - 0.1 * (hi - lo)
    x = x.to(dtype)
    if dtype != torch.float64 and bins_ok(lo, hi):
        edges = small.histc_edges(float(lo), float(hi), BINS[0], dtype)[0][1:-1]
        x = torch.cat([x, edges, torch.nextafter(edges.float(), torch.tensor(math.inf)).to(dtype)])
    return torch.cat([x, torch.tensor([math.nan, math.inf, -math.inf, lo, hi], dtype=dtype)])


BINS = [64]


def bins_ok(lo, hi):
    return lo < hi


CASES = [  # (bins, lo, hi, dtype, shape-view)
    (64, -3.0, 3.0, torch.float32, None), (64, 0.0, 63.0, torch.float32, None), (64, -0.5, 63.5, torch.float32, None),
    (1000, 0.1, 0.7, torch.float32, None), (4096, -3.0, 3.0, torch.float32, None), (301, 0, 1, torch.float32, None),
    (400, 0, 63, torch.float32, None), (7, -1e-3, 2e-3, torch.float32, None), (1, 0.0, 1.0, torch.float32, None),
    (2, -0.37, 2.9, torch.float32, None), (64, -3.0, 3.0, torch.bfloat16, None), (64, -3.0, 3.0, torch.float16, None),
    (500, -3.0, 3.0, torch.bfloat16, None), (64, -3.0, 3.0, torch.float64, None), (64, -3.0, 3.0, torch.float32, "2d"),
    (64, -3.0, 3.0, torch.float32, "slice"), (64, -3.0, 3.0, torch.float32, "strided"),
]


def reference(held, bins, lo, hi):
    """ATen's CPU histc with one thread on the values the card holds. Where ATen's 16-bit position passes the last bin
    its kernel writes outside the histogram (undefined behaviour): it is not called; the reference is then its
    positions with those in the last bin, capped as its one-thread counts are."""
    if held.dtype in (torch.bfloat16, torch.float16) and lo < hi:
        L, R = torch.tensor(lo, dtype=held.dtype), torch.tensor(hi, dtype=held.dtype)
        inside = held[(held >= L) & (held <= R)]
        pos = ((inside - L) * torch.tensor(bins, dtype=held.dtype) / (R - L)).trunc().long()
        if bool((pos > bins).any()):
            return small._cap(torch.bincount(pos.clamp(max=bins - 1), minlength=bins).float(), held.dtype)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        return torch.histc(held, bins, lo, hi)
    finally:
        torch.set_num_threads(threads)


def run(fn, compiled):
    if not compiled:
        return fn
    import habana_frameworks.torch.dynamo  # noqa: F401

    torch._dynamo.reset()  # one graph per case: no cache limit, no silent eager fallback
    return torch.compile(fn, backend="hpu_backend", dynamic=False, fullgraph=True)


def check(compiled=False):
    hpu_ops.enable()
    mode = "compile" if compiled else ("lazy" if LAZY else "eager")
    bad = 0
    for i, (bins, lo, hi, dtype, view) in enumerate(CASES):
        BINS[0] = bins
        x = data(3000, lo, hi, dtype, i)
        h = x.to(hpu)
        if view == "2d":
            h = h[: (h.numel() // 4) * 4].view(4, -1)
        elif view == "slice":
            h = h.view(-1)[5:]
        elif view == "strided":
            h = h[::3]
        before = fallbacks()
        hpu_ops._STATE["calls"].clear()
        try:
            got = run(lambda t: torch.histc(t, bins, lo, hi), compiled)(h)
            sync()
            held = h.cpu()  # what the card holds (float64: float32 values; .cpu() reads offset views right)
            ref = reference(held, bins, lo, hi)
            exact = got.dtype == ref.dtype and torch.equal(got.cpu(), ref)
            bad += not exact
            print(f"{'ok  ' if exact else 'BAD '} {mode} bins={bins} [{lo}, {hi}] {dtype} {view or ''} "
                  f"route={hpu_ops._STATE['calls']['histc']} fallbacks={fallbacks() - before}"
                  + ("" if exact else f" differ {int((got.cpu() != ref).sum())}"), flush=True)
        except Exception as e:  # noqa: BLE001
            bad += 1
            print(f"RAISE {mode} bins={bins} [{lo}, {hi}] {dtype} {view}: {str(e).splitlines()[0][:150]}", flush=True)
    # equal float bounds: the data's range. Eager / lazy: ATen's bins and errors. Compiled: read on the card, the
    # card's arithmetic (README: not ATen's at the edges; NaN / inf give counts where ATen raises).
    for x, lo in ((torch.rand(1000) * 5, 0.0), (torch.full((10,), 2.5), 2.5), (torch.tensor([1.0, math.nan]), 0.0),
                  (torch.tensor([0.0, math.inf]), 0.0)):
        try:
            ref = reference(x, 10, lo, lo)
        except RuntimeError as e:
            ref = e
        try:
            got = run(lambda t: torch.histc(t, 10, lo, lo), compiled)(x.to(hpu))
            sync()
            ok = isinstance(ref, torch.Tensor) and torch.equal(got.cpu(), ref)
            note = "" if ok or not compiled or isinstance(ref, torch.Tensor) else " (compiled: counts, documented)"
            ok = ok or bool(note)
        except RuntimeError as e:
            ok = isinstance(ref, RuntimeError) and str(ref).replace("-nan", "nan") == str(e)
            note = ""
        bad += not ok
        print(f"{'ok  ' if ok else 'BAD '} {mode} equal bounds {lo} on {x[:3].tolist()}{note}", flush=True)
    if not compiled:  # eager float64 views with an offset (read at twice the offset by Gaudi's eager kernels)
        base = torch.tensor([0.0, 1.0, 2.0, 3.0, 2.5, 0.5], dtype=torch.float64)
        hb = base.to(hpu)
        views = (("h[1:2]", lambda t: t[1:2]), ("h[2:]", lambda t: t[2:]), ("row 1", lambda t: t.view(2, 3)[1]))
        for name, view in views:
            got = torch.histc(view(hb), 4, 0.0, 4.0)
            sync()
            ref = reference(view(base), 4, 0.0, 4.0)
            ok = torch.equal(got.cpu(), ref)
            bad += not ok
            print(f"{'ok  ' if ok else 'BAD '} {mode} float64 offset view {name}: {got.cpu().tolist()} "
                  f"(ATen {ref.tolist()})", flush=True)
        x = torch.rand(2000)
        ref = reference(x, 400, 0.0, 1.0)
        with torch.inference_mode():
            got = torch.histc(x.to(hpu), 400, 0.0, 1.0)
        out = torch.empty(0, device=hpu)
        torch.histc(x.to(hpu), 400, 0.0, 1.0, out=out)
        sync()
        for name, g in (("inference_mode", got), ("out=", out)):
            ok = torch.equal(g.cpu(), ref)
            bad += not ok
            print(f"{'ok  ' if ok else 'BAD '} {mode} {name} 400 bins", flush=True)
    bad += review_inputs(mode, compiled)
    print("FAILURES", bad, flush=True)
    return bad


def review_inputs(mode, compiled):
    """The Codex review's inputs (PR #40), against ATen's one-thread CPU kernel or its error."""
    bad = 0
    cases = [("tiny range", torch.tensor([2e-38]), (400, 0.0, 1e-37)),
             ("float64 2**27 equal bounds", torch.tensor([2.0 ** 27], dtype=torch.float64), (400, 0.0, 0.0)),
             ("bf16 unreachable threshold", torch.tensor([1.0078125], dtype=torch.bfloat16), (514, 1.0, 1.0078125)),
             ("bf16 4096 x 0.25", torch.full((4096,), 0.25, dtype=torch.bfloat16), (4, 0.0, 1.0)),
             ("fp16 4096 x 0.25", torch.full((4096,), 0.25, dtype=torch.float16), (4, 0.0, 1.0)),
             ("fp16 65504 equal bounds", torch.tensor([65504.0], dtype=torch.float16), (4, 0.0, 0.0)),
             ("empty, 1e300 bound", torch.empty(0), (4, 0.0, 1e300))]
    for name, x, args in cases:
        if compiled and args[1] == args[2]:
            continue  # equal bounds under compile: the card's arithmetic (documented)
        try:
            ref = reference(x, *args)
        except RuntimeError as e:
            ref = e
        try:
            got = run(lambda t: torch.histc(t, *args), compiled)(x.to(hpu))
            sync()
            ok = isinstance(ref, torch.Tensor) and torch.equal(got.cpu(), ref)
        except Exception as e:  # noqa: BLE001
            ok = isinstance(ref, RuntimeError) and "overflow" in str(e) and "overflow" in str(ref)
        bad += not ok
        print(f"{'ok  ' if ok else 'BAD '} {mode} review input: {name}", flush=True)
    return bad


def perf():
    hpu_ops.enable()
    x = (torch.randn(32, 22, 1000)).to(hpu)
    sync()
    cases = [("native ints 64 bins [-3, 3]", lambda: torch.histc(x, 64, -3, 3)),
             ("route floats 64 bins [-3.0, 3.0]", lambda: torch.histc(x, 64, -3.0, 3.0)),
             ("route floats 256 bins", lambda: torch.histc(x, 256, -3.0, 3.0)),
             ("route 1000 bins", lambda: torch.histc(x, 1000, -3.0, 3.0)),
             ("route 4096 bins", lambda: torch.histc(x, 4096, -3.0, 3.0)),
             ("route equal float bounds (host read)", lambda: torch.histc(x, 64, 0.0, 0.0)),
             ("native default bounds 64 bins", lambda: torch.histc(x, 64))]
    for name, fn in cases:
        for _ in range(3):
            out = fn()
        sync()
        iso = []
        for _ in range(20):
            t0 = time.perf_counter()
            out = fn()
            sync()
            iso.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        outs = [fn() for _ in range(30)]
        sync()
        batch = (time.perf_counter() - t0) / 30
        del outs, out
        iso.sort()
        print(f"perf {'lazy' if LAZY else 'eager'} {name}: isolated median {iso[10] * 1e3:.3f} ms, batch of 30 "
              f"{batch * 1e3:.3f} ms/call", flush=True)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "check"
    result = {"check": check, "compile": lambda: check(compiled=True), "perf": perf}[what]()
    sys.exit(1 if result else 0)
