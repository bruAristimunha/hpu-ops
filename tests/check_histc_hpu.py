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


def check(compiled=False):
    hpu_ops.enable()
    bad = 0
    f = lambda t, b, lo, hi: torch.histc(t, b, lo, hi)  # noqa: E731
    if compiled:
        import habana_frameworks.torch.dynamo  # noqa: F401
        f = torch.compile(f, backend="hpu_backend", dynamic=False)
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
            got = f(h, bins, lo, hi)
            sync()
            held = h.cpu()  # the values the card holds (float64: float32 values)
            ref = torch.histc(held, bins, lo, hi)
            exact = torch.equal(got.cpu(), ref)
            if dtype in (torch.bfloat16, torch.float16):  # ATen counts in the 16-bit dtype; the route rounds once
                exact = exact or torch.equal(got.cpu(), torch.histc(held.float(), bins, lo, hi).to(dtype)) or \
                    torch.equal(got.cpu(), _exact16(held, bins, lo, hi))
            tag = "ok  " if exact else "BAD "
            bad += not exact
            print(f"{tag} {'compile' if compiled else ('lazy' if LAZY else 'eager')} bins={bins} [{lo}, {hi}] {dtype} "
                  f"{view or ''} route={hpu_ops._STATE['calls']['histc']} fallbacks={fallbacks() - before}"
                  + ("" if exact else f" differ {int((got.cpu() != ref).sum())}"), flush=True)
        except Exception as e:  # noqa: BLE001
            bad += 1
            print(f"RAISE bins={bins} [{lo}, {hi}] {dtype} {view}: {str(e).splitlines()[0][:150]}", flush=True)
    # equal float bounds: the data's range, ATen's errors
    for x, lo in ((torch.rand(1000) * 5, 0.0), (torch.full((10,), 2.5), 2.5), (torch.tensor([1.0, math.nan]), 0.0),
                  (torch.tensor([0.0, math.inf]), 0.0)):
        try:
            ref = torch.histc(x, 10, lo, lo)
        except RuntimeError as e:
            ref = e
        try:
            got = torch.histc(x.to(hpu), 10, lo, lo)
            sync()
            ok = isinstance(ref, torch.Tensor) and torch.equal(got.cpu(), ref)
        except RuntimeError as e:
            ok = isinstance(ref, RuntimeError) and str(ref).replace("-nan", "nan") == str(e)
        bad += not ok
        print(f"{'ok  ' if ok else 'BAD '} equal bounds {lo} on {x[:3].tolist()}", flush=True)
    print("FAILURES", bad, flush=True)


def _exact16(held, bins, lo, hi):
    """Exact counts, rounded once to the 16-bit dtype, of ATen's CPU bins (positions in the 16-bit dtype)."""
    edges, low, high = small.histc_edges(float(lo), float(hi), bins, held.dtype)
    m = (held >= low) & (held <= high)
    p = torch.bucketize(held, edges[1:-1], right=True)
    return torch.bincount(p[m], minlength=bins).to(held.dtype)


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
    {"check": check, "compile": lambda: check(compiled=True), "perf": perf}[what]()
