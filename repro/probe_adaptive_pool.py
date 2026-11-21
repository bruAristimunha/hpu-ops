"""Gaudi2: adaptive and fractional max pooling with NaN (values and indices) against torch on the CPU.
    PT_HPU_LAZY_MODE=0|1 python3 repro/probe_adaptive_pool.py"""
import os

import habana_frameworks.torch.core as htcore
import torch
import torch.nn.functional as F

LAZY = os.environ.get("PT_HPU_LAZY_MODE", "1") != "0"
NAN, INF = float("nan"), float("inf")
torch.manual_seed(0)
x1 = torch.randn(2, 3, 12)
x1[0, 0, 1], x1[0, 1, 5], x1[1, 2, -1], x1[1, 0, 3], x1[1, 0, 4] = NAN, NAN, NAN, INF, NAN
x2 = torch.randn(2, 3, 8, 8)
x2[0, 0, 1, 1], x2[0, 1, 7, 7], x2[1, 2, 3, 4], x2[1, 0, 0, 0], x2[1, 0, 0, 1] = NAN, NAN, NAN, INF, NAN
x3 = torch.randn(1, 2, 4, 6, 6)
x3[0, 0, 1, 2, 3], x3[0, 1, 3, 5, 5] = NAN, NAN
samples = torch.rand(2, 3, 2)
CASES = {f"adaptive_max_pool1d({o})": (lambda t, o=o: F.adaptive_max_pool1d(t, o, return_indices=True), x1)
         for o in (1, 2, 5)}
CASES.update({f"adaptive_max_pool2d({o})": (lambda t, o=o: F.adaptive_max_pool2d(t, o, return_indices=True), x2)
              for o in (1, 3, (2, 5))})
CASES["adaptive_max_pool3d(2)"] = (lambda t: F.adaptive_max_pool3d(t, 2, return_indices=True), x3)
CASES["fractional_max_pool2d"] = (lambda t: F.fractional_max_pool2d(
    t, 2, output_size=4, return_indices=True, _random_samples=samples.to(t.device)), x2)
for name, (f, x) in CASES.items():
    try:
        v, i = f(x.to("hpu"))
        if LAZY:
            htcore.mark_step()
        v, i = v.cpu(), i.cpu()
        wv, wi = f(x)
        nan_ok = torch.equal(v.isnan(), wv.isnan())
        val_ok = torch.equal(v.nan_to_num(), wv.nan_to_num())
        idx_ok = torch.equal(i, wi)
        print(f"ADAPT {name:28s} nan_ok={nan_ok} values_ok={val_ok} indices_ok={idx_ok}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"ADAPT {name:28s} raised {type(e).__name__}: {str(e)[:120]}", flush=True)
