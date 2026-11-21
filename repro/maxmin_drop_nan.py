"""Habana bug (Gaudi2, bridge 1.21.4, torch 2.6; eager, lazy and hpu_backend compile): the max/min family drops NaN.

    PT_HPU_LAZY_MODE=0 python maxmin_drop_nan.py     # and PT_HPU_LAZY_MODE=1

torch propagates NaN through every op below (CPU and CUDA). Gaudi returns a number, so a NaN in a model never reaches
the loss: NaN checks do not fire, training goes on with finite losses. Measured (no hpu_ops):

    x = [[nan, 1, 2, -1], [2, 1, nan, -1], [nan, nan, nan, nan], [inf, nan, -inf, 0]]
    amax(-1)                 -inf, 2, -inf, inf          torch: nan, nan, nan, nan
    max(-1)                  (-inf, 2, -inf, inf), indices 0, 0, 0, 0   torch: nan x4, indices 0, 2, 0, 1 (first NaN)
    argmax(-1) bfloat16      2, 0, 4, 0                  torch: 0, 2, 0, 1   (4 is out of range: n)
    maximum(x, y)            the first operand wherever one is NaN       torch: NaN
    fmax(x, y)               the first operand wherever one is NaN       torch: the other operand
    norm(x, inf, -1)         finite                      torch: nan
    max_pool1d(x, 2)         [inf, nan] -> inf           torch: nan
    median(-1)               finite                      torch: nan
    sort(-1)                 NaN returned as +inf, indices out of range  torch: NaN last, indices of the NaNs

Also without NaN: bfloat16 / float16 argmax and max(dim) of a row of -inf return index n (out of range; float32
returns 0, torch 0); float32 sort (not stable) of a 64 x 1000 tensor with +-inf returns out-of-range indices (stable
sort is right); max_pool indices are window-local (kernel coordinates), not torch's flat plane indices.
Right on Gaudi: softmax, log_softmax, logsumexp, cummax/cummin, sum/mean/nansum, nanmedian, relu, hardtanh.
Workaround (hpu-ops ``extrema``): run the native op, then ``where(isnan(x).any(dim), nan, result)``, the first NaN's
index from ``argmax(isnan(x))``, and for pools a second pool of the NaN positions.
"""
import os

import habana_frameworks.torch.core as htcore
import torch
import torch.nn.functional as F

NAN, INF = float("nan"), float("inf")
LAZY = os.environ.get("PT_HPU_LAZY_MODE", "1") != "0"
x = torch.tensor([[NAN, 1.0, 2.0, -1.0], [2.0, 1.0, NAN, -1.0], [NAN, NAN, NAN, NAN], [INF, NAN, -INF, 0.0]])
y = torch.tensor([[1.0, NAN, 0.0, NAN], [NAN, 1.0, NAN, 0.0], [0.0] * 4, [NAN, 0.0, 0.0, 0.0]])
CASES = {
    "amax(-1)": lambda t: t.amax(-1),
    "max(-1).values": lambda t: t.max(-1).values,
    "max(-1).indices": lambda t: t.max(-1).indices,
    "argmax(-1) bf16": lambda t: t.bfloat16().argmax(-1),
    "maximum(x, y)": lambda t: torch.maximum(t, y.to(t.device)),
    "fmax(x, y)": lambda t: torch.fmax(t, y.to(t.device)),
    "norm(inf, -1)": lambda t: t.norm(INF, -1),
    "max_pool1d(2)": lambda t: F.max_pool1d(t.unsqueeze(0), 2).squeeze(0),
    "median(-1)": lambda t: t.median(-1).values,
    "sort(-1).values": lambda t: t.sort(-1).values,
    "argmax bf16 row of -inf": lambda t: torch.full((1, 4), -INF, device=t.device).bfloat16().argmax(-1),
}
bad = 0
for name, fn in CASES.items():
    got = fn(x.to("hpu"))
    if LAZY:
        htcore.mark_step()
    got, want = got.cpu(), fn(x)
    ok = torch.equal(got.isnan(), want.isnan()) and torch.equal(
        torch.where(got.isnan(), 0, got).float(), torch.where(want.isnan(), 0, want).float())
    bad += not ok
    print(f"{name:24s} hpu {got.tolist()}  cpu {want.tolist()}  -> {'OK' if ok else 'BUG'}", flush=True)
print(f"{bad} of {len(CASES)} differ from torch (lazy={LAZY})")
