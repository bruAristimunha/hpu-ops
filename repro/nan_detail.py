"""Gaudi2 vs CPU on a few NaN vectors, printed side by side (details behind probe_reduction_nan.py's table)."""
import os

import habana_frameworks.torch.core as htcore
import torch
import torch.nn.functional as F

NAN, INF = float("nan"), float("inf")
LAZY = os.environ.get("PT_HPU_LAZY_MODE", "1") != "0"
v = torch.tensor([[NAN, 1.0, 2.0, -1.0], [2.0, 1.0, NAN, -1.0], [NAN, NAN, NAN, NAN], [INF, NAN, -INF, 0.0],
                  [1.0, 3.0, 3.0, -0.0], [-INF, -INF, -INF, -INF]])
w = torch.tensor([[1.0, NAN, 0.0, NAN], [NAN, 1.0, NAN, 0.0], [0.0, 0.0, 0.0, 0.0], [NAN, 0.0, 0.0, -0.0],
                  [-0.0, 0.0, 3.0, 0.0], [0.0, NAN, 1.0, -INF]])

CASES = {
    "clamp_min0": lambda x: x.clamp(min=0.0), "clamp_max0": lambda x: x.clamp(max=0.0),
    "clamp_min_nanbound": lambda x: x.clamp(min=NAN), "clamp_tmin_nan": lambda x: x.clamp(min=torch.full_like(x, NAN)),
    "clamp_min_t": lambda x: torch.clamp_min(x, torch.zeros_like(x)),
    "maximum": lambda x: torch.maximum(x, w.to(x)), "minimum": lambda x: torch.minimum(x, w.to(x)),
    "fmax": lambda x: torch.fmax(x, w.to(x)), "fmin": lambda x: torch.fmin(x, w.to(x)),
    "max_bin": lambda x: torch.max(x, w.to(x)),
    "gelu": F.gelu, "gelu_tanh": lambda x: F.gelu(x, approximate="tanh"),
    "argmax": lambda x: x.argmax(-1), "argmin": lambda x: x.argmin(-1),
    "max_dim": lambda x: tuple(x.max(-1)), "min_dim": lambda x: tuple(x.min(-1)),
    "sort": lambda x: tuple(x.sort(-1)), "sort_desc": lambda x: tuple(x.sort(-1, descending=True)),
    "topk2": lambda x: tuple(x.topk(2, -1)), "median": lambda x: tuple(x.median(-1)),
    "cummax": lambda x: tuple(x.cummax(-1)), "softmax": lambda x: x.softmax(-1),
    "amax_neg0": lambda x: torch.tensor([-0.0, 0.0]).to(x).amax(),
    "amax_neg0b": lambda x: torch.tensor([0.0, -0.0]).to(x).amax(),
    "max_pool1d_idx": lambda x: tuple(F.max_pool1d(x.unsqueeze(0), 2, return_indices=True)),
    "where_isnan": lambda x: torch.where(x.isnan().any(-1), NAN, x.amax(-1)),
}

for dtype in (torch.float32, torch.bfloat16, torch.float16):
    print(f"==== {dtype} lazy={LAZY}")
    for name, fn in CASES.items():
        x = v.to(dtype)
        try:
            got = fn(x.to("hpu"))
            if LAZY:
                htcore.mark_step()
            got = [t.cpu().tolist() for t in got] if isinstance(got, tuple) else got.cpu().tolist()
        except Exception as e:  # noqa: BLE001
            got = f"raised {type(e).__name__}: {str(e)[:120]}"
        want = fn(x)
        want = [t.tolist() for t in want] if isinstance(want, tuple) else want.tolist()
        print(f"{name:20s} {'SAME' if str(got) == str(want) else 'DIFF'}\n   hpu {got}\n   cpu {want}", flush=True)
