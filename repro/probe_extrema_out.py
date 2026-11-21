"""Gaudi2 with hpu_ops.enable(): out= overloads, in-place clamp_ with tensor bounds, and inference mode, with NaN,
against torch on the CPU. One ``PROBE {json}`` per case.  PT_HPU_LAZY_MODE=0|1 python3 repro/probe_extrema_out.py"""
import json
import os

import habana_frameworks.torch.core as htcore
import torch
import torch.nn.functional as F

import hpu_ops

hpu_ops.enable()
LAZY = os.environ.get("PT_HPU_LAZY_MODE", "1") != "0"
NAN, INF = float("nan"), float("inf")
x = torch.tensor([[NAN, 1.0, 2.0, -1.0], [2.0, 1.0, NAN, -1.0], [NAN] * 4, [INF, NAN, -INF, 0.0], [1.0, 3.0, 3.0, 0.0]])
y = x.flip(0)


def e(dt, d):
    return torch.empty(0, dtype=dt, device=d)


CASES = {
    "amax out": lambda t, u: torch.amax(t, -1, out=e(t.dtype, t.device)),
    "max dim out": lambda t, u: torch.stack([v.float() for v in torch.max(t, -1, out=(e(t.dtype, t.device),
                                                                                   e(torch.long, t.device)))]),
    "argmax out": lambda t, u: torch.argmax(t, -1, out=e(torch.long, t.device)),
    "maximum out": lambda t, u: torch.maximum(t, u, out=e(t.dtype, t.device)),
    "clamp out": lambda t, u: torch.clamp(t, u - 1, u + 1, out=e(t.dtype, t.device)),
    "clamp_ inplace": lambda t, u: t.clone().clamp_(min=u),
    "clamp_max_ inplace": lambda t, u: t.clone().clamp_max_(u),
    "vector_norm f64": lambda t, u: torch.linalg.vector_norm(t, INF, -1, dtype=torch.float64).float(),
}
INFER = {"amax": lambda t, u: t.amax(-1), "max dim": lambda t, u: t.max(-1).values,
         "max dim idx": lambda t, u: t.max(-1).indices, "argmax bf16": lambda t, u: t.bfloat16().argmax(-1),
         "max()": lambda t, u: t.max().reshape(1), "median": lambda t, u: t.median(-1).values,
         "maximum": lambda t, u: torch.maximum(t, u), "fmax": lambda t, u: torch.fmax(t, u),
         "clamp t": lambda t, u: t.clamp(u - 1, u + 1), "max_pool1d": lambda t, u: F.max_pool1d(t.unsqueeze(0), 2),
         "max_pool2d": lambda t, u: F.max_pool2d(t.reshape(1, 1, 5, 4), 2)}
bad = 0
for table, inference in ((CASES, False), (INFER, True)):
    for name, f in table.items():
        try:
            if inference:
                with torch.inference_mode():
                    got = f(x.to("hpu"), y.to("hpu"))
            else:
                got = f(x.to("hpu"), y.to("hpu"))
            if LAZY:
                htcore.mark_step()
            got, want = got.cpu(), f(x, y)
            ok = torch.equal(got.isnan(), want.isnan()) and torch.equal(got.nan_to_num().double(),
                                                                         want.nan_to_num().double())
            row = {"case": name, "inference": inference, "lazy": LAZY, "ok": ok}
            if not ok:
                row.update(got=got.tolist(), want=want.tolist())
        except Exception as ex:  # noqa: BLE001
            row = {"case": name, "inference": inference, "lazy": LAZY, "ok": False, "raised": str(ex)[:160]}
        bad += not row["ok"]
        print("PROBE " + json.dumps(row), flush=True)
print(f"DONE bad={bad}", flush=True)
