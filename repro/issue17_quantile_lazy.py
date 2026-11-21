"""hpu-ops #17: torch.quantile in lazy mode (bridge 1.21.4) raises FATAL 'Trying to resize tensor with non-existing
TensorImpl'. One case per process (a failed lazy graph poisons the rest):
    PT_HPU_LAZY_MODE=0|1 python3 repro/issue17_quantile_lazy.py CASE [en]
CASE: scalar | tensorq | dim | nan | nansort | sort | kth | grad | perf. ``en`` runs hpu_ops.enable() first."""
import json
import os
import sys
import time
import traceback

import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.hpu as hthpu
import torch
from habana_frameworks.torch.hpu.metrics import metric_global

case = sys.argv[1]
enable = len(sys.argv) > 2 and sys.argv[2] == "en"
if enable:
    import hpu_ops
    hpu_ops.enable()
lazy = os.environ.get("PT_HPU_LAZY_MODE", "0") != "0"
hpu = torch.device("hpu")


def fb():
    return dict(dict(metric_global("cpu_fallback").stats()).get("FallbackOps", {}) or {})


def mark():
    if lazy:
        htcore.mark_step()


torch.manual_seed(0)
x = torch.randn(4096)
x2 = torch.randn(8, 300)
info = {"case": case, "lazy": lazy, "enable": enable, "bridge": "1.21.4"}
fb0 = fb()
try:
    if case == "scalar":
        got, want = torch.quantile(x.to(hpu), 0.9), torch.quantile(x.double(), 0.9)
    elif case == "tensorq":
        q = torch.tensor([0.0, 0.1, 0.5, 0.9, 1.0])
        got, want = torch.quantile(x.to(hpu), q.to(hpu)), torch.quantile(x.double(), q.double())
    elif case == "dim":
        got = torch.quantile(x2.to(hpu), 0.25, dim=1, keepdim=True, interpolation="midpoint")
        want = torch.quantile(x2.double(), 0.25, dim=1, keepdim=True, interpolation="midpoint")
    elif case == "nan":
        xn = x2.clone()
        xn[2, 5] = float("nan")
        got = torch.stack([torch.quantile(xn.to(hpu), 0.5, dim=1), torch.nanquantile(xn.to(hpu), 0.5, dim=1)])
        want = torch.stack([torch.quantile(xn.double(), 0.5, dim=1), torch.nanquantile(xn.double(), 0.5, dim=1)])
    elif case == "nansort":  # Gaudi's sort of a NaN
        xn = torch.tensor([[3.0, float("nan"), 1.0, 2.0], [4.0, 3.0, float("inf"), 1.0]])
        got, want = torch.sort(xn.to(hpu), -1).values, torch.sort(xn.double(), -1).values
    elif case == "sort":  # the pieces quantile is built of, without its out= / resize
        s = torch.sort(x.to(hpu)).values
        r = 0.9 * (x.numel() - 1)
        lo = int(r)
        got = torch.lerp(s[lo], s[lo + 1], r - lo)
        want = torch.quantile(x.double(), 0.9)
    elif case == "kth":
        got, want = torch.kthvalue(x.to(hpu), 100).values, torch.kthvalue(x.double(), 100).values
    elif case == "grad":
        xh = x2.to(hpu).requires_grad_()
        y = torch.quantile(xh, torch.tensor([0.3, 0.7], device=hpu), dim=1)
        y.sum().backward()
        mark()
        xd = x2.double().requires_grad_()
        torch.quantile(xd, torch.tensor([0.3, 0.7], dtype=torch.float64), dim=1).sum().backward()
        info["grad_err"] = float((xh.grad.cpu().double() - xd.grad).abs().max())
        got, want = y, torch.quantile(x2.double(), torch.tensor([0.3, 0.7], dtype=torch.float64), dim=1)
    elif case == "perf":
        xh = x2.to(hpu)
        for _ in range(5):
            y = torch.quantile(xh, 0.9, dim=1)
            mark()
        hthpu.synchronize()
        keep, t0 = [], time.perf_counter()
        for _ in range(30):
            keep.append(torch.quantile(xh, 0.9, dim=1))
            mark()
        hthpu.synchronize()
        info["ms"] = (time.perf_counter() - t0) / 30 * 1e3
        got, want = keep[-1], torch.quantile(x2.double(), 0.9, dim=1)
    mark()
    got = got.cpu()
    hthpu.synchronize()
    info.update(ok=True, err=float((got.double() - want).abs().nan_to_num(0.0).max()),
                nan_same=bool(torch.equal(got.isnan(), want.isnan())))
except Exception as ex:  # noqa: BLE001
    info.update(ok=False, raised=repr(ex)[:300], where=traceback.format_exc().splitlines()[-3][:200])
info["fallbacks"] = {k: v - fb0.get(k, 0) for k, v in fb().items() if v - fb0.get(k, 0)}
if enable:
    info["calls"] = dict(hpu_ops._dispatch._STATE["calls"])
print("PROBE " + json.dumps(info), flush=True)
