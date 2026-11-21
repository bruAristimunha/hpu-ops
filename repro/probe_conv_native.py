"""Does Gaudi2's native conv2d (bridge 1.21.4) run (and give the right numbers) for one shape, forward and backward?
One case per process (a failed lazy graph poisons the rest of the process):
    PT_HPU_LAZY_MODE=0|1 python3 repro/probe_conv_native.py B C H W O KH KW SH SW [XGRAD=1] [en]
Stages, in this order, each synced and compared with CPU float64 (relative max error):
  fwd: the output alone;  bwd: the backward, weight gradient and (XGRAD=1) the input gradient.
Prints one ``PROBE {json}``: ``stage`` is the last stage that ran, ``ok`` whether every stage ran with every relative
error below 1e-3 (``bad`` names the ones above). ``en`` runs it with hpu_ops.enable() (the split) instead of native."""
import json
import os
import sys

import habana_frameworks.torch.core as htcore
import torch
import torch.nn.functional as F

b, c, h, w_, o, kh, kw, sh, sw = (int(v) for v in sys.argv[1:10])
xgrad = len(sys.argv) <= 10 or sys.argv[10] != "0"
enable = "en" in sys.argv[11:]
if enable:
    import hpu_ops
    hpu_ops.enable()
lazy = os.environ.get("PT_HPU_LAZY_MODE", "0") != "0"
torch.manual_seed(0)
x, w = torch.randn(b, c, h, w_), torch.randn(o, c, kh, kw) / (c * kh * kw) ** 0.5
xd, wd = x.double().requires_grad_(xgrad), w.double().requires_grad_()
ref = F.conv2d(xd, wd, stride=(sh, sw))
ref.pow(2).sum().backward()
info = {"x": [b, c, h, w_], "w": [o, c, kh, kw], "stride": [sh, sw], "xgrad": xgrad, "lazy": lazy, "enable": enable,
        "bridge": "1.21.4", "stage": None, "ok": False}


def rel(a, r):
    return float((a.double() - r).abs().max() / r.abs().max().clamp_min(1e-30))


def sync():
    if lazy:
        htcore.mark_step()


try:
    xh, wh = x.to("hpu").requires_grad_(xgrad), w.to("hpu").requires_grad_()
    y = F.conv2d(xh, wh, stride=(sh, sw))
    sync()
    info["out_rel"] = rel(y.detach().cpu(), ref.detach())
    info["stage"] = "fwd"
    y.pow(2).sum().backward()
    sync()
    info["gw_rel"] = rel(wh.grad.cpu(), wd.grad)
    if xgrad:
        info["gx_rel"] = rel(xh.grad.cpu(), xd.grad)
    info["stage"] = "bwd"
    info["bad"] = [k for k in ("out_rel", "gw_rel", "gx_rel") if info.get(k, 0) > 1e-3]
    info["ok"] = not info["bad"]
    if enable:
        info["calls"] = dict(hpu_ops._dispatch._STATE["calls"])
except Exception as ex:  # noqa: BLE001
    info["raised"] = repr(ex)[:160]
print("PROBE " + json.dumps(info), flush=True)
