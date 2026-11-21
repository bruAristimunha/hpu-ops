"""conv2d (1, 1, H, W) x (1, 1, kh, kw) forward + backward on Gaudi2 (bridge 1.21.4), native or with hpu_ops (en),
against CPU float64. One case per process: PT_HPU_LAZY_MODE=0|1 python3 repro/probe_conv2d_lazy.py H W KH KW [en]"""
import json
import os
import sys

import habana_frameworks.torch.core as htcore
import torch
import torch.nn.functional as F

h, w_, kh, kw = (int(v) for v in sys.argv[1:5])
enable = len(sys.argv) > 5 and sys.argv[5] == "en"
if enable:
    import hpu_ops
    hpu_ops.enable()
lazy = os.environ.get("PT_HPU_LAZY_MODE", "0") != "0"
torch.manual_seed(0)
x, w = torch.randn(1, 1, h, w_), torch.randn(1, 1, kh, kw) / (kh * kw) ** 0.5
xd, wd = x.double().requires_grad_(), w.double().requires_grad_()
F.conv2d(xd, wd).pow(2).sum().backward()
info = {"x": [h, w_], "k": [kh, kw], "lazy": lazy, "enable": enable}
try:
    xh, wh = x.to("hpu").requires_grad_(), w.to("hpu").requires_grad_()
    y = F.conv2d(xh, wh)
    y.pow(2).sum().backward()
    if lazy:
        htcore.mark_step()
    rel = lambda a, r: float((a.double() - r).abs().max() / r.abs().max())  # noqa: E731
    info.update(ok=True, out_rel=rel(y.detach().cpu(), F.conv2d(x.double(), w.double())),
                gx_rel=rel(xh.grad.cpu(), xd.grad), gw_rel=rel(wh.grad.cpu(), wd.grad))
    if enable:
        info["calls"] = dict(hpu_ops._dispatch._STATE["calls"])
except Exception as ex:  # noqa: BLE001
    info.update(ok=False, raised=repr(ex)[:160])
print("PROBE " + json.dumps(info), flush=True)
