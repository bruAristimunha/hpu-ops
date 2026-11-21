"""Which conv1d kernel sizes compile on Gaudi2 (bridge 1.21.4) forward AND backward, for (B, C, L) x (O, C, k).
One case per process (a failed lazy graph poisons the rest):
PT_HPU_LAZY_MODE=0|1 python3 repro/probe_conv_k_bwd.py K L [enable] [B C O]"""
import json
import os
import sys

import habana_frameworks.torch.core as htcore
import torch
import torch.nn.functional as F

k, length = int(sys.argv[1]), int(sys.argv[2])
enable = len(sys.argv) > 3 and sys.argv[3] == "en"
b, c, o = (int(v) for v in sys.argv[4:7]) if len(sys.argv) > 6 else (4, 8, 16)
if enable:
    import hpu_ops
    hpu_ops.enable()
lazy = os.environ.get("PT_HPU_LAZY_MODE", "0") != "0"
torch.manual_seed(0)
x, w = torch.randn(b, c, length), torch.randn(o, c, k) / k ** 0.5
xd, wd = x.double().requires_grad_(), w.double().requires_grad_()
F.conv1d(xd, wd).pow(2).sum().backward()
info = {"k": k, "L": length, "bco": [b, c, o], "lazy": lazy, "enable": enable}
try:
    xh, wh = x.to("hpu").requires_grad_(), w.to("hpu").requires_grad_()
    y = F.conv1d(xh, wh)
    y.pow(2).sum().backward()
    if lazy:
        htcore.mark_step()
    y, gx, gw = y.cpu(), xh.grad.cpu(), wh.grad.cpu()
    ref = F.conv1d(x.double(), w.double())
    rel = lambda a, r: float((a.double() - r).abs().max() / r.abs().max())  # noqa: E731
    info.update(ok=True, out_rel=rel(y, ref), gx_rel=rel(gx, xd.grad), gw_rel=rel(gw, wd.grad))
    if enable:
        info["calls"] = dict(hpu_ops._dispatch._STATE["calls"])
except Exception as ex:  # noqa: BLE001
    info.update(ok=False, raised=repr(ex)[:160])
print("PROBE " + json.dumps(info))
