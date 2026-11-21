"""Which conv1d (B, C, L) x (O, C, k) compile under torch.compile(backend="hpu_backend") on Gaudi2 (bridge 1.21.4),
forward and backward, natively or with hpu_ops.enable() (en). One case per process:
    PT_HPU_LAZY_MODE=0 python3 repro/probe_compile_conv_k.py K L [en] [B C O]"""
import json
import sys

import habana_frameworks.torch.core  # noqa: F401
import torch
import torch.nn.functional as F

k, length = int(sys.argv[1]), int(sys.argv[2])
enable = len(sys.argv) > 3 and sys.argv[3] == "en"
b, c, o = (int(v) for v in sys.argv[4:7]) if len(sys.argv) > 6 else (4, 8, 16)
if enable:
    import hpu_ops
    hpu_ops.enable()
torch.manual_seed(0)
x, w = torch.randn(b, c, length), torch.randn(o, c, k) / k ** 0.5
xd, wd = x.double().requires_grad_(), w.double().requires_grad_()
F.conv1d(xd, wd).pow(2).sum().backward()
info = {"k": k, "L": length, "bco": [b, c, o], "compile": True, "enable": enable}
try:
    f = torch.compile(lambda a, bb: F.conv1d(a, bb), backend="hpu_backend", dynamic=False)
    xh, wh = x.to("hpu").requires_grad_(), w.to("hpu").requires_grad_()
    y = f(xh, wh)
    y.pow(2).sum().backward()
    y, gx, gw = y.detach().cpu(), xh.grad.cpu(), wh.grad.cpu()
    ref = F.conv1d(x.double(), w.double())
    rel = lambda a, r: float((a.double() - r).abs().max() / r.abs().max())  # noqa: E731
    info.update(ok=True, out_rel=rel(y, ref), gx_rel=rel(gx, xd.grad), gw_rel=rel(gw, wd.grad))
    if enable:
        info["calls"] = dict(hpu_ops._dispatch._STATE["calls"])
except Exception as ex:  # noqa: BLE001
    info.update(ok=False, raised=repr(ex)[:160])
print("PROBE " + json.dumps(info), flush=True)
