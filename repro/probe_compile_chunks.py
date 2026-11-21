"""Compiled (hpu_backend) conv1d (4, 8, L) x (16, 8, k) as a sum of kernel chunks of the given widths, forward and
backward, against CPU float64: which chunkings Habana's graph compiler takes (bridge 1.21.4). One case per process:
    PT_HPU_LAZY_MODE=0 python3 repro/probe_compile_chunks.py L W1,W2,...   (k = sum of the widths)"""
import json
import sys

import habana_frameworks.torch.core  # noqa: F401
import torch
import torch.nn.functional as F

length, widths = int(sys.argv[1]), [int(v) for v in sys.argv[2].split(",")]
k = sum(widths)
out_len = length - k + 1


def chunked(x, w):
    acc, k0 = None, 0
    for c in widths:
        y = F.conv1d(x[:, :, k0:k0 + out_len + c - 1], w[:, :, k0:k0 + c])
        acc = y if acc is None else acc + y
        k0 += c
    return acc


torch.manual_seed(0)
x, w = torch.randn(4, 8, length), torch.randn(16, 8, k) / k ** 0.5
xd, wd = x.double().requires_grad_(), w.double().requires_grad_()
F.conv1d(xd, wd).pow(2).sum().backward()
info = {"L": length, "widths": widths}
try:
    xh, wh = x.to("hpu").requires_grad_(), w.to("hpu").requires_grad_()
    y = torch.compile(chunked, backend="hpu_backend", dynamic=False)(xh, wh)
    y.pow(2).sum().backward()
    rel = lambda a, r: float((a.double() - r).abs().max() / r.abs().max())  # noqa: E731
    info.update(ok=True, out_rel=rel(y.detach().cpu(), F.conv1d(x.double(), w.double())),
                gx_rel=rel(xh.grad.cpu(), xd.grad), gw_rel=rel(wh.grad.cpu(), wd.grad))
except Exception as ex:  # noqa: BLE001
    info.update(ok=False, raised=repr(ex)[:120])
print("PROBE " + json.dumps(info), flush=True)
