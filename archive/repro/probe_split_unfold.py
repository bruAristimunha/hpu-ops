"""Where the lazy split of a TensorCSPNet-like strided conv2d gets its CPU 'unfold' fallbacks (bridge 1.21.4).
(32, 1, 9, W) x (16, 1, 4, 4356), stride (1, 484), as a sum of 128-wide chunks built three ways, forward + backward.
One variant per process: PT_HPU_LAZY_MODE=1 python3 repro/probe_split_unfold.py native|slice|contig|select|plainconv"""
import json
import sys

import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.hpu as hthpu
import torch
import torch.nn.functional as F
from habana_frameworks.torch.hpu.metrics import metric_global

variant = sys.argv[1]
K, S, C = 4356, 484, 128
W = K + S * 8
out_w = (W - K) // S + 1


def fb():
    return dict(dict(metric_global("cpu_fallback").stats()).get("FallbackOps", {}) or {})


def chunked(x, w):
    acc = None
    for k0 in range(0, K, C):
        c = min(C, K - k0)
        end = k0 + (out_w - 1) * S + c
        if variant == "slice":
            xc = x[:, :, :, k0:end]
        elif variant == "contig":
            xc = x[:, :, :, k0:end].contiguous()
        else:
            xc = x.index_select(3, torch.arange(k0, end, device=x.device))
        y = F.conv2d(xc, w[:, :, :, k0:k0 + c], stride=(1, S))
        acc = y if acc is None else acc + y
    return acc


torch.manual_seed(0)
x0, w0 = torch.randn(32, 1, 9, W), torch.randn(16, 1, 4, K) / 132
xh, wh = x0.to("hpu").requires_grad_(), w0.to("hpu").requires_grad_()
f0 = fb()
try:
    if variant == "native":
        y = F.conv2d(xh, wh, stride=(1, S))
    elif variant == "plainconv":  # one 128-wide strided chunk conv on a contiguous input, no slicing
        y = F.conv2d(xh[:, :, :, :C + (out_w - 1) * S].contiguous().detach().requires_grad_(), wh[:, :, :, :C],
                     stride=(1, S))
    else:
        y = chunked(xh, wh)
    y.pow(2).sum().backward()
    htcore.mark_step()
    hthpu.synchronize()
    ref = F.conv2d(x0.double(), w0.double(), stride=(1, S))
    err = float((y.detach().cpu().double() - ref).abs().max()) if variant != "plainconv" else None
    out = {"variant": variant, "ok": True, "out_err": err}
except Exception as e:  # noqa: BLE001
    out = {"variant": variant, "ok": False, "raised": repr(e)[:160]}
out["fallbacks"] = {k: v - f0.get(k, 0) for k, v in fb().items() if v - f0.get(k, 0)}
print("PROBE " + json.dumps(out), flush=True)
