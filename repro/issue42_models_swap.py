"""Issue #42: braindecode models compiled with default dynamic shapes, batch A then batch B, three ways.

    MODE=plugins   hpu-ops enabled (group_norm rewritten while tracing with symbolic shapes)
    MODE=swap      no plugins; every nn.GroupNorm replaced by a module computing hpu_ops.norm.group_norm
    MODE=off       no plugins, models as they are (the baseline: IntImplicit)
    MODE=identity  no plugins; group norm replaced by the identity (nn.GroupNorm and F.group_norm, which
                   braindecode's masked_group_norm calls): do the other failures need group norm at all?

Tells apart a wrong composite from another Habana gap that the IntImplicit failure hid: ``swap`` runs the same math
without any plugin. Needs BD_ROOT (braindecode source + test fixtures) and the audit's probe.py on PYTHONPATH.
Usage: MODE=... python issue42_models_swap.py NAME [NAME ...]
"""
import os
import sys

import torch
from torch import nn

import habana_frameworks.torch.core  # noqa: F401
import habana_frameworks.torch.hpu as hthpu
from habana_frameworks.torch.hpu.metrics import metric_global

MODE = os.environ.get("MODE", "plugins")
if MODE == "plugins":
    import hpu_ops

    hpu_ops.enable()
from hpu_ops import norm  # noqa: E402  (importing alone installs nothing)

if MODE == "identity":
    torch.nn.functional.group_norm = lambda x, *args, **kwargs: x

import probe  # noqa: E402


class Composite(nn.Module):
    def __init__(self, gn):
        super().__init__()
        self.num_groups, self.eps, self.weight, self.bias = gn.num_groups, gn.eps, gn.weight, gn.bias

    def forward(self, x):
        return norm.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)


def swap(model):
    for name, child in model.named_children():
        if isinstance(child, nn.GroupNorm):
            setattr(model, name, nn.Identity() if MODE == "identity" else Composite(child))
        else:
            swap(child)


def fallbacks():
    return dict(dict(metric_global("cpu_fallback").stats()).get("FallbackOps", {}) or {})


for name in sys.argv[1:]:
    torch.manual_seed(0)
    model, C, T = probe.build(name)
    if MODE in ("swap", "identity"):
        swap(model)
    xc = {b: torch.randn(b, C, T) for b in (16, 128)}
    with torch.no_grad():
        ref = {b: model(xc[b]) for b in xc}
    model.to("hpu")
    xh = {b: v.to("hpu") for b, v in xc.items()}
    for order in ((16, 128), (128, 16)):
        torch._dynamo.reset()
        c = torch.compile(model, backend="hpu_backend")
        for b in order:
            before = fallbacks()
            try:
                with torch.no_grad():
                    out = c(xh[b])
                hthpu.synchronize()
                cmp = probe.compare(out, ref[b])
                res = f"{cmp['status']} diff {cmp['diff']:.1e}"
            except Exception as e:  # noqa: BLE001
                res = f"{type(e).__name__}: {str(e).splitlines()[1 if 'FATAL' in str(e) else 0][:160]}"
            new = {k: v - before.get(k, 0) for k, v in fallbacks().items() if v - before.get(k, 0)}
            print(f"{MODE:8s} {name:22s} order {order} batch {b:3d}: {res} | fallback {new}", flush=True)
