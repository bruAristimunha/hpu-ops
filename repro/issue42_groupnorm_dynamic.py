"""Issue #42: bare GroupNorm under torch.compile(backend="hpu_backend"), batch 16 then 128 (and 128 then 16).

Run with PT_HPU_LAZY_MODE=0. PLUGINS=0 runs without hpu-ops and hpu-complex. TRAIN=1 also runs a backward pass
(affine GroupNorm in train mode) and compares the gradients with the CPU's. VARY_LEN=1 gives the second batch size
another length too (60 samples, not 100), so the length is symbolic as well.
"""
import os

import torch
from torch import nn

import habana_frameworks.torch.core  # noqa: F401  (registers the device)
import habana_frameworks.torch.hpu as hthpu
from habana_frameworks.torch.hpu.metrics import metric_global

if os.environ.get("PLUGINS", "1") == "1":
    import hpu_ops

    try:
        import hpu_complex

        hpu_complex.enable()
    except ImportError:
        print("hpu_complex not importable, hpu_ops only")
    hpu_ops.enable()

TRAIN = os.environ.get("TRAIN", "0") == "1"
GROUPS, CHANNELS = int(os.environ.get("GROUPS", 4)), int(os.environ.get("CHANNELS", 16))


def fallbacks():
    return dict(dict(metric_global("cpu_fallback").stats()).get("FallbackOps", {}) or {})


torch.manual_seed(0)
module = nn.GroupNorm(GROUPS, CHANNELS)
module = module.train() if TRAIN else module.eval()
with torch.no_grad():
    module.weight.uniform_(0.5, 1.5)
    module.bias.uniform_(-0.5, 0.5)
VARY_LEN = os.environ.get("VARY_LEN", "0") == "1"
x = {b: torch.randn(b, CHANNELS, 60 if VARY_LEN and b == 128 else 100) for b in (16, 128)}
ref, ref_grads = {}, {}
for b in x:
    xr = x[b].clone().requires_grad_(TRAIN)
    out = module(xr)
    ref[b] = out.detach()
    if TRAIN:
        module.zero_grad()
        out.square().sum().backward()
        ref_grads[b] = (xr.grad, module.weight.grad.clone(), module.bias.grad.clone())
module.to("hpu")
xh = {b: v.to("hpu") for b, v in x.items()}


def run(compiled, b):
    xi = xh[b].clone().requires_grad_(TRAIN)
    if TRAIN:
        module.zero_grad()
        out = compiled(xi)
        out.square().sum().backward()
        hthpu.synchronize()
        g = (xi.grad.cpu(), module.weight.grad.cpu(), module.bias.grad.cpu())
        gd = max(float((a - r).abs().max() / (r.abs().max() + 1e-12)) for a, r in zip(g, ref_grads[b]))
        return out, f", grad rel diff {gd:.1e}"
    with torch.no_grad():
        out = compiled(xi)
    hthpu.synchronize()
    return out, ""


for tag, kwargs, order in (
    ("default", {}, (16, 128)),
    ("dynamic=False", {"dynamic": False}, (16, 128)),
    ("default, 128 first", {}, (128, 16)),
):
    torch._dynamo.reset()
    compiled = torch.compile(module, backend="hpu_backend", **kwargs)
    for b in order:
        before = fallbacks()
        try:
            out, extra = run(compiled, b)
            result = f"ok, max abs diff vs CPU {float((out.detach().cpu() - ref[b]).abs().max()):.1e}{extra}"
        except Exception as e:  # noqa: BLE001
            result = f"{type(e).__name__}: {str(e)[:300]}"
        after = fallbacks()
        new = {k: after[k] - before.get(k, 0) for k in after if after[k] - before.get(k, 0)}
        print(f"{tag:20s} batch {b:3d}: {result} | cpu_fallback: {new}", flush=True)
