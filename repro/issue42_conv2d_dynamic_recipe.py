"""Habana bug (Gaudi2, bridge 1.21.4, PT_HPU_LAZY_MODE=0), no plugin: a Conv2d compiled with a symbolic batch returns
wrong values, silently, at batch sizes other than the one Habana compiled its recipe at.

    PT_HPU_LAZY_MODE=0 python3 repro/issue42_conv2d_dynamic_recipe.py

With the default ``torch.compile(module, backend="hpu_backend")``, batch 16 compiles a static graph and batch 128
recompiles it with a symbolic batch (dynamo's automatic dynamic shapes). Habana compiles that graph into one dynamic
recipe for the batch range [128, 256] (``get_input_symbolic`` in ``compile_backend/recipe_compiler.py`` uses
``[val, 2 * val]`` for an unbounded symbol), and every later batch size launches that recipe. For ``nn.Conv2d``
the results are wrong at most other batch sizes, inside the range too (192, 256) and outside it (32, 64, 300); a
range given by the user (``torch._dynamo.mark_dynamic(x, 0, min=2, max=256)``) fails the same way (NaN at 128).
Conv1d, Linear, BatchNorm2d, AvgPool2d and pointwise ops are right at every batch size in the same test; other Conv2d
shapes fail at other batch sizes. In braindecode, batches 16, 128 then 32 give, at batch 32, DIVER1 1.46 off,
SignalJEPA_PreLocal 1.03 and SignalJEPA_PostLocal 0.27 (outputs of about 0.3 to 0.8); DIVER1 and SignalJEPA_PreLocal
are as far off without hpu-ops, 1.45 and 1.03 (``MiniPreLocal`` below: 0.20).

Two settings avoid it: Habana's ``PT_HPU_FORCE_STATIC_COMPILE=1`` (exported before Python starts: dynamo keeps one
symbolic graph, Habana compiles one static recipe per batch size) and ``torch.compile(..., dynamic=False)``.
Measured with this script (max abs difference from the CPU, batch 16 first, then 128, then the others):

    default compile   Conv2d(19, 4, 1) 16: 3.6e-07  128: 4.8e-07  144: 7.2e-07  192: 3.0e+00  256: 3.4e+00  32: 4.0e+00
                                       64: 2.9e+00
                      MiniPreLocal     16: 1.4e-05  128: 1.8e-05  32: 2.0e-01
    FORCE_STATIC=1    Conv2d(19, 4, 1) every batch within 7.2e-07
    dynamic=False     Conv2d(19, 4, 1) every batch within 7.2e-07
"""
import os
import subprocess
import sys

import torch
import torch.nn as nn

BATCHES = (16, 128, 144, 192, 256, 32, 64)


class MiniPreLocal(nn.Module):
    """SignalJEPA_PreLocal's shapes: a per-channel Conv1d stack, a written-out group norm, a spatial Conv2d."""

    def __init__(self, c=19):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 8, 10, stride=5, bias=False)
        self.conv = nn.Sequential(nn.GELU(), nn.Conv1d(8, 16, 3, stride=2, bias=False), nn.GELU(),
                                  nn.Conv1d(16, 32, 3, stride=2, bias=False), nn.GELU())
        self.spatial = nn.Conv2d(c, 4, (1, 1))
        self.head = nn.Linear(32 * 4 * 9, 4)

    def forward(self, x):
        b, c, t = x.shape
        y = self.conv1(x.reshape(b * c, 1, t))
        var, mean = torch.var_mean(y, dim=-1, correction=0, keepdim=True)  # GroupNorm(8, 8) without affine
        y = self.conv((y - mean) * torch.rsqrt(var + 1e-5))
        return self.head(self.spatial(y.reshape(b, c, 32, -1)).flatten(1))


def run(module, shape, batches, **compile_kwargs):
    import habana_frameworks.torch.core  # noqa: F401
    import habana_frameworks.torch.hpu as hthpu

    torch.manual_seed(0)
    module = module.eval()
    x = {b: torch.randn(b, *shape) for b in batches}
    with torch.no_grad():
        ref = {b: module(x[b]) for b in batches}
    compiled = torch.compile(module.to("hpu"), backend="hpu_backend", **compile_kwargs)
    out = []
    for b in batches:
        with torch.no_grad():
            y = compiled(x[b].to("hpu"))
        hthpu.synchronize()
        out.append(f"{b}: {float((y.cpu() - ref[b]).abs().max()):.1e}")
    return "  ".join(out)


if __name__ == "__main__":
    if len(sys.argv) > 1:  # one setting per process: PT_HPU_FORCE_STATIC_COMPILE is read when Habana loads
        kwargs = {"dynamic": False} if sys.argv[1] == "dynamic=False" else {}
        print(f"{sys.argv[1]:17s} Conv2d(19, 4, 1) {run(nn.Conv2d(19, 4, (1, 1)), (19, 32, 9), BATCHES, **kwargs)}",
              flush=True)
        if sys.argv[1] == "default compile":
            print(f"{'':17s} MiniPreLocal     {run(MiniPreLocal(), (19, 200), (16, 128, 32))}", flush=True)
        sys.exit()
    for setting, env in (("default compile", {}), ("FORCE_STATIC=1", {"PT_HPU_FORCE_STATIC_COMPILE": "1"}),
                         ("dynamic=False", {})):
        subprocess.run([sys.executable, __file__, setting], env={**os.environ, **env}, check=False)
