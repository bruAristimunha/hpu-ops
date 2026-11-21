"""Issue #42: GroupNorm timings on Gaudi, eager and compiled (static, and symbolic after a second batch size).

PLUGINS=0 runs without hpu-ops. Mean ms of N synced calls after warm-up, outputs kept alive. Forward under no_grad
(MODE=fwd) or forward + backward (MODE=train).
"""
import os
import time

import torch
from torch import nn

import habana_frameworks.torch.core  # noqa: F401
import habana_frameworks.torch.hpu as hthpu
from habana_frameworks.torch.hpu.metrics import metric_global

if os.environ.get("PLUGINS", "1") == "1":
    import hpu_ops

    hpu_ops.enable()
    from hpu_ops._dispatch import _STATE
else:
    _STATE = {"calls": {}}

N, MODE = 50, os.environ.get("MODE", "fwd")


def fallbacks():
    return dict(dict(metric_global("cpu_fallback").stats()).get("FallbackOps", {}) or {})


def bench(f, x):
    keep = []

    def step():
        if MODE == "train":
            xi = x.detach().requires_grad_()
            out = f(xi)
            out.square().sum().backward()
            return xi.grad
        with torch.no_grad():
            return f(x)

    for _ in range(5):
        keep.append(step())
    hthpu.synchronize()
    t = time.perf_counter()
    for _ in range(N):
        keep.append(step())
    hthpu.synchronize()
    return (time.perf_counter() - t) / N * 1e3


torch.manual_seed(0)
for groups, channels, length in ((4, 16, 100), (8, 64, 60), (8, 128, 1000)):
    m = nn.GroupNorm(groups, channels).to("hpu")
    x = {b: torch.randn(b, channels, length, device="hpu") for b in (16, 128)}
    before = fallbacks()
    eager = bench(m, x[128])
    torch._dynamo.reset()
    static = torch.compile(m, backend="hpu_backend", dynamic=False)
    t_static = bench(static, x[128])
    torch._dynamo.reset()
    dyn = torch.compile(m, backend="hpu_backend")
    bench(dyn, x[16])  # first batch size: static graph
    try:
        t_dyn = f"{bench(dyn, x[128]):.3f} ms"  # second: symbolic graph
    except RuntimeError as e:  # without hpu-ops: IntImplicit
        t_dyn = f"fails ({str(e)[:40]})"
    new = {k: v - before.get(k, 0) for k, v in fallbacks().items() if v - before.get(k, 0)}
    print(f"{MODE} GroupNorm({groups},{channels}) x(128,{channels},{length}): eager {eager:.3f} ms, "
          f"compiled static {t_static:.3f} ms, compiled symbolic {t_dyn} | "
          f"group_norm rewrites {_STATE['calls'].get('group_norm', 0)} | cpu_fallback {new}", flush=True)
