"""Issue #42: eager GroupNorm on Gaudi with hpu_ops.norm's rewrite installed and undone, alternating in one process.

Mean ms of 50 synced calls after warm-up, outputs kept alive; forward under no_grad and forward + backward.
"""
import time

import torch
from torch import nn

import habana_frameworks.torch.core  # noqa: F401
import habana_frameworks.torch.hpu as hthpu

from hpu_ops import norm


def bench(m, x, train):
    keep = []

    def step():
        if train:
            xi = x.detach().requires_grad_()
            m(xi).square().sum().backward()
            return xi.grad
        with torch.no_grad():
            return m(x)

    for _ in range(5):
        keep.append(step())
    hthpu.synchronize()
    t = time.perf_counter()
    for _ in range(50):
        keep.append(step())
    hthpu.synchronize()
    return (time.perf_counter() - t) / 50 * 1e3


for groups, channels, length in ((4, 16, 100), (8, 64, 60), (8, 128, 1000)):
    m = nn.GroupNorm(groups, channels).to("hpu")
    x = torch.randn(128, channels, length, device="hpu")
    for train in (False, True):
        times = {"with": [], "without": []}
        for _ in range(3):
            undo = norm.install_group_norm("hpu")
            times["with"].append(bench(m, x, train))
            undo()
            times["without"].append(bench(m, x, train))
        print(f"{'train' if train else 'fwd  '} GroupNorm({groups},{channels}) x(128,{channels},{length}) eager ms: "
              + " | ".join(f"{k} " + " ".join(f"{t:.3f}" for t in v) for k, v in times.items()), flush=True)
