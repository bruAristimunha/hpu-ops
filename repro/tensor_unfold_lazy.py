"""Gaudi2, bridge 1.21.4, lazy mode: ``Tensor.unfold`` (``aten::unfold``, a view op) falls back to the CPU.

    PT_HPU_LAZY_MODE=1 python3 repro/tensor_unfold_lazy.py      # fallbacks: aten::unfold
    PT_HPU_LAZY_MODE=0 python3 repro/tensor_unfold_lazy.py      # none

No hpu_ops. Each case prints the bridge's CPU-fallback ops it made, whether the result matches the CPU, and whether it
is a view of its input (``unfold`` is a view in torch: a write into the input shows in it). The fallback returns a
copy, so in lazy mode the result is not a view. ``as_strided`` with unfold's sizes and strides is the same view, made
on the card.
"""
import os

import habana_frameworks.torch.core as htcore
import torch
from habana_frameworks.torch.hpu.metrics import metric_global

LAZY = os.environ.get("PT_HPU_LAZY_MODE", "1") != "0"


def fallbacks():
    return dict(dict(metric_global("cpu_fallback").stats()).get("FallbackOps", {}) or {})


def run(name, f, x):
    before = fallbacks()
    xh = x.to("hpu")
    got = f(xh)
    if LAZY:
        htcore.mark_step()
    same = torch.equal(got.cpu(), f(x))
    xh.add_(1)  # a write into the input: a view sees it
    if LAZY:
        htcore.mark_step()
    view = torch.equal(got.cpu(), f(x + 1))
    after = fallbacks()
    made = {k: after[k] - before.get(k, 0) for k in after if after[k] - before.get(k, 0)}
    print(f"{name:38s} matches CPU {same!s:5s} view {view!s:5s} fallbacks {made}", flush=True)


def unfold_as_strided(t, dim, size, step):
    m = torch.empty_strided(t.shape, t.stride(), device="meta").unfold(dim, size, step)
    return t.as_strided(m.shape, m.stride(), t.storage_offset())


if __name__ == "__main__":
    print("lazy", LAZY, "has HPU kernel for aten::unfold",
          torch._C._dispatch_has_kernel_for_dispatch_key("aten::unfold", "HPU"), flush=True)
    x = torch.arange(2 * 3 * 40, dtype=torch.float32).reshape(2, 3, 40)
    run("unfold(2, 8, 4)", lambda t: t.unfold(2, 8, 4), x)
    run("unfold(-1, 3, 5)", lambda t: t.unfold(-1, 3, 5), x)
    run("as_strided as unfold(2, 8, 4)", lambda t: unfold_as_strided(t, 2, 8, 4), x)
    # the pattern in TensorCSPNet's step: a 2-D conv weight gradient from the input's strided patches
    run("unfold(1, 3, 1).unfold(2, 8, 4)", lambda t: t.unfold(1, 3, 1).unfold(2, 8, 4), x)
