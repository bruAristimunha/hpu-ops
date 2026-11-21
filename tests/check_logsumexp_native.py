"""Gaudi check after dropping the lazy logsumexp composite (issue #21, round 2): with hpu_ops.enable(), logsumexp has no
hpu_ops kernel, and the round-2 breaks (+inf next to 100; 3e38 twice) get ATen's CPU gradient, with no CPU fallback.

    PT_HPU_LAZY_MODE=1 python check_logsumexp_native.py    # and PT_HPU_LAZY_MODE=0
"""
import warnings

import torch
import habana_frameworks.torch.core as htcore
from habana_frameworks.torch.hpu.metrics import metric_global

import hpu_ops

inf = float("inf")


def fallbacks():
    """Total count of ops the bridge ran on the CPU so far (the "FallbackOps" stat, a dict op -> count)."""
    return sum((dict(metric_global("cpu_fallback").stats()).get("FallbackOps", {}) or {}).values())


def grad(row, device):
    x = torch.tensor([row], device=device, requires_grad=True)
    out = torch.logsumexp(x, -1)
    (g,) = torch.autograd.grad(out.sum(), x)
    if device == "hpu":
        htcore.mark_step()
    return out.cpu(), g.cpu()


from habana_frameworks.torch.utils.internal import is_lazy

has = lambda: torch._C._dispatch_has_kernel_for_dispatch_key("aten::logsumexp", "AutogradHPU")  # noqa: E731
before = has()
hpu_ops.enable()
print(f"bridge lazy {bool(is_lazy())} | AutogradHPU kernel for logsumexp before enable {before}, after {has()}",
      "(unchanged)" if before == has() else "(CHANGED)")
bad = 0
for label, row in [("+inf next to 100", [inf, 100.0, 0.0]), ("3e38 twice", [3e38, 3e38, 0.0]),
                   ("random", torch.randn(4).tolist())]:
    for _ in range(2):  # each graph twice in a row
        f0 = fallbacks()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            out, g = grad(row, "hpu")
        nwarn = sum("runs on the CPU" in str(m.message) for m in w)
        want_out, want_g = grad(row, "cpu")
        ok = torch.allclose(g, want_g, atol=1e-6, equal_nan=True) and torch.allclose(out, want_out, equal_nan=True)
        bad += not ok
        print(f"{label:18s} grad hpu {g.tolist()} cpu {want_g.tolist()} fb {fallbacks() - f0} warn {nwarn}",
              "ok" if ok else "DIFF")
print("ALL OK" if not bad else f"{bad} DIFF")
