"""hpu_ops: what Intel Gaudi's PyTorch stack lacks or gets wrong, fixed at the root or with Habana's own settings.

Model code and torch stay unmodified. On HPU, graphs compiled with ``torch.compile(backend="hpu_backend")`` get
the fixes. Eager execution and other devices run as before.

- Linear upsampling under compile. When ``torch.compile`` traces ``upsample_linear1d.vec``, PyTorch's Python
  decomposition rewrites it as gather/scatter index ops. On Gaudi their backward is a sort-based scatter-add: at
  CodeBrain's sizes, a compiled forward + backward takes 56.9 ms, against 2.8 ms in eager mode. This is unfixed in
  bridge 1.24.2, where Habana fixed only bicubic2d and trilinear3d. ``enable()`` replaces that decomposition, the
  way Habana's own ``override_composite_ops`` does for other ops. HPU tensors with static shapes get the native
  ``upsample_linear1d.default``, which Habana compiles with its own kernels (3.4 ms). Everything else, including
  symbolic shapes, whose native backward has no meta kernel, keeps PyTorch's decomposition.
- Two Habana compile passes break valid graphs. ``enable()`` turns them off through Habana's own
  ``PT_HPU_DISABLE_<pass>`` switch, which is read at every compile. Turning them off costs at most 1.7% of a step
  (EEGPT).
  - ``pass_remove_unnecessary_bmm_view`` strips einsum's views (BIOT: "Can not compile graph"). It is unfixed
    through 1.24.2; Intel uses the same switch in vllm-gaudi PR #1102.
  - ``pass_batch_as_strided_groups`` puts a node before its input (LaBraM: "used before defined"). Fixed in 1.22.

Masked attention needs a Habana setting that is read when its library loads, so it must be exported before Python
starts. The setting is per bridge version, listed in the README.
"""
import os

import torch
from torch._C import DispatchKey
from torch._decomp.decompositions import upsample_compute_output_size

__all__ = ["enable"]

aten = torch.ops.aten
BROKEN_PASSES = ("pass_remove_unnecessary_bmm_view", "pass_batch_as_strided_groups")
_KEYS = (DispatchKey.CompositeImplicitAutograd, DispatchKey.Autograd)  # where torch registers the decomposition
_STATE = {}


def _native_upsample_linear1d_vec(device_type, decomposition):
    """``upsample_linear1d.vec`` for tracing: the native op on ``device_type`` with static shapes, else ``decomposition``."""

    def kernel(input, output_size, align_corners, scale_factors):
        sizes = list(input.shape) + list(output_size or [])
        if input.device.type != device_type or not all(isinstance(s, int) for s in sizes):
            return decomposition(input, output_size, align_corners, scale_factors)
        _STATE["native_calls"] = _STATE.get("native_calls", 0) + 1
        size = upsample_compute_output_size(input.size(), output_size, scale_factors)
        scale = scale_factors[0] if scale_factors else None
        return aten.upsample_linear1d.default(input, size, align_corners, scale)

    return kernel


def _override_upsample_linear1d_vec(device_type):
    """Swap torch's Python decomposition of ``upsample_linear1d.vec`` for the native-op kernel. Returns an undo."""
    op = aten.upsample_linear1d.vec
    originals = {key: op.py_kernels.pop(key) for key in _KEYS if key in op.py_kernels}
    for key, decomposition in originals.items():
        op.py_impl(key)(_native_upsample_linear1d_vec(device_type, decomposition))  # py_impl clears the cache

    def undo():
        for key, decomposition in originals.items():
            op.py_kernels[key] = decomposition
        op._dispatch_cache.clear()

    return undo


def enable():
    """Replace the upsampling decomposition and turn off the broken passes. Idempotent; a no-op off a Gaudi stack."""
    if _STATE.get("undo"):
        return
    try:
        import habana_frameworks.torch.core  # noqa: F401
    except ImportError:
        return
    for name in BROKEN_PASSES:
        if not os.environ.get(f"PT_HPU_DISABLE_{name}"):  # unset or empty; an explicit user setting wins
            os.environ[f"PT_HPU_DISABLE_{name}"] = "True"
    _STATE["undo"] = _override_upsample_linear1d_vec("hpu")
