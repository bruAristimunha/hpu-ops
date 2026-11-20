"""HPU composites for ops whose Gaudi kernel runs on the card but is slower than a few of Gaudi's other ops (issue #21).

- ``logsumexp`` in lazy mode as ATen's own composite, ``log(sum(exp(x - m))) + m`` with ``m`` the max with its
  infinities zeroed. In lazy mode Gaudi's ``logsumexp`` kernel is slower than amax, exp, sum and log recorded into the
  same graph; in eager mode its one kernel beats their six launches, so eager calls (and ``torch.compile``, which runs
  in eager mode) keep it. The max is detached: the result does not depend on it, so autograd differentiates
  ``exp(x - m) / sum`` only, as ATen's own ``logsumexp`` backward (``grad * exp(x - result)``). The two differ where
  ``result`` rounds to the max: for ``[1e30, 1e30, -1e30]`` ATen (and Gaudi's kernel) give ``[1, 1, 0]``, the
  composite the softmax ``[0.5, 0.5, 0]``. The reductions keep
  their dims and the result is reshaped: in lazy mode the backward of a dim reduction to a 0-d tensor that is then
  summed fails (``repro/lazy_expand_garbage_size.py``).

Registered at ``AutogradHPU``. Cases it does not take (eager mode, integer or complex inputs, an empty ``dim`` list, an
empty reduced dimension, symbolic sizes) go to the op's own kernel. 16-bit inputs are reduced in float32 and returned in
their dtype, as by an autocast to float32.
"""
import os

import torch
from torch._C import DispatchKey

from ._dispatch import count, register_kernels, static

aten = torch.ops.aten
REDUCED = (torch.float16, torch.bfloat16)


def lazy_mode():
    """Whether Habana's bridge records ops into lazy graphs, as the bridge reports it; off a Gaudi stack,
    ``PT_HPU_LAZY_MODE`` (unset: eager, as bridge 1.21.4 runs with it unset)."""
    try:
        from habana_frameworks.torch.utils.internal import is_lazy
    except ImportError:
        return os.environ.get("PT_HPU_LAZY_MODE", "0") not in ("0", "")
    return bool(is_lazy())


def _dims(input, dim):
    dims = [dim] if isinstance(dim, int) else list(dim)
    return [d % max(input.dim(), 1) for d in dims]


def logsumexp(input, dim, keepdim=False):
    """``log(sum(exp(input - m), dim)) + m`` with ``m`` the max over ``dim``, detached, with its infinities zeroed as
    ATen zeroes them: a slice of -inf gives -inf, one with +inf gives +inf, one with NaN gives NaN."""
    dims = _dims(input, dim)
    if torch.is_autocast_enabled(input.device.type):
        with torch.autocast(input.device.type, enabled=False):
            return logsumexp(input, dim, keepdim)
    x = input.float() if input.dtype in REDUCED else input
    m = x.detach().amax(dims, keepdim=True).nan_to_num(0.0, 0.0, 0.0)
    out = (x - m).exp().sum(dims, keepdim=True).log() + m
    if not keepdim:
        out = out.reshape([size for d, size in enumerate(input.shape) if d not in dims])
    return out.to(input.dtype) if input.dtype in REDUCED else out


def _logsumexp_takes(input, dim, keepdim=False):
    if not input.is_floating_point():
        return False
    dims = [dim] if isinstance(dim, int) else list(dim)
    if not dims or not all(-max(input.dim(), 1) <= d < max(input.dim(), 1) for d in dims):
        return False  # torch's own errors and its empty-list meaning stay native
    if len(set(_dims(input, dims))) != len(dims):
        return False
    sizes = [input.shape[d] for d in _dims(input, dims)] if input.dim() else []
    return static(*input.shape) and all(size > 0 for size in sizes)


def install_logsumexp(device_type, lazy=None):
    """``logsumexp`` on ``device_type`` as ATen's composite, when ``lazy`` (default: Habana's lazy mode is on). Returns
    an undo."""
    if not (lazy_mode() if lazy is None else lazy):
        return lambda: None
    op = aten.logsumexp.default

    def kernel(input, dim, keepdim=False):
        if _logsumexp_takes(input, dim, keepdim):
            count("logsumexp")
            return logsumexp(input, dim, keepdim)
        return op._op_dk(DispatchKey.Autograd, input, dim, keepdim)

    return register_kernels(device_type, [(op, "Autograd{}", kernel)])
