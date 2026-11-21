"""HPU composites for ops whose Gaudi kernel fails under ``torch.compile`` (``glu``), is missing (``renorm``,
``Tensor.unfold`` in lazy mode, ``unfold_backward``) or is wrong (``embedding_dense_backward`` of a 0-d index, ``roll`` of
a strided input by a multiple of the size).

Each is registered at ``AutogradHPU``, so eager calls and traced graphs both use it and autograd differentiates the
composite; ``renorm`` is also registered at ``HPU`` for calls below autograd (inference mode). ``unfold`` and
``unfold_backward`` are registered at ``HPU`` only, below torch's own autograd kernels, which keep their formulas. Cases a composite does
not take (odd ``glu`` halves, invalid ``renorm`` arguments) go to the op's own kernel, which raises its usual error;
so do embedding gradients of indices with dimensions, which Gaudi gets right. The composites run with autocast off and
return the input dtype, as the native ops do.
"""
import math

import torch
from torch._C import DispatchKey

from ._dispatch import count, register_kernels, static

aten = torch.ops.aten
REDUCED = (torch.float16, torch.bfloat16)


def _opmath(dtype):
    return torch.float32 if dtype in REDUCED else dtype


def glu(input, dim=-1):
    """``a * sigmoid(b)`` for the two halves ``a, b`` of ``input`` along ``dim``; float32 math for 16-bit inputs."""
    half = input.shape[dim] // 2
    a, b = input.narrow(dim, 0, half), input.narrow(dim, half, half)
    dtype = _opmath(input.dtype)
    with torch.autocast(input.device.type, enabled=False):
        return (a.to(dtype) * torch.sigmoid(b.to(dtype))).to(input.dtype)


def _glu_takes(input, dim=-1):
    return input.is_floating_point() and input.dim() > 0 and input.shape[dim] % 2 == 0


def renorm(input, p, dim, maxnorm):
    """Each slice of ``input`` along ``dim`` scaled to a ``p``-norm of at most ``maxnorm``, as ``aten.renorm`` does
    it: by ``maxnorm / (norm + 1e-7)`` where ``norm > maxnorm``, else 1, with the norm in float32 for 16-bit inputs."""
    dim = dim % input.dim()
    others = [d for d in range(input.dim()) if d != dim]
    with torch.autocast(input.device.type, enabled=False):
        norm = torch.linalg.vector_norm(input, p, others, keepdim=True, dtype=_opmath(input.dtype))
        limit = torch.full_like(norm, maxnorm)
        factor = torch.where(norm > limit, limit / (norm + 1e-7), 1.0)
        return input * factor.to(input.dtype)


def _renorm_takes(input, p, dim, maxnorm):
    real = not isinstance(p, complex) and not isinstance(maxnorm, complex)
    shape = input.is_floating_point() and input.dim() > 1 and -input.dim() <= dim < input.dim()
    return real and shape and p > 0 and maxnorm >= 0


def embedding_dense_backward_one(grad_output, indices, num_weights, padding_idx, scale_grad_by_freq):
    """The weight gradient of a 0-d index's embedding as that of the one-element index, the shapes ATen's own kernels
    flatten to. Gaudi's kernel sends a 0-d index's gradient to row 0 (BIOT's channel tokens: every channel's gradient
    to the first token)."""
    return aten.embedding_dense_backward.default._op_dk(DispatchKey.Autograd, grad_output.reshape(1, -1),
                                                        indices.reshape(1), num_weights, padding_idx,
                                                        scale_grad_by_freq)


def _embedding_backward_takes(grad_output, indices, *args, **kwargs):
    return indices.dim() == 0


def roll_unshifted(input, shifts, dims=()):
    """A roll whose every shift is a multiple of its dim's size, as the copy it is: contiguous, as roll's result.
    Gaudi's eager kernel reads a strided input of such a roll wrongly (EMG2QwertyNet: a stack's gradient slice rolled
    back by 0)."""
    return input.clone(memory_format=torch.contiguous_format)


def _roll_takes(input, shifts, dims=()):
    if len(shifts) == 0 or len(shifts) != max(len(dims), 1) or not all(-input.dim() <= d < max(input.dim(), 1)
                                                                       for d in dims):
        return False  # what ATen rejects reaches it
    if all(isinstance(shift, int) and shift == 0 for shift in shifts):
        return True  # a copy at any size, symbolic too
    sizes = [input.numel()] if len(dims) == 0 else [input.shape[d] if input.dim() else 1 for d in dims]
    return static(*sizes, *shifts) and all(size == 0 or shift % size == 0 for size, shift in zip(sizes, shifts))


def unfold(input, dimension, size, step):
    """``Tensor.unfold`` as the view ATen makes of it: ``as_strided`` with the sizes and strides that ATen's own kernel
    computes, read from it on a meta tensor of ``input``'s sizes and strides (which also raises ATen's errors). The
    result is a view of ``input`` at its storage offset, as ``unfold``'s. Habana's lazy stack has no ``unfold``
    kernel: the bridge ran it on the CPU and returned a copy, not a view (``repro/tensor_unfold_lazy.py``)."""
    meta = torch.empty_strided(input.shape, input.stride(), dtype=input.dtype, device="meta")
    view = meta.unfold(dimension, size, step)
    return input.as_strided(view.shape, view.stride(), input.storage_offset())


def _pad_dims(t, pads):
    """``t`` with ``pads[d]`` zeros after its end along each dim ``d`` (one ``constant_pad_nd``: a contiguous copy)."""
    first = min(pads)
    pad = []
    for d in range(t.dim() - 1, first - 1, -1):
        pad += [0, pads.get(d, 0)]
    return aten.constant_pad_nd.default(t, pad, 0)


def unfold_backward(grad, input_sizes, dim, size, step):
    """The gradient of ``Tensor.unfold`` (``aten.unfold_backward``) as one or two pads and one sum, no index ops.

    Window ``i``, offset ``j`` goes to position ``i * step + j``. Overlaps are added by a skew: a padded buffer,
    flattened and read again with rows one shift shorter, so each row lands shifted, then one sum over the rows. Two
    skews, whichever buffer is smaller (both about the gradient's size at #43's shapes):

    - over the ``k = ceil(size / step)`` blocks of ``step`` offsets in a window (``j = q * step + r`` goes to block
      ``i + q``): the windows padded to ``k * step`` offsets, moved next to their blocks, padded from ``n`` to ``n +
      k + e`` blocks (``e`` reaches an uncovered tail); buffer ``k * (n + k + e) * step``, large when ``n << k``;
    - over the ``n`` windows: each padded to ``L + step`` and read again in rows of ``L``, so window ``i`` lands
      shifted by ``i * step``; buffer ``n * (L + step)``, large when ``L >> size``.

    Non-overlapping windows (``step >= size``) need no sum: one pad of each window to ``step`` and of the tail. The
    dims before and after ``dim`` are merged, so every pad, copy and sum sees at most 5 dims (Gaudi's lazy
    ``constant_pad_nd`` falls back above 5); a non-contiguous gradient is copied at its own rank first. The result is
    a new contiguous tensor, as ATen's; booleans add as ``or``, as ATen's CPU kernel. ``dim`` is checked as ATen
    does; ``step`` must be positive (``Tensor.unfold`` rejects 0, so autograd never passes it). The derivative stays
    torch's autograd formula (``unfold``)."""
    ndim = len(input_sizes)
    dim = torch._prims_common.canonicalize_dim(ndim, dim)  # ATen's IndexError
    if step <= 0:
        raise RuntimeError(f"unfold_backward: step is {step} but must be > 0")
    if ndim == 0:  # a 0-d input unfolds to (size,) with size 0 or 1
        return grad.any().reshape(()) if grad.dtype == torch.bool else grad.sum(dtype=grad.dtype).reshape(())
    length = input_sizes[dim]
    if size == 0 or grad.numel() == 0:
        return grad.new_zeros(input_sizes)
    n, k = grad.shape[dim], -(-size // step)
    outer, inner = math.prod(input_sizes[:dim]), math.prod(input_sizes[dim + 1:])
    g = grad.reshape(outer, n, inner, size).transpose(2, 3)  # (outer, n, size, inner), a view of a contiguous grad
    blocks = max(-(-length // step) - (n + k - 1), 0)  # e: zero blocks up to the input's end
    if k == 1:
        if blocks or step > size:
            g = _pad_dims(g, {1: blocks, 2: step - size})
        else:  # step == size and no tail: only a copy, never a view of ``grad``
            g = g.clone(memory_format=torch.contiguous_format)
        g = g.reshape(outer, (n + blocks) * step, inner).narrow(1, 0, length).contiguous()
        return g.reshape(input_sizes)
    rows = n + k - 1 + blocks
    if k * (rows + 1) * step <= n * (length + step):  # skew over the blocks
        if size < k * step:
            g = _pad_dims(g, {2: k * step - size})
        g = g.reshape(outer, n, k, step, inner).transpose(1, 2)  # (outer, k, n, step, inner), a view
        g = _pad_dims(g, {2: k + blocks})  # (outer, k, rows + 1, step, inner)
        g = g.reshape(outer, k * (rows + 1) * step * inner).narrow(1, 0, k * rows * step * inner)
        g = g.reshape(outer, k, rows * step, inner).narrow(2, 0, length)  # row q shifted by q blocks
    else:  # skew over the windows
        g = _pad_dims(g, {2: length + step - size})  # (outer, n, length + step, inner)
        g = g.reshape(outer, n * (length + step) * inner).narrow(1, 0, n * length * inner)
        g = g.reshape(outer, n, length, inner)  # window i shifted by i * step
    g = g.any(1) if g.dtype == torch.bool else g.sum(1, dtype=g.dtype)
    return g.reshape(input_sizes)


def _kernel(op, name, composite, takes, fallback):
    """``composite`` for the arguments ``takes`` accepts, else ``op``'s kernel at the ``fallback`` key."""

    def kernel(*args, **kwargs):
        if takes(*args, **kwargs):
            count(name)
            return composite(*args, **kwargs)
        return op._op_dk(fallback, *args, **kwargs)

    return kernel


def install_glu(device_type):
    """``glu`` on ``device_type`` as ``a * sigmoid(b)``. Returns an undo."""
    op = aten.glu.default
    kernel = _kernel(op, "glu", glu, _glu_takes, DispatchKey.Autograd)
    return register_kernels(device_type, [(op, "Autograd{}", kernel)])


def install_renorm(device_type):
    """``renorm`` on ``device_type`` as norm, scale factor and multiply. Returns an undo."""
    op = aten.renorm.default
    below = DispatchKey.CompositeExplicitAutogradNonFunctional  # what the HPU key resolved to: renorm.out, on the CPU
    return register_kernels(device_type, [
        (op, "Autograd{}", _kernel(op, "renorm", renorm, _renorm_takes, DispatchKey.Autograd)),
        (op, "{}", _kernel(op, "renorm", renorm, _renorm_takes, below)),
    ])


def install_embedding(device_type):
    """The embedding gradient of a 0-d index on ``device_type`` through the one-element index. Returns an undo."""
    op = aten.embedding_dense_backward.default
    return register_kernels(device_type, [(op, "Autograd{}", _kernel(
        op, "embedding_0d", embedding_dense_backward_one, _embedding_backward_takes, DispatchKey.Autograd))])


def install_roll(device_type):
    """Rolls by a multiple of the size on ``device_type`` as copies. Returns an undo. Below autograd (inference mode)
    they keep Habana's kernel, which hpu-ops does not override."""
    op = aten.roll.default
    return register_kernels(device_type, [(op, "Autograd{}", _kernel(
        op, "roll_zero", roll_unshifted, _roll_takes, DispatchKey.Autograd))])


def install_unfold(device_type):
    """``Tensor.unfold`` on ``device_type`` as ``as_strided``, at the backend key, below torch's own autograd kernel
    (whose backward is ``unfold_backward``). Habana's eager stack has its own kernel there, which is left alone, so
    only lazy mode takes it. Returns an undo."""
    op = aten.unfold.default

    def kernel(input, dimension, size, step):
        count("unfold")
        return unfold(input, dimension, size, step)

    return register_kernels(device_type, [(op, "{}", kernel)])


def install_unfold_backward(device_type):
    """``aten.unfold_backward`` (the gradient of ``Tensor.unfold``) on ``device_type`` as pads, reshapes and one sum
    (``unfold_backward`` above), at the backend key, below torch's autograd kernel (whose derivative is ``unfold``).
    No Habana stack registers a kernel there: every call ran on the CPU, eager and lazy (#43). Returns an undo."""
    op = aten.unfold_backward.default

    def kernel(grad_in, input_sizes, dim, size, step):
        count("unfold_backward")
        return unfold_backward(grad_in, input_sizes, dim, size, step)

    return register_kernels(device_type, [(op, "{}", kernel)])
