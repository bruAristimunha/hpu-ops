"""HPU composites for ops whose Gaudi kernel fails under ``torch.compile`` (``glu``), is missing (``renorm``,
``index_put`` / ``index_put_``), is wrong (``embedding_dense_backward`` of a 0-d index, ``roll`` of a strided input by
a multiple of the size), or triggers a lazy-mode DSD sections-validation failure (``index.Tensor`` + mid-backward
``index_put_`` CPU fallback in ``GBWAtt``).

Each is registered at ``AutogradHPU``, so eager calls and traced graphs both use it and autograd differentiates the
composite; ``renorm``, ``index_put_`` and ``index_put`` are also registered at ``HPU`` for calls below autograd
(inference mode and inside ``autograd.Function.backward``). Cases a composite does not take (odd ``glu`` halves,
invalid ``renorm`` arguments, non-boolean or non-scalar ``index_put``) go to the op's own kernel, which raises its
usual error; so do embedding gradients of indices with dimensions, which Gaudi gets right. The composites run with
autocast off and return the input dtype, as the native ops do.
"""
import torch
from torch._C import DispatchKey
from torch._subclasses.fake_tensor import is_fake

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


def _bool_mask_expanded(self, indices, values):
    mask = indices[0]
    if mask.dim() < self.dim():
        mask = mask.reshape(*mask.shape, *([1] * (self.dim() - mask.dim())))
    return mask, values.to(dtype=self.dtype, device=self.device)


def bool_index_put_(self, indices, values, accumulate=False):
    """Boolean-mask scalar assignment via ``torch.where`` + in-place copy/add. Habana has no HPU kernel for
    ``aten::_index_put_impl_``, so ``s[s <= threshold] = 0`` inside ``autograd.Function.backward()`` falls back to the
    CPU and flushes the partial lazy graph mid-backward (triggering ``synStatus 30`` in ``GBWAtt``)."""
    with torch.autocast(self.device.type, enabled=False):
        mask, val = _bool_mask_expanded(self, indices, values)
        # accumulate adds only where the mask is set: adding +0.0 elsewhere would turn -0.0 into +0.0
        return self.copy_(torch.where(mask, self + val if accumulate else val, self))


def bool_index_put(self, indices, values, accumulate=False):
    """Out-of-place boolean-mask scalar assignment via ``torch.where``."""
    with torch.autocast(self.device.type, enabled=False):
        mask, val = _bool_mask_expanded(self, indices, values)
        return torch.where(mask, self + val if accumulate else val, self)


def _bool_index_put_takes(self, indices, values, accumulate=False):
    if len(indices) != 1 or indices[0] is None:
        return False
    mask = indices[0]
    if mask.dtype != torch.bool or mask.device != self.device:
        return False
    if values.device.type not in (self.device.type, "cpu"):
        return False
    if values.device.type == self.device.type and values.device != self.device:
        return False
    if self.dim() == 0 or mask.dim() > self.dim() or self.shape[:mask.dim()] != mask.shape:
        return False
    if values.dim() != 0:
        return False
    if accumulate and values.dtype != self.dtype:
        return False
    if values.is_complex() and not (self.is_complex() or self.dtype == torch.bool):
        return False
    return True


def _bool_index_put_inplace_takes(self, indices, values, accumulate=False):
    # an expanded self (several elements on one memory location: overlap YES = 1) is written by ATen's kernel but
    # refused by copy_; non-dense views (TOO_HARD = 2, e.g. x[:, 1:]) are fine for copy_
    return _bool_index_put_takes(self, indices, values, accumulate) and torch._debug_has_internal_overlap(self) != 1


def _concrete(t):
    """True for a real tensor outside any tracer: its values can be read on the host."""
    return not is_fake(t) and torch._guards.detect_fake_mode() is None and not torch._is_functional_tensor(t)


def index_2d(self, indices):
    """Trailing 2-D integer indexing ``X[..., ix0, ix1]`` (``Vech``'s ``sym_to_upper``) as a flat ``index_select``
    over the last two dimensions. Avoids Habana's DSD ``index.Tensor`` backward recipe and eager ``index.Tensor_out``
    fallback.

    Out-of-range indices: host indices raise torch's ``IndexError`` (checked on the host, no card sync). Indices on
    the card cannot be checked without a sync; they are sent past the end of the flat ``index_select``, which on
    Gaudi2 (bridge 1.21.4, eager and lazy) returns 0.0 without raising, the same value Habana's native
    ``index.Tensor`` returns for them. They never read a neighbouring element."""
    ix0, ix1 = indices[-2], indices[-1]
    # host indices (Vech's triu_indices) stay on the host for the arithmetic and the bounds check
    on = self.device if self.device in (ix0.device, ix1.device) else ix0.device
    ix0, ix1 = torch.broadcast_tensors(ix0.to(on), ix1.to(on))
    h, w = self.shape[-2], self.shape[-1]
    inside = (ix0 >= -h) & (ix0 < h) & (ix1 >= -w) & (ix1 < w)
    if on.type == "cpu" and _concrete(inside) and not bool(inside.all()):
        bad0 = ~((ix0 >= -h) & (ix0 < h))
        # ATen numbers the dimension among the advanced indices (0 for ix0, 1 for ix1), not in ``self``
        dim, size, idx = (0, h, ix0[bad0]) if bad0.any() else (1, w, ix1[~((ix1 >= -w) & (ix1 < w))])
        raise IndexError(f"index {int(idx.reshape(-1)[0])} is out of bounds for dimension {dim} with size {size}")
    i0 = torch.where(ix0 < 0, ix0 + h, ix0)
    i1 = torch.where(ix1 < 0, ix1 + w, ix1)
    # an index out of its own dimension must not land on a neighbouring row: send it past the end instead
    flat = torch.where(inside, i0 * w + i1, h * w).reshape(-1).to(self.device)
    return self.flatten(-2).index_select(-1, flat).reshape((*self.shape[:-2], *ix0.shape))


def _index_2d_takes(self, indices):
    if self.dim() < 2 or len(indices) != self.dim():
        return False
    # a plain 2-D gather with card indices is one native op; the rewrite's ~10 small ops are 6-18x slower there
    # (Gaudi2 eager, 64x64, 512 indices: 0.28 vs 5.07 ms). Keep it for leading dims (Vech) or host indices.
    if self.dim() == 2 and all(idx is not None and idx.device == self.device for idx in indices):
        return False
    if self.shape[-2] <= 0 or self.shape[-1] <= 0:
        return False
    if any(idx is not None for idx in indices[:-2]):
        return False
    ix0, ix1 = indices[-2], indices[-1]
    if ix0 is None or ix1 is None:
        return False
    if ix0.dtype not in (torch.int32, torch.int64) or ix1.dtype not in (torch.int32, torch.int64):
        return False
    for idx in (ix0, ix1):
        if idx.device != self.device and idx.device.type != "cpu":
            return False
    try:
        torch.broadcast_shapes(ix0.shape, ix1.shape)
    except RuntimeError:
        return False
    return True


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


def install_index_put(device_type):
    """Boolean-mask scalar ``index_put_`` and ``index_put`` on ``device_type`` via ``torch.where``. Registered at both
    ``Autograd{}`` and ``{}`` so calls inside ``autograd.Function.backward()`` and ``inference_mode()`` avoid CPU
    fallback too. Returns an undo."""
    op_inplace = aten.index_put_.default
    op_out = aten.index_put.default
    return register_kernels(device_type, [
        (op_inplace, "Autograd{}", _kernel(op_inplace, "bool_index_put_", bool_index_put_,
                                           _bool_index_put_inplace_takes, DispatchKey.Autograd)),
        (op_inplace, "{}", _kernel(op_inplace, "bool_index_put_", bool_index_put_,
                                   _bool_index_put_inplace_takes, DispatchKey.CompositeExplicitAutograd)),
        (op_out, "Autograd{}", _kernel(op_out, "bool_index_put", bool_index_put,
                                       _bool_index_put_takes, DispatchKey.Autograd)),
        (op_out, "{}", _kernel(op_out, "bool_index_put", bool_index_put,
                               _bool_index_put_takes, DispatchKey.CompositeExplicitAutograd)),
    ])


def install_index_2d(device_type):
    """Trailing 2-D integer indexing ``aten.index.Tensor`` on ``device_type`` via flat ``index_select``. Returns an
    undo."""
    op = aten.index.Tensor
    return register_kernels(device_type, [(op, "Autograd{}", _kernel(
        op, "index_2d", index_2d, _index_2d_takes, DispatchKey.Autograd))])
