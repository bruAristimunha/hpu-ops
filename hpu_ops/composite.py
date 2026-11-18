"""HPU composites for ops whose Gaudi kernel fails under ``torch.compile`` (``glu``), is missing (``renorm``) or is
wrong (``embedding_dense_backward`` of a 0-d index, ``roll`` of a strided input by a multiple of the size).

Each is registered at ``AutogradHPU``, so eager calls and traced graphs both use it and autograd differentiates the
composite; ``renorm`` is also registered at ``HPU`` for calls below autograd (inference mode). Cases a composite does
not take (odd ``glu`` halves, invalid ``renorm`` arguments) go to the op's own kernel, which raises its usual error;
so do embedding gradients of indices with dimensions, which Gaudi gets right. The composites run with autocast off and
return the input dtype, as the native ops do.
"""
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
    """A roll whose every shift is a multiple of its dim's size, as the copy it is. Gaudi's eager kernel reads a
    strided input of such a roll wrongly (EMG2QwertyNet: a stack's gradient slice rolled back by 0)."""
    return input.clone()


def _roll_takes(input, shifts, dims=()):
    sizes = [input.numel()] if len(dims) == 0 else [input.shape[d] for d in dims]
    return static(*sizes, *shifts) and all(size == 0 or shift % size == 0 for size, shift in zip(sizes, shifts))


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
