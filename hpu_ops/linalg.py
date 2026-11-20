"""HPU composites for the linalg ops Gaudi lacks (``linalg_solve_triangular``) or runs on the CPU, built from matmuls
that stay on the card and compile into the step graph.

Each is a ``torch.autograd.Function`` registered at ``Autograd{}``: its backward is torch's formula over the same
composites, so gradients match torch's. Its forward is also registered at the backend key, for calls below autograd
(inference mode, the tracers' decompositions). Real float32 matrices of static shape are taken; float64 on ``hpu`` is
not: Gaudi2 computes float64 at float32 precision (hpu-ops#16), so those inputs keep the native kernel at ``Autograd{}``
and run on the CPU below it, with one warning per op. Complex, empty, non-square and symbolic-shape inputs go to the
op's own kernel, which raises its usual error.

Composites contain no host sync: loops run over Python ints from static shapes.
"""
import warnings

import torch
import torch.nn.functional as F
import torch.utils._pytree as pytree
from torch._C import DispatchKey

from ._dispatch import count, register_kernels, static

aten = torch.ops.aten
FLOAT64_WARNING = ("hpu_ops: {name} of float64 HPU tensors runs on the CPU: Gaudi2 computes float64 at float32 "
                   "precision (bruAristimunha/hpu-ops#16).")
_WARNED = set()


def _bind(op, args, kwargs):
    """The arguments of a call to ``op`` in schema order, defaults filled in."""
    rest = op._schema.arguments[len(args):]
    return tuple(args) + tuple(kwargs[a.name] if a.name in kwargs else a.default_value for a in rest)


def _takes(device_type, name, matrix_at=0):
    """A predicate on a call's arguments: the positional tensor at ``matrix_at`` is a square matrix and every tensor
    argument is a float32 (or float64, only off ``hpu``, with one warning per ``name``) tensor of static, non-empty
    shape."""

    def takes(*args, **kwargs):
        tensors = [x for x in (*args, *kwargs.values()) if isinstance(x, torch.Tensor)]
        if device_type == "hpu" and any(t.dtype == torch.float64 for t in tensors):
            if name not in _WARNED:
                _WARNED.add(name)
                warnings.warn(FLOAT64_WARNING.format(name=name), stacklevel=2)
            return False
        allowed = (torch.float32,) if device_type == "hpu" else (torch.float32, torch.float64)
        if not all(t.dtype in allowed and static(*t.shape) and t.numel() > 0 for t in tensors):
            return False
        matrix = args[matrix_at]
        return matrix.dim() >= 2 and matrix.shape[-1] == matrix.shape[-2]

    return takes


def _on_cpu(op, args, kwargs):
    """``op`` on CPU copies of the tensors; the outputs on the first tensor's device."""
    device = next(x.device for x in pytree.tree_leaves((args, kwargs)) if isinstance(x, torch.Tensor))
    to_cpu = lambda x: x.cpu() if isinstance(x, torch.Tensor) else x  # noqa: E731
    out = op(*pytree.tree_map(to_cpu, args), **pytree.tree_map(to_cpu, kwargs))
    return pytree.tree_map(lambda x: x.to(device) if isinstance(x, torch.Tensor) else x, out)


def _kernels(device_type, op, name, function, takes):
    """The ``(op, key, kernel)`` entries for ``register_kernels``: ``function`` for the calls ``takes`` accepts, else
    ``op``'s own kernel at autograd, and the CPU below it."""

    def autograd_kernel(*args, **kwargs):
        if takes(*args, **kwargs):
            count(name)
            return function.apply(*_bind(op, args, kwargs))
        return op._op_dk(DispatchKey.Autograd, *args, **kwargs)

    def backend_kernel(*args, **kwargs):  # below autograd: inference mode, tracers' decompositions
        if takes(*args, **kwargs):
            count(name)
            return function.compute(*_bind(op, args, kwargs))
        return _on_cpu(op, args, kwargs)

    return [(op, "Autograd{}", autograd_kernel), (op, "{}", backend_kernel)]


def tri_inverse(T, upper, unitriangular=False):
    """The inverse of the triangular part of ``T``: the 1x1 diagonal blocks' inverses merged pairwise, log2(n) levels
    of batched matmuls on ``T`` padded to a power of two. The other triangle is ignored; with ``unitriangular`` so is
    the diagonal."""
    # ponytail: explicit triangular inverse, substitution if ill-conditioned triangles need it
    if upper:
        return tri_inverse(T.mT, False, unitriangular).mT
    n = T.shape[-1]
    m = 1 << (n - 1).bit_length()  # pad to a power of two
    L = torch.tril(T, -1 if unitriangular else 0)
    L = F.pad(L, (0, m - n, 0, m - n))
    ones = torch.ones(m, dtype=T.dtype, device=T.device)
    # unit diagonal where implied or padded
    L = L + torch.diag_embed(ones if unitriangular else F.pad(ones[:n] * 0, (0, m - n), value=1.0))
    # [..., m, 1, 1]: inverses of the 1x1 diagonal blocks
    X = (1.0 / torch.diagonal(L, dim1=-2, dim2=-1)).unsqueeze(-1).unsqueeze(-1)
    s = 1
    while s < m:  # blocks of size s -> 2s
        k = m // (2 * s)
        blocks = torch.diagonal(L.unflatten(-2, (k, 2 * s)).unflatten(-1, (k, 2 * s)), dim1=-4, dim2=-2).movedim(-1, -3)
        C = blocks[..., s:, :s]  # [..., k, s, s]
        Xa, Xd = X[..., 0::2, :, :], X[..., 1::2, :, :]
        lower = -(Xd @ C @ Xa)
        top = torch.cat([Xa, torch.zeros_like(Xa)], -1)
        X = torch.cat([top, torch.cat([lower, Xd], -1)], -2)  # [..., k, 2s, 2s]
        s *= 2
    return X[..., 0, :n, :n]


def solve_triangular(A, B, upper, left=True, unitriangular=False):
    """``A^-1 B`` (``B A^-1`` if not ``left``) for triangular ``A``, as a matmul with its explicit inverse."""
    X = tri_inverse(A, upper, unitriangular)
    return X @ B if left else B @ X


class SolveTriangular(torch.autograd.Function):
    """``linalg.solve_triangular`` with torch's backward (``linalg_solve_triangular_backward``) over the composite."""

    @staticmethod
    def compute(A, B, upper, left, unitriangular):
        return solve_triangular(A, B, upper, left, unitriangular)

    @staticmethod
    def forward(ctx, A, B, upper, left, unitriangular):
        X = SolveTriangular.compute(A, B, upper, left, unitriangular)
        ctx.save_for_backward(A, X)
        ctx.options = (upper, left, unitriangular)
        ctx.B_shape = B.shape
        return X

    @staticmethod
    def backward(ctx, grad):
        A, X = ctx.saved_tensors
        upper, left, unitriangular = ctx.options
        gB = solve_triangular(A.mH, grad, not upper, left, unitriangular)
        gA = None
        if ctx.needs_input_grad[0]:
            gA = -(gB @ X.mH) if left else -(X.mH @ gB)
            gA = torch.triu(gA, int(unitriangular)) if upper else torch.tril(gA, -int(unitriangular))
            gA = gA.sum_to_size(A.shape)
        return gA, gB.sum_to_size(ctx.B_shape), None, None, None


def install_solve_triangular(device_type):
    """``linalg.solve_triangular`` on ``device_type`` as a matmul with the block-recursive inverse. Returns an undo."""
    op = aten.linalg_solve_triangular.default
    return register_kernels(device_type, _kernels(device_type, op, "solve_triangular", SolveTriangular,
                                                  _takes(device_type, "solve_triangular")))
