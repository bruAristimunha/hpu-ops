"""HPU composites for the linalg ops Gaudi lacks (``linalg_solve_triangular``, ``cholesky_solve``, ``linalg.inv``,
``det``, ``slogdet``) or runs on the CPU, built from matmuls that stay on the card and compile into the step graph.

Each is a ``torch.autograd.Function`` registered at ``Autograd{}``: its backward is torch's formula over the same
composites, so gradients match torch's. Its forward is also registered at the backend key, for calls below autograd
(inference mode, the tracers' decompositions). Real float32 matrices of static shape are taken; float64 on ``hpu`` is
not: Gaudi2 computes float64 at float32 precision (hpu-ops#16), so those inputs keep the native kernel at ``Autograd{}``
and run on the CPU below it, with one warning per op. Complex, empty, non-square and symbolic-shape inputs go to the
op's own kernel, which raises its usual error.

Composites contain no host sync: loops run over Python ints from static shapes.

On Gaudi2 these composites lose to Habana's CPU fallback (one launch per op), so with the TPC Gauss-Jordan kernel of
``hpu_ops.tpc`` loaded, float32 matrices of n <= ``tpc.MAXN`` go through it instead (solve, inv, det, slogdet and both
triangular solves). Without the kernel, or past ``MAXN``, the call on ``hpu`` is declined and Habana's fallback runs.
The composites stay the path of the CPU stand-in.
"""
import warnings

import torch
import torch.nn.functional as F
import torch.utils._pytree as pytree
from torch._C import DispatchKey

from . import tpc
from ._dispatch import count, register_kernels, static

aten = torch.ops.aten
FLOAT64_WARNING = ("hpu_ops: {name} of float64 HPU tensors runs on the CPU: Gaudi2 computes float64 at float32 "
                   "precision (bruAristimunha/hpu-ops#16).")
_WARNED = set()
# The Gauss-Jordan solver ``(A, B) -> (X, D, S)`` and triangular solver ``(A, B, upper, left, unitriangular) -> X``
# the Functions call for float32 n <= tpc.MAXN. None means ``tpc.gj_solve`` / ``tpc.tri_solve`` on HPU with the kernel
# ready; the CPU tests set them to ``tpc.gj_solve_reference`` / ``tpc.tri_solve_reference`` to exercise the wiring.
_GJ = None
_TRI = None


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


def _tpc_takes(device_type, takes, matrix_at=0):
    """``takes``, and on ``hpu`` also: n <= ``tpc.MAXN`` and the TPC kernel ready (path installed, extension built).
    Otherwise Habana's CPU fallback runs, which is faster than the composites on Gaudi2."""

    def tpc_takes(*args, **kwargs):
        if not takes(*args, **kwargs):
            return False
        return device_type != "hpu" or (args[matrix_at].shape[-1] <= tpc.MAXN and tpc.ready())

    return tpc_takes


def _use_tpc(A):
    """Whether ``A`` goes through the Gauss-Jordan kernel: n <= ``tpc.MAXN`` and, on HPU, float32 with the kernel
    ready; or, with the ``_GJ`` hook set (CPU tests), any dtype."""
    if A.shape[-1] > tpc.MAXN:
        return False
    return _GJ is not None or (A.device.type == "hpu" and A.dtype == torch.float32 and tpc.ready())


def _gj(A, B):
    """``(X, D, S)`` of the Gauss-Jordan solver, the batch dimensions of ``A`` and ``B`` broadcast. ``S[..., 0]`` is
    the sign of the permutation, ``S[..., 1]`` the info (the index follows the kernel's elimination order, can
    differ from LAPACK's, and is nonzero exactly when a pivot is exactly zero)."""
    batch = torch.broadcast_shapes(A.shape[:-2], B.shape[:-2])
    solver = tpc.gj_solve if _GJ is None else _GJ
    return solver(A.expand(*batch, *A.shape[-2:]), B.expand(*batch, *B.shape[-2:]))


def _gj_inv(A):
    """``(A^-1, D, S)``: the kernel with the identity as right-hand side, built in the kernel."""
    if _GJ is not None:
        return _gj(A, torch.eye(A.shape[-1], dtype=A.dtype, device=A.device))
    return tpc.gj_inv(A)


def _gj_solve(A, B, left=True):
    """``(X, D, S)`` for ``A X = B`` (``X A = B`` if not ``left``)."""
    if left:
        return _gj(A, B)
    X, D, S = _gj(A.mT, B.mT)
    return X.mT, D, S


def _use_tri(A):
    """Whether ``A`` goes through the triangular substitution kernel: n <= ``tpc.MAXN`` and, on HPU, float32 with the
    kernel ready; or, with the ``_TRI`` hook set (CPU tests), any dtype."""
    if A.shape[-1] > tpc.MAXN:
        return False
    return _TRI is not None or (A.device.type == "hpu" and A.dtype == torch.float32 and tpc.ready())


def _tri(A, B, upper=False, left=True, unitriangular=False):
    """Triangular substitution via ``tpc.tri_solve`` (or ``_TRI`` in CPU tests)."""
    solver = tpc.tri_solve if _TRI is None else _TRI
    return solver(A, B, upper, left, unitriangular)


def _gj_info(S, batch):
    """The ``info`` (int32; the index follows the kernel's elimination order, can differ from LAPACK's, and is nonzero
    exactly when a pivot is exactly zero) from the kernel's ``S``, shaped like the batch of ``A`` (``S`` carries the
    broadcast batch of ``A`` and ``B``; the slicing is static, no device sync)."""
    info = S[..., 1].to(torch.int32)
    if info.shape != batch:
        info = info[(0,) * (info.dim() - len(batch))]
        info = info[tuple(slice(0, 1) if a == 1 else slice(None) for a in batch)]
    return info


def _unused_factors(A):
    """The ``LU`` and ``pivots`` outputs the schemas require and the kernel does not compute: a zero scalar expanded
    to A's shape (not A's LU) and the identity permutation, expanded to the batch. Both are built on every call and
    own their storage: a tensor cached from one lazy graph breaks the next, and an alias of the input as an output
    does too. Only torch's own derivative reads them, and the Functions here replace it."""
    n = A.shape[-1]
    pivots = torch.arange(1, n + 1, dtype=torch.int32, device=A.device)
    return A.new_zeros(()).expand(A.shape), pivots.expand(*A.shape[:-1])


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
        if _use_tri(A):
            return _tri(A, B, upper, left, unitriangular)
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
        gB = SolveTriangular.apply(A.mH, grad, not upper, left, unitriangular)
        gA = None
        if ctx.needs_input_grad[0]:
            gA = -(gB @ X.mH) if left else -(X.mH @ gB)
            gA = torch.triu(gA, int(unitriangular)) if upper else torch.tril(gA, -int(unitriangular))
            gA = gA.sum_to_size(A.shape)
        return gA, gB.sum_to_size(ctx.B_shape), None, None, None


def install_solve_triangular(device_type):
    """``linalg.solve_triangular`` on ``device_type`` as a matmul with the block-recursive inverse. Returns an undo."""
    op = aten.linalg_solve_triangular.default
    base = _takes(device_type, "solve_triangular")

    def shapes_ok(A, B, upper=False, left=True, unitriangular=False):
        return base(A, B) and B.dim() >= 2 and B.shape[-2 if left else -1] == A.shape[-1]

    takes = _tpc_takes(device_type, shapes_ok)
    return register_kernels(device_type, _kernels(device_type, op, "solve_triangular", SolveTriangular, takes))


def cholesky_solve(B, L, upper=False):
    """``A^-1 B`` for ``A = L L^H`` (``U^H U`` if ``upper``): two matmuls with the explicit triangular inverse."""
    Li = tri_inverse(L, upper)
    return Li @ (Li.mH @ B) if upper else Li.mH @ (Li @ B)


class CholeskySolve(torch.autograd.Function):
    """``cholesky_solve`` with torch's backward (``cholesky_solve_backward``) over the composite."""

    @staticmethod
    def compute(B, L, upper):
        if _use_tri(L):
            Y = _tri(L.mH if upper else L, B, False, True, False)
            return _tri(L if upper else L.mH, Y, True, True, False)
        return cholesky_solve(B, L, upper)

    @staticmethod
    def forward(ctx, B, L, upper):
        X = CholeskySolve.compute(B, L, upper)
        ctx.save_for_backward(L, X)
        ctx.upper = upper
        ctx.B_shape = B.shape
        return X

    @staticmethod
    def backward(ctx, grad):
        L, X = ctx.saved_tensors
        gB = CholeskySolve.apply(grad, L, ctx.upper)
        gL = None
        if ctx.needs_input_grad[1]:
            common = gB @ X.mH
            common = common + common.mH
            gL = -(L @ common) if ctx.upper else -(common @ L)
            gL = gL.sum_to_size(L.shape)
        return gB.sum_to_size(ctx.B_shape), gL, None


def install_cholesky_solve(device_type):
    """``cholesky_solve`` on ``device_type`` as two matmuls with the block-recursive inverse. Returns an undo."""
    base = _takes(device_type, "cholesky_solve", matrix_at=1)

    def shapes_ok(B, L, upper=False):
        return base(B, L) and B.dim() >= 2 and B.shape[-2] == L.shape[-1]

    takes = _tpc_takes(device_type, shapes_ok, matrix_at=1)
    autograd = _kernels(device_type, aten.cholesky_solve.default, "cholesky_solve", CholeskySolve, takes)[0]
    # The other two entries are dropped: the helper is the backend key of the same op, so each key gets one kernel.
    backend = _kernels(device_type, aten._cholesky_solve_helper.default, "cholesky_solve", CholeskySolve, takes)[1]
    return register_kernels(device_type, [autograd, backend])


def lu(A):
    """``(LU, pivots)`` of ``A`` as ``torch.linalg.lu_factor`` returns them: L (unit diagonal) and U packed in one
    matrix, ``pivots`` int32 and 1-based. Partial pivoting, right-looking, ``n`` steps; every step works on full-size
    tensors under masks, so each step has one shape and no step indexes by a boolean mask. A zero pivot divides by 1,
    so singular input gives finite factors."""
    n = A.shape[-1]
    rows = torch.arange(n, device=A.device)
    LU, piv = A, []
    for k in range(n):
        col = LU[..., :, k].abs().masked_fill(rows < k, -1.0)  # rows above k never pivot
        p = col.argmax(-1)  # [...]
        swap = torch.where(rows == k, p[..., None], torch.where(rows == p[..., None], k, rows))
        LU = LU.gather(-2, swap[..., None].expand_as(LU))
        pivot = LU[..., k:k + 1, k:k + 1]
        l = torch.where((rows > k)[:, None], LU[..., :, k:k + 1] / torch.where(pivot == 0, 1.0, pivot), 0.0)
        LU = LU - l * torch.where((rows > k)[None, :], LU[..., k:k + 1, :], 0.0)
        LU = torch.where(((rows > k)[:, None]) & (rows == k)[None, :], l, LU)
        piv.append(p)
    return LU, (torch.stack(piv, -1) + 1).to(torch.int32)


def pivots_to_perm(pivots):
    """The row permutation ``perm`` (int64, ``[..., n]``) of ``lu``'s pivots, with ``A.gather(-2, perm[..., None]
    .expand_as(A)) == L @ U``: the swaps replayed in order on ``arange(n)``."""
    n = pivots.shape[-1]
    rows = torch.arange(n, device=pivots.device)
    # int64 gather does not compile on Gaudi2 (bridge 1.21): carry perm as float32 (exact below 2**24), cast at the end
    perm = rows.float().expand(pivots.shape)
    for k in range(n):
        p = (pivots[..., k] - 1).long()
        swap = torch.where(rows == k, p[..., None], torch.where(rows == p[..., None], k, rows))
        perm = perm.gather(-1, swap)
    return perm.long()


def lu_info(LU):
    """LAPACK's ``info`` (int32, ``[...]``): 1 + the index of the first zero on the diagonal of U, else 0."""
    return lu_info_of_diagonal(torch.diagonal(LU, dim1=-2, dim2=-1))


def lu_info_of_diagonal(diagonal):
    """``lu_info`` from the diagonal of U (the pivots) alone."""
    zero = diagonal == 0
    return torch.where(zero.any(-1), zero.int().argmax(-1) + 1, 0).to(torch.int32)


def lu_solve(LU, pivots, B, left=True):
    """``A^-1 B`` (``B A^-1`` if not ``left``) from ``lu``'s factors: two matmuls with the explicit triangular
    inverses, and a gather for the permutation."""
    perm = pivots_to_perm(pivots)
    Li = tri_inverse(LU, False, unitriangular=True)
    Ui = tri_inverse(LU, True)
    if left:
        shape = torch.broadcast_shapes(B.shape[:-2], perm.shape[:-1])
        index = perm.expand(*shape, perm.shape[-1])[..., None].expand(*shape, *B.shape[-2:])
        return Ui @ (Li @ B.expand(*shape, *B.shape[-2:]).gather(-2, index))
    Y = B @ Ui @ Li
    inverse = perm.argsort(-1)
    shape = torch.broadcast_shapes(Y.shape[:-2], inverse.shape[:-1])
    index = inverse.expand(*shape, inverse.shape[-1])[..., None, :].expand(*shape, *Y.shape[-2:])
    return Y.expand(*shape, *Y.shape[-2:]).gather(-1, index)


def _is_vector_rhs(A, B):
    """torch's rule: ``B`` is one vector, or a batch of vectors shaped like ``A``'s rows."""
    return B.dim() == 1 or (B.dim() == A.dim() - 1 and B.shape == A.shape[:-1])


class SolveEx(torch.autograd.Function):
    """``aten._linalg_solve_ex`` from the masked LU, with torch's backward (``linalg_solve_backward``) over the
    composite. ``LU``, ``pivots`` and ``info`` are not differentiable. ``LU`` and ``pivots`` are those of A (native's
    CPU kernel returns those of A^T); only torch's own derivative reads them, and this kernel replaces it. Through
    the TPC kernel ``LU`` and ``pivots`` are not computed: ``LU`` is a zero placeholder and ``pivots`` the
    identity permutation."""

    @staticmethod
    def compute(A, B, left, check_errors):
        vector = _is_vector_rhs(A, B)
        rhs = B.unsqueeze(-1) if vector else B
        if _use_tpc(A):
            result, _, S = _gj_solve(A, rhs, left)
            LU, pivots = _unused_factors(A)
            info = _gj_info(S, A.shape[:-2])
            if check_errors:
                torch.ops.aten._linalg_check_errors(info, "torch.linalg.solve_ex", is_matrix=A.dim() == 2)
        else:
            LU, pivots = lu(A)
            info = lu_info(LU)
            if check_errors:
                torch.ops.aten._linalg_check_errors(info, "torch.linalg.solve_ex", is_matrix=A.dim() == 2)
            result = lu_solve(LU, pivots, rhs, left)
        return (result.squeeze(-1) if vector else result), LU, pivots, info

    @staticmethod
    def forward(ctx, A, B, left, check_errors):
        X, LU, pivots, info = SolveEx.compute(A, B, left, check_errors)
        ctx.save_for_backward(A, X)
        ctx.left = left
        ctx.vector = _is_vector_rhs(A, B)
        ctx.B_shape = B.shape
        ctx.mark_non_differentiable(LU, pivots, info)
        return X, LU, pivots, info

    @staticmethod
    def backward(ctx, grad, *_):
        A, X = ctx.saved_tensors
        gB = SolveEx.apply(A.mH, grad, ctx.left, False)[0]
        gA = None
        if ctx.needs_input_grad[0]:
            g, x = (gB.unsqueeze(-1), X.unsqueeze(-1)) if ctx.vector else (gB, X)
            gA = (-(g @ x.mH) if ctx.left else -(x.mH @ g)).sum_to_size(A.shape)
        return gA, gB.sum_to_size(ctx.B_shape), None, None


def install_solve(device_type):
    """``linalg.solve`` (``_linalg_solve_ex``) on ``device_type`` as the masked LU and two triangular matmuls. Returns
    an undo."""
    op = aten._linalg_solve_ex.default
    base = _takes(device_type, "solve")

    def shapes_ok(A, B, left=True, check_errors=False):
        if not base(A, B):
            return False
        if _is_vector_rhs(A, B):
            return left and B.shape[-1] == A.shape[-1]
        return B.dim() >= 2 and B.shape[-2 if left else -1] == A.shape[-1]

    takes = _tpc_takes(device_type, shapes_ok)
    return register_kernels(device_type, _kernels(device_type, op, "solve", SolveEx, takes))


def _inverse(LU, pivots):
    """``A^-1`` from ``lu``'s factors: the solve against the identity."""
    n = LU.shape[-1]
    return lu_solve(LU, pivots, torch.eye(n, dtype=LU.dtype, device=LU.device))


def _parity(pivots):
    """The sign of the row permutation of ``lu``'s pivots: -1 to the number of swaps."""
    n = pivots.shape[-1]
    swaps = (pivots != torch.arange(1, n + 1, device=pivots.device, dtype=pivots.dtype)).sum(-1)
    return 1 - 2 * (swaps % 2)


class InvEx(torch.autograd.Function):
    """``aten.linalg_inv_ex`` from the masked LU, with torch's backward (``-X^H g X^H``) over the composite. ``info``
    is not differentiable."""

    @staticmethod
    def compute(A, check_errors):
        if _use_tpc(A):
            X, _, S = _gj_inv(A)
            info = _gj_info(S, A.shape[:-2])
        else:
            LU, pivots = lu(A)
            info = lu_info(LU)
            X = None
        if check_errors:
            torch.ops.aten._linalg_check_errors(info, "linalg.inv_ex", is_matrix=A.dim() == 2)
        return (_inverse(LU, pivots) if X is None else X), info

    @staticmethod
    def forward(ctx, A, check_errors):
        X, info = InvEx.compute(A, check_errors)
        ctx.save_for_backward(X)
        ctx.mark_non_differentiable(info)
        return X, info

    @staticmethod
    def backward(ctx, grad, *_):
        (X,) = ctx.saved_tensors
        return -(X.mH @ grad @ X.mH), None


class Det(torch.autograd.Function):
    """``aten._linalg_det`` as the signed product of U's diagonal, with torch's backward (``g det A^-H``), the inverse
    recomputed from A, so double backward works. A singular matrix has det 0 and a non-finite gradient. ``LU`` and
    ``pivots`` are not differentiable; they are those of A (native's CPU kernel returns those of A^T). Through the TPC
    kernel D and S come from the identity kernel (``_gj_inv``), X discarded."""

    @staticmethod
    def compute(A):
        if _use_tpc(A):
            _, D, S = _gj_inv(A)
            return S[..., 0] * D.prod(-1), *_unused_factors(A)
        LU, pivots = lu(A)
        diag = torch.diagonal(LU, dim1=-2, dim2=-1)
        return _parity(pivots).to(A.dtype) * diag.prod(-1), LU, pivots

    @staticmethod
    def forward(ctx, A):
        det, LU, pivots = Det.compute(A)
        ctx.save_for_backward(A, det)
        ctx.mark_non_differentiable(LU, pivots)
        return det, LU, pivots

    @staticmethod
    def backward(ctx, grad, *_):
        A, det = ctx.saved_tensors
        return (grad * det)[..., None, None] * InvEx.apply(A, False)[0].mH


class Slogdet(torch.autograd.Function):
    """``aten._linalg_slogdet`` as the sign and log of U's diagonal, with torch's backward (``g A^-H``) for
    ``logabsdet``, the inverse recomputed from A, so double backward works. ``sign``, ``LU`` and ``pivots`` are
    not differentiable; ``LU`` and ``pivots`` are those of A (native's CPU kernel returns those of A^T)."""

    @staticmethod
    def compute(A):
        if _use_tpc(A):
            _, D, S = _gj_inv(A)
            return (S[..., 0] * D.sign().prod(-1), D.abs().log().sum(-1), *_unused_factors(A))
        LU, pivots = lu(A)
        diag = torch.diagonal(LU, dim1=-2, dim2=-1)
        sign = _parity(pivots).to(A.dtype) * diag.sign().prod(-1)
        return sign, diag.abs().log().sum(-1), LU, pivots

    @staticmethod
    def forward(ctx, A):
        sign, logabsdet, LU, pivots = Slogdet.compute(A)
        ctx.save_for_backward(A)
        ctx.mark_non_differentiable(sign, LU, pivots)
        return sign, logabsdet, LU, pivots

    @staticmethod
    def backward(ctx, _sign, grad, *_):
        (A,) = ctx.saved_tensors
        return grad[..., None, None] * InvEx.apply(A, False)[0].mH


def install_inv_det(device_type):
    """``linalg.inv``/``inv_ex``, ``det`` and ``slogdet`` on ``device_type`` from the masked LU and two triangular
    matmuls. Returns an undo."""
    entries = []
    for op, name, function in ((aten.linalg_inv_ex.default, "inv", InvEx), (aten._linalg_det.default, "det", Det),
                               (aten._linalg_slogdet.default, "slogdet", Slogdet)):
        entries += _kernels(device_type, op, name, function, _tpc_takes(device_type, _takes(device_type, name)))
    return register_kernels(device_type, entries)
