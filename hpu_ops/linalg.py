"""HPU composites for the linalg ops Gaudi lacks (``linalg_solve_triangular``, ``cholesky_solve``, ``linalg.inv``,
``det``, ``slogdet``) or runs on the CPU, built from matmuls that stay on the card and compile into the step graph.

Each is a ``torch.autograd.Function`` registered at ``Autograd{}``: its backward is torch's formula over the same
composites, so gradients match torch's. Its forward is also registered at the backend key, for calls below autograd
(inference mode, the tracers' decompositions), except for solve_ex, det and slogdet, whose backend call reaches Habana's
own kernel (its real LU factors feed torch's derivative when a compiled graph keeps it). Real float32 matrices of static shape are taken; float64 on ``hpu`` is
not: Gaudi2 computes float64 at float32 precision (hpu-ops#16), so those inputs keep the native kernel at ``Autograd{}``
and run on the CPU below it, with one warning per op. Complex, empty, non-square and symbolic-shape inputs go to the
op's own kernel, which raises its usual error.

Composites contain no host sync: loops run over Python ints from static shapes.

On Gaudi2 these composites were measured 3-25x slower per call than Habana's CPU fallback (one launch per op), so
with the TPC Gauss-Jordan kernel of ``hpu_ops.tpc`` loaded, float32 matrices of n <= ``tpc.MAXN`` go through it
instead (solve, inv, det, slogdet), those of ``tpc.MAXN`` < n <= ``tpc.LARGE_MAXN`` through its in-place inverse
kernel (``solve`` a matmul with the inverse, which its backward reuses), and the triangular solves
(``solve_triangular``, ``cholesky_solve``) through its substitution kernel (``tpc.tri_solve``): no pivoting, so no
accuracy is lost on ill-conditioned triangles, and ``cholesky_solve`` is two substitutions, never L L^H. Without the
kernel, or past ``MAXN`` (``LARGE_MAXN`` for solve, inv, det and slogdet), the call on ``hpu`` is declined and runs
through hpu-ops' own CPU route (``_on_cpu``), which is not Habana's fallback. The composites stay the path of the
CPU stand-in.
"""
import math
import warnings

import torch
import torch.nn.functional as F
import torch.utils._pytree as pytree
from torch._C import DispatchKey

from . import tpc
from ._dispatch import count, register_paired, static

aten = torch.ops.aten
FLOAT64_WARNING = ("hpu_ops: {name} of float64 HPU tensors runs on the CPU: Gaudi2 computes float64 at float32 "
                   "precision (bruAristimunha/hpu-ops#16).")
_WARNED = set()
# The Gauss-Jordan solver ``(A, B) -> (X, D, S)`` and triangular solver ``(A, B, upper, left, unitriangular) -> X``
# the Functions call for float32 n <= tpc.MAXN. None means ``tpc.gj_solve`` / ``tpc.tri_solve`` on HPU with the kernel
# ready; the CPU tests set them to ``tpc.gj_solve_reference`` / ``tpc.tri_solve_reference`` to exercise the wiring.
_GJ = None
_TRI = None
# CPU tests: True sends n > tpc.MAXN through ``tpc.gj_inv`` / ``tpc.large_factors`` (with ``tpc._large_op`` replaced by
# the kernel's float32 replica), so the Python side of the in-place kernel path runs on the CPU too.
_LARGE_ON_CPU = False


def _bind(op, args, kwargs):
    """The arguments of a call to ``op`` in schema order, defaults filled in."""
    rest = op._schema.arguments[len(args):]
    return tuple(args) + tuple(kwargs[a.name] if a.name in kwargs else a.default_value for a in rest)


def _transformed():
    """Whether a functorch transform (vmap, grad, jvp, jacrev, ...) or forward-mode AD is active. The Functions here
    define no ``setup_context`` and no jvp, so those calls keep the op's own kernel."""
    forward_ad = getattr(torch.autograd.forward_ad, "_current_level", -1) >= 0
    return torch._C._functorch.peek_interpreter_stack() is not None or forward_ad


def _takes(device_type, name, matrix_at=0):
    """A predicate on a call's arguments: no functorch transform or forward-mode AD is active, the positional tensor
    at ``matrix_at`` is a square matrix and every tensor argument is a float32 (or float64, only off ``hpu``, with one
    warning per ``name``) tensor of static, non-empty shape."""

    def takes(*args, **kwargs):
        if _transformed():
            return False
        tensors = [x for x in (*args, *kwargs.values()) if isinstance(x, torch.Tensor)]
        if device_type == "hpu" and any(t.dtype == torch.float64 for t in tensors):
            if name not in _WARNED:
                _WARNED.add(name)
                warnings.warn(FLOAT64_WARNING.format(name=name), stacklevel=2)
            return False
        allowed = (torch.float32,) if device_type == "hpu" else (torch.float32, torch.float64)
        if not all(t.dtype in allowed and static(*t.shape) and t.numel() > 0 for t in tensors):
            return False
        if len({t.device for t in tensors}) > 1:  # torch's own kernel raises its device error
            return False
        matrix = args[matrix_at]
        return matrix.dim() >= 2 and matrix.shape[-1] == matrix.shape[-2]

    return takes


def _tpc_takes(device_type, takes, matrix_at=0, large=False):
    """``takes``, and on ``hpu`` also: n <= ``tpc.MAXN``, or with ``large`` (solve, inv, det, slogdet) n <=
    ``tpc.LARGE_MAXN`` with a batch the in-place kernel pays for (``tpc.large_pays``), and the TPC kernel ready (path
    installed, Gaudi2, extension built). Otherwise the call is declined and takes hpu-ops' own CPU route
    (``_on_cpu``), which beats the composites on Gaudi2."""

    def tpc_takes(*args, **kwargs):
        if not takes(*args, **kwargs):
            return False
        if device_type != "hpu":
            return True
        A = args[matrix_at]
        n = A.shape[-1]
        batch = A.shape[:-2]
        if large and len(args) > 1 and isinstance(args[1], torch.Tensor):  # solve: B's batch broadcasts A's
            B = args[1]
            batch = torch.broadcast_shapes(batch, B.shape[:-1] if _is_vector_rhs(A, B) else B.shape[:-2])
        fits = n <= tpc.MAXN or (large and tpc.large_pays(n, math.prod(batch)))
        return fits and tpc.ready()

    return tpc_takes


def _use_tpc(A):
    """Whether ``A`` goes through the Gauss-Jordan kernels: n <= ``tpc.LARGE_MAXN`` (n <= ``tpc.MAXN``: rows in local
    memory; past it the in-place inverse kernel, ``tpc.gj_solve`` multiplying by it) and, on HPU, float32 with the
    kernel ready; or, with the ``_GJ`` hook set (CPU tests), any dtype."""
    n = A.shape[-1]
    if n > tpc.LARGE_MAXN:
        return False
    if _LARGE_ON_CPU and n > tpc.MAXN:
        return True
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
    if _GJ is not None and not (_LARGE_ON_CPU and A.shape[-1] > tpc.MAXN):
        return _gj(A, torch.eye(A.shape[-1], dtype=A.dtype, device=A.device))
    return tpc.gj_inv(A)


def _gj_factors(A):
    """``(D, S)`` alone, for det and slogdet: past ``tpc.MAXN`` the in-place kernel without forming the inverse."""
    if A.shape[-1] > tpc.MAXN and (_GJ is None or _LARGE_ON_CPU):
        return tpc.large_factors(A)
    return _gj_inv(A)[1:]


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
    return info.contiguous()  # a slice of the broadcast batch is strided; _linalg_check_errors asserts contiguity


def _unused_factors(A):
    """The ``LU`` and ``pivots`` outputs the schemas require and the kernel does not compute: a zero scalar expanded
    to A's shape (not A's LU) and the identity permutation, expanded to the batch. Both are built on every call and
    own their storage: a tensor cached from one lazy graph breaks the next, and an alias of the input as an output
    does too. Only torch's own derivative reads them, and the Functions here replace it."""
    n = A.shape[-1]
    pivots = torch.arange(1, n + 1, dtype=torch.int32, device=A.device)
    return A.new_zeros(()).expand(A.shape), pivots.expand(*A.shape[:-1])


def _device(args, kwargs):
    """The one device of the tensors in ``args`` and ``kwargs``; torch's error when they are on several (the CPU copy
    would otherwise hide a mismatch torch rejects)."""
    devices = list(dict.fromkeys(x.device for x in pytree.tree_leaves((args, kwargs)) if isinstance(x, torch.Tensor)))
    if len(devices) > 1:
        raise RuntimeError(f"Expected all tensors to be on the same device, but found at least two devices, "
                           f"{devices[0]} and {devices[1]}!")
    return devices[0]


def _on_cpu(op, args, kwargs):
    """``op`` on CPU copies of the tensors; the outputs on the tensors' device (all on one device, else torch's
    error)."""
    device = _device(args, kwargs)
    to_cpu = lambda x: x.cpu() if isinstance(x, torch.Tensor) else x  # noqa: E731
    out = op(*pytree.tree_map(to_cpu, args), **pytree.tree_map(to_cpu, kwargs))
    return pytree.tree_map(lambda x: x.to(device) if isinstance(x, torch.Tensor) else x, out)


def _out_on_cpu(op, args, kwargs, outs):
    """The out= overload ``op`` on CPU copies of the inputs, with CPU destinations of the dtype, shape and strides of
    ``outs`` (a dict of the out= arguments), so torch's own checks run (dtype, resize warning, layout); each result is
    then written into its destination with the layout torch gave it. Returns the destinations (one, or a tuple)."""
    _device(args, {**kwargs, **outs})
    to_cpu = lambda x: x.cpu() if isinstance(x, torch.Tensor) else x  # noqa: E731
    cpu_outs = {k: torch.empty_strided(v.shape, v.stride(), dtype=v.dtype) for k, v in outs.items()}
    op(*pytree.tree_map(to_cpu, args), **pytree.tree_map(to_cpu, kwargs), **cpu_outs)
    for k, v in outs.items():
        if v.shape != cpu_outs[k].shape or v.stride() != cpu_outs[k].stride():
            v.resize_(cpu_outs[k].numel()).as_strided_(cpu_outs[k].shape, cpu_outs[k].stride())
        v.copy_(cpu_outs[k])
    values = tuple(outs.values())
    return values[0] if len(values) == 1 else values


def _kernels(device_type, op, name, function, takes, backend=True):
    """The ``(op, key, kernel)`` entries for ``register_kernels``: ``function`` for the calls ``takes`` accepts, else
    ``op``'s own kernel at autograd, and the CPU below it. With ``backend=False`` only the autograd entry is made:
    ops whose torch derivative reads the LU factors (solve_ex, det, slogdet) must reach Habana's own kernel below
    autograd, as our placeholder factors would turn a traced native backward into NaN (symbolic shapes, recompiles)."""

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

    entries = [(op, "Autograd{}", autograd_kernel), (op, "{}", backend_kernel)]
    return entries if backend else entries[:1]


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
    """``linalg.solve_triangular`` with torch's backward (``linalg_solve_triangular_backward``). On HPU, float32 with
    n <= ``tpc.MAXN`` runs the TPC substitution kernel (``tpc.tri_solve``); otherwise the composite (CPU stand-in)."""

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
    """``linalg.solve_triangular`` on ``device_type``: on HPU, float32 with n <= ``tpc.MAXN`` through the TPC
    substitution kernel (``tpc.tri_solve``), other calls through hpu-ops' CPU route; on the CPU stand-in, the
    composite (a matmul with the block-recursive inverse). Returns an undo."""
    op = aten.linalg_solve_triangular.default
    base = _takes(device_type, "solve_triangular")

    def shapes_ok(A, B, upper=False, left=True, unitriangular=False):
        return base(A, B) and B.dim() >= 2 and B.shape[-2 if left else -1] == A.shape[-1]

    takes = _tpc_takes(device_type, shapes_ok)
    groups = [_kernels(device_type, op, "solve_triangular", SolveTriangular, takes),
              _solve_triangular_out_kernels(device_type, takes), _triangular_solve_kernels(device_type, takes)]
    return register_paired(device_type, groups)


RESIZE_WARNING = ("An output with one or more elements was resized since it had shape {old}, which does not match the "
                  "required output shape {new}. This behavior is deprecated, and in a future PyTorch release outputs "
                  "will not be resized unless they have zero elements. You can explicitly reuse an out tensor t by "
                  "resizing it, inplace, to zero elements with t.resize_(0).")


def _into(out, X):
    """``X`` written into ``out``; returns ``out``. As torch's linalg out= ops: a destination of another dtype raises,
    one of another shape is resized (with torch's warning unless it was empty) to the column-major layout of the
    matrices (strides ``(..., 1, rows)``), one of the right shape keeps its layout."""
    if out.dtype != X.dtype:
        raise RuntimeError(f"Expected out tensor to have dtype {X.dtype}, but got {out.dtype} instead")
    if out.shape != X.shape:
        if out.numel() != 0:
            warnings.warn(RESIZE_WARNING.format(old=list(out.shape), new=list(X.shape)), stacklevel=3)
        out.resize_(X.mT.shape).transpose_(-2, -1)
    return out.copy_(X)


def _solve_triangular_out_kernels(device_type, takes):
    """``linalg.solve_triangular(..., out=)`` (``aten.linalg_solve_triangular.out``): the same route as the functional
    op, written into ``out``. out= ops do not differentiate, so a call with an input that requires grad keeps the
    op's own autograd kernel (which raises, as in torch)."""
    op = aten.linalg_solve_triangular.out

    def ours(A, B, *, upper, left=True, unitriangular=False, out):
        return out.dtype == A.dtype and out.device == A.device and takes(A, B, upper, left, unitriangular)

    def autograd_kernel(*args, **kwargs):
        # out= ops do not differentiate: an input or the destination that requires grad raises in torch's own kernel
        grads = torch.is_grad_enabled() and any(t.requires_grad for t in _bind(op, args, kwargs)
                                                if isinstance(t, torch.Tensor))
        if not grads and ours(*args, **kwargs):
            count("solve_triangular")
            A, B, upper, left, unitriangular, out = _bind(op, args, kwargs)
            return _into(out, SolveTriangular.compute(A, B, upper, left, unitriangular))
        return op._op_dk(DispatchKey.Autograd, *args, **kwargs)

    def backend_kernel(*args, **kwargs):
        A, B, upper, left, unitriangular, out = _bind(op, args, kwargs)
        if ours(*args, **kwargs):
            count("solve_triangular")
            return _into(out, SolveTriangular.compute(A, B, upper, left, unitriangular))
        return _out_on_cpu(op, (A, B), dict(upper=upper, left=left, unitriangular=unitriangular), dict(out=out))

    return [(op, "Autograd{}", autograd_kernel), (op, "{}", backend_kernel)]


class ClonedCoefficient(torch.autograd.Function):
    """The ``cloned_coefficient`` of the legacy ``triangular_solve``: ``A`` broadcast to the batch and copied. As in
    torch it is an output of the op, so it requires grad when ``B`` does, and its gradient to ``B`` is zero."""

    @staticmethod
    def forward(ctx, A, B, batch):
        ctx.A_shape, ctx.B_shape = A.shape, B.shape
        return A.expand(*batch, *A.shape[-2:]).clone()

    @staticmethod
    def backward(ctx, grad):
        gA = grad.sum_to_size(ctx.A_shape) if ctx.needs_input_grad[0] else None
        gB = grad.new_zeros(ctx.B_shape) if ctx.needs_input_grad[1] else None
        return gA, gB, None


def _triangular_solve_kernels(device_type, takes):
    """The legacy ``torch.triangular_solve(B, A, upper, transpose, unitriangular)`` (``aten.triangular_solve``, and
    its out= overload ``.X``): ``solution`` through the same route as ``linalg.solve_triangular`` (``transpose``
    solves with ``A^T``), and ``cloned_coefficient`` a copy of ``A`` broadcast to the batch, as torch returns it."""
    op = aten.triangular_solve.default
    op_out = aten.triangular_solve.X

    def outputs(differentiable, B, A, upper, transpose, unitriangular):
        solve = SolveTriangular.apply if differentiable else SolveTriangular.compute
        X = solve(A.mT if transpose else A, B, upper != transpose, True, unitriangular)
        batch = torch.broadcast_shapes(A.shape[:-2], B.shape[:-2])
        if differentiable:
            return X, ClonedCoefficient.apply(A, B, batch)
        return X, A.expand(*batch, *A.shape[-2:]).clone()

    def ours(B, A, upper, transpose, unitriangular):
        return takes(A.mT if transpose else A, B, upper != transpose, True, unitriangular)

    def autograd_kernel(*args, **kwargs):
        bound = _bind(op, args, kwargs)
        if ours(*bound):
            count("solve_triangular")
            return outputs(True, *bound)
        return op._op_dk(DispatchKey.Autograd, *args, **kwargs)

    def backend_kernel(*args, **kwargs):
        bound = _bind(op, args, kwargs)
        if ours(*bound):
            count("solve_triangular")
            return outputs(False, *bound)
        return _on_cpu(op, args, kwargs)

    def ours_out(B, A, upper, transpose, unitriangular, X, M):
        same = all(t.dtype == A.dtype and t.device == A.device for t in (X, M))
        return same and ours(B, A, upper, transpose, unitriangular)

    def write(bound):
        B, A, upper, transpose, unitriangular, X, M = bound
        count("solve_triangular")
        solution, coefficient = outputs(False, B, A, upper, transpose, unitriangular)
        return _into(X, solution), _into(M, coefficient)

    def autograd_kernel_out(*args, **kwargs):
        bound = _bind(op_out, args, kwargs)
        grads = torch.is_grad_enabled() and any(t.requires_grad for t in bound if isinstance(t, torch.Tensor))
        if not grads and ours_out(*bound):
            return write(bound)
        return op_out._op_dk(DispatchKey.Autograd, *args, **kwargs)

    def backend_kernel_out(*args, **kwargs):
        bound = _bind(op_out, args, kwargs)
        if ours_out(*bound):
            return write(bound)
        B, A, upper, transpose, unitriangular, X, M = bound
        return _out_on_cpu(op_out, (B, A, upper, transpose, unitriangular), {}, dict(X=X, M=M))

    return [(op, "Autograd{}", autograd_kernel), (op, "{}", backend_kernel),
            (op_out, "Autograd{}", autograd_kernel_out), (op_out, "{}", backend_kernel_out)]


def cholesky_solve(B, L, upper=False):
    """``A^-1 B`` for ``A = L L^H`` (``U^H U`` if ``upper``): two matmuls with the explicit triangular inverse."""
    Li = tri_inverse(L, upper)
    return Li @ (Li.mH @ B) if upper else Li.mH @ (Li @ B)


class CholeskySolve(torch.autograd.Function):
    """``cholesky_solve`` with torch's backward (``cholesky_solve_backward``). On HPU, float32 with n <= ``tpc.MAXN``
    runs two TPC substitutions (``tpc.tri_solve``, L then L^H); otherwise the composite (CPU stand-in)."""

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
    """``cholesky_solve`` on ``device_type``: on HPU, float32 with n <= ``tpc.MAXN`` as two TPC substitutions
    (``tpc.tri_solve``), other calls through hpu-ops' CPU route; on the CPU stand-in, the composite (two matmuls with
    the block-recursive inverse). Returns an undo."""
    base = _takes(device_type, "cholesky_solve", matrix_at=1)

    def shapes_ok(B, L, upper=False):
        return base(B, L) and B.dim() >= 2 and B.shape[-2] == L.shape[-1]

    takes = _tpc_takes(device_type, shapes_ok, matrix_at=1)
    autograd = _kernels(device_type, aten.cholesky_solve.default, "cholesky_solve", CholeskySolve, takes)[0]
    # The other two entries are dropped: the helper is the backend key of the same op, so each key gets one kernel.
    backend = _kernels(device_type, aten._cholesky_solve_helper.default, "cholesky_solve", CholeskySolve, takes)[1]
    return register_paired(device_type, [[autograd, backend]])


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


def _inv_solve(A, B, Ainv, left):
    """``A^-1 B`` (``B A^-1`` if not ``left``) from the inverse, with one residual step ``X += A^-1 (B - A X)``: on the
    in-place kernel's float32 replica (n = 220) the forward error then matches LAPACK's from condition 1 to 1e7 (the
    product with the inverse alone was up to 1.5x worse). A non-finite correction (Inf in A) is dropped."""
    X = Ainv @ B if left else B @ Ainv
    C = Ainv @ (B - A @ X) if left else (B - X @ A) @ Ainv
    return X + torch.where(torch.isfinite(C), C, 0.0)


class _InvSolve(torch.autograd.Function):
    """``A^-1 B`` (``B A^-1`` if not ``left``) as a matmul with ``Ainv``, the inverse of ``A`` the forward already
    computed (n > ``tpc.MAXN``: the in-place kernel's), so the backward of a large solve launches no kernel. ``Ainv``
    is a constant here; the derivative in ``A`` is torch's solve formula, so double backward works."""

    @staticmethod
    def forward(ctx, A, B, Ainv, left):
        X = _inv_solve(A, B, Ainv, left)
        ctx.save_for_backward(A, X, Ainv)
        ctx.left = left
        ctx.B_shape = B.shape
        return X

    @staticmethod
    def backward(ctx, grad):
        A, X, Ainv = ctx.saved_tensors
        gB = _InvSolve.apply(A.mH, grad, Ainv.mH, ctx.left)
        gA = None
        if ctx.needs_input_grad[0]:
            gA = (-(gB @ X.mH) if ctx.left else -(X.mH @ gB)).sum_to_size(A.shape)
        return gA, gB.sum_to_size(ctx.B_shape), None, None


class SolveEx(torch.autograd.Function):
    """``aten._linalg_solve_ex`` from the masked LU, with torch's backward (``linalg_solve_backward``) over the
    composite. ``LU``, ``pivots`` and ``info`` are not differentiable. ``LU`` and ``pivots`` are those of A (native's
    CPU kernel returns those of A^T); only torch's own derivative reads them, and this kernel replaces it. Through
    the TPC kernel ``LU`` and ``pivots`` are not computed: ``LU`` is a zero placeholder and ``pivots`` the
    identity permutation. Past ``tpc.MAXN`` the kernel computes ``A^-1`` (in place, partial pivoting) and the solve
    is a matmul with it; the inverse is kept for the backward, which is then matmuls only."""

    @staticmethod
    def compute(A, B, left, check_errors):
        return SolveEx.compute_with_inverse(A, B, left, check_errors)[:4]

    @staticmethod
    def compute_with_inverse(A, B, left, check_errors):
        """``compute``'s outputs and ``A^-1`` (None unless n > ``tpc.MAXN`` on the kernel path)."""
        vector = _is_vector_rhs(A, B)
        rhs = B.unsqueeze(-1) if vector else B
        Ainv = None
        if _use_tpc(A):
            if A.shape[-1] > tpc.MAXN:
                Ainv, _, S = _gj_inv(A)
                result = _inv_solve(A, rhs, Ainv, left)
            else:
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
        return (result.squeeze(-1) if vector else result), LU, pivots, info, Ainv

    @staticmethod
    def forward(ctx, A, B, left, check_errors):
        X, LU, pivots, info, Ainv = SolveEx.compute_with_inverse(A, B, left, check_errors)
        ctx.save_for_backward(A, X, Ainv)
        ctx.left = left
        ctx.vector = _is_vector_rhs(A, B)
        ctx.B_shape = B.shape
        ctx.mark_non_differentiable(LU, pivots, info)
        return X, LU, pivots, info

    @staticmethod
    def backward(ctx, grad, *_):
        A, X, Ainv = ctx.saved_tensors
        if Ainv is None:
            gB = SolveEx.apply(A.mH, grad, ctx.left, False)[0]
        else:
            g = grad.unsqueeze(-1) if ctx.vector else grad
            gB = _InvSolve.apply(A.mH, g, Ainv.mH, ctx.left)
            gB = gB.squeeze(-1) if ctx.vector else gB
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

    takes = _tpc_takes(device_type, shapes_ok, large=True)
    return register_paired(device_type, [_kernels(device_type, op, "solve", SolveEx, takes, backend=False)])


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
    recomputed from A, so double backward works. A singular matrix has det 0, and on Gaudi2 its gradient
    equals torch's (measured), also with ``create_graph``. ``LU`` and
    ``pivots`` are not differentiable; they are those of A (native's CPU kernel returns those of A^T). Through the TPC
    kernel D and S come from the identity kernel (``_gj_inv``), X discarded; a zero pivot leaves later pivots finite.
    In grad mode (double backward) a singular A (|det| < 100 eps) takes torch's SVD formula instead of ``det A^-H``,
    which is 0 there (one host sync per such backward)."""

    @staticmethod
    def compute(A):
        if _use_tpc(A):
            D, S = _gj_factors(A)
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
        cofactor = (grad * det)[..., None, None] * InvEx.apply(A, False)[0].mH
        if torch.is_grad_enabled():
            # torch's derivative in grad mode (double backward): the SVD formula for |det| < 100 eps, whose cofactor
            # is right on a singular matrix, where det * A^-H is 0. The check is a host sync, in grad mode only.
            singular = det.abs() < 100 * torch.finfo(A.dtype).eps
            if bool(singular.any()):
                return torch.where(singular[..., None, None], _det_grad_singular(A, grad), cofactor)
        return cofactor


def _det_grad_singular(A, grad):
    """torch's ``linalg_det_backward`` for singular ``A`` in grad mode: with ``A = U S V^H``, ``det(U) det(V^H) U
    diag(g prod_{j != i} s_j) V^H``, the products without a division (exclusive cumulative products). Gaudi has no
    SVD kernel, so on HPU the SVD runs on a CPU copy (differentiable ``.cpu()``/``.to()``): a host round trip, only
    for a singular matrix differentiated with ``create_graph``."""
    if A.device.type == "hpu":
        U, S, Vh = (t.to(A.device) for t in torch.linalg.svd(A.cpu()))
    else:
        U, S, Vh = torch.linalg.svd(A)
    alpha = torch.linalg.det(U) * torch.linalg.det(Vh)
    ones = torch.ones_like(S[..., :1])
    before = torch.cumprod(torch.cat([ones, S[..., :-1]], -1), -1)
    after = torch.cumprod(torch.cat([ones, S.flip(-1)[..., :-1]], -1), -1).flip(-1)
    D = (grad * alpha)[..., None] * before * after
    return (U * D[..., None, :]) @ Vh


class Slogdet(torch.autograd.Function):
    """``aten._linalg_slogdet`` as the sign and log of U's diagonal, with torch's backward (``g A^-H``) for
    ``logabsdet``, the inverse recomputed from A, so double backward works. ``sign`` is differentiable as in torch
    (its gradient is none for real input: ``sign.backward()`` leaves ``A.grad`` unset); ``LU`` and ``pivots`` are
    not; ``LU`` and ``pivots`` are those of A (native's CPU kernel returns those of A^T). On Gaudi2
    a singular matrix gives ``logabsdet = -inf`` as in torch, but a finite gradient where torch's is non-finite."""

    @staticmethod
    def compute(A):
        if _use_tpc(A):
            D, S = _gj_factors(A)
            return (S[..., 0] * D.sign().prod(-1), D.abs().log().sum(-1), *_unused_factors(A))
        LU, pivots = lu(A)
        diag = torch.diagonal(LU, dim1=-2, dim2=-1)
        sign = _parity(pivots).to(A.dtype) * diag.sign().prod(-1)
        return sign, diag.abs().log().sum(-1), LU, pivots

    @staticmethod
    def forward(ctx, A):
        sign, logabsdet, LU, pivots = Slogdet.compute(A)
        ctx.save_for_backward(A)
        ctx.mark_non_differentiable(LU, pivots)
        ctx.set_materialize_grads(False)
        return sign, logabsdet, LU, pivots

    @staticmethod
    def backward(ctx, _sign, grad, *_):
        if grad is None:  # only sign was used: real sign has no gradient
            return None
        (A,) = ctx.saved_tensors
        return grad[..., None, None] * InvEx.apply(A, False)[0].mH


def install_inv_det(device_type):
    """``linalg.inv``/``inv_ex``, ``det`` and ``slogdet`` on ``device_type`` from the masked LU and two triangular
    matmuls. Returns an undo."""
    groups = []
    for op, name, function in ((aten.linalg_inv_ex.default, "inv", InvEx), (aten._linalg_det.default, "det", Det),
                               (aten._linalg_slogdet.default, "slogdet", Slogdet)):
        takes = _tpc_takes(device_type, _takes(device_type, name), large=True)
        groups.append(_kernels(device_type, op, name, function, takes, backend=name == "inv"))
    return register_paired(device_type, groups)
