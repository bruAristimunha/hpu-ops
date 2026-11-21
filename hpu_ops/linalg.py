"""HPU composites for the linalg ops Gaudi lacks (``linalg_solve_triangular``, ``cholesky_solve``, ``linalg.inv``,
``det``, ``slogdet``) or runs on the CPU, built from matmuls that stay on the card and compile into the step graph.

Each is a ``torch.autograd.Function`` registered at ``Autograd{}``: its backward is torch's formula over the same
composites, so gradients match torch's. Its forward is also registered at the backend key, for calls below autograd
(inference mode, the tracers' decompositions), except for solve_ex, det and slogdet, whose backend call reaches Habana's
own kernel (its real LU factors feed torch's derivative when a compiled graph keeps it). Real float32 matrices of static shape are taken; float64 on ``hpu`` is
not: a float64 tensor on Gaudi2 holds float32 values (hpu-ops#16, ``tests/probe_float64.py``), so those inputs keep the native kernel at ``Autograd{}``
and run on the CPU below it, with one warning per op. Complex, empty, non-square and symbolic-shape inputs go to the
op's own kernel, which raises its usual error.

Composites contain no host sync: loops run over Python ints from static shapes.

On Gaudi2 these composites were measured 3-25x slower per call than Habana's CPU fallback (one launch per op), so
with the TPC Gauss-Jordan kernel of ``hpu_ops.tpc`` loaded, float32 matrices of n <= ``tpc.MAXN`` go through it
instead (solve, inv, det, slogdet), larger ones through its in-place inverse kernel (``solve`` a matmul with the
inverse, which its backward reuses; det and slogdet keep the forward's inverse for theirs): one square call, or the
blocked route (tall panels with partial pivoting over the whole height, matmul updates) past ``tpc.LARGE_MAXN`` or
for a batch too small to fill the card, in eager and lazy mode. The triangular solves (``solve_triangular``,
``cholesky_solve``) go through its substitution kernel (``tpc.tri_solve``, blocked past ``MAXN``): no pivoting, so no
accuracy is lost on ill-conditioned triangles, and ``cholesky_solve`` is two substitutions, never L L^H. Without the
kernel the call on ``hpu`` is declined and runs through hpu-ops' own CPU route (``_on_cpu``), which is not Habana's
fallback. The composites stay the path of the CPU stand-in. ``HPU_OPS_FLOAT64=card`` runs float64 HPU tensors on the
card too, in float32 precision (``_float64_kernel``).
"""
import math
import os
import warnings

import torch
import torch.nn.functional as F
import torch.utils._pytree as pytree
from torch._C import DispatchKey

from . import _dispatch, tpc
from ._dispatch import count, static

aten = torch.ops.aten
FLOAT64_WARNING = ("hpu_ops: {name} of float64 HPU tensors runs on the CPU: Gaudi2 stores float64 as float32, so "
                   "the input and the result hold float32 precision (bruAristimunha/hpu-ops#16).")
# Opt-in (read when a call is made): ``HPU_OPS_FLOAT64=card`` runs the linalg ops of float64 HPU tensors on the card,
# through the same float32 kernels, the inputs cast to float32 (exact: the card holds float32 values in them) and the
# floating outputs cast back to float64. Unset, float64 keeps the default above (the CPU, one warning per op).
FLOAT64_ENV = "HPU_OPS_FLOAT64"
FLOAT64_CARD_WARNING = ("hpu_ops: HPU_OPS_FLOAT64=card: float64 linalg on HPU runs on the card in float32 precision "
                        "(about 7 significant digits, not float64's 16; Gaudi2 stores float64 as float32 anyway, "
                        "bruAristimunha/hpu-ops#16).")
_WARNED = set()
# The Gauss-Jordan solver ``(A, B) -> (X, D, S)`` and triangular solver ``(A, B, upper, left, unitriangular) -> X``
# the Functions call for float32 n <= tpc.MAXN. None means ``tpc.gj_solve`` / ``tpc.tri_solve`` on HPU with the kernel
# ready; the CPU tests set them to ``tpc.gj_solve_reference`` / ``tpc.tri_solve_reference`` to exercise the wiring.
_GJ = None
_TRI = None
# CPU tests: True sends n > tpc.MAXN through ``tpc.gj_inv`` / ``tpc.large_factors`` (with ``tpc._large_op`` replaced by
# the kernel's float32 replica), so the Python side of the in-place kernel path runs on the CPU too.
_LARGE_ON_CPU = False
# The Jacobi SVD hook ``(A, compute_uv) -> (U, S, Vh)`` for CPU tests; None means ``tpc.jacobi_svd`` /
# ``tpc.jacobi_svdvals`` on HPU, or ``tpc.svd_reference`` on CPU when unset.
_SVD = None
# The Householder product hook ``(A, tau) -> Q`` for CPU tests; None means ``tpc.householder_product`` on HPU,
# or ``tpc.householder_product_reference`` on CPU when unset.
_HOUSEHOLDER = None
# The matrix_exp hooks for CPU tests; None means ``tpc.matrix_exp`` / ``tpc.matrix_exp_frechet`` on HPU,
# or Padé-13 composite on CPU when unset.
_MATRIX_EXP = None
_MATRIX_EXP_FRECHET = None
# The LU hooks for CPU tests: ``_LU`` ``A -> (LU, pivots, perm, info)`` and ``_LU_PERM`` ``pivots -> perm``. None
# means ``tpc.lu_factor`` / ``tpc.lu_perm`` on HPU with the kernel ready, else the composites (``lu``,
# ``pivots_to_perm``).
_LU = None
_LU_PERM = None


def _float64_on_card(device_type, args, kwargs):
    """Whether this call takes the float64 opt-in: ``HPU_OPS_FLOAT64=card``, on ``hpu``, a float64 tensor among the
    arguments. Warns once per process, naming the precision."""
    if device_type != "hpu" or os.environ.get(FLOAT64_ENV) != "card":
        return False
    if not any(isinstance(x, torch.Tensor) and x.dtype == torch.float64 for x in pytree.tree_leaves((args, kwargs))):
        return False
    if "float64-card" not in _WARNED:
        _WARNED.add("float64-card")
        warnings.warn(FLOAT64_CARD_WARNING, stacklevel=4)
    return True


def _writes(op):
    """Whether ``op`` writes into an argument (an out= overload or an in-place op): those keep the default."""
    return any(a.alias_info is not None and a.alias_info.is_write for a in op._schema.arguments)


def _float64_kernel(device_type, op, kernel):
    """``kernel`` with the float64 opt-in: float64 tensor arguments cast to float32 (differentiable casts, so autograd
    sees a float64 leaf), the kernel run on them, its float32 outputs cast back to float64."""
    if _writes(op):
        return kernel

    def wrapped(*args, **kwargs):
        if not _float64_on_card(device_type, args, kwargs):
            return kernel(*args, **kwargs)
        down = lambda x: x.to(torch.float32) if isinstance(x, torch.Tensor) and x.dtype == torch.float64 else x  # noqa: E731
        out = kernel(*pytree.tree_map(down, args), **pytree.tree_map(down, kwargs))
        up = lambda y: y.to(torch.float64) if isinstance(y, torch.Tensor) and y.dtype == torch.float32 else y  # noqa: E731
        return pytree.tree_map(up, out)

    return wrapped


def register_kernels(device_type, entries):
    """``_dispatch.register_kernels`` with each kernel taking the float64 opt-in (``_float64_kernel``)."""
    return _dispatch.register_kernels(device_type, [(op, key, _float64_kernel(device_type, op, k))
                                                    for op, key, k in entries])


def register_paired(device_type, groups):
    """``_dispatch.register_paired`` with each kernel taking the float64 opt-in (``_float64_kernel``)."""
    return _dispatch.register_paired(device_type, [[(op, key, _float64_kernel(device_type, op, k)) for op, key, k in g]
                                                   for g in groups])


def _is_fake(t):
    from torch._subclasses.fake_tensor import FakeTensor, is_fake
    from torch._subclasses.functional_tensor import FunctionalTensor

    if isinstance(t, FakeTensor) or is_fake(t) or torch.compiler.is_compiling():
        return True
    return isinstance(t, FunctionalTensor) and is_fake(t.elem)


def _tracing(t):
    """Whether a tracer sees ``t``: ``torch.compile`` (Dynamo, or AOT autograd's fake and functional tensors, where
    ``torch.compiler.is_compiling()`` is False), a fake mode, or ``make_fx`` (a proxy mode, also on real tensors)."""
    if _is_fake(t) or torch._guards.detect_fake_mode() is not None:
        return True
    return torch._C._get_dispatch_mode(torch._C._TorchDispatchModeKey.PROXY) is not None


def _bind(op, args, kwargs):
    """The arguments of a call to ``op`` in schema order, defaults filled in."""
    rest = op._schema.arguments[len(args):]
    return tuple(args) + tuple(kwargs[a.name] if a.name in kwargs else a.default_value for a in rest)


def _takes_tensors(device_type, name, tensors):
    if device_type == "hpu" and any(t.dtype == torch.float64 for t in tensors):
        if name not in _WARNED:
            _WARNED.add(name)
            warnings.warn(FLOAT64_WARNING.format(name=name), stacklevel=2)
        return False
    allowed = (torch.float32,) if device_type == "hpu" else (torch.float32, torch.float64)
    return all(t.dtype in allowed and static(*t.shape) and t.numel() > 0 for t in tensors)


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
        if not _takes_tensors(device_type, name, tensors):
            return False
        if len({t.device for t in tensors}) > 1:  # torch's own kernel raises its device error
            return False
        matrix = args[matrix_at]
        return matrix.dim() >= 2 and matrix.shape[-1] == matrix.shape[-2]

    return takes


def _tpc_takes(device_type, takes, matrix_at=0, maxn=None, large=False):
    """``takes``, and on ``hpu`` also: n <= ``maxn`` (default ``tpc.MAXN``), or with ``large`` (solve, inv, det,
    slogdet) any n (``tpc.large_pays``: every size and batch, eager and lazy, not while tracing), and the TPC kernel
    ready (path installed, Gaudi2, extension built). Otherwise the call is declined and takes hpu-ops' own CPU route
    (``_on_cpu``), which beats the composites on Gaudi2."""

    def tpc_takes(*args, **kwargs):
        if not takes(*args, **kwargs):
            return False
        if device_type != "hpu":
            return True
        A = args[matrix_at]
        n = A.shape[-1]
        if n <= (tpc.MAXN if maxn is None else maxn):
            return tpc.ready()
        batch = A.shape[:-2]
        if large and len(args) > 1 and isinstance(args[1], torch.Tensor):  # solve: B's batch broadcasts A's
            B = args[1]
            batch = torch.broadcast_shapes(batch, B.shape[:-1] if _is_vector_rhs(A, B) else B.shape[:-2])
        # not traced (tpc.large_pays): a traced graph (compile, AOT, make_fx) keeps main's route (codex review #19)
        return large and not _tracing(A) and tpc.large_pays(n, math.prod(batch)) and tpc.ready()

    return tpc_takes


def _use_tpc(A):
    """Whether ``A`` goes through the Gauss-Jordan kernels (n <= ``tpc.MAXN``: rows in local memory; past it the
    in-place inverse kernels, ``tpc.gj_solve`` multiplying by the inverse): on HPU, float32 with the kernel ready; or,
    with the ``_GJ`` hook set (CPU tests), any dtype."""
    n = A.shape[-1]
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


def _gj_inv(A, sign=True):
    """``(A^-1, D, S)``: the kernel with the identity as right-hand side, built in the kernel (``sign=False``: the
    in-place kernel's ``S[..., 0]`` is not computed)."""
    if _GJ is not None and not (_LARGE_ON_CPU and A.shape[-1] > tpc.MAXN):
        return _gj(A, torch.eye(A.shape[-1], dtype=A.dtype, device=A.device))
    return tpc.gj_inv(A, sign=sign)


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
    """Whether ``A`` goes through the triangular substitution kernel (past ``tpc.MAXN`` blocked, ``tpc.tri_solve``):
    on HPU, float32 with the kernel ready; or, with the ``_TRI`` hook set (CPU tests), any dtype."""
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


def _traced_with_grad(args, kwargs):
    """Whether a declined call is being traced (compile, AOT, make_fx) with its autograd recorded: grad mode on, a
    real floating tensor argument that requires grad, all on one device, no functorch transform."""
    tensors = [x for x in pytree.tree_leaves((args, kwargs)) if isinstance(x, torch.Tensor)]
    if not tensors or _transformed() or not torch.is_grad_enabled() or not any(t.requires_grad for t in tensors):
        return False
    if len({t.device for t in tensors}) > 1 or not all(t.is_floating_point() for t in tensors):
        return False
    return any(_tracing(t) for t in tensors)


def _kernels(device_type, op, name, function, takes, backend=True, traced=None):
    """The ``(op, key, kernel)`` entries for ``register_kernels``: ``function`` for the calls ``takes`` accepts, else
    ``op``'s own kernel at autograd, and the CPU below it. With ``backend=False`` only the autograd entry is made:
    ops whose torch derivative reads the LU factors (solve_ex, det, slogdet) must reach Habana's own kernel below
    autograd, as our placeholder factors would turn a traced native backward into NaN (symbolic shapes, recompiles).
    ``traced`` (those ops) takes the calls a tracer sees declined with grad recorded: the graph's node is still the
    op, but at run time it reaches the autograd kernel again with real tensors, where ``takes`` may accept it (the
    large route is eager-only, a symbolic shape is static) and return placeholder factors. torch's traced derivative
    would read them (NaN gradients of STaRNet(64)'s n = 96 Cayley solve compiled); ``traced``'s backward reads ``A``
    and the outputs only, so either run-time route gives torch's gradient."""

    def autograd_kernel(*args, **kwargs):
        if takes(*args, **kwargs):
            count(name)
            return function.apply(*_bind(op, args, kwargs))
        if traced is not None and _traced_with_grad(args, kwargs):
            return traced.apply(op, *_bind(op, args, kwargs))
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

    takes = _tpc_takes(device_type, shapes_ok, maxn=math.inf)  # past tpc.MAXN the blocked substitution
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

    takes = _tpc_takes(device_type, shapes_ok, matrix_at=1, maxn=math.inf)  # past tpc.MAXN blocked substitutions
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
    A = A.contiguous()  # a strided view here gave wrong residuals under torch.compile on Gaudi2 (see SolveEx)
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
                # one contiguous copy for the kernel and the residual: under torch.compile on Gaudi2 the residual's
                # A @ X with A a strided view of a graph input beside the kernel's copy gave 7e-2 error at n = 150
                Ac = A.contiguous()
                Ainv, _, S = _gj_inv(Ac, sign=False)
                result = _inv_solve(Ac, rhs, Ainv, left)
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
    return register_paired(device_type, [_kernels(device_type, op, "solve", SolveEx, takes, backend=False,
                                                         traced=_TracedSolveEx)])


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
            X, _, S = _gj_inv(A, sign=False)
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
        X = None
        if ctx.needs_input_grad[0] and _use_tpc(A):
            # the kernel's inverse comes with its pivots: kept for the backward, which then launches no kernel
            X, D, S = _gj_inv(A)
            det, LU, pivots = S[..., 0] * D.prod(-1), *_unused_factors(A)
        else:
            det, LU, pivots = Det.compute(A)
        ctx.save_for_backward(A, det, X)
        ctx.mark_non_differentiable(LU, pivots)
        return det, LU, pivots

    @staticmethod
    def backward(ctx, grad, *_):
        A, det, X = ctx.saved_tensors
        return _det_backward(A, det, grad, _saved_or_inverse(X))


def _saved_or_inverse(X):
    """The inverse for a backward: ``X`` saved by the forward (a constant: first-order backward only), or, in grad mode
    (double backward) or without it, ``InvEx`` on the matrix, differentiable."""
    if X is not None and not torch.is_grad_enabled():
        return lambda M: X
    return lambda M: InvEx.apply(M, False)[0]


def _det_backward(A, det, grad, inverse):
    """``g det A^-H`` with ``inverse(A)``; in grad mode (double backward) a singular A takes torch's SVD formula."""
    cofactor = (grad * det)[..., None, None] * inverse(A).mH
    if torch.is_grad_enabled():
        # torch's derivative in grad mode (double backward): the SVD formula for |det| < 100 eps, whose cofactor
        # is right on a singular matrix, where det * A^-H is 0. The check is a host sync, in grad mode only.
        singular = det.abs() < 100 * torch.finfo(A.dtype).eps
        if _tracing(A):
            # no host read in a traced graph: both formulas, selected per matrix; the cofactor of a singular matrix
            # is taken on the identity instead, so its non-finite inverse does not reach where's derivative
            mask = singular[..., None, None]
            safe = torch.where(mask, torch.eye(A.shape[-1], dtype=A.dtype, device=A.device), A)
            cofactor = (grad * det)[..., None, None] * inverse(safe).mH
            return torch.where(mask, _det_grad_singular(A, grad), cofactor)
        if bool(singular.any()):
            return torch.where(singular[..., None, None], _det_grad_singular(A, grad), cofactor)
    return cofactor


def _det_grad_singular(A, grad):
    """torch's ``linalg_det_backward`` for singular ``A`` in grad mode: with ``A = U S V^H``, ``det(U) det(V^H) U
    diag(g prod_{j != i} s_j) V^H``, the products without a division (exclusive cumulative products). The SVD is
    the public op, so on HPU it is hpu-ops' own (``install_svd``: the Jacobi kernel, differentiable), on the card."""
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
        X = None
        if ctx.needs_input_grad[0] and _use_tpc(A):
            X, D, S = _gj_inv(A)  # kept for the backward, as in Det
            sign, logabsdet, LU, pivots = (S[..., 0] * D.sign().prod(-1), D.abs().log().sum(-1), *_unused_factors(A))
        else:
            sign, logabsdet, LU, pivots = Slogdet.compute(A)
        ctx.save_for_backward(A, X)
        ctx.mark_non_differentiable(LU, pivots)
        ctx.set_materialize_grads(False)
        return sign, logabsdet, LU, pivots

    @staticmethod
    def backward(ctx, _sign, grad, *_):
        if grad is None:  # only sign was used: real sign has no gradient
            return None
        A, X = ctx.saved_tensors
        return grad[..., None, None] * _saved_or_inverse(X)(A).mH


def _below_autograd(op, *args, **kwargs):
    """``op`` at the key below autograd: the native kernel (Habana's on ``hpu``), a plain node to a tracer."""
    with torch._C._AutoDispatchBelowAutograd():
        return op(*args, **kwargs)


def _plain_inverse(A):
    """``A^-1`` through the public op: in a traced backward, whatever route ``linalg.inv_ex`` takes."""
    return torch.linalg.inv_ex(A)[0]


class _TracedSolveEx(torch.autograd.Function):
    """A traced ``_linalg_solve_ex`` that ``takes`` declined (``_kernels``' ``traced``): the op's own forward, torch's
    backward from ``A`` and ``X`` (``gB = solve(A^H, g)``, ``gA = -gB X^H``), never from ``LU`` and ``pivots``,
    which the node may return as placeholders at run time."""

    @staticmethod
    def forward(ctx, op, A, B, left, check_errors):
        X, LU, pivots, info = _below_autograd(op, A, B, left=left, check_errors=check_errors)
        ctx.save_for_backward(A, X)
        ctx.left = left
        ctx.vector = _is_vector_rhs(A, B)
        ctx.B_shape = B.shape
        ctx.mark_non_differentiable(LU, pivots, info)
        return X, LU, pivots, info

    @staticmethod
    def backward(ctx, grad, *_):
        A, X = ctx.saved_tensors
        gB = torch.linalg.solve(A.mH, grad, left=ctx.left)
        gA = None
        if ctx.needs_input_grad[1]:
            g, x = (gB.unsqueeze(-1), X.unsqueeze(-1)) if ctx.vector else (gB, X)
            gA = (-(g @ x.mH) if ctx.left else -(x.mH @ g)).sum_to_size(A.shape)
        return None, gA, gB.sum_to_size(ctx.B_shape), None, None


class _TracedDet(torch.autograd.Function):
    """A traced ``_linalg_det`` that ``takes`` declined: the op's forward, ``Det``'s backward through the public
    inverse, never from ``LU`` and ``pivots`` (see ``_TracedSolveEx``)."""

    @staticmethod
    def forward(ctx, op, A):
        det, LU, pivots = _below_autograd(op, A)
        ctx.save_for_backward(A, det)
        ctx.mark_non_differentiable(LU, pivots)
        return det, LU, pivots

    @staticmethod
    def backward(ctx, grad, *_):
        A, det = ctx.saved_tensors
        return None, _det_backward(A, det, grad, _plain_inverse)


class _TracedSlogdet(torch.autograd.Function):
    """A traced ``_linalg_slogdet`` that ``takes`` declined: the op's forward, ``Slogdet``'s backward
    (``g A^-H``) through the public inverse, never from ``LU`` and ``pivots`` (see ``_TracedSolveEx``)."""

    @staticmethod
    def forward(ctx, op, A):
        sign, logabsdet, LU, pivots = _below_autograd(op, A)
        ctx.save_for_backward(A)
        ctx.mark_non_differentiable(LU, pivots)
        ctx.set_materialize_grads(False)
        return sign, logabsdet, LU, pivots

    @staticmethod
    def backward(ctx, _sign, grad, *_):
        if grad is None:  # only sign was used: real sign has no gradient
            return None, None
        (A,) = ctx.saved_tensors
        return None, grad[..., None, None] * _plain_inverse(A).mH


def install_inv_det(device_type):
    """``linalg.inv``/``inv_ex``, ``det`` and ``slogdet`` on ``device_type`` from the masked LU and two triangular
    matmuls. Returns an undo."""
    groups = []
    for op, name, function in ((aten.linalg_inv_ex.default, "inv", InvEx), (aten._linalg_det.default, "det", Det),
                               (aten._linalg_slogdet.default, "slogdet", Slogdet)):
        takes = _tpc_takes(device_type, _takes(device_type, name), large=True)
        traced = {"det": _TracedDet, "slogdet": _TracedSlogdet}.get(name)
        groups.append(_kernels(device_type, op, name, function, takes, backend=name == "inv", traced=traced))
    return register_paired(device_type, groups)


# ---- SVD family (#9): svd, svdvals, pinv, lstsq, matrix_norm(2/-2/'nuc'), matrix_rank


def _tol_tensor_ok(tol, batch_shape):
    if not (tol.is_floating_point() and static(*tol.shape) and tol.numel() > 0):
        return False
    if tol.dim() > len(batch_shape):
        return False
    try:
        torch.broadcast_shapes(batch_shape, tol.shape)
    except RuntimeError:
        return False
    return True


def _svd_takes(device_type, name="svd"):
    def takes(A, full_matrices=False, compute_uv=True, driver=None, **_out):
        if _transformed():  # no jvp / vmap rule in these Functions: torch's own kernel
            return False
        if driver is not None or not _takes_tensors(device_type, name, [A]):
            return False
        if A.dim() < 2 or A.shape[-2] > tpc.SMALL_MAXN or A.shape[-1] > tpc.SMALL_MAXN:
            return False
        return device_type != "hpu" or tpc.ready()

    return takes


def _pinv_takes(device_type, name="pinv"):
    def takes(A, *, atol=None, rtol=None, hermitian=False):
        if _transformed():  # no jvp / vmap rule in these Functions: torch's own kernel
            return False
        if hermitian or not _takes_tensors(device_type, name, [A]):
            return False
        if A.dim() < 2 or A.shape[-2] > tpc.SMALL_MAXN or A.shape[-1] > tpc.SMALL_MAXN:
            return False
        if device_type == "hpu" and not tpc.ready():
            return False
        for tol in (atol, rtol):
            if tol is not None and not _tol_tensor_ok(tol, A.shape[:-2]):
                return False
            if tol is not None and tol.requires_grad and torch.is_grad_enabled():
                return False  # torch's own kernel raises: atol and rtol are not differentiable (codex review, minor)
        return True

    return takes


def _lstsq_takes(device_type, name="lstsq"):
    def takes(A, B, rcond=None, *, driver=None):
        if _transformed():  # no jvp / vmap rule in these Functions: torch's own kernel
            return False
        if driver not in (None, "gels", "gelsy", "gelsd", "gelss"):
            return False
        if rcond is not None and (isinstance(rcond, bool) or not isinstance(rcond, (int, float))):
            return False
        if not _takes_tensors(device_type, name, [A, B]) or A.dtype != B.dtype:
            return False
        if A.dim() < 2 or A.shape[-2] > tpc.SMALL_MAXN or A.shape[-1] > tpc.SMALL_MAXN:
            return False
        if device_type == "hpu" and not tpc.ready():
            return False
        if B.dim() == A.dim() - 1:
            return B.shape == A.shape[:-1]
        if B.dim() != A.dim() or B.shape[-2] != A.shape[-2]:
            return False
        try:
            torch.broadcast_shapes(A.shape[:-2], B.shape[:-2])
        except RuntimeError:
            return False
        return True

    return takes


def _run_svd(A, compute_uv, full_matrices=False, sort=True, flush=True):
    if _SVD is not None:
        return _SVD(A.detach(), compute_uv)
    if A.device.type == "hpu":
        if compute_uv:
            return tpc.jacobi_svd(A, full_matrices=full_matrices, sort=sort, flush=flush)
        return A.new_empty(0), tpc.jacobi_svdvals(A, sort=sort, flush=flush), A.new_empty(0)
    return tpc.svd_reference(A, compute_uv=compute_uv, full_matrices=full_matrices)


def _complete_orthogonal(U, S):
    m, k = U.shape[-2], U.shape[-1]
    eye = torch.eye(m, dtype=U.dtype, device=U.device)
    unull = _run_svd(eye - U @ U.mT, True)[0]
    return torch.cat([U, unull[..., :m - k]], -1)


class Svd(torch.autograd.Function):
    """``aten._linalg_svd`` via the Gaudi2 one-sided Jacobi TPC kernel, with torch's SVD backward."""

    @staticmethod
    def compute(A, full_matrices, compute_uv, driver=None, sort=True, flush=True):
        U, S, Vh = _run_svd(A, compute_uv, full_matrices=full_matrices, sort=sort, flush=flush)
        if compute_uv and full_matrices and A.shape[-2] != A.shape[-1]:
            if A.shape[-2] > A.shape[-1] and U.shape[-1] != A.shape[-2]:
                U = _complete_orthogonal(U, S)
            elif A.shape[-2] < A.shape[-1] and Vh.shape[-2] != A.shape[-1]:
                Vh = _complete_orthogonal(Vh.mT, S).mT
        return U, S, Vh

    @staticmethod
    def forward(ctx, A, full_matrices, compute_uv, driver):
        need_uv = compute_uv or ctx.needs_input_grad[0]
        U, S, Vh = Svd.compute(A, full_matrices, need_uv, driver)
        ctx.save_for_backward(U, S, Vh)
        ctx.set_materialize_grads(False)
        if not compute_uv:
            return A.new_empty(0), S, A.new_empty(0)
        return U, S, Vh

    @staticmethod
    def backward(ctx, gU, gS, gVh):
        if not ctx.needs_input_grad[0]:
            return None, None, None, None
        U, S, Vh = ctx.saved_tensors
        m, n, k = U.shape[-2], Vh.shape[-1], S.shape[-1]
        if k == 0 or (gU is None and gS is None and gVh is None):
            return U.new_zeros(*S.shape[:-1], m, n), None, None, None
        Uk, Vhk = U[..., :k], Vh[..., :k, :]
        gA = (Uk * gS.unsqueeze(-2)) @ Vhk if gS is not None else U.new_zeros(*S.shape[:-1], m, n)
        if gU is not None or gVh is not None:
            # torch's S_j / (S_j^2 - S_i^2) and S_i / (S_j^2 - S_i^2), as (S_j / (S_j + S_i)) / (S_j - S_i): no
            # square, so neither it nor its derivative (double backward) over- or underflows for distinct values
            # (codex review #7: diag(1e18, 2.0001, 2) was NaN); equal values give 0 (torch's are non-finite there)
            Sj, Si = S.unsqueeze(-2), S.unsqueeze(-1)
            diff, total = Sj - Si, Sj + Si
            distinct = diff != 0
            safe_diff = torch.where(distinct, diff, 1.0)
            safe_total = torch.where(total != 0, total, 1.0)
            S_inv = torch.where(S != 0, 1.0 / torch.where(S != 0, S, 1.0), 0.0)
            if gU is not None:
                gUk = gU[..., :k]
                UhgU = Uk.mH @ gUk
                G1 = torch.where(distinct, (Sj / safe_total) / safe_diff, 0.0)
                gA = gA + (Uk @ ((UhgU - UhgU.mH) * G1)) @ Vhk
                if m > k:
                    gA = gA + ((gUk - Uk @ UhgU) * S_inv.unsqueeze(-2)) @ Vhk
            if gVh is not None:
                gVhk = gVh[..., :k, :]
                gVhV = gVhk @ Vhk.mH
                G2 = torch.where(distinct, (Si / safe_total) / safe_diff, 0.0)
                gA = gA - (Uk @ ((gVhV - gVhV.mH) * G2)) @ Vhk
                if n > k:
                    gA = gA + (Uk * S_inv.unsqueeze(-2)) @ (gVhk - gVhV @ Vhk)
        return gA, None, None, None


# The backend kernels of the CompositeImplicitAutograd ops (svdvals, matrix_norm, matrix_rank) run under inference
# mode; a call they decline goes to the op's own decomposition, never to the CPU. The CPU stand-in registers them
# only when a test sets this (the CPU has no kernel at its backend key for these ops either).
_CIA_BACKENDS_ON_CPU = False


def install_svd(device_type):
    """``linalg.svd``, ``svdvals`` and ``matrix_norm(2/-2/'nuc')`` via ``aten._linalg_svd``. Returns an undo."""
    takes = _svd_takes(device_type)
    entries = _kernels(device_type, aten._linalg_svd.default, "svd", Svd, takes)

    def svd_u_backend(A, full_matrices=False, compute_uv=True, *, driver=None, U, S, Vh):
        same = all(t.dtype == A.dtype for t in (U, S, Vh))  # else torch's own kernel raises its dtype error
        if same and takes(A, full_matrices, compute_uv, driver=driver):
            count("svd")
            u, s, vh = Svd.compute(A, full_matrices, compute_uv, driver)
        else:
            u, s, vh = _on_cpu(aten._linalg_svd.U, (A, full_matrices, compute_uv),
                               {"driver": driver, "U": U, "S": S, "Vh": Vh})
        S.resize_(s.shape).copy_(s)
        if compute_uv:
            U.resize_(u.shape).copy_(u)
            Vh.resize_(vh.shape).copy_(vh)
        else:  # as torch: U and Vh resized to empty
            U.resize_(0)
            Vh.resize_(0)
        return U, S, Vh

    entries.append((aten._linalg_svd.U, "{}", svd_u_backend))

    svdvals_op = aten.linalg_svdvals.default

    def svdvals_run(A):
        count("svd")
        if torch.is_grad_enabled() and A.requires_grad:
            return Svd.apply(A, False, True, None)[1]
        return _run_svd(A.detach(), False, sort=True)[1]

    def svdvals_ag(A, driver=None):
        if takes(A, driver=driver):
            return svdvals_run(A)
        return svdvals_op._op_dk(DispatchKey.CompositeImplicitAutograd, A, driver=driver)

    def svdvals_be(A, driver=None):
        if takes(A, driver=driver):
            return svdvals_run(A)
        return svdvals_op._op_dk(DispatchKey.CompositeImplicitAutograd, A, driver=driver)

    entries.append((svdvals_op, "Autograd{}", svdvals_ag))
    if device_type != "cpu" or _CIA_BACKENDS_ON_CPU:
        entries.append((svdvals_op, "{}", svdvals_be))

    def _norm_dims(A, dim):
        if len(dim) != 2 or A.dim() < 2:
            return None
        if not all(-A.dim() <= d < A.dim() for d in dim):
            return None  # torch's own decomposition raises its IndexError (codex review, minor)
        d0, d1 = dim[0] % A.dim(), dim[1] % A.dim()
        return None if d0 == d1 else (d0, d1)

    def _norm_run(A, ord_, d0, d1, keepdim):
        count("svd")
        tail = d0 == A.dim() - 2 and d1 == A.dim() - 1
        A_mat = A if tail else A.movedim((d0, d1), (-2, -1))
        if torch.is_grad_enabled() and A_mat.requires_grad:
            S = Svd.apply(A_mat, False, True, None)[1]
        else:
            S = _run_svd(A_mat.detach(), False, sort=False)[1]
        out = S.amax(-1) if ord_ == 2 else S.amin(-1) if ord_ == -2 else S.sum(-1)
        if keepdim:
            out = out.unsqueeze(-1).unsqueeze(-1)
            if not tail:
                out = out.movedim((-2, -1), (d0, d1))
        return out

    for norm_op, check_ord in (
        (aten.linalg_matrix_norm.default, lambda o: not isinstance(o, bool) and o in (2, -2)),
        (aten.linalg_matrix_norm.str_ord, lambda o: o == "nuc"),
    ):
        def _make_norm(op_, pred_):
            def ag(*args, **kwargs):
                A, ord_, dim, keepdim, dtype = _bind(op_, args, kwargs)
                dims = _norm_dims(A, dim) if dtype is None and pred_(ord_) else None
                if dims is not None and not _transformed() and _takes_tensors(device_type, "svd", [A]) and \
                        A.shape[dims[0]] <= tpc.SMALL_MAXN and A.shape[dims[1]] <= tpc.SMALL_MAXN and \
                        (device_type != "hpu" or tpc.ready()):
                    return _norm_run(A, ord_, dims[0], dims[1], keepdim)
                return op_._op_dk(DispatchKey.CompositeImplicitAutograd, *args, **kwargs)

            return ag

        # Autograd key only: below autograd (inference mode) torch's decomposition runs on the device and reaches
        # svdvals' backend kernel for 2, -2 and nuc. A backend kernel here cost the other ords (fro, 1, inf) about
        # 0.03 ms per call of Python dispatch (Gaudi2, bridge 1.21.4: 0.78-0.88x of the decomposition).
        entries.append((norm_op, "Autograd{}", _make_norm(norm_op, check_ord)))

    return register_kernels(device_type, entries)


def _greater(x, t):
    """``x > t`` exactly: a Python number ``t`` is compared as a real number (rounded to ``x``'s dtype, ``>`` becomes
    ``>=`` where the rounding went up), a tensor in float64 off ``hpu`` (as torch's comparison with its float64
    tolerances) and in ``x``'s dtype on ``hpu``, where float64 holds float32 values."""
    if not isinstance(t, torch.Tensor):
        t = float(t)
        r = float(torch.tensor(t, dtype=x.dtype))
        return x >= r if r > t else x > r
    wide = x.dtype if x.device.type == "hpu" else torch.promote_types(x.dtype, torch.float64)
    return x.to(wide) > t.to(device=x.device, dtype=wide)


def _keep(S, m, n, atol=None, rtol=None):
    """The singular values torch keeps: ``S > max(atol, rtol * S_max)`` with torch's ``get_atol_rtol`` defaults (a
    missing ``atol`` is 0; a missing ``rtol`` is ``eps * max(m, n)``, or 0 where ``atol > 0``), compared as torch
    compares them, in float64 (``_greater``), not after rounding the tolerance to ``S``'s dtype (codex review #13:
    ``atol = 1 - 1e-9`` rounded to float32 1 dropped the identity's singular values). ``atol`` and ``rtol`` are
    Python numbers or tensors of the batch shape."""
    S_max = S[..., :1] if S.shape[-1] == 0 else S.amax(-1, keepdim=True)
    default_rtol = max(m, n) * torch.finfo(S.dtype).eps
    tensor = lambda t: t.to(S.device).unsqueeze(-1) if isinstance(t, torch.Tensor) else t  # noqa: E731
    atol, rtol = tensor(atol), tensor(rtol)
    if rtol is None:
        if atol is None:
            rtol = default_rtol
        elif isinstance(atol, torch.Tensor):
            rtol = torch.where(atol > 0, 0.0, default_rtol).double()
        else:
            rtol = 0.0 if atol > 0 else default_rtol
    keep = S > 0 if atol is None else _greater(S, atol)
    wide = S.dtype if S.device.type == "hpu" else torch.promote_types(S.dtype, torch.float64)
    if isinstance(rtol, torch.Tensor):
        bound = rtol.to(wide) * S_max.to(wide)
    else:
        bound = S_max.to(wide) * rtol
    return keep & _greater(S, bound)


def _pinv_backward(A, P, gP):
    Pt, gPt = P.mH, gP.mH
    return -(Pt @ gP @ Pt) + (gPt - A @ (P @ gPt)) @ (P @ Pt) + (Pt @ P) @ (gPt - (gPt @ P) @ A)


class Pinv(torch.autograd.Function):
    """``aten.linalg_pinv.atol_rtol_tensor`` via ``Svd`` with analytical pseudoinverse backward."""

    @staticmethod
    def compute(A, atol, rtol, hermitian):
        U, S, Vh = Svd.compute(A, False, True, sort=False)
        S_inv = torch.where(_keep(S, A.shape[-2], A.shape[-1], atol, rtol), S.reciprocal(), 0.0)
        return (Vh.mH * S_inv.unsqueeze(-2)) @ U.mH

    @staticmethod
    def forward(ctx, A, atol, rtol, hermitian):
        P = Pinv.compute(A, atol, rtol, hermitian)
        ctx.save_for_backward(A, P)
        return P

    @staticmethod
    def backward(ctx, gP):
        A, P = ctx.saved_tensors
        return _pinv_backward(A, P, gP), None, None, None


def install_pinv(device_type):
    """``linalg.pinv`` (``hermitian=False``) on ``device_type`` via ``Svd``. Returns an undo."""
    op = aten.linalg_pinv.atol_rtol_tensor
    takes = _pinv_takes(device_type)
    return register_kernels(device_type, _kernels(device_type, op, "pinv", Pinv, takes, backend=device_type != "cpu"))


def _lstsq_eval(A, B, rcond, driver):
    if driver != "gels" and _is_fake(A):
        # Tracing (torch.compile): the rank check below needs the values, and torch's rank and residuals (empty when
        # rank < n) depend on them. Only gels, which reports no rank and assumes full rank, is taken.
        return None
    m, n = A.shape[-2], A.shape[-1]
    k = min(m, n)
    need_sort = driver in ("gelsd", "gelss")
    U, S, Vh = Svd.compute(A, False, True, sort=need_sort)
    eps = torch.finfo(S.dtype).eps
    rtol_val = max(m, n) * eps if (rcond is None or rcond < 0) else (
        0.5 * eps if rcond == 0 and driver == "gelsd" else float(rcond))
    fake = _is_fake(S)
    S_max = S[..., :1] if need_sort else S.amax(-1, keepdim=True)
    if fake:
        # traced gels only (the others declined above): gels ignores rcond and assumes full rank, as in eager, where
        # a rank-deficient input is declined to torch's gels (codex review #11)
        S_inv = S.reciprocal()
        bad = None
    else:
        S_min = S[..., -1:] if need_sort else S.amin(-1, keepdim=True)
        bad = (S_min <= S_max * rtol_val).any()
        S_inv = S.reciprocal()
    vector = B.dim() == A.dim() - 1
    B_mat = B.unsqueeze(-1) if vector else B
    sol_mat = Vh.mH @ (S_inv.unsqueeze(-1) * (U.mH @ B_mat))
    sol = sol_mat.squeeze(-1) if vector else sol_mat
    if m > n and driver in ("gels", "gelsd", "gelss"):
        residuals = (B_mat - A @ sol_mat).square().sum(-2)
    else:
        residuals = A.new_empty(0)
    # a fresh tensor (torch's rank owns its storage; a stride-0 expand breaks in-place use and lazy-mode views)
    rank_out = A.new_empty(0, dtype=torch.int64) if driver == "gels" else A.new_full(
        A.shape[:-2], k, dtype=torch.int64)
    sv_out = S if driver in ("gelsd", "gelss") else A.new_empty(0)
    if bad is not None and bool(bad):
        return None
    return sol, U, S, Vh, residuals, rank_out, sv_out, rtol_val, vector


class _Lstsq(torch.autograd.Function):
    """The differentiable lstsq solution. The backward is torch's (``linalg_lstsq_backward``): through
    ``linalg.pinv(A)`` at pinv's default tolerance, whatever ``rcond`` the forward used (codex review #12)."""

    @staticmethod
    def forward(ctx, A, B, sol, U, S, Vh, residuals, rank_out, sv_out, rtol_val, vector):
        ctx.save_for_backward(A, B, U, S, Vh)
        ctx.options = (rtol_val, vector)
        ctx.mark_non_differentiable(residuals, rank_out, sv_out)
        return sol, residuals, rank_out, sv_out

    @staticmethod
    def backward(ctx, g_sol, *_):
        A, B, U, S, Vh = ctx.saved_tensors
        _, vector = ctx.options
        if torch.is_grad_enabled():
            P = Pinv.apply(A, None, None, False)
        else:
            S_inv = torch.where(_keep(S, A.shape[-2], A.shape[-1]), S.reciprocal(), 0.0)
            P = (Vh.mH * S_inv.unsqueeze(-2)) @ U.mH
        g_sol_mat = g_sol.unsqueeze(-1) if vector else g_sol
        gA = None
        if ctx.needs_input_grad[0]:
            B_mat = B.unsqueeze(-1) if vector else B
            gA = _pinv_backward(A, P, g_sol_mat @ B_mat.mH).sum_to_size(A.shape)
        gB = None
        if ctx.needs_input_grad[1]:
            gB = P.mH @ g_sol_mat
            gB = (gB.squeeze(-1) if vector else gB).sum_to_size(B.shape)
        return gA, gB, None, None, None, None, None, None, None, None, None


def install_lstsq(device_type):
    """``linalg.lstsq`` on ``device_type`` via ``pinv`` for full-rank inputs (else declines). Returns an undo."""
    op = aten.linalg_lstsq.default
    takes = _lstsq_takes(device_type)

    def autograd_kernel(*args, **kwargs):
        if takes(*args, **kwargs):
            A, B, rcond, driver = _bind(op, args, kwargs)
            res = _lstsq_eval(A, B, rcond, driver)
            if res is not None:
                count("lstsq")
                if torch.is_grad_enabled() and (A.requires_grad or B.requires_grad):
                    return _Lstsq.apply(A, B, *res)
                sol, _, _, _, residuals, rank_out, sv_out, _, _ = res
                return sol, residuals, rank_out, sv_out
        return op._op_dk(DispatchKey.Autograd, *args, **kwargs)

    def backend_kernel(*args, **kwargs):
        if takes(*args, **kwargs):
            A, B, rcond, driver = _bind(op, args, kwargs)
            res = _lstsq_eval(A, B, rcond, driver)
            if res is not None:
                count("lstsq")
                sol, _, _, _, residuals, rank_out, sv_out, _, _ = res
                return sol, residuals, rank_out, sv_out
        return _on_cpu(op, args, kwargs)

    entries = [(op, "Autograd{}", autograd_kernel)]
    if device_type != "cpu":
        entries.append((op, "{}", backend_kernel))
    return register_kernels(device_type, entries)


def install_matrix_rank(device_type):
    """``linalg.matrix_rank`` (``hermitian=False``) on ``device_type`` via ``Svd`` without float64 promotion.
    Returns an undo."""
    svd_takes = _svd_takes(device_type, name="svd")

    def make_float(op):
        def run(A, tol):
            count("svd")
            S = Svd.compute(A.detach(), False, False, sort=False)[1]
            # torch's legacy overload: atol = tol, rtol = 0, so a negative tol counts no zero (codex review, minor)
            return _keep(S, A.shape[-2], A.shape[-1], tol, 0.0).sum(-1, dtype=torch.int64)

        def kernel(*args, **kwargs):
            A, tol, hermitian = _bind(op, args, kwargs)
            if not hermitian and svd_takes(A):
                return run(A, tol)
            return op._op_dk(DispatchKey.CompositeImplicitAutograd, *args, **kwargs)

        def backend(*args, **kwargs):
            A, tol, hermitian = _bind(op, args, kwargs)
            if not hermitian and svd_takes(A):
                return run(A, tol)
            return op._op_dk(DispatchKey.CompositeImplicitAutograd, *args, **kwargs)

        return kernel, backend

    def make_tol_tensor(op):
        def takes(A, tol, hermitian=False):
            return not hermitian and svd_takes(A) and _tol_tensor_ok(tol, A.shape[:-2])

        def run(A, tol):
            count("svd")
            S = Svd.compute(A.detach(), False, False, sort=False)[1]
            return _keep(S, A.shape[-2], A.shape[-1], tol, 0.0).sum(-1, dtype=torch.int64)

        def kernel(*args, **kwargs):
            A, tol, hermitian = _bind(op, args, kwargs)
            if takes(A, tol, hermitian):
                return run(A, tol)
            return op._op_dk(DispatchKey.CompositeImplicitAutograd, *args, **kwargs)

        def backend(*args, **kwargs):
            A, tol, hermitian = _bind(op, args, kwargs)
            if takes(A, tol, hermitian):
                return run(A, tol)
            return op._op_dk(DispatchKey.CompositeImplicitAutograd, *args, **kwargs)

        return kernel, backend

    def make_atol_rtol_tensor(op):
        def takes(A, atol=None, rtol=None, hermitian=False):
            if hermitian or not svd_takes(A):
                return False
            return all(t is None or _tol_tensor_ok(t, A.shape[:-2]) for t in (atol, rtol))

        def run(A, atol, rtol):
            count("svd")
            S = Svd.compute(A.detach(), False, False, sort=False)[1]
            return _keep(S, A.shape[-2], A.shape[-1], atol, rtol).sum(-1, dtype=torch.int64)

        def kernel(*args, **kwargs):
            A, atol, rtol, hermitian = _bind(op, args, kwargs)
            if takes(A, atol, rtol, hermitian):
                return run(A, atol, rtol)
            return op._op_dk(DispatchKey.CompositeImplicitAutograd, *args, **kwargs)

        def backend(*args, **kwargs):
            A, atol, rtol, hermitian = _bind(op, args, kwargs)
            if takes(A, atol, rtol, hermitian):
                return run(A, atol, rtol)
            return op._op_dk(DispatchKey.CompositeImplicitAutograd, *args, **kwargs)

        return kernel, backend

    def make_atol_rtol_float(op):
        def run(A, atol, rtol):
            count("svd")
            S = Svd.compute(A.detach(), False, False, sort=False)[1]
            return _keep(S, A.shape[-2], A.shape[-1], atol, rtol).sum(-1, dtype=torch.int64)

        def kernel(*args, **kwargs):
            A, atol, rtol, hermitian = _bind(op, args, kwargs)
            if not hermitian and svd_takes(A):
                return run(A, atol, rtol)
            return op._op_dk(DispatchKey.CompositeImplicitAutograd, *args, **kwargs)

        def backend(*args, **kwargs):
            A, atol, rtol, hermitian = _bind(op, args, kwargs)
            if not hermitian and svd_takes(A):
                return run(A, atol, rtol)
            return op._op_dk(DispatchKey.CompositeImplicitAutograd, *args, **kwargs)

        return kernel, backend

    entries = []
    for op, maker in ((aten.linalg_matrix_rank.default, make_float),
                      (aten.linalg_matrix_rank.tol_tensor, make_tol_tensor),
                      (aten.linalg_matrix_rank.atol_rtol_tensor, make_atol_rtol_tensor),
                      (aten.linalg_matrix_rank.atol_rtol_float, make_atol_rtol_float)):
        ag, be = maker(op)
        entries.append((op, "Autograd{}", ag))
        if device_type != "cpu" or _CIA_BACKENDS_ON_CPU:
            entries.append((op, "{}", be))
    return register_kernels(device_type, entries)


# ---- matrix_exp & householder_product (#13)


_PADE13_THETA = 5.371920351148152
# The scaling cap of the matrix_exp kernels (S_MAX of matrix_exp_f32_gaudi2.c): ||A||_1 <= 3.4e38 needs <= 126 halvings
_EXPM_S_MAX = 126


def _expm_scaling(norm):
    """The kernels' scaling count of a 1-norm (a float): halvings until it is <= ``_PADE13_THETA``, at most
    ``_EXPM_S_MAX``; 0 when the norm is not finite. Halving is exact, so this is the kernel's count."""
    if not math.isfinite(norm):
        return 0
    theta = float(torch.tensor(_PADE13_THETA, dtype=torch.float32))  # the kernel compares with the float32 constant
    s = 0
    while norm > theta and s < _EXPM_S_MAX:
        norm *= 0.5
        s += 1
    return s
_PADE13_C = (
    1.0,
    0.5,
    3.0 / 25.0,
    11.0 / 600.0,
    11.0 / 5520.0,
    3.0 / 18400.0,
    1.0 / 96600.0,
    1.0 / 1932000.0,
    1.0 / 48944000.0,
    1.0 / 1585785600.0,
    1.0 / 67395888000.0,
    1.0 / 3953892096000.0,
    1.0 / 355850288640000.0,
    1.0 / 64764752532480000.0,
)
_EXPM_STRATEGY = "auto"  # "auto" | "tpc" | "item" | "masked8" | "masked12" | "masked16"


def _expm_counts(A):
    """The kernels' scaling count of each matrix of ``A`` (from its 1-norm, ``_expm_scaling``), on the host (one sync)
    and, unless all are equal, as a tensor ``[..., 1, 1]`` with the matching scales ``2^-s``: each matrix is scaled
    and squared by its own count, so one matrix's norm never changes another's result (codex review #9)."""
    counts = [_expm_scaling(x) for x in A.detach().abs().sum(-2).amax(-1).reshape(-1).tolist()]
    if len(set(counts)) <= 1:
        return (counts[0] if counts else 0), None, None
    shape = (*A.shape[:-2], 1, 1)
    s_t = torch.tensor(counts, dtype=A.dtype).reshape(shape).to(A.device)
    scale = torch.tensor([math.ldexp(1.0, -c) for c in counts], dtype=A.dtype).reshape(shape).to(A.device)
    return max(counts), s_t, scale


def _expm_traced_counts(A, s_max):
    """``(s_t, scale)`` while tracing: the counts from the 1-norms on the device, at most ``s_max``."""
    norm = A.detach().abs().sum(-2).amax((-1,), keepdim=True).unsqueeze(-1)
    s_t = torch.ceil(torch.log2(torch.clamp(norm / _PADE13_THETA, min=1.0))).clamp(0, s_max)
    return s_t, 2.0 ** (-s_t)


def _expm_s_max():
    """The traced branches' scaling cap: the kernels' (``_EXPM_S_MAX``, every finite float32 norm) unless a
    ``masked<k>`` strategy asks for ``k`` (a lower cap mis-scales larger norms: -1e10 I gave NaN at 16)."""
    return int(_EXPM_STRATEGY[6:]) if _EXPM_STRATEGY.startswith("masked") else _EXPM_S_MAX


def _pade13_uv(A_s):
    n = A_s.shape[-1]
    c = _PADE13_C
    eye = torch.eye(n, dtype=A_s.dtype, device=A_s.device)
    X2 = A_s @ A_s
    X4 = X2 @ X2
    X6 = X2 @ X4
    W1 = c[13] * X6 + c[11] * X4 + c[9] * X2
    W2 = c[7] * X6 + c[5] * X4 + c[3] * X2 + c[1] * eye
    W = X6 @ W1 + W2
    U = A_s @ W
    Z1 = c[12] * X6 + c[10] * X4 + c[8] * X2
    Z2 = c[6] * X6 + c[4] * X4 + c[2] * X2 + eye
    V = X6 @ Z1 + Z2
    return U, V


def _matrix_exp_composite(A, use_grad):
    if _is_fake(A) or _EXPM_STRATEGY.startswith("masked"):
        s_max = _expm_s_max()
        s_t, scale = _expm_traced_counts(A, s_max)
    else:
        s_max, s_t, scale = _expm_counts(A)
        scale = scale if s_t is not None else math.ldexp(1.0, -s_max)
    A_s = A if s_t is None and s_max == 0 else A * scale
    U, V = _pade13_uv(A_s)
    R = SolveEx.apply(V - U, V + U, True, False)[0] if use_grad else SolveEx.compute(V - U, V + U, True, False)[0]
    for k in range(s_max):
        R = R @ R if s_t is None else torch.where(s_t > k, R @ R, R)
    return R


def _matrix_exp_frechet_torch(A, gY):
    """Adjoint Fréchet derivative ``L_exp(A^T, gY)`` via Padé-13 + ``SolveEx`` (supports double backward)."""
    At = A.mT
    if _is_fake(A) or _is_fake(gY) or _EXPM_STRATEGY.startswith("masked"):
        s_max = _expm_s_max()
        s_t, scale = _expm_traced_counts(At, s_max)
        As, Es = At * scale, gY * scale
    else:
        s_max, s_t, scale = _expm_counts(At)
        if s_t is None:
            scale = math.ldexp(1.0, -s_max)
            As, Es = (At, gY) if s_max == 0 else (At * scale, gY * scale)
        else:
            As, Es = At * scale, gY * scale
    n = As.shape[-1]
    c = _PADE13_C
    eye = torch.eye(n, dtype=As.dtype, device=As.device)
    X2 = As @ As
    X4 = X2 @ X2
    X6 = X2 @ X4
    M2 = As @ Es + Es @ As
    M4 = X2 @ M2 + M2 @ X2
    M6 = X2 @ M4 + M2 @ X4
    W1 = c[13] * X6 + c[11] * X4 + c[9] * X2
    W2 = c[7] * X6 + c[5] * X4 + c[3] * X2 + c[1] * eye
    W = X6 @ W1 + W2
    LW1 = c[13] * M6 + c[11] * M4 + c[9] * M2
    LW2 = c[7] * M6 + c[5] * M4 + c[3] * M2
    LW = M6 @ W1 + X6 @ LW1 + LW2
    U = As @ W
    LU = Es @ W + As @ LW
    Z1 = c[12] * X6 + c[10] * X4 + c[8] * X2
    Z2 = c[6] * X6 + c[4] * X4 + c[2] * X2 + eye
    V = X6 @ Z1 + Z2
    LZ1 = c[12] * M6 + c[10] * M4 + c[8] * M2
    LZ2 = c[6] * M6 + c[4] * M4 + c[2] * M2
    LV = M6 @ Z1 + X6 @ LZ1 + LZ2
    D = V - U
    R = SolveEx.apply(D, V + U, True, False)[0]
    L = SolveEx.apply(D, (LV + LU) - (LV - LU) @ R, True, False)[0]
    for k in range(s_max):
        if s_t is None:
            L = R @ L + L @ R
            if k + 1 < s_max:
                R = R @ R
        else:
            active = s_t > k
            L = torch.where(active, R @ L + L @ R, L)
            if k + 1 < s_max:
                R = torch.where(active, R @ R, R)
    return L


class MatrixExp(torch.autograd.Function):
    """``aten.linalg_matrix_exp`` via Gaudi2 Padé-13 TPC kernels (n <= ``tpc.SMALL_MAXN``), with differentiable Padé-13
    Fréchet backward for double backward and ``torch.compile``."""

    @staticmethod
    def compute(A):
        if _is_fake(A):
            return tpc.matrix_exp(A) if (A.device.type == "hpu" and tpc.ready()) else _matrix_exp_composite(A, False)
        if _EXPM_STRATEGY in ("auto", "tpc"):
            if _MATRIX_EXP is not None:
                return _MATRIX_EXP(A.detach())
            if A.device.type == "hpu" and tpc.ready():
                return tpc.matrix_exp(A)
        return _matrix_exp_composite(A, False)

    @staticmethod
    def forward(ctx, A):
        Y = MatrixExp.compute(A)
        ctx.save_for_backward(A)
        return Y

    @staticmethod
    def backward(ctx, gY):
        (A,) = ctx.saved_tensors
        if not torch.is_grad_enabled() and not _is_fake(A) and not _is_fake(gY) and _EXPM_STRATEGY in ("auto", "tpc"):
            if _MATRIX_EXP_FRECHET is not None:
                return _MATRIX_EXP_FRECHET(A.detach(), gY.detach())
            if A.device.type == "hpu" and tpc.ready():
                return tpc.matrix_exp_frechet(A, gY)
        if not torch.is_grad_enabled() and _is_fake(A) and A.device.type == "hpu" and tpc.ready() and \
                _EXPM_STRATEGY in ("auto", "tpc"):
            return tpc.matrix_exp_frechet(A, gY)  # traced first backward: the kernel, per-matrix scaling to 126
        return _matrix_exp_frechet_torch(A, gY)


def install_matrix_exp(device_type):
    """``linalg.matrix_exp`` / ``torch.matrix_exp`` via Gaudi2 Padé-13 TPC kernels (n <= ``tpc.SMALL_MAXN``).
    Returns an undo."""
    op = aten.linalg_matrix_exp.default
    takes = _tpc_takes(device_type, _takes(device_type, "matrix_exp"), maxn=tpc.SMALL_MAXN)

    def expm_ag(A):
        if takes(A):
            count("matrix_exp")
            if _is_fake(A) and A.device.type != "hpu":
                return _matrix_exp_composite(A, torch.is_grad_enabled() and A.requires_grad)
            if _EXPM_STRATEGY in ("item", "masked8", "masked12", "masked16"):
                return _matrix_exp_composite(A, torch.is_grad_enabled() and A.requires_grad)
            return MatrixExp.apply(A)
        return op._op_dk(DispatchKey.Autograd, A)

    def expm_be(A):
        if takes(A):
            count("matrix_exp")
            return MatrixExp.compute(A)
        return _on_cpu(op, (A,), {})

    entries = [(op, "Autograd{}", expm_ag)]
    if device_type != "cpu":
        entries.append((op, "{}", expm_be))

    op_out = aten.linalg_matrix_exp.out

    def expm_out_be(A, *, out):
        if out.dtype == A.dtype and takes(A):  # else torch's own kernel raises its dtype error
            count("matrix_exp")
            res = MatrixExp.compute(A)
        elif device_type == "cpu":
            return op_out._op_dk(DispatchKey.CompositeExplicitAutograd, A, out=out)
        else:
            res = _on_cpu(op_out, (A,), {"out": out})
        out.resize_(res.shape).copy_(res)
        return out

    entries.append((op_out, "{}", expm_out_be))
    return register_kernels(device_type, entries)


_HH_FORWARD = "auto"  # "auto" | "tpc" | "tpc_noflush" | "wy"
_HH_BACKWARD_INV = "auto"  # "auto" | "tpc" | "wy"
_HH_MAX_M = None


def _unit_triu_inv(U):
    """Inverse of unit upper-triangular ``U[..., k, k]`` starting from 2x2 blocks."""
    k = U.shape[-1]
    if k <= 1:
        return torch.ones_like(U)
    if k == 2:
        return torch.eye(2, dtype=U.dtype, device=U.device) - torch.triu(U, 1)
    p = 1 << (k - 1).bit_length()
    Ip = torch.eye(p, dtype=U.dtype, device=U.device)
    U_pad = F.pad(torch.triu(U, 1), (0, p - k, 0, p - k)) + Ip if p > k else torch.triu(U, 1) + Ip
    b = p >> 1
    U2 = torch.diagonal(U_pad.unflatten(-2, (b, 2)).unflatten(-1, (b, 2)), dim1=-4, dim2=-2).movedim(-1, -3)
    X = 2.0 * torch.eye(2, dtype=U.dtype, device=U.device) - U2
    s = 2
    while s < p:
        nb = p // (2 * s)
        blocks = torch.diagonal(
            U_pad.unflatten(-2, (nb, 2 * s)).unflatten(-1, (nb, 2 * s)), dim1=-4, dim2=-2
        ).movedim(-1, -3)
        C = blocks[..., :s, s:]
        Xa, Xd = X[..., 0::2, :, :], X[..., 1::2, :, :]
        ur = -(Xa @ C @ Xd)
        X = torch.cat([torch.cat([Xa, ur], -1), F.pad(Xd, (s, 0))], -2)
        s *= 2
    return X[..., 0, :k, :k]


def _compact_wy_factors(A, tau):
    m, k = A.shape[-2], tau.shape[-1]
    Y = torch.tril(A[..., :k], -1) + torch.eye(m, k, dtype=A.dtype, device=A.device)
    S = torch.triu(Y.mH @ Y, 1)
    # a zero tau is the identity reflector: its row of tau S is 0 even where S overflowed (0 * Inf was NaN; codex #16).
    # Only there: elsewhere tau S keeps its derivative in tau (double backward at tau = 0)
    tS = tau.unsqueeze(-1) * S
    U = torch.eye(k, dtype=A.dtype, device=A.device) + torch.where((tau.unsqueeze(-1) == 0) & S.isinf(), 0.0, tS)
    return Y, S, U


def _compact_wy_forward(A, tau, Ui=None):
    m, n, k = A.shape[-2], A.shape[-1], tau.shape[-1]
    if k == 0:
        return torch.eye(m, n, dtype=A.dtype, device=A.device).expand_as(A).clone()
    Y, _, U = _compact_wy_factors(A, tau)
    if Ui is None:
        Ui = _unit_triu_inv(U)
    T = Ui * tau.unsqueeze(-2)
    return torch.eye(m, n, dtype=A.dtype, device=A.device) - Y @ (T @ Y[..., :n, :].mH)


def _run_householder_tpc(A, tau, flush=True):
    if _HOUSEHOLDER is not None:
        return _HOUSEHOLDER(A.detach(), tau.detach())
    if A.device.type == "hpu":
        return tpc.householder_product(A, tau, flush=flush)
    return tpc.householder_product_reference(A, tau)


class HouseholderProduct(torch.autograd.Function):
    """``aten.linalg_householder_product`` with TPC / Compact-WY forward and exact Compact-WY backward."""

    @staticmethod
    def compute(A, tau):
        m, n, k = A.shape[-2], A.shape[-1], tau.shape[-1]
        if k == 0:
            return torch.eye(m, n, dtype=A.dtype, device=A.device).expand_as(A).clone()
        if _is_fake(A) and A.device.type != "hpu":
            return _compact_wy_forward(A, tau)
        tpc_ok = m <= tpc.MAXHH and (_HOUSEHOLDER is not None or (A.device.type == "hpu" and tpc.ready()))
        if _HH_FORWARD == "wy":
            return _compact_wy_forward(A, tau)
        flush = _HH_FORWARD != "tpc_noflush"  # lazy mode: mark_step before the TPC op (measured race otherwise)
        return _run_householder_tpc(A, tau, flush=flush) if tpc_ok else _compact_wy_forward(A, tau)

    @staticmethod
    def forward(ctx, A, tau):
        Q = HouseholderProduct.compute(A, tau)
        ctx.save_for_backward(A, tau)
        return Q

    @staticmethod
    def backward(ctx, gQ):
        A, tau = ctx.saved_tensors
        m, n, k = A.shape[-2], A.shape[-1], tau.shape[-1]
        if k == 0 or (not ctx.needs_input_grad[0] and not ctx.needs_input_grad[1]):
            return None, None
        Y, S, U = _compact_wy_factors(A, tau)
        use_tpc_inv = (_HH_BACKWARD_INV != "wy") and k <= tpc.MAXN and _use_tpc(U)
        Ui = InvEx.apply(U, False)[0] if use_tpc_inv else _unit_triu_inv(U)
        T = Ui * tau.unsqueeze(-2)
        Yn = Y[..., :n, :]
        gQYn = gQ @ Yn
        gT = -(Y.mH @ gQYn)
        YT = Y @ T
        gY_n = -(gQ.mH @ YT)
        gY_dir = -(gQYn @ T.mH) + (F.pad(gY_n, (0, 0, 0, m - n)) if m > n else gY_n)
        W = Ui.mH @ torch.triu(gT)
        WTt = W @ T.mH
        g_tau = torch.diagonal(W, dim1=-2, dim2=-1) - (WTt * S).sum(-1) if ctx.needs_input_grad[1] else None
        gA = None
        if ctx.needs_input_grad[0]:
            Gs = torch.triu(-(tau.unsqueeze(-1) * WTt), 1)
            gY = gY_dir + Y @ (Gs + Gs.mH)
            gA_k = torch.tril(gY, -1)
            gA = F.pad(gA_k, (0, n - k)) if n > k else gA_k
        return gA, g_tau


def _householder_takes(device_type, max_m=None):
    def takes(A, tau, **_out):
        if _transformed():  # no jvp / vmap rule in these Functions: torch's own kernel
            return False
        if not _takes_tensors(device_type, "householder_product", [A]):
            return False
        if A.dim() < 2 or A.shape[-2] < A.shape[-1]:
            return False
        if tau.dtype != A.dtype or tau.device != A.device or not static(*tau.shape):
            return False
        if tau.dim() != A.dim() - 1 or tau.shape[:-1] != A.shape[:-2] or tau.shape[-1] > A.shape[-1]:
            return False
        m = A.shape[-2]
        cap = _HH_MAX_M if max_m is None else max_m
        if cap is not None and m > cap:
            return False
        if device_type == "hpu" and m <= tpc.MAXHH and tau.shape[-1] > 0 and not tpc.ready():
            return False
        return True

    return takes


def install_householder_product(device_type):
    """``linalg.householder_product`` / ``torch.orgqr`` via Gaudi2 TPC kernel & Compact-WY backward.
    Returns an undo."""
    op = aten.linalg_householder_product.default
    takes = _householder_takes(device_type)
    entries = _kernels(device_type, op, "householder_product", HouseholderProduct, takes)

    def hh_out_be(A, tau, *, out):
        if out.dtype == A.dtype and takes(A, tau):  # else torch's own kernel raises its dtype error
            count("householder_product")
            res = HouseholderProduct.compute(A, tau)
        else:
            res = _on_cpu(aten.linalg_householder_product.out, (A, tau), {"out": out})
        out.resize_(res.shape).copy_(res)
        return out

    if device_type != "cpu":
        entries.append((aten.linalg_householder_product.out, "{}", hh_out_be))
    return register_kernels(device_type, entries)


# ---- standalone LU (#10): lu_factor / lu_factor_ex, lu_solve, lu_unpack, linalg.lu


def _lu_kernel(A):
    """Whether ``A`` goes through the TPC LU kernel (or the ``_LU`` hook in the CPU tests)."""
    if A.shape[-1] > tpc.SMALL_MAXN:
        return False
    return _LU is not None or (A.device.type == "hpu" and A.dtype == torch.float32 and tpc.ready())


def _lu_factor(A):
    """``(LU, pivots, perm, info)``: LAPACK's packed factors and 1-based int32 pivots, the row permutation (``LU``'s
    row i is ``A``'s row ``perm[i]``, in ``A``'s dtype, exact below 2**24) and LAPACK's int32 info."""
    if _lu_kernel(A):
        LU, pivots, perm, info = (tpc.lu_factor if _LU is None else _LU)(A)
        return LU.to(A.dtype), pivots, perm.to(A.dtype), info
    LU, pivots = lu(A)
    return LU, pivots, pivots_to_perm(pivots).to(A.dtype), lu_info(LU)


# Variant knobs, measured on Gaudi2 (bridge 1.21.4, TEAM_REPORT): how lu_solve and lu_unpack turn pivots into the
# permutation, "kernel" (one TPC launch) or "torch" (``pivots_to_perm``, n small gathers: slower in both modes);
# how lu_solve solves, "kernel" (the TPC getrs kernel, left and not adjoint only), "gj" (A = P L U rebuilt, one
# Gauss-Jordan kernel; loses the factors' accuracy, codex review #14) or "tri" (``lu_solve``'s triangular inverses,
# not adjoint). "auto": kernel, else two triangular solves with the factors (``_lu_solve_factors``).
_LU_PERM_MODE = "auto"
_LU_SOLVE_MODE = "auto"
# The getrs hook ``(LU, pivots, B) -> X`` for the CPU tests; None means ``tpc.lu_solve`` on HPU.
_LU_SOLVE = None


def _lu_perm(pivots, dtype):
    """The row permutation of LAPACK's ``pivots`` in ``dtype``: one TPC launch on HPU, else ``pivots_to_perm``."""
    if pivots.shape[-1] <= tpc.SMALL_MAXN and _LU_PERM_MODE in ("auto", "kernel"):
        if _LU_PERM is not None:
            return _LU_PERM(pivots).to(dtype)
        if pivots.device.type == "hpu" and tpc.ready():
            return tpc.lu_perm(pivots).to(dtype)
    return pivots_to_perm(pivots).to(dtype)


def _perm_matrix(perm, n):
    """``P`` with ``A = P L U``: ``P[..., perm[i], i] = 1``, built by a float compare (no integer gather)."""
    iota = torch.arange(n, dtype=perm.dtype, device=perm.device)
    return (perm[..., None, :] == iota[:, None]).to(perm.dtype)


def _unpack(LU):
    """``(L, U)`` of packed square factors: L unit lower triangular, U upper triangular."""
    n = LU.shape[-1]
    return torch.tril(LU, -1) + torch.eye(n, dtype=LU.dtype, device=LU.device), torch.triu(LU)


def _lu_backward(A, P, L, U, gL, gU):
    """torch's ``linalg_lu_backward`` for square A = P L U: ``P L^-H [tril(L^H gL, -1) + triu(gU U^H)] U^-H``, as two
    triangular solves with the factors themselves (``SolveTriangular``: the TPC substitution kernel on Gaudi2), never
    through ``A^-1``: its sandwich ``A^-H U^H M ...`` squared the scale of A first (1e20 I overflowed, 1e-24 I
    underflowed; codex review #15). Differentiable (double backward)."""
    M = A.new_zeros(()).expand(A.shape)
    if gL is not None:
        M = M + torch.tril(L.mH @ gL, -1)
    if gU is not None:
        M = M + torch.triu(gU @ U.mH)
    X = SolveTriangular.apply(L.mH, M, True, True, True)  # L^-H M
    return P @ SolveTriangular.apply(U.mH, X, False, False, False)  # P (L^-H M) U^-H


class LuFactorEx(torch.autograd.Function):
    """``aten.linalg_lu_factor_ex`` (square, ``pivot=True``) through the TPC LU kernel: LAPACK's packed factors,
    pivots and info. ``pivots`` and ``info`` are not differentiable; the backward is torch's (``_lu_backward``)."""

    @staticmethod
    def compute(A, pivot=True, check_errors=False):
        LU, pivots, _, info = _lu_factor(A)
        if check_errors:
            torch.ops.aten._linalg_check_errors(info, "torch.linalg.lu_factor_ex", is_matrix=A.dim() == 2)
        return LU, pivots, info

    @staticmethod
    def forward(ctx, A, pivot=True, check_errors=False):
        LU, pivots, perm, info = _lu_factor(A)
        if check_errors:
            torch.ops.aten._linalg_check_errors(info, "torch.linalg.lu_factor_ex", is_matrix=A.dim() == 2)
        ctx.save_for_backward(A, LU, perm)
        ctx.mark_non_differentiable(pivots, info)
        return LU, pivots, info

    @staticmethod
    def backward(ctx, gLU, *_):
        A, LU, perm = ctx.saved_tensors
        L, U = _unpack(LU)
        gA = _lu_backward(A, _perm_matrix(perm, A.shape[-1]), L, U, torch.tril(gLU, -1), torch.triu(gLU))
        return gA, None, None


class LinalgLu(torch.autograd.Function):
    """``aten.linalg_lu`` (square, ``pivot=True``): the TPC LU kernel, unpacked. ``P`` is not differentiable; the
    backward is torch's (``_lu_backward``)."""

    @staticmethod
    def compute(A, pivot=True):
        if _LU is None and _lu_kernel(A):  # only LU and perm: no int casts of the pivots and info
            LU, perm = tpc.lu_factor(A, pivots=False)
        else:
            LU, _, perm, _ = _lu_factor(A)
        L, U = _unpack(LU)
        return _perm_matrix(perm, A.shape[-1]), L, U

    @staticmethod
    def forward(ctx, A, pivot=True):
        P, L, U = LinalgLu.compute(A)
        ctx.save_for_backward(A, P, L, U)
        ctx.mark_non_differentiable(P)
        return P, L, U

    @staticmethod
    def backward(ctx, _gP, gL, gU):
        A, P, L, U = ctx.saved_tensors
        return _lu_backward(A, P, L, U, gL, gU), None


def _lu_matrix(LU, pivots):
    """``A = P L U`` rebuilt from the packed factors: a float compare for ``P`` and two matmuls on the MME."""
    n = LU.shape[-1]
    L, U = _unpack(LU)
    return _perm_matrix(_lu_perm(pivots, LU.dtype), n) @ (L @ U)


def _lu_solve_factors(LU, pivots, B, left, adjoint):
    """``A^-1 B``, ``A^-H B``, ``B A^-1`` or ``B A^-H`` for ``A = P L U`` from the packed factors directly: two
    triangular solves (``SolveTriangular.compute``: the TPC substitution kernel on Gaudi2) and a matmul with ``P``.
    ``A`` is never rebuilt (its rounding loses what the factors hold), and ``B`` is always a matrix (no vector
    heuristic: ``lu_solve`` has none); the batches of the factors and of ``B`` broadcast as in torch."""
    P = _perm_matrix(_lu_perm(pivots, LU.dtype), LU.shape[-1])
    tri = SolveTriangular.compute
    if left and not adjoint:  # U^-1 L^-1 P^T B
        return tri(LU, tri(LU, P.mT @ B, False, True, True), True, True, False)
    if left:  # A^H = U^H L^H P^T: P L^-H U^-H B
        return P @ tri(LU.mH, tri(LU.mH, B, False, True, False), True, True, True)
    if not adjoint:  # X P L U = B: B U^-1 L^-1 P^T
        return tri(LU, tri(LU, B, True, False, False), False, False, True) @ P.mT
    # X U^H L^H P^T = B: (B P) L^-H U^-H
    return tri(LU.mH, tri(LU.mH, B @ P, True, False, True), False, False, False)


class LuSolve(torch.autograd.Function):
    """``aten.linalg_lu_solve`` (square): left and not adjoint the TPC getrs kernel on Gaudi2; the other cases (and
    without the kernel) two triangular solves with the factors themselves and a matmul with ``P``
    (``_lu_solve_factors``). ``pivots`` is not differentiable; the backward is torch's: ``gB`` the adjoint solve and
    ``gLU`` through ``A = P L U``."""

    @staticmethod
    def compute(LU, pivots, B, left=True, adjoint=False):
        mode = _LU_SOLVE_MODE
        if mode in ("auto", "kernel") and left and not adjoint and LU.shape[-1] <= tpc.SMALL_MAXN:
            if _LU_SOLVE is not None:
                return _LU_SOLVE(LU, pivots, B)
            if LU.device.type == "hpu" and LU.dtype == torch.float32 and tpc.ready():
                return tpc.lu_solve(LU, pivots, B)
        if mode == "tri" and not adjoint:
            batch = torch.broadcast_shapes(LU.shape[:-2], B.shape[:-2])
            return lu_solve(LU.expand(*batch, *LU.shape[-2:]), pivots.expand(*batch, LU.shape[-1]), B, left)
        if mode == "gj":  # measurement knob: A = P L U rebuilt, one Gauss-Jordan solve (loses the factors' accuracy)
            A = _lu_matrix(LU, pivots)
            batch = torch.broadcast_shapes(A.shape[:-2], B.shape[:-2])
            A = A.expand(*batch, *A.shape[-2:])
            B = B.expand(*batch, *B.shape[-2:])  # same rank as A: never linalg.solve's vector case
            return SolveEx.compute(A.mH if adjoint else A, B, left, False)[0]
        return _lu_solve_factors(LU, pivots, B, left, adjoint)

    @staticmethod
    def forward(ctx, LU, pivots, B, left=True, adjoint=False):
        X = LuSolve.compute(LU, pivots, B, left, adjoint)
        ctx.save_for_backward(LU, pivots, X)
        ctx.left, ctx.adjoint = left, adjoint
        ctx.B_shape = B.shape
        return X

    @staticmethod
    def backward(ctx, gX):
        LU, pivots, X = ctx.saved_tensors
        left, adjoint = ctx.left, ctx.adjoint
        gB = LuSolve.apply(LU, pivots, gX, left, not adjoint)
        gLU = None
        if ctx.needs_input_grad[0]:
            if left:
                gA = -(X @ gB.mH) if adjoint else -(gB @ X.mH)
            else:
                gA = -(gB.mH @ X) if adjoint else -(X.mH @ gB)
            gA = gA.sum_to_size(LU.shape)
            n = LU.shape[-1]
            L, U = _unpack(LU)
            PtgA = _perm_matrix(_lu_perm(pivots, LU.dtype), n).mT @ gA
            gLU = torch.tril(PtgA @ U.mH, -1) + torch.triu(L.mH @ PtgA)
        return gLU, None, gB.sum_to_size(ctx.B_shape), None, None


class LuUnpack(torch.autograd.Function):
    """``aten.lu_unpack`` (square): ``L`` and ``U`` by masks, ``P`` from the pivots by the TPC permutation kernel and a
    float compare (Habana's ``lu_unpack`` falls back to the CPU). ``P`` is not differentiable; ``gLU = tril(gL, -1) +
    triu(gU)`` as torch's."""

    @staticmethod
    def compute(LU, pivots, unpack_data=True, unpack_pivots=True):
        # distinct empty tensors (as torch's): one object returned twice would resize both (codex review, minor)
        L, U = _unpack(LU) if unpack_data else (LU.new_empty(0), LU.new_empty(0))
        P = _perm_matrix(_lu_perm(pivots, LU.dtype), LU.shape[-1]) if unpack_pivots else LU.new_empty(0)
        return P, L, U

    @staticmethod
    def forward(ctx, LU, pivots, unpack_data=True, unpack_pivots=True):
        P, L, U = LuUnpack.compute(LU, pivots, unpack_data, unpack_pivots)
        ctx.unpack_data = unpack_data
        ctx.mark_non_differentiable(P)
        return P, L, U

    @staticmethod
    def backward(ctx, _gP, gL, gU):
        if not ctx.unpack_data:
            return None, None, None, None
        return torch.tril(gL, -1) + torch.triu(gU), None, None, None


def _pivots_ok(LU, pivots):
    return (isinstance(pivots, torch.Tensor) and pivots.dtype == torch.int32 and pivots.device == LU.device
            and tuple(pivots.shape) == tuple(LU.shape[:-1]))


# Measured on Gaudi2 (bridge 1.21.4, TEAM_REPORT, same-process A/B): in lazy mode the LU ops are at parity with
# Habana's CPU fallback or slower (one graph boundary per TPC launch: lu_factor_ex 0.71-0.98x, linalg.lu 0.65x), and
# linalg.lu is at parity or slower in eager mode too (0.87x median of 5 rounds). They are installed anyway, in both
# modes: nothing runs on the CPU. ``HPU_OPS_LU_SLOW`` is no longer read.
LU_SLOW = "HPU_OPS_LU_SLOW"


def _lu_wanted(device_type, op_name):
    """Whether ``install_lu`` installs ``op_name``: every LU op, on every device and in both modes."""
    return True


def install_lu(device_type):
    """``linalg.lu_factor`` / ``lu_factor_ex``, ``lu_solve``, ``lu_unpack`` and ``linalg.lu`` on ``device_type`` for
    square float32 matrices of n <= ``tpc.SMALL_MAXN``, through the TPC LU kernel, in both modes (see ``LU_SLOW``).
    Returns an undo."""
    factor_base = _tpc_takes(device_type, _takes(device_type, "lu_factor"), maxn=tpc.SMALL_MAXN)
    lu_base = _tpc_takes(device_type, _takes(device_type, "lu"), maxn=tpc.SMALL_MAXN)
    solve_base = _tpc_takes(device_type, _takes(device_type, "lu_solve"), maxn=tpc.SMALL_MAXN)
    unpack_base = _tpc_takes(device_type, _takes(device_type, "lu_unpack"), maxn=tpc.SMALL_MAXN)

    def factor_takes(A, pivot=True, check_errors=False):
        return pivot is True and factor_base(A)

    def lu_takes(A, pivot=True):
        return pivot is True and lu_base(A)

    def solve_takes(LU, pivots, B, left=True, adjoint=False):
        if not (_pivots_ok(LU, pivots) and isinstance(B, torch.Tensor) and B.dim() >= 2 and solve_base(LU, B)):
            return False
        if B.shape[-2 if left else -1] != LU.shape[-1]:
            return False
        try:
            torch.broadcast_shapes(LU.shape[:-2], B.shape[:-2])
        except RuntimeError:
            return False
        return True

    def unpack_takes(LU_data, LU_pivots, unpack_data=True, unpack_pivots=True):
        return _pivots_ok(LU_data, LU_pivots) and unpack_base(LU_data)

    groups = [
        _kernels(device_type, op, name, function, takes) for op, name, function, takes in (
            (aten.linalg_lu_factor_ex.default, "lu_factor", LuFactorEx, factor_takes),
            (aten.linalg_lu.default, "lu", LinalgLu, lu_takes),
            (aten.linalg_lu_solve.default, "lu_solve", LuSolve, solve_takes),
            (aten.lu_unpack.default, "lu_unpack", LuUnpack, unpack_takes),
        ) if _lu_wanted(device_type, name)
    ]
    return register_paired(device_type, groups)
