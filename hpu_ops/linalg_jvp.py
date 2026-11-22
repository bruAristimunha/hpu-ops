"""Forward-mode AD (``torch.func.jvp`` and ``torch.autograd.forward_ad``) for ``solve``, ``det`` and ``slogdet``.

Under pure forward-mode AD, ``hpu_ops.linalg``'s old-style ``Function``s decline via ``_transformed()`` because they
define no ``jvp`` rule; PyTorch's native forward-AD formulas then read ``LU`` and ``pivots`` below autograd, hitting
Habana's CPU fallback (``backend=False``). The new-style ``Function``s here supply ``setup_context`` and ``jvp`` over
the same on-card ``compute`` paths while leaving mixed reverse+forward, ``vmap``, ``grad`` and nested ``jvp(jvp)`` on
the native path.
"""
import threading
import warnings

import torch
import torch.autograd.forward_ad as fwAD
from torch._C import DispatchKey

from . import _dispatch, linalg
from ._dispatch import count

aten = torch.ops.aten
_DECLINING = threading.local()


class _JvpCtx:
    """Minimal context for invoking a new-style ``Function``'s ``setup_context`` and ``jvp`` when ``torch.func.jvp``
    already lowered into ``Autograd`` (below ``FuncTorchDynamicLayerFrontMode``, where ``custom_function_call`` cannot
    dispatch).
    """

    def save_for_forward(self, *tensors):
        self.saved_for_forward = tensors

    def mark_non_differentiable(self, *tensors):
        pass


def _only_forward_ad(*tensors, allow_vmap=False):
    """Whether only single-level forward-mode AD is active (``torch.func.jvp`` or ``forward_ad``, plus ``vmap`` when
    ``allow_vmap`` is True), with no ``grad`` or nested ``jvp`` on the functorch stack, no active native decline
    re-entry, and no reverse-mode ``requires_grad`` tensors when grad is enabled.
    """
    if getattr(_DECLINING, "depth", 0) > 0:
        return False
    stack = torch._C._functorch.get_interpreter_stack() or ()
    jvp_type = torch._C._functorch.TransformType.Jvp
    vmap_type = torch._C._functorch.TransformType.Vmap
    allowed = (jvp_type, vmap_type) if allow_vmap else (jvp_type,)
    if sum(i.key() == jvp_type for i in stack) > 1 or not all(i.key() in allowed for i in stack):
        return False
    lvl = _dispatch.forward_ad_level(*tensors)
    if lvl < 0:
        return False
    if torch.is_grad_enabled() and any(isinstance(t, torch.Tensor) and t.requires_grad for t in tensors):
        return False
    # On Habana's PyTorch fork, torch.func.jvp runs JvpInterpreter above AutogradHPU and arrives here with
    # stack=[] and no dual tangents; require an actual dual tangent so native jvp keeps real LU/pivots.
    return any(isinstance(t, torch.Tensor) and fwAD.unpack_dual(t, level=lvl).tangent is not None for t in tensors)


def _own(t):
    if t._base is None:
        return t
    if torch._C._functorch.is_gradtrackingtensor(t) or (torch.is_grad_enabled() and t.requires_grad):
        return t.clone()
    return t.detach()


def _run_jvp_ctx(jvp_fn, *bound):
    lvl = _dispatch.forward_ad_level(*bound)
    primals = tuple(fwAD.unpack_dual(x, level=lvl).primal if isinstance(x, torch.Tensor) else x for x in bound)
    tangents = tuple(fwAD.unpack_dual(x, level=lvl).tangent if isinstance(x, torch.Tensor) else None for x in bound)
    out = jvp_fn.forward(*primals)
    ctx = _JvpCtx()
    jvp_fn.setup_context(ctx, primals, out)
    out_t = jvp_fn.jvp(ctx, *tangents)
    return tuple(fwAD.make_dual(o, t, level=lvl) if isinstance(o, torch.Tensor) and t is not None else o
                 for o, t in zip(out, out_t))


class _JvpSolveEx(torch.autograd.Function):
    @staticmethod
    def forward(A, B, left, check_errors):
        X, LU, pivots, info = linalg.SolveEx.compute(A, B, left, check_errors)
        return _own(X), LU, pivots, info

    @staticmethod
    def setup_context(ctx, inputs, output):
        A, B, left, _ = inputs
        X, LU, pivots, info = output
        ctx.save_for_forward(A, X)
        ctx.left = left
        ctx.vector = linalg._is_vector_rhs(A, B)
        ctx.mark_non_differentiable(LU, pivots, info)

    @staticmethod
    def jvp(ctx, gA, gB, _left, _check_errors):
        if gA is None and gB is None:
            return None, None, None, None
        A, X = ctx.saved_for_forward
        if gA is None:
            rhs = gB
        else:
            x = X.unsqueeze(-1) if ctx.vector else X
            d = gA @ x if ctx.left else x @ gA
            if ctx.vector:
                d = d.squeeze(-1)
            rhs = -d if gB is None else gB - d
        gX = linalg.SolveEx.compute(A, rhs, ctx.left, False)[0]
        return _own(gX), None, None, None


class _JvpDet(torch.autograd.Function):
    @staticmethod
    def forward(A):
        det, LU, pivots = linalg.Det.compute(A)
        return _own(det), LU, pivots

    @staticmethod
    def setup_context(ctx, inputs, output):
        det, LU, pivots = output
        ctx.save_for_forward(inputs[0], det)
        ctx.mark_non_differentiable(LU, pivots)

    @staticmethod
    def jvp(ctx, gA):
        if gA is None:
            return None, None, None
        A, det = ctx.saved_for_forward
        invA = linalg.InvEx.compute(A, False)[0]
        tan = det * (invA.mT * gA).sum((-2, -1))
        tan = torch.where(det == 0, 0.0, tan)
        return _own(tan), None, None


class _JvpSlogdet(torch.autograd.Function):
    @staticmethod
    def forward(A):
        sign, logabsdet, LU, pivots = linalg.Slogdet.compute(A)
        return _own(sign), _own(logabsdet), LU, pivots

    @staticmethod
    def setup_context(ctx, inputs, output):
        sign, _, LU, pivots = output
        ctx.save_for_forward(inputs[0], sign)
        ctx.mark_non_differentiable(LU, pivots)

    @staticmethod
    def jvp(ctx, gA):
        if gA is None:
            return None, None, None, None
        A, sign = ctx.saved_for_forward
        invA = linalg.InvEx.compute(A, False)[0]
        g_logabsdet = (invA.mT * gA).sum((-2, -1))
        return torch.zeros_like(sign), _own(g_logabsdet), None, None


class _TracedJvpSolveEx(torch.autograd.Function):
    """Traced fallback for single-level forward-AD ``solve`` when ``takes`` declines while tracing (symbolic shapes
    or traced large ``n``): runs ``op`` via ``_TracedSolveEx`` and differentiates from ``A`` and ``X`` rather than
    ``LU`` and ``pivots``, which may be placeholders at run time."""

    @staticmethod
    def forward(op, A, B, left, check_errors):
        return linalg._TracedSolveEx.apply(op, A, B, left, check_errors)

    @staticmethod
    def setup_context(ctx, inputs, output):
        _, A, B, left, _ = inputs
        X, LU, pivots, info = output
        ctx.save_for_forward(A, X)
        ctx.left = left
        ctx.vector = linalg._is_vector_rhs(A, B)
        ctx.mark_non_differentiable(LU, pivots, info)

    @staticmethod
    def jvp(ctx, _op, gA, gB, _left, _check_errors):
        if gA is None and gB is None:
            return None, None, None, None
        A, X = ctx.saved_for_forward
        if gA is None:
            rhs = gB
        else:
            x = X.unsqueeze(-1) if ctx.vector else X
            d = gA @ x if ctx.left else x @ gA
            if ctx.vector:
                d = d.squeeze(-1)
            rhs = -d if gB is None else gB - d
        gX = torch.linalg.solve_ex(A, rhs, left=ctx.left, check_errors=False)[0]
        return _own(gX), None, None, None


class _TracedJvpDet(torch.autograd.Function):
    """Traced fallback for single-level forward-AD ``det`` (see ``_TracedJvpSolveEx``)."""

    @staticmethod
    def forward(op, A):
        return linalg._TracedDet.apply(op, A)

    @staticmethod
    def setup_context(ctx, inputs, output):
        det, LU, pivots = output
        ctx.save_for_forward(inputs[1], det)
        ctx.mark_non_differentiable(LU, pivots)

    @staticmethod
    def jvp(ctx, _op, gA):
        if gA is None:
            return None, None, None
        A, det = ctx.saved_for_forward
        invA = torch.linalg.inv_ex(A)[0]
        tan = det * (invA.mT * gA).sum((-2, -1))
        tan = torch.where(det == 0, 0.0, tan)
        return _own(tan), None, None


class _TracedJvpSlogdet(torch.autograd.Function):
    """Traced fallback for single-level forward-AD ``slogdet`` (see ``_TracedJvpSolveEx``)."""

    @staticmethod
    def forward(op, A):
        return linalg._TracedSlogdet.apply(op, A)

    @staticmethod
    def setup_context(ctx, inputs, output):
        sign, _, LU, pivots = output
        ctx.save_for_forward(inputs[1], sign)
        ctx.mark_non_differentiable(LU, pivots)

    @staticmethod
    def jvp(ctx, _op, gA):
        if gA is None:
            return None, None, None, None
        A, sign = ctx.saved_for_forward
        invA = torch.linalg.inv_ex(A)[0]
        g_logabsdet = (invA.mT * gA).sum((-2, -1))
        return torch.zeros_like(sign), _own(g_logabsdet), None, None


def _base_jvp_takes(device_type, name, shape_ok=None):
    def takes(*args, **kwargs):
        tensors = [x for x in (*args, *kwargs.values()) if isinstance(x, torch.Tensor)]
        if not _only_forward_ad(*tensors):
            return False
        if not linalg._takes_tensors(device_type, name, tensors):
            return False
        if len({t.device for t in tensors}) > 1:
            return False
        A = args[0]
        if not (A.dim() >= 2 and A.shape[-1] == A.shape[-2]):
            return False
        return True if shape_ok is None else shape_ok(*args, **kwargs)

    return takes


def _traced_jvp_takes(device_type, shape_ok=None):
    allowed = (torch.float32,) if device_type == "hpu" else (torch.float32, torch.float64)

    def takes(*args, **kwargs):
        tensors = [x for x in (*args, *kwargs.values()) if isinstance(x, torch.Tensor)]
        if not _only_forward_ad(*tensors, allow_vmap=True) or not any(linalg._tracing(t) for t in tensors):
            return False
        if not all(t.dtype in allowed and t.numel() > 0 for t in tensors):
            return False
        if len({t.device for t in tensors}) > 1:
            return False
        A = args[0]
        if not (A.dim() >= 2 and A.shape[-1] == A.shape[-2]):
            return False
        return True if shape_ok is None else shape_ok(*args, **kwargs)

    return takes


def _solve_shapes_ok(A, B, left=True, check_errors=False):
    if linalg._is_vector_rhs(A, B):
        return left and B.shape[-1] == A.shape[-1]
    return B.dim() >= 2 and B.shape[-2 if left else -1] == A.shape[-1]


def _wrap_autograd(entries, device_type, op, name, takes, jvp_fn, traced_fn, shape_ok=None):
    key_op, key, orig = entries[0]
    assert key == "Autograd{}"
    traced_takes = _traced_jvp_takes(device_type, shape_ok)

    def autograd_kernel(*args, **kwargs):
        if takes(*args, **kwargs):
            count(name)
            bound = linalg._bind(op, args, kwargs)
            if torch._C._functorch.get_interpreter_stack():
                return _run_jvp_ctx(jvp_fn, *bound)
            return jvp_fn.apply(*bound)
        if traced_takes(*args, **kwargs):
            bound = linalg._bind(op, args, kwargs)
            if torch._C._functorch.get_interpreter_stack():
                return _run_jvp_ctx(traced_fn, op, *bound)
            return traced_fn.apply(op, *bound)
        _DECLINING.depth = getattr(_DECLINING, "depth", 0) + 1
        try:
            return orig(*args, **kwargs)
        finally:
            _DECLINING.depth -= 1

    return [(key_op, key, autograd_kernel), *entries[1:]]


def with_solve_jvp(device_type, op, entries):
    takes = linalg._tpc_takes(device_type, _base_jvp_takes(device_type, "solve", _solve_shapes_ok), large=True)
    return _wrap_autograd(entries, device_type, op, "solve", takes, _JvpSolveEx, _TracedJvpSolveEx, _solve_shapes_ok)


def with_inv_det_jvp(device_type, op, name, entries):
    pair = {"det": (_JvpDet, _TracedJvpDet), "slogdet": (_JvpSlogdet, _TracedJvpSlogdet)}.get(name)
    if pair is None:
        return entries
    jvp_fn, traced_fn = pair
    takes = linalg._tpc_takes(device_type, _base_jvp_takes(device_type, name), large=True)
    return _wrap_autograd(entries, device_type, op, name, takes, jvp_fn, traced_fn)


def _hpu_jvp_eligible(*tensors):
    """Whether ``tensors`` are non-empty float32 HPU tensors on the same device with a square matrix ``tensors[0]``."""
    devs = set()
    for t in tensors:
        if not isinstance(t, torch.Tensor) or t.device.type != "hpu" or t.dtype != torch.float32 or t.numel() == 0:
            return False
        devs.add(t.device)
    if len(devs) > 1:
        return False
    A = tensors[0]
    return A.dim() >= 2 and A.shape[-1] == A.shape[-2]


def _unpack(tensor, lvl):
    if not isinstance(tensor, torch.Tensor):
        return tensor, None
    u = fwAD.unpack_dual(tensor, level=lvl)
    return u.primal, u.tangent


def _any_dual(tensors, lvl):
    return any(isinstance(t, torch.Tensor) and fwAD.unpack_dual(t, level=lvl).tangent is not None for t in tensors)


def _solve_tangent(Ap, Bp, X, gA, gB, left):
    vec = linalg._is_vector_rhs(Ap, Bp)
    if gA is None:
        rhs = gB
    else:
        x = X.unsqueeze(-1) if vec else X
        d = gA @ x if left else x @ gA
        if vec:
            d = d.squeeze(-1)
        rhs = -d if gB is None else gB - d
    return torch.linalg.solve_ex(Ap, rhs, left=left, check_errors=False)[0]


def _inv_via_solve(A):
    eye = torch.eye(A.shape[-1], dtype=A.dtype, device=A.device).expand_as(A)
    return torch.linalg.solve(A, eye)


def _cia_solve_ex(A, B, *, left=True, check_errors=False):
    if _hpu_jvp_eligible(A, B) and _solve_shapes_ok(A, B, left):
        fw_lvl = _dispatch.forward_ad_level(A, B)
        if fw_lvl >= 0 and _any_dual((A, B), fw_lvl):
            Ap, gA = _unpack(A, fw_lvl)
            Bp, gB = _unpack(B, fw_lvl)
            X, _, _, info = aten._linalg_solve_ex.default(Ap, Bp, left=left, check_errors=check_errors)
            gX = _solve_tangent(Ap, Bp, X, gA, gB, left)
            return fwAD.make_dual(X, gX, level=fw_lvl), info
    X, _, _, info = aten._linalg_solve_ex.default(A, B, left=left, check_errors=check_errors)
    return X, info


def _cia_solve(A, B, *, left=True):
    return _cia_solve_ex(A, B, left=left, check_errors=True)[0]


def _other_solve_ex(A, B, *, left=True, check_errors=False):
    if _hpu_jvp_eligible(A, B) and _solve_shapes_ok(A, B, left):
        fw_lvl = _dispatch.forward_ad_level(A, B)
        if fw_lvl >= 0 and _any_dual((A, B), fw_lvl):
            Ap, gA = _unpack(A, fw_lvl)
            Bp, gB = _unpack(B, fw_lvl)
            X, LU, pivots, info = aten._linalg_solve_ex.default(Ap, Bp, left=left, check_errors=check_errors)
            gX = _solve_tangent(Ap, Bp, X, gA, gB, left)
            return fwAD.make_dual(X, gX, level=fw_lvl), LU, pivots, info
    return aten._linalg_solve_ex.default._op_dk(DispatchKey.Autograd, A, B, left=left, check_errors=check_errors)


def _cia_det(A):
    if _hpu_jvp_eligible(A):
        fw_lvl = _dispatch.forward_ad_level(A)
        if fw_lvl >= 0 and _any_dual((A,), fw_lvl):
            Ap, gA = _unpack(A, fw_lvl)
            det = aten._linalg_det.default(Ap)[0]
            if gA is None:
                return det
            invA = _inv_via_solve(Ap)
            dt = torch.where(det == 0, 0.0, det * (invA.mT * gA).sum((-2, -1)))
            return fwAD.make_dual(det, dt, level=fw_lvl)
    return aten._linalg_det.default(A)[0]


def _other_det(A):
    if _hpu_jvp_eligible(A):
        fw_lvl = _dispatch.forward_ad_level(A)
        if fw_lvl >= 0 and _any_dual((A,), fw_lvl):
            Ap, gA = _unpack(A, fw_lvl)
            det, LU, pivots = aten._linalg_det.default(Ap)
            if gA is None:
                return det, LU, pivots
            invA = _inv_via_solve(Ap)
            dt = torch.where(det == 0, 0.0, det * (invA.mT * gA).sum((-2, -1)))
            return fwAD.make_dual(det, dt, level=fw_lvl), LU, pivots
    return aten._linalg_det.default._op_dk(DispatchKey.Autograd, A)


def _cia_slogdet(A):
    if _hpu_jvp_eligible(A):
        fw_lvl = _dispatch.forward_ad_level(A)
        if fw_lvl >= 0 and _any_dual((A,), fw_lvl):
            Ap, gA = _unpack(A, fw_lvl)
            sign, logabsdet, _, _ = aten._linalg_slogdet.default(Ap)
            if gA is None:
                return sign, logabsdet
            invA = _inv_via_solve(Ap)
            dlog = (invA.mT * gA).sum((-2, -1))
            return fwAD.make_dual(sign, torch.zeros_like(sign), level=fw_lvl), fwAD.make_dual(
                logabsdet, dlog, level=fw_lvl
            )
    sign, logabsdet, _, _ = aten._linalg_slogdet.default(A)
    return sign, logabsdet


def _other_slogdet(A):
    if _hpu_jvp_eligible(A):
        fw_lvl = _dispatch.forward_ad_level(A)
        if fw_lvl >= 0 and _any_dual((A,), fw_lvl):
            Ap, gA = _unpack(A, fw_lvl)
            sign, logabsdet, LU, pivots = aten._linalg_slogdet.default(Ap)
            if gA is None:
                return sign, logabsdet, LU, pivots
            invA = _inv_via_solve(Ap)
            dlog = (invA.mT * gA).sum((-2, -1))
            return (
                fwAD.make_dual(sign, torch.zeros_like(sign), level=fw_lvl),
                fwAD.make_dual(logabsdet, dlog, level=fw_lvl),
                LU,
                pivots,
            )
    return aten._linalg_slogdet.default._op_dk(DispatchKey.Autograd, A)


def _install_impls(entries):
    """Register ``(op, key, kernel)`` handlers on ``aten`` and return an undo that restores the dispatcher."""
    library = torch.library.Library("aten", "IMPL")
    touched = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        for op, key, kernel in entries:
            if key != "CompositeImplicitAutograd" and torch._C._dispatch_has_kernel_for_dispatch_key(op.name(), key):
                continue
            library.impl(op, kernel, key)
            touched.append(op)
    for op in touched:
        op._dispatch_cache.clear()

    def undo():
        library._destroy()
        for op in touched:
            op._dispatch_cache.clear()

    return undo


def install_solve_composites(device_type):
    """On ``hpu``, handle ``torch.func.jvp``/``jacfwd`` for ``linalg.solve``/``solve_ex`` above ``AOTAutograd`` so
    ``torch.compile`` never emits ``aten.linalg_lu_solve`` against placeholder ``LU=0``. Returns an undo."""
    if device_type != "hpu":
        return lambda: None
    return _install_impls([
        (aten.linalg_solve.default, "CompositeImplicitAutograd", _cia_solve),
        (aten.linalg_solve_ex.default, "CompositeImplicitAutograd", _cia_solve_ex),
        (aten._linalg_solve_ex.default, "AutogradOther", _other_solve_ex),
    ])


def install_inv_det_composites(device_type):
    """On ``hpu``, handle ``torch.func.jvp``/``jacfwd`` for ``det``/``slogdet`` above ``AOTAutograd`` (see
    ``install_solve_composites``). Returns an undo."""
    if device_type != "hpu":
        return lambda: None
    return _install_impls([
        (aten.linalg_det.default, "CompositeImplicitAutograd", _cia_det),
        (aten.det.default, "CompositeImplicitAutograd", _cia_det),
        (aten._linalg_det.default, "AutogradOther", _other_det),
        (aten.linalg_slogdet.default, "CompositeImplicitAutograd", _cia_slogdet),
        (aten.slogdet.default, "CompositeImplicitAutograd", _cia_slogdet),
        (aten._linalg_slogdet.default, "AutogradOther", _other_slogdet),
    ])
