"""NaN through Gaudi's max/min ops, as torch propagates it.

On Gaudi2 (bridge 1.21.4, eager, lazy and compiled) the max/min family drops NaN: ``amax([nan, 1, 2])`` is ``-inf``
(a leading NaN), ``max(-1)`` returns ``2`` and an index that is not the NaN's, ``maximum(1, nan)`` is ``1``,
``max_pool`` keeps a window's finite max, ``norm(x, inf)`` is finite. A NaN in a model then never reaches the loss:
NaN checks do not fire and the loss stays finite (``repro/probe_reduction_nan.py``). In 16-bit dtypes ``argmax``,
``argmin`` and ``max(dim)`` / ``min(dim)`` also return the index ``n`` (out of range) for a row of ``-inf`` (``+inf``
for the min ops) and for an all-NaN row.

Each op here runs Gaudi's own kernel and then puts NaN where torch has it, with torch's index and torch's gradient:

- ``amax`` / ``amin``, ``max()`` / ``min()`` and ``median()`` of the whole tensor, ``aminmax``, ``norm`` /
  ``linalg.vector_norm`` with ``ord = +-inf``: NaN for each reduced slice holding a NaN (``isnan().any()`` and
  ``where``). The gradient is torch's formula over that result: for ``amax`` a NaN slice's gradient is NaN, for
  ``max()`` / ``median()`` it is split among the NaNs, for the inf-norms it is 0 (torch's ``norm_backward`` with the
  sign of NaN 0, as torch's; Gaudi's sign of NaN is NaN).
- ``max(dim)`` / ``min(dim)`` / ``median(dim)``, ``argmax`` / ``argmin``: the index of the slice's first NaN (an
  ``argmax`` of the NaN mask), as torch, and in 16-bit dtypes index 0 for the out-of-range rows. The values are read
  at that index (``gather``), whose gradient is torch's (the gradient goes to the index).
- ``maximum`` / ``minimum`` (and ``max(a, b)`` / ``min(a, b)``): ``a + b`` where either is NaN (NaN, with torch's
  gradient: the whole gradient to both); ``fmax`` / ``fmin`` the other operand where one is NaN (Gaudi returns the
  first operand, NaN or not); ``clamp`` with tensor bounds: NaN where the input or a bound is NaN (lazy mode drops the
  input's NaN, eager a bound's), with torch's zero gradient there.
- ``max_pool2d`` / ``max_pool3d`` with indices (``F.max_pool1d`` reaches the 2-D op): NaN for each window holding a
  NaN, found by pooling the NaN positions (the last NaN of a window, the one torch picks), and there the index of that
  NaN. Gaudi's indices are window-local (kernel coordinates, row-major), not torch's flat plane indices (a separate
  Gaudi difference, left as is), and its backward reads them: the fixed indices go to Gaudi's own backward, which then
  sends a NaN window's gradient to that NaN, as torch.

- With ``HPU_OPS_EXACT_SORT=1`` (off by default, see ``install``): ``sort`` / ``argsort`` / ``msort`` / ``topk`` /
  ``kthvalue`` with NaN above +inf, as torch: a stable sort of the values with NaN as +inf, then a stable sort of the
  NaN mask. Gaudi's sort returns NaN as +inf with out-of-range indices, and its non-stable sort of 1000+ elements
  with infs returns out-of-range indices even without NaN; its stable sort is right. Eager and lazy only:
  traced by ``torch.compile`` the native op stays (compiled on the card, two of the rewritten sorts came out wrong).

Registered at ``Autograd{}`` only: ``torch.inference_mode()`` calls, below autograd, keep Gaudi's kernels. Integer
and bool dtypes keep the native op (they have no NaN), and so do ``out=`` calls. The native kernels run first, so
their argument errors (empty dims, bad dims) are raised as usual.
"""
import functools
import math
import os

import torch
from torch._C import DispatchKey
from torch._subclasses.fake_tensor import is_fake

from ._dispatch import count, register_kernels

aten = torch.ops.aten
NAN = float("nan")
REDUCED = (torch.float16, torch.bfloat16)
POOL_PLANE_MAX = 1 << 24  # float32 holds every code of a pooled plane exactly up to here


def native(op):
    """``op``'s own kernel below this rewrite: torch's autograd kernel, which records the native op's gradient when
    grad mode is on and reaches Gaudi's kernel. The CPU tests swap this for a stand-in that drops NaN like Gaudi."""
    return functools.partial(op._op_dk, DispatchKey.Autograd)


def native_backward(op):
    """A backward op as dispatched (Gaudi's kernel): called inside a backward, where ``torch.compile``'s tracer
    rejects ``native``'s direct call of the autograd key. The CPU tests swap it too."""
    return op


def _floating(*ts):
    return any(isinstance(t, torch.Tensor) and t.is_floating_point() for t in ts)


def _dims(x, dim):
    """The reduced dims, normalised; every dim for ``None`` or ``[]`` (amax's "all")."""
    if x.dim() == 0:
        return []
    if dim is None or (not isinstance(dim, int) and len(dim) == 0):
        return list(range(x.dim()))
    dims = [dim] if isinstance(dim, int) else list(dim)
    for d in dims:  # torch's own error for a dim out of range (the native op raises it, called later)
        if not -x.dim() <= d < x.dim():
            raise IndexError(f"Dimension out of range (expected to be in range of [{-x.dim()}, {x.dim() - 1}], "
                             f"but got {d})")
    return sorted(d % x.dim() for d in dims)


def _kept(x, dims):
    """``x``'s shape with the reduced ``dims`` as size 1."""
    return [1 if d in dims else s for d, s in enumerate(x.shape)]


def _nan_any(x, dims, keepdim):
    """True for each reduced slice of ``x`` holding a NaN. Reduced with keepdim and reshaped: in lazy mode Gaudi's
    graph of a reduction over all dims without keepdim fails (``repro/lazy_logsumexp_all_dims_squeeze.py``)."""
    m = torch.isnan(x)
    if not dims:
        return m
    m = m.any(dims, keepdim=True)
    return m if keepdim else m.reshape([s for d, s in enumerate(x.shape) if d not in dims])


def _restore(t, x, dims, keepdim):
    """``t`` with the reduced ``dims`` of ``x`` back as size 1 (torch's restore_reduced_dims), by a reshape: in lazy
    mode the backward of ``amax()`` over all dims with unsqueezes fails ("incorrect input sizes for Copy")."""
    return t if keepdim or x.dim() == 0 else t.reshape(_kept(x, dims))


def _grad_needed(*ts):
    return torch.is_grad_enabled() and any(isinstance(t, torch.Tensor) and t.requires_grad for t in ts)


# ---- amax / amin: NaN slices, torch's scale_grad_by_count gradient


class _Extreme(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dims, keepdim, fn):
        r = _extreme(x, dims, keepdim, fn)
        ctx.save_for_backward(x, r)
        ctx.dims, ctx.keepdim = dims, keepdim
        return r

    @staticmethod
    def backward(ctx, grad):
        x, r = ctx.saved_tensors
        g, r = _restore(grad, x, ctx.dims, ctx.keepdim), _restore(r, x, ctx.dims, ctx.keepdim)
        mask = r == x  # a NaN slice matches nowhere: its gradient is 0 / 0 * 0, NaN, as torch's
        return (g / mask.sum(ctx.dims, keepdim=True)) * mask, None, None, None


def _extreme(x, dims, keepdim, fn):
    r = fn(x, dims, keepdim)  # first: the native op raises its own argument errors
    return torch.where(_nan_any(x, dims, keepdim), NAN, r)


def _amax_kernel(op, name):
    def kernel(self, dim=(), keepdim=False):
        fn = native(op)
        if not _floating(self):
            return fn(self, dim, keepdim)
        count(name)
        dims = _dims(self, dim)
        call = lambda x, _dims, k: fn(x, dim, k)  # noqa: E731  (the native op with the caller's dim)
        if _grad_needed(self):
            return _Extreme.apply(self, dims, keepdim, call)
        return _extreme(self, dims, keepdim, call)

    return kernel


# ---- max() / min() / median() of the whole tensor: torch's evenly_distribute_backward


class _Whole(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, fn):
        r = fn(x)
        r = torch.where(torch.isnan(x).any(), NAN, r)
        ctx.save_for_backward(x, r)
        return r

    @staticmethod
    def backward(ctx, grad):
        x, r = ctx.saved_tensors
        mask = (x == r) | (torch.isnan(x) & torch.isnan(r))  # a NaN result: split among the NaNs
        return mask * (grad / mask.sum()), None


def _whole_kernel(op, name):
    def kernel(self):
        fn = native(op)
        if not _floating(self):
            return fn(self)
        count(name)
        if _grad_needed(self):
            return _Whole.apply(self, fn)
        r = fn(self)
        return torch.where(torch.isnan(self).any(), NAN, r)

    return kernel


# ---- max(dim) / min(dim) / median(dim), argmax / argmin: the first NaN's index


def _fixed_index(x, d, i):
    """``i`` (keepdim) with each NaN slice's first NaN, and in 16-bit dtypes index 0 where Gaudi returns ``n``."""
    m = torch.isnan(x)
    first = native(aten.argmax.default)(m.to(torch.uint8), d, True)  # Gaudi's argmax: the first among ties
    bad = m.any(d, keepdim=True)
    if x.dtype in REDUCED:
        bad = bad | (i >= x.shape[d])  # a row of -inf (+inf for min), where ``first`` is 0
    return torch.where(bad, first, i)


def _select_kernel(op, name):
    def kernel(self, dim, keepdim=False):
        fn = native(op)
        if not _floating(self):
            return fn(self, dim, keepdim)
        if self.dim() == 0:  # as the one-element vector (the native op checks ``dim``)
            fn(self, dim, keepdim)
            values, i = kernel(self.reshape(1), 0, False)
            return values.reshape(()), i.reshape(())
        d = _dims(self, dim)[0]
        with torch.no_grad():
            _, i = fn(self, d, True)
            i = _fixed_index(self, d, i)
        count(name)
        values = self.gather(d, i)  # autograd: the gradient to the index, torch's value_selecting backward
        if not keepdim:
            values, i = values.squeeze(d), i.squeeze(d)
        return values, i

    return kernel


def _arg_kernel(op, name):
    def kernel(self, dim=None, keepdim=False):
        fn = native(op)
        if not _floating(self) or self.dim() == 0 or self.numel() == 0:
            return fn(self, dim, keepdim)
        count(name)
        x = self.reshape(-1) if dim is None else self
        d = 0 if dim is None else _dims(self, dim)[0]
        i = _fixed_index(x, d, fn(x, d, True))
        if dim is None:
            return i.reshape([1] * self.dim()) if keepdim else i.reshape(())
        return i if keepdim else i.squeeze(d)

    return kernel


# ---- aminmax (torch has no derivative: the native op's autograd node raises it)


def _aminmax_kernel(op, name):
    def kernel(self, *, dim=None, keepdim=False):
        fn = native(op)
        lo, hi = fn(self, dim=dim, keepdim=keepdim)
        if not _floating(self):
            return lo, hi
        count(name)
        m = _nan_any(self, _dims(self, dim), keepdim)
        return torch.where(m, NAN, lo), torch.where(m, NAN, hi)

    return kernel


# ---- inf-norms: NaN slices; torch's norm_backward, whose sign of NaN is 0 (Gaudi's sign of NaN is NaN)


def _inf(p):
    return p is not None and not isinstance(p, (complex, str)) and math.isinf(float(p))


class _InfNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dims, keepdim, fn):
        r = fn(x)
        r = torch.where(_nan_any(x, dims, keepdim), NAN, r)
        ctx.save_for_backward(x, r)
        ctx.dims, ctx.keepdim = dims, keepdim
        return r

    @staticmethod
    def backward(ctx, grad):
        x, r = ctx.saved_tensors
        g, r = _restore(grad, x, ctx.dims, ctx.keepdim), _restore(r, x, ctx.dims, ctx.keepdim)
        a = x.abs()
        mask = (a == r) | torch.isnan(a)
        sign = (x > 0).to(x.dtype) - (x < 0).to(x.dtype)  # torch's sgn of a real NaN is 0
        return sign * ((g / mask.sum(ctx.dims, keepdim=True)) * mask), None, None, None


def _inf_norm(self, dim, keepdim, fn, name):
    count(name)
    dims = _dims(self, dim)
    if _grad_needed(self):
        return _InfNorm.apply(self, dims, keepdim, fn)
    r = fn(self)
    return torch.where(_nan_any(self, dims, keepdim), NAN, r)


def _vector_norm_kernel(op, name):
    def kernel(self, ord=2, dim=None, keepdim=False, *, dtype=None):
        fn = native(op)
        if not _inf(ord) or not _floating(self) or self.is_complex() or dtype not in (None, self.dtype):
            return fn(self, ord, dim, keepdim, dtype=dtype)
        return _inf_norm(self, dim, keepdim, lambda x: fn(x, ord, dim, keepdim), name)

    return kernel


def _norm_kernel(op, name, with_dim):
    def kernel(self, p=2, dim=None, keepdim=False):
        fn = native(op)
        args = (p, dim, keepdim) if with_dim else (p,)
        if not _inf(p) or not _floating(self) or self.is_complex():
            return fn(self, *args)
        return _inf_norm(self, dim if with_dim else None, keepdim if with_dim else False,
                         lambda x: fn(x, *args), name)

    return kernel


# ---- binary: maximum / minimum, fmax / fmin, clamp with tensor bounds


def _binary(a, b, fn):
    return torch.where(torch.isnan(a) | torch.isnan(b), a + b, fn(a, b))


class _Binary(torch.autograd.Function):
    """``maximum`` / ``minimum`` with NaN, and torch's backward formula, which at a NaN (no comparison holds) gives
    the whole gradient to both operands: one backward instead of the native one plus the composite's."""

    @staticmethod
    def forward(ctx, a, b, fn, largest):
        ctx.save_for_backward(a, b)
        ctx.largest = largest
        return _binary(a, b, fn)

    @staticmethod
    def backward(ctx, grad):
        a, b = ctx.saved_tensors
        half = torch.where(a == b, grad / 2, grad)
        lose_a, lose_b = (a < b, a > b) if ctx.largest else (a > b, a < b)
        ga = gb = None
        if ctx.needs_input_grad[0]:
            ga = half.masked_fill(lose_a, 0).sum_to_size(a.shape).to(a.dtype)
        if ctx.needs_input_grad[1]:
            gb = half.masked_fill(lose_b, 0).sum_to_size(b.shape).to(b.dtype)
        return ga, gb, None, None


def _binary_kernel(op, name, largest):
    def kernel(self, other):
        fn = native(op)
        if not _floating(self, other):
            return fn(self, other)
        count(name)
        if _grad_needed(self, other) and self.device == other.device:
            return _Binary.apply(self, other, fn, largest)
        return _binary(self, other, fn)

    return kernel


def _fbinary_kernel(op, name):
    def kernel(self, other):
        out = native(op)(self, other)
        if not out.is_floating_point():
            return out
        count(name)
        # torch: the other operand where one is NaN; both NaN gives self (NaN), whose gradient torch keeps
        return torch.where(torch.isnan(other), self, torch.where(torch.isnan(self), other, out))

    return kernel


def _clamp_kernel(op, name, arity):
    def kernel(self, *bounds):
        out = native(op)(self, *bounds)
        if not _floating(out):
            return out
        count(name)
        m = torch.isnan(self)
        for b in bounds[:arity]:
            if isinstance(b, torch.Tensor):
                m = m | torch.isnan(b)
        return torch.where(m, NAN, out)  # a constant: torch's gradient there is 0 for the input and the bounds

    return kernel


# ---- max_pool2d / max_pool3d with indices


def _per_dim(v, nd, default=None):
    v = default if (v is None or (not isinstance(v, int) and len(v) == 0)) else v
    v = [v] if isinstance(v, int) else list(v)
    return v * nd if len(v) == 1 else v


def _codes(sizes, kernel, dilation, offset, device):
    """``sum_j ((offset_j + arange(sizes_j)) div dilation_j) * prod(kernel[j+1:])`` over the grid ``sizes``, float32.
    Within one pooling window it is the element's window-local index plus a constant of the window (row-major in
    kernel coordinates, each lower term below its kernel size), so its max over a window is the window's last
    element in scan order. Exact: below 2**24."""
    code = torch.zeros((), dtype=torch.float32, device=device)
    nd = len(sizes)
    for j in range(nd):
        a = torch.arange(sizes[j], dtype=torch.float32, device=device) * offset[j][0] + offset[j][1]
        term = torch.floor(a / dilation[j]) * math.prod(kernel[j + 1:])
        code = code + term.reshape([-1 if t == j else 1 for t in range(nd)])
    return code


def _codes_exact(spatial, kernel, nd):
    """True when every window code is below 2**24 (exact in float32)."""
    kernel = _per_dim(kernel, nd)
    return sum(size * math.prod(kernel[j + 1:]) for j, size in enumerate(spatial)) < POOL_PLANE_MAX


def _pool(x, args, fn, nd):
    """Gaudi's pool with NaN for each window holding a NaN and, there, the index of the window's last NaN (torch's
    pick), in Gaudi's window-local form: kernel coordinates, row-major (measured, eager and lazy, with stride,
    padding, dilation and 3-D)."""
    out, idx = fn(x, *args)
    kernel, stride, padding, dilation, _ = args
    kernel = _per_dim(kernel, nd)
    stride, padding, dilation = _per_dim(stride, nd, kernel), _per_dim(padding, nd), _per_dim(dilation, nd)
    spatial = x.shape[-nd:]
    code = _codes(spatial, kernel, dilation, [(1, 0)] * nd, x.device)
    marked = torch.where(torch.isnan(x), code, -1.0)  # NaN codes, -1 elsewhere: no NaN to drop
    last = fn(marked, *args)[0]  # each window's last NaN, -1 for none (padding pools as -inf)
    has = last >= 0
    start = _codes(out.shape[-nd:], kernel, dilation, [(stride[j], -padding[j]) for j in range(nd)], x.device)
    local = (last - start).to(torch.int64)
    return torch.where(has, NAN, out), torch.where(has, local, idx)


class _Pool(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, args, fn, bwd, nd):
        out, idx = _pool(x, args, fn, nd)
        ctx.save_for_backward(x, idx)
        ctx.args, ctx.bwd = args, bwd
        ctx.mark_non_differentiable(idx)
        return out, idx

    @staticmethod
    def backward(ctx, grad, _gidx):
        x, idx = ctx.saved_tensors
        # Gaudi's own backward over the fixed indices: a NaN window's gradient goes to its last NaN, as torch's
        return native_backward(ctx.bwd)(grad, x, *ctx.args, idx), None, None, None, None


def _pool_kernel(op, backward_op, name, nd):
    def kernel(self, kernel_size, stride=(), padding=0, dilation=1, ceil_mode=False):
        fn = native(op)
        args = (kernel_size, stride, padding, dilation, ceil_mode)
        if not _floating(self) or self.dim() < nd or not _codes_exact(self.shape[-nd:], kernel_size, nd):
            return fn(self, *args)
        count(name)
        if _grad_needed(self):
            return _Pool.apply(self, args, fn, backward_op, nd)
        with torch.no_grad():
            return _pool(self, args, fn, nd)

    return kernel


# ---- sort / topk / kthvalue: NaN greatest, as torch


def _sorted(x, d, descending):
    """``x`` sorted along ``d`` with NaN above +inf (torch's order), and the indices. Gaudi's sort returns NaN as
    +inf with an out-of-range index, and its non-stable sort of 1000+ elements with infs returns out-of-range indices
    (measured, also without NaN); its stable sort is right. So: a stable sort of the values with NaN as +inf, then a
    stable sort of the NaN mask, which puts the NaNs after the infs (before them when descending) and keeps the rest
    in order."""
    sort = native(aten.sort.stable)
    with torch.no_grad():
        nan = torch.isnan(x)
        _, i = sort(torch.where(nan, float("inf"), x), stable=True, dim=d, descending=descending)
        # gather of float data: Gaudi's graph compile of a bool (and int64) gather fails ("Can not compile graph")
        _, j = sort(nan.to(torch.float32).gather(d, i), stable=True, dim=d, descending=descending)
        i = i.to(torch.float32).gather(d, j).to(torch.int64)  # float data again; indices < 2**24 are exact
    return x.gather(d, i), i  # autograd: the gradient to the index, torch's backward of sort


def _concrete(t):
    """True for a real tensor outside any tracer."""
    return not is_fake(t) and torch._guards.detect_fake_mode() is None and not torch._is_functional_tensor(t)


def _sortable(x):
    # not under torch.compile: compiled on the card, sort(dim=0, descending) and msort came out wrong (measured)
    return _floating(x) and x.dim() > 0 and x.numel() > 0 and max(x.shape) < POOL_PLANE_MAX and _concrete(x)


def _sort_kernel(op, name, overload_stable):
    def kernel(self, *args, **kwargs):
        if overload_stable:
            dim, descending = kwargs.get("dim", -1), kwargs.get("descending", False)
        else:
            dim = args[0] if len(args) > 0 else kwargs.get("dim", -1)
            descending = args[1] if len(args) > 1 else kwargs.get("descending", False)
        if not _sortable(self):
            return native(op)(self, *args, **kwargs)
        count(name)
        return _sorted(self, _dims(self, dim)[0], descending)

    return kernel


def _topk_kernel(op, name):
    def kernel(self, k, dim=-1, largest=True, sorted=True):
        if not _sortable(self) or not 0 <= k <= self.shape[_dims(self, dim)[0]]:
            return native(op)(self, k, dim, largest, sorted)  # its own error for a k out of range
        count(name)
        d = _dims(self, dim)[0]
        values, i = _sorted(self, d, largest)
        return values.narrow(d, 0, k), i.narrow(d, 0, k)

    return kernel


def _kthvalue_kernel(op, name):
    def kernel(self, k, dim=-1, keepdim=False):
        if not _sortable(self) or not 1 <= k <= self.shape[_dims(self, dim)[0]]:
            return native(op)(self, k, dim, keepdim)
        count(name)
        d = _dims(self, dim)[0]
        values, i = _sorted(self, d, False)
        values, i = values.narrow(d, k - 1, 1), i.narrow(d, k - 1, 1)
        return (values, i) if keepdim else (values.squeeze(d), i.squeeze(d))

    return kernel


def install(device_type, exact_sort=None):
    """NaN propagation of Gaudi's max/min ops on ``device_type``, at its autograd key. Returns an undo. The sort
    family (``sort``, ``argsort``, ``msort``, ``topk``, ``kthvalue``) only with ``exact_sort`` (default: the
    environment's ``HPU_OPS_EXACT_SORT=1``): measured eager, it made the training steps of the SPD models up to 22%
    slower (hpu-ops' own eigh and SVD sort their eigenvalues; TensorCSPNet 205 to 250 ms), lazy within noise."""
    if exact_sort is None:
        exact_sort = os.environ.get("HPU_OPS_EXACT_SORT") == "1"
    k = []
    for op, name in ((aten.amax.default, "amax"), (aten.amin.default, "amin")):
        k.append((op, "Autograd{}", _amax_kernel(op, name)))
    for op, name in ((aten.max.default, "max"), (aten.min.default, "min"), (aten.median.default, "median")):
        k.append((op, "Autograd{}", _whole_kernel(op, name)))
    for op, name in ((aten.max.dim, "max_dim"), (aten.min.dim, "min_dim"), (aten.median.dim, "median_dim")):
        k.append((op, "Autograd{}", _select_kernel(op, name)))
    for op, name in ((aten.argmax.default, "argmax"), (aten.argmin.default, "argmin")):
        k.append((op, "Autograd{}", _arg_kernel(op, name)))
    k.append((aten.aminmax.default, "Autograd{}", _aminmax_kernel(aten.aminmax.default, "aminmax")))
    k.append((aten.linalg_vector_norm.default, "Autograd{}",
              _vector_norm_kernel(aten.linalg_vector_norm.default, "norm_inf")))
    k.append((aten.norm.ScalarOpt_dim, "Autograd{}", _norm_kernel(aten.norm.ScalarOpt_dim, "norm_inf", True)))
    k.append((aten.norm.Scalar, "Autograd{}", _norm_kernel(aten.norm.Scalar, "norm_inf", False)))
    for op, name, largest in ((aten.maximum.default, "maximum", True), (aten.minimum.default, "minimum", False)):
        k.append((op, "Autograd{}", _binary_kernel(op, name, largest)))
    for op, name in ((aten.fmax.default, "fmax"), (aten.fmin.default, "fmin")):
        k.append((op, "Autograd{}", _fbinary_kernel(op, name)))
    for op, arity in ((aten.clamp.Tensor, 2), (aten.clamp_min.Tensor, 1), (aten.clamp_max.Tensor, 1)):
        k.append((op, "Autograd{}", _clamp_kernel(op, "clamp_tensor", arity)))
    for op, bwd, nd in ((aten.max_pool2d_with_indices.default, aten.max_pool2d_with_indices_backward.default, 2),
                        (aten.max_pool3d_with_indices.default, aten.max_pool3d_with_indices_backward.default, 3)):
        k.append((op, "Autograd{}", _pool_kernel(op, bwd, f"max_pool{nd}d", nd)))
    if exact_sort:
        k.append((aten.sort.default, "Autograd{}", _sort_kernel(aten.sort.default, "sort", False)))
        k.append((aten.sort.stable, "Autograd{}", _sort_kernel(aten.sort.stable, "sort", True)))
        k.append((aten.topk.default, "Autograd{}", _topk_kernel(aten.topk.default, "topk")))
        k.append((aten.kthvalue.default, "Autograd{}", _kthvalue_kernel(aten.kthvalue.default, "kthvalue")))
    return register_kernels(device_type, k)
