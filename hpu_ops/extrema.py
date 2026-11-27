"""NaN through Gaudi's max/min ops, as torch propagates it.

On Gaudi2 (bridge 1.21.4, eager, lazy and compiled) the max/min family drops NaN: ``amax([nan, 1, 2])`` is ``-inf``
(a leading NaN), ``max(-1)`` returns ``2`` and an index that is not the NaN's, ``maximum(1, nan)`` is ``1``,
``max_pool`` keeps a window's finite max, ``norm(x, inf)`` is finite. A NaN in a model then never reaches the loss:
NaN checks do not fire and the loss stays finite (``repro/probe_reduction_nan.py``). In 16-bit dtypes ``argmax``,
``argmin`` and ``max(dim)`` / ``min(dim)`` also return an index out of range for a row of ``-inf`` (``+inf`` for the
min ops) and an all-NaN row: ``n`` eager, garbage such as INT64_MIN lazy.

Each op runs Gaudi's own kernel and then puts NaN where torch has it, with torch's index and torch's gradient:

- ``amax`` / ``amin``, ``max()`` / ``min()`` and ``median()`` of the whole tensor, ``aminmax``, ``norm`` /
  ``linalg.vector_norm`` with ``ord = +-inf``: NaN for each reduced slice holding a NaN (``isnan().any()`` and
  ``where``).
- ``max(dim)`` / ``min(dim)`` / ``median(dim)``, ``argmax`` / ``argmin``: the index of the slice's first NaN (an
  ``argmax`` of the NaN mask), as torch, and index 0 where Gaudi's index is out of range. The values are read at that
  index.
- ``maximum`` / ``minimum`` (and ``max(a, b)`` / ``min(a, b)``): ``a + b`` where either is NaN; ``fmax`` / ``fmin``
  the other operand where one is NaN (Gaudi returns the first operand, NaN or not); ``clamp`` / ``clamp_min`` /
  ``clamp_max`` with tensor bounds, in place too: NaN where the input or a bound is NaN (lazy mode drops the input's
  NaN, eager a bound's).
- ``max_pool2d`` / ``max_pool3d`` with indices (``F.max_pool1d`` reaches the 2-D op): NaN for each window holding a
  NaN, found by pooling NaN codes (the last NaN of a window, the one torch picks), and there the index of that NaN.
  Gaudi's indices are window-local (kernel coordinates, row-major), not torch's flat plane indices (a separate Gaudi
  difference, left as is), and its backward reads them: with the fixed indices Gaudi's own backward sends a NaN
  window's gradient to that NaN, as torch.

Two layers. At the device's backend key (``HPU``), for every call, inference mode and anything else below autograd
included: the values above, reaching Gaudi through each op's ``.out`` overload (Habana's lazy kernels of the
functional ops are replaced: the dispatcher cannot call a kernel it no longer routes to). torch's own autograd
formulas then run over the fixed results, which gives torch's gradients (amax: NaN; max(dim): to the first NaN;
maximum: both operands; clamp: torch's comparisons). At ``Autograd{}``: what ``torch.compile`` traces (its fake
tensors never reach a backend kernel), the same fixes as a composite; outside a tracer these kernels hand the call to
torch's autograd kernel, so nothing is fixed twice. ``max()`` / ``min()`` / ``median()`` keep their own backward
(torch's reads the NaN flag on the host, a sync per call) and the inf-norms theirs (torch's formula with Gaudi's
``sign(NaN) = NaN`` gives NaN where torch gives 0). ``out=`` calls are computed as the functional op and copied
out (with autograd on; in inference mode Gaudi's ``.out`` kernels run). A graph traced under ``inference_mode()``
(autograd's keys excluded, so neither layer sees it) gets the composites through each op's
``CompositeImplicitAutograd`` Python kernel, which the tracers' ``decompose()`` reads (``_inference_traced``). The
whole-tensor reductions and inf-norms define a ``jvp`` (forward-mode AD); under ``torch.func`` transforms torch's
autograd kernel runs over the fixed result. A CPU 0-d operand with a device tensor (``maximum(torch.tensor(0.),
x)``) is moved to the device first. Integer and bool dtypes keep the native
result. A NaN scalar bound of ``clamp``
(``clamp(x, min=nan)``) is not rewritten: Gaudi gives FLT_MAX / inf (torch: ``x`` for one NaN bound, NaN for two).

- ``sort`` / ``argsort`` / ``msort`` / ``topk`` / ``kthvalue`` (on by default, off with ``HPU_OPS_EXACT_SORT=0``):
  torch's order, NaN above +inf, and indices in range. Gaudi's float sort reads NaN as +inf with an out-of-range
  index, and its non-stable sort pads each row to a power of two with +inf (descending: -inf), so a real infinity
  ties with the padding and its index comes back out of range, at any n that is not a power of two (measured n = 3
  to 5000); its descending sort along a non-last dim returns indices that are no permutation for rows holding -max.
  sort: two sorts of finite keys along the last dim (``_sorted``); topk / kthvalue: Gaudi's own op on one finite key
  (``_select_key``; exact up to the three largest finite magnitudes, which tie). Every eager call, inference mode and
  ``out=`` included, at the ``BackendSelect`` key (``_select_kernels``: Gaudi reached through the ``.out``
  overloads at the backend key); traced by ``torch.compile`` (on the card: right in 390 of 390 cases), the composite
  at ``Autograd{}``. hpu-ops' own sorts of eigenvalues, singular values and quantile keys use ``internal_sort``
  (Gaudi's sort of ``nan_to_num(x)``: indices in range, NaN kept). A dim longer than ``SORT_MAXN`` keeps Gaudi's
  op (one warning); a stable sort past what Gaudi's stable kernel takes falls back to the CPU in Habana's bridge, as
  on main.
"""
import functools
import math
import os
import threading

import torch
from torch._C import DispatchKey

from ._dispatch import REDUCED  # noqa: F401
from ._dispatch import chain, concrete as _concrete, count, override_kernels, register_kernels, replace_py_kernels

aten = torch.ops.aten
NAN = float("nan")
POOL_PLANE_MAX = 1 << 24  # float32 holds every code of a pooled plane exactly up to here


def native(op):
    """``op``'s own kernel below this rewrite: torch's autograd kernel, which records the native op's gradient when
    grad mode is on and reaches Gaudi's kernel. The CPU tests swap this for a stand-in that drops NaN like Gaudi."""
    return functools.partial(op._op_dk, DispatchKey.Autograd)


def native_out(op, device_type):
    """``op`` (an ``.out`` overload) at the device's backend key: Gaudi's kernel, which the backend kernels here call
    (they replace the functional ops' kernels). The CPU tests swap it for a stand-in that drops NaN like Gaudi."""
    return functools.partial(op._op_dk, getattr(DispatchKey, device_type.upper()))


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


def _traced(args, kwargs):
    """True when ``torch.compile`` traces the call: its fake / functional tensors never reach a backend kernel."""
    ts = [a for a in list(args) + list(kwargs.values()) if isinstance(a, torch.Tensor)]
    return any(not _concrete(t) for t in ts)


def _onto_device(args, keep_self=False):
    """``args`` with each CPU 0-d tensor moved to the device of the others (``maximum(torch.tensor(0.), x_hpu)``,
    which torch takes): Gaudi's eager kernels below autograd get it as a wrapped Float number, which the dispatcher
    cannot hand to the Python backend kernels ("Missing cases in 'toPyObject' wrapped number handling", measured,
    bridge 1.21.4). A 0-d tensor on the device promotes as on the CPU; ``to`` passes its gradient back. ``keep_self``
    (in-place ops): ``self`` stays, so a CPU 0-d ``self`` with a device bound gets torch's device error, not a copy."""
    dev = next((a.device for a in args if isinstance(a, torch.Tensor) and a.device.type != "cpu"), None)
    if dev is None:
        return args
    return tuple(a.to(dev) if isinstance(a, torch.Tensor) and a.device.type == "cpu" and a.dim() == 0
                 and not (keep_self and i == 0) else a for i, a in enumerate(args))


def _eager_native(op, composite):
    """At ``Autograd{}``: outside a tracer, torch's autograd kernel of ``op`` (over the backend kernel's fixed result:
    torch's gradient); traced, ``composite``."""
    def kernel(*args, **kwargs):
        if _traced(args, kwargs) and not getattr(_TRACING, "active", False):
            # a CPU 0-d operand stays a graph scalar (moved, it adds a copy to every step)
            return composite(*args, **kwargs)
        args = _onto_device(args, keep_self=op._schema.is_mutable)
        if _BACKEND.get(_device(args)):
            return native(op)(*args, **kwargs)
        return composite(*args, **kwargs)

    return kernel


_BACKEND = {}  # device type -> True while the backend kernels are installed


def _device(args):
    for a in args:
        if isinstance(a, torch.Tensor):
            return a.device.type
    return None


def _grad_needed(*ts):
    return torch.is_grad_enabled() and any(isinstance(t, torch.Tensor) and t.requires_grad for t in ts)


def _functorch():
    """Whether a ``torch.func`` transform (grad, vjp, jacrev, vmap, ...) is active."""
    return torch._C._functorch.peek_interpreter_stack() is not None


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
    """With a ``jvp`` (torch's evenly_read_jvp), so forward-mode AD runs through it. Under a ``torch.func`` transform
    the kernels do not apply it (``_functorch``): a Function applied from inside a dispatcher kernel has no kernel
    there (``custom_function_call``), so torch's autograd kernel runs over the fixed result."""

    @staticmethod
    def forward(x, fn):
        r = fn(x)
        if not (_BACKEND.get(x.device.type) and _concrete(x)):  # else the backend kernel fixed it
            r = torch.where(torch.isnan(x).any(), NAN, r)
        return r

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.save_for_backward(inputs[0], output)
        ctx.save_for_forward(inputs[0], output)

    @staticmethod
    def backward(ctx, grad):
        x, r = ctx.saved_tensors
        mask = (x == r) | (torch.isnan(x) & torch.isnan(r))  # a NaN result: split among the NaNs
        return mask * (grad / mask.sum()), None

    @staticmethod
    def jvp(ctx, tangent, _):
        x, r = ctx.saved_tensors
        mask = x == r  # torch's evenly_read_jvp: a NaN result matches nowhere, 0 / 0 gives NaN
        return (mask * (tangent / mask.sum())).sum()


def _whole_kernel(op, name):
    def kernel(self):
        fn = native(op)
        if not _floating(self):
            return fn(self)
        count(name)
        if _grad_needed(self) and not _functorch():
            return _Whole.apply(self, fn)
        r = fn(self)
        if _BACKEND.get(self.device.type) and _concrete(self):
            return r
        return torch.where(torch.isnan(self).any(), NAN, r)

    return kernel


# ---- max(dim) / min(dim) / median(dim), argmax / argmin: the first NaN's index


def _fixed_index(x, d, i, argmax=None):
    """``i`` (keepdim) with each NaN slice's first NaN, and index 0 where Gaudi's index is out of range (16-bit rows
    of -inf, +inf for min: ``n`` eager, garbage such as INT64_MIN lazy; ``first`` is 0 there)."""
    m = torch.isnan(x)
    argmax = argmax or native(aten.argmax.default)
    first = argmax(m.to(torch.uint8), d, True)  # Gaudi's argmax: the first among ties (measured)
    bad = m.any(d, keepdim=True) | (i < 0) | (i >= x.shape[d])
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


def _sign(x):
    return (x > 0).to(x.dtype) - (x < 0).to(x.dtype)  # torch's sgn of a real NaN is 0 (Gaudi's is NaN)


class _InfNorm(torch.autograd.Function):
    """As ``_Whole``: torch's norm_backward and linalg_vector_norm_jvp at ``ord = +-inf``. Under a ``torch.func``
    transform torch's own backward runs (on Gaudi, whose ``sign(NaN)`` is NaN, a NaN input's gradient is NaN there,
    torch's 0)."""

    @staticmethod
    def forward(x, dims, keepdim, fn):
        r = fn(x)
        if not (_BACKEND.get(x.device.type) and _concrete(x)):  # else the backend kernel fixed it
            r = torch.where(_nan_any(x, dims, keepdim), NAN, r)
        return r

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, dims, keepdim, _ = inputs
        ctx.save_for_backward(x, output)
        ctx.save_for_forward(x, output)
        ctx.dims, ctx.keepdim = dims, keepdim

    @staticmethod
    def backward(ctx, grad):
        x, r = ctx.saved_tensors
        g, r = _restore(grad, x, ctx.dims, ctx.keepdim), _restore(r, x, ctx.dims, ctx.keepdim)
        a = x.abs()
        mask = (a == r) | torch.isnan(a)
        return _sign(x) * ((g / mask.sum(ctx.dims, keepdim=True)) * mask), None, None, None

    @staticmethod
    def jvp(ctx, tangent, *_):
        x, r = ctx.saved_tensors
        r = _restore(r, x, ctx.dims, ctx.keepdim)
        mask = ((x.abs() == r) | (torch.isnan(x) & torch.isnan(r))).to(x.dtype)
        dims = ctx.dims if ctx.dims else list(range(x.dim()))
        out = (_sign(x) * tangent * mask).sum(dims, keepdim=True) / mask.sum(dims, keepdim=True)
        return out if ctx.keepdim or x.dim() == 0 else out.reshape(_reduced(x, dims, False))


def _inf_norm(self, dim, keepdim, fn, name):
    dims = _dims(self, dim)
    if _grad_needed(self) and not _functorch():
        return _InfNorm.apply(self, dims, keepdim, fn)
    r = fn(self)
    if _BACKEND.get(self.device.type) and _concrete(self):
        return r
    count(name)
    return torch.where(_nan_any(self, dims, keepdim), NAN, r)


def _vector_norm_kernel(op, name):
    def kernel(self, ord=2, dim=None, keepdim=False, *, dtype=None):
        if self.device.type == "hpu" and _traced((self,), {}):
            # Traced (CompositeImplicitAutograd): the native call is a PyCapsule Dynamo
            # cannot trace (graph break, #115). Declined calls keep the aten op;
            # the inf-norm decomposition calls the aten op, which the tracer records.
            if not _inf(ord) or not _floating(self) or self.is_complex():
                return NotImplemented
            fn = lambda x, o, d, k, dtype=None: aten.linalg_vector_norm.default(x, o, d, k, dtype=dtype)  # noqa: E731
        else:
            fn = native(op)
            if not _inf(ord) or not _floating(self) or self.is_complex():
                return fn(self, ord, dim, keepdim, dtype=dtype)
        if dtype not in (None, self.dtype):  # the inf-norm is the largest |x|: exact in either dtype
            self = self.to(dtype)
        return _inf_norm(self, dim, keepdim, lambda x: fn(x, ord, dim, keepdim), name)

    return kernel


def _norm_kernel(op, name, with_dim):
    def kernel(self, p=2, dim=None, keepdim=False):
        args = (p, dim, keepdim) if with_dim else (p,)
        if self.device.type == "hpu" and _traced((self,), {}):
            # Traced: as _vector_norm_kernel - declined calls keep the aten op (#115)
            if not _inf(p) or not _floating(self) or self.is_complex():
                return NotImplemented
            fn = lambda x, *a: op(x, *a)  # noqa: E731  the aten op, recorded by the tracer
        else:
            fn = native(op)
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
        # out + NaN, not a NaN constant: the gradient still reaches torch's formula, which compares the raw operands
        # (clamp(5, min=nan, max=1): max's gradient 1, as torch's)
        return torch.where(m, out + NAN, out)

    return kernel


def _check_inplace(self, r):
    """torch's in-place rule: the result type must cast to ``self``'s (``arange(5).clamp_(min=full((5,), 1.5))``
    raises, it does not truncate)."""
    if not torch.can_cast(r.dtype, self.dtype):
        raise RuntimeError(f"result type {_TYPE_NAMES.get(r.dtype, r.dtype)} can't be cast to the desired output "
                           f"type {_TYPE_NAMES.get(self.dtype, self.dtype)}")
    return r


_TYPE_NAMES = {torch.float64: "Double", torch.float32: "Float", torch.float16: "Half", torch.bfloat16: "BFloat16",
               torch.int64: "Long", torch.int32: "Int", torch.int16: "Short", torch.int8: "Char",
               torch.uint8: "Byte", torch.bool: "Bool"}


def _clamp_inplace_kernel(functional, name):
    def kernel(self, *bounds):
        r = functional(self, *bounds)  # the rewritten functional op (traced: the composite)
        return self.copy_(_check_inplace(self, r))

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


SORT_MAXN = (1 << 24) // 3  # the second key (positions + 3 class offsets of n) is exact in float32 up to here
_BACKEND_KEYS = {"cpu": DispatchKey.CPU, "hpu": DispatchKey.HPU, "meta": DispatchKey.Meta}


def _backend_key(device_type):
    return _BACKEND_KEYS.get(device_type) or getattr(DispatchKey, device_type.upper())


@functools.lru_cache(maxsize=None)
def _below(op, device_type):
    """``op`` below ``BackendSelect`` for a tensor on ``device_type``: its kernel at the backend key, or its
    CompositeExplicitAutograd decomposition (whose inner calls pass ``BackendSelect`` again), or the backend's
    fallback (Habana's eager functional sorts have no kernel of their own at ``HPU``)."""
    name = op.name()
    key = _backend_key(device_type)
    if not torch._C._dispatch_has_kernel_for_dispatch_key(name, str(key).split(".")[-1]):
        # Habana's eager functional sort / topk are structured: CompositeExplicitAutogradNonFunctional over the .out
        # kernel (measured: going to HPU's key instead ran Habana's CPU fallback)
        for alias in ("CompositeExplicitAutogradNonFunctional", "CompositeExplicitAutograd"):
            if torch._C._dispatch_has_kernel_for_dispatch_key(name, alias):
                key = getattr(DispatchKey, alias)
                break
    return functools.partial(op._op_dk, key)


def native_backend(op, device_type):
    """``op``'s kernel at ``device_type``'s backend key (Gaudi's sort / topk / kthvalue, below the ``BackendSelect``
    layer here). The CPU tests swap it for a stand-in that returns Gaudi's out-of-range indices."""
    return functools.partial(op._op_dk, _backend_key(device_type))


_OUT_OF = {aten.sort.stable: aten.sort.values_stable, aten.topk.default: aten.topk.values,
           aten.kthvalue.default: aten.kthvalue.values, aten.sort.default: aten.sort.values_stable}


def _primitive(op, x):
    """Gaudi's ``op`` (``sort.stable``, ``sort.default``, ``topk``, ``kthvalue``) for ``x``: for a real tensor its
    ``.out`` overload at the backend key into fresh outputs (below this module's ``BackendSelect`` kernels; the
    ``.out`` kernels are the ones Habana registers at ``HPU`` in eager and lazy mode); when traced the autograd kernel
    (a tracer's fake tensors never reach ``BackendSelect``)."""
    if not _concrete(x):
        return native(op)
    out_op = native_backend(_OUT_OF[op], x.device.type)

    def call(t, *args, **kwargs):
        values, indices = torch.empty(0, dtype=t.dtype, device=t.device), torch.empty(0, dtype=torch.int64,
                                                                                    device=t.device)
        if op is aten.sort.default:  # (dim, descending) positional
            dim, descending = (list(args) + [-1, False][len(args):])[:2]
            return out_op(t, stable=False, dim=dim, descending=descending, values=values, indices=indices)
        return out_op(t, *args, **kwargs, values=values, indices=indices)

    return call


def _held_max(x):
    """The largest finite value of ``x``'s dtype as the card holds it: float64 is stored as float32 there
    (hpu-ops#16), so float64's max would become +inf, which ties with the sort's padding."""
    held = torch.float32 if x.dtype == torch.float64 and x.device.type != "cpu" else x.dtype
    return torch.finfo(held)


def internal_sort(x, dim=-1, descending=False, keys=False):
    """``(values, indices)`` for hpu-ops' own sorts of eigenvalues and singular values (eigh, the SVDs): one Gaudi sort
    of ``nan_to_num(x)`` (finite keys: no infinity ties with the padding, no NaN index out of range), values read
    from ``x`` (a NaN stays NaN: eigh / svd of a matrix with a NaN return NaN where torch's LAPACK raises, not an
    out-of-range gather). Below the exact rewrite: 2 ops more than Gaudi's sort alone, where the exact form takes
    about 9 (2-39 such sorts per SPD step). ``keys``: quantile's NaN-free keys, whose +-inf are deliberate and must
    not tie with +-max (``nan_to_num`` made ``quantile([inf, FLT_MAX], 0)`` NaN): Gaudi's sort of ``x`` as is; quantile
    reads only the positions its padding cannot reach."""
    with torch.no_grad():
        if keys:
            return _primitive(aten.sort.default, x)(x, dim, descending)
        mx = _held_max(x).max
        _, i = _primitive(aten.sort.default, x)(torch.nan_to_num(x, nan=mx, posinf=mx, neginf=-mx), dim, descending)
    return x.gather(dim, i), i


_POSITIONS = {}  # (n, device) -> arange(n) in float32, at most _POSITIONS_MAX, the oldest dropped first
_POSITIONS_MAX = 64


def _positions(n, device, like):
    if not _concrete(like):  # traced: a fresh one (a cached tensor belongs to another fake mode)
        return torch.arange(n, dtype=torch.float32, device=device)
    key = (n, str(device))
    if key not in _POSITIONS:
        if len(_POSITIONS) >= _POSITIONS_MAX:
            _POSITIONS.pop(next(iter(_POSITIONS)))
        _POSITIONS[key] = torch.arange(n, dtype=torch.float32, device=device)
    return _POSITIONS[key]


def _sorted(x, d, descending, stable=False):
    """``x`` sorted along ``d`` in torch's order (NaN above +inf, ascending; first, descending), and the indices.
    Gaudi's float sort reads NaN as +inf with an out-of-range index, and its non-stable sort pads a row to a power of
    two with +inf (descending: -inf): a real infinity ties with the padding and its index comes back out of range
    (measured at n = 3 to 5000). Its stable sort is right without NaN but falls back to the CPU at 1000 elements or
    along a non-last dim (14-52 ms at 22000 rows of 1000; non-stable: 0.1-0.3 ms). So two sorts of finite keys.
    First ``nan_to_num(x)`` (NaN and +-inf as +-max; stable when asked): right but for the order among max, +inf and
    NaN (among -max and -inf). Then the positions ``p`` of that order (a scatter of the first sort's indices)
    shifted by ``n`` per class, read off ``x - key`` (0 for a finite value, +-inf for +-inf, NaN for NaN): ``p + n``
    for +inf, ``p + 2n`` for NaN, ``p - n`` for -inf (mirrored descending). These are unique and finite, so a
    non-stable sort orders them exactly and keeps the first sort's order within a class (stable: the input's). An
    int32 key of the float bits would need one sort, but ``view(torch.int32)`` in a lazy graph converts the value
    instead of reinterpreting its bits (measured: inf to INT_MAX), and in eager it was no cheaper."""
    n = x.shape[d]
    if n == 1:  # Gaudi's bfloat16 sort of a dim of size 1 fails graph compile (measured); torch: x and index 0
        return x.clone(), torch.zeros(x.shape, dtype=torch.int64, device=x.device)
    sort = _primitive(aten.sort.stable, x)
    last = d == x.dim() - 1
    with torch.no_grad():
        # along the last dim: Gaudi's descending sort along another dim returns indices that are no permutation
        # when a row holds -max (measured at n = 8, 22)
        xt = x if last else x.movedim(d, -1).contiguous()
        mx = _held_max(x).max
        key = torch.nan_to_num(xt, nan=mx, posinf=mx, neginf=-mx)
        _, i = sort(key, stable=stable, dim=-1, descending=descending)
        # each element's position in that order (float data, int64 index)
        p = torch.empty(xt.shape, dtype=torch.float32, device=x.device).scatter_(
            -1, i, _positions(n, x.device, x).expand(xt.shape))
        shift = (key - xt) if descending else (xt - key)
        shift = torch.nan_to_num(shift.to(torch.float32), nan=-2.0 * n if descending else 2.0 * n,
                                 posinf=float(n), neginf=-float(n))
        _, i = sort(p + shift, stable=False, dim=-1)
        if not last:
            i = i.movedim(-1, d).contiguous()
    return x.gather(d, i), i  # autograd: the gradient to the index, torch's backward of sort


def _select_key(x):
    """A finite key in torch's order of ``x`` for Gaudi's topk / kthvalue (one call instead of the exact sort's two
    sorts and a scatter): NaN as max, +inf as the float below it (m1), -inf as -m1, finite values clamped to
    [-m2, m2] (m2: the float below m1). No key is infinite, so none ties with the padding; the order is torch's up to
    the three largest finite magnitudes (m2, m1 and max, within 2 ulp of FLT_MAX), which tie at m2 (documented)."""
    info = _held_max(x)
    ulp = 2.0 ** (math.frexp(info.max)[1] - 1) * info.eps
    m1, m2 = info.max - ulp, info.max - 2 * ulp
    key = torch.nan_to_num(x, nan=info.max, posinf=m1, neginf=-m1)
    return torch.where(torch.isfinite(x), x.clamp(-m2, m2), key)


def _selected(x, d, k, largest, op):
    """``(values, indices)`` of ``op`` (``topk`` with ``k`` and ``largest``, or ``kthvalue`` with ``k``) along ``d`` in
    torch's order: Gaudi's op on ``_select_key(x)`` along the last dim, the values read from ``x``."""
    last = d == x.dim() - 1
    with torch.no_grad():
        xt = x if last else x.movedim(d, -1).contiguous()
        key = _select_key(xt)
        if op is aten.topk.default:
            _, i = _primitive(op, x)(key, k, -1, largest, True)
        else:
            _, i = _primitive(op, x)(key, k, -1, True)
        if not last:
            i = i.movedim(-1, d).contiguous()
    return x.gather(d, i), i


_WARNED_LONG = set()


def _sortable(x, dim):
    # traced by torch.compile too: compiled on the card (hpu_backend) the composite is right (measured)
    if not (_floating(x) and x.dim() > 0 and x.numel() > 0):
        return False
    if x.shape[_dims(x, dim)[0]] > SORT_MAXN:
        if x.device.type not in _WARNED_LONG:
            _WARNED_LONG.add(x.device.type)
            import warnings
            warnings.warn(f"hpu_ops: sort / topk / kthvalue along a dim longer than {SORT_MAXN} elements runs "
                          "Gaudi's own kernel (NaN read as +inf, an infinity's index can be out of range)")
        return False
    return True


def _sort_args(overload_stable, args, kwargs):
    """``(dim, descending, stable)`` of a ``sort.stable`` (``overload_stable``) or ``sort.default`` call."""
    if overload_stable:
        stable = bool(args[0] if len(args) > 0 else kwargs.get("stable", False))
        return kwargs.get("dim", -1), kwargs.get("descending", False), stable
    dim = args[0] if len(args) > 0 else kwargs.get("dim", -1)
    descending = args[1] if len(args) > 1 else kwargs.get("descending", False)
    return dim, descending, False


def _sort_fn(self, overload_stable, args, kwargs, name="sort"):
    dim, descending, stable = _sort_args(overload_stable, args, kwargs)
    count(name)
    return _sorted(self, _dims(self, dim)[0], descending, stable)


def _topk_fn(self, k, dim=-1, largest=True, sorted=True, name="topk"):
    count(name)
    d = _dims(self, dim)[0]
    if not _concrete(self):
        # traced: from the exact sort. A graph run through the dispatcher (aot_eager, inductor's fallbacks) sends its
        # nodes through BackendSelect again: the sort of finite keys comes back unchanged there, the topk key would
        # not (a key's max clamps to the float below)
        values, i = _sorted(self, d, largest)
        return values.narrow(d, 0, k), i.narrow(d, 0, k)
    return _selected(self, d, k, largest, aten.topk.default)


def _kthvalue_fn(self, k, dim=-1, keepdim=False, name="kthvalue"):
    count(name)
    d = _dims(self, dim)[0]
    if not _concrete(self):  # traced: from the exact sort (see _topk_fn)
        values, i = _sorted(self, d, False)
        values, i = values.narrow(d, k - 1, 1), i.narrow(d, k - 1, 1)
    else:
        values, i = _selected(self, d, k, False, aten.kthvalue.default)
    return (values, i) if keepdim else (values.squeeze(d), i.squeeze(d))


def _topk_ok(self, k, dim=-1, *_, **__):
    return _sortable(self, dim) and 0 <= k <= self.shape[_dims(self, dim)[0]]


def _kth_ok(self, k, dim=-1, *_, **__):
    return _sortable(self, dim) and 1 <= k <= self.shape[_dims(self, dim)[0]]


def _sort_kernel(op, name, overload_stable):
    """At ``Autograd{}``: traced by ``torch.compile``, the composite; else torch's autograd kernel, whose call below
    reaches the ``BackendSelect`` kernel (``_select_kernels``), which computes it exactly (no rewrite twice)."""
    def kernel(self, *args, **kwargs):
        if not _sortable(self, _sort_args(overload_stable, args, kwargs)[0]) or (
                _concrete(self) and _SELECT.get(self.device.type)):
            return native(op)(self, *args, **kwargs)
        return _sort_fn(self, overload_stable, args, kwargs, name)

    return kernel


def _topk_kernel(op, name):
    def kernel(self, k, dim=-1, largest=True, sorted=True):
        if not _topk_ok(self, k, dim) or (_concrete(self) and _SELECT.get(self.device.type)):
            return native(op)(self, k, dim, largest, sorted)  # its own error for a k out of range
        return _topk_fn(self, k, dim, largest, sorted, name)

    return kernel


def _kthvalue_kernel(op, name):
    def kernel(self, k, dim=-1, keepdim=False):
        if not _kth_ok(self, k, dim) or (_concrete(self) and _SELECT.get(self.device.type)):
            return native(op)(self, k, dim, keepdim)
        return _kthvalue_fn(self, k, dim, keepdim, name)

    return kernel


# ---- the sort family at BackendSelect: every eager call (inference mode, out=, below autograd)


_SELECT = {}  # device type -> True while the BackendSelect kernels are installed for it


def _copy_outs(outs, values, check):
    """``out=`` with torch's rules: ``check(meta_values, meta_indices)`` runs the op's own ``.out`` meta kernel on
    meta copies of the outputs (its dtype errors, its warning for a nonempty output of the wrong shape); then each
    output is resized (silently, the warning is out) and written."""
    meta = [torch.empty_strided(o.shape, o.stride(), dtype=o.dtype, device="meta") for o in outs]
    check(*meta)
    for o, v in zip(outs, values):
        if o.shape != v.shape:
            o.resize_(v.shape)
        o.copy_(v)
    return tuple(outs)


def _meta(x):
    return torch.empty_strided(x.shape, x.stride(), dtype=x.dtype, device="meta")


def _out_dtypes(x, values, indices, kth=False):
    """torch's dtype errors for the sort family's ``out=`` (its CPU kernels check them; the meta kernels do not):
    on a mismatch, the same call on one-element CPU tensors of these dtypes, so the message is torch's own on this
    platform (int64 prints as "long long" on macOS, "long int" on Linux)."""
    if values.dtype == x.dtype and indices.dtype == torch.int64:
        return
    one = torch.zeros(1, dtype=x.dtype)
    v, i = torch.empty(0, dtype=values.dtype), torch.empty(0, dtype=indices.dtype)
    if kth:
        aten.kthvalue.values._op_dk(DispatchKey.CPU, one, 1, -1, False, values=v, indices=i)
    else:
        aten.sort.values_stable._op_dk(DispatchKey.CPU, one, stable=False, dim=-1, descending=False,
                                       values=v, indices=i)
    raise RuntimeError(f"out= of dtypes {values.dtype} / {indices.dtype} for a {x.dtype} input")


def _select_kernels(device_type):
    """``(op, kernel)`` for the ``BackendSelect`` key, which every eager call of an op passes on its way to the backend
    (above it only autograd, ADInplaceOrView and the tracers' keys; inference mode skips autograd, not this). For
    tensors on ``device_type``: the exact form, with Gaudi's own kernels called at the backend key; ``out=``
    overloads computed and copied out. Any other device, dtype or shape: the backend kernel of the op. torch's
    autograd formulas over the result give torch's gradients."""
    def mine(x):
        return (isinstance(x, torch.Tensor) and x.device.type == device_type and _concrete(x)
                and _SELECT.get(device_type, False))

    def sort_default(self, dim=-1, descending=False):
        if mine(self) and _sortable(self, dim):
            return _sort_fn(self, False, (dim, descending), {})
        return _below(aten.sort.default, self.device.type)(self, dim, descending)

    def sort_stable(self, *, stable=None, dim=-1, descending=False):
        if mine(self) and _sortable(self, dim):
            return _sort_fn(self, True, (), dict(stable=stable, dim=dim, descending=descending))
        return _below(aten.sort.stable, self.device.type)(self, stable=stable, dim=dim, descending=descending)

    def sort_values(self, dim=-1, descending=False, *, values, indices):
        if mine(self) and _sortable(self, dim):
            _out_dtypes(self, values, indices)
            return _copy_outs((values, indices), _sort_fn(self, False, (dim, descending), {}),
                              lambda v, i: aten.sort.values_stable._op_dk(
                                  DispatchKey.Meta, _meta(self), stable=False, dim=dim, descending=descending,
                                  values=v, indices=i))
        return _below(aten.sort.values, self.device.type)(self, dim, descending, values=values, indices=indices)

    def sort_values_stable(self, *, stable=None, dim=-1, descending=False, values, indices):
        if mine(self) and _sortable(self, dim):
            _out_dtypes(self, values, indices)
            return _copy_outs((values, indices),
                              _sort_fn(self, True, (), dict(stable=stable, dim=dim, descending=descending)),
                              lambda v, i: aten.sort.values_stable._op_dk(
                                  DispatchKey.Meta, _meta(self), stable=stable, dim=dim, descending=descending,
                                  values=v, indices=i))
        return _below(aten.sort.values_stable, self.device.type)(self, stable=stable, dim=dim, descending=descending,
                                                                 values=values, indices=indices)

    def topk_default(self, k, dim=-1, largest=True, sorted=True):
        if mine(self) and _topk_ok(self, k, dim):
            return _topk_fn(self, k, dim, largest, sorted)
        return _below(aten.topk.default, self.device.type)(self, k, dim, largest, sorted)

    def topk_values(self, k, dim=-1, largest=True, sorted=True, *, values, indices):
        if mine(self) and _topk_ok(self, k, dim):
            _out_dtypes(self, values, indices)
            return _copy_outs((values, indices), _topk_fn(self, k, dim, largest, sorted),
                              lambda v, i: aten.topk.values._op_dk(DispatchKey.Meta, _meta(self), k, dim, largest,
                                                                   sorted, values=v, indices=i))
        return _below(aten.topk.values, self.device.type)(self, k, dim, largest, sorted, values=values, indices=indices)

    def kthvalue_default(self, k, dim=-1, keepdim=False):
        if mine(self) and _kth_ok(self, k, dim):
            return _kthvalue_fn(self, k, dim, keepdim)
        return _below(aten.kthvalue.default, self.device.type)(self, k, dim, keepdim)

    def kthvalue_values(self, k, dim=-1, keepdim=False, *, values, indices):
        if mine(self) and _kth_ok(self, k, dim):
            _out_dtypes(self, values, indices, True)
            return _copy_outs((values, indices), _kthvalue_fn(self, k, dim, keepdim),
                              lambda v, i: aten.kthvalue.values._op_dk(DispatchKey.Meta, _meta(self), k, dim,
                                                                       keepdim, values=v, indices=i))
        return _below(aten.kthvalue.values, self.device.type)(self, k, dim, keepdim, values=values, indices=indices)

    return [(aten.sort.default, sort_default), (aten.sort.stable, sort_stable), (aten.sort.values, sort_values),
            (aten.sort.values_stable, sort_values_stable), (aten.topk.default, topk_default),
            (aten.topk.values, topk_values), (aten.kthvalue.default, kthvalue_default),
            (aten.kthvalue.values, kthvalue_values)]


def _install_select(device_type):
    library = torch.library.Library("aten", "IMPL")
    kernels = _select_kernels(device_type)
    for op, kernel in kernels:
        library.impl(op, kernel, "BackendSelect")
    _SELECT[device_type] = True

    def undo():
        _SELECT.pop(device_type, None)
        library._destroy()

    return undo


# ---- the backend layer: every call (inference mode too), through the ops' .out overloads


def _reduced(x, dims, keepdim):
    return [1 if d in dims else n for d, n in enumerate(x.shape)] if keepdim else \
        [n for d, n in enumerate(x.shape) if d not in dims]


def _b_amax(out_op, name, dev):
    def kernel(self, dim=(), keepdim=False):
        dims = _dims(self, dim)
        out = torch.empty(_reduced(self, dims, keepdim), dtype=self.dtype, device=self.device)
        native_out(out_op, dev)(self, dim, keepdim, out=out)
        if not _floating(self):
            return out
        count(name)
        return torch.where(_nan_any(self, dims, keepdim), NAN, out)

    return kernel


def _b_whole(out_op, name, dev):  # max() / min() as amax / amin over every dim (Gaudi has no max.unary_out)
    def kernel(self):
        out = torch.empty((), dtype=self.dtype, device=self.device)
        native_out(out_op, dev)(self, [], False, out=out)
        if not _floating(self):
            return out
        count(name)
        return torch.where(torch.isnan(self).any(), NAN, out)

    return kernel


def _b_median_whole(dev):  # median() as median(dim 0) of the flattened tensor (no median.out)
    def kernel(self):
        if self.numel() == 0:
            return torch.full((), NAN, dtype=self.dtype, device=self.device)
        # keepdim, then a reshape: Gaudi's median(dim) of a 1-D tensor to a 0-d one fails graph compile (measured,
        # its own op too: x.reshape(-1).median(0))
        v, i = torch.empty([1], dtype=self.dtype, device=self.device), torch.empty([1], dtype=torch.int64, device=self.device)
        native_out(aten.median.dim_values, dev)(self.reshape(-1), 0, True, values=v, indices=i)
        v = v.reshape(())
        if not _floating(self):
            return v
        count("median")
        return torch.where(torch.isnan(self).any(), NAN, v)

    return kernel


def _argmax_out(dev):
    def argmax(x, d, keepdim):
        x = x.reshape(x.shape)  # Habana lazy argmax.out mis-infers output dims of a permuted input (MEGGPT chain)
        out = torch.empty(_reduced(x, [d], keepdim), dtype=torch.int64, device=x.device)
        return native_out(aten.argmax.out, dev)(x, d, keepdim, out=out)

    return argmax


SELECT_OUTS = {aten.max.dim_max: ("max", "max_values"), aten.min.dim_min: ("min", "min_indices"),
               aten.median.dim_values: ("values", "indices")}


def _b_select(out_op, name, dev):
    values_name, index_name = SELECT_OUTS[out_op]

    def kernel(self, dim, keepdim=False):
        if self.dim() == 0:
            _dims(self, dim)
            v, i = kernel(self.reshape(1), 0, False)
            return v.reshape(()), i.reshape(())
        d = _dims(self, dim)[0]
        kept = _reduced(self, [d], True)
        v, i = torch.empty(kept, dtype=self.dtype, device=self.device), torch.empty(kept, dtype=torch.int64, device=self.device)
        native_out(out_op, dev)(self, d, True, **{values_name: v, index_name: i})
        if _floating(self):
            count(name)
            i = _fixed_index(self, d, i, _argmax_out(dev))
            v = self.gather(d, i)
        return (v, i) if keepdim else (v.squeeze(d), i.squeeze(d))

    return kernel


def _b_arg(out_op, name, dev):
    def kernel(self, dim=None, keepdim=False):
        if not _floating(self) or self.dim() == 0 or self.numel() == 0:
            dims = _dims(self, dim) if dim is not None else list(range(self.dim()))
            shape = _reduced(self, dims, keepdim) if dim is not None else ([1] * self.dim() if keepdim else [])
            return native_out(out_op, dev)(self, dim, keepdim, out=torch.empty(shape, dtype=torch.int64, device=self.device))
        count(name)
        x = self.reshape(-1) if dim is None else self.reshape(self.shape)  # as in _argmax_out
        d = 0 if dim is None else _dims(self, dim)[0]
        i = native_out(out_op, dev)(x, d, True, out=torch.empty(_reduced(x, [d], True), dtype=torch.int64, device=x.device))
        i = _fixed_index(x, d, i, _argmax_out(dev))
        if dim is None:
            return i.reshape([1] * self.dim()) if keepdim else i.reshape(())
        return i if keepdim else i.squeeze(d)

    return kernel


def _b_aminmax(dev):
    def kernel(self, *, dim=None, keepdim=False):
        dims = _dims(self, dim)
        shape = _reduced(self, dims, keepdim) if dim is not None else ([1] * self.dim() if keepdim else [])
        lo, hi = torch.empty(shape, dtype=self.dtype, device=self.device), torch.empty(shape, dtype=self.dtype, device=self.device)
        native_out(aten.aminmax.out, dev)(self, dim=dim, keepdim=keepdim, min=lo, max=hi)
        if not _floating(self):
            return lo, hi
        count("aminmax")
        m = _nan_any(self, dims, keepdim).reshape(shape)
        return torch.where(m, NAN, lo), torch.where(m, NAN, hi)

    return kernel


def _on(dev, *ts):
    """The first tensor of ``ts`` on the device ``dev`` (where the call was dispatched): a CPU 0-d first operand,
    which torch takes with a device tensor (``maximum(torch.tensor(0.), x_hpu)``), must not place the result."""
    for t in ts:
        if isinstance(t, torch.Tensor) and t.device.type == dev:
            return t
    return ts[0]


def _binary_out(out_op, dev, a, b):
    out = torch.empty(torch.broadcast_shapes(a.shape, b.shape), dtype=torch.result_type(a, b),
                      device=_on(dev, a, b).device)
    return native_out(out_op, dev)(a, b, out=out)


def _b_binary(out_op, name, dev, nan_first):
    def kernel(self, other):
        out = _binary_out(out_op, dev, self, other)
        if not out.is_floating_point():
            return out
        count(name)
        if nan_first:  # maximum / minimum: NaN where either is
            return torch.where(torch.isnan(self) | torch.isnan(other), self + other, out)
        return torch.where(torch.isnan(other), self, torch.where(torch.isnan(self), other, out)).to(out.dtype)

    return kernel


def _clamp_dtype(x, bounds):
    dtype = x.dtype
    for b in bounds:
        if isinstance(b, torch.Tensor):
            dtype = torch.promote_types(dtype, torch.result_type(x, b))
    return dtype


def _b_clamp(out_op, name, dev):
    def fixed(self, *bounds):
        ts = [b for b in bounds if isinstance(b, torch.Tensor)]
        out = torch.empty(torch.broadcast_shapes(self.shape, *[b.shape for b in ts]),
                          dtype=_clamp_dtype(self, bounds), device=_on(dev, self, *ts).device)
        native_out(out_op, dev)(self, *bounds, out=out)
        if not out.is_floating_point():
            return out
        count(name)
        m = torch.isnan(self)
        for b in ts:
            m = m | torch.isnan(b)
        return torch.where(m, NAN, out)

    return fixed


def _b_clamp_inplace(out_op, name, dev):
    functional = _b_clamp(out_op, name, dev)

    def kernel(self, *bounds):
        return self.copy_(_check_inplace(self, functional(self, *bounds)))

    return kernel


def _norm_dtype(x, dtype):
    if dtype is not None:
        return dtype
    return x.real.dtype if x.is_complex() else x.dtype


def _b_vector_norm(dev):
    def kernel(self, ord=2, dim=None, keepdim=False, *, dtype=None):
        dims = _dims(self, dim)
        out = torch.empty(_reduced(self, dims, keepdim), dtype=_norm_dtype(self, dtype), device=self.device)
        native_out(aten.linalg_vector_norm.out, dev)(self, ord, dim, keepdim, dtype=dtype, out=out)
        if not _inf(ord) or not _floating(self) or self.is_complex():
            return out
        count("norm_inf")
        return torch.where(_nan_any(self, dims, keepdim), NAN, out)

    return kernel


def _b_norm(dev, with_dim):
    def kernel(self, p=2, dim=None, keepdim=False):
        dims = _dims(self, dim if with_dim else None)
        keepdim = keepdim if with_dim else False
        out = torch.empty(_reduced(self, dims, keepdim), dtype=_norm_dtype(self, None), device=self.device)
        if with_dim:
            native_out(aten.norm.out, dev)(self, p, dim, keepdim, out=out)
        else:  # norm.Scalar has no .out: the vector norm over every dim (the same values)
            native_out(aten.linalg_vector_norm.out, dev)(self, 2 if p is None else p, None, False, out=out)
        if not _inf(p) or not _floating(self) or self.is_complex():
            return out
        count("norm_inf")
        return torch.where(_nan_any(self, dims, keepdim), NAN, out)

    return kernel


POOL_META = {2: aten.max_pool2d_with_indices.default, 3: aten.max_pool3d_with_indices.default}


def _b_pool(out_op, name, dev, nd):
    def fn(x, *args):  # Gaudi's pool into fresh outputs (shapes from the meta kernel)
        out, idx = POOL_META[nd](torch.empty(x.shape, dtype=x.dtype, device="meta"), *args)
        out, idx = torch.empty(out.shape, dtype=x.dtype, device=x.device), torch.empty(idx.shape, dtype=torch.int64, device=x.device)
        native_out(out_op, dev)(x, *args, out=out, indices=idx)
        return out, idx

    def kernel(self, kernel_size, stride=(), padding=0, dilation=1, ceil_mode=False):
        args = (kernel_size, stride, padding, dilation, ceil_mode)
        if not _floating(self) or self.dim() < nd or not _codes_exact(self.shape[-nd:], kernel_size, nd):
            return fn(self, *args)
        count(name)
        return _pool(self, args, fn, nd)

    return kernel


def _backend_kernels(dev):
    k = [(aten.amax.default, _b_amax(aten.amax.out, "amax", dev)),
         (aten.amin.default, _b_amax(aten.amin.out, "amin", dev)),
         (aten.max.default, _b_whole(aten.amax.out, "max", dev)),
         (aten.min.default, _b_whole(aten.amin.out, "min", dev)),
         (aten.median.default, _b_median_whole(dev)),
         (aten.max.dim, _b_select(aten.max.dim_max, "max_dim", dev)),
         (aten.min.dim, _b_select(aten.min.dim_min, "min_dim", dev)),
         (aten.median.dim, _b_select(aten.median.dim_values, "median_dim", dev)),
         (aten.argmax.default, _b_arg(aten.argmax.out, "argmax", dev)),
         (aten.argmin.default, _b_arg(aten.argmin.out, "argmin", dev)),
         (aten.aminmax.default, _b_aminmax(dev)),
         (aten.maximum.default, _b_binary(aten.maximum.out, "maximum", dev, True)),
         (aten.minimum.default, _b_binary(aten.minimum.out, "minimum", dev, True)),
         (aten.fmax.default, _b_binary(aten.fmax.out, "fmax", dev, False)),
         (aten.fmin.default, _b_binary(aten.fmin.out, "fmin", dev, False)),
         (aten.clamp.Tensor, _b_clamp(aten.clamp.Tensor_out, "clamp_tensor", dev)),
         (aten.clamp_min.Tensor, _b_clamp(aten.clamp_min.Tensor_out, "clamp_tensor", dev)),
         (aten.clamp_max.Tensor, _b_clamp(aten.clamp_max.Tensor_out, "clamp_tensor", dev)),
         (aten.clamp_.Tensor, _b_clamp_inplace(aten.clamp.Tensor_out, "clamp_tensor", dev)),
         (aten.clamp_min_.Tensor, _b_clamp_inplace(aten.clamp_min.Tensor_out, "clamp_tensor", dev)),
         (aten.clamp_max_.Tensor, _b_clamp_inplace(aten.clamp_max.Tensor_out, "clamp_tensor", dev)),
         (aten.linalg_vector_norm.default, _b_vector_norm(dev)),
         (aten.norm.ScalarOpt_dim, _b_norm(dev, True)), (aten.norm.Scalar, _b_norm(dev, False)),
         (aten.max_pool2d_with_indices.default, _b_pool(aten.max_pool2d_with_indices.out, "max_pool2d", dev, 2)),
         (aten.max_pool3d_with_indices.default, _b_pool(aten.max_pool3d_with_indices.out, "max_pool3d", dev, 3))]
    return k


# ---- out= overloads: the functional op (rewritten), copied out


OUTS = [  # (out overload, functional op, names of the out arguments)
    (aten.amax.out, aten.amax.default, ("out",)), (aten.amin.out, aten.amin.default, ("out",)),
    (aten.max.dim_max, aten.max.dim, ("max", "max_values")), (aten.min.dim_min, aten.min.dim, ("min", "min_indices")),
    (aten.median.dim_values, aten.median.dim, ("values", "indices")),
    (aten.argmax.out, aten.argmax.default, ("out",)), (aten.argmin.out, aten.argmin.default, ("out",)),
    (aten.aminmax.out, aten.aminmax.default, ("min", "max")),
    (aten.maximum.out, aten.maximum.default, ("out",)), (aten.minimum.out, aten.minimum.default, ("out",)),
    (aten.fmax.out, aten.fmax.default, ("out",)), (aten.fmin.out, aten.fmin.default, ("out",)),
    (aten.clamp.Tensor_out, aten.clamp.Tensor, ("out",)),
    (aten.clamp_min.Tensor_out, aten.clamp_min.Tensor, ("out",)),
    (aten.clamp_max.Tensor_out, aten.clamp_max.Tensor, ("out",)),
    (aten.max_pool2d_with_indices.out, aten.max_pool2d_with_indices.default, ("out", "indices")),
    (aten.max_pool3d_with_indices.out, aten.max_pool3d_with_indices.default, ("out", "indices")),
    (aten.linalg_vector_norm.out, aten.linalg_vector_norm.default, ("out",)),
    (aten.norm.out, aten.norm.ScalarOpt_dim, ("out",)),
]


OUT_OPS = {out_op for out_op, _, _ in OUTS}


def _copy_out(out, value):
    """torch's out= rules: the result cast to ``out``'s dtype where torch allows it, ``out`` resized to the result."""
    if not torch.can_cast(value.dtype, out.dtype):
        raise RuntimeError(f"result type {value.dtype} can't be cast to the desired output type {out.dtype}")
    if out.shape != value.shape:
        out.resize_(value.shape)
    return out.copy_(value)


def _out_kernel(out_op, functional, names):
    def kernel(*args, **kwargs):
        outs = [kwargs.pop(n) for n in names]
        tensors = [a for a in list(args) + list(kwargs.values()) if isinstance(a, torch.Tensor)]
        if _grad_needed(*tensors) or _traced(args, kwargs):  # torch's own error / the tracer's handling
            return native(out_op)(*args, **kwargs, **dict(zip(names, outs)))
        values = functional(*args, **kwargs)
        values = values if isinstance(values, tuple) else (values,)
        done = tuple(_copy_out(o, v) for o, v in zip(outs, values))
        return done if len(done) > 1 else done[0]

    return kernel


_TRACING = threading.local()  # .active while a composite below runs from _inference_traced


def _inference_traced(device_type, composite, original):
    """The op's ``CompositeImplicitAutograd`` Python kernel, which the tracers' ``decompose()`` call (functionalization
    and the proxy mode, under every op) reads. Under ``torch.inference_mode()`` autograd's keys are excluded, so a
    compiled graph never reaches the ``Autograd{}`` composites and its fake tensors never the backend layer: there
    this runs the composite for ``device_type`` tensors, so the graph holds the NaN fix (as ``quantile``'s traced
    kernel). Elsewhere, and for the composite's own native calls (the flag), ``NotImplemented``: the tracer records
    the op as before. None of the family has a C++ ``CompositeImplicitAutograd`` kernel; ``original`` is called only
    if torch gives it a Python one."""
    has_original = not isinstance(original, functools.partial)

    def kernel(*args, **kwargs):
        if (getattr(_TRACING, "active", False)
                or not (torch.is_inference_mode_enabled() or _traced(args, kwargs))
                or _device(args) != device_type):
            return original(*args, **kwargs) if has_original else NotImplemented
        _TRACING.active = True
        try:
            return composite(*args, **kwargs)
        finally:
            _TRACING.active = False

    return kernel


def install(device_type, exact_sort=None):
    """NaN propagation of Gaudi's max/min ops on ``device_type``: the backend layer at its backend key, the traced
    composites (and the inf-norms, the whole-tensor reductions' backward, the out= overloads) at its autograd key.
    Returns an undo. ``HPU_OPS_NAN_PROPAGATION=0`` in the environment installs nothing (Gaudi's NaN-dropping ops
    stay; measured, the rewrite costs eager SPD steps up to about 9%, lazy within noise). The sort family (``sort``,
    ``argsort``, ``msort``, ``topk``, ``kthvalue``) unless ``exact_sort`` is false (default: on, off with the
    environment's ``HPU_OPS_EXACT_SORT=0``). hpu-ops' own sorts of eigenvalues and singular values keep Gaudi's sort
    (``internal_sort``): with them through the rewrite, eager SPD steps were up to 22% slower."""
    nan = os.environ.get("HPU_OPS_NAN_PROPAGATION") != "0"  # opt-out, read when enable() runs (like HPU_OPS_TPC=0)
    if exact_sort is None:
        exact_sort = os.environ.get("HPU_OPS_EXACT_SORT") != "0"
    if not (nan or exact_sort):  # the opt-out keeps the sort family's own switch
        return lambda: None
    traced = []  # (op, composite): handed to torch's autograd kernel outside a tracer
    for op, name in ((aten.amax.default, "amax"), (aten.amin.default, "amin")):
        traced.append((op, _amax_kernel(op, name)))
    for op, name in ((aten.max.dim, "max_dim"), (aten.min.dim, "min_dim"), (aten.median.dim, "median_dim")):
        traced.append((op, _select_kernel(op, name)))
    for op, name in ((aten.argmax.default, "argmax"), (aten.argmin.default, "argmin")):
        traced.append((op, _arg_kernel(op, name)))
    traced.append((aten.aminmax.default, _aminmax_kernel(aten.aminmax.default, "aminmax")))
    for op, name, largest in ((aten.maximum.default, "maximum", True), (aten.minimum.default, "minimum", False)):
        traced.append((op, _binary_kernel(op, name, largest)))
    for op, name in ((aten.fmax.default, "fmax"), (aten.fmin.default, "fmin")):
        traced.append((op, _fbinary_kernel(op, name)))
    clamps = {}
    for op, arity in ((aten.clamp.Tensor, 2), (aten.clamp_min.Tensor, 1), (aten.clamp_max.Tensor, 1)):
        clamps[op] = _clamp_kernel(op, "clamp_tensor", arity)
        traced.append((op, clamps[op]))
    for op, functional in ((aten.clamp_.Tensor, aten.clamp.Tensor), (aten.clamp_min_.Tensor, aten.clamp_min.Tensor),
                           (aten.clamp_max_.Tensor, aten.clamp_max.Tensor)):
        traced.append((op, _clamp_inplace_kernel(clamps[functional], "clamp_tensor")))
    for op, bwd, nd in ((aten.max_pool2d_with_indices.default, aten.max_pool2d_with_indices_backward.default, 2),
                        (aten.max_pool3d_with_indices.default, aten.max_pool3d_with_indices_backward.default, 3)):
        traced.append((op, _pool_kernel(op, bwd, f"max_pool{nd}d", nd)))
    k = [(op, "Autograd{}", _eager_native(op, composite)) for op, composite in traced]
    for op, name in ((aten.max.default, "max"), (aten.min.default, "min"), (aten.median.default, "median")):
        k.append((op, "Autograd{}", _whole_kernel(op, name)))
    k.append((aten.linalg_vector_norm.default, "Autograd{}",
              _vector_norm_kernel(aten.linalg_vector_norm.default, "norm_inf")))
    k.append((aten.norm.ScalarOpt_dim, "Autograd{}", _norm_kernel(aten.norm.ScalarOpt_dim, "norm_inf", True)))
    k.append((aten.norm.Scalar, "Autograd{}", _norm_kernel(aten.norm.Scalar, "norm_inf", False)))
    k += [(out_op, "Autograd{}", _out_kernel(out_op, functional, names)) for out_op, functional, names in OUTS]
    family = len(k)  # the sort family after these: its own switch, HPU_OPS_NAN_PROPAGATION=0 does not cancel it
    undo_select = _install_select(device_type) if exact_sort else (lambda: None)
    if exact_sort:
        k.append((aten.sort.default, "Autograd{}", _sort_kernel(aten.sort.default, "sort", False)))
        k.append((aten.sort.stable, "Autograd{}", _sort_kernel(aten.sort.stable, "sort", True)))
        k.append((aten.topk.default, "Autograd{}", _topk_kernel(aten.topk.default, "topk")))
        k.append((aten.kthvalue.default, "Autograd{}", _kthvalue_kernel(aten.kthvalue.default, "kthvalue")))
    if not nan:
        return chain([register_kernels(device_type, k[family:]), undo_select])
    undo_backend = override_kernels(device_type, _backend_kernels(device_type))
    _BACKEND[device_type] = True
    undo_autograd = register_kernels(device_type, k)
    composites = traced + [(op, kernel) for op, key, kernel in k[len(traced):family] if op not in OUT_OPS]
    undo_traced = chain([replace_py_kernels(op, (DispatchKey.CompositeImplicitAutograd,),
                                            functools.partial(_inference_traced, device_type, composite))
                         for op, composite in composites])

    def undo_flag():
        _BACKEND.pop(device_type, None)

    return chain([undo_backend, undo_flag, undo_autograd, undo_traced, undo_select])
