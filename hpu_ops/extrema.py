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

- With ``HPU_OPS_EXACT_SORT=1`` (off by default, see ``install``): ``sort`` / ``argsort`` / ``msort`` / ``topk`` /
  ``kthvalue`` with NaN above +inf, as torch: a stable sort of the values with NaN as +inf, then a stable sort of the
  NaN mask. Gaudi's sort returns NaN as +inf with out-of-range indices, and its non-stable sort of 1000+ elements
  with infs returns out-of-range indices even without NaN; its stable sort is right. Eager and lazy, with autograd
  on: traced by ``torch.compile`` the native op stays (compiled on the card, two of the rewritten sorts came out
  wrong), and so in inference mode.
"""
import functools
import math
import os
import threading

import torch
from torch._C import DispatchKey
from torch._subclasses.fake_tensor import is_fake

from ._dispatch import chain, count, override_kernels, register_kernels, replace_py_kernels

aten = torch.ops.aten
NAN = float("nan")
REDUCED = (torch.float16, torch.bfloat16)
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


def _concrete(t):
    """True for a real tensor outside any tracer."""
    return not is_fake(t) and torch._guards.detect_fake_mode() is None and not torch._is_functional_tensor(t)


def _traced(args, kwargs):
    """True when ``torch.compile`` traces the call: its fake / functional tensors never reach a backend kernel."""
    ts = [a for a in list(args) + list(kwargs.values()) if isinstance(a, torch.Tensor)]
    return any(not _concrete(t) for t in ts)


def _onto_device(args):
    """``args`` with each CPU 0-d tensor moved to the device of the others (``maximum(torch.tensor(0.), x_hpu)``,
    which torch takes): Gaudi's eager kernels below autograd get it as a wrapped Float number, which the dispatcher
    cannot hand to the Python backend kernels ("Missing cases in 'toPyObject' wrapped number handling", measured,
    bridge 1.21.4). A 0-d tensor on the device promotes as on the CPU; ``to`` passes its gradient back."""
    dev = next((a.device for a in args if isinstance(a, torch.Tensor) and a.device.type != "cpu"), None)
    if dev is None:
        return args
    return tuple(a.to(dev) if isinstance(a, torch.Tensor) and a.device.type == "cpu" and a.dim() == 0 else a
                 for a in args)


def _eager_native(op, composite):
    """At ``Autograd{}``: outside a tracer, torch's autograd kernel of ``op`` (over the backend kernel's fixed result:
    torch's gradient); traced, ``composite``."""
    def kernel(*args, **kwargs):
        args = _onto_device(args)
        if _BACKEND.get(_device(args)) and not _traced(args, kwargs):
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
        fn = native(op)
        if not _inf(ord) or not _floating(self) or self.is_complex():
            return fn(self, ord, dim, keepdim, dtype=dtype)
        if dtype not in (None, self.dtype):  # the inf-norm is the largest |x|: exact in either dtype
            self = self.to(dtype)
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


# ---- the backend layer: every call (inference mode too), through the ops' .out overloads


def _reduced(x, dims, keepdim):
    return [1 if d in dims else n for d, n in enumerate(x.shape)] if keepdim else \
        [n for d, n in enumerate(x.shape) if d not in dims]


def _empty(shape, dtype, like):
    return torch.empty(shape, dtype=dtype, device=like.device)


def _b_amax(out_op, name, dev):
    def kernel(self, dim=(), keepdim=False):
        dims = _dims(self, dim)
        out = _empty(_reduced(self, dims, keepdim), self.dtype, self)
        native_out(out_op, dev)(self, dim, keepdim, out=out)
        if not _floating(self):
            return out
        count(name)
        return torch.where(_nan_any(self, dims, keepdim), NAN, out)

    return kernel


def _b_whole(out_op, name, dev):  # max() / min() as amax / amin over every dim (Gaudi has no max.unary_out)
    def kernel(self):
        out = _empty((), self.dtype, self)
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
        v, i = _empty([1], self.dtype, self), _empty([1], torch.int64, self)
        native_out(aten.median.dim_values, dev)(self.reshape(-1), 0, True, values=v, indices=i)
        v = v.reshape(())
        if not _floating(self):
            return v
        count("median")
        return torch.where(torch.isnan(self).any(), NAN, v)

    return kernel


def _argmax_out(dev):
    def argmax(x, d, keepdim):
        out = _empty(_reduced(x, [d], keepdim), torch.int64, x)
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
        v, i = _empty(kept, self.dtype, self), _empty(kept, torch.int64, self)
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
            return native_out(out_op, dev)(self, dim, keepdim, out=_empty(shape, torch.int64, self))
        count(name)
        x = self.reshape(-1) if dim is None else self
        d = 0 if dim is None else _dims(self, dim)[0]
        i = native_out(out_op, dev)(x, d, True, out=_empty(_reduced(x, [d], True), torch.int64, x))
        i = _fixed_index(x, d, i, _argmax_out(dev))
        if dim is None:
            return i.reshape([1] * self.dim()) if keepdim else i.reshape(())
        return i if keepdim else i.squeeze(d)

    return kernel


def _b_aminmax(dev):
    def kernel(self, *, dim=None, keepdim=False):
        dims = _dims(self, dim)
        shape = _reduced(self, dims, keepdim) if dim is not None else ([1] * self.dim() if keepdim else [])
        lo, hi = _empty(shape, self.dtype, self), _empty(shape, self.dtype, self)
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
    out = _empty(torch.broadcast_shapes(a.shape, b.shape), torch.result_type(a, b), _on(dev, a, b))
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
        out = _empty(torch.broadcast_shapes(self.shape, *[b.shape for b in ts]), _clamp_dtype(self, bounds),
                     _on(dev, self, *ts))
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
        out = _empty(_reduced(self, dims, keepdim), _norm_dtype(self, dtype), self)
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
        out = _empty(_reduced(self, dims, keepdim), _norm_dtype(self, None), self)
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
        out, idx = _empty(out.shape, x.dtype, x), _empty(idx.shape, torch.int64, x)
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
        if (getattr(_TRACING, "active", False) or not torch.is_inference_mode_enabled()
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
    stay; measured, the rewrite costs eager SPD steps up to about 9%, lazy within noise). The sort family (``sort``, ``argsort``, ``msort``, ``topk``, ``kthvalue``) only with
    ``exact_sort`` (default: the environment's ``HPU_OPS_EXACT_SORT=1``): measured eager, it made the training steps
    of the SPD models up to 22% slower (hpu-ops' own eigh and SVD sort their eigenvalues; TensorCSPNet 205 to 250 ms),
    lazy within noise."""
    nan = os.environ.get("HPU_OPS_NAN_PROPAGATION") != "0"  # opt-out, read when enable() runs (like HPU_OPS_TPC=0)
    if exact_sort is None:
        exact_sort = os.environ.get("HPU_OPS_EXACT_SORT") == "1"
    if not (nan or exact_sort):
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
    if exact_sort:
        k.append((aten.sort.default, "Autograd{}", _sort_kernel(aten.sort.default, "sort", False)))
        k.append((aten.sort.stable, "Autograd{}", _sort_kernel(aten.sort.stable, "sort", True)))
        k.append((aten.topk.default, "Autograd{}", _topk_kernel(aten.topk.default, "topk")))
        k.append((aten.kthvalue.default, "Autograd{}", _kthvalue_kernel(aten.kthvalue.default, "kthvalue")))
    if not nan:
        return register_kernels(device_type, k[family:])
    undo_backend = override_kernels(device_type, _backend_kernels(device_type))
    _BACKEND[device_type] = True
    undo_autograd = register_kernels(device_type, k)
    composites = traced + [(op, kernel) for op, key, kernel in k[len(traced):family] if op not in OUT_OPS]
    undo_traced = chain([replace_py_kernels(op, (DispatchKey.CompositeImplicitAutograd,),
                                            functools.partial(_inference_traced, device_type, composite))
                         for op, composite in composites])

    def undo_flag():
        _BACKEND.pop(device_type, None)

    return chain([undo_backend, undo_flag, undo_autograd, undo_traced])
