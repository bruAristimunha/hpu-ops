"""``torch.quantile`` / ``torch.nanquantile`` on HPU as sort and gather, with ``out=`` written by a copy.

In lazy mode (bridge 1.21.4) ``torch.quantile`` aborts with ``FATAL ERROR :: MODULE:PT_BRIDGE ... Trying to resize
tensor with non-existing TensorImpl`` raised from ``resize_impl_hpu_`` (hpu-ops #17): ATen's composite writes its
result into an empty ``out`` tensor that it resizes, and Habana's lazy resize of that tensor fails in the accumulation
thread. The rewrite here follows ATen's ``quantile_compute`` (aten/src/ATen/native/Sorting.cpp) step for step (sort
along the reduced dim, ranks ``q * (n - 1)`` with NaN slices sent to the last rank, the five interpolations,
``lerp``) but returns a fresh tensor, so nothing is resized; the ``out=`` overloads copy that tensor into ``out``.

Gaudi's sort returns NaN as +inf (bridge 1.21.4, eager and lazy), so native ``quantile`` of a slice with a NaN reads a
finite value where torch gives NaN, and ``nanquantile`` counts the NaNs as values (measured eager, bridge 1.21.4:
``quantile([3, nan, 1, 2], 0.5)`` 2.5 instead of nan, ``nanquantile`` 2.5 instead of 2). So the sort here never
sees a NaN: it orders a NaN-free key and only its indices are used. The sorted order torch's stable CPU sort gives
(finite values ascending, then the genuine +infs, then the NaNs, each in input order) is rebuilt from two sorts: the
values with NaN as +inf (right for every position before the first +inf) and the input positions of the +infs and
then the NaNs. The quantile values are then gathered from the input itself, at the input positions of the ranks
below and above, so their gradient reaches the same elements as torch's (a NaN slice's NaN included) and the
backward scatters into one input-sized buffer, not one per ``q``. Sort indices are carried as floating point,
exact up to the 2**24 elements ATen allows per slice: Gaudi's ``gather`` does not compile on int64 data.

It is registered at ``AutogradHPU`` (autograd differentiates gather and lerp as it does ATen's composite), at ``HPU``
(inference mode) and as the ``CompositeImplicitAutograd`` Python kernel, which ``torch.compile`` reads when it traces
an ``inference_mode()`` graph (autograd keys are excluded there and the tracer sits above ``HPU``). Arguments ATen
rejects (empty input, a ``q`` outside ``[0, 1]``, non-float dtypes, a ``q`` of another dtype or with more than one
dim, an ``out`` of another dtype) are not taken: ATen's composite runs and raises its own error. A slice longer than
2**24 raises ATen's error here. In a traced graph the values of a tensor ``q`` cannot be read on the host, so the
graph checks the range at run time: with ATen's message (``aten._assert_async``) where the device runs that op, and
on Gaudi, which has no kernel for it, by returning NaN for an out-of-range ``q``.
"""
import functools

import torch
from torch._C import DispatchKey
from torch._subclasses.fake_tensor import is_fake

from ._dispatch import chain, count, register_kernels, replace_py_kernels
from .extrema import internal_sort

aten = torch.ops.aten
INTERPOLATIONS = ("linear", "lower", "higher", "midpoint", "nearest")
MAX_LENGTH = 2 ** 24  # ATen: "quantile() input tensor is too large" past this many elements per slice
RANGE_ERROR = "quantile() q values must be in the range [0, 1]"


def _concrete(t):
    """True for a real tensor outside any tracer: its values can be read on the host."""
    return not is_fake(t) and torch._guards.detect_fake_mode() is None and not torch._is_functional_tensor(t)


def _reduce_view(self, dim, keepdim):
    """``self`` with the reduced elements on the last dim, and the output shape without the ``q`` dim."""
    if dim is None:
        return self.reshape(-1), [1] * self.dim() if keepdim else []
    if self.dim() == 0:
        return self.reshape(1), []
    d = dim % self.dim()
    shape = list(self.shape)
    if keepdim:
        shape[d] = 1
    else:
        del shape[d]
    return self.movedim(d, -1), shape


def _sorted_positions(x):
    """Gather the input position of sorted position ``pos`` (int64, any length along the last dim): torch's stable
    order, finite values ascending, then +inf, then NaN. Returns ``(at, n_valid)``, ``at(pos)`` the int64 input
    positions and ``n_valid`` the non-NaN count per slice."""
    n = x.shape[-1]
    key = x.detach()
    nan = key.isnan()
    posinf = key == float("inf")
    code = key.dtype  # sort indices as float data, exact to 2**24
    inf = float("inf")
    by_value = internal_sort(key.nan_to_num(nan=inf, posinf=inf, neginf=-inf), -1)[1].to(code)
    pos = torch.arange(n, device=key.device)
    j = pos.to(code)
    by_kind = internal_sort(torch.where(posinf, j - n, j.masked_fill(~nan, inf)), -1)[1].to(code)
    n_valid = n - nan.sum(-1, keepdim=True)
    n_finite = n_valid - posinf.sum(-1, keepdim=True)
    order = torch.where(pos < n_finite, by_value, by_kind.gather(-1, (pos - n_finite).clamp(min=0)))

    def at(p):  # float data, int64 index: Gaudi's gather does not compile on int64 data
        return order.gather(-1, p).long()

    return at, n_valid


def quantile_of(self, q, dim=None, keepdim=False, interpolation="linear", ignore_nan=False):
    """ATen's ``quantile_compute`` without the resized ``out``. ``q`` is a float or a 0-d or 1-d tensor."""
    qt = q if isinstance(q, torch.Tensor) else torch.tensor(q, dtype=self.dtype, device=self.device)
    x, out_shape = _reduce_view(self, dim, keepdim)
    n = x.shape[-1]
    at, n_valid = _sorted_positions(x)
    rank_dtype = torch.float64 if x.dtype == torch.float32 and _ranks_in_float64() else x.dtype
    qv = qt.reshape(-1).to(rank_dtype)  # (nq,), broadcast against (lead..., 1) to (lead..., nq) as in ATen
    if ignore_nan:
        ranks = qv * (n_valid - 1).to(rank_dtype)  # an all-NaN slice: rank 0, which is a NaN
        ranks = torch.where(ranks < 0, torch.zeros_like(ranks), ranks)
    else:
        last = torch.full((), n - 1, dtype=rank_dtype, device=x.device)
        has_nan = n_valid < n
        ranks = torch.where(has_nan, last, qv * (n - 1))  # a slice with a NaN: the last rank, a NaN
    if interpolation in ("linear", "midpoint"):
        below = ranks.floor()
        weights = ranks - below if interpolation == "linear" else torch.full_like(ranks, 0.5)
        lo = x.gather(-1, at(below.long()))
        hi = x.gather(-1, at(ranks.ceil().long()))
        values = torch.lerp(lo, hi, weights.to(x.dtype))
    else:
        pick = {"lower": torch.floor, "higher": torch.ceil, "nearest": torch.round}[interpolation](ranks)
        values = x.gather(-1, at(pick.long()))
    values = values.reshape(list(out_shape) + [qv.shape[0]])
    if qt.dim() == 0:
        return values.squeeze(-1)
    return values.unsqueeze(0).transpose(0, -1).squeeze(-1)  # ATen's layout: the q dim is the fastest


@functools.lru_cache(maxsize=None)
def _ranks_in_float64():
    """Whether the installed torch computes a float32 input's ranks ``q * (n - 1)`` and linear weights in float64
    (torch 2.15.0; 2.6 to 2.8 compute them in float32). Read from torch's own CPU quantile once: float32 0.7 * 5 is
    3.4999998 in float64, so nearest picks index 3, and 3.5 in float32, which rounds half to even to 4."""
    probe = aten.quantile.scalar._op_dk(DispatchKey.CompositeImplicitAutograd, torch.arange(6.0), 0.7, None, False,
                                        interpolation="nearest")  # ATen's own composite: never a rewrite
    return probe.item() == 3.0


@functools.lru_cache(maxsize=None)
def _length_limited():
    """Whether the installed torch rejects a slice longer than 2**24 (2.6 to 2.8; 2.15.0 does not). Read from ATen's
    composite on a meta tensor, which checks the size without computing."""
    try:
        aten.quantile.scalar._op_dk(DispatchKey.CompositeImplicitAutograd, torch.empty(MAX_LENGTH + 1, device="meta"),
                                    0.5, None, False, interpolation="linear")
    except RuntimeError:
        return True
    return False


_ranks_in_float64()  # read now, on the CPU, outside any tracer
_length_limited()


def _takes(self, q, dim=None, keepdim=False, interpolation="linear"):
    if not isinstance(self, torch.Tensor) or self.dtype not in (torch.float32, torch.float64) or self.numel() == 0:
        return False
    if interpolation not in INTERPOLATIONS:
        return False
    if dim is not None and not -max(self.dim(), 1) <= dim < max(self.dim(), 1):
        return False
    if self.shape and dim is not None and self.shape[dim % self.dim()] == 0:
        return False
    if isinstance(q, torch.Tensor):
        if q.dtype != self.dtype or q.dim() > 1 or q.device != self.device:
            return False
        if not _concrete(q):  # traced: checked at run time by the graph (_check_traced_range)
            return True
        return q.numel() == 0 or bool(((q >= 0) & (q <= 1)).all())
    return 0.0 <= float(q) <= 1.0


def _length(self, dim):
    return self.numel() if dim is None or self.dim() == 0 else self.shape[dim % self.dim()]


def _compute(name, ignore_nan, self, q, dim, keepdim, interpolation):
    if _length(self, dim) > MAX_LENGTH and _length_limited():
        raise RuntimeError("quantile() input tensor is too large")
    check = isinstance(q, torch.Tensor) and q.numel() and not _concrete(q)
    if check and _asserts_on(self.device.type):
        torch._assert_async(((q >= 0) & (q <= 1)).all(), RANGE_ERROR)
    count(name)
    values = quantile_of(self, q, dim, keepdim, interpolation, ignore_nan)
    if check and not _asserts_on(self.device.type):  # no device assert: an out-of-range q gives NaN, not a number
        bad = ~((q >= 0) & (q <= 1)).all()
        values = torch.where(bad, torch.full_like(values, float("nan")), values)
    return values


def _asserts_on(device_type):
    """Whether ``aten._assert_async`` runs on the device. Gaudi has no kernel for it (bridge 1.21.4: one CPU fallback
    per compiled call, 1.0 ms against 0.12 without the check), so there a traced out-of-range ``q`` gives NaN."""
    return torch._C._dispatch_has_kernel_for_dispatch_key("aten::_assert_async.msg", device_type.upper())


def _kernel(op, name, ignore_nan, fallback):
    def kernel(self, q, dim=None, keepdim=False, *, interpolation="linear"):
        if _takes(self, q, dim, keepdim, interpolation):
            return _compute(name, ignore_nan, self, q, dim, keepdim, interpolation)
        return op._op_dk(fallback, self, q, dim, keepdim, interpolation=interpolation)

    return kernel


def _out_kernel(op, name, ignore_nan, fallback):
    """The ``out=`` overloads: the fresh result copied into ``out`` (resized as ATen's ``resize_output`` does)."""

    def kernel(self, q, dim=None, keepdim=False, *, interpolation="linear", out):
        if (_takes(self, q, dim, keepdim, interpolation) and out.dtype == self.dtype
                and out.device == self.device):
            result = _compute(name, ignore_nan, self, q, dim, keepdim, interpolation)
            from torch._prims_common.wrappers import _maybe_resize_out
            _maybe_resize_out(out, result.shape)
            return out.copy_(result)
        return op._op_dk(fallback, self, q, dim, keepdim, interpolation=interpolation, out=out)

    return kernel


def _traced(device_type, name, ignore_nan, original):
    """``CompositeImplicitAutograd`` Python kernel, which tracers decompose to when autograd is excluded (compiled
    ``inference_mode()``): the rewrite for ``device_type`` tensors, ``original`` otherwise."""

    def kernel(self, q, dim=None, keepdim=False, *, interpolation="linear"):
        if self.device.type == device_type and _takes(self, q, dim, keepdim, interpolation):
            return _compute(name, ignore_nan, self, q, dim, keepdim, interpolation)
        return original(self, q, dim, keepdim, interpolation=interpolation)

    return kernel


OPS = ((aten.quantile, "quantile", False), (aten.nanquantile, "nanquantile", True))


def install(device_type):
    """``quantile`` and ``nanquantile`` (tensor and float ``q``, with and without ``out=``) on ``device_type``.
    Returns an undo."""
    kernels, undos = [], []
    fallback = DispatchKey.CompositeImplicitAutograd
    for packet, name, nan in OPS:
        for overload in (packet.default, packet.scalar):
            for key in ("Autograd{}", "{}"):
                kernels.append((overload, key, _kernel(overload, name, nan, fallback)))
            undos.append(replace_py_kernels(overload, (fallback,),
                                            functools.partial(_traced, device_type, name, nan)))
        for overload in (packet.out, packet.scalar_out):
            for key in ("Autograd{}", "{}"):
                kernels.append((overload, key, _out_kernel(overload, name, nan, fallback)))
    return chain([register_kernels(device_type, kernels)] + undos)
