"""``torch.quantile`` / ``torch.nanquantile`` on HPU as sort, gather and lerp, without an ``out=`` tensor.

In lazy mode (bridge 1.21.4) ``torch.quantile`` aborts with ``FATAL ERROR :: MODULE:PT_BRIDGE ... Trying to resize
tensor with non-existing TensorImpl`` raised from ``resize_impl_hpu_`` (hpu-ops #17): ATen's composite writes its
result into an empty ``out`` tensor that it resizes, and Habana's lazy resize of that tensor fails in the accumulation
thread. The rewrite here follows ATen's ``quantile_compute`` step for step (sort along the reduced dim, ranks
``q * (n - 1)``, the five interpolations, ``lerp``) but returns a fresh tensor, so nothing is resized. NaNs are found
in the input, not in the sorted values: Gaudi's sort returns NaN as +inf, so native ``quantile`` of a slice with a NaN
reads a finite value where torch gives NaN, and ``nanquantile`` counts the NaNs as values (measured eager, bridge
1.21.4: ``quantile([3, nan, 1, 2], 0.5)`` 2.5 instead of nan, ``nanquantile`` 2.5 instead of 2).
It is registered at ``AutogradHPU`` (autograd differentiates sort / gather / lerp as it does ATen's composite) and at
``HPU`` (inference mode). Arguments ATen rejects (empty input, a ``q`` outside ``[0, 1]``, non-float dtypes, a
``q`` of another dtype or with more than one dim) are not taken: ATen's composite runs and raises its own error.
"""
import torch
from torch._C import DispatchKey
from torch._subclasses.fake_tensor import is_fake

from ._dispatch import count, register_kernels

aten = torch.ops.aten
INTERPOLATIONS = ("linear", "lower", "higher", "midpoint", "nearest")


def _concrete(t):
    """True for a real tensor outside any tracer: its values can be read on the host."""
    return not is_fake(t) and torch._guards.detect_fake_mode() is None and not torch._is_functional_tensor(t)


def _reduce_view(self, dim, keepdim):
    """``self`` with the reduced elements on the last dim, and the output shape without the ``q`` dim."""
    if dim is None:
        return self.reshape(-1), [1] * self.dim() if keepdim else []
    d = dim % max(self.dim(), 1)
    shape = list(self.shape)
    if self.dim() == 0:
        return self.reshape(1), []
    if keepdim:
        shape[d] = 1
    else:
        del shape[d]
    return self.movedim(d, -1), shape


def quantile_of(self, q, dim=None, keepdim=False, interpolation="linear", ignore_nan=False):
    """ATen's ``quantile_compute`` (aten/src/ATen/native/Sorting.cpp) without the resized ``out``."""
    qt = q if isinstance(q, torch.Tensor) else torch.tensor(q, dtype=self.dtype, device=self.device)
    x, out_shape = _reduce_view(self, dim, keepdim)
    nan = x.isnan()
    # Gaudi's sort returns NaN as +inf (bridge 1.21.4, eager and lazy), so NaNs are sorted as +inf on purpose and
    # their slices are found from the input: a genuine +inf sorts among them and reads the same value
    s = torch.sort(torch.where(nan, torch.full_like(x, float("inf")), x), dim=-1).values
    n = s.shape[-1]
    lead = tuple(s.shape[:-1])
    nq = qt.numel()
    qv = qt.reshape((nq,) + (1,) * (len(lead) + 1)).to(s.dtype)
    if ignore_nan:
        valid = (~nan).sum(-1, keepdim=True)
        ranks = qv * (valid - 1).to(s.dtype)
        ranks = torch.where(ranks < 0, torch.zeros_like(ranks), ranks)
        is_nan = valid == 0  # an all-NaN slice gives NaN
    else:
        ranks = (qv * (n - 1)).expand((nq,) + lead + (1,))
        is_nan = nan.any(-1, keepdim=True)  # any NaN in the slice gives NaN
    full = (nq,) + lead + (1,)
    ranks = ranks.expand(full)
    sorted_q = s.unsqueeze(0).expand((nq,) + tuple(s.shape))
    if interpolation in ("linear", "midpoint"):
        below = ranks.floor()
        weights = ranks - below if interpolation == "linear" else torch.full_like(ranks, 0.5)
        lo = sorted_q.gather(-1, below.long())
        hi = sorted_q.gather(-1, ranks.ceil().long())
        values = torch.lerp(lo, hi, weights)
    else:
        pick = {"lower": torch.floor, "higher": torch.ceil, "nearest": torch.round}[interpolation](ranks)
        values = sorted_q.gather(-1, pick.long())
    values = torch.where(is_nan, torch.full_like(values, float("nan")), values)
    values = values.reshape([nq] + out_shape)
    return values.squeeze(0) if qt.dim() == 0 else values


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
        if not _concrete(q):  # traced: the range cannot be read on the host (ATen's meta check cannot either)
            return True
        return q.numel() == 0 or bool(((q >= 0) & (q <= 1)).all())
    return 0.0 <= float(q) <= 1.0


def _kernel(op, name, ignore_nan, fallback):
    def kernel(self, q, dim=None, keepdim=False, *, interpolation="linear"):
        if _takes(self, q, dim, keepdim, interpolation):
            count(name)
            return quantile_of(self, q, dim, keepdim, interpolation, ignore_nan)
        return op._op_dk(fallback, self, q, dim, keepdim, interpolation=interpolation)

    return kernel


def install(device_type):
    """``quantile`` and ``nanquantile`` (tensor and float ``q``) on ``device_type`` without ``out=``. Returns an
    undo."""
    kernels = []
    for op, name, nan in ((aten.quantile.default, "quantile", False), (aten.quantile.scalar, "quantile", False),
                          (aten.nanquantile.default, "nanquantile", True),
                          (aten.nanquantile.scalar, "nanquantile", True)):
        kernels.append((op, "Autograd{}", _kernel(op, name, nan, DispatchKey.CompositeImplicitAutograd)))
        kernels.append((op, "{}", _kernel(op, name, nan, DispatchKey.CompositeImplicitAutograd)))
    return register_kernels(device_type, kernels)
