"""Advanced indexing (``aten::index.Tensor``) and boolean-mask puts of one value, which Gaudi runs on the CPU or gets
wrong (bridge 1.21.4).

- ``index``. Three Habana defects, one kernel at ``AutogradHPU`` and ``HPU`` (Habana registers neither in eager mode,
  and in lazy mode only ``HPU``, which is left alone):
  - Eager: with an index tensor on the CPU (``torch.triu_indices`` in spd_learn's ``sym_to_upper``, i.e. ``vech``)
    ``index`` reaches ``aten::index.Tensor_out``, which falls back to the CPU (1 to 7 times per training step).
  - Lazy: the backward of ``index``, ``_index_put_impl_(zeros, indices, grad, accumulate=True)``, with a ``None``
    before the first index tensor or between two of them, returns wrong sums from batch 2 to 8 on (relative error
    0.45 to 1 for ``X[..., triu_indices]``, batch 32, n = 12 or 22). That is ``vech`` and the reads of
    ``vec_to_sym``: every gradient before them was wrong in SPDResNet, GyroSPD, CorNet, SPNN, DeepGeoCCA, SPDNetAE,
    EEGSPDNet, MAtt and CorAtt. Negative indices that do not lead have a wrong gradient eager and lazy, and
    ``X[:, mask]`` comes out transposed in lazy mode.
  - Integer indices (int64 or int32, broadcast, 0-d) run as one ``index_select`` of the dims from the first to the
    last index flattened, with the linear index of the broadcast indices (a slice between two indices joins it as an
    ``arange``, so ``self`` is not copied); its backward is ``index_add``, right in lazy mode. This only where Gaudi is
    wrong or falls back: an index tensor on the CPU, or indices that do not lead (their lazy backward; and, with or
    without a gradient, their reads in one lazy graph with index puts and a mask put fail to launch, as below).
    Leading indices on the device keep Habana's kernel. Negative
    indices are wrapped (in int64). CPU index tensors are checked against the sizes on the host (out of range: ATen's
    own error) and the linear index is built there and moved once. Index tensors on the device, and any index tensor
    while ``torch.compile`` traces (fake values), are not read on the host: an index out of range in its own dim
    becomes -1, which ``index_select`` rejects, never a neighbouring element. Compiled graphs hold the rewrite
    (forward ``index_select``, backward ``index_add`` or its decomposition).
  - One boolean mask after a slice, ``X[:, mask]``: its ``nonzero`` (on the mask's own device), into the same
    ``index_select``. Other masks and mixes on the device keep the native op; with an index tensor on the CPU they
    are moved to the device first. Index tensors on a device other than the CPU and ``self``'s reach ATen's error.
  Two other forms were measured right on ``vech`` and SPDResNet but not everywhere: a leading-index ``index`` of a
  permuted view (CorNet's gradient stayed 0.44 off), and rewriting the backward's ``_index_put_impl_`` through a
  permuted view (SPNN's launch failed, see the next item). ``gather`` on int64 data does not compile on Gaudi2.
- A boolean-mask put of one value, ``m[..., mask] = c`` (vech's sqrt(2) scale): in one lazy graph with index puts into
  a fresh matrix and index reads of it (``vech(vec_to_sym(v))``, SPNN's SPDLinear) Habana's launch fails, "synLaunch
  failed ... Sections' validation failure" (``repro/lazy_section_validation.py``, without hpu-ops). It is rewritten at
  ``AutogradHPU`` as ``self.copy_(where(mask, c, self))``: equal values, and no ``nonzero`` and host sync. Only that
  form: one bool mask, last in the index list (an explicit ``None`` after the mask is aligned differently by ATen;
  byte masks reach ATen's error). Leading masks too, ``x[x < eps] = eps``: keeping them native brought the launch
  failure back in SPNN and DeepGeoCCA (measured), at the cost of one copy of the destination. A value of another
  dtype is checked on the host with ATen's checked cast and converted with ``to`` (its forward-mode tangent is kept);
  one that ATen rejects (overflow, complex into real), one on the device or with a gradient, a mask or value that
  shares memory with the destination, and a destination whose elements overlap or leave gaps (an expanded or
  ``as_strided`` view) keep the native op.

``index_put_`` with integer indices keeps the native kernels, and so do lazy-mode calls below autograd (inference
mode), where Habana's ``HPU`` kernel of ``index`` is registered.
"""
import contextlib

import torch
from torch._C import DispatchKey

from . import _strict
from ._dispatch import count, register_kernels

aten = torch.ops.aten
INTEGER = (torch.int64, torch.int32)
MASKS = (torch.bool, torch.uint8)  # uint8: ATen's deprecated byte masks, which it still takes
VALID = INTEGER + MASKS
HOST_INDEX = "index: CPU index tensors checked and linearized on the host"


def _positions(indices):
    return [i for i, t in enumerate(indices) if t is not None]


def _in_range(t, size):
    """True when every value of a CPU index tensor is in [-size, size), False when one is not. None when it is not read
    on the host: on the device (that would need a sync), or fake while ``torch.compile`` traces."""
    if t.device.type != "cpu" or _tracing(t):
        return None  # traced: since torch 2.10 the failed read below leaves unbacked symbols that dynamo rejects
    if t.numel() == 0:
        return True
    try:
        with _strict.route(HOST_INDEX, t, detail="CPU index tensors: range checked on the CPU"):
            t = t.long()  # an int32 index of a dim past 2**31 compares in int64
            return bool(((t >= -size) & (t < size)).all())
    except _strict.CPUExecutionError:
        raise
    except Exception:  # noqa: BLE001  (data-dependent under tracing)
        return None


def _tracing(t):
    """True while ``torch.compile`` traces (dynamo, or AOT autograd's fake tensors): index arithmetic then goes on the
    device, in the graph, instead of on the host."""
    from torch._subclasses.fake_tensor import is_fake

    return torch.compiler.is_compiling() or is_fake(t)


def _off_device(self, t):
    """True for an index tensor that is not on ``self``'s device (the CPU, for HPU tensors). The CPU tests, where both
    are the CPU, patch it to stand for CPU index tensors of HPU tensors."""
    return t.device != self.device


def _plan(self, indices):
    """How ``self[indices]`` runs, only where Gaudi's own op is wrong or falls back; None: ATen's op as called.
    - "int" (checked on the host) or "int_unchecked" (on the device, or traced): integer indices as one
      ``index_select``, when an index tensor is off the device (eager fallback) or the indices do not lead (lazy:
      wrong backward, and, with a gradient or not, in one graph with index puts and a mask put Habana's launch fails,
      SPNN and DeepGeoCCA measured). Leading indices on the device keep the native op.
    - "mask": one boolean mask after a slice, through ``nonzero`` into that ``index_select`` (Habana's lazy kernel
      transposes the result of ``X[:, mask]``).
    - "move": other valid indices off the device moved to it, then ATen's op.
    Invalid indices (dtype, shape, range, a device that is neither the CPU nor ``self``'s) reach ATen's error."""
    pos = _positions(indices)
    if not pos or len(indices) > self.dim():
        return None
    if any(indices[i].device not in (self.device, torch.device("cpu")) for i in pos):
        return None  # ATen's device error
    dtypes = {indices[i].dtype for i in pos}
    if not dtypes <= set(VALID):
        return None  # float, int8 and int16 indices: ATen's error
    off = any(_off_device(self, indices[i]) for i in pos)
    if dtypes <= set(INTEGER):
        if not off and pos == list(range(len(pos))):
            return None  # Gaudi's own op is right there
        try:
            torch.broadcast_shapes(*(indices[i].shape for i in pos))
        except RuntimeError:
            return None  # ATen's error
        checked = [_in_range(indices[i], self.shape[i]) for i in pos]
        if False in checked:
            return None  # out of range: ATen's error
        return "int" if all(checked) else "int_unchecked"
    dim = 0
    for t in indices:
        if t is not None and t.dtype in MASKS and t.dim() > 0:
            if tuple(t.shape) != tuple(self.shape[dim:dim + t.dim()]):
                return None  # ATen's error
            dim += t.dim()
        elif t is None or t.dim() > 0 or t.dtype not in MASKS:
            dim += 1
    if dim > self.dim():
        return None
    if len(pos) == 1 and indices[pos[0]].dim() > 0 and pos[0] > 0 and indices[pos[0]].dtype == torch.bool:
        return "mask"
    return "move" if off else None


def index_select_flat(self, indices, checked=True):
    """``self[indices]`` for integer indices as one ``index_select`` of the dims from the first to the last index
    flattened, with the linear index of the broadcast indices; a dim between two indices (a slice) joins the linear
    index as an ``arange``, so no permuted copy of ``self`` is made. ATen puts the broadcast index dims in place of
    the indexed dims when these are adjacent, else first: a permute of the result brings them there. Negative indices
    are wrapped (in int64). Unless ``checked`` (the host has checked the range), an index out of range in its own dim
    becomes -1, which ``index_select`` rejects, never a neighbouring element. The linear index is computed on the host
    when every index tensor is there (outside tracing), then moved once."""
    pos = _positions(indices)
    start, end = pos[0], pos[-1]
    bshape = torch.broadcast_shapes(*(indices[i].shape for i in pos))
    mids = [d for d in range(start, end + 1) if indices[d] is None]  # slices between two indices
    on_host = not _tracing(self) and all(indices[i].device.type == "cpu" for i in pos)
    device = torch.device("cpu") if on_host else self.device
    host = _strict.route(HOST_INDEX, leaves=[indices[i] for i in pos],
                         detail="CPU index tensors: the linear index is computed on the CPU, then copied to the card") \
        if on_host and self.device.type != "cpu" else contextlib.nullcontext()
    shape = tuple(bshape) + tuple(self.shape[d] for d in mids)
    # dims of an expanded input (stride 0) hold one value: select from one element of them, not from a copy of all
    sizes = tuple(self.shape)
    flat_dims = [d for d in range(start, end + 1) if sizes[d] > 1 and self.stride(d) == 0]
    if flat_dims:
        self = self[tuple(slice(0, 1) if d in flat_dims else slice(None) for d in range(self.dim()))]
    with host:
        lin = ok = None
        for d in range(start, end + 1):
            size = sizes[d]
            if d in mids:
                j = mids.index(d)
                term = torch.zeros(size, device=device, dtype=torch.int64) if d in flat_dims else torch.arange(
                    size, device=device)
                term = term.reshape((size,) + (1,) * (len(mids) - 1 - j))
            else:
                t = indices[d].to(device=device, dtype=torch.int64)
                if not checked:
                    valid = (t >= -size) & (t < size)
                    ok = valid if ok is None else ok & valid
                t = torch.zeros_like(t) if d in flat_dims else torch.where(t < 0, t + size, t)
                term = t.reshape(tuple(t.shape) + (1,) * len(mids))
            lin = term if lin is None else lin * self.shape[d] + term
        if ok is not None:
            lin = torch.where(ok.reshape(tuple(ok.shape) + (1,) * len(mids)), lin, -1)
        lin = torch.broadcast_to(lin, shape)
    flat = self.flatten(start, end) if end > start else self
    out = flat.index_select(start, lin.reshape(-1).to(self.device))
    if lin.dim() == 0:
        return out.squeeze(start)  # 0-d indices (direct aten calls): the indexed dims go away
    out = out.unflatten(start, shape) if lin.dim() != 1 else out
    if mids and start > 0 and len(bshape):  # separated indices: the broadcast dims go first, as in ATen
        nb = len(bshape)
        order = list(range(start, start + nb)) + list(range(start)) + list(range(start + nb, out.dim()))
        out = out.permute(order).contiguous()
    return out


def _expand_masks(indices):
    """Each mask as the integer indices of its true elements (``nonzero`` on the mask's own device)."""
    out = []
    for t in indices:
        if t is not None and t.dtype == torch.bool and t.dim() > 0:
            out.extend(t.nonzero(as_tuple=True))
        else:
            out.append(t)
    return out


def _index_kernel(device_type, fallback):
    op = aten.index.Tensor

    def kernel(self, indices):
        plan = _plan(self, indices) if self.device.type == device_type else None
        if plan is None:
            return op._op_dk(fallback, self, indices)
        count("index")
        if plan == "move":
            return op._op_dk(fallback, self, [t if t is None else t.to(self.device) for t in indices])
        if plan == "mask":
            return index_select_flat(self, _expand_masks(indices))
        return index_select_flat(self, indices, checked=plan == "int")

    return kernel


def _dense(t):
    """True when no two elements of ``t`` share memory and it has no gaps (``copy_`` then writes it as ATen's put)."""
    try:
        return torch._debug_has_internal_overlap(t) == 0
    except Exception:  # noqa: BLE001  (a tensor subclass without strides)
        return False


def _shares_memory(a, b):
    """True when ``a`` and ``b`` may share memory (one storage), so a put through ``where`` would read what it writes.
    ATen's put raises there."""
    if a.device != b.device:
        return False  # spd_learn's case: a CPU mask and a wrapped CPU scalar into an HPU tensor
    try:
        return torch._C._is_alias_of(a, b)
    except Exception:  # noqa: BLE001  (no storage to compare, e.g. a lazy HPU tensor)
        pass
    try:
        return a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()
    except Exception:  # noqa: BLE001
        return True


def _value(self, values):
    """The put value in ``self``'s dtype, or None for the native op. A value of another dtype is checked as ATen
    checks it (``value.item()`` with its checked cast: overflow and complex into real raise), which needs it on the
    host and without a gradient, then converted with ``to``, which keeps a forward-mode tangent."""
    if values.dtype == self.dtype:
        return values
    if values.device.type != "cpu" or values.requires_grad:
        return None
    from torch._subclasses.fake_tensor import unset_fake_temporarily

    try:
        with unset_fake_temporarily():  # while tracing a fake full() would not check
            torch.full((), values.item(), dtype=self.dtype)  # ATen's checked cast, for its errors only
    except Exception:  # noqa: BLE001  (ATen's overflow error, or no value while tracing)
        return None
    return (values.real if values.is_complex() else values).to(self.dtype)


def _mask_takes(self, indices, values, accumulate=False):
    """The case Habana's lazy launch fails on (SPNN, DeepGeoCCA): ``self[..., mask] = value``, one boolean mask that
    is the last index (ATen aligns a mask followed by an explicit ``None`` differently), one value, a dense
    destination that shares no memory with the mask or the value. Leading masks too (``x[x < eps] = eps``): keeping
    those native brought the launch failure back in SPNN and DeepGeoCCA (measured), so ``where`` allocates one copy of
    the destination there instead of the in-place ``masked_fill_``."""
    pos = _positions(indices)
    if accumulate or len(pos) != 1 or values.numel() != 1 or values.dim() > 1 or self.is_complex():
        return False
    mask, p = indices[pos[0]], pos[0]
    if mask.dtype != torch.bool or p != len(indices) - 1 or len(indices) > self.dim():
        return False  # byte masks: ATen's error; a mask with an explicit None after it: ATen's alignment
    if mask.dim() == 0 or tuple(mask.shape) != tuple(self.shape[p:p + mask.dim()]):
        return False
    if mask.device not in (self.device, torch.device("cpu")) or values.device not in (self.device, torch.device("cpu")):
        return False  # ATen's device error
    if _shares_memory(self, mask) or _shares_memory(self, values) or not _dense(self):
        return False  # ATen raises on aliasing; and writes into overlapping views its own way
    return _value(self, values) is not None  # ATen raises on the value's cast (overflow, complex to real)


def mask_fill(self, indices, values, accumulate=False):
    """``self[..., mask] = value`` as ``self.copy_(where(mask, value, self))``; returns self."""
    p = _positions(indices)[0]
    mask = indices[p]
    mask = mask.to(self.device).reshape((1,) * p + tuple(mask.shape) + (1,) * (self.dim() - p - mask.dim()))
    return self.copy_(torch.where(mask, _value(self, values).reshape(()).to(self.device), self))


def install_index(device_type):
    """``index`` on ``device_type`` as above, at the autograd and backend keys. Returns an undo."""
    op = aten.index.Tensor
    return register_kernels(device_type, [
        (op, "Autograd{}", _index_kernel(device_type, DispatchKey.Autograd)),
        (op, "{}", _index_kernel(device_type, DispatchKey.CompositeExplicitAutogradNonFunctional)),
    ])


def install_mask_fill(device_type):
    """``index_put_`` of one value through one boolean mask, on ``device_type``, as ``where``. Returns an undo."""
    op = aten.index_put_.default

    def kernel(self, indices, values, accumulate=False):
        if _mask_takes(self, indices, values, accumulate):
            count("mask_fill")
            return mask_fill(self, indices, values, accumulate)
        return op._op_dk(DispatchKey.Autograd, self, indices, values, accumulate)

    return register_kernels(device_type, [(op, "Autograd{}", kernel)])
