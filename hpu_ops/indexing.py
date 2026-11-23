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
import collections
import contextlib
import math
import threading
import weakref

import torch
from torch._C import DispatchKey

from . import _dispatch, _strict
from ._dispatch import count, register_kernels

aten = torch.ops.aten
INTEGER = (torch.int64, torch.int32)
MASKS = (torch.bool, torch.uint8)  # uint8: ATen's deprecated byte masks, which it still takes
VALID = INTEGER + MASKS
HOST_INDEX = "index: CPU index tensors checked and linearized on the host"

INDEX_CACHE_SIZE = 128
INDEX_CACHE_BYTES = 64 << 20
INDEX_CACHE_ENTRY_MAX_BYTES = 16 << 20
INDEX_VALUE_MAX_NUMEL = 8192  # value-keyed cache: hashing a CPU index tensor's bytes pays below this size
_INDEX_LOCK = threading.RLock()
_INDEX_CACHE = collections.OrderedDict()
_INDEX_VALUE_CACHE = collections.OrderedDict()
_INDEX_VALUE_BYTES = 0
_DEAD_KEYS = collections.deque()
_INDEX_BYTES = 0


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


def _tracing(*tensors):
    """True while a tracer (``torch.compile`` Dynamo/AOT FakeTensor/FunctionalTensor, fake mode, or ``make_fx``
    PROXY mode) sees ``tensors``: index arithmetic then goes on the device, in the graph, instead of on the host."""
    return _dispatch.tracing(*tensors)


def _cacheable_cpu(t):
    return (
        type(t) is torch.Tensor
        and t.device.type == "cpu"
        and not t.is_inference()
        and not torch._is_functional_tensor(t)
        and not torch._C._functorch.is_functorch_wrapped_tensor(t)
    )


def _drain_dead_keys_locked():
    global _INDEX_BYTES
    while _DEAD_KEYS:
        try:
            dead_key = _DEAD_KEYS.popleft()
        except IndexError:
            break
        entry = _INDEX_CACHE.get(dead_key)
        if entry is not None and any(wr() is None for wr, _ in entry[3]):
            _INDEX_CACHE.pop(dead_key)
            _INDEX_BYTES -= entry[2]


def _clear_cache():
    global _INDEX_BYTES, _INDEX_VALUE_BYTES
    with _INDEX_LOCK:
        _INDEX_CACHE.clear()
        _INDEX_VALUE_CACHE.clear()
        _DEAD_KEYS.clear()
        _INDEX_BYTES = 0
        _INDEX_VALUE_BYTES = 0


def _cache_key_and_storages(self, indices, pos, start, end, flat_dims_rel, exact):
    storages = []
    seen_cdata = set()
    idx_meta = []
    for d in range(start, end + 1):
        t = indices[d]
        if t is None:
            idx_meta.append(None)
            continue
        st = t.untyped_storage()
        cd = st._cdata
        if cd not in seen_cdata:
            seen_cdata.add(cd)
            storages.append(st)
        idx_meta.append((t.data_ptr(), t._version, t.dtype, tuple(t.shape), tuple(t.stride()), cd))
    key = (str(self.device), tuple(self.shape[start:end + 1]), flat_dims_rel, exact, tuple(idx_meta))
    return key, storages


def _value_key(self, indices, pos, start, end, flat_dims_rel, exact):
    """The ptr-key with each index tensor's bytes hashed in place of (data_ptr, version): a caller that builds
    fresh equal indices every step (spd_learn's triu_indices) hits where the ptr key cannot. Small tensors only:
    the hash reads every value (already on the host, no sync)."""
    if sum(indices[i].numel() for i in pos) > INDEX_VALUE_MAX_NUMEL:
        return None
    idx_meta = []
    for d in range(start, end + 1):
        t = indices[d]
        if t is None:
            idx_meta.append(None)
            continue
        idx_meta.append((t.dtype, tuple(t.shape), tuple(t.stride()), hash(t.contiguous().numpy().tobytes())))
    return ("val", str(self.device), tuple(self.shape[start:end + 1]), flat_dims_rel, exact, tuple(idx_meta))


def _cache_lookup(self, indices):
    global _INDEX_BYTES
    pos = _positions(indices)
    if not pos or len(indices) > self.dim():
        return None
    if not all(indices[i].dtype in INTEGER and _cacheable_cpu(indices[i]) for i in pos):
        return None
    off = any(_off_device(self, indices[i]) for i in pos)
    if not off and pos == list(range(len(pos))):
        return None
    if (_tracing(self, *(indices[i] for i in pos)) or _transformed(self)
            or torch._C._functorch.get_dynamic_layer_stack_depth() != 0):
        return None
    try:
        bshape = torch.broadcast_shapes(*(indices[i].shape for i in pos))
    except RuntimeError:
        return None
    start, end = pos[0], pos[-1]
    sizes = tuple(self.shape)
    flat_dims_rel = tuple(d - start for d in range(start, end + 1) if sizes[d] > 1 and self.stride(d) == 0)
    exact = bool(flat_dims_rel) and torch.is_grad_enabled() and self.requires_grad
    key, storages = _cache_key_and_storages(self, indices, pos, start, end, flat_dims_rel, exact)
    with _INDEX_LOCK:
        _drain_dead_keys_locked()
        entry = _INDEX_CACHE.get(key)
        if entry is not None:
            lin_flat, lin_true, nbytes, refs = entry
            if not all(wr() is st for (wr, _), st in zip(refs, storages)):
                _INDEX_CACHE.pop(key)
                _INDEX_BYTES -= nbytes
                entry = None
            else:
                _INDEX_CACHE.move_to_end(key)
        if entry is None:
            vkey = _value_key(self, indices, pos, start, end, flat_dims_rel, exact)
            entry = _INDEX_VALUE_CACHE.get(vkey) if vkey is not None else None
            if entry is None:
                return None
            lin_flat, lin_true = entry[0], entry[1]
            _INDEX_VALUE_CACHE.move_to_end(vkey)
    mids = [d for d in range(start, end + 1) if indices[d] is None]
    flat_dims = [start + rel for rel in flat_dims_rel]
    return lin_flat, lin_true, start, end, bshape, mids, flat_dims


def _cache_store(key, storages, indices, pos, snap, lin_flat, lin_true):
    global _INDEX_BYTES
    if not all(indices[i].data_ptr() == ptr and indices[i]._version == ver for i, (ptr, ver) in zip(pos, snap)):
        return
    nbytes = lin_flat.numel() * lin_flat.element_size() + (
        0 if lin_true is None else lin_true.numel() * lin_true.element_size()
    )
    if nbytes > INDEX_CACHE_ENTRY_MAX_BYTES:
        return
    with _INDEX_LOCK:
        _drain_dead_keys_locked()
        old = _INDEX_CACHE.pop(key, None)
        if old is not None:
            _INDEX_BYTES -= old[2]
        refs = tuple((weakref.ref(st, lambda _r, k=key: _DEAD_KEYS.append(k)), st._cdata) for st in storages)
        _INDEX_CACHE[key] = (lin_flat, lin_true, nbytes, refs)
        _INDEX_BYTES += nbytes
        while len(_INDEX_CACHE) > 1 and (len(_INDEX_CACHE) > INDEX_CACHE_SIZE or _INDEX_BYTES > INDEX_CACHE_BYTES):
            _, evicted = _INDEX_CACHE.popitem(last=False)
            _INDEX_BYTES -= evicted[2]


def _cache_store_value(vkey, lin_flat, lin_true):
    global _INDEX_VALUE_BYTES
    if vkey is None:
        return
    nbytes = lin_flat.numel() * lin_flat.element_size() + (
        0 if lin_true is None else lin_true.numel() * lin_true.element_size()
    )
    if nbytes > INDEX_CACHE_ENTRY_MAX_BYTES:
        return
    with _INDEX_LOCK:
        old = _INDEX_VALUE_CACHE.pop(vkey, None)
        if old is not None:
            _INDEX_VALUE_BYTES -= old[2]
        _INDEX_VALUE_CACHE[vkey] = (lin_flat, lin_true, nbytes, ())
        _INDEX_VALUE_BYTES += nbytes
        while len(_INDEX_VALUE_CACHE) > 1 and (
                len(_INDEX_VALUE_CACHE) > INDEX_CACHE_SIZE or _INDEX_VALUE_BYTES > INDEX_CACHE_BYTES):
            _, evicted = _INDEX_VALUE_CACHE.popitem(last=False)
            _INDEX_VALUE_BYTES -= evicted[2]


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


def _finish_index_select(self, lin_flat, lin_true, start, end, bshape, mids, flat_dims):
    shape = tuple(bshape) + tuple(self.shape[d] for d in mids)
    full = self
    if flat_dims:
        self = self[tuple(slice(0, 1) if d in flat_dims else slice(None) for d in range(self.dim()))]
    flat = self.flatten(start, end) if end > start else self
    if lin_true is not None:
        out = _ExpandedSelect.apply(full, flat.detach(), lin_flat, lin_true, start, end)
    else:
        out = flat.index_select(start, lin_flat)
    if len(shape) == 0:
        return out.squeeze(start)  # 0-d indices (direct aten calls): the indexed dims go away
    out = out.unflatten(start, shape) if len(shape) != 1 else out
    if mids and start > 0 and len(bshape):  # separated indices: the broadcast dims go first, as in ATen
        nb = len(bshape)
        order = list(range(start, start + nb)) + list(range(start)) + list(range(start + nb, out.dim()))
        out = out.permute(order).contiguous()
    return out


def index_select_flat(self, indices, checked=True, cache=True):
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
    on_host = not _tracing(self, *(indices[i] for i in pos)) and all(indices[i].device.type == "cpu" for i in pos)
    device = torch.device("cpu") if on_host else self.device
    host = _strict.route(HOST_INDEX, leaves=[indices[i] for i in pos],
                         detail="CPU index tensors: the linear index is computed on the CPU, then copied to the card") \
        if on_host and self.device.type != "cpu" else contextlib.nullcontext()
    shape = tuple(bshape) + tuple(self.shape[d] for d in mids)
    # dims of an expanded input (stride 0) hold one value: select from one element of them, not from a copy of all.
    # With a gradient the backward is ``_ExpandedSelect``'s, at the true indices: through the slice autograd would
    # put it all at index 0 of the expanded dim, wrong for the expanded tensor itself (a leaf ``zeros(3, 1).expand(3,
    # 4).requires_grad_()``, ``retain_grad``, hooks, ``autograd.grad``: [3, 0, 0, 0] for [0, 1, 0, 2]). Under a
    # ``torch.func`` transform or forward-mode AD (no Function there) ``self`` is materialized instead.
    sizes = tuple(self.shape)
    flat_dims = [d for d in range(start, end + 1) if sizes[d] > 1 and self.stride(d) == 0]
    transformed = _transformed(self) if (flat_dims or (cache and checked and on_host)) else False
    if flat_dims and transformed:
        self, flat_dims = self.contiguous(), []
    exact = bool(flat_dims) and torch.is_grad_enabled() and self.requires_grad
    can_cache = (cache and checked and on_host and not transformed
                 and torch._C._functorch.get_dynamic_layer_stack_depth() == 0
                 and all(_cacheable_cpu(indices[i]) for i in pos))
    if can_cache:
        flat_dims_rel = tuple(d - start for d in flat_dims)
        key, storages = _cache_key_and_storages(self, indices, pos, start, end, flat_dims_rel, exact)
        snap = tuple((indices[i].data_ptr(), indices[i]._version) for i in pos)
    with host:
        with (torch.inference_mode(False) if can_cache else contextlib.nullcontext()), \
                (torch.no_grad() if can_cache else contextlib.nullcontext()):
            lin = lin_true = ok = None
            for d in range(start, end + 1):
                size = sizes[d]
                sliced_size = 1 if d in flat_dims else size
                if d in mids:
                    j = mids.index(d)
                    true = torch.arange(size, device=device).reshape((size,) + (1,) * (len(mids) - 1 - j))
                    term = torch.zeros_like(true) if d in flat_dims else true
                else:
                    t = indices[d].to(device=device, dtype=torch.int64)
                    if not checked:
                        valid = (t >= -size) & (t < size)
                        ok = valid if ok is None else ok & valid
                    true = torch.where(t < 0, t + size, t).reshape(tuple(t.shape) + (1,) * len(mids))
                    term = torch.zeros_like(true) if d in flat_dims else true
                lin = term if lin is None else lin * sliced_size + term
                if exact:
                    lin_true = true if lin_true is None else lin_true * size + true
            if ok is not None:
                keep = ok.reshape(tuple(ok.shape) + (1,) * len(mids))
                lin = torch.where(keep, lin, -1)
                lin_true = None if lin_true is None else torch.where(keep, lin_true, -1)
            lin = torch.broadcast_to(lin, shape)
            lin_flat = lin.reshape(-1).to(self.device)
            if exact:
                lin_true = torch.broadcast_to(lin_true, shape).reshape(-1).to(self.device)
        if can_cache:
            _cache_store(key, storages, indices, pos, snap, lin_flat, lin_true)
            _cache_store_value(_value_key(self, indices, pos, start, end, flat_dims_rel, exact), lin_flat, lin_true)
    return _finish_index_select(self, lin_flat, lin_true, start, end, bshape, mids, flat_dims)


class _ExpandedSelect(torch.autograd.Function):
    """``index_select`` of the flattened dims ``start..end`` of ``full``, computed from ``flat`` (``full`` with its
    expanded dims sliced to one element, flattened) at ``lin``, with the backward of the select of ``full`` itself at
    ``lin_true``: the gradient lands at the true indices of the expanded dims, input-sized as ATen's index
    backward."""

    @staticmethod
    def forward(ctx, full, flat, lin, lin_true, start, end):
        ctx.save_for_backward(lin_true)
        ctx.sizes, ctx.start, ctx.end = tuple(full.shape), start, end
        return flat.index_select(start, lin)

    @staticmethod
    def backward(ctx, grad):
        (lin_true,) = ctx.saved_tensors
        sizes, start, end = ctx.sizes, ctx.start, ctx.end
        flat = sizes[:start] + (math.prod(sizes[start:end + 1]),) + sizes[end + 1:]
        g = grad.new_zeros(flat).index_add(start, lin_true, grad)
        return (g.unflatten(start, sizes[start:end + 1]) if end > start else g), None, None, None, None, None
def _transformed(*tensors):
    """Whether a ``torch.func`` transform or forward-mode AD is active (an autograd.Function applied from a
    dispatcher kernel has no kernel under the transforms, and ``_ExpandedSelect`` no jvp)."""
    if _dispatch.forward_ad_level(*tensors) >= 0:
        return True
    return torch._C._functorch.peek_interpreter_stack() is not None


def _expand_masks(indices):
    """Each mask as the integer indices of its true elements (``nonzero`` on the mask's own device)."""
    out = []
    for t in indices:
        if t is not None and t.dtype == torch.bool and t.dim() > 0:
            out.extend(t.nonzero(as_tuple=True))
        else:
            out.append(t)
    return out


def _host_index_route(self, indices):
    """Strict mode's one route per call of a device tensor indexed by CPU index tensors: checked, linearized,
    masked or moved on the host (hpu_ops' choice: no device sync). A no-op when strict mode is off."""
    if not _strict._ACTIVE["n"] or self.device.type == "cpu":
        return contextlib.nullcontext()
    host = [t for t in indices if t is not None and t.device.type == "cpu"]
    if not host or _tracing(self, *host):
        return contextlib.nullcontext()
    return _strict.route(HOST_INDEX, *host,
                         detail="CPU index tensors: checked, linearized or masked on the host, then copied to the card")


def _index_kernel(device_type, fallback):
    op = aten.index.Tensor

    def kernel(self, indices):
        if self.device.type != device_type:
            return op._op_dk(fallback, self, indices)
        hit = _cache_lookup(self, indices)
        if hit is not None:
            count("index")
            return _finish_index_select(self, *hit)
        with _host_index_route(self, indices):
            plan = _plan(self, indices)
            if plan is None:
                return op._op_dk(fallback, self, indices)
            count("index")
            if plan == "move":
                return op._op_dk(fallback, self, [t if t is None else t.to(self.device) for t in indices])
            if plan == "mask":
                return index_select_flat(self, _expand_masks(indices), cache=False)
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
        with unset_fake_temporarily():  # while tracing a fake fill would not check
            # ATen's checked cast, for its errors only: a one-element masked_fill_, ATen's own put of one value
            # (``full`` stopped checking: torch 2.15 gives inf for float16 1e6, where the put raises)
            torch.zeros(1, dtype=self.dtype).masked_fill_(torch.ones(1, dtype=torch.bool), values.item())
    except Exception:  # noqa: BLE001  (ATen's overflow error, or no value while tracing)
        return None
    return (values.real if values.is_complex() else values).to(self.dtype)


def _mask_value(self, indices, values, accumulate=False):
    """Converted put value when ``self[..., mask] = value`` should run through ``where``, else None."""
    pos = _positions(indices)
    if accumulate or len(pos) != 1 or values.numel() != 1 or values.dim() > 1 or self.is_complex():
        return None
    mask, p = indices[pos[0]], pos[0]
    if mask.dtype != torch.bool or p != len(indices) - 1 or len(indices) > self.dim():
        return None  # byte masks: ATen's error; a mask with an explicit None after it: ATen's alignment
    if mask.dim() == 0 or tuple(mask.shape) != tuple(self.shape[p:p + mask.dim()]):
        return None
    if mask.device not in (self.device, torch.device("cpu")) or values.device not in (self.device, torch.device("cpu")):
        return None  # ATen's device error
    if _shares_memory(self, mask) or _shares_memory(self, values) or not _dense(self):
        return None  # ATen raises on aliasing; and writes into overlapping views its own way
    return _value(self, values)  # ATen raises on the value's cast (overflow, complex to real)


def _mask_takes(self, indices, values, accumulate=False):
    """The case Habana's lazy launch fails on (SPNN, DeepGeoCCA): ``self[..., mask] = value``, one boolean mask that
    is the last index (ATen aligns a mask followed by an explicit ``None`` differently), one value, a dense
    destination that shares no memory with the mask or the value. Leading masks too (``x[x < eps] = eps``): keeping
    those native brought the launch failure back in SPNN and DeepGeoCCA (measured), so ``where`` allocates one copy of
    the destination there instead of the in-place ``masked_fill_``."""
    return _mask_value(self, indices, values, accumulate) is not None


def mask_fill(self, indices, values, accumulate=False, value=None):
    """``self[..., mask] = value`` as ``self.copy_(where(mask, value, self))``; returns self."""
    p = _positions(indices)[0]
    mask = indices[p]
    mask = mask.to(self.device).reshape((1,) * p + tuple(mask.shape) + (1,) * (self.dim() - p - mask.dim()))
    if value is None:
        value = _value(self, values)
    return self.copy_(torch.where(mask, value.reshape(()).to(self.device), self))


def install_index(device_type):
    """``index`` on ``device_type`` as above, at the autograd and backend keys. Returns an undo."""
    op = aten.index.Tensor
    undo_kernels = register_kernels(device_type, [
        (op, "Autograd{}", _index_kernel(device_type, DispatchKey.Autograd)),
        (op, "{}", _index_kernel(device_type, DispatchKey.CompositeExplicitAutogradNonFunctional)),
    ])

    def undo():
        undo_kernels()
        _clear_cache()

    return undo


def install_mask_fill(device_type):
    """``index_put_`` of one value through one boolean mask, on ``device_type``, as ``where``. Returns an undo."""
    op = aten.index_put_.default

    def kernel(self, indices, values, accumulate=False):
        val = _mask_value(self, indices, values, accumulate)
        if val is not None:
            count("mask_fill")
            return mask_fill(self, indices, values, accumulate, value=val)
        return op._op_dk(DispatchKey.Autograd, self, indices, values, accumulate)

    return register_kernels(device_type, [(op, "Autograd{}", kernel)])
