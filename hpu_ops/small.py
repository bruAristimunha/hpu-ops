"""``bincount``, ``histc``, and scoped lazy-mode rewrites for ``logsumexp`` and ``index_add`` (hpu-ops #21).

Gaudi's ``bincount`` (bridge 1.21.4) grows with the input and steeply with the bins: 409600 elements take 7.4 ms
eager / 4.7 ms lazy with 64 bins, 230 / 225 ms with 1000 bins and 427 ms eager with 4096 bins, where one CPU core
takes 0.8 ms; even at 4096..40960 elements with >= 256 bins (eager) / >= 128 bins (lazy) it takes 2.4-24 ms.
Here, for 1-D integer inputs of 65536 to 2**24 - 1 elements (or 4096..65535 when ``minlength >= 256`` eager /
``>= 128`` lazy) without weights and with at most 4096 bins: one ``aminmax`` read (skipped for ``uint8`` with
``minlength >= 256``, since values are in ``0..255``) and ``histc`` over ``[a, a + 255]`` with 256 bins per chunk,
zero-padded when ``minlength > hi + 1``. Under ``torch.compile``, ``uint8`` inputs with ``minlength >= 256`` take
``F.one_hot(x.long(), 256).float().sum(0).long()`` (zero-padded when ``minlength > 256``), with zero host syncs
and zero graph breaks.

Scoped lazy-mode winners (eager and ``torch.compile`` stay native where Habana is already faster or parity):
- ``logsumexp`` (lazy mode, ``numel <= 2**19``): ``lse_safe`` (``amax(keepdim=True)`` + ``where(m.isinf(), 0, m)``
  + ``exp().sum().log() + m`` reshaped without ``squeeze``, avoiding Habana's ``all_dims`` squeeze and 1-D
  ``expand`` bugs) and an ``autograd.Function`` (``_LseFn``) with exact analytical backward ``g * exp(x - out)``.
- ``index_add`` (lazy mode, ``not torch.is_grad_enabled()``, ``dim == 0``, float32, ``self`` of at most 256 rows,
  ``source.shape[0] >= 40960`` and (at least 1280 source rows per row of ``self`` or rows of >= 128 elements)):
  ``index_put_((slot,), source * alpha, accumulate=True)`` into a copy of ``self`` with one scratch row, at
  ``Autograd{}`` and ``{}`` (inference mode). ``slot`` wraps ``[-rows, -1]`` and sends any other index out of range
  to the scratch row, as Habana's ``index_add`` does (ATen raises, which needs a host read); Habana's ``index_put``
  stops the device on an index of -1. Past 512 rows Habana's lazy accumulating ``index_put`` returns wrong sums
  for an index made by an ``add`` in the same graph (``repro/index_put_lazy_add_index.py``), hence the row cap.

Ops where native Habana wins or no exact rewrite beats it through the dispatcher (``trace``, ``scatter_add``,
``where``, ``lerp``, ``cumsum``, ``cumprod``, ``allclose``) stay native.
"""
import math
import struct
import threading

import torch
import torch.nn.functional as F
from torch._C import DispatchKey

from . import _strict
from ._dispatch import chain, count, override_kernels, register_kernels, static
from .linalg import _is_fake, _transformed

aten = torch.ops.aten
BINCOUNT_MIN_NUMEL = 1 << 16  # below, only high-minlength calls beat Habana's kernel (measured)
BINCOUNT_SMALL_MIN_NUMEL = 4096
BINCOUNT_SMALL_MIN_BINS_EAGER = 256
BINCOUNT_SMALL_MIN_BINS_LAZY = 128
BINCOUNT_MAX_NUMEL = 1 << 24  # histc / one_hot float32 counts are exact below 2**24
BINCOUNT_LAZY_MAX_CHUNKS = 256  # chunked lazy one_hot counting stays a bounded lazy graph
BINCOUNT_MAX_BINS = 4096  # 16 histc launches
BINCOUNT_CHUNK = 256  # histc compiles up to ~300 bins on Gaudi2
INTEGRAL = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
LOGSUMEXP_MAX_NUMEL = 1 << 19  # 524288: past ~1M elements (e.g. 6400x22x22) Habana's native fwdbwd wins
INDEX_ADD_MIN_ROWS = 40960  # source rows; below, Habana's native lazy kernel wins in isolated calls
INDEX_ADD_MAX_SELF_ROWS = 256  # Habana's lazy accumulating index_put into more than 512 rows misreads an index made
#                                by an add in the same graph (repro/index_put_lazy_add_index.py): 2x margin
INDEX_ADD_MIN_PER_ROW = 1280  # source rows per row of self below a wide row (measured: 640 is parity, 320 loses)
INDEX_ADD_WIDE_ROW = 128  # elements per row of self from which the rewrite wins at every contention measured
_MODE = {}
_SCALAR_TLS = threading.local()


def _lazy():
    """Whether Habana's bridge runs in lazy mode (read once; fixed when the bridge loads). Off a Gaudi stack, False
    (the CPU tests patch this). A failed import is not cached: gates that enable a route on ``not _lazy()`` must not
    keep a pre-bridge ``False`` for the rest of a lazy process."""
    if "lazy" not in _MODE:
        try:
            from habana_frameworks.torch.utils.internal import is_lazy
        except ImportError:
            return False
        _MODE["lazy"] = bool(is_lazy())
    return _MODE["lazy"]


def _mark_step():
    """Flush the pending lazy graph (no-op off a lazy Gaudi stack)."""
    if "mark_step" not in _MODE:
        try:
            from habana_frameworks.torch.core import mark_step

            _MODE["mark_step"] = mark_step
        except ImportError:
            _MODE["mark_step"] = None
    if _MODE["mark_step"] is not None:
        _MODE["mark_step"]()


def install_scalar_dense(device_type):
    """``.item()`` / ``bool()`` / ``int()`` of a device scalar (``aten::_local_scalar_dense``).

    The read itself stays Habana's kernel: a Python scalar cannot be produced without
    reading the device. In lazy mode the pending graph is flushed first, so the graph
    built after the read starts fresh. Habana's lazy compiler segfaults when a long
    data-dependent loop (the Braindecode quantizers' 8192-code k-means ``initialize``,
    entered right after ``bool(self.initted.item())``) is fused into the graph that
    produced its input; the same loop as its own graph compiles and runs (bridge
    1.21.4). Any model whose forward reads a scalar therefore runs stock in lazy mode;
    removing the per-forward sync itself still needs the value mirrored in Python,
    which only the model can know is safe."""
    op = aten._local_scalar_dense.default

    def kernel(x):
        if _lazy() and _concrete(x):
            _mark_step()
            count("scalar_dense")
        _SCALAR_TLS.below = True  # the backend kernel below must not flush a second time
        try:
            return op._op_dk(DispatchKey.Autograd, x)
        finally:
            _SCALAR_TLS.below = False

    def backend_kernel(x):
        # The backend key is the only one inference-mode reads reach (autograd's keys
        # are excluded there). The native read is a copy to the host: _local_scalar_dense
        # has no second overload to fall through to, and redispatching it would recurse.
        if not getattr(_SCALAR_TLS, "below", False) and _lazy() and _concrete(x):
            _mark_step()
            count("scalar_dense")
        host = torch.empty(x.shape, dtype=x.dtype, device="cpu")
        with torch._C._AutoDispatchBelowADInplaceOrView():
            aten.copy_.default(host, x)
        if x.dtype == torch.bfloat16:
            # NumPy has no bfloat16 (#102), and .item() would redispatch to this kernel:
            # bf16 is the high half of float32, so widen the bits directly
            bits = int.from_bytes(bytes(host.untyped_storage()[:2]), "little")
            return struct.unpack("f", struct.pack("I", bits << 16))[0]
        return host.numpy().item()

    return chain([register_kernels(device_type, [(op, "Autograd{}", kernel)]),
                  override_kernels(device_type, [(op, backend_kernel)])])


def _concrete(x):
    """True for a plain eager call: the host read needs real values, outside any transform or tracer. Declined, so
    Habana's kernel runs: ``torch.compile`` / fake mode (checked first so Dynamo does not graph-break on C++
    functorch queries, and since ``is_compiling()`` is false in Dynamo ``get_fake_value`` and AOT dispatch on
    2.6/2.8), functorch transforms and forward-mode AD (``linalg._transformed``; the layer depth too, since below
    ``torch.func.grad``'s layer the interpreter stack reads empty), functional tensors, and any ``__torch_dispatch__``
    mode such as ``make_fx``'s proxy mode."""
    if torch.compiler.is_compiling() or _is_fake(x) or torch._guards.detect_fake_mode() is not None:
        return False
    if _transformed() or torch._C._functorch.get_dynamic_layer_stack_depth() > 0 or _strict.foreign_modes():
        return False
    return not torch._is_functional_tensor(x) and not torch._C._functorch.is_functorch_wrapped_tensor(x)


def count_bins(xf, bins):
    """float32 counts of the integer values ``0 .. bins - 1`` of the float32 tensor ``xf``, through ``histc`` with
    Python-int bounds in chunks of at most 256 bins; values outside ``[0, bins - 1]`` (e.g. a -1 sentinel) are not
    counted. ``bins >= 2``: one bin would make ``histc`` take the data's range."""
    if bins <= BINCOUNT_CHUNK:
        return torch.histc(xf, bins, 0, bins - 1)
    starts = list(range(0, bins - BINCOUNT_CHUNK, BINCOUNT_CHUNK)) + [bins - BINCOUNT_CHUNK]
    parts = [torch.histc(xf, BINCOUNT_CHUNK, a, a + BINCOUNT_CHUNK - 1) for a in starts]
    done = starts[-2] + BINCOUNT_CHUNK  # the last chunk adds the bins from here on
    parts[-1] = parts[-1][done - starts[-1]:]
    return torch.cat(parts)


def bincount_histc(x, bins):
    """Counts of the values ``0 .. bins - 1`` of ``x`` (every value known to be in that range), int64."""
    xf = x.float()
    if bins == 1:  # every value is 0
        return torch.histc(xf, 1, 0, 0).long()
    return count_bins(xf, bins).long()


def _bincount_min_bins():
    return BINCOUNT_SMALL_MIN_BINS_LAZY if _lazy() else BINCOUNT_SMALL_MIN_BINS_EAGER


def _takes(x, weights=None, minlength=0):
    n = x.numel()
    if not (weights is None and x.dim() == 1 and x.dtype in INTEGRAL
            and isinstance(minlength, int) and 0 <= minlength <= BINCOUNT_MAX_BINS
            and static(*x.shape) and _concrete(x)):
        return False
    if BINCOUNT_MIN_NUMEL <= n < BINCOUNT_MAX_NUMEL:
        return True
    return BINCOUNT_SMALL_MIN_NUMEL <= n < BINCOUNT_MIN_NUMEL and minlength >= _bincount_min_bins()


def _compile_uint8_takes(x, weights=None, minlength=0):
    tracing = torch.compiler.is_compiling() or _is_fake(x) or torch._guards.detect_fake_mode() is not None
    return (tracing and weights is None and x.dim() == 1 and x.dtype == torch.uint8
            and isinstance(minlength, int) and minlength >= 256
            and x.numel() < BINCOUNT_MAX_NUMEL and static(*x.shape))


def _compile_minlength_takes(x, weights=None, minlength=0):
    """Traced bincount with a static output length: ``histc`` over ``[0, minlength)`` with no
    value-range read (the read is the host sync strict mode counts). Values at or above
    ``minlength`` count into the last bin instead of extending the output - the shape a
    compiled graph must have anyway; argmin-produced buckets (the measured caller) are in
    range by construction. Negative values (ATen raises on them) count into bin 0: a compiled
    graph cannot raise on data, and no measured caller produces them."""
    tracing = torch.compiler.is_compiling() or _is_fake(x) or torch._guards.detect_fake_mode() is not None
    return (tracing and not _lazy() and weights is None and x.dim() == 1 and x.dtype in INTEGRAL
            and isinstance(minlength, int) and 256 <= minlength <= BINCOUNT_MAX_BINS
            and x.numel() < BINCOUNT_MAX_NUMEL and static(*x.shape))


def _lazy_wide_takes(x, weights=None, minlength=0):
    """Lazy mode, wide static output (e.g. assignments into an 8192-code codebook):
    Habana's lazy ``bincount`` kernel segfaults there, and the eager route's
    value-range read would be a host sync. ``one_hot`` over the clamped indices,
    counted in chunks, is exact at any numel below ``BINCOUNT_MAX_NUMEL`` (per-bin
    counts stay float32-exact); above-range values count into the last bin
    (argmax/argmin buckets, the measured callers, are in range by construction).
    Concrete calls only: traced lazy graphs keep declining, as before."""
    if not (isinstance(minlength, int) and 256 <= minlength <= 16384):
        return False
    chunks = -(-x.numel() // max(1, BINCOUNT_MAX_NUMEL // minlength))  # ceil
    return (_lazy() and weights is None and x.dim() == 1 and x.dtype in INTEGRAL
            and x.numel() < BINCOUNT_MAX_NUMEL and chunks <= BINCOUNT_LAZY_MAX_CHUNKS
            and static(*x.shape) and _concrete(x))


def _bincount_lazy(x, bins):
    """int64 counts of ``x`` clamped into ``0 .. bins - 1`` for lazy mode.

    The pending graph is flushed first: fused into the graph that produced its
    input (the tokenizer's 10-iteration k-means ``initialize`` loop), the counting
    graph crashes Habana's lazy compiler (bridge 1.21.4); as its own graph it
    compiles and runs. The input is then counted in chunks so the ``one_hot``
    intermediate stays bounded at any numel."""
    _mark_step()
    xc = x.long().clamp(0, bins - 1)
    chunk = max(1, BINCOUNT_MAX_NUMEL // bins)
    if xc.numel() <= chunk:
        return F.one_hot(xc, bins).float().sum(0).long()
    counts = None
    for part in xc.split(chunk):
        part_counts = F.one_hot(part, bins).float().sum(0)
        counts = part_counts if counts is None else counts + part_counts
    return counts.long()


def install_bincount(device_type):
    """``bincount`` of 1-D integer tensors on ``device_type`` through ``histc`` (eager/lazy), ``one_hot``
    (``uint8`` with ``minlength >= 256`` under ``torch.compile``), and chunked, graph-segmented
    ``one_hot`` for wide static ``minlength`` in lazy mode. Returns an undo."""
    op = aten.bincount.default

    def kernel(x, weights=None, minlength=0):
        if _compile_uint8_takes(x, weights, minlength):
            count("bincount")
            counts = F.one_hot(x.long(), 256).float().sum(0).long()
            return counts if minlength == 256 else F.pad(counts, (0, minlength - 256))
        if _compile_minlength_takes(x, weights, minlength):
            count("bincount")
            return bincount_histc(x.clamp(0, minlength - 1), minlength)
        if _lazy_wide_takes(x, weights, minlength):
            count("bincount")
            return _bincount_lazy(x, minlength)
        if _takes(x, weights, minlength):
            if x.dtype == torch.uint8 and minlength >= 256:  # unsigned [0..255]: no host sync needed
                count("bincount")
                counts = bincount_histc(x, 256)
                return counts if minlength == 256 else F.pad(counts, (0, minlength - 256))
            with _strict.host_read("bincount: value range", x):
                lo, hi = (int(v) for v in torch.stack(torch.aminmax(x)).tolist())
            if lo < 0:  # ATen's error; Habana's kernel counts negative values instead
                raise RuntimeError("bincount only supports 1-d non-negative integral inputs.")
            active_bins = hi + 1
            bins = max(active_bins, minlength)
            if bins <= BINCOUNT_MAX_BINS:
                count("bincount")
                counts = bincount_histc(x, active_bins)
                return counts if active_bins >= minlength else F.pad(counts, (0, minlength - active_bins))
        return op._op_dk(DispatchKey.Autograd, x, weights, minlength)

    return register_kernels(device_type, [(op, "Autograd{}", kernel)])


# ---- histc --------------------------------------------------------------------------------------------------------
# Gaudi's histc (bridge 1.21.4, repro/histc_float_bounds.py, repro/probe_histc_scope.py, eager and lazy) reads its
# bounds as ints: a bound passed as a Python float (any value, 0.0 too) fails with "expected int" from
# habana::FillHistcParams (eager: in the launch thread, and the device stops); from about 350 bins its graph does not
# compile; compiled graphs abort on both. Its float32 division is not IEEE (154 of 2**20 truncations of
# (x - L) * bins / (R - L) differ from the CPU's), so the CPU's bins cannot be computed on the card:
#   ATen's CPU histc puts x in bin trunc((x - L) * bins / (R - L)) in x's dtype (L, R rounded to it), the last bin
#   closed, and skips x outside [L, R] (NaN too). That position is monotone in x, so bin k starts at the smallest
#   value of the dtype whose position is >= k (+inf when no value reaches it): ``histc_edges`` finds those on the
#   host by bisection over the dtype's ordered bit patterns, with the CPU kernel's own arithmetic (cached per bounds,
#   bins, dtype and device).
# On the card only comparisons: one ``bucketize`` of x against those edges (Gaudi's is exact for finite values, and
# linear in the number of edges), then the counts through histc with Python-int bounds up to 256 bins
# (``count_bins``), ``scatter_add_`` above, capped as ATen's one-thread kernel caps them: it adds 1 in x's dtype, so
# a bin stops at 2**p (p the significand bits: 256 in bfloat16, 2048 in float16, 2**24 in float32). float64: the
# card holds float32 values, read through an offset-free view (eager reads float64 views with an offset at twice
# the offset) and compared with the edges rounded up to float32.
HISTC_NATIVE_MAX_BINS = 300  # Gaudi's histc with int bounds: measured to 300 bins; 350 does not compile
HISTC_DTYPES = {torch.float32: (torch.int32, 32), torch.bfloat16: (torch.int16, 16), torch.float16: (torch.int16, 16),
                torch.float64: (torch.int64, 64)}
_C_NAMES = {torch.float32: "float", torch.float64: "double", torch.float16: "at::Half",
            torch.bfloat16: "at::BFloat16"}
SIGNIFICAND_BITS = {torch.float32: 24, torch.float64: 53, torch.bfloat16: 8, torch.float16: 11}
EDGE_CACHE_SIZE = 64
EDGE_CACHE_BYTES = 64 << 20
_EDGE_LOCK = threading.Lock()
_EDGE_CACHE = {}  # (lo, hi, bins, dtype, device) -> (edges on the device, low, high)


def _ordered(v):
    """int64 keys of the float tensor ``v`` (no NaN), in the order of the values (-0.0 just below +0.0)."""
    bits, width = HISTC_DTYPES[v.dtype]
    b = v.view(bits).to(torch.int64)
    magnitude = b & ((1 << (width - 1)) - 1) if width < 64 else b & 0x7FFFFFFFFFFFFFFF
    return torch.where(b >= 0, b, -magnitude - 1)


def _unordered(k, dtype):
    """The values of ``dtype`` whose ``_ordered`` keys are ``k``."""
    bits, width = HISTC_DTYPES[dtype]
    sign = -(1 << 63) if width == 64 else 1 << (width - 1)
    b = torch.where(k < 0, (-(k + 1)) | sign, k)
    if width < 64:
        b = torch.where(b >= 1 << (width - 1), b - (1 << width), b)
    return b.to(bits).view(dtype)


def _to_dtype(value, dtype):
    """``value`` (a Python float) in ``dtype``, with ATen's overflow check: a finite value past the dtype's range
    raises, also when it would round to the largest value (65505 in float16)."""
    if math.isfinite(value) and abs(value) > torch.finfo(dtype).max:
        raise RuntimeError(f"value cannot be converted to type {_C_NAMES[dtype]} without overflow")
    return torch.tensor(value, dtype=torch.float64).to(dtype)


def _float32(t, toward):
    """float64 ``t`` rounded to float32 toward ``toward`` (+-inf)."""
    near = t.float()
    off = near.double() < t if toward > 0 else near.double() > t
    return torch.where(off, torch.nextafter(near, torch.full_like(near, toward)), near)


def _positions(L, R, bins, dtype):
    """ATen's position of values of ``dtype`` in ``[L, R]``: ``trunc((x - L) * bins / (R - L))``, every step rounded
    to ``dtype`` (measured to match ATen's CPU kernel bin for bin in float16 and bfloat16). Where a step overflows
    (float16: ``(x - L) * bins`` past 65504) ATen's position is infinite and its kernel's write undefined; there the
    position is the float64 one, raised to the last finite position so that positions stay monotone in x."""
    scale = torch.tensor(bins, dtype=dtype)

    def exact(v):
        return ((v - L) * scale / (R - L)).trunc().to(torch.float64)

    if torch.isfinite(exact(R.reshape(1))).all():
        return exact
    top, finite = int(_ordered(R.reshape(1))[0]), int(_ordered(L.reshape(1))[0])
    hi_key = top  # bisection: the largest key whose position is finite
    while finite < hi_key:
        mid = (finite + hi_key + 1) // 2
        if bool(torch.isfinite(exact(_unordered(torch.tensor([mid]), dtype)))):
            finite = mid
        else:
            hi_key = mid - 1
    last = float(exact(_unordered(torch.tensor([finite]), dtype)))

    def extended(v):
        q = exact(v)
        wide = ((v.double() - L.double()) * bins / (R.double() - L.double())).floor().clamp(min=last)
        return torch.where(torch.isfinite(q), q, wide)

    return extended


def histc_edges(lo, hi, bins, dtype):
    """Host tensors ``(edges, low, high)`` for ATen's CPU histc over ``[lo, hi]`` (finite, ``lo < hi``) in ``dtype``:
    ``edges[k]`` (k = 1 .. bins - 1) is the smallest value of ``dtype`` whose position is ``k`` or more (``inf`` when
    none is), ``edges[0] = -inf``, ``edges[bins] = inf``; ``low`` / ``high`` the bounds as ATen compares them. For
    float64, everything rounded to the float32 values the card holds. Independent of the default dtype."""
    L, R = _to_dtype(lo, dtype), _to_dtype(hi, dtype)
    width = R - L
    store = torch.float32 if dtype == torch.float64 else dtype
    if torch.isinf(width):  # a width past the dtype's range: ATen counts nothing (its edges are not finite)
        return _framed(torch.full((bins - 1,), math.inf, dtype=store)), math.inf, -math.inf
    if width == 0:  # distinct bounds that round to one value: ATen's position is trunc(0 / 0), measured bin 0
        return _framed(torch.full((bins - 1,), math.inf, dtype=store)), float(L), float(R)
    position = _positions(L, R, bins, dtype)
    k = torch.arange(1, bins, dtype=torch.float64)
    top = int(_ordered(R.reshape(1))[0])
    below = torch.full((bins - 1,), int(_ordered(L.reshape(1))[0]), dtype=torch.int64)
    above = torch.full((bins - 1,), top + 1, dtype=torch.int64)  # top + 1: no value in [L, R] reaches k
    while bool((below < above).any()):
        mid = (below >> 1) + (above >> 1) + (below & above & 1)  # the floor of the mean, without int64 overflow
        reached = position(_unordered(mid.clamp(max=top), dtype)) >= k
        above = torch.where(reached, mid, above)
        below = torch.where(reached, below, mid + 1)
    edges = torch.where(below > top, torch.tensor(math.inf, dtype=dtype), _unordered(below.clamp(max=top), dtype))
    if dtype == torch.float64:  # float32 v >= e  <=>  v >= e rounded up to float32; v <= R  <=>  v <= R rounded down
        return _framed(_float32(edges, math.inf)), float(_float32(L, math.inf)), float(_float32(R, -math.inf))
    return _framed(edges), float(L), float(R)


def _framed(edges):
    """``[-inf, edges..., inf]``: entry k is bin k's lower edge, entry k + 1 its upper edge (k = 0 .. bins - 1)."""
    inf = torch.full((1,), math.inf, dtype=edges.dtype)
    return torch.cat([-inf, edges, inf])


def _host_edges(lo, hi, bins, dtype):
    from torch.utils._python_dispatch import _disable_current_modes

    with _disable_current_modes():  # plain host tensors, also while a fake or functional mode traces the call
        return histc_edges(lo, hi, bins, dtype)


def _cache_bytes():
    return sum(entry[0].numel() * entry[0].element_size() for entry in _EDGE_CACHE.values())


def _edges_on(lo, hi, bins, dtype, device, traced):
    if traced:  # a fresh constant of the graph
        edges, low, high = _host_edges(lo, hi, bins, dtype)
        return torch.tensor(edges.tolist(), dtype=edges.dtype, device=device), low, high
    key = (lo, hi, bins, dtype, str(device))
    with _EDGE_LOCK:
        entry = _EDGE_CACHE.get(key)
    if entry is None:
        edges, low, high = _host_edges(lo, hi, bins, dtype)
        entry = (edges.to(device), low, high)
        with _EDGE_LOCK:  # the newest entry always stays; the oldest go past 64 entries or EDGE_CACHE_BYTES
            _EDGE_CACHE[key] = entry
            while len(_EDGE_CACHE) > 1 and (len(_EDGE_CACHE) > EDGE_CACHE_SIZE or _cache_bytes() > EDGE_CACHE_BYTES):
                _EDGE_CACHE.pop(next(iter(_EDGE_CACHE)))
    return entry


def _card_values(x, traced):
    """The values ``x`` holds, flat, in the dtype the card compares them in (float64: float32). A float64 view with
    a storage offset is read through an offset-free view of its whole storage first: Gaudi's eager kernels read such
    views at twice the offset (``hpu_ops.wide``), ``.float()`` included."""
    if x.dtype != torch.float64:
        return x.reshape(-1)
    if not traced and x.storage_offset() != 0:
        whole = x.as_strided((x.untyped_storage().nbytes() // x.element_size(),), (1,), 0).float()
        return whole.as_strided(x.shape, x.stride(), x.storage_offset()).reshape(-1)
    return x.float().reshape(-1)


def _histc_range(flat, lo, hi):
    """ATen's ``histc_select_outer_bin_edges``: equal bounds take the data's range (one host read; the caller declines
    under a tracer), equal again are widened by 1; then ATen's checks. ``flat``: ``_card_values`` of the input."""
    lo, hi = float(lo), float(hi)
    if lo == hi and flat.numel() > 0:
        # the NaN flag as a max of x != x: with isnan().any() (or a sum) in the same lazy graph, Habana's compile fails
        lo, hi, nan = torch.stack([flat.amin(), flat.amax(), (flat != flat).to(flat.dtype).amax()]).tolist()
        if nan:
            lo = hi = math.nan
    if lo == hi:
        lo, hi = lo - 1, hi + 1
    if not (math.isfinite(lo) and math.isfinite(hi)):
        raise RuntimeError(f"torch.histc: range of [{lo:g}, {hi:g}] is not finite")
    if not lo < hi:
        raise RuntimeError("torch.histc: max must be larger than min")
    return lo, hi


def _cap(counts, dtype):
    """ATen's one-thread CPU kernel adds 1 per value in ``dtype``: a count stops at 2**p (p significand bits)."""
    return counts.clamp(max=float(2 ** SIGNIFICAND_BITS[dtype])).to(dtype)


def _count(position, inside, bins):
    """float32 counts of ``position`` (in ``[0, bins - 1]``) where ``inside``."""
    if bins == 1:
        return inside.to(torch.int32).sum().reshape(1).float()
    if bins <= BINCOUNT_CHUNK:
        return count_bins(torch.where(inside, position, -1).float(), bins)
    slots = torch.where(inside, position, bins)  # one slot past the end takes the values outside
    return torch.zeros(bins + 1, dtype=torch.float32, device=position.device).scatter_add_(
        0, slots, torch.ones_like(slots, dtype=torch.float32))[:bins]


def histc_by_edges(x, bins, lo, hi, traced=False, flat=None):
    """``torch.histc(x, bins, lo, hi)`` for finite ``lo < hi`` as ATen's one-thread CPU kernel computes it."""
    edges, low, high = _edges_on(lo, hi, bins, x.dtype, x.device, traced)
    flat = _card_values(x, traced) if flat is None else flat
    inside = (flat >= low) & (flat <= high)  # NaN is outside
    position = torch.bucketize(flat, edges[1:-1], right=True) if bins > 1 else None
    return _cap(_count(position, inside, bins), x.dtype)


def histc_traced_range(x, bins):
    """Equal bounds under a tracer: the data's range is read on the card, so the bins come from the card's own
    arithmetic (as Gaudi's kernel bins them: not ATen's at the edges), and NaN / inf give counts where ATen raises."""
    if x.numel() == 0:  # ATen: no data, bounds widened by 1, nothing counted
        return torch.zeros(bins, dtype=x.dtype, device=x.device)
    flat = _card_values(x, traced=True)
    lo, hi = flat.amin(), flat.amax()
    same = lo == hi
    lo, hi = torch.where(same, lo - 1, lo), torch.where(same, hi + 1, hi)
    inside = (flat >= lo) & (flat <= hi)
    q = ((flat - lo) * bins / (hi - lo)).nan_to_num(0.0).clamp(0, bins - 1)
    return _cap(_count(q.long(), inside, bins), x.dtype)


def _native_histc_ok(bins, lo, hi):
    """Calls Gaudi's kernel takes (measured): bounds given as Python ints (bool too), at most 300 bins."""
    return isinstance(lo, int) and isinstance(hi, int) and bins <= HISTC_NATIVE_MAX_BINS


def _native_histc(x, bins, lo, hi):
    """Gaudi's own kernel, through ``histc.out`` below autograd (``histc.default`` at the backend key is ours)."""
    out = torch.empty(bins, dtype=x.dtype, device=x.device)
    with torch._C._AutoDispatchBelowADInplaceOrView():
        return aten.histc.out(x, bins, lo, hi, out=out)


def _routed_histc(x, bins, lo, hi):
    """The rewrite's result for a call it takes, or None for one it leaves to Gaudi's kernel."""
    if not (x.dtype in HISTC_DTYPES and isinstance(bins, int) and bins > 0 and not _native_histc_ok(bins, lo, hi)
            and not _transformed()):
        return None
    traced = _is_fake(x) or not _concrete(x)
    count("histc")
    if float(lo) == float(hi) and traced:
        return histc_traced_range(x, bins)
    flat = _card_values(x, traced)
    bounds = _histc_range(flat, lo, hi)
    for value in bounds:  # ATen converts the bounds to the dtype before it looks at the data
        _to_dtype(value, x.dtype)
    if x.numel() == 0:
        return torch.zeros(bins, dtype=x.dtype, device=x.device)
    return histc_by_edges(x, bins, *bounds, traced=traced, flat=flat)


def install_histc(device_type):
    """``torch.histc`` on ``device_type`` through ``histc_by_edges`` where Gaudi's kernel fails: a bound given as a
    Python float, or more than 300 bins. At ``Autograd{}`` (eager, lazy, traced graphs) for ``histc`` and
    ``histc.out``, and at the backend key for ``histc`` (inference mode; Gaudi's kernel is then reached through
    ``histc.out``). Returns an undo."""
    op, out_op = aten.histc.default, aten.histc.out

    def autograd_kernel(x, bins=100, min=0, max=0):  # noqa: A002  (ATen's argument names)
        routed = _routed_histc(x, bins, min, max)
        return op._op_dk(DispatchKey.Autograd, x, bins, min, max) if routed is None else routed

    def out_kernel(x, bins=100, min=0, max=0, *, out):  # noqa: A002
        routed = _routed_histc(x, bins, min, max)
        if routed is None:
            return out_op._op_dk(DispatchKey.Autograd, x, bins, min, max, out=out)
        out.resize_(routed.shape)
        return out.copy_(routed)

    def backend_kernel(x, bins=100, min=0, max=0):  # noqa: A002
        routed = _routed_histc(x, bins, min, max)
        return _native_histc(x, bins, min, max) if routed is None else routed

    undos = [register_kernels(device_type, [(op, "Autograd{}", autograd_kernel), (out_op, "Autograd{}", out_kernel)]),
             override_kernels(device_type, [(op, backend_kernel)])]

    def undo():
        chain(undos)()
        with _EDGE_LOCK:
            _EDGE_CACHE.clear()

    return undo


# ---- logsumexp (lazy mode only) -----------------------------------------------------------------------------------

def _lse_dims(x, dim):
    """Normalised sorted reduced dims for ``logsumexp``, or None for calls that should fall through to ATen
    (0-D tensors, empty ``dim`` on >= 1-D, out-of-range or duplicate dims)."""
    nd = x.dim()
    if nd == 0:
        return None
    if isinstance(dim, int):
        dims = [dim]
    elif isinstance(dim, (list, tuple)) and len(dim) > 0:
        dims = list(dim)
    else:
        return None
    norm = []
    for d in dims:
        if not isinstance(d, int) or not -nd <= d < nd:
            return None
        norm.append(d % nd)
    if len(set(norm)) != len(norm):
        return None
    return sorted(norm)


def _lse_takes(x, dim):
    if not (_lazy() and x.numel() <= LOGSUMEXP_MAX_NUMEL and static(*x.shape)
            and not x.is_complex() and _concrete(x)):
        return None
    return _lse_dims(x, dim)


def lse_safe(x, dims, keepdim=False):
    """Exact ``logsumexp(x, dims, keepdim)`` via ``amax(keepdim=True)`` + ``where(m.isinf(), 0, m)``.
    Uses ``reshape`` instead of ``squeeze`` to avoid Habana lazy's ``all_dims`` squeeze and 1-D ``expand`` bugs."""
    if not x.is_floating_point():
        x = x.float()
    out_shape = [1 if d in dims else s for d, s in enumerate(x.shape)] if keepdim else [
        s for d, s in enumerate(x.shape) if d not in dims]
    if any(x.shape[d] == 0 for d in dims):
        return torch.full(out_shape, -math.inf, dtype=x.dtype, device=x.device)
    m = x.amax(dims, keepdim=True)
    m_safe = torch.where(m.isinf(), 0.0, m)
    y = (x - m_safe).exp().sum(dims, keepdim=True).log() + m
    return y if keepdim else y.reshape(out_shape)


class _LseFn(torch.autograd.Function):
    """Lazy-mode ``logsumexp`` with ``lse_safe`` forward and exact ATen-matching analytical backward."""

    @staticmethod
    def forward(ctx, x, dims, keepdim):
        out = lse_safe(x, dims, keepdim)
        ctx.save_for_backward(x, out)
        ctx.kept_shape = [1 if d in dims else s for d, s in enumerate(x.shape)]
        ctx.keepdim = keepdim
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, out = ctx.saved_tensors
        if not ctx.keepdim:
            out = out.reshape(ctx.kept_shape)
            grad_out = grad_out.reshape(ctx.kept_shape)
        return grad_out * torch.exp(x - out), None, None


def install_logsumexp(device_type):
    """Scoped lazy-mode ``logsumexp`` on ``device_type`` (``numel <= LOGSUMEXP_MAX_NUMEL``). Returns an undo."""
    if not _lazy():
        return lambda: None
    op = aten.logsumexp.default

    def autograd_kernel(self, dim, keepdim=False):
        if torch.is_grad_enabled() and self.requires_grad:
            dims = _lse_takes(self, dim)
            if dims is not None:
                count("logsumexp")
                return _LseFn.apply(self, dims, keepdim)
        return op._op_dk(DispatchKey.Autograd, self, dim, keepdim)

    def backend_kernel(self, dim, keepdim=False):
        dims = _lse_takes(self, dim)
        if dims is not None:
            count("logsumexp")
            return lse_safe(self, dims, keepdim)
        return op._op_dk(DispatchKey.CompositeExplicitAutograd, self, dim, keepdim)

    return register_kernels(device_type, [(op, "Autograd{}", autograd_kernel), (op, "{}", backend_kernel)])


# ---- index_add (lazy mode, no_grad / inference_mode, dim == 0) ----------------------------------------------------

def _index_add_wins(shape, src_rows):
    """Where the rewrite beats Habana's lazy kernel in isolated and batched calls, inside the rows where Habana's
    accumulating put is right: many source rows per row of ``self``, or wide rows."""
    rows, width = shape[0], math.prod(shape[1:])
    return (0 < rows <= INDEX_ADD_MAX_SELF_ROWS and width > 0 and src_rows >= INDEX_ADD_MIN_ROWS
            and (src_rows >= INDEX_ADD_MIN_PER_ROW * rows or width >= INDEX_ADD_WIDE_ROW))


def _index_add_takes(self, dim, index, source):
    return (_lazy() and not torch.is_grad_enabled()
            and self.dim() >= 1 and source.dim() == self.dim() and index.dim() == 1
            and index.shape[0] == source.shape[0]
            and -self.dim() <= dim < self.dim() and (dim % self.dim()) == 0
            and self.shape[1:] == source.shape[1:] and self.dtype == source.dtype == torch.float32
            and index.dtype in (torch.int32, torch.int64)
            and static(*self.shape, *index.shape, *source.shape)
            and _index_add_wins(self.shape, source.shape[0])
            and _concrete(self) and not _is_fake(index) and not _is_fake(source))


def install_index_add(device_type):
    """Scoped lazy-mode ``index_add`` on ``device_type`` (``not grad_enabled``, ``dim == 0``, float32, the shapes of
    ``_index_add_wins``) via ``index_put_(..., accumulate=True)`` into a copy of ``self`` with a scratch row.
    Returns an undo."""
    if not _lazy():
        return lambda: None
    op, out_op = aten.index_add.default, aten.index_add.out

    def _run(self, index, source, alpha):
        count("index_add")
        src = source if alpha == 1 else source * alpha
        rows = self.shape[0]
        # As Habana's index_add: [-rows, -1] wraps, any other index out of range is dropped (into the scratch row):
        # Habana's index_put stops the device on an index of -1, and ATen's error would need a host read.
        slot = torch.where((index >= -rows) & (index < rows), torch.remainder(index, rows), rows)
        padded = torch.cat([self, self.new_zeros((1,) + self.shape[1:])])  # a fresh copy: accumulate in place
        return padded.index_put_((slot,), src, accumulate=True)[:rows]

    def autograd_kernel(self, dim, index, source, *, alpha=1):
        if _index_add_takes(self, dim, index, source):
            return _run(self, index, source, alpha)
        return op._op_dk(DispatchKey.Autograd, self, dim, index, source, alpha=alpha)

    def backend_kernel(self, dim, index, source, *, alpha=1):
        if _index_add_takes(self, dim, index, source):
            return _run(self, index, source, alpha)
        out = torch.empty_like(self)
        with torch._C._AutoDispatchBelowADInplaceOrView():
            return out_op(self, dim, index, source, alpha=alpha, out=out)

    return chain([register_kernels(device_type, [(op, "Autograd{}", autograd_kernel)]),
                  override_kernels(device_type, [(op, backend_kernel)])])
