"""``bincount`` of large 1-D integer tensors through ``histc`` (hpu-ops #21), and ``histc`` itself where Gaudi's
kernel fails (see "histc" below).

Gaudi's ``bincount`` (bridge 1.21.4) grows with the input and steeply with the bins: 409600 elements take 7.4 ms
eager / 4.7 ms lazy with 64 bins, 230 / 225 ms with 1000 bins and 427 ms eager with 4096 bins, where one CPU core
takes 0.8 ms. Here, for 1-D integer inputs of 65536 to 2**24 elements without weights and with at most 4096 bins:
one ``aminmax`` read (the output length and the sign; Habana's kernel syncs the host too) and ``histc`` over
``[a, a + 255]`` with 256 bins per chunk of 256 bins (values outside a chunk's range are not counted). That counts
exactly: integers below 2**24 are exact in float32, the counts too (fewer than 2**24 elements), and on Gaudi2 every
bin count to 300 put ``arange(n)`` one per bin (``histc`` does not compile from 400 bins, and non-integer bounds abort
Habana's eager kernel: ``repro/histc_float_bounds.py``). The last chunk overlaps the one before, so no chunk has one
bin (``histc`` widens ``min == max`` by 1 on each side). Results are int64. A negative value raises ATen's error
(Habana's kernel counts it instead: ``repro/bincount_negative.py``).

Measured through the dispatcher (``tests/bench_small_dispatch.py``), isolated synced calls and a batch of calls:
65536 elements and up are faster in both modes (eager 1.04-43x, lazy 1.9-35x). Smaller inputs keep Habana's kernel:
below 65536 elements the host read costs as much as Habana's whole kernel with few bins (4096 elements, 64 bins: no
exact variant faster in both modes), and the bin count is only known after it.

Measured and not rewritten (see the README): ``logsumexp`` (faster forward in lazy mode only; forward + backward
0.80-1.21x; eager not faster), ``cumsum`` (a triangular-ones matmul is faster but turns ``[1, inf]`` into
``[nan, inf]``; exact variants 0.02-0.45x), ``index_add`` / ``scatter_add`` as one-hot matmuls (slower in at least
one mode, and ``inf * 0`` again).

Registered at ``AutogradHPU``; calls it does not take (weights, other dtypes and dims, ``minlength`` or values past
4096, traced graphs) and inference mode go to Habana's kernel.
"""
import math

import torch
from torch._C import DispatchKey

from ._dispatch import chain, count, override_kernels, register_kernels, static
from .linalg import _is_fake, _transformed

aten = torch.ops.aten
BINCOUNT_MIN_NUMEL = 1 << 16  # below, one host read plus histc is not faster than Habana's kernel (measured)
BINCOUNT_MAX_NUMEL = 1 << 24  # histc counts in float32: exact below 2**24
BINCOUNT_MAX_BINS = 4096  # 16 histc launches
BINCOUNT_CHUNK = 256  # histc compiles up to ~300 bins on Gaudi2
INTEGRAL = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)


def _concrete(x):
    """True for a plain eager call: the host read needs real values, outside any transform or tracer. Declined, so
    Habana's kernel runs: functorch transforms and forward-mode AD (``linalg._transformed``; the layer depth too,
    since below ``torch.func.grad``'s layer the interpreter stack reads empty), fake or functional tensors and
    ``torch.compile`` (``linalg._is_fake``), and any ``__torch_dispatch__`` mode such as ``make_fx``'s proxy mode
    (``is_compiling()`` is false inside AOT dispatch)."""
    if _transformed() or torch._C._functorch.get_dynamic_layer_stack_depth() > 0 or _is_fake(x):
        return False
    if torch._C._len_torch_dispatch_stack() > 0 or torch._guards.detect_fake_mode() is not None:
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


def _takes(x, weights=None, minlength=0):
    return (BINCOUNT_MIN_NUMEL <= x.numel() < BINCOUNT_MAX_NUMEL and weights is None and x.dim() == 1
            and x.dtype in INTEGRAL and isinstance(minlength, int) and 0 <= minlength <= BINCOUNT_MAX_BINS
            and static(*x.shape) and _concrete(x))


def install_bincount(device_type):
    """``bincount`` of large 1-D integer tensors with at most 4096 bins on ``device_type`` through ``histc``.
    Returns an undo."""
    op = aten.bincount.default

    def kernel(x, weights=None, minlength=0):
        if _takes(x, weights, minlength):
            lo, hi = (int(v) for v in torch.stack(torch.aminmax(x)).tolist())
            if lo < 0:  # ATen's error; Habana's kernel counts negative values instead
                raise RuntimeError("bincount only supports 1-d non-negative integral inputs.")
            bins = max(hi + 1, minlength)
            if bins <= BINCOUNT_MAX_BINS:
                count("bincount")
                return bincount_histc(x, bins)
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


def histc_edges(lo, hi, bins, dtype):
    """Host tensors ``(edges, low, high)`` for ATen's CPU histc over ``[lo, hi]`` (finite, ``lo < hi``) in ``dtype``:
    ``edges[k]`` (k = 1 .. bins - 1) is the smallest value of ``dtype`` whose position is ``k`` or more (``inf`` when
    none is), ``edges[0] = -inf``, ``edges[bins] = inf``; ``low`` / ``high`` the bounds as ATen compares them. For
    float64, everything rounded to the float32 values the card holds. Independent of the default dtype."""
    L, R = _to_dtype(lo, dtype), _to_dtype(hi, dtype)
    width = R - L
    if width == 0:  # distinct bounds that round to one value: ATen's position is trunc(0 / 0), measured bin 0
        edges = torch.full((bins - 1,), math.inf, dtype=torch.float32 if dtype == torch.float64 else dtype)
        return _framed(edges), float(L), float(R)
    scale = torch.tensor(bins, dtype=dtype)
    k = torch.arange(1, bins, dtype=torch.float64)
    top = int(_ordered(R.reshape(1))[0])
    below = torch.full((bins - 1,), int(_ordered(L.reshape(1))[0]), dtype=torch.int64)
    above = torch.full((bins - 1,), top + 1, dtype=torch.int64)  # top + 1: no value in [L, R] reaches k
    while bool((below < above).any()):
        mid = (below >> 1) + (above >> 1) + (below & above & 1)  # the floor of the mean, without int64 overflow
        reached = ((_unordered(mid.clamp(max=top), dtype) - L) * scale / width).trunc().to(torch.float64) >= k
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


def _edges_on(lo, hi, bins, dtype, device, traced):
    if traced:  # a fresh constant of the graph
        edges, low, high = _host_edges(lo, hi, bins, dtype)
        return torch.tensor(edges.tolist(), dtype=edges.dtype, device=device), low, high
    key = (lo, hi, bins, dtype, str(device))
    if key not in _EDGE_CACHE:
        edges, low, high = _host_edges(lo, hi, bins, dtype)
        _EDGE_CACHE[key] = (edges.to(device), low, high)
        while len(_EDGE_CACHE) > EDGE_CACHE_SIZE:
            _EDGE_CACHE.pop(next(iter(_EDGE_CACHE)))
    return _EDGE_CACHE[key]


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
        _EDGE_CACHE.clear()

    return undo
