"""``bincount`` of large 1-D integer tensors through ``histc`` (hpu-ops #21).

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
import torch
from torch._C import DispatchKey

from ._dispatch import count, register_kernels, static
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


def bincount_histc(x, bins):
    """Counts of the values ``0 .. bins - 1`` of ``x`` (every value known to be in that range), int64."""
    xf = x.float()
    if bins <= BINCOUNT_CHUNK:
        return torch.histc(xf, bins, 0, bins - 1).long()
    starts = list(range(0, bins - BINCOUNT_CHUNK, BINCOUNT_CHUNK)) + [bins - BINCOUNT_CHUNK]
    parts = [torch.histc(xf, BINCOUNT_CHUNK, a, a + BINCOUNT_CHUNK - 1) for a in starts]
    done = starts[-2] + BINCOUNT_CHUNK  # the last chunk adds the bins from here on
    parts[-1] = parts[-1][done - starts[-1]:]
    return torch.cat(parts).long()


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
