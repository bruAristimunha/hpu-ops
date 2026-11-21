"""1-D upsampling on HPU.

- Linear, under ``torch.compile``: the native op instead of torch's gather/scatter decomposition, whose scatter
  backward is slow on Gaudi.
- Linear, eager and compiled: from explicit source indices and weights unless the factor is a power of two without
  ``align_corners``, differentiated by autograd. Gaudi's forward kernel drifts with the length elsewhere (NeuroPose's
  last resize, 9920 to 10000 samples: 34% off), and its backward is less exact than the CPU's at some sizes (1000 to
  1009 samples: 1.4e-04 against float64, the CPU's float32 kernel 4.3e-05); the index ops are as exact as the CPU's.
- Nearest and nearest-exact by an integer factor, eager and compiled: each sample repeated (``repeat_interleave``
  along the last dim). Gaudi's nearest kernel fails after a BatchNorm1d (graph compiler, ``layout.h:75 permute``), and
  under compile torch's decomposition leaves index ops that Habana runs eagerly. Other factors keep the native op,
  under compile too.
"""
import functools
import struct
import warnings

import torch
from torch._C import DispatchKey
from torch._decomp.decompositions import upsample_compute_output_size

from ._dispatch import below_autograd, chain, count, register_kernels, replace_py_kernels, static

aten = torch.ops.aten
CIA = DispatchKey.CompositeImplicitAutograd
CIA_AND_AUTOGRAD = (CIA, DispatchKey.Autograd)  # where torch registers its upsampling decompositions
MAX_OUTPUT = 2**23  # the native kernels' float32 index arithmetic is exact below this length
SYMBOLIC_HINT = ("Linear upsampling with symbolic shapes under torch.compile keeps torch's gather/scatter "
                 "decomposition, whose backward is slow on Gaudi (CodeBrain: 20x). It happens after a new batch size, "
                 "e.g. a smaller last batch. Compile with dynamic=False, or keep one batch size (drop_last=True).")


@functools.cache
def _warn_symbolic():
    """Once per process: each trace phase calls the kernel again, and dynamo resets the warnings registry."""
    warnings.warn(SYMBOLIC_HINT, stacklevel=2)


def _native_linear_vec(device_type, decomposition):
    """``upsample_linear1d.vec`` for tracing: the native op on ``device_type`` with static shapes, else
    ``decomposition``. The native backward has no meta kernel for symbolic sizes, so these warn."""

    def kernel(input, output_size, align_corners, scale_factors):
        if input.device.type != device_type:
            return decomposition(input, output_size, align_corners, scale_factors)
        if not static(*input.shape, *(output_size or [])):
            _warn_symbolic()
            return decomposition(input, output_size, align_corners, scale_factors)
        count("linear")
        size = upsample_compute_output_size(input.size(), output_size, scale_factors)
        scale = scale_factors[0] if scale_factors else None
        if device_type in _INDEXED and below_autograd() and _linear_index_takes(input, size, align_corners, scale):
            # traced below autograd (compiled inference_mode()): neither of install_linear_forward's kernels sees the
            # .default node emitted here, which would keep Gaudi's drifting kernel; the index path goes in the graph
            count("linear_index")
            return linear_by_index(input, size, align_corners, scale)
        return aten.upsample_linear1d.default(input, size, align_corners, scale)

    return kernel


def install_linear(device_type):
    """Swap torch's decomposition of ``upsample_linear1d.vec`` for the native-op kernel. Returns an undo."""
    return replace_py_kernels(aten.upsample_linear1d.vec, CIA_AND_AUTOGRAD,
                              lambda original: _native_linear_vec(device_type, original))


def native_linear_exact(input, output_size, align_corners, scales=None):
    """Whether Gaudi's linear kernel is exact: a power-of-two factor up or down, without ``align_corners``. Its other
    results drift with the length (1000 samples to 1500: 1.6e-03; 9920 to 10000: 0.1)."""
    length, out = input.shape[-1], output_size[0]
    if align_corners or not length or not out:
        return False
    big, small = max(length, out), min(length, out)
    factor = big // small
    return big % small == 0 and factor & (factor - 1) == 0 and scales in (None, out / length)


def linear_by_index(input, output_size, align_corners, scales=None):
    """``upsample_linear1d`` from explicit source indices and weights, computed as ATen's CPU kernel computes them
    (``area_pixel_compute_scale``, ``area_pixel_compute_source_index``, ``guard_index_and_lambda``): in float32 for
    16-bit and float32 inputs, in float64 for float64 ones."""
    length, out = input.shape[-1], output_size[0]
    dtype = torch.float64 if input.dtype == torch.float64 else torch.float32
    rounded = (lambda value: value) if dtype == torch.float64 else _float32  # the scale in the kernel's precision
    with torch.autocast(input.device.type, enabled=False):
        x = input.to(dtype)
        if out == length:  # ATen copies whatever scales says, with weights 1 and 0 (an inf gives NaN)
            return (x * 1 + x * 0).to(input.dtype)
        if align_corners:
            scale = rounded((length - 1) / (out - 1)) if out > 1 else 0.0
        elif scales is not None and scales > 0:
            scale = rounded(1.0 / scales)
        else:
            scale = rounded(length / out)
        dst = torch.arange(out, device=input.device, dtype=dtype)
        src = dst * scale if align_corners else ((dst + 0.5) * scale - 0.5).clamp_min(0)
        first = src.floor().long().clamp_max(length - 1)  # in integers: float32(length - 1) can round up
        weight = (src - first.to(dtype)).clamp(0, 1)
        second = first + (first < length - 1).long()
        return (x.index_select(-1, first) * (1 - weight) + x.index_select(-1, second) * weight).to(input.dtype)


def _linear_index_takes(input, output_size, align_corners, scales=None):
    """Valid inputs whose factor Gaudi's kernel gets wrong; everything else reaches the native op (which raises on
    invalid ones)."""
    return (input.dim() == 3 and input.is_floating_point() and len(output_size) == 1
            and static(*input.shape, *output_size) and output_size[0] > 0 and input.shape[1] > 0
            and input.shape[2] > 0 and not native_linear_exact(input, output_size, align_corners, scales))


def _linear_device_kernels(op):
    """The kernels of ``upsample_linear1d.default`` at the device's autograd and backend keys."""

    def autograd(input, output_size, align_corners, scales=None):  # autograd differentiates the index ops
        if _linear_index_takes(input, output_size, align_corners, scales):
            count("linear_index")
            return linear_by_index(input, output_size, align_corners, scales)
        return op._op_dk(DispatchKey.Autograd, input, output_size, align_corners, scales)

    def backend(input, output_size, align_corners, scales=None):  # inference mode, and anything else below autograd
        if _linear_index_takes(input, output_size, align_corners, scales):
            count("linear_index")
            return linear_by_index(input, output_size, align_corners, scales)
        return op._op_dk(DispatchKey.CompositeExplicitAutogradNonFunctional, input, output_size, align_corners, scales)

    return autograd, backend


_INDEXED = set()  # device types with install_linear_forward in: the traced .vec kernel takes the index path too


def install_linear_forward(device_type):
    """Linear 1-D upsampling on ``device_type`` from explicit indices where Gaudi's kernel is inexact, eager and traced
    (compiled ``inference_mode()`` graphs through ``install_linear``'s ``.vec`` kernel). Returns an undo."""
    op = aten.upsample_linear1d.default
    autograd, backend = _linear_device_kernels(op)
    undo = register_kernels(device_type, [(op, "Autograd{}", autograd), (op, "{}", backend)])
    _INDEXED.add(device_type)

    def undo_flag():
        _INDEXED.discard(device_type)

    return chain([undo, undo_flag])


def _float32(value):
    return struct.unpack("f", struct.pack("f", value))[0]


def repeat_factor(input, output_size, scales, exact):
    """The integer ``s`` for which this nearest upsampling of the last dim repeats each sample ``s`` times, else None.

    The native kernels read sample ``floor(i * c)`` (nearest-exact: ``floor((i + 0.5) * c)``) with ``c`` the float32
    of ``1 / s``. For nearest that is ``i // s`` only when ``float32(1 / s) * s >= 1``: ``s = 41``, for one, picks
    the previous sample at some multiples of 41. Nearest-exact always gives ``i // s``.
    """
    if input.dim() != 3:
        return None
    length, out = input.shape[-1], output_size[0]
    if not static(length, out) or not 0 < length <= out <= MAX_OUTPUT or out % length:
        return None
    factor = out // length
    if scales is not None and scales > 0 and scales != factor:  # the kernels read samples at 1 / scales
        return None
    if not exact and _float32(1.0 / factor) * factor < 1.0:
        return None
    return factor


def repeat_last(input, factor):
    """``input.repeat_interleave(factor, -1)``, built as a new tensor rather than a view of an internal one."""
    *batch, length = input.shape
    return input.unsqueeze(-1).expand(*batch, length, factor).reshape(*batch, length * factor)


def _nearest_device_kernels(op, exact, decomposition):
    """The kernels of ``op`` (a nearest ``.default``) at the device's autograd and backend keys."""

    def autograd(input, output_size, scales=None):
        factor = repeat_factor(input, output_size, scales, exact)
        if factor is not None:
            count("nearest")
            return repeat_last(input, factor)
        if static(*input.shape, *output_size):
            return op._op_dk(DispatchKey.Autograd, input, output_size, scales)  # the native op and its backward
        return decomposition(input, output_size, scales)  # symbolic sizes: torch's decomposition, as before

    def backend(input, output_size, scales=None):  # inference mode, and anything else below autograd
        factor = repeat_factor(input, output_size, scales, exact)
        if factor is not None:
            count("nearest")
            return repeat_last(input, factor)
        return op._op_dk(DispatchKey.CompositeExplicitAutogradNonFunctional, input, output_size, scales)

    return autograd, backend


def _nearest_decompose(device_type, op, exact, original):
    """``op``'s CompositeImplicitAutograd kernel, which the tracers call as ``decompose()`` below autograd. On
    ``device_type`` with static shapes it keeps the native op (``NotImplemented``) or repeats; else ``original``."""

    def kernel(input, output_size, scales=None):
        if input.device.type == device_type:
            factor = repeat_factor(input, output_size, scales, exact)
            if factor is not None:
                count("nearest")
                return repeat_last(input, factor)
            if static(*input.shape, *output_size):
                if below_autograd():
                    return NotImplemented
                return op._op_dk(DispatchKey.Autograd, input, output_size, scales)
        return original(input, output_size, scales)

    return kernel


def _nearest_vec(device_type, default, original):
    """``.vec`` on ``device_type`` calls ``.default``, as its C++ kernel does in eager mode; torch's ``.vec``
    decomposition would go straight to index ops."""

    def kernel(input, output_size, scale_factors):
        if input.device.type != device_type:
            return original(input, output_size, scale_factors)
        size = upsample_compute_output_size(input.size(), output_size, scale_factors)
        return default(input, size, scale_factors[0] if scale_factors else None)

    return kernel


def install_nearest(device_type):
    """Nearest and nearest-exact 1-D upsampling on ``device_type``, eager and traced. Returns an undo."""
    undos = []
    for name, exact in (("upsample_nearest1d", False), ("_upsample_nearest_exact1d", True)):
        packet = getattr(aten, name)
        default, vec = packet.default, packet.vec
        decomposition = default.py_kernels.get(DispatchKey.Autograd) or functools.partial(
            default._op_dk, DispatchKey.Autograd)
        autograd, backend = _nearest_device_kernels(default, exact, decomposition)
        undos.append(register_kernels(device_type, [(default, "Autograd{}", autograd), (default, "{}", backend)]))
        undos.append(replace_py_kernels(default, (CIA,), functools.partial(_nearest_decompose, device_type,
                                                                           default, exact)))
        undos.append(replace_py_kernels(vec, CIA_AND_AUTOGRAD, functools.partial(_nearest_vec, device_type, default)))
    return chain(undos)
