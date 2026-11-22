"""Convolutions on HPU: ``padding="valid"`` under ``torch.compile``, the weight gradient of one-channel
convolutions with a stride, and spatial splitting of the large-kernel convolutions that fail natively.

Weight gradient: for a convolution with one input channel and a stride above 1, Gaudi's compiled
``convolution_backward`` returns about half the true weight gradient plus an unrelated part (the first convolutions of
EEGDINO, CodeBrain and LaBraM: scale 0.48-0.52, cosine 0.50-0.62 on random data). The forward, the bias and input
gradients, and eager mode at these shapes are right. On HPU that weight gradient is one contraction of the output
gradient with the input's strided patches instead, eager and compiled.

``padding="valid"``:

``conv{1,2,3}d.padding`` is a C++ composite. For ``"valid"`` it calls ``aten.convolution`` with the one-element
padding list ``[0]`` (and output padding ``[0]``), which ATen broadcasts but Habana's compiled convolution reads as a
full list: a 2-D convolution gets a garbage second padding and the graph fails to compile (ATCNet, EEGInceptionERP,
EEGITNet). On HPU, ``"valid"`` goes through ``conv{n}d.default`` with ``n`` zeros instead, which passes full lists.
Eager calls do not trace through this kernel and keep working as before.

Large kernels (``> 256``; ``> 128`` in lazy mode):

Gaudi2's MME AGU rejects convolutions whose spatial kernel dimension exceeds 256
(``gaudi2_agu_config.cpp::338 setAssociatedDimAndSize: size - 1 <= uint8 max`` in ``STaRNet(64)`` ``conv_spatial_fusion``
``(351, 1)`` and ``TensorCSPNet(22, dims=(22, 11, 11, 11))`` ``temporal_block`` ``(4, 1089)``). In lazy mode the bound is
lower (measured on bridge 1.21.4, forward and backward, native op): conv1d (4, 8, 1000) x (16, 8, 256) fails graph
compile the same way, and for 129 <= k <= 255 the input gradient is wrong (relative error 1.3 to 1.5) when
256 < L < 2k (k = 144 over 260, 240 over 300, 255 over 257). Eager, kernels up to 256 were right at every length tried
(those and k = 255, 256 over 1000). So a kernel dimension above ``MAX_KERNEL = 256`` (eager ops, compiled graphs) or
``LAZY_MAX_KERNEL = 128`` (lazy mode) is split into slices of at most that size, with exact input slicing and
``F.pad``; a 128-wide chunk whose input is longer than 256 is then longer than twice its kernel. In lazy mode a 2-D
kernel's shorter last chunk is padded with zero taps to the full chunk (native lazy (44, 44) over (54, 54) stops the
card). One class of wider kernels runs natively and is left to the native op, which is faster (see ``NATIVE_WIDE``
below and the README's sweep table): one input channel, wide only in the last dim, an even width at stride >= 32 in
eager mode (>= 8 for widths 512 to 2000), any width >= 257 at stride >= 8 in lazy mode, no wider than measured.
TensorCSPNet's default ``(4, 4356)`` stride ``(1, 484)`` is in it. 16-bit chunks are convolved and summed in float32
and the bias is added once, at the end, before the one rounding, as ATen's CPU convolution does. Compiled
``inference_mode()`` graphs split too (through ``conv{n}d``'s Python composite kernel), ``padding="same"`` and unbatched
inputs included (``"same"`` becomes ATen's int padding before it reaches the split).

Not fixed, compiled (``torch.compile(backend="hpu_backend")``): the graph compiler rejects some split graphs whatever the
chunk width. With 256-wide chunks (used): conv1d k = 351 over 1000 fails in a bundling pass (``Cannot add a node DEDW to
bundle 4 if it's already belong to bundle 3``; 176+175, 128+128+95 and 3x117 fail too, only 256+256 compiled:
``repro/probe_compile_chunks.py``), and a (4, 1089) kernel over width 1500 fails ``setAssociatedDimAndSize``. With
128-wide chunks, k = 300 over 420, 600 over 1000 and 1089 over 1500 fail instead, which pass at 256. The native op fails
for all of them. The two models' compiled steps train either way.
Autograd differentiates the slice-sum naturally.
"""
import functools
import itertools
import os

import torch
import torch.nn.functional as F
from torch._C import DispatchKey
from torch._subclasses.fake_tensor import is_fake

from ._dispatch import REDUCED, chain, count, register_kernels, replace_py_kernels, static

aten = torch.ops.aten
MAX_KERNEL = 256  # eager ops and compiled graphs
LAZY_MAX_KERNEL = 128  # lazy mode: larger chunks fail or give wrong input gradients (see the module docstring)


DEFAULTS = (None, 1, "valid", 1, 1)  # bias, stride, padding, dilation, groups; the dispatcher omits trailing defaults


def _as_list(value, n):
    return [value] * n if isinstance(value, int) else list(value)


def _same_as_ints(conv, n, input, weight, bias, stride, dilation, groups):
    """``padding="same"`` as ATen's ``_convolution_mode`` computes it: ``d * (k - 1)`` zeros per dim, half on each
    side, the odd one on the right through ``F.pad`` of the input; then ``conv{n}d.default`` with int padding, which
    the large-kernel split reads (traced under ``inference_mode()`` ATen's composite called ``aten.convolution``
    directly, so a 351-wide ``"same"`` kernel stayed whole)."""
    dil = _as_list(dilation, n)
    total = [d * (k - 1) for d, k in zip(dil, weight.shape[2:])]
    left = [t // 2 for t in total]
    extra = [t - 2 * lo for t, lo in zip(total, left)]
    if any(extra):
        input = F.pad(input, [v for e in reversed(extra) for v in (0, e)])
    return conv(input, weight, bias, _as_list(stride, n), left, dil, groups)


def _valid_as_zeros(device_type, n, original):
    """``conv{n}d.padding`` for tracing on ``device_type``: ``"valid"`` as ``n`` zeros, ``"same"`` (stride 1, static
    kernel sizes) as ATen's int padding (``_same_as_ints``); anything else ``original``, which raises ATen's errors."""
    conv = getattr(aten, f"conv{n}d").default

    def kernel(input, weight, *rest):
        bias, stride, padding, dilation, groups = (*rest, *DEFAULTS[len(rest):])
        if input.device.type != device_type:
            return original(input, weight, *rest)
        if padding == "valid":
            count("valid_conv")
            return conv(input, weight, bias, _as_list(stride, n), [0] * n, _as_list(dilation, n), groups)
        if (padding == "same" and all(s == 1 for s in _as_list(stride, n)) and weight.dim() == n + 2
                and static(*weight.shape[2:], *_as_list(dilation, n))):
            count("same_conv")
            return _same_as_ints(conv, n, input, weight, bias, stride, dilation, groups)
        return original(input, weight, *rest)

    return kernel


def install(device_type):
    """Replace the CompositeImplicitAutograd kernel of ``conv{1,2,3}d.padding``. Returns an undo."""
    return chain([replace_py_kernels(getattr(aten, f"conv{n}d").padding, (DispatchKey.CompositeImplicitAutograd,),
                                     functools.partial(_valid_as_zeros, device_type, n)) for n in (1, 2, 3)])


def one_channel_weight_grad(grad_output, input, weight, stride, padding):
    """d loss / d weight of a convolution with one input channel (groups 1, dilation 1): the output gradient contracted
    with the padded input's strided patches (``Tensor.unfold``), in float32 for 16-bit inputs."""
    n = input.dim() - 2
    dtype = torch.float32 if grad_output.dtype in (torch.float16, torch.bfloat16) else grad_output.dtype
    with torch.autocast(input.device.type, enabled=False):
        patches = F.pad(input.to(dtype), [p for p in reversed(padding) for _ in range(2)])[:, 0]
        for d in range(n):
            patches = patches.unfold(1 + d, weight.shape[2 + d], stride[d])
        space, kernel = "xyz"[:n], "ijk"[:n]
        grad = torch.einsum(f"bo{space},b{space}{kernel}->o{kernel}", grad_output.to(dtype), patches)
    return grad.unsqueeze(1).to(weight.dtype)


def _expand(values, n):
    """A one-element stride or padding list stands for every spatial dim, as in ATen (``F.conv2d(x, w, None, (2,))``)."""
    return list(values) * n if len(values) == 1 else list(values)


def _one_channel_strided(input, weight, stride, dilation, transposed, groups, output_mask):
    return (output_mask[1] and not transposed and groups == 1 and weight.shape[1] == 1 and 3 <= input.dim() <= 5
            and any(s > 1 for s in stride) and all(d == 1 for d in dilation))


def _weight_grad_kernel(op):
    """``convolution_backward`` at the device's autograd key: the weight gradient of one-channel strided convolutions
    from ``one_channel_weight_grad``, every other output (and every other convolution) from the op itself."""

    def kernel(grad_output, input, weight, bias_sizes, stride, padding, dilation, transposed, output_padding, groups,
               output_mask):
        args = (bias_sizes, stride, padding, dilation, transposed, output_padding, groups)
        if not _one_channel_strided(input, weight, stride, dilation, transposed, groups, output_mask):
            return op._op_dk(DispatchKey.Autograd, grad_output, input, weight, *args, output_mask)
        count("conv_weight_grad")
        grad_input = grad_bias = None
        if output_mask[0] or output_mask[2]:  # a first layer without bias needs nothing else from the op
            grad_input, _, grad_bias = op._op_dk(DispatchKey.Autograd, grad_output, input, weight, *args,
                                                 [output_mask[0], False, output_mask[2]])
        n = input.dim() - 2
        return grad_input, one_channel_weight_grad(grad_output, input, weight, _expand(stride, n),
                                                   _expand(padding, n)), grad_bias

    return kernel


def install_weight_grad(device_type):
    """The weight gradient of one-channel strided convolutions on ``device_type``. Returns an undo."""
    op = aten.convolution_backward.default
    return register_kernels(device_type, [(op, "Autograd{}", _weight_grad_kernel(op))])


def _lazy():
    """Whether Habana's bridge runs in lazy mode (read once; it is fixed when the bridge loads). Without the bridge
    there is no lazy mode (the CPU tests patch this)."""
    if "lazy" not in _MODE:
        try:
            from habana_frameworks.torch.utils.internal import is_lazy
            _MODE["lazy"] = bool(is_lazy())
        except ImportError:
            _MODE["lazy"] = False
    return _MODE["lazy"]


_MODE = {}


def _mode(input, device_type="hpu"):
    """How this convolution runs: ``"compile"`` (a traced ``torch.compile`` graph), ``"lazy"`` (a ``device_type`` op
    in lazy mode) or ``"eager"`` (a ``device_type`` op in eager mode)."""
    if is_fake(input) or torch._guards.detect_fake_mode() is not None:
        return "compile"
    return "lazy" if input.device.type == device_type and _lazy() else "eager"


def _max_kernel(mode):
    """The largest kernel a single Synapse ``Conv`` node gets: ``LAZY_MAX_KERNEL`` (128) in lazy mode, else
    ``MAX_KERNEL`` (256: eager ops and traced ``torch.compile`` graphs, where 128-wide chunks failed more shapes than
    256-wide ones; see the module docstring). The CPU tests patch ``MAX_KERNEL``."""
    return min(MAX_KERNEL, LAZY_MAX_KERNEL) if mode == "lazy" else MAX_KERNEL


# The one class of wide kernels Gaudi2's native convolution runs, forward and backward, with the right numbers (bridge
# 1.21.4, measured by ``repro/sweep_conv_native.sh``, table in the README): a 2-D float32 convolution with one input
# channel and groups 1, no padding or dilation, whose kernel is wider than the bound only in its last dim, ``kH <= 22``,
# ``kW`` no wider than measured, and
#   eager: an even ``kW`` up to 6002 and a stride ``>= 32`` along it (every even ``kW`` from 258 to 6002 ran at strides
#          32 to kW; every odd one tried, 257, 351, 1089, 1091 and 1501, fails ``setAssociatedDimAndSize``; (4, 4356)
#          fails at strides 1 to 16), or an even ``kW`` from 512 to 2000 with ``kH <= 4`` at a stride ``>= 8`` (512,
#          600, 700, 800, 1024, 1090, 1200, 1500 and 2000 ran at 8, 1090 and 1500 at 16; 4356 fails at 8 and 16);
#   lazy:  any ``kW`` up to 6000 and a stride ``>= 8`` (every ``kW`` from 257 to 6000 ran at strides 8 to kW; at stride 1 with
#          the input as wide as the kernel the card stops with a device critical error).
# TensorCSPNet's default ``temporal_block`` ((4, 4356), stride (1, 484)) is in it in both modes, and its
# ``dims=(22, 11, 11, 11)`` one ((4, 1089), stride (1, 121)) in lazy mode, so they keep the native op. Everything
# else above the bound is split: 4 input channels, the wide dim first ((4356, 4) or STaRNet's (351, 1)), conv1d and
# 3-D convolutions, other dtypes, and traced graphs (not measured) included.
NATIVE_WIDE = {"eager": ({"min_k": 258, "max_k": 6002, "stride": 32, "even": True, "max_kh": 22},
                         {"min_k": 512, "max_k": 2000, "stride": 8, "even": True, "max_kh": 4}),
               "lazy": ({"min_k": 257, "max_k": 6000, "stride": 8, "even": False, "max_kh": 22},)}


def _native_wide(mode, input, weight, groups, strd, pad, dil):
    """Whether this convolution is in the measured native class above."""
    if mode not in NATIVE_WIDE or input.dim() != 4 or input.dtype != torch.float32:
        return False
    if not (groups == 1 and weight.shape[1] == 1 and all(p == 0 for p in pad) and all(d == 1 for d in dil)):
        return False
    k_h, k_w = weight.shape[2:]
    return any(k_h <= min(rule["max_kh"], _max_kernel(mode)) and rule["min_k"] <= k_w <= rule["max_k"]
               and (k_w % 2 == 0 or not rule["even"]) and strd[1] >= rule["stride"] for rule in NATIVE_WIDE[mode])


def _large_kernel_takes(mode, input, weight, bias, stride, padding, dilation, transposed, output_padding, groups):
    if transposed or not (3 <= input.dim() <= 5) or input.dim() != weight.dim():
        return False
    if not input.is_floating_point() or input.dtype != weight.dtype or input.device != weight.device:
        return False
    if bias is not None and (bias.dim() != 1 or bias.shape[0] != weight.shape[0]
                             or bias.dtype != input.dtype or bias.device != input.device):
        return False
    n = input.dim() - 2
    if (len(stride) not in (1, n) or len(padding) not in (1, n)
            or len(dilation) not in (1, n) or len(output_padding) not in (1, n)):
        return False
    if not static(groups, *input.shape[2:], *weight.shape[2:], *stride, *padding, *dilation, *output_padding):
        return False
    if (groups <= 0 or weight.shape[0] <= 0 or weight.shape[1] <= 0
            or input.shape[1] != weight.shape[1] * groups or weight.shape[0] % groups != 0):
        return False
    if not all(length > 0 for length in input.shape[2:]) or not all(k > 0 for k in weight.shape[2:]):
        return False
    strd, pad, dil = _expand(stride, n), _expand(padding, n), _expand(dilation, n)
    if (any(s <= 0 for s in strd) or any(p < 0 for p in pad) or any(d <= 0 for d in dil)
            or any(p != 0 for p in output_padding)):
        return False
    if not any(k > _max_kernel(mode) for k in weight.shape[2:]):
        return False
    if _native_wide(mode, input, weight, groups, strd, pad, dil):
        count("large_conv_native")
        return False
    return all(l_in + 2 * p >= d * (k - 1) + 1
               for l_in, k, p, d in zip(input.shape[2:], weight.shape[2:], pad, dil))


def _pads_tail(mode, weight):
    """Whether the last, shorter chunk of a split dim is padded with zeros to the full chunk: in lazy mode for kernels
    with more than one dim above 1. Measured on Gaudi2, lazy, bridge 1.21.4, native forward and backward: conv2d
    (54, 54) x (44, 44), (54, 138) x (44, 128) and (64, 64) x (54, 54) stop the card (device critical error), which the
    split of a (300, 300) kernel over 310 x 310 emitted, while (138, 138) x (128, 128), (138, 54) x (128, 44) and
    (110, 110) x (100, 100) run. Padded, every chunk has the full chunk's shape. One-dim kernels (STaRNet's (351, 1):
    128 + 128 + 95) keep their measured tail."""
    return mode == "lazy" and sum(k > 1 for k in weight.shape[2:]) > 1


def _plan(mode, input, weight, out_sizes, strd, pad, dil):
    """Per spatial dim, the chunks ``(k0, k1, sc, ec, pl, pr, wpad, cpad)``: kernel slice ``k0:k1``, input slice
    ``sc:ec``, zeros ``pl`` before and ``pr`` after it, ``wpad`` zero taps after the kernel slice, the convolution's
    own padding ``cpad`` (a dim that is not split keeps it). A chunk whose window lies
    wholly in the padding is kept (an empty slice padded with zeros), so ``0 * inf`` gives NaN as natively and the
    graph keeps every weight tap."""
    max_k = _max_kernel(mode)
    pad_tail = _pads_tail(mode, weight)
    plans = []
    for l_in, k, l_out, s, p, d in zip(input.shape[2:], weight.shape[2:], out_sizes, strd, pad, dil):
        if k <= max_k:
            plans.append([(0, k, 0, l_in, 0, 0, 0, p)])
            continue
        plan = []
        for k0 in range(0, k, max_k):
            c = min(max_k, k - k0)
            wpad = max_k - c if pad_tail else 0
            x_start = k0 * d - p
            x_end = (l_out - 1) * s + k0 * d + (c - 1) * d - p + 1
            sc, ec = max(0, x_start), min(l_in, x_end)
            if sc >= ec:  # wholly in the padding
                sc = ec = 0
                pl, pr = x_end - x_start, 0
            else:
                pl, pr = sc - x_start, x_end - ec
            plan.append((k0, k0 + c, sc, ec, pl, pr + wpad * d, wpad, 0))
        plans.append(plan)
    return plans


def _native_format(input, weight):
    """The memory format ATen's convolution gives its output: channels-last if the input or the weight is."""
    from torch._prims_common import suggest_memory_format
    cl = {4: torch.channels_last, 5: torch.channels_last_3d}.get(input.dim())
    if cl is not None and cl in (suggest_memory_format(input), suggest_memory_format(weight)):
        return cl
    return torch.contiguous_format


# Where a device's native 16-bit convolution adds the bias: after rounding the sum to the input dtype (True) or before.
# Gaudi2, bridge 1.21.4, eager and lazy (repro/probe_review31.py conv_bf16_bias): bf16 conv1d of ones (1, 1, 2) by
# (256, 0.75) gives 256, with bias 0.5 still 256 (CPU: 258); fp16 256.75 and 257.25 on both.
BIAS_AFTER_ROUNDING = {"hpu": True}


def _split_conv(conv, mode, input, weight, bias, stride, padding, dilation, groups):
    """Evaluate a non-transposed convolution by partitioning spatial kernel dims above the bound into chunks of at most
    the bound with exact input slices and ``F.pad``, ``conv(x, w, stride, padding, dilation, groups)`` per chunk.
    The chunk outputs are summed in float32 for 16-bit inputs (each chunk convolved in float32 from the exact 16-bit
    values, so the products are exact) and the bias is added once, after the sum, where the device's native op adds
    it: the CPU's before the one rounding to the input dtype (bf16 conv1d of ones by (256, 0.75) with bias 0.5: 258),
    Gaudi's after it, in the input dtype (``BIAS_AFTER_ROUNDING``; measured, bridge 1.21.4, eager and lazy: 256). The sum
    runs in another order than native's: where the float32 reduction overflows midway (partial sums past 3.4e38), a
    split output can be inf or NaN where native is finite or inf, and the reverse."""
    n = input.dim() - 2
    strd, pad, dil = _expand(stride, n), _expand(padding, n), _expand(dilation, n)
    out_sizes = [(l_in + 2 * p - d * (k - 1) - 1) // s + 1
                 for l_in, k, s, p, d in zip(input.shape[2:], weight.shape[2:], strd, pad, dil)]
    plans = _plan(mode, input, weight, out_sizes, strd, pad, dil)
    acc_dtype = torch.float32 if input.dtype in REDUCED else input.dtype
    with torch.autocast(input.device.type, enabled=False):
        x_full, w_full = input.to(acc_dtype), weight.to(acc_dtype)
        acc = None
        for combo in itertools.product(*plans):
            w_chunk = w_full[(slice(None), slice(None), *(slice(k0, k1) for k0, k1, *_ in combo))]
            x_chunk = x_full[(slice(None), slice(None), *(slice(sc, ec) for _, _, sc, ec, *_ in combo))]
            if any(c[6] for c in combo):
                w_chunk = F.pad(w_chunk, [v for c in reversed(combo) for v in (0, c[6])])
            pads = [(c[4], c[5]) for c in combo]
            if any(sc >= ec for _, _, sc, ec, *_ in combo):  # wholly in the padding: zeros, joined to the input
                size = [x_chunk.shape[0], x_chunk.shape[1]] + [pl + pr + ec - sc for (pl, pr), (_, _, sc, ec, *_)
                                                               in zip(pads, combo)]
                x_chunk = torch.cat([x_full[(slice(None), slice(None), *(slice(0, 0) for _ in combo))].reshape(
                    size[:2] + [0] + size[3:]), x_full.new_zeros(size)], dim=2)
            elif any(pl or pr for pl, pr in pads):
                x_chunk = F.pad(x_chunk, [v for pair in reversed(pads) for v in pair])
            y = conv(x_chunk, w_chunk, strd, [c[7] for c in combo], dil, groups)
            acc = y if acc is None else acc + y
        late_bias = bias is not None and input.dtype in REDUCED and BIAS_AFTER_ROUNDING.get(input.device.type, False)
        if bias is not None and not late_bias:
            acc = acc + bias.to(acc_dtype).view(1, -1, *([1] * n))
        acc = acc.to(input.dtype)
        if late_bias:
            acc = acc + bias.view(1, -1, *([1] * n))
        fmt = _native_format(input, weight)
        if not acc.is_contiguous(memory_format=fmt):
            acc = acc.contiguous(memory_format=fmt)
        return acc


def _chunk_conv(op, fallback):
    def conv(x, w, stride, padding, dilation, groups):
        return op._op_dk(fallback, x, w, None, stride, padding, dilation, False, [0] * len(stride), groups)

    return conv


def _traced_conv(x, w, stride, padding, dilation, groups):
    return aten.convolution.default(x, w, None, stride, padding, dilation, False, [0] * len(stride), groups)


def _large_kernel_wrapper(op, fallback, device_type):
    """``aten.convolution.default`` kernel: split kernels with spatial size above the bound; fall back otherwise."""
    conv = _chunk_conv(op, fallback)

    def kernel(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups):
        mode = _mode(input, device_type)
        if not _large_kernel_takes(mode, input, weight, bias, stride, padding, dilation, transposed, output_padding,
                                   groups):
            return op._op_dk(fallback, input, weight, bias, stride, padding, dilation, transposed, output_padding, groups)
        count("large_conv")
        return _split_conv(conv, mode, input, weight, bias, stride, padding, dilation, groups)

    return kernel


CONV_DEFAULTS = (None, 1, 0, 1, 1)  # conv{n}d.default: bias, stride, padding, dilation, groups


def _traced_large_kernel(device_type, n, original):
    """``conv{n}d.default``'s ``CompositeImplicitAutograd`` Python kernel, which the tracers decompose to where
    autograd is excluded (a compiled ``inference_mode()`` graph: ``AutogradHPU`` is skipped and the proxy mode sits
    above ``HPU``, so neither kernel above sees the call): the same split for ``device_type`` tensors, batched or
    unbatched (``(C, L)``: split as ``(1, C, L)``, as ATen's ``batchify`` does before ``aten.convolution``)."""

    def kernel(input, weight, *rest):
        bias, stride, padding, dilation, groups = (*rest, *CONV_DEFAULTS[len(rest):])
        if (input.device.type == device_type and isinstance(padding, (int, list, tuple))
                and input.dim() in (n + 1, n + 2)):
            stride, padding, dilation = (_as_list(v, n) for v in (stride, padding, dilation))
            batched = input if input.dim() == n + 2 else input.unsqueeze(0)  # unbatched, as ATen's batchify
            mode = _mode(input, device_type)
            if _large_kernel_takes(mode, batched, weight, bias, stride, padding, dilation, False, [0] * n, groups):
                count("large_conv")
                out = _split_conv(_traced_conv, mode, batched, weight, bias, stride, padding, dilation, groups)
                return out if batched is input else out.squeeze(0)
        return original(input, weight, *rest)

    return kernel


def install_large_kernel(device_type):
    """Split convolutions with spatial kernel size above the bound on ``device_type``. Returns an undo."""
    op = aten.convolution.default
    undo = register_kernels(device_type, [
        (op, "Autograd{}", _large_kernel_wrapper(op, DispatchKey.Autograd, device_type)),
        (op, "{}", _large_kernel_wrapper(op, DispatchKey.CompositeExplicitAutograd, device_type)),
    ])
    traced = [replace_py_kernels(getattr(aten, f"conv{n}d").default, (DispatchKey.CompositeImplicitAutograd,),
                                 functools.partial(_traced_large_kernel, device_type, n)) for n in (1, 2, 3)]
    return chain([undo] + traced)
