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
``F.pad``; a 128-wide chunk whose input is longer than 256 is then longer than twice its kernel. One class of wider
kernels runs natively and is left to the native op, which is faster (see ``NATIVE_WIDE`` below and the README's
sweep table): one input channel, wide only in the last dim, an even width at stride >= 32 in eager mode, any width
>= 257 at stride >= 8 in lazy mode. TensorCSPNet's default ``(4, 4356)`` stride ``(1, 484)`` is in it.

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

from ._dispatch import chain, count, register_kernels, replace_py_kernels, static

aten = torch.ops.aten
MAX_KERNEL = 256  # eager ops and compiled graphs
LAZY_MAX_KERNEL = 128  # lazy mode: larger chunks fail or give wrong input gradients (see the module docstring)
REDUCED = (torch.float16, torch.bfloat16)


DEFAULTS = (None, 1, "valid", 1, 1)  # bias, stride, padding, dilation, groups; the dispatcher omits trailing defaults


def _as_list(value, n):
    return [value] * n if isinstance(value, int) else list(value)


def _valid_as_zeros(device_type, n, original):
    """``conv{n}d.padding`` for tracing: ``"valid"`` on ``device_type`` as ``n`` zeros; anything else ``original``."""
    conv = getattr(aten, f"conv{n}d").default

    def kernel(input, weight, *rest):
        bias, stride, padding, dilation, groups = (*rest, *DEFAULTS[len(rest):])
        if input.device.type != device_type or padding != "valid":
            return original(input, weight, *rest)
        count("valid_conv")
        return conv(input, weight, bias, _as_list(stride, n), [0] * n, _as_list(dilation, n), groups)

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
    """Whether Habana's bridge runs in lazy mode (read once; it is fixed when the bridge loads)."""
    if "lazy" not in _MODE:
        try:
            from habana_frameworks.torch.utils.internal import is_lazy
            _MODE["lazy"] = bool(is_lazy())
        except ImportError:
            _MODE["lazy"] = os.environ.get("PT_HPU_LAZY_MODE", "1") != "0"
    return _MODE["lazy"]


_MODE = {}


def _mode(input):
    """How this convolution runs: ``"compile"`` (a traced ``torch.compile`` graph), ``"lazy"`` (an HPU op in lazy
    mode) or ``"eager"`` (an HPU op in eager mode, and the CPU stand-in of the CPU tests, which patch this)."""
    if is_fake(input) or torch._guards.detect_fake_mode() is not None:
        return "compile"
    return "lazy" if input.device.type == "hpu" and _lazy() else "eager"


def _max_kernel(mode):
    """The largest kernel a single Synapse ``Conv`` node gets: ``LAZY_MAX_KERNEL`` (128) in lazy mode, else
    ``MAX_KERNEL`` (256: eager ops and traced ``torch.compile`` graphs, where 128-wide chunks failed more shapes than
    256-wide ones; see the module docstring). The CPU tests patch ``MAX_KERNEL``."""
    return min(MAX_KERNEL, LAZY_MAX_KERNEL) if mode == "lazy" else MAX_KERNEL


# The one class of wide kernels Gaudi2's native convolution runs, forward and backward, with the right numbers (bridge
# 1.21.4, measured by ``repro/sweep_conv_native.sh``, table in the README): a 2-D float32 convolution with one input
# channel and groups 1, no padding or dilation, whose kernel is wider than the bound only in its last dim, with
# ``kH <= 22`` and
#   eager: an even ``kW`` and a stride ``>= 32`` along it (every even ``kW`` from 258 to 6000 ran at strides 32 to kW;
#          every odd one tried, 257, 351, 1089, 1091 and 1501, fails ``setAssociatedDimAndSize``; (4, 4356) fails at
#          strides 1 to 16);
#   lazy:  any ``kW`` and a stride ``>= 8`` (every ``kW`` from 257 to 6000 ran at strides 8 to kW; at stride 1 with
#          the input as wide as the kernel the card stops with a device critical error).
# TensorCSPNet's default ``temporal_block`` ((4, 4356), stride (1, 484)) is in it in both modes, and its
# ``dims=(22, 11, 11, 11)`` one ((4, 1089), stride (1, 121)) in lazy mode, so they keep the native op. Everything
# else above the bound is split: 4 input channels, the wide dim first ((4356, 4) or STaRNet's (351, 1)), conv1d and
# 3-D convolutions, other dtypes, and traced graphs (not measured) included.
NATIVE_WIDE = {"eager": {"min_k": 258, "stride": 32, "even": True}, "lazy": {"min_k": 257, "stride": 8, "even": False}}
NATIVE_WIDE_MAX_KH = 22


def _native_wide(mode, input, weight, groups, strd, pad, dil):
    """Whether this convolution is in the measured native class above."""
    rule = NATIVE_WIDE.get(mode)
    if rule is None or input.dim() != 4 or input.dtype != torch.float32:
        return False
    k_h, k_w = weight.shape[2:]
    return (groups == 1 and weight.shape[1] == 1 and k_h <= min(NATIVE_WIDE_MAX_KH, _max_kernel(mode))
            and k_w >= rule["min_k"] and (k_w % 2 == 0 or not rule["even"]) and strd[1] >= rule["stride"]
            and all(p == 0 for p in pad) and all(d == 1 for d in dil))


def _large_kernel_takes(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups):
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
    mode = _mode(input)
    if not any(k > _max_kernel(mode) for k in weight.shape[2:]):
        return False
    if _native_wide(mode, input, weight, groups, strd, pad, dil):
        count("large_conv_native")
        return False
    return all(l_in + 2 * p >= d * (k - 1) + 1
               for l_in, k, p, d in zip(input.shape[2:], weight.shape[2:], pad, dil))


def _split_conv(op, fallback, input, weight, bias, stride, padding, dilation, groups):
    """Evaluate a non-transposed convolution by partitioning spatial kernel dims above ``MAX_KERNEL`` into chunks
    of at most ``MAX_KERNEL`` with exact input slices and ``F.pad`` (``conv_pad = 0`` on split dims). For 16-bit
    inputs each chunk's convolution runs and rounds in the input dtype; only the rounded chunk outputs are summed in
    float32, so the result is rounded twice (measured on the CPU, bf16 conv1d 3x1000 by 300: max error 0.50 against
    float64, the native op's 0.25). float32 and float64 splits match the native op to rounding. The sum runs in
    another order than native's: where the reduction already overflows (float32 inputs near 3e38), a split output
    can be NaN (+inf + -inf) where native gives inf; both are non-finite in the same places."""
    n = input.dim() - 2
    max_k = _max_kernel(_mode(input))
    strd, pad, dil = _expand(stride, n), _expand(padding, n), _expand(dilation, n)
    out_sizes = [(l_in + 2 * p - d * (k - 1) - 1) // s + 1
                 for l_in, k, s, p, d in zip(input.shape[2:], weight.shape[2:], strd, pad, dil)]

    dim_plans = []
    for l_in, k, l_out, s, p, d in zip(input.shape[2:], weight.shape[2:], out_sizes, strd, pad, dil):
        if k <= max_k:
            dim_plans.append([(0, k, 0, l_in, 0, 0, p)])
            continue
        plan = []
        for k0 in range(0, k, max_k):
            c = min(max_k, k - k0)
            x_start = k0 * d - p
            x_end = (l_out - 1) * s + k0 * d + (c - 1) * d - p + 1
            sc, ec = max(0, x_start), min(l_in, x_end)
            if sc >= ec:
                continue
            plan.append((k0, k0 + c, sc, ec, sc - x_start, x_end - ec, 0))
        dim_plans.append(plan)

    reduced = input.dtype in REDUCED
    acc_dtype = torch.float32 if reduced else input.dtype
    with torch.autocast(input.device.type, enabled=False):
        acc = None
        for combo in itertools.product(*dim_plans):
            w_slices = (slice(None), slice(None), *(slice(k0, k1) for k0, k1, *_ in combo))
            x_slices = (slice(None), slice(None), *(slice(sc, ec) for _, _, sc, ec, *_ in combo))
            w_chunk, x_chunk = weight[w_slices], input[x_slices]
            pads = [(pl, pr) for _, _, _, _, pl, pr, _ in combo]
            if any(pl > 0 or pr > 0 for pl, pr in pads):
                x_chunk = F.pad(x_chunk, [v for pair in reversed(pads) for v in pair])
            conv_pad = [cp for *_, cp in combo]
            chunk_bias = None if (reduced or acc is not None) else bias
            y_chunk = op._op_dk(fallback, x_chunk, w_chunk, chunk_bias, strd, conv_pad, dil, False, [0] * n, groups)
            y_val = y_chunk.float() if reduced else y_chunk
            acc = y_val if acc is None else acc + y_val

        if acc is None:
            acc = input.new_zeros((input.shape[0], weight.shape[0], *out_sizes), dtype=acc_dtype)
            if n == 2 and input.is_contiguous(memory_format=torch.channels_last):
                acc = acc.to(memory_format=torch.channels_last)
            elif n == 3 and input.is_contiguous(memory_format=torch.channels_last_3d):
                acc = acc.to(memory_format=torch.channels_last_3d)
            acc = acc + (input[:0].sum() + weight[:0].sum()).to(acc_dtype)
            if bias is not None:
                acc = acc + bias.to(acc_dtype).view(1, -1, *([1] * n))
        elif reduced and bias is not None:
            acc = acc + bias.float().view(1, -1, *([1] * n))
        return acc.to(input.dtype) if reduced else acc


def _large_kernel_wrapper(op, fallback):
    """``aten.convolution.default`` kernel: split kernels with spatial size ``> MAX_KERNEL``; fall back otherwise."""

    def kernel(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups):
        if not _large_kernel_takes(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups):
            return op._op_dk(fallback, input, weight, bias, stride, padding, dilation, transposed, output_padding, groups)
        count("large_conv")
        return _split_conv(op, fallback, input, weight, bias, stride, padding, dilation, groups)

    return kernel


def install_large_kernel(device_type):
    """Split convolutions with spatial kernel size ``> MAX_KERNEL`` on ``device_type``. Returns an undo."""
    op = aten.convolution.default
    return register_kernels(device_type, [
        (op, "Autograd{}", _large_kernel_wrapper(op, DispatchKey.Autograd)),
        (op, "{}", _large_kernel_wrapper(op, DispatchKey.CompositeExplicitAutograd)),
    ])
