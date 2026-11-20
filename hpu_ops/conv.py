"""Convolutions on HPU: ``padding="valid"`` under ``torch.compile``, the weight gradient of one-channel
convolutions with a stride, and spatial splitting of large-kernel convolutions (``> 256``).

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

Large kernels (``> 256``):

Gaudi2's MME AGU rejects convolutions whose spatial kernel dimension exceeds 256
(``gaudi2_agu_config.cpp::338 setAssociatedDimAndSize: size - 1 <= uint8 max`` in ``STaRNet(64)`` ``conv_spatial_fusion``
``(351, 1)`` and ``TensorCSPNet(22, dims=(22, 11, 11, 11))`` ``temporal_block`` ``(4, 1089)``). Splitting any kernel
dimension above ``MAX_KERNEL`` into ``<= MAX_KERNEL`` slices with exact input slicing and ``F.pad`` keeps every Synapse
``Conv`` node within the 8-bit AGU bound across eager, lazy, and ``torch.compile``, while autograd differentiates the
slice-sum naturally.
"""
import functools
import itertools

import torch
import torch.nn.functional as F
from torch._C import DispatchKey

from ._dispatch import chain, count, register_kernels, replace_py_kernels, static

aten = torch.ops.aten
MAX_KERNEL = 256
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
    if not any(k > MAX_KERNEL for k in weight.shape[2:]):
        return False
    return all(l_in + 2 * p >= d * (k - 1) + 1
               for l_in, k, p, d in zip(input.shape[2:], weight.shape[2:], pad, dil))


def _split_conv(op, fallback, input, weight, bias, stride, padding, dilation, groups):
    """Evaluate a non-transposed convolution by partitioning spatial kernel dims above ``MAX_KERNEL`` into chunks
    of at most ``MAX_KERNEL`` with exact input slices and ``F.pad`` (``conv_pad = 0`` on split dims). For 16-bit
    inputs each chunk's convolution runs and rounds in the input dtype; only the rounded chunk outputs are summed in
    float32, so the result is rounded twice (measured on the CPU, bf16 conv1d 3x1000 by 300: max error 0.50 against
    float64, the native op's 0.25). float32 and float64 splits match the native op to rounding."""
    n = input.dim() - 2
    strd, pad, dil = _expand(stride, n), _expand(padding, n), _expand(dilation, n)
    out_sizes = [(l_in + 2 * p - d * (k - 1) - 1) // s + 1
                 for l_in, k, s, p, d in zip(input.shape[2:], weight.shape[2:], strd, pad, dil)]

    dim_plans = []
    for l_in, k, l_out, s, p, d in zip(input.shape[2:], weight.shape[2:], out_sizes, strd, pad, dil):
        if k <= MAX_KERNEL:
            dim_plans.append([(0, k, 0, l_in, 0, 0, p)])
            continue
        plan = []
        for k0 in range(0, k, MAX_KERNEL):
            c = min(MAX_KERNEL, k - k0)
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
