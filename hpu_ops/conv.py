"""Convolutions on HPU: ``padding="valid"`` under ``torch.compile``, and the weight gradient of one-channel
convolutions with a stride.

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
"""
import functools

import torch
import torch.nn.functional as F
from torch._C import DispatchKey

from ._dispatch import chain, count, register_kernels, replace_py_kernels

aten = torch.ops.aten


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
