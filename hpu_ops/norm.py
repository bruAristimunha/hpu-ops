"""Group norm with symbolic shapes under ``torch.compile`` (#42).

With a symbolic size (the default ``dynamic`` setting after a second batch size), torch's ``group_norm`` traces to
``native_group_norm(input, weight, bias, s0, C, HxW, group, eps)``: the batch size reaches the op as a SymInt
argument (FX dump on bridge 1.21.4). Habana's ``hpu_backend`` then fails in its graph compile with "aten::IntImplicit
isn't registered in KernelRegistry!", presumably from converting that argument. ``aten::IntImplicit`` is a
TorchScript symbol without a dispatcher schema in torch 2.6, so ``torch.library`` cannot register a kernel for it.
There, while tracing only, group norm is computed from a reshape into groups, ``var_mean`` and the affine terms,
which take no symbolic scalar argument (the views keep the symbolic shape); autograd differentiates the composite. Static shapes and
eager calls keep Gaudi's kernel. ``count("group_norm")`` counts the tracer's calls of the kernel (dynamo's fake
propagation included), not launches on the card.
"""
import math

import torch
from torch._C import DispatchKey
from torch._prims_common import suggest_memory_format

from ._dispatch import chain, count, override_kernels, register_kernels, replace_py_kernels, static

aten = torch.ops.aten
REDUCED = (torch.float16, torch.bfloat16)


def group_norm(input, num_groups, weight=None, bias=None, eps=1e-5):
    """``F.group_norm`` from views and reductions: the biased variance over each group's channels and inner dims, the
    statistics and affine terms in float32 for 16-bit inputs, rounded once to the input dtype (bfloat16 bit for bit
    torch's CPU kernel; float16 within one ulp), in the input's memory format. float32 inputs whose variance
    overflows (about 1e19 and up) give a finite, wrong result where ATen's CPU kernel gives NaN."""
    dtype = torch.float32 if input.dtype in REDUCED else input.dtype
    with torch.autocast(input.device.type, enabled=False):
        # the group size written out, not -1: an empty batch has no -1 to infer (review of #49)
        x = input.reshape(input.shape[0], num_groups, input.shape[1] // num_groups * math.prod(input.shape[2:])).to(dtype)
        var, mean = torch.var_mean(x, dim=-1, correction=0, keepdim=True)
        out = ((x - mean) * torch.rsqrt(var + eps)).view(input.shape)
        affine = (-1,) + (1,) * (input.dim() - 2)  # per channel, broadcast over the inner dims
        if weight is not None:
            out = out * weight.view(affine)
        if bias is not None:
            out = out + bias.view(affine)
        return out.to(input.dtype).contiguous(memory_format=suggest_memory_format(input))


def _takes(input, num_groups, weight, bias):
    """Valid inputs with a symbolic size; invalid ones reach torch's kernel, which raises its usual error."""
    if not (input.dim() >= 2 and input.is_floating_point() and not static(*input.shape)):
        return False
    channels = input.shape[1]
    return (isinstance(num_groups, int) and num_groups > 0 and channels % num_groups == 0
            and all(t is None or (t.dim() == 1 and t.numel() == channels and _dtype_ok(input, t))
                    for t in (weight, bias)))


def _dtype_ok(input, param):
    """The dtype pairs ATen's kernels take: the input's, or float32 parameters of a 16-bit input."""
    return param.dtype == input.dtype or (input.dtype in REDUCED and param.dtype == torch.float32)


def _group_norm_kernel(device_type, original):
    def kernel(input, num_groups, weight=None, bias=None, eps=1e-5, cudnn_enabled=True):
        if input.device.type == device_type and _takes(input, num_groups, weight, bias):
            count("group_norm")
            return group_norm(input, num_groups, weight, bias, eps)
        return original(input, num_groups, weight, bias, eps, cudnn_enabled)

    return kernel


def install_group_norm(device_type):
    """While tracing, ``group_norm`` on ``device_type`` with a symbolic size as a composite. Returns an undo."""
    return replace_py_kernels(aten.group_norm.default, (DispatchKey.CompositeImplicitAutograd,),
                              lambda original: _group_norm_kernel(device_type, original))


def instance_norm(input, weight, bias, running_mean, running_var, use_input_stats, eps):
    """``F.instance_norm`` from views and reductions: the biased variance over each sample's and
    channel's inner dims (or the running statistics), statistics and affine in float32 for 16-bit
    inputs, in the input's memory format."""
    dtype = torch.float32 if input.dtype in REDUCED else input.dtype
    with torch.autocast(input.device.type, enabled=False):
        x = input.reshape(input.shape[0], input.shape[1], -1).to(dtype)
        if use_input_stats:
            var, mean = torch.var_mean(x, dim=-1, correction=0, keepdim=True)
        else:
            mean = running_mean.reshape(1, -1, 1).to(dtype)
            var = running_var.reshape(1, -1, 1).to(dtype)
        out = ((x - mean) * torch.rsqrt(var + eps)).view(input.shape)
        affine = (1, -1) + (1,) * (input.dim() - 2)
        if weight is not None:
            out = out * weight.view(affine)
        if bias is not None:
            out = out + bias.view(affine)
        return out.to(input.dtype).contiguous(memory_format=suggest_memory_format(input))


def _instance_norm_takes(input, weight, bias, running_mean, running_var, use_input_stats):
    """Valid calls; Habana's inference recipe rejects the op (EpiNT) and its training path
    reaches the same composite, whose var_mean autograd differentiates."""
    if not (input.dim() >= 2 and input.is_floating_point()):
        return False
    stats = use_input_stats or (running_mean is not None and running_var is not None
                                and running_mean.numel() == input.shape[1])
    params = all(t is None or (t.dim() == 1 and t.numel() == input.shape[1] and _dtype_ok(input, t))
                 for t in (weight, bias))
    return stats and params


def _instance_norm_kernel(op, device_type, fallback):
    def kernel(input, weight=None, bias=None, running_mean=None, running_var=None,
               use_input_stats=True, momentum=0.1, eps=1e-5, cudnn_enabled=True):
        if input.device.type == device_type and _instance_norm_takes(
                input, weight, bias, running_mean, running_var, use_input_stats):
            count("instance_norm")
            return instance_norm(input, weight, bias, running_mean, running_var, use_input_stats, eps)
        return op._op_dk(fallback, input, weight, bias, running_mean, running_var, use_input_stats,
                         momentum, eps, cudnn_enabled)

    return kernel


def install_instance_norm(device_type):
    """``instance_norm`` on ``device_type`` as a composite (traced, autograd and backend keys).
    Returns an undo."""
    op = aten.instance_norm.default
    below = DispatchKey.CompositeImplicitAutograd

    def traced(original):
        def kernel(input, weight=None, bias=None, running_mean=None, running_var=None,
                   use_input_stats=True, momentum=0.1, eps=1e-5, cudnn_enabled=True):
            if input.device.type == device_type and _instance_norm_takes(
                    input, weight, bias, running_mean, running_var, use_input_stats):
                count("instance_norm")
                return instance_norm(input, weight, bias, running_mean, running_var,
                                     use_input_stats, eps)
            return original(input, weight, bias, running_mean, running_var, use_input_stats,
                            momentum, eps, cudnn_enabled)

        return kernel

    return chain([
        replace_py_kernels(op, (DispatchKey.CompositeImplicitAutograd,), traced),
        register_kernels(device_type, [(op, "Autograd{}", _instance_norm_kernel(op, device_type, below))]),
        override_kernels(device_type, [(op, _instance_norm_kernel(op, device_type, below))]),
    ])
