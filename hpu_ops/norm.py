"""Group norm with symbolic shapes under ``torch.compile`` (#42).

With a symbolic size (the default ``dynamic`` setting after a second batch size), torch's ``group_norm`` traces to
``native_group_norm(input, weight, bias, s0, C, HxW, group, eps)``: the batch size reaches the op as an int argument.
Habana's ``hpu_backend`` passes that value into its fused graph as a tensor, converts it back with
``aten::IntImplicit`` (a TorchScript symbol, not a dispatcher op) and fails: "aten::IntImplicit isn't registered in
KernelRegistry!". There, while tracing only, group norm is computed from a reshape into groups, ``var_mean`` and the
affine terms, which take no size as an int; autograd differentiates the composite. Static shapes and eager calls keep
Gaudi's own kernel.
"""
import torch
from torch._C import DispatchKey

from ._dispatch import count, replace_py_kernels, static

aten = torch.ops.aten
REDUCED = (torch.float16, torch.bfloat16)


def group_norm(input, num_groups, weight=None, bias=None, eps=1e-5):
    """``F.group_norm`` from views and reductions, as torch's reference ``native_group_norm`` computes it: biased
    variance over each group's channels and inner dims, float32 math for 16-bit inputs, the input dtype back."""
    dtype = torch.float32 if input.dtype in REDUCED else input.dtype
    with torch.autocast(input.device.type, enabled=False):
        x = input.reshape(input.shape[0], num_groups, -1).to(dtype)
        var, mean = torch.var_mean(x, dim=-1, correction=0, keepdim=True)
        out = ((x - mean) * torch.rsqrt(var + eps)).view(input.shape)
        affine = (-1,) + (1,) * (input.dim() - 2)  # per channel, broadcast over the inner dims
        if weight is not None:
            out = out * weight.view(affine)
        if bias is not None:
            out = out + bias.view(affine)
        return out.to(input.dtype)


def _takes(input, num_groups):
    """Valid inputs with a symbolic size; invalid ones reach torch's kernel, which raises its usual error."""
    return (input.dim() >= 2 and input.is_floating_point() and not static(*input.shape)
            and isinstance(num_groups, int) and num_groups > 0 and input.shape[1] % num_groups == 0)


def _group_norm_kernel(device_type, original):
    def kernel(input, num_groups, weight=None, bias=None, eps=1e-5, cudnn_enabled=True):
        if input.device.type == device_type and _takes(input, num_groups):
            count("group_norm")
            return group_norm(input, num_groups, weight, bias, eps)
        return original(input, num_groups, weight, bias, eps, cudnn_enabled)

    return kernel


def install_group_norm(device_type):
    """While tracing, ``group_norm`` on ``device_type`` with a symbolic size as a composite. Returns an undo."""
    return replace_py_kernels(aten.group_norm.default, (DispatchKey.CompositeImplicitAutograd,),
                              lambda original: _group_norm_kernel(device_type, original))
