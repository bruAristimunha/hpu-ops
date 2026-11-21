"""Group norm with symbolic shapes under ``torch.compile`` (#42).

With a symbolic size (the default ``dynamic`` setting after a second batch size), torch's ``group_norm`` traces to
``native_group_norm(input, weight, bias, s0, C, HxW, group, eps)``: the batch size reaches the op as a SymInt
argument (FX dump on bridge 1.21.4). Habana's ``hpu_backend`` then fails in its graph compile with "aten::IntImplicit
isn't registered in KernelRegistry!", presumably from converting that argument. ``aten::IntImplicit`` is a
TorchScript symbol without a dispatcher schema in torch 2.6, so ``torch.library`` cannot register a kernel for it.
There, while tracing only, group norm is computed from a reshape into groups, ``var_mean`` and the affine terms,
which take no size as an argument in the forward graph; autograd differentiates the composite. Static shapes and
eager calls keep Gaudi's kernel. ``count("group_norm")`` counts the tracer's calls of the kernel (dynamo's fake
propagation included), not launches on the card.
"""
import torch
from torch._C import DispatchKey

from ._dispatch import count, replace_py_kernels, static

aten = torch.ops.aten
REDUCED = (torch.float16, torch.bfloat16)


def group_norm(input, num_groups, weight=None, bias=None, eps=1e-5):
    """``F.group_norm`` from views and reductions: the biased variance over each group's channels and inner dims, the
    statistics and affine terms in float32 for 16-bit inputs, rounded once to the input dtype (bfloat16 bit for bit
    torch's CPU kernel; float16 within one ulp). The output is contiguous."""
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


def _takes(input, num_groups, weight, bias):
    """Valid inputs with a symbolic size; invalid ones reach torch's kernel, which raises its usual error."""
    if not (input.dim() >= 2 and input.is_floating_point() and not static(*input.shape)):
        return False
    channels = input.shape[1]
    return (isinstance(num_groups, int) and num_groups > 0 and channels % num_groups == 0
            and all(t is None or (t.dim() == 1 and t.numel() == channels) for t in (weight, bias)))


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
