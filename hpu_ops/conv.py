"""Convolutions with ``padding="valid"`` on HPU under ``torch.compile``.

``conv{1,2,3}d.padding`` is a C++ composite. For ``"valid"`` it calls ``aten.convolution`` with the one-element
padding list ``[0]`` (and output padding ``[0]``), which ATen broadcasts but Habana's compiled convolution reads as a
full list: a 2-D convolution gets a garbage second padding and the graph fails to compile (ATCNet, EEGInceptionERP,
EEGITNet). On HPU, ``"valid"`` goes through ``conv{n}d.default`` with ``n`` zeros instead, which passes full lists.
Eager calls do not trace through this kernel and keep working as before.
"""
import functools

import torch
from torch._C import DispatchKey

from ._dispatch import chain, count, replace_py_kernels

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
