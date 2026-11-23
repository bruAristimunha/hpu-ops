"""Dynamic-shape tracing rewrites under ``torch.compile`` (#42).

- ``aten.expand.default``: Habana's C++ ``BroadcastOperator`` / ``HabanaLaunchOpPT`` mis-evaluates target
  dimensions when an ``expand`` has both a symbolic size (in ``size`` or ``self.shape``) and ``-1`` dims
  (e.g. ``param[1, 1, D].expand(B * K, -1, -1)`` in ``NeuroRVQ``, ``NeuroRVQTokenizer``, ``DIVER1``, and
  ``LaBraM``). Resolving ``-1`` dims to ``self.shape[i - offset]`` at ``DispatchKey.CompositeImplicitAutograd``
  during tracing eliminates the crash across ``train``, ``no_grad``, and ``inference_mode`` while keeping static
  ``expand`` and eager calls untouched.
- ``aten.constant_pad_nd.default`` on ``int64``: inside a dynamic graph, Habana lowers ``constant_pad_nd`` on
  ``int64`` tensors (even static-shaped ones such as ``NeuroRVQ``'s ``spatial_ix``) to ``hpu.constant_pad_nd_ds``,
  which fails graph compile with ``synStatus 26``. Rewriting ``int64`` ``constant_pad_nd`` during tracing via
  ``narrow``, ``torch.full``, and ``torch.cat`` removes ``constant_pad_nd_ds`` while matching ATen's ``int64``
  truncation, cropping, and error semantics.
"""
import torch
from torch._C import DispatchKey

from ._dispatch import count, replace_py_kernels, static

aten = torch.ops.aten


def _expand_kernel(device_type, original):
    def kernel(self, size, *, implicit=False):
        if self.device.type == device_type and not static(*size, *self.shape):
            offset = len(size) - self.dim()
            if offset >= 0 and any(isinstance(s, int) and s == -1 for s in size[offset:]):
                resolved = [self.shape[i - offset] if isinstance(s, int) and s == -1 and i >= offset else s
                            for i, s in enumerate(size)]
                count("expand")
                return aten.expand.default(self, resolved, implicit=implicit)
        return NotImplemented

    return kernel


def _pad_int64(self, pad, value):
    if len(pad) % 2 != 0 or len(pad) // 2 > self.dim():
        return None
    out = self
    copied = False
    for k in range(len(pad) // 2):
        left, right = pad[2 * k], pad[2 * k + 1]
        dim = self.dim() - 1 - k
        crop_left = -left if left < 0 else 0
        crop_right = -right if right < 0 else 0
        if crop_left or crop_right:
            length = out.shape[dim] - crop_left - crop_right
            if isinstance(length, int) and length < 0:
                return None
            out = out.narrow(dim, crop_left, length)
        pad_left = left if left > 0 else 0
        pad_right = right if right > 0 else 0
        if pad_left or pad_right:
            parts = []
            if pad_left:
                sz = list(out.shape)
                sz[dim] = pad_left
                parts.append(torch.full(sz, value, dtype=self.dtype, device=self.device))
            parts.append(out)
            if pad_right:
                sz = list(out.shape)
                sz[dim] = pad_right
                parts.append(torch.full(sz, value, dtype=self.dtype, device=self.device))
            out = torch.cat(parts, dim=dim)
            copied = True
    return out if copied else out.clone()


def _pad_int64_kernel(device_type, original):
    def kernel(self, pad, value=0):
        if self.device.type == device_type and self.dtype == torch.int64:
            result = _pad_int64(self, pad, value)
            if result is not None:
                count("constant_pad_int64")
                return result
        return NotImplemented

    return kernel


def install_expand(device_type):
    """While tracing, resolve ``-1`` dims in symbolic ``aten.expand.default`` on ``device_type``. Returns an undo."""
    return replace_py_kernels(aten.expand.default, (DispatchKey.CompositeImplicitAutograd,),
                              lambda original: _expand_kernel(device_type, original))


def install_pad_int64(device_type):
    """While tracing, rewrite ``int64`` ``aten.constant_pad_nd.default`` on ``device_type``. Returns an undo."""
    return replace_py_kernels(aten.constant_pad_nd.default, (DispatchKey.CompositeImplicitAutograd,),
                              lambda original: _pad_int64_kernel(device_type, original))
