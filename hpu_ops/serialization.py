"""``torch.save`` of HPU tensors that Gaudi keeps in a permuted memory layout.

In eager and compile mode, Synapse may write a result in a permuted layout and record the permutation on the
tensor's storage: a convolution weight after an optimizer step, for one. ``torch.save`` copies each storage to the
host as a flat byte tensor, and Habana's device-to-host copy cannot undo a 4-D permutation on a 1-D tensor ("Number of
dims in tensor don't match in permute", ``habana_eager/ops/copy_from.cpp``). A tensor-level copy can, since it knows
the shape.

So dense HPU tensors are reduced the way torch reduces XLA and MAIA tensors, whose storage it cannot copy either: a
tensor-level host copy, rebuilt by ``torch._utils._rebuild_device_tensor_from_cpu_tensor``. Loading puts them back
on the HPU, or where ``map_location`` says, and ``weights_only`` loading allows it. As for XLA, each tensor is saved
on its own, so views of one storage no longer share it after loading. Habana pickles HPU tensors through host copies
too, in lazy mode only (``torch_overwrites.py``).

Habana's ``PT_HPU_ENABLE_SYNAPSE_OUTPUT_PERMUTE=0`` also avoids the failure, but slows compiled convolution training
by 23-43% (EEGNet, Deep4Net).
"""
import functools
import weakref

import torch
from torch.utils.hooks import warn_if_has_hooks

from . import _strict
from ._dispatch import count

_HOST_COPIES = weakref.WeakValueDictionary()  # id -> copy; with the CPU in the role of the device (the tests), a
# copy is itself a "device" tensor and must take torch's own path


def install(device_type):
    """Reduce dense tensors of ``device_type`` through a tensor-level host copy. Returns an undo."""
    original = torch.Tensor._reduce_ex_internal

    @functools.wraps(original)
    def reduce(self, proto):
        if self.device.type != device_type or self.layout != torch.strided or _HOST_COPIES.get(id(self)) is self:
            return original(self, proto)
        warn_if_has_hooks(self)
        if torch.serialization._serialization_tls.skip_data:  # as torch does for XLA
            raise RuntimeError(f"Cannot serialize {device_type} tensors under skip_data: hpu_ops saves them through "
                               "host copies")
        count("save")
        with _strict.allowed():  # saving is I/O, not a step's compute
            host = self.detach().cpu()  # Habana's tensor-level copy undoes the permutation
        _HOST_COPIES[id(host)] = host
        return torch._utils._rebuild_device_tensor_from_cpu_tensor, (host, self.dtype, str(self.device),
                                                                       self.requires_grad)

    torch.Tensor._reduce_ex_internal = reduce

    def undo():
        if torch.Tensor._reduce_ex_internal is reduce:  # a wrapper installed later stays
            torch.Tensor._reduce_ex_internal = original

    return undo
