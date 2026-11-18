"""``torch.save`` of HPU tensors that Gaudi keeps in a permuted memory layout.

In eager and compile mode, Synapse may write a result in a permuted layout and record the permutation on the
tensor's storage: a convolution weight after an optimizer step, for one. ``torch.save`` copies each storage to the
host as a flat byte tensor, and Habana's device-to-host copy cannot undo a 4-D permutation on a 1-D tensor ("Number of
dims in tensor don't match in permute", ``habana_eager/ops/copy_from.cpp``). A tensor-level copy can, since it knows
the shape.

Habana pickles HPU tensors through such a copy in lazy mode: its ``torch_overwrites.py`` wraps
``Tensor._reduce_ex_internal`` and ``torch.save``. This does the same in eager and compile mode, at the tensor's
reduction, so tensors anywhere in a pickled object are covered. As in lazy mode, they are saved as CPU tensors:
``torch.load`` returns them on the CPU unless ``map_location`` says otherwise, ``load_state_dict`` copies them to the
module's device, and ``weights_only`` loading works. Two HPU views of one storage are saved as two copies.

Habana's ``PT_HPU_ENABLE_SYNAPSE_OUTPUT_PERMUTE=0`` also avoids the failure, but slows compiled convolution training
by 23-43% (EEGNet, Deep4Net).
"""
import functools

import torch

from ._dispatch import count


def install(device_type):
    """Pickle tensors of ``device_type`` through a tensor-level host copy. Returns an undo."""
    original = torch.Tensor._reduce_ex_internal

    @functools.wraps(original)
    def reduce(self, proto):
        if self.device.type != device_type:
            return original(self, proto)
        count("save")
        host = self.detach().cpu()  # Habana's tensor-level copy undoes the permutation
        return original(host.requires_grad_(self.requires_grad), proto)

    torch.Tensor._reduce_ex_internal = reduce

    def undo():
        torch.Tensor._reduce_ex_internal = original

    return undo
