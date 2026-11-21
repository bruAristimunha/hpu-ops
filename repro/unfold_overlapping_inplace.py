"""Gaudi2, bridge 1.21.4: an in-place op on an overlapping ``Tensor.unfold`` view (``x.unfold(1, 3, 1).add_(1)``).
torch leaves such writes undefined (it raises for some ops, not this one). The CPU adds 1 once per window that holds
an element; the card's result differs, in eager mode (Habana's own unfold) as in lazy mode.
    PT_HPU_LAZY_MODE=0|1 python3 repro/unfold_overlapping_inplace.py [--hpu-ops]"""
import sys

import habana_frameworks.torch.core as htcore
import torch

if "--hpu-ops" in sys.argv:
    import hpu_ops
    hpu_ops.enable()
x = torch.zeros(1, 6)
xh = x.to("hpu")
x.unfold(1, 3, 1).add_(1)
xh.unfold(1, 3, 1).add_(1)
htcore.mark_step()
print("cpu ", x.tolist())
print("card", xh.cpu().tolist())
