"""Habana bug (Gaudi2, bridge 1.21.4, eager and lazy): ``bincount`` of a negative value does not raise, found for
hpu-ops #21.

    PT_HPU_LAZY_MODE=0 python bincount_negative.py     # and PT_HPU_LAZY_MODE=1

torch raises ``bincount only supports 1-d non-negative integral inputs.``; Gaudi2 returns ``[2, 1, 1]`` for
``[0, 1, -1, 2]`` (the -1 is counted, it seems as 0). hpu_ops raises torch's error for the inputs its bincount rewrite
takes (65536 to 2**24 elements); smaller inputs still reach Habana's kernel.
"""
import habana_frameworks.torch.core as htcore
import torch

try:
    out = torch.bincount(torch.tensor([0, 1, -1, 2], device="hpu"))
    htcore.mark_step()
    print("hpu returns", out.cpu().tolist(), "-> BUG (torch raises)", flush=True)
except RuntimeError as e:
    print("hpu raises:", str(e)[:100], "-> OK", flush=True)
