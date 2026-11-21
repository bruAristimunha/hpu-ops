"""Gaudi2, bridge 1.21.4: a float64 HPU tensor holds float32 values, packed 4 bytes apart in an 8-byte-per-element
allocation. In eager mode an op on a float64 view with a storage offset reads at twice the offset (8 bytes per
element over 4-byte data), and silently returns shifted values (zeros past the end). int64 arithmetic in eager
mode wraps at 32 bits. Lazy mode reads the offsets right, and raises on float64 / int64 values out of the 32-bit
range instead of rounding them.
    PT_HPU_LAZY_MODE=0 python3 repro/float64_storage_offset_eager.py   (and =1 for lazy)
Expected (the CPU): every line ``ok``."""
import os

import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.hpu as hthpu
import torch

LAZY = os.environ.get("PT_HPU_LAZY_MODE", "1") != "0"
hpu = torch.device("hpu")


def run(name, fn, expected):
    try:
        got = fn()
        if LAZY:
            htcore.mark_step()
        hthpu.synchronize()
        got = got.cpu().tolist()
        print(f"{'ok  ' if got == expected else 'BAD '} {name}: {got} (expected {expected})")
    except RuntimeError as e:
        print(f"RAISE {name}: {str(e).splitlines()[0][:150]}")


v = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=torch.float64)
h = v.to(hpu)
run("float64 h[1:] + 0", lambda: h[1:] + 0, [2.0, 3.0, 4.0, 5.0, 6.0])
run("float64 h[1] + h[2]", lambda: h[1] + h[2], 5.0)
run("float64 (float32 card -> .double())[1:] * 1", lambda: v.float().to(hpu).double()[1:] * 1, [2.0, 3.0, 4.0, 5.0, 6.0])
run("float64 1 + 2**-40 round trip", lambda: torch.tensor([1 + 2.0 ** -40], dtype=torch.float64).to(hpu),
    [1 + 2.0 ** -40])
run("float64 storage as int32 on the card",
    lambda: torch.tensor([1 + 2.0 ** -40], dtype=torch.float64).to(hpu).view(torch.int32),
    torch.tensor([1 + 2.0 ** -40], dtype=torch.float64).view(torch.int32).tolist())
run("int64 (2**40 + 1) + 1", lambda: torch.tensor([2 ** 40 + 1], dtype=torch.int64).to(hpu) + 1, [2 ** 40 + 2])
