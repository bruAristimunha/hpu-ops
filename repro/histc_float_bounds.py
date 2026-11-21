"""Habana bugs in ``torch.histc`` (Gaudi2, bridge 1.21.4), found for hpu-ops #21. Run one case per process:

    PT_HPU_LAZY_MODE=0 python histc_float_bounds.py float_bounds   # aborts the process (eager)
    PT_HPU_LAZY_MODE=0 python histc_float_bounds.py int_bounds     # OK
    PT_HPU_LAZY_MODE=0 python histc_float_bounds.py bins 300       # OK
    PT_HPU_LAZY_MODE=0 python histc_float_bounds.py bins 400       # "Can not compile graph"

1. ``torch.histc(x, bins, min=-0.5, max=63.5)``: non-integer float bounds raise ``0 INTERNAL ASSERT FAILED ...
   ivalue.h:652 ... expected int`` from ``habana::FillHistcParams`` -> ``IValue::toInt``, inside the eager lowering
   thread; the error is uncatchable and Synapse then reports "a device critical error that requires a restart" and kills
   the process. Integer-valued bounds (``min=0, max=63``) work.
2. ``bins >= 400`` (300 works) fails with ``Graph compile failed ... Can not compile graph``.
"""
import sys

import habana_frameworks.torch.core as htcore
import torch

hpu = torch.device("hpu")


def main():
    case = sys.argv[1]
    x = torch.randint(0, 64, (4096,), device=hpu).float()
    if case == "float_bounds":
        out = torch.histc(x, 64, -0.5, 63.5)
    elif case == "int_bounds":
        out = torch.histc(x, 64, 0, 63)
    else:
        bins = int(sys.argv[2])
        out = torch.histc(torch.arange(bins, device=hpu).float(), bins, 0, bins - 1)
    htcore.mark_step()
    print(case, sys.argv[2:], "sum", out.sum().item(), "OK", flush=True)


if __name__ == "__main__":
    main()
