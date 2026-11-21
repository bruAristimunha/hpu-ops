"""Gaudi lazy mode (bridge 1.21.4, torch 2.6): Habana's own ``logsumexp`` of a 2-D or larger tensor over ALL its dims,
without keepdim, fails at the graph's launch with `synNodeCreateWithId failed for node: squeeze with synStatus 1
[Invalid argument]`. No gradient is needed. Eager mode and the CPU are fine.

    PT_HPU_LAZY_MODE=1 python lazy_logsumexp_all_dims_squeeze.py [shape] [dims] [plain|keepdim|flatten]
    # default: 1,1,1  0,1,2  plain; one case per process

Measured on Gaudi2 (one process per case):

    fails, lazy:   (1,1,1) dims (0,1,2); (2,1,1) dims (0,1,2); (1,1) dims (0,1); (4,3) dims (0,1)
    passes, lazy:  (1,1,1) dims (1,2); 1-D (22,) and (1,) dim 0; (4,3) dims (0,1) keepdim=True;
                   x.flatten() then dim 0
    passes, eager: every case above

Workaround: keepdim=True then reshape(()), or flatten to 1-D first.
"""
import sys

import torch
import habana_frameworks.torch.core as htcore

shape = [int(s) for s in (sys.argv[1] if len(sys.argv) > 1 else "1,1,1").split(",")]
dims = [int(d) for d in (sys.argv[2] if len(sys.argv) > 2 else "0,1,2").split(",")]
variant = sys.argv[3] if len(sys.argv) > 3 else "plain"
CALLS = {"plain": lambda x: torch.logsumexp(x, dims),
         "keepdim": lambda x: torch.logsumexp(x, dims, keepdim=True),
         "flatten": lambda x: torch.logsumexp(x.flatten(), 0)}
x = torch.randn(shape)
want = torch.logsumexp(x, dims, keepdim=variant == "keepdim")
try:
    got = CALLS[variant](x.to("hpu"))
    htcore.mark_step()
    print(shape, dims, variant, "hpu", got.cpu().tolist(), "cpu", want.tolist(), "OK" if torch.allclose(got.cpu(), want) else "DIFF")
except RuntimeError as e:
    print(shape, dims, variant, "FAILS:", str(e).splitlines()[0][:200])
