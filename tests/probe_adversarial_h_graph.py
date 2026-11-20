"""On a Gaudi card, eager mode: what reaches the aot graphs of hpu_backend for an index that does not lead, with the
index tensors on the CPU or on the device, and whether negative indices come out right, with and without hpu_ops'
index rewrite (same process). PT_HPU_LAZY_MODE=0 TORCH_LOGS=aot_graphs python3 tests/probe_adversarial_h_graph.py"""
import habana_frameworks.torch.core  # noqa: F401
import torch

import hpu_ops
from hpu_ops import indexing


def cpu_neg(x):
    return x[..., torch.tensor([-1, 0]), torch.tensor([2, -3])]


def dev_neg(x):
    return x[..., torch.tensor([-1, 0], device=x.device), torch.tensor([2, -3], device=x.device)]


def main():
    hpu_ops.FIXES = tuple(f for f in hpu_ops.FIXES if f not in (indexing.install_index, indexing.install_mask_fill))
    hpu_ops.enable()
    torch.manual_seed(0)
    x0 = torch.randn(32, 6, 6)
    for arm in ("native", "fix"):
        undo = indexing.install_index("hpu") if arm == "fix" else None
        try:
            for f in (cpu_neg, dev_neg):
                torch._dynamo.reset()
                hpu_ops._STATE["calls"].clear()
                print(f"=== BEGIN {arm} {f.__name__}", flush=True)
                x = x0.clone().to("hpu").requires_grad_(True)
                out = torch.compile(f, backend="hpu_backend")(x)
                out.sum().backward()
                ref = f(x0.double())
                e = float((out.detach().cpu().double() - ref).abs().max())
                print(f"=== END {arm} {f.__name__} abs_err={e:.2e} rewrites={dict(hpu_ops._STATE['calls'])}", flush=True)
        finally:
            if undo:
                undo()


if __name__ == "__main__":
    main()
