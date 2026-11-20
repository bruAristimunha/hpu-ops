"""Gaudi2, bridge 1.21.4, lazy mode: the output of a TPC custom op, read with ``.cpu()`` with no ATen op after the
custom op, comes back wrong. One ATen op after it (a copy) gives the right values.
    PT_HPU_LAZY_MODE=1 python3 repro/lazy_custom_op_output_cpu.py   (and =0 for the eager control)
The custom op is hpu_ops' ``linalg_tpc::tri_solve_lower`` (forward substitution, A[b, n, n] X = B[b, n, m]); any
single-output custom op registered through ``HabanaCustomOpDescriptor`` should behave the same. The script prints the
relative error of each way of reading the output against float64 on the CPU (correct is ~1e-7)."""
import os

import hpu_ops

hpu_ops.enable()  # puts liblinalg_tpc.so on GC_KERNEL_PATH before the first HPU op

import habana_frameworks.torch.core as htcore  # noqa: E402
import torch  # noqa: E402

from hpu_ops import tpc  # noqa: E402

LAZY = os.environ.get("PT_HPU_LAZY_MODE", "1") != "0"
torch.manual_seed(0)
b, n, m = 32, 22, 3
A = torch.tril(torch.randn(b, n, n)) / n + 2 * torch.eye(n)
B = torch.randn(b, n, m)
ref = torch.linalg.solve_triangular(A.double(), B.double(), upper=False)
op = tpc._tri_op(False, False)


def err(x):
    return float((x.double() - ref).abs().max() / ref.abs().max())


ways = {
    "raw output, .cpu()": lambda X: X,
    "reshape to the same shape (a view)": lambda X: X.reshape(b, n, m),
    "clone()": lambda X: X.clone(),
    "X * 1": lambda X: X * 1,
    "mark_step() then .cpu()": lambda X: (htcore.mark_step(), X)[1],
}
print("lazy", LAZY, "torch", torch.__version__)
for name, read in ways.items():
    for trial in range(2):
        Ah, Bh = A.to("hpu"), B.to("hpu")
        htcore.mark_step()  # inputs materialized: the op is the only node of its graph
        X = op(Ah, Bh)
        print(f"{name:40s} trial {trial}: rel err {err(read(X).cpu()):.3g}", flush=True)
