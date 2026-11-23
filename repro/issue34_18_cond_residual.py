"""Issue #34 (#18): float32 ``linalg.solve`` accuracy at high condition numbers (~3e6).

On Gaudi2, ``hpu_ops`` runs Gauss-Jordan without LU+triangular factor emission (``gj_solve`` for ``n <= 64`` and
``gj_inv`` + one residual refinement step for ``n > 64``). At condition number ~3e6, the solution is backward-stable
within ``cond * eps_f32`` (~0.2), while forward relative error against float64 reaches ~1.8e-2 without on-card LU
factors and blocked triangular solves.

Run on Gaudi2:
    PT_HPU_LAZY_MODE=0 python3 repro/issue34_18_cond_residual.py
    PT_HPU_LAZY_MODE=1 python3 repro/issue34_18_cond_residual.py
"""
import math
import os

import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.hpu as hthpu
import torch

import hpu_ops

hpu_ops.enable()

LAZY = os.environ.get("PT_HPU_LAZY_MODE", "1") != "0"


def sync():
    if LAZY:
        htcore.mark_step()
    hthpu.synchronize()


def make_conditioned(batch, n, cond, seed=0):
    g = torch.Generator().manual_seed(seed)
    q1, _ = torch.linalg.qr(torch.randn(batch, n, n, generator=g, dtype=torch.float64))
    q2, _ = torch.linalg.qr(torch.randn(batch, n, n, generator=g, dtype=torch.float64))
    s = torch.logspace(0.0, -math.log10(cond), n, dtype=torch.float64)
    A64 = q1 @ torch.diag_embed(s.expand(batch, n)) @ q2.mT
    x_true = torch.randn(batch, n, 4, generator=g, dtype=torch.float64)
    B64 = A64 @ x_true
    return A64.float(), B64.float(), A64, B64


def metrics(A64, B64, X):
    X64 = X.double()
    ref = torch.linalg.solve(A64, B64)
    rel_err = float((X64 - ref).abs().max() / ref.abs().max())
    res = A64 @ X64 - B64
    bwd_stab = float(res.norm(dim=(-2, -1)).max() / (
        A64.norm(dim=(-2, -1)) * X64.norm(dim=(-2, -1)) + B64.norm(dim=(-2, -1))
    ).max())
    return rel_err, bwd_stab


def main():
    eps = torch.finfo(torch.float32).eps
    print(f"Issue #34 #18 cond residual repro | lazy={LAZY} | eps_f32={eps:.3e}", flush=True)
    for n in (22, 64, 96):
        for cond in (1e2, 1e4, 1e6, 3e6, 1e7):
            A32, B32, A64, B64 = make_conditioned(4, n, cond)
            X_cpu = torch.linalg.solve(A32, B32)
            err_cpu, res_cpu = metrics(A64, B64, X_cpu)
            X_hpu = torch.linalg.solve(A32.to("hpu"), B32.to("hpu"))
            sync()
            err_hpu, res_hpu = metrics(A64, B64, X_hpu.cpu())
            print(f"n={n:3d} cond={cond:7.1e} (cond*eps={cond*eps:.2e}) | "
                  f"cpu_f32 rel_err={err_cpu:.2e} bwd={res_cpu:.2e} | "
                  f"hpu_f32 rel_err={err_hpu:.2e} bwd={res_hpu:.2e}", flush=True)


if __name__ == "__main__":
    main()
