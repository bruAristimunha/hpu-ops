"""Minimal reproducer: Habana's native torch.linalg.qr forward kernel loses precision at n=16 on Gaudi2
(bridge 1.21.4), causing linalg.qr_bwd to exceed 10x CPU float32 identically with and without hpu_ops.

Run on Gaudi2:
    PT_HPU_LAZY_MODE=0 python3 repro/habana_qr_n16_accuracy.py
    PT_HPU_LAZY_MODE=1 python3 repro/habana_qr_n16_accuracy.py
"""
import math
import os
import subprocess
import sys

import torch
from torch.utils._python_dispatch import TorchDispatchMode

LAZY = os.environ.get("PT_HPU_LAZY_MODE", "1") != "0"


def rel_err(got, ref):
    got, ref = got.detach().cpu().double(), ref.detach().double()
    scale = float(ref.abs().max())
    return float((got - ref).abs().max()) / (scale if scale > 0 else 1.0)


def qr_loss(q, r):
    return (q @ r).sum() + r.abs().sum()


class CpuQrForward(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func is torch.ops.aten.linalg_qr.default and args[0].device.type == "hpu":
            q, r = torch.linalg.qr(args[0].detach().cpu())
            return q.to(args[0].device), r.to(args[0].device)
        return func(*args, **(kwargs or {}))


def measure(label, a_cpu, enable_hpu_ops=False, cpu_qr_fwd=False):
    if enable_hpu_ops:
        import hpu_ops
        hpu_ops.enable()
    import habana_frameworks.torch.core as htcore
    import habana_frameworks.torch.hpu as hthpu

    HPU = torch.device("hpu")

    def sync():
        if LAZY:
            htcore.mark_step()
        hthpu.synchronize()

    a64 = a_cpu.double().requires_grad_()
    q64, r64 = torch.linalg.qr(a64)
    qr_loss(q64, r64).backward()

    a32 = a_cpu.clone().requires_grad_()
    q32, r32 = torch.linalg.qr(a32)
    qr_loss(q32, r32).backward()
    fwd32 = rel_err(q32 @ r32, a64)
    bwd32 = rel_err(a32.grad, a64.grad)

    ah = a_cpu.to(HPU).detach().requires_grad_()
    if cpu_qr_fwd:
        with CpuQrForward():
            qh, rh = torch.linalg.qr(ah)
    else:
        qh, rh = torch.linalg.qr(ah)
    loss = qr_loss(qh, rh)
    sync()
    fwd_hpu = rel_err(qh @ rh, a64)
    loss.backward()
    sync()
    bwd_hpu = rel_err(ah.grad, a64.grad)
    print(f"{label:38s} lazy={int(LAZY)} fwd={fwd_hpu:.3e} ({fwd_hpu / fwd32:5.1f}x cpu32) "
          f"bwd={bwd_hpu:.3e} ({bwd_hpu / bwd32:5.1f}x cpu32)", flush=True)


def main():
    mode = os.environ.get("REPRO_MODE", "all")
    if mode == "all":
        for m in ("baseline", "hpu_ops", "isolated"):
            env = {**os.environ, "REPRO_MODE": m}
            subprocess.run([sys.executable, __file__], env=env, check=True)
        return
    # Seed 6 exposes Habana's native QR precision loss on Gaudi2 bridge 1.21.4 in both eager and lazy modes
    torch.manual_seed(6)
    a = torch.randn(32, 16, 16) + 2 * math.sqrt(16) * torch.eye(16)
    if mode == "baseline":
        measure("baseline (native qr + native bwd)", a, enable_hpu_ops=False)
    elif mode == "hpu_ops":
        measure("hpu_ops (native qr + tri_solve bwd)", a, enable_hpu_ops=True)
    elif mode == "isolated":
        measure("hpu_ops (cpu32 qr + tri_solve bwd)", a, enable_hpu_ops=True, cpu_qr_fwd=True)


if __name__ == "__main__":
    main()
