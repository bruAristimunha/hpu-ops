"""Declined float64 ``linalg.cholesky`` on Gaudi2 (bridge 1.21.4) segfaulted with hpu-ops enabled:
the Autograd kernel's decline path redispatches at the Autograd key, which for the linalg ops is
the kernel itself (Habana registers no Autograd kernel below it), so the call recurses until the
stack overflows. Inference mode (backend key only) worked. Found via spd_learn GraphCSPNet's
``fit_graph`` (a float64 Cholesky of the fitted lattice); the fix routes declined float64 calls
below autograd, where the backend kernel's CPU route handles them.

    PT_HPU_LAZY_MODE=0 python cholesky_float64_declined.py   # segfault before the fix
"""
import torch

import hpu_ops

hpu_ops.enable()
torch.manual_seed(0)
A = torch.randn(36, 22, 22, dtype=torch.float64)
A = A @ A.mT + 22 * torch.eye(22, dtype=torch.float64)
Ah = A.to("hpu")
with torch.inference_mode():
    print("inference:", float(torch.linalg.cholesky(Ah).float().cpu().abs().sum()), flush=True)
Ag = Ah.detach().requires_grad_()
L = torch.linalg.cholesky(Ag)
print("grad-mode:", float(L.float().cpu().abs().sum()), flush=True)
L.sum().backward()
print("backward:", float(Ag.grad.float().cpu().abs().sum()), flush=True)
print("OK", flush=True)
