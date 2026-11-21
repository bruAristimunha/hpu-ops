"""Gaudi2, bridge 1.21.4: the gradient of ``torch.linalg.qr`` at n = 220 (batch 32, float32) is wrong on the card.
    PT_HPU_LAZY_MODE=0 python3 repro/habana_qr_n220_grad.py
Plain torch, no hpu_ops: loss ``sum(Q R) + sum(|R|)`` on a well-conditioned matrix, gradient against CPU float64.
Measured (eager): relative error 2.2e-6 at n = 128, 0.25 at n = 220 (CPU float32: 1.6e-6)."""
import habana_frameworks.torch.hpu  # noqa: F401
import torch

torch.manual_seed(0)
for n in (128, 220):
    x = torch.randn(32, n, n) + 2 * n ** 0.5 * torch.eye(n)

    def grad(x, device):
        x = x.to(device).detach().requires_grad_()
        Q, R = torch.linalg.qr(x)
        ((Q @ R).sum() + R.abs().sum()).backward()
        return x.grad.cpu().double()

    ref = grad(x.double(), "cpu")
    for device, inp in (("hpu", x), ("cpu", x)):
        print(f"n={n} {device} float32: rel err {float((grad(inp, device) - ref).abs().max() / ref.abs().max()):.2e}")
