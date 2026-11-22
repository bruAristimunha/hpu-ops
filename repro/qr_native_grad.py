"""Gaudi2 (bridge 1.21.4): the gradient through Habana's native ``torch.linalg.qr`` (reduced, batch 32, n = 22) of a
sign-invariant loss, ``(Q R).sum() + |R|.sum()``, is far off torch's on the CPU for some well-conditioned inputs
(relative error 0.51 in tests/linalg_on_hpu.py's qr_bwd row), while the same backward fed CPU float32 Q and R is right
(qr_bwd.isolated). No hpu_ops: plain torch on the card.
    python3 repro/qr_native_grad.py [--seeds 200] [--input dump.pt]
With ``--input`` (``LINALG_ON_HPU_DUMP=dir python3 tests/linalg_on_hpu.py`` saves a failing case's inputs) the dumped
batch; otherwise random inputs of the same family, seed by seed. Prints the inputs whose relative gradient error
exceeds 1e-3, and the worst."""
import argparse
import math

import habana_frameworks.torch.core as htcore
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, default=200)
ap.add_argument("--input", default=None)
args = ap.parse_args()


def loss(Q, R):
    return (Q @ R).sum() + R.abs().sum()


worst = (0.0, None)
dumped = torch.load(args.input)["inputs"][0] if args.input else None
for seed in range(1 if args.input else args.seeds):
    g = torch.Generator().manual_seed(seed)
    A = dumped if dumped is not None else torch.randn(32, 22, 22, generator=g) + 2 * math.sqrt(22) * torch.eye(22)
    ref = A.double().requires_grad_()
    loss(*torch.linalg.qr(ref)).backward()
    h = A.to("hpu").requires_grad_()
    loss(*torch.linalg.qr(h)).backward()
    htcore.mark_step()
    err = float((h.grad.cpu().double() - ref.grad).abs().max() / ref.grad.abs().max())
    if err > 1e-3:
        Qh, Rh = (t.cpu() for t in torch.linalg.qr(A.to("hpu")))
        print(f"seed {seed}: gradient error {err:.3g}; reconstruction |QR - A| {float((Qh @ Rh - A).abs().max()):.3g}, "
              f"min |diag R| {float(Rh.diagonal(dim1=-2, dim2=-1).abs().min()):.3g}", flush=True)
    worst = max(worst, (err, seed))
print(f"worst: seed {worst[1]}, gradient error {worst[0]:.3g}")
