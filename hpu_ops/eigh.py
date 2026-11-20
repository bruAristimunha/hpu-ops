"""HPU composite for ``linalg.eigh`` and ``eigvalsh`` (``aten._linalg_eigh``), which Gaudi runs on the CPU: cyclic
parallel Jacobi built from matmuls that stay on the card and compile into the step graph.

The n/2 disjoint rotations of a round-robin round are one batched matmul, and a fixed number of sweeps replaces the
convergence test, so the op count is static and nothing syncs with the host. The backward is torch's Hermitian
``linalg_eig_backward`` over differentiable ops. Input rules are those of ``hpu_ops.linalg``: real float32 matrices of
static shape, float64 off ``hpu`` only.

Opt-in, not installed by ``hpu_ops.enable()``: on Gaudi2 (bridge 1.21.4) it takes 760-7600 ms per call against 2.8-35 ms
for Habana's CPU fallback (batch 32, n = 16-128), and in lazy mode it raises a Synapse critical device error at n = 64.
"""
import torch
import torch.nn.functional as F

from ._dispatch import register_kernels
from .linalg import _kernels, _takes

aten = torch.ops.aten
JACOBI_SWEEPS = 10


def _schedule(n, device=None):
    """The round-robin (circle method) pairing of ``n`` (even) indices: ``[n - 1, n // 2, 2]``, each row ``(p, q)``
    with ``p < q``; every index once per round, every pair once over the rounds. Built on ``device``."""
    others, rounds = list(range(1, n)), []
    for _ in range(n - 1):
        line = [0] + others
        rounds.append([sorted((line[i], line[n - 1 - i])) for i in range(n // 2)])
        others = others[-1:] + others[:-1]
    return torch.tensor(rounds, device=device)


def eigh(A, upper=False):
    """``(w, V)`` of the symmetric matrix held in one triangle of ``A``: ``w`` ascending, ``V`` with orthonormal
    columns. Odd sizes are padded with a zero row and column, whose rotations are the identity."""
    # ponytail: fixed JACOBI_SWEEPS and O(n^3) rotations as matmuls; a TPC kernel (spec P5) if the model-step gate loses
    n = A.shape[-1]
    m = n + n % 2
    S = torch.triu(A) + torch.triu(A, 1).mT if upper else torch.tril(A) + torch.tril(A, -1).mT
    S = F.pad(S, (0, m - n, 0, m - n))
    batch = S.shape[:-2]
    V = torch.eye(m, dtype=S.dtype, device=S.device).expand_as(S)
    eyef = torch.eye(m, dtype=S.dtype, device=S.device).flatten().expand(*batch, m * m)
    rounds = []  # index tensors of each round, built once
    for r in _schedule(m, S.device):
        p, q = r[:, 0], r[:, 1]
        rounds.append((p * m + p, q * m + q, p * m + q, q * m + p))
    for _ in range(JACOBI_SWEEPS):
        for pp, qq, pq, qp in rounds:
            flat = S.flatten(-2)
            app, aqq, apq = (flat.gather(-1, i.expand(*batch, -1)) for i in (pp, qq, pq))
            # a converged pair (and the padded index) has apq == 0: theta 0, so J is exactly the identity
            theta = torch.where(apq == 0, 0.0, 0.5 * torch.atan(2 * apq / (aqq - app)))
            c, s = theta.cos(), theta.sin()
            index = torch.cat([pp, qq, pq, qp]).expand(*batch, -1)
            J = eyef.scatter(-1, index, torch.cat([c, c, s, -s], -1)).unflatten(-1, (m, m))
            S = J.mT @ S @ J
            V = V @ J
        S = (S + S.mT) / 2
    w, order = torch.diagonal(S, dim1=-2, dim2=-1)[..., :n].sort(-1)
    V = V[..., :n, :n]
    return w, V.gather(-1, order[..., None, :].expand_as(V))


class Eigh(torch.autograd.Function):
    """``aten._linalg_eigh`` from parallel Jacobi, with torch's Hermitian backward (``linalg_eig_backward``) over the
    result. ``V`` is empty when ``compute_v`` is false."""

    @staticmethod
    def compute(A, UPLO, compute_v):
        w, V = eigh(A, UPLO == "U")
        return w, (V if compute_v else A.new_empty(0))

    @staticmethod
    def forward(ctx, A, UPLO, compute_v):
        w, V = Eigh.compute(A, UPLO, compute_v)
        ctx.save_for_backward(w, V)
        ctx.set_materialize_grads(False)  # a zero gV would make VhgV / E 0/0 on repeated eigenvalues
        return w, V

    @staticmethod
    def backward(ctx, gw, gV):
        w, V = ctx.saved_tensors
        if gw is None and gV is None:
            return None, None, None
        if gV is None:  # eigenvalues only: V diag(gw) V^H, finite for repeated eigenvalues
            return (V * gw.unsqueeze(-2)) @ V.mH, None, None
        gw = torch.zeros_like(w) if gw is None else gw
        VhgV = V.mH @ gV
        VhgV = 0.5 * (VhgV - VhgV.mH)
        eye = torch.eye(w.shape[-1], dtype=torch.bool, device=w.device)
        # where and diag_embed, not in-place fill_ and copy_ on E and inner: autograd needs those tensors intact
        E = torch.where(eye, 1.0, w.unsqueeze(-2) - w.unsqueeze(-1))
        inner = torch.where(eye, 0.0, VhgV / E) + torch.diag_embed(gw)
        return V @ inner @ V.mH, None, None


def install_eigh(device_type):
    """``linalg.eigh`` and ``eigvalsh`` on ``device_type`` as parallel Jacobi. Returns an undo."""
    op = aten._linalg_eigh.default
    return register_kernels(device_type, _kernels(device_type, op, "eigh", Eigh, _takes(device_type, "eigh")))
