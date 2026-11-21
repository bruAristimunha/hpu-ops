"""HPU composite for ``linalg.eigh`` and ``eigvalsh`` (``aten._linalg_eigh``), which Gaudi runs on the CPU: cyclic
parallel Jacobi built from matmuls that stay on the card and compile into the step graph.

The n/2 disjoint rotations of a round-robin round are one batched matmul, and a fixed number of sweeps replaces the
convergence test, so the op count is static and nothing syncs with the host. The backward is torch's Hermitian
``linalg_eig_backward`` over differentiable ops. Input rules are those of ``hpu_ops.linalg``: real float32 matrices of
static shape, float64 off ``hpu`` only.

On Gaudi2 the composite loses to Habana's CPU fallback (760-7600 ms per call against 2.8-35 ms, batch 32, n = 16-128,
and a Synapse critical device error in lazy mode at n = 64), so float32 matrices of n <= ``tpc.EIGH_MAXN`` go through
the TPC one-sided Jacobi kernel of ``hpu_ops.tpc`` (n = 22, batch 32, lazy: 0.54 / 1.0 ms in two processes against
2.4 / 2.7 for the fallback; eager varies more, see ``tests/linalg_percall.py``). Without the kernel, or past
``EIGH_MAXN``, the call on ``hpu`` is declined and runs through hpu-ops' own CPU route
(``linalg._on_cpu``), which is not Habana's fallback. The composite stays the path of the CPU stand-in.
"""
import os

import torch
import torch.nn.functional as F

from . import tpc
from .extrema import internal_sort
from .linalg import _function, _kernels, _takes, register_paired

aten = torch.ops.aten
JACOBI_SWEEPS = 10
# The ``(A, upper) -> (w, V)`` solver ``Eigh.compute`` calls for n <= tpc.EIGH_MAXN. None means ``tpc.eigh`` on HPU with
# the kernel ready; the CPU tests set it to ``tpc.eigh_reference`` to exercise the wiring.
_EIGH = None


def _use_tpc(A):
    """Whether ``A`` goes through the Jacobi kernel: n <= ``tpc.EIGH_MAXN`` and, on HPU, float32 with the kernel ready;
    or, with the ``_EIGH`` hook set (CPU tests), any dtype."""
    if A.shape[-1] > tpc.EIGH_MAXN:
        return False
    return _EIGH is not None or (A.device.type == "hpu" and A.dtype == torch.float32 and tpc.ready())


def _tpc_takes(device_type):
    """``_takes`` for eigh, and on ``hpu`` also: the TPC kernel ready (n <= ``tpc.EIGH_MAXN`` the Jacobi kernel, past it
    the block Jacobi of ``hpu_ops.jacobi`` on the card). Otherwise the call is declined and takes hpu-ops' own CPU
    route, which beats the composite on Gaudi2."""
    takes = _takes(device_type, "eigh")

    def tpc_takes(*args, **kwargs):
        if not takes(*args, **kwargs):
            return False
        if device_type == "hpu" and args[0].shape[-1] > tpc.EIGH_MAXN and os.environ.get(LARGE_EIGH) == "cpu":
            return False  # the opt-out: hpu-ops' CPU route past the kernel (see LARGE_EIGH)
        return device_type != "hpu" or tpc.ready()

    return tpc_takes


# Past tpc.EIGH_MAXN eigh runs on the card: the one-sided Jacobi kernel jacobi_big up to tpc.BIG_EIGH_MAXN (batch 32:
# 11.6 / 21.9 / 29 ms at n = 55 / 110 / 128, hpu-ops' CPU route 9.6 / 26.2 / 34.7; tests/probe_eigh_big.py), past it
# the block Jacobi of hpu_ops.jacobi on 128-wide pairs through jacobi_big, run to convergence (see jacobi.py).
# ``HPU_OPS_LARGE_EIGH=cpu`` (read per call) asks for the CPU route past tpc.EIGH_MAXN instead.
LARGE_EIGH = "HPU_OPS_LARGE_EIGH"


# The ``(A) -> (w, V)`` solver for EIGH_MAXN < n <= BIG_EIGH_MAXN (both triangles of A filled) in the CPU tests; None
# means ``tpc.eigh_big`` on HPU.
_EIGH_BIG = None


def _use_big(A):
    """Whether ``A`` goes through the one-sided Jacobi kernel ``jacobi_big`` (``tpc.eigh_big``): ``tpc.EIGH_MAXN`` <
    n <= ``tpc.BIG_EIGH_MAXN`` on HPU, float32 with the kernel ready (or the ``_EIGH_BIG`` hook, CPU tests)."""
    if not tpc.EIGH_MAXN < A.shape[-1] <= tpc.BIG_EIGH_MAXN:
        return False
    return _EIGH_BIG is not None or (A.device.type == "hpu" and A.dtype == torch.float32 and tpc.ready())


def _use_block(A):
    """Past ``tpc.EIGH_MAXN`` on HPU (float32, the kernel ready), or with the ``linalg._BLOCK_EIGH`` hook (CPU tests):
    the block Jacobi of ``hpu_ops.jacobi``, its pairs through the eigh kernel."""
    from . import linalg

    if A.shape[-1] <= tpc.EIGH_MAXN:
        return False
    return linalg._BLOCK_EIGH is not None or (A.device.type == "hpu" and A.dtype == torch.float32 and tpc.ready())


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
    w, order = internal_sort(torch.diagonal(S, dim1=-2, dim2=-1)[..., :n], -1)
    V = V[..., :n, :n]
    return w, V.gather(-1, order[..., None, :].expand_as(V))


@_function
class Eigh(torch.autograd.Function):
    """``aten._linalg_eigh`` from parallel Jacobi, with torch's Hermitian backward (``linalg_eig_backward``) over the
    result. ``V`` is empty when ``compute_v`` is false."""

    @staticmethod
    def compute(A, UPLO, compute_v):
        if UPLO.upper() not in ("L", "U"):  # torch's own message
            raise RuntimeError(f"Expected UPLO argument to be 'L' or 'U', but got {UPLO}")
        upper = UPLO.upper() == "U"
        if _use_tpc(A):
            w, V = (tpc.eigh if _EIGH is None else _EIGH)(A, upper)
        elif _use_big(A):
            S = torch.triu(A) + torch.triu(A, 1).mT if upper else torch.tril(A) + torch.tril(A, -1).mT
            w, V = (tpc.eigh_big if _EIGH_BIG is None else _EIGH_BIG)(S)
            V = V if compute_v else A.new_empty(0)
        elif _use_block(A):
            from . import jacobi, linalg

            S = torch.triu(A) + torch.triu(A, 1).mT if upper else torch.tril(A) + torch.tril(A, -1).mT
            w, V = jacobi.block_eigh(S, linalg._block_eigh_solver(A.device), compute_v)
            V = V if compute_v else A.new_empty(0)
        else:
            w, V = eigh(A, upper)
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
        if V.device.type == "hpu" and V.shape[-1] > tpc.EIGH_MAXN:
            # lazy mode, Gaudi2 (bridge 1.21.4): linalg_on_hpu's eigh at n = 64 and 128 (batch 32) failed to compile a
            # fused graph (synStatus 26) where the isolated forward and backward compiled; a graph boundary here, as
            # in linalg._pade13_uv
            tpc._flush()
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
    """``linalg.eigh`` and ``eigvalsh`` on ``device_type``: the TPC Jacobi kernel on HPU (declined without it, then
    hpu-ops' CPU route), parallel Jacobi composites on the CPU stand-in. Returns an undo."""
    op = aten._linalg_eigh.default
    return register_paired(device_type, [_kernels(device_type, op, "eigh", Eigh, _tpc_takes(device_type))])
