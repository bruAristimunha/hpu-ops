"""CPU tests of the block Jacobi routes past the Jacobi kernels' sizes (``hpu_ops.jacobi``): eigh and SVD with the
pairs' 32 x 32 eigenproblems solved by torch's CPU eigh in float32 (the card uses the eigh kernel), against float64."""
import pytest
import torch

import hpu_ops.eigh as E
from hpu_ops import jacobi, linalg


def eigh32(X):
    return torch.linalg.eigh(X)


def rel(x, ref):
    x, ref = x.double(), ref.double()
    return float((x - ref).abs().max() / ref.abs().max())


@pytest.mark.parametrize("n", [33, 64, 100])
@pytest.mark.parametrize("kind", ["indefinite", "spd"])
def test_block_eigh_matches_cpu_float32(n, kind):
    g = torch.Generator().manual_seed(n)
    G = torch.randn(4, n, n, generator=g)
    A = (G + G.mT) / 2 if kind == "indefinite" else G @ G.mT + n * torch.eye(n)
    w64 = torch.linalg.eigvalsh(A.double())
    w32, V32 = torch.linalg.eigh(A)
    w, V = jacobi.block_eigh(A, eigh32)
    rec = lambda w_, V_: rel(V_.double() * w_.double().unsqueeze(-2) @ V_.double().mT, A)  # noqa: E731
    assert rel(w, w64) <= 4 * rel(w32, w64) + 1e-7
    assert rec(w, V) <= 4 * rec(w32, V32) + 1e-7
    assert (V.mT @ V - torch.eye(n)).abs().max() < 1e-5
    wv, none = jacobi.block_eigh(A, eigh32, compute_v=False)
    assert none is None and torch.equal(wv, w)


@pytest.mark.parametrize("shape", [(64, 64), (80, 40), (40, 80), (70, 33)])
def test_block_svd(shape):
    g = torch.Generator().manual_seed(sum(shape))
    m, n = shape
    A = torch.randn(3, m, n, generator=g) + (2 * max(m, n) ** 0.5 * torch.eye(m, n) if m == n else 0)
    S64 = torch.linalg.svdvals(A.double())
    U, S, Vh = jacobi.block_svd(A, eigh32)
    k = min(m, n)
    assert rel(S, S64) < 1e-5
    assert rel((U * S.unsqueeze(-2)) @ Vh, A) < 1e-5
    assert (U.mT @ U - torch.eye(k)).abs().max() < 1e-5 and (Vh @ Vh.mT - torch.eye(k)).abs().max() < 1e-5
    assert torch.equal(jacobi.block_svd(A, eigh32, compute_uv=False)[1], S)


def test_routes_through_the_block_hook(monkeypatch):
    """With ``linalg._BLOCK_EIGH`` (the card's eigh kernel in the CPU tests' role) eigh and the SVD of n > 32 take the
    block Jacobi; full_matrices completes U by the projector's eigenvectors."""
    monkeypatch.setattr(linalg, "_BLOCK_EIGH", eigh32)
    g = torch.Generator().manual_seed(0)
    A = torch.randn(2, 48, 48, generator=g)
    S = A @ A.mT + 48 * torch.eye(48)
    w, V = E.Eigh.compute(S, "L", True)
    assert rel(w, torch.linalg.eigvalsh(S.double())) < 1e-6
    X = torch.randn(2, 80, 40, generator=g)
    U, Sv, Vh = linalg.Svd.compute(X, True, True)
    assert U.shape == (2, 80, 80) and (U.mT @ U - torch.eye(80)).abs().max() < 1e-5
    assert rel(Sv, torch.linalg.svdvals(X.double())) < 1e-5


def test_padding_never_mixes_with_the_spectrum():
    """n = 33 pads to 64 with a value past the spectrum: a zero eigenvalue (rank-deficient input) stays the input's."""
    g = torch.Generator().manual_seed(5)
    B = torch.randn(1, 33, 20, generator=g)
    A = B @ B.mT  # rank 20: 13 zero eigenvalues
    w, V = jacobi.block_eigh(A, eigh32)
    assert w[0, :13].abs().max() < 1e-4 * w.abs().max()
    assert (V.mT @ V - torch.eye(33)).abs().max() < 1e-5


def test_large_eigh_opt_out(monkeypatch):
    """``HPU_OPS_LARGE_EIGH=cpu`` declines eigh past the kernel on HPU (hpu-ops' CPU route, faster); the default
    takes it (the block Jacobi, on the card); at or below the kernel's size the variable changes nothing."""
    from hpu_ops import tpc

    monkeypatch.setattr(tpc, "ready", lambda: True)
    takes = E._tpc_takes("hpu")
    big, small = torch.randn(64, 64), torch.randn(32, 32)
    monkeypatch.delenv(E.LARGE_EIGH, raising=False)
    assert takes(big, "L", True) is True
    monkeypatch.setenv(E.LARGE_EIGH, "cpu")
    assert takes(big, "L", True) is False and takes(small, "L", True) is True


@pytest.mark.parametrize("n", [33, 55, 110])
def test_eigh_big_wiring_on_the_reference(n, monkeypatch):
    """``tpc.eigh_big`` (shift, kernel pass, refinement pass, Newton-Schulz, Rayleigh quotients) with the kernel replaced
    by its float64 contract (``tpc.jacobi_big_reference``); and Eigh's route to it for 32 < n <= 128."""
    from hpu_ops import tpc

    monkeypatch.setattr(tpc, "_big_op", lambda refine=False: (lambda S0: tpc.jacobi_big_reference(S0, 3 if refine else 8)))
    g = torch.Generator().manual_seed(n)
    G = torch.randn(2, n, n, generator=g)
    A = (G + G.mT) / 2
    w, V = tpc.eigh_big(A)
    w64 = torch.linalg.eigvalsh(A.double())
    assert rel(w, w64) <= 2 * rel(torch.linalg.eigvalsh(A), w64) + 1e-7
    assert (V.mT @ V - torch.eye(n)).abs().max() < 1e-5
    monkeypatch.setattr(E, "_EIGH_BIG", tpc.eigh_big)
    wl, Vl = E.Eigh.compute(torch.tril(A), "L", True)
    assert torch.equal(wl, w)
    wu, _ = E.Eigh.compute(torch.triu(A), "U", False)
    assert torch.equal(wu, w)
