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


def test_routes_through_the_block_hook(monkeypatch):
    """With ``linalg._BLOCK_EIGH`` (the card's 128 x 128 eigh kernel in the CPU tests' role) eigh past
    ``tpc.BIG_EIGH_MAXN`` takes the block Jacobi, to convergence."""
    monkeypatch.setattr(linalg, "_BLOCK_EIGH", eigh32)
    g = torch.Generator().manual_seed(0)
    A = torch.randn(2, 160, 160, generator=g)
    S = A @ A.mT + 160 * torch.eye(160)
    w, V = E.Eigh.compute(S, "L", True)
    assert rel(w, torch.linalg.eigvalsh(S.double())) < 1e-6


@pytest.mark.parametrize("n, kind", [(512, "gauss"), (512, "spd"), (300, "gauss")])
def test_block_eigh_converges_at_large_n(n, kind):
    """Code review of #51: 16-wide blocks and 5 fixed sweeps left 1e-3 eigenvalue error at n = 512; 64-wide blocks run
    to convergence reach about CPU float32's."""
    g = torch.Generator().manual_seed(n)
    if kind == "gauss":
        G = torch.randn(1, n, n, generator=g)
        A = (G + G.mT) / 2
    else:
        Q = torch.linalg.qr(torch.randn(1, n, n, generator=g, dtype=torch.float64))[0]
        A = ((Q * torch.logspace(0, -6, n, dtype=torch.float64)) @ Q.mT).float()
        A = (A + A.mT) / 2
    w64 = torch.linalg.eigvalsh(A.double())
    w32 = torch.linalg.eigvalsh(A)
    w, V = jacobi.block_eigh(A, eigh32)
    assert rel(w, w64) <= 10 * rel(w32, w64)
    assert rel(V.double() * w.double()[..., None, :] @ V.double().mT, A) <= 10 * rel(w32, w64) + 1e-6


def test_padding_with_huge_entries_stays_finite():
    """Code review of #51, minor: entries near 1e19 overflowed the padding's Frobenius norm."""
    g = torch.Generator().manual_seed(2)
    G = torch.randn(1, 140, 140, generator=g)
    A = (G + G.mT) * 1e19
    w, V = jacobi.block_eigh(A, eigh32)
    assert torch.isfinite(w).all() and torch.isfinite(V).all()
    assert rel(w, torch.linalg.eigvalsh(A.double())) < 1e-5


def test_padding_never_mixes_with_the_spectrum():
    """n = 33 pads to 64 with a value past the spectrum: a zero eigenvalue (rank-deficient input) stays the input's."""
    g = torch.Generator().manual_seed(5)
    B = torch.randn(1, 33, 20, generator=g)
    A = B @ B.mT  # rank 20: 13 zero eigenvalues
    w, V = jacobi.block_eigh(A, eigh32, b=16)
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


def test_jacobi_big_replica_matches_the_contract():
    """The kernel's float32 replica against its float64 contract (tpc.jacobi_big_reference): the same rows to float32
    rounding after the sweeps."""
    import kernel_replica as kr
    from hpu_ops import tpc

    g = torch.Generator().manual_seed(3)
    G = torch.randn(1, 40, 40, generator=g)
    X = (G + G.mT) / 2
    S = X / X.abs().sum(-1).amax(-1)[:, None, None] + tpc.BIG_SHIFT * torch.eye(40)
    S0 = torch.nn.functional.pad(S, (0, 24)).contiguous()
    got = kr.jacobi_big_op()(S0)
    want = tpc.jacobi_big_reference(S0)
    norm = lambda U: U[..., :40] / U[..., :40].norm(dim=-1, keepdim=True)  # noqa: E731
    assert (norm(got).abs() - norm(want).abs()).abs().max() < 1e-4  # rows to sign (the rotation conventions differ)


@pytest.mark.parametrize("n", [129, 255])
@pytest.mark.parametrize("scale", [1e-10, 1e-19])
def test_block_eigh_small_scale_padded(n, scale):
    """Padded n > 128 with small ||A|| must not add +1 to the padding sentinel or sweep-1 off/norm exits early."""
    g = torch.Generator().manual_seed(n)
    Q = torch.linalg.qr(torch.randn(1, n, n, generator=g, dtype=torch.float64))[0]
    A = (scale * (Q * torch.logspace(0, -3, n, dtype=torch.float64)) @ Q.mT).float()
    A = (A + A.mT) / 2
    w, V = jacobi.block_eigh(A, eigh32)
    w64 = torch.linalg.eigvalsh(A.double())
    res = (A.double() @ V.double() - V.double() * w.double()[:, None, :]).abs().max() / A.double().abs().max()
    assert rel(w, w64) < 1e-5
    assert float(res) < 1e-5
    assert (V.mT @ V - torch.eye(n)).abs().max() < 1e-5


def test_block_eigh_zero_and_huge_padded_stay_finite():
    """Zero padded input falls back to pad=1.0; huge input clamps 2*norm to finfo.max without inf/NaN."""
    n = 129
    A0 = torch.zeros(1, n, n)
    w0, V0 = jacobi.block_eigh(A0, eigh32)
    assert torch.equal(w0, torch.zeros(1, n))
    assert (V0.mT @ V0 - torch.eye(n)).abs().max() < 1e-5
    A_huge = 1e37 * torch.eye(n)[None]
    wh, Vh = jacobi.block_eigh(A_huge, eigh32)
    assert torch.isfinite(wh).all() and torch.isfinite(Vh).all()
    assert rel(wh, A_huge.double().diagonal(dim1=-2, dim2=-1)) < 1e-5
