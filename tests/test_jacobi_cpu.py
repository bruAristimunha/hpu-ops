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


def test_jacobi_svd_big_and_hh_qr_replicas():
    """Float32 FTZ-faithful replicas of ``jacobi_svd_big``, ``jacobi_svdvals_big``, ``hh_qr``, ``tsqr_*``, and
    ``block_qr`` preserve singular values, orthogonality, and reconstruction across extreme scales and zero/rank-def."""
    import numpy as np
    import kernel_replica as kr

    g = torch.Generator().manual_seed(7)
    graded = torch.diag(torch.tensor([10.0 ** (-30 * i / 39) for i in range(40)], dtype=torch.float32))
    cases = [
        ("graded_1e-30_40", graded, False),
        ("tiny_1e-25_40", torch.randn(40, 40, generator=g) * 1e-25, False),
        ("huge_1e25_40", torch.randn(40, 40, generator=g) * 1e25, False),
        ("zeros_40x40", torch.zeros(40, 40), True),
        ("rank5_64x40", torch.randn(64, 5, generator=g) @ torch.randn(5, 40, generator=g), True),
        ("wide_33x64", torch.randn(33, 64, generator=g), True),
    ]
    for name, a, full in cases:
        U, S, Vh = kr.jacobi_svd_big(a.numpy(), compute_uv=True, full_matrices=full)
        ref_s = torch.linalg.svdvals(a.double()).numpy()
        scale = float(a.abs().max()) or 1.0
        k = S.shape[0]
        assert np.abs((U[:, :k] * S) @ Vh[:k] - a.numpy()).max() / scale < 1e-5, name
        assert np.abs(U.T @ U - np.eye(U.shape[1])).max() < 1e-5, name
        assert np.abs(Vh @ Vh.T - np.eye(Vh.shape[0])).max() < 1e-5, name
        assert np.abs(S - ref_s).max() / scale < 1e-5, name
        _, S_only, _ = kr.jacobi_svd_big(a.numpy(), compute_uv=False)
        assert np.allclose(S_only, S, rtol=1e-6, atol=0.0), name

    for name, a, fn in (
        ("hh_qr_128x40", torch.randn(128, 40, generator=g), lambda x: kr.hh_qr_kernel(x, "qr")),
        ("hh_qr_zeros", torch.zeros(40, 40), lambda x: kr.hh_qr_kernel(x, "qr")),
        ("tsqr_300x40", torch.randn(300, 40, generator=g), kr.tsqr_qr_replica),
        ("block_qr_96x96", torch.randn(96, 96, generator=g), kr.block_qr_replica),
    ):
        Q, R = fn(a.numpy())
        scale = float(a.abs().max()) or 1.0
        assert np.abs(Q @ R - a.numpy()).max() / scale < 1e-5, name
        assert np.abs(Q.T @ Q - np.eye(Q.shape[1])).max() < 1e-5, name
        assert np.abs(np.tril(R, -1)).max() / scale < 1e-5, name
    r_tsqr = kr.tsqr_r_replica(torch.randn(300, 40, generator=g).numpy())
    assert np.abs(np.tril(r_tsqr, -1)).max() < 1e-5


def _svd_metrics(U, S, Vh, A):
    k = S.shape[-1]
    A64 = A.double()
    S_ref = torch.linalg.svdvals(A64)
    scale = float(A64.abs().max()) or 1.0
    Uk, Vhk = U.double()[..., :k], Vh.double()[..., :k, :]
    rec = float(((Uk * S.double().unsqueeze(-2)) @ Vhk - A64).abs().max() / scale)
    eye_u = torch.eye(U.shape[-1], dtype=torch.float64).expand(*U.shape[:-2], U.shape[-1], U.shape[-1])
    eye_v = torch.eye(Vh.shape[-2], dtype=torch.float64).expand(*Vh.shape[:-2], Vh.shape[-2], Vh.shape[-2])
    orth_u = float((U.double().mT @ U.double() - eye_u).abs().max())
    orth_v = float((Vh.double() @ Vh.double().mT - eye_v).abs().max())
    s_err = float((S.double() - S_ref).abs().max() / (float(S_ref.max()) or 1.0))
    return s_err, rec, orth_u, orth_v


@pytest.mark.parametrize("full_matrices", [False, True])
@pytest.mark.parametrize("shape", [
    (2, 33, 33), (2, 40, 20), (2, 20, 40), (2, 64, 64),
    (2, 72, 64), (2, 40, 96), (2, 128, 128), (2, 3, 64, 48),
])
def test_block_svd_gram_free_shapes_and_full_matrices(shape, full_matrices):
    g = torch.Generator().manual_seed(shape[-2] * 100 + shape[-1])
    A = torch.randn(*shape, generator=g, dtype=torch.float32)
    U, S, Vh = jacobi.block_svd_gram_free(A, compute_uv=True, full_matrices=full_matrices)
    m, n = shape[-2], shape[-1]
    k = min(m, n)
    assert U.shape == (*shape[:-2], m, m if full_matrices else k)
    assert S.shape == (*shape[:-2], k)
    assert Vh.shape == (*shape[:-2], n if full_matrices else k, n)
    s_err, rec, orth_u, orth_v = _svd_metrics(U, S, Vh, A)
    assert s_err < 1e-5 and rec < 1e-5 and orth_u < 1e-5 and orth_v < 1e-5
    _, S_only, _ = jacobi.block_svd_gram_free(A, compute_uv=False)
    torch.testing.assert_close(S_only, S, rtol=1e-5, atol=1e-6)


def test_block_svd_gram_free_ill_conditioned_and_extreme_scales():
    g = torch.Generator().manual_seed(42)
    # Ill-conditioned 40x20 (cond=1e4), 128x128 (cond=1e5, 1e6), 256x256 (cond=1e5)
    for m, n, lo in ((40, 20, -4), (128, 128, -5), (128, 128, -6), (256, 256, -5)):
        Q1 = torch.linalg.qr(torch.randn(1, m, m, generator=g, dtype=torch.float64))[0]
        Q2 = torch.linalg.qr(torch.randn(1, n, n, generator=g, dtype=torch.float64))[0]
        k = min(m, n)
        spec = torch.logspace(0, lo, k, dtype=torch.float64)
        A = ((Q1[..., :k] * spec) @ Q2[..., :k, :]).float()
        U, S, Vh = jacobi.block_svd_gram_free(A, compute_uv=True, full_matrices=False)
        S_ref = torch.linalg.svdvals(A.double())
        S_cpu32 = torch.linalg.svdvals(A)
        rel_s = float((S.double() - S_ref).abs().max() / S_ref.max())
        rel_cpu32 = float((S_cpu32.double() - S_ref).abs().max() / S_ref.max())
        rel_smin = float((S[0, -1].double() - S_ref[0, -1]).abs() / S_ref[0, -1])
        _, rec, orth_u, orth_v = _svd_metrics(U, S, Vh, A)
        assert rel_s <= max(1e-6, 10 * rel_cpu32), (m, n, lo, rel_s, rel_cpu32)
        assert rel_smin < 1e-2, (m, n, lo, rel_smin)  # Gram-free preserves sigma_min without squaring kappa
        assert rec < 1e-5 and orth_u < 1e-5 and orth_v < 1e-5

    # Extreme scales, zeros, rank-deficient
    for name, A in (
        ("tiny_1e-25", torch.randn(2, 65, 65, generator=g) * 1e-25),
        ("huge_1e25", torch.randn(2, 65, 65, generator=g) * 1e25),
        ("zeros_64", torch.zeros(2, 64, 64)),
        ("rank10_128", torch.randn(2, 128, 10, generator=g) @ torch.randn(2, 10, 128, generator=g)),
    ):
        U, S, Vh = jacobi.block_svd_gram_free(A, compute_uv=True, full_matrices=True)
        s_err, rec, orth_u, orth_v = _svd_metrics(U, S, Vh, A)
        assert s_err < 1e-5 and rec < 1e-5 and orth_u < 1e-5 and orth_v < 1e-5, name


def test_block_svd_gram_free_uses_subsolver_hooks():
    """Verify sub-solver kwargs (``svd_big``, ``svdvals_big``, ``tsqr_r``, ``tsqr_qr``, ``block_qr``) are invoked."""
    seen = []
    g = torch.Generator().manual_seed(9)

    def h_svd_big(X, full_matrices=False, sort=True):
        seen.append(("svd_big", tuple(X.shape)))
        return torch.linalg.svd(X, full_matrices=full_matrices)

    def h_svdvals_big(X, sort=True):
        seen.append(("svdvals_big", tuple(X.shape)))
        return torch.linalg.svdvals(X)

    def h_tsqr_r(X):
        seen.append(("tsqr_r", tuple(X.shape)))
        return torch.linalg.qr(X, mode="r").R

    def h_tsqr_qr(X):
        seen.append(("tsqr_qr", tuple(X.shape)))
        return torch.linalg.qr(X, mode="reduced")

    def h_block_qr(X):
        seen.append(("block_qr", tuple(X.shape)))
        return torch.linalg.qr(X, mode="reduced")

    kw = dict(svd_big=h_svd_big, svdvals_big=h_svdvals_big, tsqr_r=h_tsqr_r, tsqr_qr=h_tsqr_qr, block_qr=h_block_qr)
    # Tier 1: <= 64 cols
    jacobi.block_svd_gram_free(torch.randn(1, 160, 40, generator=g), compute_uv=True, **kw)
    assert any(k == "tsqr_qr" for k, _ in seen) and any(k == "svd_big" for k, _ in seen)
    seen.clear()
    jacobi.block_svd_gram_free(torch.randn(1, 60, 60, generator=g), compute_uv=False, **kw)
    assert any(k == "svdvals_big" for k, _ in seen)
    seen.clear()
    # Tier 2: > 64 cols (block Jacobi using panel tsqr_r + svd_big + block_qr cleanup)
    jacobi.block_svd_gram_free(torch.randn(1, 96, 96, generator=g), compute_uv=True, **kw)
    assert any(k == "tsqr_r" for k, _ in seen) and any(k == "svd_big" for k, _ in seen)
