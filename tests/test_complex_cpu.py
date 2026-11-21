"""CPU tests of the complex64 TPC kernels (``hpu_ops.tpc.complex_linalg``) on their float32 replicas
(``kernel_replica.cgj_solve``, ...): the kernel op is replaced by the replica, so the layout conversions, chunking and
the slogdet/det glue run their real code. Accuracy is judged against torch's own complex64 CPU result: both against
complex128, the kernel's error within max(1e-6, 10x torch's)."""
import math

import numpy as np
import pytest
import torch

import kernel_replica as kr
from hpu_ops.tpc import complex_linalg as cl

C64, C128 = torch.complex64, torch.complex128


@pytest.fixture(autouse=True)
def replica(monkeypatch):
    monkeypatch.setenv("PT_HPU_LAZY_MODE", "0")
    monkeypatch.setattr(cl, "_cgj_op", lambda: kr.cgj_op())
    monkeypatch.setattr(cl, "_cgj_inv_op", lambda: kr.cgj_op(identity=True))


def crandn(*shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.complex(torch.randn(*shape, generator=g), torch.randn(*shape, generator=g)).to(C64)


def cond_matrix(b, n, cond, seed=0):
    """``U diag(logspace(0, -log10 cond)) V^H`` with Haar-ish unitary U, V."""
    U, _ = torch.linalg.qr(crandn(b, n, n, seed=seed).to(C128))
    V, _ = torch.linalg.qr(crandn(b, n, n, seed=seed + 1).to(C128))
    s = torch.logspace(0, -math.log10(cond), n, dtype=torch.float64).to(C128)
    return (U * s[None, None, :]) @ V.mH


def pairs(Z):
    return torch.view_as_real(Z.to(C64)).contiguous()


def cplx(P):
    return torch.view_as_complex(P.double().contiguous())


def err(x, ref):
    x, ref = x.to(C128), ref.to(C128)
    return float((x - ref).abs().max() / ref.abs().max().clamp_min(1e-300))


def within(ours, torch32, ref):
    e, t = err(ours, ref), err(torch32, ref)
    assert e <= max(1e-6, 10 * t), (e, t)
    return e, t


def cases():
    yield "randn", crandn(4, 16, 16)
    yield "cond1e6", cond_matrix(3, 16, 1e6).to(C64)
    yield "i_I_plus_small", (1j * torch.eye(8) + 1e-3 * crandn(8, 8, seed=3)).to(C64)[None]
    yield "phases", torch.diag_embed(torch.exp(1j * torch.linspace(0.1, 3.0, 12, dtype=torch.float64))).to(C64)[None]
    P = torch.eye(5)[[1, 0, 2, 3, 4]]  # an odd permutation: det(i P) = -i^5 = -i
    yield "i_odd_perm", (1j * P).to(C64)[None]
    yield "hpd", (lambda X: X @ X.mH / 22 + torch.eye(22))(crandn(2, 22, 22, seed=5))


@pytest.mark.parametrize("n", [1, 2, 3, 16, 22, 32])
def test_solve_inv_slogdet_random(n):
    A = crandn(3, n, n, seed=n)
    B = crandn(3, n, 4, seed=100 + n)
    X = cplx(cl.solve(pairs(A), pairs(B)))
    within(X, torch.linalg.solve(A, B), torch.linalg.solve(A.to(C128), B.to(C128)))
    within(cplx(cl.inv(pairs(A))), torch.linalg.inv(A), torch.linalg.inv(A.to(C128)))
    sign, lad = cl.slogdet(pairs(A))
    s32, l32 = torch.linalg.slogdet(A)
    s128, l128 = torch.linalg.slogdet(A.to(C128))
    within(cplx(sign), s32, s128)
    assert float((lad.double() - l128).abs().max()) <= max(1e-6, 10 * float((l32.double() - l128).abs().max()))
    within(cplx(cl.det(pairs(A))), torch.linalg.det(A), torch.linalg.det(A.to(C128)))


@pytest.mark.parametrize("name,A", list(cases()))
def test_structured(name, A):
    A128 = A.to(C128)
    within(cplx(cl.inv(pairs(A))), torch.linalg.inv(A), torch.linalg.inv(A128))
    sign, lad = cl.slogdet(pairs(A))
    s32, l32 = torch.linalg.slogdet(A)
    s128, l128 = torch.linalg.slogdet(A128)
    within(cplx(sign), s32, s128)
    assert float((lad.double() - l128).abs().max()) <= max(1e-6, 10 * float((l32.double() - l128).abs().max()))
    if name == "i_odd_perm":
        torch.testing.assert_close(cplx(sign), torch.tensor([-1j], dtype=C128), atol=1e-7, rtol=0)


def test_singular_and_rank_deficient():
    A = crandn(2, 6, 6, seed=7)
    A[0, :, 2] = 0  # zero column: an exactly zero pivot
    A[1, 4] = 0  # zero row
    X, D, S = cl.cgj_inv(pairs(A))
    sign, lad = cl.slogdet(pairs(A))
    assert torch.all(sign == 0) and torch.all(lad == -math.inf)
    assert torch.all(S[:, 2] > 0) and torch.isfinite(X).all()
    s32, l32 = torch.linalg.slogdet(A)
    assert torch.all(s32 == 0) and torch.all(l32 == -math.inf)
    # duplicate row, and rank n - 1 in exact arithmetic: not exactly singular in float32 (LAPACK's cgetrf leaves a
    # residue too); |det| tiny against ||A||^n, as torch's
    D2 = crandn(1, 6, 6, seed=10)
    D2[0, 3] = D2[0, 1]
    Y = crandn(1, 16, 15, seed=8) @ crandn(1, 15, 16, seed=9)
    for M in (D2, Y):
        _, lad = cl.slogdet(pairs(M))
        _, l32 = torch.linalg.slogdet(M)
        scale = M.shape[-1] * math.log(float(M.abs().max()))
        assert float(lad) - scale < -8 and float(l32) - scale < -8


@pytest.mark.parametrize("bad", [math.nan, math.inf])
def test_nonfinite(bad):
    A = crandn(2, 5, 5, seed=11)
    A[0, 2, 3] = complex(bad, 0)
    X, D, S = cl.cgj_inv(pairs(A))
    assert torch.isnan(X[0]).all() and torch.isnan(D[0]).all() and torch.isnan(S[0]).all()
    assert torch.isfinite(X[1]).all()
    assert not torch.isfinite(torch.linalg.inv(A)[0]).all()  # torch is not finite there either
    sign, lad = cl.slogdet(pairs(A))
    assert torch.isnan(sign[0]).all() and torch.isnan(lad[0])


def test_layouts_broadcast_chunks():
    A = crandn(2, 1, 7, 7, seed=12)
    B = crandn(3, 7, 70, seed=13)  # 70 columns: three kernel chunks; batch broadcast (2, 1) x (3,)
    ref = torch.linalg.solve(A.to(C128), B.to(C128))
    X = cl.solve(pairs(A), pairs(B))
    assert X.shape == (2, 3, 7, 70, 2)
    assert err(cplx(X), ref) < 1e-5
    Xi = cl.solve(pairs(A).reshape(2, 1, 7, 14), pairs(B).reshape(3, 7, 140), layout="interleaved")
    assert Xi.shape == (2, 3, 7, 140) and torch.equal(Xi, X.reshape(2, 3, 7, 140))
    assert torch.equal(cl.solve(A, B), X)  # complex64 input, viewed as pairs
    with pytest.raises(ValueError):
        cl.inv(pairs(crandn(33, 33)))
    with pytest.raises(ValueError):
        cl.inv(pairs(A).double())


def test_factors_match_lapack_contract():
    """Same pivots as LAPACK's cgetrf (cabs1, first max) on a generic matrix: D and the permutation sign."""
    A = crandn(3, 10, 10, seed=14)
    _, D, S = cl.cgj_inv(pairs(A))
    _, Dref, Sref = cl.cgj_reference(pairs(A))
    torch.testing.assert_close(D.double(), Dref, rtol=1e-5, atol=1e-5)
    assert torch.equal(S[:, 3].double(), Sref[:, 3])
    torch.testing.assert_close(S[:, :2].double(), Sref[:, :2], rtol=0, atol=1e-5)


def test_replica_ftz_and_huge():
    """Pivots near the float32 limits: 2^-64 prescale of huge pivots, scaled reciprocal of tiny ones."""
    for s in (1e-30, 1e30, 3e37):
        A = crandn(1, 6, 6, seed=15) * s
        B = crandn(1, 6, 2, seed=16) * s  # A^-1 itself is subnormal at 3e37, which the card flushes
        X = cplx(cl.solve(pairs(A), pairs(B)))
        assert err(X, torch.linalg.solve(A.to(C128), B.to(C128))) < 1e-4
        if s < 1e37:
            assert err(cplx(cl.inv(pairs(A))), torch.linalg.inv(A.to(C128))) < 1e-4
        _, lad = cl.slogdet(pairs(A))
        _, l128 = torch.linalg.slogdet(A.to(C128))
        assert abs(float(lad) - float(l128)) < 1e-4


@pytest.mark.parametrize("n", [1, 2, 5, 16, 31, 32])
def test_exact_identity(n):
    """slogdet(I) = (1, 0) and det(I) = 1 exactly (a biased rsqrt lost an ulp per pivot: 0.9999981 at n = 32)."""
    P = pairs(torch.eye(n, dtype=C64)[None].expand(2, n, n))
    sign, lad = cl.slogdet(P)
    assert sign.tolist() == [[1.0, 0.0]] * 2 and lad.tolist() == [0.0] * 2
    assert cl.det(P).tolist() == [[1.0, 0.0]] * 2
    X, D, S = cl.cgj_inv(P)
    assert torch.equal(X, P) and torch.equal(D[:, 2], torch.ones(2, n)) and S[:, :2].tolist() == [[1.0, 0.0]] * 2


@pytest.mark.parametrize("perm", [[0], [1, 0], [1, 0, 2, 3, 4], [2, 0, 1, 5, 3, 4], list(range(31, -1, -1))])
@pytest.mark.parametrize("unit", [1, 1j, -1j, -1])
def test_exact_unit_times_permutation(perm, unit):
    """det(u P) = sgn(P) u^n exactly, |sign| = 1 exactly, logabsdet = 0 exactly."""
    Z = (unit * torch.eye(len(perm))[perm]).to(C64)[None]
    ref = torch.view_as_real(torch.linalg.det(Z.to(C128))).float()
    sign, lad = cl.slogdet(pairs(Z))
    assert torch.equal(sign, ref) and lad.tolist() == [0.0]
    assert float(sign.square().sum()) == 1.0
    assert torch.equal(cl.det(pairs(Z)), ref)


def test_exact_small_integer():
    """Small-integer matrices whose elimination is exact in float32: det exact, and equal to torch's complex64 det."""
    M = torch.tensor([[[2, 1, 0], [1, 1, 0], [0, 0, 1]], [[2, 0, 0], [0, 3, 0], [0, 0, -1]],
                      [[1, 2, 3], [0, 4, 5], [0, 0, 6]], [[0, 0, 2], [0, 3, 1], [4, 1, 1]]], dtype=C64)
    M = torch.cat([M, 1j * M[:2], torch.diag_embed(torch.tensor([[2j, -3, 1j, 5]], dtype=C64))[:, :3, :3]])
    exact = torch.view_as_real(torch.linalg.det(M.to(C128))).round().float()
    d = cl.det(pairs(M))
    assert torch.equal(d, exact), (d, exact)
    assert torch.equal(d, torch.view_as_real(torch.linalg.det(M)))
    sign, lad = cl.slogdet(pairs(M))
    assert torch.equal(sign.square().sum(-1), torch.ones(len(M)))
    assert torch.equal(lad, torch.linalg.slogdet(M.to(C128))[1].float())


def test_sign_unit_modulus_random():
    sign, _ = cl.slogdet(pairs(crandn(64, 12, 12, seed=40)))
    assert float((sign.double().square().sum(-1).sqrt() - 1).abs().max()) <= 2 ** -23

