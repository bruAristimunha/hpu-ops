"""CPU tests of the complex64 Householder QR replica (``kernel_replica.cqr_kernel``)
and ``hpu_ops.tpc.complex_linalg.qr`` API contracts.
"""
import math

import kernel_replica as kr
import numpy as np
import pytest
import torch

from hpu_ops.tpc import complex_linalg as cl

C64, C128 = torch.complex64, torch.complex128


def crandn(*shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.complex(torch.randn(*shape, generator=g), torch.randn(*shape, generator=g)).to(C64)


def pairs(Z):
    return torch.view_as_real(Z.to(C64)).contiguous()


def cplx(P):
    return torch.view_as_complex(P.double().contiguous())


def _qr_errs(A, Q, R):
    """Relative reconstruction ``||Q R - A||_F / ||A||_F`` and orthogonality ``||Q^H Q - I||_F``."""
    A128 = A.to(C128)
    Q128 = Q.to(C128)
    R128 = R.to(C128)
    k = Q128.shape[-1]
    eye = torch.eye(k, dtype=C128)
    nrm = A128.abs().pow(2).sum((-2, -1)).sqrt()
    rec_abs = (Q128 @ R128 - A128).abs().pow(2).sum((-2, -1)).sqrt()
    rec = torch.where(nrm > 0, rec_abs / nrm.clamp_min(1e-300), rec_abs)
    orth = (Q128.mH @ Q128 - eye).abs().pow(2).sum((-2, -1)).sqrt()
    return float(rec.max()), float(orth.max())


@pytest.fixture
def cl_qr_replica(monkeypatch):
    monkeypatch.setenv("PT_HPU_LAZY_MODE", "0")
    monkeypatch.setattr(cl, "_cqr_op", kr.cqr_op)
    monkeypatch.setattr(cl, "_cqr_complete_op", kr.cqr_complete_op)
    monkeypatch.setattr(cl, "_cqr_r_op", kr.cqr_r_op)


def _habana_qr_nan_cases():
    g = torch.Generator().manual_seed(42)

    def cr(*s):
        return torch.complex(torch.randn(*s, generator=g), torch.randn(*s, generator=g)).to(C64)

    return [
        ("eye(2)", torch.eye(2, dtype=C64).expand(3, 2, 2).contiguous()),
        ("eye(8)", torch.eye(8, dtype=C64).expand(3, 8, 8).contiguous()),
        ("diag(2, 3)", torch.diag(torch.tensor([2.0, 3.0], dtype=C64)).expand(3, 2, 2).contiguous()),
        ("upper triangular 6x6", cr(3, 6, 6).triu() + 3 * torch.eye(6, dtype=C64)),
        (
            "first column 2 e_0",
            torch.cat(
                [torch.tensor([[[2.0], [0.0], [0.0], [0.0]]], dtype=C64).expand(3, 4, 1), cr(3, 4, 3)],
                dim=-1,
            ),
        ),
        ("ones(4, 4) (rank 1)", torch.ones(3, 4, 4, dtype=C64)),
        ("randn(6, 6) (control)", cr(3, 6, 6)),
        (
            "complex diag(2, -3, 4j, 0)",
            torch.diag(torch.tensor([2.0 + 0j, -3.0 + 0j, 4.0j, 0.0 + 0j], dtype=C64)).expand(2, 4, 4).contiguous(),
        ),
        ("complex rank-1 (1+2j)*ones", torch.full((2, 5, 4), 1.0 + 2.0j, dtype=C64)),
        (
            "complex outer rank-1",
            (cr(2, 6, 1) @ cr(2, 1, 5)).to(C64),
        ),
        (
            "complex upper tri + real diag",
            cr(2, 6, 6).triu(1) + torch.diag(torch.tensor([1.0, -2.0, 3.0, -4.0, 5.0, -6.0], dtype=C64)),
        ),
        ("1x1 complex 3+4j", torch.tensor([[[3.0 + 4.0j]]], dtype=C64)),
        ("1x1 negative real -5", torch.tensor([[[-5.0 + 0.0j]]], dtype=C64)),
    ]


@pytest.mark.parametrize("name,A", _habana_qr_nan_cases(), ids=lambda x: x if isinstance(x, str) else None)
@pytest.mark.parametrize("mode", ("reduced", "complete", "r"))
def test_habana_qr_nan_and_complex_variants(cl_qr_replica, name, A, mode):
    m, n = A.shape[-2], A.shape[-1]
    Qp, Rp = cl.qr(A, mode=mode)
    R = cplx(Rp)
    assert torch.isfinite(R).all()
    assert torch.equal(R.tril(-1), torch.zeros_like(R))
    k_diag = min(m, n)
    assert torch.equal(R[..., :k_diag, :n].diagonal(dim1=-2, dim2=-1).imag, torch.zeros_like(R[..., :k_diag, :n].diagonal(dim1=-2, dim2=-1).imag))

    if mode == "r":
        assert Qp.shape == (0,)
        assert Qp.dtype == torch.float32
        _, Rp_red = cl.qr(A, mode="reduced")
        assert torch.equal(Rp, Rp_red)
        return

    Q = cplx(Qp)
    assert torch.isfinite(Q).all()
    k_q = m if mode == "complete" else n
    assert Q.shape == (*A.shape[:-2], m, k_q)
    assert R.shape == (*A.shape[:-2], k_q, n)

    rec, orth = _qr_errs(A, Q, R)
    Q32, R32 = torch.linalg.qr(A, mode=mode)
    rec32, orth32 = _qr_errs(A, Q32, R32)
    assert rec <= max(1e-6, 10.0 * rec32)
    assert orth <= max(1e-6, 10.0 * orth32)

    if mode == "complete":
        eye_m = torch.eye(m, dtype=C128)
        orth_left = float((Q.to(C128) @ Q.to(C128).mH - eye_m).abs().pow(2).sum((-2, -1)).sqrt().max())
        assert orth_left <= max(1e-6, 10.0 * orth32)
        assert torch.equal(Rp[..., n:, :, :], torch.zeros_like(Rp[..., n:, :, :]))
        _, Rp_red = cl.qr(A, mode="reduced")
        assert torch.equal(Rp[..., :n, :, :], Rp_red)


@pytest.mark.parametrize("n", [1, 2, 3, 4, 8, 16, 22, 32])
@pytest.mark.parametrize("mode", ("reduced", "complete", "r"))
def test_square_sizes(cl_qr_replica, n, mode):
    A = crandn(3, n, n, seed=100 + n)
    Qp, Rp = cl.qr(A, mode=mode)
    R = cplx(Rp)
    assert torch.equal(R.tril(-1), torch.zeros_like(R))
    assert torch.equal(R.diagonal(dim1=-2, dim2=-1).imag, torch.zeros_like(R.diagonal(dim1=-2, dim2=-1).imag))
    if mode == "r":
        assert Qp.shape == (0,)
        return
    Q = cplx(Qp)
    rec, orth = _qr_errs(A, Q, R)
    Q32, R32 = torch.linalg.qr(A, mode=mode)
    rec32, orth32 = _qr_errs(A, Q32, R32)
    assert rec <= max(1e-6, 10.0 * rec32)
    assert orth <= max(1e-6, 10.0 * orth32)


@pytest.mark.parametrize("m,n", [(2, 1), (4, 2), (8, 1), (8, 3), (16, 5), (32, 1), (32, 17), (32, 31)])
@pytest.mark.parametrize("mode", ("reduced", "complete", "r"))
def test_tall_sizes(cl_qr_replica, m, n, mode):
    A = crandn(2, m, n, seed=200 + 37 * m + n)
    Qp, Rp = cl.qr(A, mode=mode)
    R = cplx(Rp)
    assert torch.equal(R.tril(-1), torch.zeros_like(R))
    d = R[..., :n, :n].diagonal(dim1=-2, dim2=-1)
    assert torch.equal(d.imag, torch.zeros_like(d.imag))
    if mode == "r":
        assert Qp.shape == (0,)
        _, Rp_red = cl.qr(A, mode="reduced")
        assert torch.equal(Rp, Rp_red)
        return
    Q = cplx(Qp)
    rec, orth = _qr_errs(A, Q, R)
    Q32, R32 = torch.linalg.qr(A, mode=mode)
    rec32, orth32 = _qr_errs(A, Q32, R32)
    assert rec <= max(1e-6, 10.0 * rec32)
    assert orth <= max(1e-6, 10.0 * orth32)
    if mode == "complete":
        assert torch.equal(Rp[..., n:, :, :], torch.zeros_like(Rp[..., n:, :, :]))


@pytest.mark.parametrize("scale", [1e-35, 1e-30, 1e30, 3e37, 2.5e38])
def test_extreme_scales(cl_qr_replica, scale):
    # Use unitary-scaled matrix at 2.5e38 so ||col||_2 < FLT_MAX
    rng = np.random.default_rng(77)
    Q128, _ = np.linalg.qr(rng.normal(size=(6, 4)) + 1j * rng.normal(size=(6, 4)))
    A = torch.from_numpy((Q128 * scale).astype(np.complex64))[None]
    for mode in ("reduced", "complete"):
        Qp, Rp = cl.qr(A, mode=mode)
        Q, R = cplx(Qp), cplx(Rp)
        assert torch.isfinite(Q).all() and torch.isfinite(R).all()
        rec, orth = _qr_errs(A, Q, R)
        Q32, R32 = torch.linalg.qr(A, mode=mode)
        rec32, orth32 = _qr_errs(A, Q32, R32)
        assert rec <= max(1e-6, 10.0 * rec32)
        assert orth <= max(1e-6, 10.0 * orth32)


def test_zeros_and_subnormals(cl_qr_replica):
    Z = torch.zeros(2, 6, 4, dtype=C64)
    for mode in ("reduced", "complete"):
        Qp, Rp = cl.qr(Z, mode=mode)
        Q, R = cplx(Qp), cplx(Rp)
        k_q = 6 if mode == "complete" else 4
        assert torch.equal(Q, torch.eye(6, k_q, dtype=C128).expand(2, 6, k_q))
        assert torch.equal(R, torch.zeros(2, k_q, 4, dtype=C128))

    # Subnormals flush to zero on Gaudi2 / FTZ replica -> identical to zeros
    Sub = torch.full((2, 5, 3), 1e-40 + 1e-40j, dtype=C64)
    Qp, Rp = cl.qr(Sub, mode="reduced")
    assert torch.equal(cplx(Qp), torch.eye(5, 3, dtype=C128).expand(2, 5, 3))
    assert torch.equal(cplx(Rp), torch.zeros(2, 3, 3, dtype=C128))


@pytest.mark.parametrize("bad_val", [complex(math.nan, 0.0), complex(0.0, math.inf), complex(-math.inf, 1.0)])
@pytest.mark.parametrize("mode", ("reduced", "complete", "r"))
def test_nonfinite_per_batch_isolation(cl_qr_replica, bad_val, mode):
    A = crandn(3, 6, 4, seed=55)
    A[0, 2, 1] = bad_val
    A[2, 5, 3] = bad_val
    Qp, Rp = cl.qr(A, mode=mode)
    assert Rp[0].isnan().all()
    assert Rp[2].isnan().all()
    assert Rp[1].isfinite().all()
    if mode != "r":
        assert Qp[0].isnan().all()
        assert Qp[2].isnan().all()
        assert Qp[1].isfinite().all()
        rec, orth = _qr_errs(A[1:2], cplx(Qp[1:2]), cplx(Rp[1:2]))
        assert rec < 1e-5 and orth < 1e-5


def test_layouts_views_and_batches(cl_qr_replica):
    A = crandn(4, 7, 5, seed=88)
    P = pairs(A)
    Q_p, R_p = cl.qr(P, mode="reduced", layout="pairs")
    Q_c, R_c = cl.qr(A, mode="reduced", layout="pairs")
    assert torch.equal(Q_p, Q_c)
    assert torch.equal(R_p, R_c)

    # Interleaved layout (all 3 modes)
    Pi = P.reshape(4, 7, 10)
    Q_i, R_i = cl.qr(Pi, mode="reduced", layout="interleaved")
    assert torch.equal(Q_i, Q_p.reshape(4, 7, 10))
    assert torch.equal(R_i, R_p.reshape(4, 5, 10))

    Qc_i, Rc_i = cl.qr(Pi, mode="complete", layout="interleaved")
    Qc_p, Rc_p = cl.qr(P, mode="complete", layout="pairs")
    assert torch.equal(Qc_i, Qc_p.reshape(4, 7, 14))
    assert torch.equal(Rc_i, Rc_p.reshape(4, 7, 10))

    Qr_i, Rr_i = cl.qr(Pi, mode="r", layout="interleaved")
    assert Qr_i.shape == (0,)
    assert torch.equal(Rr_i, R_i)

    # Unbatched
    Q_u, R_u = cl.qr(P[1], mode="reduced")
    assert Q_u.shape == (7, 5, 2) and R_u.shape == (5, 5, 2)
    assert torch.equal(Q_u, Q_p[1]) and torch.equal(R_u, R_p[1])

    # Multidim batch
    Q_m, R_m = cl.qr(P.reshape(2, 2, 7, 5, 2), mode="complete")
    assert Q_m.shape == (2, 2, 7, 7, 2) and R_m.shape == (2, 2, 7, 5, 2)
    assert torch.equal(Q_m.reshape(4, 7, 7, 2), Qc_p)

    # Empty batch across all modes & layouts
    for mode in ("reduced", "complete", "r"):
        for layout in ("pairs", "interleaved"):
            inp = P[:0] if layout == "pairs" else Pi[:0]
            Qe, Re = cl.qr(inp, mode=mode, layout=layout)
            k_q = 7 if mode == "complete" else 5
            assert Re.shape == ((0, k_q, 10) if layout == "interleaved" else (0, k_q, 5, 2))
            if mode == "r":
                assert Qe.shape == (0,)
            else:
                assert Qe.shape == ((0, 7, 2 * k_q) if layout == "interleaved" else (0, 7, k_q, 2))

    # Strided batch & offset view & conj/mT
    assert torch.equal(cl.qr(P[::2])[0], cl.qr(P[::2].contiguous())[0])
    assert torch.equal(cl.qr(P[2:])[1], cl.qr(P[2:].contiguous())[1])
    assert torch.equal(cl.qr(A.conj())[0], cl.qr(A.conj().resolve_conj())[0])
    A_sq = crandn(3, 5, 5, seed=89)
    assert torch.equal(cl.qr(A_sq.mT)[0], cl.qr(A_sq.mT.contiguous())[0])


def test_invalid_inputs_raise(cl_qr_replica):
    A = pairs(crandn(2, 4, 3, seed=91))
    with pytest.raises(RuntimeError, match="unrecognized mode"):
        cl.qr(A, mode="bad")
    with pytest.raises(RuntimeError, match="unrecognized mode"):
        cl.qr(A, mode=123)

    # 0D and 1D complex64 must raise ValueError (not IndexError)
    with pytest.raises(ValueError):
        cl.qr(torch.zeros((), dtype=C64))
    with pytest.raises(ValueError):
        cl.qr(torch.zeros((4,), dtype=C64))
    with pytest.raises(ValueError):
        cl.qr(torch.zeros((3, 2), dtype=torch.float32))

    # Wide m < n
    with pytest.raises(ValueError):
        cl.qr(torch.zeros((2, 3, 5, 2), dtype=torch.float32))
    # m > COMPLEX_MAXN
    with pytest.raises(ValueError):
        cl.qr(torch.zeros((1, 33, 4, 2), dtype=torch.float32))
    # n == 0
    with pytest.raises(ValueError):
        cl.qr(torch.zeros((1, 4, 0, 2), dtype=torch.float32))
    # Wrong dtypes and layout
    with pytest.raises(ValueError):
        cl.qr(torch.zeros((2, 4, 3), dtype=C128))
    with pytest.raises(ValueError):
        cl.qr(torch.zeros((2, 4, 3, 2), dtype=torch.float64))
    with pytest.raises(ValueError):
        cl.qr(A, layout="other")
