"""Adversarial CPU tests of ``hpu_ops.tpc.complex_linalg`` on the kernels' replicas: unbatched and broadcast inputs,
views (transposed, strided, conjugate), empty batches, m = 0 right-hand sides, row and column vectors for the SVD,
the size and dtype limits."""
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
    monkeypatch.setattr(cl, "_ceigh_op", kr.ceigh_op)
    monkeypatch.setattr(cl, "_ceigh_refine_op", kr.ceigh_refine_op)
    monkeypatch.setattr(cl, "_csvd_op", kr.csvd_op)
    monkeypatch.setattr(cl, "_csvdvals_op", kr.csvdvals_op)


def crandn(*shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.complex(torch.randn(*shape, generator=g), torch.randn(*shape, generator=g)).to(C64)


def cplx(P):
    return torch.view_as_complex(P.double().contiguous())


def test_views_and_unbatched():
    A = crandn(4, 6, 6, seed=1)
    P = torch.view_as_real(A)
    ref = torch.linalg.inv(A.to(C128))
    assert (cplx(cl.inv(P[0])) - ref[0]).abs().max() < 1e-5  # unbatched
    assert (cplx(cl.inv(torch.view_as_real(A.mT))) - ref.mT).abs().max() < 1e-5  # transposed view
    assert (cplx(cl.inv(P[::2])) - ref[::2]).abs().max() < 1e-5  # strided batch
    s, _ = cl.slogdet(A.conj())  # conjugate view of a complex input
    s_ref, _ = torch.linalg.slogdet(A.conj().to(C128))
    assert (cplx(s) - s_ref).abs().max() < 1e-5
    H = A + A.mH
    w, V = cl.eigh(torch.view_as_real(H)[1])
    assert w.shape == (6,) and V.shape == (6, 6, 2)
    U, S, Vh = cl.svd(torch.view_as_real(A)[2])
    assert ((cplx(U) * S.double()) @ cplx(Vh) - A[2].to(C128)).abs().max() < 1e-5


def test_empty_batches_and_rhs():
    P = torch.view_as_real(crandn(4, 6, 6, seed=2))
    assert cl.inv(P[:0]).shape == (0, 6, 6, 2)
    sign, lad = cl.slogdet(P[:0])
    assert sign.shape == (0, 2) and lad.shape == (0,)
    w, V = cl.eigh(P[:0])
    assert w.shape == (0, 6) and V.shape == (0, 6, 6, 2)
    assert cl.eigvalsh(P[:0]).shape == (0, 6)
    U, S, Vh = cl.svd(P[:0, :, :4])
    assert U.shape == (0, 6, 4, 2) and S.shape == (0, 4) and Vh.shape == (0, 4, 4, 2)
    assert cl.svdvals(P[:0]).shape == (0, 6)
    X, D, S = cl.cgj_solve(P, P[..., :0, :])
    assert X.shape == (4, 6, 0, 2) and D.shape == (4, 3, 6) and S.shape == (4, 4)


@pytest.mark.parametrize("shape", [(2, 1, 20), (2, 20, 1), (1, 32, 3)])
def test_svd_vectors_and_thin(shape):
    A = crandn(*shape, seed=3)
    U, S, Vh = cl.svd(torch.view_as_real(A))
    torch.testing.assert_close(S.double(), torch.linalg.svdvals(A.to(C128)), rtol=1e-5, atol=0)
    assert ((cplx(U) * S.double()[..., None, :]) @ cplx(Vh) - A.to(C128)).abs().max() < 1e-5


def test_limits():
    with pytest.raises(ValueError):
        cl.inv(torch.zeros(2, 0, 0, 2))
    with pytest.raises(ValueError):
        cl.eigh(torch.zeros(1, 33, 33, 2))
    with pytest.raises(ValueError):
        cl.inv(torch.zeros(2, 3, 3, 2, dtype=torch.float64))
    with pytest.raises(ValueError):
        cl.inv(torch.zeros(2, 3, 3, dtype=torch.complex128))
    with pytest.raises(ValueError):
        cl.inv(torch.zeros(2, 3, 3, 3))  # pairs need a last dim of 2
    with pytest.raises(ValueError):
        cl.inv(torch.zeros(2, 3, 5), layout="interleaved")  # odd: not pairs
    with pytest.raises(ValueError):
        cl.inv(torch.zeros(2, 3, 4, 2))  # not square
