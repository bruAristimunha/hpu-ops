"""CPU tests of the complex64 Jacobi kernels (``complex_linalg.eigh`` / ``eigvalsh`` / ``svd`` / ``svdvals``) on their
float32 replicas (``kernel_replica.ceigh_pass1/2``, ``csvd_kernel``): the kernel ops are replaced by the replicas, so
the layouts, triangles, sorting and the U/Vh assembly run their real code. Accuracy against complex128 next to torch's
complex64 CPU result: the kernel's error within max(1e-6, 10x torch's). Eigenvectors are compared phase-free:
residual ||H V - V diag(w)|| / ||H|| and orthogonality ||V^H V - I||."""
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
    monkeypatch.setattr(cl, "_ceigh_op", kr.ceigh_op)
    monkeypatch.setattr(cl, "_ceigh_refine_op", kr.ceigh_refine_op)
    monkeypatch.setattr(cl, "_csvd_op", kr.csvd_op)
    monkeypatch.setattr(cl, "_csvdvals_op", kr.csvdvals_op)


def crandn(*shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.complex(torch.randn(*shape, generator=g), torch.randn(*shape, generator=g)).to(C64)


def unitary(n, seed):
    Q, _ = torch.linalg.qr(crandn(n, n, seed=seed).to(C128))
    return Q


def herm(w, seed):
    w = torch.as_tensor(w, dtype=torch.float64)
    Q = unitary(len(w), seed)
    return ((Q * w) @ Q.mH).to(C64)


def pairs(Z):
    return torch.view_as_real(Z.to(C64)).contiguous()


def cplx(P):
    return torch.view_as_complex(P.double().contiguous())


def ok(e, t):
    assert e <= max(1e-6, 10 * t), (e, t)


def eigh_errors(H, w, V):
    """(eigenvalue, residual, orthogonality) errors, relative to max|H|, for a batch."""
    H = H.to(C128)
    nrm = H.abs().amax((-2, -1), keepdim=True)
    w128 = torch.linalg.eigvalsh(H)
    V = V.to(C128)
    ew = float(((w.double() - w128).abs() / nrm[..., 0]).max())
    res = float(((H @ V - V * w.to(C128)[..., None, :]).abs() / nrm).max())
    orth = float((V.mH @ V - torch.eye(H.shape[-1])).abs().max())
    return ew, res, orth


def eigh_cases():
    n = 16
    yield "randn_2", (lambda X: X + X.mH)(crandn(3, 2, 2, seed=1))
    yield "randn_22", (lambda X: X + X.mH)(crandn(2, 22, 22, seed=2))
    yield "randn_32", (lambda X: X + X.mH)(crandn(2, 32, 32, seed=3))
    yield "identity", torch.eye(n, dtype=C64)[None]
    yield "mult3", herm([0, 0, 0] + list(range(1, n - 2)), 4)[None]
    yield "mult_half", herm([0] * (n // 2) + [1] * (n // 2), 5)[None]
    yield "close_1e-7", herm([1, 1 + 1e-7] + list(range(2, n)), 6)[None]
    yield "cond1e6", herm(np.logspace(0, -6, n), 7)[None]
    X = crandn(1, 22, 40, seed=8)
    X = X - X.mean(-2, keepdim=True)  # average reference: rank 21, an exact zero eigenvalue
    yield "avg_ref_cov", X @ X.mH / 40
    yield "scaled_1e30", (herm(np.linspace(-1, 3, 12), 9) * 1e30)[None]
    yield "scaled_1e-30", (herm(np.linspace(-1, 3, 12), 10) * 1e-30)[None]


@pytest.mark.parametrize("name,H", list(eigh_cases()))
def test_eigh_accuracy(name, H):
    w, V = cl.eigh(pairs(H))
    ours = eigh_errors(H, w, cplx(V))
    w32, V32 = torch.linalg.eigh(H)
    ref = eigh_errors(H, w32, V32)
    for e, t in zip(ours, ref):
        ok(e, t)
    torch.testing.assert_close(cl.eigvalsh(pairs(H)), w, rtol=0, atol=0)
    assert torch.all(w[..., 1:] >= w[..., :-1])


def test_eigh_triangles_layouts_n1():
    H = (lambda X: X + X.mH)(crandn(2, 3, 6, 6, seed=11))
    junk = crandn(2, 3, 6, 6, seed=12)
    lower = torch.tril(H) + torch.triu(junk, 1) + 1j * torch.diag_embed(torch.randn(2, 3, 6))  # junk ignored
    upper = torch.triu(H) + torch.tril(junk, -1)
    w_l, V_l = cl.eigh(pairs(lower))
    w_u, _ = cl.eigh(pairs(upper), UPLO="U")
    w128 = torch.linalg.eigvalsh(H.to(C128))
    assert float((w_l.double() - w128).abs().max()) < 1e-5 and float((w_u.double() - w128).abs().max()) < 1e-5
    assert w_l.shape == (2, 3, 6) and V_l.shape == (2, 3, 6, 6, 2)
    wi, Vi = cl.eigh(pairs(lower).reshape(2, 3, 6, 12), layout="interleaved")
    assert torch.equal(wi, w_l) and torch.equal(Vi, V_l.reshape(2, 3, 6, 12))
    w1, V1 = cl.eigh(pairs(torch.tensor([[[2.5 + 3j]]])))
    assert w1.tolist() == [[2.5]] and V1.tolist() == [[[[1.0, 0.0]]]]


@pytest.mark.parametrize("uplo", ["X", "", "LU", "lower", None])
@pytest.mark.parametrize("n", [1, 4])
def test_eigh_invalid_uplo_raises(uplo, n):
    """An invalid UPLO raises as torch.linalg.eigh does (it used to be taken as "L"), also for n = 1 and empty."""
    H = pairs((lambda X: X + X.mH)(crandn(2, n, n, seed=14)))
    for fn in (cl.eigh, cl.eigvalsh):
        for A in (H, H[:0]):
            with pytest.raises(RuntimeError, match="UPLO"):
                fn(A, UPLO=uplo)
    w_lo, _ = cl.eigh(H, UPLO="l")
    assert torch.equal(w_lo, cl.eigh(H)[0]) and torch.equal(cl.eigvalsh(H, UPLO="u"), cl.eigvalsh(H, UPLO="U"))


@pytest.mark.parametrize("bad", [math.nan, math.inf])
def test_eigh_nonfinite(bad):
    H = (lambda X: X + X.mH)(crandn(2, 5, 5, seed=13))
    H[0, 3, 1] = complex(bad, 0)  # in the lower triangle, which is read
    w, V = cl.eigh(pairs(H))
    assert torch.isnan(w[0]).all() and torch.isnan(V[0]).all()
    assert torch.isfinite(w[1]).all() and torch.isfinite(V[1]).all()


def svd_errors(A, U, S, Vh):
    A = A.to(C128)
    s128 = torch.linalg.svdvals(A)
    nrm = s128[..., :1]
    U, Vh = U.to(C128), Vh.to(C128)
    es = float(((S.double() - s128).abs() / nrm).max())
    rec = float((((U * S.to(C128)[..., None, :]) @ Vh - A).abs() / nrm[..., None]).max())
    k = S.shape[-1]
    ou = float((U.mH @ U - torch.eye(k)).abs().max())
    ov = float((Vh @ Vh.mH - torch.eye(k)).abs().max())
    return es, rec, ou, ov


def svd_cases():
    n = 16
    yield "randn_1x1", crandn(2, 1, 1, seed=20)
    yield "randn_16", crandn(2, 16, 16, seed=21)
    yield "randn_22", crandn(2, 22, 22, seed=22)
    yield "randn_32", crandn(1, 32, 32, seed=23)
    yield "wide_8x20", crandn(2, 8, 20, seed=24)
    yield "tall_20x8", crandn(2, 20, 8, seed=25)
    s = torch.logspace(0, -5, n, dtype=torch.float64).to(C128)
    yield "logspace_-5", ((unitary(n, 26) * s) @ unitary(n, 27).mH).to(C64)[None]
    s = torch.logspace(0, -6, n, dtype=torch.float64).to(C128)
    yield "cond1e6", ((unitary(n, 28) * s) @ unitary(n, 29).mH).to(C64)[None]
    yield "rank10", crandn(1, 16, 10, seed=30) @ crandn(1, 10, 16, seed=31)
    s = torch.tensor([3.0, 2.0, 2.0, 2.0, 1.0, 1.0] + [0.5] * 10, dtype=torch.float64).to(C128)
    yield "repeated", ((unitary(n, 32) * s) @ unitary(n, 33).mH).to(C64)[None]


@pytest.mark.parametrize("name,A", list(svd_cases()))
def test_svd_accuracy(name, A):
    U, S, Vh = cl.svd(pairs(A))
    ours = svd_errors(A, cplx(U), S, cplx(Vh))
    ref = svd_errors(A, *torch.linalg.svd(A, full_matrices=False))
    for e, t in zip(ours, ref):
        ok(e, t)
    assert torch.all(S[..., 1:] <= S[..., :-1])
    Sv = cl.svdvals(pairs(A))
    s32 = torch.linalg.svdvals(A)
    s128 = torch.linalg.svdvals(A.to(C128))
    ok(float(((Sv.double() - s128).abs() / s128[..., :1]).max()), float(((s32.double() - s128).abs() / s128[..., :1]).max()))
    if name == "logspace_-5":  # the survey's bar: the smallest singular value to 1e-2 relative
        assert float((S[..., -1].double() - s128[..., -1]).abs() / s128[..., -1]) < 1e-2


def test_svd_zero_nonfinite_layout():
    U, S, Vh = cl.svd(pairs(torch.zeros(2, 5, 7, dtype=C64)))
    assert torch.all(S == 0)
    ok(float((cplx(Vh) @ cplx(Vh).mH - torch.eye(5)).abs().max()), 0)  # completed to orthonormal rows
    A = crandn(2, 6, 4, seed=40)
    A[0, 1, 2] = complex(math.nan, 1)
    U, S, Vh = cl.svd(pairs(A))
    assert torch.isnan(S[0]).all() and torch.isnan(U[0]).all() and torch.isfinite(S[1]).all()
    B = crandn(3, 5, 9, seed=41)
    Ui, Si, Vhi = cl.svd(pairs(B).reshape(3, 5, 18), layout="interleaved")
    U, S, Vh = cl.svd(pairs(B))
    assert torch.equal(Si, S) and torch.equal(Ui, U.reshape(3, 5, 10)) and torch.equal(Vhi, Vh.reshape(3, 5, 18))
    with pytest.raises(NotImplementedError):
        cl.svd(pairs(B), full_matrices=True)
    with pytest.raises(ValueError):
        cl.svd(pairs(crandn(33, 4)))
