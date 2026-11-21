"""CPU tests of the team-D kernels' own arithmetic (review round 3), on ``kernel_replica``: float32 replicas of
matrix_exp_f32_gaudi2.c, lu_factor_f32_gaudi2.c / lu_solve_f32_gaudi2.c and jacobi_svd_f32_gaudi2.c. The other CPU
tests stand the plain-torch references (``tpc.*_reference``) in for the kernels, so each test here also checks that
the reference has the kernel's contract.

Each fix of round 3 has a test that fails on the code before it: the replica's ``legacy*`` switches replay the old
arithmetic and show the failure, the current replica and the reference do not have it."""
import math

import numpy as np
import pytest
import torch

import kernel_replica as R
from hpu_ops import linalg, tpc


def skew(n, norm, seed=0, batch=None):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(*(() if batch is None else (batch,)), n, n, generator=g, dtype=torch.float64)
    x = x - x.mT
    return (x * (norm / torch.linalg.matrix_norm(x, 1)[..., None, None])).float()


def rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-30))


# ---- matrix_exp ---------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("n", [1, 3, 6, 22])
@pytest.mark.parametrize("norm", [0.5, 5.0, 50.0, 1e3, 1e5])
def test_expm_replica_and_reference_share_the_contract(n, norm):
    # n = 1: exp(-norm) (exp(+1e5) overflows); CPU float32 is exact there, Padé-13 near theta is a few ulp off
    a = skew(n, norm, seed=n) if n > 1 else torch.tensor([[-norm]], dtype=torch.float32)
    truth = torch.linalg.matrix_exp(a.double()).numpy()
    bar = max(10 * rel(torch.linalg.matrix_exp(a).numpy(), truth), 3e-5)
    got = R.matrix_exp(a.numpy())
    ref = tpc.matrix_exp_reference(a).numpy()
    assert R.expm_scaling(a.numpy()) == linalg._expm_scaling(float(a.abs().sum(-2).amax()))
    assert rel(got, truth) <= bar
    assert rel(ref, truth) <= bar


def test_expm_past_the_old_scaling_cap_is_finite():
    """REVISION 1: skew 6x6 of 1-norm 1e7 needs 21 halvings; the old cap of 16 left ||A|| / 2^16 = 152 for Padé-13,
    and the result was NaN (measured on the card too). With the cap at 126 it is as accurate as CPU float32."""
    a = skew(6, 1e7)
    # 21 halvings at the theta of then, 23 at the kernel's THETA = 2 (code review #8)
    assert R.expm_scaling(a.numpy(), theta=R.THETA_LEGACY) == 21 and R.expm_scaling(a.numpy()) == 23
    assert R.expm_scaling(a.numpy(), legacy=True) == 16
    assert not np.isfinite(R.matrix_exp(a.numpy(), legacy=True, legacy_rcp=True)).all()  # the kernel of then
    truth = torch.linalg.matrix_exp(a.double()).numpy()
    bar = rel(torch.linalg.matrix_exp(a).numpy(), truth)
    for got in (R.matrix_exp(a.numpy()), tpc.matrix_exp_reference(a).numpy()):
        assert np.isfinite(got).all()
        assert rel(got, truth) <= 10 * bar


def test_expm_scaling_count_covers_float32():
    theta = 2.0  # the kernels' THETA (code review #8)
    assert linalg._expm_scaling(theta) == 0
    assert linalg._expm_scaling(theta * 2 ** 20) == 20
    assert linalg._expm_scaling(theta * 2 ** 20 * 1.0000001) == 21
    assert linalg._expm_scaling(3.4028234e38) == 126 == linalg._EXPM_S_MAX
    assert linalg._expm_scaling(math.inf) == 0 and linalg._expm_scaling(math.nan) == 0
    big = np.full((2, 2), 1.7e38, np.float32)
    assert R.expm_scaling(big) == 126


def test_expm_reference_scales_each_matrix_alone():
    """The kernel scales every matrix by its own norm. The reference used the norm of the whole batch (and a cap of
    128), so a small matrix next to a large one was scaled and squared as the large one: another contract."""
    small, big = skew(5, 0.3, seed=1), skew(5, 1e4, seed=2)
    both = tpc.matrix_exp_reference(torch.stack([small, big]))
    assert torch.equal(both[0], tpc.matrix_exp_reference(small))
    assert torch.equal(both[1], tpc.matrix_exp_reference(big))
    assert rel(both[0].numpy(), R.matrix_exp(small.numpy())) < 1e-6
    g = torch.randn(2, 5, 5, generator=torch.Generator().manual_seed(3))
    fr = tpc.matrix_exp_frechet_reference(torch.stack([small, big]), g)
    assert torch.equal(fr[0], tpc.matrix_exp_frechet_reference(small, g[0]))


def test_expm_negative_definite_underflows_to_zero():
    a = (-1e30 * (torch.eye(4) + 0.01 * torch.rand(4, 4, generator=torch.Generator().manual_seed(0)))).float()
    got = R.matrix_exp(a.numpy())
    assert np.isfinite(got).all() and np.abs(got).max() == 0
    assert torch.equal(tpc.matrix_exp_reference(a), torch.zeros(4, 4))


# ---- lu_factor / lu_solve -----------------------------------------------------------------------------------------


def tied(n, seed):
    """A = L U, L unit lower with +-1 below the diagonal: every candidate of every pivot column ties and the
    arithmetic is exact, so LAPACK's first-max rule gives pivots 1..n (as in tests/probe_revision_r3.py)."""
    g = torch.Generator().manual_seed(seed)
    lower = torch.tril(torch.randint(0, 2, (n, n), generator=g).float() * 2 - 1, -1) + torch.eye(n)
    upper = torch.triu(torch.randint(-2, 3, (n, n), generator=g).float(), 1) + torch.diag(
        torch.randint(1, 3, (n,), generator=g).float())
    return lower @ upper


@pytest.mark.parametrize("n", [22, 32])
@pytest.mark.parametrize("seed", [0, 1])
def test_lu_ties_take_the_first_row_as_lapack(n, seed):
    """REVISION 2: the pivot is the first row of largest |a_ik|. Another tie order (``argmax="last"``) gives other
    pivots and factors on these matrices, so the test tells them apart."""
    a = tied(n, seed)
    LU_t, piv_t = torch.linalg.lu_factor(a)
    assert torch.equal(piv_t, torch.arange(1, n + 1, dtype=torch.int32))
    LU, piv, perm, info = R.lu_factor(a.numpy())
    assert (piv == piv_t.numpy()).all() and np.array_equal(LU, LU_t.numpy()) and info == 0
    LU_r, piv_r, perm_r, info_r = tpc.lu_factor_reference(a)
    assert torch.equal(piv_r, piv_t) and torch.equal(LU_r, LU_t)
    assert (perm == perm_r.numpy()).all()
    _, piv_last, _, _ = R.lu_factor(a.numpy(), argmax="last")
    assert not (piv_last == piv_t.numpy()).all()


@pytest.mark.parametrize("n", [1, 5, 22, 32])
def test_lu_replica_matches_reference_and_lapack(n):
    a = torch.randn(n, n, generator=torch.Generator().manual_seed(n))
    LU, piv, perm, info = R.lu_factor(a.numpy())
    LU_r, piv_r, perm_r, info_r = tpc.lu_factor_reference(a)
    LU_t, piv_t = torch.linalg.lu_factor(a)
    assert (piv == piv_r.numpy()).all() and (piv == piv_t.numpy()).all()
    assert (perm == perm_r.numpy()).all() and info == int(info_r)
    assert rel(LU, LU_t.numpy()) < 1e-5 and rel(LU_r.numpy(), LU_t.numpy()) < 1e-5
    b = torch.randn(n, 3, generator=torch.Generator().manual_seed(100 + n))
    x = R.lu_solve(LU, piv, b.numpy())
    assert rel(x, torch.linalg.lu_solve(LU_t, piv_t, b).numpy()) < 1e-4


def huge(n=6, seed=1):
    g = torch.Generator().manual_seed(seed)
    a = torch.eye(n, dtype=torch.float64) + 0.01 * torch.randn(n, n, generator=g, dtype=torch.float64)
    return (a * 2.0 ** 126.5).float()  # pivots ~2^126.5: 1/pivot ~2^-126.5 is below the smallest normal float


def test_lu_huge_pivots_keep_their_multipliers():
    """Minor (REVISION): 1/pivot is flushed to zero past |pivot| = 2^126, so the multipliers were 0 (and lu_solve's
    1/u_ii too). The kernels now scale such pivots by 2^-64 first, as gj_solve does."""
    a = huge()
    LU_t, piv_t = torch.linalg.lu_factor(a.double())
    LU_old, _, _, _ = R.lu_factor(a.numpy(), legacy_recip=True)
    LU, piv, _, _ = R.lu_factor(a.numpy())
    assert (piv == piv_t.numpy()).all()
    assert rel(LU, LU_t.numpy()) < 1e-6
    assert rel(np.tril(LU_old, -1), np.tril(LU_t.numpy(), -1)) == 1.0  # every multiplier lost
    b = (torch.randn(6, 2, generator=torch.Generator().manual_seed(2), dtype=torch.float64) * 2.0 ** 126).float()
    truth = torch.linalg.solve(a.double(), b.double()).numpy()
    assert rel(R.lu_solve(LU, piv, b.numpy()), truth) < 1e-5
    old = R.lu_solve(LU, piv, b.numpy(), legacy_recip=True)
    assert not rel(old, truth) < 1e-5


# ---- jacobi_svd ---------------------------------------------------------------------------------------------------


def check_svd(U, S, Vh, a, tol=2e-5):
    a = np.asarray(a, np.float64)
    truth = np.linalg.svd(a, compute_uv=False)
    assert rel(S, truth) < tol
    assert rel(U @ np.diag(S) @ Vh, a) < 10 * tol
    assert np.abs(U.T @ U - np.eye(U.shape[1])).max() < 10 * tol
    assert np.abs(Vh @ Vh.T - np.eye(Vh.shape[0])).max() < 10 * tol


@pytest.mark.parametrize("shape,contiguous", [((22, 22), True), ((22, 22), False), ((8, 5), True), ((5, 8), True),
                                              ((1, 1), True), ((3, 3), True)])
def test_svd_replica_and_reference_share_the_contract(shape, contiguous):
    """REVISION 3: a contiguous square A goes through the kernel as its rows (the reference took the columns, another
    rotation sequence: other signs of U and V). The replica routes as ``tpc.jacobi_svd``; with distinct singular
    values U and Vh agree with the reference column by column, signs included."""
    g = torch.Generator().manual_seed(sum(shape))
    a = torch.randn(*shape, generator=g)
    a_in = a if contiguous else a.mT.contiguous().mT
    assert a_in.is_contiguous() == contiguous
    U, S, Vh = R.jacobi_svd(a.numpy(), contiguous=contiguous)
    check_svd(U, S, Vh, a.numpy())
    Ur, Sr, Vhr = tpc.svd_reference(a_in)
    assert rel(S, Sr.numpy()) < 1e-5
    assert np.abs(U - Ur.numpy()).max() < 1e-3 and np.abs(Vh - Vhr.numpy()).max() < 1e-3
    _, s_only, _ = R.jacobi_svd(a.numpy(), compute_uv=False, contiguous=contiguous)
    assert rel(s_only, S) < 1e-6


def test_svd_rank_deficient_completes_an_orthonormal_basis():
    a = torch.randn(6, 6, generator=torch.Generator().manual_seed(7))
    a[2] = 0
    a[4] = a[1]
    U, S, Vh = R.jacobi_svd(a.numpy())
    assert np.sum(S < 1e-5 * S.max()) == 2
    check_svd(U, S, Vh, a.numpy())
    Ur, Sr, Vhr = tpc.svd_reference(a)
    assert rel(S, Sr.numpy()) < 1e-5


def test_svd_zero_matrix():
    U, S, Vh = R.jacobi_svd(np.zeros((4, 4), np.float32))
    assert (S == 0).all()
    assert np.abs(U.T @ U - np.eye(4)).max() < 1e-6 and np.abs(Vh @ Vh.T - np.eye(4)).max() < 1e-6


def test_svd_tiny_matrix_is_scaled():
    """Minor (REVISION): max|a| < 1e-36 was not scaled, so the squared row norms underflowed and every singular value
    was 0. The kernel now scales any nonzero matrix by 1/max|a| (normal for any normal max|a|)."""
    a = (torch.randn(5, 5, generator=torch.Generator().manual_seed(4)) * 1e-37).float()
    assert a.abs().max() < 1e-36
    _, s_old, _ = R.jacobi_svd(a.numpy(), compute_uv=False, legacy_tiny=True)
    assert (s_old == 0).all()
    U, S, Vh = R.jacobi_svd(a.numpy())
    flushed = torch.where(a.abs() < 2.0 ** -126, 0.0, a)  # the card's FTZ of the input (also in the replica)
    truth = np.linalg.svd(flushed.double().numpy(), compute_uv=False)
    assert rel(S, truth) < 1e-5
    assert rel(tpc.svd_reference(flushed)[1].numpy(), truth) < 1e-5
