"""CPU tests of the in-place kernel path (``gj_large``, 64 < n <= ``tpc.LARGE_MAXN``) on ``kernel_replica.gj_large``,
the kernel's arithmetic replayed in float32 (FTZ, the bit-level reciprocal, one rounding per FMA). ``tpc._large_op``
is replaced by the replica, so ``tpc.gj_inv`` / ``gj_solve`` / ``large_factors`` (gathers, division by the pivots,
the Newton step) and, with ``linalg._LARGE_ON_CPU``, SolveEx / InvEx / Det / Slogdet run their real code. The
``legacy`` replica (the kernel before the review: the pivot row normalized by 1/pivot first) shows the failures."""
import numpy as np
import pytest
import torch
from test_cpu import cpu_as_hpu

import kernel_replica as kr
from hpu_ops import linalg, tpc


def permuted(b, n, seed=0):
    """Near-identity rows in a random order: pivoting across the whole height is needed, both det signs occur."""
    g = torch.Generator().manual_seed(seed)
    M = torch.eye(n) + 0.05 * torch.randn(b, n, n, generator=g) / n ** 0.5
    perm = torch.stack([torch.randperm(n, generator=g) for _ in range(b)])
    return M.gather(1, perm[..., None].expand(b, n, n))


@pytest.fixture(autouse=True)
def eager_tpc(monkeypatch):
    monkeypatch.setenv("PT_HPU_LAZY_MODE", "0")


@pytest.fixture
def replica(monkeypatch):
    """``tpc._large_op`` -> the current kernel's replica; the list collects the shapes it was called with."""
    calls = []
    op = kr.gj_large_op()

    def fake(A):
        calls.append(tuple(A.shape))
        return op(A)

    monkeypatch.setattr(tpc, "_large_op", lambda: fake)
    return calls


def legacy(monkeypatch):
    monkeypatch.setattr(tpc, "_large_op", lambda: kr.gj_large_op(legacy=True))


def rel(x, ref):
    x, ref = x.double(), ref.double()
    return float((x - ref).abs().max() / ref.abs().max())


@pytest.mark.parametrize("n", [65, 100, 130])
def test_replica_contract(n):
    A = permuted(2, n)
    R, P, Q, D, S = (torch.from_numpy(np.stack(o)) for o in zip(*[kr.gj_large(a.numpy()) for a in A]))
    assert R.shape == (2, n, 64 * ((n + 63) // 64)) and torch.all(R[..., n:] == 0)
    rows = R[..., :n].gather(1, P.long()[..., None].expand(2, n, n))
    inv = rows.gather(2, Q.long()[:, None, :].expand(2, n, n))
    assert rel(inv, torch.linalg.inv(A.double())) < 1e-5
    assert torch.equal(Q.long(), P.long().argsort(-1))
    torch.testing.assert_close(D.double().abs().prod(-1), torch.linalg.det(A.double()).abs(), rtol=1e-4, atol=0)


@pytest.mark.parametrize("batch", [(), (2,), (2, 3)])
def test_gj_inv_solve_factors(batch, replica):
    n = 70
    A = permuted(int(np.prod(batch)) if batch else 1, n).reshape(*batch, n, n)
    B = torch.randn(*batch, n, 5)
    X, D, S = tpc.gj_inv(A)
    assert rel(X, torch.linalg.inv(A.double())) < 1e-5
    sign, logabsdet = torch.linalg.slogdet(A.double())
    assert torch.equal(S[..., 0].double(), sign)
    torch.testing.assert_close(D.double().abs().log().sum(-1), logabsdet, rtol=0, atol=1e-5)
    Y, _, _ = tpc.gj_solve(A, B)
    assert rel(Y, torch.linalg.solve(A.double(), B.double())) < 1e-5
    D2, S2 = tpc.large_factors(A)
    assert torch.equal(D2, D) and torch.equal(S2, S)


def tiny_multiplier(n=70):
    """Review finding 1: the pivot row [1e30, 1e-9] scaled by 1/1e30 flushes 1e-39 to zero."""
    A = torch.eye(n)
    A[0, 0], A[0, 1], A[1, 0], A[1, 1] = 1e30, 1e-9, 1e30, 2e-9
    return A


def huge_pivot(n=70):
    """Review finding 1: a pivot >= 2^126, whose 1/pivot is subnormal."""
    A = torch.eye(n)
    A[0, :2] = 1e38
    A[1, :2] = torch.tensor([1.0, 2.0])
    return A


@pytest.mark.parametrize("make", [tiny_multiplier, huge_pivot])
def test_extreme_pivots(make, replica):
    A = make()[None]
    X, D, S = tpc.gj_inv(A)
    det = (S[..., 0] * D.double().prod(-1))
    torch.testing.assert_close(det, torch.linalg.det(A.double()), rtol=1e-5, atol=0)
    assert rel(X, torch.linalg.inv(A.double())) < 1e-5 and S[0, 1] == 0


@pytest.mark.parametrize("make", [tiny_multiplier, huge_pivot])
def test_extreme_pivots_failed_before(make, monkeypatch):
    legacy(monkeypatch)
    A = make().mT[None]  # the route runs the kernel on A^T: this hands the kernel the matrix of the review
    X, D, S = tpc.gj_inv(A)
    det = float(S[0, 0] * D.double().prod(-1))
    true = float(torch.linalg.det(A.double()))
    assert abs(det - true) > 1e-3 * abs(true) or rel(X, torch.linalg.inv(A.double())) > 1e-2


def test_inf_keeps_a_permutation(replica):
    """Review finding 4: an Inf makes NaN candidates; the pivot order must stay a permutation."""
    torch.manual_seed(1)
    A = torch.randn(70, 70)
    A[69, 69] = float("inf")
    _, P, _, _, _ = kr.gj_large_op()(A[None])
    assert sorted(P[0].long().tolist()) == list(range(70))
    _, P_old, _, _, _ = kr.gj_large_op(legacy=True)(A[None])
    assert sorted(P_old[0].long().tolist()) != list(range(70))  # the bug, reproduced
    tpc.gj_inv(A[None])


def test_nan_gives_nan(replica):
    A = permuted(3, 70)
    A[1, 0, 0] = float("nan")
    X, D, _ = tpc.gj_inv(A)
    assert X[1].isnan().all() and D[1].isnan().all() and torch.isfinite(X[[0, 2]]).all()
    Y, _, _ = tpc.gj_solve(A, torch.randn(3, 70, 2))
    assert Y[1].isnan().all() and torch.isfinite(Y[[0, 2]]).all()


def test_singular_info_and_zero_rows(replica):
    A = permuted(3, 70)
    A[1, 7, :] = 0  # a zero row: column 7 of the A^T the kernel runs on
    X, D, S = tpc.gj_inv(A)
    assert S[1, 1] == 8 and D[1].prod() == 0 and torch.isfinite(X).all()
    assert torch.equal(S[[0, 2], 1], torch.zeros(2))


@pytest.fixture
def large_on_cpu(monkeypatch, replica):
    monkeypatch.setattr(linalg, "_LARGE_ON_CPU", True)
    return replica


OPS = {
    "solve": (linalg.install_solve, lambda a, b: torch.linalg.solve(a, b), True),
    "solve_left_false": (linalg.install_solve, lambda a, b: torch.linalg.solve(a, b.mT, left=False).mT, True),
    "inv": (linalg.install_inv_det, lambda a, b: torch.linalg.inv(a) @ b, True),
    "det": (linalg.install_inv_det, lambda a, b: torch.linalg.det(a)[..., None, None] * b, True),
    "slogdet": (linalg.install_inv_det, lambda a, b: torch.linalg.slogdet(a).logabsdet[..., None, None] * b, True),
}


@pytest.mark.parametrize("name", list(OPS))
def test_functions_through_the_replica(name, large_on_cpu):
    """SolveEx / InvEx / Det / Slogdet on the in-place path, forward and gradients against float64."""
    install, f, _ = OPS[name]
    A, B = permuted(2, 70, seed=4), torch.randn(2, 70, 3)
    xs = [A.clone().requires_grad_(), B.clone().requires_grad_()]
    refs = [A.double().requires_grad_(), B.double().requires_grad_()]
    with cpu_as_hpu(install) as calls:
        out = f(*xs)
        out.sum().backward()
    want = f(*refs)
    want.sum().backward()
    assert sum(calls.values()) >= 1 and large_on_cpu
    assert rel(out.detach(), want.detach()) < 1e-4
    for x, r in zip(xs, refs):
        assert rel(x.grad, r.grad) < 1e-4, name


def test_det_does_not_form_the_inverse(large_on_cpu, monkeypatch):
    monkeypatch.setattr(tpc, "_large", lambda *a, **k: (_ for _ in ()).throw(AssertionError("inverse formed")))
    with cpu_as_hpu(linalg.install_inv_det):
        torch.linalg.det(permuted(2, 70))
        torch.linalg.slogdet(permuted(2, 70))


def test_solve_backward_launches_no_kernel(large_on_cpu):
    A = permuted(1, 70).requires_grad_()
    B = torch.randn(1, 70, 2, requires_grad=True)
    with cpu_as_hpu(linalg.install_solve):
        X = torch.linalg.solve(A, B)
        before = len(large_on_cpu)
        X.sum().backward()
    assert len(large_on_cpu) == before


@pytest.mark.parametrize("lazy", ["0", "1"])
def test_large_route_takes_every_size_batch_and_mode(lazy, monkeypatch):
    """Nothing on the CPU: past tpc.MAXN the in-place kernels take every n (past LARGE_MAXN the blocked route), every
    batch (a small one: the blocked route, ``tpc._square_pays``) and both modes."""
    monkeypatch.setenv("PT_HPU_LAZY_MODE", lazy)
    monkeypatch.setattr(tpc, "ready", lambda: True)
    takes = linalg._tpc_takes("hpu", lambda *a, **k: True, large=True)
    for n, b in ((220, 1), (300, 1), (512, 2), (600, 1), (1500, 1)):
        assert takes(torch.empty(b, n, n), torch.empty(b, n, 3)) is True
    assert takes(torch.empty(1, 64, 64), torch.empty(1, 64, 3)) is True


def test_square_pays_by_batch_and_mode(monkeypatch):
    """The cost model: one square call for a full card of matrices, the blocked route for one large matrix (measured:
    n = 512 batch 1, square 28.2 ms against blocked 9.1 lazy) and never past LARGE_MAXN."""
    for lazy in (False, True):
        monkeypatch.setattr(tpc, "_lazy", lambda _l=lazy: _l)
        assert tpc._square_pays(220, 1) is True
        assert tpc._square_pays(130, 32) is True
        assert tpc._square_pays(512, 1) is False
    monkeypatch.setattr(tpc, "_lazy", lambda: True)
    assert tpc._square_pays(384, 1) is False
    assert tpc._square_pays_for(torch.empty(48, 600, 600)) is False


def test_panel_widths():
    for n in range(65, 1400, 7):
        widths = tpc._panel_widths(n)
        assert sum(widths) == n and all(tpc.MAXN < c <= tpc.LARGE_MAXN for c in widths)
        assert max(widths) - min(widths) <= 1 and (len(widths) == 1 or max(widths) <= tpc.PANEL + 1)


def test_large_route_declines_under_compile(monkeypatch):
    monkeypatch.setenv("PT_HPU_LAZY_MODE", "0")
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    assert tpc.large_pays(220, 1) is False


# ---- Codex review of #30/#32 (CODEX_REVIEW.txt): #1 pivot-tie growth, #17 used-row quotient overflow


def wilkinson(n=130):
    """Codex #1: ``W = I - tril(1, -1)``, last column 1, halved (condition 58.3, entries <= 0.5): partial pivoting on
    W doubles the last column at every step (2^128 at n = 130); torch factors the row-major A as LAPACK's A^T."""
    W = torch.eye(n) - torch.tril(torch.ones(n, n), -1)
    W[:, -1] = 1
    return W / 2


def growth_used_row(n=65):
    """Codex #17: ``[[2^70, 2^70], [0, 2^-70]]`` in eye(65): eliminating the used row 0 at step 1 takes 2^140."""
    A = torch.eye(n)
    A[0, 0], A[0, 1], A[1, 1] = 2.0 ** 70, 2.0 ** 70, 2.0 ** -70
    return A


@pytest.mark.parametrize("make", [wilkinson, growth_used_row, lambda: growth_used_row().mT.contiguous(),
                                  lambda: tiny_multiplier().mT.contiguous(), lambda: huge_pivot().mT.contiguous()])
def test_review_growth_inputs(make, replica):
    """#1 and #17 (and the transposes the A^T route hands the kernel): inverse, solve, det and slogdet as torch."""
    A = make()[None]
    n = A.shape[-1]
    ref = torch.linalg.inv(A.double())
    X, D, S = tpc.gj_inv(A)
    assert torch.isfinite(X).all() and S[0, 1] == 0
    assert rel(X, ref) < 1e-5
    B = torch.ones(1, n, 1)
    Y, _, _ = tpc.gj_solve(A, B)
    assert rel(Y, torch.linalg.solve(A.double(), B.double())) < 1e-5
    D2, S2 = tpc.large_factors(A)
    sign, logabsdet = torch.linalg.slogdet(A.double())
    assert torch.equal(S2[..., 0].double(), sign)
    torch.testing.assert_close(D2.double().abs().log().sum(-1), logabsdet, rtol=0, atol=1e-4)


def test_review_wilkinson_through_solve_det(large_on_cpu):
    """#1 through the public ops: det 0.5 and logabsdet -0.693147 as torch, a finite solve against ones."""
    A = wilkinson()
    with cpu_as_hpu(linalg.install_inv_det) as calls:
        det = torch.linalg.det(A)
        sl = torch.linalg.slogdet(A)
        assert sum(calls.values()) == 2
    with cpu_as_hpu(linalg.install_solve) as calls:
        x = torch.linalg.solve(A, torch.ones(130))
        assert sum(calls.values()) == 1
    torch.testing.assert_close(det, torch.tensor(0.5), rtol=1e-5, atol=0)
    torch.testing.assert_close(sl.logabsdet, torch.tensor(-0.693147), rtol=1e-5, atol=0)
    assert rel(x, torch.linalg.solve(A.double(), torch.ones(130, dtype=torch.float64))) < 1e-5


def test_review_growth_failed_before():
    """The kernel on A itself (main's route) overflows on #1; without the scaled used rows it overflows on #17."""
    _, _, _, D, S = kr.gj_large(wilkinson().numpy())
    assert S[1] == 0 and not np.isfinite(np.prod(D.astype(np.float64)))
    for A in (growth_used_row(), tiny_multiplier().mT):  # #17 itself; the review-1 matrix as the A^T route hands it
        R, _, _, _, S = kr.gj_large(A.numpy(), legacy_growth=True)
        assert S[1] == 0 and not np.isfinite(R).all()
        assert np.isfinite(kr.gj_large(A.numpy())[0]).all()


def test_review_check_script_judges_the_inverse(replica):
    """Codex #20: ``check_large_replica``'s verdict fails a wrong inverse or det even when the raw outputs match."""
    import check_large_replica as chk

    A = torch.eye(65)[None]
    X, D, S = tpc.gj_inv(A)
    base = {"bit_equal_R_P_Q_D_S": [True] * 5, "P_is_permutation": True}
    assert chk.verdict({**base, **chk.accuracy(A, X, D, S)})
    assert not chk.verdict({**base, **chk.accuracy(A, torch.zeros_like(X), D, S)})
    assert not chk.verdict({**base, **chk.accuracy(A, X, 2 * D, S)})
    assert not chk.verdict({**base, **chk.accuracy(A, X.fill_(float("nan")), D, S)})


def _inv_info_as_bridge_torch():
    """Whether the installed torch's ``inv_ex`` reports exact singularity as torch 2.6 to 2.8 (the Gaudi bridges'
    torch) do: LAPACK's factorization of A^T, where ``[[2^100, 2^100], [2^-100, 2^-100]]`` ties and cancels (info 2).
    torch 2.15 returns info 0 there (and for ``[[1, inf], [0, 1]]``); the kernels' info is checked against the
    bridges' torch, read from the parameters, on every torch."""
    return int(torch.linalg.inv_ex(torch.tensor([[2.0 ** 100, 2.0 ** 100], [2.0 ** -100, 2.0 ** -100]])).info) == 2


INV_INFO_AS_BRIDGE_TORCH = _inv_info_as_bridge_torch()


@pytest.mark.parametrize("d, info, det", [(2.0 ** -100, 2, 0.0), (2.0 ** -101, 0, -0.5)])
def test_review_exact_singularity_and_sign(d, info, det, replica):
    """Codex #2: ``[[2^100, 2^100], [2^-100, d]]`` in eye(65). On A (main's route) the multiplier 2^-200 flushed:
    info 0 at d = 2^-100 and det +0.5 at 2^-101. On A^T (torch's orientation) the rows tie and cancel exactly:
    info 2 (torch raises) and det -0.5, as torch."""
    A = torch.eye(65)
    A[0, 0], A[0, 1], A[1, 0], A[1, 1] = 2.0 ** 100, 2.0 ** 100, 2.0 ** -100, d
    D, S = tpc.large_factors(A[None])
    assert int(S[0, 1]) == info
    if INV_INFO_AS_BRIDGE_TORCH:
        assert int(torch.linalg.inv_ex(A).info) == info
    assert float(S[0, 0] * D.double().prod(-1)) == det == float(torch.linalg.det(A.double()))
    _, _, _, D0, S0 = kr.gj_large(A.numpy())  # main's route, the kernel on A
    assert (S0[1], float(np.prod(D0.astype(np.float64)))) != (info, det)


@pytest.mark.parametrize("block, info", [([[0.0, 1.0], [float("nan"), 1.0]], 0), ([[1.0, float("inf")], [0.0, 1.0]], 2)])
def test_review_nonfinite_info_as_torch(block, info, replica):
    """Codex minor: with NaN or Inf entries the info (so whether check_errors raises) is torch's, as the kernel now
    pivots A^T as LAPACK does (main reported 1 and 0)."""
    A = torch.eye(65)
    A[:2, :2] = torch.tensor(block)
    _, _, S = tpc.gj_inv(A[None])
    assert int(S[0, 1]) == info
    if INV_INFO_AS_BRIDGE_TORCH:
        assert int(torch.linalg.inv_ex(A).info) == info
    if not np.isnan(block).any():  # a NaN now gives info 0 on either orientation (the kernel's NaN rule)
        assert int(kr.gj_large(A.numpy())[4][1]) != info  # main's route, the kernel on A



# ---- (B) extreme-range findings: the kernels' arithmetic on their replicas

# ---- Codex review #8 / #10: matrix_exp_f32_gaudi2.c's reciprocal and finiteness test, on its replica


def test_review_expm_constant_component():
    """#8: exp(diag(-1e8, 0))[1, 1] is 1: the reciprocal of 1 is now exactly 1 (was 0.99999994, squared 25 times)."""
    A = np.diag([-1e8, 0.0]).astype(np.float32)
    assert kr.rcp_newton(np.float32(1))[0] == 1 and kr.rcp_newton(np.float32(1), legacy=True)[0] < 1
    assert kr.matrix_exp(A)[1, 1] == 1 and kr.matrix_exp(A)[0, 0] == 0
    # the review's value, reproduced at the theta of then (5.37; the kernel's is 2 since code review #8)
    assert abs(kr.matrix_exp(A, legacy_rcp=True, theta=kr.THETA_LEGACY)[1, 1] - 0.13530262) < 1e-6
    x = np.float32(2.0) ** np.arange(-60, 60, 7, dtype=np.float32)
    assert np.array_equal(kr.rcp_newton(x), (1 / x.astype(np.float64)).astype(np.float32))  # powers of two exact


def test_review_expm_finite_norm_at_float_max():
    """#10: the 1x1 matrix [[-FLT_MAX]]: a finite norm, 126 halvings, exp = 0 as torch (0 halvings and NaN before)."""
    big = np.float32(-np.finfo(np.float32).max)
    assert kr.expm_scaling(np.array([[big]])) == 126 and not (np.abs(big) <= np.float32(3.4e38))
    assert kr.matrix_exp(np.array([[big]]))[0, 0] == 0
    assert torch.linalg.matrix_exp(torch.tensor([[big]]))[0, 0] == 0


# ---- Codex review #16: a zero Householder coefficient next to an overflowing dot product


def zero_tau_input(m=4):
    A = torch.zeros(m, 2)
    A[2:4, 0] = 3e38
    A[2:4, 1] = 1
    return A, torch.tensor([0.0, 2 / 3])


def test_review_householder_zero_tau_replica():
    """#16 on the kernel's replica: Q's second column is torch's [0, 1/3, -2/3, -2/3] (four NaNs before)."""
    A, tau = zero_tau_input()
    want = torch.linalg.householder_product(A.double(), tau.double())
    Q = torch.from_numpy(kr.householder_product(A.numpy(), tau.numpy()))
    torch.testing.assert_close(Q.double(), want, rtol=1e-6, atol=1e-6)
    assert torch.from_numpy(kr.householder_product(A.numpy(), tau.numpy(), legacy=True))[:, 1].isnan().all()
    g = torch.Generator().manual_seed(3)  # ordinary inputs: the replica agrees with torch
    B, t = torch.randn(22, 8, generator=g), torch.rand(8, generator=g) + 0.5
    torch.testing.assert_close(torch.from_numpy(kr.householder_product(B.numpy(), t.numpy())).double(),
                               torch.linalg.householder_product(B.double(), t.double()), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("m", [4, 65])
def test_review_householder_zero_tau_route(m, monkeypatch):
    """#16 through the route: m <= 64 on the replica (``_HOUSEHOLDER``), m = 65 through the compact-WY composite."""
    A, tau = zero_tau_input(m)
    rep = lambda a, t: torch.stack([torch.from_numpy(kr.householder_product(x.numpy(), y.numpy()))  # noqa: E731
                                    for x, y in zip(a.reshape(-1, *a.shape[-2:]), t.reshape(-1, t.shape[-1]))])
    monkeypatch.setattr(linalg, "_HOUSEHOLDER", lambda a, t: rep(a, t).reshape(a.shape))
    with cpu_as_hpu(linalg.install_householder_product) as calls:
        Q = torch.linalg.householder_product(A, tau)
        assert sum(calls.values()) == 1
    want = torch.linalg.householder_product(A.double(), tau.double())
    torch.testing.assert_close(Q.double(), want, rtol=1e-6, atol=1e-6)


def test_review_lu_factor_flushed_multiplier():
    """Codex #5: ``[[2^100, 2^100], [2^-30, 2^-30]]`` is singular: U[1, 1] = 0 and info 2 as torch (the multiplier
    2^-130 flushes, so L keeps 0, but its scaled copy still cancels the row). Before: U[1, 1] = 2^-30, info 0."""
    A = torch.tensor([[2.0 ** 100, 2.0 ** 100], [2.0 ** -30, 2.0 ** -30]])
    LU, piv, _, info = kr.lu_factor(A.numpy())
    LU_t, piv_t, info_t = torch.linalg.lu_factor_ex(A)
    assert LU[1, 1] == LU_t[1, 1] == 0 and info == int(info_t) == 2 and (piv == piv_t.numpy()).all()
    LU0, _, _, info0 = kr.lu_factor(A.numpy(), legacy_tiny=True)
    assert LU0[1, 1] == 2.0 ** -30 and info0 == 0


@pytest.mark.parametrize("call", [
    lambda: tpc.matrix_exp(torch.tensor([[1 + 1j]])),
    lambda: tpc.matrix_exp_frechet(torch.eye(2), torch.eye(2, dtype=torch.complex64)),
    lambda: tpc.jacobi_svd(torch.eye(2, dtype=torch.complex64)),
    lambda: tpc.jacobi_svdvals(torch.eye(2, dtype=torch.complex128)),
    lambda: tpc.householder_product(torch.eye(2, dtype=torch.complex64), torch.zeros(1, dtype=torch.complex64)),
    lambda: tpc.lu_factor(torch.eye(2, dtype=torch.complex64)),
    lambda: tpc.lu_solve(torch.eye(2), torch.ones(2, dtype=torch.int32), torch.ones(2, 1, dtype=torch.complex64)),
    lambda: tpc.matrix_exp(torch.eye(2, dtype=torch.int64)),
])
def test_review_direct_wrappers_reject_complex(call):
    """Codex minor: the direct TPC wrappers raise on complex (and integer) data instead of casting it (complex lost
    its imaginary part: exp([[1 + 1j]]) gave 2.71828). The check runs before any kernel is touched."""
    with pytest.raises(ValueError, match="real floating"):
        call()


# ---- the blocked route (past LARGE_MAXN, or a batch too small to fill the card): gj_tall panels, matmul updates


@pytest.fixture
def blocked(monkeypatch, replica):
    monkeypatch.setattr(tpc, "_square_pays", lambda n, b: False)
    return replica


@pytest.mark.parametrize("r, c", [(130, 65), (300, 70), (200, 128)])
def test_replica_tall_panel_contract(r, c):
    """A tall panel W[r, c]: R[P] = W[P]^-1, the rows never taken hold -W[o] W[P]^-1 and Q orders them after."""
    W = torch.randn(r, c, generator=torch.Generator().manual_seed(r))
    R, P, Q, D, S = (torch.from_numpy(x) for x in kr.gj_large(W.numpy()))
    P = P.long()
    o = torch.tensor([i for i in range(r) if i not in set(P.tolist())])
    X = torch.linalg.inv(W[P].double())
    assert rel(R[P, :c], X) < 1e-4 and rel(R[o, :c], -W[o].double() @ X) < 1e-4
    assert torch.equal(Q.argsort()[:c], P) and torch.equal(Q.argsort()[c:], o)
    torch.testing.assert_close(D.double().prod(), torch.linalg.det(W[P].double()), rtol=1e-3, atol=0)


@pytest.mark.parametrize("n, b", [(130, 2), (200, 1), (300, 2)])
def test_blocked_inverse_det_and_solve(n, b, blocked):
    A = torch.randn(b, n, n, generator=torch.Generator().manual_seed(n))
    X, D, S = tpc.gj_inv(A)
    ref = torch.linalg.inv(A.double())
    assert rel(X, ref) <= max(1e-5, 10 * rel(torch.linalg.inv(A), ref))
    sign, logabsdet = torch.linalg.slogdet(A.double())
    assert torch.equal(S[..., 0] * D.sign().prod(-1), sign) and torch.equal(S[..., 1], torch.zeros(b))
    torch.testing.assert_close(D.double().abs().log().sum(-1), logabsdet, rtol=0, atol=1e-3)
    D2, S2 = tpc.large_factors(A)
    torch.testing.assert_close(D2.double().abs().log().sum(-1), logabsdet, rtol=0, atol=1e-3)
    assert torch.equal(S2[..., 0] * D2.sign().prod(-1), sign)
    B = torch.randn(b, n, 3)
    Y, _, _ = tpc.gj_solve(A, B)
    want = torch.linalg.solve(A.double(), B.double())
    assert rel(Y, want) <= max(1e-5, 10 * rel(torch.linalg.solve(A, B), want))
    assert len(blocked) == 3 * len(tpc._panel_widths(n))  # inverse, factors, solve: one launch per panel each


def test_blocked_singular_and_nan(blocked):
    A = torch.randn(3, 140, 140, generator=torch.Generator().manual_seed(3))
    A[1, 7] = 0
    A[2, 3, 4] = float("nan")
    X, D, S = tpc.gj_inv(A)
    assert S[0, 1] == 0 and S[1, 1] > 0
    assert X[2].isnan().all() and D[2].isnan().all() and X[0].isfinite().all() and X[1].isfinite().all()


def test_blocked_tournament(blocked, monkeypatch):
    """Past TALL_MAXM rows a panel's pivots come from a tournament (CALU): each chunk nominates its pivot rows."""
    monkeypatch.setattr(tpc, "TALL_MAXM", 200)
    A = torch.randn(2, 300, 300, generator=torch.Generator().manual_seed(7))
    X, D, S = tpc.gj_inv(A)
    ref = torch.linalg.inv(A.double())
    assert rel(X, ref) <= max(1e-5, 10 * rel(torch.linalg.inv(A), ref))
    assert torch.equal(S[..., 0] * D.sign().prod(-1), torch.linalg.slogdet(A.double()).sign)
    assert max(rows for _, rows, _ in blocked) <= 200


@pytest.mark.parametrize("name", list(OPS))
def test_functions_through_the_blocked_route(name, large_on_cpu, monkeypatch):
    """SolveEx / InvEx / Det / Slogdet on the blocked route (n = 140: two panels), forward and gradients."""
    monkeypatch.setattr(tpc, "_square_pays", lambda n, b: False)
    install, f, _ = OPS[name]
    A, B = permuted(2, 140, seed=5), torch.randn(2, 140, 3)
    xs = [A.clone().requires_grad_(), B.clone().requires_grad_()]
    refs = [A.double().requires_grad_(), B.double().requires_grad_()]
    with cpu_as_hpu(install) as calls:
        out = f(*xs)
        out.sum().backward()
    want = f(*refs)
    want.sum().backward()
    assert sum(calls.values()) >= 1 and all(shape[-1] == 70 for shape in large_on_cpu)
    assert rel(out.detach(), want.detach()) < 1e-4
    for x, r in zip(xs, refs):
        assert rel(x.grad, r.grad) < 1e-4, name


@pytest.mark.parametrize("name", ["det", "slogdet"])
def test_det_backward_reuses_the_forward_inverse(name, large_on_cpu):
    """det / slogdet with grad: the forward's kernel call yields the inverse, the backward launches no kernel."""
    install, f, _ = OPS[name]
    A = permuted(2, 70, seed=6).requires_grad_()
    with cpu_as_hpu(install):
        out = f(A, torch.ones(2, 70, 1))
        before = len(large_on_cpu)
        out.sum().backward()
    assert before == 1 and len(large_on_cpu) == 1


# ---- blocked substitution past tpc.MAXN (solve_triangular, cholesky_solve)


@pytest.fixture
def tri_replica(monkeypatch):
    """``tpc._tri_op`` -> the substitution kernel's replica; the list collects the diagonal blocks' shapes."""
    calls = []

    def make(upper, unitriangular):
        def op(A, B):
            calls.append(tuple(A.shape))
            out = [kr.tri_solve(a.numpy(), b.numpy(), upper, unitriangular) for a, b in zip(A, B)]
            return torch.from_numpy(np.stack(out))
        return op

    monkeypatch.setattr(tpc, "_tri_op", make)
    return calls


@pytest.mark.parametrize("n", [65, 130, 200])
@pytest.mark.parametrize("upper, left, unit", [(False, True, False), (True, True, False), (False, False, True),
                                               (True, False, True)])
def test_blocked_substitution(n, upper, left, unit, tri_replica):
    g = torch.Generator().manual_seed(n)
    T = torch.randn(2, n, n, generator=g) / n ** 0.5 + 2 * torch.eye(n)
    T = T.triu() if upper else T.tril()
    poisoned = T + (torch.full((n, n), float("nan")).tril(-1) if upper else torch.full((n, n), float("nan")).triu(1))
    if unit:
        poisoned = poisoned + torch.diag(torch.full((n,), float("nan")))
    B = torch.randn(2, n, 3, generator=g) if left else torch.randn(2, 3, n, generator=g)
    X = tpc.tri_solve(poisoned, B, upper, left, unit)
    want = torch.linalg.solve_triangular(T.double(), B.double(), upper=upper, left=left, unitriangular=unit)
    got32 = torch.linalg.solve_triangular(T, B, upper=upper, left=left, unitriangular=unit)
    assert rel(X, want) <= max(1e-6, 10 * rel(got32, want))
    assert all(shape[-1] <= tpc.MAXN for shape in tri_replica) and len(tri_replica) == -(-n // tpc.MAXN)


def test_large_triangular_ops_take_the_card(monkeypatch):
    """Past tpc.MAXN solve_triangular and cholesky_solve stay on the card (the blocked substitution)."""
    monkeypatch.setattr(tpc, "ready", lambda: True)
    takes = linalg._tpc_takes("hpu", lambda *a, **k: True, maxn=float("inf"))
    assert takes(torch.empty(1, 300, 300), torch.empty(1, 300, 2)) is True
    assert linalg._use_tri(torch.empty(300, 300)) is False  # CPU tensors keep the composite


def test_cholesky_solve_past_maxn(tri_replica, monkeypatch):
    monkeypatch.setattr(linalg, "_use_tri", lambda A: True)
    g = torch.Generator().manual_seed(1)
    M = torch.randn(2, 100, 100, generator=g)
    Lc = torch.linalg.cholesky(M @ M.mT / 100 + torch.eye(100))
    B = torch.randn(2, 100, 4, generator=g)
    for upper, F in ((False, Lc), (True, Lc.mT.contiguous())):
        got = linalg.CholeskySolve.compute(B, F, upper)
        want = torch.cholesky_solve(B.double(), F.double(), upper=upper)
        assert rel(got, want) < 1e-5


# ---- HPU_OPS_FLOAT64=card: float64 linalg on the card in float32 precision, opt-in


def test_float64_opt_in_casts_and_warns_once(monkeypatch):
    seen = []

    def kernel(A, check_errors=False):
        seen.append(A.dtype)
        return torch.linalg.inv_ex(A)

    wrapped = linalg._float64_kernel("hpu", torch.ops.aten.linalg_inv_ex.default, kernel)
    A = torch.randn(3, 5, 5, dtype=torch.float64) + 5 * torch.eye(5, dtype=torch.float64)
    monkeypatch.delenv(linalg.FLOAT64_ENV, raising=False)
    assert wrapped(A)[0].dtype == torch.float64 and seen == [torch.float64]  # default: untouched (declined later)
    monkeypatch.setenv(linalg.FLOAT64_ENV, "card")
    linalg._WARNED.discard("float64-card")
    with pytest.warns(UserWarning, match="float32 precision"):
        X, info = wrapped(A.requires_grad_())
    assert seen[-1] == torch.float32 and X.dtype == torch.float64 and info.dtype == torch.int32
    X.sum().backward()
    assert A.grad.dtype == torch.float64
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        wrapped(A.detach())  # once per process


def test_float64_opt_in_skips_out_overloads():
    k = object()
    assert linalg._float64_kernel("hpu", torch.ops.aten.linalg_solve_triangular.out, k) is k
    assert linalg._float64_kernel("cpu", torch.ops.aten.linalg_inv_ex.default, k) is not k  # wrapped, inert on cpu


def test_float64_opt_in_skips_mixed_floats(monkeypatch):
    """Mixed float64 / float32 arguments are never cast under ``HPU_OPS_FLOAT64=card`` so torch raises its dtype error."""
    seen = []

    def kernel(A, B, check_errors=False):
        seen.append((A.dtype, B.dtype))
        return torch.linalg.solve_ex(A, B)

    wrapped = linalg._float64_kernel("hpu", torch.ops.aten.linalg_solve_ex.default, kernel)
    monkeypatch.setenv(linalg.FLOAT64_ENV, "card")
    A = torch.eye(4, dtype=torch.float64)
    B = torch.ones(4, 2, dtype=torch.float32)
    assert linalg._mixed_floats((A, B), {}) is True
    with pytest.raises(RuntimeError, match="same dtype"):
        wrapped(A, B)
    assert seen == [(torch.float64, torch.float32)]


def test_eigh_big_tiny_eigenvalue_ftz_safe(monkeypatch):
    """Gershgorin-definite T with lambda_min / sigma <= 1e-19 must stay shifted so FTZ rsqrt(0) cannot yield NaN."""
    monkeypatch.setattr(tpc, "_big_op", kr.jacobi_big_op)
    A = torch.stack([
        torch.diag(torch.tensor([1.0] + [1e-20] * 32, dtype=torch.float32)),
        torch.zeros(33, 33, dtype=torch.float32),
    ])
    w, V = tpc.eigh_big(A)
    assert torch.isfinite(w).all() and torch.isfinite(V).all()
    assert (V.mT @ V - torch.eye(33)).abs().max() < 1e-5
    torch.testing.assert_close(w[0, -1], torch.tensor(1.0), rtol=1e-5, atol=0)
    torch.testing.assert_close(w[0, 0], torch.tensor(1e-20), rtol=1e-5, atol=0)
    assert torch.equal(w[1], torch.zeros(33))


def test_codex3_huge_identity_solve_is_exact(large_on_cpu):
    """Codex review of #34, #3: ``2^127 I`` (n = 65) with B its first column. Its inverse 2^-127 I flushes on the card,
    so the solve returned 0 with info 0; equilibrated by a power of two (``linalg._equilibrate``) it is exact."""
    A = 2.0 ** 127 * torch.eye(65)
    B = A[:, :1].clone()
    with cpu_as_hpu(linalg.install_solve):
        X, info = torch.linalg.solve_ex(A, B)
    assert torch.equal(X, torch.eye(65)[:, :1]) and int(info) == 0
    with cpu_as_hpu(linalg.install_inv_det):
        sign, logabsdet = torch.linalg.slogdet(A)
    assert float(sign) == 1 and abs(float(logabsdet) - 65 * 127 * float(torch.log(torch.tensor(2.0)))) < 1e-2


def test_equilibration_is_exact_on_ordinary_inputs(large_on_cpu):
    """A power of two: on ordinary inputs the equilibrated route's results equal the unscaled kernel's bit for bit."""
    A = permuted(2, 70, seed=9)
    X0, D0, S0 = tpc.gj_inv(A)
    for k in (-40, 0, 37):
        X, D, S = linalg._gj_inv(A * 2.0 ** k)
        assert torch.equal(X * 2.0 ** k, X0) and torch.equal(D * 2.0 ** -k, D0) and torch.equal(S, S0)


# ---- adversarial pass on the blocked route: odd sizes, views, broadcast, 2-D, repeated calls


@pytest.mark.parametrize("n", [65, 129, 130, 257])
def test_blocked_odd_sizes(n, blocked):
    A = torch.randn(1, n, n, generator=torch.Generator().manual_seed(n)) + n ** 0.5 * torch.eye(n)
    X, D, S = tpc.gj_inv(A)
    ref = torch.linalg.inv(A.double())
    assert rel(X, ref) <= max(1e-5, 10 * rel(torch.linalg.inv(A), ref))
    assert sum(tpc._panel_widths(n)) == n


def test_blocked_views_broadcast_and_2d(large_on_cpu, monkeypatch):
    monkeypatch.setattr(tpc, "_square_pays", lambda n, b: False)
    g = torch.Generator().manual_seed(11)
    big = torch.randn(3, 140, 140, generator=g) + 12 * torch.eye(140)
    B = torch.randn(4, 140, 2, generator=g)
    with cpu_as_hpu(linalg.install_solve), cpu_as_hpu(linalg.install_inv_det):
        X1 = torch.linalg.solve(big[1:2].mT, B)  # a transposed view of an offset slice, broadcast against batch 4
        X2 = torch.linalg.solve(big[0], B[0])  # 2-D
        X3 = torch.linalg.inv(big[2].mT)
        X3b = torch.linalg.inv(big[2].mT)  # the same call twice: same result
    want1 = torch.linalg.solve(big[1:2].mT.double(), B.double())
    assert X1.shape == (4, 140, 2) and rel(X1, want1) < 1e-5
    assert rel(X2, torch.linalg.solve(big[0].double(), B[0].double())) < 1e-5
    assert rel(X3, torch.linalg.inv(big[2].mT.double())) < 1e-5 and torch.equal(X3, X3b)


def test_blocked_extreme_scales(large_on_cpu, monkeypatch):
    """Powers of two from 2^-100 to 2^100 on the blocked route: the equilibration leaves ordinary ranges alone and the
    results scale exactly."""
    monkeypatch.setattr(tpc, "_square_pays", lambda n, b: False)
    A = permuted(1, 140, seed=12)
    X0, D0, _ = linalg._gj_inv(A)
    for k in (-100, -70, 70, 100):
        X, D, _ = linalg._gj_inv(A * 2.0 ** k)
        assert torch.equal(X * 2.0 ** k, X0) and torch.equal(D * 2.0 ** -k, D0), k


def test_backend_kernels_ignore_an_enclosing_transform(monkeypatch):
    """Below autograd the tensors are plain: a backend kernel takes the call even under forward-mode AD (it declined
    to hpu-ops' CPU route before, measured under jvp on Gaudi2); the autograd kernel still declines."""
    import torch.autograd.forward_ad as fwAD

    seen = []
    k = linalg._wrap("hpu", torch.ops.aten.linalg_inv_ex.default, "{}", lambda *a: seen.append(linalg._transformed()))
    ka = linalg._wrap("hpu", torch.ops.aten.linalg_inv_ex.default, "Autograd{}",
                      lambda *a: seen.append(linalg._transformed()))
    with fwAD.dual_level():
        k(torch.eye(2))
        ka(torch.eye(2))
    assert seen == [False, True] and linalg._transformed() is False
