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


def test_batch_threshold_counts_the_broadcast_batch(monkeypatch):
    """Review finding 7: past LARGE_ANY_BATCH_N, A's batch of 1 broadcast against B's batch of 8 takes the kernel."""
    monkeypatch.setattr(tpc, "ready", lambda: True)
    takes = linalg._tpc_takes("hpu", lambda *a, **k: True, large=True)
    n = tpc.LARGE_ANY_BATCH_N + 1
    A = torch.empty(1, n, n)
    assert takes(A, torch.empty(tpc.LARGE_MIN_BATCH, n, 3)) is True
    assert takes(A, torch.empty(1, n, 3)) is False
    assert takes(A[0], torch.empty(tpc.LARGE_MIN_BATCH, n, 1)) is True  # 2-D A



@pytest.mark.parametrize("lazy, want", [("0", True), ("1", False)])
def test_large_route_is_eager_only(lazy, want, monkeypatch):
    """Coordinator's ruling: past tpc.MAXN the in-place kernel in eager mode only; lazy mode keeps main's route."""
    monkeypatch.setenv("PT_HPU_LAZY_MODE", lazy)
    monkeypatch.setattr(tpc, "ready", lambda: True)
    takes = linalg._tpc_takes("hpu", lambda *a, **k: True, large=True)
    assert takes(torch.empty(1, 220, 220), torch.empty(1, 220, 220)) is want
    assert takes(torch.empty(1, 64, 64), torch.empty(1, 64, 3)) is True  # n <= 64: the kernel in both modes


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


@pytest.mark.parametrize("d, info, det", [(2.0 ** -100, 2, 0.0), (2.0 ** -101, 0, -0.5)])
def test_review_exact_singularity_and_sign(d, info, det, replica):
    """Codex #2: ``[[2^100, 2^100], [2^-100, d]]`` in eye(65). On A (main's route) the multiplier 2^-200 flushed:
    info 0 at d = 2^-100 and det +0.5 at 2^-101. On A^T (torch's orientation) the rows tie and cancel exactly:
    info 2 (torch raises) and det -0.5, as torch."""
    A = torch.eye(65)
    A[0, 0], A[0, 1], A[1, 0], A[1, 1] = 2.0 ** 100, 2.0 ** 100, 2.0 ** -100, d
    D, S = tpc.large_factors(A[None])
    assert int(S[0, 1]) == info == int(torch.linalg.inv_ex(A).info)
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
    assert int(S[0, 1]) == info == int(torch.linalg.inv_ex(A).info)
    assert int(kr.gj_large(A.numpy())[4][1]) != info  # main's route, the kernel on A

