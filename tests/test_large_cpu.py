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
    rows = R[..., :n].gather(1, P.long()[..., None].expand(2, n, n)) / D[..., None]
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
    A = make()[None]
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
    A[1, :, 7] = 0
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
