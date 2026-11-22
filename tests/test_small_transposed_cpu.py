"""CPU tests of the n <= 64 Gauss-Jordan path on the small kernels' float32 replicas (``kernel_replica.gj_solve``):
the kernel must factor ``A^T``, as torch's CPU solve / inv / det (LAPACK on the row-major A) and this module's n > 64
path do. On ``A`` (main) partial pivoting's growth and exact cancellations differ from torch's (code review, finding
3): Wilkinson-type matrices lose every digit, duplicate rows give det exactly 0 and a zero gradient, a singular
``[[2^100, 2^100], [2^-100, 2^-100]]`` reports det 1 and info 0, and ``[[0, 1], [nan, 1]]`` reports info 1."""
import numpy as np
import pytest
import torch
from test_cpu import cpu_as_hpu

import kernel_replica as kr
from hpu_ops import linalg, tpc


def _stack(outs):
    return tuple(torch.from_numpy(np.stack([o[j] for o in outs])) for j in range(3))


@pytest.fixture
def replica(monkeypatch):
    """``tpc._op`` / ``tpc._inv_op`` -> the small kernels' replicas; the CPU tensors take the kernel path."""
    monkeypatch.setattr(tpc, "_op", lambda: lambda A, B: _stack([kr.gj_solve(a, b) for a, b in zip(A, B)]))
    monkeypatch.setattr(tpc, "_inv_op", lambda: lambda A: _stack([kr.gj_solve(a) for a in A]))
    monkeypatch.setattr(linalg, "_use_tpc", lambda A: True)
    monkeypatch.setattr(linalg, "_GJ", None)


def rel(x, ref):
    x, ref = x.double(), ref.double()
    return float((x - ref).abs().max() / ref.abs().max())


def wilkinson(n, random_last=False):
    W = torch.eye(n) - torch.tril(torch.ones(n, n), -1)
    W[:, -1] = torch.randn(n, generator=torch.Generator().manual_seed(n)) if random_last else 1
    return W / 2


SIZES = [4, 32, 40, 64]


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("random_last", [False, True])
def test_wilkinson_solve_and_inv_keep_torchs_accuracy(n, random_last, replica):
    W = wilkinson(n, random_last)[None]
    B = torch.randn(1, n, 3, generator=torch.Generator().manual_seed(1))
    want_x, want_inv = torch.linalg.solve(W.double(), B.double()), torch.linalg.inv(W.double())
    X, _, _ = tpc.gj_solve(W, B)
    Xi, _, _ = tpc.gj_inv(W)
    assert rel(X, want_x) <= max(1e-5, 10 * rel(torch.linalg.solve(W, B), want_x))
    assert rel(Xi, want_inv) <= max(1e-5, 10 * rel(torch.linalg.inv(W), want_inv))
    with cpu_as_hpu(linalg.install_solve):
        assert rel(torch.linalg.solve(W, B), want_x) <= max(1e-5, 10 * rel(torch.linalg.solve(W, B), want_x))


@pytest.mark.parametrize("n", SIZES)
def test_duplicate_rows_det_and_gradient_like_torch(n, replica):
    g = torch.Generator().manual_seed(n)
    A = torch.randn(n, n, generator=g) / n ** 0.5 + torch.eye(n)  # det of order 1: no float32 overflow at n = 64
    A[1] = A[0]
    want = A.clone().requires_grad_()
    torch.linalg.det(want).backward()
    got = A.clone().requires_grad_()
    with cpu_as_hpu(linalg.install_inv_det):
        det = torch.linalg.det(got)
        det.backward()
    ref = torch.linalg.det(A.double())
    assert abs(float(det) - float(torch.linalg.det(A))) <= max(1e-7, 10 * abs(float(torch.linalg.det(A)) - float(ref)))
    assert got.grad.abs().max() > 0  # torch gives the adjugate, not zero
    assert rel(got.grad, want.grad) < 1e-2


def test_huge_tiny_singular_like_torch(replica):
    A = torch.tensor([[2.0 ** 100, 2.0 ** 100], [2.0 ** -100, 2.0 ** -100]])
    X, D, S = tpc.gj_inv(A)
    assert S[1] != 0  # singular: info set, as torch's inv raises
    assert float(S[0] * D.prod()) == 0
    want = float(torch.linalg.det(A))
    with cpu_as_hpu(linalg.install_inv_det):
        assert float(torch.linalg.det(A)) == want == 0
        sign, logabsdet = torch.linalg.slogdet(A)
        assert float(sign) == 0 and float(logabsdet) == float("-inf")
        with pytest.raises(torch.linalg.LinAlgError):
            torch.linalg.inv(A)


@pytest.mark.parametrize("n", [2, 64])
def test_nan_input_returns_nan_like_torch(n, replica):
    A = torch.eye(n)
    A[:2, :2] = torch.tensor([[0.0, 1.0], [float("nan"), 1.0]])
    with cpu_as_hpu(linalg.install_inv_det):
        X = torch.linalg.inv(A)  # torch returns NaN (info 0) rather than raising
        sign, logabsdet = torch.linalg.slogdet(A)
    assert X.isnan().all() and logabsdet.isnan()
