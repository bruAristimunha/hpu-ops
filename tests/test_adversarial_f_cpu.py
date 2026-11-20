"""Round-2 adversarial CPU tests: the triangular-solve overloads the Gaudi probe saw fall back to the CPU before they
were routed (bridge 1.21.4: aten::linalg_solve_triangular.out in eager and lazy, aten::triangular_solve.X in every
mode)."""
import pytest
import torch
from test_cpu import cpu_as_hpu

from hpu_ops import linalg


def _tri(n=6):
    torch.manual_seed(0)
    return torch.randn(2, n, n).tril(-1) / n + torch.diag_embed(1 + torch.rand(2, n)), torch.randn(2, n, 3)


def test_solve_triangular_out_variant_is_routed():
    A, B = _tri()
    out = torch.empty_like(B)
    with cpu_as_hpu(linalg.install_solve_triangular) as calls:
        torch.linalg.solve_triangular(A, B, upper=False, out=out)
    torch.testing.assert_close(out, torch.linalg.solve_triangular(A, B, upper=False))
    assert calls["solve_triangular"] >= 1, dict(calls)


def test_legacy_triangular_solve_is_routed():
    A, B = _tri()
    with cpu_as_hpu(linalg.install_solve_triangular) as calls:
        X = torch.triangular_solve(B, A, upper=False).solution
    torch.testing.assert_close(X, torch.linalg.solve_triangular(A, B, upper=False))
    assert calls["solve_triangular"] >= 1, dict(calls)


def test_solve_triangular_out_resizes_like_torch():
    A, B = _tri()
    out = torch.empty(0)
    with cpu_as_hpu(linalg.install_solve_triangular) as calls:
        torch.linalg.solve_triangular(A, B, upper=False, out=out)
    assert out.shape == B.shape and calls["solve_triangular"] >= 1
    torch.testing.assert_close(out, torch.linalg.solve_triangular(A, B, upper=False))


@pytest.mark.parametrize("upper", [False, True])
@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("unitriangular", [False, True])
def test_legacy_triangular_solve_options_and_grads(upper, transpose, unitriangular):
    torch.manual_seed(1)
    A0 = torch.randn(6, 6) / 6 + torch.diag(1 + torch.rand(6))  # one A broadcast against a batch of B
    B0 = torch.randn(2, 6, 3)
    expected = [None] * 4
    for k, routed in enumerate((False, True)):
        A, B = A0.clone().requires_grad_(), B0.clone().requires_grad_()
        if routed:
            with cpu_as_hpu(linalg.install_solve_triangular) as calls:
                X, C = torch.triangular_solve(B, A, upper, transpose, unitriangular)
                (X.square().sum() + C.sum()).backward()
            assert calls["solve_triangular"] >= 1
        else:
            X, C = torch.triangular_solve(B, A, upper, transpose, unitriangular)
            (X.square().sum() + C.sum()).backward()
        got = [X.detach(), C.detach(), A.grad, B.grad]
        if k == 0:
            expected = got
        else:
            assert C.shape == expected[1].shape
            for g, e in zip(got, expected):
                torch.testing.assert_close(g, e, rtol=1e-4, atol=1e-5)
