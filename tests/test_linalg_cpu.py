"""The linalg composites checked on the CPU standing in for HPU: outputs and gradients against torch's own, and what
compiled graphs contain."""
import warnings

import pytest
import torch
import torch.utils._pytree as pytree
from test_cpu import compiled, cpu_as_hpu, ops

from hpu_ops import linalg


def compare(f, install, *inputs, name, rtol=None, atol=None):
    """f's outputs and input gradients with the rewrite against torch's own, both on the CPU; returns the calls."""
    torch.manual_seed(1)

    def run():
        leaves = [x.detach().clone().requires_grad_(x.is_floating_point()) for x in inputs]
        outs = [o for o in pytree.tree_leaves(f(*leaves)) if isinstance(o, torch.Tensor) and o.is_floating_point()]
        weights = [torch.randn(o.shape, dtype=o.dtype, generator=torch.Generator().manual_seed(i))
                   for i, o in enumerate(outs)]
        sum((o * w).sum() for o, w in zip(outs, weights)).backward()
        return outs, [x.grad for x in leaves if x.requires_grad]

    want = run()
    with cpu_as_hpu(install) as calls:
        got = run()
        assert calls[name] >= 1, dict(calls)
    torch.testing.assert_close(got, want, rtol=rtol, atol=atol)
    return calls


def spd(*batch, n, dtype=torch.float64):
    A = torch.randn(*batch, n, n, dtype=dtype)
    return A @ A.mT + n * torch.eye(n, dtype=dtype)


def general(*batch, n, dtype=torch.float64):
    return torch.randn(*batch, n, n, dtype=dtype) + n * torch.eye(n, dtype=dtype)


# ---- triangular solve


@pytest.mark.parametrize("unitriangular", [False, True])
@pytest.mark.parametrize("upper", [False, True])
@pytest.mark.parametrize("n", [1, 2, 3, 8, 22, 64])
def test_tri_inverse(n, upper, unitriangular):
    torch.manual_seed(0)
    T = torch.linalg.cholesky(spd(4, n=n))
    T = T.mT if upper else T
    X = linalg.tri_inverse(T, upper, unitriangular)
    if unitriangular:
        T = (torch.triu if upper else torch.tril)(T, 1 if upper else -1) + torch.eye(n, dtype=T.dtype)
    torch.testing.assert_close(X @ T, torch.eye(n, dtype=T.dtype).expand(4, n, n))


N = 22
SHAPE_CASES = {
    "unbatched": lambda: (general(n=N), torch.randn(N, 3, dtype=torch.float64)),
    "batched": lambda: (general(2, 3, n=N), torch.randn(2, 3, N, 5, dtype=torch.float64)),
    "broadcast": lambda: (general(3, n=N), torch.randn(2, 1, N, 4, dtype=torch.float64)),
    "transposed": lambda: (general(n=N).mT, torch.randn(N, 4, dtype=torch.float64)),
    "expanded": lambda: (general(5, n=N), torch.randn(1, N, 4, dtype=torch.float64)),  # expanded to 5 inside f
}


@pytest.mark.parametrize("unitriangular", [False, True])
@pytest.mark.parametrize("left", [True, False])
@pytest.mark.parametrize("upper", [False, True])
@pytest.mark.parametrize("shape_case", list(SHAPE_CASES))
def test_solve_triangular_matches_torch(shape_case, upper, left, unitriangular):
    torch.manual_seed(0)
    A, B = SHAPE_CASES[shape_case]()
    if not left:
        B = B.mT  # (..., k, n) so that B @ A^-1 is defined

    def f(a, b):
        if shape_case == "expanded":
            b = b.expand(5, *b.shape[1:])  # stride 0 in the kernel; the gradient flows back through the expand
        return torch.linalg.solve_triangular(a, b, upper=upper, left=left, unitriangular=unitriangular)

    compare(f, linalg.install_solve_triangular, A, B, name="solve_triangular")


def test_solve_triangular_float32():
    torch.manual_seed(0)
    A, B = general(n=N, dtype=torch.float32), torch.randn(N, 3)
    f = lambda a, b: torch.linalg.solve_triangular(a, b, upper=False)  # noqa: E731
    compare(f, linalg.install_solve_triangular, A, B, name="solve_triangular", rtol=1e-4, atol=1e-4)


def test_solve_triangular_compiled_has_no_native_op():
    torch.manual_seed(0)
    A, B = general(3, n=N), torch.randn(3, N, 4, dtype=torch.float64)
    f = lambda a, b: torch.linalg.solve_triangular(a, b, upper=True)  # noqa: E731
    nodes = []
    with cpu_as_hpu(linalg.install_solve_triangular):
        got = compiled(f, nodes)(A, B)
    torch.testing.assert_close(got, f(A, B))
    assert "aten.linalg_solve_triangular.default" not in ops(nodes), ops(nodes)


def test_cholesky_and_qr_backward_use_the_composite():
    torch.manual_seed(0)
    S = spd(8, n=8).requires_grad_()
    with cpu_as_hpu(linalg.install_solve_triangular) as calls:
        torch.linalg.cholesky(S).sum().backward()
        assert calls["solve_triangular"] >= 1, dict(calls)
    G = general(n=8).requires_grad_()
    with cpu_as_hpu(linalg.install_solve_triangular) as calls:
        torch.linalg.qr(G).R.sum().backward()
        assert calls["solve_triangular"] >= 1, dict(calls)


def test_takes_declines_float64_on_hpu_and_warns_once():
    linalg._WARNED.clear()
    A = torch.randn(3, 3, dtype=torch.float64)
    takes = linalg._takes("hpu", "x")
    with pytest.warns(UserWarning, match="hpu-ops#16"):
        assert takes(A) is False
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert takes(A) is False
    assert linalg._takes("cpu", "x")(A) is True
    for bad in (torch.randn(3, 3, dtype=torch.complex64), torch.ones(3, 3, dtype=torch.int64),
                torch.randn(3, 3, dtype=torch.float16), torch.randn(3, 3, dtype=torch.bfloat16),
                torch.randn(0, 3, 3), torch.randn(3, 4), torch.randn(3)):
        assert linalg._takes("cpu", "x")(bad) is False


# ---- cholesky_solve


@pytest.mark.parametrize("upper", [False, True])
@pytest.mark.parametrize("shape_case", [k for k in SHAPE_CASES if k != "transposed"])
def test_cholesky_solve_matches_torch(shape_case, upper):
    torch.manual_seed(0)
    A, B = SHAPE_CASES[shape_case]()
    L = torch.linalg.cholesky(spd(*A.shape[:-2], n=N))
    L = L.mH if upper else L

    def f(b, l):
        if shape_case == "expanded":
            b = b.expand(5, *b.shape[1:])  # stride 0 in the kernel; the gradient flows back through the expand
        return torch.cholesky_solve(b, l, upper=upper)

    compare(f, linalg.install_cholesky_solve, B, L, name="cholesky_solve")


def test_cholesky_solve_inference_mode_matches_torch():
    torch.manual_seed(0)
    L = torch.linalg.cholesky(spd(3, n=N))
    B = torch.randn(3, N, 4, dtype=torch.float64)
    want = torch.cholesky_solve(B, L)
    with cpu_as_hpu(linalg.install_cholesky_solve) as calls, torch.inference_mode():
        got = torch.cholesky_solve(B, L)
    # The CPU stand-in never registers the `{}` key; the backend kernel is checked on Gaudi in Task 7.
    assert not calls, dict(calls)
    torch.testing.assert_close(got, want)
