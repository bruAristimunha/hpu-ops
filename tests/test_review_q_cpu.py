"""CPU tests for the Codex adversarial review of PRs #30 and #32 (CODEX_REVIEW.txt): the Python-side findings, each
with Codex's own input. Every test here failed on main (aa86bb3 / 59155d1) and passes with the fix."""
import pytest
import torch
from test_cpu import cpu_as_hpu

from hpu_ops import linalg, tpc


@pytest.fixture(autouse=True)
def eager_tpc(monkeypatch):
    monkeypatch.setenv("PT_HPU_LAZY_MODE", "0")


@pytest.fixture
def lu_kernels(gj_ref, monkeypatch):
    """The LU kernels' contracts in plain torch (as ``lu_hook`` of test_linalg_cpu) and the substitution kernel's
    (``tri_solve_reference``) behind ``_TRI``; records the hooks called."""
    calls = []

    def solve(LU, pivots, B):
        calls.append("getrs")
        batch = torch.broadcast_shapes(LU.shape[:-2], B.shape[:-2])
        return linalg.lu_solve(LU.detach().expand(*batch, *LU.shape[-2:]), pivots.expand(*batch, LU.shape[-1]),
                               B.detach(), True)

    def tri(*args):
        calls.append("tri")
        return tpc.tri_solve_reference(*args)

    monkeypatch.setattr(linalg, "_LU", tpc.lu_factor_reference)
    monkeypatch.setattr(linalg, "_LU_PERM", lambda p: linalg.pivots_to_perm(p.detach()).to(torch.float32))
    monkeypatch.setattr(linalg, "_LU_SOLVE", solve)
    monkeypatch.setattr(linalg, "_TRI", tri)
    return calls


# ---- #4: a broadcast matrix RHS of lu_solve is a matrix, never linalg.solve's batch of vectors


@pytest.mark.parametrize("left", [True, False])
@pytest.mark.parametrize("adjoint", [False, True])
def test_lu_solve_broadcast_matrix_rhs(left, adjoint, lu_kernels):
    LU, pivots = torch.linalg.lu_factor(torch.eye(2).expand(2, 2, 2).contiguous())
    B = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    want = torch.linalg.lu_solve(LU, pivots, B, left=left, adjoint=adjoint)
    with cpu_as_hpu(linalg.install_lu) as calls:
        got = torch.linalg.lu_solve(LU, pivots, B, left=left, adjoint=adjoint)
        assert calls["lu_solve"] == 1
    assert got.shape == want.shape == (2, 2, 2)
    torch.testing.assert_close(got, want)


# ---- #14: adjoint / right lu_solve from the factors, not from P @ L @ U rebuilt in float32


def wilkinson(n):
    W = torch.eye(n) - torch.tril(torch.ones(n, n), -1)
    W[:, -1] = 1
    return W


@pytest.mark.parametrize("left, adjoint", [(True, True), (False, False), (False, True)])
def test_lu_solve_uses_the_factors(left, adjoint, lu_kernels):
    n = 26
    W = wilkinson(n)
    LU, pivots = torch.linalg.lu_factor(W)
    B = torch.ones(n, 1) if left else torch.ones(1, n)
    A = W.double().mH if adjoint else W.double()
    exact = torch.linalg.solve(A, B.double(), left=left)
    # the solve with these very factors, in float64: torch's float32 lu_solve is the platform LAPACK's (on Linux
    # MKL / OpenBLAS: 0.34 off at n = 26, Wilkinson's growth; macOS Accelerate: within 1e-6), not a reference
    want = torch.linalg.lu_solve(LU.double(), pivots, B.double(), left=left, adjoint=adjoint)
    with cpu_as_hpu(linalg.install_lu):
        got = torch.linalg.lu_solve(LU, pivots, B, left=left, adjoint=adjoint)
    torch.testing.assert_close(got.double(), exact, rtol=0, atol=1e-6)
    torch.testing.assert_close(got.double(), want, rtol=0, atol=1e-6)
    assert "tri" in lu_kernels and "getrs" not in lu_kernels


def test_lu_solve_adjoint_backward_uses_the_factors(lu_kernels):
    """The backward of a left solve is the adjoint solve: the gradient of B is exact on the same input."""
    n = 26
    W = wilkinson(n)
    LU, pivots = torch.linalg.lu_factor(W)
    B = torch.ones(n, 1, requires_grad=True)
    with cpu_as_hpu(linalg.install_lu):
        torch.linalg.lu_solve(LU, pivots, B).sum().backward()
    want = torch.linalg.solve(W.double().mH, torch.ones(n, 1, dtype=torch.float64))
    torch.testing.assert_close(B.grad.double(), want, rtol=0, atol=1e-6)


# ---- #15: LU backward on scaled identities (no A^-1 sandwich squaring the scale)


@pytest.mark.parametrize("scale", [1e20, 1e-24])
def test_lu_backward_scaled_identity(scale, lu_kernels):
    A = (scale * torch.eye(2)).requires_grad_()
    with cpu_as_hpu(linalg.install_lu) as calls:
        torch.linalg.lu_factor_ex(A).LU.trace().backward()
        assert calls["lu_factor"] == 1
    torch.testing.assert_close(A.grad, torch.eye(2))


# ---- #19: the large (eager-only) route declines under every tracer, also where is_compiling() is False


def _large_route_seen(trace):
    takes = linalg._tpc_takes("hpu", lambda *a, **k: True, large=True)
    seen = []

    def f(A, B):
        seen.append(takes(A, B))
        return A @ B

    A, B = torch.eye(65), torch.ones(65, 1)
    assert takes(A, B) is True  # eager: the in-place kernel
    trace(f)(A, B)
    return seen


@pytest.mark.parametrize("tracer", ["make_fx", "make_fx_fake", "aot_function"])
def test_large_route_declines_when_traced(tracer, monkeypatch):
    from functorch.compile import aot_function, nop
    from torch.fx.experimental.proxy_tensor import make_fx

    monkeypatch.setattr(tpc, "ready", lambda: True)
    trace = {"make_fx": make_fx, "make_fx_fake": lambda f: make_fx(f, tracing_mode="fake"),
             "aot_function": lambda f: aot_function(f, fw_compiler=nop)}[tracer]
    seen = _large_route_seen(trace)
    assert seen and not any(seen)


# ---- #11 / #12: lstsq


@pytest.mark.parametrize("A, rcond", [(torch.eye(2), 1.0), (torch.diag(torch.tensor([1.0, 1e-8])), None)])
def test_compiled_gels_ignores_rcond(A, rcond, svd_hook):
    from test_cpu import compiled

    B = torch.ones(2, 1)
    want = torch.linalg.lstsq(A, B, rcond=rcond, driver="gels").solution
    with cpu_as_hpu(linalg.install_lstsq) as calls:
        eager = torch.linalg.lstsq(A, B, rcond=rcond, driver="gels").solution
        got = compiled(lambda a, b: torch.linalg.lstsq(a, b, rcond=rcond, driver="gels").solution, [])(A, B)
        assert calls["lstsq"] >= 1
    torch.testing.assert_close(eager, want)
    torch.testing.assert_close(got, want)


def test_lstsq_backward_uses_pinv_default_cutoff(svd_hook):
    A = torch.diag(torch.tensor([1.0, 1e-8]))
    grads = []
    for route in (False, True):
        B = torch.ones(2, 1, requires_grad=True)
        if route:
            with cpu_as_hpu(linalg.install_lstsq) as calls:
                torch.linalg.lstsq(A, B, rcond=1e-10, driver="gelsd").solution.sum().backward()
                assert calls["lstsq"] == 1
        else:
            torch.linalg.lstsq(A, B, rcond=1e-10, driver="gelsd").solution.sum().backward()
        grads.append(B.grad)
    torch.testing.assert_close(grads[1], grads[0])
    torch.testing.assert_close(grads[0], torch.tensor([[1.0], [0.0]]))


# ---- #13 and the legacy-tol minor: tolerances compared as torch does (float64, not rounded to S's dtype)


@pytest.fixture
def svd_routes(svd_hook, monkeypatch):
    monkeypatch.setattr(linalg, "_CIA_BACKENDS_ON_CPU", True)

    def install(d):
        undos = [linalg.install_svd(d), linalg.install_pinv(d), linalg.install_matrix_rank(d)]
        return lambda: [u() for u in reversed(undos)]

    return install


CALLS = {
    "rank_float_atol": lambda A: torch.linalg.matrix_rank(A, atol=1 - 1e-9),
    "rank_tensor_atol": lambda A: torch.linalg.matrix_rank(A, atol=torch.tensor(1 - 1e-9, dtype=torch.float64)),
    "rank_batched_atol": lambda A: torch.linalg.matrix_rank(A.expand(2, 2, 2),
                                                            atol=torch.full((2,), 1 - 1e-9, dtype=torch.float64)),
    "pinv_float_atol": lambda A: torch.linalg.pinv(A, atol=1 - 1e-9),
    "pinv_tensor_atol": lambda A: torch.linalg.pinv(A, atol=torch.tensor(1 - 1e-9, dtype=torch.float64)),
    "rank_legacy_negative_tol": lambda A: torch.linalg.matrix_rank(0 * A, tol=-1.0),
    "rank_legacy_negative_tol_tensor": lambda A: torch.linalg.matrix_rank(0 * A, tol=torch.tensor(-1.0)),
}


@pytest.mark.parametrize("name", list(CALLS))
def test_tolerance_compared_like_torch(name, svd_routes):
    A = torch.eye(2)
    want = CALLS[name](A)
    with cpu_as_hpu(svd_routes) as calls:
        got = CALLS[name](A)
        assert calls["svd"] + calls["pinv"] >= 1
    torch.testing.assert_close(got, want)


# ---- #7: SVD backward finite for distinct singular values, first and double backward


def _svd_loss(A):
    U = torch.linalg.svd(A)[0]
    return U[1, 2] * U[2, 2]  # invariant to the singular vectors' signs


def _grad_and_second(A, route):
    A = A.clone().requires_grad_()
    ctx = cpu_as_hpu(linalg.install_svd) if route else torch.enable_grad()
    with ctx:
        (g,) = torch.autograd.grad(_svd_loss(A), A, create_graph=True)
        (gg,) = torch.autograd.grad(g[1, 2], A)
    return g.detach(), gg


@pytest.mark.parametrize("diag", [(1e18, 2.0001, 2.0), (1e10, 2.0, 1.0)])
def test_svd_backward_distinct_values(diag, svd_hook):
    A = torch.diag(torch.tensor(diag))
    g, gg = _grad_and_second(A, route=True)
    want_g, want_gg = _grad_and_second(A.double(), route=False)
    assert torch.isfinite(g).all() and torch.isfinite(gg).all()
    torch.testing.assert_close(g.double(), want_g, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(gg.double(), want_gg, rtol=1e-3, atol=1e-3)


# ---- #9: matrix_exp backward, each matrix scaled by its own norm; traced branch capped as the kernels (126)


def test_matrix_exp_double_backward_scales_per_matrix(gj_ref):
    A = torch.stack([torch.eye(2), -1e8 * torch.eye(2)])
    grads = []
    for route in (False, True):
        X = A.clone().requires_grad_()
        ctx = cpu_as_hpu(linalg.install_matrix_exp) if route else torch.enable_grad()
        with ctx:
            (g,) = torch.autograd.grad(torch.linalg.matrix_exp(X).sum(), X, create_graph=True)
        grads.append(g.detach())
    torch.testing.assert_close(grads[1], grads[0])
    torch.testing.assert_close(grads[1][0], torch.full((2, 2), 2.7182815), rtol=1e-6, atol=0)


def test_matrix_exp_traced_frechet_reaches_large_norms(gj_ref):
    from torch.fx.experimental.proxy_tensor import make_fx

    A, G = -1e10 * torch.eye(2), torch.ones(2, 2)
    traced = make_fx(linalg._matrix_exp_frechet_torch, tracing_mode="fake")(A, G)
    X = A.clone().requires_grad_()
    (want,) = torch.autograd.grad(torch.linalg.matrix_exp(X), X, G)
    torch.testing.assert_close(traced(A, G), want)
    torch.testing.assert_close(linalg._matrix_exp_frechet_torch(A, G), want)


# ---- minors: lu_unpack aliasing, pinv's non-differentiable tolerance, matrix_norm dims, out= dtypes


def test_lu_unpack_returns_distinct_empties(lu_kernels):
    LU, pivots = torch.eye(2), torch.tensor([1, 2], dtype=torch.int32)
    with cpu_as_hpu(linalg.install_lu) as calls:
        _, L, U = torch.lu_unpack(LU, pivots, unpack_data=False)
        assert calls["lu_unpack"] == 1
    L.resize_(3)
    assert U.shape == (0,)


def test_pinv_differentiable_rtol_raises_as_torch(svd_hook):
    rtol = torch.tensor(0.5, requires_grad=True)
    with pytest.raises(RuntimeError):
        torch.linalg.pinv(torch.eye(2), rtol=rtol)
    with cpu_as_hpu(linalg.install_pinv), pytest.raises(RuntimeError):
        torch.linalg.pinv(torch.eye(2), rtol=rtol)


def test_matrix_norm_rejects_invalid_dim(svd_hook):
    with pytest.raises(IndexError):
        torch.linalg.matrix_norm(torch.eye(2), ord=2, dim=(0, 3))
    with cpu_as_hpu(linalg.install_svd), pytest.raises(IndexError):
        torch.linalg.matrix_norm(torch.eye(2), ord=2, dim=(0, 3))


def _backend(install, op, monkeypatch):
    captured = []
    monkeypatch.setattr(linalg, "register_kernels", lambda _d, e: captured.extend(e))
    monkeypatch.setattr(tpc, "ready", lambda: True)
    install("hpu")
    return next(fn for o, key, fn in captured if o == op and key == "{}")


def test_out_handlers_reject_other_dtypes(svd_hook, monkeypatch):
    monkeypatch.setattr(linalg, "_HOUSEHOLDER", lambda A, tau: tpc.householder_product_reference(A, tau))
    monkeypatch.setattr(linalg, "_MATRIX_EXP", lambda A: torch.linalg.matrix_exp(A))
    expm = _backend(linalg.install_matrix_exp, torch.ops.aten.linalg_matrix_exp.out, monkeypatch)
    svd = _backend(linalg.install_svd, torch.ops.aten._linalg_svd.U, monkeypatch)
    hh = _backend(linalg.install_householder_product, torch.ops.aten.linalg_householder_product.out, monkeypatch)
    monkeypatch.setattr(linalg, "_on_cpu", lambda op, args, kw: op(*args, **kw))
    A = torch.eye(2)
    with pytest.raises(RuntimeError):
        expm(A, out=torch.empty(2, 2, dtype=torch.int32))
    with pytest.raises(RuntimeError):
        svd(A, U=torch.empty(0, dtype=torch.int32), S=torch.empty(0), Vh=torch.empty(0))
    with pytest.raises(RuntimeError):
        hh(A, torch.zeros(2), out=torch.empty(2, 2, dtype=torch.int32))
    U, Vh = torch.empty(3), torch.empty(3)
    svd(torch.randn(3, 3), False, False, U=U, S=torch.empty(0), Vh=Vh)
    assert U.shape == Vh.shape == (0,)
