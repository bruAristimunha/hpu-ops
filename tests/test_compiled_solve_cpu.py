"""CPU tests: a solve / det / slogdet call a tracer sees declined, whose graph node then takes hpu-ops' route at run
time, still has torch's gradient. On Gaudi2 (bridge 1.21.4) the large route (64 < n <= 512) is eager-only, so
``torch.compile(backend="hpu_backend")`` traced STaRNet(64)'s n = 96 Cayley solve (torch's orthogonal
parametrization) as the native op, hpu_backend ran that node eagerly through our autograd kernel, which took the
large route with placeholder ``LU`` / ``pivots``, and torch's traced derivative read them: NaN gradients
(repro/compiled_solve_placeholder_factors.py). Here the CPU stand-in plays both causes of a trace-only decline: the
large route's ``not _tracing`` gate (``trace_declines``) and a symbolic shape (``dynamic=True``)."""
import pytest
import torch
from test_cpu import cpu_as_hpu

from hpu_ops import linalg, tpc


@pytest.fixture
def gj(monkeypatch):
    """The Gauss-Jordan stand-in: the route returns placeholder ``LU`` / ``pivots``, as the TPC kernels do."""
    monkeypatch.setattr(linalg, "_GJ", tpc.gj_solve_reference)


@pytest.fixture
def trace_declines(monkeypatch):
    """The large route's gate on the stand-in: taken eagerly, declined whenever a tracer sees the call."""
    tpc_takes = linalg._tpc_takes

    def gated(device_type, takes, *args, **kwargs):
        inner = tpc_takes(device_type, takes, *args, **kwargs)
        return lambda *a, **k: inner(*a, **k) and not linalg._tracing(a[0])

    monkeypatch.setattr(linalg, "_tpc_takes", gated)


def cayley(W):
    """torch's orthogonal parametrization (``orthogonal_map="cayley"``), as in spd_learn's BiMap."""
    A = W - W.mT
    Id = torch.eye(A.shape[-1], dtype=A.dtype)
    return torch.linalg.solve(torch.add(Id, A, alpha=-0.5), torch.add(Id, A, alpha=0.5))


CASES = {
    "solve": (lambda W, C: (cayley(W) * C).sum(), linalg.install_solve),
    "solve_vector": (lambda W, C: (torch.linalg.solve(torch.eye(6) + W, C[..., 0]) * C[..., 1]).sum(),
                     linalg.install_solve),
    "solve_right": (lambda W, C: (torch.linalg.solve(torch.eye(6) + W, C, left=False) * C).sum(),
                    linalg.install_solve),
    "det": (lambda W, C: (torch.linalg.det(torch.eye(6) + W) * C[..., 0, 0]).sum(), linalg.install_inv_det),
    "slogdet": (lambda W, C: (torch.linalg.slogdet(torch.eye(6) + W)[1] * C[..., 0, 0]).sum(),
                linalg.install_inv_det),
}


def grads(name, run):
    f, install = CASES[name]
    torch.manual_seed(0)
    W, C = torch.randn(2, 6, 6) * 0.1, torch.randn(2, 6, 6)
    Wr = W.clone().requires_grad_()
    f(Wr, C).backward()
    torch._dynamo.reset()
    with cpu_as_hpu(install):
        Wc = W.clone().requires_grad_()
        run(f)(Wc, C).backward()
    return Wc.grad, Wr.grad


def aot(f):
    from functorch.compile import aot_function, nop

    return aot_function(f, fw_compiler=nop, bw_compiler=nop)


@pytest.mark.parametrize("name", list(CASES))
@pytest.mark.parametrize("tracer", ["compile", "aot_function"])
def test_traced_decline_runtime_route_grad(name, tracer, gj, trace_declines):
    run = (lambda f: torch.compile(f, backend="aot_eager", fullgraph=True)) if tracer == "compile" else aot
    got, want = grads(name, run)
    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("name", list(CASES))
def test_symbolic_shape_decline_grad(name, gj):
    got, want = grads(name, lambda f: torch.compile(f, backend="aot_eager", dynamic=True, fullgraph=True))
    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-5)


def test_traced_decline_double_backward(gj, trace_declines):
    """The traced Function's backward is made of differentiable ops: a second derivative is torch's."""
    torch.manual_seed(0)
    A = (torch.eye(5) + 0.1 * torch.randn(5, 5)).requires_grad_()
    B = torch.randn(5, 2)

    def g(A):
        (gA,) = torch.autograd.grad(torch.linalg.solve(A, B).square().sum(), A, create_graph=True)
        return gA.sum()

    from torch.fx.experimental.proxy_tensor import make_fx

    want = torch.autograd.grad(g(A), A)[0]
    with cpu_as_hpu(linalg.install_solve):
        graph = make_fx(lambda A: torch.autograd.grad(g(A), A)[0])(A)
        got = graph(A)
    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-5)


def test_eager_and_untraced_routes_unchanged(gj):
    """No tracer: the call is taken (one count), as before."""
    A = (torch.eye(6) + 0.1 * torch.randn(6, 6)).requires_grad_()
    with cpu_as_hpu(linalg.install_solve) as calls:
        torch.linalg.solve(A, torch.randn(6, 2)).sum().backward()
    assert calls["solve"] >= 1


@pytest.mark.parametrize("singular", [False, True])
def test_traced_det_double_backward(singular, gj, trace_declines):
    """A det backward with ``create_graph`` under a tracer selects torch's SVD formula per matrix without a host read
    (review; it raised on ``bool(singular.any())``). Singular: the first derivative (the second is NaN in the untraced
    route too, a known limit); else the second."""
    from torch.fx.experimental.proxy_tensor import make_fx

    torch.manual_seed(0)
    A = torch.eye(4) + 0.1 * torch.randn(4, 4)
    if singular:
        A[-1] = A[0]

    def g(A):
        (gA,) = torch.autograd.grad(torch.linalg.det(A), A, create_graph=True)
        return gA if singular else torch.autograd.grad(gA.square().sum(), A)[0]

    want = g(A.clone().requires_grad_())
    with cpu_as_hpu(linalg.install_inv_det):
        X = A.clone().requires_grad_()
        got = make_fx(g)(X)(X)
    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-5)
