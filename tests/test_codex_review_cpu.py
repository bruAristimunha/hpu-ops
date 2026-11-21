"""CPU tests for the Codex review of PR #25 (REVISION.md): one or more per item.

Items 1-4 are bugs of the kernels' own float32 arithmetic, which the CPU stand-in (torch's solvers behind the
``linalg._GJ`` / ``linalg._TRI`` hooks) cannot show. They are tested on ``kernel_replica``, an operation-by-operation
float32 replica of the C kernels: its ``legacy=True`` mode reproduces the values Codex reported for the old kernels,
and the current arithmetic gives torch's answers. The same replica also stands behind the hooks, so the public calls
(``linalg.solve``, ``solve_ex``, ``det``) are checked end to end on the kernel's arithmetic."""
import sys
import types
import warnings

import numpy as np
import pytest
import torch
from kernel_replica import gj_solve as replica_gj
from kernel_replica import tri_solve as replica_tri
from test_cpu import cpu_as_hpu

from hpu_ops import linalg, tpc

S64, T64, BIG = 2.0 ** 64, 2.0 ** -64, 2.0 ** 127


@pytest.fixture(autouse=True)
def eager_tpc(monkeypatch):
    monkeypatch.setenv("PT_HPU_LAZY_MODE", "0")


def embed(block, n):
    """``eye(n)`` with ``block`` in its upper-left corner (Codex's construction)."""
    A = torch.eye(n)
    block = torch.tensor(block, dtype=torch.float32)
    A[:block.shape[0], :block.shape[1]] = block
    return A


def unit(n, i, value=1.0, m=1):
    B = torch.zeros(n, m)
    B[i] = value
    return B


def _batched(fn, A, B):
    """``fn`` (one matrix) over the broadcast batch of ``A[..., n, n]`` and ``B[..., n, m]``, as torch tensors."""
    batch = torch.broadcast_shapes(A.shape[:-2], B.shape[:-2])
    A = A.detach().expand(*batch, *A.shape[-2:]).reshape(-1, *A.shape[-2:])
    B = B.detach().expand(*batch, *B.shape[-2:]).reshape(-1, *B.shape[-2:])
    outs = [fn(a.numpy(), b.numpy()) for a, b in zip(A, B)]
    return outs, batch


@pytest.fixture
def replica_gj_hook(monkeypatch):
    """``linalg._GJ`` set to the float32 replica of the Gauss-Jordan kernel."""

    def hook(A, B):
        outs, batch = _batched(replica_gj, A.float(), B.float())
        X, D, S = (torch.from_numpy(np.stack([o[k] for o in outs])) for k in range(3))
        n, m = A.shape[-1], B.shape[-1]
        return X.reshape(*batch, n, m), D.reshape(*batch, n), S.reshape(*batch, 2)

    monkeypatch.setattr(linalg, "_GJ", hook)


@pytest.fixture
def replica_tri_hook(monkeypatch):
    """``linalg._TRI`` set to the float32 replica of the substitution kernel."""

    def hook(A, B, upper=False, left=True, unitriangular=False):
        if not left:
            return hook(A.mT, B.mT, not upper, True, unitriangular).mT
        outs, batch = _batched(lambda a, b: replica_tri(a, b, upper, unitriangular), A.float(), B.float())
        return torch.from_numpy(np.stack(outs)).reshape(*batch, *B.shape[-2:])

    monkeypatch.setattr(linalg, "_TRI", hook)


# ---- 1. Gauss-Jordan on duplicate rows: info 0 and garbage before, an exactly zero pivot now ----

@pytest.mark.parametrize("n", [2, 33])
def test_replica_duplicate_rows(n):
    A, B = embed([[3, 4], [3, 4]], n), unit(n, 1)
    X, D, S = replica_gj(A, B, legacy=True)
    assert S[1] == 0 and X[0, 0] == 11184811 and X[1, 0] == -8388608  # Codex's values: the replica is faithful
    X, D, S = replica_gj(A, B)
    assert S[1] == torch.linalg.solve_ex(A, B).info == 2
    assert D[1] == 0


@pytest.mark.parametrize("n", [2, 33])
def test_duplicate_rows_raise_through_the_kernel_arithmetic(replica_gj_hook, n):
    A, B = embed([[3, 4], [3, 4]], n), unit(n, 1)
    with cpu_as_hpu(linalg.install_solve) as calls:
        with pytest.raises(torch.linalg.LinAlgError, match="singular"):
            torch.linalg.solve(A, B)
        assert torch.linalg.solve_ex(A, B).info == 2
    assert calls["solve"] == 2
    with cpu_as_hpu(linalg.install_inv_det):
        assert torch.linalg.det(A) == 0
        with pytest.raises(torch.linalg.LinAlgError):
            torch.linalg.inv(A)


# ---- 2. substitution overflow: a separate sum overflowed where the solution does not ----

@pytest.mark.parametrize("upper", [False, True])
def test_replica_unit_triangle_overflow(upper):
    n = 33
    A, B = torch.eye(n), torch.zeros(n, 1)
    if upper:
        A[n - 3, n - 2:] = 1
        B[n - 3:] = BIG
    else:
        A[2, :2] = 1
        B[:3] = BIG
    want = torch.linalg.solve_triangular(A, B, upper=upper, unitriangular=True)
    assert torch.isfinite(want).all()
    if not upper:
        assert np.isneginf(replica_tri(A, B, upper, True, legacy=True)[2, 0])  # Codex's -inf
    got = replica_tri(A, B, upper, True)
    np.testing.assert_array_equal(got, want.numpy())


def test_unit_triangle_overflow_through_the_kernel_arithmetic(replica_tri_hook):
    n = 33
    A, B = torch.eye(n), torch.zeros(n, 1)
    A[2, :2] = 1
    B[:3] = BIG
    with cpu_as_hpu(linalg.install_solve_triangular) as calls:
        got = torch.linalg.solve_triangular(A, B, upper=False, unitriangular=True)
    assert calls["solve_triangular"] == 1
    torch.testing.assert_close(got, torch.linalg.solve_triangular(A, B, upper=False, unitriangular=True))


# ---- 3. Gauss-Jordan overflow of a well-conditioned RHS (B scaled by 1/pivot before cancellation) ----

@pytest.mark.parametrize("n", [2, 33])
def test_replica_rhs_overflow(n):
    A = embed([[0.5, 0.5], [0.5, -0.5]], n).mT.contiguous().mT
    B = unit(n, 0, BIG)
    want = torch.linalg.solve(A, B)
    assert not np.isfinite(replica_gj(A, B, legacy=True)[0][:2]).all()
    np.testing.assert_array_equal(replica_gj(A, B)[0], want.numpy())


@pytest.mark.parametrize("n", [2, 33])
def test_rhs_overflow_through_the_kernel_arithmetic(replica_gj_hook, n):
    A = embed([[0.5, 0.5], [0.5, -0.5]], n).mT.contiguous().mT
    B = unit(n, 0, BIG)
    with cpu_as_hpu(linalg.install_solve):
        got = torch.linalg.solve(A, B)
    torch.testing.assert_close(got, torch.linalg.solve(A, B))


# ---- 4. the 2^-64 rescale of a huge pivot row flushed normal entries: det doubled ----

@pytest.mark.parametrize("n", [2, 33])
def test_replica_scaled_pivots(n):
    A, B = embed([[S64, T64], [S64, 2 * T64]], n), unit(n, 1)
    X, D, S = replica_gj(A, B, legacy=True)
    assert S[0] * np.prod(D.astype(np.float64)) == 2  # Codex's doubled determinant
    X, D, S = replica_gj(A, B)
    assert S[0] * np.prod(D.astype(np.float64)) == 1
    np.testing.assert_array_equal(X, torch.linalg.solve(A, B).numpy())


@pytest.mark.parametrize("n", [2, 33])
def test_scaled_pivots_through_the_kernel_arithmetic(replica_gj_hook, n):
    A, B = embed([[S64, T64], [S64, 2 * T64]], n), unit(n, 1)
    with cpu_as_hpu(linalg.install_inv_det):
        det = torch.linalg.det(A)
        sign, logabsdet = torch.linalg.slogdet(A)
    with cpu_as_hpu(linalg.install_solve):
        X = torch.linalg.solve(A, B)
    assert det == 1 and sign == 1 and logabsdet == 0
    torch.testing.assert_close(X, torch.linalg.solve(A, B))


@pytest.mark.parametrize("n", [1, 3, 22, 64])
def test_replica_accuracy_random(n):
    """The current arithmetic on ordinary inputs: within a few float32 ulps of float64 (solve, inv, det)."""
    g = np.random.default_rng(n)
    A = g.standard_normal((n, n)).astype(np.float32) + np.float32(n ** 0.5) * np.eye(n, dtype=np.float32)
    B = g.standard_normal((n, 3)).astype(np.float32)
    X, D, S = replica_gj(A, B)
    Xi, _, _ = replica_gj(A)
    A64 = A.astype(np.float64)
    ref = np.linalg.solve(A64, B)
    assert np.abs(X - ref).max() / np.abs(ref).max() < 1e-5
    assert np.abs(Xi - np.linalg.inv(A64)).max() / np.abs(np.linalg.inv(A64)).max() < 1e-5
    assert abs(S[0] * np.prod(D.astype(np.float64)) / np.linalg.det(A64) - 1) < 1e-4
    T = np.tril(A)
    for upper, M in ((False, T), (True, T.T.copy())):
        x = replica_tri(M, B, upper)
        r = np.linalg.solve(M.astype(np.float64), B)
        assert np.abs(x - r).max() / np.abs(r).max() < 1e-5


# ---- 5. solve_triangular(out=): int destination and a destination that requires grad raise as in torch ----

def _tri(n=6):
    torch.manual_seed(0)
    return torch.randn(2, n, n).tril(-1) / n + torch.diag_embed(1 + torch.rand(2, n)), torch.randn(2, n, 3)


def _raises_like_torch(call):
    with pytest.raises(RuntimeError) as native:
        call()
    with cpu_as_hpu(linalg.install_solve_triangular):
        with pytest.raises(RuntimeError) as ours:
            call()
    assert str(ours.value) == str(native.value)


def test_solve_triangular_out_int_raises():
    _raises_like_torch(lambda: torch.linalg.solve_triangular(torch.eye(1) * 2, torch.ones(1, 1), upper=False,
                                                             out=torch.empty(1, 1, dtype=torch.long)))


def test_solve_triangular_out_requiring_grad_raises():
    leaf = torch.ones(1, 1, requires_grad=True)
    _raises_like_torch(lambda: torch.linalg.solve_triangular(torch.eye(1) * 2, torch.ones(1, 1), upper=False,
                                                             out=leaf * 2))


def test_solve_triangular_out_int_raises_past_maxn():
    """The CPU route (n > MAXN on HPU) runs torch's own out= kernel, so it raises too."""
    A, B = torch.eye(3), torch.ones(3, 1)
    out = torch.empty(3, 1, dtype=torch.long)
    with pytest.raises(RuntimeError, match="Float but found Long"):
        linalg._out_on_cpu(torch.ops.aten.linalg_solve_triangular.out, (A, B), dict(upper=False), dict(out=out))


# ---- 6. legacy cloned_coefficient keeps its autograd connection to B ----

def test_cloned_coefficient_gradient_to_B():
    A = torch.eye(2)
    for routed in (False, True):
        B = torch.ones(2, 1, requires_grad=True)
        if routed:
            with cpu_as_hpu(linalg.install_solve_triangular) as calls:
                C = torch.triangular_solve(B, A).cloned_coefficient
            assert calls["solve_triangular"] == 1
        else:
            C = torch.triangular_solve(B, A).cloned_coefficient
        assert C.requires_grad
        C.sum().backward()
        torch.testing.assert_close(B.grad, torch.zeros_like(B))


# ---- 7. mixed devices are rejected before the CPU copy ----

def test_on_cpu_rejects_mixed_devices():
    A, B = torch.eye(3), torch.ones(3, 1, device="meta")
    with pytest.raises(RuntimeError, match="same device"):
        linalg._on_cpu(torch.ops.aten.triangular_solve.default, (B, A), {})
    takes = linalg._takes("cpu", "solve_triangular")
    assert takes(A, torch.ones(3, 1)) and not takes(A, B)


# ---- 8. det double backward on a singular matrix: torch's SVD formula ----

@pytest.mark.parametrize("n", [3, 33])
def test_det_double_backward_singular(replica_gj_hook, n):
    """On the kernel arithmetic (a zero pivot gives a finite inverse row; the torch-composite reference gives inf)."""
    def grads(create_graph):
        A = torch.eye(n)
        A[-1, -1] = 0
        A = torch.stack([A, torch.eye(n) * 2]).requires_grad_()
        return torch.autograd.grad(torch.linalg.det(A).sum(), A, create_graph=create_graph)[0]

    for create_graph in (False, True):
        want = grads(create_graph)
        with cpu_as_hpu(linalg.install_inv_det) as calls:
            got = grads(create_graph)
        assert calls["det"] >= 1
        torch.testing.assert_close(got, want)
    assert want[0, -1, -1] == 1


@pytest.fixture
def gj_ref_hook(monkeypatch):
    monkeypatch.setattr(linalg, "_GJ", lambda A, B: tpc.gj_solve_reference(A.detach(), B.detach()))


def test_det_double_backward_singular_second_order(gj_ref_hook):
    """The singular cofactor stays differentiable: second-order gradients match torch's."""
    def second(A):
        A = A.clone().requires_grad_()
        g = torch.autograd.grad(torch.linalg.det(A), A, create_graph=True)[0]
        return torch.autograd.grad((g * W).sum(), A)[0]

    torch.manual_seed(0)
    A = torch.randn(4, 4, dtype=torch.float64)
    A[3] = A[0] + A[1]  # rank 3, distinct singular values
    W = torch.randn(4, 4, dtype=torch.float64)
    want = second(A)
    with cpu_as_hpu(linalg.install_inv_det):
        got = second(A)
    torch.testing.assert_close(got, want)


# ---- 9. the probe: a missing candidate gradient and the hpu-ops CPU route fail a row ----

@pytest.fixture
def probe(monkeypatch):
    """``tests/probe_adversarial_f.py`` imported with stand-ins for habana (CPU in the role of the card)."""
    core = types.SimpleNamespace(mark_step=lambda: None)
    hpu = types.SimpleNamespace(synchronize=lambda: None)
    metrics = types.SimpleNamespace(metric_global=lambda name: types.SimpleNamespace(stats=lambda: []))
    torch_mod = types.SimpleNamespace(core=core, hpu=hpu)
    for name, mod in (("habana_frameworks", types.SimpleNamespace(torch=torch_mod)),
                      ("habana_frameworks.torch", torch_mod), ("habana_frameworks.torch.core", core),
                      ("habana_frameworks.torch.hpu", hpu), ("habana_frameworks.torch.hpu.metrics", metrics)):
        monkeypatch.setitem(sys.modules, name, mod)
    import hpu_ops

    monkeypatch.setattr(hpu_ops, "enable", lambda: None)
    for name in ("_on_cpu", "_out_on_cpu"):
        monkeypatch.setattr(linalg, name, getattr(linalg, name))  # restored after the probe wraps them
    monkeypatch.delitem(sys.modules, "probe_adversarial_f", raising=False)
    import probe_adversarial_f as module

    monkeypatch.setattr(module, "HPU", torch.device("cpu"))
    return module


def _third_call_differs(normal, candidate):
    """``normal`` for the probe's two CPU references, ``candidate`` for its HPU call (the third)."""
    seen = []

    def fn(a, b):
        seen.append(1)
        return candidate(a, b) if len(seen) == 3 else normal(a, b)

    return fn


def test_probe_fails_on_missing_gradient(probe, capsys):
    solve = torch.linalg.solve
    fn = _third_call_differs(solve, lambda a, b: solve(a.detach(), b))
    assert not probe.run("missing_grad", fn, (torch.eye(33), torch.ones(33, 2)), {"grad": True})
    assert "missing on the HPU" in capsys.readouterr().out
    assert probe.run("control", solve, (torch.eye(33), torch.ones(33, 2)), {"grad": True})


def test_probe_fails_on_cpu_route(probe, capsys):
    op = torch.ops.aten.linalg_solve.default
    fn = _third_call_differs(torch.linalg.solve, lambda a, b: linalg._on_cpu(op, (a, b), {}))
    assert not probe.run("cpu_route", fn, (torch.eye(33), torch.ones(33, 2)), {})
    assert "CPU route" in capsys.readouterr().out


# ---- 10. an inference-mode offset view is copied before the kernel ----

def test_fresh_copies_inference_offset_view(monkeypatch):
    seen = []

    def fake(A, B):
        for t in (A, B):
            seen.append((t.storage_offset(), t.untyped_storage().nbytes() == t.numel() * t.element_size()))
        return tpc.gj_solve_reference(A, B)

    monkeypatch.setattr(tpc, "_op", lambda: fake)
    n = 33
    with torch.inference_mode():
        big = torch.stack([torch.eye(n), 2 * torch.eye(n)])
        rhs = torch.ones(3, n, 1)
        assert big[1:2]._base is None  # why ``_base`` alone does not see the alias
        X, _, _ = tpc.gj_solve(big[1:2], rhs[1:2])
        tpc.gj_solve(big[0:1], rhs[0:1])  # offset 0, but a prefix of a larger storage
    assert seen == [(0, True)] * 4
    torch.testing.assert_close(X, torch.full((1, n, 1), 0.5))


# ---- minor ----

def test_legacy_triangular_solve_out_is_routed_like_torch():
    A, B = _tri()
    want = torch.triangular_solve(B, A, upper=False, out=(torch.empty(0), torch.empty(0)))
    X, M = torch.empty(0), torch.empty(0)
    with cpu_as_hpu(linalg.install_solve_triangular) as calls:
        torch.triangular_solve(B, A, upper=False, out=(X, M))
    assert calls["solve_triangular"] == 1
    for got, ref in zip((X, M), want):
        torch.testing.assert_close(got, ref)
        assert got.stride() == ref.stride()


@pytest.mark.parametrize("shape", [(0,), (2, 2), (2, 6, 3)])
def test_solve_triangular_out_layout_and_warning_like_torch(shape):
    A, B = _tri()
    native_out = torch.empty(shape)
    with warnings.catch_warnings(record=True) as native_w:
        warnings.simplefilter("always")
        torch.linalg.solve_triangular(A, B, upper=False, out=native_out)
    out = torch.empty(shape)
    with cpu_as_hpu(linalg.install_solve_triangular) as calls:
        with warnings.catch_warnings(record=True) as ours_w:
            warnings.simplefilter("always")
            torch.linalg.solve_triangular(A, B, upper=False, out=out)
    assert calls["solve_triangular"] == 1
    torch.testing.assert_close(out, native_out)
    assert out.stride() == native_out.stride()
    resized = [str(w.message)[:60] for w in native_w if "resized" in str(w.message)]
    assert [str(w.message)[:60] for w in ours_w if "resized" in str(w.message)] == resized


def test_tpc_wrappers_refuse_float64(monkeypatch):
    monkeypatch.setattr(tpc, "_tri_op", lambda u, unit: pytest.fail("reached the kernel"))
    monkeypatch.setattr(tpc, "_op", lambda: pytest.fail("reached the kernel"))
    A, B = torch.tensor([[1 + 2.0 ** -30]], dtype=torch.float64), torch.ones(1, 1, dtype=torch.float64)
    for call in (lambda: tpc.tri_solve(A, B), lambda: tpc.gj_solve(A, B), lambda: tpc.gj_inv(A),
                 lambda: tpc.tri_solve(A.float(), B)):
        with pytest.raises(ValueError, match="float32"):
            call()


def test_slogdet_sign_backward_like_torch(gj_ref_hook):
    for routed in (False, True):
        A = torch.eye(33, requires_grad=True)
        if routed:
            with cpu_as_hpu(linalg.install_inv_det) as calls:
                sign, logabsdet = torch.linalg.slogdet(A)
            assert calls["slogdet"] == 1
        else:
            sign, logabsdet = torch.linalg.slogdet(A)
        sign.backward()
        assert A.grad is None
    A = torch.eye(33, requires_grad=True)
    with cpu_as_hpu(linalg.install_inv_det):
        torch.linalg.slogdet(A)[1].backward()
    torch.testing.assert_close(A.grad, torch.eye(33))
