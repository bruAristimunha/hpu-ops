"""The linalg composites checked on the CPU standing in for HPU: outputs and gradients against torch's own, and what
compiled graphs contain."""
import os
import re
import sys
import types
import warnings

import pytest
import torch
import torch.utils._pytree as pytree
from test_cpu import compiled, cpu_as_hpu, ops
from torch._dynamo.backends.common import aot_autograd

import hpu_ops
from hpu_ops import eigh as jacobi
from hpu_ops import linalg, tpc


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


# ---- pivoted LU


@pytest.mark.parametrize("batch", [(), (4,)])
@pytest.mark.parametrize("n", [1, 3, 22])
def test_lu_matches_lu_factor(n, batch):
    torch.manual_seed(0)
    A = torch.randn(*batch, n, n, dtype=torch.float64)  # no dominant diagonal: the pivots move
    LU, pivots = linalg.lu(A)
    want_LU, want_pivots = torch.linalg.lu_factor(A)
    torch.testing.assert_close(LU, want_LU)
    assert pivots.dtype == torch.int32
    assert torch.equal(pivots, want_pivots)


@pytest.mark.parametrize("batch", [(), (4,)])
@pytest.mark.parametrize("n", [1, 3, 22])
def test_perm_reorders_rows_to_LU(n, batch):
    torch.manual_seed(0)
    A = torch.randn(*batch, n, n, dtype=torch.float64)
    LU, pivots = torch.linalg.lu_factor(A)
    _, L, U = torch.lu_unpack(LU, pivots)
    perm = linalg.pivots_to_perm(pivots)
    assert perm.dtype == torch.int64
    torch.testing.assert_close(A.gather(-2, perm[..., None].expand_as(A)), L @ U)


def test_pivots_to_perm_gathers_no_integer_data():
    """Gaudi2 (bridge 1.21) cannot compile a gather or scatter whose data tensor is an integer dtype."""
    seen = []

    class Record(torch.utils._python_dispatch.TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if func.overloadpacket in (torch.ops.aten.gather, torch.ops.aten.scatter):
                seen.extend((func.overloadpacket.__name__, a.dtype) for a in args[:1] + args[3:]
                            if isinstance(a, torch.Tensor))
            return func(*args, **(kwargs or {}))

    torch.manual_seed(0)
    A, B = torch.randn(4, 5, 5), torch.randn(4, 5, 3)
    with Record():
        linalg.pivots_to_perm(linalg.lu(A)[1])
        linalg.lu_solve(*linalg.lu(A), B)
    assert seen  # the hook saw the gathers
    assert not [s for s in seen if not (s[1].is_floating_point or s[1].is_complex)], seen


@pytest.mark.parametrize("left", [True, False])
@pytest.mark.parametrize("batches", [((4,), (4,)), ((), (4,)), ((4,), ()), ((2, 1), (3,))])
@pytest.mark.parametrize("n", [1, 3, 22])
def test_lu_solve_left_and_right(n, batches, left):
    torch.manual_seed(0)
    a_batch, b_batch = batches
    A = torch.randn(*a_batch, n, n, dtype=torch.float64)
    B = torch.randn(*b_batch, *((n, 5) if left else (5, n)), dtype=torch.float64)
    want = torch.linalg.solve(A, B, left=left)
    torch.testing.assert_close(linalg.lu_solve(*linalg.lu(A), B, left=left), want)


def test_lu_singular_has_no_nan():
    torch.manual_seed(0)
    A = torch.randn(3, 4, 4, dtype=torch.float64)
    A[1, :, 1] = 0
    LU, pivots = linalg.lu(A)
    assert torch.isfinite(LU).all()
    want = torch.linalg.lu_factor_ex(A)
    torch.testing.assert_close(LU, want.LU)
    assert torch.equal(pivots, want.pivots)
    info = linalg.lu_info(LU)
    assert info.dtype == torch.int32
    assert torch.equal(info, torch.linalg.lu_factor_ex(A).info)


# ---- solve

SOLVE_CASES = {
    "vector": lambda: (general(n=N), torch.randn(N, dtype=torch.float64)),
    "batched_vector": lambda: (general(4, n=N), torch.randn(4, N, dtype=torch.float64)),
    "batched_matrix": lambda: (general(4, n=N), torch.randn(4, N, 3, dtype=torch.float64)),
    "right": lambda: (general(4, n=N), torch.randn(4, 3, N, dtype=torch.float64)),
    "broadcast": lambda: (general(3, n=N), torch.randn(2, 1, N, 2, dtype=torch.float64)),
    "transposed": lambda: (general(n=N).mT, torch.randn(N, 4, dtype=torch.float64)),
    "expanded": lambda: (general(5, n=N), torch.randn(1, N, 4, dtype=torch.float64)),  # expanded to 5 inside f
}


@pytest.mark.parametrize("case", list(SOLVE_CASES))
def test_solve_matches_torch(case):
    torch.manual_seed(0)
    A, B = SOLVE_CASES[case]()

    def f(a, b):
        if case == "expanded":
            b = b.expand(5, *b.shape[1:])  # stride 0 in the kernel; the gradient flows back through the expand
        return torch.linalg.solve(a, b, left=case != "right")

    compare(f, linalg.install_solve, A, B, name="solve")


def test_solve_float32():
    torch.manual_seed(0)
    A, B = general(4, n=N, dtype=torch.float32), torch.randn(4, N, 3)
    compare(torch.linalg.solve, linalg.install_solve, A, B, name="solve", rtol=1e-4, atol=1e-4)


def test_solve_ex_factors_rebuild_A_and_info_matches():
    torch.manual_seed(0)
    A, B = general(3, n=5), torch.randn(3, 5, 2, dtype=torch.float64)
    op = torch.ops.aten._linalg_solve_ex
    want = op(A, B)
    with cpu_as_hpu(linalg.install_solve):
        got = op(A, B)
    torch.testing.assert_close(got[0], want[0])
    assert torch.equal(got[3], want[3])
    # Native CPU packs the factors of A^T for a row-major A; ours are lu_factor's, and must rebuild A.
    P, L, U = torch.lu_unpack(got[1], got[2])
    torch.testing.assert_close(P @ L @ U, A)


def test_cayley_orthogonal_trains_through_the_composite():
    from torch.nn.utils.parametrizations import orthogonal

    x = torch.randn(32, N, dtype=torch.float64)

    def run():
        torch.manual_seed(0)
        layer = orthogonal(torch.nn.Linear(N, N, bias=False).double(), orthogonal_map="cayley")
        layer(x).sum().backward()
        return {k: p.grad for k, p in layer.named_parameters()}

    want = run()
    with cpu_as_hpu(linalg.install_solve) as calls:
        got = run()
        assert calls["solve"] >= 1, dict(calls)
    torch.testing.assert_close(got, want)


def test_solve_singular_raises_like_torch():
    A = torch.zeros(4, 4, dtype=torch.float64)
    B = torch.ones(4, 2, dtype=torch.float64)
    with pytest.raises(RuntimeError) as native:
        torch.linalg.solve(A, B)
    with cpu_as_hpu(linalg.install_solve) as calls:
        with pytest.raises(RuntimeError, match=re.escape(str(native.value))):
            torch.linalg.solve(A, B)
        assert calls["solve"] >= 1, dict(calls)


def test_solve_compiled_has_no_native_op():
    torch.manual_seed(0)
    A, B = general(3, n=N), torch.randn(3, N, 4, dtype=torch.float64)
    nodes = []
    with cpu_as_hpu(linalg.install_solve):
        got = compiled(torch.linalg.solve, nodes)(A, B)
    torch.testing.assert_close(got, torch.linalg.solve(A, B))
    assert "aten._linalg_solve_ex.default" not in ops(nodes), ops(nodes)


@pytest.mark.parametrize("b_shape, left", [((5,), False), ((3,), True)])
def test_solve_bad_shapes_raise_like_torch(b_shape, left):
    A, B = torch.randn(5, 5, dtype=torch.float64), torch.randn(*b_shape, dtype=torch.float64)
    with pytest.raises(RuntimeError) as native:
        torch.linalg.solve(A, B, left=left)
    with cpu_as_hpu(linalg.install_solve) as calls:
        with pytest.raises(RuntimeError, match=re.escape(str(native.value))):
            torch.linalg.solve(A, B, left=left)
    assert calls["solve"] == 0, dict(calls)


def test_solve_ex_check_errors_raises_like_torch():
    A, B = torch.zeros(4, 4, dtype=torch.float64), torch.ones(4, 2, dtype=torch.float64)
    with pytest.raises(RuntimeError) as native:
        torch.linalg.solve_ex(A, B, check_errors=True)
    with cpu_as_hpu(linalg.install_solve) as calls:
        with pytest.raises(RuntimeError, match=re.escape(str(native.value))):
            torch.linalg.solve_ex(A, B, check_errors=True)
        assert calls["solve"] >= 1, dict(calls)


# ---- inv, det, slogdet

INV_DET_FNS = {"inv": torch.linalg.inv, "inv_ex": torch.linalg.inv_ex, "det": torch.linalg.det,
               "slogdet": torch.linalg.slogdet}
INV_DET_COUNT = {"inv": "inv", "inv_ex": "inv", "det": "det", "slogdet": "slogdet"}


@pytest.mark.parametrize("batch", [(), (2, 3)])
@pytest.mark.parametrize("n", [1, 3, 22])
@pytest.mark.parametrize("fn", list(INV_DET_FNS))
def test_inv_det_slogdet_match_torch(fn, n, batch):
    torch.manual_seed(0)
    A = general(*batch, n=n)
    compare(INV_DET_FNS[fn], linalg.install_inv_det, A, name=INV_DET_COUNT[fn])


@pytest.mark.parametrize("fn", list(INV_DET_FNS))
def test_inv_det_float32(fn):
    torch.manual_seed(0)
    A = general(4, n=5, dtype=torch.float32)
    compare(INV_DET_FNS[fn], linalg.install_inv_det, A, name=INV_DET_COUNT[fn], rtol=1e-4, atol=1e-4)


def test_inv_det_singular():
    A = torch.zeros(3, 3, dtype=torch.float64)
    with pytest.raises(RuntimeError) as native:
        torch.linalg.inv(A)
    with cpu_as_hpu(linalg.install_inv_det) as calls:
        assert torch.equal(torch.linalg.det(A), torch.zeros((), dtype=torch.float64))
        sign, logabsdet = torch.linalg.slogdet(A)
        assert sign.item() == 0.0 and logabsdet.item() == float("-inf")
        with pytest.raises(RuntimeError, match=re.escape(str(native.value))):
            torch.linalg.inv(A)
        assert calls["det"] >= 1 and calls["slogdet"] >= 1 and calls["inv"] >= 1, dict(calls)


def test_inv_ex_check_errors_raises_like_torch():
    A = torch.zeros(3, 3, dtype=torch.float64)
    with pytest.raises(RuntimeError) as native:
        torch.linalg.inv_ex(A, check_errors=True)
    with cpu_as_hpu(linalg.install_inv_det) as calls:
        with pytest.raises(RuntimeError, match=re.escape(str(native.value))):
            torch.linalg.inv_ex(A, check_errors=True)
        assert calls["inv"] >= 1, dict(calls)


def test_inv_compiled_has_no_native_op():
    torch.manual_seed(0)
    A = general(3, n=N)
    nodes = []
    with cpu_as_hpu(linalg.install_inv_det):
        got = compiled(torch.linalg.inv, nodes)(A)
    torch.testing.assert_close(got, torch.linalg.inv(A))
    assert "aten.linalg_inv_ex.default" not in ops(nodes), ops(nodes)


@pytest.mark.parametrize("fn", [torch.linalg.det, lambda A: torch.linalg.slogdet(A).logabsdet], ids=["det", "slogdet"])
def test_det_slogdet_double_backward(fn):
    torch.manual_seed(0)
    A = general(2, n=3).requires_grad_()
    with cpu_as_hpu(linalg.install_inv_det) as calls:
        assert torch.autograd.gradgradcheck(fn, (A,))
        assert calls["det"] + calls["slogdet"] >= 1, dict(calls)


@pytest.mark.parametrize("perm", [[1, 0, 2, 3, 4], [1, 2, 0, 3, 4]], ids=["odd", "even"])
@pytest.mark.parametrize("fn", ["det", "slogdet"])
def test_det_slogdet_with_row_swaps(fn, perm):
    torch.manual_seed(0)
    A = general(4, n=5)[:, perm]
    compare(INV_DET_FNS[fn], linalg.install_inv_det, A, name=fn)
    with cpu_as_hpu(linalg.install_inv_det):
        det = torch.linalg.det(A)
    assert torch.equal(det.sign(), torch.linalg.det(A).sign())
    if perm[0] == 1 and perm[1] == 0:
        assert (det < 0).all()


# ---- eigh
def test_enable_installs_eigh():
    # it declines without the TPC kernel (n > EIGH_MAXN, kernel not ready), so it cannot be slower than the fallback
    assert jacobi.install_eigh in hpu_ops.FIXES


@pytest.fixture
def eigh_hook(monkeypatch):
    """``eigh._EIGH`` set to the reference; the list collects one entry per call."""
    calls = []

    def hook(A, upper):  # detached, like the real op: no autograd through the kernel
        calls.append((tuple(A.shape), upper))
        return tpc.eigh_reference(A.detach(), upper)

    monkeypatch.setattr(jacobi, "_EIGH", hook)
    return calls


def test_eigh_reference_contract():
    torch.manual_seed(0)
    S = spd(3, n=5)
    for upper in (False, True):
        w, V = tpc.eigh_reference(S, upper)
        torch.testing.assert_close(w, torch.linalg.eigvalsh(S))
        torch.testing.assert_close(V @ torch.diag_embed(w) @ V.mT, S)
    assert tpc.EIGH_MAXN == 32


@pytest.mark.parametrize("n", [1, 3, 22])
def test_eigh_routes_through_the_hook(n, eigh_hook):
    torch.manual_seed(0)
    S = spd(3, n=n)
    with cpu_as_hpu(jacobi.install_eigh) as calls:
        w, V = torch.linalg.eigh(S)
        assert calls["eigh"] >= 1, dict(calls)
    assert eigh_hook == [((3, n, n), False)]
    torch.testing.assert_close(w, torch.linalg.eigvalsh(S), rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(V @ torch.diag_embed(w) @ V.mT, S, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(V.mT @ V, torch.eye(n, dtype=S.dtype).expand_as(V), rtol=0, atol=1e-10)


def test_eigvalsh_routes_through_the_hook_and_skips_v(eigh_hook):
    torch.manual_seed(0)
    S = spd(2, n=6)
    with cpu_as_hpu(jacobi.install_eigh) as calls, torch.no_grad():
        w = torch.linalg.eigvalsh(S)
        assert calls["eigh"] >= 1, dict(calls)
        w2, V = jacobi.Eigh.compute(S, "L", False)
    assert len(eigh_hook) == 2 and V.numel() == 0
    torch.testing.assert_close(w, torch.linalg.eigvalsh(S), rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(w2, w)


@pytest.mark.parametrize("uplo", ["L", "U"])
def test_eigh_hook_uses_the_chosen_triangle(uplo, eigh_hook):
    torch.manual_seed(0)
    S = spd(3, n=8)
    junk = S + (torch.tril(torch.randn_like(S), -1) if uplo == "U" else torch.triu(torch.randn_like(S), 1)) * 100
    with cpu_as_hpu(jacobi.install_eigh):
        w, V = torch.linalg.eigh(junk, UPLO=uplo)
    assert eigh_hook[0][1] == (uplo == "U")
    torch.testing.assert_close(V @ torch.diag_embed(w) @ V.mT, S, rtol=1e-10, atol=1e-10)


def test_eigh_grad_through_the_hook(eigh_hook):
    torch.manual_seed(0)
    S = spd(3, n=8) / 8

    def f(S):
        w, V = torch.linalg.eigh(S)
        return V @ torch.diag_embed(w.exp()) @ V.mT, w  # sign-invariant in V

    compare(f, jacobi.install_eigh, S, name="eigh", rtol=1e-8, atol=1e-8)
    assert eigh_hook


def test_eigh_hook_degenerate(eigh_hook):
    D = torch.diag(torch.tensor([1.0, 1.0, 2.0], dtype=torch.float64))
    with cpu_as_hpu(jacobi.install_eigh):
        w, V = torch.linalg.eigh(D)
    assert eigh_hook
    assert torch.equal(w, torch.tensor([1.0, 1.0, 2.0], dtype=torch.float64))
    torch.testing.assert_close(V @ torch.diag(w) @ V.mT, D, rtol=0, atol=1e-12)


def test_eigh_is_declined_past_maxn_on_hpu_and_without_the_kernel(monkeypatch):
    A = torch.randn(2, 4, 4)
    takes = jacobi._tpc_takes("hpu")
    monkeypatch.setattr(tpc, "ready", lambda: False)
    assert takes(A, "L", True) is False
    monkeypatch.setattr(tpc, "ready", lambda: True)
    assert takes(A, "L", True) is True
    assert takes(torch.randn(tpc.EIGH_MAXN + 1, tpc.EIGH_MAXN + 1), "L", True) is False
    assert takes(torch.randn(tpc.EIGH_MAXN, tpc.EIGH_MAXN), "L", True) is True
    cpu = jacobi._tpc_takes("cpu")
    monkeypatch.setattr(tpc, "ready", lambda: False)
    assert cpu(torch.randn(tpc.EIGH_MAXN + 1, tpc.EIGH_MAXN + 1), "L", True) is True  # the stand-in keeps the composite


def test_eigh_hook_not_used_past_maxn(eigh_hook):
    torch.manual_seed(0)
    with cpu_as_hpu(jacobi.install_eigh):
        torch.linalg.eigvalsh(spd(1, n=tpc.EIGH_MAXN + 1))
    assert eigh_hook == []


@pytest.mark.parametrize("n", [2, 4, 22, 64])
def test_schedule_pairs_every_index_once_per_round_and_every_pair_once(n):
    rounds = jacobi._schedule(n)
    assert rounds.shape == (n - 1, n // 2, 2)
    assert (rounds[..., 0] < rounds[..., 1]).all()
    for r in rounds:
        assert sorted(r.flatten().tolist()) == list(range(n))
    pairs = {tuple(p) for p in rounds.reshape(-1, 2).tolist()}
    assert len(pairs) == n * (n - 1) // 2


@pytest.mark.parametrize("batch", [(), (3,)])
@pytest.mark.parametrize("n", [1, 2, 3, 8, 22, 64])
def test_eigh_matches_torch(n, batch):
    torch.manual_seed(0)
    S = spd(*batch, n=n)
    with cpu_as_hpu(jacobi.install_eigh) as calls:
        w, V = torch.linalg.eigh(S)
        assert calls["eigh"] >= 1, dict(calls)
    eye = torch.eye(n, dtype=S.dtype)
    torch.testing.assert_close(w, torch.linalg.eigvalsh(S), rtol=0, atol=1e-10)
    torch.testing.assert_close(V.mT @ V, eye.expand_as(V), rtol=0, atol=1e-10)
    torch.testing.assert_close(V @ torch.diag_embed(w) @ V.mT, S, rtol=0, atol=1e-10)


def test_eigh_float32_converges():
    torch.manual_seed(0)
    S = spd(32, n=22, dtype=torch.float32)
    with cpu_as_hpu(jacobi.install_eigh):
        w, V = torch.linalg.eigh(S)
    T = V.mT @ S @ V
    off = (T - torch.diag_embed(T.diagonal(dim1=-2, dim2=-1))).flatten(-2).norm(dim=-1)
    assert (off / S.flatten(-2).norm(dim=-1) < 1e-5).all()
    want = torch.linalg.eigvalsh(S)
    assert ((w - want).abs().amax(-1) < 1e-4 * S.flatten(-2).norm(dim=-1)).all()


def test_eigh_grad_matches_torch():
    torch.manual_seed(0)
    S = spd(3, n=8) / 8  # distinct eigenvalues

    def f(S):
        w, V = torch.linalg.eigh(S)
        return V @ torch.diag_embed(w.exp()) @ V.mT, w  # sign-invariant in V

    compare(f, jacobi.install_eigh, S, name="eigh")


def test_eigvalsh_without_grad():
    torch.manual_seed(0)
    S = spd(3, n=8)
    with cpu_as_hpu(jacobi.install_eigh) as calls, torch.no_grad():
        got = torch.linalg.eigvalsh(S)
        assert calls["eigh"] >= 1, dict(calls)
    torch.testing.assert_close(got, torch.linalg.eigvalsh(S), rtol=0, atol=1e-10)


def test_eigh_upper_ignores_the_lower_triangle():
    torch.manual_seed(0)
    S = spd(3, n=8)
    garbage = S + torch.tril(torch.randn_like(S), -1) * 100
    with cpu_as_hpu(jacobi.install_eigh) as calls:
        w, V = torch.linalg.eigh(garbage, UPLO="U")
        assert calls["eigh"] >= 1, dict(calls)
    torch.testing.assert_close(w, torch.linalg.eigvalsh(garbage, UPLO="U"), rtol=0, atol=1e-10)
    torch.testing.assert_close(V @ torch.diag_embed(w) @ V.mT, S, rtol=0, atol=1e-10)


@pytest.mark.parametrize("diagonal", [[1.0] * 5, [1.0, 1.0, 2.0], [3.0, 3.0, 3.0, 1.0]], ids=["I5", "112", "3331"])
def test_eigh_degenerate(diagonal):
    D = torch.diag(torch.tensor(diagonal, dtype=torch.float64))
    n = len(diagonal)
    with cpu_as_hpu(jacobi.install_eigh) as calls:
        w, V = torch.linalg.eigh(D)
        assert calls["eigh"] >= 1, dict(calls)
    assert torch.equal(w, torch.tensor(sorted(diagonal), dtype=torch.float64))
    torch.testing.assert_close(V.mT @ V, torch.eye(n, dtype=D.dtype), rtol=0, atol=1e-12)
    torch.testing.assert_close(V @ torch.diag(w) @ V.mT, D, rtol=0, atol=1e-12)


def test_eigh_compiled_has_no_native_op():
    torch.manual_seed(0)
    S = spd(3, n=8)
    nodes = []
    with cpu_as_hpu(jacobi.install_eigh):
        w, _ = compiled(torch.linalg.eigh, nodes)(S)
    torch.testing.assert_close(w, torch.linalg.eigvalsh(S), rtol=0, atol=1e-10)
    assert "aten._linalg_eigh.default" not in ops(nodes), ops(nodes)


@pytest.mark.parametrize("diagonal", [[1.0] * 3, [1.0, 1.0, 2.0], [3.0, 3.0, 3.0, 1.0]], ids=["I3", "112", "3331"])
@pytest.mark.parametrize("use", ["eigvalsh", "eigenvalues_pow2"])
def test_eigh_degenerate_grad_matches_torch(diagonal, use):
    D = torch.diag(torch.tensor(diagonal, dtype=torch.float64))
    fns = {"eigvalsh": lambda D: torch.linalg.eigvalsh(D).sum(),
           "eigenvalues_pow2": lambda D: torch.linalg.eigh(D).eigenvalues.pow(2).sum()}

    def grad():
        x = D.clone().requires_grad_()
        fns[use](x).backward()
        return x.grad

    want = grad()
    with cpu_as_hpu(jacobi.install_eigh) as calls:
        got = grad()
        assert calls["eigh"] >= 1, dict(calls)
    assert torch.isfinite(got).all()
    torch.testing.assert_close(got, want)


# ---- the TPC Gauss-Jordan kernel: the reference contract, the chunking and the wiring (the CPU stands in for the card)


def f32(*batch, n):
    """A well-conditioned float32 matrix: its triangles invert stably too, with or without a unit diagonal."""
    return 0.3 * torch.randn(*batch, n, n) + 2 * torch.eye(n)


@pytest.fixture(autouse=True)
def eager_tpc(monkeypatch):
    """No habana on the CPU: the TPC op's lazy flush stays off unless a test turns it on."""
    monkeypatch.setenv("PT_HPU_LAZY_MODE", "0")


@pytest.fixture
def gj_hook(monkeypatch):
    """``linalg._GJ`` set to the reference; the list collects one entry per call."""
    calls = []

    def hook(A, B):  # detached, like the real op: no autograd through the kernel
        calls.append(tuple(A.shape))
        return tpc.gj_solve_reference(A.detach(), B.detach())

    monkeypatch.setattr(linalg, "_GJ", hook)
    return calls


@pytest.fixture
def tri_hook(monkeypatch):
    """``linalg._TRI`` set to the reference; the list collects ``(shape, upper, left, unitriangular)`` per call."""
    calls = []

    def hook(A, B, upper=False, left=True, unitriangular=False):
        calls.append((tuple(A.shape), upper, left, unitriangular))
        return tpc.tri_solve_reference(A.detach(), B.detach(), upper, left, unitriangular)

    monkeypatch.setattr(linalg, "_TRI", hook)
    return calls


@pytest.mark.parametrize("m", [1, 3, 40])
@pytest.mark.parametrize("n", [1, 3, 22])
def test_tpc_reference_contract(n, m):
    torch.manual_seed(0)
    A, B = torch.randn(4, n, n), torch.randn(4, n, m)
    X, D, S = tpc.gj_solve_reference(A, B)
    assert X.shape == (4, n, m) and D.shape == (4, n) and S.shape == (4, 2)
    torch.testing.assert_close(X, torch.linalg.solve(A, B), rtol=1e-3, atol=1e-3)
    LU = torch.linalg.lu_factor(A).LU
    torch.testing.assert_close(D, torch.diagonal(LU, dim1=-2, dim2=-1), rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(S[:, 0] * D.prod(-1), torch.linalg.det(A), rtol=1e-3, atol=1e-3)
    assert torch.equal(S[:, 1], torch.zeros(4))


def test_tpc_chunks_columns(monkeypatch):
    torch.manual_seed(0)
    seen = []

    def fake(A, B):
        assert B.shape[-1] <= tpc.MAXN and B.is_contiguous()
        seen.append(B.shape[-1])
        return tpc.gj_solve_reference(A, B)

    monkeypatch.setattr(tpc, "_op", lambda: fake)
    A, B = f32(3, n=5), torch.randn(3, 70, 5).mT
    got = tpc._chunked(A, B)
    want = tpc.gj_solve_reference(A, B)
    assert seen == [64, 6]
    torch.testing.assert_close(got, want)


def test_tpc_gj_solve_flattens_batch_dims(monkeypatch):
    torch.manual_seed(0)
    shapes = []

    def fake(A, B):
        shapes.append((A.dim(), A.is_contiguous(), A.dtype))
        return tpc.gj_solve_reference(A, B)

    monkeypatch.setattr(tpc, "_op", lambda: fake)
    A, B = f32(2, 3, n=4).mT, torch.randn(2, 3, 4, 2)
    X, D, S = tpc.gj_solve(A, B)
    assert shapes == [(3, True, torch.float32)]
    assert X.shape == (2, 3, 4, 2) and D.shape == (2, 3, 4) and S.shape == (2, 3, 2)
    torch.testing.assert_close(X, torch.linalg.solve(A, B), rtol=1e-3, atol=1e-3)


def _tri_case(upper, left, unit):
    A = f32(n=6)
    B = torch.randn(6, 3) if left else torch.randn(3, 6)
    return (lambda a, b: torch.linalg.solve_triangular(a, b, upper=upper, left=left, unitriangular=unit)), (A, B)


GJ_CASES = {
    "solve": (lambda: (lambda a, b: torch.linalg.solve(a, b), (f32(2, n=6), torch.randn(2, 6, 3))),
              linalg.install_solve, "solve"),
    "solve_vector": (lambda: (lambda a, b: torch.linalg.solve(a, b), (f32(2, n=6), torch.randn(2, 6))),
                     linalg.install_solve, "solve"),
    "solve_right": (lambda: (lambda a, b: torch.linalg.solve(a, b, left=False), (f32(2, n=6), torch.randn(2, 3, 6))),
                    linalg.install_solve, "solve"),
    "solve_broadcast": (lambda: (lambda a, b: torch.linalg.solve(a, b), (f32(3, n=6), torch.randn(2, 1, 6, 2))),
                        linalg.install_solve, "solve"),
    "inv": (lambda: (torch.linalg.inv, (f32(2, 3, n=6),)), linalg.install_inv_det, "inv"),
    "det": (lambda: (torch.linalg.det, (f32(2, 3, n=6),)), linalg.install_inv_det, "det"),
    "slogdet": (lambda: (torch.linalg.slogdet, (f32(2, 3, n=6),)), linalg.install_inv_det, "slogdet"),
}

TRI_CASES = {
    "cholesky_solve_lower": (lambda: (lambda b, l: torch.cholesky_solve(b, l),
                                      (torch.randn(2, 6, 3), torch.linalg.cholesky(spd(2, n=6, dtype=torch.float32)))),
                             linalg.install_cholesky_solve, "cholesky_solve"),
    "cholesky_solve_upper": (lambda: (lambda b, l: torch.cholesky_solve(b, l, upper=True),
                                      (torch.randn(2, 6, 3),
                                       torch.linalg.cholesky(spd(2, n=6, dtype=torch.float32)).mH)),
                             linalg.install_cholesky_solve, "cholesky_solve"),
    "tri_n1": (lambda: (lambda a, b: torch.linalg.solve_triangular(a, b, upper=False),
                        (f32(2, n=1), torch.randn(2, 1, 3))),
               linalg.install_solve_triangular, "solve_triangular"),
    "tri_broadcast": (lambda: (lambda a, b: torch.linalg.solve_triangular(a, b, upper=True),
                               (f32(3, n=5), torch.randn(2, 1, 5, 4))),
                      linalg.install_solve_triangular, "solve_triangular"),
    "cholesky_broadcast": (lambda: (lambda b, l: torch.cholesky_solve(b, l),
                                    (torch.randn(2, 1, 5, 3),
                                     torch.linalg.cholesky(spd(1, 4, n=5, dtype=torch.float32)))),
                           linalg.install_cholesky_solve, "cholesky_solve"),
    "tri_m70_chunked": (lambda: (lambda a, b: torch.linalg.solve_triangular(a, b, upper=False),
                                 (f32(2, n=5), torch.randn(2, 5, 70))),
                        linalg.install_solve_triangular, "solve_triangular"),
}
for _upper in (False, True):
    for _left in (True, False):
        for _unit in (False, True):
            _key = f"tri_{'upper' if _upper else 'lower'}_{'left' if _left else 'right'}_{'unit' if _unit else 'full'}"
            TRI_CASES[_key] = (
                (lambda u=_upper, l=_left, t=_unit: _tri_case(u, l, t)), linalg.install_solve_triangular,
                "solve_triangular")


@pytest.mark.parametrize("case", list(GJ_CASES))
def test_linalg_routes_through_gj(case, gj_hook):
    torch.manual_seed(0)
    make, install, name = GJ_CASES[case]
    f, inputs = make()
    compare(f, install, *inputs, name=name, rtol=2e-3, atol=2e-3)
    assert gj_hook, "the kernel hook was not called"


@pytest.mark.parametrize("case", list(TRI_CASES))
def test_linalg_routes_through_tri(case, tri_hook):
    torch.manual_seed(0)
    make, install, name = TRI_CASES[case]
    f, inputs = make()
    compare(f, install, *inputs, name=name, rtol=2e-3, atol=2e-3)
    assert tri_hook, "the tri kernel hook was not called"


def test_gj_is_declined_past_maxn_on_hpu_and_without_the_kernel(monkeypatch):
    A = torch.randn(2, 4, 4)
    takes = linalg._tpc_takes("hpu", linalg._takes("hpu", "solve"))
    monkeypatch.setattr(tpc, "ready", lambda: False)
    assert takes(A) is False
    monkeypatch.setattr(tpc, "ready", lambda: True)
    assert takes(A) is True
    assert takes(torch.randn(tpc.MAXN + 1, tpc.MAXN + 1)) is False
    assert takes(torch.randn(tpc.MAXN, tpc.MAXN)) is True
    large = linalg._tpc_takes("hpu", linalg._takes("hpu", "solve"), large=True)
    n = tpc.LARGE_ANY_BATCH_N
    assert large(torch.randn(n, n)) is True and large(torch.randn(tpc.MAXN + 1, tpc.MAXN + 1)) is True
    assert large(torch.randn(tpc.LARGE_MIN_BATCH - 1, n + 1, n + 1)) is False  # past it, a small batch: fallback
    assert large(torch.randn(tpc.LARGE_MIN_BATCH, n + 1, n + 1)) is True
    assert large(torch.randn(tpc.LARGE_MIN_BATCH, tpc.LARGE_MAXN + 1, tpc.LARGE_MAXN + 1)) is False
    cpu = linalg._tpc_takes("cpu", linalg._takes("cpu", "solve"))
    monkeypatch.setattr(tpc, "ready", lambda: False)
    assert cpu(torch.randn(tpc.MAXN + 1, tpc.MAXN + 1)) is True  # the CPU stand-in keeps the composite


def test_gj_used_to_large_maxn(gj_hook):
    torch.manual_seed(0)
    with cpu_as_hpu(linalg.install_solve):
        torch.linalg.solve(f32(2, n=tpc.MAXN + 1), torch.randn(2, tpc.MAXN + 1, 2))
    assert gj_hook == [(2, tpc.MAXN + 1, tpc.MAXN + 1)]


def test_gj_not_used_past_large_maxn(monkeypatch, gj_hook):
    monkeypatch.setattr(tpc, "LARGE_MAXN", 80)  # the composite past it: keep the CPU test short
    torch.manual_seed(0)
    with cpu_as_hpu(linalg.install_solve):
        torch.linalg.solve(f32(2, n=81), torch.randn(2, 81, 2))
    assert gj_hook == []


# ---- the in-place kernel for 64 < n <= LARGE_MAXN, through its torch reference


def permuted(b, n, seed=0):
    """Near-identity rows in a random order: pivoting across the whole height is needed, both det signs occur."""
    g = torch.Generator().manual_seed(seed)
    M = torch.eye(n) + 0.05 * torch.randn(b, n, n, generator=g) / n ** 0.5
    perm = torch.stack([torch.randperm(n, generator=g) for _ in range(b)])
    return M.gather(1, perm[..., None].expand(b, n, n))


@pytest.fixture
def large_reference(monkeypatch):
    calls = []

    def fake(A):
        calls.append(tuple(A.shape))
        return tpc.gj_large_reference(A)

    monkeypatch.setattr(tpc, "_large_op", lambda: fake)
    return calls


@pytest.mark.parametrize("n", [65, 100, 130])
def test_large_reference_contract(n, large_reference):
    A = permuted(3, n).double()
    R, P, Q, D, S = tpc.gj_large_reference(A)
    assert R.shape == (3, n, 64 * ((n + 63) // 64)) and torch.all(R[..., n:] == 0)
    inv = R[..., :n].gather(1, P.long()[..., None].expand(3, n, n)).gather(2, Q.long()[:, None, :].expand(3, n, n))
    torch.testing.assert_close(inv, torch.linalg.inv(A))
    torch.testing.assert_close(D.prod(-1).abs(), torch.linalg.det(A).abs())
    assert torch.equal(Q.long(), P.long().argsort(-1)) and torch.equal(S[:, 1], torch.zeros(3, dtype=A.dtype))


@pytest.mark.parametrize("batch", [(), (2,), (2, 3)])
def test_large_gj_inv_and_solve(batch, large_reference):
    n = 70
    A = permuted(int(torch.tensor(batch).prod()) if batch else 1, n).reshape(*batch, n, n)
    B = torch.randn(*batch, n, 5, dtype=A.dtype)
    X, D, S = tpc.gj_inv(A)
    torch.testing.assert_close(X, torch.linalg.inv(A), rtol=1e-4, atol=1e-4)
    sign, logabsdet = torch.linalg.slogdet(A)
    torch.testing.assert_close(S[..., 0], sign)
    torch.testing.assert_close(D.abs().log().sum(-1), logabsdet, rtol=1e-4, atol=1e-4)
    Y, _, _ = tpc.gj_solve(A, B)
    torch.testing.assert_close(Y, torch.linalg.solve(A, B), rtol=1e-4, atol=1e-4)
    assert large_reference and all(shape[-1] == n for shape in large_reference)


def test_large_signs_both_occur(large_reference):
    A = permuted(16, 66)
    _, _, S = tpc.gj_inv(A)
    want = torch.linalg.slogdet(A).sign
    assert torch.equal(S[:, 0], want) and bool((want > 0).any()) and bool((want < 0).any())


def test_large_singular_info(large_reference):
    A = permuted(3, 70)
    A[1, :, 7] = 0  # a zero column: no pivot at step 7
    _, D, S = tpc.gj_inv(A)
    assert S[1, 1] == 8 and D[1].prod() == 0
    assert torch.equal(S[[0, 2], 1], torch.zeros(2))


DOUBLE_BACKWARD = {
    "solve": (linalg.install_solve, lambda a, b: torch.linalg.solve(a, b),
              lambda: (general(n=4), torch.randn(4, 2, dtype=torch.float64))),
    "solve_right": (linalg.install_solve, lambda a, b: torch.linalg.solve(a, b, left=False),
                    lambda: (general(n=4), torch.randn(2, 4, dtype=torch.float64))),
}


@pytest.mark.parametrize("case", list(DOUBLE_BACKWARD))
def test_double_backward_through_gj(case, gj_hook):
    torch.manual_seed(0)
    install, f, make = DOUBLE_BACKWARD[case]
    inputs = [x.requires_grad_() for x in make()]
    with cpu_as_hpu(install) as calls:
        assert torch.autograd.gradgradcheck(f, inputs, atol=1e-5, rtol=1e-4)
        assert sum(calls.values()) >= 1 and gj_hook


DOUBLE_BACKWARD_TRI = {
    "solve_triangular": (linalg.install_solve_triangular,
                         lambda a, b: torch.linalg.solve_triangular(a, b, upper=True),
                         lambda: (general(n=4), torch.randn(4, 2, dtype=torch.float64))),
    "solve_triangular_left_false": (linalg.install_solve_triangular,
                                    lambda a, b: torch.linalg.solve_triangular(a, b, upper=False, left=False),
                                    lambda: (general(n=4), torch.randn(2, 4, dtype=torch.float64))),
    "cholesky_solve_lower": (linalg.install_cholesky_solve, lambda b, l: torch.cholesky_solve(b, torch.tril(l)),
                             lambda: (torch.randn(4, 2, dtype=torch.float64), torch.linalg.cholesky(spd(n=4)))),
    "cholesky_solve_upper": (linalg.install_cholesky_solve,
                             lambda b, l: torch.cholesky_solve(b, torch.triu(l), upper=True),
                             lambda: (torch.randn(4, 2, dtype=torch.float64), torch.linalg.cholesky(spd(n=4)).mH)),
}


@pytest.mark.parametrize("case", list(DOUBLE_BACKWARD_TRI))
def test_double_backward_through_tri(case, tri_hook):
    torch.manual_seed(0)
    install, f, make = DOUBLE_BACKWARD_TRI[case]
    inputs = [x.requires_grad_() for x in make()]
    with cpu_as_hpu(install) as calls:
        assert torch.autograd.gradgradcheck(f, inputs, atol=1e-5, rtol=1e-4)
        assert sum(calls.values()) >= 1 and tri_hook


def singular_hook(A, B):
    """The kernel's contract on a singular A: D has a 0 and later entries stay finite; S carries the info."""
    X, D, S = tpc.gj_solve_reference(A.detach(), B.detach())
    return X, D, S


def test_singular_det_is_zero_with_gj(monkeypatch):
    monkeypatch.setattr(linalg, "_GJ", singular_hook)
    A = torch.randn(3, 4, 4)
    A[1] = 0  # the zero matrix: native and LU-of-A agree on the message
    A[2, 2] = 0  # a zero row: a zero pivot at some step
    with pytest.raises(RuntimeError) as native_solve:
        torch.linalg.solve(A, torch.ones(3, 4, 2))
    with pytest.raises(RuntimeError) as native_inv:
        torch.linalg.inv(A)
    with cpu_as_hpu(linalg.install_inv_det), cpu_as_hpu(linalg.install_solve):
        D = singular_hook(A, torch.ones(3, 4, 1))[1]
        assert (D[2] == 0).any() and torch.isfinite(D).all()
        det = torch.linalg.det(A)
        assert torch.equal(det[1:], torch.zeros(2)) and det[0] != 0
        sign, logabsdet = torch.linalg.slogdet(A)
        assert torch.equal(sign[1:], torch.zeros(2)) and torch.isinf(logabsdet[1:]).all()
        with pytest.raises(RuntimeError, match=re.escape(str(native_solve.value))):
            torch.linalg.solve(A, torch.ones(3, 4, 2))
        with pytest.raises(RuntimeError, match=re.escape(str(native_inv.value))):
            torch.linalg.inv(A)


def test_info_from_the_kernels_second_lane(gj_hook):
    A = f32(3, 2, n=4)
    A[1, 0] = 0
    want = torch.linalg.solve_ex(A, torch.ones(3, 2, 4, 1)).info
    with cpu_as_hpu(linalg.install_solve):
        got = torch.linalg.solve_ex(A, torch.ones(3, 2, 4, 1)).info
    assert got.dtype == torch.int32 and torch.equal(got, want) and want.any()


def test_lu_placeholder_has_own_storage():
    A = f32(2, 3, n=4)
    LU, pivots = linalg._unused_factors(A)
    assert LU.shape == A.shape and LU.dtype == A.dtype and pivots.shape == A.shape[:-1]
    assert LU.untyped_storage().data_ptr() != A.untyped_storage().data_ptr()


def test_chunks_and_flat_inv_contract(monkeypatch):
    monkeypatch.setattr(tpc, "_inv_op", lambda: lambda A: tpc.gj_solve_reference(
        A, torch.eye(A.shape[-1]).expand_as(A)))
    A = f32(2, 3, n=5)
    X, D, S = tpc.gj_inv(A)
    torch.testing.assert_close(X, torch.linalg.inv(A), rtol=1e-3, atol=1e-3)
    assert D.shape == (2, 3, 5) and S.shape == (2, 3, 2)


def test_ready_declines_after_a_build_failure(monkeypatch):
    monkeypatch.setattr(tpc, "_INSTALLED", True)
    monkeypatch.setattr(tpc, "_device_name", lambda: "GAUDI2")
    tpc._built.cache_clear()

    def boom():
        raise RuntimeError("no toolchain")

    monkeypatch.setattr(tpc, "_load", boom)
    with pytest.warns(UserWarning, match="no toolchain"):
        assert tpc.ready() is False
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert tpc.ready() is False
    tpc._built.cache_clear()


def test_singular_info_from_pivots(gj_hook):
    A = torch.zeros(3, 3)
    with pytest.raises(RuntimeError) as native_solve:
        torch.linalg.solve(A, torch.ones(3, 2))
    with pytest.raises(RuntimeError) as native_inv:
        torch.linalg.inv(A)
    with cpu_as_hpu(linalg.install_solve) as calls:
        with pytest.raises(RuntimeError, match=re.escape(str(native_solve.value))):
            torch.linalg.solve(A, torch.ones(3, 2))
        assert calls["solve"] >= 1, dict(calls)
    with cpu_as_hpu(linalg.install_inv_det) as calls:
        with pytest.raises(RuntimeError, match=re.escape(str(native_inv.value))):
            torch.linalg.inv(A)
        assert torch.equal(torch.linalg.det(A), torch.zeros(()))
        assert calls["inv"] >= 1 and calls["det"] >= 1, dict(calls)
    assert len(gj_hook) >= 3


def test_install_kernel_path(monkeypatch):
    default = "/usr/lib/habanalabs/libtpc_kernels.so"
    monkeypatch.setattr(tpc, "_INSTALLED", False)
    monkeypatch.setattr(tpc, "_LATE_WARNED", False)
    monkeypatch.setattr(tpc, "_initialized", lambda: False)
    monkeypatch.delenv("GC_KERNEL_PATH", raising=False)
    assert tpc.install_kernel_path() is True and tpc.available() is True
    assert os.environ["GC_KERNEL_PATH"] == f"{tpc.LIB}:{default}"
    assert tpc.install_kernel_path() is True
    assert os.environ["GC_KERNEL_PATH"] == f"{tpc.LIB}:{default}"
    monkeypatch.setenv("GC_KERNEL_PATH", "")  # empty is unset
    assert tpc.install_kernel_path() is True
    assert os.environ["GC_KERNEL_PATH"] == f"{tpc.LIB}:{default}"
    monkeypatch.setenv("GC_KERNEL_PATH", "/x/other.so")
    assert tpc.install_kernel_path() is True
    assert os.environ["GC_KERNEL_PATH"] == f"{tpc.LIB}:/x/other.so"
    assert tpc.install_kernel_path() is True
    assert os.environ["GC_KERNEL_PATH"] == f"{tpc.LIB}:/x/other.so"

    monkeypatch.setattr(tpc, "_INSTALLED", False)
    monkeypatch.setattr(tpc, "_initialized", lambda: True)
    monkeypatch.setenv("GC_KERNEL_PATH", "/x/other.so")
    with pytest.warns(UserWarning, match="used before hpu_ops.enable") as record:
        assert tpc.install_kernel_path() is False
    assert tpc.available() is False and os.environ["GC_KERNEL_PATH"] == "/x/other.so"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert tpc.install_kernel_path() is False  # warned once
    assert len(record) == 1
    assert str(record[0].message) == ("hpu_ops: the HPU was used before hpu_ops.enable(); TPC linalg kernels are off "
                                      "for this process (call enable() before the first HPU op). The linalg ops then "
                                      "run through hpu-ops' own CPU route, not Habana's fallback.")


@pytest.mark.parametrize("lazy, flushes", [("1", 1), ("0", 0)])
def test_tpc_flushes_the_lazy_graph_before_the_op(monkeypatch, lazy, flushes):
    marks = []
    core = types.SimpleNamespace(mark_step=lambda: marks.append(1))
    torch_mod = types.SimpleNamespace(core=core)
    hf = types.SimpleNamespace(torch=torch_mod)
    for name, mod in (("habana_frameworks", hf), ("habana_frameworks.torch", torch_mod),
                      ("habana_frameworks.torch.core", core)):
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.setenv("PT_HPU_LAZY_MODE", lazy)
    monkeypatch.setattr(tpc, "_op", lambda: tpc.gj_solve_reference)
    monkeypatch.setattr(tpc, "_inv_op",
                        lambda: lambda A: tpc.gj_solve_reference(A, torch.eye(A.shape[-1]).expand_as(A)))
    tpc.gj_solve(f32(2, n=4), torch.randn(2, 4, 3))
    assert len(marks) == flushes
    marks.clear()
    tpc.gj_inv(f32(2, n=4))
    assert len(marks) == flushes


def test_flush_comes_before_the_op_only(monkeypatch):
    order = []
    core = types.SimpleNamespace(mark_step=lambda: order.append("flush"))
    torch_mod = types.SimpleNamespace(core=core)
    for name, mod in (("habana_frameworks", types.SimpleNamespace(torch=torch_mod)),
                      ("habana_frameworks.torch", torch_mod), ("habana_frameworks.torch.core", core)):
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.setenv("PT_HPU_LAZY_MODE", "1")
    x = torch.ones(2)
    out = tpc._call(lambda t: order.append("op") or t, x)
    assert torch.equal(out, x) and order == ["flush", "op"]


def test_flush_is_skipped_when_compiling_or_without_habana(monkeypatch):
    monkeypatch.setenv("PT_HPU_LAZY_MODE", "1")
    monkeypatch.setitem(sys.modules, "habana_frameworks", None)  # import raises ImportError
    monkeypatch.setitem(sys.modules, "habana_frameworks.torch", None)
    monkeypatch.setitem(sys.modules, "habana_frameworks.torch.core", None)
    tpc._flush()  # no Habana: no error
    marks = []
    core = types.SimpleNamespace(mark_step=lambda: marks.append(1))
    monkeypatch.setitem(sys.modules, "habana_frameworks.torch.core", core)
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    tpc._flush()
    assert marks == []


def test_lazy_cpp_registers_eigh_with_the_legacy_api():
    assert "UserCustomOpDescriptor" not in tpc.CPP_LAZY
    assert "HabanaCustomOpDescriptor::getCustomOpDescriptor(\"linalg_tpc::jacobi_eigh\")" in tpc.CPP_LAZY
    assert "UserCustomOpDescriptor::getUserCustomOpDescriptor(\"linalg_tpc::jacobi_eigh\")" in tpc.CPP


def test_tri_solve_cpp_strings():
    assert "UserCustomOpDescriptor" not in tpc.CPP_LAZY
    for op in ("tri_solve_lower", "tri_solve_upper", "tri_solve_lower_unit", "tri_solve_upper_unit"):
        assert f'"linalg_tpc::{op}"' in tpc.CPP
        assert f'm.impl("{op}", tri_meta_tensor)' in tpc.CPP
        assert f'"linalg_tpc::{op}"' in tpc.CPP_LAZY
        assert f'm.impl("{op}", {op})' in tpc.CPP_LAZY


@pytest.mark.parametrize("upper", [False, True])
@pytest.mark.parametrize("left", [True, False])
@pytest.mark.parametrize("unitriangular", [False, True])
@pytest.mark.parametrize("batch", [(), (4,)])
@pytest.mark.parametrize("m", [1, 3, 40])
@pytest.mark.parametrize("n", [1, 3, 22])
def test_tri_solve_reference_contract(n, m, batch, upper, left, unitriangular):
    torch.manual_seed(0)
    A = f32(*batch, n=n)
    # Poison unused opposite triangle (and diagonal when unitriangular) with NaN to prove it is never read
    mask_other = torch.ones(n, n, dtype=torch.bool).tril(-1) if upper else torch.ones(n, n, dtype=torch.bool).triu(1)
    A[..., mask_other] = float("nan")
    if unitriangular:
        A.diagonal(dim1=-2, dim2=-1).zero_()
    B = torch.randn(*batch, n, m) if left else torch.randn(*batch, m, n)
    got = tpc.tri_solve_reference(A, B, upper=upper, left=left, unitriangular=unitriangular)
    want = torch.linalg.solve_triangular(A, B, upper=upper, left=left, unitriangular=unitriangular)
    assert got.shape == want.shape
    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("upper", [False, True])
@pytest.mark.parametrize("left", [True, False])
@pytest.mark.parametrize("unitriangular", [False, True])
def test_tpc_tri_solve_selects_elf_and_transposes(monkeypatch, upper, left, unitriangular):
    torch.manual_seed(0)
    picked = []

    def fake_tri_op(u_eff, unit):
        picked.append((u_eff, unit))

        def fn(A, B):
            assert A._base is None and B._base is None
            assert A.is_contiguous() and B.is_contiguous()
            return tpc.tri_solve_reference(A, B, upper=u_eff, left=True, unitriangular=unit)

        return fn

    monkeypatch.setattr(tpc, "_tri_op", fake_tri_op)
    A = f32(2, n=1).mT
    B = (torch.randn(2, 3, 1) if left else torch.randn(2, 1, 3)).mT
    got = tpc.tri_solve(A, B, upper=upper, left=left, unitriangular=unitriangular)
    want = torch.linalg.solve_triangular(A, B, upper=upper, left=left, unitriangular=unitriangular)
    assert picked == [(upper if left else not upper, unitriangular)]
    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("left", [True, False])
def test_tpc_tri_solve_chunks_columns(monkeypatch, left):
    torch.manual_seed(0)
    seen = []

    def fake_op(u, unit):
        def fake(A, B):
            assert B.shape[-1] <= tpc.MAXN and B.is_contiguous() and B._base is None
            seen.append(B.shape[-1])
            return tpc.tri_solve_reference(A, B, upper=u, left=True, unitriangular=unit)

        return fake

    monkeypatch.setattr(tpc, "_tri_op", fake_op)
    A = f32(3, n=5)
    B = torch.randn(3, 5, 70) if left else torch.randn(3, 70, 5)
    got = tpc.tri_solve(A, B, upper=False, left=left, unitriangular=False)
    want = torch.linalg.solve_triangular(A, B, upper=False, left=left, unitriangular=False)
    assert seen == [64, 6]
    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-4)


def test_tpc_tri_solve_shapes_and_broadcast(monkeypatch):
    torch.manual_seed(0)
    seen = []

    def fake_op(u, unit):
        def fn(A, B):
            seen.append((A.shape, B.shape, A.is_contiguous(), B.is_contiguous(), A.dtype, A._base is None,
                         B._base is None))
            return tpc.tri_solve_reference(A, B, upper=u, left=True, unitriangular=unit)

        return fn

    monkeypatch.setattr(tpc, "_tri_op", fake_op)
    # 2D unbatched -> batch == () -> squeezed back to 2D
    A2, B2 = f32(n=4).mT, torch.randn(4, 3)
    got2 = tpc.tri_solve(A2, B2, upper=True)
    assert got2.shape == (4, 3)
    torch.testing.assert_close(got2, torch.linalg.solve_triangular(A2, B2, upper=True), rtol=1e-4, atol=1e-4)

    # 3D batch == (1,) -> remains 3D (1, n, m)
    A1, B1 = f32(1, n=4), torch.randn(1, 4, 3)
    got1 = tpc.tri_solve(A1, B1, upper=False)
    assert got1.shape == (1, 4, 3)

    # Asymmetric batch broadcast (3,) vs (2, 1) -> (2, 3) -> flattened to (6, ...)
    Ab, Bb = f32(3, n=4), torch.randn(2, 1, 4, 5)
    gotb = tpc.tri_solve(Ab, Bb, upper=False)
    assert gotb.shape == (2, 3, 4, 5)
    torch.testing.assert_close(gotb, torch.linalg.solve_triangular(Ab, Bb, upper=False), rtol=1e-4, atol=1e-4)
    assert all(ic and bc and dt == torch.float32 and ab and bb for _, _, ic, bc, dt, ab, bb in seen)


def test_unflatten_2d_and_batched_for_gj(monkeypatch):
    torch.manual_seed(0)
    monkeypatch.setattr(tpc, "_op", lambda: tpc.gj_solve_reference)
    monkeypatch.setattr(tpc, "_inv_op",
                        lambda: lambda A: tpc.gj_solve_reference(A, torch.eye(A.shape[-1]).expand_as(A)))
    A2, B2 = f32(n=4), torch.randn(4, 3)
    X2, D2, S2 = tpc.gj_solve(A2, B2)
    assert X2.shape == (4, 3) and D2.shape == (4,) and S2.shape == (2,)
    Xi2, Di2, Si2 = tpc.gj_inv(A2)
    assert Xi2.shape == (4, 4) and Di2.shape == (4,) and Si2.shape == (2,)
    # batch=(1,) preserves 3D
    A1, B1 = f32(1, n=4), torch.randn(1, 4, 3)
    X1, D1, S1 = tpc.gj_solve(A1, B1)
    assert X1.shape == (1, 4, 3) and D1.shape == (1, 4) and S1.shape == (1, 2)
    Xi1, Di1, Si1 = tpc.gj_inv(A1)
    assert Xi1.shape == (1, 4, 4) and Di1.shape == (1, 4) and Si1.shape == (1, 2)


@pytest.mark.parametrize("lazy, flushes", [("1", 1), ("0", 0)])
def test_tpc_tri_solve_flushes_lazy(monkeypatch, lazy, flushes):
    marks = []
    core = types.SimpleNamespace(mark_step=lambda: marks.append(1))
    torch_mod = types.SimpleNamespace(core=core)
    hf = types.SimpleNamespace(torch=torch_mod)
    for name, mod in (("habana_frameworks", hf), ("habana_frameworks.torch", torch_mod),
                      ("habana_frameworks.torch.core", core)):
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.setenv("PT_HPU_LAZY_MODE", lazy)
    monkeypatch.setattr(tpc, "_tri_op", lambda u, unit: lambda A, B: tpc.tri_solve_reference(A, B, u, True, unit))
    tpc.tri_solve(f32(2, n=4), torch.randn(2, 4, 3))
    assert len(marks) == flushes


def test_cholesky_solve_two_pass_and_ignores_unused_triangle(tri_hook):
    torch.manual_seed(0)
    L_clean = torch.linalg.cholesky(spd(2, n=5, dtype=torch.float32))
    L_dirty = L_clean.clone()
    L_dirty[..., torch.ones(5, 5, dtype=torch.bool).triu(1)] = float("nan")
    B = torch.randn(2, 5, 3)
    with cpu_as_hpu(linalg.install_cholesky_solve), torch.no_grad():
        got = torch.cholesky_solve(B, L_dirty, upper=False)
    want = torch.cholesky_solve(B, L_clean, upper=False)
    assert len(tri_hook) == 2
    assert [(u, left, unit) for _, u, left, unit in tri_hook] == [(False, True, False), (True, True, False)]
    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-4)

    tri_hook.clear()
    U_clean = L_clean.mH.clone()
    U_dirty = U_clean.clone()
    U_dirty[..., torch.ones(5, 5, dtype=torch.bool).tril(-1)] = float("nan")
    with cpu_as_hpu(linalg.install_cholesky_solve), torch.no_grad():
        got_u = torch.cholesky_solve(B, U_dirty, upper=True)
    want_u = torch.cholesky_solve(B, U_clean, upper=True)
    assert len(tri_hook) == 2
    assert [(u, left, unit) for _, u, left, unit in tri_hook] == [(False, True, False), (True, True, False)]
    torch.testing.assert_close(got_u, want_u, rtol=1e-4, atol=1e-4)


def test_tri_not_used_past_maxn(tri_hook):
    torch.manual_seed(0)
    n = tpc.MAXN + 1
    with cpu_as_hpu(linalg.install_solve_triangular):
        torch.linalg.solve_triangular(f32(2, n=n), torch.randn(2, n, 2), upper=False)
    with cpu_as_hpu(linalg.install_cholesky_solve):
        torch.cholesky_solve(torch.randn(2, n, 2), f32(2, n=n).tril())
    assert tri_hook == []


def test_tri_bad_shapes_raise_like_torch(tri_hook):
    A = f32(n=4)
    with cpu_as_hpu(linalg.install_solve_triangular):
        with pytest.raises((RuntimeError, ValueError)):
            torch.linalg.solve_triangular(A, torch.randn(4), upper=False)
        with pytest.raises((RuntimeError, ValueError)):
            torch.linalg.solve_triangular(A, torch.randn(5, 2), upper=False)
        with pytest.raises((RuntimeError, ValueError)):
            torch.linalg.solve_triangular(A, torch.randn(2, 5), upper=False, left=False)
    with cpu_as_hpu(linalg.install_cholesky_solve):
        with pytest.raises((RuntimeError, ValueError)):
            torch.cholesky_solve(torch.randn(4), A)
        with pytest.raises((RuntimeError, ValueError)):
            torch.cholesky_solve(torch.randn(5, 2), A)
    assert tri_hook == []


@pytest.mark.parametrize("lazy", ["1", "0"])
def test_tpc_call_copies_outputs_in_lazy_mode(monkeypatch, lazy):
    """Lazy mode: a custom op's output read with no ATen op after it is wrong on bridge 1.21.4, so _call clones."""
    core = types.SimpleNamespace(mark_step=lambda: None)
    torch_mod = types.SimpleNamespace(core=core)
    hf = types.SimpleNamespace(torch=torch_mod)
    for name, mod in (("habana_frameworks", hf), ("habana_frameworks.torch", torch_mod),
                      ("habana_frameworks.torch.core", core)):
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.setenv("PT_HPU_LAZY_MODE", lazy)
    x, y = torch.randn(3), torch.randn(2)
    one = tpc._call(lambda: x)
    pair = tpc._call(lambda: (x, y))
    copied = lazy == "1"
    assert (one is not x) is copied and torch.equal(one, x)
    assert all((a is not b) is copied and torch.equal(a, b) for a, b in zip(pair, (x, y)))


def test_tpc_call_skips_mark_step_while_compiling(monkeypatch):
    marks = []
    core = types.SimpleNamespace(mark_step=lambda: marks.append(1))
    torch_mod = types.SimpleNamespace(core=core)
    hf = types.SimpleNamespace(torch=torch_mod)
    for name, mod in (("habana_frameworks", hf), ("habana_frameworks.torch", torch_mod),
                      ("habana_frameworks.torch.core", core)):
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.setenv("PT_HPU_LAZY_MODE", "1")
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    meta_a = torch.empty((2, 4, 4), device="meta")
    meta_b = torch.empty((2, 4, 3), device="meta")
    sentinel_meta = meta_b.new_empty(meta_b.size())
    out_meta = tpc._call(lambda a, b: sentinel_meta, meta_a, meta_b)
    assert out_meta is sentinel_meta and marks == []


def test_solve_triangular_and_cholesky_b_only_grad_route_through_tri(tri_hook):
    torch.manual_seed(0)
    A = f32(2, n=6)
    B = torch.randn(2, 6, 3)
    W = torch.randn(2, 6, 3)

    b_ref = B.clone().requires_grad_()
    (torch.linalg.solve_triangular(A, b_ref, upper=False) * W).sum().backward()
    b_got = B.clone().requires_grad_()
    with cpu_as_hpu(linalg.install_solve_triangular):
        (torch.linalg.solve_triangular(A, b_got, upper=False) * W).sum().backward()
    assert len(tri_hook) == 2
    torch.testing.assert_close(b_got.grad, b_ref.grad, rtol=2e-3, atol=2e-3)

    tri_hook.clear()
    C = torch.linalg.cholesky(spd(2, n=6, dtype=torch.float32))
    b_ref2 = B.clone().requires_grad_()
    (torch.cholesky_solve(b_ref2, C) * W).sum().backward()
    b_got2 = B.clone().requires_grad_()
    with cpu_as_hpu(linalg.install_cholesky_solve):
        (torch.cholesky_solve(b_got2, C) * W).sum().backward()
    assert len(tri_hook) == 4
    torch.testing.assert_close(b_got2.grad, b_ref2.grad, rtol=2e-3, atol=2e-3)
# ---- final-review fixes: device check, opt-out, key coupling, functorch, singular triangles, numpy-free import


@pytest.mark.parametrize("name, ok", [("GAUDI2", True), ("Gaudi2", True), ("GAUDI3", False), ("GAUDI", False)])
def test_ready_requires_a_gaudi2(monkeypatch, name, ok):
    monkeypatch.setattr(tpc, "_INSTALLED", True)
    monkeypatch.setattr(tpc, "_built", lambda: True)
    monkeypatch.setattr(tpc, "_device_name", lambda: name)
    assert tpc.ready() is ok


def test_non_gaudi2_declines_so_the_ops_keep_the_native_path(monkeypatch):
    monkeypatch.setattr(tpc, "_INSTALLED", True)
    monkeypatch.setattr(tpc, "_built", lambda: True)
    monkeypatch.setattr(tpc, "_device_name", lambda: "GAUDI3")
    takes = linalg._tpc_takes("hpu", linalg._takes("hpu", "solve"))
    assert takes(torch.randn(2, 4, 4)) is False


def test_import_hpu_ops_does_not_need_numpy():
    import subprocess
    code = "import sys; sys.modules['numpy'] = None; import hpu_ops"
    subprocess.run([sys.executable, "-c", code], check=True, cwd=os.path.dirname(os.path.dirname(__file__)))


def test_opt_out_env_turns_the_kernels_off(monkeypatch):
    monkeypatch.setattr(tpc, "_INSTALLED", True)
    monkeypatch.setattr(tpc, "_initialized", lambda: False)
    monkeypatch.setenv("HPU_OPS_TPC", "0")
    assert tpc.install_kernel_path() is False and tpc.available() is False


def test_install_kernel_path_declines_without_the_library(monkeypatch, tmp_path):
    monkeypatch.setattr(tpc, "_INSTALLED", False)
    monkeypatch.setattr(tpc, "_LATE_WARNED", False)
    monkeypatch.setattr(tpc, "_initialized", lambda: False)
    monkeypatch.setattr(tpc, "LIB", tmp_path / "missing.so")
    monkeypatch.delenv("HPU_OPS_TPC", raising=False)
    with pytest.warns(UserWarning, match="not found"):
        assert tpc.install_kernel_path() is False
    assert tpc.available() is False


def test_backend_kernel_is_registered_only_with_its_autograd_kernel(monkeypatch):
    from hpu_ops import _dispatch

    op = torch.ops.aten.linalg_inv_ex.default
    groups = [[(op, "Autograd{}", lambda *a, **k: None), (op, "{}", lambda *a, **k: None)]]
    taken, seen = set(), []
    monkeypatch.setattr(torch._C, "_dispatch_has_kernel_for_dispatch_key", lambda name, key: key in taken)
    monkeypatch.setattr(torch.library.Library, "impl", lambda self, op, fn, key: seen.append(key))
    _dispatch.register_paired("cpu", groups)
    assert sorted(seen) == ["AutogradCPU", "CPU"]
    seen.clear()
    taken.add("AutogradCPU")  # Habana's own autograd kernel: ours would split the two keys, so neither is registered
    _dispatch.register_paired("cpu", groups)
    assert seen == []
    taken.clear()
    taken.add("CPU")  # a taken backend key alone only skips that key (the CPU stand-in's case)
    _dispatch.register_paired("cpu", groups)
    assert seen == ["AutogradCPU"]


def test_functorch_jacrev_and_vmap_take_the_native_path(gj_hook):  # the kernel hook may serve calls below the transform
    A, b = general(3, n=4, dtype=torch.float32), torch.randn(3, 4)
    want = torch.func.jacrev(torch.linalg.solve)(A, b)
    with cpu_as_hpu(linalg.install_solve):
        got = torch.func.jacrev(torch.linalg.solve)(A, b)  # the transformed call is declined: no setup_context error
        v = torch.func.vmap(torch.linalg.solve)(A, b)
    torch.testing.assert_close(got, want)
    torch.testing.assert_close(v, torch.linalg.solve(A, b))


def test_forward_ad_takes_the_native_path(gj_hook):
    import torch.autograd.forward_ad as fwAD
    A = general(n=4, dtype=torch.float32)
    t = torch.randn(4, 4)
    with fwAD.dual_level():
        want = fwAD.unpack_dual(torch.linalg.det(fwAD.make_dual(A, t))).tangent
    with cpu_as_hpu(linalg.install_inv_det):
        with fwAD.dual_level():
            got = fwAD.unpack_dual(torch.linalg.det(fwAD.make_dual(A, t))).tangent
        jv = torch.func.jvp(torch.linalg.det, (A,), (t,))[1]
    torch.testing.assert_close(got, want)
    torch.testing.assert_close(jv, want)
    assert gj_hook == []


def test_singular_triangle_gives_nan_like_torch(monkeypatch):
    monkeypatch.setattr(linalg, "_TRI", tpc.tri_solve_reference)
    A = torch.tril(torch.randn(2, 4, 4)) + 3 * torch.eye(4)
    A[1, 2] = 0  # a zero row: an exact zero pivot
    B = torch.randn(2, 4, 3)
    want = torch.linalg.solve_triangular(A, B, upper=False)
    assert torch.isnan(want[1]).any() or torch.isinf(want[1]).any()
    with cpu_as_hpu(linalg.install_solve_triangular):
        got = torch.linalg.solve_triangular(A, B, upper=False)
    assert not torch.isfinite(got[1]).all() and torch.isfinite(got[0]).all()
    torch.testing.assert_close(got[0], want[0])
    L = A.clone()
    with cpu_as_hpu(linalg.install_cholesky_solve):
        out = torch.cholesky_solve(B, L, upper=False)
    assert not torch.isfinite(out[1]).all() and torch.isfinite(out[0]).all()


def test_tpc_eigh_1x1_needs_no_kernel(monkeypatch):
    """n = 1 does not compile in lazy mode: w is the entry, V is 1, and the kernel op is never built."""
    monkeypatch.setattr(tpc, "_eigh_op", lambda: pytest.fail("the kernel was used for n = 1"))
    monkeypatch.setattr(tpc, "_eigh_refine_op", lambda: pytest.fail("the refine kernel was used for n = 1"))
    A = torch.randn(4, 1, 1)
    for upper in (False, True):
        w, V = tpc.eigh(A, upper)
        assert w.shape == (4, 1) and V.shape == (4, 1, 1) and w.dtype == V.dtype == torch.float32
        assert torch.equal(w, A[..., 0, :]) and torch.equal(V, torch.ones(4, 1, 1))
        assert w.data_ptr() != A.data_ptr()  # a fresh tensor, never a view of the input


@pytest.mark.parametrize("upper", [False, True])
def test_tpc_eigh_two_pass_refine_contract(monkeypatch, upper):
    """tpc.eigh runs jacobi_eigh then jacobi_eigh_refine on symmetrized non-view S, then sorts ascending."""
    torch.manual_seed(0)
    S = spd(2, 3, n=4, dtype=torch.float32)
    garbage = S + (torch.tril(torch.randn_like(S), -1) if upper else torch.triu(torch.randn_like(S), 1)) * 100
    calls = []

    def pass1(S_flat):
        assert S_flat.is_contiguous() and S_flat.shape == (6, 4, 4) and S_flat.data_ptr() != garbage.data_ptr()
        calls.append(("pass1", S_flat.clone()))
        w, V = torch.linalg.eigh(S_flat)
        perm = torch.tensor([2, 0, 3, 1])
        return w[:, perm], V.mT[:, perm]

    def pass2(S_flat, Vt1):
        assert S_flat.is_contiguous() and Vt1.shape == (6, 4, 4)
        calls.append(("pass2", S_flat.clone(), Vt1.clone()))
        w, V = torch.linalg.eigh(S_flat)
        perm = torch.tensor([3, 1, 0, 2])
        return w[:, perm], V.mT[:, perm]

    monkeypatch.setattr(tpc, "_eigh_op", lambda: pass1)
    monkeypatch.setattr(tpc, "_eigh_refine_op", lambda: pass2)
    w, V = tpc.eigh(garbage, upper)
    assert [c[0] for c in calls] == ["pass1", "pass2"]
    torch.testing.assert_close(calls[0][1], S.reshape(-1, 4, 4))
    torch.testing.assert_close(calls[1][1], S.reshape(-1, 4, 4))
    w_ref = torch.linalg.eigvalsh(S)
    torch.testing.assert_close(w, w_ref)
    torch.testing.assert_close(V @ torch.diag_embed(w) @ V.mT, S, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("n", [3, 8, 22])
def test_generic_singular_triangle_is_non_finite(monkeypatch, n):
    """A zero diagonal entry, nonzero entries below it: substitution divides by it, so that element is non-finite."""
    monkeypatch.setattr(linalg, "_TRI", tpc.tri_solve_reference)
    torch.manual_seed(n)
    A = torch.tril(0.3 * torch.randn(3, n, n)) + 2 * torch.eye(n)
    A[1, n // 2, n // 2] = 0
    B = torch.randn(3, n, 2)
    want = torch.linalg.solve_triangular(A, B, upper=False)
    assert not torch.isfinite(want[1]).all()
    with cpu_as_hpu(linalg.install_solve_triangular):
        got = torch.linalg.solve_triangular(A, B, upper=False)
    with cpu_as_hpu(linalg.install_cholesky_solve):
        chol = torch.cholesky_solve(B, A)
    for out in (got, chol):
        assert not torch.isfinite(out[1]).all() and torch.isfinite(out[0]).all() and torch.isfinite(out[2]).all()


# ---- adversarial findings (tests/adversarial_linalg.py)


def test_solve_det_slogdet_register_no_backend_kernel(monkeypatch):
    """Their backend (``{}``) kernel would return placeholder LU factors to torch's own derivative in a compiled graph
    (NaN gradients); calls below autograd must reach Habana's kernel. inv_ex's has no such factors and stays."""
    seen = []
    monkeypatch.setattr(linalg, "register_paired", lambda device_type, groups: seen.extend(groups))
    linalg.install_solve("hpu")
    linalg.install_inv_det("hpu")
    keys = {group[0][0].name(): [key for _, key, _ in group] for group in seen}
    for name in ("aten::_linalg_solve_ex", "aten::_linalg_det", "aten::_linalg_slogdet"):
        assert keys[name] == ["Autograd{}"], keys
    assert keys["aten::linalg_inv_ex"] == ["Autograd{}", "{}"]


def test_enable_installs_the_triangular_solves():
    assert linalg.install_solve_triangular in hpu_ops.FIXES
    assert linalg.install_cholesky_solve in hpu_ops.FIXES


_COMPILE_FUNCTIONS = {
    "solve": lambda a, b: torch.linalg.solve(a, b).sum(),
    "det": lambda a, b: torch.linalg.det(a).sum() + b.sum(),
    "slogdet": lambda a, b: torch.linalg.slogdet(a)[1].sum() + b.sum(),
}


def _compiled_below_autograd(f, **options):
    """``torch.compile(f)`` through aot_autograd, its graphs run below autograd: on the card a compiled graph's ops
    reach the ``HPU`` backend key, where only Habana's kernels are registered for solve_ex, det and slogdet. (The CPU
    stand-in's inductor calls the autograd key at run time, which no card path does.)"""
    from functorch.compile import make_boxed_func

    def below(gm, example_inputs):
        def run(*args):
            with torch._C._AutoDispatchBelowAutograd():
                return gm.forward(*args)

        return make_boxed_func(run)

    torch._dynamo.reset()
    return torch.compile(f, backend=aot_autograd(fw_compiler=below, bw_compiler=below), **options)


@pytest.mark.parametrize("name", list(_COMPILE_FUNCTIONS))
@pytest.mark.parametrize("dynamic", [True, None])
def test_compiled_gradients_survive_symbolic_shapes_and_recompiles(name, dynamic, gj_hook):
    """The second batch size makes dynamo recompile with a symbolic batch, where the kernels decline and torch's own
    backward (which reads LU) stays in the graph: its gradients must be finite and torch's. Calls below autograd reach
    the native kernel (no backend kernel is registered), so the LU factors it reads are real."""
    torch.manual_seed(0)
    f = _COMPILE_FUNCTIONS[name]
    a = (general(5, n=4, dtype=torch.float32)).requires_grad_()
    b = torch.randn(5, 4, 2, requires_grad=True)
    want = torch.autograd.grad(f(a, b), (a, b))
    with cpu_as_hpu(lambda d: chain_install(d)):
        g = _compiled_below_autograd(f, dynamic=dynamic)
        g(a[:2], b[:2])
        got = torch.autograd.grad(g(a, b), (a, b))
    for x, y in zip(got, want):
        assert torch.isfinite(x).all()
    torch.testing.assert_close(got, want, rtol=1e-3, atol=1e-4)


def chain_install(device_type):
    from hpu_ops._dispatch import chain
    return chain([linalg.install_solve(device_type), linalg.install_inv_det(device_type)])


@pytest.mark.parametrize("upper", ["u", "U", "l", "L"])
@pytest.mark.parametrize("fn", [torch.linalg.eigvalsh, lambda G, UPLO: torch.linalg.eigh(G, UPLO)[0]])
def test_eigh_uplo_is_case_insensitive(upper, fn, eigh_hook):
    torch.manual_seed(0)
    G = torch.randn(3, 6, 6)  # not symmetric: the triangle read matters
    want = fn(G, UPLO=upper)
    with cpu_as_hpu(jacobi.install_eigh) as calls:
        got = fn(G, UPLO=upper)
        assert calls["eigh"] >= 1, dict(calls)
    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-4)
    assert eigh_hook[-1][1] is (upper.upper() == "U")


@pytest.mark.parametrize("a_batch, b_batch", [((1,), (4,)), ((4,), (1,)), ((3, 1), (1, 5))])
def test_solve_broadcast_batches_with_info_check(a_batch, b_batch, gj_hook):
    """``info`` sliced from the broadcast batch must be contiguous for ``_linalg_check_errors``."""
    torch.manual_seed(0)
    A = general(*a_batch, n=6, dtype=torch.float32)
    B = torch.randn(*b_batch, 6, 3)
    want = torch.linalg.solve(A, B)
    with cpu_as_hpu(linalg.install_solve) as calls:
        got = torch.linalg.solve(A, B)
        assert calls["solve"] >= 1, dict(calls)
    assert gj_hook
    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-4)


def test_enable_reinstalls_after_the_undo(monkeypatch):
    import sys
    import types

    names = ["habana_frameworks", "habana_frameworks.torch", "habana_frameworks.torch.core"]
    for name in names:
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setattr(tpc, "install_kernel_path", lambda: None)
    monkeypatch.setattr(hpu_ops, "_attention_fused", lambda device_type="hpu": True)
    monkeypatch.delitem(hpu_ops._STATE, "undo", raising=False)
    for name in hpu_ops.BROKEN_PASSES:
        monkeypatch.setenv(f"PT_HPU_DISABLE_{name}", "True")

    def registered():
        return all(torch._C._dispatch_has_kernel_for_dispatch_key(op, "AutogradHPU")
                   for op in ("aten::_linalg_solve_ex", "aten::linalg_inv_ex"))

    try:
        hpu_ops.enable()
        assert registered()
        first = hpu_ops._STATE["undo"]
        hpu_ops.enable()
        assert hpu_ops._STATE["undo"] is first  # idempotent while installed
        first()
        assert not registered() and "undo" not in hpu_ops._STATE
        first()  # a second undo is a no-op
        hpu_ops.enable()
        assert registered()
    finally:
        if "undo" in hpu_ops._STATE:
            hpu_ops._STATE["undo"]()


def test_eigh_bad_uplo_raises_like_torch(eigh_hook):
    G = torch.randn(2, 4, 4)
    with pytest.raises(RuntimeError) as native:
        torch.linalg.eigvalsh(G, UPLO="X")
    with cpu_as_hpu(jacobi.install_eigh):
        with pytest.raises(RuntimeError, match=re.escape(str(native.value))):
            torch.linalg.eigvalsh(G, UPLO="X")


LARGE_SOLVES = {
    "matrix": (lambda a, b: torch.linalg.solve(a, b), lambda: torch.randn(4, 2, dtype=torch.float64)),
    "left_false": (lambda a, b: torch.linalg.solve(a, b, left=False), lambda: torch.randn(2, 4, dtype=torch.float64)),
    "vector": (lambda a, b: torch.linalg.solve(a, b), lambda: torch.randn(4, dtype=torch.float64)),
    "broadcast": (lambda a, b: torch.linalg.solve(a, b), lambda: torch.randn(3, 4, 2, dtype=torch.float64)),
}


@pytest.mark.parametrize("case", list(LARGE_SOLVES))
def test_large_solve_backward_reuses_the_inverse(case, monkeypatch, gj_hook):
    """Past tpc.MAXN (lowered to 2 here) solve multiplies by the kernel's inverse and keeps it: the backward calls no
    kernel, first and second order equal torch's."""
    monkeypatch.setattr(tpc, "MAXN", 2)
    torch.manual_seed(0)
    f, make_b = LARGE_SOLVES[case]
    inputs = [general(n=4).requires_grad_(), make_b().requires_grad_()]
    want = f(*[x.detach() for x in inputs])
    with cpu_as_hpu(linalg.install_solve):
        X = f(*inputs)
        torch.testing.assert_close(X, want)
        before = len(gj_hook)
        X.sum().backward()
        assert len(gj_hook) == before, "the backward of a large solve called the kernel"
        assert torch.autograd.gradgradcheck(f, inputs, atol=1e-5, rtol=1e-4)


def test_large_nan_in_A_gives_nan(large_reference):
    A = permuted(3, 70)
    A[1, 0, 0] = float("nan")
    X, D, _ = tpc.gj_inv(A)
    assert X[1].isnan().all() and D[1].isnan().all()
    assert torch.isfinite(X[[0, 2]]).all() and torch.isfinite(D[[0, 2]]).all()
    Y, _, _ = tpc.gj_solve(A, torch.randn(3, 70, 2))
    assert Y[1].isnan().all() and torch.isfinite(Y[[0, 2]]).all()


def test_large_inf_in_A_keeps_the_kernel_result(large_reference):
    A = permuted(2, 70)
    A[0, 0, 0] = float("inf")
    X, _, _ = tpc.gj_inv(A)
    R, P, Q, _, _ = tpc.gj_large_reference(A)
    raw = R[..., :70].gather(1, P.long()[..., None].expand_as(A)).gather(2, Q.long()[:, None, :].expand_as(A))
    torch.testing.assert_close(X[0], raw[0], equal_nan=True)
    torch.testing.assert_close(X[1], torch.linalg.inv(A[1]), rtol=1e-4, atol=1e-4)
