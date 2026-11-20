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
    assert seen == [32, 32, 6]
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
    "cholesky_solve_lower": (lambda: (lambda b, l: torch.cholesky_solve(b, l),
                                      (torch.randn(2, 6, 3), torch.linalg.cholesky(spd(2, n=6, dtype=torch.float32)))),
                             linalg.install_cholesky_solve, "cholesky_solve"),
    "cholesky_solve_upper": (lambda: (lambda b, l: torch.cholesky_solve(b, l, upper=True),
                                      (torch.randn(2, 6, 3),
                                       torch.linalg.cholesky(spd(2, n=6, dtype=torch.float32)).mH)),
                             linalg.install_cholesky_solve, "cholesky_solve"),
}
for _upper in (False, True):
    for _left in (True, False):
        for _unit in (False, True):
            _key = f"tri_{'upper' if _upper else 'lower'}_{'left' if _left else 'right'}_{'unit' if _unit else 'full'}"
            GJ_CASES[_key] = (
                (lambda u=_upper, l=_left, t=_unit: _tri_case(u, l, t)), linalg.install_solve_triangular,
                "solve_triangular")


@pytest.mark.parametrize("case", list(GJ_CASES))
def test_linalg_routes_through_gj(case, gj_hook):
    torch.manual_seed(0)
    make, install, name = GJ_CASES[case]
    f, inputs = make()
    compare(f, install, *inputs, name=name, rtol=2e-3, atol=2e-3)
    assert gj_hook, "the kernel hook was not called"


def test_gj_is_declined_past_maxn_on_hpu_and_without_the_kernel(monkeypatch):
    A = torch.randn(2, 4, 4)
    takes = linalg._tpc_takes("hpu", linalg._takes("hpu", "solve"))
    monkeypatch.setattr(tpc, "ready", lambda: False)
    assert takes(A) is False
    monkeypatch.setattr(tpc, "ready", lambda: True)
    assert takes(A) is True
    assert takes(torch.randn(tpc.MAXN + 1, tpc.MAXN + 1)) is False
    assert takes(torch.randn(tpc.MAXN, tpc.MAXN)) is True
    cpu = linalg._tpc_takes("cpu", linalg._takes("cpu", "solve"))
    monkeypatch.setattr(tpc, "ready", lambda: False)
    assert cpu(torch.randn(tpc.MAXN + 1, tpc.MAXN + 1)) is True  # the CPU stand-in keeps the composite


def test_gj_not_used_past_maxn(gj_hook):
    torch.manual_seed(0)
    with cpu_as_hpu(linalg.install_solve):
        torch.linalg.solve(f32(2, n=tpc.MAXN + 1), torch.randn(2, tpc.MAXN + 1, 2))
    assert gj_hook == []


DOUBLE_BACKWARD = {
    "solve": (linalg.install_solve, lambda a, b: torch.linalg.solve(a, b),
              lambda: (general(n=4), torch.randn(4, 2, dtype=torch.float64))),
    "solve_right": (linalg.install_solve, lambda a, b: torch.linalg.solve(a, b, left=False),
                    lambda: (general(n=4), torch.randn(2, 4, dtype=torch.float64))),
    "solve_triangular": (linalg.install_solve_triangular,
                         lambda a, b: torch.linalg.solve_triangular(a, b, upper=True),
                         lambda: (general(n=4), torch.randn(4, 2, dtype=torch.float64))),
    "cholesky_solve": (linalg.install_cholesky_solve, lambda b, l: torch.cholesky_solve(b, torch.tril(l)),
                       lambda: (torch.randn(4, 2, dtype=torch.float64), torch.linalg.cholesky(spd(n=4)))),
}


@pytest.mark.parametrize("case", list(DOUBLE_BACKWARD))
def test_double_backward_through_gj(case, gj_hook):
    torch.manual_seed(0)
    install, f, make = DOUBLE_BACKWARD[case]
    inputs = [x.requires_grad_() for x in make()]
    with cpu_as_hpu(install) as calls:
        assert torch.autograd.gradgradcheck(f, inputs, atol=1e-5, rtol=1e-4)
        assert sum(calls.values()) >= 1 and gj_hook


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


# ---- SVD family (#9): jacobi_svd, svd, svdvals, pinv, lstsq, matrix_norm, matrix_rank


@pytest.fixture
def svd_hook(monkeypatch):
    """``linalg._SVD`` set to ``svd_reference``; collects ``(shape, compute_uv)`` per call."""
    calls = []

    def hook(A, compute_uv):
        calls.append((tuple(A.shape), compute_uv))
        return tpc.svd_reference(A.detach(), compute_uv=compute_uv)

    monkeypatch.setattr(linalg, "_SVD", hook)
    return calls


@pytest.mark.parametrize("shape", [(1, 1), (3, 3), (7, 3), (3, 7), (22, 22), (32, 17), (17, 32)])
def test_tpc_svd_reference_contract(shape):
    torch.manual_seed(0)
    m, n = shape
    k = min(m, n)
    A = torch.randn(4, m, n)
    U, S, Vh = tpc.svd_reference(A, compute_uv=True)
    assert U.shape == (4, m, k) and S.shape == (4, k) and Vh.shape == (4, k, n)
    assert (S[:, :-1] >= S[:, 1:]).all() and (S >= 0).all()
    eye = torch.eye(k).expand(4, k, k)
    torch.testing.assert_close(U.mT @ U, eye, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(Vh @ Vh.mT, eye, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close((U * S.unsqueeze(-2)) @ Vh, A, rtol=1e-4, atol=1e-4)
    _, S_only, _ = tpc.svd_reference(A, compute_uv=False)
    torch.testing.assert_close(S_only, S, rtol=1e-5, atol=1e-5)


def test_tpc_svd_flattens_batch_and_makes_contiguous(monkeypatch):
    torch.manual_seed(0)
    seen, seen_vals = [], []

    def fake_svd(A_t):
        seen.append((tuple(A_t.shape), A_t.is_contiguous(), A_t.dtype))
        U, S, Vh = tpc.svd_reference(A_t.mT, compute_uv=True)
        return U.mT.contiguous(), S, Vh

    def fake_svdvals(A_t):
        seen_vals.append((tuple(A_t.shape), A_t.is_contiguous(), A_t.dtype))
        return tpc.svd_reference(A_t.mT, compute_uv=False)[1]

    monkeypatch.setattr(tpc, "_svd_op", lambda: fake_svd)
    monkeypatch.setattr(tpc, "_svdvals_op", lambda: fake_svdvals)

    for shape in ((2, 3, 5, 4), (2, 3, 4, 5)):
        A = torch.randn(*shape[:-2], shape[-1], shape[-2], dtype=torch.float64).mT  # non-contiguous
        m, n = A.shape[-2], A.shape[-1]
        k = min(m, n)
        U, S, Vh = tpc.jacobi_svd(A)
        assert U.shape == (2, 3, m, k) and S.shape == (2, 3, k) and Vh.shape == (2, 3, k, n)
        torch.testing.assert_close((U * S.unsqueeze(-2)) @ Vh, A.float(), rtol=1e-4, atol=1e-4)
        S_only = tpc.jacobi_svdvals(A)
        torch.testing.assert_close(S_only, S, rtol=1e-5, atol=1e-5)

    assert seen == [((6, 4, 5), True, torch.float32), ((6, 4, 5), True, torch.float32)]
    assert seen_vals == [((6, 4, 5), True, torch.float32), ((6, 4, 5), True, torch.float32)]


@pytest.mark.parametrize("lazy, flushes", [("1", 1), ("0", 0)])
def test_tpc_svd_flushes_lazy_graph(monkeypatch, lazy, flushes):
    marks = []
    core = types.SimpleNamespace(mark_step=lambda: marks.append(1))
    torch_mod = types.SimpleNamespace(core=core)
    hf = types.SimpleNamespace(torch=torch_mod)
    for name, mod in (("habana_frameworks", hf), ("habana_frameworks.torch", torch_mod),
                      ("habana_frameworks.torch.core", core)):
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.setenv("PT_HPU_LAZY_MODE", lazy)
    monkeypatch.setattr(tpc, "_svd_op", lambda: lambda A_t: (
        tpc.svd_reference(A_t.mT)[0].mT, tpc.svd_reference(A_t.mT)[1], tpc.svd_reference(A_t.mT)[2]))
    monkeypatch.setattr(tpc, "_svdvals_op", lambda: lambda A_t: tpc.svd_reference(A_t.mT, compute_uv=False)[1])
    tpc.jacobi_svd(torch.randn(2, 5, 4))
    assert len(marks) == flushes
    marks.clear()
    tpc.jacobi_svdvals(torch.randn(2, 5, 4))
    assert len(marks) == flushes


@pytest.mark.parametrize("lazy, flushes", [("1", 1), ("0", 0)])
def test_tpc_expm_and_householder_flush_before_the_op(monkeypatch, lazy, flushes):
    marks = []
    core = types.SimpleNamespace(mark_step=lambda: marks.append(1))
    torch_mod = types.SimpleNamespace(core=core)
    for name, mod in (("habana_frameworks", types.SimpleNamespace(torch=torch_mod)),
                      ("habana_frameworks.torch", torch_mod), ("habana_frameworks.torch.core", core)):
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.setenv("PT_HPU_LAZY_MODE", lazy)
    monkeypatch.setattr(tpc, "_expm_op", lambda: torch.linalg.matrix_exp)
    monkeypatch.setattr(tpc, "_expm_frechet_op", lambda: lambda a, g: g)
    monkeypatch.setattr(tpc, "_hh_op", lambda: lambda a_t, t: a_t)
    A = torch.randn(2, 4, 4)
    for call in (lambda: tpc.matrix_exp(A), lambda: tpc.matrix_exp_frechet(A, A),
                 lambda: tpc.householder_product(torch.randn(2, 5, 3), torch.randn(2, 3))):
        marks.clear()
        call()
        assert len(marks) == flushes


def test_svd_declined_past_maxn_and_without_kernel(monkeypatch, svd_hook):
    A = torch.randn(2, 4, 4)
    takes = linalg._svd_takes("hpu")
    monkeypatch.setattr(tpc, "ready", lambda: False)
    assert takes(A) is False
    monkeypatch.setattr(tpc, "ready", lambda: True)
    assert takes(A) is True
    assert takes(torch.randn(tpc.MAXN + 1, 4)) is False
    assert takes(torch.randn(4, tpc.MAXN + 1)) is False
    assert takes(torch.randn(tpc.MAXN, tpc.MAXN)) is True
    assert takes(A, driver="gesvd") is False

    with cpu_as_hpu(linalg.install_svd):
        torch.linalg.svd(torch.randn(2, tpc.MAXN + 1, 4))
    assert svd_hook == []


def test_svd_family_float64_warns_once_on_hpu():
    linalg._WARNED.clear()
    A = torch.randn(4, 3, dtype=torch.float64)
    B = torch.randn(4, 2, dtype=torch.float64)
    for name, fn, args in (("svd", linalg._svd_takes("hpu"), (A,)),
                           ("pinv", linalg._pinv_takes("hpu"), (A,)),
                           ("lstsq", linalg._lstsq_takes("hpu"), (A, B))):
        with pytest.warns(UserWarning, match="hpu-ops#16"):
            assert fn(*args) is False
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert fn(*args) is False

    # pinv allows float64 atol/rtol tensors when A is float32 (PyTorch emits float64 0-D tensors)
    assert linalg._pinv_takes("cpu")(
        torch.randn(4, 3), atol=torch.tensor(1e-4, dtype=torch.float64), rtol=torch.tensor(0.0, dtype=torch.float64))

    for bad in (torch.randn(3, 3, dtype=torch.complex64), torch.ones(3, 3, dtype=torch.int64),
                torch.randn(3, 3, dtype=torch.float16), torch.randn(3, 3, dtype=torch.bfloat16),
                torch.randn(0, 3, 3), torch.randn(3)):
        assert linalg._svd_takes("cpu")(bad) is False
        assert linalg._pinv_takes("cpu")(bad) is False


def _svd_sign_invariant(full_matrices=False):
    def f(a):
        U, S, Vh = torch.linalg.svd(a, full_matrices=full_matrices)
        k = S.shape[-1]
        Uk, Vhk = U[..., :k], Vh[..., :k, :]
        rec = (Uk * S.exp().unsqueeze(-2)) @ Vhk
        if full_matrices:
            return S, rec, Uk.mT @ Uk, Vhk @ Vhk.mT, U @ U.mT, Vh.mT @ Vh
        return S, rec, U.mT @ U, Vh @ Vh.mT

    return f


@pytest.mark.parametrize("full_matrices", [False, True])
@pytest.mark.parametrize("shape", [(1, 1), (4, 4), (7, 3), (3, 7), (2, 3, 6, 4), (2, 3, 4, 6)])
def test_svd_matches_torch(shape, full_matrices, svd_hook):
    torch.manual_seed(0)
    A = torch.randn(*shape, dtype=torch.float64)
    compare(_svd_sign_invariant(full_matrices), linalg.install_svd, A, name="svd", rtol=1e-6, atol=1e-6)
    assert svd_hook


def test_svd_noncontiguous_expanded_and_float32(svd_hook):
    torch.manual_seed(0)
    A = torch.randn(3, 5, 4, dtype=torch.float32).mT
    compare(_svd_sign_invariant(False), linalg.install_svd, A, name="svd", rtol=2e-3, atol=2e-3)

    base = torch.randn(1, 5, 4, dtype=torch.float64)
    compare(lambda x: _svd_sign_invariant(True)(x.expand(3, 5, 4)), linalg.install_svd, base, name="svd",
            rtol=1e-6, atol=1e-6)


def test_svd_double_backward(svd_hook):
    torch.manual_seed(0)
    A = torch.randn(2, 5, 3, dtype=torch.float64, requires_grad=True)
    with cpu_as_hpu(linalg.install_svd) as calls:
        assert torch.autograd.gradgradcheck(lambda x: torch.linalg.svd(x, full_matrices=False)[1], (A,))
        assert torch.autograd.gradgradcheck(
            lambda x: (lambda u, s, vh: (u * s.unsqueeze(-2)) @ vh)(*torch.linalg.svd(x, full_matrices=False)),
            (A,), atol=1e-5, rtol=1e-4)
        assert calls["svd"] >= 1 and svd_hook


@pytest.mark.parametrize("A_fn", [lambda: torch.zeros(4, 3, dtype=torch.float64),
                                   lambda: torch.diag(torch.tensor([3.0, 3.0, 1.0], dtype=torch.float64))],
                         ids=["zeros", "repeated"])
def test_svd_zero_and_repeated_s_backward_is_finite(A_fn, svd_hook):
    A_ref = A_fn().requires_grad_()
    torch.linalg.svd(A_ref, full_matrices=False).S.sum().backward()
    A_got = A_fn().requires_grad_()
    with cpu_as_hpu(linalg.install_svd) as calls:
        torch.linalg.svd(A_got, full_matrices=False).S.sum().backward()
        assert calls["svd"] >= 1
    assert torch.isfinite(A_got.grad).all()
    torch.testing.assert_close(A_got.grad, A_ref.grad)


def test_svd_out_and_legacy_alias(svd_hook):
    torch.manual_seed(0)
    A = torch.randn(5, 3, dtype=torch.float64)
    want_U, want_S, want_Vh = torch.linalg.svd(A, full_matrices=False)
    U, S, Vh = torch.empty_like(want_U), torch.empty_like(want_S), torch.empty_like(want_Vh)
    with cpu_as_hpu(linalg.install_svd):
        torch.linalg.svd(A, full_matrices=False, out=(U, S, Vh))
    torch.testing.assert_close(S, want_S)
    torch.testing.assert_close((U * S.unsqueeze(-2)) @ Vh, A)
    compare(lambda x: (torch.svd(x).S, torch.svd(x).U @ torch.diag_embed(torch.svd(x).S) @ torch.svd(x).V.mT),
            linalg.install_svd, A, name="svd", rtol=1e-6, atol=1e-6)


def test_svdvals_grad_and_no_grad(svd_hook):
    torch.manual_seed(0)
    A = torch.randn(2, 6, 4, dtype=torch.float64)
    with cpu_as_hpu(linalg.install_svd) as calls, torch.no_grad():
        got = torch.linalg.svdvals(A)
        assert calls["svd"] >= 1
    torch.testing.assert_close(got, torch.linalg.svdvals(A))
    assert svd_hook[-1] == ((2, 6, 4), False)

    svd_hook.clear()
    compare(torch.linalg.svdvals, linalg.install_svd, A, name="svd", rtol=1e-6, atol=1e-6)
    assert any(cuv for _, cuv in svd_hook)


def test_svd_and_svdvals_compiled(svd_hook):
    torch.manual_seed(0)
    A = torch.randn(2, 5, 4, dtype=torch.float64)
    nodes = []
    with cpu_as_hpu(linalg.install_svd):
        got = compiled(lambda x: torch.linalg.svdvals(x), nodes)(A)
    torch.testing.assert_close(got, torch.linalg.svdvals(A))
    assert "aten._linalg_svd.default" not in ops(nodes), ops(nodes)

    nodes_big = []
    big = torch.randn(tpc.MAXN + 1, 4, dtype=torch.float64)
    with cpu_as_hpu(linalg.install_svd):
        got_big = compiled(lambda x: torch.linalg.svdvals(x), nodes_big)(big)
    torch.testing.assert_close(got_big, torch.linalg.svdvals(big))
    assert "aten._linalg_svd.default" in ops(nodes_big), ops(nodes_big)


@pytest.mark.parametrize("shape", [(4, 4), (7, 4), (4, 7), (2, 3, 5, 4)])
def test_pinv_matches_torch_and_rank_deficient(shape, svd_hook):
    torch.manual_seed(0)
    A = torch.randn(*shape, dtype=torch.float64)
    compare(torch.linalg.pinv, linalg.install_pinv, A, name="pinv", rtol=1e-6, atol=1e-6)
    compare(torch.pinverse, linalg.install_pinv, A, name="pinv", rtol=1e-6, atol=1e-6)

    # Rank-deficient + atol/rtol variants (including non-contiguous .mT)
    U_r = torch.randn(4, 2, dtype=torch.float64)
    V_r = torch.randn(2, 5, dtype=torch.float64)
    R = (U_r @ V_r).mT.contiguous().mT
    compare(lambda x: torch.linalg.pinv(x, atol=1e-3, rtol=1e-3), linalg.install_pinv, R, name="pinv",
            rtol=1e-6, atol=1e-6)
    compare(lambda x: torch.linalg.pinv(x, rtol=1e-2), linalg.install_pinv, R, name="pinv", rtol=1e-6, atol=1e-6)


def test_pinv_eye_grad_and_double_backward(svd_hook):
    torch.manual_seed(0)
    eye = torch.eye(4, dtype=torch.float64)
    compare(torch.linalg.pinv, linalg.install_pinv, eye, name="pinv", rtol=1e-6, atol=1e-6)
    A = torch.randn(2, 4, 3, dtype=torch.float64, requires_grad=True)
    with cpu_as_hpu(linalg.install_pinv) as calls:
        assert torch.autograd.gradgradcheck(torch.linalg.pinv, (A,))
        assert calls["pinv"] >= 1


def test_pinv_declines_hermitian_and_inference_mode_without_recursion(svd_hook):
    torch.manual_seed(0)
    S = spd(2, n=4)
    want = torch.linalg.pinv(S, hermitian=True)
    with cpu_as_hpu(linalg.install_pinv) as calls:
        got = torch.linalg.pinv(S, hermitian=True)
        assert calls["pinv"] == 0
    torch.testing.assert_close(got, want)

    A = torch.randn(3, 4, 4, dtype=torch.float64)
    want_inf = torch.linalg.pinv(A)
    with cpu_as_hpu(linalg.install_pinv) as calls, torch.inference_mode():
        got_inf = torch.linalg.pinv(A)
        assert calls["pinv"] == 0
    torch.testing.assert_close(got_inf, want_inf)


def test_pinv_compiled_has_no_native_op(svd_hook):
    torch.manual_seed(0)
    A = torch.randn(2, 5, 4, dtype=torch.float64)
    nodes = []
    with cpu_as_hpu(linalg.install_pinv):
        got = compiled(torch.linalg.pinv, nodes)(A)
    torch.testing.assert_close(got, torch.linalg.pinv(A))
    assert "aten.linalg_pinv.atol_rtol_tensor" not in ops(nodes), ops(nodes)
    assert "aten._linalg_svd.default" not in ops(nodes), ops(nodes)


@pytest.mark.parametrize("driver", [None, "gels", "gelsy", "gelsd", "gelss"])
@pytest.mark.parametrize("a_shape, b_shape", [((4, 4), (4, 2)), ((6, 4), (6, 2)), ((4, 6), (4, 2)),
                                              ((2, 5, 3), (2, 5)), ((3, 1, 5, 3), (1, 4, 5, 2))])
def test_lstsq_matches_torch_full_rank(a_shape, b_shape, driver, svd_hook):
    torch.manual_seed(0)
    A = torch.randn(*a_shape, dtype=torch.float64)
    B = torch.randn(*b_shape, dtype=torch.float64)
    calls = compare(lambda a, b: torch.linalg.lstsq(a, b, driver=driver).solution,
                    linalg.install_lstsq, A, B, name="lstsq", rtol=1e-6, atol=1e-6)
    assert calls["lstsq"] >= 1
    want = torch.linalg.lstsq(A, B, driver=driver)
    with cpu_as_hpu(linalg.install_lstsq):
        got = torch.linalg.lstsq(A, B, driver=driver)
    assert got.residuals.shape == want.residuals.shape
    assert got.rank.dtype == want.rank.dtype and torch.equal(got.rank, want.rank)
    assert got.singular_values.shape == want.singular_values.shape
    torch.testing.assert_close(got.residuals, want.residuals)
    torch.testing.assert_close(got.singular_values, want.singular_values)


def test_lstsq_declines_rank_deficient_and_double_backward(svd_hook):
    torch.manual_seed(0)
    A_rd = torch.randn(5, 2, dtype=torch.float64) @ torch.randn(2, 4, dtype=torch.float64)
    B_rd = torch.randn(5, 2, dtype=torch.float64)
    want_rd = torch.linalg.lstsq(A_rd, B_rd).solution
    with cpu_as_hpu(linalg.install_lstsq) as calls:
        got_rd = torch.linalg.lstsq(A_rd, B_rd).solution
        assert calls["lstsq"] == 0
    torch.testing.assert_close(got_rd, want_rd)

    A = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
    B = torch.randn(5, 2, dtype=torch.float64, requires_grad=True)
    with cpu_as_hpu(linalg.install_lstsq) as calls:
        assert torch.autograd.gradgradcheck(lambda a, b: torch.linalg.lstsq(a, b).solution, (A, B))
        assert calls["lstsq"] >= 1

    # compiled executes cleanly via eager fallback
    nodes = []
    with cpu_as_hpu(linalg.install_lstsq) as calls:
        got_c = compiled(lambda a, b: torch.linalg.lstsq(a, b).solution, nodes)(A.detach(), B.detach())
        assert calls["lstsq"] >= 1
    torch.testing.assert_close(got_c, torch.linalg.lstsq(A.detach(), B.detach()).solution)


@pytest.mark.parametrize("ord_", [2, -2, "nuc"])
@pytest.mark.parametrize("shape", [(4, 4), (7, 3), (3, 7), (2, 3, 5, 4)])
def test_matrix_norm_routes_through_svd(shape, ord_, svd_hook):
    torch.manual_seed(0)
    A = torch.randn(*shape, dtype=torch.float64)
    compare(lambda x: torch.linalg.matrix_norm(x, ord=ord_), linalg.install_svd, A, name="svd", rtol=1e-6, atol=1e-6)
    nodes = []
    with cpu_as_hpu(linalg.install_svd):
        got = compiled(lambda x: torch.linalg.matrix_norm(x, ord=ord_), nodes)(A)
    torch.testing.assert_close(got, torch.linalg.matrix_norm(A, ord=ord_))
    assert "aten._linalg_svd.default" not in ops(nodes), ops(nodes)

    with cpu_as_hpu(linalg.install_svd) as calls:
        torch.linalg.matrix_norm(A, ord="fro")
        assert calls["svd"] == 0


def test_matrix_rank_routes_through_svd(svd_hook):
    torch.manual_seed(0)
    A_full = torch.randn(2, 6, 4, dtype=torch.float32)
    A_rd = torch.randn(2, 6, 2, dtype=torch.float32) @ torch.randn(2, 2, 4, dtype=torch.float32)
    A_zero = torch.zeros(2, 4, 4, dtype=torch.float32)
    for A in (A_full, A_rd, A_zero):
        for fn in (lambda x: torch.linalg.matrix_rank(x),
                   lambda x: torch.linalg.matrix_rank(x, 1e-3),
                   lambda x: torch.linalg.matrix_rank(x, torch.tensor(1e-3)),
                   lambda x: torch.linalg.matrix_rank(x, atol=1e-3, rtol=1e-3),
                   lambda x: torch.linalg.matrix_rank(x, atol=torch.tensor(1e-3))):
            want = fn(A)
            with cpu_as_hpu(linalg.install_matrix_rank) as calls:
                got = fn(A)
                assert calls["svd"] >= 1
            assert got.dtype == torch.int64 and torch.equal(got, want)

    S = spd(2, n=4, dtype=torch.float32)
    with cpu_as_hpu(linalg.install_matrix_rank) as calls:
        got_h = torch.linalg.matrix_rank(S, hermitian=True)
        assert calls["svd"] == 0
    assert torch.equal(got_h, torch.linalg.matrix_rank(S, hermitian=True))

    nodes = []
    with cpu_as_hpu(linalg.install_matrix_rank) as calls:
        got_c = compiled(torch.linalg.matrix_rank, nodes)(A_full)
        assert calls["svd"] >= 1
    assert torch.equal(got_c, torch.linalg.matrix_rank(A_full))
    assert "aten._linalg_svd.default" not in ops(nodes), ops(nodes)


def _tiny_singular_value_matrix():
    """A batch with singular values (1, 1, 1, 1e-3, 1e-9): the last is below eps * max(m, n) * smax in float32."""
    torch.manual_seed(0)
    Q1 = torch.linalg.qr(torch.randn(3, 5, 5, dtype=torch.float64))[0]
    Q2 = torch.linalg.qr(torch.randn(3, 5, 5, dtype=torch.float64))[0]
    d = torch.tensor([1.0, 1.0, 1.0, 1e-3, 1e-9], dtype=torch.float64)
    return ((Q1 * d) @ Q2.mT).float()


ZERO_ATOL_FNS = {
    "rank atol=0.": lambda x: torch.linalg.matrix_rank(x, atol=0.0),
    "rank atol=-1.": lambda x: torch.linalg.matrix_rank(x, atol=-1.0),
    "rank atol=tensor(0.)": lambda x: torch.linalg.matrix_rank(x, atol=torch.tensor(0.0)),
    "rank atol=tensor([0., 1e-6, 0.])": lambda x: torch.linalg.matrix_rank(x, atol=torch.tensor([0.0, 1e-6, 0.0])),
    "rank atol=1e-6": lambda x: torch.linalg.matrix_rank(x, atol=1e-6),
}


@pytest.mark.parametrize("fn", list(ZERO_ATOL_FNS))
def test_matrix_rank_atol_without_rtol_keeps_default_rtol_unless_positive(fn, svd_hook):
    """torch's get_atol_rtol: a given atol zeroes the default rtol only where atol > 0."""
    A = _tiny_singular_value_matrix()
    want = ZERO_ATOL_FNS[fn](A)
    with cpu_as_hpu(linalg.install_matrix_rank) as calls:
        got = ZERO_ATOL_FNS[fn](A)
        assert calls["svd"] >= 1
    assert torch.equal(got, want), (got, want)


@pytest.mark.parametrize("atol", [0.0, torch.tensor(0.0), torch.tensor([0.0, 1e-6, 0.0])])
def test_pinv_atol_zero_keeps_default_rtol(atol, svd_hook):
    A = _tiny_singular_value_matrix()
    want = torch.linalg.pinv(A, atol=atol)
    with cpu_as_hpu(linalg.install_pinv) as calls:
        got = torch.linalg.pinv(A, atol=atol)
        assert calls["pinv"] >= 1
    torch.testing.assert_close(got, want, rtol=1e-3, atol=1e-3)
    assert got.abs().max() < 1e4  # the 1e-9 value is dropped, not inverted


@pytest.fixture
def cia_backends(monkeypatch):
    """The CompositeImplicitAutograd ops' backend kernels registered on the CPU stand-in; collects ``_on_cpu`` calls."""
    monkeypatch.setattr(linalg, "_CIA_BACKENDS_ON_CPU", True)
    seen = []
    original = linalg._on_cpu

    def spy(op, args, kwargs):
        seen.append(str(op))
        return original(op, args, kwargs)

    monkeypatch.setattr(linalg, "_on_cpu", spy)
    return seen


INFERENCE_DECLINED = {
    "matrix_norm fro": (linalg.install_svd, lambda x: torch.linalg.matrix_norm(x, "fro")),
    "matrix_norm 1": (linalg.install_svd, lambda x: torch.linalg.matrix_norm(x, 1)),
    "matrix_norm inf": (linalg.install_svd, lambda x: torch.linalg.matrix_norm(x, float("inf"))),
    "matrix_norm -1": (linalg.install_svd, lambda x: torch.linalg.matrix_norm(x, -1)),
    "matrix_norm 2 dtype": (linalg.install_svd, lambda x: torch.linalg.matrix_norm(x, 2, dtype=torch.float64)),
    "matrix_rank hermitian": (linalg.install_matrix_rank, lambda x: torch.linalg.matrix_rank(x, hermitian=True)),
    "matrix_rank hermitian tol": (linalg.install_matrix_rank,
                                  lambda x: torch.linalg.matrix_rank(x, 1e-3, hermitian=True)),
    "matrix_rank hermitian atol": (linalg.install_matrix_rank,
                                   lambda x: torch.linalg.matrix_rank(x, atol=torch.tensor(1e-3), hermitian=True)),
}


@pytest.mark.parametrize("case", list(INFERENCE_DECLINED))
def test_declined_cia_calls_decompose_under_inference_mode(case, cia_backends, svd_hook):
    """A declined call of a CompositeImplicitAutograd op below autograd decomposes on the device (as before hpu-ops),
    never through hpu-ops' CPU route."""
    install, fn = INFERENCE_DECLINED[case]
    S = spd(2, n=5, dtype=torch.float32)
    want = fn(S)
    with cpu_as_hpu(install), torch.inference_mode():
        got = fn(S)
    assert cia_backends == [], cia_backends
    torch.testing.assert_close(got, want)


def test_taken_cia_calls_use_the_kernel_under_inference_mode(cia_backends, svd_hook):
    A = torch.randn(2, 5, 4)
    for install, fn, name in ((linalg.install_svd, lambda x: torch.linalg.matrix_norm(x, "nuc"), "svd"),
                              (linalg.install_svd, torch.linalg.svdvals, "svd"),
                              (linalg.install_matrix_rank, torch.linalg.matrix_rank, "svd")):
        want = fn(A)
        with cpu_as_hpu(install) as calls, torch.inference_mode():
            got = fn(A)
            assert calls[name] >= 1
        torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)
    assert cia_backends == []


@pytest.mark.parametrize("shape", [(6, 4), (4, 6), (4, 4)])
def test_tpc_svd_full_matrices_and_unsorted(monkeypatch, shape):
    torch.manual_seed(0)
    m, n = shape
    k = min(m, n)

    def fake_svd(A_t):
        U, S, Vh = tpc.svd_reference(A_t.mT, compute_uv=True, full_matrices=False)
        return U.mT.contiguous(), S, Vh

    def fake_svd_full(A_t):
        U, S, Vh = tpc.svd_reference(A_t.mT, compute_uv=True, full_matrices=True)
        return U.mT.contiguous(), S, Vh

    monkeypatch.setattr(tpc, "_svd_op", lambda: fake_svd)
    monkeypatch.setattr(tpc, "_svd_full_op", lambda: fake_svd_full)

    A = torch.randn(2, m, n, dtype=torch.float64).mT.contiguous().mT  # non-contiguous when square/rect
    A_zero = torch.zeros(2, m, n, dtype=torch.float32)
    A_rd = torch.randn(2, m, 1, dtype=torch.float32) @ torch.randn(2, 1, n, dtype=torch.float32)
    for mat in (A, A_zero, A_rd):
        U, S, Vh = tpc.jacobi_svd(mat, full_matrices=True)
        assert U.shape == (2, m, m) and S.shape == (2, k) and Vh.shape == (2, n, n)
        assert (S[:, :-1] >= S[:, 1:]).all()
        eye_m = torch.eye(m).expand(2, m, m)
        eye_n = torch.eye(n).expand(2, n, n)
        torch.testing.assert_close(U.mT @ U, eye_m, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(U @ U.mT, eye_m, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(Vh @ Vh.mT, eye_n, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(Vh.mT @ Vh, eye_n, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close((U[..., :k] * S.unsqueeze(-2)) @ Vh[..., :k, :], mat.float(), rtol=1e-4, atol=1e-4)

    Uu, Su, Vhu = tpc.jacobi_svd(A, full_matrices=True, sort=False)
    torch.testing.assert_close((Uu[..., :k] * Su.unsqueeze(-2)) @ Vhu[..., :k, :], A.float(), rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(Su.sort(-1, descending=True).values, tpc.jacobi_svd(A)[1], rtol=1e-5, atol=1e-5)


def test_svd_u_backend_takes_and_decline_copy_back(monkeypatch, svd_hook):
    captured = []
    monkeypatch.setattr(linalg, "register_kernels", lambda _dev, entries: captured.extend(entries))
    monkeypatch.setattr(tpc, "ready", lambda: True)
    linalg.install_svd("hpu")
    svd_u = next(fn for op, key, fn in captured if op == torch.ops.aten._linalg_svd.U and key == "{}")

    def fake_on_cpu(op, args, kwargs):
        out = op(*(x.clone() if isinstance(x, torch.Tensor) else x for x in args),
                 **{k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in kwargs.items()})
        return tuple(t.clone() for t in out)

    monkeypatch.setattr(linalg, "_on_cpu", fake_on_cpu)

    # Accept path
    torch.manual_seed(0)
    A = torch.randn(2, 5, 4, dtype=torch.float32)
    U, S, Vh = torch.zeros(2, 5, 4), torch.zeros(2, 4), torch.zeros(2, 4, 4)
    rU, rS, rVh = svd_u(A, False, True, U=U, S=S, Vh=Vh)
    assert rU is U and rS is S and rVh is Vh and svd_hook
    want_U, want_S, want_Vh = torch.linalg.svd(A, full_matrices=False)
    torch.testing.assert_close(S, want_S, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close((U * S.unsqueeze(-2)) @ Vh, A, rtol=1e-4, atol=1e-4)

    # Decline path: n > MAXN, float64, and compute_uv=False (driver != None is also declined by takes)
    assert not linalg._svd_takes("hpu")(A, driver="gesvd")
    for mat, cuv in (
        (torch.randn(2, 4, tpc.MAXN + 1, dtype=torch.float32), True),
        (torch.randn(2, 4, 3, dtype=torch.float64), True),
        (torch.randn(2, 4, tpc.MAXN + 1, dtype=torch.float32), False),
    ):
        m, n = mat.shape[-2], mat.shape[-1]
        k = min(m, n)
        u_out = torch.zeros(2, m, k, dtype=mat.dtype) if cuv else mat.new_empty(0)
        s_out = torch.zeros(2, k, dtype=mat.dtype)
        vh_out = torch.zeros(2, k, n, dtype=mat.dtype) if cuv else mat.new_empty(0)
        ru, rs, rvh = svd_u(mat, False, cuv, U=u_out, S=s_out, Vh=vh_out)
        assert ru is u_out and rs is s_out and rvh is vh_out
        if cuv:
            wu, ws, wvh = torch.linalg.svd(mat, full_matrices=False)
            torch.testing.assert_close(s_out, ws)
            torch.testing.assert_close((u_out * s_out.unsqueeze(-2)) @ vh_out, mat)
        else:
            torch.testing.assert_close(s_out, torch.linalg.svdvals(mat))


@pytest.mark.parametrize("scale", [1e-20, 1e20])
@pytest.mark.parametrize("full_matrices", [False, True])
def test_svd_extreme_scale_backward_is_finite(scale, full_matrices, svd_hook):
    torch.manual_seed(0)
    A_base = torch.randn(3, 5, 4, dtype=torch.float32) * scale
    A_ref = A_base.clone().requires_grad_()
    Ur, Sr, Vhr = torch.linalg.svd(A_ref, full_matrices=full_matrices)
    ((Ur[..., :4] * Sr.unsqueeze(-2)) @ Vhr[..., :4, :]).sum().backward()

    A_got = A_base.clone().requires_grad_()
    with cpu_as_hpu(linalg.install_svd) as calls:
        Ug, Sg, Vhg = torch.linalg.svd(A_got, full_matrices=full_matrices)
        ((Ug[..., :4] * Sg.unsqueeze(-2)) @ Vhg[..., :4, :]).sum().backward()
        assert calls["svd"] >= 1
    assert torch.isfinite(A_got.grad).all()
    if torch.isfinite(A_ref.grad).all():
        torch.testing.assert_close(A_got.grad, A_ref.grad, rtol=1e-3, atol=1e-4)


def test_svdvals_and_matrix_norm_gradgradcheck_and_dims(svd_hook):
    torch.manual_seed(0)
    A = torch.randn(2, 5, 4, dtype=torch.float64, requires_grad=True)
    with cpu_as_hpu(linalg.install_svd) as calls:
        assert torch.autograd.gradgradcheck(torch.linalg.svdvals, (A,))
        for ord_ in (2, -2, "nuc"):
            assert torch.autograd.gradgradcheck(lambda x: torch.linalg.matrix_norm(x, ord=ord_), (A,))
        assert calls["svd"] >= 1

    B = torch.randn(2, 5, 3, 4, dtype=torch.float64)
    for dim in ((1, 3), (3, 1)):
        for keepdim in (False, True):
            for ord_ in (2, -2, "nuc"):
                compare(lambda x: torch.linalg.matrix_norm(x, ord=ord_, dim=dim, keepdim=keepdim),
                        linalg.install_svd, B, name="svd", rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("driver", [None, "gelsy", "gelsd", "gelss"])
@pytest.mark.parametrize("rcond", [None, -1.0, 0.0])
def test_lstsq_rcond_zero_vs_negative(driver, rcond, svd_hook):
    torch.manual_seed(0)
    eps = torch.finfo(torch.float32).eps
    B = torch.randn(6, 2)
    for s_min in (1e-3, 10.0 * eps, 3.0 * eps, 0.7 * eps, 0.2 * eps, 0.0):
        S = torch.tensor([1.0, 1.0, 1.0, s_min], dtype=torch.float32)
        A = torch.cat([torch.diag(S), torch.zeros(2, 4)])
        want = torch.linalg.lstsq(A, B, rcond=rcond, driver=driver)
        with cpu_as_hpu(linalg.install_lstsq) as calls:
            got = torch.linalg.lstsq(A, B, rcond=rcond, driver=driver)
        assert torch.equal(got.rank, want.rank)
        thresh = 6.0 * eps if (rcond is None or rcond < 0) else (0.5 * eps if driver == "gelsd" else 0.0)
        accept = s_min > thresh
        assert (calls["lstsq"] >= 1) == accept
        if s_min >= 1e-3:
            torch.testing.assert_close(got.solution, want.solution, rtol=1e-4, atol=1e-4)
        elif not accept and torch.isfinite(want.solution).all():
            torch.testing.assert_close(got.solution, want.solution)
        elif s_min >= 10.0 * eps:
            assert torch.isfinite(got.solution).all()
            torch.testing.assert_close(got.solution, want.solution, rtol=1e-3, atol=1e-3)


# ---- matrix_exp & householder_product (#13)


@pytest.fixture
def expm_hook(monkeypatch):
    """``linalg._MATRIX_EXP`` / ``linalg._MATRIX_EXP_FRECHET`` set to reference functions."""
    calls = []

    def fwd(A):
        calls.append(("fwd", tuple(A.shape)))
        return tpc.matrix_exp_reference(A)

    def bwd(A, G):
        calls.append(("bwd", tuple(A.shape)))
        return tpc.matrix_exp_frechet_reference(A, G)

    monkeypatch.setattr(linalg, "_MATRIX_EXP", fwd)
    monkeypatch.setattr(linalg, "_MATRIX_EXP_FRECHET", bwd)
    monkeypatch.setattr(linalg, "_GJ", tpc.gj_solve_reference)
    return calls


@pytest.fixture
def hh_hook(monkeypatch):
    """``linalg._HOUSEHOLDER`` set to ``tpc.householder_product_reference``."""
    calls = []

    def hook(A, tau):
        calls.append((tuple(A.shape), tuple(tau.shape)))
        return tpc.householder_product_reference(A, tau)

    monkeypatch.setattr(linalg, "_HOUSEHOLDER", hook)
    monkeypatch.setattr(linalg, "_GJ", tpc.gj_solve_reference)
    return calls


@pytest.mark.parametrize("kind", ["unit", "skew", "large", "zeros", "n1", "transposed", "expanded"])
def test_matrix_exp_matches_torch(kind, expm_hook):
    torch.manual_seed(0)
    if kind == "unit":
        A = torch.randn(2, 3, 22, 22, dtype=torch.float64) * 0.4
    elif kind == "skew":
        x = torch.randn(3, 22, 22, dtype=torch.float64)
        A = 0.1 * (x - x.mT)
    elif kind == "large":
        A = 5.0 * torch.randn(2, 16, 16, dtype=torch.float64)
    elif kind == "zeros":
        A = torch.zeros(3, 22, 22, dtype=torch.float64)
    elif kind == "n1":
        A = torch.randn(4, 1, 1, dtype=torch.float64)
    elif kind == "transposed":
        A = (0.3 * torch.randn(22, 22, dtype=torch.float64)).mT
    else:
        base = 0.3 * torch.randn(1, 6, 6, dtype=torch.float64)
        calls = compare(lambda x: torch.linalg.matrix_exp(x.expand(3, 6, 6)),
                        linalg.install_matrix_exp, base, name="matrix_exp", rtol=1e-10, atol=1e-10)
        assert calls["matrix_exp"] >= 1 and expm_hook
        return
    rtol = 1e-9 if kind == "large" else 1e-11
    calls = compare(torch.linalg.matrix_exp, linalg.install_matrix_exp, A, name="matrix_exp", rtol=rtol, atol=rtol)
    assert calls["matrix_exp"] >= 1 and expm_hook
    compare(torch.matrix_exp, linalg.install_matrix_exp, A, name="matrix_exp", rtol=rtol, atol=rtol)


def test_matrix_exp_float32_and_double_backward(expm_hook):
    torch.manual_seed(0)
    for A in (0.1 * (lambda x: x - x.mT)(torch.randn(2, 22, 22)), 5.0 * torch.randn(2, 22, 22)):
        Ad = A.double().requires_grad_()
        ref = torch.linalg.matrix_exp(Ad)
        ref.sum().backward()
        Ac = A.clone().requires_grad_()
        want = torch.linalg.matrix_exp(Ac)
        want.sum().backward()
        Ag = A.clone().requires_grad_()
        with cpu_as_hpu(linalg.install_matrix_exp) as calls:
            got = torch.linalg.matrix_exp(Ag)
            got.sum().backward()
            assert calls["matrix_exp"] >= 1
        e_got = (got.double() - ref).abs().max() / ref.abs().max()
        e_want = (want.double() - ref).abs().max() / ref.abs().max()
        ge_got = (Ag.grad.double() - Ad.grad).abs().max() / Ad.grad.abs().max()
        ge_want = (Ac.grad.double() - Ad.grad).abs().max() / Ad.grad.abs().max()
        assert e_got <= max(1e-6, 10 * e_want)
        assert ge_got <= max(1e-6, 10 * ge_want)

    A_dbl = torch.randn(2, 5, 5, dtype=torch.float64, requires_grad=True)
    A_big = (4.0 * torch.randn(4, 4, dtype=torch.float64)).detach().requires_grad_()
    with cpu_as_hpu(linalg.install_matrix_exp) as calls:
        assert torch.autograd.gradcheck(torch.linalg.matrix_exp, (A_dbl,))
        assert torch.autograd.gradgradcheck(torch.linalg.matrix_exp, (A_dbl,))
        assert torch.autograd.gradgradcheck(torch.matrix_exp, (A_big,))
        assert calls["matrix_exp"] >= 1 and expm_hook


def test_matrix_exp_compiled_and_out_and_declines(monkeypatch, expm_hook):
    torch.manual_seed(0)
    A = torch.randn(2, 8, 8, dtype=torch.float64, requires_grad=True)
    nodes = []
    with cpu_as_hpu(linalg.install_matrix_exp) as calls:
        got = compiled(torch.linalg.matrix_exp, nodes, fullgraph=True)(A)
        got.sum().backward()
        assert calls["matrix_exp"] >= 1
    assert "aten.linalg_matrix_exp.default" not in ops(nodes), ops(nodes)
    want = torch.linalg.matrix_exp(A.detach())
    torch.testing.assert_close(got.detach(), want)

    # Declines: float64 warns once on hpu, oversize, empty, invalid shape raises
    linalg._WARNED.clear()
    takes_hpu = linalg._tpc_takes("hpu", linalg._takes("hpu", "matrix_exp"))
    with pytest.warns(UserWarning, match="hpu-ops#16"):
        assert takes_hpu(torch.randn(4, 4, dtype=torch.float64)) is False
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert takes_hpu(torch.randn(4, 4, dtype=torch.float64)) is False
    assert takes_hpu(torch.randn(tpc.MAXN + 1, tpc.MAXN + 1)) is False
    assert takes_hpu(torch.randn(0, 4, 4)) is False

    for empty_shape in ((0, 0), (2, 0, 0), (0, 3, 3)):
        Ae = torch.empty(empty_shape, dtype=torch.float64, requires_grad=True)
        with cpu_as_hpu(linalg.install_matrix_exp) as calls:
            ge = torch.linalg.matrix_exp(Ae)
            ge.sum().backward()
            assert calls["matrix_exp"] == 0
        assert ge.shape == empty_shape

    with cpu_as_hpu(linalg.install_matrix_exp):
        with pytest.raises(RuntimeError):
            torch.linalg.matrix_exp(torch.randn(3, 4))
        with pytest.raises(RuntimeError):
            torch.linalg.matrix_exp(torch.randn(4))

    # out= overload (both accepted and declined on cpu_as_hpu and mocked HPU backend)
    out = torch.empty(0, dtype=torch.float64)
    with cpu_as_hpu(linalg.install_matrix_exp) as calls:
        r = torch.ops.aten.linalg_matrix_exp.out(A.detach(), out=out)
        assert r is out and calls["matrix_exp"] >= 1
    torch.testing.assert_close(out, want)

    captured = []
    monkeypatch.setattr(linalg, "register_kernels", lambda _d, e: captured.extend(e))
    monkeypatch.setattr(tpc, "ready", lambda: True)
    linalg.install_matrix_exp("hpu")
    out_be = next(fn for op, key, fn in captured if op == torch.ops.aten.linalg_matrix_exp.out and key == "{}")
    monkeypatch.setattr(linalg, "_on_cpu", lambda op, args, kw: op(*(x.clone() for x in args), **kw))
    out_acc = torch.empty(0, dtype=torch.float32)
    A32 = torch.randn(2, 6, 6, dtype=torch.float32)
    assert out_be(A32, out=out_acc) is out_acc
    torch.testing.assert_close(out_acc, torch.linalg.matrix_exp(A32), rtol=1e-5, atol=1e-5)
    out_dec = torch.empty(0, dtype=torch.float32)
    A_big = torch.randn(2, tpc.MAXN + 1, tpc.MAXN + 1, dtype=torch.float32)
    assert out_be(A_big, out=out_dec) is out_dec
    torch.testing.assert_close(out_dec, torch.linalg.matrix_exp(A_big))


@pytest.mark.parametrize("fmode", ["tpc", "wy"])
@pytest.mark.parametrize("m, n, k", [
    (1, 1, 1),
    (4, 4, 4),
    (6, 4, 4),
    (22, 22, 22),
    (22, 22, 21),
    (22, 10, 10),
    (22, 10, 6),
    (64, 32, 32),
])
def test_householder_product_matches_torch(m, n, k, fmode, hh_hook, monkeypatch):
    torch.manual_seed(0)
    monkeypatch.setattr(linalg, "_HH_FORWARD", fmode)
    A = torch.randn(2, m, n, dtype=torch.float64)
    tau = 2.0 / (1.0 + (A.tril(-1)[..., :k] ** 2).sum(-2))
    calls = compare(torch.linalg.householder_product, linalg.install_householder_product, A, tau,
                    name="householder_product", rtol=1e-10, atol=1e-10)
    assert calls["householder_product"] >= 1
    if fmode == "tpc":
        assert hh_hook
    compare(torch.orgqr, linalg.install_householder_product, A, tau,
            name="householder_product", rtol=1e-10, atol=1e-10)


def test_householder_product_noncontig_oversize_and_arbitrary_tau(hh_hook):
    torch.manual_seed(0)
    # Non-contiguous A.mT and strided tau, arbitrary non-unitary tau, batched 4D
    A = torch.randn(2, 3, 4, 6, dtype=torch.float64).mT
    tau = (2.0 * torch.rand(2, 3, 6, dtype=torch.float64))[..., ::2]
    compare(torch.linalg.householder_product, linalg.install_householder_product, A, tau,
            name="householder_product", rtol=1e-9, atol=1e-9)

    # m > MAXHH (68 > 64) falls through to Compact-WY cleanly
    A_big = torch.randn(2, 68, 32, dtype=torch.float64)
    tau_big = 2.0 / (1.0 + (A_big.tril(-1) ** 2).sum(-2))
    compare(torch.linalg.householder_product, linalg.install_householder_product, A_big, tau_big,
            name="householder_product", rtol=1e-10, atol=1e-10)

    # Singular H_0 (tau[0] = 1/||v_0||^2) and tau[1] = 0: native returns NaN, ours stays finite and matches gradcheck!
    A_s = torch.randn(2, 8, 5, dtype=torch.float64)
    v0 = torch.cat([torch.ones(2, 1, dtype=torch.float64), A_s[:, 1:, 0]], -1)
    tau_s = 2.0 / (1.0 + (A_s.tril(-1) ** 2).sum(-2))
    tau_s[:, 0] = 1.0 / (v0 * v0).sum(-1)
    tau_s[:, 1] = 0.0
    A_req = A_s.clone().requires_grad_()
    tau_req = tau_s.clone().requires_grad_()
    with cpu_as_hpu(linalg.install_householder_product) as calls:
        assert torch.autograd.gradcheck(torch.linalg.householder_product, (A_req, tau_req))
        assert torch.autograd.gradgradcheck(torch.linalg.householder_product, (A_req, tau_req))
        assert calls["householder_product"] >= 1


def test_householder_product_k0_empty_gradgrad_compiled_and_declines(monkeypatch, hh_hook):
    torch.manual_seed(0)
    # k = 0 accepted with zero fallbacks; forward == eye(m, n), backward == (None, None)
    A0 = torch.randn(2, 22, 10, dtype=torch.float64, requires_grad=True)
    tau0 = torch.empty(2, 0, dtype=torch.float64, requires_grad=True)
    want0 = torch.linalg.householder_product(A0, tau0)
    want0.sum().backward()
    assert A0.grad is None and tau0.grad is None
    with cpu_as_hpu(linalg.install_householder_product) as calls:
        got0 = torch.linalg.householder_product(A0, tau0)
        got0.sum().backward()
        assert calls["householder_product"] >= 1
    torch.testing.assert_close(got0, want0)
    assert A0.grad is None and tau0.grad is None
    assert linalg._householder_takes("hpu")(A0.float(), tau0.float()) is True

    # Double backward (k < n)
    A = torch.randn(2, 5, 3, dtype=torch.float64, requires_grad=True)
    t = (2.0 * torch.rand(2, 2, dtype=torch.float64)).detach().requires_grad_()
    with cpu_as_hpu(linalg.install_householder_product) as calls:
        assert torch.autograd.gradgradcheck(torch.linalg.householder_product, (A, t))
        assert calls["householder_product"] >= 1

    # compiled(fullgraph=True)
    nodes = []
    with cpu_as_hpu(linalg.install_householder_product) as calls:
        got_c = compiled(torch.linalg.householder_product, nodes, fullgraph=True)(A.detach(), t.detach())
        assert calls["householder_product"] >= 1
    assert "aten.linalg_householder_product.default" not in ops(nodes), ops(nodes)
    torch.testing.assert_close(got_c, torch.linalg.householder_product(A.detach(), t.detach()))

    # Declines and native errors
    A32, t32 = torch.randn(2, 8, 5), torch.rand(2, 4)
    A64, t64 = A32.double(), t32.double()
    linalg._WARNED.clear()
    takes_hpu = linalg._householder_takes("hpu")
    with pytest.warns(UserWarning, match="hpu-ops#16"):
        assert takes_hpu(A64, t64) is False
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert takes_hpu(A64, t64) is False

    for empty_a, empty_t in ((torch.empty(0, 0), torch.empty(0)),
                             (torch.empty(5, 0), torch.empty(0)),
                             (torch.empty(0, 5, 4), torch.empty(0, 4))):
        with cpu_as_hpu(linalg.install_householder_product) as calls:
            ge = torch.linalg.householder_product(empty_a, empty_t)
            assert calls["householder_product"] == 0
        assert torch.equal(ge, torch.linalg.householder_product(empty_a, empty_t))

    with cpu_as_hpu(linalg.install_householder_product):
        with pytest.raises(RuntimeError):
            torch.linalg.householder_product(torch.randn(4, 6), torch.rand(4))
        with pytest.raises(RuntimeError):
            torch.linalg.householder_product(torch.randn(6, 4), torch.rand(5))
        with pytest.raises(RuntimeError):
            torch.linalg.householder_product(torch.randn(2, 6, 4), torch.rand(3, 4))

    # out= backend overload (accept and decline)
    captured = []
    monkeypatch.setattr(linalg, "register_kernels", lambda _d, e: captured.extend(e))
    monkeypatch.setattr(tpc, "ready", lambda: True)
    linalg.install_householder_product("hpu")
    hh_out = next(fn for op, key, fn in captured if op == torch.ops.aten.linalg_householder_product.out and key == "{}")
    monkeypatch.setattr(linalg, "_on_cpu", lambda op, args, kw: op(*(x.clone() for x in args), **kw))
    out_acc = torch.empty(0)
    assert hh_out(A32, t32, out=out_acc) is out_acc
    torch.testing.assert_close(out_acc, torch.linalg.householder_product(A32, t32), rtol=1e-5, atol=1e-5)
    out_dec = torch.empty(0, dtype=torch.float64)
    assert hh_out(A64, t64, out=out_dec) is out_dec
    torch.testing.assert_close(out_dec, torch.linalg.householder_product(A64, t64))


@pytest.mark.parametrize("omap", ["matrix_exp", "householder"])
@pytest.mark.parametrize("shape", [(22, 22), (22, 10), (10, 22)])
def test_orthogonal_matrix_exp_and_householder(omap, shape, expm_hook, hh_hook):
    from torch.nn.utils.parametrizations import orthogonal

    rows, cols = shape
    x = torch.randn(4, cols, dtype=torch.float64)

    def run():
        torch.manual_seed(0)
        layer = orthogonal(torch.nn.Linear(cols, rows, bias=False).double(),
                           orthogonal_map=omap, use_trivialization=False)
        out = layer(x)
        out.sum().backward()
        w = layer.weight.detach()
        return out, w, [p.grad for p in layer.parameters()]

    want_out, want_w, want_grads = run()
    install = linalg.install_matrix_exp if omap == "matrix_exp" else linalg.install_householder_product
    key = "matrix_exp" if omap == "matrix_exp" else "householder_product"
    with cpu_as_hpu(install) as calls:
        got_out, got_w, got_grads = run()
        assert calls[key] >= 1
    torch.testing.assert_close(got_out, want_out, rtol=1e-9, atol=1e-9)
    torch.testing.assert_close(got_w, want_w, rtol=1e-9, atol=1e-9)
    for g, wg in zip(got_grads, want_grads, strict=True):
        torch.testing.assert_close(g, wg, rtol=1e-9, atol=1e-9)
    k = min(rows, cols)
    gram = got_w.mT @ got_w if rows >= cols else got_w @ got_w.mT
    torch.testing.assert_close(gram, torch.eye(k, dtype=torch.float64), rtol=1e-10, atol=1e-10)


def test_flush_comes_before_the_op_only(monkeypatch):
    order = []
    core = types.SimpleNamespace(mark_step=lambda: order.append("flush"))
    torch_mod = types.SimpleNamespace(core=core)
    for name, mod in (("habana_frameworks", types.SimpleNamespace(torch=torch_mod)),
                      ("habana_frameworks.torch", torch_mod), ("habana_frameworks.torch.core", core)):
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.setenv("PT_HPU_LAZY_MODE", "1")
    out = tpc._call(lambda x: order.append("op") or x, 1)
    assert out == 1 and order == ["flush", "op"]


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


def zeroing_hook(A, B):
    """The kernel's contract on a singular triangle: the pivot row is zeroed (a zero row in X), info is nonzero."""
    X, D, S = tpc.gj_solve_reference(A.detach(), B.detach())
    zero = (D == 0)
    X = torch.where(zero.any(-1)[..., None, None], torch.zeros_like(X), X)
    return X, D, S


def test_singular_triangle_gives_nan_like_torch(monkeypatch):
    monkeypatch.setattr(linalg, "_GJ", zeroing_hook)
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
    """A zero diagonal entry, nonzero entries below it: pivots are rarely exactly zero, the diagonal flags it."""
    monkeypatch.setattr(linalg, "_GJ", tpc.gj_solve_reference)
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


def test_enable_does_not_install_the_triangular_solves():
    assert linalg.install_solve_triangular not in hpu_ops.FIXES
    assert linalg.install_cholesky_solve not in hpu_ops.FIXES


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


# ---- standalone LU (#10): lu_factor / lu_factor_ex, lu_solve, lu_unpack, linalg.lu


@pytest.fixture
def lu_hook(monkeypatch):
    """``linalg._LU`` and ``_LU_PERM`` set to the kernel's contract in plain torch; collects the shapes per call."""
    calls = []

    def factor(A):
        calls.append(("factor", tuple(A.shape)))
        return tpc.lu_factor_reference(A)

    def perm(pivots):
        calls.append(("perm", tuple(pivots.shape)))
        return linalg.pivots_to_perm(pivots.detach()).to(torch.float32)

    def solve(LU, pivots, B):
        calls.append(("solve", tuple(B.shape)))
        batch = torch.broadcast_shapes(LU.shape[:-2], B.shape[:-2])
        return linalg.lu_solve(LU.detach().expand(*batch, *LU.shape[-2:]), pivots.expand(*batch, LU.shape[-1]),
                               B.detach(), True)

    monkeypatch.setattr(linalg, "_LU", factor)
    monkeypatch.setattr(linalg, "_LU_PERM", perm)
    monkeypatch.setattr(linalg, "_LU_SOLVE", solve)
    return calls


def _pivoting(*shape, dtype=torch.float64):
    g = torch.Generator().manual_seed(3)
    return torch.randn(*shape, generator=g, dtype=dtype) + 0.5 * torch.eye(shape[-1], dtype=dtype)


@pytest.mark.parametrize("shape", [(1, 1), (3, 3), (4, 22, 22), (2, 3, 7, 7), (32, 32)])
def test_lu_factor_matches_torch(shape, lu_hook):
    A = _pivoting(*shape)
    compare(lambda a: torch.linalg.lu_factor(a)[0], linalg.install_lu, A, name="lu_factor", rtol=1e-8, atol=1e-8)
    want = torch.linalg.lu_factor_ex(A)
    with cpu_as_hpu(linalg.install_lu) as calls:
        got = torch.linalg.lu_factor_ex(A)
        assert calls["lu_factor"] >= 1
    assert got.pivots.dtype == torch.int32 and torch.equal(got.pivots, want.pivots)
    assert got.info.dtype == torch.int32 and torch.equal(got.info, want.info)
    assert ("factor", tuple(shape)) in lu_hook


def test_lu_factor_singular_info_and_check_errors(lu_hook):
    A = _pivoting(3, 5, 5)
    A[1] = 0.0
    A[2, :, 2] = 0.0
    want = torch.linalg.lu_factor_ex(A)
    with cpu_as_hpu(linalg.install_lu):
        got = torch.linalg.lu_factor_ex(A)
        assert torch.equal(got.info, want.info)
        with pytest.raises(RuntimeError):
            torch.linalg.lu_factor_ex(A, check_errors=True)
        with pytest.raises(RuntimeError):
            torch.linalg.lu_factor(A)


def test_lu_factor_double_backward(lu_hook):
    A = _pivoting(2, 4, 4).requires_grad_()
    with cpu_as_hpu(linalg.install_lu) as calls:
        assert torch.autograd.gradcheck(lambda a: torch.linalg.lu_factor(a)[0], (A,))
        assert torch.autograd.gradgradcheck(lambda a: torch.linalg.lu_factor(a)[0], (A,))
        assert calls["lu_factor"] >= 1


@pytest.mark.parametrize("left", [True, False])
@pytest.mark.parametrize("adjoint", [False, True])
@pytest.mark.parametrize("a_shape, b_cols", [((5, 5), 3), ((2, 6, 6), 1), ((2, 1, 4, 4), 40)])
def test_lu_solve_matches_torch(a_shape, b_cols, left, adjoint, lu_hook, gj_hook):
    A = _pivoting(*a_shape)
    n = a_shape[-1]
    LU, pivots = torch.linalg.lu_factor(A)
    batch = (1, 3) if len(a_shape) == 4 else a_shape[:-2]  # (2, 1) against (1, 3): broadcast batch
    B = torch.randn(*batch, *((n, b_cols) if left else (b_cols, n)), dtype=torch.float64)
    compare(lambda lu_, b: torch.linalg.lu_solve(lu_, pivots, b, left=left, adjoint=adjoint), linalg.install_lu,
            LU, B, name="lu_solve", rtol=1e-7, atol=1e-7)
    # the getrs kernel for left, not adjoint (also the backward of left adjoint), else A = P L U and Gauss-Jordan
    assert any(c[0] == "solve" for c in lu_hook) == left
    assert any(c[0] == "perm" for c in lu_hook) and gj_hook  # one direction of every pair runs Gauss-Jordan


def test_lu_solve_double_backward_and_legacy(lu_hook):
    A = _pivoting(2, 4, 4)
    LU, pivots = torch.linalg.lu_factor(A)
    LU = LU.requires_grad_()
    B = torch.randn(2, 4, 3, dtype=torch.float64, requires_grad=True)
    with cpu_as_hpu(linalg.install_lu) as calls:
        assert torch.autograd.gradgradcheck(lambda lu_, b: torch.linalg.lu_solve(lu_, pivots, b), (LU, B))
        got = torch.lu_solve(B, LU, pivots)
        assert calls["lu_solve"] >= 1
    torch.testing.assert_close(got, torch.linalg.solve(A, B))


@pytest.mark.parametrize("unpack_data, unpack_pivots", [(True, True), (False, True), (True, False), (False, False)])
def test_lu_unpack_matches_torch(unpack_data, unpack_pivots, lu_hook):
    A = _pivoting(2, 3, 6, 6)
    LU, pivots = torch.linalg.lu_factor(A)
    want = torch.lu_unpack(LU, pivots, unpack_data, unpack_pivots)
    with cpu_as_hpu(linalg.install_lu) as calls:
        got = torch.lu_unpack(LU, pivots, unpack_data, unpack_pivots)
        assert calls["lu_unpack"] == 1
    for g, w in zip(got, want, strict=True):
        assert g.shape == w.shape and g.dtype == w.dtype
        torch.testing.assert_close(g, w)
    if unpack_data:
        compare(lambda lu_: torch.lu_unpack(lu_, pivots, unpack_data, unpack_pivots)[1:], linalg.install_lu, LU,
                name="lu_unpack")


@pytest.mark.parametrize("shape", [(1, 1), (5, 5), (2, 3, 7, 7)])
def test_linalg_lu_matches_torch(shape, lu_hook):
    A = _pivoting(*shape)
    compare(lambda a: torch.linalg.lu(a), linalg.install_lu, A, name="lu", rtol=1e-8, atol=1e-8)
    with cpu_as_hpu(linalg.install_lu):
        A2 = A.clone().requires_grad_()
        assert torch.autograd.gradgradcheck(lambda a: torch.linalg.lu(a)[1:], (A2,))


def test_lu_declines_what_the_kernel_does_not_take(monkeypatch, lu_hook):
    with cpu_as_hpu(linalg.install_lu) as calls:
        torch.linalg.lu_factor(torch.randn(4, 6, dtype=torch.float64))  # not square
        torch.linalg.lu(torch.randn(3, 3, dtype=torch.complex128))
        assert calls["lu_factor"] == 0 and calls["lu"] == 0
    monkeypatch.setattr(tpc, "ready", lambda: False)
    assert linalg._lu_kernel(torch.randn(tpc.MAXN + 1, tpc.MAXN + 1)) is False
    monkeypatch.setattr(linalg, "_LU", None)
    assert linalg._lu_kernel(torch.randn(4, 4)) is False  # CPU without the hook: the composite
    assert linalg.install_lu  # pivot=False: torch's own kernel (the CPU raises; on HPU Habana's)
    takes = linalg._tpc_takes("cpu", linalg._takes("cpu", "lu_factor"))
    assert takes(torch.randn(4, 4, dtype=torch.float64))


def test_lu_float64_warns_once_on_hpu():
    linalg._WARNED.discard("lu_factor")
    takes = linalg._tpc_takes("hpu", linalg._takes("hpu", "lu_factor"))
    with pytest.warns(UserWarning, match="hpu-ops#16"):
        assert takes(torch.randn(3, 3, dtype=torch.float64)) is False


def test_tpc_lu_flushes_before_the_op_and_casts(monkeypatch):
    marks = []
    core = types.SimpleNamespace(mark_step=lambda: marks.append(1))
    torch_mod = types.SimpleNamespace(core=core)
    for name, mod in (("habana_frameworks", types.SimpleNamespace(torch=torch_mod)),
                      ("habana_frameworks.torch", torch_mod), ("habana_frameworks.torch.core", core)):
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.setenv("PT_HPU_LAZY_MODE", "1")

    def fake_lu(A):
        LU, piv, perm, info = tpc.lu_factor_reference(A)
        pv = torch.zeros(A.shape[0], 3, A.shape[-1])
        pv[:, 0], pv[:, 1], pv[:, 2, 0] = piv.float(), perm.float(), info.float()
        return LU, pv

    monkeypatch.setattr(tpc, "_lu_op", lambda: fake_lu)
    monkeypatch.setattr(tpc, "_lu_perm_op", lambda: lambda p: linalg.pivots_to_perm(p.int()).float())
    A = _pivoting(2, 3, 5, 5, dtype=torch.float32)
    LU, pivots, perm, info = tpc.lu_factor(A)
    assert len(marks) == 1
    want_LU, want_piv, want_info = torch.linalg.lu_factor_ex(A)
    torch.testing.assert_close(LU, want_LU)
    assert pivots.dtype == torch.int32 and torch.equal(pivots, want_piv)
    assert info.dtype == torch.int32 and info.shape == (2, 3) and torch.equal(info, want_info)
    assert perm.dtype == torch.float32 and perm.shape == (2, 3, 5)
    torch.testing.assert_close(A.gather(-2, perm.long()[..., None].expand_as(A)),
                               torch.tril(LU, -1) @ torch.triu(LU) + torch.triu(LU), rtol=1e-5, atol=1e-5)
    marks.clear()
    p = tpc.lu_perm(pivots)
    assert len(marks) == 1 and torch.equal(p, perm)


@pytest.mark.parametrize("perm_mode", ["kernel", "torch"])
@pytest.mark.parametrize("solve_mode", ["kernel", "gj", "tri"])
@pytest.mark.parametrize("left", [True, False])
def test_lu_solve_and_unpack_modes(perm_mode, solve_mode, left, monkeypatch, lu_hook, gj_hook):
    monkeypatch.setattr(linalg, "_LU_PERM_MODE", perm_mode)
    monkeypatch.setattr(linalg, "_LU_SOLVE_MODE", solve_mode)
    A = _pivoting(3, 6, 6)
    LU, pivots = torch.linalg.lu_factor(A)
    B = torch.randn(3, *((6, 2) if left else (2, 6)), dtype=torch.float64)
    compare(lambda lu_, b: torch.linalg.lu_solve(lu_, pivots, b, left=left), linalg.install_lu, LU, B,
            name="lu_solve", rtol=1e-7, atol=1e-7)
    compare(lambda lu_: torch.lu_unpack(lu_, pivots), linalg.install_lu, LU, name="lu_unpack")
    assert any(c[0] == "perm" for c in lu_hook) == (perm_mode == "kernel")


LU_OPS = ("linalg_lu_factor_ex", "linalg_lu", "linalg_lu_solve", "lu_unpack")


@pytest.mark.parametrize("lazy, knob, installed", [
    ("1", None, ()), (None, None, ()), ("1", "0", ()), ("1", "1", LU_OPS), ("0", "1", LU_OPS),
    ("0", None, ("linalg_lu_factor_ex", "linalg_lu_solve", "lu_unpack"))])
def test_lu_ops_installed_only_where_measured_faster(monkeypatch, lazy, knob, installed):
    """Lazy-mode LU, and linalg.lu in both modes, were measured at parity or slower than Habana's fallback: on HPU they
    are installed only with HPU_OPS_LU_SLOW=1."""
    for name, value in (("PT_HPU_LAZY_MODE", lazy), (linalg.LU_SLOW, knob)):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)

    def present():
        return {op for op in LU_OPS if torch._C._dispatch_has_kernel_for_dispatch_key(f"aten::{op}", "AutogradHPU")}

    assert present() == set()
    undo = linalg.install_lu("hpu")
    try:
        assert present() == set(installed)
    finally:
        undo()
    assert present() == set()
    monkeypatch.setenv("PT_HPU_LAZY_MODE", "1")
    monkeypatch.delenv(linalg.LU_SLOW, raising=False)
    assert all(linalg._lu_wanted("cpu", name) for name in ("lu", "lu_factor"))  # the CPU stand-in always installs
