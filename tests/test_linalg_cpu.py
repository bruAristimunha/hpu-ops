"""The linalg composites checked on the CPU standing in for HPU: outputs and gradients against torch's own, and what
compiled graphs contain."""
import re
import warnings

import pytest
import torch
import torch.utils._pytree as pytree
from test_cpu import compiled, cpu_as_hpu, ops

import hpu_ops
from hpu_ops import eigh as jacobi
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
def test_enable_does_not_install_eigh():
    # measured on Gaudi2 (1.21.4): slower than the CPU fallback, and a device error in lazy mode at n >= 64
    assert jacobi.install_eigh not in hpu_ops.FIXES



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
