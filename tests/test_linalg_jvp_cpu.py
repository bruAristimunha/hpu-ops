"""CPU stand-in tests for forward-mode AD (``torch.func.jvp`` and ``torch.autograd.forward_ad``) of
``solve``, ``det`` and ``slogdet`` (``hpu_ops.linalg_jvp``).
"""
import pytest
import torch
import torch.autograd.forward_ad as fwAD
from test_cpu import cpu_as_hpu

from hpu_ops import _dispatch, linalg, linalg_jvp, tpc


def _general(*batch, n=4, dtype=torch.float32):
    return torch.randn(*batch, n, n, dtype=dtype) + n * torch.eye(n, dtype=dtype)


@pytest.fixture
def gj_hook(monkeypatch):
    calls = []

    def hook(A, B):
        calls.append(tuple(A.shape))
        return tpc.gj_solve_reference(A.detach(), B.detach())

    monkeypatch.setattr(linalg, "_GJ", hook)
    return calls


def _fwad(fn, primals, tangents):
    with fwAD.dual_level():
        duals = [fwAD.make_dual(p, t) if t is not None else p for p, t in zip(primals, tangents)]
        out = fn(*duals)
        if isinstance(out, tuple):
            return tuple(o.detach() if isinstance(o, torch.Tensor) else o for o in out), tuple(
                fwAD.unpack_dual(o).tangent if isinstance(o, torch.Tensor) else None for o in out
            )
        return out.detach(), fwAD.unpack_dual(out).tangent


def test_only_forward_ad_predicate():
    x = torch.eye(3)
    t = torch.randn(3, 3)
    assert linalg_jvp._only_forward_ad(x) is False

    with fwAD.dual_level():
        d = fwAD.make_dual(x, t)
        assert linalg_jvp._only_forward_ad(d) is True
        assert linalg_jvp._only_forward_ad(x) is False
        dr = fwAD.make_dual(x.clone().requires_grad_(True), t)
        assert linalg_jvp._only_forward_ad(dr) is False
        with torch.no_grad():
            assert linalg_jvp._only_forward_ad(dr) is True

    seen = {}

    def record(tag):
        def f(a):
            seen[tag] = linalg_jvp._only_forward_ad(a)
            return a.sum()

        return f

    torch.func.jvp(record("jvp"), (x,), (t,))
    torch.func.vmap(lambda a: torch.func.jvp(record("vmap_jvp"), (a,), (t,))[1])(x.unsqueeze(0))
    torch.func.jvp(lambda a: torch.func.vmap(record("jvp_vmap"))(a), (x.unsqueeze(0),), (t.unsqueeze(0),))
    torch.func.grad(lambda a: torch.func.jvp(record("grad_jvp"), (a,), (t,))[1])(x)
    torch.func.jvp(lambda a: torch.func.grad(record("jvp_grad"))(a), (x,), (t,))
    torch.func.jvp(lambda a: torch.func.jvp(record("jvp_jvp"), (a,), (t,))[1], (x,), (t,))

    assert seen == {
        "jvp": True,
        "vmap_jvp": False,
        "jvp_vmap": False,
        "grad_jvp": False,
        "jvp_grad": False,
        "jvp_jvp": False,
    }


@pytest.mark.parametrize("case", ["matrix", "left_false", "vector_1d", "vector_batch", "broadcast"])
@pytest.mark.parametrize("tangent_mode", ["both", "A_only", "B_only", "neither"])
def test_solve_jvp_and_fwad_match_torch(case, tangent_mode, gj_hook):
    torch.manual_seed(0)
    if case == "matrix":
        A, B = _general(2, 3, n=4), torch.randn(2, 3, 4, 2)

        def fn(a, b):
            return torch.linalg.solve(a, b)
    elif case == "left_false":
        A, B = _general(3, n=4), torch.randn(3, 2, 4)

        def fn(a, b):
            return torch.linalg.solve(a, b, left=False)
    elif case == "vector_1d":
        A, B = _general(3, n=4), torch.randn(4)

        def fn(a, b):
            return torch.linalg.solve(a, b)
    elif case == "vector_batch":
        A, B = _general(3, n=4), torch.randn(3, 4)

        def fn(a, b):
            return torch.linalg.solve(a, b)
    else:
        A, B = _general(1, 3, n=4), torch.randn(2, 1, 4, 3)

        def fn(a, b):
            return torch.linalg.solve(a, b)

    tA = torch.randn_like(A) if tangent_mode in ("both", "A_only") else None
    tB = torch.randn_like(B) if tangent_mode in ("both", "B_only") else None
    want_val, want_tan = _fwad(fn, (A, B), (tA, tB))

    with cpu_as_hpu(linalg.install_solve) as calls:
        got_val, got_tan = _fwad(fn, (A, B), (tA, tB))
        if tangent_mode == "neither":
            assert dict(calls) == {}
            assert gj_hook == []
        else:
            assert calls["solve"] == 1
            assert len(gj_hook) >= 1
            jv_val, jv_tan = torch.func.jvp(
                fn,
                (A, B),
                (torch.zeros_like(A) if tA is None else tA, torch.zeros_like(B) if tB is None else tB),
            )
            torch.testing.assert_close(jv_val, want_val, rtol=1e-4, atol=1e-4)
            torch.testing.assert_close(jv_tan, want_tan, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(got_val, want_val, rtol=1e-4, atol=1e-4)
    if want_tan is None:
        assert got_tan is None
    else:
        torch.testing.assert_close(got_tan, want_tan, rtol=1e-4, atol=1e-4)


def test_solve_jvp_large_path_and_inplace_output(monkeypatch, gj_hook):
    import kernel_replica as kr

    monkeypatch.setattr(tpc, "MAXN", 2)
    monkeypatch.setattr(linalg, "_LARGE_ON_CPU", True)
    monkeypatch.setattr(tpc, "_large_op", lambda: kr.gj_large_op())
    torch.manual_seed(1)
    A = _general(3, n=4)
    B = torch.randn(4)
    tA = torch.randn_like(A)
    tB = torch.randn_like(B)
    want_val, want_tan = _fwad(torch.linalg.solve, (A, B), (tA, tB))
    with cpu_as_hpu(linalg.install_solve) as calls:
        with fwAD.dual_level():
            out = torch.linalg.solve(fwAD.make_dual(A, tA), fwAD.make_dual(B, tB))
            got_val = out.detach().clone()
            got_tan = fwAD.unpack_dual(out).tangent.clone()
            out.add_(1.0)
        assert calls["solve"] == 1
    assert gj_hook == []  # large path used gj_inv rather than gj_solve
    torch.testing.assert_close(got_val, want_val, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(got_tan, want_tan, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shape", [(4, 4), (2, 3, 4, 4)])
def test_det_and_slogdet_jvp_and_fwad_match_torch(shape, gj_hook):
    torch.manual_seed(2)
    A = _general(*shape[:-2], n=shape[-1])
    A[..., 0, :] = -A[..., 0, :]  # negative determinant in the batch
    t = torch.randn_like(A)

    want_det = torch.func.jvp(torch.linalg.det, (A,), (t,))
    want_slog = torch.func.jvp(lambda a: tuple(torch.linalg.slogdet(a)), (A,), (t,))

    with cpu_as_hpu(linalg.install_inv_det) as calls:
        got_det_fw = _fwad(torch.linalg.det, (A,), (t,))
        got_det_jv = torch.func.jvp(torch.linalg.det, (A,), (t,))
        got_slog_fw = _fwad(lambda a: tuple(torch.linalg.slogdet(a)), (A,), (t,))
        got_slog_jv = torch.func.jvp(lambda a: tuple(torch.linalg.slogdet(a)), (A,), (t,))
        assert calls["det"] == 2
        assert calls["slogdet"] == 2
        assert len(gj_hook) >= 4

    for got in (got_det_fw, got_det_jv):
        torch.testing.assert_close(got[0], want_det[0], rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(got[1], want_det[1], rtol=1e-4, atol=1e-4)
    for got in (got_slog_fw, got_slog_jv):
        torch.testing.assert_close(got[0][0], want_slog[0][0])
        torch.testing.assert_close(got[0][1], want_slog[0][1], rtol=1e-4, atol=1e-4)
        assert torch.equal(got[1][0], torch.zeros_like(want_slog[0][0]))
        torch.testing.assert_close(got[1][1], want_slog[1][1], rtol=1e-4, atol=1e-4)


def test_det_jvp_singular_returns_zero_like_torch(gj_hook):
    A = torch.stack([
        _general(n=3),
        torch.tensor([[1.0, 2.0, 0.0], [2.0, 4.0, 0.0], [0.0, 0.0, 3.0]]),
        torch.diag(torch.tensor([1.0, 2.0, 0.0])),
    ])
    t = torch.ones_like(A)
    want_val, want_tan = _fwad(torch.linalg.det, (A,), (t,))
    assert want_tan[1] == 0.0 and want_tan[2] == 0.0
    with cpu_as_hpu(linalg.install_inv_det):
        got_val, got_tan = _fwad(torch.linalg.det, (A,), (t,))
        jv_val, jv_tan = torch.func.jvp(torch.linalg.det, (A,), (t,))
    torch.testing.assert_close(got_val, want_val, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(got_tan, want_tan, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(jv_val, want_val, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(jv_tan, want_tan, rtol=1e-4, atol=1e-4)


def test_declined_transforms_and_float64_hpu_warning(monkeypatch, gj_hook):
    torch.manual_seed(3)
    A = _general(3, n=4)
    B = torch.randn(3, 4, 2)
    tA = torch.randn_like(A)
    tB = torch.randn_like(B)

    jvp_used = []
    for cls in (linalg_jvp._JvpSolveEx, linalg_jvp._JvpDet, linalg_jvp._JvpSlogdet):
        orig_fwd = cls.forward
        monkeypatch.setattr(cls, "forward", staticmethod(lambda *a, _f=orig_fwd, _n=cls.__name__: (jvp_used.append(_n), _f(*a))[1]))

    def chain(d):
        from hpu_ops._dispatch import chain as _chain
        return _chain([linalg.install_solve(d), linalg.install_inv_det(d)])

    want_vmap_jvp = torch.func.vmap(lambda a, b, ta, tb: torch.func.jvp(torch.linalg.solve, (a, b), (ta, tb))[1])(
        A, B, tA, tB
    )
    want_grad_jvp = torch.func.grad(lambda a: torch.func.jvp(torch.linalg.det, (a,), (tA[0],))[1])(A[0])
    want_jvp_jvp = torch.func.jvp(
        lambda a: torch.func.jvp(torch.linalg.det, (a,), (tA[0],))[1], (A[0],), (tA[0],)
    )[1]

    with cpu_as_hpu(chain):
        got_vmap_jvp = torch.func.vmap(lambda a, b, ta, tb: torch.func.jvp(torch.linalg.solve, (a, b), (ta, tb))[1])(
            A, B, tA, tB
        )
        got_grad_jvp = torch.func.grad(lambda a: torch.func.jvp(torch.linalg.det, (a,), (tA[0],))[1])(A[0])
        got_jvp_jvp = torch.func.jvp(
            lambda a: torch.func.jvp(torch.linalg.det, (a,), (tA[0],))[1], (A[0],), (tA[0],)
        )[1]
    assert jvp_used == []
    torch.testing.assert_close(got_vmap_jvp, want_vmap_jvp)
    torch.testing.assert_close(got_grad_jvp, want_grad_jvp)
    torch.testing.assert_close(got_jvp_jvp, want_jvp_jvp)

    linalg._WARNED.clear()
    takes_det = linalg_jvp._base_jvp_takes("hpu", "det")
    with fwAD.dual_level():
        f64 = fwAD.make_dual(A[0].double(), tA[0].double())
        with pytest.warns(UserWarning, match="hpu-ops#16"):
            assert takes_det(f64) is False


def test_forward_ad_level_detects_cpp_dual_level():
    assert _dispatch.forward_ad_level() == -1
    with fwAD.dual_level():
        assert _dispatch.forward_ad_level() == 0
    lvl = torch._C._enter_dual_level()
    try:
        assert getattr(fwAD, "_current_level", -1) == -1
        x = torch.eye(2)
        d = torch._VF._make_dual(x, torch.ones_like(x), lvl)
        assert _dispatch.forward_ad_level(d) == 0
        from torch._subclasses.fake_tensor import FakeTensorMode

        with FakeTensorMode():
            assert _dispatch.forward_ad_level() == 0
    finally:
        torch._C._exit_dual_level(lvl)
    assert _dispatch.forward_ad_level() == -1


def _assert_tree_close(got, want):
    if isinstance(want, tuple):
        assert isinstance(got, tuple) and len(got) == len(want)
        for g, w in zip(got, want):
            _assert_tree_close(g, w)
    elif want is None:
        assert got is None
    else:
        torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("op_name", ["solve", "solve_left_false", "solve_vec", "det", "slogdet"])
def test_compile_jvp_and_fwad_match_torch(dynamic, op_name, gj_hook):
    torch.manual_seed(4)
    A = _general(2, n=4)
    A[..., 0, :] = -A[..., 0, :]
    tA = torch.randn_like(A)

    if op_name == "solve":
        B, tB = torch.randn(2, 4, 3), torch.randn(2, 4, 3)

        def fn(a, b):
            return torch.linalg.solve(a, b)

        primals, tangents = (A, B), (tA, tB)
    elif op_name == "solve_left_false":
        B, tB = torch.randn(2, 3, 4), torch.randn(2, 3, 4)

        def fn(a, b):
            return torch.linalg.solve(a, b, left=False)

        primals, tangents = (A, B), (tA, tB)
    elif op_name == "solve_vec":
        B, tB = torch.randn(2, 4), torch.randn(2, 4)

        def fn(a, b):
            return torch.linalg.solve(a, b)

        primals, tangents = (A, B), (tA, tB)
    elif op_name == "det":
        fn = torch.linalg.det
        primals, tangents = (A,), (tA,)
    else:

        def fn(a):
            return tuple(torch.linalg.slogdet(a))

        primals, tangents = (A,), (tA,)

    def run_jvp(*args):
        n = len(primals)
        return torch.func.jvp(fn, args[:n], args[n:])

    def run_fwad(*args):
        n = len(primals)
        with fwAD.dual_level():
            duals = [fwAD.make_dual(p, t) for p, t in zip(args[:n], args[n:])]
            out = fn(*duals)
            if isinstance(out, tuple):
                return tuple(o.detach() for o in out), tuple(fwAD.unpack_dual(o).tangent for o in out)
            return out.detach(), fwAD.unpack_dual(out).tangent

    want_jvp = run_jvp(*primals, *tangents)
    want_fwad = run_fwad(*primals, *tangents)

    def install(d):
        return _dispatch.chain([linalg.install_solve(d), linalg.install_inv_det(d)])

    key = "solve" if op_name.startswith("solve") else op_name
    with cpu_as_hpu(install) as calls:
        for runner, want in ((run_jvp, want_jvp), (run_fwad, want_fwad)):
            torch._dynamo.reset()
            got = torch.compile(runner, backend="aot_eager", dynamic=dynamic)(*primals, *tangents)
            _assert_tree_close(got, want)
        if not dynamic:
            assert calls[key] >= 2


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("tangent_mode", ["A_only", "B_only", "neither"])
def test_compile_fwad_solve_partial_and_no_tangents(dynamic, tangent_mode, gj_hook):
    torch.manual_seed(5)
    A = _general(n=4)
    B = torch.randn(4, 2)
    tA = torch.randn_like(A) if tangent_mode == "A_only" else None
    tB = torch.randn_like(B) if tangent_mode == "B_only" else None

    def run_fwad(a, b):
        with fwAD.dual_level():
            da = fwAD.make_dual(a, tA) if tA is not None else a
            db = fwAD.make_dual(b, tB) if tB is not None else b
            out = torch.linalg.solve(da, db)
            return out.detach(), fwAD.unpack_dual(out).tangent

    want = run_fwad(A, B)
    with cpu_as_hpu(linalg.install_solve):
        torch._dynamo.reset()
        got = torch.compile(run_fwad, backend="aot_eager", dynamic=dynamic)(A, B)
    _assert_tree_close(got, want)


@pytest.mark.parametrize("op_name", ["solve", "det", "slogdet"])
def test_compile_jvp_with_requires_grad_and_traced_large_path(monkeypatch, op_name, gj_hook):
    torch.manual_seed(6)
    A = _general(2, n=4)
    B = torch.randn(2, 4, 2)
    tA = torch.randn_like(A)
    tB = torch.randn_like(B)

    def run(a, b, ta, tb):
        if op_name == "solve":
            return torch.func.jvp(torch.linalg.solve, (a, b), (ta, tb))
        if op_name == "det":
            return torch.func.jvp(torch.linalg.det, (a,), (ta,))
        return torch.func.jvp(lambda x: tuple(torch.linalg.slogdet(x)), (a,), (ta,))

    want = run(A, B, tA, tB)

    def install(d):
        return _dispatch.chain([linalg.install_solve(d), linalg.install_inv_det(d)])

    # Declined via requires_grad=True under compile
    with cpu_as_hpu(install):
        torch._dynamo.reset()
        got_rg = torch.compile(run, backend="aot_eager")(A.clone().requires_grad_(True), B, tA, tB)
    _assert_tree_close(got_rg, want)

    # Declined while tracing (e.g. HPU large route n > MAXN) -> _TracedJvp* at trace time, gj_hook at runtime
    tpc_takes = linalg._tpc_takes

    def gated(dt, takes, *a, **k):
        inner = tpc_takes(dt, takes, *a, **k)
        return lambda *args, **kw: inner(*args, **kw) and not linalg._tracing(args[0])

    monkeypatch.setattr(linalg, "_tpc_takes", gated)
    with cpu_as_hpu(install):
        torch._dynamo.reset()
        got_traced = torch.compile(run, backend="aot_eager")(A, B, tA, tB)
    _assert_tree_close(got_traced, want)


def _fd_grad_a(scalar_fn, A, eps=1e-6):
    A64 = A.detach().double()
    grad = torch.zeros_like(A64)
    flat = A64.reshape(-1)
    gflat = grad.reshape(-1)
    for i in range(flat.numel()):
        e = torch.zeros_like(flat)
        e[i] = eps
        A_p = (flat + e).reshape(A64.shape)
        A_m = (flat - e).reshape(A64.shape)
        gflat[i] = (scalar_fn(A_p) - scalar_fn(A_m)) / (2 * eps)
    return grad.to(A.dtype)


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("op_name", ["solve", "det", "slogdet"])
def test_compile_jvp_and_fwad_requires_grad_backward(dynamic, op_name, gj_hook):
    """Bug 1: ``torch.compile(torch.func.jvp, dynamic=False/True)`` and ``torch.compile(fwAD)`` with
    ``requires_grad=True`` + ``.backward()`` must match float64 finite differences without reading placeholder
    ``LU=0``."""
    torch.manual_seed(7)
    A = _general(2, n=3)
    B = torch.randn(2, 3, 2)
    tA = torch.randn_like(A)
    tB = torch.randn_like(B)

    def make_jvp_scalar(a_in, b_in, ta_in, tb_in):
        if op_name == "solve":
            p, t = torch.func.jvp(torch.linalg.solve, (a_in, b_in), (ta_in, tb_in))
            return p.sum() + t.sum()
        if op_name == "det":
            p, t = torch.func.jvp(torch.linalg.det, (a_in,), (ta_in,))
            return p.sum() + t.sum()
        p, t = torch.func.jvp(lambda x: tuple(torch.linalg.slogdet(x)), (a_in,), (ta_in,))
        return p[1].sum() + t[1].sum()

    def make_fwad_scalar(a_in, b_in, ta_in, tb_in):
        with fwAD.dual_level():
            da = fwAD.make_dual(a_in, ta_in)
            db = fwAD.make_dual(b_in, tb_in)
            if op_name == "solve":
                out = torch.linalg.solve(da, db)
            elif op_name == "det":
                out = torch.linalg.det(da)
            else:
                out = torch.linalg.slogdet(da)[1]
            return out + fwAD.unpack_dual(out).tangent

    want_grad = _fd_grad_a(lambda a64: make_jvp_scalar(a64, B.double(), tA.double(), tB.double()), A)

    def install(d):
        return _dispatch.chain([linalg.install_solve(d), linalg.install_inv_det(d)])

    with cpu_as_hpu(install):
        for runner in (make_jvp_scalar, make_fwad_scalar):
            torch._dynamo.reset()
            Ag = A.clone().requires_grad_(True)
            loss = torch.compile(runner, backend="aot_eager", dynamic=dynamic)(Ag, B, tA, tB)
            loss.sum().backward()
            assert torch.isfinite(Ag.grad).all()
            torch.testing.assert_close(Ag.grad, want_grad, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("op_name", ["solve", "det", "slogdet"])
def test_compile_jacfwd_match_torch(monkeypatch, dynamic, op_name, gj_hook):
    """Bug 2: ``torch.compile(torch.func.jacfwd(...), dynamic=False/True)`` must match native ``jacfwd`` for both
    normal and traced-declined paths."""
    torch.manual_seed(8)
    A = _general(2, n=3)
    B = torch.randn(2, 3, 2)

    if op_name == "solve":
        def jac_fn(a, b):
            return torch.func.jacfwd(torch.linalg.solve)(a, b)
        args = (A, B)
    elif op_name == "det":
        def jac_fn(a):
            return torch.func.jacfwd(torch.linalg.det)(a)
        args = (A,)
    else:
        def jac_fn(a):
            return torch.func.jacfwd(lambda x: torch.linalg.slogdet(x)[1])(a)
        args = (A,)

    want = jac_fn(*args)

    def install(d):
        return _dispatch.chain([linalg.install_solve(d), linalg.install_inv_det(d)])

    with cpu_as_hpu(install):
        torch._dynamo.reset()
        got = torch.compile(jac_fn, backend="aot_eager", dynamic=dynamic)(*args)
        assert torch.isfinite(got).all()
        torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-4)

        # Also exercise _TracedJvp* + _Traced*.apply when _tpc_takes declines while tracing.
        tpc_takes = linalg._tpc_takes
        monkeypatch.setattr(
            linalg,
            "_tpc_takes",
            lambda dt, takes, *a, **k: (lambda *ar, **kw: tpc_takes(dt, takes, *a, **k)(*ar, **kw) and not linalg._tracing(ar[0])),
        )
        torch._dynamo.reset()
        got_traced = torch.compile(jac_fn, backend="aot_eager", dynamic=dynamic)(*args)
        assert torch.isfinite(got_traced).all()
        torch.testing.assert_close(got_traced, want, rtol=1e-4, atol=1e-4)


def test_backend_solve_safe():
    """Bug 3: ``_backend_solve_safe`` allows inference_mode and eager/lazy no-tangent ``fwAD.dual_level()`` without
    reverse grad, while declining ``torch.func.*`` (including ``JVP_NESTING > 0``), tangents, and ``requires_grad``."""
    A = _general(n=3)
    B = torch.randn(3, 2)
    tA = torch.randn_like(A)

    assert linalg._backend_solve_safe((A, B, True, False)) is True
    with torch.inference_mode():
        assert linalg._backend_solve_safe((A, B, True, False)) is True

    with fwAD.dual_level():
        assert linalg._backend_solve_safe((A, B, True, False)) is True
        assert linalg._backend_solve_safe((fwAD.make_dual(A, tA), B, True, False)) is False
        Ar = A.clone().requires_grad_(True)
        assert linalg._backend_solve_safe((Ar, B, True, False)) is False
        with torch.no_grad():
            assert linalg._backend_solve_safe((Ar, B, True, False)) is True

    seen = {}
    torch.func.jvp(lambda a, b: (seen.setdefault("jvp", linalg._backend_solve_safe((a, b, True, False))), a)[1], (A, B), (tA, B))
    torch.func.vmap(lambda a, b: (seen.setdefault("vmap", linalg._backend_solve_safe((a, b, True, False))), a)[1])(
        A.unsqueeze(0), B.unsqueeze(0)
    )
    torch.func.grad(lambda a: (seen.setdefault("grad", linalg._backend_solve_safe((a, B, True, False))), a.sum())[1])(A)
    assert seen == {"jvp": False, "vmap": False, "grad": False}

    # Simulate Habana fork where torch.func.jvp strips DualTensor & stack above AutogradHPU, leaving JVP_NESTING > 0.
    import torch._functorch.eager_transforms as _et

    with fwAD.dual_level(), _et.jvp_increment_nesting():
        assert linalg._backend_solve_safe((A, B, True, False)) is False

    # C++ _enter_dual_level() without Python fwAD._current_level >= 0 must decline.
    lvl = torch._C._enter_dual_level()
    try:
        assert linalg._backend_solve_safe((A, B, True, False)) is False
    finally:
        torch._C._exit_dual_level(lvl)

    # Below autograd (where VariableType strips requires_grad & tangents), _solve_declined_unsafe propagates
    # via _BACKEND_SOLVE_UNSAFE, and _backend_solve_safe does not call _unpack_dual on below-autograd tensors.
    with fwAD.dual_level():
        assert linalg._solve_declined_unsafe((A, B), {}) is False
        assert linalg._solve_declined_unsafe((Ar, B), {}) is True
        assert linalg._solve_declined_unsafe((fwAD.make_dual(A, tA), B), {}) is True
        with torch._C._AutoDispatchBelowAutograd():
            assert linalg._backend_solve_safe((A, B, True, False)) is True
            linalg._BACKEND_SOLVE_UNSAFE.depth = 1
            try:
                assert linalg._backend_solve_safe((A, B, True, False)) is False
            finally:
                linalg._BACKEND_SOLVE_UNSAFE.depth = 0


def test_compile_eager_backend_inside_fwad_dual_level_linalg():
    """Bug 4: ``torch.compile(..., backend="eager")`` lowers ``fwAD.dual_level()`` to C++
    ``_enter_dual_level()`` with ``fwAD._current_level == -1`` and ``_len_torch_dispatch_stack() == 0``;
    ``linalg.svd`` must detect C++ dual level 0 and decline to native instead of calling ``_Svd`` on DualTensors."""
    A = _general(n=3)
    tA = torch.randn_like(A)

    def run(a, ta):
        with fwAD.dual_level():
            s = torch.linalg.svd(fwAD.make_dual(a, ta))[1]
            return s.detach(), fwAD.unpack_dual(s).tangent

    want_p, want_t = run(A, tA)
    torch._dynamo.reset()
    compiled_run = torch.compile(run, backend="eager", fullgraph=True)

    with cpu_as_hpu(linalg.install_svd) as calls:
        got_p, got_t = compiled_run(A, tA)
        assert calls["svd"] == 0

    torch.testing.assert_close(got_p, want_p, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(got_t, want_t, rtol=1e-4, atol=1e-4)


def test_solve_backend_kernel_output_strides(monkeypatch):
    """Bug 5: ``SolveEx.compute_with_inverse`` (and thus both ``SolveEx.forward`` and ``backend_kernel``) must match
    ``aten._linalg_solve_ex``'s ``mT.contiguous().mT`` stride convention when ``left=True`` and ``X.dim() >= 2``, and
    row-major contiguous when ``left=False``."""
    monkeypatch.setattr(linalg, "_GJ", tpc.gj_solve_reference)
    monkeypatch.setattr(linalg, "_tpc_takes", lambda dt, fn, **kw: fn)
    groups = []
    orig_register = linalg.register_paired
    monkeypatch.setattr(linalg, "register_paired", lambda dt, gs: (groups.extend(gs), orig_register(dt, []))[1])
    linalg.install_solve("hpu")
    backend_fn = {key: fn for _, key, fn in groups[0]}["{}"]

    torch.manual_seed(9)
    cases = [
        (_general(n=6), torch.randn(6, 2), True),
        (_general(2, n=6), torch.randn(2, 6, 3), True),
        (_general(2, 3, n=6), torch.randn(2, 3, 6, 4), True),
        (_general(n=6), torch.randn(6), True),
        (_general(2, n=6), torch.randn(2, 6), True),
        (_general(2, n=6), torch.randn(2, 3, 6), False),
    ]
    for A, B, left in cases:
        want_X = torch.ops.aten._linalg_solve_ex.default(A, B, left=left)[0]
        got_fwd_X = linalg.SolveEx.apply(A, B, left, False)[0]
        assert got_fwd_X.stride() == want_X.stride(), (A.shape, B.shape, left, got_fwd_X.stride(), want_X.stride())
        torch.testing.assert_close(got_fwd_X, want_X, rtol=1e-4, atol=1e-4)
        with torch.inference_mode():
            got_X = backend_fn(A, B, left=left)[0]
            assert got_X.stride() == want_X.stride(), (A.shape, B.shape, left, got_X.stride(), want_X.stride())
            torch.testing.assert_close(got_X, want_X, rtol=1e-4, atol=1e-4)
