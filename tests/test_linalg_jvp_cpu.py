"""CPU stand-in tests for forward-mode AD (``torch.func.jvp`` and ``torch.autograd.forward_ad``) of
``solve``, ``det`` and ``slogdet`` (``hpu_ops.linalg_jvp``).
"""
import pytest
import torch
import torch.autograd.forward_ad as fwAD
from hpu_ops import linalg, linalg_jvp, tpc
from test_cpu import cpu_as_hpu


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
        fn = lambda a, b: torch.linalg.solve(a, b)
    elif case == "left_false":
        A, B = _general(3, n=4), torch.randn(3, 2, 4)
        fn = lambda a, b: torch.linalg.solve(a, b, left=False)
    elif case == "vector_1d":
        A, B = _general(3, n=4), torch.randn(4)
        fn = lambda a, b: torch.linalg.solve(a, b)
    elif case == "vector_batch":
        A, B = _general(3, n=4), torch.randn(3, 4)
        fn = lambda a, b: torch.linalg.solve(a, b)
    else:
        A, B = _general(1, 3, n=4), torch.randn(2, 1, 4, 3)
        fn = lambda a, b: torch.linalg.solve(a, b)

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
