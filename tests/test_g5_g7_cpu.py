"""CPU tests for G7 (im2col/col2im HPU backend override + AOT FunctionalTensor _tracing) and G5 (_linalg_solve_ex
HPU backend key for inference_mode + _raw_transformed)."""
import pytest
import torch
import torch.nn.functional as F
from test_cpu import compiled, cpu_as_hpu

from hpu_ops import _dispatch, im2col, linalg, tpc

aten = torch.ops.aten


# ---- G7: im2col / col2im backend override & _tracing


def test_im2col_install_hpu_registers_and_undoes_backend_keys():
    ops_all = ("aten::im2col", "aten::im2col.out", "aten::col2im", "aten::col2im.out")
    ops_autograd = ("aten::im2col", "aten::col2im")
    for name in ops_all:
        assert not torch._C._dispatch_has_kernel_for_dispatch_key(name, "HPU")
    for name in ops_autograd:
        assert not torch._C._dispatch_has_kernel_for_dispatch_key(name, "AutogradHPU")

    undo = im2col.install("hpu")
    try:
        for name in ops_all:
            assert torch._C._dispatch_has_kernel_for_dispatch_key(name, "HPU"), name
        for name in ops_autograd:
            assert torch._C._dispatch_has_kernel_for_dispatch_key(name, "AutogradHPU"), name
    finally:
        undo()

    for name in ops_all:
        assert not torch._C._dispatch_has_kernel_for_dispatch_key(name, "HPU"), name
    for name in ops_autograd:
        assert not torch._C._dispatch_has_kernel_for_dispatch_key(name, "AutogradHPU"), name


def test_im2col_install_hpu_overrides_existing_backend_stub(monkeypatch):
    """Simulates Habana's lazy mode where HPU already reports a C++ kernel on im2col/col2im."""
    orig_has = torch._C._dispatch_has_kernel_for_dispatch_key
    monkeypatch.setattr(
        torch._C,
        "_dispatch_has_kernel_for_dispatch_key",
        lambda name, key: True if key == "HPU" else orig_has(name, key),
    )
    registered = []
    orig_override = im2col.override_kernels
    monkeypatch.setattr(
        im2col,
        "override_kernels",
        lambda dt, ks: (registered.extend((dt, op.name()) for op, _ in ks), orig_override(dt, ks))[1],
    )
    undo = im2col.install("hpu")
    try:
        assert registered == [
            ("hpu", "aten::im2col"),
            ("hpu", "aten::im2col.out"),
            ("hpu", "aten::col2im"),
            ("hpu", "aten::col2im.out"),
        ]
    finally:
        undo()


def test_im2col_col2im_override_inference_mode_and_undo():
    """override_kernels on CPU + inference_mode hits im2col/col2im rewrites for F.unfold/F.fold and undo restores."""
    torch.manual_seed(0)
    x = torch.randn(2, 3, 7, 9, dtype=torch.float64)
    y = torch.randn(2, 18, 24, dtype=torch.float64)
    with torch.inference_mode():
        want_u = F.unfold(x, (2, 3), stride=(1, 2))
        want_f = F.fold(y, (7, 9), (2, 3), stride=(1, 2))
        want_u_out = aten.im2col.out(x, [2, 3], [1, 1], [0, 0], [1, 2], out=torch.empty(1, dtype=torch.float64))
        want_f_out = aten.col2im.out(y, [7, 9], [2, 3], [1, 1], [0, 0], [1, 2], out=torch.empty(1, dtype=torch.float64))

    _dispatch._STATE["calls"].clear()
    undo = _dispatch.override_kernels("cpu", [
        (aten.im2col.default, im2col._im2col_kernel("cpu", aten.im2col.default)),
        (aten.im2col.out, im2col._im2col_out_kernel("cpu", aten.im2col.out)),
        (aten.col2im.default, im2col._col2im_kernel("cpu", aten.col2im.default)),
        (aten.col2im.out, im2col._col2im_out_kernel("cpu", aten.col2im.out)),
    ])
    try:
        with torch.inference_mode():
            got_u = F.unfold(x, (2, 3), stride=(1, 2))
            got_f = F.fold(y, (7, 9), (2, 3), stride=(1, 2))
            got_u_out = aten.im2col.out(x, [2, 3], [1, 1], [0, 0], [1, 2], out=torch.empty(1, dtype=torch.float64))
            got_f_out = aten.col2im.out(y, [7, 9], [2, 3], [1, 1], [0, 0], [1, 2],
                                        out=torch.empty(1, dtype=torch.float64))
        assert _dispatch._STATE["calls"]["im2col"] == 2
        assert _dispatch._STATE["calls"]["col2im"] == 2
        torch.testing.assert_close(got_u, want_u, atol=0, rtol=0)
        torch.testing.assert_close(got_f, want_f, atol=0, rtol=0)
        torch.testing.assert_close(got_u_out, want_u_out, atol=0, rtol=0)
        torch.testing.assert_close(got_f_out, want_f_out, atol=0, rtol=0)
    finally:
        undo()

    _dispatch._STATE["calls"].clear()
    with torch.inference_mode():
        after_u = F.unfold(x, (2, 3), stride=(1, 2))
        after_f = F.fold(y, (7, 9), (2, 3), stride=(1, 2))
    assert not _dispatch._STATE["calls"]["im2col"] and not _dispatch._STATE["calls"]["col2im"]
    torch.testing.assert_close(after_u, want_u, atol=0, rtol=0)
    torch.testing.assert_close(after_f, want_f, atol=0, rtol=0)


def test_im2col_tracing_detects_aot_functional_tensor_and_modes():
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch._subclasses.functional_tensor import (
        FunctionalTensor,
        FunctionalTensorMode,
    )
    from torch.fx.experimental.proxy_tensor import make_fx

    x = torch.randn(2, 3, 7, 9)
    assert im2col._tracing(x) is False

    seen = []

    def fn(t):
        seen.append((type(t).__name__, torch._is_functional_tensor(t), im2col._tracing(t)))
        return F.unfold(t, (2, 3), padding=1)

    with FakeTensorMode():
        fx = torch.randn(2, 3, 7, 9)
        assert im2col._tracing(fx) is True

    with FakeTensorMode(), FunctionalTensorMode():
        ft = FunctionalTensor.to_functional(torch.randn(2, 3, 7, 9))
        assert im2col._tracing(ft) is True

    torch.func.functionalize(fn)(x)
    assert seen and all(res is True for _, _, res in seen), seen

    proxy_seen = []

    def pfn(t):
        proxy_seen.append(im2col._tracing(t))
        return t + 1

    make_fx(pfn)(x)
    assert proxy_seen == [True]


def test_compiled_unfold_fold_fx_graph_has_no_cpu_arange_or_to_copy(monkeypatch):
    """With AOT FunctionalTensor detected as tracing, Tier 2 im2col/col2im always passes tracing=True to
    _flat_indices (so idx_device == self.device, never CPU + _to_copy) and emits 0 _to_copy nodes."""
    tracing_flags = []
    orig_flat = im2col._flat_indices

    def spy_flat(height, width, kh, kw, dh, dw, ph, pw, sh, sw, device, tracing):
        tracing_flags.append(tracing)
        return orig_flat(height, width, kh, kw, dh, dw, ph, pw, sh, sw, device, tracing)

    monkeypatch.setattr(im2col, "_flat_indices", spy_flat)

    x = torch.randn(2, 3, 7, 9, requires_grad=True)
    y = torch.randn(2, 12, 48, requires_grad=True)
    nodes_u, nodes_f = [], []
    with cpu_as_hpu(im2col.install) as calls:
        gu = compiled(lambda t: F.unfold(t, (2, 2), padding=1), nodes_u, fullgraph=True)
        gf = compiled(lambda t: F.fold(t, (7, 9), (2, 2), stride=(1, 1)), nodes_f, fullgraph=True)
        gu(x).sum().backward()
        gf(y).sum().backward()
    assert calls["im2col"] >= 1 and calls["col2im"] >= 1
    assert tracing_flags and all(tracing_flags), tracing_flags
    for label, nodes in (("unfold", nodes_u), ("fold", nodes_f)):
        cpu_to_copy = [
            n for n in nodes
            if "_to_copy" in str(n.target)
            and any(isinstance(a, torch.fx.Node) and "arange" in str(a.target) for a in n.args)
        ]
        assert not cpu_to_copy, f"{label} emitted CPU _to_copy: {cpu_to_copy}"


# ---- G5: _linalg_solve_ex HPU backend key & _raw_transformed


def _capture_solve_backend(device_type, monkeypatch):
    groups = []
    orig = linalg.register_paired
    monkeypatch.setattr(linalg, "register_paired", lambda dt, gs: (groups.extend(gs), orig(dt, []))[1])
    linalg.install_solve(device_type)
    by_key = {key: fn for _, key, fn in groups[0]}
    return by_key["{}"]


def test_raw_transformed_ignores_below_guard_and_detects_transforms():
    import torch.autograd.forward_ad as fwAD

    assert linalg._raw_transformed() is False
    with fwAD.dual_level():
        assert linalg._raw_transformed() is True

    seen = []
    x = torch.randn(2, 3)
    torch.func.vmap(lambda t: (seen.append(("vmap", linalg._raw_transformed())), t)[1])(x)
    torch.func.grad(lambda t: (seen.append(("grad", linalg._raw_transformed())), t.sum())[1])(x)
    torch.func.jvp(lambda t: (seen.append(("jvp", linalg._raw_transformed())), t)[1], (x,), (x,))
    assert seen == [("vmap", True), ("grad", True), ("jvp", True)]


def test_solve_backend_kernel_inference_mode_and_undo(monkeypatch):
    monkeypatch.setattr(linalg, "_GJ", tpc.gj_solve_reference)
    backend_fn = _capture_solve_backend("cpu", monkeypatch)

    torch.manual_seed(0)
    A = torch.eye(4) + 0.1 * torch.randn(2, 4, 4)
    B = torch.randn(2, 4, 3)
    with torch.inference_mode():
        want = torch.linalg.solve(A, B)

    _dispatch._STATE["calls"].clear()
    undo = _dispatch.override_kernels("cpu", [(aten._linalg_solve_ex.default, backend_fn)])
    try:
        with torch.inference_mode():
            got_eager = torch.linalg.solve(A, B)
            torch._dynamo.reset()
            got_compiled = torch.compile(torch.linalg.solve, backend="aot_eager", fullgraph=True)(A, B)
        assert _dispatch._STATE["calls"]["solve"] == 2
        torch.testing.assert_close(got_eager, want, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(got_compiled, want, rtol=1e-4, atol=1e-5)
    finally:
        undo()

    _dispatch._STATE["calls"].clear()
    with torch.inference_mode():
        after = torch.linalg.solve(A, B)
    assert not _dispatch._STATE["calls"]["solve"]
    torch.testing.assert_close(after, want, atol=0, rtol=0)


def test_solve_backend_kernel_routes_transforms_and_float64_to_on_cpu(monkeypatch):
    monkeypatch.setattr(linalg, "_GJ", tpc.gj_solve_reference)
    monkeypatch.setattr(linalg, "_WARNED", set())
    monkeypatch.setattr(linalg, "_tpc_takes", lambda dt, fn, **kw: fn)  # avoid tpc.ready() check on "hpu"
    backend_fn = _capture_solve_backend("hpu", monkeypatch)

    on_cpu_calls = []
    orig_on_cpu = linalg._on_cpu

    def spy_on_cpu(op, args, kwargs):
        on_cpu_calls.append(op)
        return orig_on_cpu(op, args, kwargs)

    monkeypatch.setattr(linalg, "_on_cpu", spy_on_cpu)

    torch.manual_seed(0)
    A = torch.eye(4) + 0.1 * torch.randn(3, 4, 4)
    B = torch.randn(3, 4, 2)

    # Normal (non-transformed) float32 inference_mode call goes to SolveEx.compute, NOT _on_cpu.
    _dispatch._STATE["calls"].clear()
    with torch.inference_mode():
        res = backend_fn(A, B)
    assert _dispatch._STATE["calls"]["solve"] == 1
    assert on_cpu_calls == []
    torch.testing.assert_close(res[0], torch.linalg.solve(A, B), rtol=1e-4, atol=1e-5)

    # Functorch transforms (vmap, grad, jacrev) route to _on_cpu even inside backend_kernel.
    want_v = torch.func.vmap(torch.linalg.solve)(A, B)
    got_v = torch.func.vmap(lambda a, b: backend_fn(a, b)[0])(A, B)
    torch.testing.assert_close(got_v, want_v, rtol=1e-4, atol=1e-5)

    want_g = torch.func.grad(lambda a, b: torch.linalg.solve(a, b).sum())(A[0], B[0])
    got_g = torch.func.grad(lambda a, b: backend_fn(a, b)[0].sum())(A[0], B[0])
    torch.testing.assert_close(got_g, want_g, rtol=1e-4, atol=1e-5)

    want_j = torch.func.jacrev(torch.linalg.solve)(A[0], B[0])
    got_j = torch.func.jacrev(lambda a, b: backend_fn(a, b)[0])(A[0], B[0])
    torch.testing.assert_close(got_j, want_j, rtol=1e-4, atol=1e-5)
    assert len(on_cpu_calls) >= 3

    # Float64 on "hpu" declines in takes() with one warning and routes to _on_cpu.
    on_cpu_calls.clear()
    A64, B64 = A.double(), B.double()
    with pytest.warns(UserWarning, match="hpu-ops#16"):
        res64 = backend_fn(A64, B64)
    assert len(on_cpu_calls) == 1
    torch.testing.assert_close(res64[0], torch.linalg.solve(A64, B64))


def test_solve_grad_jacrev_vmap_and_compiled_requires_grad(monkeypatch):
    monkeypatch.setattr(linalg, "_GJ", tpc.gj_solve_reference)
    torch.manual_seed(0)
    A = torch.eye(4) + 0.1 * torch.randn(3, 4, 4)
    B = torch.randn(3, 4, 2)
    want_g = torch.func.grad(lambda a, b: torch.linalg.solve(a, b).sum())(A[0], B[0])
    want_j = torch.func.jacrev(torch.linalg.solve)(A[0], B[0])
    want_v = torch.func.vmap(torch.linalg.solve)(A, B)

    Ag = A[0].clone().requires_grad_()
    Bg = B[0].clone().requires_grad_()
    torch.linalg.solve(Ag, Bg).sum().backward()

    with cpu_as_hpu(linalg.install_solve) as calls:
        torch.testing.assert_close(
            torch.func.grad(lambda a, b: torch.linalg.solve(a, b).sum())(A[0], B[0]), want_g, rtol=1e-4, atol=1e-5
        )
        torch.testing.assert_close(torch.func.jacrev(torch.linalg.solve)(A[0], B[0]), want_j, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(torch.func.vmap(torch.linalg.solve)(A, B), want_v, rtol=1e-4, atol=1e-5)

        torch._dynamo.reset()
        Ah = A[0].clone().requires_grad_()
        Bh = B[0].clone().requires_grad_()
        torch.compile(torch.linalg.solve, backend="aot_eager", fullgraph=True)(Ah, Bh).sum().backward()
        assert calls["solve"] >= 1
        torch.testing.assert_close(Ah.grad, Ag.grad, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(Bh.grad, Bg.grad, rtol=1e-4, atol=1e-5)
