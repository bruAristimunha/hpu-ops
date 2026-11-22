"""Adversarial CPU fake-HPU pytest suite AND standalone Gaudi2 HPU runner/probe for M3B-A (Feature #8a:
Gram-free on-card SVD past ``n=32``).

- Run CPU fake-HPU tests via pytest:
    ``/path/to/.venv/bin/python -m pytest tests/test_svd_gt32_adv.py -q``
- Run live Gaudi2 suite (eager or lazy):
    ``PT_HPU_LAZY_MODE=0|1 python3 tests/test_svd_gt32_adv.py``
"""
from __future__ import annotations

import contextlib
import json
import math
import os
import sys
import warnings

try:
    import pytest
except ImportError:
    class _PytestStub:
        fixture = staticmethod(lambda fn: fn)

        class mark:
            @staticmethod
            def parametrize(*_a, **_kw):
                return lambda fn: fn

    pytest = _PytestStub()
import torch

import hpu_ops
from hpu_ops import jacobi, linalg, tpc

L = torch.linalg


@contextlib.contextmanager
def _cpu_as_hpu(install):
    undo = install("cpu")
    hpu_ops._STATE["calls"].clear()
    try:
        yield hpu_ops._STATE["calls"]
    finally:
        undo()


@pytest.fixture
def svd_gt32_hook(monkeypatch):
    """Traceable/float64-capable ``linalg._SVD`` hook for CPU fake-HPU autograd & compile tests."""
    calls = []

    def hook(A, compute_uv):
        calls.append((tuple(A.shape), compute_uv))
        if max(A.shape[-2:]) <= tpc.SMALL_MAXN and A.dtype == torch.float32 and not linalg._is_fake(A):
            return tpc.svd_reference(A.detach(), compute_uv=compute_uv)
        with torch._C._AutoDispatchBelowAutograd():
            return torch.ops.aten._linalg_svd.default(A.detach(), False, compute_uv)

    monkeypatch.setattr(linalg, "_SVD", hook)
    return calls


def _rel_err(got, ref):
    got, ref = got.detach().cpu().double(), ref.detach().cpu().double()
    if got.shape != ref.shape:
        return math.inf
    if got.numel() == 0:
        return 0.0
    scale = float(ref.abs().max())
    e = float((got - ref).abs().max()) / (scale if scale > 0 else 1.0)
    return math.inf if math.isnan(e) else e


def _ill_cond(shape, lo, seed=0, batch=()):
    g = torch.Generator().manual_seed(seed)
    m, n = shape
    k = min(m, n)
    q1 = L.qr(torch.randn(*batch, m, m, generator=g, dtype=torch.float64))[0]
    q2 = L.qr(torch.randn(*batch, n, n, generator=g, dtype=torch.float64))[0]
    s = torch.logspace(0, lo, k, dtype=torch.float64)
    return ((q1[..., :k] * s) @ q2[..., :k, :]).float()


def _singular_distinct_sv(n, seed=0):
    g = torch.Generator().manual_seed(seed + n)
    q1 = L.qr(torch.randn(n, n, generator=g, dtype=torch.float64))[0]
    q2 = L.qr(torch.randn(n, n, generator=g, dtype=torch.float64))[0]
    s = torch.linspace(0.8, 1.2, n, dtype=torch.float64)
    s[-1] = 0.0
    return ((q1 * s) @ q2).float()


def _smin_rel_err(S_got, S_ref):
    smin_ref = S_ref[..., -1].double()
    smin_got = S_got[..., -1].detach().cpu().double()
    return float(((smin_got - smin_ref).abs() / smin_ref.abs()).max())


def _check_svd_factors(U, S, Vh, A, full_matrices, tol=1e-4):
    m, n = A.shape[-2], A.shape[-1]
    k = min(m, n)
    batch = A.shape[:-2]
    assert U.shape == (*batch, m, m if full_matrices else k)
    assert S.shape == (*batch, k)
    assert Vh.shape == (*batch, n if full_matrices else k, n)
    A64 = A.detach().cpu().double()
    S_ref = L.svdvals(A64)
    S64 = S.detach().cpu().double()
    U64 = U.detach().cpu().double()
    Vh64 = Vh.detach().cpu().double()
    scale = float(A64.abs().max()) or 1.0
    s_err = float((S64 - S_ref).abs().max()) / scale
    rec = float(((U64[..., :k] * S64.unsqueeze(-2)) @ Vh64[..., :k, :] - A64).abs().max()) / scale
    eye_u = torch.eye(U.shape[-1], dtype=torch.float64).expand(*batch, U.shape[-1], U.shape[-1])
    eye_v = torch.eye(Vh.shape[-2], dtype=torch.float64).expand(*batch, Vh.shape[-2], Vh.shape[-2])
    orth_ut_u = float((U64.mT @ U64 - eye_u).abs().max())
    orth_vh_vht = float((Vh64 @ Vh64.mT - eye_v).abs().max())
    errs = [s_err, rec, orth_ut_u, orth_vh_vht]
    if full_matrices:
        eye_m = torch.eye(m, dtype=torch.float64).expand(*batch, m, m)
        eye_n = torch.eye(n, dtype=torch.float64).expand(*batch, n, n)
        errs.append(float((U64 @ U64.mT - eye_m).abs().max()))
        errs.append(float((Vh64.mT @ Vh64 - eye_n).abs().max()))
    assert max(errs) < tol, f"SVD check failed: {errs}"


# ---- Pytest CPU fake-HPU suite ----


@pytest.mark.parametrize("full_matrices", [False, True])
@pytest.mark.parametrize("shape", [
    (33, 33), (40, 20), (72, 64), (200, 40), (40, 96), (33, 128),
    (63, 63), (64, 64), (65, 65), (96, 96), (127, 127), (128, 128), (256, 256),
    (4, 40, 35), (2, 3, 64, 48),
])
def test_svd_gt32_shapes_and_full_matrices_cpu(shape, full_matrices):
    g = torch.Generator().manual_seed(sum(shape))
    A = torch.randn(*shape, generator=g, dtype=torch.float32)
    with _cpu_as_hpu(linalg.install_svd) as calls:
        U, S, Vh = L.svd(A, full_matrices=full_matrices)
        assert calls["svd"] >= 1
    _check_svd_factors(U, S, Vh, A, full_matrices, tol=1e-4)


def test_svd_gt32_noncontiguous_and_sliced_views_cpu():
    g = torch.Generator().manual_seed(11)
    A_mt = torch.randn(4, 35, 40, generator=g, dtype=torch.float32).mT
    assert not A_mt.is_contiguous()
    big = torch.randn(8, 70, 68, generator=g, dtype=torch.float32)
    A_slice = big[2:5, 3:67, 1:65]
    assert A_slice.storage_offset() > 0 and not A_slice.is_contiguous()
    for A in (A_mt, A_slice):
        with _cpu_as_hpu(linalg.install_svd) as calls:
            U, S, Vh = L.svd(A, full_matrices=True)
            assert calls["svd"] >= 1
        _check_svd_factors(U, S, Vh, A, True, tol=1e-4)


@pytest.mark.parametrize("shape, lo", [
    ((40, 20), -4), ((40, 20), -5),
    ((64, 64), -4), ((64, 64), -5),
    ((128, 128), -4), ((128, 128), -5), ((128, 128), -6),
    ((256, 256), -4), ((256, 256), -5),
])
def test_svd_gt32_ill_conditioned_cpu(shape, lo):
    A = _ill_cond(shape, lo, seed=abs(lo) + shape[0])
    S_ref = L.svdvals(A.double())
    S_cpu32 = L.svdvals(A)
    bar = max(1e-6, 10 * _rel_err(S_cpu32, S_ref))
    with _cpu_as_hpu(linalg.install_svd) as calls:
        U, S, Vh = L.svd(A, full_matrices=False)
        assert calls["svd"] >= 1
    assert _rel_err(S, S_ref) <= bar
    smin_c32 = _smin_rel_err(S_cpu32, S_ref)
    smin_got = _smin_rel_err(S, S_ref)
    smin_bar = 5e-3 if lo >= -5 else max(5e-3, 10 * smin_c32)
    assert smin_got < smin_bar, f"sigma_min rel err {smin_got:.3g} >= {smin_bar:.3g} (cpu32={smin_c32:.3g})"
    _check_svd_factors(U, S, Vh, A, False, tol=1e-4)


def test_svd_gt32_extreme_scales_zero_rank_and_nan_inf_cpu():
    g = torch.Generator().manual_seed(23)
    for name, A in (
        ("tiny_1e-25", torch.randn(2, 40, 40, generator=g) * 1e-25),
        ("huge_1e25", torch.randn(2, 40, 40, generator=g) * 1e25),
        ("zeros_64x64", torch.zeros(2, 64, 64)),
        ("zeros_40x20", torch.zeros(2, 40, 20)),
        ("rank1_64x64", torch.randn(2, 64, 1, generator=g) @ torch.randn(2, 1, 64, generator=g)),
        ("rank10_128x128", torch.randn(1, 128, 10, generator=g) @ torch.randn(1, 10, 128, generator=g)),
    ):
        with _cpu_as_hpu(linalg.install_svd) as calls:
            U, S, Vh = L.svd(A, full_matrices=True)
            assert calls["svd"] >= 1, name
        _check_svd_factors(U, S, Vh, A, True, tol=1e-4)

    # NaN/Inf parity: finite/non-finite pattern matches without crash
    A_nan = torch.randn(40, 40, generator=g)
    A_nan[3, 7] = float("nan")
    A_inf = torch.randn(40, 40, generator=g)
    A_inf[0, 0] = float("inf")
    for mat in (A_nan, A_inf):
        with _cpu_as_hpu(linalg.install_svd):
            S = L.svdvals(mat)
        assert S.shape == (40,) and not torch.isfinite(S).all()


@pytest.mark.parametrize("shape, full_matrices", [
    ((40, 40), False),
    ((72, 72), False),
    ((160, 40), False),
    ((40, 20), True),
])
def test_svd_gt32_batched_mixed_nan_inf_isolation_cpu(shape, full_matrices):
    g = torch.Generator().manual_seed(29 + sum(shape))
    m, n = shape
    A_fin = torch.randn(m, n, generator=g, dtype=torch.float32)
    A_nan = torch.randn(m, n, generator=g, dtype=torch.float32)
    A_nan[3, 7] = float("nan")
    A_inf = torch.randn(m, n, generator=g, dtype=torch.float32)
    A_inf[0, 0] = float("inf")
    A = torch.stack([A_fin, A_nan, A_inf])
    S_ref = L.svdvals(A_fin.double())

    with _cpu_as_hpu(linalg.install_svd) as calls:
        U, S, Vh = L.svd(A, full_matrices=full_matrices)
        assert calls["svd"] >= 1
    assert torch.isfinite(U[0]).all() and torch.isfinite(S[0]).all() and torch.isfinite(Vh[0]).all()
    assert _rel_err(S[0], S_ref) < 1e-5
    _check_svd_factors(U[0], S[0], Vh[0], A_fin, full_matrices, tol=1e-4)
    for idx in (1, 2):
        assert U[idx].isnan().all() and S[idx].isnan().all() and Vh[idx].isnan().all()

    with _cpu_as_hpu(linalg.install_svd) as calls:
        Sv = L.svdvals(A)
        assert calls["svd"] >= 1
    assert torch.isfinite(Sv[0]).all() and _rel_err(Sv[0], S_ref) < 1e-5
    assert Sv[1].isnan().all() and Sv[2].isnan().all()


def test_svd_gt32_autograd_and_double_backward_cpu(svd_gt32_hook):
    g = torch.Generator().manual_seed(31)
    # 1. gradgradcheck at n = 34 x 33 in float64 (uses svd_gt32_hook for float64 finite diff)
    A64 = torch.randn(2, 34, 33, generator=g, dtype=torch.float64, requires_grad=True)
    with _cpu_as_hpu(linalg.install_svd) as calls:
        assert torch.autograd.gradgradcheck(lambda x: L.svd(x, full_matrices=False)[1], (A64,), atol=1e-4, rtol=1e-3)
        assert torch.autograd.gradgradcheck(
            lambda x: (lambda u, s, vh: (u * s.unsqueeze(-2)) @ vh)(*L.svd(x, full_matrices=False)),
            (A64,), atol=1e-4, rtol=1e-3)
        assert calls["svd"] >= 2

    # 2. First + second order gradients across svd, svdvals, pinv, lstsq, matrix_norm at n = 64
    A32 = (torch.randn(2, 64, 48, dtype=torch.float64, generator=g) + 4 * torch.eye(64, 48, dtype=torch.float64))
    B32 = torch.randn(2, 64, 3, dtype=torch.float64, generator=g)
    w_s = torch.linspace(1.0, 2.0, 48, dtype=torch.float64)
    cases = [
        ("svd_s", L.svd, lambda o: (o[1] * w_s.to(o[1].dtype)).sum(), (A32.float(),)),
        ("svd_uv", lambda x: L.svd(x, full_matrices=False),
         lambda o: ((o[0] * o[1].square().unsqueeze(-2)) @ o[2]).sum(), (A32.float(),)),
        ("svdvals", L.svdvals, lambda o: (o * w_s.to(o.dtype)).sum(), (A32.float(),)),
        ("pinv", L.pinv, lambda o: o.sum(), (A32.float(),)),
        ("lstsq", lambda a, b: L.lstsq(a, b).solution, lambda o: o.sum(), (A32.float(), B32.float())),
        ("norm_2", lambda x: L.matrix_norm(x, ord=2), lambda o: o.sum(), (A32.float(),)),
        ("norm_neg2", lambda x: L.matrix_norm(x, ord=-2), lambda o: o.sum(), (A32.float(),)),
        ("norm_nuc", lambda x: L.matrix_norm(x, ord="nuc"), lambda o: o.sum(), (A32.float(),)),
    ]
    for name, fn, loss_fn, ins in cases:
        inst = linalg.install_pinv if name == "pinv" else linalg.install_lstsq if name == "lstsq" else linalg.install_svd
        ws = [torch.randn(x.shape, generator=g, dtype=torch.float64) for x in ins]
        ref_ins = [x.double().requires_grad_() for x in ins]
        ref_g1 = torch.autograd.grad(loss_fn(fn(*ref_ins)), ref_ins, create_graph=True)
        ref_g2 = torch.autograd.grad(sum((t * w).sum() for t, w in zip(ref_g1, ws)), ref_ins)

        got_ins = [x.clone().requires_grad_() for x in ins]
        with _cpu_as_hpu(inst) as calls:
            g1 = torch.autograd.grad(loss_fn(fn(*got_ins)), got_ins, create_graph=True)
            g2 = torch.autograd.grad(sum((t * w.to(t.dtype)).sum() for t, w in zip(g1, ws)), got_ins)
            assert sum(calls.values()) >= 1, name
        for a, b in zip(g1, ref_g1):
            assert _rel_err(a, b) < 1e-2, f"{name} 1st bwd"
        for a, b in zip(g2, ref_g2):
            assert torch.isfinite(a).all() and _rel_err(a, b) < 5e-2, f"{name} 2nd bwd"

    # 3. det create_graph (singular n = 33, matching codex8) & double backward (n = 33)
    singular_eye = torch.eye(33, dtype=torch.float64)
    singular_eye[-1, -1] = 0.0
    S_eye = torch.stack([singular_eye, 2 * torch.eye(33, dtype=torch.float64)]).requires_grad_()
    ref_eye_g1 = torch.autograd.grad(L.det(S_eye).sum(), S_eye, create_graph=True)[0]
    S_eye_got = S_eye.detach().clone().requires_grad_()
    with _cpu_as_hpu(linalg.install_inv_det), _cpu_as_hpu(linalg.install_svd) as svd_calls:
        got_eye_g1 = torch.autograd.grad(L.det(S_eye_got).sum(), S_eye_got, create_graph=True)[0]
        assert svd_calls["svd"] >= 1
    torch.testing.assert_close(got_eye_g1, ref_eye_g1, rtol=1e-5, atol=1e-5)

    S_det = torch.stack([torch.diag(torch.linspace(1.0, 2.0, 33, dtype=torch.float64)),
                         torch.diag(torch.linspace(1.5, 3.0, 33, dtype=torch.float64))]).requires_grad_()
    ref_det_g1 = torch.autograd.grad(L.det(S_det).sum(), S_det, create_graph=True)[0]
    ref_det_g2 = torch.autograd.grad(ref_det_g1.square().sum(), S_det)[0]
    S_det_got = S_det.detach().clone().requires_grad_()
    with _cpu_as_hpu(linalg.install_inv_det), _cpu_as_hpu(linalg.install_svd):
        g1 = torch.autograd.grad(L.det(S_det_got).sum(), S_det_got, create_graph=True)[0]
        g2 = torch.autograd.grad(g1.square().sum(), S_det_got)[0]
    torch.testing.assert_close(g1, ref_det_g1, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(g2, ref_det_g2, rtol=1e-5, atol=1e-5)


def test_svd_gt32_double_backward_losses_and_det_singular_cpu():
    # Exercises block_svd_gram_free + Svd.backward (no _SVD hook) across 40x28 & 65x48.
    def _ws(s):
        return torch.linspace(1.0, 2.0, s.shape[-1], dtype=s.dtype, device=s.device)

    svd_losses = (
        ("u_s_vh_recon", lambda o: o[1].sum() + ((o[0] * o[1].unsqueeze(-2)) @ o[2]).sum(), "sq"),
        ("u_outer_u", lambda o: (o[0] @ o[0].mT).sum(), "sq"),
        ("s_sq_recon", lambda o: ((o[0] * o[1].square().unsqueeze(-2)) @ o[2]).sum(), "sq"),
        ("s_only", lambda o: (o[1] * _ws(o[1])).sum(), "hvp"),
    )
    for m, n in ((40, 28), (65, 48)):
        g = torch.Generator().manual_seed(123 + m)
        A = (torch.randn(2, m, n, generator=g, dtype=torch.float64)
             + 4 * torch.eye(m, n, dtype=torch.float64)).float()
        w = torch.randn(2, m, n, generator=torch.Generator().manual_seed(99 + m), dtype=torch.float64)
        for lname, loss_fn, mode in svd_losses:
            xr = A.double().requires_grad_()
            g1r = torch.autograd.grad(loss_fn(L.svd(xr, full_matrices=False)), xr, create_graph=True)[0]
            g2r = torch.autograd.grad((g1r * g1r).sum() if mode == "sq" else (g1r * w).sum(), xr)[0]

            xc = A.float().requires_grad_()
            g1c = torch.autograd.grad(loss_fn(L.svd(xc, full_matrices=False)), xc, create_graph=True)[0]
            g2c = torch.autograd.grad((g1c * g1c).sum() if mode == "sq" else (g1c * w.float()).sum(), xc)[0]

            xh = A.float().requires_grad_()
            with _cpu_as_hpu(linalg.install_svd) as calls:
                g1h = torch.autograd.grad(loss_fn(L.svd(xh, full_matrices=False)), xh, create_graph=True)[0]
                g2h = torch.autograd.grad((g1h * g1h).sum() if mode == "sq" else (g1h * w.float()).sum(), xh)[0]
                assert calls["svd"] >= 1
            assert torch.isfinite(g1h).all() and torch.isfinite(g2h).all(), f"{m}x{n} {lname}"
            assert _rel_err(g1h, g1r) < max(1e-3, 10 * _rel_err(g1c, g1r)), f"{m}x{n} {lname} g1"
            assert _rel_err(g2h, g2r) < max(1e-2, 10 * _rel_err(g2c, g2r)), f"{m}x{n} {lname} g2"

    # Distinct-SV singular det double backward at n = 33 and n = 65.
    for n in (33, 65):
        A_sing = _singular_distinct_sv(n)
        xr = A_sing.double().requires_grad_()
        g1r = torch.autograd.grad(L.det(xr).sum(), xr, create_graph=True)[0]
        g2r = torch.autograd.grad((g1r * g1r).sum(), xr)[0]

        xc = A_sing.float().requires_grad_()
        g1c = torch.autograd.grad(L.det(xc).sum(), xc, create_graph=True)[0]
        g2c = torch.autograd.grad((g1c * g1c).sum(), xc)[0]

        xh = A_sing.float().requires_grad_()
        with _cpu_as_hpu(linalg.install_inv_det), _cpu_as_hpu(linalg.install_svd) as svd_calls:
            g1h = torch.autograd.grad(L.det(xh).sum(), xh, create_graph=True)[0]
            g2h = torch.autograd.grad((g1h * g1h).sum(), xh)[0]
            assert svd_calls["svd"] >= 1
        assert torch.isfinite(g1h).all() and torch.isfinite(g2h).all(), f"det singular n={n}"
        assert _rel_err(g1h, g1r) < max(1e-3, 10 * _rel_err(g1c, g1r)), f"det singular n={n} g1"
        assert _rel_err(g2h, g2r) < max(1e-2, 10 * _rel_err(g2c, g2r)), f"det singular n={n} g2"


def test_matrix_rank_all_overloads_and_inference_mode_gt32_cpu(monkeypatch):
    monkeypatch.setattr(linalg, "_CIA_BACKENDS_ON_CPU", True)
    g = torch.Generator().manual_seed(45)
    A = torch.randn(2, 40, 35, generator=g, dtype=torch.float32)
    A_rd = torch.randn(2, 40, 10, generator=g) @ torch.randn(2, 10, 35, generator=g)
    for mat in (A, A_rd):
        for fn in (
            lambda x: L.matrix_rank(x),
            lambda x: L.matrix_rank(x, 1e-3),
            lambda x: L.matrix_rank(x, torch.tensor(1e-3)),
            lambda x: L.matrix_rank(x, atol=1e-3, rtol=1e-3),
            lambda x: L.matrix_rank(x, atol=torch.tensor([1e-3, 1e-2])),
        ):
            want = fn(mat)
            with _cpu_as_hpu(linalg.install_matrix_rank) as calls, torch.inference_mode():
                got = fn(mat)
                assert calls["svd"] >= 1
            assert torch.equal(got, want)

    monkeypatch.setattr(tpc, "ready", lambda: True)
    assert linalg._svd_takes("hpu")(A) and linalg._pinv_takes("hpu")(A)
    with _cpu_as_hpu(linalg.install_svd) as calls, torch.no_grad():
        got_s = L.svd(A, full_matrices=False)[1]
        assert calls["svd"] >= 1
    torch.testing.assert_close(got_s, L.svdvals(A), rtol=1e-4, atol=1e-4)

    for install, fn, key in (
        (linalg.install_svd, L.svdvals, "svd"),
        (linalg.install_svd, lambda x: L.matrix_norm(x, ord=2), "svd"),
        (linalg.install_svd, lambda x: L.matrix_norm(x, ord="nuc"), "svd"),
    ):
        want = fn(A)
        with _cpu_as_hpu(install) as calls, torch.inference_mode():
            got = fn(A)
            assert calls[key] >= 1
        torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-4)


# ---- Live Gaudi2 runner / probe ----


def _run_gaudi_suite():
    hpu_ops.enable()
    import habana_frameworks.torch.core as htcore
    import habana_frameworks.torch.hpu as hthpu
    from habana_frameworks.torch.hpu.metrics import metric_global

    assert tpc.ready(), "hpu_ops.tpc.ready() is False"
    lazy = os.environ.get("PT_HPU_LAZY_MODE", "1") != "0"
    hpu = torch.device("hpu")
    cpu_route = []
    for name in ("_on_cpu", "_out_on_cpu"):
        inner = getattr(linalg, name)

        def counted(*args, _inner=inner, _name=name, **kwargs):
            cpu_route.append(_name)
            return _inner(*args, **kwargs)

        setattr(linalg, name, counted)

    def sync():
        if lazy:
            htcore.mark_step()
        hthpu.synchronize()

    def fb_ops():
        return dict(dict(metric_global("cpu_fallback").stats()).get("FallbackOps", {}) or {})

    results = []

    def check_case(name, fn, make_in, *, strict_host=True, verify=None, grad_loss=None,
                   double_bwd=False, check_g2_accuracy=True):
        row = {"case": name, "lazy": lazy, "strict_host": strict_host}
        problems = []
        try:
            ins = make_in()
            ref_ins = [x.double() for x in ins]
            try:
                ref_out = fn(*ref_ins)
            except torch.linalg.LinAlgError:
                if verify is None:
                    raise
                ref_out = None
            h_ins = [x.to(hpu).detach() for x in ins]
            sync()
            allow = () if strict_host else ("host sync",)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                f0, r0 = fb_ops(), len(cpu_route)
                with hpu_ops.strict("raise", allow=allow, watch_cpu_ops=True):
                    got = fn(*h_ins)
                    if lazy:
                        htcore.mark_step()
                hthpu.synchronize()
                fb = {k: v - f0.get(k, 0) for k, v in fb_ops().items() if v - f0.get(k, 0)}
                rc = len(cpu_route) - r0
            warns = [str(w.message) for w in caught if "runs on the CPU" in str(w.message)]
            row.update(fallback=fb, cpu_route=rc, cpu_warn=warns)
            if fb:
                problems.append(f"FallbackOps={fb}")
            if rc:
                problems.append(f"cpu_route={rc}")
            if warns:
                problems.append(f"cpu_warn={warns}")
            if verify is not None:
                problems.extend(verify(got, ref_out, ins))
            else:
                got_t = [got] if isinstance(got, torch.Tensor) else list(got)
                ref_t = [ref_out] if isinstance(ref_out, torch.Tensor) else list(ref_out)
                cpu_t = [fn(*ins)] if isinstance(ref_out, torch.Tensor) else list(fn(*ins))
                for g, r, c in zip(got_t, ref_t, cpu_t):
                    if g.dtype == torch.int64:
                        if not torch.equal(g.cpu(), r.cpu()):
                            problems.append("int64 mismatch")
                    else:
                        e = _rel_err(g, r)
                        bar = max(1e-5, 10 * _rel_err(c, r))
                        if not e <= bar:
                            problems.append(f"err {e:.3g} > {bar:.3g}")
            if grad_loss is not None:
                req_ins = [x.to(hpu).detach().requires_grad_() for x in ins]
                ws = []
                h_ws = []
                if double_bwd and double_bwd != "sq":
                    wgen = torch.Generator().manual_seed(999)
                    ws = [torch.randn(x.shape, generator=wgen, dtype=torch.float64) for x in ins]
                    h_ws = [w.to(device=hpu, dtype=x.dtype).detach() for w, x in zip(ws, ins)]
                sync()
                with warnings.catch_warnings(record=True) as gcaught:
                    warnings.simplefilter("always")
                    f0, r0 = fb_ops(), len(cpu_route)
                    with hpu_ops.strict("raise", allow=allow, watch_cpu_ops=True):
                        loss = grad_loss(fn(*req_ins))
                        if double_bwd:
                            g1 = torch.autograd.grad(loss, req_ins, create_graph=True)
                            g2_in = (sum(t.square().sum() for t in g1) if double_bwd == "sq"
                                     else sum((t * hw).sum() for t, hw in zip(g1, h_ws)))
                            g2 = torch.autograd.grad(g2_in, req_ins)
                        else:
                            g1 = torch.autograd.grad(loss, req_ins)
                            g2 = ()
                        if lazy:
                            htcore.mark_step()
                    hthpu.synchronize()
                    gfb = {k: v - f0.get(k, 0) for k, v in fb_ops().items() if v - f0.get(k, 0)}
                    grc = len(cpu_route) - r0
                gwarns = [str(w.message) for w in gcaught if "runs on the CPU" in str(w.message)]
                if gfb or grc or gwarns:
                    problems.append(f"grad fb={gfb} route={grc} warn={gwarns}")

                ref_g_ins = [x.double().detach().requires_grad_() for x in ins]
                c32_ins = [x.float().detach().requires_grad_() for x in ins]
                ref_g1 = torch.autograd.grad(grad_loss(fn(*ref_g_ins)), ref_g_ins, create_graph=bool(double_bwd))
                c32_g1 = torch.autograd.grad(grad_loss(fn(*c32_ins)), c32_ins, create_graph=bool(double_bwd))
                g1_rels = []
                for idx, (gh, gc, gr) in enumerate(zip(g1, c32_g1, ref_g1)):
                    if not torch.isfinite(gh.cpu()).all():
                        problems.append(f"non-finite g1[{idx}]")
                    elif torch.isfinite(gr).all() and torch.isfinite(gc).all():
                        e = _rel_err(gh, gr)
                        g1_rels.append(e)
                        bar = max(1e-3, 10 * _rel_err(gc, gr))
                        if not e <= bar:
                            problems.append(f"g1[{idx}] err {e:.3g} > {bar:.3g}")
                if g1_rels:
                    row["g1_rel"] = max(g1_rels)

                if double_bwd:
                    for idx, gh in enumerate(g2):
                        if not torch.isfinite(gh.cpu()).all():
                            problems.append(f"non-finite g2[{idx}]")
                    if check_g2_accuracy:
                        ref_g2 = torch.autograd.grad(
                            sum(t.square().sum() for t in ref_g1) if double_bwd == "sq"
                            else sum((t * w).sum() for t, w in zip(ref_g1, ws)),
                            ref_g_ins,
                        )
                        c32_g2 = torch.autograd.grad(
                            sum(t.square().sum() for t in c32_g1) if double_bwd == "sq"
                            else sum((t * w.to(t.dtype)).sum() for t, w in zip(c32_g1, ws)),
                            c32_ins,
                        )
                        g2_rels = []
                        for idx, (gh, gc, gr) in enumerate(zip(g2, c32_g2, ref_g2)):
                            if not (torch.isfinite(gr).all() and torch.isfinite(gc).all()):
                                problems.append(f"non-finite cpu ref g2[{idx}]")
                            elif torch.isfinite(gh.cpu()).all():
                                e = _rel_err(gh, gr)
                                g2_rels.append(e)
                                bar = max(1e-2, 10 * _rel_err(gc, gr))
                                if not e <= bar:
                                    problems.append(f"g2[{idx}] err {e:.3g} > {bar:.3g}")
                        if g2_rels:
                            row["g2_rel"] = max(g2_rels)
        except Exception as e:  # noqa: BLE001
            problems.append(f"exc {e!r}"[:240])
        row["ok"] = not problems
        row["problems"] = problems
        print("RESULT " + json.dumps(row), flush=True)
        results.append(row["ok"])

    def verify_svd(full, lo=None):
        def _v(got, _ref, ins):
            U, S, Vh = got
            A = ins[0]
            A64 = A.double()
            S_ref = L.svdvals(A64)
            S_cpu = L.svdvals(A)
            bar = max(1e-5, 10 * _rel_err(S_cpu, S_ref))
            probs = []
            se = _rel_err(S, S_ref)
            if not se <= bar:
                probs.append(f"S err {se:.3g} > {bar:.3g}")
            if lo is not None:
                smin_c32 = _smin_rel_err(S_cpu, S_ref)
                smin_got = _smin_rel_err(S, S_ref)
                smin_bar = 5e-3 if lo >= -5 else max(5e-3, 10 * smin_c32)
                if not smin_got < smin_bar:
                    probs.append(f"sigma_min err {smin_got:.3g} >= {smin_bar:.3g} (cpu32={smin_c32:.3g})")
            try:
                _check_svd_factors(U, S, Vh, A, full, tol=1e-4)
            except AssertionError as e:
                probs.append(str(e))
            return probs
        return _v

    gen = torch.Generator().manual_seed(0)
    # 1. Shapes & orientations (full_matrices False & True)
    for shape in (
        (33, 33), (40, 20), (72, 64), (200, 40), (40, 96), (33, 128),
        (63, 63), (64, 64), (65, 65), (96, 96), (127, 127), (128, 128), (256, 256),
        (4, 40, 35), (2, 3, 64, 48),
    ):
        for full in (False, True):
            check_case(f"svd_{'full' if full else 'thin'}_{'x'.join(map(str, shape))}",
                       lambda x, _f=full: L.svd(x, full_matrices=_f),
                       lambda _s=shape: [torch.randn(*_s, generator=gen, dtype=torch.float32)],
                       verify=verify_svd(full))

    # Non-contiguous .mT and sliced offset view
    check_case("svd_noncontig_mT_40x35", lambda x: L.svd(x, full_matrices=True),
               lambda: [torch.randn(4, 35, 40, generator=gen, dtype=torch.float32).mT], verify=verify_svd(True))
    check_case("svd_sliced_offset_64", lambda x: L.svd(x, full_matrices=False),
               lambda: [torch.randn(6, 70, 68, generator=gen, dtype=torch.float32)[2:5, 3:67, 1:65]],
               verify=verify_svd(False))

    # 2. Ill-conditioned, extreme scales, zeros, rank-1, rank-10, NaN/Inf
    for shape, lo in (
        ((40, 20), -4), ((40, 20), -5),
        ((64, 64), -4), ((64, 64), -5),
        ((128, 128), -4), ((128, 128), -5), ((128, 128), -6),
        ((256, 256), -4), ((256, 256), -5),
    ):
        check_case(f"svd_illcond_{shape[0]}x{shape[1]}_1e{lo}",
                   lambda x: L.svd(x, full_matrices=False),
                   lambda _sh=shape, _lo=lo: [_ill_cond(_sh, _lo, seed=abs(_lo) + _sh[0])],
                   verify=verify_svd(False, lo=lo))
    for name, fn_in in (
        ("tiny_1e-25_64", lambda: [torch.randn(2, 64, 64, generator=gen) * 1e-25]),
        ("huge_1e25_64", lambda: [torch.randn(2, 64, 64, generator=gen) * 1e25]),
        ("zeros_64", lambda: [torch.zeros(2, 64, 64)]),
        ("rank1_64", lambda: [torch.randn(2, 64, 1, generator=gen) @ torch.randn(2, 1, 64, generator=gen)]),
        ("rank10_128", lambda: [torch.randn(1, 128, 10, generator=gen) @ torch.randn(1, 10, 128, generator=gen)]),
    ):
        check_case(f"svd_{name}", lambda x: L.svd(x, full_matrices=True), fn_in, verify=verify_svd(True))

    def _nan_verify(got, ref, _ins):
        g = got.cpu()
        if ref is None:
            return [] if not torch.isfinite(g).all() else ["expected non-finite output"]
        return [] if torch.equal(torch.isnan(g), torch.isnan(ref.cpu())) else ["NaN pattern differs"]

    check_case("svdvals_nan_40", L.svdvals,
               lambda: [(lambda m: (m.__setitem__((0, 0), float("nan")), m)[1])(torch.randn(40, 40))],
               verify=_nan_verify)
    check_case("svdvals_inf_40", L.svdvals,
               lambda: [(lambda m: (m.__setitem__((0, 0), float("inf")), m)[1])(torch.randn(40, 40))],
               verify=lambda g, _r, _: [] if not torch.isfinite(g.cpu()).all() else ["expected non-finite output"])

    # 3. Backward + double backward + matrix_rank + inference_mode
    w_s_cpu = {n: torch.linspace(1.0, 2.0, n, dtype=torch.float64) for n in (28, 48)}
    w_s_hpu = {n: w.float().to(hpu).detach() for n, w in w_s_cpu.items()}
    sync()

    def _ws(s):
        n = s.shape[-1]
        if s.device.type == "hpu":
            return w_s_hpu[n]
        return w_s_cpu[n].to(dtype=s.dtype)

    def make_well(b=2, m=64, n=48):
        return [torch.randn(b, m, n, generator=gen) + 4 * torch.eye(m, n)]

    for m, n in ((40, 28), (65, 48)):
        def _make_distinct(_m=m, _n=n):
            lg = torch.Generator().manual_seed(123 + _m)
            return [(torch.randn(2, _m, _n, generator=lg, dtype=torch.float64)
                     + 4 * torch.eye(_m, _n, dtype=torch.float64)).float()]

        for lname, lfn, mode in (
            ("u_s_vh_recon", lambda o: o[1].sum() + ((o[0] * o[1].unsqueeze(-2)) @ o[2]).sum(), "sq"),
            ("u_outer_u", lambda o: (o[0] @ o[0].mT).sum(), "sq"),
            ("s_sq_recon", lambda o: ((o[0] * o[1].square().unsqueeze(-2)) @ o[2]).sum(), "sq"),
            ("s_only", lambda o: (o[1] * _ws(o[1])).sum(), True),
        ):
            check_case(f"svd_bwd2_{lname}_{m}x{n}", lambda x: L.svd(x, full_matrices=False), _make_distinct,
                       verify=verify_svd(False), grad_loss=lfn, double_bwd=mode)

    check_case("svd_bwd_double_64x48", lambda x: L.svd(x, full_matrices=False), make_well,
               verify=verify_svd(False), grad_loss=lambda o: o[1].sum() + ((o[0] * o[1].unsqueeze(-2)) @ o[2]).sum(),
               double_bwd="sq")
    check_case("svdvals_bwd_64", L.svdvals, make_well, grad_loss=lambda o: o.sum(), double_bwd=True)
    check_case("pinv_bwd_64", L.pinv, make_well, grad_loss=lambda o: o.sum(), double_bwd=True)
    check_case("lstsq_bwd_64", lambda a, b: L.lstsq(a, b).solution,
               lambda: [torch.randn(2, 64, 48, generator=gen) + 4 * torch.eye(64, 48), torch.randn(2, 64, 3)],
               strict_host=False, grad_loss=lambda o: o.sum(), double_bwd=True)
    for ord_ in (2, -2, "nuc"):
        check_case(f"matrix_norm_{ord_}_64", lambda x, _o=ord_: L.matrix_norm(x, ord=_o),
                   make_well, grad_loss=lambda o: o.sum(), double_bwd=True)

    check_case("det_double_bwd_singular_eye_n33", L.det,
               lambda: [(lambda s: (s.__setitem__((-1, -1), 0.0), s)[1])(torch.eye(33))],
               strict_host=False, grad_loss=lambda o: o.sum(), double_bwd="sq", check_g2_accuracy=False)
    check_case("det_double_bwd_singular_mixed_n33", L.det,
               lambda: [torch.stack([(lambda s: (s.__setitem__((-1, -1), 0.0), s)[1])(torch.eye(33)),
                                     2 * torch.eye(33)])],
               strict_host=False, grad_loss=lambda o: o.sum(), double_bwd="sq", check_g2_accuracy=False)
    for n in (33, 65):
        check_case(f"det_double_bwd_singular_distinct_n{n}", L.det,
                   lambda _n=n: [_singular_distinct_sv(_n)],
                   strict_host=False, grad_loss=lambda o: o.sum(), double_bwd="sq", check_g2_accuracy=True)

    for idx, fn in enumerate((
        lambda x: L.matrix_rank(x),
        lambda x: L.matrix_rank(x, 1e-3),
        lambda x: L.matrix_rank(x, torch.tensor(1e-3, device=x.device)),
        lambda x: L.matrix_rank(x, atol=1e-3, rtol=1e-3),
    )):
        check_case(f"matrix_rank_ov{idx}_64", fn, make_well)

    with torch.inference_mode():
        for name, fn in (
            ("inf_svd_64", lambda x: L.svd(x, full_matrices=False)[1]),
            ("inf_svdvals_128", L.svdvals),
            ("inf_pinv_64", L.pinv),
            ("inf_norm_nuc_64", lambda x: L.matrix_norm(x, ord="nuc")),
            ("inf_matrix_rank_64", L.matrix_rank),
        ):
            check_case(name, fn, lambda: [torch.randn(2, 64, 48, generator=gen) + 4 * torch.eye(64, 48)]
                       if "128" not in name else [torch.randn(2, 128, 128, generator=gen)])

    print(f"SUMMARY {len(results)} cases, {results.count(False)} failed", flush=True)
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(_run_gaudi_suite())
