"""On a Gaudi card: adversarial round 2 probes of team D's SVD family, matrix_exp, householder_product and LU ops.
    PT_HPU_LAZY_MODE=0|1 python3 tests/probe_adversarial_d.py [--compile] [--only SUBSTR]
Cases the earlier probe (tests/adversarial_team_d.py) did not cover: rank-deficient and zero inputs (orthonormal
singular vectors, full_matrices completion), entries near float32 overflow and in the subnormal range, forward-mode
AD and functorch transforms, double backward against CPU float64, inference mode, lstsq outputs (rank, residuals)
against the same call with hpu_ops undone (Habana's own path, same process), rank-deficient lstsq under
torch.compile(hpu_backend), strided views into the kernels with grad, and a 6-iteration loop of fresh inputs
without a sync between iterations (lazy-mode races). One ``RESULT {json}`` line per probe."""
import argparse
import json
import math
import os
import sys
import warnings

import torch

import hpu_ops
from hpu_ops import _dispatch

LAZY = os.environ.get("PT_HPU_LAZY_MODE", "0") == "1"  # bridge 1.21.4: unset is eager
L = torch.linalg
HPU = "hpu"


def fb_ops():
    from habana_frameworks.torch.hpu.metrics import metric_global

    return dict(dict(metric_global("cpu_fallback").stats()).get("FallbackOps", {}) or {})


def sync():
    import habana_frameworks.torch.core as htcore
    import habana_frameworks.torch.hpu as hthpu

    if LAZY:
        htcore.mark_step()
    hthpu.synchronize()


def gen(seed, *shape, scale=1.0, dtype=torch.float64):
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed), dtype=dtype) * scale


def rel(got, want):
    got, want = got.detach().cpu().double(), want.detach().cpu().double()
    if got.shape != want.shape:
        return math.inf
    fw, fg = torch.isfinite(want), torch.isfinite(got)
    if not torch.equal(fw, fg):
        return math.inf
    if not fw.all():
        same = torch.equal(got[~fw].nan_to_num(), want[~fw].nan_to_num())
        if not same:
            return math.inf
        got, want = got[fw], want[fw]
    if got.numel() == 0:
        return 0.0
    scale = float(want.abs().max())
    d = float((got - want).abs().max())
    return d / scale if scale > 0 else d


def orth_err(Q):
    Q = Q.detach().cpu().double()
    k = Q.shape[-1]
    return float((Q.mT @ Q - torch.eye(k, dtype=Q.dtype)).abs().max())


def h(x):
    return x.to(HPU, torch.float32) if x.is_floating_point() else x.to(HPU)


def record(name, fn):
    row = {"probe": name, "lazy": LAZY}
    calls0 = dict(_dispatch._STATE["calls"])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        f0 = fb_ops()
        try:
            info = fn()
            sync()
            row.update(info)
        except Exception as e:  # noqa: BLE001
            row["status"] = "error"
            row["error"] = f"{type(e).__name__}: {str(e)[:300]}"
        fb = {k: v - f0.get(k, 0) for k, v in fb_ops().items() if v - f0.get(k, 0)}
    row["fallback"] = fb
    row["cpu_warn"] = sorted({str(w.message)[:100] for w in caught if "runs on the CPU" in str(w.message)})
    calls = {k: v - calls0.get(k, 0) for k, v in _dispatch._STATE["calls"].items() if v - calls0.get(k, 0)}
    row["hpu_ops_calls"] = calls
    print("RESULT", json.dumps(row, default=str), flush=True)


def verdict(**checks):
    """``checks``: name -> (value, bar, 'le'|'ge'|'eq'). Status ok when all hold."""
    out, bad = {}, []
    for k, (v, bar, how) in checks.items():
        ok = (v <= bar) if how == "le" else (v >= bar) if how == "ge" else (v == bar)
        out[k] = v
        if not ok:
            bad.append(f"{k}={v} vs {bar} ({how})")
    out["status"] = "ok" if not bad else "fail"
    out["problems"] = bad
    return out


# ---- probes


def p_svd_rank_deficient():
    A = gen(1, 8, 6, 2) @ gen(2, 8, 2, 4)  # rank 2, 6x4
    U, S, Vh = L.svd(h(A), full_matrices=False)
    Uc, Sc, Vhc = L.svd(A, full_matrices=False)
    return verdict(U_orth=(orth_err(U), 1e-4, "le"), V_orth=(orth_err(Vh.mT), 1e-4, "le"),
                   S_rel=(rel(S, Sc), 1e-5, "le"),
                   rec=(rel((U * S.unsqueeze(-2)) @ Vh, A), 1e-5, "le"))


def p_svd_zero():
    A = torch.zeros(3, 5, 4, dtype=torch.float64)
    U, S, Vh = L.svd(h(A), full_matrices=False)
    return verdict(U_orth=(orth_err(U), 1e-4, "le"), V_orth=(orth_err(Vh.mT), 1e-4, "le"),
                   S_max=(float(S.abs().max().cpu()), 0.0, "le"))


def p_svd_full_rank_deficient():
    A = gen(3, 4, 7, 2) @ gen(4, 4, 2, 3)  # 7x3 rank 2, full_matrices -> U 7x7
    U, S, Vh = L.svd(h(A), full_matrices=True)
    return verdict(U_orth=(orth_err(U), 1e-4, "le"), V_orth=(orth_err(Vh.mT), 1e-4, "le"),
                   U_shape=(tuple(U.shape), (4, 7, 7), "eq"),
                   rec=(rel((U[..., :3] * S.unsqueeze(-2)) @ Vh, A), 1e-5, "le"))


def p_svd_wide_full_rank_deficient():
    A = gen(5, 4, 2, 2) @ gen(6, 4, 2, 7)  # 2x7 rank 2... and a rank-1 one
    A[1] = gen(7, 2, 1) @ gen(8, 1, 7)
    U, S, Vh = L.svd(h(A), full_matrices=True)
    return verdict(U_orth=(orth_err(U), 1e-4, "le"), V_orth=(orth_err(Vh.mT), 1e-4, "le"),
                   rec=(rel((U * S.unsqueeze(-2)) @ Vh[..., :2, :], A), 1e-5, "le"))


def _scaled(scale, n=8, b=4):
    return gen(9, b, n, n) * scale


def p_svd_near_overflow():
    A = (_scaled(1.0, n=32) / 4.0).clamp(-1, 1) * 3e38  # entries up to 3e38 (float32 max 3.4e38)
    S = L.svdvals(h(A))
    Sc = L.svdvals(A)
    S32 = L.svdvals(A.float())
    return verdict(S_rel=(rel(S, Sc), max(1e-5, 10 * rel(S32, Sc)), "le"),
                   finite=(bool(torch.isfinite(S.cpu()).all()), bool(torch.isfinite(S32).all()), "eq"))


def p_svd_subnormal():
    A = _scaled(1e-39)  # float32 subnormals (min normal 1.18e-38)
    S = L.svdvals(h(A))
    Sc = L.svdvals(A)
    S32 = L.svdvals(A.float())
    return verdict(S_rel=(rel(S, Sc), max(1e-3, 10 * rel(S32, Sc)), "le"),
                   S_cpu32_max=(float(S32.max()), 0.0, "ge"), S_hpu_max=(float(S.max().cpu()), 0.0, "ge"))


def p_pinv_subnormal_scale():
    A = _scaled(1e-20)  # pinv entries ~1e20, A^T A ~1e-40 in float32
    P = L.pinv(h(A))
    Pc = L.pinv(A)
    P32 = L.pinv(A.float())
    return verdict(rel=(rel(P, Pc), max(1e-5, 10 * rel(P32, Pc)), "le"))


def p_matrix_rank_subnormal():
    A = _scaled(1e-39)
    r = L.matrix_rank(h(A)).cpu()
    rc = L.matrix_rank(A.float())
    return verdict(rank=(r.tolist(), rc.tolist(), "eq"))


def p_matrix_exp_overflow():
    A = torch.stack([torch.eye(4, dtype=torch.float64) * 100, torch.eye(4, dtype=torch.float64) * -200,
                     gen(10, 4, 4) * 30])
    Y = L.matrix_exp(h(A))
    Y32 = L.matrix_exp(A.float())
    return verdict(rel_vs_cpu32=(rel(Y, Y32), 1e-4, "le"))


def p_householder_transposed_grad():
    base = gen(11, 3, 5, 7).tril(-1)  # (3,5,7) -> mT (3,7,5)
    tau = (2.0 / (1.0 + (base.mT * base.mT).sum(-2)))[..., :4]

    def f(Ab, t):
        return L.householder_product(Ab.mT, t)

    def run(dev):
        Ab = (h(base) if dev == HPU else base.clone()).requires_grad_()
        t = (h(tau) if dev == HPU else tau.clone()).requires_grad_()
        Q = f(Ab, t)
        g = torch.autograd.grad((Q * torch.arange(Q.numel(), device=Q.device, dtype=Q.dtype).reshape(Q.shape)).sum(),
                                (Ab, t))
        return Q, g

    Q, (gA, gt) = run(HPU)
    Qc, (gAc, gtc) = run("cpu")
    return verdict(Q=(rel(Q, Qc), 1e-5, "le"), gA=(rel(gA, gAc), 1e-4, "le"), gt=(rel(gt, gtc), 1e-4, "le"))


def p_matrix_norm_nontail_grad():
    X = gen(12, 6, 4, 5)

    def run(dev):
        x = (h(X) if dev == HPU else X.clone()).requires_grad_()
        n2 = L.matrix_norm(x, 2, dim=(0, 2))
        nn = L.matrix_norm(x, "nuc", dim=(2, 0), keepdim=True)
        g = torch.autograd.grad(n2.sum() + nn.sum(), x)[0]
        return n2, nn, g

    a, b, g = run(HPU)
    ac, bc, gc = run("cpu")
    return verdict(n2=(rel(a, ac), 1e-5, "le"), nuc=(rel(b, bc), 1e-5, "le"), grad=(rel(g, gc), 1e-4, "le"))


def _hvp(fn, A, V, dev):
    x = (h(A) if dev == HPU else A.clone()).requires_grad_()
    v = h(V) if dev == HPU else V
    out = fn(x)
    w = torch.arange(1, out.numel() + 1, dtype=out.dtype, device=out.device).reshape(out.shape) / out.numel()
    g = torch.autograd.grad((out * w).sum(), x, create_graph=True)[0]
    return torch.autograd.grad((g * v).sum(), x)[0]


def _double_backward(fn, A):
    V = gen(99, *A.shape)
    got = _hvp(fn, A, V, HPU)
    want = _hvp(fn, A, V, "cpu")
    want32 = _hvp(fn, A.float(), V.float(), "cpu")
    return verdict(hvp_rel=(rel(got, want), max(1e-4, 10 * rel(want32, want)), "le"))


def p_pinv_double_backward():
    return _double_backward(L.pinv, gen(13, 3, 6, 4))


def p_svdvals_double_backward():
    return _double_backward(L.svdvals, gen(14, 3, 6, 4))


def p_matrix_exp_double_backward():
    return _double_backward(L.matrix_exp, gen(15, 3, 6, 6) * 0.5)


def p_lu_factor_double_backward():
    return _double_backward(lambda a: L.lu_factor(a)[0], gen(16, 3, 6, 6) + 2 * torch.eye(6, dtype=torch.float64))


def p_svdvals_jvp():
    A, T = gen(17, 3, 5, 4), gen(18, 3, 5, 4)
    _, t = torch.func.jvp(L.svdvals, (h(A),), (h(T),))
    _, tc = torch.func.jvp(L.svdvals, (A,), (T,))
    return verdict(tangent_rel=(rel(t, tc), 1e-4, "le"))


def p_func_grad_pinv():
    A = gen(19, 3, 5, 4)
    g = torch.func.grad(lambda a: L.pinv(a).sum())(h(A))
    gc = torch.func.grad(lambda a: L.pinv(a).sum())(A)
    return verdict(grad_rel=(rel(g, gc), 1e-4, "le"))


def p_inference_mode():
    A = gen(20, 4, 6, 5)
    S6 = gen(21, 4, 6, 6) * 0.3
    H = gen(22, 4, 6, 5).tril(-1)
    tau = 2.0 / (1.0 + (H * H).sum(-2))
    B = gen(23, 4, 6, 2)
    G = gen(24, 4, 6, 6) + 2 * torch.eye(6, dtype=torch.float64)
    with torch.inference_mode():
        outs = {
            "svd": L.svd(h(A), full_matrices=False)[1], "pinv": L.pinv(h(A)), "lstsq": L.lstsq(h(A), h(B)).solution,
            "rank": L.matrix_rank(h(A)), "expm": L.matrix_exp(h(S6)), "hh": L.householder_product(h(H), h(tau)),
            "lu": L.lu_factor(h(G))[0],
        }
    want = {"svd": L.svd(A, full_matrices=False)[1], "pinv": L.pinv(A), "lstsq": L.lstsq(A, B).solution,
            "rank": L.matrix_rank(A), "expm": L.matrix_exp(S6), "hh": L.householder_product(H, tau),
            "lu": L.lu_factor(G)[0]}
    errs = {k: (rel(outs[k].double(), want[k].double()), 1e-5, "le") for k in outs}
    return verdict(**errs)


def p_zz_lstsq_vs_stock():  # last: re-enabling after the undo is checked by p_zzz_reenable
    rows = {}
    A = gen(25, 4, 7, 3)
    B = gen(26, 4, 7, 2)
    for drv in (None, "gelsd", "gels"):
        kw = {} if drv is None else {"driver": drv}
        ours = L.lstsq(h(A), h(B), **kw)  # against torch on the CPU (Habana's own path: the origin/main run)
        stock = L.lstsq(A, B, **kw)
        sync()
        rows[f"{drv}.residuals_shape"] = (tuple(ours.residuals.shape), tuple(stock.residuals.shape), "eq")
        rows[f"{drv}.rank"] = (ours.rank.cpu().tolist(), stock.rank.cpu().tolist(), "eq")
        rows[f"{drv}.sv_shape"] = (tuple(ours.singular_values.shape), tuple(stock.singular_values.shape), "eq")
        rows[f"{drv}.sol"] = (rel(ours.solution, stock.solution), 1e-4, "le")
        if ours.residuals.numel() and stock.residuals.numel():
            rows[f"{drv}.res"] = (rel(ours.residuals, stock.residuals), 1e-3, "le")
    return verdict(**rows)


def p_lstsq_compile_rank_deficient():
    A = gen(27, 7, 2) @ gen(28, 2, 4)  # rank 2, 7x4
    B = gen(29, 7, 3)
    rows = {}
    for drv in ("gelsd", "gelss"):
        torch._dynamo.reset()
        f = torch.compile(lambda a, b: L.lstsq(a, b, driver=drv), backend="hpu_backend")
        got = f(h(A), h(B))
        sync()
        want = L.lstsq(A, B, driver=drv)
        rows[f"{drv}.rank"] = (got.rank.cpu().tolist(), want.rank.tolist(), "eq")
        rows[f"{drv}.residuals_shape"] = (tuple(got.residuals.shape), tuple(want.residuals.shape), "eq")
    return verdict(**rows)


def p_lazy_loop_no_sync():
    """Six iterations of fresh inputs through every op, no sync between them; then one sync and a check of all."""
    outs, wants = [], []
    for i in range(6):
        A = gen(100 + i, 4, 6, 5)
        S6 = gen(200 + i, 4, 6, 6) * 0.3
        H = gen(300 + i, 4, 6, 5).tril(-1)
        tau = 2.0 / (1.0 + (H * H).sum(-2))
        G = gen(400 + i, 4, 6, 6) + 2 * torch.eye(6, dtype=torch.float64)
        B = gen(500 + i, 4, 6, 3)
        Ah, Gh = h(A), h(G)
        LUh, ph = L.lu_factor(Gh)
        o = [L.svdvals(Ah), L.pinv(Ah), L.matrix_exp(h(S6)), L.householder_product(h(H), h(tau)), LUh,
             L.lu_solve(LUh, ph, h(B)), L.matrix_norm(Ah, "nuc"), L.lstsq(Ah, h(B)).solution]
        LUc, pc = L.lu_factor(G)
        w = [L.svdvals(A), L.pinv(A), L.matrix_exp(S6), L.householder_product(H, tau), LUc, L.lu_solve(LUc, pc, B),
             L.matrix_norm(A, "nuc"), L.lstsq(A, B).solution]
        outs.append(o)
        wants.append(w)
    sync()
    names = ["svdvals", "pinv", "expm", "hh", "lu_factor", "lu_solve", "nuc", "lstsq"]
    worst = {n: max(rel(outs[i][j], wants[i][j]) for i in range(6)) for j, n in enumerate(names)}
    return verdict(**{n: (v, 2e-4, "le") for n, v in worst.items()})


def p_small_edges():
    one = gen(30, 1, 1) * 5
    rows = {
        "pinv_1x1": (rel(L.pinv(h(one)), L.pinv(one)), 1e-6, "le"),
        "rank_zero_1x1": (L.matrix_rank(h(torch.zeros(1, 1))).cpu().tolist(), 0, "eq"),
        "lstsq_1x1": (rel(L.lstsq(h(one), h(one)).solution, torch.ones(1, 1)), 1e-6, "le"),
        "expm_1x1_100": (rel(L.matrix_exp(h(torch.full((1, 1), 100.0))), L.matrix_exp(torch.full((1, 1), 100.0))), 1e-5,
                         "le"),
        "svd_1x7": (rel(L.svdvals(h(gen(31, 2, 1, 7))), L.svdvals(gen(31, 2, 1, 7))), 1e-6, "le"),
        "svd_7x1": (rel(L.svdvals(h(gen(32, 2, 7, 1))), L.svdvals(gen(32, 2, 7, 1))), 1e-6, "le"),
        "hh_1x1": (rel(L.householder_product(h(torch.ones(1, 1)), h(torch.full((1,), 2.0))),
                       torch.full((1, 1), -1.0)), 1e-6, "le"),
        "pinv_32x32_singular": (rel(L.pinv(h(gen(33, 32, 3) @ gen(34, 3, 32))), L.pinv(gen(33, 32, 3) @ gen(34, 3, 32))),
                                1e-3, "le"),
    }
    return verdict(**rows)


def p_float64_on_hpu():
    A = gen(35, 3, 5, 4)
    S = L.svdvals(A.to(HPU))  # float64 on hpu: by design hpu_ops' CPU route, one warning
    return verdict(dtype=(str(S.dtype), "torch.float64", "eq"), rel=(rel(S, L.svdvals(A)), 1e-6, "le"))


def _fad(f, A, T):
    from torch.autograd import forward_ad

    with forward_ad.dual_level():
        return forward_ad.unpack_dual(f(forward_ad.make_dual(A, T))).tangent


def p_forward_ad_svd_family():
    A, T = gen(37, 3, 5, 4), gen(38, 3, 5, 4)
    rows = {}
    for nm, f in (("svdvals", L.svdvals), ("nuc", lambda a: L.matrix_norm(a, "nuc")), ("pinv", L.pinv),
                  ("svd_S", lambda a: L.svd(a, full_matrices=False)[1])):
        try:
            t = _fad(f, h(A), h(T))
            rows[nm] = (math.inf if t is None else rel(t, _fad(f, A, T)), 1e-4, "le")
        except Exception as e:  # noqa: BLE001
            rows[nm] = (f"{type(e).__name__}: {str(e)[:80]}", "a tangent", "eq")
    return verdict(**rows)


def p_func_transforms():
    A = gen(39, 3, 5, 4)
    H = gen(40, 3, 6, 4).tril(-1)
    tau = torch.rand(3, 4, dtype=torch.float64, generator=torch.Generator().manual_seed(41))
    cases = {
        "grad_svdvals": lambda dev: torch.func.grad(lambda a: L.svdvals(a).sum())(dev(A)),
        "grad_householder": lambda dev: torch.func.grad(lambda a: L.householder_product(a, dev(tau)).sum())(dev(H)),
        "jacrev_svdvals": lambda dev: torch.func.jacrev(L.svdvals)(dev(A[0])),
        "jvp_pinv": lambda dev: torch.func.jvp(L.pinv, (dev(A),), (dev(A),))[1],
        "jvp_nuc": lambda dev: torch.func.jvp(lambda a: L.matrix_norm(a, "nuc"), (dev(A),), (dev(A),))[1],
        "vmap_pinv": lambda dev: torch.vmap(L.pinv)(dev(A)),
    }
    rows = {}
    for nm, f in cases.items():
        try:
            rows[nm] = (rel(f(h), f(lambda x: x)), 1e-4, "le")
        except Exception as e:  # noqa: BLE001
            rows[nm] = (f"{type(e).__name__}: {str(e)[:80]}", "a result", "eq")
    return verdict(**rows)


# ---- round 2b: gaps of the round-0 probe (tests/adversarial_team_d.py) and of the probes above


def p_forward_ad_more():
    """forward-mode AD of householder_product and lstsq (their ``takes`` do not read ``_transformed()``)."""
    H = gen(42, 3, 6, 4).tril(-1)
    tau = torch.rand(3, 4, dtype=torch.float64, generator=torch.Generator().manual_seed(43))
    A, T, B = gen(44, 3, 6, 4), gen(45, 3, 6, 4), gen(46, 3, 6, 2)
    rows = {}
    for nm, f, x in (("householder", lambda a, t=None: L.householder_product(a, t), H),
                     ("lstsq", lambda a, b=None: L.lstsq(a, b).solution, A)):
        other = tau if nm == "householder" else B
        try:
            t = _fad(lambda a: f(a, h(other)), h(x), h(T))
            rows[nm] = (math.inf if t is None else rel(t, _fad(lambda a: f(a, other), x, T)), 1e-4, "le")
        except Exception as e:  # noqa: BLE001
            rows[nm] = (f"{type(e).__name__}: {str(e)[:80]}", "a tangent", "eq")
    return verdict(**rows)


def _grad_cmp(fn, *xs, ref=None):
    """``fn`` and the gradient of a weighted sum of its output on HPU float32 vs ``ref`` (default ``fn``) on CPU
    float64."""

    def run(dev, xs_=None):
        leaves = [(h(x) if dev == HPU else x.clone()).requires_grad_() for x in (xs_ if xs_ is not None else xs)]
        out = (fn if dev == HPU or ref is None else ref)(*leaves)
        w = torch.linspace(0.5, 1.5, out.numel(), dtype=out.dtype, device=out.device).reshape(out.shape)
        return out, torch.autograd.grad((out * w).sum(), leaves)

    o, g = run(HPU)
    oc, gc = run("cpu")
    o32, g32 = run("cpu", tuple(x.float() if x.is_floating_point() else x for x in xs))
    err = max(rel(o, oc), *(rel(a, b) for a, b in zip(g, gc)))
    err32 = max(rel(o32, oc), *(rel(a, b) for a, b in zip(g32, gc)))
    return err, max(1e-4, 10 * err32)


def _row(fn, *xs, ref=None):
    """A verdict row: the HPU error against max(1e-4, 10x CPU float32's)."""
    err, bar = _grad_cmp(fn, *xs, ref=ref)
    return err, bar, "le"


def p_compile_grad():
    """torch.compile(backend="hpu_backend") with backward, a second call on fresh data of the same shape."""
    A = gen(47, 4, 7, 5)
    S6 = gen(48, 4, 6, 6) * 0.4
    H = gen(49, 4, 7, 5).tril(-1)
    tau = 2.0 / (1.0 + (H * H).sum(-2))
    B = gen(50, 4, 7, 2)
    G = gen(51, 4, 6, 6) + 2 * torch.eye(6, dtype=torch.float64)
    cases = {
        "svdvals": (lambda a: L.svdvals(a), (A,)),
        "svd_UVh": (lambda a: (lambda U, S, Vh: torch.cat([(U @ U.mT).flatten(-2), (Vh.mT @ Vh).flatten(-2),
                                                           ((U * S.unsqueeze(-2)) @ Vh).flatten(-2)], -1))(
            *L.svd(a, full_matrices=False)), (A,)),
        "pinv": (L.pinv, (A,)),
        "nuc": (lambda a: L.matrix_norm(a, "nuc"), (A,)),
        "lstsq": (lambda a, b: L.lstsq(a, b).solution, (A, B)),
        "expm": (L.matrix_exp, (S6,)),
        "hh": (lambda a, t: L.householder_product(a, t), (H, tau)),
        "lu_solve": (lambda a, b: L.lu_solve(*L.lu_factor(a), b[..., :6, :]), (G, B)),
    }
    rows = {}
    only = os.environ.get("PROBE_CASE")  # one case per process: in lazy mode one case aborts the device
    for nm, (f, xs) in cases.items():
        if only and nm != only:
            continue
        if os.environ.get("PROBE_NO_GRAD"):
            torch._dynamo.reset()
            out = torch.compile(f, backend="hpu_backend")(*(h(x) for x in xs))
            sync()
            rows[nm] = (rel(out, f(*xs)), 1e-4, "le")
            continue
        torch._dynamo.reset()
        cf = torch.compile(f, backend="hpu_backend")
        try:
            e1, b1 = _grad_cmp(cf, *xs, ref=f)
            e2, b2 = _grad_cmp(cf, *(x * 1.1 for x in xs), ref=f)
            rows[nm] = (max(e1, e2), max(b1, b2), "le")
        except Exception as e:  # noqa: BLE001
            rows[nm] = (f"{type(e).__name__}: {str(e)[:120]}", "a result", "eq")
    return verdict(**rows)


def p_views_grad():
    """Offset, step and transposed views of an HPU leaf into every kernel, with grad (lazy: views of graph inputs)."""
    X = gen(52, 6, 8, 8) * 0.4
    G = X + 2 * torch.eye(8, dtype=torch.float64)
    Bx = gen(53, 6, 8, 3)
    views = {"offset": lambda x: x[2:5], "step": lambda x: x[::2], "mT": lambda x: x.mT, "col": lambda x: x[..., 1:7]}
    fns = {
        "svdvals": lambda a: L.svdvals(a), "pinv": L.pinv, "expm": lambda a: L.matrix_exp(a[..., :6, :6]),
        "lu_factor": lambda a: L.lu_factor(a[..., :6, :6] + 2 * torch.eye(6, dtype=a.dtype, device=a.device))[0],
    }
    rows = {}
    for vn, v in views.items():
        for fn_name, f in fns.items():
            src = G if fn_name == "lu_factor" else X
            try:
                rows[f"{fn_name}.{vn}"] = _row(lambda a: f(v(a)), src)
            except Exception as e:  # noqa: BLE001
                rows[f"{fn_name}.{vn}"] = (f"{type(e).__name__}: {str(e)[:100]}", "a result", "eq")
    try:
        rows["lu_solve.offset_B"] = _row(lambda a, b: L.lu_solve(*L.lu_factor(a[1:4]), b[1:4]), G, Bx)
        rows["lstsq.offset"] = _row(lambda a, b: L.lstsq(a[1:4, :, :5], b[1:4]).solution, X, Bx)
        Hb = gen(54, 6, 8, 5).tril(-1)
        tb = 2.0 / (1.0 + (Hb * Hb).sum(-2))
        rows["hh.offset"] = _row(lambda a, t: L.householder_product(a[2:5], t[2:5]), Hb, tb)
    except Exception as e:  # noqa: BLE001
        rows["solve_lstsq_hh"] = (f"{type(e).__name__}: {str(e)[:100]}", "a result", "eq")
    return verdict(**rows)


def p_lu_variants_grad():
    """lu_solve: LU batch broadcast, 2-D B, left=False, adjoint, grads wrt LU and B; lu_unpack flags; singular info."""
    G = gen(55, 3, 5, 5) + 2 * torch.eye(5, dtype=torch.float64)
    LU1, p1 = L.lu_factor(G[:1])
    LU3, p3 = L.lu_factor(G)
    rows = {}
    for nm, (lu, p, b, kw) in {
        "bcast_LU1": (LU1, p1, gen(56, 3, 5, 2), {}), "B_2d": (LU3, p3, gen(57, 5, 2), {}),
        "right": (LU3, p3, gen(58, 3, 2, 5), {"left": False}), "adjoint": (LU3, p3, gen(59, 3, 5, 2), {"adjoint": True}),
        "right_adjoint": (LU3, p3, gen(60, 3, 2, 5), {"left": False, "adjoint": True}),
    }.items():
        def f(lu_, b_, p=p, kw=kw):
            return L.lu_solve(lu_, p.to(lu_.device), b_, **kw)
        rows[nm] = _row(f, lu, b)
    P, Lo, U = torch.lu_unpack(h(LU3), p3.to(HPU), unpack_pivots=False)
    rows["unpack_no_piv_P_shape"] = (tuple(P.shape), (0,), "eq")
    P, Lo, U = torch.lu_unpack(h(LU3), p3.to(HPU), unpack_data=False)
    rows["unpack_no_data_L_shape"] = (tuple(Lo.shape), (0,), "eq")
    S = G.clone()
    S[1, :, 2] = 0
    _, _, info = L.lu_factor_ex(h(S))
    rows["singular_info"] = (info.cpu().tolist(), L.lu_factor_ex(S)[2].tolist(), "eq")
    try:
        L.lu_factor(h(S))
        sync()
        rows["lu_factor_singular_raises"] = ("no error", "RuntimeError", "eq")
    except RuntimeError as e:
        rows["lu_factor_singular_raises"] = ("RuntimeError" if "zero" in str(e) else str(e)[:80], "RuntimeError", "eq")
    return verdict(**rows)


def _finite_pattern(x):
    return torch.isfinite(x.detach().cpu()).all(-1).flatten().tolist() if x.dim() else [bool(torch.isfinite(x))]


def p_nonfinite_inputs():
    """NaN / inf entries: the finite pattern of the output as torch float64 (or torch's error)."""
    A = gen(61, 3, 4, 4)
    A[1, 2, 3] = float("nan")
    A[2, 0, 0] = float("inf")
    rows = {}
    for nm, f in {"svdvals": L.svdvals, "pinv": L.pinv, "expm": L.matrix_exp,
                  "lu_factor": lambda a: L.lu_factor_ex(a)[0], "hh": lambda a: L.householder_product(a, a[..., 0, :])}.items():
        try:
            want = _finite_pattern(f(A))
        except Exception as e:  # noqa: BLE001
            want = type(e).__name__
        try:
            got = _finite_pattern(f(h(A)))
        except Exception as e:  # noqa: BLE001
            got = type(e).__name__
        rows[nm] = (got, want, "eq")
    return verdict(**rows)


def p_ill_conditioned():
    """cond 1e5 inputs: error vs CPU float64 within 10x of CPU float32's."""
    g = torch.Generator().manual_seed(62)
    Q1, _ = torch.linalg.qr(torch.randn(4, 16, 16, dtype=torch.float64, generator=g))
    Q2, _ = torch.linalg.qr(torch.randn(4, 16, 16, dtype=torch.float64, generator=g))
    A = Q1 @ torch.diag_embed(torch.logspace(0, -5, 16, dtype=torch.float64)) @ Q2.mT
    B = gen(63, 4, 16, 3)
    rows = {}
    for nm, f in {"pinv": lambda a, b: L.pinv(a), "svdvals": lambda a, b: L.svdvals(a),
                  "lu_solve": lambda a, b: L.lu_solve(*L.lu_factor(a), b),
                  "lstsq": lambda a, b: L.lstsq(a, b).solution}.items():
        want = f(A, B)
        e32 = rel(f(A.float(), B.float()), want)
        rows[nm] = (rel(f(h(A), h(B)), want), max(1e-6, 10 * e32), "le")
    return verdict(**rows)


def p_sizes():
    """Unbatched, batch 1, two batch dims, n = 32 on both sides, large batches."""
    rows = {}
    for nm, shape in {"unbatched": (5, 3), "b1": (1, 4, 4), "b2x3": (2, 3, 6, 4), "n32": (2, 32, 32),
                      "32x1": (3, 32, 1), "big65536": (65536, 4, 4), "big4096x22": (4096, 22, 22)}.items():
        A = gen(64, *shape)
        try:
            rows[f"svdvals.{nm}"] = (rel(L.svdvals(h(A)), L.svdvals(A)), 1e-5, "le")
            if shape[-1] == shape[-2]:
                Gs = A + 3 * torch.eye(shape[-1], dtype=A.dtype)
                rows[f"lu.{nm}"] = (rel(L.lu_factor(h(Gs))[0], L.lu_factor(Gs)[0]), 1e-4, "le")
                rows[f"expm.{nm}"] = (rel(L.matrix_exp(h(A * 0.2)), L.matrix_exp(A * 0.2)), 1e-5, "le")
        except Exception as e:  # noqa: BLE001
            rows[nm] = (f"{type(e).__name__}: {str(e)[:100]}", "a result", "eq")
    return verdict(**rows)


def p_complex64():
    """complex64 on HPU (the kernels take float32 only: the call is declined)."""
    A = torch.complex(gen(65, 3, 4, 4), gen(66, 3, 4, 4))
    rows = {}
    for nm, f in {"svdvals": L.svdvals, "pinv": L.pinv, "expm": lambda a: L.matrix_exp(a * 0.3),
                  "lu_factor": lambda a: L.lu_factor(a)[0]}.items():
        try:
            rows[nm] = (rel(f(A.to(torch.complex64).to(HPU)).abs(), f(A).abs()), 1e-4, "le")
        except Exception as e:  # noqa: BLE001
            rows[nm] = (f"{type(e).__name__}: {str(e)[:100]}", "a result", "eq")
    return verdict(**rows)


def p_zzzz_reenable_cause():
    """After undo + enable on an initialized device: tpc state (``install_kernel_path`` turns ``_INSTALLED`` off)."""
    tpc = getattr(hpu_ops, "tpc", None)
    if tpc is None:
        return verdict(skipped=("no hpu_ops.tpc", "no hpu_ops.tpc", "eq"))
    before = (tpc.available(), tpc.ready())
    _dispatch._STATE["undo"]()
    hpu_ops.enable()
    after = (tpc.available(), tpc.ready())
    c0 = dict(_dispatch._STATE["calls"])
    X = gen(67, 4, 6, 6)
    out = L.solve(h(X + 3 * torch.eye(6, dtype=X.dtype)), h(X)), L.svdvals(h(X))
    sync()
    calls = {k: v - c0.get(k, 0) for k, v in _dispatch._STATE["calls"].items() if v - c0.get(k, 0)}
    return verdict(ready_before=(before, (True, True), "eq"), ready_after=(after, (True, True), "eq"),
                   calls_after=(sorted(calls), ["solve", "svd"], "eq"))


PROBES = [(k[2:], v) for k, v in sorted(globals().items()) if k.startswith("p_")]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", default="")
    args = parser.parse_args()
    hpu_ops.enable()
    import habana_frameworks.torch.utils.internal as internal
    lu = getattr(getattr(hpu_ops, "linalg", None), "_lu_wanted", None)
    print(f"MODE PT_HPU_LAZY_MODE={os.environ.get('PT_HPU_LAZY_MODE')} is_lazy={internal.is_lazy()} "
          f"hpu_ops={hpu_ops.__file__} lu_installed={lu('hpu', 'lu_factor') if lu else None} "
          f"torch={torch.__version__}", flush=True)
    for name, fn in PROBES:
        if args.only in name:
            record(name, fn)
    print("DONE", flush=True)


if __name__ == "__main__":
    sys.exit(main())
