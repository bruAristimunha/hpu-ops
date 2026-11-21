"""On a Gaudi card: round-2 adversarial probe of team F's TPC solve / triangular-solve path.
    PT_HPU_LAZY_MODE=0|1 python3 tests/probe_adversarial_f.py [--only REGEX] [--habana]
Per case: the HPU call, torch on the CPU in float64 (reference, on the float32-rounded inputs) and in float32 (the
error bar and the finiteness pattern). A row fails when its error exceeds max(1e-6, 10x CPU float32's), its NaN/Inf
pattern differs from CPU float32's, it raises where the CPU does not, a gradient the CPU has is missing on the HPU (or
the other way round), the bridge cpu_fallback metric moved, a warning says "runs on the CPU", or the call took
hpu-ops' own CPU route (``hpu_ops.linalg._on_cpu`` / ``_out_on_cpu``, which neither of the former sees).
``--habana`` reruns every case after ``undo()`` (Habana's own path, origin/main-like baseline) in the same process. One RESULT {json} line per row, then SUMMARY. Bridge 1.21.4."""
import argparse
import json
import math
import os
import re
import sys
import warnings

import hpu_ops

hpu_ops.enable()

import habana_frameworks.torch.core as htcore  # noqa: E402
import habana_frameworks.torch.hpu as hthpu  # noqa: E402
import torch  # noqa: E402
from habana_frameworks.torch.hpu.metrics import metric_global  # noqa: E402

LAZY = os.environ.get("PT_HPU_LAZY_MODE", "1") != "0"
HPU = torch.device("hpu")
L = torch.linalg
G = torch.Generator().manual_seed(11)
PHASE = "kernel"
ON_CPU = []  # one entry per call of hpu-ops' own CPU route (counted by wrapping hpu_ops.linalg._on_cpu)


def _count_cpu_route():
    from hpu_ops import linalg as ops_linalg

    for name in ("_on_cpu", "_out_on_cpu"):
        inner = getattr(ops_linalg, name, None)
        if inner is None:
            continue

        def counted(*args, _inner=inner, _name=name, **kwargs):
            ON_CPU.append(_name)
            return _inner(*args, **kwargs)

        setattr(ops_linalg, name, counted)


_count_cpu_route()


def sync():
    if LAZY:
        htcore.mark_step()
    hthpu.synchronize()


def fb_ops():
    return dict(dict(metric_global("cpu_fallback").stats()).get("FallbackOps", {}) or {})


def leaves(x):
    if isinstance(x, torch.Tensor):
        return [x]
    if isinstance(x, (tuple, list)):
        return [t for y in x for t in leaves(y)]
    return [getattr(x, k) for k in ("sign", "logabsdet", "solution", "Q", "R") if hasattr(x, k)]


def rel_err(got, ref):
    got, ref = got.detach().cpu().double(), ref.detach().cpu().double()
    if got.shape != ref.shape:
        return math.inf
    fin = torch.isfinite(ref)
    if not fin.any():
        return 0.0
    d = (got - ref)[fin].abs().max()
    e = float(d) / max(float(ref[fin].abs().max()), 1e-300)
    return math.inf if math.isnan(e) else e


def pattern(t):
    t = t.detach().cpu()
    return (int(torch.isnan(t).sum()), int(torch.isinf(t).sum()))


def tri(*shape, upper=False, scale=1.0, diag=None):
    n = shape[-1]
    off = torch.randn(*shape, generator=G) / n
    off = off.triu(1) if upper else off.tril(-1)
    d = (1 + torch.rand(*shape[:-1], generator=G)) if diag is None else diag * torch.ones(*shape[:-1])
    return (off + torch.diag_embed(d)) * scale


def rn(*shape, scale=1.0):
    return torch.randn(*shape, generator=G) * scale


def spd(*shape):
    n = shape[-1]
    X = torch.randn(*shape[:-1], 2 * n, generator=G)
    return X @ X.mT / (2 * n) + 0.1 * torch.eye(n)


def st(upper=False, left=True, unit=False):
    return lambda a, b: L.solve_triangular(a, b, upper=upper, left=left, unitriangular=unit)


def chol(upper=False):
    return lambda b, c: torch.cholesky_solve(b, c, upper=upper)


def ill_lower(b, n, cond):
    """Lower triangle with a geometric diagonal 1 .. 1/cond and O(1) off-diagonal entries (row-scaled)."""
    d = torch.logspace(0, -math.log10(cond), n)
    off = torch.randn(b, n, n, generator=G).tril(-1) * d[:, None] * 0.5
    return off + torch.diag_embed(d.expand(b, n))


def double_bwd(fn):
    """grad of <dA, v> w.r.t. (A, B), dA = d<fn(A,B), g>/dA with create_graph: second-order through the Function."""
    def f(a, b):
        y = fn(a, b)
        g = torch.ones_like(y)
        ga, gb = torch.autograd.grad((y * g).sum(), (a, b), create_graph=True)
        return (ga * ga).sum() + (gb * gb).sum()
    return f


def chain(fn, k):
    def f(a, b):
        x = b
        for _ in range(k):
            x = fn(a, x)
        return x
    return f


def cases():
    big = tri(2, 8, 8, diag=2e38)
    yield "tri_diag_2e38_lower", st(), (big, rn(2, 8, 3, scale=1e10)), {}
    yield "tri_diag_2e38_upper", st(upper=True), (tri(2, 8, 8, upper=True, diag=2e38), rn(2, 8, 3, scale=1e10)), {}
    yield "tri_diag_subnormal", st(), (tri(2, 8, 8, diag=1e-39) * 1.0, rn(2, 8, 3, scale=1e-30)), {}
    yield "tri_B_subnormal", st(), (tri(2, 8, 8), rn(2, 8, 3, scale=1e-39)), {}
    yield "solve_diag_2e38", L.solve, (torch.diag_embed(torch.full((2, 8), 2e38)) + rn(2, 8, 8) * 1e30,
                                       rn(2, 8, 3, scale=1e10)), {}
    yield "inv_diag_2e37", L.inv, (torch.diag_embed(torch.full((2, 8), 2e37)),), {}
    yield "det_diag_subnormal_result", L.det, (torch.diag_embed(torch.full((2, 4), 1e-10)),), {}
    yield "solve_scale_1e-38", L.solve, (spd(4, 22, 22) * 1e-38, rn(4, 22, 3) * 1e-38), {}
    nanA = tri(2, 16, 16)
    nanA[0, 5, 2] = float("nan")
    yield "tri_nan_in_A", st(), (nanA, rn(2, 16, 3)), {}
    nanU = tri(2, 16, 16)
    nanU[:, 2, 9] = float("nan")
    yield "tri_nan_opposite_triangle", st(), (nanU, rn(2, 16, 3)), {}
    infB = rn(2, 16, 3)
    infB[1, 4, 0] = float("inf")
    yield "tri_inf_in_B", st(), (tri(2, 16, 16), infB), {}
    zd = tri(2, 16, 16)
    zd[0, 7, 7] = 0.0
    yield "tri_zero_diag", st(), (zd, rn(2, 16, 3)), {}
    nanS = spd(2, 16, 16)
    nanS[1, 3, 3] = float("nan")
    yield "solve_nan_in_A", L.solve, (nanS, rn(2, 16, 3)), {}
    yield "inv_nan_in_A", L.inv, (nanS,), {}
    yield "det_nan_in_A", L.det, (nanS,), {}
    infS = spd(2, 16, 16)
    infS[0, 0, 0] = float("inf")
    yield "solve_inf_in_A", L.solve, (infS, rn(2, 16, 3)), {}
    yield "tri_m65_lower", st(), (tri(3, 40, 40), rn(3, 40, 65)), {}
    yield "tri_m130_upper", st(upper=True), (tri(3, 40, 40, upper=True), rn(3, 40, 130)), {}
    yield "tri_left_false_m65", st(left=False), (tri(3, 40, 40), rn(3, 65, 40)), {}
    yield "solve_m129", L.solve, (spd(3, 40, 40), rn(3, 40, 129)), {}
    yield "chol_m65", chol(), (rn(2, 64, 65), L.cholesky(spd(2, 64, 64))), {}
    yield "tri_n64", st(), (tri(5, 64, 64), rn(5, 64, 64)), {}
    yield "tri_n65_declined", st(), (tri(5, 65, 65), rn(5, 65, 3)), {"route": "declined"}
    yield "chol_n65_declined", chol(), (rn(2, 65, 3), L.cholesky(spd(2, 65, 65))), {"route": "declined"}
    yield "tri_n2", st(upper=True), (tri(7, 2, 2, upper=True), rn(7, 2, 1)), {}
    yield "solve_n1_b37", L.solve, (spd(37, 1, 1), rn(37, 1, 2)), {}
    yield "inv_n1", L.inv, (spd(1, 1),), {}
    # empty inputs are declined (``_takes`` needs numel > 0) and take hpu-ops' CPU route: a copy of nothing
    yield "tri_batch_empty", st(), (tri(0, 22, 22), rn(0, 22, 3)), {"route": "declined"}
    yield "tri_m0", st(), (tri(2, 22, 22), rn(2, 22, 0)), {"route": "declined"}
    yield "tri_batch_1000_n64", st(), (tri(1000, 64, 64), rn(1000, 64, 5)), {}
    yield "solve_batch_997_n40", L.solve, (spd(997, 40, 40), rn(997, 40, 1)), {}
    yield "tri_expand_A", lambda a, b: L.solve_triangular(a.expand(16, 40, 40), b, upper=False), \
        (tri(40, 40), rn(16, 40, 3)), {}
    yield "tri_expand_B", lambda a, b: L.solve_triangular(a, b.expand(6, 40, 3), upper=True), \
        (tri(6, 40, 40, upper=True), rn(40, 3)), {}
    yield "tri_permuted_batch", lambda a, b: L.solve_triangular(a.permute(1, 0, 2, 3), b, upper=False), \
        (tri(3, 4, 22, 22), rn(4, 3, 22, 2)), {}
    yield "tri_step_slice", lambda a, b: L.solve_triangular(a[::2, 1:23, 1:23], b[::3], upper=False), \
        (tri(8, 24, 24), rn(12, 22, 2)), {}
    for c in (1e4, 1e6, 1e7):
        yield f"tri_ill_cond_{c:.0e}", st(), (ill_lower(8, 40, c), rn(8, 40, 3)), {}
        # the ill-conditioned lower triangle is itself a Cholesky factor (positive diagonal): cond(L L^T) ~ c
        yield f"chol_ill_cond_{c:.0e}", chol(), (rn(4, 40, 3), ill_lower(4, 40, math.sqrt(c)).abs()), {}
    yield "tri_grad", st(), (tri(4, 40, 40), rn(4, 40, 3)), {"grad": True}
    yield "tri_grad_left_false_unit", st(upper=True, left=False, unit=True), \
        (tri(4, 40, 40, upper=True), rn(4, 5, 40)), {"grad": True}
    yield "chol_grad_upper", chol(upper=True), (rn(4, 40, 3), L.cholesky(spd(4, 40, 40), upper=True)), {"grad": True}
    yield "tri_double_bwd", double_bwd(st()), (tri(4, 22, 22), rn(4, 22, 3)), {"grad": True, "scalar": True}
    yield "chol_double_bwd", double_bwd(chol()), (rn(4, 22, 3), L.cholesky(spd(4, 22, 22))), \
        {"grad": True, "scalar": True}
    yield "solve_double_bwd", double_bwd(L.solve), (spd(4, 40, 40), rn(4, 40, 3)), {"grad": True, "scalar": True}
    yield "cholesky_bwd_n40", lambda a: L.cholesky(a), (spd(4, 40, 40),), {"grad": True, "allow_fb": "cholesky"}
    yield "qr_bwd_n40", lambda a: L.qr(a), (rn(4, 40, 40),), {"grad": True, "allow_fb": "qr"}
    yield "tri_inference_mode", st(), (tri(4, 40, 40), rn(4, 40, 3)), {"inference": True}
    yield "chol_inference_mode", chol(), (rn(4, 40, 3), L.cholesky(spd(4, 40, 40))), {"inference": True}
    yield "tri_out_variant", lambda a, b: L.solve_triangular(a, b, upper=False, out=torch.empty_like(b)), \
        (tri(4, 22, 22), rn(4, 22, 3)), {}
    yield "triangular_solve_legacy", lambda a, b: torch.triangular_solve(b, a, upper=False).solution, \
        (tri(4, 22, 22), rn(4, 22, 3)), {}
    yield "tri_chain10_lazy", chain(st(), 10), (tri(4, 22, 22, diag=1.0), rn(4, 22, 3)), {}
    yield "chol_chain10", chain(lambda c, b: torch.cholesky_solve(b, c), 10), \
        (L.cholesky(spd(4, 22, 22) + 2 * torch.eye(22)), rn(4, 22, 3)), {}
    yield "tri_complex64", st(), (tri(2, 8, 8).to(torch.complex64), rn(2, 8, 3).to(torch.complex64)), {"route": "any"}


def run(name, fn, ins, opt, compiled=False):
    row = {"op": name, "phase": PHASE, "lazy": LAZY, "compiled": compiled, "bridge": "1.21.4"}
    problems = []
    grad = opt.get("grad", False)
    try:
        cplx = ins[0].is_complex()
        ref_in = [x.detach().to(torch.complex128 if cplx else torch.float64).requires_grad_(grad) for x in ins]
        cpu_in = [x.detach().clone().requires_grad_(grad) for x in ins]

        def go(f, xs):
            out = leaves(f(*xs))
            gs = []
            if grad:
                tot = sum((o.abs() if o.is_complex() else o).sum() * (1 + i) for i, o in enumerate(out)
                          if o.requires_grad)
                gs = list(torch.autograd.grad(tot, xs, allow_unused=True))
            return out, gs

        cpu_raised = None
        try:
            ref, rg = go(fn, ref_in)
            c32, cg = go(fn, cpu_in)
        except Exception as e:  # noqa: BLE001
            cpu_raised = repr(e)[:160]
        h_in = [x.detach().to(HPU).requires_grad_(grad) for x in ins]
        f = torch.compile(fn, backend="hpu_backend") if compiled else fn
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            sync()
            f0 = fb_ops()
            routed0 = len(ON_CPU)
            try:
                if opt.get("inference"):
                    with torch.inference_mode():
                        out, hg = go(f, h_in)
                else:
                    out, hg = go(f, h_in)
                sync()
                out = [o.detach().cpu() for o in out]
                hg = [None if g is None else g.detach().cpu() for g in hg]
                hpu_raised = None
            except Exception as e:  # noqa: BLE001
                hpu_raised = repr(e)[:200]
            sync()
            f1 = fb_ops()
            on_cpu = len(ON_CPU) - routed0
        fb = {k: f1[k] - f0.get(k, 0) for k in f1 if f1[k] - f0.get(k, 0)}
        allow = opt.get("allow_fb")
        fb_bad = {k: v for k, v in fb.items() if not (allow and allow in k)}
        cpu_w = [str(w.message)[:140] for w in caught if "runs on the CPU" in str(w.message)]
        row.update(fallback=fb, cpu_warnings=cpu_w, cpu_raised=cpu_raised, hpu_raised=hpu_raised, on_cpu=on_cpu)
        if cpu_raised and hpu_raised:
            pass
        elif hpu_raised:
            problems.append(f"hpu raised {hpu_raised}")
        elif cpu_raised:
            problems.append(f"cpu raised {cpu_raised} but hpu returned")
        else:
            errs, bars = [], []
            if len(hg) != len(rg):
                problems.append(f"{len(hg)} HPU gradients, {len(rg)} CPU")
            for k, (h, r) in enumerate(zip(hg, rg)):
                if (h is None) != (r is None):
                    problems.append(f"gradient {k} {'missing' if h is None else 'extra'} on the HPU")
            for h, r, c in zip(out + hg, ref + rg, c32 + cg):
                if r is None or h is None:
                    continue
                errs.append(rel_err(h, r))
                bars.append(rel_err(c, r))
                if pattern(h) != pattern(c):
                    problems.append(f"nonfinite pattern hpu {pattern(h)} cpu32 {pattern(c)}")
                if h.shape != c.shape:
                    problems.append(f"shape {tuple(h.shape)} vs {tuple(c.shape)}")
            err, bar = max(errs, default=0.0), max(bars, default=0.0)
            limit = max(1e-6, 10 * bar)
            row.update(err=err, err_cpu32=bar)
            if not err <= limit:
                problems.append(f"err {err:.3g} > {limit:.3g}")
        if opt.get("route") != "any":
            problems += [f"fell back {fb_bad}"] * bool(fb_bad) + cpu_w
        if opt.get("route") not in ("any", "declined"):
            problems += [f"took hpu-ops' CPU route {on_cpu}x"] * bool(on_cpu)
    except Exception as e:  # noqa: BLE001
        problems.append(f"probe exception {e!r}"[:300])
    print("RESULT " + json.dumps({**row, "status": "fail" if problems else "ok", "problems": problems},
                                 default=str), flush=True)
    return not problems


def compile_recompile():
    """torch.compile with default dynamic=None: the 2nd/3rd distinct batch size marks dims dynamic."""
    rows = []
    f = torch.compile(st(), backend="hpu_backend")
    for b in (4, 8, 12):
        A, B = tri(b, 40, 40), rn(b, 40, 3)
        ref = L.solve_triangular(A.double(), B.double(), upper=False)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            sync()
            f0 = fb_ops()
            try:
                out = f(A.to(HPU), B.to(HPU))
                sync()
                err = rel_err(out, ref)
                exc = None
            except Exception as e:  # noqa: BLE001
                err, exc = math.inf, repr(e)[:200]
            f1 = fb_ops()
        fb = {k: f1[k] - f0.get(k, 0) for k in f1 if f1[k] - f0.get(k, 0)}
        cpu_w = [str(w.message)[:140] for w in caught if "runs on the CPU" in str(w.message)]
        problems = ([f"err {err:.3g}"] if not err <= 1e-5 else []) + ([f"fell back {fb}"] if fb else []) + cpu_w
        if exc:
            problems.append(exc)
        print("RESULT " + json.dumps({"op": f"compile_dyn_batch{b}", "phase": PHASE, "lazy": LAZY, "compiled": True,
                                      "err": err, "fallback": fb, "status": "fail" if problems else "ok",
                                      "problems": problems}), flush=True)
        rows.append(not problems)
    return rows


def main():
    global PHASE
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--habana", action="store_true")
    ap.add_argument("--compile", action="store_true")
    args = ap.parse_args()
    print("PLUGINS", hpu_ops.__file__, "tpc_ready", hpu_ops.tpc.ready(), "lazy", LAZY, flush=True)
    selected = [c for c in cases() if not args.only or re.search(args.only, c[0])]
    results = []
    for name, fn, ins, opt in selected:
        results.append(run(name, fn, ins, opt))
    if args.compile and not LAZY:
        for name, fn, ins, opt in selected:
            if not opt.get("inference") and "chain" not in name:
                results.append(run(name, fn, ins, opt, compiled=True))
        results += compile_recompile()
    if args.habana:
        hpu_ops._STATE["undo"]()
        PHASE = "habana"
        for name, fn, ins, opt in selected:
            results.append(run(name, fn, ins, {**opt, "route": "any"}))
    print(f"SUMMARY {len(results)} rows, {results.count(False)} failed", flush=True)
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
