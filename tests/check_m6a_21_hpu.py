"""Gaudi2 verification for M6A (#21) rewrites after hpu_ops.enable(), eager (PT_HPU_LAZY_MODE=0) and lazy
(PT_HPU_LAZY_MODE=1). Checks exact semantics, gradients, dispatch counts, FallbackOps == 0, and
"runs on the CPU" == 0. Exits 1 on any failure.
"""
import sys
import warnings

import habana_frameworks.torch.core as htcore
import torch
from habana_frameworks.torch.hpu.metrics import metric_global

import hpu_ops
from hpu_ops import small

hpu_ops.enable()
LAZY = hpu_ops._lazy()
HPU = torch.device("hpu")
CALLS = hpu_ops._STATE["calls"]
FAILED = []
INF, NAN = float("inf"), float("nan")


def fallbacks():
    return sum((dict(metric_global("cpu_fallback").stats()).get("FallbackOps", {}) or {}).values())


def run_checked(name, fn, expected_calls=None):
    CALLS.clear()
    fb0 = fallbacks()
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        ok, detail = fn()
    htcore.mark_step()
    fb = fallbacks() - fb0
    cw = sum("runs on the CPU" in str(w.message) for w in rec)
    taken_ok = True
    if expected_calls is not None:
        for k, v in expected_calls.items():
            if CALLS[k] != v:
                taken_ok = False
                detail += f" [calls[{k!r}]={CALLS[k]} != {v}]"
    passed = bool(ok) and taken_ok and fb == 0 and cw == 0
    print(f"{name:52s} {'ok' if passed else 'FAIL'} fb={fb} cw={cw} {detail}", flush=True)
    if not passed:
        FAILED.append(name)


def same(a, b, rtol=1e-4, atol=1e-4, exact=False):
    if a.dtype != b.dtype or a.shape != b.shape:
        return False, f"meta mismatch {a.dtype}/{a.shape} vs {b.dtype}/{b.shape}"
    if exact:
        ok = torch.equal(a, b)
        return ok, "" if ok else "not equal"
    ok = torch.allclose(a.double(), b.double(), rtol=rtol, atol=atol, equal_nan=True)
    if ok:
        return True, ""
    diff = (a.double() - b.double()).abs().max().item()
    return False, f"max_diff={diff:.3e}"


def check_bincount():
    torch.manual_seed(0)
    # 1. Small high-minlength cases per mode
    for n, ml, taken in [
        (4096, 64, 0),
        (4096, 128, 1 if LAZY else 0),
        (4096, 256, 1),
        (40960, 128, 1 if LAZY else 0),
        (40960, 512, 1),
        (65536, 64, 1),
        (131072, 1000, 1),
    ]:
        for dt in (torch.uint8, torch.int8, torch.int32, torch.int64):
            hi = 120 if dt == torch.int8 else min(ml or 64, 255 if dt == torch.uint8 else ml)
            x = torch.randint(0, hi, (n,), dtype=dt)
            ref = torch.bincount(x, minlength=ml)

            def _case(x=x, ml=ml, ref=ref):
                out = torch.bincount(x.to(HPU), minlength=ml)
                htcore.mark_step()
                return same(out.cpu(), ref, exact=True)

            run_checked(f"bincount n={n} ml={ml} {dt}", _case, {"bincount": taken})

    # Negative check raises ATen error
    def _neg():
        x = torch.randint(0, 10, (4096,), dtype=torch.int32)
        x[5] = -1
        try:
            torch.bincount(x.to(HPU), minlength=256)
            htcore.mark_step()
            return False, "did not raise"
        except RuntimeError as e:
            return "non-negative" in str(e), str(e)[:40]

    run_checked("bincount 4096 ml=256 negative raises", _neg)

    # Compile uint8 (eager mode only: Habana compile backend does not run in lazy mode)
    if not LAZY:
        for n, ml in [(0, 256), (4096, 256), (40960, 512), (409600, 1000)]:
            x = torch.randint(0, 256, (n,), dtype=torch.uint8) if n else torch.empty(0, dtype=torch.uint8)
            ref = torch.bincount(x, minlength=ml)
            torch._dynamo.reset()

            def _comp(x=x, ml=ml, ref=ref):
                fn = torch.compile(lambda t: torch.bincount(t, minlength=ml), backend="hpu_backend", fullgraph=True)
                out = fn(x.to(HPU))
                return same(out.cpu(), ref, exact=True)

            run_checked(f"bincount compile uint8 n={n} ml={ml}", _comp)


def check_logsumexp():
    want_taken = 1 if LAZY else 0
    # Non-finite + boundary edge cases
    cases = [
        ("all_ninf", [-INF, -INF, -INF], 1.0),
        ("mix_ninf", [-INF, 2.0, -INF], 1.0),
        ("pinf_mix", [INF, 100.0, 0.0], 1.0),
        ("pinf_g0", [INF, 100.0, 0.0], 0.0),
        ("pinf_ninf", [INF, -INF], 1.0),
        ("pinf_pinf", [INF, INF], 1.0),
        ("nan_mix", [NAN, 1.0], 1.0),
        ("nan_g0", [NAN, 1.0], 0.0),
        ("large_3e38", [3e38, 3e38, 0.0], 1.0),
    ]
    for cname, vals, gval in cases:
        xc = torch.tensor(vals, dtype=torch.float32, requires_grad=True)
        yc = torch.logsumexp(xc, 0)
        (gc,) = torch.autograd.grad(yc, xc, torch.tensor(gval))

        def _edge(vals=vals, gval=gval, yc=yc, gc=gc):
            xh = torch.tensor(vals, dtype=torch.float32, device=HPU, requires_grad=True)
            yh = torch.logsumexp(xh, 0)
            (gh,) = torch.autograd.grad(yh, xh, torch.tensor(gval, device=HPU))
            htcore.mark_step()
            ok1, d1 = same(yh.cpu(), yc)
            ok2, d2 = same(gh.cpu(), gc)
            return ok1 and ok2, f"{d1} {d2}".strip()

        run_checked(f"logsumexp edge {cname}", _edge, {"logsumexp": want_taken})

    # Repro cases: all_dims >= 2D & 1D backward+sum, plus tuple dim, keepdim, inference_mode, double bwd
    for shape, dim, keepdim in [
        ((64, 22, 22), -1, False),
        ((64, 22, 22), (0, 2), True),
        ((4, 3, 2), (0, 1, 2), False),
        ((22,), 0, False),
    ]:
        torch.manual_seed(0)
        xc = torch.randn(*shape, dtype=torch.float32, requires_grad=True)
        yc = torch.logsumexp(xc, dim, keepdim=keepdim)
        (g1c,) = torch.autograd.grad(yc.sum(), xc, create_graph=True)
        (g2c,) = torch.autograd.grad((g1c ** 2).sum(), xc)

        def _shape(xc=xc, dim=dim, keepdim=keepdim, yc=yc, g1c=g1c, g2c=g2c):
            xh = xc.detach().to(HPU).requires_grad_()
            yh = torch.logsumexp(xh, dim, keepdim=keepdim)
            (g1h,) = torch.autograd.grad(yh.sum(), xh, create_graph=True)
            (g2h,) = torch.autograd.grad((g1h ** 2).sum(), xh)
            with torch.inference_mode():
                yim = torch.logsumexp(xh.detach(), dim, keepdim=keepdim)
            htcore.mark_step()
            for a, b in [(yh.cpu(), yc.detach()), (yim.cpu(), yc.detach()), (g1h.cpu(), g1c.detach()), (g2h.cpu(), g2c)]:
                ok, d = same(a, b, rtol=1e-3, atol=1e-3)
                if not ok:
                    return False, d
            return True, ""

        run_checked(f"logsumexp {shape} dim={dim} kd={keepdim}", _shape, {"logsumexp": 2 * want_taken})

    # Dtypes + empty + above max numel
    for dt in (torch.float16, torch.bfloat16, torch.int32, torch.bool):
        xc = torch.tensor([[1, 0, 2], [0, 1, 0]]).to(dt)
        yc = torch.logsumexp(xc, -1)

        def _dt(xc=xc, yc=yc):
            yh = torch.logsumexp(xc.to(HPU), -1)
            htcore.mark_step()
            return same(yh.cpu(), yc, rtol=2e-2, atol=2e-2)

        run_checked(f"logsumexp dtype {dt}", _dt, {"logsumexp": want_taken})

    for shape, dim in [((0,), 0), ((0, 3), 0), ((3, 0), 1)]:
        xc = torch.empty(shape, dtype=torch.float32, requires_grad=True)
        yc = torch.logsumexp(xc, dim)
        (gc,) = torch.autograd.grad(yc.sum(), xc)

        def _empty(shape=shape, dim=dim, yc=yc, gc=gc):
            xh = torch.empty(shape, dtype=torch.float32, device=HPU, requires_grad=True)
            yh = torch.logsumexp(xh, dim)
            (gh,) = torch.autograd.grad(yh.sum(), xh)
            htcore.mark_step()
            ok1, d1 = same(yh.cpu(), yc, exact=True)
            ok2, d2 = same(gh.cpu(), gc, exact=True)
            return ok1 and ok2, f"{d1} {d2}".strip()

        run_checked(f"logsumexp empty {shape} dim={dim}", _empty, {"logsumexp": want_taken})

    def _above_max():
        x = torch.randn(small.LOGSUMEXP_MAX_NUMEL + 1, device=HPU)
        y = torch.logsumexp(x, 0)
        htcore.mark_step()
        return y.shape == (), ""

    run_checked("logsumexp > MAX_NUMEL stays native", _above_max, {"logsumexp": 0})


def check_index_add():
    want_taken = 1 if LAZY else 0
    rows = small.INDEX_ADD_MIN_ROWS
    torch.manual_seed(0)
    for shape_dst, dim, alpha in [((64,), 0, 1.0), ((64, 22), 0, 1.0), ((64, 22), -2, 2.5), ((16, 4, 3), 0, 1.0)]:
        dst = torch.randn(shape_dst)
        idx = torch.randint(0, shape_dst[0], (rows,))
        src = torch.randn((rows,) + shape_dst[1:])
        src[0], src[1], src[2] = INF, -INF, NAN
        ref = dst.index_add(dim, idx, src, alpha=alpha)

        def _ia(dst=dst, idx=idx, src=src, dim=dim, alpha=alpha, ref=ref):
            dh, ih, sh = dst.to(HPU), idx.to(HPU), src.to(HPU)
            with torch.no_grad():
                o1 = dh.index_add(dim, ih, sh, alpha=alpha)
            with torch.inference_mode():
                o2 = dh.index_add(dim, ih, sh, alpha=alpha)
            htcore.mark_step()
            ok1, d1 = same(o1.cpu(), ref, rtol=1e-4, atol=1e-3)
            ok2, d2 = same(o2.cpu(), ref, rtol=1e-4, atol=1e-3)
            return ok1 and ok2, f"{d1} {d2}".strip()

        run_checked(f"index_add {shape_dst} dim={dim} a={alpha}", _ia, {"index_add": 2 * want_taken})

    # grad_enabled, small, and dim != 0 stay native
    def _declined():
        dst = torch.randn(64, 8, device=HPU, requires_grad=True)
        idx = torch.randint(0, 64, (rows,), device=HPU)
        src = torch.randn(rows, 8, device=HPU, requires_grad=True)
        y = dst.index_add(0, idx, src)
        gd, gs = torch.autograd.grad(y.sum(), (dst, src))
        with torch.no_grad():
            dst.detach().index_add(0, idx[:1024], src.detach()[:1024])
            dst2 = torch.randn(8, 64, device=HPU)
            dst2.index_add(1, torch.randint(0, 64, (64,), device=HPU), torch.randn(8, 64, device=HPU))
        htcore.mark_step()
        return gd.shape == dst.shape and gs.shape == src.shape, ""

    run_checked("index_add grad/small/dim!=0 stay native", _declined, {"index_add": 0})


def check_trace_native():
    def _tr():
        x = torch.randn(64, 64, device=HPU)
        out = torch.trace(x)
        htcore.mark_step()
        return out.shape == (), ""

    run_checked("trace stays native", _tr, {"trace": 0})


def main():
    print(f"bridge 1.21.4 torch {torch.__version__} mode {'lazy' if LAZY else 'eager'}", flush=True)
    check_bincount()
    check_logsumexp()
    check_index_add()
    check_trace_native()
    print("ALL OK" if not FAILED else f"FAILED {FAILED}", flush=True)
    if FAILED:
        sys.exit(1)


if __name__ == "__main__":
    main()
