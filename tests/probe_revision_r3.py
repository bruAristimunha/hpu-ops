"""Gaudi probe for the review-round-3 items (REVISION.md): matrix_exp past the old scaling cap, LU pivot ties against
LAPACK's first-max rule, the LU pivot reciprocal near the float32 maximum, and per-call timing of matrix_exp.

Run on the card (hx-d 'python .../tests/probe_revision_r3.py'), eager or lazy. One JSON row per check, then SUMMARY."""
import json
import os
import sys
import time

import torch

import hpu_ops
from hpu_ops import tpc

LAZY = os.environ.get("PT_HPU_LAZY_MODE", "0") == "1"
ROWS = []


def emit(**row):
    row.setdefault("lazy", LAZY)
    ROWS.append(row)
    print("RESULT " + json.dumps(row, default=str), flush=True)


def sync():
    if LAZY:
        import habana_frameworks.torch.core as htcore

        htcore.mark_step()
    torch.hpu.synchronize()


def fallbacks():
    from habana_frameworks.torch.hpu.metrics import metric_global

    for k, v in metric_global("cpu_fallback").stats():
        if k == "FallbackOps":
            return dict(v)
    return {}


def skew(batch, n, norm, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(batch, n, n, generator=g, dtype=torch.float64)
    x = x - x.mT
    return (x * (norm / torch.linalg.matrix_norm(x, 1)[..., None, None])).float()


def check_expm_large():
    """skew matrices of 1-norm 1e2..1e30: exp is orthogonal, so finite with error ~ cond * eps."""
    for norm in (1e2, 3.5e5, 1e7, 1e12, 1e20, 1e30):
        for n in (6, 22):
            a = skew(4, n, norm)
            ref = torch.linalg.matrix_exp(a.double())
            before = fallbacks()
            got = torch.linalg.matrix_exp(a.to("hpu"))
            sync()
            fb = {k: v - before.get(k, 0) for k, v in fallbacks().items() if v != before.get(k, 0)}
            got = got.cpu().double()
            cpu32 = torch.linalg.matrix_exp(a).double()
            err = float(((got - ref).abs().amax() / ref.abs().amax()))
            err32 = float(((cpu32 - ref).abs().amax() / ref.abs().amax()))
            finite = bool(torch.isfinite(got).all())
            # past ~1e7 float32 cannot resolve the angles (norm * eps > 1): CPU float32 is itself NaN from 1e12, so the
            # bar is CPU float32's (finite where it is finite, within 10x of its error); where it is NaN, recorded only
            cpu_ok = err32 == err32
            ok = (not fb) and ((finite and err <= max(10 * err32, 1e-5)) if cpu_ok else True)
            emit(check="expm_large_norm", norm=norm, n=n, finite=finite, err=err, err_cpu32=err32, fallback=fb,
                 status=("ok" if ok else "fail") if cpu_ok else ("info" if ok else "fail"))
    # mixed scaling counts in one batch, repeated with small-norm calls in between (Y buffers reused): the count read
    # back from Y must be this call's, or the power is wrong (error ~1, not ~CPU float32's)
    worst, worst32, bad = 0.0, 0.0, 0
    for rep in range(30):
        g = torch.Generator().manual_seed(100 + rep)
        norms = 10 ** (5.5 + 2.5 * torch.rand(32, generator=g, dtype=torch.float64))
        x = torch.randn(32, 8, 8, generator=g, dtype=torch.float64)
        x = x - x.mT
        a = (x * (norms / torch.linalg.matrix_norm(x, 1))[..., None, None]).float()
        torch.linalg.matrix_exp(skew(32, 8, 1.0).to("hpu"))
        got = torch.linalg.matrix_exp(a.to("hpu"))
        sync()
        ref = torch.linalg.matrix_exp(a.double())
        e = ((got.cpu().double() - ref).abs().amax((-2, -1)))
        e32 = ((torch.linalg.matrix_exp(a).double() - ref).abs().amax((-2, -1)))
        bad += int((~torch.isfinite(e) | (e > torch.clamp(10 * e32, min=1e-5))).sum())
        worst, worst32 = max(worst, float(e.max())), max(worst32, float(e32.max()))
    emit(check="expm_mixed_counts_reused_buffers", matrices=30 * 32, over_bar=bad, worst_err=worst,
         worst_err_cpu32=worst32, status="ok" if bad == 0 else "fail")
    for scale in (1e3, 1e20, 1e38):  # exp of a large negative-definite matrix underflows to 0, also on the CPU
        g = torch.Generator().manual_seed(2)
        a = (-scale * (torch.eye(6) + 0.01 * torch.rand(2, 6, 6, generator=g))).float()
        got = torch.linalg.matrix_exp(a.to("hpu"))
        sync()
        got = got.cpu()
        ref = torch.linalg.matrix_exp(a)
        finite = bool(torch.isfinite(got).all())
        err = float((got - ref).abs().amax())
        emit(check="expm_negative_definite", scale=scale, finite=finite, abs_err=err, cpu32_finite=bool(
            torch.isfinite(ref).all()), status="ok" if finite and err < 1e-6 else "fail")


def tied(n, seed, batch=8):
    """A = L U, L unit lower with +-1 below the diagonal, U small-integer upper with diagonal 1 or 2: at every step all
    candidates of the pivot column tie (|l_ik u_kk| = |u_kk|) and the arithmetic is exact (integer Schur
    complements), so LAPACK's first-max rule gives pivots 1..n and the factors L, U exactly."""
    g = torch.Generator().manual_seed(seed)
    lower = torch.tril(torch.randint(0, 2, (batch, n, n), generator=g).float() * 2 - 1, -1) + torch.eye(n)
    upper = torch.triu(torch.randint(-2, 3, (batch, n, n), generator=g).float(), 1) + torch.diag_embed(
        torch.randint(1, 3, (batch, n), generator=g).float())
    return lower @ upper


def check_lu_ties():
    for n in (22, 32):
        for seed in range(3):
            a = tied(n, seed)
            LU_ref, piv_ref = torch.linalg.lu_factor(a)
            LU, piv = torch.linalg.lu_factor(a.to("hpu"))
            sync()
            piv = piv.cpu()
            same = bool((piv == piv_ref).all())
            first = int((piv != piv_ref).any(-1).nonzero()[0, 0]) if not same else -1
            emit(check="lu_factor_tied_pivots", n=n, seed=seed, pivots_match_lapack=same,
                 first_diff_matrix=first, lu_err=float((LU.cpu() - LU_ref).abs().amax()),
                 status="ok" if same else "fail")


def check_lu_huge_pivot():
    """|pivot| > 2^126: 1/pivot is subnormal, flushed to zero on the card unless the pivot is scaled first."""
    n = 6
    g = torch.Generator().manual_seed(1)
    a = torch.eye(n, dtype=torch.float64) + 0.01 * torch.randn(4, n, n, generator=g, dtype=torch.float64)
    a = (a * 2.0 ** 126.5).float()  # pivots ~2^126.5: 1/pivot ~2^-126.5 is subnormal; no entry overflows
    b = (torch.randn(4, n, 2, generator=g, dtype=torch.float64) * 2.0 ** 126).float()  # x ~ 1, not subnormal
    LU_ref, piv_ref = torch.linalg.lu_factor(a.double())
    LU, piv = torch.linalg.lu_factor(a.to("hpu"))
    x = torch.linalg.lu_solve(LU, piv, b.to("hpu"))
    sync()
    rel = float(((LU.cpu().double() - LU_ref).abs() / LU_ref.abs().clamp_min(1e-30)).amax())
    x_ref = torch.linalg.solve(a.double(), b.double())
    xerr = float(((x.cpu().double() - x_ref).abs().amax() / x_ref.abs().amax()))
    emit(check="lu_huge_pivot", lu_rel_err=rel, solve_rel_err=xerr, finite=bool(torch.isfinite(x.cpu()).all()),
         status="ok" if rel < 1e-5 and xerr < 1e-4 else "fail")


def time_expm(norm, n=22, batch=32, iters=30):
    a = skew(batch, n, norm).to("hpu")
    outs = []
    for _ in range(5):
        outs.append(torch.linalg.matrix_exp(a))
    sync()
    t = time.perf_counter()
    for _ in range(iters):
        outs.append(torch.linalg.matrix_exp(a))
        if LAZY:
            sync()
    sync()
    return (time.perf_counter() - t) / iters * 1e3


def check_expm_time():
    for norm in (1.0, 5.0, 50.0, 1e3):
        emit(check="expm_ms", norm=norm, n=22, batch=32, ms=time_expm(norm), status="info")
    a = skew(32, 22, 50.0).to("hpu").requires_grad_()
    for _ in range(3):
        torch.linalg.matrix_exp(a).sum().backward()
    sync()
    t = time.perf_counter()
    for _ in range(20):
        torch.linalg.matrix_exp(a).sum().backward()
        sync()
    emit(check="expm_fwd_bwd_ms", norm=50.0, n=22, batch=32, ms=(time.perf_counter() - t) / 20 * 1e3, status="info")


def main():
    if "--baseline" not in sys.argv:
        hpu_ops.enable()
    import habana_frameworks.torch  # noqa: F401

    emit(check="setup", tpc_ready=tpc.ready(), baseline="--baseline" in sys.argv, status="info")
    only = [a[7:] for a in sys.argv if a.startswith("--only=")]
    for fn in (check_expm_large, check_lu_ties, check_lu_huge_pivot, check_expm_time):
        if only and fn.__name__ not in only:
            continue
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            emit(check=fn.__name__, status="fail", error=repr(e)[:300])
    bad = [r for r in ROWS if r["status"] == "fail"]
    print(f"SUMMARY {len(ROWS)} rows, {len(bad)} failed", flush=True)


if __name__ == "__main__":
    main()
