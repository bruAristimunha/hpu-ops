"""Gaudi probe for team D review round 1 (one process, mode from PT_HPU_LAZY_MODE): atol=0 semantics of matrix_rank
and pinv, declined CompositeImplicitAutograd calls under inference mode (no hpu-ops CPU route), the lazy-mode LU
default, invalid LU pivots, and same-process A/B timings (3 alternating rounds of 20 synced calls) with fallback
counts. Run: ``python tests/gaudi_review_r1.py``."""
import json
import os
import time
import warnings

import torch

import hpu_ops

os.environ.pop("HPU_OPS_LU_SLOW", None)
hpu_ops.enable()
import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.hpu as hthpu
from habana_frameworks.torch.hpu.metrics import metric_global

from hpu_ops import _dispatch, linalg, tpc

LAZY = os.environ.get("PT_HPU_LAZY_MODE", "1") != "0"


def say(**kv):
    print("RESULT", json.dumps(kv), flush=True)


def fallbacks():
    return dict(dict(metric_global("cpu_fallback").stats()).get("FallbackOps", {}) or {})


def delta(a, b):
    return {k: v - a.get(k, 0) for k, v in b.items() if v - a.get(k, 0)}


def sync():
    if LAZY:
        htcore.mark_step()
    hthpu.synchronize()


ON_CPU = []
_original_on_cpu = linalg._on_cpu


def _spy(op, args, kwargs):
    ON_CPU.append(str(op))
    return _original_on_cpu(op, args, kwargs)


linalg._on_cpu = _spy


def reinstall():
    hpu_ops._STATE["undo"]()
    hpu_ops._STATE["undo"] = hpu_ops._install("hpu")


def probe_routing(A):
    say(probe="mode", lazy=LAZY, tpc_ready=tpc.ready(), bridge="1.21.4")
    Ah = A.to("hpu")
    S = A @ A.mT
    cases = {
        "matrix_norm fro": lambda x: torch.linalg.matrix_norm(x, "fro"),
        "matrix_norm 1": lambda x: torch.linalg.matrix_norm(x, 1),
        "matrix_norm inf": lambda x: torch.linalg.matrix_norm(x, float("inf")),
        "matrix_norm -1": lambda x: torch.linalg.matrix_norm(x, -1),
        "matrix_norm nuc": lambda x: torch.linalg.matrix_norm(x, "nuc"),
        "matrix_norm 2": lambda x: torch.linalg.matrix_norm(x, 2),
        "svdvals": torch.linalg.svdvals,
    }
    for name, fn in cases.items():
        ON_CPU.clear()
        f0 = fallbacks()
        c0 = _dispatch._STATE["calls"].get("svd", 0)
        with torch.inference_mode():
            r = fn(Ah)
            sync()
        err = float((r.cpu() - fn(A)).abs().max() / fn(A).abs().max())
        say(probe=f"inference_mode {name}", hpu_ops_on_cpu=ON_CPU[:], bridge_fb=delta(f0, fallbacks()), rel_err=err,
            svd_calls=_dispatch._STATE["calls"].get("svd", 0) - c0)
    for name, fn in {"hermitian": lambda x: torch.linalg.matrix_rank(x, hermitian=True),
                     "hermitian tol": lambda x: torch.linalg.matrix_rank(x, 1e-3, hermitian=True)}.items():
        ON_CPU.clear()
        f0 = fallbacks()
        with torch.inference_mode():
            r = fn(S.to("hpu"))
            sync()
        say(probe=f"inference_mode matrix_rank {name}", hpu_ops_on_cpu=ON_CPU[:], bridge_fb=delta(f0, fallbacks()),
            equal=bool(torch.equal(r.cpu(), fn(S))))


def probe_hermitian_rank(A):
    """Why matrix_rank(hermitian=True) of A A^T can differ from the CPU: the eigenvalues near the tolerance."""
    S = A @ A.mT
    Sh = S.to("hpu")
    want = torch.linalg.matrix_rank(S, hermitian=True)
    with torch.inference_mode():
        got_inf = torch.linalg.matrix_rank(Sh, hermitian=True).cpu()
    got_grad = torch.linalg.matrix_rank(Sh, hermitian=True).cpu()
    w = torch.linalg.eigvalsh(S.double()).abs()
    tol = w.amax(-1) * 22 * torch.finfo(torch.float32).eps
    w_hpu = torch.linalg.eigvalsh(Sh).cpu().double().abs()
    bad = (got_inf != want).nonzero().flatten().tolist()
    say(probe="hermitian rank diag", cpu=want.tolist(), hpu_inference=got_inf.tolist(), hpu_grad=got_grad.tolist(),
        rank_f64=(w > tol[:, None]).sum(-1).tolist(),
        differing=[{"i": i, "min_eig_f64": float(w[i].min()), "tol": float(tol[i]),
                    "min_eig_hpu": float(w_hpu[i].min()),
                    "min_eig_cpu32": float(torch.linalg.eigvalsh(S[i]).abs().min())} for i in bad[:4]])


def tiny_sv(batch=4, n=5):
    g = torch.Generator().manual_seed(1)
    Q1 = torch.linalg.qr(torch.randn(batch, n, n, dtype=torch.float64, generator=g))[0]
    Q2 = torch.linalg.qr(torch.randn(batch, n, n, dtype=torch.float64, generator=g))[0]
    d = torch.tensor([1.0, 1.0, 1.0, 1e-3, 1e-9], dtype=torch.float64)
    return ((Q1 * d) @ Q2.mT).float()


def probe_atol():
    Ar = tiny_sv()
    Ah = Ar.to("hpu")
    for label, atol_cpu, atol_hpu in (("0.", 0.0, 0.0), ("-1.", -1.0, -1.0),
                                       ("tensor(0.)", torch.tensor(0.0), torch.tensor(0.0, device="hpu")),
                                       ("tensor([0,1e-6,0,0])", torch.tensor([0.0, 1e-6, 0.0, 0.0]),
                                        torch.tensor([0.0, 1e-6, 0.0, 0.0], device="hpu")),
                                       ("1e-6", 1e-6, 1e-6)):
        ON_CPU.clear()
        c0 = dict(_dispatch._STATE["calls"])
        f0 = fallbacks()
        want = torch.linalg.matrix_rank(Ar, atol=atol_cpu)
        got = torch.linalg.matrix_rank(Ah, atol=atol_hpu).cpu()
        pw = torch.linalg.pinv(Ar, atol=atol_cpu)
        pg = torch.linalg.pinv(Ah, atol=atol_hpu).cpu()
        sync()
        calls = {k: v - c0.get(k, 0) for k, v in _dispatch._STATE["calls"].items() if v - c0.get(k, 0)}
        say(probe=f"atol={label}", rank_cpu=want.tolist(), rank_hpu=got.tolist(),
            pinv_max_abs_cpu=float(pw.abs().max()), pinv_max_abs_hpu=float(pg.abs().max()),
            pinv_rel_err=float((pg - pw).abs().max() / pw.abs().max()), calls=calls,
            hpu_ops_on_cpu=ON_CPU[:], bridge_fb=delta(f0, fallbacks()))


def timed(fn, x):
    f0 = fallbacks()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        for _ in range(3):
            fn(x)
        sync()
        outs = []
        t = time.perf_counter()
        for _ in range(20):
            outs.append(fn(x))
            if LAZY:
                sync()
        sync()
        ms = (time.perf_counter() - t) / 20 * 1e3
    return ms, delta(f0, fallbacks()), sum("runs on the CPU" in str(m.message) for m in w)


def ab(name, fn, x, rounds=3):
    ours, stock, ofb, sfb, warn = [], [], {}, {}, 0
    c0 = sum(_dispatch._STATE["calls"].values())
    ON_CPU.clear()
    for _ in range(rounds):
        ms, f, w = timed(fn, x)
        ours.append(ms)
        ofb.update(f)
        warn += w
        hpu_ops._STATE["undo"]()
        ms, f, _ = timed(fn, x)
        stock.append(ms)
        sfb.update(f)
        hpu_ops._STATE["undo"] = hpu_ops._install("hpu")
    say(op=name, lazy=LAZY, ours_ms=[round(v, 3) for v in ours], stock_ms=[round(v, 3) for v in stock],
        speedup_median=round(sorted(stock)[rounds // 2] / sorted(ours)[rounds // 2], 2),
        speedup_range=[round(min(stock) / max(ours), 2), round(max(stock) / min(ours), 2)],
        ours_fb=ofb, ours_cpu_warn=warn, stock_fb=sfb, ours_calls=sum(_dispatch._STATE["calls"].values()) - c0,
        hpu_ops_on_cpu=sorted(set(ON_CPU)))


def probe_lu_default(P):
    c0 = dict(_dispatch._STATE["calls"])
    f0 = fallbacks()
    torch.linalg.lu_factor_ex(P.to("hpu"))
    sync()
    say(probe="lu default", lazy=LAZY,
        knob=os.environ.get("HPU_OPS_LU_SLOW"),
        installed={op: torch._C._dispatch_has_kernel_for_dispatch_key(f"aten::{op}", "AutogradHPU")
                   for op in ("linalg_lu_factor_ex", "linalg_lu", "linalg_lu_solve", "lu_unpack")},
        lu_calls=_dispatch._STATE["calls"].get("lu_factor", 0) - c0.get("lu_factor", 0),
        bridge_fb=delta(f0, fallbacks()))


def probe_bad_pivots(P):
    Ph = P[:2, :5, :5].to("hpu")
    LU, piv = torch.linalg.lu_factor(Ph)
    B = torch.randn(2, 5, 3).to("hpu")
    for bad in (0, 9):
        p = piv.clone()
        p[0, 1] = bad
        for name, fn in (("lu_solve", lambda: torch.linalg.lu_solve(LU, p, B)),
                         ("lu_unpack", lambda: torch.lu_unpack(LU, p)[0])):
            try:
                r = fn().cpu()
                say(probe=f"bad pivot {bad} {name}", raised=None, finite=bool(r.isfinite().all()))
            except Exception as e:  # noqa: BLE001
                say(probe=f"bad pivot {bad} {name}", raised=f"{type(e).__name__}: {str(e)[:120]}")
            sync()


def main():
    g = torch.Generator().manual_seed(0)
    A = torch.randn(32, 22, 22, generator=g)
    P = torch.randn(32, 22, 22, generator=g) + 22 ** 0.5 * torch.eye(22)
    probe_lu_default(P)
    probe_routing(A)
    probe_hermitian_rank(A)
    probe_atol()
    Ah, Ph = A.to("hpu"), P.to("hpu")

    def inference(fn):
        def run(x):
            with torch.inference_mode():
                return fn(x)
        return run

    ab("svd", lambda a: torch.linalg.svd(a, full_matrices=False), Ah)
    ab("pinv atol=tensor(0.)", lambda a: torch.linalg.pinv(a, atol=torch.tensor(0.0, device="hpu")), Ah)
    ab("matrix_rank", torch.linalg.matrix_rank, Ah)
    ab("matrix_rank atol=0.", lambda a: torch.linalg.matrix_rank(a, atol=0.0), Ah)
    ab("inference matrix_norm fro", inference(lambda a: torch.linalg.matrix_norm(a, "fro")), Ah)
    ab("inference matrix_norm 1", inference(lambda a: torch.linalg.matrix_norm(a, 1)), Ah)
    ab("inference matrix_norm nuc", inference(lambda a: torch.linalg.matrix_norm(a, "nuc")), Ah)
    ab("matrix_norm nuc", lambda a: torch.linalg.matrix_norm(a, "nuc"), Ah)
    ab("inference matrix_rank hermitian", inference(lambda a: torch.linalg.matrix_rank(a @ a.mT, hermitian=True)), Ah)
    os.environ["HPU_OPS_LU_SLOW"] = "1"
    reinstall()
    probe_lu_default(P)
    for name, fn in (("lu_factor_ex", lambda a: torch.linalg.lu_factor_ex(a)[:2]),
                     ("linalg.lu", torch.linalg.lu)):
        ab(name, fn, Ph, rounds=5)
    probe_bad_pivots(P)


if __name__ == "__main__":
    main()
