"""On a Gaudi card: same-process A/B of the hpu_ops TPC routes against the stock stack (hpu_ops undone).
    PT_HPU_LAZY_MODE=0|1 python3 tests/bench_ab.py [--only NAME ...] [--n 22] [--batch 32]
For each op: REPS alternating rounds (ours, then stock) of three warm-up calls and TIMED synced calls with the outputs kept alive (lazy mode skips work whose outputs are
dropped), first with ``hpu_ops``' rewrites installed (ours), then with them removed (the stock stack: Habana's kernel
or its CPU fallback), in the same process. ``enable()`` runs once: a second call after the first HPU op turns the TPC
kernels off (the kernel path is read when the device initialises), so the rewrites are re-registered directly. Each
row records how many times the rewrite ran (``ours_calls``), so a declined call (hpu-ops' own CPU route, which the
bridge metric does not see) shows. Records the bridge ``cpu_fallback`` delta and "runs on the CPU" warnings of both
sides and the relative error ``max|x - ref| / max|ref|`` of ours against torch on the CPU in float64 (the reference
is computed on sign-invariant quantities where the op's output is defined up to signs). One ``RESULT {json}`` line
per op."""
import argparse
import json
import os
import time
import warnings

import torch

import hpu_ops  # enable() must run before the first HPU op (kernel path)
from hpu_ops import _dispatch

LAZY = os.environ.get("PT_HPU_LAZY_MODE", "0") == "1"  # bridge 1.21.4: unset is eager
TIMED = 20
REPS = 3  # timed blocks per side; the minimum block mean is reported, all are recorded
CPU_TEXT = "runs on the CPU"


def fb_ops():
    from habana_frameworks.torch.hpu.metrics import metric_global

    stats = dict(metric_global("cpu_fallback").stats())
    return dict(stats.get("FallbackOps", {}) or {})


def delta(before, after):
    return {k: v - before.get(k, 0) for k, v in after.items() if v - before.get(k, 0)}


def sync():
    import habana_frameworks.torch.core as htcore
    import habana_frameworks.torch.hpu as hthpu

    if LAZY:
        htcore.mark_step()
    hthpu.synchronize()


def leaves(x):
    if isinstance(x, torch.Tensor):
        return [x]
    if isinstance(x, (tuple, list)):
        return [t for y in x for t in leaves(y)]
    return [getattr(x, f) for f in getattr(x, "_fields", ()) if isinstance(getattr(x, f), torch.Tensor)]


def timed(fn, args, warmup=3):
    """``(first output, ms per call, fallback delta, "runs on the CPU" warnings)``: ``warmup`` calls, then TIMED
    synced calls with the outputs kept alive. The fallback delta and warnings cover the timed calls only (the
    warm-up's are also counted: a fallback anywhere fails ours)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        f0 = fb_ops()
        first = fn(*args)
        for _ in range(warmup - 1):
            fn(*args)
        sync()
        outs = []
        t = time.perf_counter()
        for _ in range(TIMED):
            outs.append(fn(*args))
            if LAZY:
                sync()
        sync()
        ms = (time.perf_counter() - t) / TIMED * 1e3
        fb = delta(f0, fb_ops())
    cpu_warn = sum(CPU_TEXT in str(w.message) for w in caught)
    return first, ms, fb, cpu_warn


def merge(a, b):
    return {k: a.get(k, 0) + b.get(k, 0) for k in {*a, *b}}


def rel_err(got, want):
    got = [g.detach().cpu().double() for g in leaves(got)]
    want = [w.detach().double() for w in leaves(want)]
    err = 0.0
    for g, w in zip(got, want, strict=True):
        if g.dtype.is_floating_point:
            scale = w.abs().max().clamp(min=1e-30)
            err = max(err, float((g - w).abs().max() / scale))
        else:
            err = max(err, float((g.long() != w.long()).any()))
    return err


def grad_of(f):
    """``A -> d(sum(f(A) * w))/dA`` with a fixed weight: forward and backward in one call."""

    def g(*args):
        xs = [a.detach().requires_grad_() if a.dtype.is_floating_point else a for a in args]
        out = leaves(f(*xs))
        loss = sum((o * torch.linspace(0.5, 1.5, o.numel(), device=o.device).reshape(o.shape)).sum()
                   for o in out if o.dtype.is_floating_point)
        grads = torch.autograd.grad(loss, [x for x in xs if x.requires_grad])
        return grads

    return g


def svd_inv(x):
    U, S, Vh = torch.linalg.svd(x, full_matrices=False)
    return S, U @ torch.diag_embed(S) @ Vh, U.mT @ U


def cases(n, b):
    g = torch.Generator().manual_seed(0)
    A = torch.randn(b, n, n, generator=g)
    R = torch.randn(b, n + 6, n, generator=g)
    Bm = torch.randn(b, n + 6, 3, generator=g)
    K = torch.randn(b, n, n, generator=g)
    skew = (K - K.mT) * 0.5
    H = torch.randn(b, n, n, generator=g).tril(-1)
    tau = 2.0 / (1.0 + (H * H).sum(-2))  # exact reflectors: Q orthogonal, as orthogonal(householder) builds them
    P = torch.randn(b, n, n, generator=g) + n ** 0.5 * torch.eye(n)
    Bs = torch.randn(b, n, 4, generator=g)
    L = torch.linalg
    return [
        ("svd", svd_inv, (A,)),
        ("svd_bwd", grad_of(lambda x: torch.linalg.svd(x, full_matrices=False)[1]), (A,)),
        ("svdvals", L.svdvals, (A,)),
        ("pinv", L.pinv, (R,)),
        ("lstsq", lambda a, bb: L.lstsq(a, bb).solution, (R, Bm)),
        ("matrix_norm_2", lambda x: L.matrix_norm(x, ord=2), (A,)),
        ("matrix_norm_nuc", lambda x: L.matrix_norm(x, ord="nuc"), (A,)),
        ("matrix_rank", L.matrix_rank, (A,)),
        ("matrix_exp", L.matrix_exp, (skew,)),
        ("matrix_exp_bwd", grad_of(L.matrix_exp), (skew,)),
        ("householder_product", L.householder_product, (H, tau)),
        ("householder_product_bwd", grad_of(L.householder_product), (H, tau)),
        ("lu_factor", lambda x: (lambda lu, piv: (lu, piv))(*L.lu_factor(x)), (P,)),
        ("lu_factor_ex", lambda x: L.lu_factor_ex(x)[:2], (P,)),
        ("lu_solve", lambda x, bb: L.lu_solve(*L.lu_factor(x), bb), (P, Bs)),
        ("lu_solve_adjoint", lambda x, bb: L.lu_solve(*L.lu_factor(x), bb, adjoint=True), (P, Bs)),
        ("lu_solve_right", lambda x, bb: L.lu_solve(*L.lu_factor(x), bb.mT, left=False), (P, Bs)),
        ("lu_solve_bwd", grad_of(lambda x, bb: L.lu_solve(*L.lu_factor(x), bb)), (P, Bs)),
        ("lu_factor_bwd", grad_of(lambda x: L.lu_factor(x)[0]), (P,)),
        ("lu_unpack", lambda x: torch.lu_unpack(*L.lu_factor(x)), (P,)),
        ("lu", lambda x: L.lu(x), (P,)),
        # what #13 is for: one forward and backward of an orthogonal-parametrized layer (spd_learn's BiMap pattern)
        ("orthogonal_matrix_exp_step", orth_step("matrix_exp", n), (torch.randn(b, n, generator=g),)),
        ("orthogonal_householder_step", orth_step("householder", n), (torch.randn(b, n, generator=g),)),
    ]


def orth_step(kind, n):
    """``x -> d(sum(layer(x)))/dW`` of ``orthogonal(Linear(n, n), orthogonal_map=kind)``, one layer per device and
    dtype (same seed), so ours, stock and the CPU float64 reference use the same weight."""
    layers = {}

    def step(x):
        key = (x.device.type, x.dtype)
        if key not in layers:
            torch.manual_seed(0)
            lin = torch.nn.Linear(n, n, bias=False)
            layers[key] = torch.nn.utils.parametrizations.orthogonal(lin, orthogonal_map=kind).to(x.device, x.dtype)
        layer = layers[key]
        weight = layer.parametrizations.weight.original
        return torch.autograd.grad(layer(x).sum(), weight)

    return step


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--n", type=int, default=22)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--set", action="append", default=[],
                        help="a variant knob of hpu_ops.linalg, e.g. _HH_FORWARD=wy (recorded in every row)")
    args = parser.parse_args()
    from hpu_ops import linalg as ops_linalg

    for knob in args.set:
        key, value = knob.split("=", 1)
        setattr(ops_linalg, key, value)
    hpu_ops.enable()
    from hpu_ops import tpc

    hpu_ops._STATE["undo"]()  # enable()'s own undo; the kernel path stays set, the rewrites are re-registered below
    state = {"undo": hpu_ops._install("hpu")}

    print("PLUGINS hpu_ops", hpu_ops.__file__, "tpc_ready", tpc.ready(), "lazy", LAZY, "bridge 1.21.4", flush=True)
    for name, fn, cpu_args in cases(args.n, args.batch):
        if args.only and not any(s in name for s in args.only):
            continue
        row = {"op": name, "n": args.n, "batch": args.batch, "lazy": LAZY, "set": args.set}
        try:
            ref = fn(*[a.double() if a.dtype.is_floating_point else a for a in cpu_args])
            ref32 = fn(*cpu_args)
            row["err_cpu32"] = rel_err(ref32, ref)
            h_args = [a.to("hpu") for a in cpu_args]
            calls0 = sum(_dispatch._STATE["calls"].values())
            ours, stock = [], []
            row["ours_fallback"], row["stock_fallback"] = {}, {}
            row["ours_cpu_warn"] = row["stock_cpu_warn"] = 0
            for _ in range(REPS):  # alternate, so drift and first-call costs hit both sides
                got, ms, fb, warn = timed(fn, h_args)
                ours.append(ms)
                row["ours_fallback"] = merge(row["ours_fallback"], fb)
                row["ours_cpu_warn"] += warn
                state["undo"]()
                try:
                    _, ms, fb, warn = timed(fn, h_args)
                finally:
                    state["undo"] = hpu_ops._install("hpu")
                stock.append(ms)
                row["stock_fallback"] = merge(row["stock_fallback"], fb)
                row["stock_cpu_warn"] += warn
            row["ours_calls"] = sum(_dispatch._STATE["calls"].values()) - calls0  # counts both sides: stock adds 0
            row["ours_err"] = rel_err(got, ref)
            row["ours_ms"], row["stock_ms"] = min(ours), min(stock)
            row["ours_ms_all"], row["stock_ms_all"] = [round(x, 3) for x in ours], [round(x, 3) for x in stock]
            row["speedup"] = row["stock_ms"] / row["ours_ms"]
            row["status"] = "ok" if not row["ours_fallback"] and not row["ours_cpu_warn"] else "fallback"
            if not row["ours_calls"]:
                row["status"] = "declined"
            if not row["ours_err"] <= max(1e-6, 10 * row["err_cpu32"]):
                row["status"] = "inaccurate"
        except Exception as e:  # noqa: BLE001  (one bad op must not hide the others)
            row["status"] = "error"
            row["error"] = f"{type(e).__name__}: {str(e)[:300]}"
        print("RESULT", json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
