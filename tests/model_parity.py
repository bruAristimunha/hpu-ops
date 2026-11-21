"""On a Gaudi card: model-level parity of every spd_diag model against the CPU.
    PT_HPU_LAZY_MODE=0|1 python3 tests/model_parity.py [--models A,B] [--steps 50] [--parts sweep,grads,train,compile]
Each model of ``spd_diag.models.MODELS`` (all of them unless ``--models``) at batch 32 on BNCI2014_001 (22 channels),
four parts, each in its own try/except so a failure is a RESULT row and not a crash:
  sweep    one warm-up plus 10 timed fwd+bwd+Adam steps on the HPU: ms per step, bridge fallbacks per step, host syncs
           (``_local_scalar_dense`` in one profiled step), "runs on the CPU" warnings, hpu_ops rewrites per step.
           Status ok, fallback (recorded, not a failure) or error.
  grads    one fwd+bwd on the same batch with the same initial weights on the HPU, on the CPU in float64 (the reference)
           and on the CPU in float32. Per parameter rel_err = max|g - g64| / max(max|g64|, abs_floor), with abs_floor =
           1e-3 x the largest gradient max of the model (near-zero gradients, such as a bias before a normalization,
           would otherwise dominate). A parameter fails when rel_err > max(1e-3, 10 x rel_err_cpu32) or is non-finite.
  train    ``--steps`` Adam steps, fixed seed, same data order, on the HPU and on the CPU in float32. Fails when the HPU
           curve has a NaN or ends above 1.5 x the CPU loss; ``final_within_5pct`` records the 5% criterion.
  compile  torch.compile(model, backend="hpu_backend", dynamic=False): one fwd+bwd against eager HPU (loss 1e-4
           relative, grads 1e-3). ``unavailable`` (not a failure) when the backend is not registered.
All variants of a model start from one CPU build (same seed), converted to the target device and dtype, so the weights
are identical by construction. grads, train and compile set dropout probabilities to 0, since the HPU and CPU random
streams differ. For models with ``fixed_input`` or ``fit_graph``, both run once on the float32 CPU model before the
conversion, so every variant sees the same input and the same graph. One PLUGINS line, one ``RESULT {json}`` per
(model, part), a last SUMMARY line with counts per status. Exits 1 when a part errored or failed, except models in
KNOWN_ISSUES (failing on origin/main for a known reason): those rows carry ``known_issue`` and do not count."""
import argparse
import collections
import contextlib
import gc
import json
import math
import os
import sys
import time
import traceback
import types
import warnings

import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.hpu as hthpu
import torch
from habana_frameworks.torch.hpu.metrics import metric_global

LAZY = os.environ.get("PT_HPU_LAZY_MODE", "1") != "0"
HPU = torch.device("hpu")
CPU = torch.device("cpu")
torch.manual_seed(0)

DATASET = "BNCI2014_001"
BATCH = 32
MAX_BATCHES = 4  # trials loaded: up to 4 batches, cycled in a fixed order by train
SEED = 0
LR = 1e-3
WARMUP = 1
TIMED = 10
GRAD_TOL = 1e-3  # grads: floor of the per-parameter threshold (HPU or CPU float32 vs CPU float64)
GRAD_CPU32_FACTOR = 10
FLOOR_FRAC = 1e-3  # abs_floor = this x the largest gradient max of the model
COMPILE_LOSS_TOL = 1e-4
COMPILE_GRAD_TOL = 1e-3
DIVERGE = 1.5
WITHIN = 1.05
PARTS = ("sweep", "grads", "train", "compile")
SYNC_KEY = "_local_scalar_dense"
# (lower-case model, lazy) -> reason: errors on origin/main too, recorded under known_issue and not counted as failures
KNOWN_ISSUES = {("gbwatt", True): "GBWAtt errors in lazy mode on origin/main (#20)"}


def sync():
    if LAZY:
        htcore.mark_step()
    hthpu.synchronize()


def fb_ops():
    return dict(dict(metric_global("cpu_fallback").stats()).get("FallbackOps", {}) or {})


def delta(a, b):
    return {k: b[k] - a.get(k, 0) for k in b if b[k] - a.get(k, 0)}


def clean(o):
    """Strict JSON: non-finite floats become the strings "nan", "inf", "-inf"."""
    if isinstance(o, float) and not math.isfinite(o):
        return str(o)
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    return o


def emit(**kw):
    print("RESULT " + json.dumps(clean(kw), default=str), flush=True)


def resolve(name, models):
    """The key of ``models`` that is ``name``, up to case."""
    found = [k for k in models if k.lower() == name.lower()]
    if not found:
        raise KeyError(f"{name} is not in spd_diag.models.MODELS: {sorted(models)}")
    return found[0]


@contextlib.contextmanager
def default_dtype(dtype):
    old = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(old)


def make(name, ctx, device, dtype=torch.float32, deterministic=True):
    """(model, inputs of all trials, number of dropout modules zeroed): one CPU build from SEED, then converted."""
    torch.manual_seed(SEED)
    model = ctx.P.build(name, ctx.X.shape[1], ctx.info, None)
    if hasattr(model, "fit_graph"):
        model.fit_graph(ctx.X)
    xin = model.fixed_input(ctx.X) if hasattr(model, "fixed_input") else ctx.X
    n_drop = 0
    if deterministic:
        for m in model.modules():
            if isinstance(m, torch.nn.modules.dropout._DropoutNd):
                m.p = 0.0
                n_drop += 1
    if dtype == torch.float64:
        model = model.double()
        xin = xin.double() if xin.is_floating_point() else xin
    return model.to(device), xin.to(device), n_drop


def batches(ctx, xin, device):
    xs = [xin[i * BATCH:(i + 1) * BATCH] for i in range(ctx.nb)]
    ys = [ctx.y[i * BATCH:(i + 1) * BATCH].to(device) for i in range(ctx.nb)]
    return xs, ys


def forward_loss(model, xb, yb):
    out = model(xb)
    out = out[0] if isinstance(out, tuple) else out
    return torch.nn.functional.cross_entropy(out, yb)


def on_hpu(t):
    return t.device.type == HPU.type


def fwd_bwd(model, xb, yb, dtype=torch.float32):
    """One forward and backward; the loss as a Python float (read after the sync)."""
    model.zero_grad(set_to_none=True)
    with default_dtype(dtype):
        loss = forward_loss(model, xb, yb)
        loss.backward()
    if on_hpu(xb):
        sync()
    return float(loss.detach().double().cpu())


def flat(t):
    t = t.detach()
    if t.is_complex():
        t = torch.view_as_real(t)
    return t.double().cpu()


def grads_of(model):
    return {n: (None if p.grad is None else flat(p.grad)) for n, p in model.named_parameters()}


def peak(g):
    return float(g.abs().max()) if g.numel() else 0.0


def finite(g):
    return bool(torch.isfinite(g).all())


def abs_floor(ref):
    """FLOOR_FRAC x the largest gradient max of the model (tiny when every gradient is zero)."""
    peaks = [peak(g) for g in ref.values() if g is not None and finite(g)]
    return FLOOR_FRAC * max(peaks, default=0.0) or 1e-30


def rel_errs(test, ref, floor):
    """Per parameter max|test - ref| / max(max|ref|, floor); nan for a non-finite gradient, inf for a gradient present
    on one side only or of another shape."""
    out = {}
    for n, r in ref.items():
        t = test.get(n)
        if r is None and t is None:
            continue
        if r is None or t is None or t.shape != r.shape:
            out[n] = float("inf")
        elif not (finite(t) and finite(r)):
            out[n] = float("nan")
        else:
            out[n] = (peak(t - r) / max(peak(r), floor)) if r.numel() else 0.0
    return out


def rel(a, b):
    return abs(a - b) / max(abs(b), 1e-12)


def part_sweep(name, ctx):
    import hpu_ops
    ops_linalg = sys.modules.get("hpu_ops.linalg")
    if hasattr(ops_linalg, "_WARNED"):
        ops_linalg._WARNED.clear()  # the float64 warning is once per process: let each model see it
    model, xin, _ = make(name, ctx, HPU, deterministic=False)
    xs, ys = batches(ctx, xin, HPU)
    xb, yb = xs[0], ys[0]
    opt = torch.optim.Adam(model.parameters(), LR)

    def step():
        opt.zero_grad(set_to_none=True)
        forward_loss(model, xb, yb).backward()
        if LAZY:
            htcore.mark_step()
        opt.step()
        if LAZY:
            htcore.mark_step()

    row = {}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(WARMUP):
            step()
        sync()
        f0, c0 = fb_ops(), dict(hpu_ops._STATE["calls"])
        t = time.perf_counter()
        for _ in range(TIMED):
            step()
        sync()
        row["ms_per_step"] = round((time.perf_counter() - t) / TIMED * 1e3, 1)
        row["fallback_per_step"] = {k: v / TIMED for k, v in delta(f0, fb_ops()).items()}
        row["hpu_ops_calls_per_step"] = {k: v / TIMED for k, v in delta(c0, dict(hpu_ops._STATE["calls"])).items()}
        try:  # one more step, profiled: it is not part of the timing
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
                step()
                sync()
            row["host_syncs_per_step"] = sum(e.count for e in prof.key_averages() if SYNC_KEY in e.key)
        except Exception as e:  # noqa: BLE001
            row["host_syncs_per_step"] = None
            row["host_syncs_error"] = repr(e)[:200]
    row["cpu_warnings"] = sorted({str(w.message) for w in caught if "runs on the CPU" in str(w.message)})
    row["status"] = "fallback" if row["fallback_per_step"] or row["cpu_warnings"] else "ok"
    return row


def grad_failures(eh, e32):
    """Per-parameter thresholds max(GRAD_TOL, GRAD_CPU32_FACTOR x the CPU float32 error) and the failing parameters,
    worst first. A nan error fails, and so does a nan e32 (nan threshold). An inf e32 (a gradient missing or misshapen
    on CPU float32 too) does not make the threshold infinite, so a gradient missing on the HPU (inf) still fails."""
    thr = {n: GRAD_TOL if e32.get(n, 0.0) == math.inf else max(GRAD_TOL, GRAD_CPU32_FACTOR * e32.get(n, 0.0))
           for n in eh}
    failed = sorted((n for n in eh if not eh[n] <= thr[n]), key=lambda n: -(eh[n] if eh[n] == eh[n] else math.inf))
    return thr, failed


def part_grads(name, ctx):
    runs, n_drop = {}, 0
    for tag, device, dtype in (("hpu", HPU, torch.float32), ("cpu64", CPU, torch.float64),
                               ("cpu32", CPU, torch.float32)):
        model, xin, n_drop = make(name, ctx, device, dtype)
        xs, ys = batches(ctx, xin, device)
        loss = fwd_bwd(model, xs[0], ys[0], dtype)
        runs[tag] = (loss, grads_of(model))
    (lh, gh), (l64, g64), (l32, g32) = runs["hpu"], runs["cpu64"], runs["cpu32"]
    floor = abs_floor(g64)
    eh, e32 = rel_errs(gh, g64, floor), rel_errs(g32, g64, floor)
    thr, failed = grad_failures(eh, e32)
    loss_err, loss_err32 = rel(lh, l64), rel(l32, l64)
    loss_ok = loss_err <= max(GRAD_TOL, GRAD_CPU32_FACTOR * loss_err32)
    worst = sorted(eh, key=lambda n: -(eh[n] if eh[n] == eh[n] else math.inf))[:5]
    return {"status": "ok" if not failed and loss_ok else "fail", "n_params": len(eh), "n_failed": len(failed),
            "abs_floor": floor, "dropout_modules_zeroed": n_drop,
            "loss_hpu": lh, "loss_cpu64": l64, "loss_cpu32": l32, "loss_rel_err": loss_err,
            "loss_rel_err_cpu32": loss_err32, "loss_ok": loss_ok,
            "failed": [{"param": n, "rel_err": eh[n], "rel_err_cpu32": e32.get(n), "threshold": thr[n]}
                       for n in failed[:10]],
            "worst": [{"param": n, "rel_err": eh[n], "rel_err_cpu32": e32.get(n)} for n in worst]}


def curve(name, ctx, device, steps, order):
    model, xin, _ = make(name, ctx, device)
    xs, ys = batches(ctx, xin, device)
    opt = torch.optim.Adam(model.parameters(), LR)
    losses = []
    for i in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = forward_loss(model, xs[order[i]], ys[order[i]])
        loss.backward()
        if on_hpu(xs[0]) and LAZY:
            htcore.mark_step()
        opt.step()
        if on_hpu(xs[0]) and LAZY:
            htcore.mark_step()
        losses.append(loss.detach())
    if on_hpu(xs[0]):
        sync()
    return [float(v) for v in torch.stack(losses).double().cpu()]


def part_train(name, ctx):
    gen = torch.Generator().manual_seed(SEED)
    order = [int(j) for _ in range(-(-ctx.steps // ctx.nb)) for j in torch.randperm(ctx.nb, generator=gen)]
    order = order[:ctx.steps]
    ch = curve(name, ctx, HPU, ctx.steps, order)
    cc = curve(name, ctx, CPU, ctx.steps, order)
    bad = any(not math.isfinite(v) for v in ch)
    fh, fc = ch[-1], cc[-1]
    max_rel = max((abs(h - c) / max(abs(c), 1e-8) for h, c in zip(ch, cc)), default=0.0)
    diverged = bad or fh > DIVERGE * fc + 1e-3
    return {"status": "fail" if diverged else "ok", "steps": ctx.steps, "n_batches": ctx.nb,
            "loss_hpu": [round(v, 6) for v in ch] if not bad else ch, "loss_cpu": [round(v, 6) for v in cc],
            "max_rel_dloss": max_rel, "final_hpu": fh, "final_cpu": fc, "hpu_nan": bad,
            "final_within_5pct": bool(not bad and fh <= WITHIN * fc + 1e-6)}


def hpu_backend_available():
    try:
        import habana_frameworks.torch.dynamo.compile_backend  # noqa: F401  (registers hpu_backend)
    except Exception:  # noqa: BLE001
        pass
    try:
        import torch._dynamo as dynamo
        return "hpu_backend" in dynamo.list_backends(exclude_tags=())
    except Exception:  # noqa: BLE001
        return False


def part_compile(name, ctx):
    if not hpu_backend_available():
        return {"status": "unavailable", "reason": "hpu_backend is not registered with torch._dynamo"}
    eager, xin, _ = make(name, ctx, HPU)
    xs, ys = batches(ctx, xin, HPU)
    le = fwd_bwd(eager, xs[0], ys[0])
    base, _, _ = make(name, ctx, HPU)
    compiled = torch.compile(base, backend="hpu_backend", dynamic=False)
    t = time.perf_counter()
    lc = fwd_bwd(compiled, xs[0], ys[0])
    compile_s = round(time.perf_counter() - t, 2)
    ge, gc_ = grads_of(eager), grads_of(base)
    errs = rel_errs(gc_, ge, abs_floor(ge))
    failed = sorted(n for n in errs if not errs[n] <= COMPILE_GRAD_TOL)
    loss_err = rel(lc, le)
    ok = not failed and loss_err <= COMPILE_LOSS_TOL
    return {"status": "ok" if ok else "fail", "compile_s": compile_s, "loss_eager": le, "loss_compiled": lc,
            "loss_rel_err": loss_err, "n_params": len(errs), "n_failed": len(failed),
            "failed": [{"param": n, "rel_err": errs[n]} for n in failed[:10]],
            "worst": sorted(errs.values(), key=lambda v: -(v if v == v else math.inf))[:3]}


PART_FNS = {"sweep": part_sweep, "grads": part_grads, "train": part_train, "compile": part_compile}


def run_part(name, part, ctx):
    row = {"model": name, "part": part, "dataset": DATASET, "n_ch": ctx.X.shape[1], "batch": BATCH, "lazy": LAZY}
    try:
        row.update(PART_FNS[part](name, ctx))
    except Exception as e:  # noqa: BLE001  (a failing case is a row, not a crash)
        row.update(status="error", error=repr(e)[:400],
                   where=[f"{f.filename.split('site-packages/')[-1]}:{f.lineno}"
                          for f in traceback.extract_tb(e.__traceback__)][-4:])
    known = KNOWN_ISSUES.get((name.lower(), LAZY))
    if known and row["status"] in ("error", "fail"):
        row["known_issue"] = known
    emit(**row)
    gc.collect()
    return row


def counts(rows):
    out = collections.Counter(r["status"] for r in rows)
    out["known_issue"] = sum("known_issue" in r for r in rows)
    return dict(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", default="", help="comma-separated names (default: every model)")
    parser.add_argument("--steps", type=int, default=50, help="Adam steps of the train part")
    parser.add_argument("--parts", default=",".join(PARTS), help=f"subset of {','.join(PARTS)}")
    parser.add_argument("--large-maxn", type=int, default=None,
                        help="set hpu_ops.tpc.LARGE_MAXN before enable() (64: the parent branch's route past n = 64)")
    args = parser.parse_args()
    parts = [p for p in args.parts.split(",") if p]
    if not parts or not set(parts) <= set(PARTS):
        raise SystemExit(f"--parts must be a subset of {PARTS}, got {args.parts!r}")
    import hpu_complex
    from spd_diag import protocol as P
    from spd_diag.cache import open_cache
    from spd_diag.models import MODELS

    import hpu_ops
    if args.large_maxn is not None:
        hpu_ops.tpc.LARGE_MAXN = args.large_maxn
    P._enable_device("hpu")  # hpu_ops.enable() and hpu_complex.enable()
    print("PLUGINS hpu_ops", getattr(hpu_ops, "__version__", "?"), hpu_ops.__file__, "hpu_complex",
          getattr(hpu_complex, "__version__", "?"), "tpc_ready",
          getattr(getattr(hpu_ops, "tpc", None), "ready", lambda: None)(), "lazy", LAZY, flush=True)
    cache = open_cache(DATASET)
    info, sd = cache.info, cache.subject(cache.subjects[0], "X")
    X = torch.as_tensor(sd.X[:BATCH * MAX_BATCHES], dtype=torch.float32)
    X = X / X.std()
    y = torch.as_tensor(sd.y[:BATCH * MAX_BATCHES]).long()
    ctx = types.SimpleNamespace(P=P, info=info, X=X, y=y, nb=max(1, len(X) // BATCH), steps=args.steps)
    names = [n for n in args.models.split(",") if n] or list(MODELS)
    rows = []
    for name in names:
        try:
            name = resolve(name, MODELS)
        except KeyError as e:
            rows.append({"status": "error"})
            emit(model=name, part="resolve", dataset=DATASET, lazy=LAZY, status="error", error=str(e)[:400])
            continue
        for part in parts:
            rows.append(run_part(name, part, ctx))
    bad = [r for r in rows if r["status"] in ("error", "fail") and "known_issue" not in r]
    print(f"SUMMARY {len(rows)} rows, {json.dumps(counts(rows))}, {len(bad)} failing", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
