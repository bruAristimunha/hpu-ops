"""On a Gaudi card: one train step (forward, backward, Adam) of the SPD models that use the linalg composites.
    PT_HPU_LAZY_MODE=0|1 python3 tests/linalg_steps.py [--baseline]
SPDNet, TSMNet, TensorCSPNet, EEGSPDNet and GBWAtt on a real batch of 32 at BNCI2014_001's width (22 channels):
five warm-up steps, then the mean ms of 20 steps and the CPU-fallback ops per step (bridge ``cpu_fallback`` metric).
A model fails when its step errors or when one of the ops of the linalg composites falls back: solve_triangular,
cholesky_solve, solve, inv, det, slogdet, eigh and eigvalsh. One ``RESULT {json}`` line per model; the last line
counts them; exits 1 if any model failed. ``--baseline`` skips ``hpu_ops.enable()`` to time the stock stack: those
fallbacks are then recorded and do not fail a model. Needs the spd_diag package (``spd_diag.protocol``,
``spd_diag.cache``, ``spd_diag.models``)."""
import argparse
import json
import os
import sys
import time
import traceback

import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.hpu as hthpu
import torch
from habana_frameworks.torch.hpu.metrics import metric_global

LAZY = os.environ.get("PT_HPU_LAZY_MODE", "1") != "0"
HPU = torch.device("hpu")
torch.manual_seed(0)

DATASET = "BNCI2014_001"
MODEL_NAMES = ("SPDNet", "TSMNet", "TensorCSPNet", "EEGSPDNet", "GBWAtt")
BATCH = 32
WARMUP = 5
STEPS = 20
LINALG_OPS = ("solve_triangular", "cholesky_solve", "linalg_solve", "linalg_inv", "inverse", "linalg_det",
              "linalg_slogdet", "eigh")  # substrings of the bridge's fallback op names


def sync():
    if LAZY:
        htcore.mark_step()
    hthpu.synchronize()


def fb_ops():
    return dict(dict(metric_global("cpu_fallback").stats()).get("FallbackOps", {}) or {})


def delta(a, b):
    return {k: b[k] - a.get(k, 0) for k in b if b[k] - a.get(k, 0)}


def emit(**kw):
    print("RESULT " + json.dumps(kw, default=str), flush=True)


def resolve(name, models):
    """The key of ``models`` that is ``name``, up to case."""
    found = [k for k in models if k.lower() == name.lower()]
    if not found:
        raise KeyError(f"{name} is not in spd_diag.models.MODELS: {sorted(models)}")
    return found[0]


def run_model(name, P, info, X, y, baseline):
    row = {"model": name, "dataset": DATASET, "n_ch": X.shape[1], "batch": BATCH, "lazy": LAZY,
           "baseline": baseline}
    try:
        torch.manual_seed(0)
        model = P.build(name, X.shape[1], info, None).to(HPU)
        xb = X.to(HPU)
        if hasattr(model, "fixed_input"):
            xb = model.fixed_input(X).to(HPU)
        if hasattr(model, "fit_graph"):
            model.fit_graph(X)
        opt = torch.optim.Adam(model.parameters(), 1e-3)
        yb = y.to(HPU)

        def step():
            opt.zero_grad(set_to_none=True)
            out = model(xb)
            out = out[0] if isinstance(out, tuple) else out
            torch.nn.functional.cross_entropy(out, yb).backward()
            if LAZY:
                htcore.mark_step()
            opt.step()
            if LAZY:
                htcore.mark_step()
        for _ in range(WARMUP):
            step()
        sync()
        f0 = fb_ops()
        t = time.perf_counter()
        for _ in range(STEPS):
            step()
        sync()
        row["ms_per_step"] = round((time.perf_counter() - t) / STEPS * 1e3, 1)
        row["fallback_per_step"] = {k: v / STEPS for k, v in delta(f0, fb_ops()).items()}
        row["linalg_fallback_per_step"] = {k: v for k, v in row["fallback_per_step"].items()
                                           if any(s in k for s in LINALG_OPS)}
        row["status"] = "fallback" if row["linalg_fallback_per_step"] else "ok"
    except Exception as e:  # noqa: BLE001  (a failing case is a row, not a crash)
        row.update(status="error", error=repr(e)[:400],
                   where=[f"{f.filename.split('site-packages/')[-1]}:{f.lineno}"
                          for f in traceback.extract_tb(e.__traceback__)][-4:])
    emit(**row)
    return row["status"] == "ok" or (baseline and row["status"] == "fallback")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--baseline", action="store_true", help="skip hpu_ops.enable()")
    baseline = parser.parse_args().baseline
    if not baseline:
        import hpu_ops
        hpu_ops.enable()
        try:
            import hpu_complex
            hpu_complex.enable()
        except ImportError:
            pass
    from spd_diag import protocol as P
    from spd_diag.cache import open_cache
    from spd_diag.models import MODELS
    P._enable_device("hpu")
    cache = open_cache(DATASET)
    info, sd = cache.info, cache.subject(cache.subjects[0], "X")
    X = torch.as_tensor(sd.X[:BATCH], dtype=torch.float32)
    X = X / X.std()
    y = torch.as_tensor(sd.y[:BATCH]).long()
    results = []
    for name in MODEL_NAMES:
        try:
            name = resolve(name, MODELS)
        except KeyError as e:
            emit(model=name, dataset=DATASET, lazy=LAZY, baseline=baseline, status="error", error=str(e)[:400])
            results.append(False)
            continue
        results.append(run_model(name, P, info, X, y, baseline))
    print(f"SUMMARY {len(results)} models, {results.count(False)} failed", flush=True)
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
