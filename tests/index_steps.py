"""On a Gaudi card: train-step A/B of hpu_ops' indexing rewrites (indexing.install_index, install_mask_fill) in one
process. PT_HPU_LAZY_MODE=0|1 python3 tests/index_steps.py [--models A,B] [--rounds 2]
enable() runs without the two rewrites; then, per model and round, the step (forward, backward, Adam) is timed without
them ("before") and with them installed on top ("after"): 5 warm-up steps, the mean ms of 20 synced steps, the
bridge's cpu_fallback ops per step and the "runs on the CPU" warnings. One RESULT {json} line per model and arm."""
import argparse
import json
import os
import time
import warnings

import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.hpu as hthpu
import torch
from habana_frameworks.torch.hpu.metrics import metric_global

LAZY = os.environ.get("PT_HPU_LAZY_MODE", "1") != "0"
NINE = "SPDResNet,GyroSPD,CorNet,SPNN,DeepGeoCCA,SPDNetAE,EEGSPDNet,MAtt,CorAtt"
WARMUP, STEPS, BATCH, SEED = 5, 20, 32, 0


def sync():
    if LAZY:
        htcore.mark_step()
    hthpu.synchronize()


def fb():
    return dict(dict(metric_global("cpu_fallback").stats()).get("FallbackOps", {}) or {})


def time_steps(P, name, X, y, info):
    torch.manual_seed(SEED)
    model = P.build(name, X.shape[1], info, None)
    if hasattr(model, "fit_graph"):
        model.fit_graph(X)
    xin = model.fixed_input(X) if hasattr(model, "fixed_input") else X
    model = model.to("hpu")
    xb, yb = xin[:BATCH].to("hpu"), y[:BATCH].to("hpu")
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    losses = []

    def step():
        opt.zero_grad(set_to_none=True)
        out = model(xb)
        loss = torch.nn.functional.cross_entropy(out[0] if isinstance(out, tuple) else out, yb)
        loss.backward()
        if LAZY:
            htcore.mark_step()
        opt.step()
        if LAZY:
            htcore.mark_step()
        losses.append(loss.detach())

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(WARMUP):
            step()
        sync()
        f0 = fb()
        t = time.perf_counter()
        for _ in range(STEPS):
            step()
        sync()
        ms = (time.perf_counter() - t) / STEPS * 1e3
        f1 = fb()
    fallback = {k: (f1[k] - f0.get(k, 0)) / STEPS for k in f1 if f1[k] - f0.get(k, 0)}
    cpu_warn = sorted({str(w.message)[:120] for w in caught if "runs on the CPU" in str(w.message)})
    return round(ms, 2), fallback, cpu_warn, float(torch.stack(losses).sum().cpu())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=NINE)
    ap.add_argument("--rounds", type=int, default=2)
    a = ap.parse_args()
    import hpu_ops
    from hpu_ops import indexing
    hpu_ops.FIXES = tuple(f for f in hpu_ops.FIXES if f not in (indexing.install_index, indexing.install_mask_fill))
    from spd_diag import protocol as P
    from spd_diag.cache import open_cache
    P._enable_device("hpu")
    cache = open_cache("BNCI2014_001")
    sd = cache.subject(cache.subjects[0], "X")
    X = torch.as_tensor(sd.X[:BATCH], dtype=torch.float32)
    X = X / X.std()
    y = torch.as_tensor(sd.y[:BATCH]).long()
    print("PLUGINS hpu_ops", hpu_ops.__file__, "lazy", LAZY, flush=True)
    for name in a.models.split(","):
        for rnd in range(a.rounds):
            for arm in ("before", "after"):
                undo = None
                if arm == "after":
                    undo = hpu_ops._dispatch.chain([indexing.install_index("hpu"), indexing.install_mask_fill("hpu")])
                hpu_ops._STATE["calls"].clear()
                try:
                    ms, fallback, cpu_warn, lsum = time_steps(P, name, X, y, cache.info)
                    row = {"ms_per_step": ms, "fallback_per_step": fallback, "cpu_warnings": cpu_warn,
                           "rewrites_per_step": {k: v / (WARMUP + STEPS) for k, v in hpu_ops._STATE["calls"].items()
                                                 if k in ("index", "mask_fill")},
                           "loss_sum": lsum}
                except Exception as e:  # noqa: BLE001
                    row = {"error": repr(e)[:300]}
                finally:
                    if undo:
                        undo()
                print("RESULT " + json.dumps({"model": name, "lazy": LAZY, "round": rnd, "arm": arm, **row}),
                      flush=True)


if __name__ == "__main__":
    main()
