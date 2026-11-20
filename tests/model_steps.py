"""On a Gaudi card: train steps (forward, backward, Adam) of one spd_learn model, with or without some hpu_ops fixes.
    PT_HPU_LAZY_MODE=0|1 python3 tests/model_steps.py --model GBWAtt [--ch 22] [--k resolve] [--compile]
        [--without install_index_put,install_index_2d] [--ab install_index_put,install_index_2d]
``--without`` leaves those fixes out of ``hpu_ops.enable()`` (the baseline). ``--ab`` runs the steps with them left
out, then installs them in the same process and runs again (same-process A/B; only for baselines that do not abort).
22 channels read a real BNCI2014_001 batch from the spd_diag cache; other widths use ``torch.randn``. Prints one
``RESULT {json}`` per run: the loss of the first step (same seed, same batch: comparable across eager/lazy), the loss
of the last step, ms per step (mean of ``--steps`` synced steps after ``--warmup``), CPU-fallback ops per step (bridge
``cpu_fallback`` metric) and "runs on the CPU" warnings."""
import argparse
import json
import os
import sys
import time
import traceback
import warnings

import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.hpu as hthpu
import torch
from habana_frameworks.torch.hpu.metrics import metric_global

LAZY = os.environ.get("PT_HPU_LAZY_MODE", "0") != "0"
HPU = torch.device("hpu")


def mark():
    if LAZY:
        htcore.mark_step()


def fb_ops():
    return dict(dict(metric_global("cpu_fallback").stats()).get("FallbackOps", {}) or {})


def delta(a, b):
    return {k: b[k] - a.get(k, 0) for k in b if b[k] - a.get(k, 0)}


def batch(n_ch, size):
    from spd_diag.cache import open_cache
    torch.manual_seed(0)
    if n_ch == 22:
        cache = open_cache("BNCI2014_001")
        sd = cache.subject(cache.subjects[0], "X")
        X = torch.as_tensor(sd.X[:size], dtype=torch.float32)
        return cache.info, X / X.std(), torch.as_tensor(sd.y[:size]).long()
    info = {"classes": [0, 1, 2, 3], "sfreq": 128.0, "paradigm": "imagery"}
    return info, torch.randn(size, n_ch, 1000), torch.randint(4, (size,))


def run(args, label):
    from spd_diag import protocol as P
    row = {"run": label, "model": args.model, "n_ch": args.ch, "batch": args.batch, "lazy": LAZY,
           "compile": args.compile, "k": args.k, "bridge": "1.21.4"}
    try:
        info, X, y = batch(args.ch, args.batch)
        k = P.resolve_k(args.model, args.ch, len(info["classes"]), info["sfreq"], None) if args.k == "resolve" else None
        torch.manual_seed(0)
        model = P.build(args.model, args.ch, info, k).to(HPU)
        xb = model.fixed_input(X).to(HPU) if hasattr(model, "fixed_input") else X.to(HPU)
        if hasattr(model, "fit_graph"):
            model.fit_graph(X)
        fwd = torch.compile(model, backend="hpu_backend", dynamic=False) if args.compile else model
        opt = torch.optim.Adam(model.parameters(), 1e-3)
        yb = y.to(HPU)
        losses = []

        def step():
            opt.zero_grad(set_to_none=True)
            out = fwd(xb)
            out = out[0] if isinstance(out, tuple) else out
            loss = torch.nn.functional.cross_entropy(out, yb)
            loss.backward()
            mark()
            opt.step()
            mark()
            losses.append(loss.detach())  # kept alive: lazy mode skips work whose outputs are dropped

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            step()
            hthpu.synchronize()
            row["loss_first"] = float(losses[0].cpu())
            for _ in range(args.warmup - 1):
                step()
            hthpu.synchronize()
            f0 = fb_ops()
            t = time.perf_counter()
            for _ in range(args.steps):
                step()
            hthpu.synchronize()
            row["ms_per_step"] = round((time.perf_counter() - t) / args.steps * 1e3, 1)
        row["loss_last"] = float(losses[-1].cpu())
        row["fallback_per_step"] = {k_: v / args.steps for k_, v in delta(f0, fb_ops()).items()}
        row["cpu_warnings"] = sorted({str(w.message)[:160] for w in caught if "runs on the CPU" in str(w.message)})
        row["status"] = "ok"
    except Exception as e:  # noqa: BLE001  (a failing run is a row)
        row.update(status="error", error=repr(e)[:600],
                   where=[f"{f.filename.split('site-packages/')[-1]}:{f.lineno}"
                          for f in traceback.extract_tb(e.__traceback__)][-5:])
    print("RESULT " + json.dumps(row, default=str), flush=True)
    return row["status"] == "ok"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--ch", type=int, default=22)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--k", choices=("none", "resolve"), default="none")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--without", default="", help="comma-separated hpu_ops install_* fixes to leave out")
    parser.add_argument("--ab", default="", help="leave these out, run, install them, run again (same process)")
    args = parser.parse_args()
    import hpu_complex
    import hpu_ops
    skip = {s for s in (args.without + "," + args.ab).split(",") if s}
    names = {f.__name__ for f in hpu_ops.FIXES}
    if skip - names:
        raise SystemExit(f"unknown fixes {skip - names}; known: {sorted(names)}")
    later = [f for f in hpu_ops.FIXES if f.__name__ in set(args.ab.split(","))]
    hpu_ops.FIXES = tuple(f for f in hpu_ops.FIXES if f.__name__ not in skip)
    hpu_ops.enable()
    hpu_complex.enable()
    print("PLUGINS", hpu_ops.__file__, "lazy", LAZY, "left out", sorted(skip), flush=True)
    ok = run(args, "baseline" if skip else "fixed")
    if later:
        for fix in later:
            fix("hpu")
        ok = run(args, "fixed") and ok
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
