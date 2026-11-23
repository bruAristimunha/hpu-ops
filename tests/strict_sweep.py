"""On a Gaudi card: the CPU executions of one training step and one inference forward per model, from strict mode.
    PT_HPU_LAZY_MODE=0|1 python3 tests/strict_sweep.py --family spd|bd [--models A,B] [--compile] [--out file.jsonl]
spd: the models of ``spd_diag.models.MODELS`` on a real batch of 32 of BNCI2014_001 (22 channels), cross-entropy.
bd: the braindecode zoo (``braindecode.models.util.models_mandatory_parameters``, the signal parameters of its tests),
random input of batch 32, mean-square loss; ``braindecode_kernels.patch_braindecode()`` after ``hpu_complex.enable()``.
Per model: one warm-up step (forward, backward, Adam), then ``hpu_ops.strict("report")`` over ``STEPS`` steps (counts
per step), the mean ms of ``TIMED`` synced steps without strict mode, and the inference_mode forward under strict
mode. ``--compile``: ``torch.compile(model, backend="hpu_backend")`` (eager bridge); the first step's rows (tracing,
compiling) are recorded apart; ops inside Habana's recipes are seen through the bridge's counter, per step. One ``RESULT {json}`` line per model, also appended to ``--out``."""
import argparse
import gc
import json
import os
import sys
import time
import traceback
import warnings

import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.hpu as hthpu
import torch

LAZY = os.environ.get("PT_HPU_LAZY_MODE", "1") != "0"
HPU = torch.device("hpu")
BATCH, STEPS, TIMED = 32, 3, 5
PER_OP = True  # --step-check: no dispatch mode, the counters read per block (the program as it runs unwatched)
BRIDGE = "1.21.4"


def sync():
    if LAZY:
        htcore.mark_step()
    hthpu.synchronize()


def rows(hpu_ops, per=1):
    return [dict(source=r[0], op=r[1], tensors=r[2], per_step=round(r[3] / per, 3), where=_short(r[4]))
            for r in hpu_ops.strict_report()]


def _short(where):
    for cut in ("site-packages/", "/src/_ext/", "/src/"):
        if cut in where:
            return where.split(cut)[-1]
    return where


def build_spd(names):
    from spd_diag import protocol as P
    from spd_diag.cache import open_cache
    from spd_diag.models import MODELS

    cache = open_cache("BNCI2014_001")
    info, sd = cache.info, cache.subject(cache.subjects[0], "X")
    X = torch.as_tensor(sd.X[:BATCH], dtype=torch.float32)
    X = X / X.std()
    y = torch.as_tensor(sd.y[:BATCH]).long()

    def make(name):
        torch.manual_seed(0)
        model = P.build(name, X.shape[1], info, None)
        xb = model.fixed_input(X) if hasattr(model, "fixed_input") else X
        if hasattr(model, "fit_graph"):
            model.fit_graph(X)
        return model, xb, y, "ce"

    return (names or sorted(MODELS)), make


def build_bd(names):
    try:
        from braindecode.models.util import _get_signal_params
    except ImportError:  # moved to the test package (braindecode master, Nov 2026)
        from test.unit_tests.models._fixtures import _get_signal_params
    from braindecode.models.util import models_dict, models_mandatory_parameters

    specs = {name: (required, params) for name, required, params in models_mandatory_parameters}

    def make(name):
        torch.manual_seed(0)
        required, params = specs[name]
        sp = _get_signal_params(params, required)
        model = models_dict[name](**sp)
        full = _get_signal_params(params)
        n_chans = sp.get("n_chans") or (len(sp["chs_info"]) if "chs_info" in sp else full["n_chans"])
        n_times = sp.get("n_times") or full["n_times"]
        X = torch.randn(BATCH, n_chans, n_times)
        return model, X, None, "mse"

    return (names or [n for n, _, _ in models_mandatory_parameters]), make


def first_tensor(out):
    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, dict):
        out = list(out.values())
    for o in out:
        t = first_tensor(o) if not isinstance(o, torch.Tensor) else o
        if t is not None and t.is_floating_point():
            return t
    return None


def run(name, make, hpu_ops, compiled):
    mode = "compile" if compiled else ("lazy" if LAZY else "eager")
    row = dict(model=name, mode=mode, batch=BATCH, bridge=BRIDGE)
    stage = "build"
    try:
        model, X, y, loss_kind = make(name)
        model = model.to(HPU).train()
        xb = X.to(HPU)
        yb = y.to(HPU) if y is not None else None
        fwd = torch.compile(model, backend="hpu_backend") if compiled else model
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.Adam(params, 1e-3) if params else None

        def step():
            if opt is not None:
                opt.zero_grad(set_to_none=True)
            out = first_tensor(fwd(xb))
            if loss_kind == "ce" and out.dim() == 2:
                loss = torch.nn.functional.cross_entropy(out, yb)
            else:
                loss = out.float().square().mean()
            loss.backward()
            if LAZY:
                htcore.mark_step()
            if opt is not None:
                opt.step()
            if LAZY:
                htcore.mark_step()
            return loss

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            stage = "warm-up"
            hpu_ops.strict_reset()
            if compiled:  # tracing and compiling: hpu-ops' routes are seen here, once
                with hpu_ops.strict("report", per_op=PER_OP):
                    step()
                    sync()
                row["trace_rows"] = rows(hpu_ops)
                hpu_ops.strict_reset()
            else:
                step()
                sync()
            stage = "strict steps"
            with hpu_ops.strict("report", per_op=PER_OP):
                for _ in range(STEPS):
                    keep = step()
                sync()
            row["step_rows"] = rows(hpu_ops, STEPS)
            hpu_ops.strict_reset()
            stage = "timed steps"
            t = time.perf_counter()
            for _ in range(TIMED):
                keep = step()
            sync()
            row["ms_per_step"] = round((time.perf_counter() - t) / TIMED * 1e3, 2)
            del keep
            stage = "inference"
            model.eval()
            with torch.inference_mode():
                fwd(xb)  # warm-up (compile: a graph of its own)
                sync()
                with hpu_ops.strict("report", per_op=PER_OP):
                    out = fwd(xb)
                    sync()
                row["infer_rows"] = rows(hpu_ops)
                hpu_ops.strict_reset()
                del out
        row["cpu_warnings"] = sorted({str(w.message)[:200] for w in caught if "CPU" in str(w.message)
                                      or "cpu" in str(w.message)})
        row["status"] = "ok"
    except Exception as e:  # noqa: BLE001  (a failing model is a row, not a crash)
        hpu_ops.strict_reset()
        row.update(status="error", stage=stage, error=repr(e)[:500],
                   where=[f"{_short(f.filename)}:{f.lineno}" for f in traceback.extract_tb(e.__traceback__)][-5:])
    finally:
        gc.collect()
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--family", choices=("spd", "bd"), required=True)
    parser.add_argument("--models", default="")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--out", default="")
    parser.add_argument("--step-check", action="store_true",
                        help="strict(per_op=False): no dispatch mode at all, bridge and hpu_complex counters per block")
    parser.add_argument("--resume", action="store_true",
                        help="skip the models --out has; a model started and never finished is recorded as crashed")
    args = parser.parse_args()
    global PER_OP
    PER_OP = not args.step_check
    import hpu_complex

    hpu_complex.enable()
    import hpu_ops

    hpu_ops.enable()
    if args.family == "bd":
        import braindecode_kernels

        braindecode_kernels.patch_braindecode()
    names = [n for n in args.models.split(",") if n]
    names, make = (build_spd if args.family == "spd" else build_bd)(names)
    print("PLUGINS", hpu_ops.__file__, hpu_complex.__file__, "lazy", LAZY, "compile", args.compile, "models",
          len(names), flush=True)
    mode = "compile" if args.compile else ("lazy" if LAZY else "eager")
    done, started = set(), set()
    if args.resume and os.path.exists(args.out):
        for line in open(args.out):
            r = json.loads(line)
            (started if r.get("status") == "started" else done).add(r["model"])
    for name in names:
        if name in done:
            continue
        if name in started:  # the process died in this model (abort, segfault, timeout)
            row = dict(family=args.family, model=name, mode=mode, batch=BATCH, bridge=BRIDGE, status="crashed",
                       error="the process died while running this model")
        else:
            if args.out:
                with open(args.out, "a") as f:
                    f.write(json.dumps(dict(model=name, status="started")) + "\n")
            row = dict(family=args.family, **run(name, make, hpu_ops, args.compile))
        line = json.dumps(row, default=str)
        print("RESULT " + line, flush=True)
        if args.out:
            with open(args.out, "a") as f:
                f.write(line + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
