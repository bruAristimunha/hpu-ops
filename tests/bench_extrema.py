"""On a Gaudi card: per-call cost of hpu_ops' NaN rewrites of the max/min family (``hpu_ops.extrema``) against
Gaudi's native ops. One mode per process (installing and undoing in one process skews eager timings), alternated by
the caller: ``repro/run_extrema_bench.sh``.
    PT_HPU_LAZY_MODE=0|1 python3 tests/bench_extrema.py --fix on|off [--reps 50] [--round r]
Per case: mean ms per synced call (forward, and forward + backward where differentiable) over ``--reps`` calls after
warm-up, outputs kept alive; each call is a ``mark_step`` in lazy mode. CPU fallbacks per call from the bridge
``cpu_fallback`` metric (must be 0). One ``BENCH {json}`` line per case and round."""
import argparse
import json
import os
import time

import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.hpu as hthpu
import torch
import torch.nn.functional as F
from habana_frameworks.torch.hpu.metrics import metric_global

from hpu_ops import extrema
from hpu_ops._dispatch import _STATE

LAZY = os.environ.get("PT_HPU_LAZY_MODE", "1") != "0"
HPU = torch.device("hpu")


def sync():
    if LAZY:
        htcore.mark_step()
    hthpu.synchronize()


def fallbacks():
    return sum((dict(dict(metric_global("cpu_fallback").stats()).get("FallbackOps", {}) or {})).values())


torch.manual_seed(0)
EEG = torch.randn(32, 22, 1000)
EEG2 = torch.randn(32, 22, 1000)
CASES = {  # name: (function of the inputs, inputs, differentiable)
    "amax(-1) 32x22x1000": (lambda x: x.amax(-1), (EEG,), True),
    "amax() 32x22x1000": (lambda x: x.amax(), (EEG,), True),
    "max(-1) 32x22x1000": (lambda x: x.max(-1).values, (EEG,), True),
    "argmax(-1) 32x22x1000": (lambda x: x.argmax(-1), (EEG,), False),
    "argmax(-1) bf16": (lambda x: x.argmax(-1), (EEG.bfloat16(),), False),
    "max() 32x22x1000": (lambda x: x.max(), (EEG,), True),
    "maximum 32x22x1000": (lambda x, y: torch.maximum(x, y), (EEG, EEG2), True),
    "maximum bf16": (lambda x, y: torch.maximum(x, y), (EEG.bfloat16(), EEG2.bfloat16()), True),
    "clamp(t, t) 32x22x1000": (lambda x, y: x.clamp(y - 1, y + 1), (EEG, EEG2), True),
    "norm(inf, -1) 32x22x1000": (lambda x: x.norm(float("inf"), -1), (EEG,), True),
    "max_pool2d (1,3) 32x40x1x1000": (lambda x: F.max_pool2d(x, (1, 3), (1, 3)), (torch.randn(32, 40, 1, 1000),), True),
    "max_pool2d 2 32x64x56x56": (lambda x: F.max_pool2d(x, 2), (torch.randn(32, 64, 56, 56),), True),
    "max_pool2d 2 bf16 32x64x56x56": (lambda x: F.max_pool2d(x, 2), (torch.randn(32, 64, 56, 56).bfloat16(),), True),
    "max_pool1d 3,2 32x22x1000": (lambda x: F.max_pool1d(x, 3, 2), (EEG,), True),
    "sort(-1) 32x22x1000": (lambda x: x.sort(-1).values, (EEG,), True),
    "sort(-1) 64x64": (lambda x: x.sort(-1).values, (torch.randn(64, 64),), True),
    "topk(8, -1) 32x22x1000": (lambda x: x.topk(8, -1).values, (EEG,), True),
    "argsort(-1) 32x22x1000": (lambda x: x.argsort(-1), (EEG,), False),
}


def timed(f, xs, train, reps, inference=False):
    def call():
        if inference:
            with torch.inference_mode():
                return f(*xs)
        if train:
            ins = [x.detach().requires_grad_() for x in xs]
            out = f(*ins)
            out.float().sum().backward()
            return [out] + [x.grad for x in ins]
        return f(*xs)

    for _ in range(3):
        call()
        sync()
    fb = fallbacks()
    keep = []
    t = time.perf_counter()
    for _ in range(reps):
        keep.append(call())
        if LAZY:
            htcore.mark_step()
    sync()
    ms = 1e3 * (time.perf_counter() - t) / reps
    return ms, (fallbacks() - fb) / reps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fix", choices=("on", "off"), required=True)
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--round", type=int, default=0)
    ap.add_argument("--inference", action="store_true", help="forward under torch.inference_mode() only")
    a = ap.parse_args()
    on = a.fix == "on"
    # keep the undo alive: its torch.library objects are referenced only by it, and they deregister the rewrite when
    # it is freed (dropped, the "on" arm timed Gaudi's native kernels)
    undo = extrema.install("hpu") if on else None
    inputs = {name: tuple(x.to(HPU) for x in xs) for name, (_, xs, _) in CASES.items()}
    for name, (f, _, diff) in CASES.items():
        for train in ((False,) if a.inference else (False, True) if diff else (False,)):
            _STATE["calls"].clear()
            ms, fb = timed(f, inputs[name], train, a.reps, a.inference)
            rewrites = sum(_STATE["calls"].values())
            opt_in = name.startswith(("sort", "topk", "argsort")) and os.environ.get("HPU_OPS_EXACT_SORT") != "1"
            if on and not rewrites and not opt_in:
                raise RuntimeError(f"{name}: the rewrite did not run in the 'on' arm")
            print("BENCH " + json.dumps({"case": name + (" [inference]" if a.inference else ""), "train": train,
                                         "fix": on, "round": a.round, "ms": round(ms, 4),
                                         "fallbacks_per_call": fb, "lazy": LAZY, "rewrites": rewrites}), flush=True)
    del undo


if __name__ == "__main__":
    main()
