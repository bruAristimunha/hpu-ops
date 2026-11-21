"""On a Gaudi card: per-call time of the TPC linalg kernels against Habana's own path (its CPU fallback), in one
process.
    PT_HPU_LAZY_MODE=0|1 python3 tests/linalg_percall.py [--n 22] [--batch 32] [--rhs 3] [--iters 30] [--rounds 3]
Ops: ``linalg.solve`` (``--rhs`` right-hand sides), ``inv``, ``det``, ``slogdet`` and ``eigh`` (``--no-eigh``: not
timed), float32. ``--rounds``
rounds time every op with the kernels (``hpu_ops.enable()``); then the fixes are undone and as many rounds time the
same calls through Habana's path. (Not interleaved: ``enable()`` again after the device is up leaves the TPC kernels
off, so alternate processes to see drift.) Time = mean ms of ``--iters`` synced calls after one warm-up call,
outputs kept alive. Per phase the bridge ``cpu_fallback`` delta and any "runs on the CPU" warning are recorded. One
``RESULT {json}`` line per op and phase with the per-round means and their mean."""
import argparse
import json
import os
import time
import warnings

import hpu_ops

hpu_ops.enable()  # before the first HPU op: Habana reads the kernel path when the device initialises

import habana_frameworks.torch.core as htcore  # noqa: E402
import habana_frameworks.torch.hpu as hthpu  # noqa: E402
import torch  # noqa: E402
from habana_frameworks.torch.hpu.metrics import metric_global  # noqa: E402

from hpu_ops import tpc  # noqa: E402

LAZY = os.environ.get("PT_HPU_LAZY_MODE", "1") != "0"
BATCH = 32


def sync():
    if LAZY:
        htcore.mark_step()
    hthpu.synchronize()


def fb_ops():
    return dict(dict(metric_global("cpu_fallback").stats()).get("FallbackOps", {}) or {})


def ops(n, rhs=3, eigh=True):
    gen = torch.Generator().manual_seed(0)
    A = (torch.randn(BATCH, n, n, generator=gen) + 2 * n ** 0.5 * torch.eye(n)).to("hpu")
    S = torch.randn(BATCH, n, n, generator=gen)
    S = (S + S.mT).to("hpu")
    B = torch.randn(BATCH, n, rhs, generator=gen).to("hpu")
    L = torch.linalg
    calls = {"solve": lambda: L.solve(A, B), "inv": lambda: L.inv(A), "det": lambda: L.det(A),
             "slogdet": lambda: L.slogdet(A), "eigh": lambda: L.eigh(S)}
    if not eigh:
        del calls["eigh"]
    return calls


def time_op(fn, iters):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        f0 = fb_ops()
        fn()
        sync()
        keep = []
        t = time.perf_counter()
        for _ in range(iters):
            keep.append(fn())
        sync()
        ms = (time.perf_counter() - t) * 1e3 / iters
        f1 = fb_ops()
    fb = {k: f1[k] - f0.get(k, 0) for k in f1 if f1[k] - f0.get(k, 0)}
    cpu = sum("runs on the CPU" in str(w.message) for w in caught)
    return ms, fb, cpu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=22)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--rhs", type=int, default=3)
    ap.add_argument("--no-eigh", action="store_true")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--rounds", type=int, default=3)
    args = ap.parse_args()
    global BATCH
    BATCH = args.batch
    print("PLUGINS hpu_ops", hpu_ops.__file__, "tpc_ready", tpc.ready(), "lazy", LAZY, "MAXN", tpc.MAXN, flush=True)
    calls = ops(args.n, args.rhs, not args.no_eigh)
    res = {}
    for phase in ("kernel", "habana"):
        if phase == "habana":
            hpu_ops._STATE["undo"]()  # one way: enable() after the device is up leaves the TPC kernels off
        for _ in range(args.rounds):
            for name, fn in calls.items():
                ms, fb, cpu = time_op(fn, args.iters)
                r = res.setdefault((name, phase), dict(ms=[], fb={}, cpu=0))
                r["ms"].append(round(ms, 3))
                r["cpu"] += cpu
                for k, v in fb.items():
                    r["fb"][k] = r["fb"].get(k, 0) + v
    for (name, phase), r in res.items():
        print("RESULT " + json.dumps(dict(op=name, phase=phase, n=args.n, batch=BATCH, rhs=args.rhs, lazy=LAZY,
                                          bridge="1.21.4",
                                          iters=args.iters, ms_rounds=r["ms"],
                                          ms_mean=round(sum(r["ms"]) / len(r["ms"]), 3),
                                          fallback=r["fb"], cpu_warnings=r["cpu"])), flush=True)


if __name__ == "__main__":
    main()
