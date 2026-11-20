"""On a Gaudi card: the TPC substitution kernel behind ``linalg.solve_triangular`` and ``cholesky_solve`` against
Habana's own path (its CPU fallback), in the same process.
    PT_HPU_LAZY_MODE=0|1 python3 tests/tri_solve_probe.py [--iters 30]
Per op, variant and n (batch 32, float32): the relative error ``max|x - ref| / max|ref|`` against float64 on the CPU,
next to the same error of CPU float32 (the bar), well- and ill-conditioned (diagonal logspace(-3, 0): condition about
1e3 per factor, so 1e6 for cholesky_solve), the bridge ``cpu_fallback`` delta and "runs on the CPU" warnings, and
the mean ms of ``--iters`` synced calls with the outputs kept alive. The kernel runs first (``hpu_ops.enable()``),
then the fixes are undone and the same calls time Habana's fallback. One ``RESULT {json}`` line per row; the last line
counts the failed rows (kernel error above 10x the CPU float32 bar, or any fallback/CPU warning on the kernel path)."""
import argparse
import json
import math
import os
import sys
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
HPU = torch.device("hpu")
BATCH = 32
FACTOR = 10
L = torch.linalg


def sync():
    if LAZY:
        htcore.mark_step()
    hthpu.synchronize()


def fb_ops():
    return dict(dict(metric_global("cpu_fallback").stats()).get("FallbackOps", {}) or {})


def rel_err(got, ref):
    got, ref = got.detach().cpu().double(), ref.double()
    e = float((got - ref).abs().max()) / max(float(ref.abs().max()), 1e-300)
    return math.inf if math.isnan(e) else e


def triangle(n, upper, unit, ill, gen):
    off = torch.randn(BATCH, n, n, generator=gen) / n
    off = off.triu(1) if upper else off.tril(-1)
    diag = torch.logspace(-3, 0, n) if ill else 1.0 + torch.rand(BATCH, n, generator=gen)
    diag = diag[torch.randperm(n, generator=gen)] if ill else diag
    return off + (torch.zeros(BATCH, n, n) if unit else torch.diag_embed(diag.expand(BATCH, n)))


def variants(n, gen):
    """``(name, fn, cpu_inputs)``: every kernel variant, both sides, and cholesky_solve lower and upper."""
    out = []
    for ill in (False, True):
        for upper in (False, True):
            for unit in (False, True):
                if unit and ill:
                    continue
                for left in (True, False):
                    A = triangle(n, upper, unit, ill, gen)
                    B = torch.randn(BATCH, n, 3, generator=gen) if left else torch.randn(BATCH, 3, n, generator=gen)
                    fn = (lambda a, b, u=upper, lf=left, un=unit: L.solve_triangular(a, b, upper=u, left=lf,
                                                                                    unitriangular=un))
                    out.append((f"solve_triangular.u{int(upper)}.left{int(left)}.unit{int(unit)}.ill{int(ill)}",
                                fn, (A, B)))
        for upper in (False, True):
            C = triangle(n, upper, False, ill, gen)
            B = torch.randn(BATCH, n, 3, generator=gen)
            out.append((f"cholesky_solve.u{int(upper)}.ill{int(ill)}",
                        lambda b, c, u=upper: torch.cholesky_solve(b, c, upper=u), (B, C)))
    return out


def run(fn, h_in, iters):
    """First call (checked) and the mean ms of ``iters`` synced calls, outputs kept alive."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        f0 = fb_ops()
        out = fn(*h_in)
        sync()
        first = out.cpu()
        keep = []
        t = time.perf_counter()
        for _ in range(iters):
            keep.append(fn(*h_in))
        sync()
        ms = (time.perf_counter() - t) * 1e3 / iters
        f1 = fb_ops()
    fb = {k: f1[k] - f0.get(k, 0) for k in f1 if f1[k] - f0.get(k, 0)}
    cpu = [str(w.message)[:120] for w in caught if "runs on the CPU" in str(w.message)]
    return first, ms, fb, cpu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--sizes", default="1,3,16,22,40,64")
    args = ap.parse_args()
    sizes = [int(s) for s in args.sizes.split(",")]
    print("PLUGINS hpu_ops", hpu_ops.__file__, "tpc_ready", tpc.ready(), "lazy", LAZY, "MAXN", tpc.MAXN, flush=True)
    gen = torch.Generator().manual_seed(0)
    cases = [(n, *v) for n in sizes for v in variants(n, gen)]
    rows = {}
    for phase in ("kernel", "habana"):
        if phase == "habana":
            hpu_ops._STATE["undo"]()  # the same calls through Habana's own path
        for n, name, fn, cpu_in in cases:
            ref = fn(*(x.double() for x in cpu_in))
            bar = rel_err(fn(*cpu_in), ref)
            h_in = [x.to(HPU) for x in cpu_in]
            sync()
            try:
                first, ms, fb, cpu = run(fn, h_in, args.iters)
                err = rel_err(first, ref)
            except Exception as e:  # noqa: BLE001  (a failing case is a row)
                ms, fb, cpu, err = None, {}, [f"exception {e!r}"[:300]], math.inf
            rows.setdefault((n, name), {})[phase] = dict(err=err, ms=ms, fb=fb, cpu=cpu, bar=bar)
    failed = 0
    for (n, name), r in rows.items():
        k, h = r["kernel"], r["habana"]
        bar = max(1e-6, FACTOR * k["bar"])
        problems = []
        problems += [f"err {k['err']:.3g} > {bar:.3g}"] * (not k["err"] <= bar)
        problems += [f"fell back {k['fb']}"] * bool(k["fb"]) + k["cpu"]
        failed += bool(problems)
        speed = h["ms"] / k["ms"] if k["ms"] and h["ms"] else None
        print("RESULT " + json.dumps(dict(op=name, n=n, batch=BATCH, lazy=LAZY, bridge="1.21.4",
                                          err=k["err"], err_cpu32=k["bar"], err_habana=h["err"],
                                          ms_kernel=k["ms"], ms_habana=h["ms"], speedup=speed,
                                          fb_kernel=k["fb"], fb_habana=h["fb"], status="fail" if problems else "ok",
                                          problems=problems)), flush=True)
    print(f"SUMMARY {len(rows)} rows, {failed} failed", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
