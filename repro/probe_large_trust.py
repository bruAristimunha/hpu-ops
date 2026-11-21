"""Gaudi2 eager: the in-place kernel route past n = 64 (solve, inv) on Codex review #3 / #18's inputs against torch on
the CPU, and its per-call time (mean of 20 synced calls after warm-up, outputs kept) for ordinary inputs; run it with
this branch's hpu_ops or origin/main's first on PYTHONPATH (the A/B is by process).
    PT_HPU_LAZY_MODE=0 python3 repro/probe_large_trust.py
CHECK lines: the result's largest difference from torch's float32 CPU result and from float64; TIME lines: ms per call
and the CPU fallback ops and hpu-ops' redone matrices (``_redo_on_cpu`` calls) in the timed calls."""
import json
import time

import hpu_ops
import habana_frameworks.torch.hpu as hthpu
import torch
from habana_frameworks.torch.hpu.metrics import metric_global

hpu_ops.enable()
from hpu_ops import linalg  # noqa: E402

REDONE = []
if hasattr(linalg, "_redo_on_cpu"):
    real = linalg._redo_on_cpu

    def spy(bad, *a, **k):
        REDONE.append(int(bad.sum()))
        return real(bad, *a, **k)

    linalg._redo_on_cpu = spy


def fb():
    return dict(dict(metric_global("cpu_fallback").stats()).get("FallbackOps", {}) or {})


def codex18(s=1e7):
    A = torch.eye(65)
    A[:2, :2] = 0.25 * torch.tensor([[7.0, s], [0.0, 0.9 * s]])
    return A


def check(name, f, *xs):
    got = f(*[x.to("hpu") for x in xs])
    hthpu.synchronize()
    got = got.cpu()
    want32, want64 = f(*xs), f(*[x.double() for x in xs])
    print("CHECK " + json.dumps(dict(case=name, path=hpu_ops.__file__.split("/")[-3],
                                     diff_cpu32=float((got - want32).abs().max()),
                                     diff_f64=float((got.double() - want64).abs().max()),
                                     finite=bool(torch.isfinite(got).all()), redone=list(REDONE))), flush=True)
    REDONE.clear()


big = torch.eye(65) * 2.0 ** 127
check("codex3_solve_2^127I", torch.linalg.solve, big, big[:, :1].clone())
check("codex3_inv_2^127I", torch.linalg.inv, big)
A18 = codex18()
check("codex18_solve", torch.linalg.solve, A18, A18[:, 1:2].clone())
for s in (1e4, 1e5, 1e6):
    A = codex18(s)
    check(f"codex18_s={s:.0e}", torch.linalg.solve, A, A[:, 1:2].clone())
g = torch.Generator().manual_seed(0)
batch = torch.stack([torch.randn(65, 65, generator=g) + 16 * torch.eye(65), codex18(), big])
check("batch_good_codex18_codex3", torch.linalg.solve, batch, torch.randn(3, 65, 2, generator=g))

for n, b in ((65, 1), (65, 32), (128, 32), (256, 32), (256, 1)):
    for kind in ("diag_dominant", "gaussian"):
        A = torch.randn(b, n, n, generator=g) + (2 * n ** 0.5 * torch.eye(n) if kind == "diag_dominant" else 0)
        B = torch.randn(b, n, 3, generator=g)
        Ah, Bh = A.to("hpu"), B.to("hpu")
        for op, f in (("solve", lambda: torch.linalg.solve(Ah, Bh)), ("inv", lambda: torch.linalg.inv(Ah))):
            for _ in range(3):
                f()
            hthpu.synchronize()
            REDONE.clear()
            f0 = fb()
            keep = []
            t = time.perf_counter()
            for _ in range(20):
                keep.append(f())
            hthpu.synchronize()
            ms = (time.perf_counter() - t) / 20 * 1e3
            f1 = fb()
            print("TIME " + json.dumps(dict(op=op, n=n, batch=b, kind=kind, path=hpu_ops.__file__.split("/")[-3],
                                            ms=round(ms, 3), redone_per_call=sum(REDONE) / 20,
                                            fallback={k: v - f0.get(k, 0) for k, v in f1.items()
                                                      if v - f0.get(k, 0)})), flush=True)
print("DONE", flush=True)
