"""Gaudi2 eager: where the large route's trust check costs (solve n = 65 and 256, batch 1 and 32, ordinary inputs):
the check off, its ops without the host read, and the full check; 20 synced calls after warm-up, 5 alternating
rounds, medians. Also the same with one synchronize per call (the latency the host read imposes).
    PT_HPU_LAZY_MODE=0 python3 repro/probe_large_trust_cost.py"""
import statistics
import time

import hpu_ops
import habana_frameworks.torch.hpu as hthpu
import torch

hpu_ops.enable()
from hpu_ops import linalg  # noqa: E402

real_untrusted, real_any = linalg._large_untrusted, None


class NoRead(torch.Tensor):
    pass


def variant(name):
    if name == "off":
        linalg._large_untrusted = lambda A, Ainv: torch.zeros((), dtype=torch.bool)  # host-side False
    elif name == "ops_only":
        def f(A, Ainv):
            real_untrusted(A, Ainv)  # the ops, result dropped
            return torch.zeros((), dtype=torch.bool)
        linalg._large_untrusted = f
    else:
        linalg._large_untrusted = real_untrusted


def bench(f, sync_each):
    for _ in range(3):
        f()
    hthpu.synchronize()
    keep = []
    t = time.perf_counter()
    for _ in range(20):
        keep.append(f())
        if sync_each:
            hthpu.synchronize()
    hthpu.synchronize()
    return (time.perf_counter() - t) / 20 * 1e3


g = torch.Generator().manual_seed(0)
for n, b in ((65, 1), (65, 32), (256, 32)):
    A = (torch.randn(b, n, n, generator=g) + 2 * n ** 0.5 * torch.eye(n)).to("hpu")
    B = torch.randn(b, n, 3, generator=g).to("hpu")
    times = {}
    for _ in range(5):
        for name in ("off", "ops_only", "full"):
            for sync_each in (False, True):
                variant(name)
                times.setdefault((name, sync_each), []).append(bench(lambda: torch.linalg.solve(A, B), sync_each))
    variant("full")
    print(f"COST n={n} b={b} " + " ".join(f"{k[0]}{'_sync' if k[1] else ''}={statistics.median(v):.2f}"
                                          for k, v in times.items()), flush=True)
print("DONE", flush=True)
