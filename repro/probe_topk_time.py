"""Gaudi2: per-call time of topk / kthvalue on wide rows (logits-like 32x50257, 256x1000, 32x22): Gaudi's own
(``native``) against hpu-ops' exact form, interleaved rounds, medians; outputs kept, synced.
    PT_HPU_LAZY_MODE=0|1 python3 repro/probe_topk_time.py"""
import json
import os
import statistics
import time

import hpu_ops
import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.hpu as hthpu
import torch
from hpu_ops import extrema

hpu_ops.enable()
LAZY = os.environ.get("PT_HPU_LAZY_MODE", "1") != "0"
A = torch.ops.aten


def sync():
    if LAZY:
        htcore.mark_step()
    hthpu.synchronize()


def bench(f, reps=20):
    for _ in range(3):
        f()
    sync()
    keep = []
    t = time.perf_counter()
    for _ in range(reps):
        keep.append(f())
        if LAZY:
            htcore.mark_step()
    sync()
    return (time.perf_counter() - t) / reps * 1e3


g = torch.Generator().manual_seed(0)
for shape, k in (((32, 50257), 5), ((32, 50257), 50), ((256, 1000), 5), ((32, 22), 3)):
    x = torch.randn(*shape, generator=g).to("hpu")
    forms = {
        "topk_native": lambda: extrema.native(A.topk.default)(x, k, -1, True, True),
        "topk_exact": lambda: x.topk(k, -1),
        "kth_native": lambda: extrema.native(A.kthvalue.default)(x, k, -1, False),
        "kth_exact": lambda: x.kthvalue(k, -1),
        "sort_native": lambda: extrema.internal_sort(x, -1),
    }
    times = {n: [] for n in forms}
    for _ in range(5):
        for n, f in forms.items():
            times[n].append(bench(f))
    print("TIME " + json.dumps(dict(shape=shape, k=k, lazy=LAZY, **{n: round(statistics.median(v), 3)
                                                                    for n, v in times.items()})), flush=True)
