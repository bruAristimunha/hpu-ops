"""Gaudi2: per-call time of linalg.svd / svdvals through hpu-ops' Jacobi kernel (batch 32, n = 8, 22, 32), mean of
20 synced calls after warm-up, outputs kept; run with this branch's or origin/main's hpu_ops first on PYTHONPATH.
    PT_HPU_LAZY_MODE=0|1 python3 repro/probe_svd_time.py"""
import json
import os
import time

import hpu_ops
import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.hpu as hthpu
import torch

hpu_ops.enable()
LAZY = os.environ.get("PT_HPU_LAZY_MODE", "1") != "0"


def sync():
    if LAZY:
        htcore.mark_step()
    hthpu.synchronize()


g = torch.Generator().manual_seed(0)
for n in (8, 22, 32):
    A = torch.randn(32, n, n, generator=g).to("hpu")
    for name, f in (("svd", lambda: torch.linalg.svd(A, full_matrices=False)), ("svdvals", lambda: torch.linalg.svdvals(A))):
        for _ in range(3):
            f()
        sync()
        keep = []
        t = time.perf_counter()
        for _ in range(20):
            keep.append(f())
            if LAZY:
                htcore.mark_step()
        sync()
        print("TIME " + json.dumps(dict(op=name, n=n, lazy=LAZY, path=hpu_ops.__file__.split("/")[-3],
                                        ms=round((time.perf_counter() - t) / 20 * 1e3, 3))), flush=True)
