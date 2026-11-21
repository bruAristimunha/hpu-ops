"""Counting integer bin positions on Gaudi2: histc chunks vs index_add_ vs scatter_add_ (exactness and cost).
    PT_HPU_LAZY_MODE=0|1 python repro/bench_histc_count.py"""
import os
import time

import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.hpu as hthpu
import torch

from hpu_ops import small

LAZY = os.environ.get("PT_HPU_LAZY_MODE", "1") != "0"
hpu = torch.device("hpu")


def sync():
    if LAZY:
        htcore.mark_step()
    hthpu.synchronize()


def index_add(pos, bins):
    ones = torch.ones_like(pos, dtype=torch.float32)
    slot = torch.where(pos < 0, bins, pos)
    return torch.zeros(bins + 1, device=pos.device).index_add_(0, slot, ones)[:bins]


def scatter_add(pos, bins):
    ones = torch.ones_like(pos, dtype=torch.float32)
    slot = torch.where(pos < 0, bins, pos)
    return torch.zeros(bins + 1, device=pos.device).scatter_add_(0, slot, ones)[:bins]


def chunks(pos, bins):
    return small.count_bins(pos.float(), bins)


def timed(fn):
    for _ in range(3):
        out = fn()
    sync()
    t0 = time.perf_counter()
    outs = [fn() for _ in range(20)]
    sync()
    return (time.perf_counter() - t0) / 20 * 1e3, out


for bins in (64, 256, 1000, 4096):
    n = 32 * 22 * 1000
    pos_cpu = torch.randint(-1, bins, (n,))
    ref = torch.bincount(pos_cpu[pos_cpu >= 0], minlength=bins).float()
    pos = pos_cpu.to(hpu)
    thr = torch.sort(torch.randn(bins - 1)).values.to(hpu)
    x = torch.randn(n).to(hpu)
    sync()
    line = [f"bins={bins}"]
    for name, fn in (("chunks", lambda: chunks(pos, bins)), ("index_add", lambda: index_add(pos, bins)),
                     ("scatter_add", lambda: scatter_add(pos, bins)),
                     ("bucketize", lambda: torch.bucketize(x, thr, right=True))):
        try:
            ms, out = timed(fn)
            exact = "" if name == "bucketize" else (" exact" if torch.equal(out.cpu(), ref) else " WRONG")
            line.append(f"{name} {ms:.3f} ms{exact}")
        except Exception as e:  # noqa: BLE001
            line.append(f"{name} RAISE {str(e).splitlines()[0][:60]}")
    print(("lazy " if LAZY else "eager ") + " | ".join(line), flush=True)
