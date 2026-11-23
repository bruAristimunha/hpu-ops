"""Standalone repro (bridge 1.21.4, Gaudi2, no hpu_ops): Habana's accumulating ``index_put`` on dim 0.

1. Lazy mode, wrong sums: ``d.index_put((i + 0,), s, accumulate=True)`` into more than 512 rows returns wrong sums
   (max |hpu - cpu| ~ 10..100) when the index is the output of an ``add`` in the same graph (shown here at 40960
   source rows of 22 columns). The same index straight from a graph input, or made by ``where``, ``clamp``,
   ``remainder`` or ``mul``, gives the right sums; so does every index at 512 rows or fewer.
   Habana's ``index_add`` is right for every index.
2. Eager and lazy, device stop: an index of -1 stops the device ("Compute or dma timeout", exit 137), where
   ``index_add`` wraps it to the last row.

    PT_HPU_LAZY_MODE=1 python3 index_put_lazy_add_index.py         # 1: prints a table, exits 0
    PT_HPU_LAZY_MODE=1 python3 index_put_lazy_add_index.py crash   # 2: the process is killed (exit 137)
"""
import os
import sys

import habana_frameworks.torch.core as htcore
import torch

LAZY = os.environ.get("PT_HPU_LAZY_MODE", "1") != "0"
HPU = torch.device("hpu")


def sync():
    if LAZY:
        htcore.mark_step()
    torch.hpu.synchronize()


def wrong_sums():
    torch.manual_seed(0)
    print(f"torch {torch.__version__} lazy={LAZY}: max |hpu - cpu| of d.index_put((index,), s, accumulate=True)")
    for rows in (512, 513, 1024, 4096):
        d, i, s = torch.randn(rows, 22), torch.randint(0, rows, (40960,)), torch.randn(40960, 22)
        ref = d.index_add(0, i, s)
        dh, ih, sh = d.to(HPU), i.to(HPU), s.to(HPU)
        out = {}
        with torch.no_grad():
            for name, index in (("input", lambda: ih), ("i + 0", lambda: ih + 0), ("i * 1", lambda: ih * 1),
                                ("where", lambda: torch.where(ih < 0, rows, ih))):
                o = dh.index_put((index(),), sh, accumulate=True)
                sync()
                out[name] = (o.cpu() - ref).abs().max().item()
            o = dh.index_add(0, ih + 0, sh)
            sync()
            out["index_add(i + 0)"] = (o.cpu() - ref).abs().max().item()
        print(f"rows={rows:5d} " + "  ".join(f"{k}: {v:.1e}" for k, v in out.items()), flush=True)


def device_stop():
    d, s = torch.zeros(16, 8, device=HPU), torch.ones(40960, 8, device=HPU)
    i = torch.zeros(40960, dtype=torch.long)
    i[0] = -1
    print("index_add:", d.index_add(0, i.to(HPU), s).cpu()[15, 0].item(), "(-1 wraps to row 15)", flush=True)
    print("index_put with -1 next: the process is killed", flush=True)
    o = d.index_put((i.to(HPU),), s, accumulate=True)
    sync()
    print(o.cpu()[15, 0].item())


if __name__ == "__main__":
    device_stop() if sys.argv[1:] == ["crash"] else wrong_sums()
