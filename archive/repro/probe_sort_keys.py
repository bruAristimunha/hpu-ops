"""Gaudi2: sort / topk without NaN (inf, ties, -0.0), and int32 / float32 sort and topk speed, for a NaN-exact sort."""
import os
import time

import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.hpu as hthpu
import torch

LAZY = os.environ.get("PT_HPU_LAZY_MODE", "1") != "0"
INF = float("inf")


def sync():
    if LAZY:
        htcore.mark_step()
    hthpu.synchronize()


def check(name, f, x):
    got = f(x.to("hpu"))
    sync()
    got = [g.cpu() for g in got]
    want = f(x)
    vals = torch.equal(got[0], want[0])
    picked = torch.equal(x.gather(-1, got[1].clamp(0, x.shape[-1] - 1)), got[0])
    in_range = bool(((got[1] >= 0) & (got[1] < x.shape[-1])).all())
    print(f"CHECK {name:28s} values={vals} indices_in_range={in_range} indices_pick_values={picked}", flush=True)


for dt in (torch.float32, torch.bfloat16, torch.int32):
    for n in (8, 1000, 5000):
        g = torch.Generator().manual_seed(0)
        x = torch.randn(64, n, generator=g)
        if dt.is_floating_point:
            x[:, ::7] = INF
            x[:, 3::11] = -INF
            x[:, 5::13] = 0.5
            x[:, 6::17] = -0.0
        x = (x * 1000).to(dt) if not dt.is_floating_point else x.to(dt)
        check(f"sort {dt} n={n}", lambda t: tuple(t.sort(-1)), x)
        check(f"sort desc {dt} n={n}", lambda t: tuple(t.sort(-1, descending=True)), x)
        check(f"sort stable {dt} n={n}", lambda t: tuple(t.sort(dim=-1, stable=True)), x)
        check(f"topk5 {dt} n={n}", lambda t: tuple(t.topk(5, -1)), x)
        check(f"topk5 small {dt} n={n}", lambda t: tuple(t.topk(5, -1, largest=False)), x)


def bench(name, f, x, reps=30):
    xs = x.to("hpu")
    for _ in range(5):
        out = f(xs)
    sync()
    keep = []
    t = time.perf_counter()
    for _ in range(reps):
        keep.append(f(xs))
    sync()
    print(f"TIME {name:32s} {1e3 * (time.perf_counter() - t) / reps:.3f} ms", flush=True)
    del out


for shape in ((64, 64), (256, 1000), (32, 22, 1000)):
    x = torch.randn(*shape)
    for dt in (torch.float32, torch.int32):
        xx = x.to(dt) if dt.is_floating_point else (x * 1e6).to(dt)
        bench(f"sort {dt} {shape}", lambda t: t.sort(-1), xx)
        bench(f"topk8 {dt} {shape}", lambda t: t.topk(8, -1), xx)
    bench(f"view-int key {shape}", lambda t: torch.where(torch.isnan(t), 2 ** 31 - 1, (t + 0.0).view(torch.int32) ^ (
        ((t + 0.0).view(torch.int32) >> 31) & 0x7FFFFFFF)), x)
print("DONE", flush=True)
