"""Gaudi2 eager, one harness: native sort vs the int32-key exact form vs the two-finite-key exact form
(``extrema._sorted``), 9 interleaved rounds of 30 calls, median / min ms, shapes 32x22, 32x64, 32x22x1000."""
import statistics
import time

import hpu_ops
import habana_frameworks.torch.hpu as hthpu
import torch
from hpu_ops import extrema

hpu_ops.enable()
I32_MAX, NAN_KEY = 2 ** 31 - 1, 0x7F800001
sort = extrema.native(torch.ops.aten.sort.stable)


def intkey(x):
    b = x.view(torch.int32)
    s = b >> 31
    k = torch.where(torch.isnan(x), NAN_KEY, (b ^ (s & I32_MAX)) - s)
    _, i = sort(k, stable=False, dim=-1, descending=False)
    return x.gather(-1, i), i


def two_where(x):  # the two-key form with where/isnan in place of nan_to_num
    n = x.shape[-1]
    nan = torch.isnan(x)
    key = torch.where(nan, 3.0e38, x).clamp(-3.0e38, 3.0e38)
    _, i = sort(key, stable=False, dim=-1, descending=False)
    p = torch.empty(x.shape, dtype=torch.float32, device=x.device).scatter_(
        -1, i, extrema._positions(n, x.device, x).expand(x.shape))
    shift = torch.where(nan, 2.0 * n, torch.where(x == key, 0.0, (x - key).sign() * n))
    _, i = sort(p + shift, stable=False, dim=-1)
    return x.gather(-1, i), i


FORMS = {"native": lambda x: extrema.internal_sort(x, -1), "intkey": intkey,
         "two_key": lambda x: extrema._sorted(x, -1, False), "two_where": two_where}
for shape in ((32, 22), (32, 64), (32, 22, 1000)):
    x = torch.randn(*shape).to("hpu")
    times = {k: [] for k in FORMS}
    for _ in range(9):
        for name, f in FORMS.items():
            for _ in range(3):
                f(x)
            hthpu.synchronize()
            keep = []
            t = time.perf_counter()
            for _ in range(30):
                keep.append(f(x))
            hthpu.synchronize()
            times[name].append((time.perf_counter() - t) / 30 * 1e3)
    print(f"FORMS {str(shape):14s} " + " ".join(f"{k}={statistics.median(v):.3f}/{min(v):.3f}"
                                                 for k, v in times.items()), flush=True)
