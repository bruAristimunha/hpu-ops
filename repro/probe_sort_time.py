"""Gaudi2: per-call time of the candidate sort forms (native ops, hpu_ops not imported), 7 interleaved rounds of 30
synced calls with the outputs kept, medians; then the forms compiled (``--compile``: correctness only).
    PT_HPU_LAZY_MODE=0|1 python3 repro/probe_sort_time.py [--compile]
native / native_st: Gaudi's sort, non-stable / stable (wrong with NaN, non-stable wrong with +-inf)
key_un / key_st:    one int32 sort (non-stable / stable) of the total-order key, values gathered: the exact form
key_topk:           key_un narrowed to k (topk's exact form); native_topk Gaudi's topk
int_ntn:            nan_to_num(nan=FLT_MAX), Gaudi's non-stable sort, values gathered from x (an internal candidate)"""
import os
import statistics
import sys
import time

import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.hpu as hthpu
import torch

LAZY = os.environ.get("PT_HPU_LAZY_MODE", "1") != "0"
INF, NAN = float("inf"), float("nan")
I32_MAX = 2 ** 31 - 1
NAN_KEY = 0x7F800001
FMAX = torch.finfo(torch.float32).max


def sync():
    if LAZY:
        htcore.mark_step()
    hthpu.synchronize()


def key(x):
    f = x if x.dtype == torch.float32 else x.to(torch.float32)
    b = f.view(torch.int32)
    s = b >> 31
    return torch.where(torch.isnan(f), NAN_KEY, (b ^ (s & I32_MAX)) - s)


def key_sort(stable):
    def f(x, desc=False):
        _, i = torch.sort(key(x), stable=stable, dim=-1, descending=desc)
        return x.gather(-1, i), i
    return f


def key_topk(x, desc=True, k=5):
    _, i = torch.sort(key(x), dim=-1, descending=desc)
    i = i.narrow(-1, 0, k)
    return x.gather(-1, i), i


def int_ntn(x, desc=False):
    _, i = torch.sort(torch.nan_to_num(x, nan=FMAX), dim=-1, descending=desc)
    return x.gather(-1, i), i


FORMS = {
    "native": lambda x, desc=False: tuple(torch.sort(x, -1, desc)),
    "native_st": lambda x, desc=False: tuple(torch.sort(x, stable=True, dim=-1, descending=desc)),
    "key_un": key_sort(False), "key_st": key_sort(True),
    "native_topk": lambda x, desc=True: tuple(torch.topk(x, 5, -1)), "key_topk": key_topk,
    "int_ntn": int_ntn,
}


def bench(f, x, reps=30):
    for _ in range(3):
        out = f(x)
    sync()
    keep = []
    t = time.perf_counter()
    for _ in range(reps):
        keep.append(f(x))
        if LAZY:
            htcore.mark_step()
    sync()
    del out
    return 1e3 * (time.perf_counter() - t) / reps


print(f"MODE {'lazy' if LAZY else 'eager'}", flush=True)
if "--compile" in sys.argv:
    torch._dynamo.config.cache_size_limit = 256
    for dt in (torch.float32, torch.bfloat16):
        for shape, d in (((8, 3), -1), ((8, 22), -1), ((22, 8), 0), ((16, 1000), -1)):
            g = torch.Generator().manual_seed(1)
            x = torch.randn(*shape, generator=g)
            fl = x.movedim(d, -1).reshape(-1, x.shape[d])
            fl[:, ::7], fl[:, 1::5], fl[:, 3::11], fl[:, 6::4] = INF, NAN, -INF, -0.0
            x = fl.reshape(x.movedim(d, -1).shape).movedim(-1, d).to(dt).contiguous()
            for desc in (False, True):
                want = torch.sort(x, dim=d, descending=desc, stable=True)
                for stable in (False, True):
                    def f(t, desc=desc, stable=stable, d=d):
                        _, i = torch.sort(key(t), stable=stable, dim=d, descending=desc)
                        return t.gather(d, i), i
                    try:
                        got = torch.compile(f, backend="hpu_backend", dynamic=False)(x.to("hpu"))
                        sync()
                        gv, gi = got[0].cpu(), got[1].cpu()
                        ok_v = torch.equal(gv.nan_to_num(7.0), want.values.nan_to_num(7.0))
                        ok_i = torch.equal(gi, want.indices) if stable else torch.equal(
                            x.gather(d, gi.clamp(0, x.shape[d] - 1)).nan_to_num(7.0), want.values.nan_to_num(7.0))
                        res = f"values={ok_v} indices={ok_i}"
                    except Exception as e:  # noqa: BLE001
                        res = f"ERROR {type(e).__name__}: {str(e)[:150]}"
                    print(f"COMPILE {str(dt):15s} {str(shape):10s} d={d} desc={desc} stable={stable} {res}",
                          flush=True)
    sys.exit(0)

# int_ntn correctness on eigenvalue-like rows with NaN: indices in range, NaN in the values
g = torch.Generator().manual_seed(0)
for n in (3, 22, 32):
    x = torch.randn(8, n, generator=g)
    x[2, 1] = NAN
    x[5] = NAN
    v, i = int_ntn(x.to("hpu"))
    sync()
    v, i = v.cpu(), i.cpu()
    print(f"INTERNAL n={n} in_range={bool(((i >= 0) & (i < n)).all())} nan_rows_nan={bool(v[2].isnan().any())},"
          f"{bool(v[5].isnan().all())} clean_rows_exact={torch.equal(v[0], x[0].sort().values)}", flush=True)

for shape in ((32, 22), (32, 64), (32, 22, 1000), (256, 1000), (4, 65536)):
    g = torch.Generator().manual_seed(0)
    x = torch.randn(*shape, generator=g)
    h = x.to("hpu")
    hb = h.to(torch.bfloat16)
    times = {k: [] for k in FORMS}
    times["key_un_bf16"], times["native_bf16"] = [], []
    for _ in range(7):
        for name, f in FORMS.items():
            times[name].append(bench(f, h))
        times["key_un_bf16"].append(bench(FORMS["key_un"], hb))
        times["native_bf16"].append(bench(FORMS["native"], hb))
    print(f"TIME {str(shape):14s} " + " ".join(f"{k}={statistics.median(v):.3f}" for k, v in times.items()),
          flush=True)
print("DONE", flush=True)
