"""Gaudi2: adversarial cases of the exact sort family: float64 (held as float32), float16, non-contiguous and offset
views, a dim of size 1, n = 2, all-NaN / all-inf rows, -0.0 against 0.0 (stable order), two lazy runs in a row, and
inference mode (not rewritten: Gaudi's sort, recorded). Values and indices against torch on the CPU.
    PT_HPU_LAZY_MODE=0|1 python3 repro/probe_exact_sort_adv.py"""
import json
import os

import hpu_ops
import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.hpu as hthpu
import torch

hpu_ops.enable()
LAZY = os.environ.get("PT_HPU_LAZY_MODE", "1") != "0"
INF, NAN = float("inf"), float("nan")


def sync():
    if LAZY:
        htcore.mark_step()
    hthpu.synchronize()


def eq(a, b):
    return a.shape == b.shape and torch.equal(a.isnan(), b.isnan()) and torch.equal(a.nan_to_num(7.0), b.nan_to_num(7.0))


def check(name, x, f, exact_index=True, inference=False):
    want = f(x)
    try:
        h = x.to("hpu")
        if inference:
            with torch.inference_mode():
                got = f(h)
        else:
            got = f(h)
        sync()
        gv, gi = got[0].cpu(), got[1].cpu()
        if exact_index:
            ok = eq(gv, want[0]) and torch.equal(gi, want[1])
        else:
            d = -1
            inr = bool(((gi >= 0) & (gi < x.shape[d])).all())
            ok = eq(gv, want[0]) and inr and eq(x.gather(d, gi.clamp(0, x.shape[d] - 1)), want[0])
    except Exception as e:  # noqa: BLE001
        ok = f"ERROR {type(e).__name__}: {str(e)[:100]}"
    print("ADV " + json.dumps(dict(case=name, lazy=LAZY, ok=ok)), flush=True)
    return ok is True


g = torch.Generator().manual_seed(0)
base = torch.randn(6, 22, generator=g)
base[:, ::5], base[:, 2::7], base[:, 3::4] = INF, -INF, NAN
base[1], base[2] = NAN, INF
base[3, ::2], base[3, 1::2] = -0.0, 0.0
stable = lambda t: tuple(t.sort(dim=-1, stable=True))  # noqa: E731
stable_desc = lambda t: tuple(t.sort(dim=-1, stable=True, descending=True))  # noqa: E731
res = []
res.append(check("float64", base.double(), stable))
res.append(check("float64_desc", base.double(), stable_desc))
res.append(check("float16", base.half(), stable))
res.append(check("bfloat16_desc", base.bfloat16(), stable_desc))
big = torch.randn(6, 40, generator=g)
big[:, ::3] = NAN
res.append(check("noncontig_slice", big[:, ::2], stable))
res.append(check("offset_view", big[2:, 5:27], stable))
res.append(check("transposed", big.mT.contiguous().mT, stable))
res.append(check("dim_size_1", base[:, :1].contiguous(), stable))
res.append(check("n2", base[:, :2].contiguous(), stable))
res.append(check("all_nan_all_inf_rows", base[1:3].contiguous(), stable))
res.append(check("topk_nan_largest", base, lambda t: tuple(t.topk(5, -1)), exact_index=False))
res.append(check("kthvalue_last", base, lambda t: tuple(t.kthvalue(22, -1, keepdim=True)), exact_index=False))
for i in range(2):
    res.append(check(f"repeat_{i}", base, lambda t: tuple(t.sort(-1)), exact_index=False))
for nm, f, ex in (("sort", lambda t: tuple(t.sort(-1)), False), ("sort_stable", stable, True),
                  ("topk", lambda t: tuple(t.topk(5, -1)), False),
                  ("kthvalue", lambda t: tuple(t.kthvalue(4, -1, keepdim=True)), False)):
    res.append(check(f"inference_mode_{nm}", base, f, exact_index=ex, inference=True))


def with_out(call):
    def f(t):
        o = (torch.empty(0, dtype=t.dtype, device=t.device), torch.empty(0, dtype=torch.int64, device=t.device))
        got = call(t, o)
        assert got[0].data_ptr() == o[0].data_ptr()
        return tuple(got)
    return f


for nm, call, ex in (("sort_out", lambda t, o: torch.sort(t, -1, out=o), False),
                     ("sort_stable_out", lambda t, o: torch.sort(t, dim=-1, stable=True, out=o), True),
                     ("topk_out", lambda t, o: torch.topk(t, 5, -1, out=o), False),
                     ("kthvalue_out", lambda t, o: torch.kthvalue(t, 4, -1, keepdim=True, out=o), False)):
    for inf_mode in (False, True):
        res.append(check(f"{nm}{'_inference' if inf_mode else ''}", base, with_out(call), exact_index=ex,
                         inference=inf_mode))
big = torch.randn(4, 50257, generator=g)
big[:, ::97], big[:, 5::131], big[:, 7::50] = NAN, INF, -INF
res.append(check("topk_logits_50257", big, lambda t: tuple(t.topk(50, -1)), exact_index=False))
res.append(check("topk_smallest_50257", big, lambda t: tuple(t.topk(50, -1, largest=False)), exact_index=False))
res.append(check("kthvalue_50257", big, lambda t: tuple(t.kthvalue(1000, -1, keepdim=True)), exact_index=False))
from hpu_ops import extrema  # noqa: E402
v, i = extrema.internal_sort(base.to("hpu"), -1)
sync()
res.append(bool(((i.cpu() >= 0) & (i.cpu() < 22)).all()) and torch.isnan(v.cpu()[1]).all().item())
print("ADV " + json.dumps(dict(case="internal_sort_nan_inf", lazy=LAZY, ok=res[-1])), flush=True)
print(f"SUMMARY {sum(res)}/{len(res)} ok", flush=True)
