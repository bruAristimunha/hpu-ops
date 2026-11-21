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
inf_ok = check("inference_mode_not_rewritten", base, lambda t: tuple(t.sort(-1)), exact_index=False, inference=True)
print(f"SUMMARY {sum(res)}/{len(res)} ok; inference mode (not rewritten) ok={inf_ok}", flush=True)
