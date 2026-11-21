"""Gaudi check of hpu_ops.wide (int64 / float64 checks at the copy to the card) and its per-copy cost.
    PT_HPU_LAZY_MODE=0|1 python tests/check_wide_hpu.py"""
import os
import time
import warnings

import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.hpu as hthpu
import torch

import hpu_ops
from hpu_ops import wide

LAZY = os.environ.get("PT_HPU_LAZY_MODE", "1") != "0"
MODE = "lazy" if LAZY else "eager"
hpu = torch.device("hpu")


def sync():
    if LAZY:
        htcore.mark_step()
    hthpu.synchronize()


def caught(fn):
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        try:
            out = fn()
            sync()
            err = None
        except RuntimeError as e:
            out, err = None, str(e).splitlines()[0][:90]
    return [str(w.message)[:60] for w in record if "HPU" in str(w.message)], out, err


hpu_ops.enable()
print(f"{MODE}: copy_ kernel at HPU: {torch._C._dispatch_dump_table('aten::copy_').splitlines()[0][:110]}",
      flush=True)
for name, fn in [("int64 2**40 .to(hpu)", lambda: torch.tensor([2 ** 40 + 1, 3]).to(hpu)),
                 ("int64 2**40 again", lambda: torch.tensor([2 ** 41]).to(hpu)),
                 ("int64 small", lambda: torch.arange(5).to(hpu)),
                 ("float64 .to(hpu)", lambda: torch.arange(4.0, dtype=torch.float64).to(hpu)),
                 ("float32 .to(hpu).double()", lambda: torch.arange(4.0).to(hpu).double()),
                 ("torch.tensor(int64 big, device=hpu)", lambda: torch.tensor([2 ** 33], device=hpu))]:
    w, out, err = caught(fn)
    value = None if out is None else out.cpu().tolist()
    print(f"{MODE}: {name}: warnings {w} value {value} error {err}", flush=True)
wide._WARNED.clear()
os.environ["HPU_OPS_INT64_CHECK"] = "raise"
print(f"{MODE}: raise setting:", caught(lambda: torch.tensor([2 ** 40]).to(hpu))[2], flush=True)
del os.environ["HPU_OPS_INT64_CHECK"]


def bench(fn, n=200):
    for _ in range(10):
        fn()
    sync()
    t0 = time.perf_counter()
    keep = [fn() for _ in range(n)]
    sync()
    return (time.perf_counter() - t0) / n * 1e6, keep


labels = torch.randint(0, 4, (32,))
batch = torch.randn(32, 22, 1000)
h = torch.randn(64, 22, 22).to(hpu)
cases = {"int64 labels (32,) .to(hpu)": lambda: labels.to(hpu),
         "float32 batch (32,22,1000) .to(hpu)": lambda: batch.to(hpu),
         "hpu transpose .contiguous()": lambda: h.transpose(1, 2).contiguous(),
         "hpu .to(bfloat16)": lambda: h.to(torch.bfloat16)}
rounds = {k: {"on": [], "off": []} for k in cases}
hpu_ops._STATE["undo"]()  # the copy checks alone against no hpu-ops at all
for r in range(3):
    for state in ("on", "off") if r % 2 == 0 else ("off", "on"):
        undo = wide.install_copy_checks("hpu") if state == "on" else (lambda: None)
        for k, fn in cases.items():
            rounds[k][state].append(bench(fn)[0])
        undo()
for k, v in rounds.items():
    on, off = sorted(v["on"])[1], sorted(v["off"])[1]
    print(f"{MODE}: cost {k}: with checks {on:.1f} us, without {off:.1f} us (medians of 3 alternating rounds)",
          flush=True)
