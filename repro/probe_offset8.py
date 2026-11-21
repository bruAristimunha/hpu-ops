"""Scope of the 8-byte storage-offset bug (Gaudi2, bridge 1.21.4). One process per mode:
    PT_HPU_LAZY_MODE=0|1 python repro/probe_offset8.py [compile]
Each line: ok / BAD / RAISE, op, dtype, case."""
import os
import sys

import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.hpu as hthpu
import torch

LAZY = os.environ.get("PT_HPU_LAZY_MODE", "1") != "0"
COMPILE = "compile" in sys.argv
hpu = torch.device("hpu")
DTYPES = [torch.float64, torch.int64, torch.complex64, torch.float32, torch.int32, torch.bfloat16, torch.int16,
          torch.uint8, torch.bool]
if COMPILE:
    import habana_frameworks.torch.dynamo  # noqa: F401

VIEWS = {
    "slice[1:]": lambda t: t[1:],
    "slice[2:5]": lambda t: t[2:5],
    "select row1": lambda t: t.view(2, 6)[1],
    "narrow": lambda t: t.narrow(0, 3, 4),
    "as_strided off1": lambda t: t.as_strided((5,), (1,), 1),
    "col slice [:,1:]": lambda t: t.view(3, 4)[:, 1:],
    "transpose row1": lambda t: t.view(2, 2, 3)[1].t(),
    "step [1::2]": lambda t: t[1::2],
    "unbind[1]": lambda t: t.view(2, 6).unbind(0)[1],
    "split[1]": lambda t: t.split(4)[1],
}
OPS = {
    "add0": lambda v: v + 0,
    "mul1": lambda v: v * 1,
    "neg": lambda v: -v if v.dtype != torch.bool else ~v,
    "clone": lambda v: v.clone(),
    "contiguous": lambda v: v.contiguous(),
    "copy_into": lambda v: torch.empty_like(v, memory_format=torch.contiguous_format).copy_(v),
    "to_same": lambda v: v.to(v.dtype, copy=True),
    "to_f32": lambda v: v.to(torch.float32),
    "sum": lambda v: v.sum(),
    "eq_self_cpu": None,  # compared after .cpu()
    "cpu": lambda v: v,
    "cat": lambda v: torch.cat([v, v]),
    "where": lambda v: torch.where(torch.ones_like(v, dtype=torch.bool), v, v),
    "index_select": lambda v: v.reshape(-1).index_select(0, torch.tensor([0], device=v.device)),
    "matmul": lambda v: (v.reshape(1, -1) @ v.reshape(-1, 1)),
}


def base(dtype):
    t = torch.arange(1, 13)
    if dtype == torch.bool:
        return t % 3 == 0
    return t.to(dtype)


def check(opname, op, dtype, vname, view):
    t = base(dtype)
    ref_v = view(t)
    try:
        ref = op(ref_v)
    except Exception:
        return  # op not defined for this dtype on CPU
    try:
        h = t.to(hpu)
        if LAZY:
            htcore.mark_step()
        f = (lambda h: op(view(h)))
        if COMPILE:
            torch._dynamo.reset()
            f = torch.compile(f, backend="hpu_backend")
        got = f(h)
        if LAZY:
            htcore.mark_step()
        hthpu.synchronize()
        got = got.cpu()
        same = got.shape == ref.shape and torch.equal(got.to(ref.dtype), ref)
        tag = "ok  " if same else "BAD "
        detail = "" if same else f" got {got.reshape(-1)[:6].tolist()} want {ref.reshape(-1)[:6].tolist()}"
        print(f"{tag} {opname:12s} {str(dtype)[6:]:10s} {vname}{detail}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"RAISE {opname:12s} {str(dtype)[6:]:10s} {vname}: {str(e).splitlines()[0][:120]}", flush=True)


print("mode", "compile" if COMPILE else ("lazy" if LAZY else "eager"), torch.__version__, flush=True)
only = [a for a in sys.argv[1:] if a != "compile"]
for dtype in DTYPES:
    if only and str(dtype)[6:] not in only:
        continue
    for opname, op in OPS.items():
        if op is None:
            continue
        for vname, view in VIEWS.items():
            check(opname, op, dtype, vname, view)
