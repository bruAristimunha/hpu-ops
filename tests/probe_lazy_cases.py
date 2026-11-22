"""On a Gaudi card in lazy mode: linalg_on_hpu's failing cases step by step (first call, synced; then repeated calls),
with a traceback, to see which call fails.
    PT_HPU_LAZY_MODE=1 python3 tests/probe_lazy_cases.py"""
import sys
import traceback

import hpu_ops

hpu_ops.enable()
import torch  # noqa: E402

import linalg_on_hpu as T  # noqa: E402

T.ops_linalg = hpu_ops.linalg
L = torch.linalg
names = sys.argv[1:] or ["linalg.eigh", "linalg.matrix_exp", "linalg.lu_factor"]
for case in T.CASES:
    if case.name not in names:
        continue
    for n in [int(x) for x in __import__("os").environ.get("SIZES", "64").split(",")]:
        for b in (32,):
            cpu_in = case.make(b, n, torch.float32)
            ref = T.as_tensors(case.fn(*[x.double() for x in cpu_in]))
            h = [x.to("hpu") for x in cpu_in]
            for call in range(int(__import__("os").environ.get("CALLS", "3"))):
                try:
                    out = T.as_tensors(case.fn(*h))
                    T.sync()
                    err, _ = case.err(out, ref, [x.double() for x in cpu_in])
                    print("RESULT", case.name, n, b, "call", call, "err", err, flush=True)
                except Exception as e:  # noqa: BLE001
                    print("RESULT", case.name, n, b, "call", call, "EXC", repr(e)[:200], flush=True)
                    traceback.print_exc(limit=6)
                    break
