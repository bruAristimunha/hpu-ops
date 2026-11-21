"""Issue #42: dump the FX graphs of GroupNorm compiled at batch 16, then recompiled at batch 128 (symbolic batch).

Run: TORCH_LOGS=graph_code,aot_graphs PT_HPU_LAZY_MODE=0 python repro/issue42_dump_graph.py
"""
import traceback

import torch
from torch import nn

import habana_frameworks.torch.core  # noqa: F401

module = nn.GroupNorm(4, 16).eval().to("hpu")
compiled = torch.compile(module, backend="hpu_backend")
for b in (16, 128):
    print(f"===== batch {b}", flush=True)
    try:
        with torch.no_grad():
            out = compiled(torch.randn(b, 16, 100, device="hpu"))
        print("ok", out.shape, flush=True)
    except Exception:  # noqa: BLE001
        traceback.print_exc()
