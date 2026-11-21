"""Which conv1d kernel sizes compile on Gaudi2 (bridge 1.21.4), natively, for (4, 8, L) x (16, 8, k). One size per
process (a failed lazy graph poisons the rest): PT_HPU_LAZY_MODE=0|1 python3 repro/probe_adv_conv_k.py K L [enable]"""
import json
import os
import sys

import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.hpu as hthpu
import torch
import torch.nn.functional as F

k, length = int(sys.argv[1]), int(sys.argv[2])
if len(sys.argv) > 3:
    import hpu_ops
    hpu_ops.enable()
lazy = os.environ.get("PT_HPU_LAZY_MODE", "0") != "0"
torch.manual_seed(0)
x, w = torch.randn(4, 8, length), torch.randn(16, 8, k) / k ** 0.5
try:
    y = F.conv1d(x.to("hpu"), w.to("hpu"))
    if lazy:
        htcore.mark_step()
    y = y.cpu()
    hthpu.synchronize()
    e = float((y.double() - F.conv1d(x.double(), w.double())).abs().max())
    print("PROBE " + json.dumps({"k": k, "L": length, "lazy": lazy, "enable": len(sys.argv) > 3, "ok": True,
                                 "max_err": e}))
except Exception as ex:  # noqa: BLE001
    print("PROBE " + json.dumps({"k": k, "L": length, "lazy": lazy, "enable": len(sys.argv) > 3, "ok": False,
                                 "raised": repr(ex)[:200]}))
