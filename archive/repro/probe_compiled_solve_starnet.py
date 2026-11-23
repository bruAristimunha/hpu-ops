"""On a Gaudi card: one compiled training step of spd_learn STaRNet(64), logging every call of the solve route
(shapes, dtype, tracing, which route) and printing the full traceback of a failure.
    PT_HPU_LAZY_MODE=0 python3 repro/probe_compiled_solve_starnet.py [--eager]"""
import sys
import traceback

import torch

import hpu_ops
from hpu_ops import linalg as L

LOG = []
_orig = L._tpc_takes


def _logged_tpc_takes(device_type, takes, matrix_at=0, maxn=None, large=False):
    inner = _orig(device_type, takes, matrix_at, maxn, large)

    def tpc_takes(*args, **kwargs):
        res = inner(*args, **kwargs)
        A = args[matrix_at]
        shapes = [tuple(a.shape) for a in args if isinstance(a, torch.Tensor)]
        LOG.append((shapes, str(A.dtype), A.device.type, type(A).__name__, L._tracing(A), res))
        print("CALL", shapes, A.dtype, A.device.type, type(A).__name__, "tracing", L._tracing(A), "takes", res,
              flush=True)
        return res

    return tpc_takes


L._tpc_takes = _logged_tpc_takes
hpu_ops.enable()
import hpu_complex  # noqa: E402

hpu_complex.enable()
from spd_learn.models import STaRNet  # noqa: E402

torch.manual_seed(0)
model = STaRNet(64, 4).to("hpu")
X = torch.randn(32, 64, 1000).to("hpu")
y = torch.randint(4, (32,)).to("hpu")
fwd = model if "--eager" in sys.argv else torch.compile(model, backend="hpu_backend", dynamic=False)
try:
    out = fwd(X)
    loss = torch.nn.functional.cross_entropy(out, y)
    loss.backward()
    print("OK loss", float(loss.cpu()))
except Exception:  # noqa: BLE001
    traceback.print_exc()
    print("FAILED")
