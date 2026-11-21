"""Print every aten.convolution a model's forward makes (input, weight, stride, padding, dilation, groups), on the CPU.
    python3 repro/conv_shapes.py MODEL CH [resolve]   (needs spd_diag; same builder as tests/model_steps.py)"""
import sys

import torch
from torch.utils._python_dispatch import TorchDispatchMode

sys.argv += [""] * 3
name, ch, k_arg = sys.argv[1], int(sys.argv[2]), sys.argv[3]
sys.path.insert(0, __file__.rsplit("/repro/", 1)[0] + "/tests")
from model_steps import batch  # noqa: E402
from spd_diag import protocol as P  # noqa: E402

info, X, _ = batch(ch, 4)
k = P.resolve_k(name, ch, len(info["classes"]), info["sfreq"], None) if k_arg == "resolve" else None
torch.manual_seed(0)
model = P.build(name, ch, info, k)
xb = model.fixed_input(X) if hasattr(model, "fixed_input") else X


class Log(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func is torch.ops.aten.convolution.default:
            x, w, _, s, p, d, t, _, g = args
            print("CONV", name, ch, k_arg or "-", "x", list(x.shape), "w", list(w.shape), "stride", list(s),
                  "pad", list(p), "dil", list(d), "transposed", t, "groups", g, "contig", x.is_contiguous(),
                  "xstride", list(x.stride()), flush=True)
        return func(*args, **(kwargs or {}))


with Log():
    out = model(xb)
print("DONE", name, ch, k_arg, flush=True)
