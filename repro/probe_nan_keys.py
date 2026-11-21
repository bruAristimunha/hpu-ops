"""Which dispatch keys Habana registers for the max-like ops, and which aten ops a Python call reaches on HPU."""
import habana_frameworks.torch.core  # noqa: F401
import torch
import torch.nn.functional as F
from torch.utils._python_dispatch import TorchDispatchMode

OPS = ["amax", "amin", "max", "max.dim", "min", "min.dim", "aminmax", "argmax", "argmin", "maximum", "minimum",
       "fmax", "fmin", "clamp", "clamp.Tensor", "clamp_min", "clamp_max", "clamp_min.Tensor", "clamp_max.Tensor",
       "linalg_vector_norm", "max_pool1d", "max_pool2d", "max_pool3d", "max_pool1d_with_indices",
       "max_pool2d_with_indices", "max_pool3d_with_indices", "max_pool2d_with_indices_backward",
       "max_pool3d_with_indices_backward", "sort", "sort.stable", "topk", "kthvalue", "median", "median.dim",
       "nanmedian", "argsort", "argsort.stable", "max.other", "min.other", "norm.ScalarOpt_dim"]
for name in OPS:
    full = f"aten::{name.split('.')[0]}" + (f".{name.split('.')[1]}" if "." in name else "")
    keys = {k: torch._C._dispatch_has_kernel_for_dispatch_key(full, k)
            for k in ("HPU", "AutogradHPU", "CompositeImplicitAutograd", "CompositeExplicitAutograd", "Autograd")}
    print(f"KEYS {name:34s} " + " ".join(f"{k}={int(v)}" for k, v in keys.items()), flush=True)


class Ops(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        seen.append(str(func))
        return func(*args, **(kwargs or {}))


x = torch.randn(2, 3, 8, 8, device="hpu")
calls = {"max_pool1d": lambda: F.max_pool1d(x[0], 2),
         "max_pool1d_idx": lambda: F.max_pool1d(x[0], 2, return_indices=True),
         "max_pool2d": lambda: F.max_pool2d(x, 2), "max_pool3d": lambda: F.max_pool3d(x.unsqueeze(1), 2),
         "norm_inf": lambda: x.norm(float("inf"), -1), "max_other": lambda: torch.max(x, x),
         "clamp_t": lambda: x.clamp(torch.zeros_like(x), torch.ones_like(x)), "median": lambda: x.median(),
         "adaptive_max_pool2d": lambda: F.adaptive_max_pool2d(x, 1), "nn_MaxPool2d": lambda: torch.nn.MaxPool2d(2)(x),
         "topk": lambda: x.topk(2, -1), "argsort": lambda: x.argsort(-1)}
for name, f in calls.items():
    for rg in (False, True):
        seen = []
        xx = x.detach().requires_grad_(rg)
        globals()["x"] = xx
        with Ops():
            f()
        print(f"CALLS {name:20s} grad={int(rg)} {seen}", flush=True)
