"""Issue #42 round 3: the round-2 fixes on the card. channels_last 4-D GroupNorm training and bfloat16 input with
float32 parameters, default compile at batch 16 then 128, against the CPU."""
import torch
import torch.nn.functional as F

import habana_frameworks.torch.core  # noqa: F401
import habana_frameworks.torch.hpu as hthpu
from habana_frameworks.torch.hpu.metrics import metric_global

import hpu_ops

hpu_ops.enable()
from hpu_ops._dispatch import _STATE  # noqa: E402


def fallbacks():
    return dict(dict(metric_global("cpu_fallback").stats()).get("FallbackOps", {}) or {})


torch.manual_seed(0)
w, b = torch.rand(8) + 0.5, torch.randn(8)
cases = {
    "channels_last train": (lambda n: torch.randn(n, 8, 6, 10).to(memory_format=torch.channels_last), torch.float32),
    "bf16 input, f32 params": (lambda n: torch.randn(n, 8, 50).bfloat16(), torch.bfloat16),
}
for tag, (make, dtype) in cases.items():
    torch._dynamo.reset()
    f = lambda x, w, b: F.group_norm(x, 4, w, b)  # noqa: E731
    g = torch.compile(f, backend="hpu_backend")
    for n in (16, 128):
        x = make(n)
        xr = x.float().clone().requires_grad_()
        ref = F.group_norm(xr, 4, w, b)
        ref.square().sum().backward()
        xh = x.to("hpu").detach().requires_grad_(dtype == torch.float32)
        before = fallbacks()
        try:
            out = g(xh, w.to("hpu"), b.to("hpu"))
            extra = ""
            if dtype == torch.float32:
                out.square().sum().backward()
                hthpu.synchronize()
                extra = f", grad diff {float((xh.grad.cpu() - xr.grad).abs().max()):.1e}"
            hthpu.synchronize()
            res = (f"ok, diff {float((out.float().cpu() - ref.detach()).abs().max()):.1e}{extra}, dtype {out.dtype}, "
                   f"channels_last {out.is_contiguous(memory_format=torch.channels_last)}")
        except Exception as e:  # noqa: BLE001
            res = f"{type(e).__name__}: {str(e)[:200]}"
        new = {k: v - before.get(k, 0) for k, v in fallbacks().items() if v - before.get(k, 0)}
        print(f"{tag:24s} batch {n:3d}: {res} | rewrites {_STATE['calls']['group_norm']} | fallback {new}", flush=True)
