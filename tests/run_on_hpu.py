"""On a Gaudi card: the fixes against torch on the CPU (float32 math), eager and hpu_backend-compiled, through
different callers. One line per check; exits 1 if any check fails.
Export the attention setting before running (PT_HPU_USE_OVERRIDE_ATEN_SDPA=1 on bridge 1.21)."""
import math
import sys

import torch
import torch.nn.functional as F
from torch.utils._python_dispatch import TorchDispatchMode

import hpu_ops

hpu_ops.enable()
hpu = torch.device("hpu")
torch.manual_seed(0)
failures = []


class Ops(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.seen = set()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.seen.add(str(func.overloadpacket).replace("aten.", ""))
        return func(*args, **(kwargs or {}))


def max_error(got, ref):
    errors = [(a - b).abs().max().item() for a, b in zip(got, ref)]
    return math.inf if any(math.isnan(e) for e in errors) else max(errors)  # max() would skip a NaN


def check(name, fn, *cpu_input_sets, atol, n_grad=None, modes=("eager", "compiled")):
    """fn's outputs and the gradients of its first ``n_grad`` inputs on HPU against the CPU, for each input set
    (several sets exercise recompilation on new shapes)."""
    for mode in modes:
        torch._dynamo.reset()
        f = fn if mode == "eager" else torch.compile(fn, backend="hpu_backend")
        for cpu_inputs in cpu_input_sets:
            grads = len(cpu_inputs) if n_grad is None else n_grad

            def run(g, device):
                xs = [t.detach().to(device).requires_grad_(i < grads) for i, t in enumerate(cpu_inputs)]
                out = g(*xs)
                out.backward(torch.ones_like(out))
                return [out.detach().float().cpu()] + [x.grad.float().cpu() for x in xs[:grads]]

            label = f"{name} {tuple(cpu_inputs[0].shape)}"
            try:
                err = max_error(run(f, hpu), run(fn, "cpu"))
                ok = err <= atol
                print(f"{label:52s} {mode:8s} max|err| {err:.1e} {'OK' if ok else 'FAIL'}", flush=True)
            except Exception as e:  # noqa: BLE001
                ok = False
                cause = " | ".join(line for line in str(e).splitlines() if line.strip())[:300] or repr(e)
                print(f"{label:52s} {mode:8s} ERROR {type(e).__name__}: {cause}", flush=True)
            if not ok:
                failures.append((label, mode))


sdpa = lambda q, k, v, m: F.scaled_dot_product_attention(q, k, v, attn_mask=m)  # noqa: E731
q, k, v = (torch.randn(6, 8, 120, 64) for _ in range(3))
keep = (torch.arange(120) < torch.tensor([120, 90, 60, 30, 10, 1])[:, None])[:, None, None, :]
check("F.sdpa, boolean key padding", sdpa, (q, k, v, keep), atol=2e-3, n_grad=3)
check("F.sdpa, additive key padding", sdpa, (q, k, v, torch.zeros(keep.shape).masked_fill(~keep, float("-inf"))),
      atol=2e-3, n_grad=3)
check("F.sdpa, no mask", lambda q, k, v: F.scaled_dot_product_attention(q, k, v), (q, k, v), atol=2e-3)
mha = torch.nn.MultiheadAttention(64, 4, batch_first=True)
pad = ~keep[:, 0, 0, :]
check("nn.MultiheadAttention, key_padding_mask",
      lambda x: mha.to(x.device)(x, x, x, key_padding_mask=pad.to(x.device), need_weights=False)[0],
      (torch.randn(6, 120, 64),), atol=2e-3)
with Ops() as ops:
    sdpa(*(t.to(hpu) for t in (q, k, v, keep)))
fused = "hpu.sdpa_fwd" in ops.seen
print(f"eager attention ops: {sorted(ops.seen)} {'OK' if fused else 'FAIL: not the fused kernel'}", flush=True)
if not fused:
    failures.append(("eager attention ops", "eager"))

hpu_ops._STATE["native_calls"] = 0
check("F.interpolate linear x256 (CodeBrain)", lambda x: F.interpolate(x, scale_factor=256.0, mode="linear"),
      (torch.randn(8, 400, 2),), (torch.randn(5, 400, 2),), atol=1e-4)
native = hpu_ops._STATE.get("native_calls", 0)
print(f"compiled upsampling took the native op {native} times {'OK' if native else 'FAIL: override inactive'}",
      flush=True)
if not native:
    failures.append(("upsampling override", "compiled"))
check("nn.Upsample linear x2, long", torch.nn.Upsample(scale_factor=2.0, mode="linear"), (torch.randn(2, 4, 4096),),
      atol=1e-4)
try:
    from braindecode.modules.linear_attention import _SelfAttention

    block = _SelfAttention(256, 8)
    mask = torch.arange(120) < torch.tensor([120, 100, 80, 60, 40, 20, 10, 1])[:, None]
    check("braindecode _SelfAttention (BIOT einsums)", lambda x: block.to(x.device)(x, mask.to(x.device)),
          (torch.randn(8, 120, 256),), atol=1e-4)
except ImportError as e:
    print("braindecode not importable, BIOT check skipped:", e, flush=True)

print(f"{len(failures)} failed: {failures}" if failures else "all checks passed", flush=True)
sys.exit(1 if failures else 0)
