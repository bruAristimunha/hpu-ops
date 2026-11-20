"""On a Gaudi card, adversarial checks of the logsumexp rewrite (lazy) against torch on the CPU in float64: odd shapes,
views, dtypes, extreme values, a summed 1-D result, and the same graph twice in a row. One line per check; exits 1 on
a failure. Also prints whether Habana's bridge reports lazy mode, and what hpu_ops reads."""
import sys

import torch
import habana_frameworks.torch.core as htcore

import hpu_ops
from hpu_ops import perf

hpu = torch.device("hpu")
calls = hpu_ops._STATE["calls"]
failures = []
torch.manual_seed(0)

try:
    from habana_frameworks.torch.utils.internal import is_lazy
    bridge_lazy = is_lazy()
except Exception as e:  # noqa: BLE001
    bridge_lazy = f"unavailable ({e})"
print(f"bridge is_lazy(): {bridge_lazy}  hpu_ops lazy_mode(): {perf.lazy_mode()}", flush=True)


def check(label, make, dim, keepdim=False, reduce=lambda y: y.sum(), dtype=torch.float32, atol=1e-5):
    for run in (1, 2):  # the same lazy graph twice in a row
        x_cpu = make()
        ref_x = x_cpu.double().detach().requires_grad_()
        ref = torch.logsumexp(ref_x, dim, keepdim)
        reduce(ref).backward()
        c0 = calls["logsumexp"]
        try:
            x = make_device(x_cpu, label, dtype)
            if not x.is_leaf:
                x.retain_grad()
            y = torch.logsumexp(x, dim, keepdim)
            reduce(y).backward()
            htcore.mark_step()
            leaf = x
            e_out = (y.detach().double().cpu() - ref.detach()).abs().nan_to_num(0.0).max().item() if y.numel() else 0.0
            nan_ok = torch.equal(torch.isnan(y.detach().cpu()), torch.isnan(ref.detach()))
            e_grad = (leaf.grad.double().cpu() - ref_x.grad).abs().nan_to_num(0.0).max().item() if leaf.numel() else 0.0
            taken = calls["logsumexp"] - c0
            ok = e_out <= atol and e_grad <= atol and nan_ok and tuple(y.shape) == tuple(ref.shape) and y.dtype == dtype
            print(f"{label:46s} run {run}  out {e_out:.1e} grad {e_grad:.1e} shape {tuple(y.shape)} {y.dtype} "
                  f"taken {taken}  {'OK' if ok else 'FAIL'}", flush=True)
        except Exception as e:  # noqa: BLE001
            ok = False
            print(f"{label:46s} run {run}  ERROR {type(e).__name__}: {str(e).splitlines()[0][:200]}", flush=True)
        if not ok:
            failures.append(label)


x_leaf = [None]


def make_device(x_cpu, label, dtype):
    """The device input, as a leaf, or a view of a leaf for the view cases."""
    if "permuted" in label:
        leaf = x_cpu.permute(2, 1, 0).contiguous().to(hpu, dtype).requires_grad_()
        x_leaf[0] = leaf
        return leaf.permute(2, 1, 0)
    if "offset" in label:
        base = torch.cat([torch.zeros(5), x_cpu.flatten()]).to(hpu, dtype).requires_grad_()
        x_leaf[0] = base
        return base[5:].view(x_cpu.shape)
    return x_cpu.to(hpu, dtype).requires_grad_()


inf = float("inf")
special = torch.tensor([[-inf, -inf, -inf], [inf, 1.0, 2.0], [-1e30, 0.0, 88.0], [1e30, 1e30, -1e30],
                        [-80.0, -90.0, -100.0]])
undo = perf.install_logsumexp("hpu", lazy=True)
try:
    check("(64,22,22) dim -1", lambda: torch.randn(64, 22, 22), -1)
    check("(22,) dim 0 summed (the lazy 0-d trap)", lambda: torch.randn(22), 0)
    check("(1,) dim 0", lambda: torch.randn(1), 0)
    check("(1,1) dim 1 keepdim", lambda: torch.randn(1, 1), 1, True)
    check("0-d dim 0", lambda: torch.randn(()), 0)
    check("0-d dim -1 keepdim", lambda: torch.randn(()), -1, True)
    check("(1,22,22) batch 1 dims (1,2)", lambda: torch.randn(1, 22, 22), (1, 2))
    check("(64,22,22) dims (0,2) keepdim", lambda: torch.randn(64, 22, 22), (0, 2), True)
    check("(8,6,4) permuted dim 1", lambda: torch.randn(8, 6, 4), 1)
    check("(8,6) offset view dim -1", lambda: torch.randn(8, 6), -1)
    check("(64,22,22) broadcast-expanded dim 0", lambda: torch.randn(1, 22, 22).expand(64, 22, 22), 0)
    check("(64,22,22) mean-weighted loss", lambda: torch.randn(64, 22, 22), -1,
          reduce=lambda y: (y * torch.linspace(0, 1, y.shape[-1], device=y.device, dtype=y.dtype)).mean())
    check("float64 (64,22,22) dim -1", lambda: torch.randn(64, 22, 22), -1, dtype=torch.float64)
    check("bfloat16 (64,22,22) dim -1", lambda: torch.randn(64, 22, 22), -1, dtype=torch.bfloat16, atol=6e-2)
    # [1e30, 1e30, -1e30] is printed below only: ATen's backward grad * exp(x - result) loses log 2 next to 1e30
    # (CPU float64: [1, 1, 0]); the composite's gradient is the softmax, [0.5, 0.5, 0].
    check("extreme values dim 1 (no -inf/+inf grads)", lambda: special[[2, 4]], 1)
    for row in special[2:]:
        for name in ("ours", "habana"):
            inner = (lambda: None) if name == "ours" else undo
            xr = row.to(hpu).requires_grad_()
            if name == "habana":
                undo()
            torch.logsumexp(xr, 0).backward()
            htcore.mark_step()
            if name == "habana":
                undo = perf.install_logsumexp("hpu", lazy=True)
            xd = row.double().requires_grad_()
            torch.logsumexp(xd, 0).backward()
            print(f"   grad {name:6s} row {row.tolist()}: {xr.grad.cpu().tolist()}  cpu f64 {xd.grad.tolist()}",
                  flush=True)
    check("large magnitude 1e4 dim -1", lambda: torch.randn(16, 22) * 1e4, -1, atol=5e-3)
    out = torch.logsumexp(special.to(hpu), 1)
    htcore.mark_step()
    want = torch.logsumexp(special, 1)
    same = torch.equal(out.cpu().isinf(), want.isinf()) and torch.allclose(out.cpu(), want)
    print(f"special values forward {out.cpu().tolist()}  {'OK' if same else 'FAIL'}", flush=True)
    if not same:
        failures.append("special")
    with torch.inference_mode():
        c0 = calls["logsumexp"]
        y = torch.logsumexp(torch.randn(4, 5).to(hpu), 1)
        htcore.mark_step()
        print(f"inference mode: rewrite taken {calls['logsumexp'] - c0} (expect 0) shape {tuple(y.shape)}", flush=True)
finally:
    undo()
print("FAILURES:", failures or "none", flush=True)
sys.exit(1 if failures else 0)
