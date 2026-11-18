"""On a Gaudi card: the fixes against torch on the CPU (float32 math), eager and hpu_backend-compiled, through
different callers. One line per check; exits 1 if any check fails.
Export the attention setting before running (PT_HPU_USE_OVERRIDE_ATEN_SDPA=1 on bridge 1.21)."""
import copy
import io
import math
import os
import sys

import torch
import torch.nn.functional as F
from torch.utils._python_dispatch import TorchDispatchMode

import sdpa_checks  # braindecode's attention shapes: strided q/k/v, bfloat16, dropout, fully masked rows

sdpa_checks.probe_sliced_views()  # known Habana bug, in processes of its own, before this one takes the card

import hpu_ops  # noqa: E402
from hpu_ops import autocast  # noqa: E402

hpu_ops.enable()
hpu = torch.device("hpu")
calls = hpu_ops._STATE["calls"]
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


def check(name, fn, *cpu_input_sets, atol, n_grad=None, modes=("eager", "compiled"), context=None):
    """fn's outputs and the gradients of its first ``n_grad`` inputs on HPU against the CPU, for each input set
    (several sets exercise recompilation on new shapes). With ``context`` (e.g. ``torch.inference_mode``), the
    forward only, inside it."""
    for mode in modes:
        torch._dynamo.reset()
        f = fn if mode == "eager" else torch.compile(fn, backend="hpu_backend")
        for cpu_inputs in cpu_input_sets:
            grads = 0 if context else len(cpu_inputs) if n_grad is None else n_grad

            def run(g, device):
                xs = [t.detach().to(device).requires_grad_(i < grads) for i, t in enumerate(cpu_inputs)]
                if context:
                    with context():
                        return [g(*xs).float().cpu()]
                out = g(*xs)
                out.backward(torch.ones_like(out))
                return [out.detach().float().cpu()] + [x.grad.float().cpu() for x in xs[:grads]]

            label = f"{name} {tuple(cpu_inputs[0].shape)}"
            try:
                err = max_error(run(f, hpu), run(fn, "cpu"))
                ok = err <= atol
                print(f"{label:60s} {mode:8s} max|err| {err:.1e} {'OK' if ok else 'FAIL'}", flush=True)
            except Exception as e:  # noqa: BLE001
                ok = False
                cause = " | ".join(line for line in str(e).splitlines() if line.strip())[:300] or repr(e)
                print(f"{label:60s} {mode:8s} ERROR {type(e).__name__}: {cause}", flush=True)
            if not ok:
                failures.append((label, mode))


def expect(label, ok, detail=""):
    print(f"{label}: {detail} {'OK' if ok else 'FAIL'}", flush=True)
    if not ok:
        failures.append((label, "-"))


def per_device(module, call=None):
    """A function of the input that runs a CPU or an HPU copy of ``module``. Moving one module between devices would
    copy compiled-step state back to the CPU reference."""
    copies = {"cpu": module, "hpu": copy.deepcopy(module).to(hpu)}
    return lambda x: (call or (lambda m, y: m(y)))(copies[x.device.type], x)


def cpu_fallbacks(fn):
    """Habana's count of the ATen calls that ``fn`` ran on the CPU (op name -> calls)."""
    from habana_frameworks.torch.hpu.metrics import metric_global

    metric = metric_global("cpu_fallback")
    before = dict(dict(metric.stats()).get("FallbackOps") or {})
    fn()
    torch.hpu.synchronize()
    after = dict(dict(metric.stats()).get("FallbackOps") or {})
    return {k: v - before.get(k, 0) for k, v in after.items() if v != before.get(k, 0)}


# ---- attention (Habana's setting)
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
      per_device(mha, lambda m, x: m(x, x, x, key_padding_mask=pad.to(x.device), need_weights=False)[0]),
      (torch.randn(6, 120, 64),), atol=2e-3)
with Ops() as ops:
    sdpa(*(t.to(hpu) for t in (q, k, v, keep)))
expect(f"eager attention ops {sorted(ops.seen)}", "hpu.sdpa_fwd" in ops.seen, "fused kernel")
failures.extend(sdpa_checks.run(hpu))

# ---- linear upsampling: the native op under compile
calls.clear()
check("F.interpolate linear x256 (CodeBrain)", lambda x: F.interpolate(x, scale_factor=256.0, mode="linear"),
      (torch.randn(8, 400, 2),), (torch.randn(5, 400, 2),), atol=1e-4)
expect("compiled linear upsampling took the native op", calls["linear"] > 0, f"{calls['linear']} times")
check("nn.Upsample linear x2, long", torch.nn.Upsample(scale_factor=2.0, mode="linear"), (torch.randn(2, 4, 4096),),
      atol=1e-4)

# ---- nearest upsampling: repeats for integer factors (USleep: BatchNorm1d then nearest x2)
calls.clear()
bn_up = torch.nn.Sequential(torch.nn.BatchNorm1d(302), torch.nn.Upsample(scale_factor=2))
check("BatchNorm1d > nn.Upsample nearest x2 (USleep)", per_device(bn_up), (torch.randn(4, 302, 125),),
      (torch.randn(4, 302, 1),), atol=1e-4)
check("BatchNorm1d > nn.Upsample nearest x2, no_grad", per_device(bn_up), (torch.randn(4, 302, 125),), atol=1e-4,
      context=torch.no_grad)
check("BatchNorm1d (eval) > nn.Upsample nearest x2, inference_mode", per_device(copy.deepcopy(bn_up).eval()),
      (torch.randn(4, 302, 125),), atol=1e-4, context=torch.inference_mode)
expect("nearest x2 took the repeat", calls["nearest"] > 0, f"{calls['nearest']} times")
check("F.interpolate nearest-exact x3", lambda x: F.interpolate(x, scale_factor=3.0, mode="nearest-exact"),
      (torch.randn(4, 6, 50),), atol=1e-6)
check("F.interpolate nearest x2.5 (native op)", lambda x: F.interpolate(x, scale_factor=2.5, mode="nearest"),
      (torch.randn(4, 6, 40),), atol=1e-6)

# ---- padding="valid" (EEGITNet, ATCNet, EEGInceptionERP)
calls.clear()
check('Conv2d padding="valid", depthwise (1, 4) (EEGITNet)',
      per_device(torch.nn.Conv2d(14, 14, (1, 4), padding="valid", groups=14, bias=False)),
      (torch.randn(4, 14, 1, 253),), atol=1e-4)
check('Conv2d padding="valid", depthwise (8, 1) (ATCNet)',
      per_device(torch.nn.Conv2d(16, 32, (8, 1), padding="valid", groups=16, bias=False)),
      (torch.randn(4, 16, 8, 250),), atol=1e-4)
check('Conv1d padding="valid"', per_device(torch.nn.Conv1d(8, 16, 5, padding="valid")), (torch.randn(4, 8, 250),),
      atol=1e-4)
expect('compiled padding="valid" took full zero lists', calls["valid_conv"] > 0, f"{calls['valid_conv']} times")

# ---- glu (BrainModule: a convolution, then a gated linear unit over the channels)
calls.clear()
check("Conv1d > F.glu (BrainModule)", per_device(torch.nn.Conv1d(8, 640, 3, padding=1), lambda m, x: F.glu(m(x), 1)),
      (torch.randn(4, 8, 1000),), atol=1e-4)
expect("glu took a * sigmoid(b)", calls["glu"] > 0, f"{calls['glu']} times")

# ---- renorm (SSTDPN), and none of it on the CPU
calls.clear()
check("torch.renorm p=2 dim=0 (SSTDPN)", lambda w: torch.renorm(w, 2, 0, 0.5), (torch.randn(4, 64),), atol=1e-5)
check("torch.renorm p=2 dim=0, inference_mode", lambda w: torch.renorm(w, 2, 0, 0.5), (torch.randn(4, 64),),
      atol=1e-5, context=torch.inference_mode)
proto = torch.randn(4, 64, device=hpu)
compiled_renorm = torch.compile(lambda w: torch.renorm(w, 2, 0, 0.5), backend="hpu_backend")
for label, run in (("eager", lambda: torch.renorm(proto, 2, 0, 0.5)), ("compiled", lambda: compiled_renorm(proto)),
                   ("eager inference_mode", lambda: torch.inference_mode()(torch.renorm)(proto, 2, 0, 0.5)),
                   ("compiled inference_mode", lambda: torch.inference_mode()(compiled_renorm)(proto))):
    fallbacks = cpu_fallbacks(run)
    expect(f"renorm {label} on the card", not fallbacks, f"CPU fallbacks {fallbacks}")

# ---- the weight gradient of one-channel strided convolutions: about half of it compiled (EEGDINO, CodeBrain, LaBraM)
for label, make in (("EEGDINO/CodeBrain proj_in", lambda: torch.nn.Conv2d(1, 25, (1, 49), (1, 25), (0, 24))),
                    ("LaBraM temporal conv1", lambda: torch.nn.Conv2d(1, 8, (1, 15), (1, 8), (0, 7))),
                    ("Conv2d(1, 25, (1, 3), stride (1, 2))", lambda: torch.nn.Conv2d(1, 25, (1, 3), (1, 2)))):
    torch.manual_seed(0)
    layer, x = make(), torch.randn(8, 1, 64, 200)
    g = torch.randn(layer(x).shape)
    # With an input gradient (the op computes it and the bias gradient), and as a first layer without bias (the op is
    # not called).
    for bias, input_grad in ((True, True), (False, False)):
        if not bias:
            layer.bias = None
        ref, xr = copy.deepcopy(layer).double(), x.double().requires_grad_(input_grad)
        (ref(xr) * g.double()).sum().backward()
        for mode in ("eager", "compiled"):
            torch._dynamo.reset()
            calls.clear()
            card, xh = copy.deepcopy(layer).to(hpu), x.to(hpu, copy=True).requires_grad_(input_grad)
            step = card if mode == "eager" else torch.compile(card, backend="hpu_backend")
            step(xh).mul(g.to(hpu)).sum().backward()
            w, r = card.weight.grad.cpu().double(), ref.weight.grad
            err = ((w - r).abs().max() / r.abs().max()).item()
            others = ([(card.bias.grad, ref.bias.grad)] if bias else []) + ([(xh.grad, xr.grad)] if input_grad else [])
            other = max([((a.cpu().double() - b).abs().max() / b.abs().max()).item() for a, b in others] or [0.0])
            expect(f"{label} weight grad, {'bias and input grad' if bias else 'first layer without bias'}, {mode}",
                   err <= 1e-4 and other <= 1e-4 and calls["conv_weight_grad"] > 0,
                   f"max|err| / max {err:.1e}, scale {((w * r).sum() / (r * r).sum()).item():.4f}, "
                   f"bias/input {other:.1e}")

# ---- the embedding gradient of a 0-d index: Gaudi's kernel sent each one to row 0 (BIOT's channel tokens)
for mode in ("eager", "compiled"):
    calls.clear()
    check("F.embedding of 0-d indices (BIOT channel tokens)",
          lambda w: torch.stack([F.embedding(torch.arange(6, device=w.device)[i], w) * (i + 1) for i in range(6)]),
          (torch.randn(6, 4),), atol=1e-5, modes=(mode,))
    expect(f"embedding gradient of a 0-d index took the one-element index, {mode}", calls["embedding_0d"] > 0,
           f"{calls['embedding_0d']} times")
check("F.embedding of a 0-d index with max_norm (embedding_renorm_)",
      lambda w: F.embedding(torch.tensor(2, device=w.device), w, max_norm=1.0), (3 * torch.randn(5, 4),), atol=1e-5)

# ---- roll by a multiple of the size: Gaudi's eager kernel read a strided input of one wrongly (EMG2QwertyNet's
# rotation-invariant MLP, whose backward rolls each slice of the stack's gradient back). Random output gradients: a
# misread roll of ones is still ones.
slices = torch.randn(497, 2, 3, 16, 33)
for shift in (0, 16, -16, 1):
    calls.clear()
    err = (slices.to(hpu)[:, :, 1].roll(shift, 2).cpu() - slices[:, :, 1].roll(shift, 2)).abs().max().item()
    expect(f"roll of a strided slice by {shift:+d} (size 16)", err == 0 and (calls["roll_zero"] > 0) == (shift % 16 == 0),
           f"max|err| {err:.1e}")
x, g = torch.randn(4, 3, 16), torch.randn(4, 3, 3, 16)
for mode in ("eager", "compiled"):
    torch._dynamo.reset()
    calls.clear()
    rotations = lambda t: torch.stack([t.roll(offset, 2) for offset in (-1, 0, 1)], 2)  # noqa: E731
    step = rotations if mode == "eager" else torch.compile(rotations, backend="hpu_backend")
    xh, xc = x.to(hpu).requires_grad_(), x.clone().requires_grad_()
    step(xh).backward(g.to(hpu))
    rotations(xc).backward(g)
    err = (xh.grad.cpu() - xc.grad).abs().max().item()
    expect(f"input gradient of a stack of rolls by -1, 0, +1, {mode}", err <= 1e-6 and calls["roll_zero"] > 0,
           f"max|err| {err:.1e}")

# ---- linear upsampling by other than a power of two: Gaudi's forward kernel drifts with the length (NeuroPose's
# 9920 to 10000: 34% off), and its backward is less exact than the CPU's at some sizes. Output and input gradient as
# close to float64 as the CPU's float32 kernel.
for length, out, align in ((200, 300, False), (1000, 1009, False), (9920, 10000, False), (9920, 7440, True),
                           (1000, 2000, False)):
    x, g = torch.randn(2, 4, length), torch.randn(2, 4, out)
    resize = lambda t: F.interpolate(t, size=out, mode="linear", align_corners=align)  # noqa: E731

    def output_and_grad(step, device, dtype=torch.float32):
        xr = x.to(device, dtype, copy=True).requires_grad_()
        y = step(xr)
        y.backward(g.to(device, dtype))
        return y.detach().cpu().double(), xr.grad.cpu().double()

    reference, cpu = output_and_grad(resize, "cpu", torch.float64), output_and_grad(resize, "cpu")
    error = lambda pair, i: ((pair[i] - reference[i]).abs().max() / reference[i].abs().max()).item()  # noqa: E731
    for mode in ("eager", "compiled"):
        torch._dynamo.reset()
        calls.clear()
        card = output_and_grad(resize if mode == "eager" else torch.compile(resize, backend="hpu_backend",
                                                                            fullgraph=True), hpu)
        power_of_two = not align and out % length == 0 and (out // length) & (out // length - 1) == 0
        expect(f"linear {length} -> {out}, align_corners {align}, {mode}",
               all(error(card, i) <= 1.5 * error(cpu, i) + 1e-6 for i in (0, 1))
               and (calls["linear_index"] > 0) != power_of_two,
               f"vs float64: output {error(card, 0):.1e} (CPU float32 {error(cpu, 0):.1e}), input grad "
               f"{error(card, 1):.1e} (CPU {error(cpu, 1):.1e}), {'native op' if power_of_two else 'index ops'}")
x = torch.randn(2, 4, 1000)
reference = F.interpolate(x.double(), size=1009, mode="linear")
calls.clear()
with torch.inference_mode():  # below autograd: the kernel at the HPU key
    card = F.interpolate(x.to(hpu), size=1009, mode="linear").cpu().double()
cpu_err, err = (F.interpolate(x, size=1009, mode="linear").double() - reference).abs().max(), (card - reference).abs().max()
expect("linear 1000 -> 1009 in inference mode", err <= 1.5 * cpu_err + 1e-6 and calls["linear_index"] > 0,
       f"vs float64 {err:.1e} (CPU float32 {cpu_err:.1e})")

# ---- torch.save after an optimizer step: conv weights in a permuted layout, eager and compiled. The reference is the
# same step on the CPU, not a host copy of the card's weights (the path under test).
for mode in ("eager", "compiled"):
    torch._dynamo.reset()
    nets = {"cpu": torch.nn.Sequential(torch.nn.Conv2d(1, 8, (1, 25)), torch.nn.BatchNorm2d(8), torch.nn.ELU(),
                                       torch.nn.Conv2d(8, 16, (8, 1)))}
    nets["hpu"] = copy.deepcopy(nets["cpu"]).to(hpu)
    x = torch.randn(16, 1, 8, 1000)
    optimizers = {}
    for device, net in nets.items():
        optimizers[device] = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
        (torch.compile(net, backend="hpu_backend") if mode == "compiled" and device == "hpu" else net)(
            x.to(device)).square().mean().backward()
        optimizers[device].step()
    calls.clear()
    label = f"torch.save after an optimizer step, {mode}"
    try:
        buffer = io.BytesIO()
        torch.save({"model": nets["hpu"].state_dict(), "optimizer": optimizers["hpu"].state_dict()}, buffer)
        buffer.seek(0)
        loaded = torch.load(buffer, weights_only=True)
        devices = {v.device.type for v in loaded["model"].values()}
        err = max(((loaded["model"][k].cpu().float() - v.float()).abs().max() / v.float().abs().max().clamp_min(1e-12))
                  .item() for k, v in nets["cpu"].state_dict().items())
        expect(label, err <= 1e-4 and devices == {"hpu"} and calls["save"] > 0,
               f"max|loaded - CPU step| / max {err:.1e}, loaded on {devices}, {calls['save']} host copies")
    except Exception as e:  # noqa: BLE001
        expect(label, False, f"ERROR {type(e).__name__}: {str(e).splitlines()[0][:300]}")
# ---- bfloat16 autocast: CUDA's dtypes once hpu_ops.autocast's lists are exported; Habana's are only printed
exported = {name: os.environ.get(name) for name in autocast.FILES}
label = "bfloat16 autocast with CUDA's lists"
try:
    x, h = torch.randn(4, 16, 32, device=hpu), torch.randn(32, device=hpu)
    with torch.autocast("hpu", dtype=torch.bfloat16):
        low = F.linear(x, torch.randn(32, 32, device=hpu))
        got = {"linear": low, "sdpa": F.scaled_dot_product_attention(x, x, x), "layer_norm": F.layer_norm(x, (32,)),
               "group_norm": F.group_norm(x, 4), "batch_norm": F.batch_norm(x, None, None, training=True),
               "softmax": x.softmax(-1), "log_softmax": x.log_softmax(-1), "sum": x.sum(), "mean": x.mean(),
               "mul": x * x, "dot, bfloat16 and float32": torch.dot(low[0, 0], h),
               "atan2, bfloat16 and float32": torch.atan2(low, x),
               "binary_cross_entropy": F.binary_cross_entropy(low.sigmoid(), torch.rand_like(x))}
    dtypes = {op: str(out.dtype).replace("torch.", "") for op, out in got.items()}
    if exported == autocast.paths():
        cuda = {op: "bfloat16" if op in ("linear", "sdpa") else "float32" for op in got}
        expect(label, dtypes == cuda, str(dtypes))
    elif not any(exported.values()):
        print(f"bfloat16 autocast with Habana's lists (not checked; export hpu_ops.autocast's for CUDA's): {dtypes}",
              flush=True)
    else:
        expect(label, False, f"not hpu_ops.autocast's pair of lists: {exported}")
except Exception as e:  # noqa: BLE001
    expect(label, False, f"ERROR {type(e).__name__}: {str(e).splitlines()[0][:300]}")

def check_model_compiles(name, model, x, rtol=1e-3):
    """``model``'s compiled training step runs and matches its eager one on the card: the output, and the parameter
    gradients, each relative to its largest value."""
    model, x = model.to(hpu).train(), x.to(hpu)
    results, mode = {}, "eager"
    try:
        for mode in ("eager", "compiled"):
            torch._dynamo.reset()
            model.zero_grad(set_to_none=True)
            out = (model if mode == "eager" else torch.compile(model, backend="hpu_backend"))(x)
            out.square().mean().backward()
            grads = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
            results[mode] = [out.detach().float().cpu().flatten(), grads.float().cpu()]
        err = max((a - b).abs().max().item() / b.abs().max().item() for a, b in zip(results["compiled"],
                                                                                  results["eager"]))
        expect(f"{name} compiled vs eager", err <= rtol, f"max|err| / max {err:.1e}")
    except Exception as e:  # noqa: BLE001
        expect(f"{name} {mode}", False, f"ERROR {type(e).__name__}: {str(e).splitlines()[0][:300]}")


# ---- pass_weight_permutation off: FBCNet failed with "create tensor invalid permutation 4 2"
try:
    from braindecode.models import FBCNet

    check_model_compiles("braindecode FBCNet", FBCNet(n_chans=8, n_outputs=2, n_times=1000, sfreq=200.0),
                         torch.randn(4, 8, 1000))
except ImportError as e:
    print("braindecode not importable, FBCNet check skipped:", e, flush=True)

# ---- braindecode's BIOT attention block (einsum views)
try:
    from braindecode.modules.linear_attention import _SelfAttention

    block = _SelfAttention(256, 8)
    mask = torch.arange(120) < torch.tensor([120, 100, 80, 60, 40, 20, 10, 1])[:, None]
    check("braindecode _SelfAttention (BIOT einsums)", per_device(block, lambda m, x: m(x, mask.to(x.device))),
          (torch.randn(8, 120, 256),), atol=1e-4)
except ImportError as e:
    print("braindecode not importable, BIOT check skipped:", e, flush=True)

print(f"{len(failures)} failed: {failures}" if failures else "all checks passed", flush=True)
sys.exit(1 if failures else 0)
