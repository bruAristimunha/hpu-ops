"""On a Gaudi card: torch's scaled_dot_product_attention under Habana's attention setting, at the shapes braindecode's
models use, against torch on the CPU in float32. braindecode copies q/k/v to contiguous tensors before Gaudi's
FusedSDPA because strided views were wrong "at some shapes"; its branch can go only if these checks pass.

Cases, eager and hpu_backend-compiled:
- q/k/v as views of one (batch, seq, 3 * heads * head_dim) projection, split the way the models split it, and as
  contiguous copies. All layouts take the same values, so they share one reference;
- bfloat16 inputs, against float32 on the same rounded inputs;
- dropout in training, on ``n`` copies of one sample: the mean must be the output without dropout, the variance
  ``p / (1 - p) * sum_j a_ij^2 v_j^2`` (torch's dropout on the attention weights), the gradients finite, and the
  backward must use the forward's draws: the output is linear in v, so ``sum(grad_v * v) = sum(grad_out * out)``
  for each (sample, head) only then;
- fully masked query rows (a sample with no real key): what the card returns is printed next to torch's result
  (a record, not a check); the other samples must still match.

A known Habana bug is probed in processes of its own and never counted (probe_sliced_views): eager attention on q/k/v
that view part of a larger tensor fails to launch after an earlier call on views of a tensor of the same size. Every
outcome of every other check counts.

Standalone: ``python tests/sdpa_checks.py`` (exits 1 on a failure). run_on_hpu.py runs it too. Export the attention
setting first (PT_HPU_USE_OVERRIDE_ATEN_SDPA=1 on bridge 1.21).
"""
import math
import os
import subprocess
import sys

import torch
import torch.nn.functional as F
from torch.utils._python_dispatch import TorchDispatchMode

TOL = {torch.float32: 5e-3, torch.bfloat16: 5e-2}  # max |error| / max |reference|; a wrong layout gives about 1
DTYPES = (torch.float32, torch.bfloat16)
# name, batch, heads, seq, head_dim, the model's layout, the model's mask, as recorded from the models on 18 Nov
# (braindecode 7d45007a): a 100-sample micro-batch of 32 channels x 30 one-second patches (REVE: 50).
SHAPES = (
    ("LaBraM", 100, 10, 961, 20, "qk_normed", "additive"),  # 960 tokens + cls; 200 = 10 heads x 20
    ("REVE", 50, 8, 960, 64, "packed", "bool"),
    ("CBraMod criss-cross, space", 3000, 4, 32, 25, "seqfirst", "additive per head"),  # nn.MultiheadAttention
    ("CBraMod criss-cross, time", 3200, 4, 30, 25, "seqfirst", None),
    ("EEGPT", 3000, 8, 36, 64, "packed", None),  # 32 channels + 4 summary tokens per patch
    ("tiny", 3, 2, 5, 8, "separate", "bool"),
)
LAYOUTS = ("packed", "qk_normed", "separate", "seqfirst", "contiguous")
# Bridge 1.21.4, eager: attention on q/k/v viewing part of a larger tensor (x[:1], x[:, :n]) fails with this error
# after an earlier call on views of a tensor of the same size; compiled, or on a copy (.clone()), it runs.
# Repro: SLICED_VIEW_PROBE below.
KNOWN_SLICED_VIEW_BUG = "synStatus 30 [Sections' validation failure]"
# argv: eager or compiled. One process per mode: a failed launch can leave the card unusable for what follows.
SLICED_VIEW_PROBE = """
import sys
import habana_frameworks.torch.core
import torch
import torch.nn.functional as F

def attend(x):  # q, k, v: (batch, 8, seq, 64) views of one (batch, seq, 3 * 8 * 64) projection
    return F.scaled_dot_product_attention(*x.unflatten(-1, (3, 8, 64)).permute(2, 0, 3, 1, 4).unbind(0))

fn = torch.compile(attend, backend="hpu_backend") if sys.argv[1] == "compiled" else attend
gen = torch.Generator().manual_seed(7)
whole, part = (torch.randn(3000, 36, 1536, generator=gen) for _ in range(2))
card = part.to("hpu")
with torch.no_grad():
    fn(whole.to("hpu"))
    for name, view in (("batch slice [:1]", lambda t: t[:1]), ("token slice [:, :18]", lambda t: t[:, :18])):
        try:
            got = fn(view(card)).float().cpu()
            ref = attend(view(part))
            print(f"PROBE {name}: runs, max|err| / max {((got - ref).abs().max() / ref.abs().max()).item():.1e}")
        except RuntimeError as exc:
            print(f"PROBE {name}: {str(exc).splitlines()[0][:110]}")
"""


def split_qkv(x, layout, heads):
    """(batch, heads, seq, head_dim) q, k, v of the projection ``x``; every layout takes the same values.

    packed: (B, L, 3, H, D) permuted, or three chunks with the heads moved forward (EEGPT, TimmAttention, REVE: the
    same strides); qk_normed: q and k contiguous, as LayerNorms leave them, v a view (LaBraM); separate: one
    contiguous projection each (einops-split modules); seqfirst: nn.MultiheadAttention's split (CBraMod);
    contiguous: copies.
    """
    width = x.shape[-1] // 3
    if layout == "seqfirst":
        batch, seq = x.shape[:2]
        proj = x.transpose(0, 1).unflatten(-1, (3, width)).unsqueeze(0).transpose(0, -2).squeeze(-2).contiguous()
        return [t.view(seq, batch * heads, -1).transpose(0, 1).view(batch, heads, seq, -1) for t in proj.unbind(0)]
    if layout == "separate":
        return [t.contiguous().unflatten(-1, (heads, -1)).transpose(1, 2) for t in x.chunk(3, dim=-1)]
    q, k, v = x.unflatten(-1, (3, heads, -1)).permute(2, 0, 3, 1, 4).unbind(0)
    if layout == "qk_normed":
        return [q.contiguous(), k.contiguous(), v]
    return [q, k, v] if layout == "packed" else [t.contiguous() for t in (q, k, v)]


def v_part(t, heads):
    """The value part of a projection-shaped tensor (the projection, or its gradient), (B, H, L, D)."""
    return t[..., 2 * t.shape[-1] // 3:].unflatten(-1, (heads, -1)).transpose(1, 2)


def attention(layout, heads, dropout_p=0.0):
    def fn(x, mask=None):
        q, k, v = split_qkv(x, layout, heads)
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=dropout_p)

    return fn


def key_mask(lengths, seq, kind, heads):
    """Mask of the first ``lengths`` keys of each sample: boolean (True attends) or additive, (B, 1, 1, L); "additive
    per head" is (B, H, 1, L), as nn.MultiheadAttention passes a key padding mask on."""
    keep = torch.arange(seq) < lengths[:, None]
    if kind == "bool":
        return keep[:, None, None, :]
    mask = torch.zeros(keep.shape).masked_fill(~keep, float("-inf"))[:, None, None, :]
    return mask.expand(-1, heads, -1, -1).contiguous() if kind == "additive per head" else mask


def inputs(batch, heads, seq, dim, kind, seed, fully_masked=False):
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(batch, seq, 3 * heads * dim, generator=gen)
    g = torch.randn(batch, heads, seq, dim, generator=gen)
    lengths = torch.randint(1, seq + 1, (batch,), generator=gen)
    lengths[-1] = seq
    if fully_masked:
        lengths[0] = 0
    return x, g, None if kind is None else key_mask(lengths, seq, kind, heads)


def cast(t, dtype, device):
    if t is None:
        return None
    return t.to(device=device, dtype=dtype) if t.is_floating_point() else t.to(device)


def forward_backward(fn, x, g, mask, dtype, device):
    """Output and gradient of the projection, both in float32 on the CPU."""
    x = cast(x, dtype, device).detach().requires_grad_(True)  # detach: ``to`` returns ``x`` itself when nothing moves
    out = fn(x, cast(mask, dtype, device))
    out.backward(cast(g, dtype, device))
    return out.detach().float().cpu(), x.grad.float().cpu()


def reference(x, g, mask, heads, dtype):
    """torch on the CPU in float32, from the inputs rounded to ``dtype``."""
    rounded = [None if t is None else t.to(dtype).float() if t.is_floating_point() else t for t in (x, g, mask)]
    return forward_backward(attention("contiguous", heads), *rounded, torch.float32, "cpu")


def rel_error(got, ref):
    if not torch.isfinite(got).all():
        return math.inf
    return ((got - ref).abs().max() / ref.abs().max().clamp_min(1e-30)).item()


def compiled(fn, mode):
    if mode == "eager":
        return fn
    torch._dynamo.reset()  # a fresh cache: one function, many variants, would reach the recompile limit
    return torch.compile(fn, backend="hpu_backend")


class FusedOps(TorchDispatchMode):
    """Names of the Habana attention kernels that one eager call runs."""

    def __init__(self):
        super().__init__()
        self.seen = set()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        name = str(func.overloadpacket)
        if name.startswith("hpu.") and "sdpa" in name:
            self.seen.add(name)
        return func(*args, **(kwargs or {}))


class Report:
    def __init__(self):
        self.failures = []

    def line(self, label, ok, detail):
        print(f"{label:78s} {detail} {'OK' if ok else 'FAIL'}", flush=True)
        if not ok:
            self.failures.append(label)

    def error(self, label, exc):
        cause = " | ".join(line for line in str(exc).splitlines() if line.strip())[:300] or repr(exc)
        self.line(label, False, f"ERROR {type(exc).__name__}: {cause}")


def check_layouts(report, device, name, batch, heads, seq, dim, model_layout, kind):
    """Every layout, both dtypes, eager and compiled (compiled: the model's layout and contiguous copies)."""
    x, g, mask = inputs(batch, heads, seq, dim, kind, seed=1)
    for dtype in DTYPES:
        ref = reference(x, g, mask, heads, dtype)
        dt = str(dtype).replace("torch.", "")
        for mode in ("eager", "compiled"):
            for layout in LAYOUTS if mode == "eager" else (model_layout, "contiguous"):
                label = f"{name} {(batch, heads, seq, dim)} {layout} {kind or 'no mask'} {dt} {mode}"
                try:
                    got = forward_backward(compiled(attention(layout, heads), mode), x, g, mask, dtype, device)
                    err = [rel_error(a, b) for a, b in zip(got, ref)]
                    report.line(label, max(err) <= TOL[dtype], f"out {err[0]:.1e} grad {err[1]:.1e}")
                except Exception as exc:  # noqa: BLE001
                    report.error(label, exc)
        with FusedOps() as ops:
            attention(model_layout, heads)(cast(x, dtype, device), cast(mask, dtype, device))
        report.line(f"{name} {dt} eager runs {sorted(ops.seen)}", bool(ops.seen), "fused kernel")


def check_dropout(report, device, name, heads, seq, dim, layout, kind, n, p):
    """``n`` copies of one sample with dropout: torch's distribution and a backward on the forward's draws."""
    x1, _, mask1 = inputs(1, heads, seq, dim, kind, seed=2)
    if mask1 is not None:  # a real key padding, as in the models
        mask1 = key_mask(torch.tensor([max(1, (3 * seq) // 4)]), seq, kind, heads)
    for dtype in DTYPES:
        dt = str(dtype).replace("torch.", "")
        q, k, v = split_qkv(x1.to(dtype).float(), "contiguous", heads)
        scores = q @ k.transpose(-2, -1) / math.sqrt(dim)
        if mask1 is not None:
            scores = scores.masked_fill(~mask1, float("-inf")) if kind == "bool" else scores + mask1
        weights = scores.softmax(-1)
        variance = p / (1 - p) * (weights.square() @ v.square())  # (1, H, L, D)
        for mode in ("eager", "compiled"):
            label = f"{name} dropout {p} x{n} {(heads, seq, dim)} {layout} {kind or 'no mask'} {dt} {mode}"
            try:
                with_dropout = compiled(attention(layout, heads, p), mode)
                x = cast(x1, dtype, device).repeat(n, 1, 1).requires_grad_(True)
                mask = cast(None if mask1 is None else mask1.expand(n, -1, -1, -1).contiguous(), dtype, device)
                if mode == "eager":
                    with FusedOps() as ops:
                        out = with_dropout(x, mask)
                    kernels = sorted(ops.seen)
                else:
                    out, kernels = with_dropout(x, mask), "-"
                g = torch.randn(out.shape, generator=torch.Generator().manual_seed(3)).to(device=device,
                                                                                       dtype=out.dtype)
                out.backward(g)
                with torch.no_grad():
                    again = with_dropout(x, mask)
                    # a fresh copy: views of x[:1] hit the sliced-view bug (check_sliced_views)
                    mean_ref = compiled(attention(layout, heads), mode)(cast(x1, dtype, device), cast(mask1, dtype,
                                                                                                    device))
                draws, mean_ref = out.detach().float().cpu(), mean_ref.float().cpu()
                mean, var = draws.mean(0, keepdim=True), draws.var(0, keepdim=True)
                z = (mean - mean_ref) / (variance / n).sqrt().clamp_min(1e-30)
                rms_z = z.square().mean().sqrt().item()
                slope = ((mean * mean_ref).sum() / mean_ref.square().sum()).item()
                ratio = (var.sum() / variance.sum()).item()
                lhs = (v_part(x.grad.float().cpu(), heads) * v_part(x.detach().float().cpu(), heads)).sum((-1, -2))
                rhs = (draws * g.float().cpu()).sum((-1, -2))
                same_draws = ((lhs - rhs).abs().max() / rhs.square().mean().sqrt()).item()
                finite = bool(torch.isfinite(x.grad).all())
                fresh = not torch.equal(out.detach(), again)
                ok = (abs(slope - 1) <= 0.02 and rms_z <= 1.5 and 0.9 <= ratio <= 1.1 and same_draws <= 0.1
                      and finite and fresh)
                report.line(label, ok, f"mean slope {slope:.4f} rms z {rms_z:.2f} var ratio {ratio:.3f} "
                            f"bwd-vs-fwd draws {same_draws:.1e} finite grads {finite} new draws {fresh} "
                            f"kernels {kernels}")
            except Exception as exc:  # noqa: BLE001
                report.error(label, exc)


def probe_sliced_views():
    """The known Habana bug, each mode in a process of its own; printed, never counted. Run it before this process
    takes the card: a Gaudi card serves one process at a time."""
    for mode in ("eager", "compiled"):
        try:
            done = subprocess.run([sys.executable, "-c", SLICED_VIEW_PROBE, mode], env=dict(os.environ),
                                  capture_output=True, text=True, timeout=900)
            found = [line[6:] for line in done.stdout.splitlines() if line.startswith("PROBE ")]
            status, tail = done.returncode, " | ".join(done.stderr.strip().splitlines()[-3:])[-400:]
        except subprocess.TimeoutExpired as exc:
            found, status, tail = [], "timeout", str(exc)[-400:]
        known = sum(KNOWN_SLICED_VIEW_BUG in line for line in found)
        print(f"known Habana bug probe, sliced views after a whole tensor, {mode} (own process, not counted): "
              f"exit {status}; {'; '.join(found) or 'no result'}; {known} known-bug failures; stderr tail: {tail}",
              flush=True)


def record_fully_masked(report, device, name, batch, heads, seq, dim, layout, kind):
    """Sample 0 has no real key: print what each side returns for it; the other samples must match."""
    x, g, mask = inputs(batch, heads, seq, dim, kind, seed=4, fully_masked=True)
    for dtype in DTYPES:
        dt = str(dtype).replace("torch.", "")
        ref = reference(x, g, mask, heads, dtype)
        for mode in ("eager", "compiled"):
            label = f"{name} fully masked sample {(batch, heads, seq, dim)} {layout} {kind} {dt} {mode}"
            try:
                got = forward_backward(compiled(attention(layout, heads), mode), x, g, mask, dtype, device)
                err = [rel_error(a[1:], b[1:]) for a, b in zip(got, ref)]
                print(f"RECORD {label}: card out {describe(got[0][0])}, grad {describe(got[1][0])}; "
                      f"torch CPU out {describe(ref[0][0])}, grad {describe(ref[1][0])}", flush=True)
                report.line(f"{label}, other samples", max(err) <= TOL[dtype],
                            f"out {err[0]:.1e} grad {err[1]:.1e}")
            except Exception as exc:  # noqa: BLE001
                report.error(label, exc)


def describe(t):
    nan = int(torch.isnan(t).sum())
    finite = t[torch.isfinite(t)]
    peak = finite.abs().max().item() if finite.numel() else float("nan")
    return f"{nan}/{t.numel()} NaN, max|finite| {peak:.2e}"


def check_multihead(report, device, name, batch, seq, embed, heads):
    """nn.MultiheadAttention with a key padding mask, the way CBraMod calls it, eager and compiled; float32 and
    bfloat16 autocast, against float32 on the CPU."""
    torch.manual_seed(5)
    module = torch.nn.MultiheadAttention(embed, heads, batch_first=True)
    on_card = torch.nn.MultiheadAttention(embed, heads, batch_first=True).to(device)
    on_card.load_state_dict(module.state_dict())
    gen = torch.Generator().manual_seed(6)
    x, g = torch.randn(batch, seq, embed, generator=gen), torch.randn(batch, seq, embed, generator=gen)
    pad = torch.arange(seq) >= torch.randint(1, seq + 1, (batch,), generator=gen)[:, None]

    def self_attention(m):
        return lambda x, pad: m(x, x, x, key_padding_mask=pad, need_weights=False)[0]

    def run(fn, x, pad, autocast):
        x = x.detach().requires_grad_(True)
        with torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=autocast):
            out = fn(x, pad)
        out.backward(g.to(x.device, out.dtype))
        return out.detach().float().cpu(), x.grad.float().cpu()

    ref = run(self_attention(module), x.clone(), pad, False)
    for dtype in DTYPES:
        dt = "bfloat16 autocast" if dtype == torch.bfloat16 else "float32"
        for mode in ("eager", "compiled"):
            label = f"{name} nn.MultiheadAttention {(batch, seq, embed)} {heads} heads key padding {dt} {mode}"
            try:
                fn = compiled(self_attention(on_card), mode)
                got = run(fn, x.to(device), pad.to(device), dtype == torch.bfloat16)
                err = [rel_error(a, b) for a, b in zip(got, ref)]
                report.line(label, max(err) <= TOL[dtype], f"out {err[0]:.1e} grad {err[1]:.1e}")
            except Exception as exc:  # noqa: BLE001
                report.error(label, exc)


def run(device):
    """All checks on ``device``; returns the labels of the failed ones."""
    report = Report()
    for name, batch, heads, seq, dim, layout, kind in SHAPES:
        check_layouts(report, device, name, batch, heads, seq, dim, layout, kind)
    for name, batch, heads, seq, dim, layout, kind in SHAPES:
        check_dropout(report, device, name, heads, seq, dim, layout, kind, n=4096 if name == "tiny" else batch,
                      p=0.5 if name == "tiny" else 0.1)
    for name, batch, heads, seq, dim, layout, kind in SHAPES:
        for mask_kind in ("bool", "additive") if kind is None or name == "tiny" else (kind,):
            record_fully_masked(report, device, name, batch, heads, seq, dim, layout, mask_kind)
    check_multihead(report, device, "CBraMod space", 3000, 32, 100, 4)
    check_multihead(report, device, "CBraMod time", 3200, 30, 100, 4)
    return report.failures


if __name__ == "__main__":
    probe_sliced_views()  # before this process takes the card
    import habana_frameworks.torch.core  # noqa: F401

    import hpu_ops

    hpu_ops.enable()
    failed = run(torch.device("hpu"))
    print(f"{len(failed)} failed: {failed}" if failed else "all attention checks passed", flush=True)
    sys.exit(1 if failed else 0)
