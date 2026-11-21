"""Gaudi2 checks of the non-linalg review fixes (gate/code-review-hpu-ops.txt), each with the review's input.

    PT_HPU_LAZY_MODE=0|1 python3 repro/probe_review_rest.py [--only name,...]

One ``CHECK {json}`` line per case: ``ok`` (the card matches the CPU reference), ``detail``, and the CPU fallbacks
of the call (bridge ``cpu_fallback`` metric). Compiled cases run in eager mode only (in lazy mode hpu_ops runs
``torch.compile`` uncompiled). ``import`` runs in fresh processes."""
import argparse
import json
import os
import subprocess
import sys

LAZY = os.environ.get("PT_HPU_LAZY_MODE", "1") != "0"


def check_import():
    """(2) import hpu_ops under a bf16 default dtype and an hpu default device: no error, the device not initialised
    before enable()."""
    out = {}
    for setup in ("torch.set_default_dtype(torch.bfloat16)", "torch.set_default_device('hpu')"):
        code = ("import torch, habana_frameworks.torch.core, habana_frameworks.torch.hpu as h; " + setup +
                "; import hpu_ops; print(h.is_initialized())")
        done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        out[setup] = done.stdout.strip().splitlines()[-1:] if done.returncode == 0 else done.stderr.strip()[-200:]
    ok = all(v == ["False"] for v in out.values())
    return ok, out


import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import habana_frameworks.torch.core as htcore  # noqa: E402
from habana_frameworks.torch.hpu.metrics import metric_global  # noqa: E402

import hpu_ops  # noqa: E402

hpu_ops.enable()
HPU = torch.device("hpu")
NAN, INF = float("nan"), float("inf")


def fallbacks():
    return sum((dict(dict(metric_global("cpu_fallback").stats()).get("FallbackOps", {}) or {})).values())


def sync(x):
    if LAZY:
        htcore.mark_step()
    return x


def cpu(t):
    if isinstance(t, (tuple, list)):
        return [cpu(v) for v in t]
    return t.detach().cpu() if isinstance(t, torch.Tensor) else t


def close(got, want, atol=1e-4):
    if isinstance(want, (tuple, list)):
        return all(close(g, w, atol) for g, w in zip(got, want))
    if got.shape != want.shape or got.dtype != want.dtype:
        return False
    if want.is_floating_point():
        return bool(torch.equal(torch.isnan(got), torch.isnan(want))) and torch.allclose(
            got.double(), want.double(), atol=atol, rtol=1e-4, equal_nan=True)
    return torch.equal(got, want)


def compiled(f):
    torch._dynamo.reset()
    return torch.compile(f, backend="hpu_backend", dynamic=False)


def check_conv():
    """(5) compiled under inference_mode: padding='same' and unbatched large kernels (natively one 351-wide conv,
    which Gaudi2's MME rejects)."""
    torch.manual_seed(0)
    cases = {"conv1d k351 same": (nn.Conv1d(1, 2, 351, padding="same"), torch.randn(2, 1, 400)),
             "conv1d k350 same": (nn.Conv1d(1, 2, 350, padding="same"), torch.randn(2, 1, 400)),
             "conv2d (1,301) same": (nn.Conv2d(1, 4, (1, 301), padding="same"), torch.randn(2, 1, 3, 400)),
             "conv1d k351 unbatched": (nn.Conv1d(1, 2, 351), torch.randn(1, 400)),
             "conv2d (351,1) unbatched": (nn.Conv2d(1, 4, (351, 1)), torch.randn(1, 400, 3))}
    res = {}
    for name, (layer, x) in cases.items():
        with torch.no_grad():
            want = layer(x)
        dev = layer.to(HPU)
        try:
            with torch.inference_mode():
                got = cpu(sync(compiled(dev) if not LAZY else dev)(x.to(HPU)))
            res[name] = bool(close(got, want, 1e-3))
        except Exception as e:  # noqa: BLE001
            res[name] = f"{type(e).__name__}: {str(e)[:120]}"
        layer.cpu()
    return all(v is True for v in res.values()), res


def check_upsample():
    """(6) compiled under inference_mode, 9920 -> 10000 (NeuroPose): error against CPU float64, within 1.5x of the CPU
    float32 kernel's (native: 3.4e-2). Lazy mode runs it uncompiled under inference_mode, where Habana's own HPU
    kernel is registered and hpu-ops' backend kernel is not (not this fix)."""
    x = torch.randn(2, 3, 9920)
    want = F.interpolate(x.double(), size=10000, mode="linear")
    f = lambda t: F.interpolate(t, size=10000, mode="linear")  # noqa: E731
    with torch.inference_mode():
        got = cpu(sync((compiled(f) if not LAZY else f)(x.to(HPU))))
    err = float((got.double() - want).abs().max() / want.abs().max())
    cpu32 = float((f(x).double() - want).abs().max() / want.abs().max())  # CPU float32's own error
    return err <= 1.5 * cpu32, {"rel_err": err, "cpu_float32_rel_err": cpu32}


def check_zero_d():
    """(12) a CPU 0-d first operand: torch.maximum(torch.tensor(0.), x_hpu) and its relatives, grad and inference."""
    x = torch.tensor([-1.0, NAN, 2.0, INF])
    calls = {"maximum": lambda c, d: torch.maximum(c, d), "minimum": lambda c, d: torch.minimum(c, d),
             "max(a, b)": lambda c, d: torch.max(c, d), "fmax": lambda c, d: torch.fmax(c, d),
             "fmin": lambda c, d: torch.fmin(c, d), "clamp(cpu0d, min=d)": lambda c, d: torch.clamp(c, min=d),
             "clamp_min(cpu0d, d)": lambda c, d: torch.clamp_min(c, d)}
    res = {}
    for name, f in calls.items():
        want = f(torch.tensor(0.0), x)
        for mode in ("grad", "inference"):
            try:
                if mode == "inference":
                    with torch.inference_mode():
                        got = sync(f(torch.tensor(0.0), x.to(HPU)))
                else:
                    got = sync(f(torch.tensor(0.0), x.to(HPU).requires_grad_()))
                res[f"{name} {mode}"] = got.device.type == "hpu" and close(cpu(got), want)
            except Exception as e:  # noqa: BLE001
                res[f"{name} {mode}"] = f"{type(e).__name__}: {str(e)[:100]}"
    return all(v is True for v in res.values()), res


def check_functorch():
    """(7) torch.func and forward-mode AD of max() / median() / inf-norms."""
    import torch.autograd.forward_ad as fwAD

    x = torch.tensor([[1.0, 3.0, 2.0], [0.5, 4.0, -1.0]])
    cases = {"func.grad(max)": lambda t: torch.func.grad(lambda r: r.max())(t),
             "func.grad(median)": lambda t: torch.func.grad(lambda r: r.median())(t),
             "func.grad(vector_norm inf)": lambda t: torch.func.grad(lambda r: torch.linalg.vector_norm(r, INF))(t),
             "vmap(grad(max))": lambda t: torch.func.vmap(torch.func.grad(lambda r: r.max()))(t)}

    def fwd(f, t):
        with fwAD.dual_level():
            return fwAD.unpack_dual(f(fwAD.make_dual(t, torch.ones_like(t)))).tangent

    cases["forward AD max"] = lambda t: fwd(lambda d: d.max(), t.clone().requires_grad_())
    cases["forward AD vector_norm inf"] = lambda t: fwd(lambda d: torch.linalg.vector_norm(d, INF, -1),
                                                        t.clone().requires_grad_())
    res = {}
    for name, f in cases.items():
        try:
            res[name] = close(cpu(sync(f(x.to(HPU)))), f(x))
        except Exception as e:  # noqa: BLE001
            res[name] = f"{type(e).__name__}: {str(e)[:100]}"
    return all(v is True for v in res.values()), res


def check_clamp_inplace():
    """clamp_ of an int tensor with float tensor bounds raises torch's cast error."""
    res = {}
    for mode in ("grad", "inference"):
        try:
            if mode == "inference":
                with torch.inference_mode():
                    sync(torch.arange(5, device=HPU).clamp_(min=torch.full((5,), 1.5, device=HPU)))
            else:
                sync(torch.arange(5, device=HPU).clamp_(min=torch.full((5,), 1.5, device=HPU)))
            res[mode] = "no error"
        except RuntimeError as e:
            res[mode] = "can't be cast" in str(e)
    return all(v is True for v in res.values()), res


def check_expanded_index():
    """(15) leaf = zeros(3, 1).expand(3, 4).requires_grad_(); leaf[:, [1, 3, 3]].sum().backward(): leaf.grad rows
    [0, 1, 0, 2] (was [3, 0, 0, 0]); CPU index tensor (the rewritten case) and a device one."""
    res = {}
    for where in ("cpu index", "hpu index"):
        idx = torch.tensor([1, 3, 3])
        idx = idx if where == "cpu index" else idx.to(HPU)
        leaf = torch.zeros(3, 1, device=HPU).expand(3, 4).requires_grad_()
        (leaf[:, idx] * torch.tensor([1.0, 10.0, 100.0], device=HPU)).sum().backward()
        sync(leaf.grad)
        res[where] = cpu(leaf.grad)[0].tolist()
        base = torch.randn(3, 1, device=HPU, requires_grad=True)  # a non-leaf expand: the base's gradient
        (base.expand(3, 4)[:, idx] * 2).sum().backward()
        sync(base.grad)
        res[where + " base"] = cpu(base.grad).flatten().tolist()
    ok = all(res[k] == [0.0, 1.0, 0.0, 110.0] for k in ("cpu index", "hpu index")) and all(
        res[k] == [6.0, 6.0, 6.0] for k in ("cpu index base", "hpu index base"))
    return ok, res


def check_compiled_index():
    """efficiency: compiled x[:, idx] with a CPU index tensor and F.unfold, values and gradients."""
    idx = torch.tensor([0, 2, 3])
    f = lambda t: (t[:, idx] ** 2).sum() + F.unfold(t[None, None], (2, 2), padding=1).sum()  # noqa: E731
    x = torch.randn(5, 4)
    xc = x.clone().requires_grad_()
    f(xc).backward()
    xd = x.to(HPU).requires_grad_()
    out = (compiled(f) if not LAZY else f)(xd)
    out.backward()
    sync(xd.grad)
    return close(cpu(xd.grad), xc.grad, 1e-4), {"loss": [float(cpu(out)), float(f(x))]}


CHECKS = {"import": check_import, "conv": check_conv, "upsample": check_upsample, "zero_d": check_zero_d,
          "functorch": check_functorch, "clamp_inplace": check_clamp_inplace, "expanded_index": check_expanded_index,
          "compiled_index": check_compiled_index}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    names = a.only.split(",") if a.only else list(CHECKS)
    for name in names:
        fb = fallbacks()
        try:
            ok, detail = CHECKS[name]()
        except Exception as e:  # noqa: BLE001
            ok, detail = False, f"raised {type(e).__name__}: {str(e)[:200]}"
        print("CHECK " + json.dumps({"case": name, "lazy": LAZY, "ok": ok, "detail": detail,
                                     "fallbacks": fallbacks() - fb}, default=str), flush=True)


if __name__ == "__main__":
    main()
