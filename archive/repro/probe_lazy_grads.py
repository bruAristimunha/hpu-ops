"""Gaudi probe: is the lazy-mode gradient error of a model real, and which module first gets it wrong.
    PT_HPU_LAZY_MODE=1 python3 repro/probe_lazy_grads.py MODEL [--checks] [--bisect] [--noplugins]
--checks  harness checks: loss vs CPU float64; grads read as float32 .cpu() (no .double() on the card); a second
          identical fwd+bwd; a mark_step between forward and backward.
--bisect  per-module forward output and output-gradient error vs CPU float64 (tensor hooks), in forward order.
"""
import argparse
import os
import sys
import types

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests"))
import model_parity as M  # noqa: E402
import habana_frameworks.torch.core as htcore  # noqa: E402


def ctx_and_setup(plugins):
    from spd_diag import protocol as P
    from spd_diag.cache import open_cache
    if plugins:
        P._enable_device("hpu")
    cache = open_cache(M.DATASET)
    sd = cache.subject(cache.subjects[0], "X")
    X = torch.as_tensor(sd.X[:M.BATCH * M.MAX_BATCHES], dtype=torch.float32)
    X = X / X.std()
    y = torch.as_tensor(sd.y[:M.BATCH * M.MAX_BATCHES]).long()
    return types.SimpleNamespace(P=P, info=cache.info, X=X, y=y, nb=max(1, len(X) // M.BATCH), steps=1)


def grads32(model):
    return {n: (None if p.grad is None else p.grad.detach().cpu().double()) for n, p in model.named_parameters()}


def report(tag, g, ref, floor):
    e = M.rel_errs(g, ref, floor)
    worst = sorted(e.items(), key=lambda kv: -(kv[1] if kv[1] == kv[1] else 1e30))[:4]
    print(f"CHECK {tag}: max_rel={max(e.values()):.3g} n>1e-3={sum(v > 1e-3 for v in e.values())}/{len(e)} worst="
          + ", ".join(f"{k}={v:.3g}" for k, v in worst), flush=True)


def run(model, xb, yb, mark_between=False):
    model.zero_grad(set_to_none=True)
    loss = M.forward_loss(model, xb, yb)
    if mark_between and M.on_hpu(xb):
        htcore.mark_step()
    loss.backward()
    if M.on_hpu(xb):
        M.sync()
    return float(loss.detach().cpu())


def checks(name, ctx):
    ref_m, xin, _ = M.make(name, ctx, M.CPU, torch.float64)
    xs, ys = M.batches(ctx, xin, M.CPU)
    with M.default_dtype(torch.float64):
        l64 = run(ref_m, xs[0], ys[0])
    g64 = M.grads_of(ref_m)
    floor = M.abs_floor(g64)
    model, xin, _ = M.make(name, ctx, M.HPU)
    xs, ys = M.batches(ctx, xin, M.HPU)
    l1 = run(model, xs[0], ys[0])
    print(f"CHECK loss hpu={l1:.7f} cpu64={l64:.7f}", flush=True)
    report("harness .double() read", M.grads_of(model), g64, floor)
    report("float32 .cpu() read", grads32(model), g64, floor)
    run(model, xs[0], ys[0])
    report("2nd identical fwd+bwd", grads32(model), g64, floor)
    run(model, xs[0], ys[0], mark_between=True)
    report("mark_step between fwd and bwd", grads32(model), g64, floor)


def capture(model):
    """Forward hooks: each module's tensor output and the gradient that reaches it, keyed by call order."""
    store, handles = [], []

    def hook(mod, inp, out):
        t = out[0] if isinstance(out, (tuple, list)) else out
        if not isinstance(t, torch.Tensor) or not t.is_floating_point():
            return
        rec = {"name": mod._probe_name, "out": t.detach(), "grad": None}
        store.append(rec)
        if t.requires_grad:
            def gh(g, rec=rec):
                rec["grad"] = g.detach()
            t.register_hook(gh)

    for n, m in model.named_modules():
        m._probe_name = n or "<root>"
        handles.append(m.register_forward_hook(hook))
    return store, handles


def rel(a, b):
    a, b = a.detach().cpu().double(), b.detach().cpu().double()
    if a.is_complex() or b.is_complex():
        a, b = torch.view_as_real(a), torch.view_as_real(b)
    if a.shape != b.shape:
        return float("inf")
    return float((a - b).abs().max() / max(float(b.abs().max()), 1e-12))


def bisect(name, ctx):
    recs = {}
    for tag, dev, dt in (("cpu64", M.CPU, torch.float64), ("hpu", M.HPU, torch.float32)):
        model, xin, _ = M.make(name, ctx, dev, dt)
        xs, ys = M.batches(ctx, xin, dev)
        store, handles = capture(model)
        with M.default_dtype(dt):
            run(model, xs[0], ys[0])
        for h in handles:
            h.remove()
        recs[tag] = (store, M.grads_of(model))
    (s64, g64), (sh, gh) = recs["cpu64"], recs["hpu"]
    report("with hooks", gh, g64, M.abs_floor(g64))
    for i, (a, b) in enumerate(zip(sh, s64)):
        ge = rel(a["grad"], b["grad"]) if a["grad"] is not None and b["grad"] is not None else None
        print(f"BISECT {i:3d} {a['name']:<40} out={rel(a['out'], b['out']):.3g} "
              f"grad_out={'-' if ge is None else f'{ge:.3g}'} shape={tuple(a['out'].shape)}", flush=True)


def split(name, ctx, where):
    """One HPU fwd+bwd with a mark_step after each module's forward (F) and/or at each module output's gradient (B)."""
    model, xin, _ = M.make(name, ctx, M.HPU)
    xs, ys = M.batches(ctx, xin, M.HPU)
    names = []

    def fwd(mod, inp, out):
        t = out[0] if isinstance(out, (tuple, list)) else out
        if "F" in where:
            htcore.mark_step()
        only = where.split("=", 1)[1].split(",") if "=" in where else None
        if "B" in where and isinstance(t, torch.Tensor) and t.requires_grad and (only is None or
                                                                                mod._probe_name in only):
            def gh(g, n=mod._probe_name):
                names.append(n)
                htcore.mark_step()
            t.register_hook(gh)

    for n, m in model.named_modules():
        m._probe_name = n or "<root>"
        m.register_forward_hook(fwd)
    try:
        run(model, xs[0], ys[0])
        ref, _, _ = M.make(name, ctx, M.CPU, torch.float64)
        xr, yr = M.batches(ctx, M.make(name, ctx, M.CPU, torch.float64)[1], M.CPU)
        with M.default_dtype(torch.float64):
            run(ref, xr[0], yr[0])
        g64 = M.grads_of(ref)
        report(f"split {where}", grads32(model), g64, M.abs_floor(g64))
        print("CHECK backward hook order:", ",".join(names), flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"CHECK split {where}: ERROR {repr(e)[:160]} after backward hooks {names[-3:]}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--checks", action="store_true")
    ap.add_argument("--bisect", action="store_true")
    ap.add_argument("--noplugins", action="store_true")
    ap.add_argument("--split", default="", help="F, B or FB: mark_steps between modules (forward, backward)")
    a = ap.parse_args()
    ctx = ctx_and_setup(not a.noplugins)
    from spd_diag.models import MODELS
    name = M.resolve(a.model, MODELS)
    print("PROBE", name, "lazy", M.LAZY, "plugins", not a.noplugins, flush=True)
    if a.checks:
        checks(name, ctx)
    if a.bisect:
        bisect(name, ctx)
    if a.split:
        split(name, ctx, a.split)


if __name__ == "__main__":
    main()
