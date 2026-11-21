"""Gaudi2: hpu_ops' exact sort family (on by default) against torch on the CPU, and its per-call cost against
Gaudi's own sort (``extrema.internal_sort`` / the native op with the rewrite skipped) in the same process.
    PT_HPU_LAZY_MODE=0|1 python3 repro/probe_exact_sort.py [--compile] [--no-time]
Correctness: sort (stable / not, ascending / descending, dim -1 and 0), argsort, msort, topk, kthvalue at
n = 1 .. 5000 in float32 and bfloat16, rows with NaN, +-inf, -0.0, ties, all-NaN and all-inf rows; values exact,
indices torch's for stable sorts, else in range and picking the values; the backward of sort/topk values against the
CPU. ``--compile``: the same through torch.compile(backend="hpu_backend") with the rewrite allowed under tracing (the
installed kernel declines traced calls; this measures whether it could take them). CPU fallbacks counted."""
import os
import statistics
import sys
import time

import hpu_ops  # noqa: F401  enable() before the first HPU op
import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.hpu as hthpu
import torch
from habana_frameworks.torch.hpu.metrics import metric_global
from hpu_ops import extrema

hpu_ops.enable()
LAZY = os.environ.get("PT_HPU_LAZY_MODE", "1") != "0"
COMPILE = "--compile" in sys.argv
INF, NAN = float("inf"), float("nan")
if COMPILE:
    torch._dynamo.config.cache_size_limit = 1024
    extrema._sortable = lambda x, dim: x.is_floating_point() and x.dim() > 0 and x.numel() > 0


def sync():
    if LAZY:
        htcore.mark_step()
    hthpu.synchronize()


def fb():
    return dict(dict(metric_global("cpu_fallback").stats()).get("FallbackOps", {}) or {})


def data(n, dt, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(8, n, generator=g)
    x[:, ::7], x[:, 3::5], x[:, 1::4], x[:, 2::6] = INF, -INF, NAN, -0.0
    x[:, 5::9] = 0.5
    x[1], x[2] = NAN, INF
    x[3, ::2] = -INF
    x[4] = torch.randn(n, generator=g)  # a clean row
    return x.to(dt)


def picks(x, d, got_v, got_i, want_v):
    n = x.shape[d]
    if not bool(((got_i >= 0) & (got_i < n)).all()):
        return False
    return equal(x.gather(d, got_i), want_v) and equal(got_v, want_v)


def equal(a, b):
    return a.shape == b.shape and torch.equal(a.isnan(), b.isnan()) and torch.equal(a.nan_to_num(7.0),
                                                                                    b.nan_to_num(7.0))


CASES = {
    "sort_stable": (lambda t, d, desc: tuple(t.sort(dim=d, descending=desc, stable=True)), True),
    "sort": (lambda t, d, desc: tuple(t.sort(d, desc)), False),
    "argsort_stable": (lambda t, d, desc: (None, t.argsort(dim=d, descending=desc, stable=True)), True),
    "msort": (lambda t, d, desc: (t.msort(), None), None),
    "topk3": (lambda t, d, desc: tuple(t.topk(min(3, t.shape[d]), d, largest=desc)), False),
    "kthvalue": (lambda t, d, desc: tuple(t.kthvalue(min(2, t.shape[d]), d, keepdim=True)), False),
}


def run(f, x):
    fn = torch.compile(f, backend="hpu_backend", dynamic=False) if COMPILE else f
    return fn(x)


bad = total = 0
f0 = fb()
for dt in (torch.float32, torch.bfloat16):
    for n in (1, 3, 22, 64, 65, 300, 1000, 1024, 5000):
        x = data(n, dt, n)
        for d in (-1, 0):
            xd = x if d == -1 else x.mT.contiguous()
            for desc in (False, True):
                for name, (f, exact_index) in CASES.items():
                    if name == "msort" and (d != 0 or desc):
                        continue
                    want = f(xd, d, desc)
                    try:
                        got = run(lambda t: f(t, d, desc), xd.to("hpu"))
                        sync()
                        gv = None if got[0] is None else got[0].cpu()
                        gi = None if got[1] is None else got[1].cpu()
                        if gv is None:
                            ok = torch.equal(gi, want[1])
                        elif gi is None:
                            ok = equal(gv, want[0])
                        elif exact_index:
                            ok = equal(gv, want[0]) and torch.equal(gi, want[1])
                        else:
                            ok = picks(xd, d, gv, gi, want[0])
                    except Exception as e:  # noqa: BLE001
                        ok = f"ERROR {type(e).__name__}: {str(e)[:120]}"
                    total += 1
                    if ok is not True:
                        bad += 1
                        print(f"BAD {name} {dt} n={n} d={d} desc={desc} {ok}", flush=True)

# gradients of the values (sort stable / topk) against the CPU
for dt in (torch.float32, torch.bfloat16):
    for n in (3, 22, 1000):
        x = data(n, dt, 7)
        w = torch.linspace(0.5, 1.5, x.numel()).reshape(x.shape).to(dt)
        for name, f in (("sort_stable", lambda t: t.sort(dim=-1, stable=True).values),
                        ("topk", lambda t: t.topk(min(3, n), -1).values)):
            xc = x.clone().requires_grad_()
            (f(xc).nan_to_num(0.0) * w[..., :f(xc).shape[-1]]).sum().backward()
            xh = x.to("hpu").requires_grad_()
            try:
                g = run(lambda t: f(t), xh) if COMPILE else f(xh)
                (g.nan_to_num(0.0) * w[..., :g.shape[-1]].to("hpu")).sum().backward()
                sync()
                ok = torch.equal(xh.grad.cpu(), xc.grad) if name == "sort_stable" else torch.equal(
                    xh.grad.cpu().sum(-1), xc.grad.sum(-1))
            except Exception as e:  # noqa: BLE001
                ok = f"ERROR {type(e).__name__}: {str(e)[:120]}"
            total += 1
            if ok is not True:
                bad += 1
                print(f"BAD grad {name} {dt} n={n} {ok}", flush=True)
print(f"CORRECT mode={'lazy' if LAZY else 'eager'} compile={COMPILE} bad={bad} of {total} fallbacks={dict((k, v - f0.get(k, 0)) for k, v in fb().items() if v - f0.get(k, 0))}", flush=True)
if COMPILE or "--no-time" in sys.argv:
    sys.exit(0)


def bench(f, x, reps=30):
    for _ in range(3):
        out = f(x)
    sync()
    keep = []
    t = time.perf_counter()
    for _ in range(reps):
        keep.append(f(x))
        if LAZY:
            htcore.mark_step()
    sync()
    del out
    return 1e3 * (time.perf_counter() - t) / reps


FORMS = {
    "gaudi_sort": lambda t: extrema.internal_sort(t, -1),
    "exact_sort": lambda t: tuple(t.sort(-1)),
    "gaudi_stable": lambda t: extrema.native(torch.ops.aten.sort.stable)(t, stable=True, dim=-1),
    "exact_stable": lambda t: tuple(t.sort(dim=-1, stable=True)),
    "gaudi_topk": lambda t: extrema.native(torch.ops.aten.topk.default)(t, 5, -1),
    "exact_topk": lambda t: tuple(t.topk(5, -1)),
    "exact_sort_fwd_bwd": None,
}


def fwd_bwd(exact):
    def f(t):
        t = t.detach().requires_grad_()
        v = t.sort(-1).values if exact else extrema.internal_sort(t, -1)[0]
        return torch.autograd.grad(v.sum(), t)[0]
    return f


for shape in ((32, 22), (32, 64), (32, 22, 1000), (256, 1000)):
    x = torch.randn(*shape).to("hpu")
    times = {k: [] for k in FORMS}
    times["gaudi_sort_fwd_bwd"] = []
    for _ in range(5):
        for name, f in FORMS.items():
            if f is None:
                continue
            if "stable" in name and shape[-1] > 64:
                continue
            times[name].append(bench(f, x))
        times["exact_sort_fwd_bwd"].append(bench(fwd_bwd(True), x))
        times["gaudi_sort_fwd_bwd"].append(bench(fwd_bwd(False), x))
    print(f"TIME {'lazy' if LAZY else 'eager'} {str(shape):14s} " + " ".join(
        f"{k}={statistics.median(v):.3f}" for k, v in times.items() if v), flush=True)
print("DONE", flush=True)
