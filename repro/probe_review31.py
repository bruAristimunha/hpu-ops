"""Gaudi probe of the #31 review fixes (CODEX_REVIEW.txt inputs), one case per process:
    PT_HPU_LAZY_MODE=0|1 python3 repro/probe_review31.py CASE [en]
``en`` runs hpu_ops.enable() first; without it the native op runs. Each case compares the card with torch on the CPU
(the same float32 inputs; float64 where named) and prints one ``PROBE {json}`` line with the CPU-fallback delta.
CASE: q_grad_nan_row | grad_to_nan | nan_inf_grad | out | layout | big_q_backward | compiled_q_range | compiled_infer
      | perf | conv_bf16_bias | conv_fp16_cancel | conv_bias_once | conv_pad_inf | conv_lazy_300 | conv_cl_weight"""
import json
import os
import sys
import time
import traceback

import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.hpu as hthpu
import torch
import torch.nn.functional as F
from habana_frameworks.torch.hpu.metrics import metric_global

case = sys.argv[1]
enable = len(sys.argv) > 2 and sys.argv[2] == "en"
if enable:
    import hpu_ops
    hpu_ops.enable()
LAZY = os.environ.get("PT_HPU_LAZY_MODE", "0") != "0"
HPU = torch.device("hpu")
torch.zeros(1, device=HPU).cpu()  # the device up before mark_step() (the CPU reference runs first)
NAN, INF = float("nan"), float("inf")


def mark():
    if LAZY:
        htcore.mark_step()


def fb():
    return dict(dict(metric_global("cpu_fallback").stats()).get("FallbackOps", {}) or {})


def cpu(t):
    return [cpu(v) for v in t] if isinstance(t, (list, tuple)) else (None if t is None else t.detach().cpu())


def same(a, b):
    """Exact match, NaN where torch has NaN."""
    if isinstance(a, list):
        return all(same(x, y) for x, y in zip(a, b))
    return a.shape == b.shape and bool(torch.equal(a.isnan(), b.isnan())) and bool(
        (a.double() - b.double()).nan_to_num(0.0).abs().max() <= 1e-6 * max(1.0, float(b.nan_to_num(0).abs().max())))


def both(f):
    """f on the CPU and on the card (same code, device argument)."""
    want = cpu(f(torch.device("cpu")))
    got = cpu(f(HPU))
    return got, want


def q_grad_nan_row(d):
    q = torch.tensor(0.25, device=d, requires_grad=True)
    y = torch.quantile(torch.tensor([[1.0, 3.0], [2.0, NAN]], device=d), q, dim=1)[0]
    g = torch.autograd.grad(y, q)[0]
    mark()
    return [y, g]


def grad_to_nan(d):
    x = torch.tensor([1.0, NAN, 2.0], device=d, requires_grad=True)
    y = torch.quantile(x, 0.5)
    g = torch.autograd.grad(y, x)[0]
    mark()
    return [y, g]


def nan_inf_grad(d):
    x = torch.tensor([NAN, INF], device=d, requires_grad=True)
    y = torch.nanquantile(x, 1.0, interpolation="lower")
    g = torch.autograd.grad(y, x)[0]
    mark()
    return [y, g]


def out(d):
    x = torch.tensor([3.0, NAN, 1.0, 2.0], device=d)
    o1, o2 = torch.empty(0, device=d), torch.empty(2, device=d)
    torch.quantile(x, 0.5, out=o1)
    torch.nanquantile(x, torch.tensor([0.25, 0.5], device=d), out=o2)
    mark()
    return [o1, o2]


def layout(d):
    torch.manual_seed(0)
    r = torch.quantile(torch.randn(2, 3, 5).to(d), torch.tensor([0.25, 0.75], device=d), dim=1)
    mark()
    return [r.movedim(0, -1).reshape(-1), torch.tensor(r.stride(), dtype=torch.float32)]


def big_q_backward(d):
    torch.manual_seed(0)
    x = torch.randn(64, 4096).to(d).requires_grad_()
    q = torch.linspace(0, 1, 257).to(d)
    y = torch.quantile(x, q, dim=1)
    g = torch.autograd.grad(y.sum(), x)[0]
    mark()
    return [y, g]


def run_case():
    info = {"case": case, "lazy": LAZY, "enable": enable, "bridge": "1.21.4"}
    f0 = fb()
    try:
        if case in ("q_grad_nan_row", "grad_to_nan", "nan_inf_grad", "out", "layout", "big_q_backward"):
            got, want = both(globals()[case])
            err = max(float((g.double() - w.double()).nan_to_num(0).abs().max()) for g, w in zip(got, want))
            info.update(ok=same(got, want), max_abs=err, got=[g.reshape(-1)[:6].tolist() for g in got],
                        want=[w.reshape(-1)[:6].tolist() for w in want])
            if case == "big_q_backward":
                info["mem_peak_mb"] = hthpu.memory_stats().get("MaxInUse", 0) / 2 ** 20
        elif case.startswith("compiled") and LAZY:
            info.update(ok=None, skipped="hpu_backend runs in eager mode (lazy: native fails GetHbLazyTensor too)")
        elif case == "compiled_q_range":
            f = torch.compile(lambda x, q: torch.quantile(x, q, interpolation="nearest"), backend="hpu_backend",
                              fullgraph=True)
            x = torch.tensor([2.0, 6.0], device=HPU)
            ok_val = f(x, torch.tensor(0.25, device=HPU)).cpu().item()
            try:  # Gaudi has no _assert_async kernel: hpu_ops gives NaN there (raising needs a CPU fallback per call)
                bad = f(x, torch.tensor(1.1, device=HPU)).cpu().item()
                info.update(ok=ok_val == 2.0 and bad != bad, in_range=ok_val, out_of_range=bad)
            except RuntimeError as e:
                info.update(ok=ok_val == 2.0 and "range" in str(e), in_range=ok_val, raised=str(e)[:120])
        elif case == "compiled_perf":  # the cost of the run-time q range check (_assert_async falls back to the CPU)
            x = torch.randn(8, 300, device=HPU)
            q = torch.tensor([0.1, 0.5, 0.9], device=HPU)
            rows = {}
            for label in ("checked", "unchecked"):
                if label == "unchecked":
                    torch._assert_async = lambda *a, **k: None
                torch._dynamo.reset()
                f = torch.compile(lambda t, qq: torch.quantile(t, qq, dim=1), backend="hpu_backend", fullgraph=True)
                keep = [f(x, q) for _ in range(5)]
                hthpu.synchronize()
                f0 = fb()
                t = time.perf_counter()
                keep += [f(x, q) for _ in range(30)]
                hthpu.synchronize()
                rows[label] = {"ms": (time.perf_counter() - t) / 30 * 1e3,
                               "fb_per_call": {k: (v - f0.get(k, 0)) / 30 for k, v in fb().items() if v - f0.get(k, 0)}}
            info.update(ok=True, rows=rows)
        elif case == "compiled_infer":
            outs = []
            for fn in (torch.quantile, torch.nanquantile):
                f = torch.compile(lambda t, fn=fn: fn(t, 0.5), backend="hpu_backend", fullgraph=True)
                with torch.inference_mode():
                    outs.append(f(torch.tensor([3.0, NAN, 1.0, 2.0], device=HPU)).cpu().item())
            info.update(ok=str(outs) == str([NAN, 2.0]), got=outs, want=[NAN, 2.0])
        elif case == "perf":
            torch.manual_seed(0)
            x = torch.randn(8, 300).to(HPU)
            keep = []
            for _ in range(5):
                keep.append(torch.quantile(x, 0.9, dim=1))
                mark()
            hthpu.synchronize()
            t = time.perf_counter()
            for _ in range(30):
                keep.append(torch.quantile(x, 0.9, dim=1))
                mark()
            hthpu.synchronize()
            info.update(ok=True, ms=(time.perf_counter() - t) / 30 * 1e3)
            xg = torch.randn(8, 300).to(HPU).requires_grad_()
            for i in range(25):
                if i == 5:
                    hthpu.synchronize()
                    t = time.perf_counter()
                y = torch.quantile(xg, torch.tensor([0.1, 0.5, 0.9], device=HPU), dim=1)
                keep.append(torch.autograd.grad(y.sum(), xg)[0])
                mark()
            hthpu.synchronize()
            info["ms_fwd_bwd_3q"] = (time.perf_counter() - t) / 20 * 1e3
        elif case.startswith("conv_"):
            conv_case(info)
        else:
            raise SystemExit(f"unknown case {case}")
    except Exception as e:  # noqa: BLE001
        info.update(ok=False, raised=repr(e)[:300], at=traceback.format_exc().splitlines()[-3:-1])
    f1 = fb()
    info["fb"] = {k: f1[k] - f0.get(k, 0) for k in f1 if f1[k] - f0.get(k, 0)}
    if enable:
        info["calls"] = dict(hpu_ops._STATE["calls"])
    print("PROBE " + json.dumps(info, default=str), flush=True)


def conv_case(info):
    if case == "conv_bf16_bias":  # what the card's own bf16 conv does with a bias (k = 2: native everywhere)
        rows = {}
        for dt in (torch.bfloat16, torch.float16):
            x = torch.ones(1, 1, 2, dtype=dt)
            w = torch.tensor([[[256.0, 0.75]]], dtype=dt)
            b = torch.tensor([0.5], dtype=dt)
            rows[str(dt)] = {"card": [F.conv1d(x.to(HPU), w.to(HPU)).cpu().item(),
                                      F.conv1d(x.to(HPU), w.to(HPU), b.to(HPU)).cpu().item()],
                             "cpu": [F.conv1d(x, w).item(), F.conv1d(x, w, b).item()]}
            # the 512-tap cancellation: native card (k > 256 splits with hpu_ops) and a native k = 256 analogue
            x2 = torch.full((1, 1, 256), 256.0, dtype=dt)
            w2 = torch.cat([torch.ones(128), -torch.ones(128)]).to(dt).view(1, 1, 256)
            rows[str(dt)]["cancel256_card"] = F.conv1d(x2.to(HPU), w2.to(HPU)).cpu().item()
        info.update(ok=True, rows=rows)
        return
    if case == "conv_fp16_cancel":
        def f(d):
            out = []
            for dt in (torch.float16, torch.bfloat16):
                x = torch.full((1, 1, 512), 256.0, dtype=dt, device=d)
                w = torch.cat([torch.ones(256), -torch.ones(256)]).to(dt).view(1, 1, 512).to(d)
                out.append(F.conv1d(x, w).float())
                x1, w1 = torch.ones(1, 1, 512, dtype=dt, device=d), w.clone()
                w1[0, 0, 0] = 2
                out.append(F.conv1d(x1, w1).float())
            mark()
            return out
    elif case == "conv_bias_once":
        def f(d):
            x = torch.ones(1, 1, 512, device=d)
            w = torch.cat([torch.ones(256), -torch.ones(256)]).view(1, 1, 512).to(d)
            y = F.conv1d(x, w, torch.tensor([1e-5], device=d))
            mark()
            return [y]
    elif case == "conv_pad_inf":
        def f(d):
            x = torch.ones(1, 1, 1, device=d)
            w = torch.ones(1, 1, 257)
            w[0, 0, 0] = INF
            y = F.conv1d(x, w.to(d), stride=513, padding=256)
            mark()
            return [y]
    elif case == "conv_lazy_300":  # the review's 300 x 300 over 310 x 310 (lazy: 44 x 44 tail before)
        def f(d):
            torch.manual_seed(0)
            x = torch.randn(1, 1, 310, 310).to(d).requires_grad_()
            w = (torch.randn(1, 1, 300, 300) / 300).to(d).requires_grad_()
            y = F.conv2d(x, w)
            gx, gw = torch.autograd.grad(y.pow(2).sum(), (x, w))
            mark()
            return [y, gx, gw]
    elif case == "conv_cl_weight":
        def f(d):
            torch.manual_seed(0)
            x = torch.randn(2, 2, 3, 1).to(d)
            w = torch.randn(3, 2, 1, 257).to(d).contiguous(memory_format=torch.channels_last)
            y = F.conv2d(x, w, padding=(0, 257), stride=(1, 513))
            mark()
            return [y, torch.tensor([float(y.is_contiguous(memory_format=torch.channels_last))])]
    else:
        raise SystemExit(f"unknown case {case}")
    got, want = both(f)
    rel = [float((g.double() - w.double()).nan_to_num(0).abs().max() / max(1e-30, float(w.nan_to_num(0).abs().max())))
           for g, w in zip(got, want)]
    ok = all(torch.equal(g.isnan(), w.isnan()) for g, w in zip(got, want)) and max(rel) < 1e-3
    info.update(ok=ok, rel=rel, got=[g.reshape(-1)[:4].tolist() for g in got],
                want=[w.reshape(-1)[:4].tolist() for w in want])


run_case()
