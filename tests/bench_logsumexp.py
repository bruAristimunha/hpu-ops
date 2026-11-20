"""On a Gaudi card: torch.logsumexp with hpu_ops' rewrite installed against Habana's kernel (rewrite undone), in one
process, in alternating blocks of N synced calls (outputs kept alive; lazy calls end with a mark_step). Forward, and
forward + backward. Also: errors against float64 on the CPU, fallback counts, "runs on the CPU" warnings, the rewrite's
call count, and an hpu_backend-compiled check. Mode follows PT_HPU_LAZY_MODE."""
import os
import sys
import time
import warnings

import torch
import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.hpu as hthpu
from habana_frameworks.torch.hpu.metrics import metric_global

import hpu_ops
from hpu_ops import perf

LAZY = os.environ.get("PT_HPU_LAZY_MODE", "1") == "1"
N = int(os.environ.get("BENCH_N", "30"))
ROUNDS = 7
hpu = torch.device("hpu")
calls = hpu_ops._STATE["calls"]
torch.manual_seed(0)


def fallback_total():
    return sum((dict(metric_global("cpu_fallback").stats()).get("FallbackOps") or {}).values())


def step():
    if LAZY:
        htcore.mark_step()


def run(fn, x, n, keep):
    for _ in range(n):
        keep.append(fn(x))
        step()
    hthpu.synchronize()


def bench(label, fn, x, taken=True):
    """Median ms of ROUNDS alternating blocks with (ours) and without (habana) the rewrite."""
    times = {"habana": [], "ours": []}
    fb = {"habana": 0, "ours": 0}
    warn = {"habana": 0, "ours": 0}
    outs = {}
    for r in range(ROUNDS + 1):
        for name in (("ours", "habana") if r % 2 else ("habana", "ours")):
            undo = perf.install_logsumexp("hpu") if name == "ours" else (lambda: None)
            try:
                keep = []
                if r == 0:  # warm-up round: compile both graphs
                    run(fn, x, 10, keep)
                    continue
                before, c0 = fallback_total(), calls["logsumexp"]
                with warnings.catch_warnings(record=True) as record:
                    warnings.simplefilter("always")
                    t = time.perf_counter()
                    run(fn, x, N, keep)
                    times[name].append((time.perf_counter() - t) / N * 1e3)
                fb[name] += fallback_total() - before
                warn[name] += sum("runs on the CPU" in str(w.message) for w in record)
                if name == "ours" and perf.lazy_mode() and taken:
                    assert calls["logsumexp"] - c0 >= N, "rewrite not taken"
                elif name == "ours":
                    assert calls["logsumexp"] == c0, "rewrite taken in eager mode or without a gradient"
                else:
                    assert calls["logsumexp"] == c0, "rewrite taken while undone"
                outs[name] = keep[-1]
            finally:
                undo()
    med = {k: sorted(v)[len(v) // 2] for k, v in times.items()}
    print(f"{label:44s} habana {med['habana']:7.3f} ms  ours {med['ours']:7.3f} ms  speedup "
          f"{med['habana'] / med['ours']:5.2f}x  fallbacks {fb['habana']}/{fb['ours']}  cpu-warn "
          f"{warn['habana']}/{warn['ours']}  ranges habana [{min(times['habana']):.3f}-{max(times['habana']):.3f}] "
          f"ours [{min(times['ours']):.3f}-{max(times['ours']):.3f}]", flush=True)
    return outs


def cpu_ms(fn, x):
    for _ in range(5):
        fn(x)
    t = time.perf_counter()
    for _ in range(N):
        fn(x)
    return (time.perf_counter() - t) / N * 1e3


def accuracy(x_cpu, dim, keepdim):
    """Max error of forward and input gradient against float64 on the CPU, Habana's kernel and ours."""
    ref_x = x_cpu.double().requires_grad_()
    ref = torch.logsumexp(ref_x, dim, keepdim)
    ref.backward(torch.ones_like(ref))
    errs = {}
    for name in ("habana", "ours"):
        undo = perf.install_logsumexp("hpu") if name == "ours" else (lambda: None)
        try:
            x = x_cpu.to(hpu).requires_grad_()
            y = torch.logsumexp(x, dim, keepdim)
            y.backward(torch.ones_like(y))
            step()
            e_out = (y.detach().cpu().double() - ref.detach()).abs().max().item()
            e_grad = (x.grad.cpu().double() - ref_x.grad).abs().max().item()
            errs[name] = (e_out, e_grad)
        finally:
            undo()
    return errs


def main():
    print(f"bridge 1.21.4  torch {torch.__version__}  mode {'lazy' if LAZY else 'eager'}  N={N} rounds={ROUNDS}  "
          f"Habana AutogradHPU logsumexp kernel: "
          f"{torch._C._dispatch_has_kernel_for_dispatch_key('aten::logsumexp', 'AutogradHPU')}  HPU kernel: "
          f"{torch._C._dispatch_has_kernel_for_dispatch_key('aten::logsumexp', 'HPU')}", flush=True)
    cases = [((64, 22, 22), -1), ((4096,), 0), ((64, 22, 22), 0), ((64, 22, 22), (-2, -1)), ((32, 22), -1),
             ((640, 22, 22), -1), ((64, 22, 220), -1), ((40960,), 0), ((409600,), 0), ((256, 64, 1000), -1),
             ((4096, 4096), -1)]
    only = sys.argv[1:]
    for shape, dim in cases:
        if only and f"{shape} {dim}".replace(" ", "") not in [o.replace(" ", "") for o in only]:
            continue
        x_cpu = torch.randn(shape)
        x = x_cpu.to(hpu)
        fwd = lambda t, d=dim: torch.logsumexp(t, d)  # noqa: E731
        fwd_grad = lambda t, d=dim: torch.logsumexp(t.detach().requires_grad_(), d)  # noqa: E731

        def fwd_bwd(t, d=dim):
            t = t.detach().requires_grad_()
            y = torch.logsumexp(t, d)
            (g,) = torch.autograd.grad(y.sum(), t)
            return g

        c_ms = cpu_ms(fwd, x_cpu)
        print(f"-- {shape} dim={dim}: cpu forward {c_ms:.3f} ms", flush=True)
        bench(f"   forward, no gradient (not taken)", fwd, x, taken=False)
        bench(f"   forward, input requires grad", fwd_grad, x)
        bench(f"   forward+backward", fwd_bwd, x)
        errs = accuracy(x_cpu, dim, False)
        print(f"   max err vs CPU float64 (out, grad): habana {errs['habana'][0]:.1e} {errs['habana'][1]:.1e}  "
              f"ours {errs['ours'][0]:.1e} {errs['ours'][1]:.1e}", flush=True)
    inf, nan = float("inf"), float("nan")
    special = torch.tensor([[-inf, -inf, -inf], [inf, 1.0, 2.0], [inf, -inf, 0.0], [nan, 1.0, 2.0],
                            [1e30, 1e30, -1e30], [-1e30, 0.0, 88.0]])
    want = torch.logsumexp(special, 1)
    undo = perf.install_logsumexp("hpu")
    try:
        got = torch.logsumexp(special.to(hpu).requires_grad_(), 1).detach()
        step()
        bf = torch.logsumexp(torch.randn(64, 22, 22).to(hpu).bfloat16().requires_grad_(), -1)
        step()
        xc = torch.randn(64, 22, 22)
        if LAZY:
            yc = torch.logsumexp(xc, -1)  # hpu_backend runs in eager mode only
        else:
            torch._dynamo.reset()
            comp = torch.compile(lambda t: torch.logsumexp(t, -1), backend="hpu_backend")
            yc = comp(xc.to(hpu))
    finally:
        undo()
    hab = torch.logsumexp(special.to(hpu), 1).cpu()
    print(f"special values: cpu {want.tolist()}\n  ours {got.cpu().tolist()}\n  habana {hab.tolist()}  "
          f"ours==cpu {torch.allclose(got.cpu(), want, equal_nan=True)}", flush=True)
    print(f"bfloat16 dtype {bf.dtype}  compiled max err {(yc.cpu() - torch.logsumexp(xc, -1)).abs().max().item():.1e}",
          flush=True)


if __name__ == "__main__":
    main()
