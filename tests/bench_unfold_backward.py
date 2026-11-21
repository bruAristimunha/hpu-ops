"""On a Gaudi card: ``aten.unfold_backward`` (#43), the candidate formulations and the CPU fallback, eager or lazy.
    PT_HPU_LAZY_MODE=0|1 python3 tests/bench_unfold_backward.py [rounds]
For the issue's four shapes: each candidate alone on the op (``skew`` is hpu_ops' kernel), then the issue's timed call
``(x.unfold(dim, size, step) * w).sum().backward()`` with hpu_ops' kernel installed (``after``) and uninstalled
(``before``: the bridge's CPU fallback), in alternating rounds in one process. Time = mean of 20 synced calls after
one warm-up, outputs kept; prints the median over rounds, the bridge's fallbacks and the error against the CPU."""
import os
import statistics
import sys
import time

import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.hpu as hthpu
import torch
import torch.nn.functional as F
from habana_frameworks.torch.hpu.metrics import metric_global

import hpu_ops
from hpu_ops import composite

LAZY = os.environ.get("PT_HPU_LAZY_MODE", "0") == "1"
hpu_ops.FIXES = tuple(f for f in hpu_ops.FIXES if f is not composite.install_unfold_backward)
hpu_ops.enable()
aten = torch.ops.aten
SHAPES = [((16, 52, 64), 1, 8, 1), ((16, 52, 64), 1, 4, 4), ((16, 8, 64), -1, 4, 2), ((32, 128), 1, 16, 8)]


def fb():
    return dict(dict(metric_global("cpu_fallback").stats()).get("FallbackOps", {}) or {})


def sync():
    if LAZY:
        htcore.mark_step()
    hthpu.synchronize()


def index_add(grad, sizes, dim, size, step):
    """torch's own decomposition (``torch._decomp``): index_add of the windows onto zeros."""
    dim = dim % len(sizes)
    idx = torch.arange(sizes[dim], device=grad.device, dtype=torch.int32).unfold(0, size, step).flatten()
    g = grad.movedim(-1, dim + 1).flatten(dim, dim + 1)
    return grad.new_zeros(sizes).index_add_(dim, idx, g)


def slices(grad, sizes, dim, size, step):
    """``size`` in-place adds of strided slices onto zeros."""
    dim = dim % len(sizes)
    out = grad.new_zeros(sizes)
    n = grad.shape[dim]
    for j in range(size):
        out.narrow(dim, j, (n - 1) * step + 1)[(slice(None),) * dim + (slice(None, None, step),)] += grad[..., j]
    return out


def conv_t(grad, sizes, dim, size, step):
    """One transposed convolution with an identity kernel (the windows as channels)."""
    dim = dim % len(sizes)
    g = grad.movedim(dim, -1)  # (..., size, n)
    batch = g.shape[:-2]
    g = g.reshape(-1, size, g.shape[-1])
    w = torch.eye(size, device=grad.device, dtype=grad.dtype).reshape(size, 1, size)
    out = F.conv_transpose1d(g, w, stride=step)  # (B, 1, (n - 1) * step + size)
    length = sizes[dim]
    out = F.pad(out, (0, length - out.shape[-1])).reshape(*batch, length)
    return out.movedim(-1, dim).contiguous()


CANDIDATES = {"skew": composite.unfold_backward, "index_add": index_add, "slices": slices, "conv_t": conv_t,
              "fallback": aten.unfold_backward.default}


def timed(f, calls=20):
    keep = [f()]  # warm-up (compiles the graph)
    sync()
    t0 = time.perf_counter()
    for _ in range(calls):
        keep.append(f())
        sync()
    return (time.perf_counter() - t0) / calls * 1e3, keep


def main(rounds):
    print("bench_unfold_backward lazy", LAZY, "torch", torch.__version__, flush=True)
    results = {}
    for shape, dim, size, step in SHAPES:
        torch.manual_seed(0)
        xc = torch.randn(shape, requires_grad=True)
        w = torch.randn(xc.unfold(dim, size, step).shape)
        (xc.unfold(dim, size, step) * w).sum().backward()
        g = torch.randn(w.shape)
        want_op = aten.unfold_backward(g, list(shape), dim, size, step)
        gh, wh = g.to("hpu"), w.to("hpu")
        key = (shape, dim, size, step)
        for r in range(rounds):
            for name, f in CANDIDATES.items():  # the op alone (the fallback with hpu_ops' kernel not installed)
                f0 = fb()
                try:
                    t, keep = timed(lambda: f(gh, list(shape), dim, size, step))
                    err = float((keep[-1].cpu() - want_op).abs().max())
                except Exception as e:  # noqa: BLE001
                    t, err = float("nan"), repr(e)[:120]
                d = fb()
                results.setdefault((key, "op " + name), []).append((t, err, {k: d[k] - f0.get(k, 0) for k in d
                                                                             if d[k] - f0.get(k, 0)}))
            for side in ("before", "after") if r % 2 == 0 else ("after", "before"):
                undo = composite.install_unfold_backward("hpu") if side == "after" else (lambda: None)
                try:
                    x = xc.detach().to("hpu").requires_grad_()
                    f0 = fb()

                    def call():
                        x.grad = None
                        (x.unfold(dim, size, step) * wh).sum().backward()
                        return x.grad

                    t, keep = timed(call)
                    d = fb()
                    err = float((keep[-1].cpu() - xc.grad).abs().max())
                finally:
                    undo()
                results.setdefault((key, "step " + side), []).append((t, err, {k: d[k] - f0.get(k, 0) for k in d
                                                                               if d[k] - f0.get(k, 0)}))
    for (key, name), rows in results.items():
        times = [r[0] for r in rows]
        print("RESULT", key, f"{name:16s}", "median_ms", f"{statistics.median(times):.3f}",
              "rounds", [f"{t:.3f}" for t in times], "max_abs_err", rows[-1][1], "fallbacks", rows[-1][2], flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3)
