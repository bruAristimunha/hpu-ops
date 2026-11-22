"""G15: ``aten::unique_dim`` falls back to CPU in eager and ``torch.compile(backend="hpu_backend")``.

In BrainOmni / BrainTokenizer (``braindecode/modules/quantization.py:150``, ``EuclideanCodebook._kmeans``), the first
forward runs ``rows, inverse = torch.unique(samples, dim=0, return_inverse=True)`` once before setting
``self.inited = 1``. In ``tests/strict_sweep.py``, eager/lazy warm-up runs that initialization before
``strict("report")``, while ``--compile`` traces step 1 inside ``strict("report")`` and records ``aten::unique_dim``
in ``trace_rows``.

Run on Gaudi2:
    PT_HPU_LAZY_MODE=0 python3 repro/g15_unique_dim_compile.py
    PT_HPU_LAZY_MODE=1 python3 repro/g15_unique_dim_compile.py
"""
import os

import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.hpu as hthpu
import torch
from habana_frameworks.torch.hpu.metrics import metric_global

import hpu_ops

hpu_ops.enable()

LAZY = os.environ.get("PT_HPU_LAZY_MODE", "1") != "0"


def fallbacks():
    return dict(dict(metric_global("cpu_fallback").stats()).get("FallbackOps", {}) or {})


def delta(before):
    now = fallbacks()
    return {k: now[k] - before.get(k, 0) for k in now if now[k] - before.get(k, 0)}


def sync():
    if LAZY:
        htcore.mark_step()
    hthpu.synchronize()


def unique_fn(x):
    return torch.unique(x, dim=0, return_inverse=True)


class MiniCodebookInit(torch.nn.Module):
    """Minimal EuclideanCodebook._kmeans pattern from braindecode."""

    def __init__(self, codebook_size=16, dim=8, kmeans_iters=2):
        super().__init__()
        self.codebook_size = codebook_size
        self.kmeans_iters = kmeans_iters
        self.register_buffer("inited", torch.zeros(1))
        self.register_buffer("embed", torch.zeros(codebook_size, dim))

    def _kmeans(self, samples):
        centers = samples[:self.codebook_size].clone()
        rows, inverse = torch.unique(samples, dim=0, return_inverse=True)
        for _ in range(self.kmeans_iters):
            dist = torch.cdist(rows, centers, compute_mode="donot_use_mm_for_euclid_dist")
            buckets = dist.argmin(dim=-1)[inverse]
            bins = torch.bincount(buckets, minlength=self.codebook_size).clamp_min(1)
            new_centers = torch.zeros_like(centers)
            new_centers.scatter_add_(0, buckets.unsqueeze(-1).expand(-1, samples.shape[-1]), samples)
            centers = new_centers / bins.unsqueeze(-1)
        return centers

    def forward(self, x):
        if not bool(self.inited.item()):
            with torch.no_grad():
                self.embed.copy_(self._kmeans(x))
                self.inited.fill_(1)
        return x @ self.embed.t()


def main():
    torch.manual_seed(0)
    print(f"G15 unique_dim repro | torch={torch.__version__} | lazy={LAZY}", flush=True)
    # 1. Bare unique_dim in eager and compile
    x_cpu = torch.randn(24, 8)
    x_cpu = torch.cat([x_cpu, x_cpu[:8]], dim=0)  # 32 rows, 24 distinct
    x = x_cpu.to("hpu")

    b = fallbacks()
    rows_e, inv_e = unique_fn(x)
    sync()
    fb_eager = delta(b)
    torch.testing.assert_close(rows_e[inv_e].cpu(), x_cpu)
    print(f"bare eager unique_dim: shape={tuple(rows_e.shape)} fallback={fb_eager}", flush=True)

    if not LAZY:
        compiled_u = torch.compile(unique_fn, backend="hpu_backend")
        for step in (1, 2):
            b = fallbacks()
            rows_c, inv_c = compiled_u(x)
            sync()
            fb_c = delta(b)
            torch.testing.assert_close(rows_c[inv_c].cpu(), x_cpu)
            print(f"bare compile step {step} unique_dim: shape={tuple(rows_c.shape)} fallback={fb_c}", flush=True)

    # 2. EuclideanCodebook._kmeans one-shot pattern
    for mode in (("eager",) if LAZY else ("eager", "compile")):
        mod = MiniCodebookInit().to("hpu")
        fn = mod if mode == "eager" else torch.compile(mod, backend="hpu_backend")
        for step in (1, 2):
            b = fallbacks()
            out = fn(x)
            sync()
            fb = delta(b)
            print(f"codebook {mode} step {step}: out={tuple(out.shape)} inited={int(mod.inited.item())} "
                  f"fallback={fb}", flush=True)


if __name__ == "__main__":
    main()
