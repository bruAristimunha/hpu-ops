"""Gaudi2 with hpu_ops.enable(): outputs and input gradients of the max/min family on NaN inputs against torch on
the CPU (same dtype), eager / lazy (PT_HPU_LAZY_MODE) and ``--compile`` (hpu_backend). One ``PROBE {json}`` per case.
    PT_HPU_LAZY_MODE=0|1 python3 repro/probe_extrema_grads.py [--compile] [--baseline]
"""
import argparse
import json
import os

import habana_frameworks.torch.core as htcore
import torch
import torch.nn.functional as F

LAZY = os.environ.get("PT_HPU_LAZY_MODE", "1") != "0"
NAN, INF = float("nan"), float("inf")


def rows(n=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(10, n, generator=g)
    x[0, 0] = NAN
    x[1, n // 2] = NAN
    x[2, -1] = NAN
    x[3, 0], x[3, -1] = INF, NAN
    x[4, -1], x[4, 0] = -INF, NAN
    x[5] = NAN
    x[6, 0], x[6, 1] = NAN, NAN
    x[7] = -INF
    x[8] = 1.5
    x[9, 0], x[9, 1] = -0.0, 0.0
    return x


CASES = {
    "amax": lambda x: x.amax(-1), "amax_all": lambda x: x.amax(), "amin_dim0": lambda x: x.amin(0),
    "max_all": lambda x: x.max(), "median_all": lambda x: x.median(),
    "max_dim": lambda x: x.max(-1).values, "min_dim": lambda x: x.min(-1).values,
    "median_dim": lambda x: x.median(-1).values, "norm_inf": lambda x: x.norm(INF, -1),
    "vnorm_ninf": lambda x: torch.linalg.vector_norm(x, -INF, -1),
    "maximum": lambda x: torch.maximum(x, x.flip(0)), "minimum": lambda x: torch.minimum(x, x.flip(-1)),
    "fmax": lambda x: torch.fmax(x, x.flip(0)), "clamp_t": lambda x: x.clamp(x.flip(0) - 1, x.flip(0) + 1),
    "pool1d": lambda x: F.max_pool1d(x.unsqueeze(0), 2),
    "pool1d_k3s1p1": lambda x: F.max_pool1d(x.unsqueeze(0), 3, 1, 1),
    "pool2d": lambda x: F.max_pool2d(x.reshape(1, 5, 2, 8), 2),
    "pool2d_ceil": lambda x: F.max_pool2d(x.reshape(1, 5, 2, 8), (2, 3), ceil_mode=True),
    "pool3d": lambda x: F.max_pool3d(x.reshape(1, 5, 2, 2, 4), 2),
    "maxpool_module": lambda x: torch.nn.MaxPool2d((1, 3), (1, 2))(x.reshape(1, 1, 10, 8)),
}
INDEX = {"argmax": lambda x: x.argmax(-1), "argmin_dim0": lambda x: x.argmin(0), "argmax_all": lambda x: x.argmax(),
         "max_dim_idx": lambda x: x.max(-1).indices, "median_dim_idx": lambda x: x.median(-1).indices,
         "aminmax": lambda x: torch.stack(torch.aminmax(x, dim=-1))}


# Gaudi breaks ties its own way (measured without hpu_ops too): median's index among equal values, and a padded
# max_pool window of all -inf sends its gradient to the padding. Those cases get tie-free rows 7 and 8.
TIES_NATIVE = {"median_dim", "median_dim_idx", "pool1d_k3s1p1"}


def untied(x):
    x = x.clone()
    x[7] = torch.linspace(-3, 3, x.shape[1])
    x[8] = torch.linspace(2, -2, x.shape[1])
    return x


def weighted(o):
    return (o * torch.arange(1, o.numel() + 1, dtype=o.dtype, device=o.device).reshape(o.shape)).sum()


def run(f, x, grad):
    x = x.detach().requires_grad_(grad)
    out = f(x)
    if grad:
        weighted(out).backward()
    return out.detach(), (x.grad if grad else None)


def differs(a, b):
    a, b = a.cpu(), b.cpu()
    if a.shape != b.shape:
        return f"shape {tuple(a.shape)} vs {tuple(b.shape)}"
    if not a.is_floating_point():
        return None if torch.equal(a, b) else f"index differs at {int((a != b).sum())}"
    na, nb = a.isnan(), b.isnan()
    if not torch.equal(na, nb):
        return f"NaN lost {int((nb & ~na).sum())} extra {int((na & ~nb).sum())}"
    ok = ~na
    err = (a[ok].double() - b[ok].double()).abs()
    fin = torch.isfinite(b[ok])
    if not torch.equal(a[ok][~fin], b[ok][~fin]):
        return "inf differs"
    tol = 1e-5 if a.dtype == torch.float32 else 2e-2
    return None if not fin.any() or err[fin].max() <= tol * max(1.0, b[ok][fin].abs().max().item()) else \
        f"err {err[fin].max():.3g}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--dtypes", default="float32,bfloat16")
    a = ap.parse_args()
    if not a.baseline:
        import hpu_ops

        hpu_ops.enable()
    bad = 0
    only = set(a.only.split(",")) if a.only else None
    for dtype in [getattr(torch, d) for d in a.dtypes.split(",")]:
        x0 = rows().to(dtype)
        for table, grad in ((CASES, True), (INDEX, False)):
            for name, f in table.items():
                if only and name not in only:
                    continue
                row = {"case": name, "dtype": str(dtype)[6:], "lazy": LAZY, "compile": a.compile}
                x = untied(x0) if name in TIES_NATIVE else x0
                try:
                    g = f
                    if a.compile:
                        torch._dynamo.reset()
                        g = torch.compile(f, backend="hpu_backend", dynamic=False)
                    out, gx = run(g, x.to("hpu"), grad)
                    if LAZY:
                        htcore.mark_step()
                    wout, wgx = run(f, x, grad)
                    issues = [f"out {d}" for d in [differs(out, wout)] if d]
                    if grad:
                        issues += [f"grad {d}" for d in [differs(gx, wgx)] if d]
                    row["bad"] = issues
                except Exception as e:  # noqa: BLE001
                    row["bad"] = [f"raised {type(e).__name__}: {str(e).splitlines()[0][:160]}"]
                bad += bool(row["bad"])
                print("PROBE " + json.dumps(row), flush=True)
    print(f"DONE bad={bad}", flush=True)


if __name__ == "__main__":
    main()
