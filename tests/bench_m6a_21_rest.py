"""On a Gaudi card (issue #21): the ops the issue names besides bincount, logsumexp and index_add, against
rewrites, at the issue's sizes (4096 elements, 64x22x22) and at 10x and 100x those sizes. The ops are
``scatter_add``, ``trace`` / ``diagonal().sum()``, ``where`` and ``cumsum``. It uses bench_small_ops2's harness:
interleaved blocks with the order rotated every round, the median block mean in ms per call, outputs kept alive,
and fallbacks counted from metric_global("cpu_fallback") and "runs on the CPU" warnings. ``iso`` syncs every call
(lazy: mark_step + synchronize); ``batch`` runs N calls per sync. ``exact`` marks a rewrite that gives ATen's
result for inf and NaN as well (only the summation order differs); the other rewrites multiply by 0 (inf * 0 is NaN)
or lose the sign of zero. Mode follows PT_HPU_LAZY_MODE. The candidates are plain torch, so the CPU tests can import
this file.

    python bench_m6a_21_rest.py [op ...]
"""
import os
import sys

import torch

_CONST = {}
SIZES_3D = ((64, 22, 22), (640, 22, 22), (6400, 22, 22))
SIZES_1D = (4096, 40960, 409600)


def const(key, make):
    if key not in _CONST:
        _CONST[key] = make()
    return _CONST[key]


def scatter_last_put(s, i, v):
    """exact: ``s.scatter_add(-1, i, v)`` (``i`` the shape of ``s``) as one accumulating ``index_put`` of the
    flattened tensors at linear indices."""
    n, rows = s.shape[-1], s.numel() // s.shape[-1]
    base = const(("scatter_base", s.shape, s.device),
                 lambda: (torch.arange(rows) * n).view(*s.shape[:-1], 1).to(s.device))
    return s.reshape(-1).index_put(((i + base).reshape(-1),), v.reshape(-1), accumulate=True).view(s.shape)


def scatter_last_where(s, i, v):
    """exact: one-hot ``where`` then a sum over the source dim."""
    hit = i.unsqueeze(-1) == const(("arange", s.shape[-1], s.device), lambda: torch.arange(s.shape[-1]).to(s.device))
    return s + torch.where(hit, v.unsqueeze(-1), 0.0).sum(-2)


def scatter_last_bmm(s, i, v):
    """inf * 0: a one-hot ``baddbmm``."""
    n, k = s.shape[-1], v.shape[-1]
    hit = (i.unsqueeze(-1) == const(("arange", n, s.device), lambda: torch.arange(n).to(s.device))).to(v.dtype)
    b = v.numel() // k
    return torch.baddbmm(s.reshape(b, 1, n), v.reshape(b, 1, k), hit.reshape(b, k, n)).view(s.shape)


def scatter_1d_put(s, i, v):
    """exact: ``s.scatter_add(0, i, v)`` of 1-D tensors as one accumulating ``index_put``."""
    return s.index_put((i,), v, accumulate=True)


def trace_strided(x):
    """exact: the diagonal as a strided slice of the flattened matrix."""
    return x.reshape(-1)[:: x.shape[-1] + 1].sum()


def trace_einsum(x):
    return torch.einsum("ii", x)


def btrace_native(x):
    return x.diagonal(dim1=-2, dim2=-1).sum(-1)


def btrace_where(x):
    """exact: ``where`` with a cached boolean eye, then one sum over the last two dims."""
    n = x.shape[-1]
    eye = const(("eye_bool", n, x.device), lambda: torch.eye(n, dtype=torch.bool).to(x.device))
    return torch.where(eye, x, 0.0).sum((-2, -1))


def btrace_strided(x):
    """exact: the diagonal as a strided slice of each flattened matrix."""
    n = x.shape[-1]
    return x.reshape(-1, n * n)[:, :: n + 1].sum(-1).view(x.shape[:-2])


def btrace_eye_mul(x):
    """inf * 0: a product with a cached float eye."""
    n = x.shape[-1]
    return (x * const(("eye", n, x.device), lambda: torch.eye(n).to(x.device))).sum((-2, -1))


def where_masked_fill(c, x, y):
    """signed zero: ``-0.0`` where ``c`` comes out as ``+0.0``."""
    return x.masked_fill(~c, 0.0) + y.masked_fill(c, 0.0)


def where_mul(c, x, y):
    """inf * 0: arithmetic select."""
    m = c.to(x.dtype)
    return x * m + y * (1 - m)


def cumsum_last_mm(x):
    """inf * 0: ``x @ triu(ones)``."""
    n = x.shape[-1]
    return x @ const(("triu", n, x.device), lambda: torch.ones(n, n).triu().to(x.device))


def cumsum_last_where(x):
    """exact: ``where(j <= k, x_j, 0)`` summed over j."""
    n = x.shape[-1]
    mask = const(("tril_bool", n, x.device), lambda: torch.ones(n, n, dtype=torch.bool).tril().to(x.device))
    return torch.where(mask, x.unsqueeze(-2), 0.0).sum(-1)


def cumsum_1d_blocks(x, b=64):
    """inf * 0: rows of 64 through ``y @ triu(ones)``, then the native cumsum of the row totals."""
    m = x.shape[0] // b
    inner = x.view(m, b) @ const(("triu", b, x.device), lambda: torch.ones(b, b).triu().to(x.device))
    tot = inner[:, -1]
    return (inner + (tot.cumsum(0) - tot).unsqueeze(1)).view(-1)


def cases():
    r = torch.randn
    c = {"scatter_add": [], "trace": [], "where": [], "cumsum": []}
    for shape in SIZES_3D:
        c["scatter_add"].append((f"{shape} dim-1", lambda s, i, v: s.scatter_add(-1, i, v),
                                 (torch.zeros(shape), torch.randint(0, shape[-1], shape), r(*shape)),
                                 {"put exact": scatter_last_put, "where exact": scatter_last_where,
                                  "bmm": scatter_last_bmm}))
    for n in SIZES_1D:
        c["scatter_add"].append((f"(64,)<-({n},) dim0", lambda s, i, v: s.scatter_add(0, i, v),
                                 (torch.zeros(64), torch.randint(0, 64, (n,)), r(n)), {"put exact": scatter_1d_put}))
    for n in (64, 202, 640):  # 4096, ~40960, 409600 elements
        c["trace"].append((f"trace ({n},{n})", torch.trace, (r(n, n),),
                           {"diag_sum exact": lambda x: x.diagonal().sum(), "strided exact": trace_strided,
                            "einsum": trace_einsum}))
    for shape in SIZES_3D:
        c["trace"].append((f"diagonal.sum {shape}", btrace_native, (r(*shape),),
                           {"where exact": btrace_where, "strided exact": btrace_strided, "eye_mul": btrace_eye_mul}))
    for shape in SIZES_3D:
        c["where"].append((f"where {shape}", torch.where, (r(*shape) > 0, r(*shape), r(*shape)),
                           {"masked_fill": where_masked_fill, "mul": where_mul}))
    for shape in SIZES_3D:
        c["cumsum"].append((f"{shape} dim-1", lambda x: torch.cumsum(x, -1), (r(*shape),),
                            {"where exact": cumsum_last_where, "mm": cumsum_last_mm}))
    for n in SIZES_1D:
        c["cumsum"].append((f"({n},) dim0", lambda x: torch.cumsum(x, 0), (r(n),), {"blocks": cumsum_1d_blocks}))
    return c


def main():
    import bench_small_ops2 as h  # imports the Gaudi bridge

    torch.manual_seed(0)
    want = sys.argv[1:]
    styles = os.environ.get("BENCH_STYLES", "iso,batch").split(",")
    print(f"bridge 1.21.4 torch {torch.__version__} mode {'lazy' if h.LAZY else 'eager'} N={h.N} rounds={h.ROUNDS}",
          flush=True)
    for op, rows in cases().items():
        if want and op not in want:
            continue
        for label, native, args, variants in rows:
            cpu_ms, ref = h.cpu_time(native, args)
            hargs = [a.to(h.hpu) for a in args]
            for style in styles:
                res = h.race({"habana": native, **variants}, hargs, style)
                base = res["habana"][0] if not isinstance(res["habana"], Exception) else float("nan")
                print(f"{op:12s} {label:30s} {style:5s} cpu {cpu_ms:7.3f}", flush=True)
                for name, x in res.items():
                    if isinstance(x, Exception):
                        print(f"    {name:16s} ERROR {type(x).__name__}: {str(x)[:200]}", flush=True)
                        continue
                    ms, lo, hi, fb, cw, out = x
                    err = (out.cpu().double() - ref.double()).abs().max().item()
                    print(f"    {name:16s} {ms:8.3f} [{lo:.3f}-{hi:.3f}] habana/this {base / ms:5.2f} "
                          f"hpu/cpu {ms / cpu_ms:6.2f} err {err:.1e} fb {fb} cpuwarn {cw}", flush=True)


if __name__ == "__main__":
    main()
