"""On a Gaudi card: issue #21's small ops, Habana's kernel against one CPU core and against candidate rewrites, in one
process. Mode follows PT_HPU_LAZY_MODE. Each time is the mean of N synced calls after warm-up, outputs kept alive;
lazy calls end with a mark_step. Fallbacks: metric_global("cpu_fallback") delta and "runs on the CPU" warnings.

    python bench_small_ops.py [op ...]
"""
import os
import sys
import time
import warnings

import torch
import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.hpu as hthpu
from habana_frameworks.torch.hpu.metrics import metric_global

LAZY = os.environ.get("PT_HPU_LAZY_MODE", "1") == "1"
N = int(os.environ.get("BENCH_N", "30"))
hpu = torch.device("hpu")
torch.manual_seed(0)


def fallbacks():
    stats = dict(metric_global("cpu_fallback").stats())
    return stats.get("FallbackOps", {}) or {}


def step(device="hpu"):
    if LAZY and device == "hpu":
        htcore.mark_step()


def timed(fn, args, device, blocks=5):
    """Median over ``blocks`` of the mean ms of N synced calls, plus fallback ops and CPU warnings seen while timing."""
    sync = hthpu.synchronize if device == "hpu" else (lambda: None)
    keep = []
    for _ in range(10):
        keep.append(fn(*args))
        step(device)
    sync()
    before = dict(fallbacks()) if device == "hpu" else {}
    means = []
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        for _ in range(blocks):
            keep = keep[-1:]
            t = time.perf_counter()
            for _ in range(N):
                keep.append(fn(*args))
                step(device)
            sync()
            means.append((time.perf_counter() - t) / N * 1e3)
    after = fallbacks() if device == "hpu" else {}
    fb = {k: v - before.get(k, 0) for k, v in after.items() if v - before.get(k, 0)}
    cpu_warn = sum("runs on the CPU" in str(w.message) for w in record)
    return sorted(means)[len(means) // 2], fb, cpu_warn, keep[-1]


def race(contenders, args, rounds=7):
    """``contenders`` ({name: fn}) on the HPU in interleaved blocks of N synced calls, rotating the order each round so
    every one sees the same host conditions. Returns {name: (median block-mean ms, fallback ops, CPU warnings, out)}.
    A contender that raises is reported with its error."""
    keep, result, live = {}, {}, {}
    for name, fn in contenders.items():
        try:
            keep[name] = [fn(*args) for _ in range(10)]
            step()
            hthpu.synchronize()
            live[name] = fn
        except Exception as e:  # noqa: BLE001
            result[name] = e
    means = {name: [] for name in live}
    fb = {name: {} for name in live}
    warn = {name: 0 for name in live}
    names = list(live)
    for r in range(rounds):
        for name in names[r % len(names):] + names[:r % len(names)]:
            fn = live[name]
            before = dict(fallbacks())
            with warnings.catch_warnings(record=True) as record:
                warnings.simplefilter("always")
                out = keep[name][-1:]
                t = time.perf_counter()
                for _ in range(N):
                    out.append(fn(*args))
                    step()
                hthpu.synchronize()
                means[name].append((time.perf_counter() - t) / N * 1e3)
            keep[name] = out
            for k, v in fallbacks().items():
                if v - before.get(k, 0):
                    fb[name][k] = fb[name].get(k, 0) + v - before.get(k, 0)
            warn[name] += sum("runs on the CPU" in str(w.message) for w in record)
    for name in live:
        m = sorted(means[name])
        result[name] = (m[len(m) // 2], fb[name], warn[name], keep[name][-1], m[0], m[-1])
    return result


def first(out):
    return out[0] if isinstance(out, tuple) else out


# ---- candidate rewrites -------------------------------------------------------------------------------------------

def cumsum_tri(x, dim):
    """cumsum along dim as a matmul with an upper-triangular ones matrix."""
    x = x.movedim(dim, -1)
    n = x.shape[-1]
    tri = torch.ones(n, n, device=x.device, dtype=x.dtype).triu_()
    return (x @ tri).movedim(-1, dim)


def cumsum_block(x, dim, b=64):
    """cumsum along dim of length n = b*m: within-block triangular matmul, plus the exclusive cumsum of block sums."""
    x = x.movedim(dim, -1)
    n = x.shape[-1]
    m = n // b
    y = x.reshape(*x.shape[:-1], m, b)
    tri = torch.ones(b, b, device=x.device, dtype=x.dtype).triu_()
    inner = y @ tri
    totals = inner[..., -1]
    strict = torch.ones(m, m, device=x.device, dtype=x.dtype).triu_(1)
    offsets = totals @ strict
    return (inner + offsets.unsqueeze(-1)).reshape(x.shape).movedim(-1, dim)


def bincount_onehot(x, minlength):
    """bincount with a known output length, as a one-hot sum (no host sync)."""
    return (x.unsqueeze(-1) == torch.arange(minlength, device=x.device)).sum(0)


def bincount_onehot_f(x, minlength):
    ar = torch.arange(minlength, device=x.device, dtype=torch.int32)
    return (x.int().unsqueeze(-1) == ar).float().sum(0).long()


def bincount_scatter(x, minlength):
    out = torch.zeros(minlength, device=x.device, dtype=torch.float32)
    return out.index_add(0, x, torch.ones_like(x, dtype=torch.float32)).long()


def bincount_sync(x):
    """bincount's own semantics: length max(x)+1 read from the device (a host sync), then a one-hot sum."""
    return bincount_onehot_f(x, int(x.max()) + 1)


def index_add_onehot(self, dim, index, src):
    """self.index_add(0, index, src) as self + onehot(index)^T @ src."""
    onehot = (index.unsqueeze(0) == torch.arange(self.shape[0], device=self.device).unsqueeze(1)).to(src.dtype)
    return self + (onehot @ src.reshape(src.shape[0], -1)).reshape(self.shape)


def scatter_add_onehot(self, dim, index, src):
    """self.scatter_add(-1, index, src) with index of src's shape along the last dim: onehot contraction."""
    n = self.shape[-1]
    onehot = (index.unsqueeze(-1) == torch.arange(n, device=self.device)).to(src.dtype)  # (..., k, n)
    return self + (src.unsqueeze(-2) @ onehot).squeeze(-2)


def logsumexp_fused(x, dim):
    m = x.amax(dim, keepdim=True)
    m = torch.where(torch.isinf(m), torch.zeros_like(m), m)
    return (x - m).exp().sum(dim).log() + m.squeeze(dim)


def logsumexp_masked(x, dim):
    """ATen's own composite: the max with infinities replaced by 0, so all -inf rows give -inf and +inf gives +inf."""
    m = x.amax(dim, keepdim=True)
    m = m.masked_fill(m.abs() == float("inf"), 0)
    return (x - m).exp().sum(dim).log() + m.squeeze(dim)


def logsumexp_nan2num(x, dim):
    """ATen's composite with the infinities of the max zeroed by one nan_to_num (a NaN max gives NaN either way)."""
    m = x.amax(dim, keepdim=True).nan_to_num(0.0, 0.0, 0.0)
    return (x - m).exp().sum(dim).log() + m.squeeze(dim)


def logsumexp_nofinite(x, dim):
    m = x.amax(dim, keepdim=True)
    return (x - m).exp().sum(dim).log() + m.squeeze(dim)


def trace_sum(x):
    return x.diagonal(dim1=-2, dim2=-1).sum(-1)


def trace_eye(x):
    eye = torch.eye(x.shape[-1], device=x.device, dtype=x.dtype)
    return (x * eye).sum((-2, -1))


def cumprod_log(x, dim):
    return x.log().cumsum(dim).exp()


def histc_onehot(x, bins, lo, hi):
    """histc with explicit bounds as a one-hot sum (no host sync); the last bin includes ``hi``."""
    pos = ((x - lo) * (bins / (hi - lo))).floor().clamp(max=bins - 1)
    inside = (x >= lo) & (x <= hi)
    onehot = (pos.unsqueeze(-1) == torch.arange(bins, device=x.device, dtype=x.dtype)) & inside.unsqueeze(-1)
    return onehot.to(x.dtype).sum(0)


def lerp_composite(a, b, w):
    return a + w * (b - a)


# ---- the cases ----------------------------------------------------------------------------------------------------

V, B = 4096, (64, 22, 22)


def cases():
    vec = lambda *s: torch.randn(*s)  # noqa: E731
    out = {}
    for n in (V, 10 * V, 100 * V):
        x = vec(n)
        out.setdefault("cumsum", []).append((f"({n},) dim0", lambda x: torch.cumsum(x, 0), (x,),
                                             {"tri": lambda x: cumsum_tri(x, 0) if x.shape[0] <= 4096 else None,
                                              "block64": lambda x: cumsum_block(x, 0, 64)}))
    for shape in (B, (640, 22, 22), (64, 22, 220)):
        x = vec(*shape)
        out["cumsum"].append((f"{shape} dim-1", lambda x: torch.cumsum(x, -1), (x,),
                              {"tri": lambda x: cumsum_tri(x, -1)}))
    for n in (V, 10 * V, 100 * V):
        x = torch.randint(0, 64, (n,))
        out.setdefault("bincount", []).append((f"({n},) 64 bins", lambda x: torch.bincount(x, minlength=64), (x,),
                                               {"onehot_known_len": lambda x: bincount_onehot_f(x, 64),
                                                "onehot_int64": lambda x: bincount_onehot(x, 64),
                                                "index_add": lambda x: bincount_scatter(x, 64),
                                                "onehot_sync_len": bincount_sync}))
    for n, rows in ((V, 64), (10 * V, 64), (V, 1024)):
        s = torch.zeros(rows, 22)
        idx = torch.randint(0, rows, (n,))
        src = vec(n, 22)
        out.setdefault("index_add", []).append((f"self({rows},22) src({n},22)",
                                                lambda s, i, v: s.index_add(0, i, v), (s, idx, src),
                                                {"onehot_mm": lambda s, i, v: index_add_onehot(s, 0, i, v)}))
    for shape, k in ((B, 22), ((64, 22, 64), 64), ((640, 22, 22), 22)):
        s = torch.zeros(*shape)
        idx = torch.randint(0, shape[-1], shape[:-1] + (k,))
        src = vec(*shape[:-1], k)
        out.setdefault("scatter_add", []).append((f"{shape} dim-1", lambda s, i, v: s.scatter_add(-1, i, v),
                                                  (s, idx, src),
                                                  {"onehot_mm": lambda s, i, v: scatter_add_onehot(s, -1, i, v)}))
    for shape, dim in ((B, -1), ((V,), 0), ((640, 22, 22), -1), ((64, 22, 220), -1), (B, 0)):
        x = vec(*shape)
        out.setdefault("logsumexp", []).append((f"{shape} dim{dim}", lambda x, d=dim: torch.logsumexp(x, d), (x,),
                                                {"fused": lambda x, d=dim: logsumexp_fused(x, d),
                                                 "masked": lambda x, d=dim: logsumexp_masked(x, d),
                                                 "nan2num": lambda x, d=dim: logsumexp_nan2num(x, d),
                                                 "nofinite": lambda x, d=dim: logsumexp_nofinite(x, d)}))
    for shape in (B, (640, 22, 22)):
        x = vec(*shape)
        out.setdefault("trace_diagonal_sum", []).append((f"{shape}", trace_sum, (x,), {"eye_mul": trace_eye}))
    for shape in (B, (640, 22, 22)):
        x = vec(*shape).abs() + 0.5
        out.setdefault("cumprod", []).append((f"{shape} dim-1", lambda x: torch.cumprod(x, -1), (x,),
                                              {"exp_cumsum_log": lambda x: cumprod_log(x, -1)}))
    for n in (V, 100 * V):
        x = vec(n)
        out.setdefault("histc", []).append((f"({n},) 64 bins [-3,3]", lambda x: torch.histc(x, 64, -3, 3), (x,),
                                            {"onehot": lambda x: histc_onehot(x, 64, -3.0, 3.0)}))
    a, b = vec(*B), vec(*B)
    out["lerp"] = [(f"{B} w=0.3", lambda a, b: torch.lerp(a, b, 0.3), (a, b),
                    {"a+w(b-a)": lambda a, b: lerp_composite(a, b, 0.3)})]
    out["median"] = [(f"{B} dim-1", lambda x: torch.median(x, -1).values, (vec(*B),), {}),
                     (f"({V},)", lambda x: torch.median(x), (vec(V),), {})]
    out["topk"] = [(f"{B} k=5 dim-1", lambda x: torch.topk(x, 5, -1).values, (vec(*B),), {}),
                   (f"({V},) k=10", lambda x: torch.topk(x, 10).values, (vec(V),), {})]
    out["where"] = [(f"{B}", lambda x: torch.where(x > 0, x, 0.0), (vec(*B),), {})]
    out["matmul"] = [(f"{B} @ {B}", lambda x: x @ x, (vec(*B),), {})]
    return out


def main():
    want = sys.argv[1:]
    print(f"bridge 1.21.4  torch {torch.__version__}  mode {'lazy' if LAZY else 'eager'}  N={N}  "
          f"cpu threads {torch.get_num_threads()}", flush=True)
    for op, rows in cases().items():
        if want and op not in want:
            continue
        for label, native, args, variants in rows:
            cpu_ms, _, _, ref = timed(native, args, "cpu")
            hargs = [a.to(hpu) for a in args]
            contenders = {"habana": native}
            contenders.update({k: v for k, v in variants.items() if v(*args) is not None})
            res = race(contenders, hargs)
            h_ms = res["habana"][0] if not isinstance(res["habana"], Exception) else float("nan")
            print(f"{op:18s} {label:28s} cpu {cpu_ms:8.3f} ms", flush=True)
            for name, r in res.items():
                if isinstance(r, Exception):
                    print(f"{'':18s} {'':28s} {name:16s} ERROR {type(r).__name__}: {str(r)[:200]}", flush=True)
                    continue
                ms, fb, cw, out, lo, hi = r
                err = (first(out).cpu().double() - first(ref).double()).abs().max().item()
                print(f"{'':18s} {'':28s} {name:16s} {ms:8.3f} ms [{lo:.3f}-{hi:.3f}]  /cpu {ms / cpu_ms:6.1f}x  "
                      f"habana/this {h_ms / ms:5.2f}  err {err:.1e} fb {fb} cpuwarn {cw}", flush=True)


if __name__ == "__main__":
    main()
