"""On a Gaudi card (issue #21, round 2): Habana's kernel against minimal-op rewrites with cached constants, in one
process, in two timing styles. Mode follows PT_HPU_LAZY_MODE.

- ``iso``: each call is synced on its own (lazy: ``mark_step`` then ``synchronize`` after every call): the latency a
  user sees when calling the op once.
- ``batch``: N calls, then one ``mark_step`` and one ``synchronize`` (lazy: one graph of N calls; eager: N queued
  launches): the throughput when many calls run back to back.

Contenders run in interleaved blocks, the order rotated every round; the median block mean (ms per call) is reported.
Outputs are kept alive. Fallbacks: metric_global("cpu_fallback") delta and "runs on the CPU" warnings.

    python bench_small_ops2.py [op ...]
"""
import os
import sys
import time
import warnings

import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.hpu as hthpu
import torch
from habana_frameworks.torch.hpu.metrics import metric_global

LAZY = os.environ.get("PT_HPU_LAZY_MODE", "1") == "1"
N = int(os.environ.get("BENCH_N", "30"))
ROUNDS = int(os.environ.get("BENCH_ROUNDS", "7"))
hpu = torch.device("hpu")
torch.manual_seed(0)
_CONST = {}


def const(key, make):
    if key not in _CONST:
        _CONST[key] = make()
    return _CONST[key]


def fallbacks():
    return sum((dict(metric_global("cpu_fallback").stats()).get("FallbackOps", {}) or {}).values())


def step():
    if LAZY:
        htcore.mark_step()


def block(fn, args, style):
    out = []
    t = time.perf_counter()
    for _ in range(N):
        out.append(fn(*args))
        if style == "iso":
            step()
            hthpu.synchronize()
    if style == "batch":
        step()
        hthpu.synchronize()
    return (time.perf_counter() - t) / N * 1e3, out[-1]


def cpu_time(fn, args):
    for _ in range(5):
        fn(*args)
    ts = []
    for _ in range(5):
        t = time.perf_counter()
        for _ in range(N):
            out = fn(*args)
        ts.append((time.perf_counter() - t) / N * 1e3)
    return sorted(ts)[2], out


def race(contenders, args, style):
    live, res = {}, {}
    for name, fn in contenders.items():
        try:
            for _ in range(3):
                block(fn, args, style)
            live[name] = fn
        except Exception as e:  # noqa: BLE001
            res[name] = e
    times = {k: [] for k in live}
    outs, fb, warn = {}, {k: 0 for k in live}, {k: 0 for k in live}
    names = list(live)
    for r in range(ROUNDS):
        for name in names[r % len(names):] + names[:r % len(names)]:
            before = fallbacks()
            with warnings.catch_warnings(record=True) as rec:
                warnings.simplefilter("always")
                ms, outs[name] = block(live[name], args, style)
            fb[name] += fallbacks() - before
            warn[name] += sum("runs on the CPU" in str(w.message) for w in rec)
            times[name].append(ms)
    for name in live:
        t = sorted(times[name])
        res[name] = (t[len(t) // 2], t[0], t[-1], fb[name], warn[name], outs[name])
    return res


# ---- candidates -----------------------------------------------------------------------------------------------------

def tri(n, dev, dtype=torch.float32, diagonal=0):
    return const(("triu", n, diagonal, dev, dtype), lambda: torch.ones(n, n, dtype=dtype).triu(diagonal).to(dev))


def cumsum_last_mm(x):
    return x @ tri(x.shape[-1], x.device)


def cumsum_last_where(x):
    """Exact for non-finite input (no inf * 0): where(j <= k, x_j, 0) summed over j."""
    n = x.shape[-1]
    mask = const(("trimask", n, x.device), lambda: torch.ones(n, n, dtype=torch.bool).tril().to(x.device))
    return torch.where(mask, x.unsqueeze(-2), 0.0).sum(-1)


def cumsum_last_mm_safe(x):
    """Exact for non-finite input: the finite part and an integer code of the non-finite part (+inf 1, -inf 128,
    NaN 16384; exact sums in float32 for n <= 64) through one matmul with the triangle, decoded after."""
    fin = x.isfinite()
    code = torch.where(fin, 0.0, torch.where(x.isnan(), 16384.0, torch.where(x > 0, 1.0, 128.0)))
    both = torch.stack([torch.where(fin, x, 0.0), code], -2) @ tri(x.shape[-1], x.device)
    out, c = both.unbind(-2)
    pos, neg, nan = (c % 128) > 0, (c % 16384) >= 128, c >= 16384
    special = torch.where(nan | (pos & neg), float("nan"), torch.where(pos, float("inf"), float("-inf")))
    return torch.where(c > 0, special, out)


def cumsum_dim1_mm(x):  # dim -2 of a 3-D tensor: lower-triangular ones on the left
    n = x.shape[-2]
    return const(("tril", n, x.device), lambda: torch.ones(n, n).tril().to(x.device)) @ x


def cumsum_1d_block_a(x, b=64):
    """1-D, n = m*b: inner y @ T, plus the exclusive cumsum of the row totals, Ls @ totals."""
    m = x.shape[0] // b
    y = x.view(m, b)
    inner = y @ tri(b, x.device)
    ls = const(("tril_strict", m, x.device), lambda: torch.ones(m, m).tril(-1).to(x.device))
    return (inner + ls @ inner[:, -1:]).view(-1)


def cumsum_1d_block_b(x, b=64):
    """1-D, n = m*b: addmm(Ls @ (y @ J), y, T)."""
    m = x.shape[0] // b
    y = x.view(m, b)
    ls = const(("tril_strict", m, x.device), lambda: torch.ones(m, m).tril(-1).to(x.device))
    j = const(("ones", b, x.device), lambda: torch.ones(b, b).to(x.device))
    return torch.addmm(ls @ (y @ j), y, tri(b, x.device)).view(-1)


def cumsum_1d_block_native(x, b=64):
    """1-D: inner y @ T, the native cumsum of the m row totals, exclusive by subtraction."""
    m = x.shape[0] // b
    y = x.view(m, b)
    inner = y @ tri(b, x.device)
    tot = inner[:, -1]
    return (inner + (tot.cumsum(0) - tot).unsqueeze(1)).view(-1)


def cumsum_1d_block_mm2(x, b=64):
    """1-D, n = m*b with m <= ~1024: inner y @ T; offsets = Ls @ totals (one matvec)."""
    m = x.shape[0] // b
    y = x.view(m, b)
    inner = y @ tri(b, x.device)
    ls = const(("tril_strict", m, x.device), lambda: torch.ones(m, m).tril(-1).to(x.device))
    return torch.addmm(inner, ls, inner[:, -1:].expand(m, b)).view(-1)


def arange_col(n, dev, dtype=torch.int64):
    return const(("arcol", n, dev, dtype), lambda: torch.arange(n, dtype=dtype).to(dev).unsqueeze(1))


def arange_row(n, dev, dtype=torch.int64):
    return const(("arrow", n, dev, dtype), lambda: torch.arange(n, dtype=dtype).to(dev))


def index_add_oh(s, i, v):
    oh = (arange_col(s.shape[0], s.device) == i).to(v.dtype)  # (R, M)
    return torch.addmm(s, oh, v)


def index_add_oh32(s, i, v):
    oh = (arange_col(s.shape[0], s.device, torch.int32) == i.int()).to(v.dtype)
    return torch.addmm(s, oh, v)


def index_add_put(s, i, v):
    return s.index_put((i,), v, accumulate=True)


def index_add_scatter(s, i, v):
    return s.scatter_add(0, i.unsqueeze(1).expand_as(v), v)


def scatter_last_bmm(s, i, v):
    n, k = s.shape[-1], v.shape[-1]
    oh = (i.unsqueeze(-1) == arange_row(n, s.device)).to(v.dtype)  # (..., k, n)
    b = v.numel() // k
    return torch.baddbmm(s.reshape(b, 1, n), v.reshape(b, 1, k), oh.reshape(b, k, n)).view(s.shape)


def scatter_last_where(s, i, v):
    n = s.shape[-1]
    hit = i.unsqueeze(-1) == arange_row(n, s.device)
    return s + torch.where(hit, v.unsqueeze(-1), 0.0).sum(-2)


def scatter_1d_mv(s, i, v):
    oh = (arange_col(s.shape[0], s.device) == i).to(v.dtype)
    return torch.addmv(s, oh, v)


def bincount_histc(x, minlength=0):
    lo, hi = torch.aminmax(x)
    lo, hi = torch.stack([lo, hi]).tolist()
    if lo < 0:
        raise RuntimeError("bincount only supports 1-d non-negative integral inputs.")
    n = max(hi + 1, minlength)
    return torch.histc(x.float(), n, 0, n - 1).long()


def bincount_histc_max(x, minlength=0):
    n = max(int(x.max()) + 1, minlength)
    return torch.histc(x.float(), n, 0, n - 1).long()


def bincount_oh(x, minlength=0):
    n = max(int(x.max()) + 1, minlength)
    return (x.unsqueeze(1) == arange_row(n, x.device)).sum(0, dtype=torch.int32).long()


def bincount_minmax_cat(x, minlength=0):
    lo, hi = torch.cat([x.min().reshape(1), x.max().reshape(1)]).tolist()
    if lo < 0:
        raise RuntimeError("bincount only supports 1-d non-negative integral inputs.")
    n = max(hi + 1, minlength)
    return torch.histc(x.float(), n, 0, n - 1).long()


def bincount_two_items(x, minlength=0):
    lo, hi = x.min(), x.max()
    hi = int(hi)
    if int(lo) < 0:
        raise RuntimeError("bincount only supports 1-d non-negative integral inputs.")
    n = max(hi + 1, minlength)
    return torch.histc(x.float(), n, 0, n - 1).long()


def bincount_float_minmax(x, minlength=0):
    xf = x.float()
    lo, hi = torch.stack([xf.amin(), xf.amax()]).tolist()
    if lo < 0:
        raise RuntimeError("bincount only supports 1-d non-negative integral inputs.")
    n = max(int(hi) + 1, minlength)
    return torch.histc(xf, n, 0, n - 1).long()


def bincount_known_histc(x, n):  # the length given (not bincount's semantics; for the floor)
    return torch.histc(x.float(), n, 0, n - 1).long()


def lse_logsoftmax(x, dim):
    return x.amax(dim) - torch.log_softmax(x, dim).amax(dim)


def lse_logsoftmax_inf(x, dim):
    """amax - amax(log_softmax), with an infinite max as the result (all -inf rows, a +inf entry)."""
    m = x.amax(dim)
    return torch.where(m.isinf(), m, m - torch.log_softmax(x, dim).amax(dim))


def lse_composite(x, dim):
    m = x.amax(dim, keepdim=True).nan_to_num(0.0, 0.0, 0.0)
    return (x - m).exp().sum(dim).log() + m.squeeze(dim)


# ---- cases --------------------------------------------------------------------------------------------------------

V, B = 4096, (64, 22, 22)


def cases():
    r = torch.randn
    c = {}
    c["cumsum"] = [
        ("(64,22,22) dim-1", lambda x: torch.cumsum(x, -1), (r(*B),), {"mm": cumsum_last_mm, "mm_safe": cumsum_last_mm_safe}),
        ("(640,22,22) dim-1", lambda x: torch.cumsum(x, -1), (r(640, 22, 22),), {"mm": cumsum_last_mm, "mm_safe": cumsum_last_mm_safe}),
        ("(64,22,64) dim-1", lambda x: torch.cumsum(x, -1), (r(64, 22, 64),), {"mm": cumsum_last_mm, "mm_safe": cumsum_last_mm_safe}),
        ("(64,22,22) dim1", lambda x: torch.cumsum(x, 1), (r(*B),), {"mm": cumsum_dim1_mm}),
        ("(4096,) dim0", lambda x: torch.cumsum(x, 0), (r(V),),
         {"blk_a": cumsum_1d_block_a, "blk_b": cumsum_1d_block_b, "blk_native": cumsum_1d_block_native,
          "blk_mm2": cumsum_1d_block_mm2}),
        ("(40960,) dim0", lambda x: torch.cumsum(x, 0), (r(10 * V),),
         {"blk_a": cumsum_1d_block_a, "blk_native": cumsum_1d_block_native, "blk_mm2": cumsum_1d_block_mm2}),
        ("(409600,) dim0", lambda x: torch.cumsum(x, 0), (r(100 * V),),
         {"blk_native": cumsum_1d_block_native}),
    ]
    c["index_add"] = []
    for rows, n in ((64, V), (64, 10 * V), (1024, V)):
        args = (torch.zeros(rows, 22), torch.randint(0, rows, (n,)), r(n, 22))
        c["index_add"].append((f"self({rows},22) src({n},22)", lambda s, i, v: s.index_add(0, i, v), args,
                               {"oh": index_add_oh, "oh32": index_add_oh32, "put_acc": index_add_put,
                                "scatter": index_add_scatter}))
    c["scatter_add"] = []
    for shape in (B, (640, 22, 22), (64, 22, 64)):
        args = (torch.zeros(*shape), torch.randint(0, shape[-1], shape), r(*shape))
        c["scatter_add"].append((f"{shape} dim-1", lambda s, i, v: s.scatter_add(-1, i, v), args,
                                 {"bmm": scatter_last_bmm, "where": scatter_last_where}))
    args = (torch.zeros(64), torch.randint(0, 64, (V,)), r(V))
    c["scatter_add"].append(("self(64,) src(4096,) dim0", lambda s, i, v: s.scatter_add(0, i, v), args,
                             {"mv": scatter_1d_mv, "put_acc": index_add_put}))
    c["bincount"] = []
    for n, bins in ((V, 64), (V, 256), (10 * V, 64), (10 * V, 256), (16384, 128)):
        args = (torch.randint(0, bins, (n,)),)
        c["bincount"].append((f"({n},) {bins} bins", lambda x: torch.bincount(x), args,
                              {"histc_aminmax": bincount_histc, "histc_max": bincount_histc_max,
                               "minmax_cat": bincount_minmax_cat, "two_items": bincount_two_items,
                               "float_minmax": bincount_float_minmax}))
    c["logsumexp"] = []
    for shape, dim in ((B, -1), ((V,), 0), (B, 0), ((640, 22, 22), -1)):
        c["logsumexp"].append((f"{shape} dim{dim}", lambda x, d=dim: torch.logsumexp(x, d), (r(*shape),),
                               {"logsoftmax": lambda x, d=dim: lse_logsoftmax(x, d),
                                "logsoftmax_inf": lambda x, d=dim: lse_logsoftmax_inf(x, d),
                                "composite": lambda x, d=dim: lse_composite(x, d)}))
    return c


def first(o):
    return o[0] if isinstance(o, tuple) else o


def main():
    want = sys.argv[1:]
    styles = os.environ.get("BENCH_STYLES", "iso,batch").split(",")
    print(f"bridge 1.21.4 torch {torch.__version__} mode {'lazy' if LAZY else 'eager'} N={N} rounds={ROUNDS} "
          f"cpu threads {torch.get_num_threads()}", flush=True)
    for op, rows in cases().items():
        if want and op not in want:
            continue
        for label, native, args, variants in rows:
            cpu_ms, ref = cpu_time(native, args)
            hargs = [a.to(hpu) for a in args]
            for style in styles:
                res = race({"habana": native, **variants}, hargs, style)
                h = res["habana"][0] if not isinstance(res["habana"], Exception) else float("nan")
                print(f"{op:12s} {label:28s} {style:5s} cpu {cpu_ms:7.3f}", flush=True)
                for name, x in res.items():
                    if isinstance(x, Exception):
                        print(f"    {name:16s} ERROR {type(x).__name__}: {str(x)[:300]}", flush=True)
                        continue
                    ms, lo, hi, fb, cw, out = x
                    o, rf = first(out).cpu(), first(ref)
                    err = (o.double() - rf.double()).abs().max().item() if o.shape == rf.shape else float("nan")
                    same_dtype = o.dtype == rf.dtype
                    print(f"    {name:16s} {ms:8.3f} [{lo:.3f}-{hi:.3f}] habana/this {h / ms:5.2f} "
                          f"err {err:.1e} dtype_ok {same_dtype} fb {fb} cpuwarn {cw}", flush=True)


if __name__ == "__main__":
    main()
