"""Gaudi2 damage map: which max/min-like ops drop NaN (hpu-ops reduction-NaN issue).

    PT_HPU_LAZY_MODE=0|1 python3 repro/probe_reduction_nan.py [--enable] [--compile] [--only amax,max_dim]

Each case builds a float32 tensor on the CPU with NaN at the first, a middle and the last position of some rows, NaN
with +inf and -inf, an all-NaN row, -0.0 against 0.0 and finite rows, casts it to each dtype, runs the op on the
card and on the CPU (the CPU in float32 for 16-bit dtypes it lacks) and compares: NaN positions exactly, indices
exactly where torch defines them (the first NaN of a NaN row, else a position holding the extreme value), values
to the dtype's tolerance. One ``PROBE {json}`` line per case; ``bad`` lists what differs.
"""
import argparse
import json
import os
import sys

import torch

try:
    import habana_frameworks.torch.core as htcore
    DEV = "hpu"
except ImportError:  # a CPU dry run of the probe itself
    htcore, DEV = None, "cpu"
import torch.nn.functional as F

LAZY = os.environ.get("PT_HPU_LAZY_MODE", "1") != "0" and DEV == "hpu"
NAN, INF = float("nan"), float("inf")
DTYPES = (torch.float32, torch.bfloat16, torch.float16)
SIZES = (3, 64, 1000, 4099, 65537)


def rows(n, seed=0):
    """(9, n): NaN first / middle / last, NaN+inf, NaN-inf, all NaN, finite, -inf row, NaN only at a far position."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(9, n, generator=g)
    x[0, 0] = NAN
    x[1, n // 2] = NAN
    x[2, -1] = NAN
    x[3, 0], x[3, -1] = INF, NAN
    x[4, -1], x[4, 0] = -INF, NAN
    x[5] = NAN
    x[7] = -INF
    x[8, (n * 3) // 4] = NAN
    return x


def put(t):
    out = t.to(DEV)
    return out


def to_cpu(v):
    if isinstance(v, torch.Tensor):
        return v.cpu()
    if isinstance(v, (tuple, list)):
        return [to_cpu(e) for e in v]
    return v


def compare(got, want, dtype, kinds):
    """Returns a list of mismatch descriptions. ``kinds`` names each output: "v" value, "i" index (checked by the
    value it points to, through ``kinds``' "src"), "x" exact."""
    bad = []
    got, want = to_cpu(got), to_cpu(want)
    if not isinstance(got, list):
        got, want = [got], [want]
    tol = {torch.float32: 1e-4, torch.bfloat16: 2e-2, torch.float16: 2e-3}.get(dtype, 0)
    for k, (g, w) in enumerate(zip(got, want)):
        if g.shape != w.shape:
            bad.append(f"out{k} shape {tuple(g.shape)} vs {tuple(w.shape)}")
            continue
        if g.is_floating_point():
            gn, wn = g.isnan(), w.isnan()
            lost, extra = int((wn & ~gn).sum()), int((gn & ~wn).sum())
            if lost:
                sample = g[wn & ~gn].flatten()[:3].float().tolist()
                bad.append(f"out{k} NaN lost at {lost} (got {sample})")
            if extra:
                bad.append(f"out{k} NaN extra at {extra}")
            ok = ~(gn | wn)
            gv, wv = g[ok].double(), w.to(g.dtype)[ok].double()
            inf_bad = int(((gv.isinf() | wv.isinf()) & (gv != wv)).sum())
            fin = ~(gv.isinf() | wv.isinf())
            err = (gv[fin] - wv[fin]).abs().max().item() if fin.any() else 0.0
            scale = max(wv[fin].abs().max().item(), 1.0) if fin.any() else 1.0
            if inf_bad:
                bad.append(f"out{k} inf mismatch at {inf_bad}")
            if err > tol * scale * 10:
                bad.append(f"out{k} value err {err:.3g}")
            # sign of zero
            zs = (gv == 0) & (wv == 0)
            if zs.any() and int((torch.signbit(gv[zs]) != torch.signbit(wv[zs])).sum()):
                n_sign = int((torch.signbit(gv[zs]) != torch.signbit(wv[zs])).sum())
                bad.append(f"out{k} sign of zero differs at {n_sign}")
        else:
            if kinds and kinds[k] == "i_free":
                continue  # checked by the caller
            if not torch.equal(g, w.to(g.dtype)):
                bad.append(f"out{k} index differs at {int((g != w.to(g.dtype)).sum())}")
    return bad


def ref(fn, x, *rest):
    """``fn`` on the CPU in x's dtype, or in float32 then cast back when the CPU lacks the dtype."""
    try:
        return fn(x, *rest)
    except (RuntimeError, NotImplementedError):
        out = fn(x.float(), *[r.float() if isinstance(r, torch.Tensor) and r.is_floating_point() else r for r in rest])
        return out


def index_check(gi, x, dim, largest, want_i):
    """Indices are right when a NaN row's index equals torch's (the first NaN) and a finite row's points to an
    extreme value (ties may pick any)."""
    gi = gi.cpu()
    if gi.shape != want_i.shape:
        return [f"index shape {tuple(gi.shape)} vs {tuple(want_i.shape)}"]
    xs = x.float()
    if dim is None:
        xs, dim = xs.reshape(-1), 0
    has_nan = xs.isnan().any(dim, keepdim=True)
    picked = xs.gather(dim, gi.unsqueeze(dim) if gi.dim() < xs.dim() else gi)
    clean = torch.where(xs.isnan(), torch.zeros_like(xs), xs)
    ext = clean.amax(dim, keepdim=True) if largest else clean.amin(dim, keepdim=True)
    has_nan_s = has_nan if gi.dim() == xs.dim() else has_nan
    if gi.dim() < xs.dim():
        wi = want_i.unsqueeze(dim)
        gii = gi.unsqueeze(dim)
    else:
        wi, gii = want_i, gi
    bad = []
    nan_wrong = int((has_nan_s & (gii != wi)).sum())
    if nan_wrong:
        wrong = has_nan_s & (gii != wi)
        bad.append(f"index of NaN row not torch's (first NaN) at {nan_wrong}, got {gii[wrong][:3].tolist()} "
                   f"want {wi[wrong][:3].tolist()}")
    fin_wrong = int((~has_nan_s & (picked != ext)).sum())
    if fin_wrong:
        bad.append(f"index of finite row not an extreme at {fin_wrong}")
    return bad


def cases():
    """(name, fn, input builder, kinds, index-spec) per case. Builders take (n, dtype) and return CPU tensors."""
    two = lambda n, d: rows(n).to(d)  # noqa: E731
    two_t = lambda n, d: rows(n).t().contiguous().to(d)  # reduce dim 0  # noqa: E731
    three = lambda n, d: rows(n).reshape(9, 1, n).expand(9, 4, n).contiguous().to(d)  # noqa: E731
    out = []
    for nm, fn in (("amax_last", lambda x: x.amax(-1)), ("amax_dim0", None), ("amax_full", lambda x: x.amax()),
                   ("amax_mid", lambda x: x.amax(1)), ("amax_keep", lambda x: x.amax(-1, keepdim=True)),
                   ("amin_last", lambda x: x.amin(-1)), ("amin_full", lambda x: x.amin()),
                   ("max_full", lambda x: x.max()), ("min_full", lambda x: x.min()),
                   ("aminmax_last", lambda x: torch.aminmax(x, dim=-1)), ("aminmax_full", lambda x: torch.aminmax(x)),
                   ("maximum_bin", lambda x: torch.maximum(x, x.flip(-1))),
                   ("minimum_bin", lambda x: torch.minimum(x, x.flip(-1))),
                   ("maximum_nan_right", lambda x: torch.maximum(x.nan_to_num(0.5), x)),
                   ("fmax_bin", lambda x: torch.fmax(x, x.flip(-1))),
                   ("clamp_both", lambda x: x.clamp(-0.5, 0.5)), ("clamp_min", lambda x: x.clamp(min=0.0)),
                   ("clamp_max", lambda x: x.clamp(max=0.0)),
                   ("clamp_tensor", lambda x: x.clamp(torch.zeros_like(x), torch.ones_like(x))),
                   ("relu", F.relu), ("relu6", F.relu6), ("hardtanh", F.hardtanh), ("leaky_relu", F.leaky_relu),
                   ("elu", F.elu), ("gelu", F.gelu), ("silu", F.silu),
                   ("logsumexp_last", lambda x: x.logsumexp(-1)), ("logsumexp_dim0", lambda x: x.logsumexp(0)),
                   ("softmax_last", lambda x: x.softmax(-1)), ("log_softmax_last", lambda x: x.log_softmax(-1)),
                   ("softmax_dim0", lambda x: x.softmax(0)), ("log_softmax_dim0", lambda x: x.log_softmax(0)),
                   ("norm_inf", lambda x: x.norm(INF, -1)),
                   ("norm_ninf", lambda x: torch.linalg.vector_norm(x, -INF, -1)),
                   ("vnorm_inf_full", lambda x: torch.linalg.vector_norm(x, INF)),
                   ("nansum", lambda x: x.nansum(-1)), ("nanmean", lambda x: x.nanmean(-1)),
                   ("median_dim", lambda x: x.median(-1).values), ("nanmedian_dim", lambda x: x.nanmedian(-1).values),
                   ("sum", lambda x: x.sum(-1)), ("mean", lambda x: x.mean(-1)), ("prod", lambda x: x.prod(-1)),
                   ("std", lambda x: x.std(-1)), ("isnan_any", lambda x: x.isnan().any(-1))):
        if nm == "amax_dim0":
            out.append((nm, lambda x: x.amax(0), two_t, None, None))
        elif nm == "amax_mid":
            out.append((nm, fn, three, None, None))
        else:
            out.append((nm, fn, two, None, None))
    # values + indices: index checked by index_check
    out += [
        ("max_dim", lambda x: x.max(-1), two, ("v", "i_free"), (-1, True)),
        ("min_dim", lambda x: x.min(-1), two, ("v", "i_free"), (-1, False)),
        ("max_dim0", lambda x: x.max(0), two_t, ("v", "i_free"), (0, True)),
        ("max_dim_keep", lambda x: x.max(-1, keepdim=True), two, ("v", "i_free"), (-1, True)),
        ("argmax_last", lambda x: x.argmax(-1), two, ("i_free",), (-1, True)),
        ("argmin_last", lambda x: x.argmin(-1), two, ("i_free",), (-1, False)),
        ("argmax_full", lambda x: x.argmax(), two, ("i_free",), (None, True)),
        ("argmax_dim0", lambda x: x.argmax(0), two_t, ("i_free",), (0, True)),
        ("cummax", lambda x: x.cummax(-1), two, None, None),
        ("cummin", lambda x: x.cummin(-1), two, None, None),
        ("sort", lambda x: x.sort(-1), two, ("v", "i_free"), "sort"),
        ("sort_desc", lambda x: x.sort(-1, descending=True), two, ("v", "i_free"), "sort"),
        ("topk3", lambda x: x.topk(3, -1), two, ("v", "i_free"), "sort"),
        ("topk3_small", lambda x: x.topk(3, -1, largest=False), two, ("v", "i_free"), "sort"),
        ("kthvalue2", lambda x: x.kthvalue(2, -1, keepdim=True), two, ("v", "i_free"), "sort"),
        ("argsort", lambda x: x.argsort(-1), two, ("i_free",), "sort"),
    ]
    pool_in1 = lambda n, d: rows(n).reshape(3, 3, n).to(d)  # noqa: E731
    pool_in2 = lambda n, d: rows(16).reshape(9, 1, 4, 4).contiguous().to(d)  # noqa: E731
    pool_in3 = lambda n, d: rows(64).reshape(9, 1, 4, 4, 4).to(d)  # noqa: E731
    out += [
        ("max_pool1d", lambda x: F.max_pool1d(x, 2), pool_in1, None, None),
        ("max_pool1d_k3s1", lambda x: F.max_pool1d(x, 3, 1), pool_in1, None, None),
        ("max_pool1d_idx", lambda x: F.max_pool1d(x, 2, return_indices=True), pool_in1, None, None),
        ("max_pool2d", lambda x: F.max_pool2d(x, 2), pool_in2, None, None),
        ("max_pool2d_idx", lambda x: F.max_pool2d(x, 2, return_indices=True), pool_in2, None, None),
        ("max_pool3d", lambda x: F.max_pool3d(x, 2), pool_in3, None, None),
        ("adaptive_max_pool1d", lambda x: F.adaptive_max_pool1d(x, 1), pool_in1, None, None),
        ("adaptive_max_pool1d_idx", lambda x: F.adaptive_max_pool1d(x, 1, return_indices=True), pool_in1, None, None),
        ("adaptive_max_pool2d", lambda x: F.adaptive_max_pool2d(x, 1), pool_in2, None, None),
        ("adaptive_avg_pool1d", lambda x: F.adaptive_avg_pool1d(x, 1), pool_in1, None, None),
    ]
    return out


INFERENCE = False


def run_case(name, fn, build, kinds, spec, n, dtype, compile_):
    x = build(n, dtype)
    row = {"case": name, "n": n, "dtype": str(dtype).split(".")[-1], "lazy": LAZY, "compile": compile_}
    try:
        f = torch.compile(fn, backend="hpu_backend", dynamic=False) if compile_ else fn
        if INFERENCE:
            with torch.inference_mode():
                got = f(put(x))
        else:
            got = f(put(x))
        if LAZY:
            htcore.mark_step()
        got = to_cpu(got)
        want = ref(fn, x)
        want_l = to_cpu(want)
        bad = compare(got, want_l, dtype, kinds)
        if spec is not None and spec != "sort":
            dim, largest = spec
            gi = got[-1] if isinstance(got, list) else got
            wi = want_l[-1] if isinstance(want_l, list) else want_l
            bad += index_check(gi, x, dim, largest, wi)
        elif spec == "sort":
            gi = got[-1] if isinstance(got, list) else got
            wi = want_l[-1] if isinstance(want_l, list) else want_l
            gv = x.float().gather(-1, gi)
            wv = x.float().gather(-1, wi)
            if not (torch.equal(gv.isnan(), wv.isnan()) and torch.equal(gv.nan_to_num(), wv.nan_to_num())):
                bad.append(f"indices pick other values at {int(((gv != wv) & ~(gv.isnan() & wv.isnan())).sum())}")
        row["bad"] = bad
    except Exception as e:  # noqa: BLE001
        row["bad"] = [f"raised {type(e).__name__}: {str(e).splitlines()[0][:200]}"]
    print("PROBE " + json.dumps(row), flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enable", action="store_true")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--inference", action="store_true", help="run the card's calls under torch.inference_mode()")
    ap.add_argument("--only", default="")
    ap.add_argument("--sizes", default=",".join(map(str, SIZES)))
    a = ap.parse_args()
    global INFERENCE
    INFERENCE = a.inference
    if a.enable:
        import hpu_ops
        hpu_ops.enable()
    only = set(a.only.split(",")) if a.only else None
    sizes = [int(s) for s in a.sizes.split(",")]
    n_bad = 0
    for name, fn, build, kinds, spec in cases():
        if only and name not in only:
            continue
        for dtype in DTYPES:
            for n in (sizes if not name.startswith(("max_pool", "adaptive")) else [sizes[0]]):
                if a.compile:
                    torch._dynamo.reset()
                row = run_case(name, fn, build, kinds, spec, n, dtype, a.compile)
                n_bad += bool(row["bad"])
    print(f"DONE bad={n_bad}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
