"""On a Gaudi card: adversarial round 2 of hpu_ops' indexing rewrites (install_index, install_mask_fill).
    PT_HPU_LAZY_MODE=0|1 python3 tests/probe_adversarial_h.py [--only substr] [--compile] [--fused]
enable() runs without the two rewrites; each case runs in two arms in the same process: "native" (Habana's kernels)
and "fix" (the two rewrites installed on top). Per case and arm: max relative error of every output and gradient
against the CPU in float64 (float32 for integer data), two calls in a row, the bridge's cpu_fallback delta and the
"runs on the CPU" warnings. One CASE line per (case, arm), SUMMARY last."""
import argparse
import os
import warnings

import habana_frameworks.torch.core as htcore
import torch
from habana_frameworks.torch.hpu.metrics import metric_global

import hpu_ops
from hpu_ops import indexing

LAZY = os.environ.get("PT_HPU_LAZY_MODE", "1") != "0"
TRIU = {}


def triu(n):
    if n not in TRIU:
        TRIU[n] = torch.triu_indices(n, n)
    return TRIU[n]


def vech(x):
    r, c = triu(x.shape[-1])
    v = x[..., r, c]
    m = torch.ones_like(v)
    m[..., r != c] = 2 ** 0.5
    return v * m


def vech_plain(x):
    r, c = triu(x.shape[-1])
    return x[..., r, c]


def vec_to_sym(v, n):
    r, c = triu(n)
    m = torch.zeros(v.shape[:-1] + (n, n), dtype=v.dtype, device=v.device)
    m[..., r, c] = v
    m[..., c, r] = v
    return m


def dup(x):
    idx = torch.tensor([0, 0, 0, 1, 2, 2])
    return x[..., idx, idx]


def mask_value_grad(x, v):
    m = x * 1
    mask = torch.arange(x.shape[-1]) % 2 == 0
    m[..., mask] = v
    return m


def mask_2d(x):
    m = x * 1
    mask = torch.rand(x.shape[-2], x.shape[-1], generator=torch.Generator().manual_seed(3)) > 0.5
    m[:, mask] = 0.25
    return m


def mask_dev(x):
    m = x * 1
    mask = (torch.arange(x.shape[-1]) % 3 == 0).to(x.device)
    m[..., mask] = -1.5
    return m


def mask_complex_into_real(x):  # ATen raises: "value cannot be converted to type float without overflow"
    m = x * 1
    m[..., torch.arange(x.shape[-1]) % 2 == 0] = torch.tensor(1 + 2j)
    return m


def mask_fp16_overflow(x):  # ATen raises: 1e6 does not fit in float16
    m = x.half() * 1
    m[..., torch.arange(x.shape[-1]) % 2 == 0] = torch.tensor(1e6)
    return m


def mask_other_dtype(x):  # a float64 value into float32: ATen casts it
    m = x * 1
    m[..., torch.arange(x.shape[-1]) % 2 == 0] = torch.tensor(1.25, dtype=torch.float64)
    return m


def mask_expanded(x):  # ATen writes into a stride-0 view (with a deprecation warning)
    m = (x[:1] * 1).expand(x.shape)
    m[..., torch.arange(x.shape[-1]) % 2 == 0] = 0.5
    return m[:1].clone()


def mask_overlapping(x):  # as_strided view whose elements share memory
    b = x.flatten()[:12] * 1
    b.as_strided((3, 4), (1, 1))[:, torch.tensor([True, False, True, False])] = 1.0
    return b


def double_bwd(x):  # returns d/dx of sum(g^2), g = d/dx sum(vech(x)^3)
    y = (vech(x) ** 3).sum()
    g, = torch.autograd.grad(y, x, create_graph=True)
    return g


def oob_dev(x):
    return x[..., torch.tensor([0, x.shape[-1]], device=x.device)]


def neg_dev(x):
    return x[..., torch.tensor([-1, 0], device=x.device), torch.tensor([2, -3], device=x.device)]


def int32_dev(x):
    r, c = triu(x.shape[-1])
    return x[..., r.int().to(x.device), c.int().to(x.device)]


def noncontig_dev_index(x):
    r, c = triu(x.shape[-1])
    return x[..., r.to(x.device)[::2], c.to(x.device)[::2]]


def sym_roundtrip(x):
    n = x.shape[-1]
    return vech(vec_to_sym(vech(x), n))


def rand(*s, dtype=torch.float32, scale=1.0):
    return lambda: (torch.randn(*s) * scale).to(dtype)


def ints(*s, dtype):
    return lambda: torch.randint(-1000, 1000, s).to(dtype)


# name: (f, make (one tensor or a tuple), grad, tol, extra_args)
CASES = {
    "bf16 vech": (vech, rand(32, 22, 22, dtype=torch.bfloat16), True, 2e-2),
    "fp16 vech": (vech, rand(32, 22, 22, dtype=torch.float16), True, 2e-3),
    "float64 vech": (vech, rand(32, 22, 22, dtype=torch.float64), True, 1e-5),
    "int64 data": (vech_plain, ints(32, 12, 12, dtype=torch.int64), False, 0),
    "int32 data": (vech_plain, ints(32, 12, 12, dtype=torch.int32), False, 0),
    "bool data": (vech_plain, lambda: torch.rand(32, 12, 12) > 0.5, False, 0),
    "uint8 data": (vech_plain, lambda: torch.randint(0, 255, (32, 12, 12), dtype=torch.uint8), False, 0),
    "int8 data": (vech_plain, lambda: torch.randint(-100, 100, (32, 12, 12), dtype=torch.int8), False, 0),
    "dup idx b32": (dup, rand(32, 6, 6), True, 1e-5),
    "dup idx b512": (dup, rand(512, 6, 6), True, 1e-5),
    **{f"vech b{b}": (vech, rand(b, 12, 12), True, 1e-5) for b in (2, 3, 7, 8, 9, 16, 33, 500)},
    "vech n64 b64": (vech, rand(64, 64, 64), True, 1e-5),
    "vech 5-d broadcast": (vech, lambda: torch.randn(2, 1, 3, 12, 12).expand(2, 4, 3, 12, 12), True, 1e-5),
    "vech subnormal 1e-40": (vech_plain, rand(32, 12, 12, scale=1e-40), True, 1e-5),
    "vech near overflow 3e38": (vech_plain, rand(32, 12, 12, scale=3e38 / 5), True, 1e-5),
    "sym roundtrip": (sym_roundtrip, rand(32, 12, 12), True, 1e-5),
    "double backward": (double_bwd, rand(32, 8, 8), True, 1e-4),
    "mask value grad": (mask_value_grad, lambda: (torch.randn(32, 6, 6), torch.randn(())), True, 1e-5),
    "mask 2-d p=1": (mask_2d, rand(4, 9, 7), True, 1e-5),
    "mask on device": (mask_dev, rand(32, 6, 6), True, 1e-5),
    "mask bf16": (mask_dev, rand(32, 6, 6, dtype=torch.bfloat16), True, 2e-2),
    "mask int32 self": (mask_dev, ints(32, 6, 6, dtype=torch.int32), False, 0),
    "mask complex into real": (mask_complex_into_real, rand(32, 6, 6), False, 0),
    "mask fp16 overflow": (mask_fp16_overflow, rand(32, 6, 6), False, 0),
    "mask float64 value": (mask_other_dtype, rand(32, 6, 6), False, 0),
    "mask expanded view": (mask_expanded, rand(32, 6, 6), False, 0),
    "mask overlapping view": (mask_overlapping, rand(32, 6, 6), False, 0),
    "device idx out of range": (oob_dev, rand(32, 6, 6), False, 0),
    "device negative idx": (neg_dev, rand(32, 6, 6), True, 1e-5),
    "device int32 idx": (int32_dev, rand(32, 12, 12), True, 1e-5),
    "device strided idx": (noncontig_dev_index, rand(32, 12, 12), True, 1e-5),
}


def to_dev(xs, dev, grad, double=False):
    out = []
    for x in xs:
        x = x.detach().clone()
        if double and x.is_floating_point():
            x = x.double()
        x = x.to(dev)
        if grad and x.is_floating_point():
            x.requires_grad_(True)
        out.append(x)
    return out


def call(f, xs0, dev, grad, double=False, compiled=False, sync=True):
    xs = to_dev(xs0, dev, grad, double)
    g = torch.compile(f, backend="hpu_backend") if compiled else f
    out = g(*xs)
    res = [out]
    if grad:
        w = torch.linspace(0.5, 1.5, out.numel(), device=dev, dtype=out.dtype).reshape(out.shape)  # no cancellation
        (out * w).sum().backward()
        res += [x.grad for x in xs if x.requires_grad]
    if dev == "hpu" and sync:
        if LAZY:
            htcore.mark_step()
        torch.hpu.synchronize()
    return [r.detach() if dev == "hpu" and not sync else r.detach().cpu() for r in res]


def err(a, b):
    a, b = a.double(), b.double()
    if a.shape != b.shape:
        return float("inf")
    if not b.numel():
        return 0.0
    if torch.isnan(a).any() and not torch.isnan(b).any():
        return float("nan")  # a NaN the reference does not have fails (nan <= tol is False)
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        same = torch.equal(torch.isnan(a), torch.isnan(b)) and torch.equal(torch.nan_to_num(a), torch.nan_to_num(b))
        return 0.0 if same else float("inf")
    return float((a - b).abs().max() / b.abs().max().clamp_min(1e-300))


def fb():
    return dict(dict(metric_global("cpu_fallback").stats()).get("FallbackOps", {}) or {})


def run_case(name, f, make, grad, tol, compiled, fused=False):
    torch.manual_seed(0)
    xs0 = make()
    xs0 = list(xs0) if isinstance(xs0, tuple) else [xs0]
    try:
        ref = call(f, xs0, "cpu", grad, double=True)
        ref_note = ""
    except Exception as ex:  # noqa: BLE001
        ref, ref_note = None, "cpu_ref_error=" + repr(ex)[:80]
    f0 = fb()
    hpu_ops._STATE["calls"].clear()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            if fused:  # two calls in one lazy graph: no mark_step between them
                first = call(f, xs0, "hpu", grad, compiled=compiled, sync=False)
                second = call(f, xs0, "hpu", grad, compiled=compiled)
                runs = [[r.cpu() for r in first], second]
            else:
                runs = [call(f, xs0, "hpu", grad, compiled=compiled) for _ in range(2)]
            if ref is None:
                e, status = float("nan"), "NO_ERROR_RAISED"
            else:
                es = [err(a, b) for res in runs for a, b in zip(res, ref)]
                e = float("nan") if any(v != v for v in es) else max(es)  # max() drops a NaN after a number
                status = "ok" if e <= tol else "FAIL"
            dtypes = [str(r.dtype).replace("torch.", "") for r in runs[0]]
            msg = f"err={e:.2e} dtypes={dtypes}"
        except Exception as ex:  # noqa: BLE001
            status, msg = ("raised" if ref is None else "ERROR"), "exc=" + repr(ex)[:160]
    f1 = fb()
    fbd = {k: f1[k] - f0.get(k, 0) for k in f1 if f1[k] - f0.get(k, 0)}
    cpuw = sorted({str(w.message)[:90] for w in caught if "runs on the CPU" in str(w.message)})
    if (fbd or cpuw) and status == "ok":
        status = "FALLBACK"
    return status, f"{msg} {ref_note} fallbacks={fbd} cpu_warn={cpuw} rewrites={dict(hpu_ops._STATE['calls'])}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--fused", action="store_true", help="lazy: both calls in one graph (no mark_step between)")
    a = ap.parse_args()
    hpu_ops.FIXES = tuple(f for f in hpu_ops.FIXES if f not in (indexing.install_index, indexing.install_mask_fill))
    hpu_ops.enable()
    print("PLUGINS hpu_ops", hpu_ops.__file__, "lazy", int(LAZY), "compile", int(a.compile), "fused", int(a.fused),
          flush=True)
    bad = {"native": 0, "fix": 0}
    for name, (f, make, grad, tol) in CASES.items():
        if a.only and a.only not in name:
            continue
        for arm in ("native", "fix"):
            undo = None
            if arm == "fix":
                undo = hpu_ops._dispatch.chain([indexing.install_index("hpu"), indexing.install_mask_fill("hpu")])
            try:
                torch._dynamo.reset()
                status, msg = run_case(name, f, make, grad, tol, a.compile, a.fused)
            finally:
                if undo:
                    undo()
            bad[arm] += status not in ("ok", "raised")
            print(f"CASE lazy={int(LAZY)} compiled={int(a.compile)} {arm:<6} {name:<24} {status:<8} {msg}", flush=True)
    print(f"SUMMARY lazy={int(LAZY)} compiled={int(a.compile)} not_ok={bad}", flush=True)


if __name__ == "__main__":
    main()
