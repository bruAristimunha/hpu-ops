"""On a Gaudi card: adversarial probes of the SVD family, matrix_exp, householder_product and the LU ops.
    PT_HPU_LAZY_MODE=0|1 python3 tests/adversarial_team_d.py [--compile]
Odd shapes (n = 1, a 2-D input, batch 1, broadcast batches), views (a transpose, an offset slice of a batch),
extreme scales (1e30, 1e-30), singular and rank-deficient inputs, two calls in a row, and with ``--compile``
the same calls under ``torch.compile(backend="hpu_backend")``. Each probe: no bridge fallback, no "runs on the CPU"
warning, the hpu_ops rewrite ran, finite outputs where torch's are finite, and the relative error against torch on
the CPU in float64 within max(1e-5, 10x CPU float32). One ``RESULT {json}`` line per probe; exits 1 on a failure."""
import argparse
import json
import math
import os
import sys
import warnings

import torch

import hpu_ops
from hpu_ops import _dispatch

LAZY = os.environ.get("PT_HPU_LAZY_MODE", "1") != "0"
L = torch.linalg


def fb_ops():
    from habana_frameworks.torch.hpu.metrics import metric_global

    return dict(dict(metric_global("cpu_fallback").stats()).get("FallbackOps", {}) or {})


def sync():
    import habana_frameworks.torch.core as htcore
    import habana_frameworks.torch.hpu as hthpu

    if LAZY:
        htcore.mark_step()
    hthpu.synchronize()


def leaves(x):
    if isinstance(x, torch.Tensor):
        return [x]
    if isinstance(x, (tuple, list)):
        return [t for y in x for t in leaves(y)]
    return [getattr(x, f) for f in getattr(x, "_fields", ()) if isinstance(getattr(x, f), torch.Tensor)]


def rel(got, want):
    got, want = got.detach().cpu().double(), want.detach().double()
    if got.shape != want.shape:
        return math.inf
    if not torch.isfinite(want).all():
        return 0.0 if torch.equal(torch.isfinite(got), torch.isfinite(want)) else math.inf
    if not torch.isfinite(got).all():
        return math.inf
    scale = float(want.abs().max()) if want.numel() else 0.0
    return float((got - want).abs().max()) / scale if scale > 0 else float((got - want).abs().max())


def gen(seed, *shape, scale=1.0):
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed)) * scale


def piv(seed, *shape, scale=1.0):
    return (gen(seed, *shape) + 0.5 * torch.eye(shape[-1])) * scale


def svd_inv(x):
    U, S, Vh = L.svd(x, full_matrices=False)
    return S, U @ torch.diag_embed(S) @ Vh


def probes():
    big = piv(5, 8, 6, 6)
    rank2 = gen(6, 4, 7, 2) @ gen(7, 4, 2, 5)
    sing = piv(8, 3, 5, 5)
    sing[1, :, 2] = 0.0
    H = gen(9, 3, 8, 5).tril(-1)
    tau = 2.0 / (1.0 + (H * H).sum(-2))
    skew = gen(10, 2, 7, 7)
    LU1, p1 = L.lu_factor(piv(11, 2, 1, 6, 6))  # factors with a batch of (2, 1), CPU
    yield "svd.n1", svd_inv, (gen(1, 1, 1),), lambda x: (x,)
    yield "svd.2d", svd_inv, (gen(1, 9, 4),), lambda x: (x,)
    yield "svd.transposed_view", svd_inv, (gen(2, 3, 4, 9),), lambda x: (x.mT,)
    yield "svd.scale_1e30", svd_inv, (gen(3, 4, 6, 6, scale=1e30),), lambda x: (x,)
    yield "svd.scale_1e-30", svd_inv, (gen(3, 4, 6, 6, scale=1e-30),), lambda x: (x,)
    yield "svdvals.rank_deficient", L.svdvals, (rank2,), lambda x: (x,)
    yield "pinv.rank_deficient", L.pinv, (rank2,), lambda x: (x,)
    yield "matrix_rank.rank_deficient", L.matrix_rank, (rank2,), lambda x: (x,)
    yield "matrix_exp.n1", L.matrix_exp, (gen(4, 1, 1),), lambda x: (x,)
    yield "matrix_exp.large_norm", L.matrix_exp, (skew * 20,), lambda x: (x,)
    yield "matrix_exp.offset_slice", L.matrix_exp, (gen(12, 6, 5, 5) * 0.3,), lambda x: (x[2:4],)
    yield "householder.batch1", L.householder_product, (H[:1], tau[:1]), lambda a, t: (a, t)
    yield "householder.k_lt_n", L.householder_product, (H, tau[:, :3]), lambda a, t: (a, t)
    yield "lu_factor.n1", lambda x: L.lu_factor(x), (gen(13, 1, 1),), lambda x: (x,)
    yield "lu_factor.2d", lambda x: L.lu_factor(x), (piv(14, 7, 7),), lambda x: (x,)
    yield "lu_factor.offset_slice", lambda x: L.lu_factor(x), (big,), lambda x: (x[2:5],)
    yield "lu_factor.transposed_view", lambda x: L.lu_factor(x), (big,), lambda x: (x.mT,)
    yield "lu_factor.singular_info", lambda x: L.lu_factor_ex(x), (sing,), lambda x: (x,)
    yield "lu_factor.scale_1e30", lambda x: L.lu_factor(x), (piv(15, 4, 6, 6, scale=1e30),), lambda x: (x,)
    yield "lu_factor.n32", lambda x: L.lu_factor(x), (piv(16, 2, 32, 32),), lambda x: (x,)
    yield "lu_solve.broadcast", lambda lu, p, b: L.lu_solve(lu, p, b), (LU1, p1, gen(17, 1, 3, 6, 2)), \
        lambda a, p, b: (a, p, b)
    yield "lu_solve.40_columns", lambda a, b: L.lu_solve(*L.lu_factor(a), b), (big, gen(18, 8, 6, 40)), \
        lambda a, b: (a, b)
    yield "lu_solve.adjoint_left", lambda a, b: L.lu_solve(*L.lu_factor(a), b, adjoint=True), \
        (big, gen(19, 8, 6, 3)), lambda a, b: (a, b)
    yield "lu_unpack.n1", lambda x: torch.lu_unpack(*L.lu_factor(x)), (gen(20, 3, 1, 1),), lambda x: (x,)
    yield "lu.2d", L.lu, (piv(21, 9, 9),), lambda x: (x,)
    yield "lu_factor.grad", lambda x: torch.autograd.grad(L.lu_factor(x)[0].sum(), x), (big,), lambda x: (x,)
    # no generator inside: torch 2.6's dynamo fails on sum(<genexpr>) under autograd.grad (also on the CPU)
    yield "lu.grad", lambda x: torch.autograd.grad((lambda P, L_, U: L_.sum() + U.sum())(*L.lu(x)), x), (big,), \
        lambda x: (x,)
    yield "lu_solve.grad", lambda a, b: torch.autograd.grad(L.lu_solve(*L.lu_factor(a), b).sum(), (a, b)), \
        (big, gen(22, 8, 6, 3)), lambda a, b: (a, b)


def run(name, fn, inputs, view, compile_):
    row = {"probe": name, "lazy": LAZY, "compile": compile_}
    grad = name.endswith(".grad")

    def prep(xs, dtype, device):
        xs = [x.to(device, dtype) if x.dtype.is_floating_point else x.to(device) for x in xs]
        if grad:
            xs = [x.requires_grad_() if x.dtype.is_floating_point else x for x in xs]
        return view(*xs)

    try:
        ref = leaves(fn(*prep(inputs, torch.float64, "cpu")))
        ref32 = leaves(fn(*prep(inputs, torch.float32, "cpu")))
        bar = max(1e-5, 10 * max((rel(a, b) for a, b in zip(ref32, ref, strict=True)), default=0.0))
        f = torch.compile(fn, backend="hpu_backend") if compile_ else fn
        calls0 = sum(_dispatch._STATE["calls"].values())
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            f0 = fb_ops()
            outs = []
            for _ in range(2):  # two calls in a row
                outs.append(leaves(f(*prep(inputs, torch.float32, "hpu"))))
                sync()
            fb = {k: v - f0.get(k, 0) for k, v in fb_ops().items() if v - f0.get(k, 0)}
        row["calls"] = sum(_dispatch._STATE["calls"].values()) - calls0
        row["fallback"] = fb
        row["cpu_warn"] = [str(w.message)[:120] for w in caught if "runs on the CPU" in str(w.message)]
        row["err"] = max(rel(a, b) for out in outs for a, b in zip(out, ref, strict=True))
        row["bar"] = bar
        problems = []
        if fb or row["cpu_warn"]:
            problems.append("fallback")
        if not row["calls"] and not compile_:
            problems.append("hpu_ops did not run")
        if not row["err"] <= bar:
            problems.append(f"error {row['err']:.2e} > {bar:.2e}")
        row["status"] = "ok" if not problems else "fail"
        row["problems"] = problems
    except Exception as e:  # noqa: BLE001
        row["status"] = "error"
        row["problems"] = [f"{type(e).__name__}: {str(e)[:300]}"]
    print("RESULT", json.dumps(row), flush=True)
    return row["status"] == "ok"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()
    hpu_ops.enable()
    results = [run(name, fn, inputs, view, args.compile) for name, fn, inputs, view in probes()]
    print(f"SUMMARY {len(results)} probes, {results.count(False)} failed", flush=True)
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
