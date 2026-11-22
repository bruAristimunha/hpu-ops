"""On a Gaudi card: adversarial inputs for the triangular solves on the TPC substitution kernel.
    PT_HPU_LAZY_MODE=0|1 python3 tests/tri_solve_adversarial.py [--compile]
Odd shapes (2-D, batch 1, broadcast batches both ways, a vector of right-hand sides as a column), strided and
offset views (a transposed triangle, a slice of a bigger batch, a transposed B), extreme scales (1e30, 1e-30),
gradients of A and B, float64 (hpu-ops#16: one warning, the CPU route, float64 accuracy), and with ``--compile``
the same calls under ``torch.compile(backend="hpu_backend")``. A row fails when its relative error against float64
on the CPU exceeds max(1e-6, 10x CPU float32's) (float64: 1e-12 against the CPU result over the inputs and output as the card stores them), when the bridge ``cpu_fallback`` delta is not empty (float32), or
when a warning says "runs on the CPU" (float32). One ``RESULT {json}`` line per row, then a SUMMARY."""
import argparse
import json
import math
import os
import sys
import warnings

import hpu_ops

hpu_ops.enable()

import habana_frameworks.torch.core as htcore  # noqa: E402
import habana_frameworks.torch.hpu as hthpu  # noqa: E402
import torch  # noqa: E402
from habana_frameworks.torch.hpu.metrics import metric_global  # noqa: E402

LAZY = os.environ.get("PT_HPU_LAZY_MODE", "1") != "0"
HPU = torch.device("hpu")
L = torch.linalg
G = torch.Generator().manual_seed(7)


def sync():
    if LAZY:
        htcore.mark_step()
    hthpu.synchronize()


def fb_ops():
    return dict(dict(metric_global("cpu_fallback").stats()).get("FallbackOps", {}) or {})


def rel_err(got, ref):
    got, ref = got.detach().cpu().double(), ref.detach().double()
    if got.shape != ref.shape:
        return math.inf
    e = float((got - ref).abs().max()) / max(float(ref.abs().max()), 1e-300)
    return math.inf if math.isnan(e) else e


def tri(*shape, upper=False, scale=1.0):
    n = shape[-1]
    off = torch.randn(*shape, generator=G) / n
    off = off.triu(1) if upper else off.tril(-1)
    return (off + torch.diag_embed(1 + torch.rand(*shape[:-1], generator=G))) * scale


def rn(*shape, scale=1.0):
    return torch.randn(*shape, generator=G) * scale


def st(upper=False, left=True, unit=False):
    return lambda a, b: L.solve_triangular(a, b, upper=upper, left=left, unitriangular=unit)


def chol(upper=False):
    return lambda b, c: torch.cholesky_solve(b, c, upper=upper)


def cases():
    big = tri(40, 22, 22)
    yield "2d", st(), (tri(22, 22), rn(22, 3))
    yield "batch1", st(upper=True), (tri(1, 22, 22, upper=True), rn(1, 22, 3))
    yield "n1_m1_batch1", st(), (tri(1, 1, 1), rn(1, 1, 1))
    yield "broadcast_A", st(), (tri(1, 22, 22), rn(4, 22, 5))
    yield "broadcast_both", st(), (tri(3, 1, 22, 22), rn(1, 5, 22, 2))
    yield "column_rhs", st(left=False), (tri(4, 40, 40), rn(4, 1, 40))
    yield "transposed_A", lambda a, b: L.solve_triangular(a.mT, b, upper=False), (tri(4, 40, 40, upper=True),
                                                                                     rn(4, 40, 3))
    yield "offset_slice_A", lambda a, b: L.solve_triangular(a[2:34], b, upper=False), (big, rn(32, 22, 3))
    yield "transposed_B", lambda a, b: L.solve_triangular(a, b.mT, upper=False), (tri(4, 64, 64), rn(4, 3, 64))
    yield "scale_1e30", st(), (tri(8, 40, 40, scale=1e30), rn(8, 40, 3, scale=1e30))
    yield "scale_1e-30", st(upper=True), (tri(8, 40, 40, upper=True, scale=1e-30), rn(8, 40, 3, scale=1e-30))
    yield "unit_ignores_diag", st(unit=True), (tri(8, 40, 40) * 1e20, rn(8, 40, 3))
    yield "chol_2d", chol(), (rn(40, 3), tri(40, 40))
    yield "chol_broadcast", chol(upper=True), (rn(4, 22, 3), tri(1, 22, 22, upper=True))
    yield "chol_scale_1e15", chol(), (rn(8, 22, 3), tri(8, 22, 22, scale=1e15))


def run(name, fn, ins, grad=False, dtype=torch.float32, compiled=False):
    row = {"op": name, "lazy": LAZY, "dtype": str(dtype).split(".")[1], "grad": grad, "compiled": compiled,
           "bridge": "1.21.4"}
    problems = []
    try:
        ins = [x.to(dtype) for x in ins]
        if dtype == torch.float64:  # the card holds float64 at float32: the reference takes the stored inputs
            ins = [x.to(HPU).cpu() for x in ins]
        ref_in = [x.detach().double().requires_grad_(grad) for x in ins]
        ref = fn(*ref_in)
        if dtype == torch.float64:  # the decline path returns through .to(hpu), which rounds the same way
            ref = ref.detach().to(HPU).cpu()
        cpu_in = [x.detach().float().clone().requires_grad_(grad) for x in ins]
        bar = rel_err(fn(*cpu_in), ref)
        g = rn(*ref.shape)
        if grad:
            (ref * g.double()).sum().backward()
            (fn(*cpu_in) * g).sum().backward()
            bar = max(bar, *(rel_err(c.grad, r.grad) for c, r in zip(cpu_in, ref_in)))
        h_in = [x.detach().to(HPU).requires_grad_(grad) for x in ins]
        f = torch.compile(fn, backend="hpu_backend") if compiled else fn
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            sync()
            f0 = fb_ops()
            out = f(*h_in)
            if grad:
                (out * g.to(HPU, dtype)).sum().backward()
            sync()
            err = rel_err(out, ref)
            if grad:
                err = max(err, *(rel_err(h.grad, r.grad) for h, r in zip(h_in, ref_in)))
            f1 = fb_ops()
        fb = {k: f1[k] - f0.get(k, 0) for k in f1 if f1[k] - f0.get(k, 0)}
        cpu = [str(w.message)[:160] for w in caught if "runs on the CPU" in str(w.message)]
        limit = 1e-12 if dtype == torch.float64 else max(1e-6, 10 * bar)
        row.update(err=err, err_cpu32=bar, fallback=fb, cpu_warnings=cpu)
        problems += [f"err {err:.3g} > {limit:.3g}"] * (not err <= limit)
        if dtype == torch.float32:
            problems += [f"fell back {fb}"] * bool(fb) + cpu
    except Exception as e:  # noqa: BLE001  (a failing case is a row)
        problems.append(f"exception {e!r}"[:300])
    print("RESULT " + json.dumps({**row, "status": "fail" if problems else "ok", "problems": problems}), flush=True)
    return not problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compile", action="store_true")
    args = ap.parse_args()
    print("PLUGINS", hpu_ops.__file__, "tpc_ready", hpu_ops.tpc.ready(), "lazy", LAZY, flush=True)
    results = []
    for name, fn, ins in cases():
        results.append(run(name, fn, ins))
        results.append(run(name, fn, ins, grad=True))
    results.append(run("float64", st(), (tri(4, 22, 22), rn(4, 22, 3)), dtype=torch.float64))
    results.append(run("chol_float64", chol(), (rn(4, 22, 3), tri(4, 22, 22)), dtype=torch.float64))
    if args.compile:
        for name, fn, ins in list(cases())[:6]:
            results.append(run(name, fn, ins, compiled=True))
            results.append(run(name, fn, ins, grad=True, compiled=True))
    print(f"SUMMARY {len(results)} rows, {results.count(False)} failed", flush=True)
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
