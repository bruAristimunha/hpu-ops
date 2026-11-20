"""On a Gaudi card: the linalg composites through their public torch calls, against torch on the CPU in float64.
    PT_HPU_LAZY_MODE=0|1 python3 tests/linalg_on_hpu.py [--baseline] [--with-eigh]
Per op and n in 16, 22, 64, 128 (batch 32, float32): forward, and backward where differentiable. A case fails when
the bridge ``cpu_fallback`` delta is not empty, when a warning says "runs on the CPU" (hpu_complex's boxed fallback
does not move the bridge metric, so both are read), or when the error, in forward or backward, exceeds
max(1e-3, 2 * the error of the same op in float32 on the CPU): no worse than CPU float32 by more than 2x. One float64
case per op: every output (eigh: w and V) matches the native CPU float64 result to 1e-12, and hpu-ops#16's warning
comes once. Two probes of Gaudi behavior: argmax
returns the first index among ties, and int64 argsort stays on the card.
One ``RESULT {json}`` line per case; the last line counts them; exits 1 if any case failed.
eigh is opt-in (not in hpu_ops.FIXES): its two cases are "expected fallback" rows, timed and never failed on the fallback,
unless ``--with-eigh`` installs it and checks it strictly (n <= 22 in lazy mode: n = 64 caused a device critical error).
``--baseline`` skips ``hpu_ops.enable()`` to time the stock stack: fallbacks are then recorded (status "fallback")
and do not fail a case, and the float64 cases are skipped (no warning to expect)."""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import sys
import time
import warnings
from typing import Callable, NamedTuple

import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.hpu as hthpu
import torch
from habana_frameworks.torch.hpu.metrics import metric_global

LAZY = os.environ.get("PT_HPU_LAZY_MODE", "1") != "0"
HPU = torch.device("hpu")
torch.manual_seed(0)

SIZES = (16, 22, 64, 128)
BATCH = 32
F64_N = 22
TOL32 = 1e-3
TOL64 = 1e-12
CPU_TEXT = "runs on the CPU"
F64_TEXT = "hpu-ops#16"
BASELINE = False
EIGH_FALLBACKS = []  # baseline: the fallback deltas seen for eigh (the positive control)
EIGH_STRICT = False  # eigh and eigvalsh are checked like the others (installed, or --with-eigh)
ops_linalg = None  # hpu_ops.linalg, imported in main
L = torch.linalg


def sync():
    if LAZY:
        htcore.mark_step()
    hthpu.synchronize()


def fb_ops():
    return dict(dict(metric_global("cpu_fallback").stats()).get("FallbackOps", {}) or {})


def delta(a, b):
    return {k: b[k] - a.get(k, 0) for k in b if b[k] - a.get(k, 0)}


def emit(**kw):
    print("RESULT " + json.dumps(kw, default=str), flush=True)


def spd(b, n, dtype, dev):
    a = torch.randn(b, n, n, dtype=dtype, device=dev)
    return a @ a.mH + n * torch.eye(n, dtype=dtype, device=dev)  # Hermitian PD for complex


def as_tensors(out):
    if isinstance(out, torch.Tensor):
        return [out]
    if isinstance(out, (tuple, list)) or hasattr(out, "_fields"):
        return [t for t in out if isinstance(t, torch.Tensor)]
    return []


# Inputs: SPD as in kernel_sweep, or a unit-scaled matrix (I + 0.05 G G^T / n, det near 1: float32 holds up to 3e38)
# with its rows permuted by a different random permutation per batch element. The diagonal is dominant, so only
# the permutation makes LU pivot; the det sign is the permutation's parity, and both signs occur in a batch.
def make_spd(b, n, dtype):
    return [spd(b, n, dtype, "cpu")]


def make_spd_unit(b, n, dtype):
    g = torch.randn(b, n, n, dtype=dtype)
    return [torch.eye(n, dtype=dtype) + 0.05 * (g @ g.mT) / n]


def pivoting(b, n, dtype):
    """Unit-scaled matrices with permuted rows; asserts that LU swaps and that both det signs occur."""
    M = make_spd_unit(b, n, dtype)[0]
    perm = torch.stack([torch.randperm(n) for _ in range(b)])
    A = M.gather(1, perm.unsqueeze(-1).expand(b, n, n))
    pivots = L.lu_factor(A).pivots
    swaps = int((pivots != torch.arange(1, n + 1, dtype=pivots.dtype)).sum())
    signs = L.slogdet(A).sign
    assert swaps > 0 and bool((signs > 0).any()) and bool((signs < 0).any()), (
        f"inputs do not exercise pivoting: {swaps} swaps, signs {signs.unique().tolist()}")
    return A


def general(b, n, dtype):
    return torch.randn(b, n, n, dtype=dtype) + 2 * math.sqrt(n) * torch.eye(n, dtype=dtype)


def make_triangular(b, n, dtype):
    return [spd(b, n, dtype, "cpu").tril(), torch.randn(b, n, 3, dtype=dtype)]


def make_cholesky_solve(b, n, dtype):
    return [torch.randn(b, n, 3, dtype=dtype), L.cholesky(spd(b, n, dtype, "cpu"))]


def make_solve(b, n, dtype):
    return [pivoting(b, n, dtype), torch.randn(b, n, 3, dtype=dtype)]


def make_pivoting(b, n, dtype):
    return [pivoting(b, n, dtype)]


def make_general(b, n, dtype):
    return [general(b, n, dtype)]


def rel_err(got, ref):
    """``max|got - ref| / max(1, max|ref|)`` in float64; NaN counts as infinite."""
    got, ref = got.detach().cpu().double(), ref.detach().double()
    if got.shape != ref.shape:
        return math.inf
    e = float((got - ref).abs().max() / max(1.0, float(ref.abs().max())))
    return math.inf if math.isnan(e) else e


def err_all(out, ref, ins):
    return max(rel_err(o, r) for o, r in zip(out, ref)), []


def err_slogdet(out, ref, ins):
    same = torch.equal(out[0].detach().cpu().double(), ref[0])
    return rel_err(out[1], ref[1]), ([] if same else ["sign differs"])


def err_eigh(out, ref, ins):
    """``w`` directly; V through ``V diag(w) V^T`` against S, which does not depend on the sign of V's columns."""
    w, V = (o.detach().cpu().double() for o in out)
    return max(rel_err(w, ref[0]), rel_err(V * w.unsqueeze(-2) @ V.mT, ins[0])), []


def err_qr(out, ref, ins):
    Q, R = (o.detach().cpu().double() for o in out)
    return rel_err(Q @ R, ins[0]), []


def loss_sum(out):
    return sum(o.sum() for o in out if o.requires_grad)


def loss_slogdet(out):
    return out[1].sum()


def loss_eigh(out):
    w, V = out
    return w.sum() + (V * w.unsqueeze(-2) @ V.mT).sum()  # no dependence on the sign of V's columns


def loss_qr(out):
    """Q and R are fixed only up to the signs of Q's columns, which differ between LAPACK on float64 and the card's
    float32 QR: ``sum(Q)`` or ``sum(R)`` alone gives gradients of opposite sign. ``Q R`` and ``|R|`` do not depend on them."""
    Q, R = out
    return (Q @ R).sum() + R.abs().sum()


class Case(NamedTuple):
    name: str
    make: Callable
    fn: Callable
    err: Callable = err_all
    loss: Callable | None = loss_sum  # None: not differentiable
    scope: str = "all"  # "bwd": only the backward must stay on the card (the forward is not ours)
    opt_in: bool = False  # eigh: not in hpu_ops.FIXES, so its fallback is expected unless --with-eigh
    float64: bool = True  # one of the ops that warns for float64 (hpu-ops#16)


CASES = (
    Case("linalg.solve_triangular", make_triangular, lambda a, b: L.solve_triangular(a, b, upper=False)),
    Case("cholesky_solve", make_cholesky_solve, lambda b, c: torch.cholesky_solve(b, c)),
    Case("linalg.solve", make_solve, lambda a, b: L.solve(a, b)),
    Case("linalg.inv", make_pivoting, lambda x: L.inv(x)),
    Case("linalg.det", make_pivoting, lambda x: L.det(x)),
    Case("linalg.slogdet", make_pivoting, lambda x: L.slogdet(x), err_slogdet, loss_slogdet),
    Case("linalg.eigh", make_spd, lambda x: L.eigh(x), err_eigh, loss_eigh, opt_in=True),
    Case("linalg.eigvalsh", make_spd, lambda x: L.eigvalsh(x), opt_in=True),
    Case("linalg.cholesky_bwd", make_spd, lambda x: L.cholesky(x), scope="bwd", float64=False),
    Case("linalg.qr_bwd", make_general, lambda x: L.qr(x), err_qr, loss_qr, scope="bwd", float64=False),
)


@contextlib.contextmanager
def watch():
    """Fills a dict with the bridge's CPU-fallback delta and the warnings raised inside the block. Sync inside."""
    rec = {}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        f0 = fb_ops()
        try:
            yield rec
        finally:
            rec["fallback"] = delta(f0, fb_ops())
            rec["caught"] = [(w.category, str(w.message)) for w in caught]


def cpu_warnings(rec, allow=None):
    return [m for _, m in rec["caught"] if CPU_TEXT in m and not (allow and allow in m)]


def forward(case, h_in):
    """The first call (the warm-up) and five timed calls, synced. Returns the first output and the ms per call."""
    with watch() as rec:
        out = as_tensors(case.fn(*h_in))
        sync()
        t = time.perf_counter()
        for _ in range(5):
            case.fn(*h_in)
        sync()
        ms = (time.perf_counter() - t) / 5 * 1e3
    return out, ms, rec


def reference_grads(case, ref_in):
    xs = [x.detach().clone().requires_grad_() for x in ref_in]
    case.loss(as_tensors(case.fn(*xs))).backward()
    return [x.grad for x in xs]


def grad_error(xs, ref_grads):
    return max(rel_err(x.grad, g) for x, g in zip(xs, ref_grads))


def backward(case, cpu_in, ref_in):
    """Backward alone inside the watch: the forward and the loss run, and sync, before it opens."""
    xs = [x.to(HPU).detach().requires_grad_() for x in cpu_in]
    loss = case.loss(as_tensors(case.fn(*xs)))
    sync()
    with watch() as rec:
        loss.backward()
        sync()
    ref_grads = reference_grads(case, ref_in)
    # the bar for the gradient is CPU float32's own error, measured on the same op
    cpu_xs = [x.detach().clone().requires_grad_() for x in cpu_in]
    case.loss(as_tensors(case.fn(*cpu_xs))).backward()
    return grad_error(xs, ref_grads), grad_error(cpu_xs, ref_grads), rec


def run_case(case, n):
    row = {"op": case.name, "n": n, "batch": BATCH, "dtype": "float32", "lazy": LAZY, "baseline": BASELINE}
    hard, soft = [], []  # wrong results or errors; CPU fallbacks
    try:
        cpu_in = case.make(BATCH, n, torch.float32)
        ref_in = [x.double() for x in cpu_in]
        ref = as_tensors(case.fn(*ref_in))
        out, row["hpu_ms"], rec = forward(case, [x.to(HPU) for x in cpu_in])
        row["fwd_fallback"] = rec["fallback"]
        if BASELINE and case.name == "linalg.eigh" and rec["fallback"]:
            EIGH_FALLBACKS.append(rec["fallback"])
        row["max_rel_err"], notes = case.err(out, ref, ref_in)
        row["err_cpu32"], _ = case.err(as_tensors(case.fn(*cpu_in)), ref, ref_in)
        tol = max(TOL32, 2 * row["err_cpu32"])
        hard += notes + ([f"error {row['max_rel_err']:.3g} > {tol:.3g}"] if not row["max_rel_err"] <= tol else [])
        if case.scope == "all":
            soft += ["forward fell back"] * bool(rec["fallback"]) + cpu_warnings(rec)
        if case.loss:
            row["grad_err"], row["grad_err_cpu32"], rec = backward(case, cpu_in, ref_in)
            gtol = max(TOL32, 2 * row["grad_err_cpu32"])
            row["bwd_fallback"] = rec["fallback"]
            hard += [f"gradient error {row['grad_err']:.3g} > {gtol:.3g}"] * (not row["grad_err"] <= gtol)
            soft += ["backward fell back"] * bool(rec["fallback"]) + cpu_warnings(rec)
    except Exception as e:  # noqa: BLE001  (a failing case is a row, not a crash)
        hard.append(f"exception {e!r}"[:300])
    expected = BASELINE or (case.opt_in and not EIGH_STRICT)  # fallbacks recorded, not failed
    problems = hard + ([] if expected else soft)
    row.update(status="fail" if problems else "fallback" if soft else "ok", problems=hard + soft)
    emit(**row)
    return not problems


def run_float64(case):
    """float64 on the card: declined, run on the CPU, with one warning naming hpu-ops#16 on the first call."""
    row = {"op": case.name, "n": F64_N, "batch": BATCH, "dtype": "float64", "lazy": LAZY, "baseline": BASELINE}
    if BASELINE or (case.opt_in and not EIGH_STRICT):
        emit(**row, status="skipped", problems=["no hpu_ops install for this op, no warning to check"])
        return True
    problems = []
    try:
        ops_linalg._WARNED.clear()  # the warning comes once per op name and process
        cpu_in = case.make(BATCH, F64_N, torch.float64)
        h_in = [x.to(HPU) for x in cpu_in]
        # Gaudi2 rounds float64 to float32 at every host-to-card copy (hpu-ops#16): the reference is the native CPU op
        # over what the card holds, its outputs rounded by the same copy (the decline path returns through .to(hpu)).
        cpu_in = [x.cpu() for x in h_in]
        ref = [r.to(HPU).cpu() for r in as_tensors(case.fn(*cpu_in))]
        with watch() as first:
            out = as_tensors(case.fn(*h_in))
            sync()
        with watch() as second:
            case.fn(*h_in)
            sync()
        row["max_rel_err"], problems = err_all(out, ref, cpu_in)  # every output directly, no reconstruction
        said = [m for c, m in first["caught"] if issubclass(c, UserWarning) and F64_TEXT in m]
        again = [m for _, m in second["caught"] if F64_TEXT in m]
        row.update(fwd_fallback=first["fallback"], warned=len(said), warned_again=len(again))
        problems += [f"error {row['max_rel_err']:.3g} > {TOL64}"] * (not row["max_rel_err"] <= TOL64)
        problems += [f"{len(said)} warnings naming {F64_TEXT} on the first call, expected 1"] * (len(said) != 1)
        problems += ["the warning came again"] * bool(again)
        problems += ["forward fell back"] * bool(first["fallback"] or second["fallback"])
        problems += cpu_warnings(first, F64_TEXT) + cpu_warnings(second, F64_TEXT)
    except Exception as e:  # noqa: BLE001  (a failing case is a row, not a crash)
        problems.append(f"exception {e!r}"[:300])
    emit(**row, status="fail" if problems else "ok", problems=problems)
    return not problems


def probe_argmax_ties():
    """Gaudi's argmax returns the first index among ties: [1, 3, 3, 0] gives 1."""
    row = {"op": "probe.argmax_first_of_ties", "lazy": LAZY, "baseline": BASELINE}
    problems = []
    try:
        with watch() as rec:
            got = int(torch.tensor([1.0, 3.0, 3.0, 0.0]).to(HPU).argmax().cpu())
            sync()
        row.update(got=got, fwd_fallback=rec["fallback"])
        problems += [f"argmax returned {got}, expected 1"] * (got != 1)
        problems += ["fell back"] * bool(rec["fallback"]) + cpu_warnings(rec)
    except Exception as e:  # noqa: BLE001  (a failing case is a row, not a crash)
        problems.append(f"exception {e!r}"[:300])
    emit(**row, status="fail" if problems else "ok", problems=problems)
    return not problems


def probe_argsort_int64():
    """int64 argsort runs on the card without a fallback (distinct keys, so the order is defined)."""
    row = {"op": "probe.argsort_int64", "lazy": LAZY, "baseline": BASELINE}
    problems = []
    try:
        x = torch.randperm(4096).to(torch.int64)
        with watch() as rec:
            got = x.to(HPU).argsort()
            sync()
        row.update(fwd_fallback=rec["fallback"])
        problems += ["argsort differs from the CPU's"] * (not torch.equal(got.cpu(), x.argsort()))
        problems += ["fell back"] * bool(rec["fallback"]) + cpu_warnings(rec)
    except Exception as e:  # noqa: BLE001  (a failing case is a row, not a crash)
        problems.append(f"exception {e!r}"[:300])
    emit(**row, status="fail" if problems else "ok", problems=problems)
    return not problems


def main():
    global BASELINE, EIGH_STRICT, ops_linalg
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--baseline", action="store_true", help="skip hpu_ops.enable()")
    parser.add_argument("--with-eigh", action="store_true",
                        help="install hpu_ops.eigh.install_eigh and check eigh and eigvalsh strictly (n <= 22 in lazy mode)")
    args = parser.parse_args()
    BASELINE = args.baseline
    import hpu_ops
    from hpu_ops import linalg as ops_linalg
    if not BASELINE:
        hpu_ops.enable()
        if args.with_eigh and hpu_ops.eigh.install_eigh not in hpu_ops.FIXES:
            hpu_ops.eigh.install_eigh("hpu")
        EIGH_STRICT = args.with_eigh or hpu_ops.eigh.install_eigh in hpu_ops.FIXES
    complex_state = "skipped (baseline)"
    if not BASELINE:
        try:
            import hpu_complex
            hpu_complex.enable()
            complex_state = "enabled " + str(getattr(hpu_complex, "__version__", "?"))
        except ImportError:
            complex_state = "not found"
    print("PLUGINS hpu_ops", getattr(hpu_ops, "__version__", "?"), "hpu_complex", complex_state, "baseline",
          BASELINE, "lazy", LAZY, flush=True)
    results = [probe_argmax_ties(), probe_argsort_int64()]
    for case in CASES:
        sizes = [n for n in SIZES if n <= 22] if case.opt_in and EIGH_STRICT and LAZY else SIZES  # n=64: device error
        results += [run_case(case, n) for n in sizes]
        if case.float64:
            results.append(run_float64(case))
    if BASELINE:  # positive control: without hpu_ops eigh falls back, else the metric key is wrong
        ok = bool(EIGH_FALLBACKS)
        emit(op="metric_check", lazy=LAZY, baseline=True, status="ok" if ok else "error",
             problems=[] if ok else ["no cpu_fallback delta for eigh under --baseline: the metric key is wrong"])
        results.append(ok)
    print(f"SUMMARY {len(results)} cases, {results.count(False)} failed", flush=True)
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
