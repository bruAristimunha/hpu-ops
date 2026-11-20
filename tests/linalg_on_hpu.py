"""On a Gaudi card: the linalg composites through their public torch calls, against torch on the CPU in float64.
    PT_HPU_LAZY_MODE=0|1 python3 tests/linalg_on_hpu.py [--baseline]
Per op and n in 16, 22, 64, 128 (batch 32, float32): forward, and backward where differentiable. A case fails when
the bridge ``cpu_fallback`` delta is not empty, when a warning says "runs on the CPU" (hpu_complex's boxed fallback
does not move the bridge metric, so both are read), when an error or its CPU float32 bar is not finite, or when the
relative error ``max|x - ref| / max|ref|``, in forward or backward, exceeds max(1e-6, 10 * the error of the same op in
float32 on the CPU): no worse than 10x CPU float32. One float64 case per op: every output (eigh: w and V) matches the
native CPU float64 result to 1e-12, and hpu-ops#16's warning comes once. Two probes of Gaudi behavior: argmax returns
the first index among ties, and int64 argsort stays on the card.
One ``RESULT {json}`` line per case; the last line counts them; exits 1 if any case failed.
eigh and eigvalsh go through the TPC Jacobi kernel for n <= hpu_ops.tpc.EIGH_MAXN and are checked strictly there; past
it (n = 64, 128) Habana's fallback is expected and recorded. ``--with-eigh`` is accepted and ignored (eigh is in
hpu_ops.FIXES).
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
TOL32 = 1e-6  # floor of the float32 bar
CPU32_FACTOR = 10  # the card may be this many times worse than CPU float32
TOL64 = 1e-12
CPU_TEXT = "runs on the CPU"
F64_TEXT = "hpu-ops#16"
BASELINE = False
EIGH_FALLBACKS = []  # baseline: the fallback deltas seen for eigh (the positive control)
ops_linalg = None  # hpu_ops.linalg, imported in main
tpc_maxn = 32  # hpu_ops.tpc.MAXN, set in main
eigh_maxn = 32  # hpu_ops.tpc.EIGH_MAXN, set in main
SINGULAR_N = 22
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


def make_solve_right(b, n, dtype):
    return [pivoting(b, n, dtype), torch.randn(b, 3, n, dtype=dtype)]


def make_solve_vector(b, n, dtype):
    return [pivoting(b, n, dtype), torch.randn(b, n, dtype=dtype)]


def make_solve_2d(b, n, dtype):
    return [pivoting(b, n, dtype)[0], torch.randn(n, 3, dtype=dtype)]


def make_solve_chunked(b, n, dtype):
    return [pivoting(b, n, dtype), torch.randn(b, n, 70, dtype=dtype)]


def make_solve_broadcast(b, n, dtype):
    return [pivoting(b, n, dtype)[:1].contiguous(), torch.randn(b, n, 3, dtype=dtype)]


def make_solve_big(b, n, dtype):
    """A batch of ``b + 4`` matrices; the case solves with the slice ``[2:2 + b]``, a view with an offset."""
    return [pivoting(b + 4, n, dtype), torch.randn(b, n, 3, dtype=dtype)]


def make_triangular_upper_unit(b, n, dtype):
    return [torch.triu(torch.randn(b, n, n, dtype=dtype), 1) / n, torch.randn(b, n, 3, dtype=dtype)]


def make_cholesky_solve_upper(b, n, dtype):
    return [torch.randn(b, n, 3, dtype=dtype), L.cholesky(spd(b, n, dtype, "cpu"), upper=True)]


def make_pivoting(b, n, dtype):
    return [pivoting(b, n, dtype)]


def make_general(b, n, dtype):
    return [general(b, n, dtype)]


def rel_err(got, ref):
    """``max|got - ref| / max|ref|`` in float64 (absolute when ref is all zeros); NaN counts as infinite."""
    got, ref = got.detach().cpu().double(), ref.detach().double()
    if got.shape != ref.shape:
        return math.inf
    scale = float(ref.abs().max())
    e = float((got - ref).abs().max()) / (scale if scale > 0 else 1.0)
    return math.inf if math.isnan(e) else e


def err_all(out, ref, ins):
    if len(out) != len(ref):
        return math.inf, [f"{len(out)} outputs, expected {len(ref)}"]
    return max(rel_err(o, r) for o, r in zip(out, ref, strict=True)), []


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
    float32 QR: ``sum(Q)`` or ``sum(R)`` alone gives gradients of opposite sign. ``Q R`` and ``|R|`` do not depend on
    them."""
    Q, R = out
    return (Q @ R).sum() + R.abs().sum()


class Case(NamedTuple):
    name: str
    make: Callable
    fn: Callable
    err: Callable = err_all
    loss: Callable | None = loss_sum  # None: not differentiable
    scope: str = "all"  # "bwd": only the backward must stay on the card (the forward is not ours)
    float64: bool = True  # one of the ops that warns for float64 (hpu-ops#16)
    factor: int = CPU32_FACTOR  # the card may be this many times worse than CPU float32 on this op
    sizes: tuple | None = None  # only these n (the variants run at n = 22)
    # routed through a TPC kernel (n <= hpu_ops.tpc.MAXN, EIGH_MAXN for eigh); past it Habana's fallback is expected
    tpc: bool = False


CASES = (
    Case("linalg.solve_triangular", make_triangular, lambda a, b: L.solve_triangular(a, b, upper=False),
         tpc=True),
    Case("cholesky_solve", make_cholesky_solve, lambda b, c: torch.cholesky_solve(b, c), tpc=True),
    Case("linalg.solve", make_solve, lambda a, b: L.solve(a, b), tpc=True),
    Case("linalg.inv", make_pivoting, lambda x: L.inv(x), tpc=True),
    Case("linalg.det", make_pivoting, lambda x: L.det(x), tpc=True),
    Case("linalg.slogdet", make_pivoting, lambda x: L.slogdet(x), err_slogdet, loss_slogdet, tpc=True),
    Case("linalg.eigh", make_spd, lambda x: L.eigh(x), err_eigh, loss_eigh, tpc=True),
    Case("linalg.eigvalsh", make_spd, lambda x: L.eigvalsh(x), tpc=True),
    Case("linalg.solve.left_false", make_solve_right, lambda a, b: L.solve(a, b, left=False), float64=False,
         tpc=True, sizes=(22,)),
    Case("linalg.solve.vector_rhs", make_solve_vector, lambda a, b: L.solve(a, b), float64=False, tpc=True,
         sizes=(22,)),
    Case("linalg.solve.matrix_2d", make_solve_2d, lambda a, b: L.solve(a, b), float64=False, tpc=True, sizes=(22,)),
    Case("linalg.solve.m70_chunked", make_solve_chunked, lambda a, b: L.solve(a, b), float64=False, tpc=True,
         sizes=(22,)),
    Case("linalg.solve.broadcast_batch", make_solve_broadcast, lambda a, b: L.solve(a, b), float64=False, tpc=True,
         sizes=(22,)),
    Case("linalg.solve.aliased", make_pivoting, lambda a: L.solve(a, a), float64=False, tpc=True, sizes=(22,)),
    Case("linalg.solve.sliced_offset", make_solve_big, lambda big, b: L.solve(big[2:2 + b.shape[0]], b),
         float64=False, tpc=True, sizes=(22,)),
    Case("linalg.inv.of_batch_item", make_pivoting, lambda big: L.inv(big[3]), float64=False, tpc=True, sizes=(22,)),
    Case("linalg.solve_triangular.upper_unit", make_triangular_upper_unit,
         lambda a, b: L.solve_triangular(a, b, upper=True, unitriangular=True), float64=False, tpc=True, sizes=(22,)),
    Case("cholesky_solve.upper", make_cholesky_solve_upper, lambda b, c: torch.cholesky_solve(b, c, upper=True),
         float64=False, tpc=True, sizes=(22,)),
    Case("linalg.cholesky_bwd", make_spd, lambda x: L.cholesky(x), scope="bwd", float64=False),
    # factor 20: the forward is Habana's native qr, not an hpu_ops kernel (measured 10.2x at n=16)
    Case("linalg.qr_bwd", make_general, lambda x: L.qr(x), err_qr, loss_qr, scope="bwd", float64=False, factor=20),
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
    return max(rel_err(x.grad, g) for x, g in zip(xs, ref_grads, strict=True))


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


def over_bar(what, err, err_cpu32, factor=CPU32_FACTOR):
    """The problems of one error against its bar, max(TOL32, 10 * CPU float32's error); both must be finite."""
    if not (math.isfinite(err) and math.isfinite(err_cpu32)):
        return [f"{what} {err:.3g} or its CPU float32 bar {err_cpu32:.3g} is not finite"]
    tol = max(TOL32, factor * err_cpu32)
    return [f"{what} {err:.3g} > {tol:.3g}"] * (not err <= tol)


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
        hard += notes
        if case.scope == "all":  # a "bwd" case's forward is Habana's native op, not ours: recorded, not gated
            hard += over_bar("error", row["max_rel_err"], row["err_cpu32"], case.factor)
        if case.scope == "all":
            soft += ["forward fell back"] * bool(rec["fallback"]) + cpu_warnings(rec)
        if case.loss:
            row["grad_err"], row["grad_err_cpu32"], rec = backward(case, cpu_in, ref_in)
            row["bwd_fallback"] = rec["fallback"]
            hard += over_bar("gradient error", row["grad_err"], row["grad_err_cpu32"], case.factor)
            soft += ["backward fell back"] * bool(rec["fallback"]) + cpu_warnings(rec)
    except Exception as e:  # noqa: BLE001  (a failing case is a row, not a crash)
        hard.append(f"exception {e!r}"[:300])
    # fallbacks recorded, not failed: baseline, and the TPC ops past the kernel's n limit
    limit = eigh_maxn if case.name in ("linalg.eigh", "linalg.eigvalsh") else tpc_maxn
    expected = BASELINE or (case.tpc and n > limit)
    problems = hard + ([] if expected else soft)
    row.update(status="fail" if problems else "fallback" if soft else "ok", problems=hard + soft)
    emit(**row)
    return not problems


def run_float64(case):
    """float64 on the card: declined, run on the CPU, with one warning naming hpu-ops#16 on the first call."""
    row = {"op": case.name, "n": F64_N, "batch": BATCH, "dtype": "float64", "lazy": LAZY, "baseline": BASELINE}
    if BASELINE:
        emit(**row, status="skipped", problems=["baseline: no hpu_ops install, no warning to check"])
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


def singular_input():
    """float32 matrices, batch 32 at n = 22: element 1 all zero (native and LU of A report the same message), element
    2 with one zero row (a zero pivot in U, so det is exactly 0)."""
    A = pivoting(BATCH, SINGULAR_N, torch.float32)
    A[1] = 0
    A[2, 5] = 0
    return A


def run_singular(op, fn):
    """A singular batch on the card: ``det`` exactly 0 and ``slogdet`` (0, -inf) for elements 1 and 2; ``solve`` and
    ``inv`` raise torch's own message, captured from a CPU run; no fallback, no CPU warning."""
    row = {"op": f"singular.{op}", "n": SINGULAR_N, "batch": BATCH, "dtype": "float32", "lazy": LAZY,
           "baseline": BASELINE}
    problems = []
    try:
        A = singular_input()
        B = torch.randn(BATCH, SINGULAR_N, 3)
        call = (lambda x: fn(x, B.to(HPU))) if op == "solve" else fn
        call_cpu = (lambda x: fn(x, B)) if op == "solve" else fn
        raises, want = op in ("solve", "inv"), None
        if raises:
            try:
                call_cpu(A)
            except RuntimeError as e:
                want = str(e)
            assert want is not None, "torch did not raise on the CPU"
        with watch() as rec:
            try:
                got = as_tensors(call(A.to(HPU)))
                sync()
                err = None
            except RuntimeError as e:
                got, err = [], str(e)
        row.update(fwd_fallback=rec["fallback"], raised=err)
        if raises:
            problems += [f"raised {err!r}, expected {want!r}"] * (err != want)
        else:
            problems += [f"raised {err!r}"] * (err is not None)
            if not err:
                out = [t.cpu() for t in got]
                if op == "det":
                    problems += ["det of a singular element is not exactly 0"] * bool((out[0][1:3] != 0).any())
                else:
                    zero = bool((out[0][1:3] != 0).any() or not torch.isneginf(out[1][1:3]).all())
                    problems += ["slogdet of a singular element is not (0, -inf)"] * zero
        problems += ["fell back"] * bool(rec["fallback"]) + cpu_warnings(rec)
    except Exception as e:  # noqa: BLE001  (a failing case is a row, not a crash)
        problems.append(f"exception {e!r}"[:300])
    emit(**row, status="fail" if problems else "ok", problems=problems)
    return not problems


def double_backward_grads(solve, A, B, wx, wa, wb):
    """Gradients of a second-order loss through ``solve``: the first gradients kept in the graph, weighted, summed."""
    A, B = A.detach().requires_grad_(), B.detach().requires_grad_()
    x = solve(A, B)
    ga, gb = torch.autograd.grad((x * wx).sum(), (A, B), create_graph=True)
    (g2a, g2b) = torch.autograd.grad((ga * wa).sum() + (gb * wb).sum(), (A, B))
    return g2a, g2b


def run_double_backward(tol=1e-2):
    """``solve`` differentiated twice on the card in float32 against the CPU float64 result, bar ``tol``; no fallback,
    no CPU warning."""
    row = {"op": "linalg.solve.double_backward", "n": F64_N, "batch": BATCH, "dtype": "float32", "lazy": LAZY,
           "baseline": BASELINE}
    problems = []
    try:
        A, B = make_solve(BATCH, F64_N, torch.float32)
        wx, wa, wb = torch.randn_like(B), torch.randn_like(A), torch.randn_like(B)
        ref = double_backward_grads(L.solve, A.double(), B.double(), wx.double(), wa.double(), wb.double())
        with watch() as rec:
            got = double_backward_grads(L.solve, A.to(HPU), B.to(HPU), wx.to(HPU), wa.to(HPU), wb.to(HPU))
            sync()
        err = max(rel_err(g, r) for g, r in zip(got, ref, strict=True))
        row.update(max_rel_err=err, fwd_fallback=rec["fallback"])
        problems += [f"double-backward error {err:.3g} > {tol:.3g}"] * (not err <= tol)
        problems += ["fell back"] * bool(rec["fallback"]) + cpu_warnings(rec)
    except Exception as e:  # noqa: BLE001  (a failing case is a row, not a crash)
        problems.append(f"exception {e!r}"[:300])
    emit(**row, status="fail" if problems else "ok", problems=problems)
    return not problems


def rotated(b, n, diagonal, seed):
    """``Q diag(d) Q^T`` for a random orthogonal Q per batch element (QR of a normal matrix), float32."""
    g = torch.Generator().manual_seed(seed)
    Q = torch.linalg.qr(torch.randn(b, n, n, generator=g, dtype=torch.float64)).Q
    S = Q @ torch.diag_embed(diagonal.double()).expand(b, n, n) @ Q.mT
    return ((S + S.mT) / 2).float().contiguous()


def adversarial_input(kind, n, seed):
    """The eigh inputs the first kernel versions failed: dense repeated eigenvalues, condition 1e6 (max 1), and an
    extreme scale (1e30 times a unit-scale matrix)."""
    if kind == "deg_rotated":
        return rotated(BATCH, n, torch.tensor([1.0] * (n // 2) + [2.0] * (n - n // 2)), seed)
    if kind == "illcond_1e6":
        d = torch.logspace(0, 6, n, dtype=torch.float64)
        return rotated(BATCH, n, d / d.max(), seed)
    S = rotated(BATCH, n, torch.linspace(0.1, 1.0, n, dtype=torch.float64), seed)
    return (S.double() * 1e30).float().contiguous()


def eigh_metrics(w, V, S, w_ref):
    """Eigenvalue error and reconstruction error relative to ``max|w_ref|`` and ``max|S|``, and the orthogonality
    error ``max|V^T V - I|`` (absolute), all in float64."""
    w, V, S = w.detach().cpu().double(), V.detach().cpu().double(), S.double()
    eye = torch.eye(S.shape[-1], dtype=torch.float64)
    return {"w": rel_err(w, w_ref), "recon": rel_err(V * w.unsqueeze(-2) @ V.mT, S),
            "orth": float((V.mT @ V - eye).abs().max()) if torch.isfinite(V).all() else math.inf}


def run_eigh_adversarial(kind, n, seed):
    """eigh on the card for one adversarial input against the bar of 10x CPU float32's error on w, ``V diag(w) V^T``
    and orthogonality (floor 1e-6); finite outputs; no fallback, no CPU warning."""
    row = {"op": f"linalg.eigh.{kind}", "n": n, "seed": seed, "batch": BATCH, "dtype": "float32", "lazy": LAZY,
           "baseline": BASELINE}
    hard = []
    try:
        S = adversarial_input(kind, n, seed)
        w_ref = torch.linalg.eigvalsh(S.double())
        with watch() as rec:
            w, V = L.eigh(S.to(HPU))
            sync()
        got = eigh_metrics(w, V, S, w_ref)
        cw, cV = L.eigh(S)
        cpu = eigh_metrics(cw, cV, S, w_ref)
        row.update(errors=got, errors_cpu32=cpu, fwd_fallback=rec["fallback"])
        for name in got:
            hard += over_bar(f"{name} error", got[name], cpu[name])
        hard += ["fell back"] * bool(rec["fallback"]) + cpu_warnings(rec)
    except Exception as e:  # noqa: BLE001  (a failing case is a row, not a crash)
        hard.append(f"exception {e!r}"[:300])
    problems = [] if BASELINE else hard
    emit(**row, status="fail" if problems else "fallback" if hard else "ok", problems=hard)
    return not problems


def check_tpc(tpc):
    """After the first op: with ``hpu_ops.enable()`` the TPC extension must be ready, else every kernel-routed op
    silently fell back."""
    ready = tpc.ready()
    emit(op="tpc_check", lazy=LAZY, baseline=False, ready=ready, status="ok" if ready else "error",
         problems=[] if ready else ["hpu_ops.tpc.ready() is False: the TPC kernels are not in use (late enable() or "
                                    "the extension failed to build)"])
    return ready


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
    global BASELINE, ops_linalg, tpc_maxn, eigh_maxn
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--baseline", action="store_true", help="skip hpu_ops.enable()")
    parser.add_argument("--with-eigh", action="store_true",
                        help="no-op: eigh is installed by hpu_ops.enable()")
    args = parser.parse_args()
    BASELINE = args.baseline
    import hpu_ops
    from hpu_ops import linalg as ops_linalg
    from hpu_ops import tpc
    tpc_maxn, eigh_maxn = tpc.MAXN, tpc.EIGH_MAXN
    if not BASELINE:
        hpu_ops.enable()
    complex_state = "skipped (baseline)"
    if not BASELINE:
        try:
            import hpu_complex
            hpu_complex.enable()
            complex_state = "enabled " + str(getattr(hpu_complex, "__version__", "?"))
        except ImportError:
            complex_state = "not found"
    print("PLUGINS hpu_ops", getattr(hpu_ops, "__version__", "?"), "hpu_complex", complex_state, "baseline",
          BASELINE, "tpc_ready", None if BASELINE else tpc.ready(), "lazy", LAZY, flush=True)
    results = [probe_argmax_ties(), probe_argsort_int64()]
    if not BASELINE:
        results.append(check_tpc(tpc))
    for case in CASES:
        results += [run_case(case, n) for n in case.sizes or SIZES]
        if case.float64:
            results.append(run_float64(case))
    if not BASELINE:
        results.append(run_double_backward())
        results += [run_eigh_adversarial(kind, n, seed) for kind in ("deg_rotated", "illcond_1e6", "scaled_1e30")
                    for n in (16, 22) for seed in range(3)]
        for op, fn in (("solve", L.solve), ("inv", L.inv), ("det", L.det), ("slogdet", L.slogdet)):
            results.append(run_singular(op, fn))
    if BASELINE:  # positive control: without hpu_ops eigh falls back, else the metric key is wrong
        ok = bool(EIGH_FALLBACKS)
        emit(op="metric_check", lazy=LAZY, baseline=True, status="ok" if ok else "error",
             problems=[] if ok else ["no cpu_fallback delta for eigh under --baseline: the metric key is wrong"])
        results.append(ok)
    print(f"SUMMARY {len(results)} cases, {results.count(False)} failed", flush=True)
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
