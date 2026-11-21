"""On a Gaudi card: the linalg composites through their public torch calls, against torch on the CPU in float64.
    PT_HPU_LAZY_MODE=0|1 python3 tests/linalg_on_hpu.py [--baseline]
Per op and n in 16, 22, 64, 128 (batch 32, float32): forward, and backward where differentiable. The eight kernel ops
(solve, inv, det, slogdet, solve_triangular, cholesky_solve, eigh, eigvalsh) also run at the kernel's edges, n in
1, 3, 21, 31, 32, for correctness and fallback only (no timing loop), and eigh and eigvalsh on an indefinite
(G + G^T) / 2 at n = 3, 22, 32. A case fails when
the bridge ``cpu_fallback`` delta holds an op the case does not expect (see below), when a warning says "runs on the
CPU" (hpu_complex's boxed fallback does not move the bridge metric, so both are read), when an error or its CPU
float32 bar is not finite, or when the relative error ``max|x - ref| / max|ref|``, in forward or backward, exceeds
max(1e-6, 10 * the error of the same op in float32 on the CPU): no worse than 10x CPU float32. One float64 case per
op: every output (eigh: w and V) matches the native CPU float64 result to 1e-12, and hpu-ops#16's warning comes once.
Two probes of Gaudi behavior: argmax returns
the first index among ties, and int64 argsort stays on the card.
One ``RESULT {json}`` line per case; the last line counts them; exits 1 if any case failed.
eigh and eigvalsh go through the TPC Jacobi kernel for n <= hpu_ops.tpc.EIGH_MAXN and are checked strictly there; past
it (n = 64, 128) hpu-ops' CPU route is expected and recorded. ``--with-eigh`` is accepted and ignored (eigh is in
hpu_ops.FIXES).
``linalg.solve_triangular`` and ``cholesky_solve`` (and so the backward of ``linalg.cholesky`` and ``linalg.qr``) run
through the TPC substitution kernel for n <= hpu_ops.tpc.MAXN and are checked strictly there.
Expected fallbacks (the defaults of hpu_ops.FIXES): float64 ``solve``,
``det`` and ``slogdet`` have no backend kernel and their autograd kernel declines float64 (one hpu-ops#16 warning), so
Habana's fallback runs them. Each Case names these ops in ``fallback`` (float32) and ``fallback64`` (float64); a
fallback op outside the set fails the case, an expected op that did not fall back is a note (the kernel may have come
back). The accuracy bars still apply.
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
from torch.utils._python_dispatch import TorchDispatchMode

LAZY = os.environ.get("PT_HPU_LAZY_MODE", "1") != "0"
HPU = torch.device("hpu")
torch.manual_seed(0)

SIZES = (16, 22, 64, 128)
GATE_SIZES = (1, 3, 21, 31, 32)  # the kernels' edges: n = 1, odd sizes, one below, at the limit; no timing
INDEFINITE_SIZES = (3, 22, 32)
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
ops_tpc = None  # hpu_ops.tpc, imported in main
tpc_maxn = 64  # hpu_ops.tpc.MAXN, set in main
eigh_maxn = 32  # hpu_ops.tpc.EIGH_MAXN, set in main
small_maxn = 32  # hpu_ops.tpc.SMALL_MAXN (SVD family, matrix_exp, LU ops), set in main
SMALL_OPS = ("linalg.svd", "linalg.svdvals", "linalg.pinv", "linalg.lstsq", "linalg.matrix_norm", "linalg.matrix_rank",
             "linalg.matrix_exp", "linalg.lu", "lu_unpack")
SINGULAR_N = 22
L = torch.linalg
SOLVE_FB = frozenset({"aten::_linalg_solve_ex.result"})  # float64: no backend kernel, so Habana's fallback runs the op
DET_FB = frozenset({"aten::_linalg_det.result"})
SLOGDET_FB = frozenset({"aten::_linalg_slogdet.sign"})


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


def make_indefinite(b, n, dtype):
    g = torch.randn(b, n, n, dtype=dtype)
    return [(g + g.mT) / 2]


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
    # n = 1 has nothing to swap and one sign
    assert n < 2 or (swaps > 0 and bool((signs > 0).any()) and bool((signs < 0).any())), (
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


def make_triangular_right(b, n, dtype):
    return [spd(b, n, dtype, "cpu").tril(), torch.randn(b, 3, n, dtype=dtype)]


def make_triangular_upper_right(b, n, dtype):
    return [spd(b, n, dtype, "cpu").tril().mT.contiguous(), torch.randn(b, 3, n, dtype=dtype)]


def make_triangular_ill_cond(b, n, dtype):
    return [torch.tril(torch.randn(b, n, n, dtype=dtype) / n, -1) + torch.diag_embed(torch.logspace(-3, 0, n, dtype=dtype)),
            torch.randn(b, n, 3, dtype=dtype)]


def make_cholesky_ill_cond(b, n, dtype):
    return [torch.randn(b, n, 3, dtype=dtype),
            torch.tril(torch.randn(b, n, n, dtype=dtype) * (0.1 / n), -1)
            + torch.diag_embed(torch.logspace(-3, 0, n, dtype=dtype))]


def make_triangular_chunked(b, n, dtype):
    return [spd(b, n, dtype, "cpu").tril(), torch.randn(b, n, 70, dtype=dtype)]


def make_triangular_2d(b, n, dtype):
    return [spd(b, n, dtype, "cpu")[0].tril(), torch.randn(n, 3, dtype=dtype)]


def make_cholesky_2d(b, n, dtype):
    return [torch.randn(n, 3, dtype=dtype), L.cholesky(spd(b, n, dtype, "cpu")[0])]


def make_triangular_nan_other(b, n, dtype):
    a = spd(b, n, dtype, "cpu").tril()
    a[..., torch.ones(n, n, dtype=torch.bool).triu(1)] = float("nan")
    return [a, torch.randn(b, n, 3, dtype=dtype)]


def make_triangular_upper_unit_nan(b, n, dtype):
    a = torch.triu(torch.randn(b, n, n, dtype=dtype), 1) / n
    a[..., torch.ones(n, n, dtype=torch.bool).tril(0)] = float("nan")
    return [a, torch.randn(b, n, 3, dtype=dtype)]


def make_cholesky_nan_other(b, n, dtype):
    c = L.cholesky(spd(b, n, dtype, "cpu"))
    c[..., torch.ones(n, n, dtype=torch.bool).triu(1)] = float("nan")
    return [torch.randn(b, n, 3, dtype=dtype), c]


class _CpuQrForward(TorchDispatchMode):
    """Substitutes accurate CPU float32 QR factors in forward so backward exercises HPU solve_triangular alone."""

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func is torch.ops.aten.linalg_qr.default and args[0].device.type == "hpu":
            q, r = L.qr(args[0].detach().cpu())
            return q.to(args[0].device), r.to(args[0].device)
        return func(*args, **(kwargs or {}))


def qr_isolated(x):
    if x.device.type != "hpu":
        return L.qr(x)
    with _CpuQrForward():
        return L.qr(x)


def make_pivoting(b, n, dtype):
    return [pivoting(b, n, dtype)]


def make_general(b, n, dtype):
    return [general(b, n, dtype)]


def rel_err(got, ref):
    """``max|got - ref| / max|ref|`` in float64 (absolute when ref is all zeros); NaN counts as infinite."""
    got, ref = got.detach().cpu().double(), ref.detach().double()
    if got.shape != ref.shape:
        return math.inf
    if got.numel() == 0:
        return 0.0
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


def err_svd(out, ref, ins):
    U, S, Vh = (o.detach().cpu().double() for o in out)
    k = S.shape[-1]
    Uk, Vhk = U[..., :k], Vh[..., :k, :]
    rec_err = rel_err((Uk * S.unsqueeze(-2)) @ Vhk, ins[0])
    s_err = rel_err(S, ref[1])
    eye_u = torch.eye(U.shape[-1], dtype=torch.float64).expand(*U.shape[:-2], U.shape[-1], U.shape[-1])
    eye_v = torch.eye(Vh.shape[-2], dtype=torch.float64).expand(*Vh.shape[:-2], Vh.shape[-2], Vh.shape[-2])
    u_orth = rel_err(U.mT @ U, eye_u)
    v_orth = rel_err(Vh @ Vh.mT, eye_v)
    return max(s_err, rec_err, u_orth, v_orth), []


def err_lstsq(out, ref, ins):
    return rel_err(out[0], ref[0]), []


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


def loss_svd(out):
    U, S, Vh = out
    k = S.shape[-1]
    return S.sum() + ((U[..., :k] * S.unsqueeze(-2)) @ Vh[..., :k, :]).sum()


def loss_lstsq(out):
    return out[0].sum()


def make_rect_tall(b, n, dtype):
    return [torch.randn(b, n + 8, n, dtype=dtype)]


def make_rect_wide(b, n, dtype):
    return [torch.randn(b, n, n + 8, dtype=dtype)]


def make_lstsq(b, n, dtype):
    return [torch.randn(b, n + 6, n, dtype=dtype), torch.randn(b, n + 6, 3, dtype=dtype)]


def make_skew(b, n, dtype):
    a = torch.randn(b, n, n, dtype=dtype)
    return [0.1 * (a - a.mT)]


def make_householder(b, n, dtype):
    a = torch.randn(b, n, n, dtype=dtype).tril(-1)
    tau = 2.0 / (1.0 + (a * a).sum(-2))
    return [a, tau]


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
    # routed through a TPC kernel (n <= hpu_ops.tpc.MAXN, EIGH_MAXN for eigh); past it hpu-ops' CPU route is expected
    tpc: bool = False
    gate: bool = False  # also run at GATE_SIZES, without timing
    fallback: frozenset = frozenset()  # bridge ops expected in the float32 forward and backward fallback delta
    fallback64: frozenset = frozenset()  # the same for the float64 case
    warn64: bool = True  # the float64 case expects hpu-ops#16's warning once (else none: the op is not installed)


def fallback_problems(got, expected, what):
    """``(problems, notes)`` of a fallback delta ``got`` against the ops a case ``expected``: an op outside the set is
    a problem, an expected op that did not fall back a note (the kernel may have come back)."""
    extra, missing = sorted(set(got) - expected), sorted(expected - set(got))
    return [f"{what} fell back on {extra}"] * bool(extra), [f"{what}: expected {k} did not fall back" for k in missing]


# The LU ops are installed in eager mode only and linalg.lu not at all (measured slower than the fallback; see
# install_lu and HPU_OPS_LU_SLOW): their Habana fallback is expected, not a failure
LU_OFF_FB = frozenset({"aten::linalg_lu.out"})
LU_LAZY_FB = (frozenset({"aten::linalg_lu_factor_ex.out", "aten::linalg_lu_solve.out", "aten::lu_unpack.out"})
              if LAZY and os.environ.get("HPU_OPS_LU_SLOW") != "1" else frozenset())

CASES = (
    Case("linalg.solve_triangular", make_triangular, lambda a, b: L.solve_triangular(a, b, upper=False), tpc=True,
         gate=True),
    Case("cholesky_solve", make_cholesky_solve, lambda b, c: torch.cholesky_solve(b, c), tpc=True, gate=True),
    Case("linalg.solve", make_solve, lambda a, b: L.solve(a, b), tpc=True, gate=True, fallback64=SOLVE_FB),
    Case("linalg.inv", make_pivoting, lambda x: L.inv(x), tpc=True, gate=True),
    Case("linalg.det", make_pivoting, lambda x: L.det(x), tpc=True, gate=True, fallback64=DET_FB),
    Case("linalg.slogdet", make_pivoting, lambda x: L.slogdet(x), err_slogdet, loss_slogdet, tpc=True, gate=True,
         fallback64=SLOGDET_FB),
    Case("linalg.eigh", make_spd, lambda x: L.eigh(x), err_eigh, loss_eigh, tpc=True, gate=True),
    Case("linalg.eigvalsh", make_spd, lambda x: L.eigvalsh(x), tpc=True, gate=True),
    Case("linalg.eigh.indefinite", make_indefinite, lambda x: L.eigh(x), err_eigh, loss_eigh, float64=False, tpc=True,
         sizes=INDEFINITE_SIZES),
    Case("linalg.eigvalsh.indefinite", make_indefinite, lambda x: L.eigvalsh(x), float64=False, tpc=True,
         sizes=INDEFINITE_SIZES),
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
         lambda a, b: L.solve_triangular(a, b, upper=True, unitriangular=True), float64=False, tpc=True,
         sizes=(22,)),
    Case("cholesky_solve.upper", make_cholesky_solve_upper, lambda b, c: torch.cholesky_solve(b, c, upper=True),
         float64=False, tpc=True, sizes=(22,)),
    Case("linalg.cholesky_bwd", make_spd, lambda x: L.cholesky(x), scope="bwd", float64=False),
    # factor 20: the forward is Habana's native qr, not an hpu_ops kernel (measured 10.2x at n=16)
    Case("linalg.qr_bwd", make_general, lambda x: L.qr(x), err_qr, loss_qr, scope="bwd", float64=False, factor=20),
    # the same backward with CPU float32 QR factors: only hpu_ops' solve_triangular is measured
    Case("linalg.qr_bwd.isolated", make_general, qr_isolated, err_qr, loss_qr, scope="bwd", float64=False,
         factor=10, tpc=True),
    Case("linalg.solve_triangular.left_false", make_triangular_right,
         lambda a, b: L.solve_triangular(a, b, upper=False, left=False), float64=False, tpc=True, sizes=(16, 22, 64)),
    Case("linalg.solve_triangular.upper_left_false", make_triangular_upper_right,
         lambda a, b: L.solve_triangular(a, b, upper=True, left=False), float64=False, tpc=True, sizes=(16, 22, 64)),
    Case("linalg.solve_triangular.ill_cond", make_triangular_ill_cond,
         lambda a, b: L.solve_triangular(a, b, upper=False), float64=False, tpc=True, sizes=(16, 22, 64)),
    Case("cholesky_solve.ill_cond", make_cholesky_ill_cond,
         lambda b, c: torch.cholesky_solve(b, c), float64=False, tpc=True, sizes=(16, 22, 64)),
    Case("solve_triangular.m70_chunked", make_triangular_chunked,
         lambda a, b: L.solve_triangular(a, b, upper=False), float64=False, tpc=True, sizes=(16, 22, 64)),
    Case("solve_triangular.2d", make_triangular_2d,
         lambda a, b: L.solve_triangular(a, b, upper=False), float64=False, tpc=True, sizes=(16, 22, 64)),
    Case("cholesky_solve.2d", make_cholesky_2d,
         lambda b, c: torch.cholesky_solve(b, c), float64=False, tpc=True, sizes=(16, 22, 64)),
    Case("linalg.solve_triangular.ignored_upper_nan", make_triangular_nan_other,
         lambda a, b: L.solve_triangular(a, b, upper=False), float64=False, tpc=True, sizes=(22,)),
    Case("linalg.solve_triangular.ignored_diag_lower_nan", make_triangular_upper_unit_nan,
         lambda a, b: L.solve_triangular(a, b, upper=True, unitriangular=True), float64=False, tpc=True, sizes=(22,)),
    Case("cholesky_solve.ignored_upper_nan", make_cholesky_nan_other,
         lambda b, c: torch.cholesky_solve(b, c), loss=None, float64=False, tpc=True, sizes=(22,)),
    Case("linalg.svd", make_general, lambda x: L.svd(x, full_matrices=False), err_svd, loss_svd, tpc=True),
    Case("linalg.svd_tall", make_rect_tall, lambda x: L.svd(x, full_matrices=False), err_svd, loss_svd, tpc=True),
    Case("linalg.svd_tall_full", make_rect_tall, lambda x: L.svd(x, full_matrices=True), err_svd, loss_svd, tpc=True),
    Case("linalg.svd_wide", make_rect_wide, lambda x: L.svd(x, full_matrices=False), err_svd, loss_svd, tpc=True),
    Case("linalg.svd_wide_full", make_rect_wide, lambda x: L.svd(x, full_matrices=True), err_svd, loss_svd, tpc=True),
    Case("linalg.svdvals", make_general, lambda x: L.svdvals(x), tpc=True),
    Case("linalg.pinv", make_general, lambda x: L.pinv(x), tpc=True),
    Case("linalg.lstsq", make_lstsq, lambda a, b: L.lstsq(a, b), err_lstsq, loss_lstsq, tpc=True),
    Case("linalg.matrix_norm_2", make_general, lambda x: L.matrix_norm(x, ord=2), tpc=True),
    Case("linalg.matrix_norm_neg2", make_general, lambda x: L.matrix_norm(x, ord=-2), tpc=True),
    # matrix_norm_nuc sums S on HPU at float32 precision after _linalg_svd declines float64 (~6.7e-8 vs CPU f64)
    Case("linalg.matrix_norm_nuc", make_general, lambda x: L.matrix_norm(x, ord="nuc"), float64=False, tpc=True),
    Case("linalg.matrix_rank", make_general, lambda x: L.matrix_rank(x), err_all, None, tpc=True),
    Case("linalg.matrix_exp", make_skew, lambda x: L.matrix_exp(x), tpc=True),
    Case("linalg.householder_product", make_householder, lambda a, t: L.householder_product(a, t)),
    # standalone LU (#10): the TPC LU kernel (LAPACK's factors and pivots) and the pivots-to-permutation kernel
    Case("linalg.lu_factor", make_pivoting, lambda x: L.lu_factor(x), float64=False, tpc=True, gate=True,
         fallback=LU_LAZY_FB),
    Case("linalg.lu_factor_ex", make_pivoting, lambda x: L.lu_factor_ex(x), float64=False, tpc=True, gate=True,
         fallback=LU_LAZY_FB),
    Case("linalg.lu", make_pivoting, lambda x: L.lu(x), float64=False, tpc=True, gate=True,
         fallback=LU_OFF_FB | LU_LAZY_FB),
    Case("linalg.lu_solve", make_solve, lambda a, b: L.lu_solve(*L.lu_factor(a), b), float64=False, tpc=True,
         gate=True, fallback=LU_LAZY_FB),
    Case("linalg.lu_solve.adjoint_right", make_solve_right,
         lambda a, b: L.lu_solve(*L.lu_factor(a), b, left=False, adjoint=True), float64=False, tpc=True, sizes=(22,),
         fallback=LU_LAZY_FB),
    Case("lu_unpack", make_pivoting, lambda x: torch.lu_unpack(*L.lu_factor(x)), float64=False, tpc=True,
         gate=True, fallback=LU_LAZY_FB),
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


TIMED_CALLS = 20


def forward(case, h_in, timed=True):
    """The first call (the warm-up) and, if ``timed``, TIMED_CALLS synced calls, keeping the outputs alive (lazy mode
    skips work whose outputs are dropped). Returns the first output and the ms per call (None when not timed)."""
    ms = None
    with watch() as rec:
        out = as_tensors(case.fn(*h_in))
        sync()
        if timed:
            outs = []
            t = time.perf_counter()
            for _ in range(TIMED_CALLS):
                outs.append(case.fn(*h_in))
                if LAZY:
                    sync()
            if not LAZY:
                sync()
            ms = (time.perf_counter() - t) / TIMED_CALLS * 1e3
            del outs
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


def run_case(case, n, timed=True):
    row = {"op": case.name, "n": n, "batch": BATCH, "dtype": "float32", "lazy": LAZY, "baseline": BASELINE,
           "timed": timed}
    hard, soft, notes = [], [], []  # wrong results or errors; unexpected fallbacks; expected ones that did not occur
    try:
        cpu_in = case.make(BATCH, n, torch.float32)
        ref_in = [x.double() for x in cpu_in]
        ref = as_tensors(case.fn(*ref_in))
        out, row["hpu_ms"], rec = forward(case, [x.to(HPU) for x in cpu_in], timed)
        row["fwd_fallback"] = rec["fallback"]
        if BASELINE and case.name == "linalg.eigh" and rec["fallback"]:
            EIGH_FALLBACKS.append(rec["fallback"])
        row["max_rel_err"], wrong = case.err(out, ref, ref_in)
        row["err_cpu32"], _ = case.err(as_tensors(case.fn(*cpu_in)), ref, ref_in)
        hard += wrong
        if case.scope == "all":  # a "bwd" case's forward is Habana's native op, not ours: recorded, not gated
            hard += over_bar("error", row["max_rel_err"], row["err_cpu32"], case.factor)
            bad, note = fallback_problems(rec["fallback"], case.fallback, "forward")
            soft += bad + cpu_warnings(rec)
            notes += note
        if case.loss:
            row["grad_err"], row["grad_err_cpu32"], rec = backward(case, cpu_in, ref_in)
            row["bwd_fallback"] = rec["fallback"]
            hard += over_bar("gradient error", row["grad_err"], row["grad_err_cpu32"], case.factor)
            bad, note = fallback_problems(rec["fallback"], case.fallback, "backward")
            soft += bad + cpu_warnings(rec)
            notes += note
    except Exception as e:  # noqa: BLE001  (a failing case is a row, not a crash)
        hard.append(f"exception {e!r}"[:300])
    # fallbacks recorded, not failed: baseline, and the TPC ops past the kernel's n limit
    limit = (eigh_maxn if case.name in ("linalg.eigh", "linalg.eigvalsh")
             else small_maxn if case.name.startswith(SMALL_OPS) else tpc_maxn)
    expected = BASELINE or (case.tpc and n > limit)
    problems = hard + ([] if expected else soft)
    row.update(status="fail" if problems else "fallback" if soft else "ok", problems=hard + soft, notes=notes)
    emit(**row)
    return not problems


def run_float64(case):
    """float64 on the card: declined, with one warning naming hpu-ops#16 on the first call. eigh, eigvalsh and inv run
    on hpu-ops' CPU route, and so do solve_triangular and cholesky_solve. solve, det and slogdet warn too (the
    autograd kernel's check), then Habana's fallback runs them. The case's
    ``fallback64`` is the expected delta, and an op outside it fails."""
    row = {"op": case.name, "n": F64_N, "batch": BATCH, "dtype": "float64", "lazy": LAZY, "baseline": BASELINE}
    if BASELINE:
        emit(**row, status="skipped", problems=["baseline: no hpu_ops install, no warning to check"])
        return True
    problems, notes = [], []
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
        want = int(case.warn64)
        problems += [f"{len(said)} warnings naming {F64_TEXT} on the first call, expected {want}"] * (len(said) != want)
        problems += ["the warning came again"] * bool(again)
        for rec in (first, second):
            bad, note = fallback_problems(rec["fallback"], case.fallback64, "forward")
            problems += bad
            notes += note
        problems += cpu_warnings(first, F64_TEXT) + cpu_warnings(second, F64_TEXT)
    except Exception as e:  # noqa: BLE001  (a failing case is a row, not a crash)
        problems.append(f"exception {e!r}"[:300])
    emit(**row, status="fail" if problems else "ok", problems=problems, notes=sorted(set(notes)))
    return not problems


def singular_input():
    """float32 matrices, batch 32 at n = 22: element 1 all zero (native and LU of A report the same message), element
    2 with one zero row (a zero pivot in U, so det is exactly 0)."""
    A = pivoting(BATCH, SINGULAR_N, torch.float32)
    A[1] = 0
    A[2, 5] = 0
    return A


def singular_triangle(op):
    """Batch 32 at n = 22: well-conditioned triangles (the lower one, as a Cholesky factor too), element 1 with a zero
    diagonal entry and nonzero entries below it."""
    n = SINGULAR_N
    T = torch.tril(0.3 * torch.randn(BATCH, n, n)) + 2 * torch.eye(n)
    T[1, 5, 5] = 0
    return T


def run_singular_triangle(op, fn, expected):
    """``solve_triangular`` and ``cholesky_solve`` of a triangle with a zero diagonal entry: non-finite for that
    element, finite for the others, as torch's; no fallback outside ``expected``, no CPU warning."""
    row = {"op": f"singular.{op}", "n": SINGULAR_N, "batch": BATCH, "dtype": "float32", "lazy": LAZY,
           "baseline": BASELINE}
    problems, notes = [], []
    try:
        T, B = singular_triangle(op), torch.randn(BATCH, SINGULAR_N, 3)
        with watch() as rec:
            got = fn(T.to(HPU), B.to(HPU)).cpu()
            sync()
        row["fwd_fallback"] = rec["fallback"]
        problems += ["the singular element is finite"] * bool(torch.isfinite(got[1]).all())
        problems += ["a regular element is not finite"] * (not bool(torch.isfinite(got[2:]).all()))
        bad, notes = fallback_problems(rec["fallback"], expected, "forward")
        problems += bad + cpu_warnings(rec)
    except Exception as e:  # noqa: BLE001  (a failing case is a row, not a crash)
        problems.append(f"exception {e!r}"[:300])
    emit(**row, status="fail" if problems else "ok", problems=problems, notes=notes)
    return not problems


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


# ---- the Codex review of PR #25 (REVISION.md): its inputs, each against torch on the CPU ----
S64, T64, BIG = 2.0 ** 64, 2.0 ** -64, 2.0 ** 127
CPU_ROUTE = []  # calls of hpu-ops' own CPU route (hpu_ops.linalg._on_cpu / _out_on_cpu), counted in main


def embed(block, n):
    """``eye(n)`` with ``block`` in its upper-left corner."""
    A = torch.eye(n)
    block = torch.tensor(block, dtype=torch.float32)
    A[:block.shape[0], :block.shape[1]] = block
    return A


def unit_rhs(n, rows, value=1.0):
    B = torch.zeros(n, 1)
    B[rows] = value
    return B


def _det_grad_create_graph(A):
    A = A.detach().requires_grad_()
    return torch.autograd.grad(L.det(A).sum(), A, create_graph=True)[0]


def _cloned_coefficient_grad(A, B):
    B = B.detach().requires_grad_()
    torch.triangular_solve(B, A, upper=False).cloned_coefficient.sum().backward()
    return B.grad


def _slogdet_sign_backward(A):
    A = A.detach().requires_grad_()
    sign, _ = L.slogdet(A)
    sign.backward()
    return torch.tensor(float(A.grad is None))


def _out_int(A, B):
    return L.solve_triangular(A, B, upper=False, out=torch.empty(B.shape, dtype=torch.long, device=B.device))


def _out_nonleaf(A, B):
    leaf = torch.ones(B.shape, device=B.device, requires_grad=True)
    return L.solve_triangular(A, B, upper=False, out=leaf * 2)


def _out_layout(A, B):
    out = torch.empty(0, device=B.device)
    L.solve_triangular(A, B, upper=False, out=out)
    return out, torch.tensor(out.stride(), dtype=torch.float32)


def _legacy_out(A, B):
    X, M = torch.empty(0, device=B.device), torch.empty(0, device=B.device)
    torch.triangular_solve(B, A, upper=False, out=(X, M))
    return X, M, torch.tensor(X.stride() + M.stride(), dtype=torch.float32)


def _inference_offset_view(n):
    def f(big, B):
        # inv and solve_triangular: their backend kernels take inference mode (solve's is Habana's, by design)
        with torch.inference_mode():
            big = big * 1  # an inference tensor: its slices have ``_base`` None
            view = big[1:2]
            return (L.inv(view), L.inv(view), L.solve_triangular(view, B, upper=False),
                    L.solve_triangular(view, B, upper=True))  # two calls in a row each: lazy
    return f


def _tri_upper_unit_overflow(n):
    A, B = torch.eye(n), torch.zeros(n, 1)
    A[n - 3, n - 2:] = 1
    B[n - 3:] = BIG
    return A, B


def _tri_lower_unit_overflow(n):
    A, B = torch.eye(n), torch.zeros(n, 1)
    A[2, :2] = 1
    B[:3] = BIG
    return A, B


def codex_cases():
    """``(name, fn, inputs, expect)``: ``expect`` "torch" compares with CPU torch on the same inputs (both raise, or
    the outputs match to 1e-6 relative with the same NaN/Inf pattern); "float64" the same against CPU torch in
    float64, rounded to float32 (the 2^127 triangles: CPU float32's own trsm overflows on the pod's BLAS, measured
    -inf, while it is finite on Apple's); "raise" expects an error on the card."""
    for n in (2, 33):  # 2: the n <= 32 range of main's kernel too
        dup = embed([[3, 4], [3, 4]], n)
        yield f"codex1.solve_duplicate_rows.n{n}", L.solve, (dup, unit_rhs(n, 1)), "torch"
        yield f"codex1.solve_ex_info_duplicate_rows.n{n}", lambda a, b: L.solve_ex(a, b).info.float(), \
            (dup, unit_rhs(n, 1)), "torch"
        yield f"codex1.det_duplicate_rows.n{n}", L.det, (dup,), "torch"
        yield f"codex1.inv_duplicate_rows.n{n}", L.inv, (dup,), "torch"
        colmajor = embed([[0.5, 0.5], [0.5, -0.5]], n).mT.contiguous().mT
        yield f"codex3.solve_rhs_2^127.n{n}", L.solve, (colmajor, unit_rhs(n, 0, BIG)), "torch"
        scaled = embed([[S64, T64], [S64, 2 * T64]], n)
        yield f"codex4.det_pivot_2^64.n{n}", L.det, (scaled,), "torch"
        yield f"codex4.slogdet_pivot_2^64.n{n}", L.slogdet, (scaled,), "torch"
        yield f"codex4.solve_pivot_2^64.n{n}", L.solve, (scaled, unit_rhs(n, 1)), "torch"
        rank1 = embed([[2.0 ** 30, 2.0 ** 127], [2.0 ** -97, 1.0]], n)  # row 2 = 2^-127 x row 1
        yield f"rereview.solve_ex_info_tiny_multiplier.n{n}", lambda a, b: L.solve_ex(a, b).info.float(), \
            (rank1, torch.ones(n, 1)), "torch"
        yield f"rereview.det_tiny_multiplier.n{n}", L.det, (rank1,), "torch"
        det1016 = embed([[1024.0, 2.0 ** 120], [2.0 ** -117, 1.0]], n)  # det 1016
        yield f"rereview.det_1016.n{n}", L.det, (det1016,), "float64"
        yield f"rereview.solve_det_1016.n{n}", L.solve, (det1016, torch.ones(n, 1)), "float64"
    yield "codex2.tri_lower_unit_2^127.n33", lambda a, b: L.solve_triangular(a, b, upper=False, unitriangular=True), \
        _tri_lower_unit_overflow(33), "float64"
    yield "codex2.tri_upper_unit_2^127.n33", lambda a, b: L.solve_triangular(a, b, upper=True, unitriangular=True), \
        _tri_upper_unit_overflow(33), "float64"
    tri = torch.tril(torch.randn(2, 22, 22)) / 22 + 2 * torch.eye(22)
    rhs = torch.randn(2, 22, 3)
    yield "codex5.solve_triangular_out_int", _out_int, (tri, rhs), "torch"
    yield "codex5.solve_triangular_out_nonleaf_grad", _out_nonleaf, (tri, rhs), "torch"
    yield "codex6.cloned_coefficient_B_grad", _cloned_coefficient_grad, (tri, rhs), "torch"
    big = torch.tril(torch.randn(65, 65)) / 65 + 2 * torch.eye(65)
    yield "codex7.legacy_mixed_devices_n65", lambda a, b: torch.triangular_solve(b.cpu(), a), \
        (big, torch.ones(65, 1)), "raise"
    yield "codex7.legacy_mixed_devices_n33", lambda a, b: torch.triangular_solve(b.cpu(), a), \
        (big[:33, :33], torch.ones(33, 1)), "raise"
    singular = torch.eye(33)
    singular[-1, -1] = 0
    yield "codex8.det_double_backward_singular.n33", _det_grad_create_graph, \
        (torch.stack([singular, 2 * torch.eye(33)]),), "torch"
    yield "codex10.inference_offset_view.n33", _inference_offset_view(33), \
        (torch.stack([torch.eye(33), 2 * torch.eye(33)]), torch.ones(1, 33, 2)), "torch"
    yield "codex.minor.legacy_out_X", _legacy_out, (tri, rhs), "torch"
    yield "codex.minor.out_layout", _out_layout, (tri, rhs), "torch"
    yield "codex.minor.slogdet_sign_backward.n33", _slogdet_sign_backward, (torch.eye(33) * 2,), "torch"
    yield "codex.minor.tpc_tri_solve_float64", lambda a, b: ops_tpc.tri_solve(a.double(), b.double()), \
        (tri, rhs), "raise"
    yield "codex.minor.raw_tri_solve_4d", lambda a, b: ops_tpc._tri_op(False, False)(
        a.reshape(1, 2, 22, 22).expand(2, 2, 22, 22).contiguous(), b.reshape(1, 2, 22, 3).expand(2, 2, 22, 3)
        .contiguous()), (tri, rhs), "raise"


def run_codex_case(name, fn, ins, expect):
    row = {"op": name, "dtype": "float32", "lazy": LAZY, "baseline": BASELINE, "expect": expect}
    problems = []
    try:
        want, want_err = None, None
        if expect in ("torch", "float64"):
            cast = (lambda x: x.double()) if expect == "float64" else (lambda x: x.clone())  # noqa: E731
            try:
                want = [t.detach().cpu().float() if expect == "float64" else t.detach().cpu()
                        for t in as_tensors(fn(*[cast(x) for x in ins]))]
            except Exception as e:  # noqa: BLE001
                want_err = type(e).__name__
        routed0 = len(CPU_ROUTE)
        with watch() as rec:
            try:
                got = [t.detach().cpu() for t in as_tensors(fn(*[x.to(HPU) for x in ins]))]
                sync()
                got_err = None
            except Exception as e:  # noqa: BLE001
                got, got_err = None, f"{type(e).__name__}: {str(e)[:160]}"
                with contextlib.suppress(Exception):
                    sync()
        row.update(fwd_fallback=rec["fallback"], raised=got_err, want_raised=want_err,
                   cpu_route=len(CPU_ROUTE) - routed0)
        if expect == "raise":
            problems += ["did not raise"] * (got_err is None)
        elif want_err:
            problems += [f"torch raised {want_err}, the card did not"] * (got_err is None)
        elif got_err:
            problems.append(f"raised {got_err}")
        else:
            if len(got) != len(want):
                problems.append(f"{len(got)} outputs, torch {len(want)}")
            errs = []
            for g, w in zip(got, want):
                g, w = g.double(), w.double()
                if g.shape != w.shape:
                    problems.append(f"shape {tuple(g.shape)} vs {tuple(w.shape)}")
                    continue
                if not torch.equal(torch.isnan(g), torch.isnan(w)) or not torch.equal(torch.isinf(g), torch.isinf(w)):
                    problems.append("NaN/Inf pattern differs from torch's")
                errs.append(rel_err(g, w))
            err = max(errs, default=0.0)
            row["max_rel_err"] = err
            problems += [f"error {err:.3g} > 1e-6"] * (not err <= 1e-6)
        problems += ["fell back"] * bool(rec["fallback"]) + cpu_warnings(rec)
        problems += [f"took hpu-ops' CPU route {row['cpu_route']}x"] * bool(row["cpu_route"] and want_err is None
                                                                            and expect != "raise")
    except Exception as e:  # noqa: BLE001  (a failing case is a row, not a crash)
        problems.append(f"exception {e!r}"[:300])
    emit(**row, status="fail" if problems else "ok", problems=problems)
    return not problems


def count_cpu_route():
    for name in ("_on_cpu", "_out_on_cpu"):
        inner = getattr(ops_linalg, name)

        def counted(*args, _inner=inner, _name=name, **kwargs):
            CPU_ROUTE.append(_name)
            return _inner(*args, **kwargs)

        setattr(ops_linalg, name, counted)


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


def run_svd_edges():
    """Edge cases for Issue #9 SVD family: zeros, cond=1e4, rank-1, repeated spectrum, no_grad svdvals, declines."""
    out_results = []

    def check(label, fn, cpu_in, err_fn=err_all, grad_fn=None, expect_fb=False):
        row = {"op": f"svd_edge.{label}", "n": SINGULAR_N, "batch": BATCH, "dtype": "float32", "lazy": LAZY}
        problems = []
        try:
            ref_in = [x.double() for x in cpu_in]
            ref = as_tensors(fn(*ref_in))
            with watch() as rec:
                got = as_tensors(fn(*(x.to(HPU) for x in cpu_in)))
                sync()
            row["fwd_fallback"] = rec["fallback"]
            e, notes = err_fn(got, ref, ref_in)
            e32, _ = err_fn(as_tensors(fn(*cpu_in)), ref, ref_in)
            row["max_rel_err"], row["err_cpu32"] = e, e32
            problems += notes + over_bar("error", e, e32)
            if not expect_fb:
                problems += ["fell back"] * bool(rec["fallback"]) + cpu_warnings(rec)
            if grad_fn is not None:
                xs = [x.to(HPU).detach().requires_grad_() for x in cpu_in]
                loss = grad_fn(as_tensors(fn(*xs)))
                sync()
                with watch() as brec:
                    loss.backward()
                    sync()
                row["bwd_fallback"] = brec["fallback"]
                for x in xs:
                    if not torch.isfinite(x.grad).all():
                        problems.append("non-finite grad")
                if not expect_fb:
                    problems += ["bwd fell back"] * bool(brec["fallback"]) + cpu_warnings(brec)
        except Exception as e:  # noqa: BLE001
            problems.append(f"exception {e!r}"[:300])
        emit(**row, status="fail" if problems else "ok", problems=problems)
        out_results.append(not problems)

    n = SINGULAR_N
    zeros = [torch.zeros(BATCH, n, n)]
    check("zeros_svd", lambda x: L.svd(x, full_matrices=False), zeros, err_svd)
    check("zeros_svdvals", lambda x: L.svdvals(x), zeros, grad_fn=loss_sum)

    q1, _ = L.qr(torch.randn(BATCH, n, n))
    q2, _ = L.qr(torch.randn(BATCH, n, n))
    s_cond = torch.logspace(0, -4, n).expand(BATCH, n)
    cond1e4 = [(q1 * s_cond.unsqueeze(-2)) @ q2.mT]
    check("cond1e4_svd", lambda x: L.svd(x, full_matrices=False), cond1e4, err_svd, grad_fn=loss_svd)

    u1 = torch.randn(BATCH, n, 1)
    v1 = torch.randn(BATCH, 1, n)
    rank1 = [u1 @ v1]
    check("rank1_svd", lambda x: L.svd(x, full_matrices=False), rank1, err_svd)
    check("rank1_pinv", lambda x: L.pinv(x, atol=1e-3, rtol=1e-3), rank1)
    check("rank1_matrix_rank", lambda x: L.matrix_rank(x, atol=1e-3, rtol=1e-3), rank1)

    rep_vals = torch.tensor([3.0, 3.0] + [1.0] * (n - 2))
    rep = [torch.diag_embed(rep_vals.expand(BATCH, n))]
    check("repeated_svdvals", lambda x: L.svdvals(x), rep, grad_fn=loss_sum)

    with torch.no_grad():
        check("svdvals_no_grad", lambda x: L.svdvals(x), [general(BATCH, n, torch.float32)])

    # Declines (expected fallback, must run cleanly without error)
    check("decline_pinv_hermitian", lambda x: L.pinv(x, hermitian=True), [spd(BATCH, n, torch.float32, "cpu")],
          expect_fb=True)
    check("decline_matrix_rank_hermitian", lambda x: L.matrix_rank(x, hermitian=True),
          [spd(BATCH, n, torch.float32, "cpu")], expect_fb=True)
    check("decline_lstsq_rank_deficient", lambda a, b: L.lstsq(a, b),
          [rank1[0], torch.randn(BATCH, n, 3)], err_lstsq, expect_fb=True)

    def svd_out_decline(x):
        u = torch.zeros(BATCH, 4, 4, device=x.device, dtype=x.dtype)
        s = torch.zeros(BATCH, 4, device=x.device, dtype=x.dtype)
        vh = torch.zeros(BATCH, 4, 34, device=x.device, dtype=x.dtype)
        L.svd(x, full_matrices=False, out=(u, s, vh))
        return u, s, vh

    check("decline_svd_out_big_n", svd_out_decline, [torch.randn(BATCH, 4, 34, dtype=torch.float32)], err_svd,
          expect_fb=True)
    return out_results


def run_expm_hh_edges():
    """Edge cases for Issue #13: matrix_exp s>0/zeros/n=1, householder rect/k<n/k=0/singular, and orthogonal layers."""
    out_results = []

    def check(label, fn, cpu_in, grad_fn=loss_sum, expect_fb=False):
        row = {"op": f"expm_hh_edge.{label}", "n": SINGULAR_N, "batch": BATCH, "dtype": "float32", "lazy": LAZY}
        problems = []
        try:
            ref_in = [x.double() for x in cpu_in]
            ref = as_tensors(fn(*ref_in))
            with watch() as rec:
                got = as_tensors(fn(*(x.to(HPU) for x in cpu_in)))
                sync()
            row["fwd_fallback"] = rec["fallback"]
            e, notes = err_all(got, ref, ref_in)
            e32, _ = err_all(as_tensors(fn(*cpu_in)), ref, ref_in)
            row["max_rel_err"], row["err_cpu32"] = e, e32
            problems += notes + over_bar("error", e, e32)
            if not expect_fb:
                problems += ["fell back"] * bool(rec["fallback"]) + cpu_warnings(rec)
            if grad_fn is not None:
                xs = [x.to(HPU).detach().requires_grad_() for x in cpu_in]
                loss = grad_fn(as_tensors(fn(*xs)))
                sync()
                with watch() as brec:
                    loss.backward()
                    sync()
                row["bwd_fallback"] = brec["fallback"]
                ref_grads = reference_grads(
                    Case(label, lambda *_: cpu_in, fn, loss=grad_fn), ref_in)
                cpu_xs = [x.detach().clone().requires_grad_() for x in cpu_in]
                grad_fn(as_tensors(fn(*cpu_xs))).backward()
                ge = grad_error(xs, ref_grads)
                ge32 = grad_error(cpu_xs, ref_grads)
                row["grad_err"], row["grad_err_cpu32"] = ge, ge32
                problems += over_bar("gradient error", ge, ge32)
                if not expect_fb:
                    problems += ["bwd fell back"] * bool(brec["fallback"]) + cpu_warnings(brec)
        except Exception as e:  # noqa: BLE001
            problems.append(f"exception {e!r}"[:300])
        emit(**row, status="fail" if problems else "ok", problems=problems)
        out_results.append(not problems)

    # matrix_exp edge cases
    check("expm_large_s_gt_0", lambda x: L.matrix_exp(x), [5.0 * torch.randn(BATCH, SINGULAR_N, SINGULAR_N)])
    check("expm_zeros", lambda x: L.matrix_exp(x), [torch.zeros(BATCH, SINGULAR_N, SINGULAR_N)])
    check("expm_n1", lambda x: L.matrix_exp(x), [torch.randn(BATCH, 1, 1)])
    check("expm_torch_alias", lambda x: torch.matrix_exp(x), make_skew(BATCH, SINGULAR_N, torch.float32))

    # householder_product edge cases (Issue #13 exact shapes)
    a22 = torch.randn(BATCH, 22, 22).tril(-1)
    t21 = 2.0 / (1.0 + (a22[..., :21] ** 2).sum(-2))
    check("hh_22_22_k21", lambda a, t: L.householder_product(a, t), [a22, t21])

    a10 = torch.randn(BATCH, 22, 10).tril(-1)
    t10 = 2.0 / (1.0 + (a10 ** 2).sum(-2))
    t6 = 2.0 / (1.0 + (a10[..., :6] ** 2).sum(-2))
    check("hh_22_10_k10", lambda a, t: L.householder_product(a, t), [a10, t10])
    check("hh_22_10_k6", lambda a, t: L.householder_product(a, t), [a10, t6])
    check("hh_22_10_k0", lambda a, t: L.householder_product(a, t), [a10, torch.empty(BATCH, 0)], grad_fn=None)

    a64_32 = torch.randn(BATCH, 64, 32).tril(-1)
    t64_32 = 2.0 / (1.0 + (a64_32 ** 2).sum(-2))
    check("hh_64_32_k32", lambda a, t: L.householder_product(a, t), [a64_32, t64_32])

    # orthogonal parametrizations (matrix_exp & householder) on (22, 22) & (22, 10)
    from torch.nn.utils.parametrizations import orthogonal

    for omap in ("matrix_exp", "householder"):
        for rows, cols in ((22, 22), (22, 10)):
            row = {"op": f"orthogonal.{omap}_{rows}x{cols}", "n": rows, "batch": BATCH,
                   "dtype": "float32", "lazy": LAZY}
            problems = []
            try:
                torch.manual_seed(0)
                layer_ref = orthogonal(torch.nn.Linear(cols, rows, bias=False).double(),
                                       orthogonal_map=omap, use_trivialization=False)
                layer_cpu32 = orthogonal(torch.nn.Linear(cols, rows, bias=False),
                                         orthogonal_map=omap, use_trivialization=False)
                layer_hpu = orthogonal(torch.nn.Linear(cols, rows, bias=False),
                                         orthogonal_map=omap, use_trivialization=False)
                with torch.no_grad():
                    w0 = torch.randn(rows, cols)
                    layer_ref.parametrizations.weight.original.copy_(w0.double())
                    layer_cpu32.parametrizations.weight.original.copy_(w0)
                    layer_hpu.parametrizations.weight.original.copy_(w0)
                layer_hpu = layer_hpu.to(HPU)
                x = torch.randn(BATCH, cols)
                xh = x.to(HPU)
                sync()
                layer_ref(x.double()).sum().backward()
                layer_cpu32(x).sum().backward()
                with watch() as rec:
                    out_h = layer_hpu(xh)
                    out_h.sum().backward()
                    sync()
                row["fwd_fallback"] = rec["fallback"]
                row["max_rel_err"] = rel_err(out_h, layer_ref(x.double()))
                row["err_cpu32"] = rel_err(layer_cpu32(x), layer_ref(x.double()))
                row["grad_err"] = rel_err(layer_hpu.parametrizations.weight.original.grad,
                                          layer_ref.parametrizations.weight.original.grad)
                row["grad_err_cpu32"] = rel_err(layer_cpu32.parametrizations.weight.original.grad,
                                                layer_ref.parametrizations.weight.original.grad)
                problems += over_bar("error", row["max_rel_err"], row["err_cpu32"])
                problems += over_bar("gradient error", row["grad_err"], row["grad_err_cpu32"])
                problems += ["fell back"] * bool(rec["fallback"]) + cpu_warnings(rec)
            except Exception as e:  # noqa: BLE001
                problems.append(f"exception {e!r}"[:300])
            emit(**row, status="fail" if problems else "ok", problems=problems)
            out_results.append(not problems)
    return out_results


def _lazy_check_row(name, n, batch, check_fn):
    row = {"op": name, "n": n, "batch": batch, "dtype": "float32", "lazy": LAZY, "baseline": BASELINE}
    problems = []
    try:
        err, bar, rec = check_fn()
        row.update(max_rel_err=err, err_cpu32=bar, fwd_fallback=rec["fallback"])
        problems += over_bar("error", err, bar)
        problems += ["fell back"] * bool(rec["fallback"]) + cpu_warnings(rec)
    except Exception as e:  # noqa: BLE001
        problems.append(f"exception {e!r}"[:300])
    emit(**row, status="fail" if problems else "ok", problems=problems)
    return not problems


def run_lazy_single_call_checks(tpc):
    """Single-call immediate .cpu() and 4x-repeated synced .cpu() checks on 3D (len(batch)==1, m<=64, left=True)
    paths with zero trailing ops, B-only grad backward, and direct tpc.tri_solve / tpc.gj_solve / tpc.gj_inv."""
    gen = torch.Generator().manual_seed(123)
    out = []

    def randn(*shape):
        return torch.randn(*shape, generator=gen, dtype=torch.float32)

    def make_tri(b, n, m, upper, unit):
        A = randn(b, n, n) / max(n, 1)
        A = (A.triu(1) if upper else A.tril(-1)) if unit else ((A.triu() if upper else A.tril()) + 2.0 * torch.eye(n))
        return A, randn(b, n, m)

    # 1. Public solve_triangular: immediate single-call .cpu() (no sync) + 4x repeated sync(); .cpu()
    for b, n, m, upper, unit in ((1, 1, 1, False, False), (4, 22, 5, False, False),
                                 (32, 16, 3, True, False), (4, 64, 64, False, True)):
        A, B = make_tri(b, n, m, upper, unit)
        fn = lambda a, b_: L.solve_triangular(a, b_, upper=upper, left=True, unitriangular=unit)  # noqa: E731
        ref = fn(A.double(), B.double())
        bar = rel_err(fn(A, B), ref)

        def check_tri(A=A, B=B, fn=fn, ref=ref, bar=bar):
            Ah, Bh = A.to(HPU), B.to(HPU)
            sync()
            errs = []
            with watch() as rec:
                errs.append(rel_err(fn(Ah, Bh).cpu(), ref))  # immediate .cpu(), no sync/trailing op
                for _ in range(4):
                    x = fn(Ah, Bh)
                    sync()
                    errs.append(rel_err(x.cpu(), ref))
            return max(errs), bar, rec

        out.append(_lazy_check_row(f"lazy_single.solve_triangular.b{b}_m{m}_u{int(upper)}_unit{int(unit)}",
                                   n, b, check_tri))

    # 2. Public cholesky_solve: immediate single-call .cpu() + 4x repeated sync(); .cpu()
    for b, n, m, upper in ((1, 1, 1, False), (4, 22, 5, False), (32, 16, 3, True), (4, 64, 64, False)):
        g = randn(b, n, n)
        C = L.cholesky(g @ g.mT + n * torch.eye(n), upper=upper)
        B = randn(b, n, m)
        fn = lambda b_, c_: torch.cholesky_solve(b_, c_, upper=upper)  # noqa: E731
        ref = fn(B.double(), C.double())
        bar = rel_err(fn(B, C), ref)

        def check_chol(B=B, C=C, fn=fn, ref=ref, bar=bar):
            Bh, Ch = B.to(HPU), C.to(HPU)
            sync()
            errs = []
            with watch() as rec:
                errs.append(rel_err(fn(Bh, Ch).cpu(), ref))
                for _ in range(4):
                    x = fn(Bh, Ch)
                    sync()
                    errs.append(rel_err(x.cpu(), ref))
            return max(errs), bar, rec

        out.append(_lazy_check_row(f"lazy_single.cholesky_solve.b{b}_m{m}_u{int(upper)}", n, b, check_chol))

    # 3. SolveTriangular B-only grad (A.requires_grad=False, B.requires_grad=True)
    for b, n, m, upper in ((4, 22, 5, False), (32, 16, 3, True)):
        A, B = make_tri(b, n, m, upper, False)
        W = randn(b, n, m)
        ref_b = B.double().requires_grad_()
        (L.solve_triangular(A.double(), ref_b, upper=upper) * W.double()).sum().backward()
        cpu_b = B.clone().requires_grad_()
        (L.solve_triangular(A, cpu_b, upper=upper) * W).sum().backward()
        bar = rel_err(cpu_b.grad, ref_b.grad)

        def check_b_grad(A=A, B=B, W=W, upper=upper, ref_g=ref_b.grad, bar=bar):
            Ah, Wh = A.to(HPU), W.to(HPU)
            sync()
            errs = []
            with watch() as rec:
                for i in range(4):
                    Bh = B.to(HPU).detach().requires_grad_()
                    loss = (L.solve_triangular(Ah, Bh, upper=upper) * Wh).sum()
                    sync()
                    loss.backward()
                    if i > 0:
                        sync()
                    errs.append(rel_err(Bh.grad.cpu(), ref_g))
            return max(errs), bar, rec

        out.append(_lazy_check_row(f"lazy_single.solve_triangular.b_only_grad.u{int(upper)}", n, b, check_b_grad))

    # 4. Direct tpc.tri_solve, tpc.gj_solve, tpc.gj_inv single-call + 4x repeated .cpu()
    A_t, B_t = make_tri(4, 22, 5, False, False)
    ref_t = L.solve_triangular(A_t.double(), B_t.double(), upper=False)
    bar_t = rel_err(L.solve_triangular(A_t, B_t, upper=False), ref_t)

    def check_raw_tri():
        Ah, Bh = A_t.to(HPU), B_t.to(HPU)
        sync()
        errs = []
        with watch() as rec:
            errs.append(rel_err(tpc.tri_solve(Ah, Bh, upper=False, left=True, unitriangular=False).cpu(), ref_t))
            for _ in range(4):
                x = tpc.tri_solve(Ah, Bh, upper=False, left=True, unitriangular=False)
                sync()
                errs.append(rel_err(x.cpu(), ref_t))
        return max(errs), bar_t, rec

    out.append(_lazy_check_row("lazy_single.tpc_tri_solve", 22, 4, check_raw_tri))

    g_gj = randn(4, 22, 22)
    A_gj = g_gj @ g_gj.mT + 22 * torch.eye(22)
    B_gj = randn(4, 22, 5)
    ref_gj = L.solve(A_gj.double(), B_gj.double())
    bar_gj = rel_err(L.solve(A_gj, B_gj), ref_gj)

    def check_raw_gj():
        Ah, Bh = A_gj.to(HPU), B_gj.to(HPU)
        sync()
        errs = []
        with watch() as rec:
            x0, d0, s0 = tpc.gj_solve(Ah, Bh)
            errs.append(rel_err(x0.cpu(), ref_gj))
            assert not bool((s0.cpu()[..., 1] != 0).any()) and torch.isfinite(d0.cpu()).all()
            for _ in range(4):
                x, d, s = tpc.gj_solve(Ah, Bh)
                sync()
                errs.append(rel_err(x.cpu(), ref_gj))
                assert not bool((s.cpu()[..., 1] != 0).any()) and torch.isfinite(d.cpu()).all()
        return max(errs), bar_gj, rec

    out.append(_lazy_check_row("lazy_single.tpc_gj_solve", 22, 4, check_raw_gj))

    ref_inv = L.inv(A_gj.double())
    bar_inv = rel_err(L.inv(A_gj), ref_inv)

    def check_raw_inv():
        Ah = A_gj.to(HPU)
        sync()
        errs = []
        with watch() as rec:
            x0, d0, s0 = tpc.gj_inv(Ah)
            errs.append(rel_err(x0.cpu(), ref_inv))
            assert not bool((s0.cpu()[..., 1] != 0).any()) and torch.isfinite(d0.cpu()).all()
            for _ in range(4):
                x, d, s = tpc.gj_inv(Ah)
                sync()
                errs.append(rel_err(x.cpu(), ref_inv))
        return max(errs), bar_inv, rec

    out.append(_lazy_check_row("lazy_single.tpc_gj_inv", 22, 4, check_raw_inv))
    return out


def main():
    global BASELINE, ops_linalg, ops_tpc, tpc_maxn, eigh_maxn, small_maxn
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--baseline", action="store_true", help="skip hpu_ops.enable()")
    parser.add_argument("--with-eigh", action="store_true",
                        help="no-op: eigh is installed by hpu_ops.enable()")
    parser.add_argument("--only", action="append", default=[], help="only run cases whose name contains this substring")
    args = parser.parse_args()
    BASELINE = args.baseline
    import hpu_ops
    from hpu_ops import linalg as ops_linalg
    from hpu_ops import tpc
    ops_tpc = tpc
    tpc_maxn, eigh_maxn, small_maxn = tpc.MAXN, tpc.EIGH_MAXN, tpc.SMALL_MAXN
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
          BASELINE, "tpc_ready", None if BASELINE else tpc.ready(), "lazy", LAZY, "bridge 1.21.4", flush=True)
    results = [probe_argmax_ties(), probe_argsort_int64()] if not args.only else []
    if not BASELINE and not args.only:
        results.append(check_tpc(tpc))
    cases = [c for c in CASES if not args.only or any(s in c.name for s in args.only)]
    for case in cases:
        results += [run_case(case, n) for n in case.sizes or SIZES]
        if case.gate:
            results += [run_case(case, n, timed=False) for n in GATE_SIZES]
        if case.float64:
            results.append(run_float64(case))
    if not BASELINE and not args.only:
        results.append(run_double_backward())
        results += [run_eigh_adversarial(kind, n, seed) for kind in ("deg_rotated", "illcond_1e6", "scaled_1e30")
                    for n in (16, 22) for seed in range(3)]
        for op, fn in (("solve", L.solve), ("inv", L.inv), ("det", L.det), ("slogdet", L.slogdet)):
            results.append(run_singular(op, fn))
        results += run_lazy_single_call_checks(tpc)
        for op, fn in (("solve_triangular", lambda T, B: L.solve_triangular(T, B, upper=False)),
                       ("cholesky_solve", lambda T, B: torch.cholesky_solve(B, T))):
            results.append(run_singular_triangle(op, fn, frozenset()))
        results += run_svd_edges()
        results += run_expm_hh_edges()
        count_cpu_route()  # from here on, calls of hpu-ops' CPU route are counted (the Codex rows check them)
        results += [run_codex_case(*case) for case in codex_cases()]
    if BASELINE:  # positive control: without hpu_ops eigh falls back, else the metric key is wrong
        ok = bool(EIGH_FALLBACKS)
        emit(op="metric_check", lazy=LAZY, baseline=True, status="ok" if ok else "error",
             problems=[] if ok else ["no cpu_fallback delta for eigh under --baseline: the metric key is wrong"])
        results.append(ok)
    print(f"SUMMARY {len(results)} cases, {results.count(False)} failed", flush=True)
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
