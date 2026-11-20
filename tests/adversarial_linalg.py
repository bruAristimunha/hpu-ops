"""Adversarial fuzz of hpu-ops' linalg rewrites (PR #23, the TPC path): every routed op against torch on the CPU.

    ../.venv-t26/bin/python tests/adversarial_linalg.py                  # CPU stands in for HPU (reference hooks)
    HPU_OPS_TPC_EIGH=1 PT_HPU_LAZY_MODE=0|1 python3 tests/adversarial_linalg.py --hpu   # on a Gaudi2 card

Routed ops: linalg.solve / solve_ex, inv / inv_ex, det, slogdet (and logdet), solve_triangular, cholesky_solve,
eigh, eigvalsh. CPU mode installs the rewrites with "cpu" in the role of "hpu" (``cpu_as_hpu``) and sets the
``linalg._GJ`` / ``eigh._EIGH`` hooks to the kernels' reference contracts, so n <= 32 goes through the kernel wiring
and n = 33 through the composites. ``--hpu`` calls ``hpu_ops.enable()`` before the first HPU op and runs on "hpu".

Per case, three runs: torch on the CPU in float64 (the accuracy reference, on the float32-rounded inputs), torch on
the CPU in the case's dtype (the behaviour reference: errors, non-finite values, shapes, dtypes, and the error bar),
and the rewrite. A case FAILs when
  - wrong-result: an output or gradient is further from float64 than max(FLOOR, 10 x torch CPU float32's error)
    (the gate's bar), its finiteness differs from torch's, its shape or dtype differs, or torch raises and the
    rewrite returns finite values;
  - crash: the rewrite raises where torch does not;
  - fallback: a call the README says runs through the rewrite did not (no count), or, on the card, the Habana
    ``cpu_fallback`` metric moved;
  - message: both raise with different messages, or torch raises and the rewrite returns non-finite values.
Each case prints ``PASS|FAIL|SKIP <id> seed=<seed> ...``; ``--only REGEX`` reruns a subset. The summary groups
failures by root-cause hypothesis.
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import math
import os
import re
import sys
import traceback
import types
import warnings
import zlib

ARGS = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ARGS.add_argument("--hpu", action="store_true", help="run on a Gaudi card (hpu_ops.enable(), device hpu)")
ARGS.add_argument("--seed", type=int, default=0, help="base seed; each case derives its own from its id")
ARGS.add_argument("--only", default=None, help="regex on case ids")
ARGS.add_argument("--quick", action="store_true", help="fewer shape sweeps")
ARGS.add_argument("--verbose", action="store_true", help="print tracebacks of unexpected errors")
OPTS = ARGS.parse_args()

if OPTS.hpu:
    os.environ.setdefault("HPU_OPS_TPC_EIGH", "1")  # asked for by the controller; hpu_ops does not read it
    import habana_frameworks.torch.core as htcore  # noqa: E402

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import hpu_ops  # noqa: E402
from hpu_ops import eigh as jacobi  # noqa: E402
from hpu_ops import linalg, tpc  # noqa: E402

if OPTS.hpu:
    hpu_ops.enable()  # before the first HPU op: the kernel path is read when the device initialises
    DEV = torch.device("hpu")
else:
    from test_cpu import cpu_as_hpu  # noqa: E402
    DEV = torch.device("cpu")

warnings.simplefilter("ignore")
FLOOR = 1e-6
FACTOR = 10
MAXN = tpc.MAXN
LAZY = os.environ.get("PT_HPU_LAZY_MODE", "1") != "0"

CAUSES = {
    "uplo-case": "eigh.py:100,102 test UPLO == 'U' case-sensitively; torch accepts 'u' and 'l' (it upper-cases), "
                 "so UPLO='u' reads the lower triangle",
    "chol-normal-equations": "linalg.py:274-277 CholeskySolve.compute forms L @ L^H and solves it by Gauss-Jordan: "
                             "squares cond(L) (torch does two triangular solves) and overflows/underflows once "
                             "|L|^2 leaves float32",
    "gj-on-triangle": "linalg.py:230-232 SolveTriangular.compute solves the triangle by pivoted elimination "
                      "(Gauss-Jordan / LU) instead of substitution: forward error ~cond(T)*eps, torch's trsm is "
                      "componentwise accurate on ill-conditioned triangles",
    "out-not-routed": "linalg.py:258,304,306,427,547 and eigh.py:132 register only the .default overloads: out= "
                      "calls (_linalg_solve_ex.out, linalg_inv_ex.inverse, _linalg_det.result, ...) keep the "
                      "native kernel (on the card: Habana's CPU fallback)",
    "shape-check": "linalg.py:52-72 _takes checks only the matrix argument; a bad B reaches _gj (linalg.py:100-102)"
                   " / tpc.gj_solve (tpc/__init__.py:364-367) and fails there with another error",
    "mixed-dtype": "linalg.py:66-67 _takes accepts float32 and float64 together off hpu; the composite then "
                   "matmuls mixed dtypes (torch promotes in solve_triangular, raises its own error elsewhere)",
    "reference-hook": "tpc/__init__.py:411-421 gj_solve_reference keeps inf/NaN from a zero pivot (the kernel "
                      "zeroes the row): CPU stand-in only",
    "singular-grad": "linalg.py:508-511 Det.backward multiplies det by the GJ inverse; at det == 0 torch returns 0",
    "nan-triangle": "linalg.py:139-142 _nan_if_singular sets the whole matrix to NaN where torch keeps finite "
                    "entries (and +-inf)",
    "enable-undo": "__init__.py:93-103 enable() returns early while _STATE['undo'] is set; the undo never clears "
                   "it, so enable() after the undo is a no-op",
    "eigh-error": "eigh.py:98-103 / tpc/__init__.py:383-399: no info from the Jacobi kernel, so a NaN input "
                  "returns NaN where torch raises 'failed to converge'",
    "float64-backward": "linalg.py:171 / 61-65: float64 on hpu keeps torch's derivative, whose backward "
                        "(linalg_lu_solve) has no HPU kernel: Habana fallback, at float32 storage precision",
    "past-maxn-backward": "linalg.py:83: n > MAXN keeps torch's derivative; its backward (linalg_lu_solve) has no "
                          "HPU kernel: Habana fallback, not hpu-ops' CPU route",
    "slogdet-sign": "linalg.py:533 Slogdet.forward marks sign non-differentiable; torch's real sign requires grad "
                    "(zero gradient), so sign.backward() raises 'does not require grad'",
    "bcast-info": "linalg.py:120-128 _gj_info slices S's broadcast batch down to A's batch: a non-contiguous info, "
                  "and aten._linalg_check_errors (linalg.py:394) asserts infos.is_contiguous()",
    "eigh-uplo-check": "eigh.py:100-102 UPLO is never validated: 'X' reads the lower triangle where torch raises",
    "autocast-backward": "linalg.py:244-253,413-421: the Functions' backward runs under the forward's autocast "
                         "state; its matmuls run in bf16/fp16 (CPU stand-in: the reference's lu_solve then mixes "
                         "dtypes and raises)",
    "info-rounding": "gj_solve_f32_gaudi2.c:77 / tpc/__init__.py:411-421: info is set only for an exactly-zero "
                     "pivot of the kernel's own elimination; rounding differs from LAPACK, so solve/inv raise where "
                     "torch returns, or return where torch raises (documented as an index difference only)",
    "nonfinite-pivot": "tpc/__init__.py:411-421 / linalg.py:310-328: an inf pivot or a subnormal scale makes the "
                       "elimination's non-finite pattern differ from LAPACK's; the TPC kernel (reciprocal of inf = 0,"
                       " flush-to-zero) decides it on the card",
    "known-eigh-grad":"known: the eigh kernel degrades gradients (being fixed elsewhere)",
}


# ------------------------------------------------------------------------------------------------ infrastructure


def seed_of(cid):
    return (zlib.crc32(cid.encode()) + OPTS.seed) % (2 ** 31)


NOT_INSTALLED = {"solve_triangular", "cholesky_solve"}


def install_linalg(device_type):
    from hpu_ops._dispatch import chain
    # solve_triangular / cholesky_solve are not in hpu_ops.FIXES (accuracy): their calls stay native; eigh is opt-in
    # in enable() but fuzzed here
    return chain([fix(device_type) for fix in (linalg.install_solve, linalg.install_inv_det, jacobi.install_eigh)])


def eigh_hook(A, upper):
    """The kernel's contract through numpy; under torch.compile's fake tensors, outputs of the right shape (numpy
    cannot trace: a CPU stand-in limit, the card's op has a Meta kernel)."""
    if type(A) is not torch.Tensor:  # FakeTensor / FunctionalTensor while tracing
        return A.new_empty(A.shape[:-1]), A.new_empty(A.shape)
    return tpc.eigh_reference(A.detach(), upper)


@contextlib.contextmanager
def rewrites_on():
    """CPU: the rewrites installed for "cpu" with the reference hooks; HPU: already on (enable())."""
    if OPTS.hpu:
        hpu_ops._STATE["calls"].clear()
        yield hpu_ops._STATE["calls"]
        return
    saved = linalg._GJ, jacobi._EIGH
    linalg._GJ = lambda A, B: tpc.gj_solve_reference(A.detach(), B.detach())
    jacobi._EIGH = eigh_hook
    try:
        with cpu_as_hpu(install_linalg) as calls:
            yield calls
    finally:
        linalg._GJ, jacobi._EIGH = saved


def fallback_ops():
    if not OPTS.hpu:
        return {}
    try:
        from habana_frameworks.torch.hpu.metrics import metric_global
        return dict(dict(metric_global("cpu_fallback").stats()).get("FallbackOps", {}) or {})
    except Exception:  # noqa: BLE001
        return {}


def sync():
    if OPTS.hpu:
        if LAZY:
            htcore.mark_step()
        torch.hpu.synchronize()


class Run(types.SimpleNamespace):
    """tensors: outputs then gradients (CPU copies); error: the exception, if any."""


def leaves_of(x):
    if isinstance(x, torch.Tensor):
        return [x]
    if isinstance(x, (tuple, list)) or hasattr(x, "_fields"):
        return [t for item in x for t in leaves_of(item)]
    return []


def execute(case, dtype, device, ctx_device):
    """One run of ``case``: inputs cast to ``dtype`` (per-input overrides in case.dtypes) on ``device``."""
    try:
        inputs = []
        for i, x in enumerate(case.inputs):
            dt = case.dtypes[i] if case.dtypes and dtype != torch.float64 else dtype
            if case.dtypes and dtype == torch.float64 and case.keep_mixed:
                dt = case.dtypes[i]
            t = x.to(dt).to(device)
            if case.grad and case.grad[i] and t.is_floating_point():
                t.requires_grad_()
            inputs.append(t)
        with case.ctx(ctx_device):
            out = case.fn(*inputs)
        outs = [o for o in leaves_of(out)]
        grads = []
        # one weight per floating output, whether or not it requires grad, so torch and the rewrite get the same
        # weights when they disagree on which outputs are differentiable (slogdet's sign: torch True, rewrite False)
        gen = torch.Generator().manual_seed(case.seed + 1)
        weighted = [(o, torch.randn(o.shape, generator=gen, dtype=torch.float64).to(o.dtype).to(o.device))
                    for o in outs if o.is_floating_point()]
        weighted = [(o, w) for o, w in weighted if o.requires_grad]
        if case.grad and weighted:
            loss = sum((torch.where(torch.isfinite(o), o, 0) * w).sum() for o, w in weighted)
            wrt = [t for t in inputs if t.requires_grad]
            g = torch.autograd.grad(loss, wrt, create_graph=case.double, allow_unused=True)
            g = [torch.zeros_like(t) if gi is None else gi for t, gi in zip(wrt, g)]
            grads.extend(g)
            if case.double:
                v = [torch.randn(t.shape, generator=gen, dtype=torch.float64).to(t.dtype).to(t.device) for t in wrt]
                second = sum((gi * vi).sum() for gi, vi in zip(g, v) if gi.requires_grad)
                if isinstance(second, torch.Tensor):
                    h = torch.autograd.grad(second, wrt, allow_unused=True)
                    grads.extend(torch.zeros_like(t) if hi is None else hi for t, hi in zip(wrt, h))
        sync()
        return Run(tensors=[t.detach().cpu() for t in outs + grads], error=None, n_out=len(outs))
    except Exception as error:  # noqa: BLE001
        if OPTS.verbose:
            traceback.print_exc()
        return Run(tensors=None, error=error, n_out=0)


def rel_err(x, ref):
    x, ref = x.double(), ref.double()
    ok = torch.isfinite(ref) & torch.isfinite(x)
    if not ok.any():
        return 0.0
    scale = ref[torch.isfinite(ref)].abs().max().item() if torch.isfinite(ref).any() else 1.0
    return (x[ok] - ref[ok]).abs().max().item() / (scale or 1.0)


def short(error):
    return f"{type(error).__name__}: {str(error).splitlines()[0][:110]}" if error else ""


def same_message(a, b):
    sa, sb = str(a).splitlines()[0], str(b).splitlines()[0]
    head = lambda s: re.sub(r"\d+", "#", s)[:40]  # noqa: E731
    return type(a) is type(b) or head(sa) == head(sb)


def judge(case, ref64, ref32, ours, calls, fb):
    """(kind, detail) of the first problem, or (None, detail)."""
    if fb:
        return "fallback", f"Habana cpu_fallback moved: {fb}"
    if ref32.error is not None:
        if ours.error is None:
            finite = all(torch.isfinite(t).all() for t in ours.tensors if t.is_floating_point())
            return ("wrong-result" if finite else "message"), f"torch raised [{short(ref32.error)}], rewrite returned"
        if not same_message(ref32.error, ours.error):
            return "message", f"torch [{short(ref32.error)}] vs rewrite [{short(ours.error)}]"
        return None, "both raise"
    if ours.error is not None:
        return "crash", short(ours.error)
    # solve_triangular / cholesky_solve are not installed by enable(): their calls are never routed
    if case.route is not None and not set(case.route_names) & NOT_INSTALLED:
        routed = sum(calls.get(name, 0) for name in case.route_names) > 0
        if case.route and not routed:
            return "fallback", f"not routed (calls {dict(calls)})"
    if len(ours.tensors) != len(ref32.tensors):
        return "wrong-result", f"{len(ours.tensors)} tensors vs torch's {len(ref32.tensors)}"
    worst = (0.0, 0.0, -1)
    for k, (o, r32) in enumerate(zip(ours.tensors, ref32.tensors)):
        if o.shape != r32.shape or o.dtype != r32.dtype:
            return "wrong-result", f"tensor {k}: {tuple(o.shape)} {o.dtype} vs torch {tuple(r32.shape)} {r32.dtype}"
        if not o.is_floating_point():
            if not torch.equal(o != 0, r32 != 0):
                return "wrong-result", f"tensor {k} (int): nonzero pattern {o.flatten()[:6].tolist()} vs torch " \
                                       f"{r32.flatten()[:6].tolist()}"
            continue
        if case.finiteness and not torch.equal(torch.isfinite(o), torch.isfinite(r32)):
            fo, fr = torch.isfinite(o).all().item(), torch.isfinite(r32).all().item()
            if fo != fr or case.finiteness == "elementwise":
                return "wrong-result", f"tensor {k}: finite={fo} where torch finite={fr} " \
                                       f"(ours {o.flatten()[:4].tolist()} torch {r32.flatten()[:4].tolist()})"
        ref = ref32 if ref64.error is not None or ref64.tensors is None else ref64
        r = ref.tensors[k]
        e_ours, e_32 = rel_err(o, r), rel_err(r32, r)
        bar = max(case.floor, FACTOR * e_32)
        if e_ours > bar and (worst[2] < 0 or e_ours / bar > worst[0] / worst[1]):
            worst = (e_ours, bar, k)
    if worst[2] >= 0:
        k = worst[2]
        what = "output" if k < ours.n_out else "grad"
        return "wrong-result", f"{what} {k}: rel err {worst[0]:.2e} > bar {worst[1]:.2e}"
    return None, ""


RULES = [  # (case id regex, failure detail regex, cause) for cases added without one
    (r"^bcast/.*/solve$|^bcast/.*solve_vec_batched_A|^bcast/.*rhs_wide", r"infos.is_contiguous", "bcast-info"),
    (r"eigh_bad_uplo", r"", "eigh-uplo-check"),
    (r"^autocast/", r"expected scalar type|grad", "autocast-backward"),
    (r"^singular/.*(solve|inv|arange)", r"nonzero pattern|raised|Singular|singular", "info-rounding"),
    (r"^nonfinite/|^scale/", r"finite=", "nonfinite-pivot"),
]


def classify(case, kind, detail):
    """The root-cause key of a failure: the case's own, else the first matching rule, else family:op."""
    if case.cause:
        return case.cause
    for id_re, detail_re, cause in RULES:
        if re.search(id_re, case.id) and re.search(detail_re, detail or ""):
            return cause
    return f"{case.family}:{case.op}"


SEVERITY = ["wrong-result", "crash", "fallback", "message"]
RESULTS = []


def run_case(case):
    if OPTS.only and not re.search(OPTS.only, case.id):
        return
    if case.skip:
        print(f"SKIP {case.id} seed={case.seed} ({case.skip})", flush=True)
        RESULTS.append((case, "skip", case.skip))
        return
    if case.custom is not None:
        kind, detail = case.custom()
    else:
        ref64 = execute(case, torch.float64 if not case.keep_mixed else case.dtype, "cpu", "cpu")
        ref32 = execute(case, case.dtype, "cpu", "cpu")
        before = fallback_ops()
        with rewrites_on() as calls:
            ours = execute(case, case.dtype, DEV, DEV.type)
            calls = dict(calls)
        after = fallback_ops()
        fb = {k: after[k] - before.get(k, 0) for k in after if after[k] - before.get(k, 0)}
        kind, detail = judge(case, ref64, ref32, ours, calls, fb)
    if kind is not None and case.known and OPTS.hpu:
        detail = f"[{case.known}] " + detail
    status = "PASS" if kind is None else "FAIL"
    cause = classify(case, kind, detail)
    tag = "" if kind is None else f" [{kind}] cause={cause}"
    print(f"{status} {case.id} seed={case.seed}{tag} {detail}".rstrip(), flush=True)
    RESULTS.append((case, kind, detail))


def Case(cid, family, op, fn, inputs, **kw):
    seed = seed_of(cid)
    defaults = dict(id=cid, family=family, op=op, fn=fn, inputs=inputs, seed=seed, dtype=torch.float32, dtypes=None,
                    keep_mixed=False, grad=None, double=False, ctx=lambda dev: contextlib.nullcontext(), route=True,
                    route_names=(op,), finiteness="all", floor=FLOOR, cause=None, skip=None, custom=None, known=None)
    defaults.update(kw)
    return types.SimpleNamespace(**defaults)


# ------------------------------------------------------------------------------------------------ inputs


def gen(cid):
    return torch.Generator().manual_seed(seed_of(cid))


def f32r(x):
    """float64 values exactly representable in float32 (the reference sees what the card sees)."""
    return x.float().double()


def randn(g, *shape):
    return torch.randn(*shape, generator=g, dtype=torch.float64)


def general(g, batch, n):
    return f32r(randn(g, *batch, n, n) + math.sqrt(n) * torch.eye(n, dtype=torch.float64) * 0.5)


def spd(g, batch, n, cond=None):
    if cond is None:
        G = randn(g, *batch, n, n)
        return f32r(G @ G.mT / n + torch.eye(n, dtype=torch.float64))
    Q, _ = torch.linalg.qr(randn(g, *batch, n, n))
    d = torch.logspace(0, -math.log10(cond), n, dtype=torch.float64)
    return f32r((Q * d) @ Q.mT)


def chol(g, batch, n, cond=None):
    return f32r(torch.linalg.cholesky(spd(g, batch, n, cond)))


def sym(g, batch, n):
    G = randn(g, *batch, n, n)
    return f32r((G + G.mT) / 2)


def route_expected(n, numel=1, dtype=torch.float32):
    """Whether the README says the call runs through the rewrite: on the card float32 n <= 32 (else hpu-ops' own CPU
    route, no claim); on the CPU stand-in every float32/float64 static non-empty call."""
    if numel == 0:
        return None
    if OPTS.hpu:
        return True if (dtype == torch.float32 and n <= MAXN) else None
    return True if dtype in (torch.float32, torch.float64) else None


# ------------------------------------------------------------------------------------------------ ops


def eigh_fn(uplo="L", invariant=False):
    def fn(A):
        w, V = torch.linalg.eigh(A, UPLO=uplo)
        if invariant:
            return w, (V * w.unsqueeze(-2)) @ V.mT
        return w, V * V  # sign-invariant for distinct eigenvalues
    return fn


def op_table(g, batch, n, k=3):
    """name -> (fn, inputs, grad mask, route names)."""
    A, B = general(g, batch, n), f32r(randn(g, *batch, n, k))
    T = chol(g, batch, n)
    L = chol(g, batch, n)
    S = sym(g, batch, n)
    return {
        "solve": (lambda a, b: torch.linalg.solve(a, b), [A, B], [True, True], ("solve",)),
        "solve_vec": (lambda a, b: torch.linalg.solve(a, b), [A, B[..., 0]], [True, True], ("solve",)),
        "solve_right": (lambda a, b: torch.linalg.solve(a, b, left=False), [A, B.mT.contiguous()], [True, True],
                        ("solve",)),
        "solve_ex": (lambda a, b: torch.linalg.solve_ex(a, b), [A, B], [True, True], ("solve",)),
        "inv": (lambda a: torch.linalg.inv(a), [A], [True], ("inv",)),
        "inv_ex": (lambda a: torch.linalg.inv_ex(a), [A], [True], ("inv",)),
        "det": (lambda a: torch.linalg.det(a), [A], [True], ("det",)),
        "slogdet": (lambda a: torch.linalg.slogdet(a), [A], [True], ("slogdet",)),
        "logdet": (lambda a: torch.logdet(a @ a.mT), [A], [True], ("slogdet",)),
        "tri_lower": (lambda t, b: torch.linalg.solve_triangular(t, b, upper=False), [T, B], [True, True],
                      ("solve_triangular",)),
        "tri_upper_right": (lambda t, b: torch.linalg.solve_triangular(t.mT, b, upper=True, left=False),
                            [T, B.mT.contiguous()], [True, True], ("solve_triangular",)),
        "tri_unit": (lambda t, b: torch.linalg.solve_triangular(t, b, upper=False, unitriangular=True), [T, B],
                     [True, True], ("solve_triangular",)),
        "chol_lower": (lambda b, l: torch.cholesky_solve(b, l), [B, L], [True, True], ("cholesky_solve",)),
        "chol_upper": (lambda b, l: torch.cholesky_solve(b, l.mT, upper=True), [B, L], [True, True],
                       ("cholesky_solve",)),
        "eigh": (eigh_fn(), [S], [True], ("eigh",)),
        "eigvalsh": (lambda a: torch.linalg.eigvalsh(a), [S], [True], ("eigh",)),
    }


def add(cases, *args, **kw):
    cases.append(Case(*args, **kw))


def eigh_known(op):
    return "known-eigh-grad" if op.startswith("eig") else None


# ------------------------------------------------------------------------------------------------ families


def family_shapes(cases):
    """Random batch shapes (empty, size-1, broadcast) x n = 1..33, every op, forward and backward."""
    ns = (1, 2, 3, 5, 8, 16, 21, 22, 31, 32, 33) if not OPTS.quick else (1, 3, 22, 32, 33)
    batches = [(), (1,), (0,), (4,), (2, 1), (1, 3), (0, 2)]
    for n in ns:
        for bi, batch in enumerate(batches):
            if bi > 3 and n not in (1, 3, 22, 32, 33):
                continue
            cid0 = f"shapes/n{n}/b{'x'.join(map(str, batch)) or 'none'}"
            table = op_table(gen(cid0), batch, n)
            numel = math.prod(batch) * n * n
            for op, (fn, inputs, grad, names) in table.items():
                add(cases, f"{cid0}/{op}", "shapes", op, fn, inputs, grad=grad, route_names=names,
                    route=route_expected(n, numel), known=eigh_known(op))
    # broadcasting between A and B batches
    for n in (4, 32, 33):
        g = gen(f"bcast{n}")
        A, B = general(g, (3, 1), n), f32r(randn(g, 1, 5, n, 2))
        T, L = chol(g, (3, 1), n), chol(g, (1,), n)
        r = route_expected(n)
        add(cases, f"bcast/n{n}/solve", "bcast", "solve", lambda a, b: torch.linalg.solve(a, b), [A, B],
            grad=[True, True], route=r)
        add(cases, f"bcast/n{n}/solve_ex_info", "bcast", "solve", lambda a, b: torch.linalg.solve_ex(a, b), [A, B],
            grad=[True, True], route=r)
        add(cases, f"bcast/n{n}/solve_2d_A_batched_B", "bcast", "solve", lambda a, b: torch.linalg.solve_ex(a, b),
            [A[0, 0], B], grad=[True, True], route=r)
        add(cases, f"bcast/n{n}/solve_vec_batched_A", "bcast", "solve", lambda a, b: torch.linalg.solve(a, b),
            [A, B[0, 0, :, 0]], grad=[True, True], route=r)
        add(cases, f"bcast/n{n}/tri", "bcast", "solve_triangular",
            lambda t, b: torch.linalg.solve_triangular(t, b, upper=False), [T, B], grad=[True, True], route=r)
        add(cases, f"bcast/n{n}/chol", "bcast", "cholesky_solve", lambda b, l: torch.cholesky_solve(b, l),
            [B[0], L], grad=[True, True], route=r)
        add(cases, f"bcast/n{n}/rhs_wide_m{2 * MAXN + 1}", "bcast", "solve", lambda a, b: torch.linalg.solve(a, b),
            [A[0, 0], f32r(randn(g, n, 2 * MAXN + 1))], grad=[True, True], route=r)


def family_layouts(cases):
    """Non-contiguous, transposed, expanded, sliced with an offset, contiguous with a storage offset, aliased."""

    def offset_view(x):  # contiguous, storage_offset 7
        flat = torch.cat([x.new_zeros(7), x.reshape(-1)])
        return flat[7:].view(x.shape)

    def sliced(x):  # non-contiguous view with an offset
        big = F.pad(x, (1, 2, 3, 0))
        return big[..., 3:3 + x.shape[-2], 1:1 + x.shape[-1]]

    def strided(x):  # every other column of a wider tensor
        return torch.stack([x, x], -1).flatten(-2)[..., ::2]

    layouts = {"T": lambda x: x.mT, "expand": lambda x: x.expand(4, *x.shape[-2:]), "offset": offset_view,
               "sliced": sliced, "strided": strided, "neg_stride_free_flip": lambda x: x.flip(-1).flip(-1)}
    for n in (3, 22, 32):
        for lname, lay in layouts.items():
            table = op_table(gen(f"layout{n}{lname}"), (), n)
            for op, (fn, inputs, grad, names) in table.items():
                def wrapped(*xs, fn=fn, lay=lay):
                    return fn(*[lay(x) if x.dim() >= 2 else x for x in xs])
                if op == "solve_vec" and lname in ("T", "expand"):
                    continue
                add(cases, f"layout/{lname}/n{n}/{op}", "layout", op, wrapped, inputs, grad=grad, route_names=names,
                    route=route_expected(n), known=eigh_known(op))
    # aliasing: the same tensor as both arguments
    for n in (4, 32):
        g = gen(f"alias{n}")
        A, T = general(g, (), n), chol(g, (), n)
        add(cases, f"alias/n{n}/solve_A_AT", "alias", "solve", lambda a: torch.linalg.solve(a, a.mT), [A], grad=[True],
            route=route_expected(n))
        add(cases, f"alias/n{n}/solve_A_ATview", "alias", "solve", lambda a: torch.linalg.solve(a, a.mT[:, :2]), [A],
            grad=[True], route=route_expected(n))
        add(cases, f"alias/n{n}/tri_T_TT", "alias", "solve_triangular",
            lambda t: torch.linalg.solve_triangular(t, t.mT, upper=False), [T], grad=[True], route=route_expected(n))
        add(cases, f"alias/n{n}/chol_L_L", "alias", "cholesky_solve", lambda t: torch.cholesky_solve(t, t), [T],
            grad=[True], route=route_expected(n))
        add(cases, f"alias/n{n}/inv_of_inv", "alias", "inv", lambda a: torch.linalg.inv(torch.linalg.inv(a)), [A],
            grad=[True], route=route_expected(n))
        add(cases, f"alias/n{n}/solve_A_invA", "alias", "solve", lambda a: torch.linalg.solve(a, torch.linalg.inv(a)),
            [A], grad=[True], route=route_expected(n))


def family_out(cases):
    """out= variants and in-place use of outputs."""
    n = 6
    g = gen("out")
    A, B, T, S = general(g, (2,), n), f32r(randn(g, 2, n, 3)), chol(g, (2,), n), sym(g, (2,), n)
    outs = {
        "solve": (lambda a, b: torch.linalg.solve(a, b, out=torch.empty_like(b)), [A, B], ("solve",)),
        "solve_ex": (lambda a, b: torch.linalg.solve_ex(a, b, out=(torch.empty_like(b),
                                                                   torch.empty(2, dtype=torch.int32))), [A, B],
                     ("solve",)),
        "inv": (lambda a: torch.linalg.inv(a, out=torch.empty_like(a)), [A], ("inv",)),
        "inverse": (lambda a: torch.inverse(a, out=torch.empty_like(a)), [A], ("inv",)),
        "det": (lambda a: torch.linalg.det(a, out=a.new_empty(2)), [A], ("det",)),
        "slogdet": (lambda a: torch.linalg.slogdet(a, out=(a.new_empty(2), a.new_empty(2))), [A], ("slogdet",)),
        "tri": (lambda t, b: torch.linalg.solve_triangular(t, b, upper=False, out=torch.empty_like(b)), [T, B],
                ("solve_triangular",)),
        "chol": (lambda b, l: torch.cholesky_solve(b, l, out=torch.empty_like(b)), [B, T], ("cholesky_solve",)),
        "eigh": (lambda a: torch.linalg.eigh(a, out=(a.new_empty(2, n), torch.empty_like(a)))[0], [S], ("eigh",)),
        "eigvalsh": (lambda a: torch.linalg.eigvalsh(a, out=a.new_empty(2, n)), [S], ("eigh",)),
    }
    for op, (fn, inputs, names) in outs.items():
        def on_device(*xs, fn=fn):  # out tensors are made on the inputs' device inside fn
            return fn(*xs)
        add(cases, f"out/{op}", "out", op, on_device, inputs, route_names=names, cause="out-not-routed")
    # in-place use of outputs (both should raise at backward, or both succeed with the same gradients)
    inplace = {
        "inv_mul_": lambda a: torch.linalg.inv(a).mul_(2),
        "solve_add_": lambda a, b: torch.linalg.solve(a, b).add_(1),
        "det_mul_": lambda a: torch.linalg.det(a).mul_(3),
        "slogdet_logabs_add_": lambda a: torch.linalg.slogdet(a)[1].add_(1),
        "slogdet_sign_mul_": lambda a: torch.linalg.slogdet(a)[0].mul_(1) + torch.linalg.slogdet(a)[1],
        "tri_add_": lambda t, b: torch.linalg.solve_triangular(t, b, upper=False).add_(1),
        "chol_add_": lambda b, l: torch.cholesky_solve(b, l).add_(1),
        "eigh_w_add_": lambda a: torch.linalg.eigh(a)[0].add_(1),
        "input_modified_after": None,
    }
    arg = {"inv_mul_": [A], "solve_add_": [A, B], "det_mul_": [A], "slogdet_logabs_add_": [A],
           "slogdet_sign_mul_": [A], "tri_add_": [T, B], "chol_add_": [B, T], "eigh_w_add_": [S]}
    for name, fn in inplace.items():
        if fn is None:
            continue
        op = name.split("_")[0]
        op = {"tri": "solve_triangular", "chol": "cholesky_solve"}.get(op, op)
        add(cases, f"inplace/{name}", "inplace", op, fn, arg[name], grad=[True] * len(arg[name]),
            route_names=(op,), known=eigh_known(op))

    def modified_input(a, b):  # A changed in place after the solve: torch's saved A trips the version check
        x = torch.linalg.solve(a * 1, b)
        return x

    add(cases, "inplace/solve_out_feeds_next", "inplace", "solve",
        lambda a, b: torch.linalg.solve(a, torch.linalg.solve(a, b)), [A, B], grad=[True, True])


def family_modes(cases):
    """inference_mode, no_grad, requires_grad on one input, double backward."""
    for n in (3, 22, 32, 33):
        table = op_table(gen(f"modes{n}"), (2,), n)
        for op, (fn, inputs, grad, names) in table.items():
            r = route_expected(n)
            # CPU stand-in: the backend key stays torch's, so inference mode is checked for values only
            r_inf = r if OPTS.hpu else None
            add(cases, f"modes/inference/n{n}/{op}", "inference", op, fn, inputs, route_names=names, route=r_inf,
                ctx=lambda dev: torch.inference_mode())
            add(cases, f"modes/no_grad/n{n}/{op}", "no_grad", op, fn, inputs, route_names=names, route=r,
                ctx=lambda dev: torch.no_grad())
            if len(inputs) == 2:
                for i in (0, 1):
                    mask = [j == i for j in range(2)]
                    add(cases, f"modes/grad_only_{i}/n{n}/{op}", "one_grad", op, fn, inputs, grad=mask,
                        route_names=names, route=r, known=eigh_known(op))
            if n != 22:
                add(cases, f"modes/double/n{n}/{op}", "double", op, fn, inputs, grad=grad, double=True,
                    route_names=names, route=r, floor=1e-5, known=eigh_known(op))


def family_autocast(cases):
    """autocast bf16/fp16, float32 inputs and low-precision inputs made inside the region."""
    n = 8
    for dt in (torch.bfloat16, torch.float16):
        tag = str(dt).split(".")[-1]
        table = op_table(gen(f"ac{tag}"), (2,), n)
        for op, (fn, inputs, grad, names) in table.items():
            ctx = lambda dev, dt=dt: torch.autocast(dev, dtype=dt)  # noqa: E731
            add(cases, f"autocast/{tag}/f32_inputs/{op}", "autocast", op, fn, inputs, grad=grad, route_names=names,
                ctx=ctx, route=None, floor=1e-2, known=eigh_known(op))

            def lowp(*xs, fn=fn):  # a matmul output (low precision under autocast) feeds the op
                eye = torch.eye(xs[0].shape[-1], dtype=xs[0].dtype, device=xs[0].device)
                return fn(*[(x @ eye if x.dim() >= 2 and x.shape[-1] == eye.shape[0] else x) for x in xs])
            add(cases, f"autocast/{tag}/lowp_inputs/{op}", "autocast", op, lowp, inputs, grad=grad,
                route_names=names, ctx=ctx, route=None, floor=1e-2, known=eigh_known(op))


def family_mixed(cases):
    """Mixed dtypes: torch's behaviour (promotion or its error) is the reference."""
    n = 5
    g = gen("mixed")
    A, B, T = general(g, (), n), f32r(randn(g, n, 2)), chol(g, (), n)
    two = {"solve": (lambda a, b: torch.linalg.solve(a, b), [A, B]),
           "tri": (lambda t, b: torch.linalg.solve_triangular(t, b, upper=False), [T, B]),
           "chol": (lambda b, l: torch.cholesky_solve(b, l), [B, T])}
    combos = {"f32_f64": (torch.float32, torch.float64), "f64_f32": (torch.float64, torch.float32),
              "f32_f16": (torch.float32, torch.float16), "f32_bf16": (torch.float32, torch.bfloat16)}
    for op, (fn, inputs) in two.items():
        for cname, dts in combos.items():
            add(cases, f"mixed/{cname}/{op}", "mixed", op, fn, inputs, dtypes=dts, keep_mixed=True, route=None,
                grad=[True, True], cause="mixed-dtype")


def family_scales(cases):
    """Very large and very small scales (a matrix and its right-hand side scaled together)."""
    scales = (1e-38, 1e-30, 1e-20, 1e-10, 1e10, 1e19, 1e20, 1e30, 2e38)
    for n in (4, 22):
        for s in scales:
            g = gen(f"scale{n}{s}")
            A, B, T, L, S = general(g, (2,), n), randn(g, 2, n, 3), chol(g, (2,), n), chol(g, (2,), n), sym(g, (2,), n)
            sa = s / n if s > 1 else s  # keep n*|A| inside float32
            As, Bs, Ts, Ls, Ss = (f32r(x * sa) for x in (A, B, T, L, S))
            cs = {
                "solve": (lambda a, b: torch.linalg.solve(a, b), [As, Bs]),
                "inv": (lambda a: torch.linalg.inv(a), [As]),
                "det_scaled": (lambda a: torch.linalg.det(a), [f32r(A * (s ** (1 / n)))]),
                "slogdet": (lambda a: torch.linalg.slogdet(a)[1], [As]),
                "tri": (lambda t, b: torch.linalg.solve_triangular(t, b, upper=False), [Ts, Bs]),
                "chol": (lambda b, l: torch.cholesky_solve(b, l), [Bs, f32r(L * math.sqrt(sa) if s > 1 else L * sa)]),
                "chol_L_scaled": (lambda b, l: torch.cholesky_solve(b, l), [Bs, Ls]),
                "eigvalsh": (lambda a: torch.linalg.eigvalsh(a), [Ss]),
            }
            for op, (fn, inputs) in cs.items():
                name = {"tri": "solve_triangular", "chol": "cholesky_solve", "chol_L_scaled": "cholesky_solve",
                        "eigvalsh": "eigh", "det_scaled": "det"}.get(op, op)
                cause = "chol-normal-equations" if op.startswith("chol") else None
                add(cases, f"scale/{s:.0e}/n{n}/{op}", "scale", op, fn, inputs, route_names=(name,),
                    route=route_expected(n), cause=cause)


def family_nonfinite(cases):
    """NaN and Inf at one entry: torch's behaviour (error, non-finite pattern) is the reference."""
    for n in (3, 22):
        for val, vname in ((float("nan"), "nan"), (float("inf"), "inf"), (-float("inf"), "-inf")):
            for where in ("diag", "off"):
                g = gen(f"nf{n}{vname}{where}")
                A, B, T, S = general(g, (2,), n), f32r(randn(g, 2, n, 2)), chol(g, (2,), n), sym(g, (2,), n)
                i, j = (1, 1) if where == "diag" else (n - 1, 0)
                for X in (A, T, S):
                    X[1, i, j] = val
                    if X is S:
                        S[1, j, i] = val
                cs = {"solve": (lambda a, b: torch.linalg.solve(a, b), [A, B], ("solve",)),
                      "solve_ex": (lambda a, b: torch.linalg.solve_ex(a, b)[0], [A, B], ("solve",)),
                      "inv": (lambda a: torch.linalg.inv(a), [A], ("inv",)),
                      "det": (lambda a: torch.linalg.det(a), [A], ("det",)),
                      "slogdet": (lambda a: torch.linalg.slogdet(a), [A], ("slogdet",)),
                      "tri": (lambda t, b: torch.linalg.solve_triangular(t, b, upper=False), [T, B],
                              ("solve_triangular",)),
                      "chol": (lambda b, l: torch.cholesky_solve(b, l), [B, T], ("cholesky_solve",)),
                      "eigh": (lambda a: torch.linalg.eigh(a)[0], [S], ("eigh",)),
                      "eigvalsh": (lambda a: torch.linalg.eigvalsh(a), [S], ("eigh",))}
                for op, (fn, inputs, names) in cs.items():
                    cause = "eigh-error" if op.startswith("eig") else None
                    add(cases, f"nonfinite/{vname}/{where}/n{n}/{op}", "nonfinite", op, fn,
                        [x.clone() for x in inputs], route_names=names, route=route_expected(n), cause=cause)


def family_singular(cases):
    """Exactly singular and near-singular matrices."""
    for n in (3, 8, 22, 32):
        g = gen(f"sing{n}")
        A = general(g, (4,), n)
        A[0] = 0  # zero matrix
        A[1, n // 2] = 0  # zero row
        A[2, -1] = A[2, 0]  # duplicate rows
        A[3] = f32r(randn(g, n, 1) @ randn(g, 1, n))  # rank one
        B = f32r(randn(g, 4, n, 2))
        T = chol(g, (4,), n)
        T[1, n // 2, n // 2] = 0  # zero diagonal: torch gives inf/NaN
        T[2, -1, -1] = 0
        r = route_expected(n)
        add(cases, f"singular/n{n}/solve", "singular", "solve", lambda a, b: torch.linalg.solve(a, b), [A, B],
            route=r)
        add(cases, f"singular/n{n}/solve_ex", "singular", "solve",
            lambda a, b: torch.linalg.solve_ex(a, b).info != 0, [A, B], route=r)
        add(cases, f"singular/n{n}/inv", "singular", "inv", lambda a: torch.linalg.inv(a), [A], route=r)
        add(cases, f"singular/n{n}/inv_ex_info", "singular", "inv", lambda a: torch.linalg.inv_ex(a).info != 0, [A],
            route=r)
        add(cases, f"singular/n{n}/det", "singular", "det", lambda a: torch.linalg.det(a), [A], route=r,
            floor=1e-5)
        add(cases, f"singular/n{n}/det_grad", "singular", "det", lambda a: torch.linalg.det(a), [A], grad=[True],
            route=r, cause="singular-grad")
        add(cases, f"singular/n{n}/slogdet", "singular", "slogdet", lambda a: torch.linalg.slogdet(a), [A], route=r)
        add(cases, f"singular/n{n}/tri_zero_diag", "singular", "solve_triangular",
            lambda t, b: torch.linalg.solve_triangular(t, b, upper=False), [T, B], route=r, finiteness="elementwise",
            cause="nan-triangle")
        add(cases, f"singular/n{n}/chol_zero_diag", "singular", "cholesky_solve",
            lambda b, l: torch.cholesky_solve(b, l), [B, T], route=r, cause="nan-triangle")
        # integer-valued rank-deficient matrix: LAPACK's float32 pivot is exactly zero, the kernel's may not be
        M = f32r(torch.arange(n * n, dtype=torch.float64).reshape(n, n) + 1)
        add(cases, f"singular/n{n}/arange_det", "singular", "det", lambda a: torch.linalg.det(a), [M], route=r,
            floor=1e-4)
        add(cases, f"singular/n{n}/arange_solve_ex_info", "singular", "solve",
            lambda a: torch.linalg.solve_ex(a, a[:, :1]).info != 0, [M], route=r)
    # near-singular / ill-conditioned
    for n in (8, 22, 32):
        for cond in (1e3, 1e5, 1e6):
            g = gen(f"cond{n}{cond}")
            S = spd(g, (2,), n, cond)
            B = f32r(randn(g, 2, n, 3))
            L = f32r(torch.linalg.cholesky(S))
            r = route_expected(n)
            tag = f"cond/{cond:.0e}/n{n}"
            add(cases, f"{tag}/solve", "cond", "solve", lambda a, b: torch.linalg.solve(a, b), [S, B],
                grad=[True, True], route=r)
            add(cases, f"{tag}/inv", "cond", "inv", lambda a: torch.linalg.inv(a), [S], grad=[True], route=r)
            add(cases, f"{tag}/slogdet", "cond", "slogdet", lambda a: torch.linalg.slogdet(a)[1], [S], grad=[True],
                route=r)
            add(cases, f"{tag}/chol", "cond", "cholesky_solve", lambda b, l: torch.cholesky_solve(b, l), [B, L],
                grad=[True, True], route=r, cause="chol-normal-equations")
            add(cases, f"{tag}/tri_cholfactor", "cond", "solve_triangular",
                lambda t, b: torch.linalg.solve_triangular(t, b, upper=False), [L, B], grad=[True, True], route=r,
                cause="gj-on-triangle")
            add(cases, f"{tag}/eigvalsh", "cond", "eigh", lambda a: torch.linalg.eigvalsh(a), [S], route=r)
    # random triangles (no dominant diagonal): ill-conditioned, but trsm stays accurate
    for n in (8, 16, 22, 32):
        for k in range(2):
            g = gen(f"tri_rand{n}{k}")
            T, B = f32r(torch.tril(randn(g, n, n))), f32r(randn(g, n, 3))
            add(cases, f"tri_random/n{n}/k{k}/lower", "tri_random", "solve_triangular",
                lambda t, b: torch.linalg.solve_triangular(t, b, upper=False), [T, B], route=route_expected(n),
                cause="gj-on-triangle")
            add(cases, f"tri_random/n{n}/k{k}/upper_right", "tri_random", "solve_triangular",
                lambda t, b: torch.linalg.solve_triangular(t.mT, b.mT, upper=True, left=False), [T, B],
                route=route_expected(n), cause="gj-on-triangle")


def family_eigh(cases):
    """Repeated eigenvalues, UPLO (both cases of the letter), garbage in the unread triangle."""
    for n in (2, 3, 6, 22, 32, 33):
        g = gen(f"eig{n}")
        Q, _ = torch.linalg.qr(randn(g, 2, n, n))
        spectra = {"identity": torch.ones(n), "zeros": torch.zeros(n), "two_values": (torch.arange(n) % 2).double(),
                   "clusters": torch.cat([torch.full((n - n // 2,), 1.0), torch.full((n // 2,), 1 + 1e-6)]),
                   "neg_scaled_identity": -3 * torch.ones(n)}
        for sname, d in spectra.items():
            M = f32r((Q * d.double()) @ Q.mT)
            M = (M + M.mT) / 2
            r = route_expected(n)
            add(cases, f"eigh/repeated/{sname}/n{n}/w_recon", "eigh_repeated", "eigh", eigh_fn(invariant=True), [M],
                route=r, floor=1e-5)
            add(cases, f"eigh/repeated/{sname}/n{n}/eigvalsh_grad", "eigh_repeated", "eigh",
                lambda a: torch.linalg.eigvalsh(a), [M], grad=[True], route=r, floor=1e-5, known="known-eigh-grad")
        G = f32r(randn(g, 2, n, n))  # not symmetric: each UPLO reads its own triangle
        for uplo in ("L", "U", "l", "u"):
            r = route_expected(n)
            cause = "uplo-case" if uplo.islower() else None
            add(cases, f"eigh/uplo_{uplo}/n{n}/eigh", "uplo", "eigh", eigh_fn(uplo), [G], route=r, cause=cause)
            add(cases, f"eigh/uplo_{uplo}/n{n}/eigvalsh", "uplo", "eigh",
                lambda a, u=uplo: torch.linalg.eigvalsh(a, UPLO=u), [G], grad=[True], route=r, cause=cause,
                known="known-eigh-grad")
    for b in (1, 2, 3, 5, 33, 65):  # the kernel packs two matrices per vector: odd batches leave a half
        g = gen(f"eigb{b}")
        add(cases, f"eigh/batch{b}/n7", "eigh_batch", "eigh", eigh_fn(invariant=True), [sym(g, (b,), 7)],
            route=route_expected(7))


def below_autograd_backend():
    """CPU stand-in of the card's compiled graphs, whose ops reach the ``HPU`` backend key, not ``AutogradHPU``:
    aot_autograd with both graphs run below autograd (plain aot_eager would call the stand-in's autograd kernels,
    which on the card no compiled graph does)."""
    from functorch.compile import make_boxed_func
    from torch._dynamo.backends.common import aot_autograd

    def below(gm, example_inputs):
        def run(*args):
            with torch._C._AutoDispatchBelowAutograd():
                return gm.forward(*args)

        return make_boxed_func(run)

    return aot_autograd(fw_compiler=below, bw_compiler=below)


def family_compile(cases):
    """torch.compile of a function that uses every routed op, forward and backward."""
    backend = "hpu_backend" if OPTS.hpu else below_autograd_backend()
    skip = "torch.compile needs PT_HPU_LAZY_MODE=0" if OPTS.hpu and LAZY else None

    def model(a, b, t, s):
        x = torch.linalg.solve(a, b)
        y = torch.linalg.inv(a) @ b
        z = torch.linalg.solve_triangular(t, x, upper=False)
        c = torch.cholesky_solve(y, t)
        w = torch.linalg.eigvalsh(s)
        return x.sum() + y.square().sum() + z.sum() + c.sum() + torch.linalg.det(a).sum() + \
            torch.linalg.slogdet(a)[1].sum() + w.square().sum()

    per_op = {
        "solve": (lambda a, b, t, s: torch.linalg.solve(a, b).sum(), ("solve",)),
        "inv": (lambda a, b, t, s: (torch.linalg.inv(a) @ b).square().sum(), ("inv",)),
        "det": (lambda a, b, t, s: torch.linalg.det(a).sum(), ("det",)),
        "slogdet": (lambda a, b, t, s: torch.linalg.slogdet(a)[1].sum(), ("slogdet",)),
        "tri": (lambda a, b, t, s: torch.linalg.solve_triangular(t, b, upper=False).sum(), ("solve_triangular",)),
        "chol": (lambda a, b, t, s: torch.cholesky_solve(b, t).sum(), ("cholesky_solve",)),
        "eigvalsh": (lambda a, b, t, s: torch.linalg.eigvalsh(s).square().sum(), ("eigh",)),
        "model": (model, ("solve", "inv", "det", "eigh", "solve_triangular", "cholesky_solve")),
    }
    for n in ((4, 22, 33) if OPTS.hpu else (4, 33)):
        for dynamic in ("static", "dynamic", "auto"):
            if not OPTS.hpu and n == 33 and dynamic == "static":
                continue  # tracing the 33-step composites on the CPU takes many minutes; the card declines n = 33
            g = gen(f"compile{n}{dynamic}")
            inputs = [general(g, (3,), n), f32r(randn(g, 3, n, 2)), chol(g, (3,), n), sym(g, (3,), n)]
            for op, (body, names) in per_op.items():
                if n == 33 and op in ("eigvalsh", "model"):
                    continue  # the 33x33 Jacobi composite takes minutes to trace; not a card path

                def fn(*xs, body=body, dynamic=dynamic):  # traced anew per run: the rewrites are read when tracing
                    torch._dynamo.reset()
                    if dynamic == "auto":  # a first call at batch 2, then batch 3: dynamo recompiles with a symbolic
                        compiled = torch.compile(body, backend=backend)  # batch, as a model's last short batch does
                        compiled(*[x[:2].detach() for x in xs])
                        return compiled(*xs)
                    return torch.compile(body, backend=backend, dynamic=dynamic == "dynamic")(*xs)
                route = route_expected(n) if dynamic == "static" else None
                cause = "compile-symbolic-derivative" if dynamic != "static" else None
                why = None
                if not OPTS.hpu and dynamic == "static" and op in ("eigvalsh", "model"):
                    why = "CPU stand-in: the numpy eigh reference cannot be traced (the card's op can)"
                add(cases, f"compile/{dynamic}/n{n}/{op}", "compile", op, fn, inputs, grad=[True] * 4, route=route,
                    route_names=names, skip=skip or why, floor=1e-5, cause=cause,
                    known="known-eigh-grad" if op in ("eigvalsh", "model") else None)


def family_functorch(cases):
    """vmap, grad, jacrev, jacfwd, hessian, jvp, vjp."""
    f = torch.func
    n = 4
    g = gen("functorch")
    A, B, T, S = general(g, (3,), n), f32r(randn(g, 3, n, 2)), chol(g, (3,), n), sym(g, (3,), n)
    cs = {
        "vmap_solve": (lambda a, b: f.vmap(torch.linalg.solve)(a, b), [A, B]),
        "vmap_inv": (lambda a: f.vmap(torch.linalg.inv)(a), [A]),
        "vmap_det": (lambda a: f.vmap(torch.linalg.det)(a), [A]),
        "vmap_slogdet": (lambda a: f.vmap(torch.linalg.slogdet)(a), [A]),
        "vmap_tri": (lambda t, b: f.vmap(lambda x, y: torch.linalg.solve_triangular(x, y, upper=False))(t, b), [T, B]),
        "vmap_chol": (lambda b, t: f.vmap(torch.cholesky_solve)(b, t), [B, T]),
        "vmap_eigvalsh": (lambda s: f.vmap(torch.linalg.eigvalsh)(s), [S]),
        "grad_det": (lambda a: f.vmap(f.grad(torch.linalg.det))(a), [A]),
        "grad_logabsdet": (lambda a: f.grad(lambda x: torch.linalg.slogdet(x)[1].sum())(a), [A]),
        "grad_solve": (lambda a, b: f.grad(lambda x: torch.linalg.solve(x, b).square().sum())(a), [A, B]),
        "grad_tri": (lambda t, b: f.grad(lambda x: torch.linalg.solve_triangular(x, b, upper=False).sum())(t), [T, B]),
        "grad_chol": (lambda b, t: f.grad(lambda x: torch.cholesky_solve(b, x).sum())(t), [B, T]),
        "grad_eigvalsh": (lambda s: f.grad(lambda x: torch.linalg.eigvalsh(x).square().sum())(s), [S]),
        "jacrev_inv": (lambda a: f.jacrev(torch.linalg.inv)(a[0]), [A]),
        "jacfwd_inv": (lambda a: f.jacfwd(torch.linalg.inv)(a[0]), [A]),
        "jacrev_solve": (lambda a, b: f.jacrev(torch.linalg.solve, argnums=(0, 1))(a[0], b[0]), [A, B]),
        "hessian_det": (lambda a: f.hessian(torch.linalg.det)(a[0]), [A]),
        "hessian_logdet": (lambda a: f.hessian(lambda x: torch.linalg.slogdet(x)[1])(a[0]), [A]),
        "jvp_solve": (lambda a, b: f.jvp(torch.linalg.solve, (a, b), (torch.ones_like(a), b))[1], [A, B]),
        "vjp_inv": (lambda a: f.vjp(torch.linalg.inv, a)[1](torch.ones_like(a))[0], [A]),
        "jacrev_eigvalsh": (lambda s: f.jacrev(torch.linalg.eigvalsh)(s[0]), [S]),
        "grad_of_grad_det": (lambda a: f.grad(lambda x: f.grad(torch.linalg.det)(x).sum())(a[0]), [A]),
    }
    for name, (fn, inputs) in cs.items():
        add(cases, f"functorch/{name}", "functorch", name, fn, inputs, route=None, floor=1e-5,
            known="known-eigh-grad" if "eig" in name else None)


def family_devices(cases):
    """An HPU matrix with a CPU right-hand side: torch raises; the rewrite must not compute."""
    if not OPTS.hpu:
        add(cases, "device/mixed_hpu_cpu", "device", "solve", None, [], skip="needs the card")
        return
    g = gen("device")
    A, B = general(g, (), 4).float(), f32r(randn(g, 4, 2)).float()

    def custom(op_fn, name):
        def check():  # torch's linalg ops raise on an HPU A with a CPU B (no implicit device move)
            want = op_fn(A, B)
            try:
                got = op_fn(A.to("hpu"), B)
            except Exception as error:  # noqa: BLE001
                text = short(error)
                ok = re.search(r"device|same device|Expected all tensors", text)
                return (None, text) if ok else ("message", f"raised [{text}], torch raises a device mismatch")
            got = got.cpu()
            if torch.allclose(got, want, rtol=1e-4, atol=1e-5):
                return "message", f"computed with mixed devices (on {got.device}); torch raises a device mismatch"
            return "wrong-result", f"computed with mixed devices, max diff {(got - want).abs().max().item():.2e}"
        return check

    for name, op_fn in {"solve": lambda a, b: torch.linalg.solve(a, b),
                        "tri": lambda a, b: torch.linalg.solve_triangular(a, b, upper=False),
                        "chol": lambda a, b: torch.cholesky_solve(b, a)}.items():
        add(cases, f"device/mixed_hpu_cpu/{name}", "device", name, None, [], custom=custom(op_fn, name))


def family_float64_backward(cases):
    """float64 on the card: forward on the CPU; does the backward stay float64-exact and off Habana's fallback?"""
    for n, cause in ((6, "float64-backward"), (40, "past-maxn-backward")):
        g = gen(f"f64bw{n}")
        A, B = general(g, (2,), n), randn(g, 2, n, 2)
        dt = torch.float64 if n == 6 else torch.float32
        for op, fn, inputs in (("solve", lambda a, b: torch.linalg.solve(a, b), [A, B]),
                               ("det", lambda a: torch.linalg.det(a), [A]),
                               ("slogdet", lambda a: torch.linalg.slogdet(a)[1], [A]),
                               ("inv", lambda a: torch.linalg.inv(a), [A])):
            floor = 1e-12 if dt == torch.float64 else FLOOR
            add(cases, f"{cause}/n{n}/{op}", cause, op, fn, [x if dt == torch.float64 else f32r(x) for x in inputs],
                dtype=dt, grad=[True] * len(inputs), route=None, floor=floor, cause=cause)


def family_errors(cases):
    """Bad shapes: torch's error is the reference."""
    g = gen("errors")
    A, T = general(g, (), 3), chol(g, (), 3)
    bad = {
        "tri_B_1d": (lambda t, b: torch.linalg.solve_triangular(t, b, upper=False), [T, randn(g, 3)]),
        "tri_B_rows": (lambda t, b: torch.linalg.solve_triangular(t, b, upper=False), [T, randn(g, 4, 2)]),
        "tri_B_right_cols": (lambda t, b: torch.linalg.solve_triangular(t, b, upper=False, left=False),
                             [T, randn(g, 2, 4)]),
        "chol_B_1d": (lambda b, l: torch.cholesky_solve(b, l), [randn(g, 3), T]),
        "chol_B_rows": (lambda b, l: torch.cholesky_solve(b, l), [randn(g, 4, 2), T]),
        "solve_B_rows": (lambda a, b: torch.linalg.solve(a, b), [A, randn(g, 4, 2)]),
        "solve_vec_right": (lambda a, b: torch.linalg.solve(a, b, left=False), [A, randn(g, 3)]),
        "inv_nonsquare": (lambda a: torch.linalg.inv(a[:, :2]), [A]),
        "eigh_bad_uplo": (lambda a: torch.linalg.eigh(a, UPLO="X"), [A]),
        "det_1d": (lambda a: torch.linalg.det(a[0]), [A]),
    }
    def sign_backward(a):  # torch's real sign requires grad (a zero gradient); the rewrite marks it non-differentiable
        s = torch.linalg.slogdet(a).sign
        s.sum().backward()
        return s

    for n in (3, 33):
        add(cases, f"errors/slogdet_sign_backward/n{n}", "errors", "slogdet", sign_backward, [general(g, (), n)],
            grad=[True], route=None, cause="slogdet-sign")
    for name, (fn, inputs) in bad.items():
        cause = "shape-check" if name.startswith(("tri", "chol")) else None
        add(cases, f"errors/{name}", "errors", name, fn, inputs, route=None, cause=cause)


def family_enable(cases):
    """enable() twice, the undo, and enable() after the undo."""
    key = "AutogradHPU"
    ops_ = ("aten::_linalg_solve_ex", "aten::linalg_inv_ex")  # the default install (eigh is opt-in, triangular off)

    def has():
        return [torch._C._dispatch_has_kernel_for_dispatch_key(o, key) for o in ops_]

    @contextlib.contextmanager
    def fake_habana():
        """CPU: a stub habana_frameworks.torch.core, so enable() runs its real code and registers at HPU keys."""
        if OPTS.hpu:
            yield
            return
        names = ["habana_frameworks", "habana_frameworks.torch", "habana_frameworks.torch.core"]
        saved = {k: sys.modules.get(k) for k in names}
        env = dict(os.environ)
        for k in names:
            sys.modules[k] = types.ModuleType(k)
        try:
            yield
        finally:
            for k, v in saved.items():
                if v is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v
            os.environ.clear()
            os.environ.update(env)

    def enable_twice():
        with fake_habana():
            hpu_ops.enable()
            first = has()
            undo = hpu_ops._STATE["undo"]
            hpu_ops.enable()
            same = hpu_ops._STATE["undo"] is undo and has() == first
            if not OPTS.hpu:
                hpu_ops._STATE["undo"]()
                hpu_ops._STATE.pop("undo", None)
        if not all(first):
            return "fallback", f"enable() registered {dict(zip(ops_, first))}"
        return (None, "idempotent") if same else ("crash", "second enable() changed the registrations")

    def enable_after_undo():
        with fake_habana():
            hpu_ops.enable()
            hpu_ops._STATE["undo"]()
            off = has()
            hpu_ops.enable()
            again = has()
            try:
                hpu_ops._STATE["undo"]()  # a second undo of the same registrations
                second_undo = None
            except Exception as error:  # noqa: BLE001
                second_undo = short(error)
            if not OPTS.hpu:
                hpu_ops._STATE.pop("undo", None)
        if any(off):
            return "wrong-result", f"undo left kernels: {dict(zip(ops_, off))}"
        if not all(again):
            return "fallback", f"enable() after the undo registered nothing: {dict(zip(ops_, again))}" + \
                (f"; second undo: {second_undo}" if second_undo else "")
        if second_undo:
            return "crash", f"second undo raised {second_undo}"
        return None, "re-enabled"

    def install_twice_cpu():
        if OPTS.hpu:
            return None, "CPU only"
        u1 = install_linalg("cpu")
        u2 = install_linalg("cpu")
        A = torch.randn(3, 3) + 3 * torch.eye(3)
        hpu_ops._STATE["calls"].clear()
        torch.linalg.inv(A)
        routed = hpu_ops._STATE["calls"].get("inv", 0)
        u2()
        u1()
        hpu_ops._STATE["calls"].clear()
        torch.linalg.inv(A)
        after = hpu_ops._STATE["calls"].get("inv", 0)
        if routed != 1:
            return "wrong-result", f"inv counted {routed} times with two installs"
        if after:
            return "wrong-result", "undo did not restore torch"
        return None, "one registration, undo restores torch"

    add(cases, "enable/twice", "enable", "enable", None, [], custom=enable_twice, cause="enable-undo")
    add(cases, "enable/install_twice_cpu", "enable", "enable", None, [], custom=install_twice_cpu,
        cause="enable-undo")
    # last: on the card this leaves hpu-ops in whatever state enable() after the undo gives
    add(cases, "enable/after_undo", "enable", "enable", None, [], custom=enable_after_undo, cause="enable-undo")


# ------------------------------------------------------------------------------------------------ main


def main():
    cases = []
    for family in (family_shapes, family_layouts, family_out, family_modes, family_autocast, family_mixed,
                   family_scales, family_nonfinite, family_singular, family_eigh, family_compile, family_functorch,
                   family_devices, family_float64_backward, family_errors, family_enable):
        family(cases)
    print(f"# {len(cases)} cases on {DEV} (lazy={LAZY if OPTS.hpu else 'n/a'}), base seed {OPTS.seed}", flush=True)
    for case in cases:
        run_case(case)
    ran = [r for r in RESULTS if r[1] != "skip"]
    failed = [r for r in ran if r[1] is not None]
    print("\n# ---- summary")
    print(f"# {len(ran)} run, {len(ran) - len(failed)} PASS, {len(failed)} FAIL, "
          f"{len(RESULTS) - len(ran)} SKIP")
    by_sev = collections.Counter(kind for _, kind, _ in failed)
    print("# by severity: " + ", ".join(f"{k} {by_sev[k]}" for k in SEVERITY if by_sev[k]))
    groups = collections.defaultdict(list)
    for case, kind, detail in failed:
        cause = case.known if (case.known and OPTS.hpu and case.op.startswith("eig") and kind == "wrong-result"
                               and "grad" in detail) else classify(case, kind, detail)
        groups[cause].append((case, kind, detail))
    order = lambda item: (min(SEVERITY.index(k) for _, k, _ in item[1]), -len(item[1]))  # noqa: E731
    for cause, items in sorted(groups.items(), key=order):
        kinds = collections.Counter(k for _, k, _ in items)
        print(f"# [{', '.join(f'{k} {v}' for k, v in kinds.items())}] {cause}")
        if cause in CAUSES:
            print(f"#     hypothesis: {CAUSES[cause]}")
        for case, kind, detail in items[:4]:
            print(f"#     e.g. {case.id} seed={case.seed}: {detail[:140]}")
        if len(items) > 4:
            print(f"#     ... and {len(items) - 4} more")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
