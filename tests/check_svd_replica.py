"""On a Gaudi card: the Jacobi SVD kernel (Codex review #6: relative rotation cutoffs, exact 45-degree rotation of
equal norms, row norms without squared underflow, completion of the zero rows only) against its float32 replica
``kernel_replica.jacobi_svd_kernel``, unsorted (``sort=False``): thin (U, S, Vh), full_matrices and svdvals (the -DNO_UV
build, two matrices per iteration), on the review's inputs and ordinary ones, and the public routes' answers on the
review's inputs against torch on the CPU.
    PT_HPU_LAZY_MODE=0|1 python3 tests/check_svd_replica.py
    SVD_REPLICA_LEGACY=1 with origin/main's hpu_ops first on PYTHONPATH: main's kernel against the legacy replica (the
    baseline: v_rsqrt_f32 and v_reciprocal_f32 are modelled as correctly rounded, so neither is bit for bit)
One ``RESULT {json}`` per case: ``bit_equal`` (every output equals the replica's bits), the largest difference
(``max_diff``; ``S_diff`` for the singular values), the singular values' error against torch in float64 (relative to
each value, not to the largest), and for the factors ``recon`` (max|U S Vh - A| / max|A|) and ``orth``. The replica
models ``v_rsqrt_f32`` / ``v_reciprocal_f32`` as correctly rounded, so outputs past them can differ by ulps (also on
origin/main's kernel and replica). Pass: the routes give torch's answers, the factors are an SVD to 1e-5, and the
singular values are the replica's bits or within 2e-5 max|A|. Last line ``SUMMARY ok`` or ``MISMATCH``."""
import json
import os
import warnings

import numpy as np
import torch

import hpu_ops
import kernel_replica as kr

warnings.filterwarnings("ignore")
hpu_ops.enable()
from hpu_ops import tpc  # noqa: E402

LAZY = os.environ.get("PT_HPU_LAZY_MODE", "1") != "0"
LEGACY = os.environ.get("SVD_REPLICA_LEGACY") == "1"  # origin/main's kernel against the replica's legacy_cutoffs


def sync():
    import habana_frameworks.torch.core as htcore
    import habana_frameworks.torch.hpu as hthpu

    if LAZY:
        htcore.mark_step()
    hthpu.synchronize()


def replica(a, compute_uv, full):
    """The kernel's unsorted outputs mapped like ``tpc.jacobi_svd(sort=False)`` (contiguous input)."""
    m, n = a.shape
    k, big = min(m, n), max(m, n)
    full = full and m != n
    swap = m < n or m == n
    Ut, S, Vt = kr.jacobi_svd_kernel(a if swap else a.T, compute_uv, full, legacy_cutoffs=LEGACY)
    if not compute_uv:
        return (S,)
    U, Vh = (Vt.T, Ut) if swap else (Ut.T, Vt)
    return U, S, Vh


def cases():
    g = torch.Generator().manual_seed(0)
    yield "codex6_identical_small_rows", torch.tensor([[1.0, 0, 0], [0, 1e-10, 0], [0, 1e-10, 0]])
    yield "codex6_diag_1e20_1", torch.diag(torch.tensor([1e20, 1.0]))
    yield "identical_rows_signs", torch.tensor([[2.0, 1, 0], [-2.0, -1, 0], [0, 0, 3]])
    yield "graded_1e-30", torch.diag(torch.tensor([1.0, 1e-15, 1e-30]))
    for shape in ((3, 3), (8, 8), (22, 22), (32, 32), (7, 3), (3, 7), (32, 17), (17, 32)):
        yield f"randn_{shape[0]}x{shape[1]}", torch.randn(*shape, generator=g)
    A = torch.randn(22, 22, generator=g)
    A[:, -1] = A[:, 0]
    yield "rank_deficient_22", A


rows, ok = [], True
for name, a in cases():
    for mode in ("thin", "full", "svdvals"):
        compute_uv, full = mode != "svdvals", mode == "full"
        want = replica(a.numpy(), compute_uv, full)
        h = a[None].to("hpu")
        if mode == "svdvals":
            got = (tpc.jacobi_svdvals(h, sort=False),)
        else:
            got = tpc.jacobi_svd(h, full_matrices=full, sort=False)
        sync()
        got = [g[0].cpu().numpy() for g in got]
        diff = max(float(np.abs(g.astype(np.float64) - w).max()) if g.size else 0.0 for g, w in zip(got, want))
        bit = all(np.array_equal(g.view(np.uint32), np.asarray(w, np.float32).view(np.uint32)) for g, w in zip(got, want))
        S = np.sort(got[1] if compute_uv else got[0])[::-1].astype(np.float64)
        ref = torch.linalg.svdvals(a.double()).numpy()
        floor = 1e-6 * ref.max()  # values below float32's resolution of the largest: absolute
        sv_err = float(np.max(np.abs(S - ref) / np.maximum(ref, floor)))
        S_diff = float(np.abs(np.sort(got[1] if compute_uv else got[0]) - np.sort(want[1] if compute_uv else want[0])).max())
        scale = float(np.abs(a.numpy()).max()) or 1.0
        row = dict(case=name, mode=mode, lazy=LAZY, legacy=LEGACY, bit_equal=bit, max_diff=diff, S_diff=S_diff,
                   sv_rel_err=sv_err)
        if compute_uv:  # the card's factors are an SVD (a rank-deficient null vector may differ from the replica's)
            U, Sg, Vh = (x.astype(np.float64) for x in got)
            k = Sg.shape[0]
            row["recon"] = float(np.abs((U[:, :k] * Sg) @ Vh[:k] - a.numpy()).max() / scale)
            row["orth"] = float(max(np.abs(U.T @ U - np.eye(U.shape[1])).max(), np.abs(Vh @ Vh.T - np.eye(Vh.shape[0])).max()))
            ok &= row["recon"] < 1e-5 and row["orth"] < 1e-5
        # bit for bit where no hardware rsqrt / reciprocal rounding differs; else within float32 noise of the matrix
        ok &= bit or S_diff <= 2e-5 * scale
        print("RESULT " + json.dumps(row), flush=True)

# the public routes on the review's inputs, against torch on the CPU
A = torch.tensor([[1.0, 0, 0], [0, 1e-10, 0], [0, 1e-10, 0]])
D = torch.diag(torch.tensor([1e20, 1.0]))
for name, f in (("rank_rtol0", lambda x: torch.linalg.matrix_rank(x, rtol=0.0)), ("svdvals", torch.linalg.svdvals),
                ("norm_-2", lambda x: torch.linalg.matrix_norm(x, ord=-2)),
                ("pinv_rtol0", lambda x: torch.linalg.pinv(x, rtol=0.0))):
    for xn, x in (("codex6_rows", A), ("codex6_diag", D)):
        got = f(x.to("hpu"))
        sync()
        got, want = got.cpu(), f(x)
        good = torch.equal(got, want) if got.dtype == torch.int64 else bool(torch.allclose(got, want, rtol=1e-5, atol=0))
        ok &= good
        print("RESULT " + json.dumps(dict(case=xn, route=name, lazy=LAZY, ok=good, got=got.flatten().tolist()[:4],
                                          want=want.flatten().tolist()[:4])), flush=True)
print("SUMMARY " + ("ok" if ok else "MISMATCH"), flush=True)
