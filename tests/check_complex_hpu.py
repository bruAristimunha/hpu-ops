"""On a Gaudi card: the complex64 TPC kernels of ``hpu_ops.tpc.complex_linalg`` against their float32 replicas
(``kernel_replica``) bit for bit, their error against complex128 next to torch's complex64 CPU error, CPU fallback
counts, and the per-call time against torch's CPU complex route (copy to the host, the complex64 op, copy back).
    PT_HPU_LAZY_MODE=0|1 python3 tests/check_complex_hpu.py [gj|eigh|svd|eig|adv|all] [--time]
One ``RESULT {json}`` per case; the last line is ``SUMMARY ok`` or ``SUMMARY MISMATCH``."""
import json
import math
import sys
import time
import warnings

import numpy as np
import torch

import kernel_replica as kr

warnings.filterwarnings("ignore")
C64, C128 = torch.complex64, torch.complex128


def crandn(*shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.complex(torch.randn(*shape, generator=g), torch.randn(*shape, generator=g)).to(C64)


def cond_matrix(b, n, cond, seed=0):
    U, _ = torch.linalg.qr(crandn(b, n, n, seed=seed).to(C128))
    V, _ = torch.linalg.qr(crandn(b, n, n, seed=seed + 1).to(C128))
    s = torch.logspace(0, -math.log10(cond), n, dtype=torch.float64).to(C128)
    return ((U * s[None, None, :]) @ V.mH).to(C64)


def pairs(Z):
    return torch.view_as_real(Z.to(C64)).contiguous()


def cplx(P):
    return torch.view_as_complex(P.double().contiguous())


def err(x, ref):
    x, ref = x.to(C128), ref.to(C128)
    return float((x - ref).abs().max() / ref.abs().max().clamp_min(1e-300))


def fallbacks():
    try:
        from habana_frameworks.torch.hpu.metrics import metric_global
        stats = dict(metric_global("cpu_fallback").stats())
        return sum(stats.get("FallbackOps", {}).values()) if isinstance(stats.get("FallbackOps"), dict) else \
            int(stats.get("TotalNumber", 0))
    except Exception:  # noqa: BLE001
        return -1


def sync(x):
    import habana_frameworks.torch.core as htcore
    htcore.mark_step()
    torch.hpu.synchronize()
    return x


def timed(fn, reps=50):
    for _ in range(3):
        sync(fn())
    keep = []
    t0 = time.perf_counter()
    for _ in range(reps):
        keep.append(fn())
        sync(keep[-1])
    return (time.perf_counter() - t0) / reps * 1e3


def gj_cases():
    yield "randn_16", crandn(4, 16, 16, seed=1)
    yield "randn_32", crandn(4, 32, 32, seed=2)
    yield "n1", crandn(3, 1, 1, seed=3)
    yield "cond1e6_22", cond_matrix(3, 22, 1e6, seed=4)
    yield "i_I_plus_small", (1j * torch.eye(8) + 1e-3 * crandn(8, 8, seed=5)).to(C64)[None]
    A = crandn(2, 6, 6, seed=6)
    A[0, :, 2] = 0
    yield "singular", A
    A = crandn(2, 5, 5, seed=7)
    A[0, 2, 3] = complex(math.nan, 0)
    A[1, 1, 1] = complex(0, math.inf)
    yield "nonfinite", A
    yield "huge_3e37", crandn(2, 6, 6, seed=8) * 3e37
    yield "tiny_1e-30", crandn(2, 6, 6, seed=9) * 1e-30
    yield "eye_32", torch.eye(32, dtype=C64)[None]
    yield "i_perm_6", (1j * torch.eye(6)[[2, 0, 1, 5, 3, 4]]).to(C64)[None]
    yield "int_exact", exact_int_matrices()


def exact_int_matrices():
    """Small-integer matrices whose elimination is exact in float32, and their determinants."""
    M = torch.tensor([[[2, 1, 0], [1, 1, 0], [0, 0, 1]], [[2, 0, 0], [0, 3, 0], [0, 0, -1]],
                      [[1, 2, 3], [0, 4, 5], [0, 0, 6]], [[0, 0, 2], [0, 3, 1], [4, 1, 1]]], dtype=C64)
    return M


def planar(Z):
    P = pairs(Z)
    b, n, m = P.shape[0], P.shape[1], P.shape[2]
    return P.permute(0, 3, 1, 2).reshape(b, 2 * n, m)


def run_gj(emit, time_it):
    from hpu_ops.tpc import complex_linalg as cl
    for name, A in gj_cases():
        n = A.shape[-1]
        B = crandn(A.shape[0], n, 3, seed=50)
        f0 = fallbacks()
        X, D, S = cl.cgj_solve(pairs(A).to("hpu"), pairs(B).to("hpu"))
        Xi, Di, Si = cl.cgj_inv(pairs(A).to("hpu"))
        X, D, S, Xi, Di, Si = (t.cpu() for t in (X, D, S, Xi, Di, Si))
        fb = fallbacks() - f0
        reps = [kr.cgj_solve(a, b) for a, b in zip(planar(A).numpy(), planar(B).numpy())]
        repi = [kr.cgj_solve(a) for a in planar(A).numpy()]
        rX = torch.from_numpy(np.stack([r[0] for r in reps])).reshape(-1, 2, n, 3).permute(0, 2, 3, 1)
        rD, rS = torch.from_numpy(np.stack([r[1] for r in reps])), torch.from_numpy(np.stack([r[2] for r in reps]))
        rXi = torch.from_numpy(np.stack([r[0] for r in repi])).reshape(-1, 2, n, n).permute(0, 2, 3, 1)
        same = all(torch.equal(a.nan_to_num(), b.nan_to_num()) for a, b in
                   ((X, rX), (D, rD), (S, rS), (Xi, rXi)))
        ndiff = int(sum((a.nan_to_num() != b.nan_to_num()).sum() for a, b in ((X, rX), (D, rD), (S, rS), (Xi, rXi))))
        row = {"op": "cgj", "case": name, "n": n, "bit_equal": same, "n_diff": ndiff, "fallbacks": fb, "ok": same}
        if torch.isfinite(A).all() and name != "singular":
            sign, lad = cl.slogdet(pairs(A).to("hpu"))
            s128, l128 = torch.linalg.slogdet(A.to(C128))
            s32, l32 = torch.linalg.slogdet(A)
            row.update(err_inv=err(cplx(Xi), torch.linalg.inv(A.to(C128))),
                       err_inv_cpu32=err(torch.linalg.inv(A), torch.linalg.inv(A.to(C128))),
                       err_sign=err(cplx(sign.cpu()), s128), err_sign_cpu32=err(s32, s128),
                       err_lad=float((lad.cpu().double() - l128).abs().max()),
                       err_lad_cpu32=float((l32.double() - l128).abs().max()))
        emit(row)
    if not time_it:
        return
    for n in (16, 22, 32):
        A = crandn(32, n, n, seed=n)
        B = crandn(32, n, n, seed=n + 1)
        Ah, Bh = pairs(A).to("hpu"), pairs(B).to("hpu")

        def cpu_route(op, *ts):
            return lambda: torch.view_as_real(op(*[torch.view_as_complex(t.cpu()) for t in ts])).to("hpu")

        def cpu_slogdet():
            s, l = torch.linalg.slogdet(torch.view_as_complex(Ah.cpu()))
            return torch.view_as_real(s).to("hpu"), l.to("hpu")

        f0 = fallbacks()
        row = {"op": "timing_gj", "n": n, "batch": 32,
               "solve_ms": timed(lambda: cl.solve(Ah, Bh)), "solve_cpu_ms": timed(cpu_route(torch.linalg.solve, Ah, Bh)),
               "inv_ms": timed(lambda: cl.inv(Ah)), "inv_cpu_ms": timed(cpu_route(torch.linalg.inv, Ah)),
               "slogdet_ms": timed(lambda: cl.slogdet(Ah)), "slogdet_cpu_ms": timed(cpu_slogdet),
               "ok": True}
        row["fallbacks_in_kernel_calls"] = fallbacks() - f0
        emit(row)


def unitary(n, seed):
    return torch.linalg.qr(crandn(n, n, seed=seed).to(C128))[0]


def herm(w, seed):
    Q = unitary(len(w), seed)
    return ((Q * torch.as_tensor(w, dtype=torch.float64)) @ Q.mH).to(C64)


def eigh_cases():
    yield "randn_16", (lambda X: X + X.mH)(crandn(2, 16, 16, seed=1))
    yield "randn_22", (lambda X: X + X.mH)(crandn(2, 22, 22, seed=2))
    yield "randn_32", (lambda X: X + X.mH)(crandn(2, 32, 32, seed=3))
    yield "odd_5_batch3", (lambda X: X + X.mH)(crandn(3, 5, 5, seed=4))
    yield "mult3_16", herm([0, 0, 0] + list(range(1, 14)), 5)[None]
    yield "close_1e-7", herm([1, 1 + 1e-7] + list(range(2, 16)), 6)[None]
    yield "cond1e6", herm(np.logspace(0, -6, 16), 7)[None]
    yield "scaled_1e30", herm(np.linspace(-1, 3, 12), 8)[None] * 1e30
    H = (lambda X: X + X.mH)(crandn(2, 5, 5, seed=9))
    H[0, 3, 1] = complex(math.nan, 0)
    yield "nan", H


def run_eigh(emit, time_it):
    from hpu_ops.tpc import complex_linalg as cl
    for name, H in eigh_cases():
        n = H.shape[-1]
        a = planar(H)
        f0 = fallbacks()
        V1, Hk = cl._call(cl._ceigh_op(), a.to("hpu"))
        W, V = cl._call(cl._ceigh_refine_op(), Hk, V1)
        w, Vp = cl.eigh(pairs(H).to("hpu"))
        V1, Hk, W, V, w, Vp = (t.cpu() for t in (V1, Hk, W, V, w, Vp))
        fb = fallbacks() - f0
        rep1 = [kr.ceigh_pass1(x) for x in a.numpy()]
        rep2 = [kr.ceigh_pass2(h, v) for (v, h) in rep1]
        pairs_ = ((V1, np.stack([r[0] for r in rep1])), (Hk, np.stack([r[1] for r in rep1])),
                  (W, np.stack([r[0] for r in rep2])), (V, np.stack([r[1] for r in rep2])))
        ndiff = int(sum((x.nan_to_num() != torch.from_numpy(y).nan_to_num()).sum() for x, y in pairs_))
        row = {"op": "ceigh", "case": name, "n": n, "bit_equal": ndiff == 0, "n_diff": ndiff, "fallbacks": fb,
               "ok": ndiff == 0}
        if torch.isfinite(H).all():
            H128 = H.to(C128)
            nrm = H128.abs().amax((-2, -1), keepdim=True)
            w128 = torch.linalg.eigvalsh(H128)
            w32, V32 = torch.linalg.eigh(H)

            def errs(wv, Vv):
                Vv = Vv.to(C128)
                return (float(((wv.double() - w128).abs() / nrm[..., 0]).max()),
                        float(((H128 @ Vv - Vv * wv.to(C128)[..., None, :]).abs() / nrm).max()),
                        float((Vv.mH @ Vv - torch.eye(n)).abs().max()))
            ours, ref = errs(w, cplx(Vp)), errs(w32, V32)
            row.update(err_w=ours[0], err_w_cpu32=ref[0], res=ours[1], res_cpu32=ref[1], orth=ours[2],
                       orth_cpu32=ref[2])
            row["ok"] = row["ok"] and all(e <= max(1e-6, 10 * t) for e, t in zip(ours, ref))
        emit(row)
    if not time_it:
        return
    for n in (16, 22, 32):
        H = (lambda X: X + X.mH)(crandn(32, n, n, seed=n))
        Hh = pairs(H).to("hpu")

        def cpu_route():
            w, V = torch.linalg.eigh(torch.view_as_complex(Hh.cpu()))
            return w.to("hpu"), torch.view_as_real(V).to("hpu")

        f0 = fallbacks()
        row = {"op": "timing_eigh", "n": n, "batch": 32, "eigh_ms": timed(lambda: cl.eigh(Hh)),
               "eigh_cpu_ms": timed(cpu_route), "eigvalsh_ms": timed(lambda: cl.eigvalsh(Hh)),
               "eigvalsh_cpu_ms": timed(lambda: torch.linalg.eigvalsh(torch.view_as_complex(Hh.cpu())).to("hpu")),
               "ok": True}
        row["fallbacks_in_kernel_calls"] = fallbacks() - f0
        emit(row)


def svd_cases():
    yield "randn_16", crandn(2, 16, 16, seed=11)
    yield "randn_32", crandn(2, 32, 32, seed=12)
    yield "wide_8x20", crandn(3, 8, 20, seed=13)
    yield "tall_20x8", crandn(2, 20, 8, seed=14)
    yield "n1", crandn(2, 1, 1, seed=15)
    s = torch.logspace(0, -5, 16, dtype=torch.float64).to(C128)
    yield "logspace_-5", ((unitary(16, 16) * s) @ unitary(16, 17).mH).to(C64)[None]
    yield "rank10", crandn(1, 16, 10, seed=18) @ crandn(1, 10, 16, seed=19)
    yield "zeros", torch.zeros(2, 5, 7, dtype=C64)
    A = crandn(2, 6, 4, seed=20)
    A[0, 1, 2] = complex(math.inf, 0)
    yield "inf", A


def run_svd(emit, time_it):
    from hpu_ops.tpc import complex_linalg as cl
    for name, A in svd_cases():
        m, n = A.shape[-2:]
        P, _, _, wide, a = cl._svd_input(pairs(A), "pairs")
        f0 = fallbacks()
        Wn, S, J = cl._call(cl._csvd_op(), a.to("hpu"))
        Sv = cl._call(cl._csvdvals_op(), a.to("hpu"))
        U, Ss, Vh = cl.svd(pairs(A).to("hpu"))
        Wn, S, J, Sv, U, Ss, Vh = (t.cpu() for t in (Wn, S, J, Sv, U, Ss, Vh))
        fb = fallbacks() - f0
        rep = [kr.csvd_kernel(x) for x in a.numpy()]
        repv = [kr.csvd_kernel(x, compute_uv=False)[1] for x in a.numpy()]
        pairs_ = ((Wn, np.stack([r[0] for r in rep])), (S, np.stack([r[1] for r in rep])),
                  (J, np.stack([r[2] for r in rep])), (Sv, np.stack(repv)))
        ndiff = int(sum((x.nan_to_num() != torch.from_numpy(y).nan_to_num()).sum() for x, y in pairs_))
        row = {"op": "csvd", "case": name, "shape": [m, n], "bit_equal": ndiff == 0, "n_diff": ndiff,
               "fallbacks": fb, "ok": ndiff == 0}
        if torch.isfinite(A).all() and A.abs().max() > 0:
            A128 = A.to(C128)
            s128 = torch.linalg.svdvals(A128)
            nrm = s128[..., :1]
            k = min(m, n)

            def errs(Uv, Sv_, Vv):
                Uv, Vv = Uv.to(C128), Vv.to(C128)
                return (float(((Sv_.double() - s128).abs() / nrm).max()),
                        float((((Uv * Sv_.to(C128)[..., None, :]) @ Vv - A128).abs() / nrm[..., None]).max()),
                        float((Uv.mH @ Uv - torch.eye(k)).abs().max()),
                        float((Vv @ Vv.mH - torch.eye(k)).abs().max()))
            ours = errs(cplx(U), Ss, cplx(Vh))
            ref = errs(*torch.linalg.svd(A, full_matrices=False))
            row.update(err_s=ours[0], err_s_cpu32=ref[0], rec=ours[1], rec_cpu32=ref[1], orth_u=ours[2],
                       orth_u_cpu32=ref[2], orth_v=ours[3], orth_v_cpu32=ref[3])
            row["ok"] = row["ok"] and all(e <= max(1e-6, 10 * t) for e, t in zip(ours, ref))
        emit(row)
    if not time_it:
        return
    for n in (16, 22, 32):
        Ah = pairs(crandn(32, n, n, seed=n)).to("hpu")

        def cpu_route():
            U, S, Vh = torch.linalg.svd(torch.view_as_complex(Ah.cpu()), full_matrices=False)
            return torch.view_as_real(U).to("hpu"), S.to("hpu"), torch.view_as_real(Vh).to("hpu")

        f0 = fallbacks()
        row = {"op": "timing_svd", "n": n, "batch": 32, "svd_ms": timed(lambda: cl.svd(Ah)),
               "svd_cpu_ms": timed(cpu_route), "svdvals_ms": timed(lambda: cl.svdvals(Ah)),
               "svdvals_cpu_ms": timed(lambda: torch.linalg.svdvals(torch.view_as_complex(Ah.cpu())).to("hpu")),
               "ok": True}
        row["fallbacks_in_kernel_calls"] = fallbacks() - f0
        emit(row)


def run_exact(emit, time_it):
    """Exact inputs give exact results on the card: slogdet(I) = (1, 0), det(i P) = sgn(P) i^n, det of small-integer
    matrices with exact elimination; and a NaN in a later 32-column chunk of B poisons X, D and S of its member."""
    from hpu_ops.tpc import complex_linalg as cl
    for n in (1, 2, 7, 32):
        sign, lad = (t.cpu() for t in cl.slogdet(pairs(torch.eye(n, dtype=C64)[None]).to("hpu")))
        d = cl.det(pairs(torch.eye(n, dtype=C64)[None]).to("hpu")).cpu()
        ok = sign.tolist() == [[1.0, 0.0]] and lad.tolist() == [0.0] and d.tolist() == [[1.0, 0.0]]
        emit({"op": "exact", "case": f"eye_{n}", "sign": sign.tolist(), "logabsdet": lad.tolist(), "ok": ok})
    for perm in ([0], [1, 0], [1, 0, 2, 3, 4], [2, 0, 1, 5, 3, 4], list(range(31, -1, -1))):
        P = torch.eye(len(perm))[perm]
        Z = (1j * P).to(C64)[None]
        sign, lad = (t.cpu() for t in cl.slogdet(pairs(Z).to("hpu")))
        d = cl.det(pairs(Z).to("hpu")).cpu()
        ref = torch.view_as_real(torch.linalg.det(Z.to(C128))).float()
        ok = torch.equal(d, ref) and torch.equal(sign, ref) and lad.tolist() == [0.0]
        emit({"op": "exact", "case": f"i_perm_{len(perm)}", "det": d.tolist(), "ref": ref.tolist(), "ok": ok})
    M = exact_int_matrices()
    d = cl.det(pairs(M).to("hpu")).cpu()
    ref = torch.view_as_real(torch.linalg.det(M.to(C128))).round().float()
    emit({"op": "exact", "case": "int_det", "det": d.tolist(), "ref": ref.tolist(),
          "ok": torch.equal(d, ref) and torch.equal(d, torch.view_as_real(torch.linalg.det(M)))})
    A, B = crandn(3, 6, 6, seed=70), crandn(3, 6, 70, seed=71)
    B[1, 2, 50] = complex(math.nan, 0)  # in the second 32-column chunk
    X, D, S = (t.cpu() for t in cl.cgj_solve(pairs(A).to("hpu"), pairs(B).to("hpu")))
    ok = (X[1].isnan().all() and D[1].isnan().all() and S[1].isnan().all() and X[[0, 2]].isfinite().all()
          and D[[0, 2]].isfinite().all() and S[[0, 2]].isfinite().all())
    emit({"op": "exact", "case": "nan_in_later_chunk", "ok": bool(ok)})


def run_adv(emit, time_it):
    """Views, unbatched, empty and odd batches, and the same call twice (lazy: two graphs in a row), on the card
    against the same call on contiguous inputs (bit equality) or the expected shapes."""
    from hpu_ops.tpc import complex_linalg as cl
    A = crandn(5, 7, 7, seed=60)
    H = A + A.mH
    P, PH = pairs(A).to("hpu"), pairs(H).to("hpu")
    checks = {
        "inv_transposed_pairs_view": (lambda: cl.inv(pairs(A.mT).to("hpu")), lambda: cl.inv(P.transpose(1, 2))),
        "inv_strided_batch": (lambda: cl.inv(pairs(A[::2]).to("hpu")), lambda: cl.inv(P[::2])),
        "inv_unbatched": (lambda: cl.inv(P[1:2])[0], lambda: cl.inv(P[1])),
        "eigh_twice": (lambda: cl.eigh(PH)[1], lambda: (cl.eigh(PH), cl.eigh(PH))[1][1]),
        "svd_unbatched": (lambda: cl.svd(P[2:3])[1][0], lambda: cl.svd(P[2])[1]),
        "slogdet_odd_batch3": (lambda: cl.slogdet(P[:3])[1], lambda: torch.cat([cl.slogdet(P[i:i + 1])[1]
                                                                                 for i in range(3)])),
    }
    for name, (ref_fn, fn) in checks.items():
        try:
            a, b = ref_fn().cpu(), fn().cpu()
            same = bool(torch.equal(a.nan_to_num(), b.nan_to_num()))
            emit({"op": "adv", "case": name, "bit_equal": same, "ok": same})
        except Exception as e:  # noqa: BLE001
            emit({"op": "adv", "case": name, "error": str(e)[:300], "ok": False})
    try:
        shapes = [tuple(cl.inv(P[:0]).shape), tuple(cl.eigh(PH[:0])[1].shape), tuple(cl.svdvals(P[:0]).shape)]
        emit({"op": "adv", "case": "empty_batch", "shapes": shapes,
              "ok": shapes == [(0, 7, 7, 2), (0, 7, 7, 2), (0, 7)]})
    except Exception as e:  # noqa: BLE001
        emit({"op": "adv", "case": "empty_batch", "error": str(e)[:300], "ok": False})


def _linear_sum_assignment_np(cost):
    """Exact O(n^3) Hungarian minimum-sum assignment for ``(n, n)`` cost matrix (no scipy)."""
    cost = np.asarray(cost, dtype=np.float64)
    n, m = cost.shape
    assert n == m
    if n == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    u = np.zeros(n + 1, dtype=np.float64)
    v = np.zeros(n + 1, dtype=np.float64)
    p = np.zeros(n + 1, dtype=np.int64)
    way = np.zeros(n + 1, dtype=np.int64)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(n + 1, np.inf, dtype=np.float64)
        used = np.zeros(n + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = np.inf
            j1 = 0
            for j in range(1, n + 1):
                if not used[j]:
                    cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    col_ind = np.empty(n, dtype=np.int64)
    for j in range(1, n + 1):
        col_ind[p[j] - 1] = j - 1
    return np.arange(n, dtype=np.int64), col_ind


def _eig_match_err(w_cand, w_ref):
    w1 = np.asarray(w_cand, dtype=np.complex128).ravel()
    w2 = np.asarray(w_ref, dtype=np.complex128).ravel()
    if w1.size == 0:
        return 0.0
    cost = np.abs(w1[:, None] - w2[None, :])
    r, c = _linear_sum_assignment_np(cost)
    max_diff = float(cost[r, c].max())
    denom = float(np.abs(w2).max())
    return max_diff if denom == 0.0 else max_diff / max(1e-30, denom)


def _eig_res(A, w, V):
    A128 = np.asarray(A, dtype=np.complex128)
    w128 = np.asarray(w, dtype=np.complex128)
    V128 = np.asarray(V, dtype=np.complex128)
    res = float(np.linalg.norm(A128 @ V128 - V128 * w128[None, :], ord="fro"))
    nrm = float(np.linalg.norm(A128, ord="fro"))
    return res if nrm == 0.0 else res / nrm


def _eig_backward(A, w, V):
    """Largest backward errors relative to ||A||_2: sigma_min(A - w_k I) and ||A v_k - w_k v_k|| / ||v_k||."""
    A128 = np.asarray(A, dtype=np.complex128)
    w128 = np.asarray(w, dtype=np.complex128)
    V128 = np.asarray(V, dtype=np.complex128)
    eye = np.eye(A128.shape[-1])
    pseudo = max(np.linalg.svd(A128 - wk * eye, compute_uv=False)[-1] for wk in w128)
    pair = np.max(np.linalg.norm(A128 @ V128 - V128 * w128[None, :], axis=0) / np.linalg.norm(V128, axis=0))
    nrm = np.linalg.norm(A128, 2)
    return float(pseudo / nrm), float(pair / nrm)


def _eig_first_order(A, w, eps32):
    """The largest err_i / (kappa_i * n * eps32 * ||A||_2) over the eigenvalues where first-order theory applies (the
    bound under a quarter of the gap to the nearest other eigenvalue, cond(X) <= 1e12); 0.0 if there are none."""
    A128 = np.asarray(A, dtype=np.complex128)
    n = A128.shape[-1]
    lam, X = np.linalg.eig(A128)
    kappa = np.linalg.norm(X, axis=0) * np.linalg.norm(np.linalg.inv(X), axis=1)
    bound = kappa * n * eps32 * np.linalg.norm(A128, 2)
    dist = np.abs(lam[:, None] - lam[None, :]) + np.diag(np.full(n, np.inf))
    valid = (bound < 0.25 * dist.min(axis=1)) & (np.linalg.cond(X) <= 1e12)
    cost = np.abs(np.asarray(w, dtype=np.complex128)[:, None] - lam[None, :])
    r, c = _linear_sum_assignment_np(cost)
    err = np.empty(n)
    err[c] = cost[r, c]
    return float(np.max(err[valid] / bound[valid])) if valid.any() else 0.0


def eig_cases():
    yield "zero_16", torch.zeros(2, 16, 16, dtype=C64), None
    yield "eye_32", torch.eye(32, dtype=C64)[None], None
    yield "diag_8", torch.diag(torch.tensor([1, -2, 3j, -4j, 1 + 2j, -1e-3, 1e3, 0], dtype=C64))[None], None
    U = torch.triu(crandn(2, 12, 12, seed=21))
    yield "upper_tri_12", U, None
    yield "lower_tri_12", U.mT.contiguous(), None
    C8 = torch.zeros(1, 8, 8, dtype=C64)
    C8[0, 1:, :-1] = torch.eye(7, dtype=C64)
    C8[0, 0, -1] = 1.0
    yield "cyclic_8", C8, None
    C16 = torch.zeros(1, 16, 16, dtype=C64)
    C16[0, 1:, :-1] = torch.eye(15, dtype=C64)
    C16[0, :, -1] = -0.2 * crandn(16, seed=22)
    yield "companion_16", C16, None
    J4 = torch.diag(torch.full((4,), 2.0 + 1j, dtype=C128))
    for i in range(3):
        J4[i, i + 1] = 1.0
    Q4 = unitary(4, 23)
    yield "jordan_4_sim", (Q4 @ J4 @ Q4.mH).to(C64)[None], 4
    yield "near_def_2", torch.tensor([[[1.0, 1.0], [1e-6, 1.0 + 1e-6]]], dtype=C64), 2
    rep = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0 + 1e-6, 2.0, -3j, 5.0 + 2j], dtype=C128)
    Q8 = unitary(8, 24)
    yield "repeated_8", (Q8 @ torch.diag(rep) @ Q8.mH).to(C64)[None], None
    R6 = torch.zeros(1, 6, 6, dtype=C64)
    for k, th in enumerate([0.3, 1.1, 2.5]):
        R6[0, 2 * k:2 * k + 2, 2 * k:2 * k + 2] = torch.tensor(
            [[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]], dtype=C64)
    yield "real_rot_6", R6, None
    D_bad = torch.diag(2.0 ** torch.linspace(-30, 30, 8, dtype=torch.float64)).to(C128)
    M8 = crandn(8, 8, seed=25).to(C128)
    yield "bad_scale_8", (D_bad @ M8 @ torch.linalg.inv(D_bad)).to(C64)[None], None
    yield "tiny_1e-30", crandn(2, 6, 6, seed=26) * 1e-30, None
    yield "huge_1e30", crandn(2, 6, 6, seed=27) * 1e30, None
    yield "huge_3e37", crandn(2, 6, 6, seed=28) * 3e37, None
    yield "scale_eye_2p5e38", (torch.eye(4, dtype=C64) * 2.5e38)[None], None
    yield "huge_2p5e38", (unitary(6, 30) * 2.5e38).to(C64)[None], None
    for fn in (16, 24, 32):
        Fo = torch.diag(torch.ones(fn - 1, dtype=C64), 1)
        Fo[fn - 1, 0] = 0.1
        yield f"forsythe_n{fn}_eps0.1", Fo[None], None
    F16 = torch.zeros(1, 16, 16, dtype=C64)
    for i in range(16):
        for j in range(max(0, i - 1), 16):
            F16[0, i, j] = float(16 - max(i, j))
    yield "frank_n16", F16, None
    for cname, fv in (
        ("clartg_f_1p1e-19", 1.1e-19),
        ("clartg_f_1p2e-19", 1.2e-19),
        ("clartg_f_1p3e-19", 1.3e-19),
        ("clartg_f_1p4e-19", 1.4e-19),
        ("clartg_f_1p5e-19", 1.5e-19),
        ("clartg_f_1p2e-19_cpx", 1.2e-19 + 0.3e-19j),
    ):
        H = torch.tensor(
            [[[fv, 0.2, 0.3, 0.4], [1.0, 0.5, 0.1, 0.2], [0.0, 0.5, 0.7, 0.0], [0.0, 0.0, 0.5, 0.0]]],
            dtype=C64,
        )
        yield cname, H, None
    yield (
        "clartg_g_1p2e-19",
        torch.tensor(
            [[[1.0, 0.2, 0.3, 0.4], [1.2e-19, 0.5, 0.1, 0.2], [0.0, 0.5, 0.7, 0.0], [0.0, 0.0, 0.5, 0.0]]],
            dtype=C64,
        ),
        None,
    )
    for n in (1, 2, 4, 8, 16, 22, 32, 33, 48, 64):
        yield f"randn_{n}", crandn(2 if n > 32 else 3, n, n, seed=100 + n), None
    for n in (33, 48, 64):
        R_n = torch.randn(1, n, n, generator=torch.Generator().manual_seed(140 + n), dtype=torch.float32)
        yield f"rand_real_{n}", R_n.to(C64), None
    yield "zero_64", torch.zeros(1, 64, 64, dtype=C64), None
    yield "eye_64", torch.eye(64, dtype=C64)[None], None
    yield "subnormal_64", (torch.randn(1, 64, 64, generator=torch.Generator().manual_seed(31), dtype=torch.float32) * 1e-40).to(C64), None
    rep64 = torch.cat([
        torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0 + 1e-6, -3j, 5.0 + 2j], dtype=C128),
        torch.arange(10, 10 + 57, dtype=torch.float64).to(C128),
    ])
    Q64 = unitary(64, 34)
    yield "repeated_64", (Q64 @ torch.diag(rep64) @ Q64.mH).to(C64)[None], None
    J48 = torch.diag(torch.cat([torch.full((4,), 2.0 + 1j, dtype=C128), torch.arange(5, 49, dtype=torch.float64).to(C128)]))
    for i in range(3):
        J48[i, i + 1] = 1.0
    Q48 = unitary(48, 32)
    yield "jordan_4_sim_48", (Q48 @ J48 @ Q48.mH).to(C64)[None], 4
    D_bad64 = torch.diag(2.0 ** torch.linspace(-30, 30, 64, dtype=torch.float64)).to(C128)
    M64 = crandn(64, 64, seed=35).to(C128)
    yield "bad_scale_64", (D_bad64 @ M64 @ torch.linalg.inv(D_bad64)).to(C64)[None], None
    Q64b = unitary(64, 36)
    T64 = torch.diag(torch.tensor(np.geomspace(1.0, 1e-5, 64), dtype=C128))
    iu64 = torch.triu_indices(64, 64, offset=1)
    T64[iu64[0], iu64[1]] = 0.05 * crandn(iu64.shape[1], seed=37).to(C128)
    yield "illcond_1e5_64", (Q64b @ T64 @ Q64b.mH).to(C64)[None], None
    Qn64 = unitary(64, 38)
    Dn64 = torch.diag(torch.tensor(np.geomspace(1.0, 1e-5, 64), dtype=C128))
    yield "illcond_norm_1e5_64", (Qn64 @ Dn64 @ Qn64.mH).to(C64)[None], None
    Fo64 = torch.diag(torch.ones(63, dtype=C64), 1)
    Fo64[63, 0] = 0.1
    yield "forsythe_n64_eps0.1", Fo64[None], None
    NF = crandn(3, 5, 5, seed=29)
    NF[0, 2, 3] = complex(math.nan, 0.0)
    NF[2, 1, 4] = complex(0.0, math.inf)
    yield "nonfinite", NF, None
    NF48 = crandn(3, 48, 48, seed=39)
    NF48[0, 35, 10] = complex(math.nan, 0.0)
    NF48[2, 12, 41] = complex(0.0, math.inf)
    yield "nonfinite_48", NF48, None


def run_eig(emit, time_it):
    from hpu_ops.tpc import complex_linalg as cl
    eps32 = 2.0 ** -23
    for name, A, jordan_m in eig_cases():
        n = A.shape[-1]
        a = planar(A)
        with warnings.catch_warnings(record=True) as wrec:
            warnings.simplefilter("always", UserWarning)
            f0 = fallbacks()
            eig_op = cl._ceig_op() if n <= cl.COMPLEX_MAXN else cl._ceig_big_op()
            ev_op = cl._ceigvals_op() if n <= cl.COMPLEX_MAXN else cl._ceigvals_big_op()
            raw_eig = cl._call(eig_op, a.to("hpu"))
            raw_ev = cl._call(ev_op, a.to("hpu"))
            w, Vp = cl.eig(pairs(A).to("hpu"))
            wv = cl.eigvals(pairs(A).to("hpu"))
            W_raw = raw_eig[0].cpu()
            V_raw = raw_eig[1].cpu()
            Wv_raw = (raw_ev[0] if isinstance(raw_ev, (tuple, list)) else raw_ev).cpu()
            w, Vp, wv = w.cpu(), Vp.cpu(), wv.cpu()
            fb = fallbacks() - f0
        cpu_warns = sum(1 for m in wrec if "runs on the CPU" in str(m.message))

        kfn = kr.ceig_kernel if n <= cl.COMPLEX_MAXN else kr.ceig_big_kernel
        rep = [kfn(x, compute_vectors=True) for x in a.numpy()]
        repv = [kfn(x, compute_vectors=False) for x in a.numpy()]
        rW = torch.from_numpy(np.stack([r[0] for r in rep]))
        rV = torch.from_numpy(np.stack([r[1] for r in rep]))
        rWv = torch.from_numpy(np.stack([r[0] for r in repv]))
        ndiff = int(sum((x.nan_to_num() != y.nan_to_num()).sum()
                        for x, y in ((W_raw, rW), (V_raw, rV), (Wv_raw, rWv))))
        ev_parity = bool(torch.equal(w.nan_to_num(), wv.nan_to_num())
                         and torch.equal(w.isnan(), wv.isnan()))
        row = {"op": "ceig", "case": name, "n": n, "bit_equal": ndiff == 0, "n_diff": ndiff,
               "eigvals_parity": ev_parity, "fallbacks": fb, "cpu_route_warns": cpu_warns,
               "ok": ndiff == 0 and ev_parity and fb == 0 and cpu_warns == 0}

        if name.startswith("subnormal_"):
            w_c = cplx(w).numpy()
            V_c = cplx(Vp).numpy()
            ftz_ok = bool(np.all(w_c == 0) and np.array_equal(V_c[0], np.eye(n, dtype=np.complex64)))
            row.update(ftz_ok=ftz_ok)
            row["ok"] = row["ok"] and ftz_ok
        elif torch.isfinite(A).all():
            w_c = cplx(w).numpy()
            V_c = cplx(Vp).numpy()
            w32_t, _ = torch.linalg.eig(A)
            w128_t = torch.linalg.eigvals(A.to(C128))
            max_res, max_vnorm_err, max_ew, max_ew32 = 0.0, 0.0, 0.0, 0.0
            for b in range(A.shape[0]):
                a_np = A[b].numpy()
                max_res = max(max_res, _eig_res(a_np, w_c[b], V_c[b]))
                v_norms = np.linalg.norm(V_c[b], axis=0)
                max_vnorm_err = max(max_vnorm_err, float(np.max(np.abs(v_norms - 1.0))))
                w128_np = w128_t[b].numpy()
                max_ew = max(max_ew, _eig_match_err(w_c[b], w128_np))
                max_ew32 = max(max_ew32, _eig_match_err(w32_t[b].numpy(), w128_np))
            if jordan_m:
                ew_tol = 2.0 * (eps32 ** (1.0 / jordan_m))
            elif name.startswith("lower_tri"):
                ew_tol = max(1e-5, 100.0 * n * eps32, 20.0 * max_ew32)
            elif name.startswith("frank"):
                ew_tol = 0.02
            elif name.startswith("illcond_1e5_"):
                ew_tol = math.inf  # non-normal: only the first-order test below bounds it
            else:
                ew_tol = max(1e-5, 20.0 * max_ew32)
            row.update(res=max_res, vnorm_err=max_vnorm_err, err_w=max_ew, err_w_cpu32=max_ew32, ew_tol=ew_tol)
            row["ok"] = row["ok"] and (max_res < 50 * n * eps32) and (max_vnorm_err < 1e-5) and (max_ew <= ew_tol)
            if n > cl.COMPLEX_MAXN and A.abs().max() > 0:
                # ceig_big: backward errors at most n * eps32 * ||A||_2 (eigenvalues and pairs), and each eigenvalue
                # where first-order theory applies within kappa_i times that of the exact one.
                bw = [_eig_backward(A[b].numpy(), w_c[b], V_c[b]) for b in range(A.shape[0])]
                pseudo, pair = max(x[0] for x in bw), max(x[1] for x in bw)
                fo = max(_eig_first_order(A[b].numpy(), w_c[b], eps32) for b in range(A.shape[0]))
                row.update(pseudo_n_eps=pseudo / (n * eps32), pair_n_eps=pair / (n * eps32), first_order=fo)
                row["ok"] = row["ok"] and pseudo <= n * eps32 and pair <= n * eps32 and fo <= 1.0
        else:
            nan_ok = bool(w[0].isnan().all() and Vp[0].isnan().all() and wv[0].isnan().all()
                          and w[2].isnan().all() and Vp[2].isnan().all() and wv[2].isnan().all()
                          and w[1].isfinite().all() and Vp[1].isfinite().all() and wv[1].isfinite().all())
            res1 = _eig_res(A[1].numpy(), cplx(w[1]).numpy(), cplx(Vp[1]).numpy())
            vnorm1 = float(np.max(np.abs(np.linalg.norm(cplx(Vp[1]).numpy(), axis=0) - 1.0)))
            row.update(nan_ok=nan_ok, clean_res=res1, clean_vnorm_err=vnorm1)
            row["ok"] = row["ok"] and nan_ok and (res1 < 50 * n * eps32) and (vnorm1 < 1e-5)
        emit(row)

    # Layouts, non-contiguous views, unbatched, multi-dim & empty batch, lazy repeat
    def _pairs_eq(a, b):
        return torch.equal(a[0].cpu(), b[0].cpu()) and torch.equal(a[1].cpu(), b[1].cpu())

    def _raises_value_error(fn, *args):
        try:
            fn(*args)
        except ValueError:
            return True
        return False

    A4 = crandn(4, 6, 6, seed=60)
    P4 = pairs(A4).to("hpu")
    with warnings.catch_warnings(record=True) as wrec:
        warnings.simplefilter("always", UserWarning)
        f0 = fallbacks()
        w_p, V_p = cl.eig(P4)
        w_i, V_i = cl.eig(P4.reshape(4, 6, 12), layout="interleaved")
        wv_i = cl.eigvals(P4.reshape(4, 6, 12), layout="interleaved")
        u_eig = cl.eig(P4[1])
        m_eig = cl.eig(P4.reshape(2, 2, 6, 6, 2))
        view_checks = {
            "interleaved": (
                torch.equal(w_i.cpu(), w_p.cpu().reshape(4, 12))
                and torch.equal(V_i.cpu(), V_p.cpu().reshape(4, 6, 12))
                and torch.equal(wv_i.cpu(), w_i.cpu())
            ),
            "strided_batch": _pairs_eq(cl.eig(P4[::2]), cl.eig(P4[::2].contiguous())),
            "mT_view": _pairs_eq(cl.eig(P4.transpose(1, 2)), cl.eig(pairs(A4.mT).to("hpu"))),
            "conj_view": _pairs_eq(
                cl.eig(torch.stack([P4[..., 0], -P4[..., 1]], dim=-1)),
                cl.eig(pairs(A4.conj().resolve_conj()).to("hpu")),
            ),
            "unbatched": torch.equal(u_eig[0].cpu(), w_p[1].cpu()) and torch.equal(u_eig[1].cpu(), V_p[1].cpu()),
            "eig_twice": _pairs_eq((w_p, V_p), (cl.eig(P4), cl.eig(P4))[1]),
            "empty_batch": (
                tuple(cl.eig(P4[:0])[0].shape) == (0, 6, 2)
                and tuple(cl.eig(P4[:0])[1].shape) == (0, 6, 6, 2)
                and tuple(cl.eigvals(P4[:0]).shape) == (0, 6, 2)
            ),
            "multidim_batch": tuple(m_eig[0].shape) == (2, 2, 6, 2) and tuple(m_eig[1].shape) == (2, 2, 6, 6, 2),
            "invalid_1d_c64": (
                _raises_value_error(cl.eig, torch.zeros(4, dtype=C64))
                and _raises_value_error(cl.eigvals, torch.zeros(4, dtype=C64))
            ),
            "invalid_n65": (
                _raises_value_error(cl.eig, torch.zeros(1, 65, 65, 2))
                and _raises_value_error(cl.eigvals, torch.zeros(1, 65, 65, 2))
            ),
        }
        fb_adv = fallbacks() - f0
    cpu_warns_adv = sum(1 for m in wrec if "runs on the CPU" in str(m.message))

    A48 = crandn(4, 48, 48, seed=61)
    P48 = pairs(A48).to("hpu")
    with warnings.catch_warnings(record=True) as wrec48:
        warnings.simplefilter("always", UserWarning)
        f0_48 = fallbacks()
        w48, V48 = cl.eig(P48)
        u48 = cl.eig(P48[1])
        view_checks["eig_big_twice"] = _pairs_eq((w48, V48), (cl.eig(P48), cl.eig(P48))[1])
        view_checks["strided_batch_48"] = _pairs_eq(cl.eig(P48[::2]), cl.eig(P48[::2].contiguous()))
        view_checks["unbatched_48"] = torch.equal(u48[0].cpu(), w48[1].cpu()) and torch.equal(u48[1].cpu(), V48[1].cpu())
        view_checks["empty_batch_48"] = (
            tuple(cl.eig(P48[:0])[0].shape) == (0, 48, 2)
            and tuple(cl.eig(P48[:0])[1].shape) == (0, 48, 48, 2)
            and tuple(cl.eigvals(P48[:0]).shape) == (0, 48, 2)
        )
        fb_adv += fallbacks() - f0_48
    cpu_warns_adv += sum(1 for m in wrec48 if "runs on the CPU" in str(m.message))
    for cname, cok in view_checks.items():
        emit({"op": "ceig_adv", "case": cname, "fallbacks": fb_adv, "cpu_route_warns": cpu_warns_adv,
              "ok": bool(cok) and fb_adv == 0 and cpu_warns_adv == 0})

    if not time_it:
        return
    for n in (16, 22, 32, 48, 64):
        Ah = pairs(crandn(32, n, n, seed=30 + n)).to("hpu")

        def cpu_route_eig(t=Ah):
            w_, V_ = torch.linalg.eig(torch.view_as_complex(t.cpu()))
            return torch.view_as_real(w_).to("hpu"), torch.view_as_real(V_).to("hpu")

        def cpu_route_eigvals(t=Ah):
            return torch.view_as_real(torch.linalg.eigvals(torch.view_as_complex(t.cpu()))).to("hpu")

        with warnings.catch_warnings(record=True) as wrec:
            warnings.simplefilter("always", UserWarning)
            f0 = fallbacks()
            eig_ms = timed(lambda t=Ah: cl.eig(t))
            eigvals_ms = timed(lambda t=Ah: cl.eigvals(t))
            fb_t = fallbacks() - f0
        cw_t = sum(1 for m in wrec if "runs on the CPU" in str(m.message))
        row = {"op": "timing_eig", "n": n, "batch": 32,
               "eig_ms": eig_ms, "eig_cpu_ms": timed(cpu_route_eig),
               "eigvals_ms": eigvals_ms, "eigvals_cpu_ms": timed(cpu_route_eigvals),
               "fallbacks_in_kernel_calls": fb_t, "cpu_route_warns": cw_t,
               "ok": fb_t == 0 and cw_t == 0}
        emit(row)


def main():
    from hpu_ops import tpc
    tpc.install_kernel_path()
    import habana_frameworks.torch.core  # noqa: F401
    assert tpc.ready(), "tpc kernels not ready"
    which = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else "all"
    time_it = "--time" in sys.argv
    ok = True

    def emit(row):
        nonlocal ok
        ok &= bool(row["ok"])
        print("RESULT " + json.dumps(row), flush=True)

    if which in ("gj", "all"):
        run_gj(emit, time_it)
        run_exact(emit, time_it)
    if which in ("eigh", "all"):
        run_eigh(emit, time_it)
    if which in ("svd", "all"):
        run_svd(emit, time_it)
    if which in ("eig", "all"):
        run_eig(emit, time_it)
    if which in ("adv", "all"):
        run_adv(emit, time_it)
    print("SUMMARY " + ("ok" if ok else "MISMATCH"), flush=True)


if __name__ == "__main__":
    main()
