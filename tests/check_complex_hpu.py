"""On a Gaudi card: the complex64 TPC kernels of ``hpu_ops.tpc.complex_linalg`` against their float32 replicas
(``kernel_replica``) bit for bit, their error against complex128 next to torch's complex64 CPU error, CPU fallback
counts, and the per-call time against torch's CPU complex route (copy to the host, the complex64 op, copy back).
    PT_HPU_LAZY_MODE=0|1 python3 tests/check_complex_hpu.py [gj|eigh|svd|all] [--time]
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
    if which in ("eigh", "all"):
        run_eigh(emit, time_it)
    if which in ("svd", "all"):
        run_svd(emit, time_it)
    if which in ("adv", "all"):
        run_adv(emit, time_it)
    print("SUMMARY " + ("ok" if ok else "MISMATCH"), flush=True)


if __name__ == "__main__":
    main()
