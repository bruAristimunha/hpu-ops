"""On a Gaudi card: complex64 Householder QR TPC kernels (``cqr``, ``cqr_complete``, ``cqr_r``)
against ``kernel_replica.cqr_kernel`` bit for bit, accuracy against ``complex128``, CPU fallback counts,
and zero CPU-route warnings.

    PT_HPU_LAZY_MODE=0|1 python3 tests/check_cqr_hpu.py [--time]

One ``RESULT {json}`` per case; the last line is ``SUMMARY ok`` or ``SUMMARY MISMATCH``.
"""
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


def pairs(Z):
    return torch.view_as_real(Z.to(C64)).contiguous()


def cplx(P):
    return torch.view_as_complex(P.double().contiguous())


def planar_t(Z):
    P = pairs(Z)
    b, m, n = P.shape[0], P.shape[1], P.shape[2]
    return P.permute(0, 3, 2, 1).reshape(b, 2 * n, m).contiguous()


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


def qr_errs(A, Q, R):
    A128, Q128, R128 = A.to(C128), Q.to(C128), R.to(C128)
    k = Q128.shape[-1]
    eye = torch.eye(k, dtype=C128)
    nrm = A128.abs().pow(2).sum((-2, -1)).sqrt()
    rec_abs = (Q128 @ R128 - A128).abs().pow(2).sum((-2, -1)).sqrt()
    rec = torch.where(nrm > 0, rec_abs / nrm.clamp_min(1e-300), rec_abs)
    orth = (Q128.mH @ Q128 - eye).abs().pow(2).sum((-2, -1)).sqrt()
    return float(rec.max()), float(orth.max())


def qr_cases():
    # All 7 habana_qr_nan.py cases in complex64 + complex variants
    yield "nan_repro_eye2", torch.eye(2, dtype=C64).expand(3, 2, 2).contiguous()
    yield "nan_repro_eye8", torch.eye(8, dtype=C64).expand(3, 8, 8).contiguous()
    yield "nan_repro_diag23", torch.diag(torch.tensor([2.0, 3.0], dtype=C64)).expand(3, 2, 2).contiguous()
    yield "nan_repro_triu6", (crandn(3, 6, 6, seed=10).triu() + 3 * torch.eye(6, dtype=C64))
    yield "nan_repro_col0_e0", torch.cat(
        [torch.tensor([[[2.0], [0.0], [0.0], [0.0]]], dtype=C64).expand(3, 4, 1), crandn(3, 4, 3, seed=11)],
        dim=-1,
    )
    yield "nan_repro_ones4", torch.ones(3, 4, 4, dtype=C64)
    yield "nan_repro_randn6", crandn(3, 6, 6, seed=12)

    yield "cpx_diag4", torch.diag(torch.tensor([2.0, -3.0, 4.0j, 0.0], dtype=C64)).expand(2, 4, 4).contiguous()
    yield "cpx_1x1", torch.tensor([[[3.0 + 4.0j]], [[-5.0 + 0.0j]]], dtype=C64)
    yield "cpx_rank1", torch.full((2, 5, 4), 1.0 + 2.0j, dtype=C64)

    for n in (1, 2, 4, 8, 16, 22, 32):
        yield f"sq_{n}", crandn(2, n, n, seed=100 + n)
    for m, n in ((4, 2), (8, 3), (16, 5), (32, 17), (32, 31)):
        yield f"tall_{m}x{n}", crandn(2, m, n, seed=200 + 37 * m + n)

    yield "zeros_6x4", torch.zeros(2, 6, 4, dtype=C64)
    yield "subnormals_5x3", torch.full((2, 5, 3), 1e-40 + 1e-40j, dtype=C64)
    yield "tiny_1e-35", crandn(2, 6, 4, seed=301) * 1e-35
    yield "huge_1e35", crandn(2, 6, 4, seed=302) * 1e35
    Q6, _ = torch.linalg.qr(crandn(2, 6, 4, seed=303).to(C128))
    yield "huge_2p5e38", (Q6 * 2.5e38).to(C64)

    A_nf = crandn(3, 6, 4, seed=304)
    A_nf[0, 2, 1] = complex(math.nan, 0.0)
    A_nf[2, 4, 3] = complex(0.0, math.inf)
    yield "nonfinite_mixed", A_nf


def run_cqr(emit, time_it):
    from hpu_ops.tpc import complex_linalg as cl

    for name, A in qr_cases():
        m, n = A.shape[-2], A.shape[-1]
        a_t_cpu = planar_t(A)
        a_t_hpu = a_t_cpu.to("hpu")
        P_hpu = pairs(A).to("hpu")
        is_fin = bool(torch.isfinite(A).all())

        for mode in ("reduced", "complete", "r"):
            with warnings.catch_warnings(record=True) as wrec:
                warnings.simplefilter("always", UserWarning)
                f0 = fallbacks()
                if mode == "r":
                    R_raw = cl._call(cl._cqr_r_op(), a_t_hpu).cpu()
                    Qp, Rp = cl.qr(P_hpu, mode="r")
                    Q_raw = None
                    raw_tensors = (R_raw,)
                    rep_list = [kr.cqr_kernel(x, mode="r") for x in a_t_cpu.numpy()]
                else:
                    op = cl._cqr_complete_op() if mode == "complete" else cl._cqr_op()
                    Q_raw, R_raw = (t.cpu() for t in cl._call(op, a_t_hpu))
                    Qp, Rp = cl.qr(P_hpu, mode=mode)
                    raw_tensors = (Q_raw, R_raw)
                    rep_list = [kr.cqr_kernel(x, mode=mode) for x in a_t_cpu.numpy()]
                fb = fallbacks() - f0
            cw = sum(1 for msg in wrec if "runs on the CPU" in str(msg.message))
            ndiff = 0
            for idx, got in enumerate(raw_tensors):
                rep = torch.from_numpy(np.stack([r[idx] for r in rep_list]))
                ndiff += int((got.nan_to_num() != rep.nan_to_num()).sum())
                ndiff += int((got.isnan() != rep.isnan()).sum())

            Qp_cpu = Qp.cpu()
            Rp_cpu = Rp.cpu()
            row = {
                "op": f"cqr_{mode}",
                "case": name,
                "m": m,
                "n": n,
                "bit_equal": ndiff == 0,
                "n_diff": ndiff,
                "fallbacks": fb,
                "cpu_route_warns": cw,
                "ok": (ndiff == 0 and fb == 0 and cw == 0),
            }

            if is_fin:
                R = cplx(Rp_cpu)
                fin_ok = bool(torch.isfinite(R).all())
                tril_ok = bool(torch.equal(R.tril(-1), torch.zeros_like(R)))
                d = R[..., :n, :n].diagonal(dim1=-2, dim2=-1)
                diag_real_ok = bool(torch.equal(d.imag, torch.zeros_like(d.imag)))
                row.update(finite=fin_ok, tril_zero=tril_ok, diag_real=diag_real_ok)
                row["ok"] = row["ok"] and fin_ok and tril_ok and diag_real_ok

                if mode == "r":
                    row["ok"] = row["ok"] and (Qp_cpu.shape == (0,))
                elif name.startswith("subnormals"):
                    k_q = m if mode == "complete" else n
                    Q = cplx(Qp_cpu)
                    ftz_ok = bool(
                        torch.equal(Q, torch.eye(m, k_q, dtype=C128).expand(*A.shape[:-2], m, k_q))
                        and torch.equal(R, torch.zeros(*A.shape[:-2], k_q, n, dtype=C128))
                    )
                    row["ftz_ok"] = ftz_ok
                    row["ok"] = row["ok"] and ftz_ok
                else:
                    Q = cplx(Qp_cpu)
                    rec, orth = qr_errs(A, Q, R)
                    Q32, R32 = torch.linalg.qr(A, mode=mode)
                    rec32, orth32 = qr_errs(A, Q32, R32)
                    tol_rec = 1e-5 if math.isnan(rec32) else max(1e-6, 10.0 * rec32)
                    tol_orth = 1e-5 if math.isnan(orth32) else max(1e-6, 10.0 * orth32)
                    row.update(rec=rec, rec_cpu32=rec32, orth=orth, orth_cpu32=orth32)
                    row["ok"] = row["ok"] and bool(torch.isfinite(Q).all()) and (rec <= tol_rec) and (orth <= tol_orth)
                    if mode == "complete":
                        row["ok"] = row["ok"] and bool(torch.equal(Rp_cpu[..., n:, :, :], torch.zeros_like(Rp_cpu[..., n:, :, :])))
            else:
                R = cplx(Rp_cpu)
                nan_ok = bool(R[0].isnan().all() and R[2].isnan().all() and R[1].isfinite().all())
                if mode != "r":
                    Q = cplx(Qp_cpu)
                    nan_ok = nan_ok and bool(Q[0].isnan().all() and Q[2].isnan().all() and Q[1].isfinite().all())
                    rec1, orth1 = qr_errs(A[1:2], Q[1:2], R[1:2])
                    nan_ok = nan_ok and (rec1 < 1e-5) and (orth1 < 1e-5)
                row["nan_ok"] = nan_ok
                row["ok"] = row["ok"] and nan_ok

            emit(row)

    # Layout / view / empty-batch adversarial checks
    A4 = crandn(4, 7, 5, seed=401)
    P4 = pairs(A4).to("hpu")
    with warnings.catch_warnings(record=True) as wrec:
        warnings.simplefilter("always", UserWarning)
        f0 = fallbacks()
        Qp, Rp = cl.qr(P4, mode="reduced")
        Qi, Ri = cl.qr(P4.reshape(4, 7, 10), mode="reduced", layout="interleaved")
        Qcp, Rcp = cl.qr(P4, mode="complete")
        Qci, Rci = cl.qr(P4.reshape(4, 7, 10), mode="complete", layout="interleaved")
        Qrp, Rrp = cl.qr(P4, mode="r")
        Qri, Rri = cl.qr(P4.reshape(4, 7, 10), mode="r", layout="interleaved")
        Qu, Ru = cl.qr(P4[1])
        Qm, Rm = cl.qr(P4.reshape(2, 2, 7, 5, 2))
        Qs, Rs = cl.qr(P4[::2])
        Qsc, Rsc = cl.qr(P4[::2].contiguous())
        Qe, Re = cl.qr(P4[:0], mode="complete")
        Qre, Rre = cl.qr(P4[:0], mode="r")
        fb_adv = fallbacks() - f0
    cw_adv = sum(1 for msg in wrec if "runs on the CPU" in str(msg.message))
    adv_checks = {
        "interleaved_reduced": torch.equal(Qi.cpu(), Qp.cpu().reshape(4, 7, 10)) and torch.equal(Ri.cpu(), Rp.cpu().reshape(4, 5, 10)),
        "interleaved_complete": torch.equal(Qci.cpu(), Qcp.cpu().reshape(4, 7, 14)) and torch.equal(Rci.cpu(), Rcp.cpu().reshape(4, 7, 10)),
        "interleaved_r": Qrp.shape == (0,) and Qri.shape == (0,) and torch.equal(Rri.cpu(), Ri.cpu()) and torch.equal(Rrp.cpu(), Rp.cpu()),
        "unbatched": torch.equal(Qu.cpu(), Qp[1].cpu()) and torch.equal(Ru.cpu(), Rp[1].cpu()),
        "multidim_batch": tuple(Qm.shape) == (2, 2, 7, 5, 2) and tuple(Rm.shape) == (2, 2, 5, 5, 2),
        "strided_batch": torch.equal(Qs.cpu(), Qsc.cpu()) and torch.equal(Rs.cpu(), Rsc.cpu()),
        "empty_batch": tuple(Qe.shape) == (0, 7, 7, 2) and tuple(Re.shape) == (0, 7, 5, 2) and Qre.shape == (0,) and tuple(Rre.shape) == (0, 5, 5, 2),
    }
    for cname, cok in adv_checks.items():
        emit({"op": "cqr_adv", "case": cname, "fallbacks": fb_adv, "cpu_route_warns": cw_adv,
              "ok": bool(cok) and fb_adv == 0 and cw_adv == 0})

    if not time_it:
        return
    for m, n in ((16, 16), (32, 32), (32, 16)):
        Ah = pairs(crandn(32, m, n, seed=500 + n)).to("hpu")

        def cpu_route(t=Ah):
            Q_, R_ = torch.linalg.qr(torch.view_as_complex(t.cpu()))
            return torch.view_as_real(Q_).to("hpu"), torch.view_as_real(R_).to("hpu")

        f0 = fallbacks()
        cqr_ms = timed(lambda t=Ah: cl.qr(t))
        fb_t = fallbacks() - f0
        emit({"op": "timing_cqr", "m": m, "n": n, "batch": 32, "cqr_ms": cqr_ms,
              "cpu_ms": timed(cpu_route), "fallbacks": fb_t, "ok": fb_t == 0})


def main():
    from hpu_ops import tpc
    tpc.install_kernel_path()
    import habana_frameworks.torch.core  # noqa: F401
    assert tpc.ready(), "tpc kernels not ready"
    time_it = "--time" in sys.argv
    ok = True

    def emit(row):
        nonlocal ok
        ok &= bool(row["ok"])
        print("RESULT " + json.dumps(row), flush=True)

    run_cqr(emit, time_it)
    print("SUMMARY " + ("ok" if ok else "MISMATCH"), flush=True)


if __name__ == "__main__":
    main()
