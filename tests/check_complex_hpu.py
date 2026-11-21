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
    if which in ("eigh", "all") and hasattr(sys.modules[__name__], "run_eigh"):
        run_eigh(emit, time_it)  # noqa: F821
    if which in ("svd", "all") and hasattr(sys.modules[__name__], "run_svd"):
        run_svd(emit, time_it)  # noqa: F821
    print("SUMMARY " + ("ok" if ok else "MISMATCH"), flush=True)


if __name__ == "__main__":
    main()
