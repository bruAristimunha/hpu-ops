"""On a Gaudi card: the kernels changed for the Codex review's extreme-range findings against their float32 replicas
(``kernel_replica``), on the review's inputs and on ordinary ones: matrix_exp (#8 reciprocal, #10 finiteness),
householder_product (#16 zero tau) and lu_factor (#5 flushed multiplier).
    PT_HPU_LAZY_MODE=0|1 python3 tests/check_review_replica.py
One ``RESULT {json}`` per case: bit equality with the replica (matrix_exp, householder_product; lu_factor's
``v_reciprocal_f32`` is modelled as correctly rounded, so its rows report the largest difference) and the error
against torch on the CPU in float64. Last line ``SUMMARY ok`` or ``MISMATCH``."""
import json
import warnings

import numpy as np
import torch

import kernel_replica as kr

warnings.filterwarnings("ignore")


def rel(x, ref):
    x, ref = torch.as_tensor(x).double(), torch.as_tensor(ref).double()
    if not torch.isfinite(x[torch.isfinite(ref)]).all():
        return float("nan")
    return float((x - ref).abs().max() / ref.abs().max().clamp_min(1e-300))


def expm_cases():
    g = torch.Generator().manual_seed(0)
    yield "codex8_diag_-1e8_0", torch.diag(torch.tensor([-1e8, 0.0]))[None]
    yield "codex8_diag_-1e10_0", torch.diag(torch.tensor([-1e10, 0.0]))[None]
    yield "codex10_-flt_max", torch.tensor([[[-torch.finfo(torch.float32).max]]])
    yield "randn_22", torch.randn(4, 22, 22, generator=g)
    yield "randn_6_x50", 50 * torch.randn(4, 6, 6, generator=g)


def hh_cases():
    g = torch.Generator().manual_seed(1)
    A = torch.zeros(4, 2)
    A[2:4, 0] = 3e38
    A[2:4, 1] = 1
    yield "codex16_zero_tau", A[None], torch.tensor([[0.0, 2 / 3]])
    yield "randn_22x8", torch.randn(3, 22, 8, generator=g), torch.rand(3, 8, generator=g) + 0.5
    t = torch.rand(3, 8, generator=g) + 0.5
    t[:, 2] = 0
    yield "randn_22x8_one_zero_tau", torch.randn(3, 22, 8, generator=g), t


def lu_cases():
    g = torch.Generator().manual_seed(2)
    yield "codex5_flushed_multiplier", torch.tensor([[[2.0 ** 100, 2.0 ** 100], [2.0 ** -30, 2.0 ** -30]]])
    yield "randn_22", torch.randn(4, 22, 22, generator=g)
    yield "tiny_rows_32", torch.randn(2, 32, 32, generator=g) * torch.logspace(-30, 30, 32)[:, None]


def main():
    from hpu_ops import tpc

    tpc.install_kernel_path()
    import habana_frameworks.torch.core as htcore

    assert tpc.ready()
    ok = True

    def emit(row):
        nonlocal ok
        ok &= row["ok"]
        print("RESULT " + json.dumps(row), flush=True)

    for name, A in expm_cases():
        got = tpc.matrix_exp(A.to("hpu")).cpu()
        htcore.mark_step()
        rep = torch.from_numpy(np.stack([kr.matrix_exp(a.numpy()) for a in A]))
        same = bool(torch.equal(got.nan_to_num(), rep.nan_to_num()))
        emit({"op": "matrix_exp", "case": name, "bit_equal": same, "err": rel(got, torch.linalg.matrix_exp(A.double())),
              "err_cpu32": rel(torch.linalg.matrix_exp(A), torch.linalg.matrix_exp(A.double())), "ok": same})
    for name, A, tau in hh_cases():
        got = tpc.householder_product(A.to("hpu"), tau.to("hpu")).cpu()
        htcore.mark_step()
        rep = torch.from_numpy(np.stack([kr.householder_product(a.numpy(), t.numpy()) for a, t in zip(A, tau)]))
        same = bool(torch.equal(got.nan_to_num(), rep.nan_to_num()))
        ref = torch.linalg.householder_product(A.double(), tau.double())
        emit({"op": "householder_product", "case": name, "bit_equal": same, "err": rel(got, ref),
              "finite": bool(torch.isfinite(got).all()), "ok": same and bool(torch.isfinite(got).all())})
    for name, A in lu_cases():
        LU, piv, _, info = tpc.lu_factor(A.to("hpu"))
        LU, piv, info = LU.cpu(), piv.cpu(), info.cpu()
        htcore.mark_step()
        reps = [kr.lu_factor(a.numpy()) for a in A]
        rep_lu = torch.from_numpy(np.stack([r[0] for r in reps]))
        LU_t, piv_t, info_t = torch.linalg.lu_factor_ex(A)
        row = {"op": "lu_factor", "case": name, "pivots_equal_replica": bool(torch.equal(
            piv.long(), torch.from_numpy(np.stack([r[1] for r in reps])).long())),
               "card_vs_replica": rel(LU, rep_lu), "info": info.tolist(), "info_torch": info_t.tolist(),
               "err": rel(LU, LU_t)}
        row["ok"] = row["pivots_equal_replica"] and row["card_vs_replica"] < 1e-5 and row["info"] == row["info_torch"]
        emit(row)
    print("SUMMARY", "ok" if ok else "MISMATCH", flush=True)


if __name__ == "__main__":
    main()
