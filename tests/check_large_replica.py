"""On a Gaudi card: the raw ``gj_large`` kernel against its float32 replica (``kernel_replica.gj_large``), and the
inverse / det through ``tpc.gj_inv`` against float64, on random inputs and the review's matrices.
    PT_HPU_LAZY_MODE=0|1 python3 tests/check_large_replica.py"""
import json
import warnings

from hpu_ops import tpc

tpc.install_kernel_path()
import habana_frameworks.torch.core as htcore  # noqa: E402
import torch  # noqa: E402

import kernel_replica as kr  # noqa: E402

warnings.filterwarnings("ignore")


def cases():
    g = torch.Generator().manual_seed(0)
    for n in (65, 70, 130, 220, 300):
        yield f"randn_n{n}", torch.randn(2, n, n, generator=g)
    for n in (70, 220):
        M = torch.eye(n) + 0.05 * torch.randn(2, n, n, generator=g) / n ** 0.5
        perm = torch.stack([torch.randperm(n, generator=g) for _ in range(2)])
        yield f"permuted_n{n}", M.gather(1, perm[..., None].expand(2, n, n))
    A = torch.eye(70)
    A[0, 0], A[0, 1], A[1, 0], A[1, 1] = 1e30, 1e-9, 1e30, 2e-9
    yield "review_tiny_multiplier", A[None]
    A = torch.eye(70)
    A[0, :2] = 1e38
    A[1, :2] = torch.tensor([1.0, 2.0])
    yield "review_huge_pivot", A[None]
    A = torch.randn(1, 70, 70, generator=g)
    A[0, 69, 69] = float("inf")
    yield "inf_69_69", A


def main():
    assert tpc.ready()
    ok = True
    for name, A in cases():
        n = A.shape[-1]
        got = [t.cpu() for t in torch.ops.linalg_tpc.gj_large(A.to("hpu").contiguous())]
        htcore.mark_step()
        want = kr.gj_large_op()(A)
        same = [bool(torch.equal(g.nan_to_num(), w.nan_to_num())) for g, w in zip(got, want)]
        diff = float((got[0] - want[0]).nan_to_num().abs().max())
        perm = sorted(got[1][0].long().tolist()) == list(range(n))
        row = {"case": name, "n": n, "bit_equal_R_P_Q_D_S": same, "R_maxdiff": diff, "P_is_permutation": perm}
        if torch.isfinite(A).all():
            X, D, S = tpc.gj_inv(A.to("hpu"))
            ref = torch.linalg.inv(A.double())
            row["inv_err"] = float((X.cpu().double() - ref).abs().max() / ref.abs().max())
            row["inv_err_cpu32"] = float((torch.linalg.inv(A).double() - ref).abs().max() / ref.abs().max())
            det, det_ref = S[..., 0].cpu().double() * D.cpu().double().prod(-1), torch.linalg.det(A.double())
            row["det_rel_err"] = float(((det - det_ref).abs() / det_ref.abs()).max())
            row["info"] = S[..., 1].cpu().tolist()
        row["ok"] = all(same) and perm
        ok &= row["ok"]
        print("RESULT " + json.dumps(row), flush=True)
    print("SUMMARY", "ok" if ok else "MISMATCH", flush=True)


if __name__ == "__main__":
    main()
