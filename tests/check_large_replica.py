"""On a Gaudi card: the raw ``gj_large`` kernel against its float32 replica (``kernel_replica.gj_large``), and the
inverse / det through ``tpc.gj_inv`` against float64, on random inputs and the review's matrices.
    PT_HPU_LAZY_MODE=0|1 python3 tests/check_large_replica.py
A case is ok when the raw outputs are bit-equal to the replica's, ``P`` is a permutation and, for a finite input,
the inverse and det are within ``max(FLOOR, 10x)`` CPU float32's error against float64 (Codex review #20: the
verdict once ignored the inverse and det, so a wrong inverse passed)."""
import json
import warnings

import torch

import kernel_replica as kr

warnings.filterwarnings("ignore")
FLOOR = 1e-5


def wilkinson(n):
    W = torch.eye(n) - torch.tril(torch.ones(n, n), -1)
    W[:, -1] = 1
    return W / 2


def cases():
    g = torch.Generator().manual_seed(0)
    for n in (65, 70, 130, 220, 300, 512):
        yield f"randn_n{n}", torch.randn(2, n, n, generator=g)
    for n in (70, 220):
        M = torch.eye(n) + 0.05 * torch.randn(2, n, n, generator=g) / n ** 0.5
        perm = torch.stack([torch.randperm(n, generator=g) for _ in range(2)])
        yield f"permuted_n{n}", M.gather(1, perm[..., None].expand(2, n, n))
    A = torch.eye(70)
    A[0, 0], A[0, 1], A[1, 0], A[1, 1] = 1e30, 1e-9, 1e30, 2e-9
    yield "review_tiny_multiplier", A[None]
    yield "review_tiny_multiplier_T", A.mT.contiguous()[None]
    A = torch.eye(70)
    A[0, :2] = 1e38
    A[1, :2] = torch.tensor([1.0, 2.0])
    yield "review_huge_pivot", A[None]
    yield "review_huge_pivot_T", A.mT.contiguous()[None]
    yield "codex1_wilkinson_n130", wilkinson(130)[None]
    yield "codex1_wilkinson_T_n130", wilkinson(130).mT.contiguous()[None]
    A = torch.eye(65)
    A[0, 0], A[0, 1], A[1, 1] = 2.0 ** 70, 2.0 ** 70, 2.0 ** -70
    yield "codex17_used_row", A[None]
    yield "codex17_used_row_T", A.mT.contiguous()[None]
    A = torch.randn(1, 70, 70, generator=g)
    A[0, 69, 69] = float("inf")
    yield "inf_69_69", A


def rel_err(x, ref):
    """max |x - ref| / max |ref|, NaN when x is not finite where ref is."""
    x = x.double()
    if not torch.isfinite(x[torch.isfinite(ref)]).all():
        return float("nan")
    return float((x - ref).abs().max() / ref.abs().max())


def accuracy(A, X, D, S):
    """The inverse's and det's errors (``X``, ``D``, ``S`` from ``tpc.gj_inv``, on the CPU) and CPU float32's."""
    ref = torch.linalg.inv(A.double())
    det_ref = torch.linalg.det(A.double())
    det = S[..., 0].double() * D.double().prod(-1)
    return {"inv_err": rel_err(X, ref), "inv_err_cpu32": rel_err(torch.linalg.inv(A), ref),
            "det_rel_err": float(((det - det_ref).abs() / det_ref.abs()).max()),
            "det_rel_err_cpu32": float(((torch.linalg.det(A).double() - det_ref).abs() / det_ref.abs()).max()),
            "info": S[..., 1].tolist()}


def verdict(row):
    """Whether a case passed: raw outputs bit-equal, ``P`` a permutation, and (finite inputs) the inverse and det
    within ``max(FLOOR, 10x CPU float32)`` (a NaN error fails; where CPU float32's is NaN too, it is not judged)."""
    ok = all(row["bit_equal_R_P_Q_D_S"]) and row["P_is_permutation"]
    for key in ("inv_err", "det_rel_err"):
        cpu = row.get(key + "_cpu32")
        if key in row and cpu == cpu:  # where CPU float32 itself overflows (NaN error), the error is only reported
            ok &= row[key] <= max(FLOOR, 10 * cpu)  # NaN compares False
    return bool(ok)


def main():
    from hpu_ops import tpc

    tpc.install_kernel_path()
    import habana_frameworks.torch.core as htcore

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
            htcore.mark_step()
            row.update(accuracy(A, X.cpu(), D.cpu(), S.cpu()))
        row["ok"] = verdict(row)
        ok &= row["ok"]
        print("RESULT " + json.dumps(row), flush=True)
    print("SUMMARY", "ok" if ok else "MISMATCH", flush=True)


if __name__ == "__main__":
    main()
