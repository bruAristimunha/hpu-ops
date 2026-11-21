"""On a Gaudi card in lazy mode: the composites past the kernels' sizes one by one, to find which lazy graph fails to
compile or computes wrong values (the LU composite, block Jacobi eigh, the matrix_exp composite), with and without a
graph boundary (mark_step) after each step.
    PT_HPU_LAZY_MODE=1 python3 tests/probe_lazy_composites.py"""
import json
import traceback

import hpu_ops

hpu_ops.enable()
import habana_frameworks.torch.core as htcore  # noqa: E402
import torch  # noqa: E402

from hpu_ops import jacobi, linalg, tpc  # noqa: E402


def rel(x, ref):
    return float((x.detach().cpu().double() - ref.double()).abs().max() / ref.double().abs().max())


def run(name, f, ref):
    row = {"case": name}
    try:
        out = f()
        htcore.mark_step()
        torch.hpu.synchronize()
        row["err"] = rel(out, ref)
    except Exception as e:  # noqa: BLE001
        row["error"] = repr(e)[:240]
        traceback.print_exc(limit=3)
    print("RESULT " + json.dumps(row), flush=True)


def lu_stepped(A, every):
    n = A.shape[-1]
    rows = torch.arange(n, device=A.device)
    LU = A
    for k in range(n):
        col = LU[..., :, k].abs().masked_fill(rows < k, -1.0)
        p = col.argmax(-1)
        swap = torch.where(rows == k, p[..., None], torch.where(rows == p[..., None], k, rows))
        LU = LU.gather(-2, swap[..., None].expand_as(LU))
        pivot = LU[..., k:k + 1, k:k + 1]
        lcol = torch.where((rows > k)[:, None], LU[..., :, k:k + 1] / torch.where(pivot == 0, 1.0, pivot), 0.0)
        LU = LU - lcol * torch.where((rows > k)[None, :], LU[..., k:k + 1, :], 0.0)
        LU = torch.where(((rows > k)[:, None]) & (rows == k)[None, :], lcol, LU)
        if every and (k + 1) % every == 0:
            htcore.mark_step()
    return LU


def main():
    torch.manual_seed(0)
    A = torch.randn(8, 64, 64)
    ref = torch.linalg.lu_factor(A)[0]
    for every in (0, 1, 8):
        run(f"lu_composite_every{every}", lambda: lu_stepped(A.to("hpu"), every), ref)
    run("lu_public", lambda: torch.linalg.lu_factor(A.to("hpu"))[0], ref)
    G = torch.randn(4, 64, 64)
    S = G @ G.mT + 64 * torch.eye(64)
    w_ref = torch.linalg.eigvalsh(S)
    eig = lambda X: tpc.eigh(X, False)  # noqa: E731
    run("block_eigh_64", lambda: jacobi.block_eigh(S.to("hpu"), eig)[0], w_ref)
    orig = jacobi._rotate_pairs

    def flushed(*a, **k):
        out = orig(*a, **k)
        htcore.mark_step()
        return out

    jacobi._rotate_pairs = flushed
    run("block_eigh_64_flush_each_round", lambda: jacobi.block_eigh(S.to("hpu"), eig)[0], w_ref)
    jacobi._rotate_pairs = orig
    M = 0.1 * torch.randn(4, 64, 64)
    run("expm_composite_64", lambda: linalg._matrix_exp_composite(M.to("hpu"), False), torch.linalg.matrix_exp(M))


if __name__ == "__main__":
    main()
