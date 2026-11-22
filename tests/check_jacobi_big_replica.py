"""On a Gaudi card: the raw ``jacobi_big`` kernel (full and refinement builds) against its float32 replica
(``kernel_replica.jacobi_big``), bit for bit, on shifted positive definite inputs of n = 33, 64, 65, 110, 128.
    PT_HPU_LAZY_MODE=0|1 python3 tests/check_jacobi_big_replica.py"""
import json

import torch

import kernel_replica as kr
from hpu_ops import tpc


def main():
    tpc.install_kernel_path()
    import habana_frameworks.torch.core as htcore

    assert tpc.ready()
    ok = True
    for n in (33, 64, 65, 110, 128):
        g = torch.Generator().manual_seed(n)
        G = torch.randn(2, n, n, generator=g)
        X = (G + G.mT) / 2
        sigma = X.abs().sum(-1).amax(-1)
        S = X / sigma[:, None, None] + tpc.BIG_SHIFT * torch.eye(n)
        w = 64 * (-(-n // 64))
        S0 = torch.nn.functional.pad(S, (0, w - n)).contiguous()
        for refine in (False, True):
            got = tpc._big_op(refine)(S0.to("hpu")).cpu()
            htcore.mark_step()
            want = kr.jacobi_big_op(refine)(S0)
            row = {"n": n, "refine": refine, "bit_equal": bool(torch.equal(got, want)),
                   "max_diff": float((got - want).abs().max())}
            ok &= row["bit_equal"]
            print("RESULT " + json.dumps(row), flush=True)
    print("SUMMARY", "ok" if ok else "MISMATCH", flush=True)


if __name__ == "__main__":
    main()
