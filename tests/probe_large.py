"""On a Gaudi card: adversarial probe of solve, solve_ex, inv, det and slogdet past n = 64 (the in-place kernel).
    PT_HPU_LAZY_MODE=0|1 python3 tests/probe_large.py [--only REGEX] [--compile]
The row runner and its bar are ``tests/probe_adversarial_f.py``'s: error within max(1e-6, 10x CPU float32's) against
float64 on the float32 inputs, the NaN/Inf pattern of CPU float32, an error where the CPU raises, no bridge fallback,
no "runs on the CPU" warning. Bridge 1.21.4."""
import argparse
import math
import re
import sys

import probe_adversarial_f as base  # enables hpu_ops before the first HPU op
import torch

L = torch.linalg
G = torch.Generator().manual_seed(23)


def rn(*shape, scale=1.0):
    return torch.randn(*shape, generator=G) * scale


def permuted(b, n):
    M = torch.eye(n) + 0.05 * torch.randn(b, n, n, generator=G) / n ** 0.5
    perm = torch.stack([torch.randperm(n, generator=G) for _ in range(b)])
    return M.gather(1, perm[..., None].expand(b, n, n))


def cayley(b, n):
    g = torch.randn(b, n, n, generator=G) / math.sqrt(n)
    S = g - g.mT
    eye = torch.eye(n)
    return eye - S / 2, eye + S / 2


def ill(b, n, cond):
    """``U diag(logspace(0, -log10 cond)) V^T`` with random orthogonal U, V: condition ``cond``."""
    U = torch.linalg.qr(torch.randn(b, n, n, generator=G, dtype=torch.float64)).Q
    V = torch.linalg.qr(torch.randn(b, n, n, generator=G, dtype=torch.float64)).Q
    d = torch.logspace(0, -math.log10(cond), n, dtype=torch.float64)
    return (U * d @ V.mT).float()


def slog(a):
    return L.slogdet(a)


def cases():
    A, B = cayley(1, 220)
    yield "cayley_b1_n220", L.solve, (A, B), {}
    yield "cayley_2d_n220", L.solve, (A[0], B[0]), {}
    yield "cayley_grad_b1_n220", L.solve, (A, B), {"grad": True}
    for n in (65, 127, 128, 129, 300, 449, 512):
        yield f"solve_n{n}", L.solve, (permuted(2, n), rn(2, n, 3)), {}
    yield "inv_n512_b1", L.inv, (permuted(1, 512),), {}
    yield "det_n512", L.det, (permuted(2, 512),), {}
    yield "slogdet_n512", slog, (permuted(2, 512),), {}
    yield "solve_m600", L.solve, (permuted(2, 100), rn(2, 100, 600)), {}
    yield "solve_vector_b1", L.solve, (permuted(1, 220), rn(1, 220)), {}
    yield "solve_left_false_b1", lambda a, b: L.solve(a, b, left=False), (permuted(1, 220), rn(1, 4, 220)), {}
    yield "solve_broadcast_A", L.solve, (permuted(1, 130), rn(8, 130, 3)), {}
    yield "solve_broadcast_B", L.solve, (permuted(6, 130), rn(130, 3)), {}
    yield "solve_expand_A", lambda a, b: L.solve(a.expand(5, 130, 130), b), (permuted(1, 130)[0], rn(5, 130, 2)), {}
    yield "solve_transposed_A", lambda a, b: L.solve(a.mT, b), (permuted(3, 150), rn(3, 150, 2)), {}
    yield "solve_step_slice", lambda a, b: L.solve(a[::2, 1:221, 1:221], b[::3]), (permuted(8, 224), rn(12, 220, 2)), {}
    yield "solve_permuted_batch", lambda a, b: L.solve(a.permute(1, 0, 2, 3), b), \
        (permuted(6, 100).reshape(3, 2, 100, 100), rn(2, 3, 100, 2)), {}
    yield "inv_of_batch_item", lambda a: L.inv(a[3]), (permuted(5, 220),), {}
    for c in (1e4, 1e6):
        yield f"solve_ill_{c:.0e}_n220", L.solve, (ill(4, 220, c), rn(4, 220, 3)), {}
        yield f"inv_ill_{c:.0e}_n128", L.inv, (ill(4, 128, c),), {}
        yield f"slogdet_ill_{c:.0e}_n220", slog, (ill(4, 220, c),), {}
    P = torch.eye(220)[torch.randperm(220, generator=G)]
    yield "solve_permutation_matrix", L.solve, (P.expand(2, 220, 220).clone(), rn(2, 220, 3)), {}
    yield "det_permutation_matrix", L.det, (P.expand(2, 220, 220).clone(),), {}
    yield "solve_scale_1e30", L.solve, (permuted(2, 130) * 1e30, rn(2, 130, 3)), {}
    yield "solve_scale_1e-30", L.solve, (permuted(2, 130) * 1e-30, rn(2, 130, 3)), {}
    nanA = permuted(2, 100)
    nanA[1, 3, 5] = float("nan")
    yield "solve_nan_in_A", L.solve, (nanA, rn(2, 100, 3)), {}
    infA = permuted(2, 100)
    infA[0, 0, 0] = float("inf")
    yield "solve_inf_in_A", L.solve, (infA, rn(2, 100, 3)), {}
    S0 = permuted(3, 220)
    S0[1] = 0
    yield "solve_singular_raises", L.solve, (S0, rn(3, 220, 3)), {}
    yield "inv_singular_raises", L.inv, (S0,), {}
    S1 = permuted(3, 220)
    S1[2, :, 17] = 0
    yield "det_singular_column", L.det, (S1,), {}
    yield "solve_ex_info", lambda a, b: L.solve_ex(a, b).info.float(), (S1, rn(3, 220, 3)), {}
    yield "inv_ex_info", lambda a: L.inv_ex(a).info.float(), (S1,), {}
    yield "solve_grad_n128", L.solve, (permuted(4, 128), rn(4, 128, 3)), {"grad": True}
    yield "inv_grad_n128", L.inv, (permuted(4, 128),), {"grad": True}
    yield "det_grad_n96", L.det, (permuted(4, 96),), {"grad": True}
    yield "slogdet_grad_n220", slog, (permuted(4, 220),), {"grad": True}
    yield "solve_double_bwd_n96", base.double_bwd(L.solve), (permuted(4, 96), rn(4, 96, 3)), \
        {"grad": True, "scalar": True}
    yield "solve_chain5_n220", base.chain(L.solve, 5), cayley(2, 220), {}
    yield "solve_inference_mode", L.solve, (permuted(4, 220), rn(4, 220, 3)), {"inference": True}
    yield "inv_inference_mode", L.inv, (permuted(4, 220),), {"inference": True}
    yield "solve_batch_100_n96", L.solve, (permuted(100, 96), rn(100, 96, 1)), {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--compile", action="store_true")
    args = ap.parse_args()
    print("PLUGINS", base.hpu_ops.__file__, "tpc_ready", base.hpu_ops.tpc.ready(), "lazy", base.LAZY, flush=True)
    selected = [c for c in cases() if not args.only or re.search(args.only, c[0])]
    results = [base.run(name, fn, ins, opt) for name, fn, ins, opt in selected]
    if args.compile and not base.LAZY:
        results += [base.run(name, fn, ins, opt, compiled=True) for name, fn, ins, opt in selected
                    if not opt.get("inference") and "chain" not in name]
    print(f"SUMMARY {len(results)} rows, {results.count(False)} failed", flush=True)
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
