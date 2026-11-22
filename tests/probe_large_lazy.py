"""On a Gaudi card: where the time of a large (64 < n <= 512) solve / inv / det goes, ours against Habana's fallback.
    PT_HPU_LAZY_MODE=0|1 python3 tests/probe_large_lazy.py [--n 220] [--batch 1] [--reps 20]
Same process, alternating: ``square`` (one square in-place kernel call), ``blocked`` (the blocked route) and ``stock``
(the route declined: Habana's fallback for solve/det/slogdet, hpu-ops' CPU route for inv), forward + backward of
``sum(solve(A, B))`` and friends; and the raw kernel alone. Mean ms of ``reps`` synced iterations after warm-up,
outputs kept alive. One ``RESULT {json}`` line per (op, side)."""
import argparse
import json
import os
import time

import hpu_ops

hpu_ops.enable()
import habana_frameworks.torch.core as htcore  # noqa: E402
import habana_frameworks.torch.hpu as hthpu  # noqa: E402
import torch  # noqa: E402

from hpu_ops import tpc  # noqa: E402

LAZY = os.environ.get("PT_HPU_LAZY_MODE", "0") == "1"
ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=220)
ap.add_argument("--batch", type=int, default=1)
ap.add_argument("--reps", type=int, default=20)
ap.add_argument("--rounds", type=int, default=3)
ap.add_argument("--ops", default="raw,solve,inv,det,slogdet")
args = ap.parse_args()


def sync():
    if LAZY:
        htcore.mark_step()
    hthpu.synchronize()


def timed(f):
    keep = [f() for _ in range(3)]
    sync()
    t = time.perf_counter()
    keep = []
    for _ in range(args.reps):
        keep.append(f())
        if LAZY:
            htcore.mark_step()
    sync()
    return (time.perf_counter() - t) / args.reps * 1e3


def main():
    g = torch.Generator().manual_seed(0)
    n, b = args.n, args.batch
    A0 = torch.randn(b, n, n, generator=g) + n ** 0.5 * torch.eye(n)
    B0 = torch.randn(b, n, 4, generator=g)
    A, B = A0.to("hpu"), B0.to("hpu")

    def fwd_bwd(fn):
        def run():
            a = A.detach().requires_grad_()
            out = fn(a)
            out.sum().backward()
            return out.detach(), a.grad
        return run

    cases = {
        "solve": fwd_bwd(lambda a: torch.linalg.solve(a, B)),
        "inv": fwd_bwd(lambda a: torch.linalg.inv(a)),
        "det": fwd_bwd(lambda a: torch.linalg.slogdet(a)[1] + 0 * torch.linalg.det(a[..., :1, :1])),
        "slogdet": fwd_bwd(lambda a: torch.linalg.slogdet(a)[1]),
    }
    original, square = tpc.large_pays, tpc._square_pays
    sides = (("square", True, lambda n_, b_: True), ("blocked", True, lambda n_, b_: False),
             ("stock", False, square))
    for r in range(args.rounds):
        for op in args.ops.split(","):
            if op == "raw":
                if n > tpc.LARGE_MAXN:
                    continue
                ms = timed(lambda: tpc._call(tpc._large_op(), A.mT.contiguous()))
                print("RESULT " + json.dumps(dict(op="raw", side="ours", n=n, batch=b, lazy=LAZY, round=r,
                                                  ms=round(ms, 3))), flush=True)
                continue
            for side, ours, route in sides:
                if side == "square" and n > tpc.LARGE_MAXN:
                    continue
                tpc.large_pays = (lambda n_, b_: True) if ours else (lambda n_, b_: False)
                tpc._square_pays = route
                try:
                    ms = timed(cases[op])
                    err = None
                except Exception as e:  # noqa: BLE001
                    ms, err = None, repr(e)[:200]
                tpc.large_pays, tpc._square_pays = original, square
                print("RESULT " + json.dumps(dict(op=op, side=side, n=n, batch=b, lazy=LAZY, round=r,
                                                  ms=None if ms is None else round(ms, 3), error=err)), flush=True)


if __name__ == "__main__":
    main()
