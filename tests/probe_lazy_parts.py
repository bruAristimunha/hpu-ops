"""On a Gaudi card (lazy mode): what each part of the square in-place route costs per call, for an A/B of glue
variants (timing only: a variant may return wrong values).
    PT_HPU_LAZY_MODE=1 python3 tests/probe_lazy_parts.py [--n 220] [--batch 1] [--reps 30]
Variants of ``inv`` forward + backward: the route as is; no output clones in ``tpc._call``; no gathers (``R``
returned as the inverse); the raw kernel after a mark_step; and the stock route (declined). Mean ms of ``reps``
synced iterations (each closed by a mark_step) after warm-up, three alternating rounds."""
import argparse
import json
import time

import hpu_ops

hpu_ops.enable()
import habana_frameworks.torch.core as htcore  # noqa: E402
import habana_frameworks.torch.hpu as hthpu  # noqa: E402
import torch  # noqa: E402

from hpu_ops import tpc  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=220)
ap.add_argument("--batch", type=int, default=1)
ap.add_argument("--reps", type=int, default=30)
args = ap.parse_args()


def timed(f):
    keep = [f() for _ in range(3)]
    htcore.mark_step()
    hthpu.synchronize()
    t = time.perf_counter()
    keep = []
    for _ in range(args.reps):
        keep.append(f())
        htcore.mark_step()
    hthpu.synchronize()
    return (time.perf_counter() - t) / args.reps * 1e3


def main():
    n, b = args.n, args.batch
    A = (torch.randn(b, n, n) + n ** 0.5 * torch.eye(n)).to("hpu")

    def inv_fb():
        a = A.detach().requires_grad_()
        out = torch.linalg.inv(a)
        out.sum().backward()
        return out.detach(), a.grad

    def inv_fwd():
        return torch.linalg.inv(A)

    call, square, large_pays = tpc._call, tpc._square, tpc.large_pays

    def no_clone(op, *a):
        tpc._flush()
        return op(*a)

    def no_gather(A_, sign=True):
        R, P, Q, D, S = call(tpc._large_op(), A_)
        return R, P, Q, D, S

    variants = {
        "route": {},
        "no_clone": {"_call": no_clone},
        "raw_only": {"_call": no_clone, "_rows": lambda X, i: X},
        "stock": {"large_pays": lambda n_, b_: False},
    }
    for r in range(3):
        for name, patch in variants.items():
            for k, v in patch.items():
                setattr(tpc, k, v)
            try:
                ms_fb, ms_f, err = timed(inv_fb), timed(inv_fwd), None
            except Exception as e:  # noqa: BLE001
                ms_fb = ms_f = None
                err = repr(e)[:200]
            tpc._call, tpc._square, tpc.large_pays = call, square, large_pays
            tpc._rows = _ROWS
            print("RESULT " + json.dumps(dict(variant=name, n=n, batch=b, round=r, fwd_bwd_ms=ms_fb, fwd_ms=ms_f,
                                              error=err)), flush=True)
    raw = timed(lambda: call(tpc._large_op(), A.mT.contiguous()))
    print("RESULT " + json.dumps(dict(variant="raw_kernel", n=n, batch=b, fwd_ms=raw)), flush=True)


_ROWS = tpc._rows

if __name__ == "__main__":
    main()
