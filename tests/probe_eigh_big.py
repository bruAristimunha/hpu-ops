"""On a Gaudi card: ``tpc.eigh_big`` (the jacobi_big kernel, 32 < n <= 128) against torch's eigh in float64 and
float32 on the CPU, and its time against hpu-ops' CPU route and the block Jacobi. Mean ms of synced calls.
    PT_HPU_LAZY_MODE=0|1 python3 tests/probe_eigh_big.py [--sizes 33,55,64,110,128] [--batch 32]"""
import argparse
import json
import os
import time

import hpu_ops

hpu_ops.enable()
import habana_frameworks.torch.core as htcore  # noqa: E402
import torch  # noqa: E402

from hpu_ops import jacobi, tpc  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--sizes", default="33,55,64,110,128")
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--reps", type=int, default=5)
args = ap.parse_args()
LAZY = os.environ.get("PT_HPU_LAZY_MODE", "0") == "1"


def rel(x, r):
    return float((x.cpu().double() - r).abs().max() / r.abs().max())


def timed(f):
    keep = [f() for _ in range(2)]
    htcore.mark_step()
    torch.hpu.synchronize()
    t = time.perf_counter()
    keep = [f() for _ in range(args.reps)]
    htcore.mark_step()
    torch.hpu.synchronize()
    del keep
    return round((time.perf_counter() - t) / args.reps * 1e3, 2)


for n in [int(x) for x in args.sizes.split(",")]:
    torch.manual_seed(n)
    for kind in ("spd", "indefinite"):
        G = torch.randn(args.batch, n, n)
        A = G @ G.mT / n + 1e-3 * torch.eye(n) if kind == "spd" else (G + G.mT) / 2
        w64, V64 = torch.linalg.eigh(A.double())
        w32, V32 = torch.linalg.eigh(A)
        Ah = A.to("hpu")
        row = {"n": n, "batch": args.batch, "kind": kind, "lazy": LAZY}
        try:
            w, V = tpc.eigh_big(Ah)
            htcore.mark_step()
            row["w_err"], row["w_err_cpu32"] = rel(w, w64), rel(w32, w64)
            rec = lambda w_, V_: rel(V_.cpu().double() * w_.cpu().double()[..., None, :] @ V_.cpu().double().mT, A.double())  # noqa: E731
            row["rec_err"], row["rec_err_cpu32"] = rec(w, V), rec(w32, V32)
            row["orth"] = float((V.cpu().mT @ V.cpu() - torch.eye(n)).abs().max())
            row["ms_big"] = timed(lambda: tpc.eigh_big(Ah))
            if kind == "spd":
                row["ms_block"] = timed(lambda: jacobi.block_eigh(Ah, lambda X: tpc.eigh(X, False)))
                os.environ["HPU_OPS_LARGE_EIGH"] = "cpu"
                row["ms_cpu_route"] = timed(lambda: torch.linalg.eigh(Ah))
                os.environ.pop("HPU_OPS_LARGE_EIGH")
        except Exception as e:  # noqa: BLE001
            row["error"] = repr(e)[:300]
        print("RESULT " + json.dumps(row), flush=True)
