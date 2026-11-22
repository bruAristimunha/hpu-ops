"""On a Gaudi card: the linalg calls of one EEGSPDNet train step (shapes and counts of eigh and the Gauss-Jordan
routes), then eigh at those shapes timed on its routes: block Jacobi on the card, hpu-ops' CPU route, and torch on the
host. Mean ms of synced calls after warm-up.
    PT_HPU_LAZY_MODE=0|1 python3 tests/probe_eegspdnet_linalg.py [--model EEGSPDNet] [--reps 5]"""
import argparse
import collections
import json
import os
import time

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="EEGSPDNet")
ap.add_argument("--reps", type=int, default=5)
args = ap.parse_args()
LAZY = os.environ.get("PT_HPU_LAZY_MODE", "0") == "1"


def main():
    import habana_frameworks.torch.core as htcore
    from spd_diag import protocol as P
    from spd_diag.cache import open_cache

    import hpu_ops.eigh as E
    from hpu_ops import linalg

    P._enable_device("hpu")
    seen = collections.Counter()
    orig = E.Eigh.compute

    def logged(A, UPLO, compute_v):
        seen[("eigh", tuple(A.shape), bool(compute_v))] += 1
        return orig(A, UPLO, compute_v)

    E.Eigh.compute = staticmethod(logged)
    for cls in ("SolveEx", "InvEx", "Det", "Slogdet"):
        f = getattr(linalg, cls)
        o = f.forward

        def fw(ctx, A, *a, _o=o, _c=cls):
            seen[(_c, tuple(A.shape))] += 1
            return _o(ctx, A, *a)

        f.forward = staticmethod(fw)
    cache = open_cache("BNCI2014_001")
    info, sd = cache.info, cache.subject(cache.subjects[0], "X")
    X = torch.as_tensor(sd.X[:32], dtype=torch.float32)
    X = (X / X.std()).to("hpu")
    y = torch.as_tensor(sd.y[:32]).long().to("hpu")
    model = P.build(args.model, X.shape[1], info, None).to("hpu")
    out = model(X)
    out = out[0] if isinstance(out, tuple) else out
    torch.nn.functional.cross_entropy(out, y).backward()
    htcore.mark_step()
    torch.hpu.synchronize()
    E.Eigh.compute = staticmethod(orig)
    print("RESULT " + json.dumps({"calls": {str(k): v for k, v in seen.items()}, "lazy": LAZY}), flush=True)
    for (name, shape, *_), _count in list(seen.items()):
        if name != "eigh":
            continue
        g = torch.randn(*shape)
        S = g @ g.mT / shape[-1] + 1e-3 * torch.eye(shape[-1])
        Sh = S.to("hpu")
        for route in ("card", "cpu_route", "host"):
            os.environ[E.LARGE_EIGH] = "cpu" if route == "cpu_route" else ""

            def call():
                if route == "host":
                    return torch.linalg.eigh(S)
                return torch.linalg.eigh(Sh)

            keep = [call() for _ in range(2)]
            htcore.mark_step()
            torch.hpu.synchronize()
            t = time.perf_counter()
            keep = [call() for _ in range(args.reps)]
            htcore.mark_step()
            torch.hpu.synchronize()
            ms = (time.perf_counter() - t) / args.reps * 1e3
            del keep
            print("RESULT " + json.dumps({"shape": shape, "route": route, "ms": round(ms, 2), "lazy": LAZY}), flush=True)
        os.environ.pop(E.LARGE_EIGH, None)


if __name__ == "__main__":
    main()
