"""Which hpu_ops.extrema rewrites one training step of each SPD model of tests/linalg_steps.py runs, and how often.
    PT_HPU_LAZY_MODE=0|1 python3 repro/extrema_counts.py [--models SPDNet,TSMNet]
One ``COUNTS {json}`` line per model: the rewrite call counts of one step after warm-up (forward, backward, Adam)."""
import argparse
import json
import os

import habana_frameworks.torch.core as htcore
import torch

LAZY = os.environ.get("PT_HPU_LAZY_MODE", "1") != "0"
HPU = torch.device("hpu")
EXTREMA = {"amax", "amin", "max", "min", "median", "max_dim", "min_dim", "median_dim", "argmax", "argmin", "aminmax",
           "norm_inf", "maximum", "minimum", "fmax", "fmin", "clamp_tensor", "max_pool2d", "max_pool3d", "sort",
           "topk", "kthvalue"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="SPDNet,TSMNet,TensorCSPNet,EEGSPDNet,GBWAtt")
    ap.add_argument("--no-sort", action="store_true", help="the sort family keeps the native op (bisection)")
    a = ap.parse_args()
    if a.no_sort:
        from hpu_ops import extrema

        extrema._sortable = lambda x: False
    from spd_diag import protocol as P
    from spd_diag.cache import open_cache

    import hpu_ops

    P._enable_device("hpu")
    cache = open_cache("BNCI2014_001")
    info, sd = cache.info, cache.subject(cache.subjects[0], "X")
    X = torch.as_tensor(sd.X[:32], dtype=torch.float32)
    X = X / X.std()
    y = torch.as_tensor(sd.y[:32]).long().to(HPU)
    for name in a.models.split(","):
        model = P.build(name, X.shape[1], info, None).to(HPU)
        xb = model.fixed_input(X).to(HPU) if hasattr(model, "fixed_input") else X.to(HPU)
        if hasattr(model, "fit_graph"):
            model.fit_graph(X)
        opt = torch.optim.Adam(model.parameters(), 1e-3)

        def step():
            opt.zero_grad(set_to_none=True)
            out = model(xb)
            out = out[0] if isinstance(out, tuple) else out
            torch.nn.functional.cross_entropy(out, y).backward()
            opt.step()
            if LAZY:
                htcore.mark_step()

        step()
        hpu_ops._STATE["calls"].clear()
        step()
        calls = {k: v for k, v in hpu_ops._STATE["calls"].items() if k in EXTREMA}
        print("COUNTS " + json.dumps({"model": name, "lazy": LAZY, "calls": calls}), flush=True)


if __name__ == "__main__":
    main()
