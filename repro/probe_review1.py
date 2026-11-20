"""Gaudi probe (review round 1): which keys hpu_ops registers before ``enable()``, and what an out-of-range index does
in index_select / index_2d with indices on the card.
    PT_HPU_LAZY_MODE=0|1 python3 repro/probe_review1.py
Prints one ``PROBE {json}`` line per check."""
import json
import os

import habana_frameworks.torch.core as htcore
import torch

LAZY = os.environ.get("PT_HPU_LAZY_MODE", "0") != "0"
OPS = ("aten::index_put_", "aten::index_put", "aten::_index_put_impl_", "aten::index.Tensor", "aten::convolution")


def show(name, **row):
    print("PROBE " + json.dumps({"check": name, "lazy": LAZY, "bridge": "1.21.4", **row}, default=str), flush=True)


def keys():
    has = torch._C._dispatch_has_kernel_for_dispatch_key
    return {op: {k: has(op, k) for k in ("HPU", "AutogradHPU")} for op in OPS}


def attempt(name, fn):
    try:
        out = fn()
        if LAZY:
            htcore.mark_step()
        show(name, raised=None, value=out.cpu().tolist())
    except Exception as e:  # noqa: BLE001  (an exception is the expected outcome)
        show(name, raised=repr(e)[:300])


def main():
    show("keys before enable", keys=keys())
    xn = torch.arange(12.0).reshape(3, 4).to("hpu")
    attempt("native x[[0],[4]] device indices (before enable)",
            lambda: xn[torch.tensor([0], device="hpu"), torch.tensor([4], device="hpu")])
    attempt("native x[[0],[4]] host indices (before enable)", lambda: xn[torch.tensor([0]), torch.tensor([4])])
    import hpu_ops
    hpu_ops.enable()
    hpu_ops._STATE["calls"].clear()
    x = torch.arange(12.0).reshape(3, 4).to("hpu")
    flat = x.flatten()
    attempt("index_select flat index h*w (12) on the card", lambda: flat.index_select(0, torch.tensor([12], device="hpu")))
    attempt("index_select flat index 100 on the card", lambda: flat.index_select(0, torch.tensor([100], device="hpu")))
    attempt("x[[0],[4]] device indices (index_2d)", lambda: x[torch.tensor([0], device="hpu"), torch.tensor([4], device="hpu")])
    attempt("x[[0],[4]] host indices (index_2d)", lambda: x[torch.tensor([0]), torch.tensor([4])])
    attempt("x[[1],[2]] device indices (in range)", lambda: x[torch.tensor([1], device="hpu"), torch.tensor([2], device="hpu")])
    show("rewrite calls", calls=dict(hpu_ops._STATE["calls"]))


if __name__ == "__main__":
    main()
