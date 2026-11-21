"""On a Gaudi card: bridge CPU fallbacks of spd_learn's vech (CPU triu indices), X[:, mask] and F.unfold / F.fold
at the SPD models' shapes, under training, torch.no_grad() and torch.inference_mode(), with hpu_ops' fixes.
    PT_HPU_LAZY_MODE=0|1 python3 tests/index_im2col_modes.py
One CASE line per (case, mode): fallbacks of 3 calls, max error against the CPU, rewrites counted."""
import contextlib
import os

import habana_frameworks.torch.core as htcore
import torch
import torch.nn.functional as F
from habana_frameworks.torch.hpu.metrics import metric_global

import hpu_ops

LAZY = os.environ.get("PT_HPU_LAZY_MODE", "1") != "0"
R, C = torch.triu_indices(22, 22)
MASK = torch.ones(22, 22, dtype=torch.bool).triu()
CASES = {
    "vech": (lambda x: x[..., R, C], lambda: torch.randn(32, 22, 22)),
    "X[:, mask]": (lambda x: x[:, MASK], lambda: torch.randn(32, 22, 22)),
    "unfold MAtt": (lambda x: F.unfold(x, (257, 1), stride=(257, 1)), lambda: torch.randn(32, 20, 514, 1)),
    "unfold k3x5": (lambda x: F.unfold(x, (3, 5)), lambda: torch.randn(8, 4, 22, 64)),
    "fold k85": (lambda x: F.fold(x, (513, 1), (85, 1), stride=(85, 1)), lambda: torch.randn(32, 1700, 6)),
}
MODES = {"train": contextlib.nullcontext, "no_grad": torch.no_grad, "inference_mode": torch.inference_mode}


def fb():
    return dict(dict(metric_global("cpu_fallback").stats()).get("FallbackOps", {}) or {})


def main():
    hpu_ops.enable()
    print("PLUGINS", hpu_ops.__file__, "lazy", int(LAZY), flush=True)
    for name, (f, make) in CASES.items():
        x0 = make()
        ref = f(x0.double())
        for mode, ctx in MODES.items():
            hpu_ops._STATE["calls"].clear()
            f0 = fb()
            try:
                outs = []
                for _ in range(3):
                    x = x0.to("hpu").requires_grad_(mode == "train")
                    with ctx():
                        out = f(x)
                    if mode == "train":
                        out.square().sum().backward()
                    if LAZY:
                        htcore.mark_step()
                    outs.append(out.detach().cpu().double())
                err = max(float((o - ref).abs().max() / ref.abs().max()) for o in outs)
                msg = f"err={err:.2e}"
            except Exception as e:  # noqa: BLE001
                msg = "ERROR " + repr(e)[:150]
            f1 = fb()
            d = {k: f1[k] - f0.get(k, 0) for k in f1 if f1[k] - f0.get(k, 0)}
            print(f"CASE lazy={int(LAZY)} {name:<12} {mode:<15} fallbacks/3 calls={d} {msg} "
                  f"rewrites={dict(hpu_ops._STATE['calls'])}", flush=True)


if __name__ == "__main__":
    main()
