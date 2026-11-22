"""On a Gaudi card: braindecode ``MetaNeuromotorHand``'s SGD step (#43) with and without hpu_ops' ``unfold_backward``.
    PT_HPU_LAZY_MODE=0|1 PYTHONPATH=<braindecode src>:$PYTHONPATH python3 tests/bench_unfold_backward_model.py [rounds]
        [--no-random] [--sides=before,after]
``--no-random`` builds the model with ``drop_prob=0`` and no SpecAugment masks (still in train mode), so the two sides
draw no random numbers; ``--sides`` times only the sides named.
The issue's sweep step: batch 16, the registered test signal (16 channels, 32000 samples at 2 kHz), loss = mean square
of the output, ``SGD(lr=1e-5)``. ``after`` installs ``composite.install_unfold_backward``, ``before`` leaves the
bridge's CPU fallback, alternating per round in one process. Per round: 3 warm-up steps, then the mean of 20 synced
steps; the bridge's fallbacks per step. First, the input and parameter gradients of one backward, both sides from the
same weights and input, against each other (``before`` runs ATen's CPU kernel: the reference)."""
import os
import statistics
import sys
import time

import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.hpu as hthpu
import torch
from habana_frameworks.torch.hpu.metrics import metric_global

import hpu_ops
from hpu_ops import composite

LAZY = os.environ.get("PT_HPU_LAZY_MODE", "0") == "1"
hpu_ops.FIXES = tuple(f for f in hpu_ops.FIXES if f is not composite.install_unfold_backward)
hpu_ops.enable()

from braindecode.models import MetaNeuromotorHand  # noqa: E402


def fb():
    return dict(dict(metric_global("cpu_fallback").stats()).get("FallbackOps", {}) or {})


def sync():
    if LAZY:
        htcore.mark_step()
    hthpu.synchronize()


def side(name):
    return composite.install_unfold_backward("hpu") if name == "after" else (lambda: None)


def grads(model, x):
    model.zero_grad(set_to_none=True)
    xg = x.detach().clone().requires_grad_()
    model(xg).float().pow(2).mean().backward()
    sync()
    return [xg.grad.cpu()] + [p.grad.cpu() for p in model.parameters() if p.grad is not None]


def main(rounds, no_random=False, sides=("before", "after")):
    torch.manual_seed(0)
    extra = {"drop_prob": 0.0, "mask_max_num_masks": (0, 0)} if no_random else {}
    model = MetaNeuromotorHand(n_chans=16, n_outputs=100, n_times=32000, sfreq=2000.0, **extra).to("hpu").train()
    print("no_random", no_random, "sides", sides, flush=True)
    x = torch.randn(16, 16, 32000).to("hpu")
    print("bench_unfold_backward_model lazy", LAZY, "torch", torch.__version__, flush=True)
    for mode in ("eval", "train"):  # eval: no SpecAugment masks, so no random numbers on either side
        model.train(mode == "train")
        got = {}
        for name in ("before", "before2", "after", "after2"):
            undo = side(name[:5] if name.startswith("after") else "before")
            try:
                torch.manual_seed(1)  # the same SpecAugment masks on both sides (eager: measured identical)
                got[name] = grads(model, x)
            finally:
                undo()

        def rel(a, b):
            return max(float((u - v).norm() / v.norm().clamp_min(1e-30)) for u, v in zip(got[a], got[b]))

        print("GRADS", mode, len(got["after"]), "tensors, max relative error: after vs before",
              f"{rel('after', 'before'):.2e}", "before vs before", f"{rel('before2', 'before'):.2e}",
              "after vs after", f"{rel('after2', 'after'):.2e}", flush=True)
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=1e-5)

    def step():
        opt.zero_grad(set_to_none=True)
        loss = model(x).float().pow(2).mean()
        loss.backward()
        opt.step()
        return loss

    times = {name: [] for name in sides}
    for r in range(rounds):
        for name in sides if r % 2 == 0 else sides[::-1]:
            undo = side(name)
            try:
                for _ in range(3):
                    step()
                sync()
                f0 = fb()
                keep, t0 = [], time.perf_counter()
                for _ in range(20):
                    keep.append(step())
                    sync()
                ms = (time.perf_counter() - t0) / 20 * 1e3
                d = fb()
            finally:
                undo()
            times[name].append(ms)
            per = {k: (d[k] - f0.get(k, 0)) / 20 for k in d if d[k] - f0.get(k, 0)}
            print("ROUND", r, name, f"{ms:.1f} ms/step", "fallbacks/step", per, "loss", float(keep[-1]), flush=True)
    for name, ts in times.items():
        print("RESULT", name, "median_ms", f"{statistics.median(ts):.1f}", "rounds", [f"{t:.1f}" for t in ts],
              flush=True)


if __name__ == "__main__":
    args = sys.argv[1:]
    sides = next((a.split("=", 1)[1] for a in args if a.startswith("--sides=")), "before,after")
    numbers = [a for a in args if not a.startswith("--")]
    main(int(numbers[0]) if numbers else 3, "--no-random" in args, tuple(sides.split(",")))
