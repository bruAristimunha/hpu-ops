"""Issue #42: standalone pure-PyTorch repros for the two Habana ``hpu_backend`` dynamic-shape bugs
uncovered in braindecode after fixing ``GroupNorm`` (``hpu_ops/norm.py``), with NO ``hpu_ops`` loaded.

1. ``BroadcastOperator`` crash (NeuroRVQ, NeuroRVQTokenizer, DIVER1, Labram):
   Expanding a ``[1, 1, 200]`` parameter along a symbolic dimension derived from a reshaped input
   (``x.reshape(B * 4, 9, 200)``) and concatenating passes batch 16 cleanly, then crashes at batch 128 in
   ``BroadcastOperator::AllocateAndAddSynapseNode`` with
   ``The expanded size of the tensor (9) must match the existing size (200) ... Target sizes: [512, 4, 9]. Tensor sizes: [1, 1, 200]``
   (Habana ``LaunchRecipe`` then aborts the child process).
2. Silent wrong output on ``128 -> 16`` after ``(16, 128)`` (SignalJEPA_PreLocal):
   A pure ``Conv1d + Conv2d + Linear`` module (no ``GroupNorm``) compiled with ``hpu_backend`` across
   ``(16, 128)``, then ``torch._dynamo.reset()`` and recompiled across ``(128, 16)``, matches CPU to ~6e-6 on
   the first three calls and silently returns ~0.42 wrong values on the fourth (``(128, 16)`` batch 16).

Run on Gaudi2:
    PT_HPU_LAZY_MODE=0 python3 repro/issue42_habana_dynamic_shapes.py
"""
import subprocess
import sys

import habana_frameworks.torch.core  # noqa: F401
import habana_frameworks.torch.hpu as hthpu
import torch
from torch import nn


class ExpandAfterReshape(nn.Module):
    """Minimal pure-PyTorch pattern matching NeuroRVQ / Labram / DIVER1 token expand."""

    def __init__(self):
        super().__init__()
        self.tok = nn.Parameter(torch.randn(1, 1, 200))

    def forward(self, x):
        x4 = x.reshape(x.shape[0] * 4, 9, 200)
        t = self.tok.expand(x4.shape[0], -1, -1)
        return torch.cat([t, x4], dim=1)


class MiniPreLocal(nn.Module):
    """Minimal pure-PyTorch pattern matching SignalJEPA_PreLocal (no GroupNorm)."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 8, 10, stride=5, bias=False),
            nn.GELU(),
            nn.Conv1d(8, 16, 3, stride=2, bias=False),
            nn.GELU(),
            nn.Conv1d(16, 32, 3, stride=2, bias=False),
            nn.GELU(),
        )
        self.spatial = nn.Conv2d(19, 4, (1, 1))
        self.head = nn.Linear(32 * 4 * 9, 4)

    def forward(self, x):
        B, C, T = x.shape
        x = x.reshape(B * C, 1, T)
        x = self.conv(x)
        x = x.reshape(B, C, 32, -1)
        x = self.spatial(x).flatten(1)
        return self.head(x)


def run_broadcast_child():
    torch.manual_seed(0)
    m = ExpandAfterReshape().eval().to("hpu")
    c = torch.compile(m, backend="hpu_backend")
    for b in (16, 128):
        with torch.no_grad():
            out = c(torch.randn(b, 4, 9, 200, device="hpu"))
        hthpu.synchronize()
        print(f"broadcast batch {b:3d}: ok {tuple(out.shape)}", flush=True)


def run_broadcast():
    # Habana LaunchRecipe aborts the process after throwing BroadcastOperator's c10::Error, so isolate in a child.
    proc = subprocess.run(
        [sys.executable, __file__, "--child-broadcast"],
        check=False,
        capture_output=True,
        text=True,
    )
    print("=== 1. BroadcastOperator crash (no hpu_ops) ===", flush=True)
    for line in proc.stdout.splitlines():
        if line.startswith("broadcast "):
            print(" ", line, flush=True)
    err_lines = [ln for ln in proc.stderr.splitlines() if "expanded size" in ln or "BroadcastOperator" in ln]
    for ln in err_lines[:4]:
        print(" ", ln, flush=True)
    print(f"  child returncode={proc.returncode}", flush=True)


def run_silent_mismatch():
    print("=== 2. Silent 128->16 mismatch after (16, 128) (no hpu_ops) ===", flush=True)
    torch.manual_seed(0)
    m = MiniPreLocal().eval()
    xc = {b: torch.randn(b, 19, 200) for b in (16, 128)}
    with torch.no_grad():
        ref = {b: m(xc[b]) for b in xc}
    m.to("hpu")
    xh = {b: xc[b].to("hpu") for b in xc}
    for order in ((16, 128), (128, 16)):
        torch._dynamo.reset()
        c = torch.compile(m, backend="hpu_backend")
        for b in order:
            with torch.no_grad():
                out = c(xh[b])
            hthpu.synchronize()
            diff = float((out.cpu() - ref[b]).abs().max())
            print(f"  order {order} batch {b:3d}: max_abs_diff={diff:.3e}", flush=True)


def main():
    if "--child-broadcast" in sys.argv:
        run_broadcast_child()
        return
    print(f"Issue #42 Habana dynamic-shape repro | torch={torch.__version__} | hpu_ops loaded=False", flush=True)
    run_broadcast()
    run_silent_mismatch()


if __name__ == "__main__":
    main()
