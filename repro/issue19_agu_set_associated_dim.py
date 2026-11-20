"""Minimal standalone reproducer for Issue #19: gaudi2_agu_config setAssociatedDimAndSize (size - 1 <= uint8 max).

Root cause (bridge 1.21.4):
  Gaudi2's MME AGU rejects single-op Conv graph lowering when a spatial kernel dimension exceeds 256:
  1. ``STaRNet(64)`` ``conv_spatial_fusion``: ``Conv2d(1, 8, kernel_size=(351, 1))`` on ``(32, 1, 351, 1000)``
     fails in eager, lazy, and ``torch.compile``.
  2. ``TensorCSPNet(22, dims=(22, 11, 11, 11))`` ``temporal_block``: ``Conv2d(1, 16, kernel_size=(4, 1089),
     stride=(1, 121))`` on ``(32, 1, 4, 1089)`` fails in eager mode (async ``EagerLoweringTask`` surfaces the
     forward compile error during backward).

Run on Gaudi2:
  python3 repro/issue19_agu_set_associated_dim.py --case starnet64            # fails with setAssociatedDimAndSize
  python3 repro/issue19_agu_set_associated_dim.py --case tensorcsp22          # fails with setAssociatedDimAndSize
  python3 repro/issue19_agu_set_associated_dim.py --enable                    # both pass with hpu_ops.enable()
"""
import argparse
import os

try:
    import habana_frameworks.torch.core as htcore
    import habana_frameworks.torch.hpu as hthpu
except ImportError:
    htcore = None
    hthpu = None
import torch
import torch.nn.functional as F


def run_case(name, device, lazy):
    if name == "starnet64":
        x = torch.randn(32, 1, 351, 1000, device=device, requires_grad=True)
        w = torch.randn(8, 1, 351, 1, device=device, requires_grad=True)
        b = None
        stride = (1, 1)
    elif name == "tensorcsp22":
        x = torch.randn(32, 1, 4, 1089, device=device, requires_grad=True)
        w = torch.randn(16, 1, 4, 1089, device=device, requires_grad=True)
        b = torch.randn(16, device=device, requires_grad=True)
        stride = (1, 121)
    else:
        raise ValueError(f"Unknown case: {name}")

    y = F.conv2d(x, w, b, stride=stride)
    if lazy:
        htcore.mark_step()
    y.square().mean().backward()
    if lazy:
        htcore.mark_step()
    if device.type == "hpu":
        hthpu.synchronize()
    print(f"OK ({name}, device={device.type}, lazy={lazy}, out={tuple(y.shape)}, w_grad={w.grad.norm().item():.6f})")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--enable", action="store_true", help="Call hpu_ops.enable() before running")
    parser.add_argument("--device", default=None, help="Target device (default: hpu if available else cpu)")
    parser.add_argument("--case", choices=("both", "starnet64", "tensorcsp22"), default="both",
                        help="Which minimal conv case to run (default: both)")
    args = parser.parse_args()

    if args.enable:
        try:
            import hpu_ops
        except ImportError:
            import sys
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            import hpu_ops
        hpu_ops.enable()

    torch.manual_seed(0)
    has_hpu = hthpu is not None and hthpu.is_available()
    device = torch.device(args.device or ("hpu" if has_hpu else "cpu"))
    lazy = device.type == "hpu" and os.environ.get("PT_HPU_LAZY_MODE", "0") != "0"

    cases = ("starnet64", "tensorcsp22") if args.case == "both" else (args.case,)
    for case in cases:
        run_case(case, device, lazy)


if __name__ == "__main__":
    main()
