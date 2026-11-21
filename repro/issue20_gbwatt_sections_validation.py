"""Minimal standalone reproducer for Issue #20: lazy-mode synStatus 30 [Sections' validation failure].

Root cause (bridge 1.21.4):
  1. Forward computes a broadcast matmul ``root @ x @ root`` from ``(18, 18)`` onto ``(32, 3, 18, 18)`` followed by
     ``Vech``'s trailing 2-D integer indexing ``z[..., ixs[0], ixs[1]]`` (``aten::index.Tensor``).
  2. In backward, ``aten::index.Tensor`` backward creates a DSD (``SHAPE_TENSOR``) recipe, and an
     ``autograd.Function.backward()`` (``matrix_sqrt`` in ``spd_learn``) executes ``s[s <= threshold] = 0``
     (``aten::index_put_`` -> ``aten::_index_put_impl_``).
  3. Habana has no HPU kernel for ``aten::_index_put_impl_``, so it triggers a mid-backward CPU fallback that flushes
     the partial backward lazy graph (``HabanaFusedOpLazy_*``) missing a ``SHAPE_TENSOR`` -> ``synStatus 30``.

Run on Gaudi2 in lazy mode:
  PT_HPU_LAZY_MODE=1 python3 repro/issue20_gbwatt_sections_validation.py            # fails with synStatus 30
  PT_HPU_LAZY_MODE=1 python3 repro/issue20_gbwatt_sections_validation.py --enable   # passes with hpu_ops.enable()
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


class _MaskInBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x * 1.0

    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        s = x.diagonal(dim1=-2, dim2=-1).clone()
        s[s <= 0.5] = 0
        return g + s.sum() * 0


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--enable", action="store_true", help="Call hpu_ops.enable() before running")
    parser.add_argument("--device", default=None, help="Target device (default: hpu if available else cpu)")
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
    use_hpu = device.type == "hpu"
    lazy = use_hpu and os.environ.get("PT_HPU_LAZY_MODE", "0") != "0"

    m = torch.eye(18, device=device, requires_grad=True)
    x = torch.randn(32, 3, 18, 18, device=device)
    ixs = torch.triu_indices(18, 18)

    root = _MaskInBackward.apply(m)
    z = root @ x @ root
    y = z[..., ixs[0], ixs[1]]
    if lazy:
        htcore.mark_step()
    y.sum().backward()
    if lazy:
        htcore.mark_step()
    if use_hpu:
        hthpu.synchronize()
    print(f"OK (device={device.type}, enable={args.enable}, lazy={lazy}, grad_norm={m.grad.norm().item():.6f})")


if __name__ == "__main__":
    main()
