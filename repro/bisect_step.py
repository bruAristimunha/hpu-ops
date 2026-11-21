"""Bisect a lazy-mode train step: a mark_step + synchronize after every leaf module's forward (and, with --bwd,
after every leaf module's backward), printing where it got to. The last line before an abort names the culprit.
    PT_HPU_LAZY_MODE=1 python3 repro/bisect_step.py --model TensorCSPNet [--ch 22] [--k none|resolve] [--bwd]"""
import argparse
import os
import sys

import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.hpu as hthpu
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests"))
from model_steps import batch  # noqa: E402


def flush(tag):
    htcore.mark_step()
    hthpu.synchronize()
    print("PASSED", tag, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="TensorCSPNet")
    parser.add_argument("--ch", type=int, default=22)
    parser.add_argument("--k", choices=("none", "resolve"), default="none")
    parser.add_argument("--bwd", action="store_true")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--no-enable", action="store_true")
    args = parser.parse_args()
    if not args.no_enable:
        import hpu_complex
        import hpu_ops
        hpu_ops.enable()
        hpu_complex.enable()
    from spd_diag import protocol as P
    info, X, y = batch(args.ch, 32)
    k = P.resolve_k(args.model, args.ch, len(info["classes"]), info["sfreq"], None) if args.k == "resolve" else None
    torch.manual_seed(0)
    model = P.build(args.model, args.ch, info, k).to("hpu")
    print("MODEL", type(model).__name__, model.__class__.__mro__[1].__name__, flush=True)
    xb = model.fixed_input(X).to("hpu") if hasattr(model, "fixed_input") else X.to("hpu")
    yb = y.to("hpu")
    for name, mod in model.named_modules():
        if len(list(mod.children())) == 0:
            mod.register_forward_hook(lambda m, i, o, name=name: flush(f"fwd {name} {type(m).__name__}"))
            if args.bwd:
                mod.register_full_backward_hook(lambda m, gi, go, name=name: flush(f"bwd {name} {type(m).__name__}"))
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    flush("setup")
    for s in range(args.steps):
        opt.zero_grad(set_to_none=True)
        out = model(xb)
        out = out[0] if isinstance(out, tuple) else out
        loss = torch.nn.functional.cross_entropy(out, yb)
        flush(f"step{s} forward+loss {float(loss.detach().cpu()):.6f}")
        loss.backward()
        flush(f"step{s} backward")
        opt.step()
        flush(f"step{s} optimizer")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
