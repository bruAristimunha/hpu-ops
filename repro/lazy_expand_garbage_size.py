"""Gaudi lazy mode (bridge 1.21.4, torch 2.6): the backward of a dim reduction of a 1-D tensor to a 0-d tensor that is
then summed fails at once with `expand(HPUFloatType{[1, 1]}, size=[<garbage>]): the number of sizes provided (1) must
be greater or equal to the number of dimensions in the tensor (2)`. Eager mode and the CPU are fine.

    PT_HPU_LAZY_MODE=1 python lazy_expand_garbage_size.py sum   # one case per process: a failed graph can poison the next

Measured on Gaudi2 (one process per case, 3 iterations of forward + backward, a mark_step after each):

    fails, lazy:   x.sum(0).sum(), x.mean(0).sum(), x.exp().sum(0).log().sum(), ATen's logsumexp composite
    passes, lazy:  x.sum(0) alone, x.sum(), x.amax(0).sum(), x.sum(0, keepdim=True).reshape(()).sum(),
                   torch.logsumexp(x, 0).sum() (Habana's kernel), x.view(1, -1).exp().sum(1).log().sum()
    passes, eager: x.sum(0).sum(), the logsumexp composite

Workaround: reduce with keepdim=True and reshape to the result's shape (hpu_ops.perf.logsumexp does).
"""
import sys

import torch
import habana_frameworks.torch.core as htcore

CASES = {
    "sum": lambda x: x.sum(0),
    "exp_sum": lambda x: x.exp().sum(0),
    "exp_sum_log": lambda x: x.exp().sum(0).log(),
    "sub_keepdim_max": lambda x: (x - x.amax(0, keepdim=True).detach()).sum(0),
    "plus_squeezed_max": lambda x: x.sum(0) + x.amax(0, keepdim=True).detach().squeeze(0),
    "logsumexp_composite": lambda x: (lambda m: (x - m).exp().sum(0).log() + m.squeeze(0))(
        x.amax(0, keepdim=True).detach()),
    "native_logsumexp": lambda x: torch.logsumexp(x, 0),
    "softmax_like_2d": lambda x: x.view(1, -1).exp().sum(1).log().sum(),
    "sum_all": lambda x: x.sum(),
    "sum_keepdim_reshape": lambda x: x.sum(0, keepdim=True).reshape(()),
    "composite_keepdim": lambda x: (lambda m: ((x - m).exp().sum(0, keepdim=True).log() + m).reshape(()))(
        x.amax(0, keepdim=True).detach()),
    "amax_0d": lambda x: x.amax(0),
    "mean_0d": lambda x: x.mean(0),
}
DIRECT = {"sum_backward_direct"}


def run(name, n=22):
    x0 = torch.randn(n).to("hpu")
    for i in range(3):
        try:
            x = x0.detach().requires_grad_()
            if name == "sum_backward_direct":
                y = x.sum(0)
                (g,) = torch.autograd.grad(y, x)
            else:
                y = CASES[name](x)
                (g,) = torch.autograd.grad(y.sum(), x)
            htcore.mark_step()
            g.cpu()
        except Exception as e:  # noqa: BLE001
            return f"ERROR at iteration {i}: {str(e).splitlines()[0][:160]}"
    return "OK"


if __name__ == "__main__":
    for name in sys.argv[1:] or [*CASES, *DIRECT]:
        print(f"{name:22s} {run(name)}", flush=True)
