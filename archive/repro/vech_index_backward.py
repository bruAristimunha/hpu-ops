"""Standalone repro: the gradient of X[..., rows, cols] (spd_learn's vech) on Gaudi2 in lazy mode.
    PT_HPU_LAZY_MODE=1 python3 vech_index_backward.py      (and PT_HPU_LAZY_MODE=0 for eager)
No hpu_ops, no spd_learn: torch + habana_frameworks only. Each case prints max|g_hpu - g_cpu| / max|g_cpu| for the
gradient of X, for several heads (what produces the gradient that reaches IndexBackward).
"""
import os

import habana_frameworks.torch.core as htcore
import torch

LAZY = os.environ.get("PT_HPU_LAZY_MODE", "1") != "0"
HPU = torch.device("hpu")


def sync():
    if LAZY:
        htcore.mark_step()
    torch.hpu.synchronize()


def vech(X, scale):
    n = X.shape[-1]
    ixs = torch.triu_indices(n, n)  # on the CPU, as in spd_learn.functional.sym_to_upper
    v = X[..., ixs[0], ixs[1]]
    if scale:
        m = torch.ones_like(v)
        m[..., ixs[0] != ixs[1]] = 2 ** 0.5
        v = v * m
    return v


def head_expand(v, p):  # grad of v = p["w"] expanded over the batch: a stride-0 view
    return (v * p["w"]).sum()


def head_dense(v, p):  # grad of v = a dense (batch, k) tensor
    return (v * p["g"]).sum()


def head_linear_ce(v, p):  # as in the models: Linear + cross_entropy
    return torch.nn.functional.cross_entropy(v @ p["W"].t() + p["b"], p["y"])


HEADS = {"expanded grad": head_expand, "dense grad": head_dense, "linear+CE": head_linear_ce}


def run(X0, params, scale, head, dev):
    X = X0.detach().to(dev).clone().requires_grad_(True)
    p = {k: v.to(dev) for k, v in params.items()}
    out = vech(X, scale)
    head(out, p).backward()
    if dev == HPU:
        sync()
    return out.detach().cpu().double(), X.grad.detach().cpu().double()


def main():
    torch.manual_seed(0)
    bad = 0
    for b, n in ((32, 12), (32, 22), (1, 4)):
        k = n * (n + 1) // 2
        X0 = torch.randn(b, n, n)
        X0 = X0 @ X0.mT
        params = {"w": torch.randn(k), "g": torch.randn(b, k), "W": torch.randn(4, k) / k ** 0.5,
                  "b": torch.randn(4), "y": torch.randint(0, 4, (b,))}
        for scale in (True, False):
            for hname, head in HEADS.items():
                for rep in range(2):
                    oc, gc = run(X0, params, scale, head, "cpu")
                    oh, gh = run(X0, params, scale, head, HPU)
                    eo = float((oh - oc).abs().max() / oc.abs().max())
                    eg = float((gh - gc).abs().max() / gc.abs().max())
                    bad += eg > 1e-4
                    print(f"lazy={int(LAZY)} b={b:2d} n={n:2d} sqrt2_scale={int(scale)} head={hname:<13} rep={rep} "
                          f"out_err={eo:.2e} grad_err={eg:.2e} {'WRONG' if eg > 1e-4 else 'ok'}", flush=True)
    print(f"SUMMARY lazy={int(LAZY)} wrong={bad}", flush=True)


if __name__ == "__main__":
    main()
