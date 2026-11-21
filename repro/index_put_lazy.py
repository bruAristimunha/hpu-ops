"""Standalone repro: aten.index_put(zeros(b, n, n), [None, rows, cols], g, accumulate=True), the backward of
X[..., rows, cols], on Gaudi2. PT_HPU_LAZY_MODE=1|0 python3 index_put_lazy.py
Cases vary where g comes from and where the graph is cut. Prints max|hpu - cpu| / max|cpu|."""
import os

import habana_frameworks.torch.core as htcore
import torch

LAZY = os.environ.get("PT_HPU_LAZY_MODE", "1") != "0"
HPU = torch.device("hpu")


def sync():
    if LAZY:
        htcore.mark_step()
    torch.hpu.synchronize()


def ip(z, i0, i1, g, acc=True):
    return torch.ops.aten.index_put(z, [None, i0, i1], g, acc)


def cases(b, n, dev, g0, X0):
    i0, i1 = torch.triu_indices(n, n)
    i0d, i1d = i0.to(dev), i1.to(dev)
    z = lambda: torch.zeros(b, n, n, device=dev)  # noqa: E731
    g = g0.to(dev)
    X = X0.to(dev)
    out = {
        "g from host, CPU idx": lambda: ip(z(), i0, i1, g),
        "g from host, dev idx": lambda: ip(z(), i0d, i1d, g),
        "g from host, accumulate=False": lambda: ip(z(), i0, i1, g, False),
        "g = 2*g in graph": lambda: ip(z(), i0, i1, 2 * g),
        "g = X[..., i0, i1] in graph": lambda: ip(z(), i0, i1, X[..., i0, i1]),
        "g = X[..., i0, i1], mark_step between": lambda: (lambda t: (htcore.mark_step() if dev == HPU and LAZY else None,
                                                                    ip(z(), i0, i1, t))[1])(X[..., i0, i1]),
        "index_put on z.permute(1,2,0) leading idx": lambda: torch.ops.aten.index_put(
            z().permute(1, 2, 0), [i0d, i1d], g.t(), True).permute(2, 0, 1),
        "index_put on z.permute(1,2,0).contiguous()": lambda: torch.ops.aten.index_put(
            z().permute(1, 2, 0).contiguous(), [i0d, i1d], g.t().contiguous(), True).permute(2, 0, 1),
        "index_put flat, [None, lin]": lambda: torch.ops.aten.index_put(
            torch.zeros(b, n * n, device=dev), [None, (i0 * n + i1).to(dev)], g, True).view(b, n, n),
        "index_add on flat (n*n) dim": lambda: torch.zeros(b, n * n, device=dev).index_add(
            1, (i0 * n + i1).to(dev), g).view(b, n, n),
    }
    return out


def main():
    torch.manual_seed(0)
    bad = 0
    for b, n in ((32, 12), (32, 22), (8, 12), (2, 12), (1, 12)):
        k = n * (n + 1) // 2
        g0, X0 = torch.randn(b, k), torch.randn(b, n, n)
        ref = {k_: f().double() for k_, f in cases(b, n, "cpu", g0, X0).items()}
        for name, f in cases(b, n, HPU, g0, X0).items():
            r = f()
            sync()
            r = r.cpu().double()
            e = float((r - ref[name]).abs().max() / ref[name].abs().max())
            bad += e > 1e-5
            print(f"lazy={int(LAZY)} b={b:2d} n={n:2d} {name:<44} err={e:.2e} {'WRONG' if e > 1e-5 else 'ok'}",
                  flush=True)
    print(f"SUMMARY lazy={int(LAZY)} wrong={bad}", flush=True)


if __name__ == "__main__":
    main()
