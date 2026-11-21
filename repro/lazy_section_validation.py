"""Standalone repro (torch + habana_frameworks only, no hpu_ops): in lazy mode, one graph that writes into a zeros
matrix through advanced-index puts, reads it back through an advanced index and builds a boolean-mask put fails to
launch on Gaudi2 (bridge 1.21.4): "synLaunch failed. synStatus=synStatus 30 [Sections' validation failure]".
Eager mode, or a mark_step after the writes, runs. This is spd_learn's vech(vec_to_sym(v)) (SPNN's SPDLinear).
    PT_HPU_LAZY_MODE=1 python3 lazy_section_validation.py"""
import os

import habana_frameworks.torch.core as htcore
import torch

LAZY = os.environ.get("PT_HPU_LAZY_MODE", "1") != "0"


def write(v, n, ixs, read_back):
    X = torch.zeros(v.shape[0], n, n, device=v.device)
    X[..., ixs[0], ixs[1]] = v
    if read_back:  # vec_to_sym's symmetrization: an index read of X, put back at the transposed places
        od = ixs[0] != ixs[1]
        X[..., ixs[1, od], ixs[0, od]] = X[..., ixs[0, od], ixs[1, od]]
    return X


def read(X, ixs, mask):
    out = X[..., ixs[0], ixs[1]]
    if mask:  # vech's sqrt(2) scale: a boolean-mask put on a ones tensor
        m = torch.ones_like(out)
        m[..., ixs[0] != ixs[1]] = 2 ** 0.5
        out = out * m
    return out


def main():
    torch.manual_seed(0)
    n = 12
    ixs = torch.triu_indices(n, n)
    v0 = torch.randn(32, n * (n + 1) // 2)
    for read_back in (False, True):
        for mask in (False, True):
            for cut in (False, True):
                want = read(write(v0, n, ixs, read_back), ixs, mask)
                try:
                    X = write(v0.to("hpu"), n, ixs, read_back)
                    if cut and LAZY:
                        htcore.mark_step()
                    got = read(X, ixs, mask).cpu()
                    res = f"ok, err={float((got - want).abs().max()):.1e}"
                except RuntimeError as e:
                    res = "FAILS: " + str(e).splitlines()[0][:110]
                print(f"lazy={int(LAZY)} read_back={int(read_back)} mask={int(mask)} mark_step_between={int(cut)} {res}",
                      flush=True)


if __name__ == "__main__":
    main()
