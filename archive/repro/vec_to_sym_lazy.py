"""Standalone: fwd+bwd of spd_learn's vec_to_sym (in-place index writes and reads on a zeros matrix), and of
vech(vec_to_sym(v)), on Gaudi2. PT_HPU_LAZY_MODE=1 python3 vec_to_sym_lazy.py [--hpu-ops] [--variant select]"""
import argparse
import math
import os

import torch

LAZY = os.environ.get("PT_HPU_LAZY_MODE", "1") != "0"


def vec_to_sym(v):
    n = int((math.sqrt(1 + 8 * v.shape[-1]) - 1) / 2)
    ixs = torch.triu_indices(n, n)
    od = ixs[0] != ixs[1]
    X = torch.zeros((*v.shape[:-1], n, n), device=v.device, dtype=v.dtype)
    X[..., ixs[0], ixs[1]] = v
    X[..., ixs[0, od], ixs[1, od]] /= math.sqrt(2)
    X[..., ixs[1, od], ixs[0, od]] = X[..., ixs[0, od], ixs[1, od]]
    return X


def vech(X):
    n = X.shape[-1]
    ixs = torch.triu_indices(n, n)
    v = X[..., ixs[0], ixs[1]]
    m = torch.ones_like(v)
    m[..., ixs[0] != ixs[1]] = math.sqrt(2)
    return v * m


def _cut(t):
    if LAZY and t.device.type == "hpu":
        import habana_frameworks.torch.core as htcore
        htcore.mark_step()
    return t


class Fwd(Exception):
    pass


CASES = {
    "vech(vec_to_sym) fwd only": lambda v, W: (_ for _ in ()).throw(Fwd(_cut(vech(vec_to_sym(v)).sum()))),
    "vech(cut(vec_to_sym))": lambda v, W: (vech(_cut(vec_to_sym(v))) * v).sum(),
    "vech(vec_to_sym(v).clone())": lambda v, W: (vech(vec_to_sym(v).clone()) * v).sum(),
    "vech(vec_to_sym(v) + 0)": lambda v, W: (vech(vec_to_sym(v) + 0) * v).sum(),
    "vec_to_sym": lambda v, W: (vec_to_sym(v) * W).sum(),
    "vech(vec_to_sym)": lambda v, W: (vech(vec_to_sym(v)) * v).sum(),
    "vech(expm(vec_to_sym))": lambda v, W: (vech(torch.matrix_exp(vec_to_sym(v) * 0.1)) * v).sum(),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hpu-ops", action="store_true")
    a = ap.parse_args()
    if a.hpu_ops:
        import hpu_ops
        hpu_ops.enable()
    import habana_frameworks.torch.core as htcore
    torch.manual_seed(0)
    b, n = 32, 12
    v0, W0 = torch.randn(b, n * (n + 1) // 2), torch.randn(b, n, n)
    for name, f in CASES.items():
        out = {}
        for dev in ("cpu", "hpu"):
            v = v0.detach().clone().to(dev).requires_grad_(True)
            try:
                f(v, W0.to(dev)).backward()
                if dev == "hpu":
                    if LAZY:
                        htcore.mark_step()
                    torch.hpu.synchronize()
                out[dev] = v.grad.cpu().double()
            except Fwd as e:
                try:
                    float(e.args[0].cpu())
                    out[dev] = "forward ok"
                except Exception as e2:  # noqa: BLE001
                    out[dev] = repr(e2)[:150]
            except Exception as e:  # noqa: BLE001
                out[dev] = repr(e)[:150]
        e = out["hpu"] if isinstance(out["cpu"], str) else (float((out["hpu"] - out["cpu"]).abs().max() / out["cpu"].abs().max())
             if isinstance(out["hpu"], torch.Tensor) else out["hpu"])
        print(f"lazy={int(LAZY)} hpu_ops={int(a.hpu_ops)} {name:<24} grad_err={e}", flush=True)


if __name__ == "__main__":
    main()
