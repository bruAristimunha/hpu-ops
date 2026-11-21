"""On a Gaudi card: the routes past the Jacobi, LU and Padé kernels' sizes (block Jacobi eigh and SVD, the LU and
matrix_exp composites) through the public ops, forward and backward, against torch on the CPU in float64; the bridge
fallback delta and hpu-ops' own CPU route must stay empty. Timing: mean ms of 5 synced calls.
    PT_HPU_LAZY_MODE=0|1 python3 tests/probe_block_routes.py"""
import json
import os
import time

import hpu_ops

hpu_ops.enable()
import habana_frameworks.torch.core as htcore  # noqa: E402
import torch  # noqa: E402
from habana_frameworks.torch.hpu.metrics import metric_global  # noqa: E402

from hpu_ops import linalg  # noqa: E402

LAZY = os.environ.get("PT_HPU_LAZY_MODE", "0") == "1"
CPU = []
for name in ("_on_cpu", "_out_on_cpu"):
    inner = getattr(linalg, name)
    setattr(linalg, name, lambda *a, _i=inner, **k: (CPU.append(1), _i(*a, **k))[1])


def fb():
    return dict(dict(metric_global("cpu_fallback").stats()).get("FallbackOps", {}) or {})


def sync():
    htcore.mark_step()
    torch.hpu.synchronize()


def spd(b, n):
    a = torch.randn(b, n, n)
    return a @ a.mT + n * torch.eye(n)


CASES = {
    "eigh_64": (lambda: [spd(8, 64)], lambda x: torch.linalg.eigh(x), lambda o: o[0].sum() + (o[1] * o[0][..., None, :] @ o[1].mT).sum()),
    "eigvalsh_128": (lambda: [spd(4, 128)], lambda x: torch.linalg.eigvalsh(x), lambda o: o.sum()),
    "svd_64": (lambda: [torch.randn(8, 64, 64) + 16 * torch.eye(64)], lambda x: torch.linalg.svd(x, full_matrices=False),
               lambda o: o[1].sum() + ((o[0] * o[1][..., None, :]) @ o[2]).sum()),
    "svd_tall_full_72x64": (lambda: [torch.randn(4, 72, 64)], lambda x: torch.linalg.svd(x, full_matrices=True),
                            lambda o: o[1].sum()),
    "svdvals_wide_40x96": (lambda: [torch.randn(4, 40, 96)], lambda x: torch.linalg.svdvals(x), lambda o: o.sum()),
    "pinv_64": (lambda: [torch.randn(4, 64, 64) + 16 * torch.eye(64)], lambda x: torch.linalg.pinv(x), lambda o: o.sum()),
    "lu_factor_64": (lambda: [torch.randn(8, 64, 64)], lambda x: torch.linalg.lu_factor(x)[0], lambda o: o.sum()),
    "lu_solve_64": (lambda: [torch.randn(8, 64, 64), torch.randn(8, 64, 3)],
                    lambda a, b: torch.linalg.lu_solve(*torch.linalg.lu_factor(a), b), lambda o: o.sum()),
    "linalg_lu_48": (lambda: [torch.randn(4, 48, 48)], lambda x: torch.linalg.lu(x)[2], lambda o: o.sum()),
    "matrix_exp_64": (lambda: [0.1 * torch.randn(4, 64, 64)], lambda x: torch.linalg.matrix_exp(x), lambda o: o.sum()),
    "solve_triangular_128": (lambda: [torch.linalg.cholesky(spd(4, 128)), torch.randn(4, 128, 3)],
                             lambda a, b: torch.linalg.solve_triangular(a, b, upper=False), lambda o: o.sum()),
}


def as_list(o):
    return [o] if isinstance(o, torch.Tensor) else [t for t in o if isinstance(t, torch.Tensor)]


def main():
    torch.manual_seed(0)
    for name, (make, fn, loss) in CASES.items():
        row = {"case": name, "lazy": LAZY, "bridge": "1.21.4"}
        try:
            ins = make()
            ref_in = [x.double().requires_grad_() for x in ins]
            ref = fn(*ref_in)
            loss(ref).backward()
            h = [x.to("hpu").requires_grad_() for x in ins]
            f0, c0 = fb(), len(CPU)
            out = fn(*h)
            loss(out).backward()
            sync()
            t = time.perf_counter()
            keep = [fn(*[x.detach() for x in h]) for _ in range(5)]
            sync()
            row["ms"] = round((time.perf_counter() - t) / 5 * 1e3, 2)
            del keep
            row["fallback"] = {k: v - f0.get(k, 0) for k, v in fb().items() if v - f0.get(k, 0)}
            row["cpu_route"] = len(CPU) - c0
            errs = []
            for g, r in zip(as_list(out), as_list(ref)):
                if name.startswith(("svd", "eigh")) and g.dim() == 3 and g.shape[-1] == g.shape[-2]:
                    continue  # vectors: sign-free quantities below
                errs.append(float((g.detach().cpu().double() - r.detach()).abs().max() / r.detach().abs().max()))
            gerr = [float((x.grad.cpu().double() - r.grad).abs().max() / r.grad.abs().max()) for x, r in zip(h, ref_in)]
            row.update(fwd_err=max(errs) if errs else None, grad_err=max(gerr))
            row["ok"] = not row["fallback"] and not row["cpu_route"] and max(errs + gerr) < 1e-3
        except Exception as e:  # noqa: BLE001
            row.update(ok=False, error=repr(e)[:300])
        print("RESULT " + json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
