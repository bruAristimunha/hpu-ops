"""On a Gaudi card: which linalg calls that hpu-ops declines under a functorch transform or forward-mode AD
(``linalg._transformed``) fall back to the CPU in Habana's native op. Per op and transform (vmap, grad, jacrev, jvp,
forward-mode ``dual``): the bridge ``cpu_fallback`` delta, hpu-ops' own CPU route, and the error against torch on the
CPU in float64.
    PT_HPU_LAZY_MODE=0|1 python3 tests/probe_functorch.py"""
import json
import os

import hpu_ops

hpu_ops.enable()
import habana_frameworks.torch.core as htcore  # noqa: E402
import torch  # noqa: E402
import torch.autograd.forward_ad as fwAD  # noqa: E402
from habana_frameworks.torch.hpu.metrics import metric_global  # noqa: E402
from torch.func import grad, jacrev, jvp, vmap  # noqa: E402

from hpu_ops import linalg  # noqa: E402

LAZY = os.environ.get("PT_HPU_LAZY_MODE", "0") == "1"
CPU = []
for name in ("_on_cpu", "_out_on_cpu"):
    inner = getattr(linalg, name)
    setattr(linalg, name, lambda *a, _i=inner, **k: (CPU.append(1), _i(*a, **k))[1])
L = torch.linalg


def fb():
    return dict(dict(metric_global("cpu_fallback").stats()).get("FallbackOps", {}) or {})


def spd(n):
    a = torch.randn(n, n)
    return a @ a.mT + n * torch.eye(n)


N = 8
OPS = {  # name: (input, scalar function of one matrix)
    "solve": (lambda: torch.randn(N, N) + 4 * torch.eye(N), lambda a: L.solve(a, torch.ones(N, 1, device=a.device, dtype=a.dtype)).sum()),
    "inv": (lambda: torch.randn(N, N) + 4 * torch.eye(N), lambda a: L.inv(a).sum()),
    "det": (lambda: torch.randn(N, N) + 4 * torch.eye(N), lambda a: L.det(a)),
    "slogdet": (lambda: torch.randn(N, N) + 4 * torch.eye(N), lambda a: L.slogdet(a)[1]),
    "solve_triangular": (lambda: spd(N).tril(), lambda a: L.solve_triangular(a, torch.ones(N, 1, device=a.device, dtype=a.dtype), upper=False).sum()),
    "cholesky_solve": (lambda: L.cholesky(spd(N)), lambda a: torch.cholesky_solve(torch.ones(N, 1, device=a.device, dtype=a.dtype), a).sum()),
    "eigh": (lambda: spd(N), lambda a: L.eigvalsh(a).sum()),
    "svdvals": (lambda: torch.randn(N, N) + 4 * torch.eye(N), lambda a: L.svdvals(a).sum()),
    "matrix_exp": (lambda: 0.1 * torch.randn(N, N), lambda a: L.matrix_exp(a).sum()),
    "lu_factor": (lambda: torch.randn(N, N), lambda a: L.lu_factor(a)[0].sum()),
}


def transforms(f, x):
    t = torch.ones_like(x)
    return {
        "vmap": lambda: vmap(f)(x.expand(3, *x.shape).contiguous()),
        "grad": lambda: grad(f)(x),
        "jacrev": lambda: jacrev(f)(x),
        "jvp": lambda: jvp(f, (x,), (t,))[1],
        "forward_ad": lambda: _dual(f, x, t),
    }


def _dual(f, x, t):
    with fwAD.dual_level():
        return fwAD.unpack_dual(f(fwAD.make_dual(x, t))).tangent


def main():
    torch.manual_seed(0)
    for op, (make, f) in OPS.items():
        x = make()
        refs = transforms(f, x.double())
        for name, run in transforms(f, x.to("hpu")).items():
            row = {"op": op, "transform": name, "lazy": LAZY, "bridge": "1.21.4"}
            try:
                want = refs[name]()
                f0, c0 = fb(), len(CPU)
                got = run()
                htcore.mark_step()
                torch.hpu.synchronize()
                row["fallback"] = {k: v - f0.get(k, 0) for k, v in fb().items() if v - f0.get(k, 0)}
                row["cpu_route"] = len(CPU) - c0
                row["err"] = float((got.cpu().double() - want).abs().max() / want.abs().max().clamp_min(1e-30))
            except Exception as e:  # noqa: BLE001
                row["error"] = repr(e)[:200]
            print("RESULT " + json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
