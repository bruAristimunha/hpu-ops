"""On a Gaudi card: hpu_ops' indexing rewrites on odd shapes and layouts, eager or lazy, and compiled (eager only),
against the CPU. PT_HPU_LAZY_MODE=0|1 python3 tests/adversarial_index.py
Each case: output and input gradient of a sum-of-squares-weighted loss, max|hpu - cpu| / max|cpu|; two runs in a row;
bridge fallbacks counted. One CASE line each, SUMMARY last."""
import os

import habana_frameworks.torch.core as htcore
import torch
from habana_frameworks.torch.hpu.metrics import metric_global

import hpu_ops

LAZY = os.environ.get("PT_HPU_LAZY_MODE", "1") != "0"


def vech(x):
    r, c = torch.triu_indices(x.shape[-1], x.shape[-1])
    v = x[..., r, c]
    m = torch.ones_like(v)
    m[..., r != c] = 2 ** 0.5
    return v * m


CASES = {
    "n=1": (vech, lambda: torch.randn(4, 1, 1)),
    "batch 1": (vech, lambda: torch.randn(1, 22, 22)),
    "batch 1024 n=22": (vech, lambda: torch.randn(1024, 22, 22)),
    # layouts: (base, view); the view is taken on the device, of the base that gets the gradient
    "offset view": (vech, lambda: (torch.randn(40, 12, 12), lambda b: b[8:])),
    "stride-0 batch": (vech, lambda: (torch.randn(12, 12), lambda b: b[None].expand(32, 12, 12))),
    "4-d transposed": (vech, lambda: (torch.randn(8, 3, 12, 12), lambda b: b.transpose(-1, -2))),
    "extreme values": (vech, lambda: torch.randn(32, 12, 12) * 1e30),
    "middle dim": (lambda x: x[:, torch.tensor([3, 0, 3, 1])], lambda: torch.randn(32, 5, 7)),
    "separated": (lambda x: x[torch.tensor([0, 1, 1]), :, torch.tensor([3, 0, 3])], lambda: torch.randn(4, 9, 5)),
    "device index": (lambda x: x[..., torch.tensor([0, 5, 5], device=x.device)], lambda: torch.randn(32, 6, 6)),
    "negative index": (lambda x: x[..., torch.tensor([-1, 0]), torch.tensor([2, -3])], lambda: torch.randn(32, 6, 6)),
    "0-d, direct aten": (lambda x: torch.ops.aten.index.Tensor(x, [None, torch.tensor(1)]), lambda: torch.randn(32, 6, 6)),
    "0-d and 1-d": (lambda x: torch.ops.aten.index.Tensor(x, [None, torch.tensor(-1), torch.tensor([0, 2])]),
                    lambda: torch.randn(32, 6, 6)),
    # PR #24's forms, composed into the same kernel
    "leading CPU index": (lambda x: x[torch.tensor([0, -1, 3])], lambda: torch.randn(32, 6, 6)),
    "separated negative": (lambda x: x[torch.tensor([-1, 0, 2]), :, torch.tensor([1, -2, 4])],
                           lambda: torch.randn(4, 9, 5)),
    "int32 index": (lambda x: x[..., torch.tensor([1, 1, 0], dtype=torch.int32)], lambda: torch.randn(32, 6, 6)),
    "CPU mask after slice": (lambda x: x[:, torch.ones(6, 6, dtype=torch.bool).triu()], lambda: torch.randn(32, 6, 6)),
    "CPU mask leading": (lambda x: x[torch.arange(32) % 3 == 0], lambda: torch.randn(32, 6)),
    "device mask after slice": (lambda x: x[:, torch.ones(6, 6, dtype=torch.bool, device=x.device).triu()],
                                lambda: torch.randn(32, 6, 6)),
}


def step(f, x0, dev, compiled=False):
    x0, view = x0 if isinstance(x0, tuple) else (x0, lambda b: b)
    x = x0.detach().clone().to(dev).requires_grad_(True)
    g = torch.compile(f, backend="hpu_backend") if compiled else f
    out = g(view(x))
    w = torch.linspace(-1, 1, out.numel(), device=dev).reshape(out.shape)
    (out * w).sum().backward()
    if dev == "hpu":
        if LAZY:
            htcore.mark_step()
        torch.hpu.synchronize()
    return out.detach().cpu().double(), x.grad.detach().cpu().double()


def err(a, b):
    """max|a - b| / max|b|; NaN (a failure: NaN < tol is False) when either has a NaN."""
    if torch.isnan(a).any() or torch.isnan(b).any():
        return float("nan")
    return float((a - b).abs().max() / b.abs().max().clamp_min(1e-30)) if b.numel() else 0.0


def main():
    hpu_ops.enable()
    torch.manual_seed(0)
    bad = 0
    modes = [False] if LAZY else [False, True]
    for name, (f, make) in CASES.items():
        x0 = make()
        x64 = (x0[0].double(), x0[1]) if isinstance(x0, tuple) else x0.double()
        oc, gc = step(f, x64, "cpu")
        for compiled in modes:
            fb0 = dict(dict(metric_global("cpu_fallback").stats()).get("FallbackOps", {}) or {})
            hpu_ops._STATE["calls"].clear()
            try:
                res = [step(f, x0, "hpu", compiled) for _ in range(2)]
                es = [err(o, oc) for o, _ in res] + [err(g, gc) for _, g in res]
                e = float("nan") if any(v != v for v in es) else max(es)  # max() would drop a NaN
                fb1 = dict(dict(metric_global("cpu_fallback").stats()).get("FallbackOps", {}) or {})
                fbd = {k: fb1[k] - fb0.get(k, 0) for k in fb1 if fb1[k] - fb0.get(k, 0)}
                ok = e < 1e-5 and not fbd
                msg = f"err={e:.2e} fallbacks={fbd} rewrites={dict(hpu_ops._STATE['calls'])}"
            except Exception as ex:  # noqa: BLE001
                ok, msg = False, "ERROR " + repr(ex)[:200]
            bad += not ok
            print(f"CASE lazy={int(LAZY)} compiled={int(compiled)} {name:<18} {'ok' if ok else 'FAIL'} {msg}",
                  flush=True)
    print(f"SUMMARY lazy={int(LAZY)} failing={bad}", flush=True)


if __name__ == "__main__":
    main()
