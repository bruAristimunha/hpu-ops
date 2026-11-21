"""On a Gaudi card: does strict mode see each CPU execution, per op, eager and lazy, and what does the watch cost?
    PT_HPU_LAZY_MODE=0|1 python3 tests/strict_probe.py
One ``RESULT {json}`` line per check: the rows strict mode reported (source, op, caller) and whether they match what
the bridge's ``cpu_fallback`` metric and hpu_complex's counter saw. Needs hpu_complex for the complex checks."""
import json
import os
import sys
import time
import traceback

import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.hpu as hthpu
import torch
from habana_frameworks.torch.hpu.metrics import metric_global

LAZY = os.environ.get("PT_HPU_LAZY_MODE", "1") != "0"
HPU = torch.device("hpu")


def sync():
    if LAZY:
        htcore.mark_step()
    hthpu.synchronize()


def bridge():
    return dict(dict(metric_global("cpu_fallback").stats()).get("FallbackOps", {}) or {})


def emit(**kw):
    print("RESULT " + json.dumps(kw, default=str), flush=True)


def rows(hpu_ops):
    return [dict(source=r[0], op=r[1], tensors=r[2], n=r[3], where=r[4].split("/")[-1]) for r in hpu_ops.strict_report()]


def check(name, hpu_ops, fn, expect_source=None):
    hpu_ops.strict_reset()
    b0 = bridge()
    try:
        with hpu_ops.strict("report"):
            fn()
            sync()
        got = rows(hpu_ops)
        b1 = bridge()
        fell = {k: b1[k] - b0.get(k, 0) for k in b1 if b1[k] - b0.get(k, 0)}
        seen = sum(r["n"] for r in got if r["source"] == "bridge fallback")
        ok = (expect_source is None or any(r["source"] == expect_source for r in got)) and seen == sum(fell.values())
        emit(check=name, lazy=LAZY, ok=ok, bridge_delta=fell, rows=got)
    except Exception as e:  # noqa: BLE001
        emit(check=name, lazy=LAZY, ok=False, error=repr(e)[:300], where=traceback.format_exc()[-600:])
    try:  # raise mode: the error names the op and the caller
        with hpu_ops.strict("raise", allow=("cpu op",)):
            fn()
            sync()
        emit(check=name + " raise", lazy=LAZY, raised=False)
    except hpu_ops.CPUExecutionError as e:
        emit(check=name + " raise", lazy=LAZY, raised=True, message=str(e)[:400])
    except Exception as e:  # noqa: BLE001
        emit(check=name + " raise", lazy=LAZY, raised="other", message=repr(e)[:400])


class SvdInBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x * 2

    @staticmethod
    def backward(ctx, g):
        return torch.linalg.svdvals(g.reshape(-1, 4, 4).cpu().to(g.device)).sum() * g  # a host copy, then svdvals


def main():
    import hpu_ops

    try:
        import hpu_complex

        hpu_complex.enable()
        has_complex = True
    except Exception as e:  # noqa: BLE001
        print("no hpu_complex:", repr(e)[:200])
        has_complex = False
    hpu_ops.enable()
    print("PLUGINS", hpu_ops.__file__, "hpu_complex", has_complex and hpu_complex.__file__, "lazy", LAZY, flush=True)
    x = torch.randn(8, 4, 4, device=HPU)

    # a bridge fallback Habana has (svd of a 3-D batch of 4x4 goes through hpu_ops' route when declined; use an op
    # Habana has no kernel for and hpu_ops leaves alone)
    check("bridge: special.zeta", hpu_ops, lambda: torch.special.zeta(x.abs() + 1.5, x.abs() + 1.0), "bridge fallback")
    check("bridge: cummin", hpu_ops, lambda: torch.cummin(x, -1), None)
    check("hpu_ops route: float64 inv", hpu_ops, lambda: torch.linalg.inv(x.double() + 4 * torch.eye(4, device=HPU,
                                                                                    dtype=torch.float64)),
          "hpu_ops route")
    check("host sync: item", hpu_ops, lambda: x.sum().item(), "host sync")
    check("host copy: cpu()", hpu_ops, lambda: x.cpu(), "host copy")

    def backward_case():
        y = x.clone().requires_grad_()
        SvdInBackward.apply(y).sum().backward()

    check("backward thread: host copy + svdvals", hpu_ops, backward_case, "host copy")
    if has_complex:
        z = torch.randn(8, 4, 4, dtype=torch.complex64).to(HPU)
        check("hpu_complex: small complex64 bmm", hpu_ops, lambda: z @ z, "hpu_complex")
        check("hpu_complex: complex128", hpu_ops, lambda: (z.to(torch.complex128) * 2).abs(), "hpu_complex")

    # the watch's cost: 200 small adds, eager or lazy, with and without the per-op watch
    a = torch.randn(64, device=HPU)

    def adds():
        b = a
        for _ in range(200):
            b = b + 1.0
        sync()
        return b

    for _ in range(3):
        adds()
    t = time.perf_counter()
    for _ in range(5):
        adds()
    base = (time.perf_counter() - t) / 1000
    with hpu_ops.strict("report"):
        adds()
        t = time.perf_counter()
        for _ in range(5):
            adds()
        watched = (time.perf_counter() - t) / 1000
    emit(check="watch cost per op", lazy=LAZY, us_plain=round(base * 1e6, 2), us_watched=round(watched * 1e6, 2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
