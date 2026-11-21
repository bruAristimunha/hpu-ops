"""Gaudi check of hpu_ops.small's bincount (issue #21) after hpu_ops.enable(), eager or lazy (PT_HPU_LAZY_MODE):
exact counts against ATen on the CPU (heavy bins, chunk boundaries, dtypes, minlength), the cases left to Habana's
kernel, the negative-input error; CPU fallbacks. Prints one line per case and ALL OK at the end."""
import habana_frameworks.torch.core as htcore
import torch
from habana_frameworks.torch.hpu.metrics import metric_global

import hpu_ops

hpu_ops.enable()
hpu = torch.device("hpu")
calls = hpu_ops._STATE["calls"]
FAILED = []


def fallbacks():
    return sum((dict(metric_global("cpu_fallback").stats()).get("FallbackOps", {}) or {}).values())


def close(a, b, exact=False):
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    if exact:
        return torch.equal(a, b)
    return torch.allclose(a.double(), b.double(), rtol=1e-5, atol=1e-5, equal_nan=True)


def bincount_case(name, x, taken, **kw):
    ref = torch.bincount(x, **kw)
    calls.clear()
    before = fallbacks()
    out = torch.bincount(x.to(hpu), **{k: v.to(hpu) if isinstance(v, torch.Tensor) else v for k, v in kw.items()})
    htcore.mark_step()
    ok = close(out.cpu(), ref, exact=ref.dtype == torch.int64) and calls["bincount"] == taken
    fb = fallbacks() - before
    print(f"bincount  {name:40s} {'ok' if ok else 'FAIL'} taken {calls['bincount']} fb {fb}", flush=True)
    if not ok:
        FAILED.append(name)


def main():
    torch.manual_seed(0)
    n = 1 << 16
    bincount_case("65536 64 bins", torch.randint(0, 64, (n,)), 1)
    heavy = torch.randint(0, 256, (409600,))
    heavy[:300000] = 7
    bincount_case("409600 256 bins heavy bin", heavy, 1)
    bincount_case("65536 int32 minlength 200", torch.randint(0, 10, (n,), dtype=torch.int32), 1, minlength=200)
    bincount_case("65536 one bin", torch.zeros(n, dtype=torch.long), 1)
    for bins in (257, 300, 511, 512, 513, 1000, 4095, 4096):
        bincount_case(f"arange({bins}) repeated, chunked", torch.arange(bins).repeat(n // bins + 1), 1)
    bincount_case("131072 1000 bins random", torch.randint(0, 1000, (2 * n,)), 1)
    bincount_case("65536 5000 bins (native)", torch.randint(0, 5000, (n,)), 0)
    bincount_case("4096 (native)", torch.randint(0, 64, (4096,)), 0)
    bincount_case("65536 weights (native)", torch.randint(0, 10, (n,)), 0, weights=torch.rand(n))
    bincount_case("empty minlength 4 (native)", torch.zeros(0, dtype=torch.long), 0, minlength=4)
    neg = torch.randint(0, 10, (n,))
    neg[3] = -1
    try:
        torch.bincount(neg.to(hpu))
        htcore.mark_step()
        print("bincount  negative input                             FAIL no error", flush=True)
        FAILED.append("negative")
    except RuntimeError as e:
        print(f"bincount  negative input raises                      ok ({str(e)[:60]})", flush=True)
    print("ALL OK" if not FAILED else f"FAILED {FAILED}", flush=True)


if __name__ == "__main__":
    main()
