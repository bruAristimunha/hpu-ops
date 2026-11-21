"""Scope of Gaudi's histc failures (Gaudi2, bridge 1.21.4). One case per process (some abort it):
    PT_HPU_LAZY_MODE=0|1 python repro/probe_histc_scope.py <case> [args]
cases: native <bins> <min> <max> [dtype] [data]   data: int (randint 0..63), unif (uniform -3..3), nan, inf, const
       compile <bins> <min> <max>
       arith                      bitwise check of the card's float32 (x - L) * bins / (R - L) against the CPU
Prints OK / BAD against torch.histc on the CPU (float64 input is compared as float32)."""
import os
import sys

import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.hpu as hthpu
import torch

LAZY = os.environ.get("PT_HPU_LAZY_MODE", "1") != "0"
hpu = torch.device("hpu")
torch.manual_seed(0)


def data(kind, dtype):
    if kind == "int":
        x = torch.randint(0, 64, (4096,)).float()
    elif kind == "unif":
        x = torch.rand(4096) * 6 - 3
    elif kind == "nan":
        x = torch.rand(4096) * 6 - 3
        x[::7] = float("nan")
    elif kind == "inf":
        x = torch.rand(4096) * 6 - 3
        x[::7] = float("inf")
        x[1::7] = -float("inf")
    elif kind == "const":
        x = torch.full((4096,), 2.5)
    return x.to(dtype)


def sync():
    if LAZY:
        htcore.mark_step()
    hthpu.synchronize()


def report(tag, got, ref):
    got = got.cpu().float()
    ref = ref.float()
    same = got.shape == ref.shape and torch.equal(got, ref)
    diff = "" if same or got.shape != ref.shape else f" bins differing {int((got != ref).sum())}, sum {got.sum()} / {ref.sum()}"
    print(f"{'OK ' if same else 'BAD'} {tag}{diff}", flush=True)


def main():
    case = sys.argv[1]
    if case in ("native", "compile"):
        num = lambda a: float(a) if "." in a or "e" in a else int(a)  # noqa: E731  (Python int vs float bound)
        bins, lo, hi = int(sys.argv[2]), num(sys.argv[3]), num(sys.argv[4])
        dtype = getattr(torch, sys.argv[5]) if len(sys.argv) > 5 else torch.float32
        kind = sys.argv[6] if len(sys.argv) > 6 else "unif"
        x = data(kind, dtype)
        ref = torch.histc(x.float() if dtype == torch.float64 else x, bins, lo, hi) if not (
            kind in ("nan", "inf") and lo == hi) else None
        f = lambda t: torch.histc(t, bins, lo, hi)  # noqa: E731
        if case == "compile":
            import habana_frameworks.torch.dynamo  # noqa: F401
            f = torch.compile(f, backend="hpu_backend")
        tag = f"{case} {'lazy' if LAZY else 'eager'} bins={bins} [{lo}, {hi}] {dtype} {kind} ({type(lo).__name__} bounds)"
        try:
            got = f(x.to(hpu))
            sync()
            if ref is None:
                print("GOT", tag, got.cpu()[:4].tolist())
            else:
                report(tag, got, ref)
        except Exception as e:  # noqa: BLE001
            print("RAISE", tag, str(e).splitlines()[0][:160], flush=True)
    elif case == "arith":
        n = 1 << 20
        for lo, hi, bins in ((-3.0, 3.0, 64), (-0.5, 63.5, 64), (0.1, 0.7, 1000), (-1e-3, 2e-3, 7), (0, 1, 3)):
            x = torch.rand(n) * (hi - lo) * 1.2 + lo - 0.1 * (hi - lo)
            edges = torch.linspace(lo, hi, bins + 1)
            x = torch.cat([x, edges, torch.nextafter(edges, edges + 1), torch.nextafter(edges, edges - 1)])
            L, R = torch.tensor(lo, dtype=torch.float32), torch.tensor(hi, dtype=torch.float32)
            ref = (x - L) * float(bins) / (R - L)
            h = x.to(hpu)
            got = (h - L.to(hpu)) * float(bins) / (R - L).to(hpu)
            got2 = (h - lo) * float(bins) / (hi - lo)
            sync()
            got, got2 = got.cpu(), got2.cpu()
            bad = int((got != ref).sum())
            badtrunc = int((got.trunc() != ref.trunc()).sum())
            bad2 = int((got2.trunc() != ref.trunc()).sum())
            print(f"arith [{lo}, {hi}] bins={bins}: {bad} of {x.numel()} values differ bitwise, {badtrunc} "
                  f"truncations differ (tensor scalars), {bad2} (python scalars)", flush=True)
            hc = torch.histc(x, bins, lo, hi)
            ok_native = float(lo).is_integer() and float(hi).is_integer() and bins < 300
            hg = torch.histc(h, bins, int(lo), int(hi)) if ok_native else None
            if hg is not None:
                sync()
                print(f"   native histc integer bounds: bins differing {int((hg.cpu() != hc).sum())}", flush=True)


def bucketize():
    for n_thr in (63, 255, 4095):
        thr = torch.sort(torch.rand(n_thr) * 6 - 3).values
        x = torch.cat([torch.rand(1 << 20) * 7 - 3.5, thr, torch.nextafter(thr, thr + 1), torch.nextafter(thr, thr - 1),
                       torch.tensor([float("nan"), float("inf"), -float("inf")])])
        for right in (False, True):
            ref = torch.bucketize(x, thr, right=right)
            got = torch.bucketize(x.to(hpu), thr.to(hpu), right=right)
            got2 = torch.searchsorted(thr.to(hpu), x.to(hpu), right=right)
            sync()
            print(f"bucketize thr={n_thr} right={right}: {int((got.cpu() != ref).sum())} differ, searchsorted "
                  f"{int((got2.cpu() != ref).sum())} differ, dtype {got.dtype}", flush=True)
            bad = (got.cpu() != ref).nonzero().reshape(-1)[:4]
            for i in bad.tolist():
                print(f"  diff x={x[i].item()!r} cpu {ref[i].item()} card {got.cpu()[i].item()}", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "bucketize":
        bucketize()
        sys.exit()
    main()
