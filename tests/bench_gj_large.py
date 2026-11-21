"""On a Gaudi card: time of the raw ``gj_large`` kernel (and ``tpc.gj_inv``) per call, for an A/B of two builds of
``liblinalg_tpc.so`` in alternating processes (the kernel library is read once per process).
    PT_HPU_LAZY_MODE=0 python3 tests/bench_gj_large.py [--lib PATH] [--reps 30]
Prints one ``RESULT`` JSON line per shape: mean ms of ``reps`` synced calls after warm-up, outputs kept alive."""
import argparse
import json
import pathlib
import time

from hpu_ops import tpc

ap = argparse.ArgumentParser()
ap.add_argument("--lib", default=None)
ap.add_argument("--reps", type=int, default=30)
args = ap.parse_args()
if args.lib:
    tpc.LIB = pathlib.Path(args.lib)
tpc.install_kernel_path()
import habana_frameworks.torch.core as htcore  # noqa: E402
import torch  # noqa: E402


def timed(f, reps):
    keep = [f() for _ in range(3)]
    htcore.mark_step()
    torch.hpu.synchronize()
    t = time.perf_counter()
    keep = [f() for _ in range(reps)]
    torch.hpu.synchronize()
    return (time.perf_counter() - t) / reps * 1e3, keep


def main():
    assert tpc.ready()
    g = torch.Generator().manual_seed(0)
    for n, b in ((65, 8), (130, 8), (220, 8), (300, 8), (512, 2)):
        A = (torch.randn(b, n, n, generator=g) + n ** 0.5 * torch.eye(n)).to("hpu")
        raw, _ = timed(lambda: torch.ops.linalg_tpc.gj_large(A), args.reps)
        inv, _ = timed(lambda: tpc.gj_inv(A), args.reps)
        print("RESULT " + json.dumps({"lib": args.lib or "src", "n": n, "batch": b, "raw_ms": round(raw, 4),
                                      "gj_inv_ms": round(inv, 4)}), flush=True)


if __name__ == "__main__":
    main()
