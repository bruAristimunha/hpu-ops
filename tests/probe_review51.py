"""On a Gaudi card: the code review of #51's counterexamples through the public ops (hpu_ops enabled), each against
torch on the CPU in float64, next to CPU float32's error; plus the block Jacobi eigh's time against hpu-ops' CPU route.
    PT_HPU_LAZY_MODE=0|1 python3 tests/probe_review51.py [--sizes 256,512,1024] [--batch 4]"""
import argparse
import json
import os
import time
import warnings

import hpu_ops

hpu_ops.enable()
import habana_frameworks.torch.core as htcore  # noqa: E402
import torch  # noqa: E402

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--sizes", default="256,512,1024")
ap.add_argument("--batch", type=int, default=4)
args = ap.parse_args()
LAZY = os.environ.get("PT_HPU_LAZY_MODE", "0") == "1"
L = torch.linalg


def rel(x, r):
    x, r = x.detach().cpu().double(), r.detach().double()
    return float((x - r).abs().max() / r.abs().max())


def sync():
    htcore.mark_step()
    torch.hpu.synchronize()


def emit(**kw):
    print("RESULT " + json.dumps({**kw, "lazy": LAZY}), flush=True)


def spd_spectrum(b, n, lo, seed):
    g = torch.Generator().manual_seed(seed)
    Q = torch.linalg.qr(torch.randn(b, n, n, generator=g, dtype=torch.float64))[0]
    A = (Q * torch.logspace(0, lo, n, dtype=torch.float64)) @ Q.mT
    return ((A + A.mT) / 2).float()


def eigh_case(name, A, timed=False):
    w64 = L.eigvalsh(A.double())
    w32, V32 = L.eigh(A)
    h = A.to("hpu")
    w, V = L.eigh(h)
    sync()
    res = rel(V.cpu().double() * w.cpu().double()[..., None, :] @ V.cpu().double().mT, A.double())
    res32 = rel(V32.double() * w32.double()[..., None, :] @ V32.double().mT, A.double())
    row = {"case": name, "w_err": rel(w, w64), "w_err_cpu32": rel(w32, w64), "res": res, "res_cpu32": res32}
    row["ok"] = row["w_err"] <= max(1e-6, 10 * row["w_err_cpu32"]) and res <= max(1e-6, 10 * res32)
    if timed:
        for route in ("card", "cpu_route"):
            os.environ["HPU_OPS_LARGE_EIGH"] = "cpu" if route == "cpu_route" else ""
            L.eigh(h)
            sync()
            t = time.perf_counter()
            L.eigh(h)
            sync()
            row[f"ms_{route}"] = round((time.perf_counter() - t) * 1e3, 1)
        os.environ.pop("HPU_OPS_LARGE_EIGH")
    emit(**row)


def main():
    g = torch.Generator().manual_seed(0)
    for n in [int(x) for x in args.sizes.split(",")]:
        G = torch.randn(args.batch, n, n, generator=g)
        eigh_case(f"eigh_gauss_n{n}", (G + G.mT) / 2, timed=True)
        eigh_case(f"eigh_spd_1e-6_n{n}", spd_spectrum(args.batch, n, -6, n), timed=True)
    # LOW 3: eigh_big's small eigenvalues (n = 127, spectrum 1 to 1e-6)
    A = spd_spectrum(4, 127, -6, 127)
    w = L.eigvalsh(A.to("hpu")).cpu()
    w64 = L.eigvalsh(A.double())
    emit(case="eigh_big_small_eigs_n127", smallest=float(w[0, 0]), smallest_ref=float(w64[0, 0]),
         smallest_cpu32=float(L.eigvalsh(A)[0, 0]), w_err=rel(w, w64), w_err_cpu32=rel(L.eigvalsh(A), w64))
    # LOW 5: a 40 x 40 symmetric matrix of entries near 1e19
    G = torch.randn(2, 40, 40, generator=g)
    eigh_case("eigh_1e19_n40", (G + G.mT) * 1e19)
    # HIGH 2: svdvals 512 of spectrum 1 to 1e-5, lstsq 40 x 20 at condition 1e4
    U = torch.linalg.qr(torch.randn(1, 512, 512, generator=g, dtype=torch.float64))[0]
    Vq = torch.linalg.qr(torch.randn(1, 512, 512, generator=g, dtype=torch.float64))[0]
    M = ((U * torch.logspace(0, -5, 512, dtype=torch.float64)) @ Vq.mT).float()
    s = L.svdvals(M.to("hpu"))
    sync()
    s64 = L.svdvals(M.double())
    emit(case="svdvals_512_1e-5", err=rel(s, s64), err_cpu32=rel(L.svdvals(M), s64))
    U = torch.linalg.qr(torch.randn(1, 40, 20, generator=g, dtype=torch.float64))[0]
    Vq = torch.linalg.qr(torch.randn(1, 20, 20, generator=g, dtype=torch.float64))[0]
    A = ((U * torch.logspace(0, -4, 20, dtype=torch.float64)) @ Vq.mT).float()
    B = torch.randn(1, 40, 2, generator=g)
    x = L.lstsq(A.to("hpu"), B.to("hpu")).solution
    sync()
    x64 = L.lstsq(A.double(), B.double()).solution
    emit(case="lstsq_40x20_cond1e4", err=rel(x, x64), err_cpu32=rel(L.lstsq(A, B).solution, x64))
    # LOW 4: HPU_OPS_FLOAT64=card with mixed float64 / float32 arguments raises, as torch
    os.environ["HPU_OPS_FLOAT64"] = "card"
    try:
        L.solve(torch.eye(3, dtype=torch.float64).to("hpu"), torch.ones(3, 1).to("hpu"))
        sync()
        raised = None
    except Exception as e:  # noqa: BLE001
        raised = type(e).__name__
    os.environ.pop("HPU_OPS_FLOAT64")
    try:
        L.solve(torch.eye(3, dtype=torch.float64), torch.ones(3, 1))
        want = None
    except Exception as e:  # noqa: BLE001
        want = type(e).__name__
    emit(case="float64_card_mixed_dtypes", raised=raised, torch_raises=want, ok=raised == want)


if __name__ == "__main__":
    main()
