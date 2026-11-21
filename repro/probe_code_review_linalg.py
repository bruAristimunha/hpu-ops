"""On a Gaudi card: the code review's linalg/TPC findings (1, 8, 10, 11, 13), each with the review's exact input.
    PT_HPU_LAZY_MODE=0|1 python3 repro/probe_code_review_linalg.py [--time]
Run it with the fixed tree and with origin/main first on PYTHONPATH (same pod) to compare. One ``RESULT {json}`` line per
check: ``ok`` true when the behavior is torch's (dtype, no error, accuracy within max(1e-6, 10 x torch's CPU float32
error against CPU float64), in-place edits that torch allows). ``--time`` adds matrix_exp timings (mean of 100 synced
calls after warm-up, outputs kept alive). Bridge fallbacks per check are recorded (``fallback``)."""
import argparse
import json
import os
import time
import warnings

import habana_frameworks.torch.core as htcore
import torch
from habana_frameworks.torch.hpu.metrics import metric_global

import hpu_ops

hpu_ops.enable()
HPU = torch.device("hpu")
LAZY = os.environ.get("PT_HPU_LAZY_MODE", "0") == "1"
warnings.simplefilter("ignore")


def fallbacks():
    return dict(dict(metric_global("cpu_fallback").stats()).get("FallbackOps", {}) or {})


def sync():
    if LAZY:
        htcore.mark_step()
    torch.hpu.synchronize()


def rel(x, ref):
    x, ref = x.detach().cpu().double(), ref.detach().cpu().double()
    return ((x - ref).abs().max() / ref.abs().max().clamp(min=1e-300)).item()


def report(name, **fields):
    print("RESULT " + json.dumps({"check": name, "lazy": LAZY, **fields}), flush=True)


def check(name, fn):
    before = fallbacks()
    try:
        fields = fn()
        sync()
    except Exception as error:  # noqa: BLE001  (a row, not a crash)
        fields = {"ok": False, "error": f"{type(error).__name__}: {str(error)[:160]}"}
    after = fallbacks()
    delta = {k: v - before.get(k, 0) for k, v in after.items() if v != before.get(k, 0)}
    report(name, **fields, fallback=delta)


def autocast_checks():
    """Finding 1: hpu-ops' linalg Functions under bf16 autocast return torch's dtype (float32) and accuracy."""
    torch.manual_seed(0)
    X = torch.randn(3, 6, 6)
    M = X @ X.mT + 6 * torch.eye(6)
    LU, piv = torch.linalg.lu_factor(M)
    B = torch.randn(3, 6, 6)
    for left in (True, False):
        for adjoint in (False, True):
            ref = torch.linalg.lu_solve(LU.double(), piv, B.double(), left=left, adjoint=adjoint)
            bar = rel(torch.linalg.lu_solve(LU, piv, B, left=left, adjoint=adjoint), ref)

            def run(left=left, adjoint=adjoint, ref=ref, bar=bar):
                with torch.autocast("hpu", dtype=torch.bfloat16):
                    out = torch.linalg.lu_solve(LU.to(HPU), piv.to(HPU), B.to(HPU), left=left, adjoint=adjoint)
                e = rel(out, ref)
                return {"ok": out.dtype == torch.float32 and e <= max(1e-6, 10 * bar), "dtype": str(out.dtype),
                        "rel": e, "bar": bar}

            check(f"autocast lu_solve left={left} adjoint={adjoint}", run)
    n = 96
    A = torch.randn(8, n, n) + n * torch.eye(n)
    Bn = torch.randn(8, n, 4)
    ref = torch.linalg.solve(A.double(), Bn.double())
    bar = rel(torch.linalg.solve(A, Bn), ref)

    def solve96():
        with torch.autocast("hpu", dtype=torch.bfloat16):
            out = torch.linalg.solve(A.to(HPU), Bn.to(HPU))
        e = rel(out, ref)
        return {"ok": out.dtype == torch.float32 and e <= max(1e-6, 10 * bar), "dtype": str(out.dtype), "rel": e,
                "bar": bar}

    check("autocast solve n=96 batch 8 (large kernel)", solve96)
    A5, B5 = torch.randn(4, 6, 5), torch.randn(4, 6, 2)
    ref5 = torch.linalg.lstsq(A5.double(), B5.double()).solution
    bar5 = rel(torch.linalg.lstsq(A5, B5).solution, ref5)

    def lstsq():
        Ah = A5.to(HPU).requires_grad_()
        with torch.autocast("hpu", dtype=torch.bfloat16):
            out = torch.linalg.lstsq(Ah, B5.to(HPU)).solution
            out.sum().backward()
        e = rel(out, ref5)
        return {"ok": out.dtype == torch.float32 and Ah.grad.dtype == torch.float32 and e <= max(1e-6, 10 * bar5),
                "dtype": str(out.dtype), "rel": e, "bar": bar5}

    check("autocast lstsq + backward", lstsq)


def expm_case(A):
    """(forward rel, grad rel, torch f32 forward rel, torch f32 grad rel) against CPU float64."""
    G = torch.ones_like(A) + 0.1 * torch.arange(A.numel(), dtype=A.dtype).reshape(A.shape) / A.numel()
    Ad = A.double().requires_grad_()
    (torch.linalg.matrix_exp(Ad) * G.double()).sum().backward()
    At = A.clone().requires_grad_()
    Yt = torch.linalg.matrix_exp(At)
    (Yt * G).sum().backward()
    Ah = A.to(HPU).requires_grad_()
    Yh = torch.linalg.matrix_exp(Ah)
    (Yh * G.to(HPU)).sum().backward()
    sync()
    ref = Ad.detach().matrix_exp()
    return rel(Yh, ref), rel(Ah.grad, Ad.grad), rel(Yt, ref), rel(At.grad, Ad.grad)


def expm_checks():
    """Finding 8: matrix_exp float32 accuracy with a dominant positive eigenvalue, forward and gradient."""
    torch.manual_seed(1)
    cases = {f"diag({x}, .5, -1)": torch.diag(torch.tensor([x, 0.5, -1.0])) for x in (3.0, 5.0, 20.0, 80.0)}
    cases["diag(-80, .5, 1)"] = torch.diag(torch.tensor([-80.0, 0.5, 1.0]))
    M = torch.randn(22, 22)
    cases["spd n=22 ||A||~10"] = (lambda S: 10 * S / S.abs().sum(0).max())(M @ M.T)
    cases["randn n=22 *0.3"] = 0.3 * M
    cases["skew n=16 *2"] = (lambda K: 2 * (K - K.T))(torch.randn(16, 16))
    for name, A in cases.items():
        def run(A=A):
            f, g, tf, tg = expm_case(A)
            return {"ok": f <= max(1e-6, 10 * tf) and g <= max(1e-6, 10 * tg), "rel": f, "bar": tf, "grad_rel": g,
                    "grad_bar": tg}

        check(f"matrix_exp {name}", run)


def aso_checks():
    """Finding 11: s > 16 squarings read back by a scalar load after the vector store (||A||_1 > 3.5e5), batched,
    after other matrix_exp calls; forward and Frechet (backward)."""
    torch.manual_seed(2)
    for _ in range(3):
        torch.linalg.matrix_exp(torch.randn(32, 8, 8, device=HPU))
    sync()
    A = torch.tensor([[0.0, 1e6], [0.0, 0.0]]).expand(64, 2, 2).contiguous()
    want = torch.tensor([[1.0, 1e6], [0.0, 1.0]])

    def forward():
        Y = torch.linalg.matrix_exp(A.to(HPU)).cpu()
        bad = int((~torch.isclose(Y, want.expand_as(Y), rtol=1e-6, atol=0)).any(-1).any(-1).sum())
        return {"ok": bad == 0, "bad_matrices": bad, "first": Y[0].flatten().tolist()}

    check("aso matrix_exp [[0,1e6],[0,0]] x64", forward)

    def frechet():
        Ah = A.to(HPU).requires_grad_()
        torch.linalg.matrix_exp(Ah).sum().backward()
        Ad = A.double().requires_grad_()
        torch.linalg.matrix_exp(Ad).sum().backward()
        e = rel(Ah.grad, Ad.grad)
        return {"ok": e <= 1e-5, "rel": e}

    check("aso matrix_exp backward [[0,1e6],[0,0]] x64", frechet)


def nan_checks():
    """Finding 13: NaN eigenvalues / singular values stay NaN (not +-inf), and every mode agrees."""
    A = torch.randn(4, 4)
    A = A + A.T
    A[1, 2] = A[2, 1] = float("nan")
    Ah = A.to(HPU)

    def eigvalsh():
        w = torch.linalg.eigvalsh(Ah).cpu()
        return {"ok": bool(w.isnan().any()) and not bool(w.isinf().any()), "w": w.tolist()}

    check("nan eigvalsh 4x4", eigvalsh)

    def eigh():
        w, V = torch.linalg.eigh(Ah)
        w, V = w.cpu(), V.cpu()
        return {"ok": not bool(w.isinf().any()) and bool(w.isnan().any()), "w": w.tolist()}

    check("nan eigh 4x4", eigh)

    def norms():
        plain = torch.linalg.matrix_norm(Ah, 2).item()
        g = Ah.clone().requires_grad_()
        grad = torch.linalg.matrix_norm(g, 2).item()
        with torch.inference_mode():
            inf_mode = torch.linalg.matrix_norm(Ah, 2).item()
        sv = torch.linalg.svdvals(Ah).cpu()
        vals = [plain, grad, inf_mode]
        return {"ok": all(v != v for v in vals) and not bool(sv.isinf().any()), "no_grad": plain, "grad": grad,
                "inference": inf_mode, "svdvals": sv.tolist()}

    check("nan matrix_norm(2) by mode, svdvals", norms)

    def svd():
        U, S, Vh = torch.linalg.svd(Ah)
        return {"ok": not bool(S.cpu().isinf().any()), "S": S.cpu().tolist()}

    check("nan svd 4x4", svd)


def view_checks():
    """Finding 10: in-place edits of the results, as torch allows them."""
    torch.manual_seed(3)

    def lstsq_mul():
        A = torch.randn(6, 4, device=HPU, requires_grad=True)
        X = torch.linalg.lstsq(A, torch.randn(6, 2, device=HPU)).solution
        X.mul_(0.5)
        X.sum().backward()
        return {"ok": bool(torch.isfinite(A.grad).all())}

    check("inplace lstsq solution mul_", lstsq_mul)

    def lstsq_clamp():
        A = torch.randn(6, 4, device=HPU, requires_grad=True)
        r = torch.linalg.lstsq(A, torch.randn(6, 2, device=HPU), driver="gelsd")
        r.singular_values.clamp_(min=1e-3)
        r.solution.sum().backward()
        return {"ok": bool(torch.isfinite(A.grad).all())}

    check("inplace lstsq gelsd singular_values clamp_", lstsq_clamp)

    def expm_diag():
        A = (0.3 * torch.randn(2, 4, 4)).to(HPU).requires_grad_()
        Y = torch.linalg.matrix_exp(A)
        Y.diagonal(dim1=-2, dim2=-1).add_(1.0)
        Y[..., :3, 3] = 0.0
        Y.sum().backward()
        return {"ok": bool(torch.isfinite(A.grad).all())}

    check("inplace matrix_exp diagonal add_", expm_diag)

    def lu_factor():
        A = torch.randn(5, 5, device=HPU, requires_grad=True)
        LU, piv = torch.linalg.lu_factor(A)
        LU.mul_(2.0)
        return {"ok": True}

    check("inplace lu_factor LU mul_", lu_factor)

    def inv2d():
        A = (torch.randn(5, 5) + 5 * torch.eye(5)).to(HPU).requires_grad_()
        X = torch.linalg.inv(A)
        X.add_(1.0)  # as torch: allowed (a backward through X then raises in torch too: the inverse is saved)
        return {"ok": bool(X.requires_grad)}

    check("inplace inv 2-D add_", inv2d)


def ortho(Q):
    """max |Q^T Q - I| of the columns of ``Q`` (float64, on the CPU)."""
    Q = Q.detach().cpu().double()
    return (Q.mT @ Q - torch.eye(Q.shape[-1], dtype=torch.float64)).abs().amax().item()


def ortho_checks():
    """Team Z's rsqrt bias: orthogonality of the Jacobi SVD's U, V and eigh's V on the card against torch's CPU
    float32, with the values' error against float64 (batch 32, random)."""
    torch.manual_seed(5)
    for n in (4, 8, 16, 22, 32):
        A = torch.randn(32, n, n)
        S_ref = torch.linalg.svdvals(A.double())
        U, S, Vh = torch.linalg.svd(A.to(HPU))
        Ut, St, Vht = torch.linalg.svd(A)
        report(f"ortho svd n={n}", ok=True, U=ortho(U), V=ortho(Vh.mT), torch_U=ortho(Ut), torch_V=ortho(Vht.mT),
               S_rel=rel(S, S_ref), torch_S_rel=rel(St, S_ref))
        M = (A + A.mT) / 2
        w_ref = torch.linalg.eigvalsh(M.double())
        w, V = torch.linalg.eigh(M.to(HPU))
        wt, Vt = torch.linalg.eigh(M)
        report(f"ortho eigh n={n}", ok=True, V=ortho(V), torch_V=ortho(Vt), w_rel=rel(w, w_ref),
               torch_w_rel=rel(wt, w_ref))


def timed(fn, reps=100):
    out = [fn() for _ in range(3)]
    sync()
    t0 = time.perf_counter()
    out = [fn() for _ in range(reps)]
    sync()
    return (time.perf_counter() - t0) / reps * 1e3, out


def expm_times():
    torch.manual_seed(4)
    for scale, tag in ((0.1, "||A||~0.5"), (1.0, "||A||~5"), (10.0, "||A||~50")):
        A = (scale * torch.randn(32, 22, 22) / 22 ** 0.5).to(HPU)
        G = torch.randn(32, 22, 22, device=HPU)
        fwd, _ = timed(lambda: torch.linalg.matrix_exp(A))
        Ag = A.clone().requires_grad_()

        def both():
            Y = torch.linalg.matrix_exp(Ag)
            return torch.autograd.grad(Y, Ag, G)[0]

        bwd, _ = timed(both)
        report(f"time matrix_exp n=22 b=32 {tag}", ok=True, forward_ms=fwd, fwd_bwd_ms=bwd)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--time", action="store_true")
    parser.add_argument("--only", default="autocast,expm,aso,nan,view")
    args = parser.parse_args()
    report("setup", ok=True, hpu_ops=os.path.dirname(hpu_ops.__file__), bridge=__import__(
        "habana_frameworks.torch").torch.__version__ if hasattr(__import__("habana_frameworks.torch").torch,
                                                                 "__version__") else "?")
    groups = {"autocast": autocast_checks, "expm": expm_checks, "aso": aso_checks, "nan": nan_checks,
              "view": view_checks, "ortho": ortho_checks}
    for name in filter(None, args.only.split(",")):
        groups[name]()
    if args.time:
        expm_times()


if __name__ == "__main__":
    main()
