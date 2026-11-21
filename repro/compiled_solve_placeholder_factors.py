"""On a Gaudi card: the gradient of torch.linalg.solve (torch's Cayley map, as STaRNet(64)'s BiMap at n = 96)
compiled with hpu_backend against CPU torch, per n, and the LU / pivots _linalg_solve_ex returns.
    PT_HPU_LAZY_MODE=0 python3 repro/compiled_solve_placeholder_factors.py [--no-ops] [--dynamic]
Bridge 1.21.4, origin/main 9016bbe, compiled grad rel err against CPU: solve NaN at n = 65, 96 (batch 1, 4), 200
(3e-7 at n <= 64 and without hpu_ops); slogdet NaN there; det "synLaunch failed ... Sections' validation failure"
at n = 65 and 96, 4.8e12 at n = 200. With --dynamic (symbolic shapes) every n fails, n = 16 too: solve and slogdet
NaN, det 7.9e6 at n = 16, else "tensor size should match the storage size". The trace declines the call (eager-only
large route, symbolic shape), so the graph holds the native op and torch's derivative reads its LU / pivots;
hpu_backend runs that node eagerly through hpu_ops' autograd kernel, which then takes hpu_ops' route and returns
placeholder factors (zero LU expanded from a scalar, identity pivots: the FACTORS lines). Fixed (the declined traced
call's backward reads A and the outputs only): every case as eager, 1e-7 to 3e-6, static and --dynamic."""
import sys

import torch

if "--no-ops" not in sys.argv:
    import hpu_ops

    hpu_ops.enable()


def cayley(W):
    A = W - W.mT
    Id = torch.eye(A.shape[-1], dtype=A.dtype, device=A.device)
    return torch.linalg.solve(torch.add(Id, A, alpha=-0.5), torch.add(Id, A, alpha=0.5))


def loss(W, C):
    return (cayley(W) * C).sum()


def loss_det(W, C):
    return (torch.linalg.det(torch.eye(W.shape[-1], device=W.device) + W) * C[..., 0, 0]).sum()


def loss_slogdet(W, C):
    return (torch.linalg.slogdet(torch.eye(W.shape[-1], device=W.device) + W)[1] * C[..., 0, 0]).sum()


LOSSES = {"solve": loss, "det": loss_det, "slogdet": loss_slogdet}


torch._dynamo.config.cache_size_limit = 64
for name, loss_fn in LOSSES.items():
    for n, batch in ((16, 1), (64, 1), (65, 1), (96, 1), (96, 4), (200, 1)):
        torch.manual_seed(0)
        W = torch.randn(batch, n, n) * (0.1 if name == "solve" else 0.3 / n ** 0.5)  # det: I + W stays O(1)
        C = torch.randn(batch, n, n)
        Wc = W.clone().requires_grad_()
        loss_fn(Wc, C).backward()
        ref = Wc.grad
        row = [f"{name} n={n} batch={batch}"]
        for mode in ("eager", "compiled"):
            Wh = W.to("hpu").requires_grad_()
            f = loss_fn if mode == "eager" else torch.compile(loss_fn, backend="hpu_backend",
                                                              dynamic="--dynamic" in sys.argv)
            try:
                out = f(Wh, C.to("hpu"))
                out.backward()
                g = Wh.grad.cpu()
                row.append(f"{mode}: fwd err {abs(float(out.cpu()) - float(loss_fn(W, C))):.2e} "
                           f"grad rel err {float((g - ref).norm() / ref.norm()):.2e}")
            except Exception as e:  # noqa: BLE001
                row.append(f"{mode}: ERROR {repr(e)[:200]}")
        print("GRAD", " | ".join(row), flush=True)

# Habana's (or hpu_ops') _linalg_solve_ex factors against CPU's, contiguous A
for n in (16, 96):
    torch.manual_seed(1)
    A = torch.randn(1, n, n) + n ** 0.5 * torch.eye(n)
    B = torch.randn(1, n, n)
    with torch.no_grad():
        Xc, LUc, Pc, Ic = torch.ops.aten._linalg_solve_ex(A, B)
        Xh, LUh, Ph, Ih = [t.cpu() for t in torch.ops.aten._linalg_solve_ex(A.to("hpu"), B.to("hpu"))]
    LUt, Pt = torch.linalg.lu_factor(A.mT)
    print(f"FACTORS n={n} X err {float((Xh - Xc).abs().max()):.2e} LU vs CPU {float((LUh - LUc).abs().max()):.2e} "
          f"LU vs lu(A^T) {float((LUh - LUt).abs().max()):.2e} LU vs lu(A) "
          f"{float((LUh - torch.linalg.lu_factor(A)[0]).abs().max()):.2e} pivots equal CPU {bool((Ph == Pc).all())} "
          f"pivots[:8] {Ph.flatten()[:8].tolist()} cpu {Pc.flatten()[:8].tolist()}", flush=True)
