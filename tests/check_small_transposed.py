"""On a Gaudi card: the n <= 64 Gauss-Jordan kernels on the code review's inputs (finding 3) and random ones.
    PT_HPU_LAZY_MODE=0|1 python3 tests/check_small_transposed.py
Per input: the raw ``gj_inv`` / ``gj_solve`` kernels against their float32 replica (bit-equal), and the public
``solve``, ``inv``, ``det``, ``slogdet`` and ``det``'s gradient (hpu_ops enabled) against torch on the CPU: within
``max(FLOOR, 10x)`` CPU float32's error against float64, the same singular verdict (det 0, inv raises) and NaN where
torch gives NaN. One ``RESULT {json}`` line per input; a last SUMMARY line."""
import json
import os
import warnings

import torch

import hpu_ops

hpu_ops.enable()
import habana_frameworks.torch.core as htcore  # noqa: E402

import kernel_replica as kr  # noqa: E402
from hpu_ops import tpc  # noqa: E402

warnings.filterwarnings("ignore")
LAZY = os.environ.get("PT_HPU_LAZY_MODE", "0") == "1"
FLOOR = 1e-5


def wilkinson(n, random_last=False):
    W = torch.eye(n) - torch.tril(torch.ones(n, n), -1)
    W[:, -1] = torch.randn(n, generator=torch.Generator().manual_seed(n)) if random_last else 1
    return W / 2


def cases():
    for n in (4, 32, 40, 64):
        yield f"wilkinson_n{n}", wilkinson(n)
        yield f"wilkinson_randlast_n{n}", wilkinson(n, True)
        A = torch.randn(n, n, generator=torch.Generator().manual_seed(n)) / n ** 0.5 + torch.eye(n)
        A[1] = A[0]
        yield f"duplicate_rows_n{n}", A
        yield f"random_n{n}", torch.randn(n, n, generator=torch.Generator().manual_seed(100 + n)) / n ** 0.5 + torch.eye(n)
    yield "huge_tiny_singular", torch.tensor([[2.0 ** 100, 2.0 ** 100], [2.0 ** -100, 2.0 ** -100]])
    yield "huge_tiny_101", torch.tensor([[2.0 ** 100, 2.0 ** 100], [2.0 ** -100, 2.0 ** -101]])
    yield "nan_2", torch.tensor([[0.0, 1.0], [float("nan"), 1.0]])
    A = torch.eye(64)
    A[:2, :2] = torch.tensor([[0.0, 1.0], [float("nan"), 1.0]])
    yield "nan_in_eye64", A


def rel(x, ref):
    x, ref = x.double(), ref.double()
    if not torch.isfinite(x[torch.isfinite(ref)]).all():
        return float("nan")
    scale = float(ref.abs().max())
    return float((x - ref).abs().max()) / (scale if scale > 0 else 1.0)


def sync():
    htcore.mark_step()
    torch.hpu.synchronize()


def outcome(fn, A):
    """``fn(A)`` on the CPU float32, CPU float64 and the card (each result on the CPU), or the exception's type."""
    out = {}
    for name, x in (("cpu32", A), ("cpu64", A.double()), ("hpu", A.to("hpu"))):
        try:
            r = fn(x)
            if x.device.type == "hpu":
                sync()
            out[name] = r.detach().cpu() if isinstance(r, torch.Tensor) else [t.detach().cpu() for t in r]
        except Exception as e:  # noqa: BLE001
            out[name] = type(e).__name__
    return out


def judge(name, o):
    if isinstance(o["cpu32"], str) or isinstance(o["hpu"], str):
        return [] if o["cpu32"] == o["hpu"] or (isinstance(o["cpu32"], str) and isinstance(o["hpu"], str)) \
            else [f"{name}: cpu {o['cpu32']} hpu {o['hpu'] if isinstance(o['hpu'], str) else 'returned'}"]
    got, want, cpu = (torch.as_tensor(o[k]) if isinstance(o[k], torch.Tensor) else torch.stack(o[k])
                      for k in ("hpu", "cpu64", "cpu32"))
    if want.isnan().any() or cpu.isnan().any():
        return [] if got.isnan().any() else [f"{name}: torch gives NaN, the card does not"]
    if (want == 0).all():
        return [] if (got == 0).all() else [f"{name}: torch gives 0, the card {got.flatten()[:2].tolist()}"]
    inf = ~torch.isfinite(want)
    if inf.any():  # -inf logabsdet of a singular matrix: the same entries, then the finite ones judged
        if not torch.equal(got[inf].double(), want[inf]):
            return [f"{name}: torch's non-finite entries {want[inf].tolist()}, the card's {got[inf].tolist()}"]
        got, want, cpu = got[~inf], want[~inf], cpu[~inf]
        if want.numel() == 0 or (want == 0).all():
            return [] if (got == 0).all() else [f"{name}: torch gives 0, the card {got.flatten()[:2].tolist()}"]
    err, bar = rel(got, want), max(FLOOR, 10 * rel(cpu, want))
    return [] if err <= bar else [f"{name}: error {err:.3g} > {bar:.3g}"]


def det_grad(A):
    A = A.detach().clone().requires_grad_()
    torch.linalg.det(A).backward()
    return A.grad


def main():
    assert tpc.ready()
    ok = True
    for name, A in cases():
        n = A.shape[-1]
        row = {"case": name, "n": n, "lazy": LAZY, "bridge": "1.21.4"}
        problems = []
        At = A.mT.contiguous()[None]
        got = [t.cpu() for t in torch.ops.linalg_tpc.gj_inv(At.to("hpu"))]
        sync()
        want = [torch.from_numpy(x)[None] for x in kr.gj_solve(At[0].numpy())]
        row["raw_bit_equal"] = [bool(torch.equal(g.nan_to_num(), w.nan_to_num())) for g, w in zip(got, want)]
        problems += [] if all(row["raw_bit_equal"]) else ["raw gj_inv differs from its replica"]
        B = torch.randn(n, 3, generator=torch.Generator().manual_seed(7))
        for op, fn in (("solve", lambda x: torch.linalg.solve(x, B.to(x.device, x.dtype))),
                       ("inv", torch.linalg.inv), ("det", torch.linalg.det),
                       ("slogdet", lambda x: torch.stack(list(torch.linalg.slogdet(x)))),
                       ("det_grad", det_grad)):
            problems += judge(op, outcome(fn, A))
        row["problems"] = problems
        row["ok"] = not problems
        ok &= row["ok"]
        print("RESULT " + json.dumps(row), flush=True)
    print("SUMMARY", "ok" if ok else "MISMATCH", flush=True)


if __name__ == "__main__":
    main()
