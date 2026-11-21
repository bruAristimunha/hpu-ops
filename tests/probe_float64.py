"""On a Gaudi card: what a float64 HPU tensor holds, and whether float32 matmul on the MME is IEEE float32.
    PT_HPU_LAZY_MODE=0|1 python3 tests/probe_float64.py
    GAUDI2_FORCE_MME_FP32_IEEE=1 PT_HPU_LAZY_MODE=0 python3 tests/probe_float64.py --only mme
    PT_HPU_LAZY_MODE=0|1 python3 tests/probe_float64.py --only linalg   (hpu_ops enabled)
(a) storage: element size and storage bytes of a float64 HPU tensor; the host -> card -> host round trip of
values that need more than 24 mantissa bits (1 + 2^-40, random float64); the same bits viewed as int64 / int32
on the card; int64 values >= 2^32; float32 -> float64 on the card; float64 arithmetic on the card; 64-bit
patterns carried in an int32 tensor and viewed as float64 on the card.
(b) MME: the float32 product of 64 x 64 matrices whose exact result needs mantissa bit k (k = 8..23): the largest
k kept tells IEEE float32 (23), TF32 (10) or bf16 (7); and the error of random / cancelling products against
CPU float64 next to the CPU float32 error, under torch.set_float32_matmul_precision("highest" | "high" | "medium").
One ``RESULT {json}`` line per probe."""
import argparse
import json
import os
import sys
import warnings

if "linalg" in sys.argv:  # before the first HPU op: Habana reads the kernel path when the device initialises
    import hpu_ops

    hpu_ops.enable()

import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.hpu as hthpu
import torch

LAZY = os.environ.get("PT_HPU_LAZY_MODE", "1") != "0"
HPU = torch.device("hpu")


def sync():
    if LAZY:
        htcore.mark_step()
    hthpu.synchronize()


def emit(name, **kv):
    row = {"probe": name, "lazy": LAZY, "mme_ieee_env": os.environ.get("GAUDI2_FORCE_MME_FP32_IEEE"), **kv}
    print("RESULT " + json.dumps(row, default=str), flush=True)


def try_(name, fn):
    try:
        emit(name, **fn())
    except Exception as e:  # a probe that raises is itself a result
        emit(name, error=f"{type(e).__name__}: {str(e).splitlines()[0][:200]}")


def card(t):
    out = t.to(HPU)
    sync()
    return out


def back(t):
    sync()
    return t.cpu()


def storage():
    x = torch.tensor([1.0 + 2.0 ** -40, 1.0 + 2.0 ** -30, 1.0 / 3.0, 1e300, 1e-300], dtype=torch.float64)
    r = torch.randn(4096, dtype=torch.float64, generator=torch.Generator().manual_seed(0))

    def meta():
        h = card(x)
        return {"dtype": str(h.dtype), "element_size": h.element_size(),
                "storage_nbytes": h.untyped_storage().nbytes(), "cpu_storage_nbytes": x.untyped_storage().nbytes()}

    def roundtrip():
        got = back(card(x))
        gr = back(card(r))
        return {"values": got.tolist(), "exact": got.tolist() == x.tolist(),
                "one_plus_2m40_minus_1": float(got[0] - 1), "random_max_abs_diff": float((gr - r).abs().max()),
                "random_bit_equal": float((gr.view(torch.int64) == r.view(torch.int64)).double().mean()),
                "random_equals_f32_rounding": bool((gr == r.float().double()).all())}

    def view_int64_on_card():
        got = back(card(x).view(torch.int64))
        return {"card": got.tolist(), "cpu": x.view(torch.int64).tolist(),
                "equal": got.tolist() == x.view(torch.int64).tolist()}

    def view_int32_on_card():
        got = back(card(x).view(torch.int32))
        return {"shape": list(got.shape), "card": got.tolist(), "cpu": x.view(torch.int32).tolist()}

    def int64_values():
        i = torch.tensor([2 ** 40 + 1, -(2 ** 33) - 3, 2 ** 62 + 7], dtype=torch.int64)
        got = back(card(i))
        added = back(card(i) + 1)
        return {"roundtrip": got.tolist(), "plus_one": added.tolist(), "expected": i.tolist()}

    def f32_to_f64_on_card():
        a = card(torch.tensor([1.0, 2.0 ** -30, 1.0 / 3.0], dtype=torch.float32))
        d = a.to(torch.float64)
        s = d[0] + d[1]  # 1 + 2^-30: exact in float64, rounds to 1 in float32
        p = d[2] * 3.0  # (1/3 in f32) * 3: exact in f64 is 1 - 2^-25 approx; f32 rounds to 1
        return {"to_f64": back(d).tolist(), "sum_minus_1": float(back(s) - 1.0),
                "prod_minus_1": float(back(p) - 1.0),
                "expected_prod_minus_1": float(torch.tensor(1.0 / 3.0, dtype=torch.float32).double() * 3 - 1)}

    def f64_arith_on_card():
        one = card(torch.ones(4, dtype=torch.float64))
        e = card(torch.full((4,), 2.0 ** -40, dtype=torch.float64))
        return {"(1+2^-40)-1": back((one + e) - one).tolist(), "expected": 2.0 ** -40}

    def bits_via_int32():
        bits = x.view(torch.int32)  # 2 int32 per float64
        h = card(bits)
        v = h.view(torch.float64)
        got = back(v)
        twice = back(v * 1.0)
        return {"shape": list(v.shape), "card_view_f64": got.tolist(), "exact": got.tolist() == x.tolist(),
                "times_one": twice.tolist(), "back_as_int32_equal": back(h).tolist() == bits.tolist()}

    def offset_views():
        v = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=torch.float64)
        h = card(v)
        return {"h[1:].cpu()": back(h[1:]).tolist(), "h[1:]+0": back(h[1:] + 0).tolist(),
                "h[3].item()": back(h[3]).item(), "h[1]+h[2]": back(h[1] + h[2]).item(),
                "h[::2]": back(h[::2]).tolist(), "h.double()[1:]*1 (f32 card -> f64)":
                    back(card(v.float()).double()[1:] * 1).tolist(), "expected_h[1:]": v[1:].tolist()}

    for name, fn in [("a.meta", meta), ("a.roundtrip", roundtrip), ("a.offset_views", offset_views),
                     ("a.view_int64", view_int64_on_card),
                     ("a.view_int32", view_int32_on_card), ("a.int64_values", int64_values),
                     ("a.f32_to_f64", f32_to_f64_on_card), ("a.f64_arith", f64_arith_on_card),
                     ("a.bits_via_int32", bits_via_int32)]:
        try_(name, fn)


def mme():
    n = 64
    gen = torch.Generator().manual_seed(1)

    def mantissa_bits():
        kept = []
        for k in range(6, 24):
            A = torch.eye(n)
            A[:, 0] += 2.0 ** -k  # column 0 = 1 + 2^-k at (0, 0), 2^-k below
            B = torch.eye(n)
            got = back(card(A) @ card(B))
            kept.append((k, bool(got[0, 0].item() == 1.0 + 2.0 ** -k)))
        prod = []
        for k in range(6, 24):  # product rounding: (1 + 2^-k)(1 + 2^-k) = 1 + 2^(1-k) + 2^-2k
            a = torch.eye(n)
            a[0, 0] = 1 + 2.0 ** -k
            got = back(card(a) @ card(a))
            ref = torch.tensor((1 + 2.0 ** -k) ** 2, dtype=torch.float32).item()
            prod.append((k, bool(got[0, 0].item() == ref)))
        acc = []
        for k in range(6, 30):  # accumulation: 1 + 2^-k summed exactly in f32 iff k <= 23
            a = torch.zeros(n, n)
            a[0, 0], a[0, 1] = 1.0, 2.0 ** -k
            b = torch.zeros(n, n)
            b[0, 0], b[1, 0] = 1.0, 1.0
            got = back(card(a) @ card(b))[0, 0].item()
            acc.append((k, got - 1.0))
        return {"input_bit_kept": kept, "product_exact": prod, "acc_1_plus_2mk_minus_1": acc,
                "max_input_bit": max([k for k, ok in kept if ok], default=None)}

    def errors():
        out = {}
        A = torch.randn(32, n, n, generator=gen)
        B = torch.randn(32, n, n, generator=gen)
        ref = A.double() @ B.double()
        cpu = (A @ B).double()
        hpu = back(card(A) @ card(B)).double()
        scale = float(ref.abs().max())
        out["random_rel_cpu_f32"] = float((cpu - ref).abs().max()) / scale
        out["random_rel_hpu_f32"] = float((hpu - ref).abs().max()) / scale
        # cancelling product: rows of A orthogonal-ish to columns of B at large magnitude
        Q = torch.linalg.qr(torch.randn(n, n, generator=gen, dtype=torch.float64))[0]
        C = (Q * 1e3).float()
        D = (Q.mT * 1e3).float()
        refc = C.double() @ D.double()
        offdiag = ~torch.eye(n, dtype=torch.bool)
        out["cancel_abs_cpu_f32"] = float(((C @ D).double() - refc)[offdiag].abs().max())
        out["cancel_abs_hpu_f32"] = float((back(card(C) @ card(D)).double() - refc)[offdiag].abs().max())
        # batched matvec (bmm with one column), the shape refinement uses
        x = torch.randn(32, n, 1, generator=gen)
        refv = A.double() @ x.double()
        out["matvec_rel_cpu_f32"] = float(((A @ x).double() - refv).abs().max() / refv.abs().max())
        out["matvec_rel_hpu_f32"] = float((back(card(A) @ card(x)).double() - refv).abs().max() / refv.abs().max())
        return out

    base = torch.get_float32_matmul_precision()
    for prec in ("highest", "high", "medium"):
        torch.set_float32_matmul_precision(prec)
        try_(f"b.mme_bits[{prec}]", mantissa_bits)
        try_(f"b.mme_errors[{prec}]", errors)
    torch.set_float32_matmul_precision(base)


def linalg():
    """What a user gets today: float64 solve/inv/det on HPU tensors (hpu_ops enabled) against CPU float64 on the
    original float64 data, at condition 1e2 / 1e6 (batch 8, n = 22)."""
    gen = torch.Generator().manual_seed(2)
    n = 22
    for cond in (1e2, 1e6):
        Q = torch.linalg.qr(torch.randn(8, n, n, generator=gen, dtype=torch.float64))[0]
        A = Q @ torch.diag_embed(torch.logspace(0, -torch.log10(torch.tensor(cond)).item(), n,
                                                dtype=torch.float64).expand(8, n)) @ Q.mT
        b = torch.randn(8, n, 1, generator=gen, dtype=torch.float64)

        def rel(got, ref):
            return float((got.cpu().double() - ref).abs().max() / ref.abs().max())

        def run():
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                Ah, bh = card(A), card(b)
                x, Ai, d = torch.linalg.solve(Ah, bh), torch.linalg.inv(Ah), torch.linalg.det(Ah)
                sync()
            A32 = A.float().double()  # what the card holds
            return {"cond": cond, "out_dtype": str(x.dtype),
                    "solve_rel_vs_f64_of_input": rel(x, torch.linalg.solve(A, b)),
                    "inv_rel_vs_f64_of_input": rel(Ai, torch.linalg.inv(A)),
                    "det_rel_vs_f64_of_input": rel(d, torch.linalg.det(A)),
                    "solve_rel_vs_f64_of_card_values": rel(x, torch.linalg.solve(A32, b.float().double())),
                    "cpu_f32_solve_rel": rel(torch.linalg.solve(A.float(), b.float()), torch.linalg.solve(A, b)),
                    "warnings": sorted({str(m.message)[:90] for m in w})}

        try_(f"a.linalg[cond={cond:g}]", run)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--only", choices=("storage", "mme", "linalg"), default=None)
    a = p.parse_args()
    emit("env", torch=torch.__version__, int64_support=os.environ.get("PT_ENABLE_INT64_SUPPORT"))
    if a.only in (None, "storage"):
        storage()
    if a.only in (None, "mme"):
        mme()
    if a.only == "linalg":
        linalg()


if __name__ == "__main__":
    main()
