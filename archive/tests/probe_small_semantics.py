"""Gaudi probe (issue #21, round 2): the semantics the candidate rewrites rely on, eager or lazy (PT_HPU_LAZY_MODE).

- matmul precision: fp32 accumulation, or TF32/bf16 (a ones-triangle matmul of values with 23-bit mantissas).
- inf * 0 inside matmul: IEEE gives NaN, which breaks cumsum / one-hot rewrites by matmul on non-finite input.
- histc with bins n over [0, n - 1] (integer bounds: float bounds such as -0.5 abort Habana's eager histc) of integer-valued floats: exact counts, equal to bincount.
"""
import habana_frameworks.torch.core as htcore
import torch

hpu = torch.device("hpu")


def sync(t):
    htcore.mark_step()
    return t.cpu()


def main():
    torch.manual_seed(0)
    x = torch.randn(64, 22, 22)
    tri = torch.ones(22, 22).triu()
    got = sync(x.to(hpu) @ tri.to(hpu))
    ref = torch.cumsum(x.double(), -1)
    print("matmul tri err vs float64 cumsum", (got.double() - ref).abs().max().item(),
          "cpu f32 cumsum err", (torch.cumsum(x, -1).double() - ref).abs().max().item(), flush=True)
    big = torch.tensor([[1.0 + 2 ** -20, 2 ** -22, 2 ** -23]])
    print("fine-mantissa matmul", sync(big.to(hpu) @ torch.ones(3, 1, device=hpu)).item() - 1.0,
          "expected", (big.double().sum() - 1).item(), flush=True)
    inf = float("inf")
    v = torch.tensor([[1.0, inf, 2.0]])
    print("inf in matmul with triangle", sync(v.to(hpu) @ torch.ones(3, 3).triu().to(hpu)).tolist(),
          "cumsum", torch.cumsum(v, -1).tolist(), flush=True)
    for n, bins in ((4096, 64), (409600, 64), (100, 3)):
        i = torch.randint(0, bins, (n,))
        h = sync(torch.histc(i.to(hpu).float(), bins, 0, bins - 1)).long()
        print(f"histc exact n={n} bins={bins}", torch.equal(h, torch.bincount(i, minlength=bins)), flush=True)
    ns = list(range(1, 130)) + [200, 255, 256, 257, 300, 400, 511, 512, 513, 600, 800, 1000, 1024, 2048, 4096]
    bad, fail = [], []
    for n in ns:
        try:
            h = sync(torch.histc(torch.arange(n, device=hpu).float(), n, 0, n - 1)).long()
            if not torch.equal(h, torch.ones(n, dtype=torch.long)):
                bad.append(n)
        except RuntimeError:
            fail.append(n)
    print("histc arange(n) one per bin: inexact at n =", bad, "fails to compile at n =", fail, flush=True)
    i = torch.randint(0, 64, (4096,))
    print("habana bincount exact", torch.equal(sync(torch.bincount(i.to(hpu))), torch.bincount(i)),
          "dtype", torch.bincount(i.to(hpu)).dtype, flush=True)


if __name__ == "__main__":
    main()
