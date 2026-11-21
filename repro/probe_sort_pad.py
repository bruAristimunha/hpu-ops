"""Gaudi2: which values tie with the non-stable sort's padding (index out of range): +-inf, +-max, the float just
below max, per dtype, n and direction."""
import habana_frameworks.torch.hpu as hthpu
import torch

INF = float("inf")
for dt in (torch.float32, torch.bfloat16, torch.float16):
    mx = torch.finfo(dt).max
    below = torch.nextafter(torch.tensor(mx, dtype=torch.float32), torch.tensor(0.0)).item() if dt == torch.float32 \
        else float(torch.tensor(mx, dtype=dt).float() - torch.finfo(dt).eps * 2 ** (torch.finfo(dt).bits - 1))
    below = torch.tensor(mx, dtype=dt)
    below = torch.nextafter(below.float(), torch.tensor(0.0)).to(dt) if dt == torch.float32 else \
        (below.float() * (1 - torch.finfo(dt).eps)).to(dt)
    for n in (3, 22, 100, 1000):
        for name, val in (("inf", INF), ("max", mx), ("below_max", below.item())):
            for desc in (False, True):
                g = torch.Generator().manual_seed(n)
                x = torch.randn(16, n, generator=g)
                x[:, ::3] = -val if desc else val
                x = x.to(dt)
                v, i = x.to("hpu").sort(-1, desc)
                hthpu.synchronize()
                i = i.cpu()
                oor = int(((i < 0) | (i >= n)).sum())
                print(f"PAD {str(dt):15s} n={n:5d} {'-' if desc else '+'}{name:9s} desc={desc} out_of_range={oor}",
                      flush=True)
