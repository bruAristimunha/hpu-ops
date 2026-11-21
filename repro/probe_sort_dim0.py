"""Gaudi2: Gaudi's non-stable float sort along a non-last dim: are the indices a permutation picking the values?
Data with ties, -0.0, +-max; per n, direction, ingredient."""
import habana_frameworks.torch.hpu as hthpu
import torch

MX = torch.finfo(torch.float32).max
for n in (3, 8, 22, 64, 100):
    for ing in ("randn", "ties", "neg0", "max", "-max", "all"):
        for desc in (False, True):
            g = torch.Generator().manual_seed(n)
            x = torch.randn(8, n, generator=g)
            if ing in ("ties", "all"):
                x[:, 5::9] = 0.5
            if ing in ("neg0", "all"):
                x[:, 2::6] = -0.0
            if ing in ("max", "all"):
                x[:, ::7] = MX
            if ing in ("-max", "all"):
                x[:, 3::5] = -MX
            xd = x.mT.contiguous()  # (n, 8): sort along dim 0
            v, i = xd.to("hpu").sort(0, desc)
            hthpu.synchronize()
            v, i = v.cpu(), i.cpu()
            inr = bool(((i >= 0) & (i < n)).all())
            perm = inr and torch.equal(i.sort(0).values, torch.arange(n).unsqueeze(1).expand(n, 8))
            picks = inr and torch.equal(xd.gather(0, i), v)
            ok_v = torch.equal(v, xd.sort(0, desc).values)
            if not (perm and picks and ok_v):
                print(f"DIM0 n={n:4d} {ing:6s} desc={desc} values={ok_v} in_range={inr} perm={perm} picks={picks}",
                      flush=True)
print("DONE", flush=True)
