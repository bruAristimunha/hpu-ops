"""float64 offset views (writes, correct-copy candidates, autograd) and int64 past int32 (Gaudi2, bridge 1.21.4).
    PT_HPU_LAZY_MODE=0|1 python repro/probe_f64_int64.py f64|i64"""
import os
import sys

import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.hpu as hthpu
import torch

LAZY = os.environ.get("PT_HPU_LAZY_MODE", "1") != "0"
hpu = torch.device("hpu")


def run(name, fn, ref):
    try:
        got = fn()
        if LAZY:
            htcore.mark_step()
        hthpu.synchronize()
        got = got.cpu() if isinstance(got, torch.Tensor) else got
        same = torch.equal(got.to(ref.dtype), ref) if got.shape == ref.shape else False
        print(f"{'ok  ' if same else 'BAD '} {name}" + ("" if same else
              f": got {got.reshape(-1)[:6].tolist()} want {ref.reshape(-1)[:6].tolist()}"), flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"RAISE {name}: {str(e).splitlines()[0][:140]}", flush=True)


def f64():
    c = torch.arange(1, 13, dtype=torch.float64)

    def fresh():
        h = c.to(hpu)
        if LAZY:
            htcore.mark_step()
        return h
    # writes through offset views
    for name, w in {
        "copy_ into h[1:4]": lambda h: h[1:4].copy_(torch.tensor([-1.0, -2, -3], dtype=torch.float64, device=hpu)),
        "setitem h[6:] = 0": lambda h: h.__setitem__(slice(6, None), 0.0),
        "add_ on h[2:5]": lambda h: h[2:5].add_(100),
        "fill_ row1": lambda h: h.view(2, 6)[1].fill_(7.5),
        "add_ col slice": lambda h: h.view(3, 4)[:, 1:].add_(100),
    }.items():
        ref = c.clone()
        w(ref)
        run("write " + name, lambda: (lambda h: (w(h), h)[1])(fresh()), ref)
    # candidate correct copies of a contiguous offset view
    v = c[3:9]
    cands = {
        "clone": lambda h: h[3:9].clone(),
        "expand2 clone [...,0]": lambda h: h[3:9].unsqueeze(-1).expand(6, 2).clone()[:, 0].contiguous(),
        "stride0 as_strided clone": lambda h: h[3:9].as_strided((6, 2), (1, 0)).clone()[:, 0],
        "base index_select": lambda h: h.index_select(0, torch.arange(3, 9, device=hpu)),
        "base[...] narrow via view(n,1)": lambda h: h.view(12, 1)[3:9, 0].clone(),
        "cpu roundtrip": lambda h: h[3:9].cpu().to(hpu),
        "to f32 base then slice": lambda h: h.float()[3:9].double(),
        "2d col view clone": lambda h: h.view(6, 2)[:, :].clone().view(-1)[6:].clone(),
    }
    for name, f in cands.items():
        run("copy " + name, lambda: f(fresh()), v)
    # autograd: gradient of a slice and of a sum over rows
    def grad_case(h):
        h = h.clone().requires_grad_()
        (h[3:9] * torch.arange(6, dtype=torch.float64, device=hpu)).sum().backward()
        return h.grad
    ref = torch.zeros(12, dtype=torch.float64)
    ref[3:9] = torch.arange(6, dtype=torch.float64)
    run("grad of slice", lambda: grad_case(fresh()), ref)
    def chunk_case(h):
        a, b = h.view(2, 6).unbind(0)
        return a * b
    run("unbind rows product", lambda: chunk_case(fresh()), c[:6] * c[6:])
    # float64 created on the card (not copied)
    run("arange on card [2:]+0", lambda: torch.arange(1, 13, dtype=torch.float64, device=hpu)[2:] + 0, c[2:])
    run("f32->double on card [2:]+0", lambda: torch.arange(1, 13, device=hpu).double()[2:] + 0, c[2:])
    run("int64 [2:]+0", lambda: torch.arange(1, 13, device=hpu)[2:] + 0, torch.arange(3, 13))


def i64():
    big = torch.tensor([2 ** 40 + 1, -(2 ** 35) - 3, 2 ** 31, 7], dtype=torch.int64)
    small = torch.tensor([5, -3, 2, 7], dtype=torch.int64)

    def h(t=big):
        x = t.to(hpu)
        if LAZY:
            htcore.mark_step()
        return x
    run("roundtrip", lambda: h(), big)
    run("clone", lambda: h().clone(), big)
    run("view[1:]", lambda: h()[1:].clone(), big[1:])
    run("add 1", lambda: h() + 1, big + 1)
    run("add tensor", lambda: h() + h(small), big + small)
    run("sub", lambda: h() - 1, big - 1)
    run("mul 3", lambda: h() * 3, big * 3)
    run("neg", lambda: -h(), -big)
    run("abs", lambda: h().abs(), big.abs())
    run("sum", lambda: h().sum(), big.sum())
    run("cumsum", lambda: h().cumsum(0), big.cumsum(0))
    run("max", lambda: h().max(), big.max())
    run("eq", lambda: h() == h(), big == big)
    run("gt 2**32", lambda: h() > 2 ** 32, big > 2 ** 32)
    run("floor_divide 2", lambda: torch.div(h(), 2, rounding_mode="floor"), torch.div(big, 2, rounding_mode="floor"))
    run("remainder 10", lambda: h() % 10, big % 10)
    run("bitwise_and", lambda: h() & 0xFFFF, big & 0xFFFF)
    run("shift >> 8", lambda: h() >> 8, big >> 8)
    run("to float32", lambda: h().float(), big.float())
    run("float32 -> int64", lambda: big.float().to(hpu).long(), big.float().long())
    run("where", lambda: torch.where(h() > 0, h(), h(small)), torch.where(big > 0, big, small))
    run("cat", lambda: torch.cat([h(), h()]), torch.cat([big, big]))
    run("sort", lambda: h().sort().values, big.sort().values)
    run("index_select", lambda: h().index_select(0, torch.tensor([3, 0], device=hpu)), big[[3, 0]])
    run("gather data", lambda: torch.gather(h(), 0, torch.tensor([0, 2], device=hpu)), big[[0, 2]])
    run("arange big on card", lambda: torch.arange(2 ** 33, 2 ** 33 + 4, device=hpu), torch.arange(2 ** 33, 2 ** 33 + 4))
    run("full big on card", lambda: torch.full((3,), 2 ** 40, device=hpu, dtype=torch.int64), torch.full((3,), 2 ** 40))
    run("small mul to 2**40", lambda: h(torch.tensor([2 ** 20])) * (2 ** 20), torch.tensor([2 ** 40]))
    run("small add to past int32", lambda: h(torch.tensor([2 ** 31 - 1])) + 1, torch.tensor([2 ** 31]))
    run("index with small int64", lambda: torch.arange(10.0, device=hpu)[h(torch.tensor([9, 1]))], torch.tensor([9.0, 1.0]))


print("mode", "lazy" if LAZY else "eager", os.environ.get("PT_ENABLE_INT64_SUPPORT"), flush=True)
{"f64": f64, "i64": i64}[sys.argv[1]]()
