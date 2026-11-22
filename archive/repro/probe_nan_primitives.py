"""Gaudi2 primitives the NaN rewrites rely on: any over dims, argmax ties (first index), argmax out of range,
max_pool index format."""
import os

import habana_frameworks.torch.core as htcore
import torch
import torch.nn.functional as F

LAZY = os.environ.get("PT_HPU_LAZY_MODE", "1") != "0"
NAN, INF = float("nan"), float("inf")


def show(name, f, *xs):
    try:
        got = f(*[x.to("hpu") for x in xs])
        if LAZY:
            htcore.mark_step()
        got = [g.cpu().tolist() for g in got] if isinstance(got, tuple) else got.cpu().tolist()
    except Exception as e:  # noqa: BLE001
        got = f"raised {type(e).__name__}: {str(e)[:150]}"
    want = f(*xs)
    want = [w.tolist() for w in want] if isinstance(want, tuple) else want.tolist()
    print(f"{'SAME' if str(got) == str(want) else 'DIFF'} {name}\n   hpu {got}\n   cpu {want}", flush=True)


m = torch.tensor([[0.0, 1, 1, 0], [0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 1]])
for n in (4, 1000, 70000):
    z = torch.zeros(3, n)
    z[0, n // 2:] = 1
    z[2] = 1
    show(f"argmax f32 0/1 ties n={n}", lambda t: t.argmax(-1), z)
    show(f"argmax f32 0/1 ties dim0 n={n}", lambda t: t.argmax(0), z.t().contiguous())
    show(f"argmax uint8 n={n}", lambda t: t.to(torch.uint8).argmax(-1), z)
    show(f"any dims n={n}", lambda t: (t > 0).any((0, 1)), z)
    show(f"any dim keep n={n}", lambda t: (t > 0).any(-1, keepdim=True), z)
for dt in (torch.float32, torch.bfloat16, torch.float16):
    low = -3e38 if dt != torch.float16 else -6e4
    r = torch.tensor([[-INF] * 4, [INF] * 4, [1.0] * 4, [-INF, -INF, 2.0, -INF], [low] * 4])
    show(f"argmax {dt} edge rows", lambda t: t.to(dt).argmax(-1), r)
    show(f"argmin {dt} edge rows", lambda t: t.to(dt).argmin(-1), r)
    show(f"max.dim {dt} edge rows", lambda t: tuple(t.to(dt).max(-1)), r)
    big = torch.full((2, 5000), -INF)
    show(f"argmax {dt} all -inf n=5000", lambda t: t.to(dt).argmax(-1), big)
x = torch.arange(16.0).reshape(1, 1, 4, 4)
show("max_pool2d idx no NaN", lambda t: tuple(F.max_pool2d(t, 2, return_indices=True)), x)
show("max_pool2d idx shuffled", lambda t: tuple(F.max_pool2d(t, 2, return_indices=True)), x.flip(-1))
show("max_pool1d idx", lambda t: tuple(F.max_pool1d(t, 2, return_indices=True)), torch.arange(8.0).reshape(1, 1, 8))
show("max_pool2d idx k3 s1 p1", lambda t: tuple(F.max_pool2d(t, 3, 1, 1, return_indices=True)), x)
show("max_pool3d idx", lambda t: tuple(F.max_pool3d(t, 2, return_indices=True)),
     torch.arange(64.0).reshape(1, 1, 4, 4, 4))
xn = torch.tensor([[[NAN, NAN, 1.0, NAN, 2.0, 0.0]]])
show("max_pool1d nan idx (cpu: last NaN)", lambda t: tuple(F.max_pool1d(t, 3, return_indices=True)), xn)
# max_pool backward through MaxUnpool path: does the native backward accept its own indices
xg = torch.randn(2, 3, 8, 8)
for dev in ("cpu", "hpu"):
    t = xg.to(dev).requires_grad_()
    F.max_pool2d(t, 2).sum().backward()
    print(f"pool grad {dev} nonzero={int((t.grad != 0).sum())}", flush=True)
