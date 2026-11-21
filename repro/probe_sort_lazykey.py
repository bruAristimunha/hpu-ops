"""Gaudi2 lazy mode: which int32 total-order key of float32 builds (the view + bitshift form fails graph compile:
"bitshift_fwd_f32"), exact against the CPU key.
    PT_HPU_LAZY_MODE=1 python3 repro/probe_sort_lazykey.py"""
import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.hpu as hthpu
import torch

INF, NAN = float("inf"), float("nan")
I32_MIN, I32_MAX = -2 ** 31, 2 ** 31 - 1
NAN_KEY = 0x7F800001


def ref(x):
    b = x.view(torch.int32)
    s = b >> 31
    return torch.where(torch.isnan(x), NAN_KEY, (b ^ (s & I32_MAX)) - s)


def k_where(x):
    b = x.view(torch.int32)
    return torch.where(torch.isnan(x), NAN_KEY, torch.where(b < 0, I32_MIN - b, b))


def k_clone(x):
    b = x.view(torch.int32).clone()
    s = b >> 31
    return torch.where(torch.isnan(x), NAN_KEY, (b ^ (s & I32_MAX)) - s)


def k_signbit(x):
    b = x.view(torch.int32)
    return torch.where(torch.isnan(x), NAN_KEY, torch.where(torch.signbit(x), I32_MIN - b, b))


def k_mul(x):  # no shift: s = -(b < 0)
    b = x.view(torch.int32)
    neg = b < 0
    return torch.where(torch.isnan(x), NAN_KEY, torch.where(neg, (b ^ I32_MAX) + 1, b))


def k_mark(x):  # the view materialised in its own graph
    b = x.view(torch.int32) + 0
    htcore.mark_step()
    return torch.where(torch.isnan(x), NAN_KEY, torch.where(b < 0, I32_MIN - b, b))


g = torch.Generator().manual_seed(0)
x = torch.randn(16, 1000, generator=g)
x[:, ::7], x[:, 3::11], x[:, 5::13], x[:, 6::17], x[:, 1::5] = INF, -INF, 0.5, -0.0, NAN
x[:, 9::23] = 1e-40
want = ref(x)
for name, f in (("where", k_where), ("clone", k_clone), ("signbit", k_signbit), ("mul", k_mul), ("mark", k_mark)):
    try:
        h = x.to("hpu")
        k = f(h)
        _, i = torch.sort(k, dim=-1)
        _, j = torch.sort(k, dim=-1, stable=True)
        htcore.mark_step()
        hthpu.synchronize()
        ok = torch.equal(k.cpu(), want)
        ws = torch.sort(x, dim=-1, stable=True).indices
        print(f"LAZYKEY {name:8s} key_exact={ok} stable_exact={torch.equal(j.cpu(), ws)} "
              f"unstable_in_range={bool(((i.cpu() >= 0) & (i.cpu() < 1000)).all())}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"LAZYKEY {name:8s} ERROR {type(e).__name__}: {str(e)[:160]}", flush=True)
print("DONE", flush=True)
