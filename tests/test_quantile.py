"""CPU stand-in tests for hpu_ops.quantile (#17): the sort/gather/lerp rewrite matches ATen's quantile and
nanquantile exactly, forward and backward, and leaves arguments ATen rejects to ATen."""
import itertools

import pytest
import torch
from test_cpu import compiled, cpu_as_hpu, ops

from hpu_ops import quantile

INTERP = ("linear", "lower", "higher", "midpoint", "nearest")


def _cases():
    torch.manual_seed(0)
    x3 = torch.randn(3, 5, 7, dtype=torch.float64)
    xn = x3.clone()
    xn[0, 1, 2] = float("nan")
    xn[2, :, 4] = float("nan")  # an all-NaN slice along dim 1
    xt = torch.tensor([1.0, 2.0, 2.0, 3.0, 4.0, 4.0])  # ties, nearest at a .5 rank (round half to even)
    qs = [0.0, 0.25, 0.5, 0.9, 1.0, torch.tensor(0.3, dtype=torch.float64),
          torch.tensor([0.0, 0.1, 0.5, 0.75, 1.0], dtype=torch.float64)]
    for x, q, dim, keepdim in itertools.product((x3, xn), qs, (None, 0, 1, -1), (False, True)):
        yield x, q, dim, keepdim
    for q in (0.5, 0.3, 0.7, 0.1):
        yield xt, q, None, False
        yield xt.double(), torch.tensor([q, 1 - q], dtype=torch.float64), 0, True
    yield x3[:, ::2, 1:].transpose(0, 2), 0.4, 1, False  # non-contiguous view with an offset
    yield torch.tensor(2.5, dtype=torch.float64), 0.3, None, True  # 0-d
    yield torch.tensor(2.5, dtype=torch.float64), 0.3, 0, True
    yield torch.randn(1, 1), 0.7, 1, False  # n = 1
    yield torch.randn(4, 9), torch.tensor([], dtype=torch.float32), 1, False  # empty q


@pytest.mark.parametrize("fn", (torch.quantile, torch.nanquantile))
def test_quantile_matches_aten_forward(fn):
    n = 0
    for (x, q, dim, keepdim), interp in itertools.product(_cases(), INTERP):
        want = fn(x, q, dim=dim, keepdim=keepdim, interpolation=interp)
        with cpu_as_hpu(quantile.install) as calls:
            got = fn(x, q, dim=dim, keepdim=keepdim, interpolation=interp)
        assert sum(calls.values()) == 1, (dim, keepdim, interp)
        assert got.shape == want.shape and got.dtype == want.dtype, (x.shape, q, dim, keepdim, interp)
        torch.testing.assert_close(got, want, rtol=0, atol=0, equal_nan=True)
        n += 1
    assert n > 500


@pytest.mark.parametrize("fn", (torch.quantile, torch.nanquantile))
@pytest.mark.parametrize("interp", INTERP)
def test_quantile_gradient_matches_aten(fn, interp):
    torch.manual_seed(1)
    x = torch.randn(4, 11, dtype=torch.float64)
    x[1, 3] = float("nan")
    q = torch.tensor([0.1, 0.5, 0.95], dtype=torch.float64)
    x[2, 5] = float("inf")
    x[3, :] = float("nan")  # an all-NaN row

    def grads():
        a, qa = x.clone().requires_grad_(), q.clone().requires_grad_()
        y = fn(a, qa, dim=1, interpolation=interp)
        gx, gq = torch.autograd.grad(y.sum(), (a, qa), allow_unused=True)  # no nan_to_num: NaN rows count
        return [y, gx, torch.zeros_like(q) if gq is None else gq]

    want = grads()
    with cpu_as_hpu(quantile.install) as calls:
        got = grads()
    assert calls[fn.__name__] == 1
    for g, w in zip(got, want):
        torch.testing.assert_close(g, w, rtol=0, atol=0, equal_nan=True)


def test_quantile_inference_mode_and_float32():
    x = torch.randn(6, 50)
    want = torch.quantile(x, 0.9, dim=1)
    with cpu_as_hpu(quantile.install) as calls, torch.inference_mode():
        got = torch.quantile(x, 0.9, dim=1)
    assert calls["quantile"] == 1
    torch.testing.assert_close(got, want, rtol=0, atol=0)


@pytest.mark.parametrize("call", [
    lambda: torch.quantile(torch.tensor([]), 0.5),
    lambda: torch.quantile(torch.randn(3), 1.5),
    lambda: torch.quantile(torch.randn(3), -0.1),
    lambda: torch.quantile(torch.randn(3), torch.tensor([0.2, 1.5])),
    lambda: torch.quantile(torch.randn(3), torch.tensor([0.2], dtype=torch.float64)),
    lambda: torch.quantile(torch.randn(3), torch.tensor([[0.2]])),
    lambda: torch.quantile(torch.arange(3), 0.5),
    lambda: torch.quantile(torch.randn(3, 0), 0.5, dim=1),
    lambda: torch.quantile(torch.randn(3), 0.5, interpolation="cubic"),
    lambda: torch.quantile(torch.randn(3, 2), 0.5, dim=2),
])
def test_quantile_invalid_arguments_raise_aten_error(call):
    with pytest.raises((RuntimeError, IndexError)) as native:
        call()
    with cpu_as_hpu(quantile.install) as calls:
        with pytest.raises(type(native.value)) as rewritten:
            call()
    assert str(rewritten.value) == str(native.value)
    assert sum(calls.values()) == 0


def test_quantile_compiled_has_no_out_resize():
    x = torch.randn(5, 40)
    q = torch.tensor([0.25, 0.75])
    f = lambda t: torch.quantile(t, q, dim=1)  # noqa: E731
    want = f(x)
    nodes = []
    with cpu_as_hpu(quantile.install) as calls:
        got = compiled(f, nodes, fullgraph=True)(x)
    torch.testing.assert_close(got, want, rtol=0, atol=0)
    assert calls["quantile"] > 0
    assert not any("resize" in o for o in ops(nodes)), ops(nodes)


@pytest.mark.parametrize("fn", (torch.quantile, torch.nanquantile))
def test_quantile_right_when_sort_returns_nan_as_inf(fn, monkeypatch):
    """Gaudi's sort returns NaN as +inf (bridge 1.21.4): the rewrite must not read NaNs from the sorted values."""
    x = torch.tensor([[3.0, float("nan"), 1.0, 2.0], [float("nan")] * 4, [4.0, 3.0, float("inf"), 1.0]])
    want = [fn(x, q, dim=1, interpolation=i) for q in (0.0, 0.5, 1.0) for i in INTERP]
    sort = torch.sort

    def gaudi_sort(t, *args, **kwargs):
        return sort(torch.nan_to_num(t, nan=float("inf"), posinf=float("inf"), neginf=float("-inf")), *args, **kwargs)

    monkeypatch.setattr(torch, "sort", gaudi_sort)
    with cpu_as_hpu(quantile.install):
        got = [fn(x, q, dim=1, interpolation=i) for q in (0.0, 0.5, 1.0) for i in INTERP]
    for g, w in zip(got, want):
        torch.testing.assert_close(g, w, rtol=0, atol=0, equal_nan=True)


# ---- review of #31 (CODEX_REVIEW.txt), with its inputs

def _both(f):
    want = f()
    with cpu_as_hpu(quantile.install) as calls:
        got = f()
        calls = dict(calls)
    return got, want, calls


def test_q_gradient_is_not_poisoned_by_a_nan_in_another_row():
    def f():
        q = torch.tensor(0.25, requires_grad=True)
        y = torch.quantile(torch.tensor([[1.0, 3.0], [2.0, float("nan")]]), q, dim=1)[0]
        return y.detach(), torch.autograd.grad(y, q)[0]

    got, want, calls = _both(f)
    assert calls == {"quantile": 1} and want[1].item() == 2.0
    for g, w in zip(got, want):
        torch.testing.assert_close(g, w, rtol=0, atol=0)


def test_gradient_reaches_the_nan_as_in_torch():
    def f():
        x = torch.tensor([1.0, float("nan"), 2.0], requires_grad=True)
        return torch.autograd.grad(torch.quantile(x, 0.5), x)[0]

    got, want, _ = _both(f)
    assert want.tolist() == [0.0, 1.0, 0.0] and torch.equal(got, want)


def test_nanquantile_genuine_inf_gets_the_gradient():
    def f():
        x = torch.tensor([float("nan"), float("inf")], requires_grad=True)
        y = torch.nanquantile(x, 1.0, interpolation="lower")
        return y.detach(), torch.autograd.grad(y, x)[0]

    got, want, _ = _both(f)
    assert want[1].tolist() == [0.0, 1.0]
    for g, w in zip(got, want):
        torch.testing.assert_close(g, w, rtol=0, atol=0)


def test_backward_memory_does_not_scale_with_q_times_input():
    """Review: 257 quantiles of 4096 values made two [257, 4096] scatter buffers in the backward; torch uses [4096]."""
    from torch.utils._python_dispatch import TorchDispatchMode

    class Largest(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            out = func(*args, **(kwargs or {}))
            for t in out if isinstance(out, (tuple, list)) else (out,):
                if isinstance(t, torch.Tensor):
                    self.most = max(self.most, t.numel())
            return out

    x = torch.randn(4096, requires_grad=True)
    q = torch.linspace(0, 1, 257)
    with cpu_as_hpu(quantile.install) as calls:
        mode = Largest()
        mode.most = 0
        with mode:
            torch.quantile(x, q).sum().backward()
    assert calls["quantile"] == 1
    assert mode.most <= 4 * x.numel(), mode.most  # 257 * 4096 before


def test_traced_q_outside_the_range_raises_at_run_time():
    f = lambda x, q: torch.quantile(x, q, interpolation="nearest")  # noqa: E731
    x = torch.tensor([2.0, 6.0])
    with cpu_as_hpu(quantile.install) as calls:
        c = compiled(f, [], fullgraph=True)
        assert c(x, torch.tensor(0.25)).item() == 2.0
        with pytest.raises(RuntimeError, match=r"q values must be in the range \[0, 1\]"):
            c(x, torch.tensor(1.1))
    assert calls["quantile"] > 0


@pytest.mark.skipif(not quantile._length_limited(), reason="this torch has no 2**24 quantile limit")
def test_reduction_length_guard_raises_aten_error():
    for call in (lambda: torch.quantile(torch.zeros(2 ** 24 + 1), 0.5),
                 lambda: torch.nanquantile(torch.zeros(2, 2 ** 24 + 1), 0.5, dim=1)):
        with pytest.raises(RuntimeError, match="input tensor is too large") as native:
            call()
        with cpu_as_hpu(quantile.install), pytest.raises(RuntimeError) as got:
            call()
        assert str(got.value) == str(native.value)


@pytest.mark.parametrize("fn", (torch.quantile, torch.nanquantile))
def test_compiled_inference_mode_takes_the_rewrite(fn):
    x = torch.tensor([3.0, float("nan"), 1.0, 2.0])
    want = fn(x, 0.5)
    f = lambda t: fn(t, 0.5)  # noqa: E731
    with cpu_as_hpu(quantile.install) as calls, torch.inference_mode():
        got = compiled(f, [], fullgraph=True)(x)
    assert calls[fn.__name__] > 0
    torch.testing.assert_close(got, want, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("fn", (torch.quantile, torch.nanquantile))
def test_out_overloads_take_the_rewrite(fn, monkeypatch):
    """With Gaudi's sort emulated (NaN returned as +inf), ATen's out= composite gave 2.5 for both."""
    x = torch.tensor([3.0, float("nan"), 1.0, 2.0])
    want = fn(x, 0.5)
    want_v = fn(x, torch.tensor([0.25, 0.5]))
    sort = torch.sort
    monkeypatch.setattr(torch, "sort", lambda t, *a, **k: sort(torch.nan_to_num(t, nan=float("inf")), *a, **k))
    with cpu_as_hpu(quantile.install) as calls:
        out, out_v = torch.empty(0), torch.empty(2)
        r = fn(x, 0.5, out=out)
        r_v = fn(x, torch.tensor([0.25, 0.5]), out=out_v)
        with pytest.raises(RuntimeError, match="same dtype"):
            fn(x, 0.5, out=torch.empty(0, dtype=torch.float64))
    assert calls[fn.__name__] == 2
    assert r is out and r_v is out_v
    torch.testing.assert_close(out, want, rtol=0, atol=0, equal_nan=True)
    torch.testing.assert_close(out_v, want_v, rtol=0, atol=0, equal_nan=True)


def test_out_keeps_autograd_like_aten():
    x = torch.tensor([1.0, 2.0, 4.0], requires_grad=True)
    want = torch.quantile(x, 0.5, out=torch.empty(0))
    with cpu_as_hpu(quantile.install):
        got = torch.quantile(x, 0.5, out=torch.empty(0))
    assert got.requires_grad == want.requires_grad
    torch.testing.assert_close(torch.autograd.grad(got, x)[0], torch.autograd.grad(want, x)[0])


def test_vector_q_result_has_atens_layout():
    x = torch.randn(2, 3, 5)
    q = torch.tensor([0.25, 0.75])
    want = torch.quantile(x, q, dim=1)
    with cpu_as_hpu(quantile.install):
        got = torch.quantile(x, q, dim=1)
    assert got.stride() == want.stride()
    torch.testing.assert_close(got.movedim(0, -1).view(-1), want.movedim(0, -1).view(-1), rtol=0, atol=0)


def test_traced_q_outside_the_range_is_nan_where_the_device_has_no_assert(monkeypatch):
    """Gaudi has no _assert_async kernel (one CPU fallback per call): there an out-of-range traced q gives NaN."""
    monkeypatch.setattr(quantile, "_asserts_on", lambda device_type: False)
    f = lambda x, q: torch.quantile(x, q, interpolation="nearest")  # noqa: E731
    x = torch.tensor([2.0, 6.0])
    nodes = []
    with cpu_as_hpu(quantile.install):
        c = compiled(f, nodes, fullgraph=True)
        assert c(x, torch.tensor(0.25)).item() == 2.0
        assert torch.isnan(c(x, torch.tensor(1.1))).all()
    assert "aten._assert_async.msg" not in ops(nodes)


def test_rank_precision_follows_the_installed_torch():
    """torch 2.15 computes a float32 input's ranks in float64 (nearest of 0.7 over 6 values: index 3), 2.6-2.8 in
    float32 (index 4); the rewrite reads which from ATen and matches it."""
    x = torch.arange(6.0)
    for q in (0.1, 0.7, torch.tensor([0.1, 0.7])):
        for interp in INTERP:
            want = torch.quantile(x, q, interpolation=interp)
            with cpu_as_hpu(quantile.install):
                got = torch.quantile(x, q, interpolation=interp)
            torch.testing.assert_close(got, want, rtol=0, atol=0)
