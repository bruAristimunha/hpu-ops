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
    a = x.clone().requires_grad_()
    torch.nan_to_num(fn(a, q, dim=1, interpolation=interp)).sum().backward()
    b = x.clone().requires_grad_()
    with cpu_as_hpu(quantile.install):
        torch.nan_to_num(fn(b, q, dim=1, interpolation=interp)).sum().backward()
    torch.testing.assert_close(b.grad, a.grad, rtol=0, atol=0)


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
