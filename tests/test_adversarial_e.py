"""Adversarial CPU checks (review round 2, team E) of the #20 index rewrites and the #19 large-kernel conv split, with
"cpu" in the role of "hpu". Each test compares the rewrite with the native op on the same inputs."""
import itertools

import pytest
import torch
import torch.nn.functional as F
from test_cpu import compiled, cpu_as_hpu, ops

from hpu_ops import composite, conv


def native_and_rewrite(install, fn, *args):
    want = fn(*args)
    with cpu_as_hpu(install) as calls:
        got = fn(*args)
    return want, got, dict(calls)


# ---- index_2d


def test_index_2d_broadcast_index_shapes_and_batch_one():
    x = torch.randn(1, 5, 6, dtype=torch.float64)
    i0, i1 = torch.tensor([[0], [4], [-1]]), torch.tensor([1, -6, 5, 0])
    want, got, calls = native_and_rewrite(composite.install_index_2d, lambda t: t[..., i0, i1], x)
    assert calls.get("index_2d") == 1
    assert got.shape == want.shape
    torch.testing.assert_close(got, want, atol=0, rtol=0)


def test_index_2d_one_by_one_and_offset_view():
    base = torch.randn(4, 7, 7, dtype=torch.float64)
    x = base[1:3, 2:3, 3:4]  # offset, non-contiguous, 1 x 1 trailing
    i0, i1 = torch.tensor([0, -1, 0]), torch.tensor([0, 0, -1])
    want, got, calls = native_and_rewrite(composite.install_index_2d, lambda t: t[..., i0, i1], x)
    assert calls.get("index_2d") == 1
    torch.testing.assert_close(got, want, atol=0, rtol=0)


def test_index_2d_transposed_trailing_dims_and_grad():
    x = torch.randn(3, 6, 5, dtype=torch.float64).transpose(-1, -2)
    i0, i1 = torch.triu_indices(5, 6)
    xa = x.detach().clone().requires_grad_()
    want = xa[..., i0, i1]
    want.pow(2).sum().backward()
    xb = x.detach().clone().requires_grad_()
    with cpu_as_hpu(composite.install_index_2d) as calls:
        got = xb[..., i0, i1]
        got.pow(2).sum().backward()
    assert calls["index_2d"] == 1
    torch.testing.assert_close(got, want, atol=0, rtol=0)
    torch.testing.assert_close(xb.grad, xa.grad, atol=0, rtol=0)


def test_index_2d_duplicate_indices_double_backward():
    x = torch.randn(2, 4, 4, dtype=torch.float64, requires_grad=True)
    i0, i1 = torch.tensor([0, 0, 3, 3]), torch.tensor([1, 1, 2, -2])
    f = lambda t: t[..., i0, i1].pow(3).sum()  # noqa: E731
    with cpu_as_hpu(composite.install_index_2d) as calls:
        assert torch.autograd.gradgradcheck(lambda t: t[..., i0, i1].pow(3), (x,))
        torch.autograd.gradcheck(f, (x,))
    assert calls["index_2d"] > 0


def test_index_2d_int64_and_bool_data_and_complex():
    for x in (torch.arange(24).reshape(2, 3, 4), torch.arange(24).reshape(2, 3, 4) % 3 == 0,
              torch.randn(2, 3, 4, dtype=torch.complex128), torch.randn(2, 3, 4).to(torch.bfloat16)):
        i0, i1 = torch.tensor([0, 2, -1]), torch.tensor([3, 0, -4])
        want, got, calls = native_and_rewrite(composite.install_index_2d, lambda t: t[..., i0, i1], x)
        assert calls.get("index_2d") == 1
        assert got.dtype == want.dtype
        assert torch.equal(got, want)


def test_index_2d_vmap_matches_native():
    x = torch.randn(3, 4, 5, dtype=torch.float64)
    i0, i1 = torch.tensor([0, 3]), torch.tensor([4, 1])
    f = torch.vmap(lambda t: t[i0, i1])
    want, got, _ = native_and_rewrite(composite.install_index_2d, f, x)
    torch.testing.assert_close(got, want, atol=0, rtol=0)


def test_index_2d_index_error_message_names_the_same_dimension_as_native():
    x = torch.arange(24.0).reshape(2, 3, 4)
    i0, i1 = torch.tensor([0]), torch.tensor([7])
    with pytest.raises(IndexError) as native:
        _ = x[..., i0, i1]
    with cpu_as_hpu(composite.install_index_2d):
        with pytest.raises(IndexError) as rewritten:
            _ = x[..., i0, i1]
    assert str(rewritten.value) == str(native.value)


def test_index_2d_compiled_device_index_tensor_inputs():
    # index tensors passed as graph inputs (not constants), dynamic batch, fullgraph
    x = torch.randn(4, 6, 6, dtype=torch.float64)
    i0, i1 = torch.triu_indices(6, 6)
    f = lambda t, a, b: t[..., a, b]  # noqa: E731
    want = f(x, i0, i1)
    nodes = []
    with cpu_as_hpu(composite.install_index_2d) as calls:
        got = compiled(f, nodes, dynamic=True, fullgraph=True)(x, i0, i1)
    torch.testing.assert_close(got, want, atol=0, rtol=0)
    assert calls["index_2d"] > 0 and "aten.index.Tensor" not in ops(nodes)


# ---- bool index_put


def test_bool_index_put_non_accumulate_dtype_mismatch_matches_native():
    # native index_put_ (not __setitem__) checks that source and destination dtypes match
    x = torch.zeros(3, 4)
    mask = torch.tensor([True, False, True])
    v = torch.tensor(1.5, dtype=torch.float64)
    try:
        want = x.clone().index_put_([mask], v)
        native = None
    except RuntimeError as e:
        native = e
    with cpu_as_hpu(composite.install_index_put) as calls:
        try:
            got = x.clone().index_put_([mask], v)
            rewritten = None
        except RuntimeError as e:
            rewritten = e
    if native is None:
        assert rewritten is None and torch.equal(got, want)
    else:
        assert rewritten is not None, f"native raised {native!r}; rewrite returned {got} ({dict(calls)})"


def test_bool_index_put_float_into_int_truncates_as_native():
    x = torch.zeros(4, dtype=torch.int32)
    mask = torch.tensor([True, False, True, False])
    for v in (torch.tensor(-2.7), torch.tensor(3.9)):
        want = x.clone()
        want[mask] = v
        with cpu_as_hpu(composite.install_index_put):
            got = x.clone()
            got[mask] = v
        assert torch.equal(got, want), (v, got, want)


def test_bool_index_put_value_aliases_self():
    x = torch.arange(6.0).reshape(2, 3)
    mask = torch.tensor([[False, True, True], [True, False, True]])
    want = x.clone()
    want.index_put_([mask], want[0, 0])
    got = x.clone()
    with cpu_as_hpu(composite.install_index_put):
        got.index_put_([mask], got[0, 0])
    assert torch.equal(got, want)


def test_bool_index_put_value_grad_and_double_backward():
    x = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
    v = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    mask = torch.randn(3, 4) > 0

    def f(t, w, acc):
        return (t * 1.0).index_put([mask], w, accumulate=acc).pow(3)

    for acc in (False, True):
        with cpu_as_hpu(composite.install_index_put) as calls:
            assert torch.autograd.gradgradcheck(lambda t, w: f(t, w, acc), (x, v))
        assert calls["bool_index_put"] > 0


def test_bool_index_put_empty_and_all_false_mask_and_offset_view():
    """torch._debug_has_internal_overlap returns 2 (TOO_HARD, not 1 = YES) for non-dense views such as x[1:4, 2:5]:
    the in-place guard must still take them (only an expanded self, YES, goes back to ATen)."""
    for shape in ((0, 4), (3, 0), (1, 1)):
        x = torch.randn(*shape)
        mask = torch.zeros(shape[0], dtype=torch.bool)
        want, got, _ = native_and_rewrite(composite.install_index_put,
                                          lambda t: t.clone().index_put_([mask], torch.tensor(5.0)), x)
        assert torch.equal(got, want)
    base = torch.randn(5, 6)
    want = base.clone()
    want[1:4, 2:5][torch.tensor([True, False, True])] = -1.0
    got = base.clone()
    with cpu_as_hpu(composite.install_index_put) as calls:
        got[1:4, 2:5].index_put_([torch.tensor([True, False, True])], torch.tensor(-1.0))
    assert calls["bool_index_put_"] == 1
    assert torch.equal(got, want)


def test_bool_index_put_nan_inf_and_negative_zero_value():
    x = torch.tensor([1.0, -0.0, float("inf"), float("nan")])
    mask = torch.tensor([True, True, False, True])
    for v, acc in itertools.product((float("nan"), float("-inf"), -0.0), (False, True)):
        val = torch.tensor(v)
        want = x.clone().index_put_([mask], val, accumulate=acc)
        with cpu_as_hpu(composite.install_index_put):
            got = x.clone().index_put_([mask], val, accumulate=acc)
        torch.testing.assert_close(got, want, equal_nan=True, atol=0, rtol=0)
        assert torch.equal(torch.signbit(got), torch.signbit(want)), (v, acc, got, want)


def test_bool_index_put_compiled_inplace_on_input_view():
    def f(t):
        y = t.clone()
        y[:, 1:][y[:, 1:] > 0] = 0.0  # mask over a column-offset view of an intermediate
        return y

    x = torch.randn(4, 5, dtype=torch.float64)
    want = f(x)
    nodes = []
    with cpu_as_hpu(composite.install_index_put):
        got = compiled(f, nodes, fullgraph=True)(x)
    assert torch.equal(got, want)


# ---- large-kernel conv split


@pytest.fixture
def small_max(monkeypatch):
    monkeypatch.setattr(conv, "MAX_KERNEL", 4)


CONV_CASES = [
    # (input shape, weight shape, stride, padding, dilation, groups)
    ((2, 3, 17), (4, 3, 9), 1, 0, 1, 1),
    ((1, 2, 30), (2, 1, 11), 3, 5, 2, 2),  # stride, padding, dilation, grouped
    ((0, 3, 20), (4, 3, 9), 1, 0, 1, 1),  # empty batch
    ((2, 2, 9), (3, 2, 9), 1, 0, 1, 1),  # kernel == input length (output length 1)
    ((2, 1, 5), (2, 1, 9), 1, 8, 1, 1),  # padding larger than input: chunks fully in padding
    ((1, 2, 6, 13), (3, 2, 5, 9), (2, 3), (2, 4), (1, 2), 1),
    ((1, 1, 9, 9, 9), (1, 1, 5, 6, 7), 1, 1, 1, 1),  # every dim split
]


@pytest.mark.parametrize("case", CONV_CASES)
def test_large_kernel_split_forward_grad_and_double_backward(small_max, case):
    ishape, wshape, stride, padding, dilation, groups = case
    torch.manual_seed(0)
    x = torch.randn(*ishape, dtype=torch.float64, requires_grad=True)
    w = torch.randn(*wshape, dtype=torch.float64, requires_grad=True)
    b = torch.randn(wshape[0], dtype=torch.float64, requires_grad=True)
    f = {3: F.conv1d, 4: F.conv2d, 5: F.conv3d}[len(ishape)]
    fn = lambda a, c, d: f(a, c, d, stride, padding, dilation, groups)  # noqa: E731
    want = fn(x, w, b)
    gw = torch.autograd.grad(want.pow(2).sum(), (x, w, b), allow_unused=True)
    with cpu_as_hpu(conv.install_large_kernel) as calls:
        got = fn(x, w, b)
        gg = torch.autograd.grad(got.pow(2).sum(), (x, w, b), allow_unused=True)
        if x.numel():
            assert torch.autograd.gradgradcheck(fn, (x, w, b))
    assert calls["large_conv"] > 0
    assert got.shape == want.shape
    torch.testing.assert_close(got, want, atol=1e-10, rtol=1e-10)
    for a, c in zip(gg, gw):
        torch.testing.assert_close(a, c, atol=1e-10, rtol=1e-10)


def test_large_kernel_split_forward_mode_ad(small_max):
    x = torch.randn(1, 2, 20, dtype=torch.float64)
    w = torch.randn(3, 2, 9, dtype=torch.float64)
    tx, tw = torch.randn_like(x), torch.randn_like(w)
    want = torch.func.jvp(lambda a, c: F.conv1d(a, c), (x, w), (tx, tw))
    with cpu_as_hpu(conv.install_large_kernel):
        got = torch.func.jvp(lambda a, c: F.conv1d(a, c), (x, w), (tx, tw))
    for a, c in zip(got, want):
        torch.testing.assert_close(a, c, atol=1e-10, rtol=1e-10)


def test_large_kernel_split_near_overflow_float32_is_non_finite_where_native_is(small_max):
    """Known difference (documented in conv._split_conv): the split sums chunk outputs in another order, so where the
    float32 reduction already overflows, the split can give NaN (+inf + -inf) where native gives inf. Both are wrong
    against float64 (0 here); the split must stay non-finite exactly where native is."""
    x = torch.full((1, 1, 12), 3e38)
    w = torch.tensor([[[1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 0.0]]])
    want = F.conv1d(x, w)
    with cpu_as_hpu(conv.install_large_kernel) as calls:
        got = F.conv1d(x, w)
    assert calls["large_conv"] == 1
    assert torch.equal(torch.isfinite(got), torch.isfinite(want))
    xs = torch.randn(1, 1, 12)  # away from the limit the split matches native
    with cpu_as_hpu(conv.install_large_kernel):
        got = F.conv1d(xs, w)
    torch.testing.assert_close(got, F.conv1d(xs, w))


def test_large_kernel_split_subnormal_float32(small_max):
    x = torch.full((1, 1, 12), 1e-39)  # subnormal: FTZ on the card would flush it
    w = torch.ones(1, 1, 9)
    want = F.conv1d(x, w)
    with cpu_as_hpu(conv.install_large_kernel):
        got = F.conv1d(x, w)
    torch.testing.assert_close(got, want, atol=0, rtol=1e-6)


def test_large_kernel_split_padding_mode_circular_module(small_max):
    m = torch.nn.Conv1d(2, 3, 9, padding=4, padding_mode="circular").double()
    x = torch.randn(2, 2, 15, dtype=torch.float64)
    want = m(x)
    with cpu_as_hpu(conv.install_large_kernel) as calls:
        got = m(x)
    assert calls["large_conv"] == 1
    torch.testing.assert_close(got, want, atol=1e-10, rtol=1e-10)


def test_bool_index_put_expanded_self_still_goes_to_aten():
    x = torch.zeros(1).expand(3)  # overlap YES: copy_ refuses it, ATen writes it
    assert torch._debug_has_internal_overlap(x) == 1
    with cpu_as_hpu(composite.install_index_put) as calls:
        x.index_put_([torch.tensor([True, False, True])], torch.tensor(1.0))
    assert "bool_index_put_" not in calls


def test_index_2d_plain_2d_gather_with_device_indices_keeps_native():
    g = torch.randn(8, 8, dtype=torch.float64)
    i0, i1 = torch.randint(0, 8, (20,)), torch.randint(0, 8, (20,))
    # "cpu" plays "hpu" here, so CPU indices are on self's device (card indices): the plain 2-D gather keeps ATen
    want, got, calls = native_and_rewrite(composite.install_index_2d, lambda t: t[i0, i1], g)
    assert "index_2d" not in calls
    torch.testing.assert_close(got, want, atol=0, rtol=0)
    want, got, calls = native_and_rewrite(composite.install_index_2d, lambda t: t[..., i0, i1], g.expand(2, 8, 8))
    assert calls.get("index_2d") == 1  # leading dims (Vech) keep the rewrite
    torch.testing.assert_close(got, want, atol=0, rtol=0)
