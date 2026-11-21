"""Which convolutions conv.install_large_kernel splits and which it leaves to the native op, per mode (CPU stand-in).
The native class is the one Gaudi2 runs right (bridge 1.21.4, repro/sweep_conv_native.sh, table in the README);
everything else with a kernel dim above the bound is split."""
import pytest
import torch
import torch.nn.functional as F
from test_cpu import compiled, cpu_as_hpu

from hpu_ops import conv


def _set_mode(monkeypatch, mode):
    """The real ``conv._mode`` with the bridge's lazy flag patched: "eager" or "lazy" for the CPU stand-in."""
    monkeypatch.setattr(conv, "_lazy", lambda: mode == "lazy")


def _reference(f, x, w, b, *args):
    """f in float64 from the same values: output and gradients of a sum-of-squares loss."""
    xd, wd = x.detach().double().requires_grad_(), w.detach().double().requires_grad_()
    bd = None if b is None else b.detach().double()
    y = f(xd, wd, bd, *args)
    return [y] + list(torch.autograd.grad(y.square().sum(), (xd, wd)))


def route(monkeypatch, mode, x_shape, w_shape, stride=1, padding=0, dilation=1, groups=1, dtype=torch.float32):
    """Run F.conv{n}d through install_large_kernel as ``mode`` would; return "split" or "native" and check the output
    against the native op (16-bit: elementwise to one rounding) and the gradients against float64."""
    _set_mode(monkeypatch, mode)
    torch.manual_seed(0)
    fan_in = w_shape[1] * torch.tensor(w_shape[2:]).prod().item()
    x = torch.randn(*x_shape).to(dtype).requires_grad_()
    w = (torch.randn(*w_shape) / fan_in ** 0.5).to(dtype).requires_grad_()
    b = torch.randn(w_shape[0]).to(dtype)
    f = {3: F.conv1d, 4: F.conv2d, 5: F.conv3d}[len(x_shape)]
    args = (stride, padding, dilation, groups)
    want = f(x, w, b, *args)
    ref = _reference(f, x, w, b, *args)
    with cpu_as_hpu(conv.install_large_kernel) as calls:
        got = f(x, w, b, *args)
        grads = torch.autograd.grad(got.to(torch.promote_types(dtype, torch.float32)).square().sum(), (x, w))
        calls = dict(calls)
    assert got.dtype == dtype and got.shape == want.shape
    if dtype in (torch.float16, torch.bfloat16):
        eps = torch.finfo(dtype).eps
        torch.testing.assert_close(got.float(), want.float(), rtol=eps, atol=1e-5)
        bound = 4 * eps
    else:
        torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)
        bound = 1e-5 if dtype is torch.float32 else 1e-12
    for g, r in zip([got] + list(grads), ref):
        assert float((g.double() - r).abs().max() / r.abs().max()) < bound, (dtype, x_shape, w_shape)
    assert not (calls.get("large_conv") and calls.get("large_conv_native")), calls
    return "split" if calls.get("large_conv") else "native"


TCSP_DEFAULT = ((2, 1, 4, 4356), (16, 1, 4, 4356), (1, 484))  # TensorCSPNet(22) temporal_block, default dims
TCSP_K11 = ((2, 1, 4, 1089), (16, 1, 4, 1089), (1, 121))  # TensorCSPNet(22, dims=(22, 11, 11, 11))
STARNET64 = ((2, 1, 351, 40), (8, 1, 351, 1), 1)  # STaRNet(64) conv_spatial_fusion (input width cut to 40)


@pytest.mark.parametrize("mode, case, want", [
    ("eager", TCSP_DEFAULT, "native"),  # runs natively in both modes: the split only slowed it
    ("lazy", TCSP_DEFAULT, "native"),
    ("eager", TCSP_K11, "split"),  # eager native fails setAssociatedDimAndSize
    ("lazy", TCSP_K11, "native"),  # lazy native runs it right
    ("eager", STARNET64, "split"),  # native fails in both modes
    ("lazy", STARNET64, "split"),
])
def test_model_shapes_route_per_mode(monkeypatch, mode, case, want):
    assert route(monkeypatch, mode, *case) == want


@pytest.mark.parametrize("mode, want", [("eager", "native"), ("lazy", "split")])
def test_kernel_256_over_1000_splits_in_lazy_mode_only(monkeypatch, mode, want):
    # lazy native k=256 over 1000 fails graph compile, and 129..255 give a wrong input gradient; eager is right
    assert route(monkeypatch, mode, (2, 8, 1000), (4, 8, 256)) == want


@pytest.mark.parametrize("mode", ["eager", "lazy"])
@pytest.mark.parametrize("x_shape, w_shape, stride, padding, dilation, why", [
    ((2, 4, 4, 4356), (16, 4, 4, 4356), (1, 484), 0, 1, "4 input channels: native fails"),
    ((2, 1, 4356, 4), (16, 1, 4356, 4), (484, 1), 0, 1, "wide dim first: native fails"),
    ((2, 1, 4, 4356), (16, 1, 4, 4356), (1, 4), 0, 1, "stride 4: eager fails below 32, lazy stops the card at 1"),
    ((2, 1, 4, 4400), (16, 1, 4, 4356), (1, 484), (0, 22), 1, "padding: not measured"),
    ((2, 1, 4, 4356), (16, 1, 4, 2178), (1, 484), 0, (1, 2), "dilation: not measured"),
    ((2, 1, 23, 4356), (16, 1, 23, 4356), (1, 484), 0, 1, "kH above 22: not measured"),
    ((2, 1, 4356), (16, 1, 4356), 484, 0, 1, "conv1d: not measured"),
])
def test_outside_the_measured_native_class_splits(monkeypatch, mode, x_shape, w_shape, stride, padding, dilation, why):
    assert route(monkeypatch, mode, x_shape, w_shape, stride, padding, dilation) == "split", why


def test_groups_and_dtype_outside_the_class_split(monkeypatch):
    assert route(monkeypatch, "eager", (2, 2, 4, 4356), (16, 1, 4, 4356), (1, 484), groups=2) == "split"
    assert route(monkeypatch, "eager", *TCSP_DEFAULT, dtype=torch.float64) == "split"
    assert route(monkeypatch, "eager", *TCSP_DEFAULT, dtype=torch.bfloat16) == "split"


@pytest.mark.parametrize("k_w, stride, eager, lazy", [
    (257, 32, "split", "native"),  # odd: eager native fails
    (258, 32, "native", "native"),
    (1091, 32, "split", "native"),
    (1090, 32, "native", "native"),
    (1090, 16, "native", "native"),  # eager: stride 8 and 16 measured for 512..2000 (kH 4)
    (4356, 16, "split", "native"),  # eager 4356 fails at 8 and 16
    (1090, 7, "split", "split"),  # lazy stride floor 8
])
def test_width_parity_and_stride_bounds_per_mode(monkeypatch, k_w, stride, eager, lazy):
    case = ((2, 1, 4, k_w), (16, 1, 4, k_w), (1, stride))
    assert route(monkeypatch, "eager", *case) == eager
    assert route(monkeypatch, "lazy", *case) == lazy


def test_lazy_split_band_129_to_256_is_kept(monkeypatch):
    """In lazy mode 129..256 stay split even with one input channel (the class starts at 257)."""
    assert route(monkeypatch, "lazy", (2, 1, 4, 400), (16, 1, 4, 200), (1, 32)) == "split"
    assert route(monkeypatch, "eager", (2, 1, 4, 400), (16, 1, 4, 200), (1, 32)) == "native"


def test_compiled_graphs_keep_splitting_the_default_tensorcspnet_shape():
    """Traced graphs are not in the measured class: compile keeps the split as before."""
    x_shape, w_shape, stride = TCSP_DEFAULT
    layer = torch.nn.Conv2d(1, 16, w_shape[2:], stride=stride, bias=False)
    x = torch.randn(*x_shape)
    nodes = []
    with cpu_as_hpu(conv.install_large_kernel) as calls:
        got = compiled(layer, nodes)(x)
        calls = dict(calls)
    torch.testing.assert_close(got, layer(x), atol=1e-3, rtol=1e-3)
    assert calls.get("large_conv") and not calls.get("large_conv_native")


def test_real_mode_reads_traced_graphs_as_compile():
    from torch._subclasses.fake_tensor import FakeTensorMode
    with FakeTensorMode():
        assert conv._mode(torch.empty(2, 1, 4, 8)) == "compile"
    assert conv._mode(torch.empty(2, 1, 4, 8)) == "eager"  # a CPU tensor is the eager stand-in


# ---- review of #31 (CODEX_REVIEW.txt): routing by the sweep rows, lazy chunk shapes, numerics, compiled inference

def test_mode_reads_the_bridge_lazy_flag_for_the_device_type(monkeypatch):
    """Not mocked: a forced-eager bridge must route lazy cases as eager (mutation of the review: _lazy() False)."""
    case = ((2, 1, 4, 1089), (16, 1, 4, 1089), (1, 121))
    assert route(monkeypatch, "lazy", *case) == "native"
    assert route(monkeypatch, "eager", *case) == "split"
    monkeypatch.setattr(conv, "_lazy", lambda: True)
    assert conv._mode(torch.empty(1), "cpu") == "lazy" and conv._mode(torch.empty(1), "hpu") == "eager"


@pytest.mark.parametrize("mode", ["eager", "lazy"])
def test_width_ceiling_is_the_widest_measured(monkeypatch, mode):
    widest = {"eager": 6002, "lazy": 6000}[mode]
    assert route(monkeypatch, mode, (2, 1, 4, widest), (16, 1, 4, widest), (1, 32)) == "native"
    assert route(monkeypatch, mode, (2, 1, 4, 6004), (16, 1, 4, 6004), (1, 32)) == "split"  # past the sweep


def test_eager_stride_8_rows_of_the_sweep_run_natively(monkeypatch):
    """Sweep: eager (4, 512) .. (4, 2000) run at stride 8; (4, 4356) fails there."""
    assert route(monkeypatch, "eager", (2, 1, 4, 512), (16, 1, 4, 512), (1, 8)) == "native"
    assert route(monkeypatch, "eager", (2, 1, 4, 1500), (16, 1, 4, 1500), (1, 8)) == "native"
    assert route(monkeypatch, "eager", (2, 1, 4, 4356), (16, 1, 4, 4356), (1, 8)) == "split"
    assert route(monkeypatch, "eager", (2, 1, 22, 1500), (16, 1, 22, 1500), (1, 8)) == "split"  # kH 22: only at 32


# conv2d (x H, W) x (k H, W), native, forward + backward, lazy, bridge 1.21.4: stops the card (device critical error)
LAZY_DIED = {((54, 54), (44, 44)), ((54, 138), (44, 128)), ((64, 64), (54, 54))}


def _chunk_shapes(monkeypatch, mode, x, w, **kw):
    seen = []
    make = conv._chunk_conv

    def recording(op, fallback):
        inner = make(op, fallback)

        def run(xc, wc, *args):
            seen.append((tuple(xc.shape[2:]), tuple(wc.shape[2:])))
            return inner(xc, wc, *args)

        return run

    monkeypatch.setattr(conv, "_chunk_conv", recording)
    _set_mode(monkeypatch, mode)
    want = F.conv2d(x, w, **kw)
    with cpu_as_hpu(conv.install_large_kernel) as calls:
        got = F.conv2d(x, w, **kw)
    assert calls["large_conv"] == 1
    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-4)
    return seen


def test_lazy_split_never_emits_a_chunk_that_stopped_the_card(monkeypatch):
    """Review: 300 x 300 over 310 x 310 in lazy mode emitted (44, 44) over (54, 54). The tail is padded to 128."""
    torch.manual_seed(0)
    x, w = torch.randn(1, 1, 310, 310), torch.randn(1, 1, 300, 300) / 300
    seen = _chunk_shapes(monkeypatch, "lazy", x, w)
    assert set(seen) == {((138, 138), (128, 128))}, set(seen)
    assert not set(seen) & LAZY_DIED


def test_lazy_one_dim_kernel_keeps_its_measured_tail(monkeypatch):
    """STaRNet(64)'s (351, 1): 128 + 128 + 95 in lazy mode, as measured on the card (92 ms/step)."""
    x, w = torch.randn(2, 1, 351, 40), torch.randn(8, 1, 351, 1)
    assert [k for _, k in _chunk_shapes(monkeypatch, "lazy", x, w)] == [(128, 1), (128, 1), (95, 1)]


def test_lazy_tail_padding_keeps_inf_and_gradients(monkeypatch):
    """The zero taps added to the tail meet added zeros only: an inf in the input row and column that stride 3 leaves
    unused stays out of the output, as natively (130 = 128 + 2, the tail padded to 128 past the window's end)."""
    _set_mode(monkeypatch, "lazy")
    torch.manual_seed(0)
    x = torch.randn(1, 1, 140, 140, dtype=torch.float64)
    x[..., -1, :] = float("inf")
    x[..., :, -1] = float("inf")
    w = torch.randn(1, 1, 130, 130, dtype=torch.float64)

    def run():
        a, b = x.clone().requires_grad_(), w.clone().requires_grad_()
        y = F.conv2d(a, b, stride=3)
        return [y] + list(torch.autograd.grad(y.sum(), (a, b)))

    want = run()
    with cpu_as_hpu(conv.install_large_kernel) as calls:
        got = run()
    assert calls["large_conv"] == 1 and torch.isfinite(want[0]).all()
    for g, r in zip(got, want):
        torch.testing.assert_close(g, r, rtol=1e-10, atol=1e-9)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_16_bit_split_sums_in_float32_like_native(monkeypatch, dtype):
    """Review: fp16 256 * (+1, -1) over 512 taps: native 0, the split NaN (partials overflowed in fp16)."""
    _set_mode(monkeypatch, "eager")
    x = torch.full((1, 1, 512), 256.0, dtype=dtype)
    w = torch.cat([torch.ones(256), -torch.ones(256)]).to(dtype).view(1, 1, 512)
    x1 = torch.ones(1, 1, 512, dtype=dtype)
    w1 = w.clone()
    w1[0, 0, 0] = 2  # bf16: 257 - 256 rounds to 1 only when the partials are not rounded first
    xb, wb, bb = torch.ones(1, 1, 258, dtype=dtype), torch.zeros(1, 1, 258, dtype=dtype), torch.tensor([0.5], dtype=dtype)
    wb[0, 0, 0], wb[0, 0, 257] = 256, 0.75  # 256.75 + 0.5: bias added before the one rounding (bf16: 258)
    want = [F.conv1d(x, w), F.conv1d(x1, w1), F.conv1d(xb, wb, bb)]
    with cpu_as_hpu(conv.install_large_kernel) as calls:
        got = [F.conv1d(x, w), F.conv1d(x1, w1), F.conv1d(xb, wb, bb)]
    assert calls["large_conv"] == 3
    for g, r in zip(got, want):
        assert g.dtype == dtype and torch.equal(g, r), (g, r)


def test_bias_is_added_once_after_the_chunks(monkeypatch):
    _set_mode(monkeypatch, "eager")
    x = torch.ones(1, 1, 512)
    w = torch.cat([torch.ones(256), -torch.ones(256)]).view(1, 1, 512)
    b = torch.tensor([1e-5])
    with cpu_as_hpu(conv.install_large_kernel):
        got = F.conv1d(x, w, b)
    assert torch.equal(got, F.conv1d(x, w, b))  # 1e-5; bias in the first chunk gave 0


def test_wholly_padded_chunk_keeps_zero_times_inf_and_double_backward(monkeypatch):
    _set_mode(monkeypatch, "eager")
    x = torch.ones(1, 1, 1)
    w = torch.ones(1, 1, 257)
    w[0, 0, 0] = float("inf")
    want = F.conv1d(x, w, stride=513, padding=256)
    with cpu_as_hpu(conv.install_large_kernel):
        got = F.conv1d(x, w, stride=513, padding=256)
    assert torch.isnan(want).all() and torch.isnan(got).all()

    def second(installed):
        xd = torch.ones(1, 1, 1, dtype=torch.float64, requires_grad=True)
        wd = torch.ones(1, 1, 257, dtype=torch.float64, requires_grad=True)
        y = F.conv1d(xd, wd, stride=513, padding=257)
        (dx,) = torch.autograd.grad(y.sum(), xd, create_graph=True)
        return torch.autograd.grad(dx.sum(), wd)[0]

    want = second(False)
    with cpu_as_hpu(conv.install_large_kernel):
        got = second(True)
    torch.testing.assert_close(got, want, rtol=0, atol=0)


def test_channels_last_weight_gives_channels_last_output(monkeypatch):
    _set_mode(monkeypatch, "eager")
    x = torch.randn(2, 2, 3, 1)
    w = torch.randn(3, 2, 1, 257).to(memory_format=torch.channels_last)
    want = F.conv2d(x, w, padding=(0, 257), stride=(1, 513))
    with cpu_as_hpu(conv.install_large_kernel) as calls:
        got = F.conv2d(x, w, padding=(0, 257), stride=(1, 513))
        got2 = F.conv2d(torch.randn(2, 2, 3, 300), w)
    assert calls["large_conv"] == 2
    assert want.is_contiguous(memory_format=torch.channels_last)
    assert got.is_contiguous(memory_format=torch.channels_last) and got2.is_contiguous(memory_format=torch.channels_last)
    torch.testing.assert_close(got, want)


def test_compiled_inference_mode_splits_like_ordinary_capture(monkeypatch):
    """Review: compiled inference_mode() kept one width-351 convolution; ordinary capture emitted 256 + 95."""
    torch.manual_seed(0)
    layer = torch.nn.Conv1d(1, 2, 351, bias=True)
    x = torch.randn(2, 1, 400)
    want = layer(x)
    for context in (torch.no_grad, torch.inference_mode):
        nodes = []
        with cpu_as_hpu(conv.install_large_kernel) as calls, context():
            got = compiled(layer, nodes)(x)
            calls = dict(calls)
        widths = [n.args[1].meta["val"].shape[-1] for n in nodes if n.target is torch.ops.aten.convolution.default]
        assert sorted(widths) == [95, 256], (context, widths)
        assert calls.get("large_conv")
        torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)


def test_16_bit_bias_where_the_device_adds_it(monkeypatch):
    """Gaudi's native bf16 convolution rounds the sum, then adds the bias in bf16 (256 + 0.5 -> 256; CPU: 258)."""
    _set_mode(monkeypatch, "eager")
    x, b = torch.ones(1, 1, 258, dtype=torch.bfloat16), torch.tensor([0.5], dtype=torch.bfloat16)
    w = torch.zeros(1, 1, 258, dtype=torch.bfloat16)
    w[0, 0, 0], w[0, 0, 257] = 256, 0.75
    monkeypatch.setitem(conv.BIAS_AFTER_ROUNDING, "cpu", True)
    with cpu_as_hpu(conv.install_large_kernel) as calls:
        got = F.conv1d(x, w, b)
    assert calls["large_conv"] == 1 and got.item() == 256.0
    assert F.conv1d(x[..., :2], w[..., [0, 257]], b).item() == 258.0  # the CPU's own convention
