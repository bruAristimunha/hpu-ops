"""Which convolutions conv.install_large_kernel splits and which it leaves to the native op, per mode (CPU stand-in).
The native class is the one Gaudi2 runs right (bridge 1.21.4, repro/sweep_conv_native.sh, table in the README);
everything else with a kernel dim above the bound is split."""
import pytest
import torch
import torch.nn.functional as F
from test_cpu import compiled, cpu_as_hpu

from hpu_ops import conv


def route(monkeypatch, mode, x_shape, w_shape, stride=1, padding=0, dilation=1, groups=1, dtype=torch.float32):
    """Run F.conv{n}d through install_large_kernel as ``mode`` would; return "split" or "native" and check values."""
    monkeypatch.setattr(conv, "_mode", lambda _input: mode)
    torch.manual_seed(0)
    x = torch.randn(*x_shape, dtype=dtype, requires_grad=True)
    w = torch.randn(*w_shape, dtype=dtype, requires_grad=True)
    f = {3: F.conv1d, 4: F.conv2d, 5: F.conv3d}[len(x_shape)]
    want = f(x, w, None, stride, padding, dilation, groups)
    gx, gw = torch.autograd.grad(want.sum(), (x, w))
    with cpu_as_hpu(conv.install_large_kernel) as calls:
        got = f(x, w, None, stride, padding, dilation, groups)
        hx, hw = torch.autograd.grad(got.sum(), (x, w))
        calls = dict(calls)
    tol = {"atol": 1e-3, "rtol": 1e-3} if dtype in (torch.float32, torch.bfloat16) else {}
    if dtype is not torch.bfloat16:
        for a, b in ((got, want), (hx, gx), (hw, gw)):
            torch.testing.assert_close(a, b, **tol)
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
    (1090, 16, "split", "native"),  # eager stride floor 32
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
