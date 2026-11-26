"""The rewrites checked on the CPU standing in for HPU: what compiled graphs contain, that eager and compiled results
match the native ops, and that other devices and symbolic shapes keep torch's own path."""
import contextlib
import copy
import importlib.util
import os
import pathlib
import re
import shlex

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from functorch.compile import make_boxed_func
from torch._dynamo.backends.common import aot_autograd

import hpu_ops
from hpu_ops import autocast, composite, conv, im2col, indexing, serialization, upsample


@contextlib.contextmanager
def cpu_as_hpu(install):
    """The rewrite ``install`` with "cpu" in the role of "hpu", undone afterwards; yields the rewrite call counts."""
    undo = install("cpu")
    hpu_ops._STATE["calls"].clear()
    try:
        yield hpu_ops._STATE["calls"]
    finally:
        undo()


def _scalar_mask_put_reaches_index_put():
    """True when ``x[mask] = <Python number>`` reaches ``aten.index_put_``, the op install_mask_fill rewrites (torch 2.6
    to 2.14, the Gaudi bridges' 2.6 among them). From torch 2.15.0 Python indexing sends it to
    ``aten.masked_fill_.Scalar`` instead (any single bool mask, leading, middle or trailing), which hpu-ops leaves
    native: Habana's launch failure was measured on the ``index_put_`` form only. A tensor value still reaches
    ``index_put_``."""
    from torch.utils._python_dispatch import TorchDispatchMode

    class Ops(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            seen.add(func)
            return func(*args, **(kwargs or {}))

    seen = set()
    x = torch.zeros(2, 2)
    with Ops():
        x[..., torch.tensor([True, False])] = 1.0
    return torch.ops.aten.index_put_.default in seen


# mask_fill count of one Python-number mask put: 1 where it reaches index_put_, 0 from torch 2.15 (masked_fill_)
SCALAR_MASK_PUT = int(_scalar_mask_put_reaches_index_put())


@pytest.fixture
def cpu_indices_off_device(monkeypatch):
    """Index tensors count as off the device, as CPU index tensors of HPU tensors are (the case that falls back)."""
    monkeypatch.setattr(indexing, "_off_device", lambda self, t: True)


def compiled(f, nodes, **options):
    """``torch.compile(f, **options)`` through aot_autograd, collecting the forward and backward graph nodes into
    ``nodes``."""

    def record(gm, example_inputs):
        nodes.extend(n for n in gm.graph.nodes if n.op == "call_function")
        return make_boxed_func(gm.forward)

    torch._dynamo.reset()
    return torch.compile(f, backend=aot_autograd(fw_compiler=record, bw_compiler=record), **options)


def ops(nodes):
    return {str(n.target) for n in nodes}


@pytest.fixture
def torch26_tracers(monkeypatch):
    """The tracers of torch <= 2.6 (Gaudi's is 2.6.0): every CompositeImplicitAutograd op that reaches them is
    decomposed, even one the device has a kernel for. Newer torch decomposes only what autograd would have."""
    import torch._subclasses.functional_tensor as functional_tensor
    import torch.fx.experimental.proxy_tensor as proxy_tensor

    for module in (functional_tensor, proxy_tensor):
        if hasattr(module, "autograd_would_have_decomposed"):
            monkeypatch.setattr(module, "autograd_would_have_decomposed", lambda func, args: True)


def train_step(f, *inputs):
    """Output and input gradients of a sum-of-squares loss."""
    inputs = [x.detach().requires_grad_() for x in inputs]
    out = f(*inputs)
    out.float().square().sum().backward()
    return [out.detach()] + [x.grad for x in inputs]


def assert_same(got, want, tol=1e-12):
    for a, b in zip(got, want):
        torch.testing.assert_close(a, b, atol=tol, rtol=tol)


# ---- linear upsampling: the native op under compile


@pytest.mark.parametrize("align_corners", [False, True])
@pytest.mark.parametrize("scale", [2.0, 256.0, 2.7])
def test_linear_compiled_graph_keeps_the_native_op(scale, align_corners):
    f = lambda x: F.interpolate(x, scale_factor=scale, mode="linear", align_corners=align_corners)  # noqa: E731
    x = torch.randn(8, 5, 3, dtype=torch.float64)
    nodes = []
    with cpu_as_hpu(upsample.install_linear) as calls:
        got = train_step(compiled(f, nodes), x)
    assert_same(got, train_step(f, x))
    assert {"aten.upsample_linear1d.default", "aten.upsample_linear1d_backward.default"} <= ops(nodes)
    assert not any("index_put" in op for op in ops(nodes)) and calls["linear"], ops(nodes)


def test_linear_new_batch_size_recompiles_through_the_decomposition():
    """A second shape marks the batch dimension dynamic: torch's decomposition with a warning, not a crash."""
    f = lambda x: F.interpolate(x, scale_factor=2.0, mode="linear")  # noqa: E731
    upsample._warn_symbolic.cache_clear()
    with cpu_as_hpu(upsample.install_linear), pytest.warns(UserWarning, match="dynamic=False") as record:
        g = compiled(f, [])
        for batch in (8, 5, 3):
            x = torch.randn(batch, 5, 3, dtype=torch.float64)
            assert_same(train_step(g, x), train_step(f, x))
    assert sum(str(w.message) == upsample.SYMBOLIC_HINT for w in record) == 1


def test_linear_static_compile_keeps_the_native_op_for_each_batch_size(recwarn):
    f = lambda x: F.interpolate(x, scale_factor=2.0, mode="linear")  # noqa: E731
    nodes = []
    upsample._warn_symbolic.cache_clear()
    with cpu_as_hpu(upsample.install_linear) as calls:
        g = compiled(f, nodes, dynamic=False)
        for batch in (8, 5, 3):
            x = torch.randn(batch, 5, 3, dtype=torch.float64)
            assert_same(train_step(g, x), train_step(f, x))
    assert calls["linear"] >= 3 and not any("index_put" in op for op in ops(nodes)), ops(nodes)
    assert not [w for w in recwarn if str(w.message) == upsample.SYMBOLIC_HINT]


# ---- linear upsampling: the forward from explicit indices where Gaudi's kernel is inexact


def linear(out, align_corners, scale):
    if scale is not None:
        return lambda x: F.interpolate(x, scale_factor=scale, mode="linear", align_corners=align_corners)
    return lambda x: F.interpolate(x, size=out, mode="linear", align_corners=align_corners)


def forward_backward(f, x):
    """Output, and input gradient for a fixed output gradient."""
    x = x.detach().requires_grad_()
    out = f(x)
    out.backward(torch.randn(out.shape, dtype=out.dtype, generator=torch.Generator().manual_seed(1)))
    return out.detach(), x.grad


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("length, out, align_corners, scale", [
    (200, 300, False, None), (200, 300, True, None), (1000, 1009, False, None), (9920, 10000, False, None),
    (1000, 750, False, None), (1000, 750, True, None), (200, None, False, 2.7), (2, 5, True, None),
])
def test_linear_index_is_as_accurate_as_atens(length, out, align_corners, scale, dtype):
    """Source positions in the kernel's precision: output and input gradient against float64, as close as ATen's own
    float32 kernel (a few positions round to the other side of an integer where a compiler fuses the multiply-add);
    float64 to rounding."""
    torch.manual_seed(0)
    x, f = torch.randn(3, 4, length, dtype=dtype), linear(out, align_corners, scale)
    want, reference = forward_backward(f, x), forward_backward(f, x.double())
    with cpu_as_hpu(upsample.install_linear_forward) as calls:
        got = forward_backward(f, x)
    assert calls["linear_index"] > 0
    if dtype == torch.float64:
        assert_same(got, want, tol=1e-10)
    else:
        for ours, atens, exact in zip(got, want, reference):
            error = lambda a: (a.double() - exact).abs().max().item()  # noqa: E731
            assert error(ours) <= 1.5 * error(atens) + 1e-7, (error(ours), error(atens))


@pytest.mark.parametrize("length, out", [(200, 400), (200, 100), (100, 800), (100, 100)])
def test_linear_power_of_two_factors_keep_the_native_op(length, out):
    with cpu_as_hpu(upsample.install_linear_forward) as calls:
        F.interpolate(torch.randn(3, 4, length), size=out, mode="linear")
    assert not calls["linear_index"]


def test_linear_index_forward_compiled():
    """Under compile with the native-op path: index ops in the graph instead of the native op."""
    x, f, nodes = torch.randn(3, 4, 200, dtype=torch.float64), linear(300, False, None), []
    want = train_step(f, x)
    with cpu_as_hpu(upsample.install_linear), cpu_as_hpu(upsample.install_linear_forward) as calls:
        got = train_step(compiled(f, nodes), x)
    assert_same(got, want)
    assert "aten.upsample_linear1d.default" not in ops(nodes) and calls["linear_index"], ops(nodes)


@pytest.mark.parametrize("align_corners", [False, True])
def test_linear_index_compiles_in_one_graph(align_corners):
    """The scale is computed in Python, not read from a tensor: size= and align_corners compile with fullgraph."""
    x, f = torch.randn(3, 4, 200, dtype=torch.float64), linear(300, align_corners, None)
    want = train_step(f, x)
    with cpu_as_hpu(upsample.install_linear), cpu_as_hpu(upsample.install_linear_forward) as calls:
        got = train_step(compiled(f, [], fullgraph=True), x)
    assert_same(got, want)
    assert calls["linear_index"]


@pytest.mark.parametrize("scale", [1.3, 0.5])
def test_linear_same_length_is_atens_copy(scale):
    """An output as long as the input is ATen's copy whatever the scale, with weights 1 and 0: an inf gives NaN."""
    x, op = torch.tensor([[[0.0, 1.0, 4.0, float("inf")]]]), torch.ops.aten.upsample_linear1d.default
    want = op(x, [4], False, scale)
    with cpu_as_hpu(upsample.install_linear_forward) as calls:
        got = op(x, [4], False, scale)
    torch.testing.assert_close(got, want, equal_nan=True)
    assert calls["linear_index"]


@pytest.mark.parametrize("length, out, align_corners, scale", [(200, 400, True, None), (5, 10, False, 2.1),
                                                               (100, 150, False, -1.0)])
def test_linear_index_takes_align_corners_other_scales_and_ignores_nonpositive_ones(length, out, align_corners,
                                                                                      scale):
    """A power-of-two size is exact on Gaudi only without align_corners and at its own scale; a scale <= 0 means
    the sizes' ratio, as in ATen."""
    x, op = torch.randn(3, 4, length, dtype=torch.float64), torch.ops.aten.upsample_linear1d.default
    want = op(x, [out], align_corners, scale)
    with cpu_as_hpu(upsample.install_linear_forward) as calls:
        got = op(x, [out], align_corners, scale)
    assert_same([got], [want], tol=1e-10)
    assert calls["linear_index"]


@pytest.mark.parametrize("call", [
    lambda x: torch.ops.aten.upsample_linear1d.default(x, [0], False),
    lambda x: F.interpolate(x[:, :0], size=150, mode="linear"),
    lambda x: F.interpolate(x.long(), size=150, mode="linear"),
    lambda x: torch.ops.aten.upsample_linear1d.default(x, [150, 2], False),
])
def test_linear_invalid_inputs_reach_the_native_op(call):
    x = torch.randn(2, 4, 100)
    with pytest.raises(Exception) as want:
        call(x)
    with cpu_as_hpu(upsample.install_linear_forward) as calls, pytest.raises(type(want.value)):
        call(x)
    assert not calls["linear_index"]


def test_linear_index_long_input_stays_in_range():
    """Past 2**24 samples float32(length - 1) rounds up to length: the index is clamped in integers."""
    x = torch.randn(1, 1, 2**24 + 4)
    want = F.interpolate(x, size=3, mode="linear", align_corners=True)
    with cpu_as_hpu(upsample.install_linear_forward) as calls:
        got = F.interpolate(x, size=3, mode="linear", align_corners=True)
    torch.testing.assert_close(got, want)
    assert calls["linear_index"]


def test_linear_backend_kernel_below_autograd():
    """The kernel registered at the device's backend key (inference mode on HPU), called directly on the CPU."""
    op = torch.ops.aten.upsample_linear1d.default
    backend = upsample._linear_device_kernels(op)[1]
    x = torch.randn(4, 3, 100, dtype=torch.float64)
    with torch.inference_mode():
        for size in ([200], [150]):  # the native kernel, then the index ops
            torch.testing.assert_close(backend(x, size, False), op(x, size, False), atol=1e-10, rtol=1e-10)


def test_linear_index_forward_in_bfloat16():
    x = torch.randn(3, 4, 200)
    want = F.interpolate(x.double(), size=300, mode="linear")
    with cpu_as_hpu(upsample.install_linear_forward):
        got = F.interpolate(x.bfloat16(), size=300, mode="linear")
    assert got.dtype == torch.bfloat16
    torch.testing.assert_close(got.double(), want, rtol=2e-2, atol=2e-2)


# ---- nearest upsampling: repeats for integer factors, the native op otherwise


def nearest(mode, **how):
    return lambda x: F.interpolate(x, mode=mode, **how)


INTEGER = [("nearest", {"scale_factor": 2.0}), ("nearest", {"size": 21}), ("nearest", {"scale_factor": 24.0}),
           ("nearest-exact", {"scale_factor": 2.0}), ("nearest-exact", {"scale_factor": 41.0})]


@pytest.mark.parametrize("mode, how", INTEGER)
def test_nearest_integer_factor_repeats_eager_and_compiled(mode, how):
    f = nearest(mode, **how)
    x = torch.randn(4, 3, 7, dtype=torch.float64)
    want = train_step(f, x)
    nodes = []
    with cpu_as_hpu(upsample.install_nearest) as calls:
        eager = train_step(f, x)
        assert calls["nearest"] == 1
        got = train_step(compiled(f, nodes), x)
    assert_same(eager, want, tol=0)
    assert_same(got, want, tol=0)
    assert not any("upsample" in op or "index" in op for op in ops(nodes)), ops(nodes)


@pytest.mark.parametrize("mode, how", [("nearest", {"scale_factor": 2.5}), ("nearest", {"scale_factor": 41.0}),
                                       ("nearest", {"size": 9}), ("nearest-exact", {"scale_factor": 1.5})])
def test_nearest_other_factors_keep_the_native_op_under_compile(torch26_tracers, mode, how):
    """Non-integer factors, and integer ones whose float32 index arithmetic is not exact (41)."""
    f = nearest(mode, **how)
    x = torch.randn(4, 3, 7, dtype=torch.float64)
    nodes = []
    with cpu_as_hpu(upsample.install_nearest) as calls:
        got = train_step(compiled(f, nodes), x)
    assert_same(got, train_step(f, x), tol=0)
    name = "upsample_nearest1d" if mode == "nearest" else "_upsample_nearest_exact1d"
    assert {f"aten.{name}.default", f"aten.{name}_backward.default"} <= ops(nodes), ops(nodes)
    assert not any("index" in op for op in ops(nodes)) and not calls["nearest"], ops(nodes)


@pytest.mark.parametrize("scale", [2.0, 2.5])
def test_nearest_new_batch_size_recompiles(scale):
    """Dynamic batch: integer factors still repeat; other factors take torch's decomposition (no SymInt meta)."""
    f = nearest("nearest", scale_factor=scale)
    with cpu_as_hpu(upsample.install_nearest):
        g = compiled(f, [])
        for batch in (8, 5, 3):
            x = torch.randn(batch, 3, 6, dtype=torch.float64)
            assert_same(train_step(g, x), train_step(f, x), tol=0)


def test_nearest_permuted_and_bf16_inputs():
    f = nearest("nearest", scale_factor=2.0)
    x = torch.randn(4, 7, 3).transpose(1, 2)
    want = [f(x), f(x.bfloat16())]
    with cpu_as_hpu(upsample.install_nearest):
        got = [f(x), compiled(f, [])(x), f(x.bfloat16())]
    assert_same(got[:2], [want[0]] * 2, tol=0)
    assert got[2].dtype == torch.bfloat16 and torch.equal(got[2], want[1])


def test_nearest_backend_kernel_below_autograd():
    """The kernel registered at the device's backend key (inference mode on HPU), called directly on the CPU."""
    op = torch.ops.aten.upsample_nearest1d.default
    backend = upsample._nearest_device_kernels(op, False, None)[1]
    x = torch.randn(4, 3, 7)
    with torch.inference_mode():
        for size, scale in (([14], 2.0), ([17], None)):  # a repeat, then the native kernel
            torch.testing.assert_close(backend(x, size, scale), op(x, size, scale), atol=0, rtol=0)


def test_repeat_factor_matches_the_native_index_arithmetic():
    x = torch.arange(997.0).view(1, 1, -1)
    for factor in range(1, 130):
        repeated = x.repeat_interleave(factor, -1)
        for mode, exact in (("nearest", False), ("nearest-exact", True)):
            rule = upsample.repeat_factor(x, [997 * factor], float(factor), exact)
            assert rule == factor if exact or factor <= 24 else rule in (None, factor)
            native = F.interpolate(x, scale_factor=float(factor), mode=mode)
            assert rule is None or torch.equal(native, repeated), (mode, factor)
    assert upsample.repeat_factor(x, [1994], None, exact=False) == 2
    assert upsample.repeat_factor(torch.zeros(1, 1, 5), [10], 2.1, exact=False) is None  # samples at 1 / 2.1


# ---- padding="valid": full zero lists under compile


@pytest.mark.parametrize("n, kwargs", [(1, {}), (2, {}), (3, {}), (2, {"groups": 3, "stride": (1, 2)}),
                                       (2, {"dilation": (1, 2)})])
def test_valid_padding_traces_full_zero_lists(n, kwargs):
    layer = getattr(nn, f"Conv{n}d")(3, 6, (2,) * n, padding="valid", **kwargs).double()
    x = torch.randn((2, 3) + (7,) * n, dtype=torch.float64)
    nodes = []
    with cpu_as_hpu(conv.install) as calls:
        got = train_step(compiled(layer, nodes), x)
    assert_same(got, train_step(layer, x))
    convs = [node for node in nodes if str(node.target) == "aten.convolution.default"]
    assert convs and calls["valid_conv"]
    assert all(list(node.args[4]) == [0] * n and list(node.args[7]) == [0] * n for node in convs), convs


def test_valid_padding_elsewhere_keeps_the_one_element_list():
    """The root cause, on a device the rewrite does not apply to: ATen passes ``[0]`` to a 2-D convolution."""
    layer = nn.Conv2d(3, 6, 2, padding="valid")
    nodes = []
    with cpu_as_hpu(lambda device: conv.install("hpu")):
        compiled(layer, nodes)(torch.randn(2, 3, 7, 7))
    convs = [node for node in nodes if str(node.target) == "aten.convolution.default"]
    assert convs and all(list(node.args[4]) == [0] for node in convs)


def test_same_padding_is_left_alone():
    layer = nn.Conv1d(3, 6, 3, padding="same")
    with cpu_as_hpu(conv.install) as calls:
        compiled(layer, [])(torch.randn(2, 3, 9))
    assert not calls["valid_conv"]


# ---- large-kernel convolutions (> MAX_KERNEL): spatial kernel splitting


@pytest.mark.parametrize("mode", ["eager", "compiled"])
@pytest.mark.parametrize("make, shape", [
    (lambda: nn.Conv2d(1, 8, (351, 1), bias=False), (2, 1, 351, 40)),  # STaRNet(64) conv_spatial_fusion
    (lambda: nn.Conv2d(1, 16, (4, 1089), stride=(1, 121)), (2, 1, 4, 1089)),  # TensorCSPNet(22, k=11) temporal_block
])
def test_large_kernel_real_shapes_match_native_conv(mode, make, shape):
    torch.manual_seed(0)
    layer, x = make().double(), torch.randn(*shape, dtype=torch.float64)
    want, nodes = module_grads(layer, x), []
    with cpu_as_hpu(conv.install_large_kernel) as calls:
        got = module_grads(layer if mode == "eager" else compiled(layer, nodes), x)
    assert_same(got, want)
    assert calls["large_conv"] > 0
    if mode == "compiled":
        convs = [n for n in nodes if n.target in (torch.ops.aten.convolution.default,
                                                 torch.ops.aten.convolution_backward.default)]
        assert convs
        for node in convs:
            w_idx = 1 if node.target is torch.ops.aten.convolution.default else 2
            assert all(k <= conv.MAX_KERNEL for k in node.args[w_idx].meta["val"].shape[2:])


@pytest.mark.parametrize("mode", ["eager", "compiled"])
@pytest.mark.parametrize("make, shape", [
    (lambda: nn.Conv2d(2, 3, (9, 1)), (2, 2, 12, 5)),  # kH > M, L_out > 1
    (lambda: nn.Conv2d(2, 3, (9, 1)), (2, 2, 9, 5)),  # L_out == 1
    (lambda: nn.Conv2d(2, 3, (9, 7), padding=(2, 1)), (2, 2, 10, 8)),  # both kH > M and kW > M
    (lambda: nn.Conv2d(2, 3, (3, 9), stride=(1, 3), padding=(1, 2)), (2, 2, 6, 15)),  # stride > 1
    (lambda: nn.Conv2d(2, 3, (3, 6), dilation=(1, 2), padding=(0, 1)), (2, 2, 5, 14)),  # dilation > 1
    (lambda: nn.Conv2d(4, 6, (3, 7), groups=2), (2, 4, 5, 11)),  # groups > 1
    (lambda: nn.Conv2d(4, 4, (3, 7), groups=4, bias=False), (2, 4, 5, 11)),  # depthwise, no bias
    (lambda: nn.Conv1d(2, 3, 9, stride=2, padding=3), (2, 2, 15)),  # 1D
    (lambda: nn.Conv3d(2, 3, (3, 6, 5), padding=(1, 2, 1)), (2, 2, 4, 7, 6)),  # 3D
    (lambda: nn.Conv2d(2, 3, (1, 7), padding="valid"), (2, 2, 4, 11)),  # padding="valid"
    (lambda: nn.Conv2d(2, 3, (1, 7), padding="same"), (2, 2, 4, 11)),  # padding="same"
    (lambda: nn.Conv2d(2, 3, (1, 7), padding=(0, 3), padding_mode="reflect"), (2, 2, 4, 11)),  # reflect pad
    (lambda: nn.Conv2d(2, 3, (1, 7)), (0, 2, 4, 11)),  # batch == 0
    (lambda: nn.Conv2d(2, 3, (1, 9), stride=(1, 20), padding=(0, 10)), (2, 2, 4, 6)),  # zero-overlap (acc is None)
])
def test_large_kernel_splitting_variants_match_native(monkeypatch, mode, make, shape):
    monkeypatch.setattr(conv, "MAX_KERNEL", 4)
    torch.manual_seed(0)
    layer, x = make().double(), torch.randn(*shape, dtype=torch.float64)
    want, nodes = module_grads(layer, x), []
    with cpu_as_hpu(conv.install), cpu_as_hpu(conv.install_large_kernel) as calls:
        got = module_grads(layer if mode == "eager" else compiled(layer, nodes), x)
    assert_same(got, want)
    assert calls["large_conv"] > 0
    if mode == "compiled":
        convs = [n for n in nodes if n.target in (torch.ops.aten.convolution.default,
                                                 torch.ops.aten.convolution_backward.default)]
        for node in convs:
            w_idx = 1 if node.target is torch.ops.aten.convolution.default else 2
            assert all(k <= 4 for k in node.args[w_idx].meta["val"].shape[2:])


def test_large_kernel_one_element_lists_and_coexistence(monkeypatch):
    monkeypatch.setattr(conv, "MAX_KERNEL", 4)
    torch.manual_seed(0)
    x = torch.randn(2, 1, 6, 15, dtype=torch.float64)
    w = torch.randn(3, 1, 2, 7, dtype=torch.float64)
    b = torch.randn(3, dtype=torch.float64)

    def fn(xr, wr, br):
        return F.conv2d(xr, wr, br, stride=(2,), padding=(1,), dilation=(1,))

    want = train_step(fn, x, w, b)
    nodes = []
    with (cpu_as_hpu(conv.install) as c_valid, cpu_as_hpu(conv.install_weight_grad) as c_wg,
          cpu_as_hpu(conv.install_large_kernel) as c_lk):
        eager = train_step(fn, x, w, b)
        assert c_lk["large_conv"] == 1 and c_wg["conv_weight_grad"] > 0
        c_lk.clear()
        c_wg.clear()
        got = train_step(compiled(fn, nodes), x, w, b)
        assert c_lk["large_conv"] > 0 and c_wg["conv_weight_grad"] > 0

        # Also test valid padding + large kernel + one-channel strided weight grad together
        layer = nn.Conv2d(1, 4, (1, 7), stride=(1, 2), padding="valid").double()
        want_v = module_grads(layer, x)
        nodes_v = []
        c_valid.clear()
        c_wg.clear()
        c_lk.clear()
        got_v = module_grads(compiled(layer, nodes_v), x)
        assert c_valid["valid_conv"] > 0 and c_lk["large_conv"] > 0 and c_wg["conv_weight_grad"] > 0
    assert_same(eager, want)
    assert_same(got, want)
    assert_same(got_v, want_v)


def test_large_kernel_channels_last_non_contiguous_bf16_and_inference_mode(monkeypatch):
    monkeypatch.setattr(conv, "MAX_KERNEL", 4)
    torch.manual_seed(0)
    x_cl = torch.randn(2, 3, 6, 12, dtype=torch.float64).to(memory_format=torch.channels_last)
    w_cl = torch.randn(4, 3, 2, 7, dtype=torch.float64).to(memory_format=torch.channels_last)
    b = torch.randn(4, dtype=torch.float64)
    want_cl = train_step(lambda a, k: F.conv2d(a, k, b), x_cl, w_cl)

    x_nc = torch.randn(2, 12, 6, 3, dtype=torch.float64).permute(0, 3, 2, 1)
    w_nc = torch.randn(4, 7, 2, 3, dtype=torch.float64).permute(0, 3, 2, 1)
    want_nc = train_step(lambda a, k: F.conv2d(a, k), x_nc, w_nc)

    want_bf16 = train_step(lambda a, k, bias: F.conv2d(a, k, bias),
                           x_cl.float().bfloat16(), w_cl.float().bfloat16(), b.float().bfloat16())
    with cpu_as_hpu(conv.install_large_kernel) as calls:
        got_cl = train_step(lambda a, k: F.conv2d(a, k, b), x_cl, w_cl)
        assert got_cl[0].is_contiguous(memory_format=torch.channels_last)
        got_nc = train_step(lambda a, k: F.conv2d(a, k), x_nc, w_nc)

        # bfloat16 & autocast with float32 accumulation
        x_f32, w_f32, b_f32 = x_cl.float(), w_cl.float(), b.float()
        want_f64 = train_step(lambda a, k, bias: F.conv2d(a, k, bias), x_cl, w_cl, b)
        got_bf16 = train_step(lambda a, k, bias: F.conv2d(a, k, bias),
                              x_f32.bfloat16(), w_f32.bfloat16(), b_f32.bfloat16())
        with torch.autocast("cpu", dtype=torch.bfloat16):
            got_ac = train_step(lambda a, k, bias: F.conv2d(a, k, bias), x_f32, w_f32, b_f32)

        # inference_mode (hits "{}" backend key)
        calls.clear()
        with torch.inference_mode():
            got_inf = F.conv2d(x_cl, w_cl, b)
            assert got_inf.is_contiguous(memory_format=torch.channels_last)
        assert calls["large_conv"] == 1
    assert_same(got_cl, want_cl)
    assert_same(got_nc, want_nc)
    assert_same([got_inf], [want_cl[0]])
    assert got_bf16[0].dtype == torch.bfloat16
    # the output against the native bf16 op elementwise (both sum in float32 and round once), gradients against
    # float64 relative to their largest entry: a split that zeroed one bf16 output entry fails both
    eps = torch.finfo(torch.bfloat16).eps
    for got16 in (got_bf16, got_ac):
        torch.testing.assert_close(got16[0].float(), want_bf16[0].float(), rtol=eps, atol=1e-5)
        for a, r in zip(got16[1:], want_f64[1:]):
            assert float((a.double() - r).abs().max() / r.abs().max()) < 8 * eps


def test_large_kernel_dynamic_batch_vs_dynamic_spatial(monkeypatch):
    monkeypatch.setattr(conv, "MAX_KERNEL", 4)
    torch.manual_seed(0)
    layer = nn.Conv2d(2, 3, (1, 7)).double()
    inputs = [torch.randn(b, 2, 4, 11, dtype=torch.float64) for b in (2, 5, 3)]
    wants = [module_grads(layer, x) for x in inputs]
    nodes = []
    with cpu_as_hpu(conv.install_large_kernel) as calls:
        c = compiled(layer, nodes, fullgraph=True)
        for x, want in zip(inputs, wants):
            assert_same(module_grads(c, x), want)
        assert calls["large_conv"] > 0
        convs = [n for n in nodes if n.target is torch.ops.aten.convolution.default]
        assert convs and all(n.args[1].meta["val"].shape[-1] <= 4 for n in convs)

        # Dynamic spatial size falls back cleanly via static(...) during tracing
        dyn_nodes = []
        x_sp = torch.randn(2, 2, 4, 13, dtype=torch.float64)
        c_sp = compiled(layer, dyn_nodes, dynamic=True, fullgraph=True)
        want_sp = F.conv2d(x_sp.detach(), layer.weight.detach(), layer.bias.detach())
        got_sp = c_sp(x_sp)
        dyn_convs = [n for n in dyn_nodes if n.target is torch.ops.aten.convolution.default]
        assert dyn_convs and any(n.args[1].meta["val"].shape[-1] == 7 for n in dyn_convs)
    assert_same([got_sp], [want_sp])


def test_large_kernel_skips_preregistered_backend_key_and_still_fixes_autograd(monkeypatch):
    monkeypatch.setattr(conv, "MAX_KERNEL", 4)
    op = torch.ops.aten.convolution.default
    backend_calls = 0

    def dummy_cpu_conv(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups):
        nonlocal backend_calls
        backend_calls += 1
        return op._op_dk(torch._C.DispatchKey.CompositeExplicitAutograd, input, weight, bias, stride, padding,
                         dilation, transposed, output_padding, groups)

    lib = torch.library.Library("aten", "IMPL")
    lib.impl(op, dummy_cpu_conv, "CPU")
    op._dispatch_cache.clear()
    try:
        assert torch._C._dispatch_has_kernel_for_dispatch_key("aten::convolution", "CPU")
        x = torch.randn(2, 2, 4, 10, dtype=torch.float64)
        w = torch.randn(3, 2, 1, 7, dtype=torch.float64)
        want = train_step(lambda a, k: F.conv2d(a, k), x, w)
        nodes = []
        with cpu_as_hpu(conv.install_large_kernel) as calls:
            eager = train_step(lambda a, k: F.conv2d(a, k), x, w)
            with torch.no_grad():
                ng = F.conv2d(x, w)
            comp = train_step(compiled(lambda a, k: F.conv2d(a, k), nodes), x, w)
        assert calls["large_conv"] >= 3 and backend_calls >= 6
        assert_same(eager, want)
        assert_same(comp, want)
        assert_same([ng], [want[0]])
    finally:
        lib._destroy()
        op._dispatch_cache.clear()


def test_large_kernel_invalid_and_passthrough_fall_through_to_aten(monkeypatch):
    monkeypatch.setattr(conv, "MAX_KERNEL", 4)
    op = torch.ops.aten.convolution.default
    invalid_calls = [
        lambda: F.conv2d(torch.randn(2, 2, 4, 4), torch.randn(3, 2, 1, 6)),  # k > L_in + 2p
        lambda: F.conv2d(torch.randn(2, 2, 0, 10), torch.randn(3, 2, 1, 6), padding=(1, 0)),  # L_in == 0
        lambda: F.conv2d(torch.randn(2, 2, 4, 10), torch.randn(0, 2, 1, 6)),  # C_out == 0
        lambda: F.conv2d(torch.randn(2, 2, 4, 10), torch.randn(3, 2, 0, 6)),  # k == 0
        lambda: op(torch.randn(2, 2, 4, 10), torch.randn(3, 2, 1, 6), None, [1, 1], [0, 0], [1, 1], False, [0, 0], 0),
        lambda: F.conv2d(torch.randn(2, 2, 4, 10), torch.randn(3, 2, 1, 6), torch.randn(4)),  # bad bias
        lambda: F.conv2d(torch.randn(2, 2, 4, 10), torch.randn(3, 2, 1, 6), stride=(1, 0)),  # stride == 0
        lambda: op(torch.randn(2, 2, 4, 10), torch.randn(3, 2, 1, 6), None, [1, 1], [0, -1], [1, 1], False, [0, 0], 1),
        lambda: F.conv2d(torch.randn(2, 2, 4, 10), torch.randn(3, 2, 1, 6), dilation=(1, 0)),  # dilation == 0
        lambda: F.conv2d(torch.randn(2, 2, 4, 10), torch.randn(3, 2, 1, 6).double()),  # dtype mismatch
        lambda: op(torch.randn(2, 2, 4, 10), torch.randn(3, 2, 1, 6), None, [1, 1], [0, 0], [1, 1], False, [0, 0, 0], 1),
        lambda: op(torch.randn(2, 2, 4, 10), torch.randn(3, 2, 1, 6), None, [1, 1], [0, 0], [1, 1], False, [-1, 0], 1),
        lambda: op(torch.randn(2, 2, 4, 10, requires_grad=True), torch.randn(3, 2, 1, 6, requires_grad=True),
                   None, [1, 1], [0, 0], [1, 1], False, [1, 0], 1).sum().backward(),
        lambda: op(torch.randn(2, 2, 4, 10, requires_grad=True), torch.randn(3, 2, 1, 6, requires_grad=True),
                   None, [1, 1], [0, 0], [1, 1], False, [1], 1).sum().backward(),
    ]
    for call in invalid_calls:
        with pytest.raises(Exception) as want:
            call()
        with cpu_as_hpu(conv.install_large_kernel) as calls, pytest.raises(type(want.value)):
            call()
        assert not calls["large_conv"]

    # Non-split / transposed / non-float / C_in=0 / non-zero output_padding pass through cleanly
    with cpu_as_hpu(conv.install_large_kernel) as calls:
        F.conv2d(torch.randn(2, 2, 4, 10), torch.randn(3, 2, 1, 4))
        F.conv_transpose2d(torch.randn(2, 2, 4, 10), torch.randn(2, 3, 1, 6))
        F.conv2d(torch.randint(0, 5, (2, 2, 4, 10)), torch.randint(0, 5, (3, 2, 1, 6)))
        y_c0 = F.conv2d(torch.randn(2, 0, 4, 10), torch.randn(3, 0, 1, 6))
        assert y_c0.shape == (2, 0, 4, 5)
        y_op = op(torch.randn(2, 2, 4, 10), torch.randn(3, 2, 1, 6), None, [1, 1], [0, 0], [1, 1], False, [1, 0], 1)
        assert y_op.shape == (2, 3, 4, 5)
    assert not calls["large_conv"]


def test_large_kernel_zero_overlap_with_inf_input_matches_native(monkeypatch):
    monkeypatch.setattr(conv, "MAX_KERNEL", 4)
    x = torch.tensor([[[[float("inf"), 1.0, 2.0, 3.0]]]], dtype=torch.float64)
    w = torch.ones(1, 1, 1, 9, dtype=torch.float64)
    b = torch.tensor([1.5], dtype=torch.float64)

    def f(a, k, bias=None):
        return F.conv2d(a, k, bias, stride=(1, 20), padding=(0, 10))

    want = train_step(f, x, w)
    want_b = train_step(f, x, w, b)
    x_cl = x.expand(2, 3, 2, 4).clone().to(memory_format=torch.channels_last)
    w_cl = torch.ones(2, 3, 1, 9, dtype=torch.float64).to(memory_format=torch.channels_last)
    want_cl = train_step(f, x_cl, w_cl)
    want_bf16 = train_step(f, x_cl.bfloat16(), w_cl.bfloat16(), torch.ones(2, dtype=torch.bfloat16))
    x_b0 = torch.empty(0, 3, 2, 4, dtype=torch.float64)
    want_b0 = train_step(f, x_b0, w_cl)

    nodes = []
    with cpu_as_hpu(conv.install_large_kernel) as calls:
        got = train_step(f, x, w)
        got_c = train_step(compiled(f, nodes), x, w)
        got_b = train_step(f, x, w, b)
        got_cl = train_step(f, x_cl, w_cl)
        got_bf16 = train_step(f, x_cl.bfloat16(), w_cl.bfloat16(), torch.ones(2, dtype=torch.bfloat16))
        got_b0 = train_step(f, x_b0, w_cl)
    assert calls["large_conv"] >= 6
    assert got[0].item() == 0.0 and not torch.isnan(got[0]).any()
    assert torch.equal(got[1], torch.zeros_like(x)) and torch.equal(got[2], torch.zeros_like(w))
    assert got_cl[0].is_contiguous(memory_format=torch.channels_last)
    assert_same(got, want)
    assert_same(got_c, want)
    assert_same(got_b, want_b)
    assert_same(got_cl, want_cl)
    assert_same(got_bf16, want_bf16)
    assert_same(got_b0, want_b0)


# ---- glu as a * sigmoid(b)


@pytest.mark.parametrize("dim", [1, -1])
def test_glu_is_a_composite_eager_and_compiled(dim):
    f = lambda x: F.glu(x, dim)  # noqa: E731
    x = torch.randn(4, 6, 8, dtype=torch.float64)
    want = train_step(f, x)
    nodes = []
    with cpu_as_hpu(composite.install_glu) as calls:
        eager = train_step(f, x)
        assert calls["glu"] == 1
        got = train_step(compiled(f, nodes), x)
    assert_same(eager, want)
    assert_same(got, want)
    assert "aten.sigmoid.default" in ops(nodes) and not any("glu" in op for op in ops(nodes)), ops(nodes)


def test_glu_dtypes_layouts_and_errors():
    x = torch.randn(4, 6, 8)
    permuted = torch.randn(8, 6, 4).permute(2, 1, 0)
    want = [F.glu(x, 1), F.glu(permuted, 1)]
    with cpu_as_hpu(composite.install_glu):
        half = F.glu(x.bfloat16(), 1)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            autocast = F.glu(x, 1)
        strided = F.glu(permuted, 1)
        with pytest.raises(RuntimeError, match="Halving dimension must be even"):
            F.glu(torch.randn(4, 5), 1)
    assert half.dtype == torch.bfloat16 and (half.float() - want[0]).abs().max() < 2e-2
    assert autocast.dtype == torch.float32
    torch.testing.assert_close(strided, want[1], atol=1e-6, rtol=1e-6)


def test_glu_inference_mode_uses_the_backend_composite(monkeypatch):
    """Inference mode skips autograd: BrainModule's GLU along dim 1 must not fall back to Habana's kernel."""
    x = torch.randn(4, 6, 8)
    want = F.glu(x, 1)
    monkeypatch.setenv("PT_HPU_LAZY_MODE", "1")
    with cpu_as_hpu(composite.install_glu) as calls:
        with torch.inference_mode():
            got = F.glu(x, 1)
    assert calls["glu"] == 1
    torch.testing.assert_close(got, want, atol=0, rtol=0)


def test_glu_elsewhere_stays_glu():
    nodes = []
    with cpu_as_hpu(lambda device: composite.install_glu("hpu")) as calls:
        train_step(compiled(lambda x: F.glu(x, 1), nodes), torch.randn(4, 6, 8))
    assert "aten.glu.default" in ops(nodes) and not calls["glu"]


def test_sdpa_math_matches_aten_eager_inference_and_gradients():
    torch.manual_seed(0)
    q = torch.randn(2, 4, 6, 8, requires_grad=True)
    k = torch.randn(2, 4, 6, 8, requires_grad=True)
    v = torch.randn(2, 4, 6, 8, requires_grad=True)
    mask = torch.rand(2, 1, 6, 6) > 0.3
    want = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    want.sum().backward()
    grads = [t.grad.clone() for t in (q, k, v)]
    for t in (q, k, v):
        t.grad = None
    with cpu_as_hpu(composite.install_sdpa) as calls:
        got = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        got.sum().backward()
        assert calls["sdpa"] >= 1
        with torch.inference_mode():
            inferred = F.scaled_dot_product_attention(q.detach(), k.detach(), v.detach(), is_causal=True)
    torch.testing.assert_close(got, want, atol=1e-6, rtol=1e-6)
    for t, grad in zip((q, k, v), grads):
        torch.testing.assert_close(t.grad, grad, atol=1e-5, rtol=1e-5)
    with torch.no_grad():
        causal = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    torch.testing.assert_close(inferred, causal, atol=1e-6, rtol=1e-6)


def test_sdpa_math_gqa_and_declines():
    torch.manual_seed(0)
    q = torch.randn(1, 4, 5, 8)
    k = torch.randn(1, 2, 5, 8)
    v = torch.randn(1, 2, 5, 8)
    want = F.scaled_dot_product_attention(q, k, v, enable_gqa=True)
    with cpu_as_hpu(composite.install_sdpa) as calls:
        got = F.scaled_dot_product_attention(q, k, v, enable_gqa=True)
        assert calls["sdpa"] == 1
    torch.testing.assert_close(got, want, atol=1e-6, rtol=1e-6)
    # head mismatches decline to the native kernel (its errors); dropout is taken
    # (Habana's lazy SDPA segfaults with it) and matches ATen's distribution
    assert not composite._sdpa_takes(q, k, v)
    assert composite._sdpa_takes(q, k, v, enable_gqa=True)
    assert composite._sdpa_takes(q, q, q, dropout_p=0.5)
    with cpu_as_hpu(composite.install_sdpa) as calls:
        dropped = F.scaled_dot_product_attention(q, q, q, dropout_p=0.5)
    assert calls["sdpa"] == 1 and dropped.shape == q.shape
    assert torch.isfinite(dropped).all()


def test_addmm_reduced_matches_aten_and_declines():
    torch.manual_seed(0)
    inp = torch.randn(16, 16, dtype=torch.bfloat16)
    a = torch.randn(16, 16, dtype=torch.bfloat16, requires_grad=True)
    b = torch.randn(16, 16, dtype=torch.bfloat16, requires_grad=True)
    want = torch.addmm(inp, a, b, beta=-4.7750, alpha=2.0315)
    want.sum().backward()
    ga, gb = a.grad.clone(), b.grad.clone()
    a.grad = b.grad = None
    with cpu_as_hpu(composite.install_addmm) as calls:
        got = torch.addmm(inp, a, b, beta=-4.7750, alpha=2.0315)
        got.sum().backward()
        assert calls["addmm"] >= 1
        with torch.inference_mode():
            inferred = torch.addmm(inp, a.detach(), b.detach(), beta=3.4445)
        zero_beta = torch.addmm(inp, a, b, beta=0)  # declined: native is exact there
        assert calls["addmm"] >= 1
    torch.testing.assert_close(got.float(), want.float(), atol=0, rtol=0)  # same fp32 math, one rounding
    torch.testing.assert_close(a.grad, ga, atol=6e-2, rtol=2e-2)  # bf16 ulps: ATen rounds its grads to bf16
    torch.testing.assert_close(b.grad, gb, atol=6e-2, rtol=2e-2)
    assert inferred.dtype == torch.bfloat16
    assert not composite._addmm_takes(inp, a, b, beta=0)
    assert not composite._addmm_takes(inp.float(), a.float(), b.float(), beta=1)


def test_unique_dim_and_flat_match_aten():
    torch.manual_seed(0)
    x = torch.randint(0, 3, (40, 5)).float()
    x[::7] = x[1]  # duplicate rows
    want = torch.unique(x, dim=0, return_inverse=True, return_counts=True)
    flat_want = torch.unique(x, return_inverse=True, return_counts=True)
    with cpu_as_hpu(composite.install_unique) as calls:
        got = torch.unique(x, dim=0, return_inverse=True, return_counts=True)
        flat = torch.unique(x, return_inverse=True, return_counts=True)
        assert calls["unique"] == 2
        with torch.inference_mode():
            inferred = torch.unique(x, dim=1)
    for a, b in zip(got, want):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    for a, b in zip(flat, flat_want):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    torch.testing.assert_close(inferred, torch.unique(x, dim=1), atol=0, rtol=0)
    assert not composite._unique_takes(x, 0, False)  # sorted=False keeps ATen


def test_unique_default_without_inverse_on_nonempty_tensor():
    x = torch.tensor([3, 1, 3, 2], dtype=torch.int64)
    want = torch.unique(x)
    with cpu_as_hpu(composite.install_unique) as calls:
        got = torch.unique(x)
        got_2d = torch.unique(x.reshape(2, 2))
        assert calls["unique"] == 2
    torch.testing.assert_close(got, want, atol=0, rtol=0)
    torch.testing.assert_close(got_2d, want, atol=0, rtol=0)
@pytest.mark.parametrize("alpha_val", [0.1, 0.3, 0.7, 1.0, 2.5, 10.0])
def test_standard_gamma_distribution_matches_moments(alpha_val):
    gen = torch.Generator(device="cpu").manual_seed(1234 + int(alpha_val * 100))
    alpha = torch.full((4000,), alpha_val, dtype=torch.float32)
    with cpu_as_hpu(composite.install_standard_gamma) as calls:
        samples = torch._standard_gamma(alpha, generator=gen)
        assert calls["standard_gamma"] == 1
    assert samples.shape == alpha.shape
    assert torch.isfinite(samples).all()
    assert (samples > 0).all()
    sample_mean = float(samples.mean().item())
    sample_var = float(samples.var(unbiased=True).item())
    # Standard Gamma(alpha, 1) has mean = alpha and var = alpha; check within 4.5 sigma
    std_err_mean = (alpha_val / samples.numel()) ** 0.5
    assert abs(sample_mean - alpha_val) < 4.5 * std_err_mean + 1e-4
    assert abs(sample_var - alpha_val) / max(alpha_val, 0.1) < 0.25


def test_standard_gamma_edge_values_and_bfloat16():
    x = torch.tensor([-1.0, 0.0, float("inf"), float("nan")], dtype=torch.float32)
    want = torch._standard_gamma(x)
    with cpu_as_hpu(composite.install_standard_gamma) as calls:
        got = torch._standard_gamma(x)
        bf = torch._standard_gamma(torch.full((16,), 0.5, dtype=torch.bfloat16))
        assert calls["standard_gamma"] == 2
    torch.testing.assert_close(got, want, atol=0, rtol=0)
    assert bf.dtype == torch.bfloat16
    assert torch.isfinite(bf).all() and (bf > 0).all()


def test_standard_gamma_declines_int_and_grad_tracking():
    with cpu_as_hpu(composite.install_standard_gamma) as calls:
        with pytest.raises(RuntimeError):
            torch._standard_gamma(torch.tensor([1, 2], dtype=torch.int64))
        x = torch.full((64,), 2.0, requires_grad=True)
        y = torch._standard_gamma(x)
        y.sum().backward()
        assert calls["standard_gamma"] == 0
    assert x.grad is not None and torch.isfinite(x.grad).all()



# ---- renorm as norm, scale factor and multiply


def renorm_cases():
    w = torch.randn(5, 4, 3, dtype=torch.float64)
    w[1] *= 0.01  # one slice under the limit
    return w


@pytest.mark.parametrize("p, dim, maxnorm", [(2, 0, 1.0), (1, 1, 0.5), (3.5, -1, 2.0)])
def test_renorm_is_a_composite_eager_and_compiled(p, dim, maxnorm):
    f = lambda w: torch.renorm(w, p, dim, maxnorm)  # noqa: E731
    w = renorm_cases()
    want = train_step(f, w)
    nodes = []
    with cpu_as_hpu(composite.install_renorm) as calls:
        eager = train_step(f, w)
        assert calls["renorm"] == 1
        got = train_step(compiled(f, nodes), w)
    assert_same(eager, want)
    assert_same(got, want)
    assert not any("renorm" in op for op in ops(nodes)), ops(nodes)


class Prototypes(nn.Module):
    """SSTDPN's pattern: prototypes renormalized in place under no_grad, then used."""

    def __init__(self):
        super().__init__()
        self.proto = nn.Parameter(torch.randn(3, 8, dtype=torch.float64))

    def forward(self, x):
        with torch.no_grad():
            self.proto.copy_(torch.renorm(self.proto, p=2, dim=0, maxnorm=0.5))
        return x @ self.proto.T


def test_renorm_in_place_under_no_grad_compiles_without_renorm():
    x = torch.randn(4, 8, dtype=torch.float64)
    ref, model = Prototypes(), Prototypes()
    model.load_state_dict(ref.state_dict())
    want = train_step(ref, x) + [ref.proto.grad]
    nodes = []
    with cpu_as_hpu(composite.install_renorm):
        got = train_step(compiled(model, nodes), x) + [model.proto.grad]
    assert_same(got, want)
    assert not any("renorm" in op for op in ops(nodes)), ops(nodes)


def test_renorm_dtypes_layouts_and_errors():
    w = renorm_cases().float()
    want = torch.renorm(w, 2, 0, 1.0)
    want_strided = torch.renorm(w.transpose(0, 2), 2, 1, 1.0)
    with cpu_as_hpu(composite.install_renorm):
        half = torch.renorm(w.bfloat16(), 2, 0, 1.0)
        same = torch.renorm(w, 2, 0, 1.0)
        strided = torch.renorm(w.transpose(0, 2), 2, 1, 1.0)
        torch.testing.assert_close(strided, want_strided, atol=1e-6, rtol=1e-6)
        for bad in ((w, 0, 0, 1.0), (w, 2, 0, -1.0), (w[0, 0], 2, 0, 1.0)):
            with pytest.raises(RuntimeError, match="renorm"):
                torch.renorm(*bad)
        with pytest.raises(IndexError, match="out of range"):
            torch.renorm(w, 2, 3, 1.0)
    torch.testing.assert_close(same, want, atol=0, rtol=0)
    assert half.dtype == torch.bfloat16 and (half.float() - want).abs().max() < 2e-2


def test_renorm_backend_kernel_below_autograd():
    """The kernel registered at the device's backend key (inference mode on HPU), called directly on the CPU."""
    op = torch.ops.aten.renorm.default
    backend = composite._kernel(op, "renorm", composite.renorm, composite._renorm_takes,
                                torch._C.DispatchKey.CompositeExplicitAutogradNonFunctional)
    w = renorm_cases()
    with torch.inference_mode():
        torch.testing.assert_close(backend(w, 2, 0, 1.0), op(w, 2, 0, 1.0), atol=0, rtol=0)
        with pytest.raises(RuntimeError, match="non-positive"):
            backend(w, 0, 0, 1.0)


def test_renorm_elsewhere_stays_renorm():
    nodes = []
    with cpu_as_hpu(lambda device: composite.install_renorm("hpu")) as calls:
        train_step(compiled(lambda w: torch.renorm(w, 2, 0, 1.0), nodes), renorm_cases())
    assert "aten.renorm.default" in ops(nodes) and not calls["renorm"]


# ---- one_hot with an inferred class count through ATen's composite


def test_one_hot_inferred_classes_eager_and_inference():
    x = torch.tensor([0, 2, 1, 2])
    want = F.one_hot(x)
    with cpu_as_hpu(composite.install_one_hot) as calls:
        got = F.one_hot(x)
        assert calls["one_hot"] == 1  # the autograd kernel (the backend is not reached)
        explicit = F.one_hot(x, 5)
        assert calls["one_hot"] == 1  # explicit counts go to ATen's composite
        with torch.inference_mode():
            inferred = F.one_hot(x)
        assert calls["one_hot"] == 2  # below autograd, the backend kernel
    torch.testing.assert_close(got, want, atol=0, rtol=0)
    torch.testing.assert_close(explicit, F.one_hot(x, 5), atol=0, rtol=0)
    torch.testing.assert_close(inferred, want, atol=0, rtol=0)


def test_one_hot_inferred_classes_errors_match_native():
    with cpu_as_hpu(composite.install_one_hot):
        with pytest.raises(RuntimeError, match="non-negative"):
            F.one_hot(torch.tensor([-1, 0]))
        with pytest.raises(RuntimeError, match="Can not infer"):
            F.one_hot(torch.empty(0, dtype=torch.long))


# ---- the weight gradient of one-channel strided convolutions (EEGDINO, CodeBrain, LaBraM)


def module_grads(module, x):
    """Output, input gradient and parameter gradients of a sum-of-squares loss."""
    module.zero_grad(set_to_none=True)
    x = x.detach().requires_grad_()
    out = module(x)
    out.float().square().sum().backward()
    return [out.detach(), x.grad] + [p.grad for p in module.parameters()]


ONE_CHANNEL_STRIDED = [
    (lambda: nn.Conv2d(1, 25, (1, 49), (1, 25), (0, 24)), (3, 1, 8, 200)),  # EEGDINO / CodeBrain proj_in
    (lambda: nn.Conv2d(1, 8, (1, 15), (1, 8), (0, 7)), (3, 1, 8, 200)),  # LaBraM temporal conv1
    (lambda: nn.Conv1d(1, 4, 5, 3, 2), (3, 1, 50)),
    (lambda: nn.Conv3d(1, 2, (2, 3, 3), (1, 2, 2), (0, 1, 1), bias=False), (2, 1, 4, 9, 9)),
]


@pytest.mark.parametrize("mode", ["eager", "compiled"])
@pytest.mark.parametrize("make, shape", ONE_CHANNEL_STRIDED)
def test_one_channel_strided_conv_weight_grad_is_a_contraction(mode, make, shape):
    torch.manual_seed(0)
    layer, x = make().double(), torch.randn(*shape, dtype=torch.float64)
    want, nodes = module_grads(layer, x), []
    with cpu_as_hpu(conv.install_weight_grad) as calls:
        got = module_grads(layer if mode == "eager" else compiled(layer, nodes), x)
    assert_same(got, want)
    assert calls["conv_weight_grad"] > 0
    backward = [n for n in nodes if n.target is torch.ops.aten.convolution_backward.default]
    assert all(not n.args[-1][1] for n in backward)  # the native backward computes no weight gradient


@pytest.mark.parametrize("make, shape", [
    (lambda: nn.Conv2d(2, 4, (1, 5), (1, 2)), (2, 2, 4, 40)),  # two input channels
    (lambda: nn.Conv2d(1, 4, (1, 5)), (2, 1, 4, 40)),  # stride 1
    (lambda: nn.Conv2d(1, 4, (1, 5), (1, 2), dilation=2), (2, 1, 4, 40)),
    (lambda: nn.ConvTranspose2d(1, 4, (1, 5), (1, 2)), (2, 1, 4, 40)),
    (lambda: nn.Conv2d(4, 4, (1, 5), (1, 2), groups=4), (2, 4, 4, 40)),  # depthwise: weight.shape[1] == 1 too
    (lambda: nn.ConvTranspose2d(4, 1, (1, 5), (1, 2)), (2, 4, 4, 40)),  # transposed: weight.shape[1] == 1 too
])
def test_other_convolutions_keep_the_native_backward(make, shape):
    torch.manual_seed(0)
    layer, x = make().double(), torch.randn(*shape, dtype=torch.float64)
    want = module_grads(layer, x)
    with cpu_as_hpu(conv.install_weight_grad) as calls:
        got = module_grads(layer, x)
    assert_same(got, want)
    assert calls["conv_weight_grad"] == 0


@pytest.mark.parametrize("stride, padding", [((2,), (1,)), ([4, 2], [1]), ([2, 2], [1]), ([2], [0, 1])])
def test_one_channel_strided_conv_weight_grad_of_one_element_lists(stride, padding):
    """F.conv2d passes one-element stride and padding lists through; ATen broadcasts them to every spatial dim."""
    torch.manual_seed(0)
    x, w = torch.randn(2, 1, 8, 20, dtype=torch.float64), torch.randn(4, 1, 3, 3, dtype=torch.float64)

    def grads():
        xr, wr = x.clone().requires_grad_(), w.clone().requires_grad_()
        F.conv2d(xr, wr, None, stride, padding).square().sum().backward()
        return [xr.grad, wr.grad]

    want = grads()
    with cpu_as_hpu(conv.install_weight_grad) as calls:
        got = grads()
    assert_same(got, want)
    assert calls["conv_weight_grad"] > 0


@pytest.mark.parametrize("mode", ["eager", "compiled"])
def test_one_channel_strided_first_layer_without_bias(mode):
    """No input gradient and no bias: the weight gradient alone, without calling the native op."""
    torch.manual_seed(0)
    layer, x = nn.Conv2d(1, 8, (1, 15), (1, 8), (0, 7), bias=False).double(), torch.randn(3, 1, 8, 200).double()

    def grads(module):
        layer.zero_grad(set_to_none=True)
        module(x).square().sum().backward()
        return [layer.weight.grad]

    want, nodes = grads(layer), []
    with cpu_as_hpu(conv.install_weight_grad) as calls:
        got = grads(layer if mode == "eager" else compiled(layer, nodes))
    assert_same(got, want)
    assert calls["conv_weight_grad"] > 0


def test_one_channel_strided_conv_weight_grad_in_bfloat16():
    """16-bit inputs are contracted in float32 and the gradient comes back in the weight's dtype."""
    torch.manual_seed(0)
    layer, x = nn.Conv2d(1, 8, (1, 15), (1, 8), (0, 7)), torch.randn(3, 1, 8, 200)
    want = module_grads(copy.deepcopy(layer).double(), x.double())[2]  # .double() converts in place, grads too
    with cpu_as_hpu(conv.install_weight_grad):
        got = module_grads(layer.bfloat16(), x.bfloat16())[2]
    assert got.dtype == torch.bfloat16
    torch.testing.assert_close(got.double(), want, rtol=2e-2, atol=2e-2 * want.abs().max().item())


# ---- the embedding gradient of a 0-d index (BIOT's channel tokens)


@pytest.mark.parametrize("mode", ["eager", "compiled"])
def test_embedding_gradient_of_a_0d_index_goes_through_the_one_element_index(mode):
    """BIOT's pattern, one 0-d index per channel from an arange buffer: torch's output and weight gradient; compiled,
    the backward graph's embedding_dense_backward takes one-element indices."""
    torch.manual_seed(0)
    weight, index = torch.randn(6, 4), torch.arange(6)

    def channel_tokens(weight):
        return torch.stack([F.embedding(index[i], weight) * (i + 1) for i in range(6)])

    want, nodes = train_step(channel_tokens, weight), []
    with cpu_as_hpu(composite.install_embedding) as calls:
        got = train_step(channel_tokens if mode == "eager" else compiled(channel_tokens, nodes), weight)
    assert_same(got, want)
    assert calls["embedding_0d"] > 0
    if mode == "compiled":
        backward = [n for n in nodes if n.target is torch.ops.aten.embedding_dense_backward.default]
        assert backward and all(n.args[1].meta["val"].dim() == 1 for n in backward)


@pytest.mark.parametrize("kwargs", [{"padding_idx": 2}, {"padding_idx": 3}, {"scale_grad_by_freq": True}])
def test_embedding_gradient_of_a_0d_index_keeps_the_arguments(kwargs):
    weight = torch.randn(5, 4)
    f = lambda w: F.embedding(torch.tensor(2), w, **kwargs)  # noqa: E731
    with cpu_as_hpu(composite.install_embedding) as calls:
        got = train_step(f, weight)
    assert_same(got, train_step(f, weight))
    assert calls["embedding_0d"] == 1


def test_embedding_forward_stays_native():
    """Only the backward changes: a 0-d embedding under no_grad is no view and can be changed in place."""
    weight = torch.randn(5, 4, requires_grad=True)
    with cpu_as_hpu(composite.install_embedding) as calls:
        with torch.no_grad():
            token = F.embedding(torch.tensor(2), weight)
        assert token._base is None
        token.add_(torch.ones(4, requires_grad=True))
        got = train_step(lambda w: F.embedding(torch.tensor([[1, 2], [3, 4]]), w), weight)
    assert_same(got, train_step(lambda w: F.embedding(torch.tensor([[1, 2], [3, 4]]), w), weight))
    assert calls["embedding_0d"] == 0


# ---- roll by a multiple of the size as a copy (EMG2QwertyNet's stacked gradient)


def strided_roll(shifts, dims):
    """A roll of a strided slice, as the gradient of a stack's slice is."""
    return lambda t: t[:, :, 1].roll(shifts, dims) if dims is not None else t[:, :, 1].roll(shifts)


@pytest.mark.parametrize("shifts, dims", [(0, 2), (16, 2), (-16, 2), (32, 2), ((0, 0), (1, 2)), ((16, 2), (2, 1)),
                                          (0, None)])
def test_roll_by_a_multiple_of_the_size_is_a_copy(shifts, dims):
    torch.manual_seed(0)
    x, f = torch.randn(7, 2, 3, 16, 5, dtype=torch.float64), strided_roll(shifts, dims)
    want = train_step(f, x)
    with cpu_as_hpu(composite.install_roll) as calls:
        got = train_step(f, x)
    assert_same(got, want)
    assert calls["roll_zero"] > 0


@pytest.mark.parametrize("shifts, dims", [(1, 2), (-1, 2), (17, 2), ((0, 1), (1, 2))])
def test_other_rolls_keep_the_native_op(shifts, dims):
    torch.manual_seed(0)
    x, f = torch.randn(7, 2, 3, 16, 5, dtype=torch.float64), strided_roll(shifts, dims)
    want = train_step(f, x)
    with cpu_as_hpu(composite.install_roll) as calls:
        got = train_step(f, x)
    assert_same(got, want)
    assert not calls["roll_zero"]


def test_roll_copy_is_contiguous_as_roll_is():
    for x in (torch.randn(3, 4).t(), torch.randn(2, 3, 4, 5).to(memory_format=torch.channels_last)):
        want = x.roll(0, 0)
        with cpu_as_hpu(composite.install_roll) as calls:
            got = x.roll(0, 0)
        assert got.stride() == want.stride() and torch.equal(got, want) and calls["roll_zero"]


@pytest.mark.parametrize("shifts, dims", [((0, 1), ()), ((0,), (1, 2)), ((16, 1), (2,)), ((), ()), ((0,), (3,))])
def test_roll_arguments_aten_rejects_reach_it(shifts, dims):
    x = torch.randn(4, 3, 16)
    with pytest.raises(Exception) as want:
        x.roll(shifts, dims)
    with cpu_as_hpu(composite.install_roll) as calls, pytest.raises(type(want.value)):
        x.roll(shifts, dims)
    assert not calls["roll_zero"]


def test_roll_without_dims_by_other_than_a_multiple_of_the_numel():
    x, f = torch.randn(7, 2, 3, 16, 5, dtype=torch.float64), strided_roll(5, None)
    want = train_step(f, x)
    with cpu_as_hpu(composite.install_roll) as calls:
        got = train_step(f, x)
    assert_same(got, want)
    assert not calls["roll_zero"]


def test_roll_of_an_empty_dim_and_of_a_scalar():
    with cpu_as_hpu(composite.install_roll) as calls:
        empty, scalar = torch.empty(2, 0, 3).roll(3, 1), torch.tensor(2.0).roll(1, 0)
    assert empty.shape == (2, 0, 3) and torch.equal(scalar, torch.tensor(2.0)) and calls["roll_zero"] == 2


def test_roll_by_zero_with_dynamic_shapes():
    """A literal 0 is a copy at symbolic sizes too."""
    x, nodes = torch.randn(4, 3, 16, dtype=torch.float64), []
    f = lambda t: torch.stack([t.roll(offset, 2) for offset in (-1, 0, 1)], 2)  # noqa: E731
    want = train_step(f, x)
    with cpu_as_hpu(composite.install_roll) as calls:
        got = train_step(compiled(f, nodes, dynamic=True, fullgraph=True), x)
    assert_same(got, want)
    assert "aten.clone.default" in ops(nodes) and calls["roll_zero"]


@pytest.mark.parametrize("mode", ["eager", "compiled"])
def test_roll_of_a_stack_by_zero(mode):
    """A rotation-invariant MLP's rolls: the backward rolls each slice of the stack's gradient back."""
    torch.manual_seed(0)
    x, nodes = torch.randn(4, 3, 16, dtype=torch.float64), []
    f = lambda t: torch.stack([t.roll(offset, 2) for offset in (-1, 0, 1)], 2)  # noqa: E731
    want = train_step(f, x)
    with cpu_as_hpu(composite.install_roll) as calls:
        got = train_step(f if mode == "eager" else compiled(f, nodes), x)
    assert_same(got, want)
    assert calls["roll_zero"] > 0


# ---- boolean-mask index_put_ / index_put and trailing 2-D index.Tensor (GBWAtt)


def test_save_reduces_device_tensors_through_a_host_copy(tmp_path):
    """A checkpoint after an optimizer step: state dicts, a Parameter and the momentum, loaded with weights_only, on
    the device the tensors came from or where map_location says."""
    model = nn.Sequential(nn.Conv2d(1, 4, 3), nn.BatchNorm2d(4))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    model(torch.randn(2, 1, 8, 8)).square().mean().backward()
    optimizer.step()
    weight = model[0].weight
    with cpu_as_hpu(serialization.install) as calls:
        assert weight.detach().__reduce_ex__(2)[0] is torch._utils._rebuild_device_tensor_from_cpu_tensor
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "weight": weight},
                   tmp_path / "checkpoint.pt")
    assert calls["save"] >= len(model.state_dict())
    loaded = torch.load(tmp_path / "checkpoint.pt", weights_only=True)
    for name, value in model.state_dict().items():
        assert torch.equal(loaded["model"][name], value), name
    assert isinstance(loaded["weight"], nn.Parameter) and loaded["weight"].requires_grad
    assert torch.equal(loaded["weight"], weight)
    momentum = optimizer.state_dict()["state"][0]["momentum_buffer"]
    assert torch.equal(loaded["optimizer"]["state"][0]["momentum_buffer"], momentum)
    elsewhere = torch.load(tmp_path / "checkpoint.pt", weights_only=True, map_location="meta")
    assert {v.device.type for v in elsewhere["model"].values()} == {"meta"}


def test_save_keeps_torch_checks_for_device_tensors(tmp_path):
    """As for XLA tensors: hooks warn that they are not saved, and skip_data refuses a host copy."""
    x = torch.randn(3, requires_grad=True)
    x.register_hook(lambda grad: grad)
    with cpu_as_hpu(serialization.install), pytest.warns(UserWarning, match="hook"):
        torch.save(x, tmp_path / "x.pt")
    with cpu_as_hpu(serialization.install), torch.serialization.skip_data(), pytest.raises(RuntimeError,
                                                                                            match="skip_data"):
        torch.save(torch.randn(3), tmp_path / "skipped.pt")


# ---- CUDA's autocast lists for Habana


def test_autocast_files_hold_torchs_cuda_lists():
    """The installed torch's CUDA lists, composed as hpu_ops.autocast says (rms_norm joined CUDA's float32 list after
    2.6), in the files that the quoted exports name."""
    header = pathlib.Path(torch.__file__).parent / "include" / "ATen" / "autocast_mode.h"
    if not header.exists():
        pytest.skip("torch installed without its C++ headers")
    text = header.read_text()

    def listed(macro, entry=r"_\(([a-z0-9_]+)"):
        body = re.search(rf"#define {macro}\(_\)((?:.*\\\n)*.*)", text)
        assert body, f"{macro} not in {header}"
        return set(re.findall(entry, body.group(1)))

    files = autocast.paths()
    lower, fp32 = (set(pathlib.Path(files[name]).read_text().split()) for name in autocast.FILES)
    assert lower == listed("AT_FORALL_LOWER_PRECISION_FP")
    cuda_fp32 = (listed("AT_FORALL_FP32") | listed("AT_FORALL_FP32_SET_OPT_DTYPE")
                 | listed("AT_FORALL_DIFFERENT_REDISPATCH_SIGNATURE", r"ADD_NS\(([a-z0-9_]+)\)")
                 | (listed("AT_FORALL_PROMOTE") - autocast.HABANA_PROMOTE) | autocast.REFUSED_BY_CUDA)
    assert cuda_fp32 <= fp32 <= cuda_fp32 | {"rms_norm"}, fp32 ^ cuda_fp32
    assert autocast.exports() == [f"export {name}={shlex.quote(path)}" for name, path in files.items()]


# ---- indexing whose indexed dims do not lead: one index_select; a boolean-mask put of one value: where


def _vech(x):
    rows, cols = torch.triu_indices(x.shape[-1], x.shape[-1])
    return x[..., rows, cols]


def _vech_scaled(x):  # spd_learn's sym_to_upper: an index, then a boolean-mask put of sqrt(2)
    rows, cols = torch.triu_indices(x.shape[-1], x.shape[-1])
    v = x[..., rows, cols]
    m = torch.ones_like(v)
    m[..., rows != cols] = 2 ** 0.5
    return v * m


def _vec_to_sym(v):  # spd_learn's vec_to_sym: index puts into zeros and index reads of them
    n = int(((1 + 8 * v.shape[-1]) ** 0.5 - 1) / 2)
    ixs = torch.triu_indices(n, n)
    od = ixs[0] != ixs[1]
    x = torch.zeros((*v.shape[:-1], n, n), dtype=v.dtype)
    x[..., ixs[0], ixs[1]] = v
    x[..., ixs[0, od], ixs[1, od]] /= 2 ** 0.5
    x[..., ixs[1, od], ixs[0, od]] = x[..., ixs[0, od], ixs[1, od]]
    return x


INDEXINGS = {
    "vech": _vech,
    "middle dim": lambda x: x[:, torch.tensor([2, 0, 2, 1])],
    "last dim, 2-d index": lambda x: x[..., torch.tensor([[0, 3], [3, 3]])],
    "two adjacent, broadcast": lambda x: x[:, torch.tensor([[0], [2]]), torch.tensor([1, 3, 1])],
    "separated by a slice": lambda x: x[torch.tensor([0, 1, 1]), :, torch.tensor([3, 0, 3])],
    "separated, not first": lambda x: x[:, torch.tensor([0, 2]), :, torch.tensor([1, 1])],
    "negative indices": lambda x: x[..., torch.tensor([-1, 0, -4]), torch.tensor([2, -2, 1])],
    "int32 index": lambda x: x[..., torch.tensor([1, 1, 0], dtype=torch.int32)],
    "empty index": lambda x: x[..., torch.tensor([], dtype=torch.long)],
    "non-contiguous input": lambda x: x.transpose(0, 2)[:, torch.tensor([0, 0, 3])],
}


@pytest.mark.usefixtures("cpu_indices_off_device")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16])
@pytest.mark.parametrize("name", sorted(INDEXINGS))
def test_index_goes_through_index_select(name, dtype):
    torch.manual_seed(0)
    x = torch.randn(3, 4, 4, 4, dtype=dtype) if name == "separated, not first" else torch.randn(5, 4, 4, dtype=dtype)
    f = INDEXINGS[name]
    want = train_step(f, x)
    with cpu_as_hpu(indexing.install_index) as calls:
        got = train_step(f, x)
        out = f(x)
    assert calls["index"] == 2
    assert_same(got, want, tol=0)
    if name != "non-contiguous input":  # ATen's result follows the input's layout; index_select's is contiguous
        assert out.stride() == f(x).stride()


def test_only_the_patterns_gaudi_gets_wrong_are_rewritten(monkeypatch):
    """On the device: indices that do not lead (lazy backward and launch) and a mask after a slice (lazy transpose).
    Leading indices keep the native op, unless an index tensor is on the CPU (eager fallback)."""
    x = torch.randn(5, 4, 4)
    rewritten = [lambda x: x[:, torch.tensor([0, 3])], lambda x: x[torch.tensor([0, 1]), :, torch.tensor([3, 3])],
                 lambda x: x[:, x[0, 0] > 0]]
    native = [lambda x: x[torch.tensor([0, 4, 4])], lambda x: x[torch.tensor([0, 1]), torch.tensor([3, 3])],
              lambda x: x[x > 0], lambda x: x[x[:, 0, 0] > 0], lambda x: x[:, 1:3], lambda x: x[..., 2]]
    for f in rewritten + native:
        want = train_step(f, x)
        with cpu_as_hpu(indexing.install_index) as calls:
            got = train_step(f, x)
            f(x)  # no gradient: rewritten too
        assert calls["index"] == 2 * (f in rewritten), calls
        assert_same(got, want, tol=0)
    monkeypatch.setattr(indexing, "_off_device", lambda self, t: True)
    for f in native[:2]:
        with cpu_as_hpu(indexing.install_index) as calls:
            got = f(x)
        assert calls["index"] == 1 and torch.equal(got, f(x))


def test_index_errors_reach_the_native_op():
    x = torch.randn(5, 4, 4)
    with cpu_as_hpu(indexing.install_index) as calls:
        for f in (lambda: x[:, torch.tensor([4])], lambda: x[..., torch.tensor([0, 1]), torch.tensor([-5])],
                  lambda: x[:, torch.tensor([0, 1]), torch.tensor([0, 1, 2])]):
            with pytest.raises((IndexError, RuntimeError)):
                f()
    assert not calls["index"]


def test_index_under_no_grad_and_inference_mode():
    x = torch.randn(5, 4, 4)
    with cpu_as_hpu(indexing.install_index) as calls:
        with torch.no_grad():
            a = _vech(x)
        with torch.inference_mode():
            b = _vech(x)
    assert calls["index"] == 1  # inference mode skips the autograd key
    assert_same([a, b], [_vech(x)] * 2, tol=0)


def test_index_compiled():
    x = torch.randn(5, 6, 6)
    want = train_step(_vech_scaled, x)
    nodes = []
    with cpu_as_hpu(hpu_ops._install):
        got = train_step(compiled(_vech_scaled, nodes), x)
    assert_same(got, want, tol=0)
    # CPU index tensors are fake while tracing: the graph gets the rewrite with the range turned into -1 in the graph
    assert {"aten.index_select.default", "aten.where.self"} <= ops(nodes)
    assert not ops(nodes) & {"aten.index.Tensor", "aten.index_put.default"}


@pytest.mark.parametrize("make", [
    lambda: torch.randn(1, 1, 1),  # n = 1
    lambda: torch.randn(1, 7, 7),  # batch 1
    lambda: torch.randn(6, 5, 5)[2:],  # an offset view
    lambda: torch.randn(5, 5)[None].expand(3, 5, 5),  # a stride-0 batch
    lambda: torch.randn(2, 3, 6, 6, dtype=torch.float64).transpose(-1, -2),  # 4-d, transposed
])
def test_vech_adversarial_shapes_and_layouts(make):
    torch.manual_seed(0)
    x = make()
    want = train_step(_vech_scaled, x)
    with cpu_as_hpu(hpu_ops._install) as calls:
        got = train_step(_vech_scaled, x)
    assert calls["index"] == 1 and calls["mask_fill"] == SCALAR_MASK_PUT
    assert_same(got, want, tol=0)


def test_index_twice_in_a_row_and_extreme_values():
    x = torch.tensor([[[1e30, -1e-30], [float("inf"), 3.0]]]).repeat(4, 1, 1)
    with cpu_as_hpu(indexing.install_index):
        a, b = _vech(x), _vech(x)
    assert torch.equal(a, _vech(x)) and torch.equal(b, a)


def test_index_compiled_with_the_index_as_an_input():
    x, rows = torch.randn(5, 6, 6), torch.tensor([0, 5, 2])
    f = lambda x, rows: x[:, rows, rows.flip(0)]  # noqa: E731
    want = train_step(lambda x: f(x, rows), x)
    nodes = []
    with cpu_as_hpu(indexing.install_index):
        got = train_step(lambda x: compiled(f, nodes)(x, rows), x)
    assert_same(got, want, tol=0)
    assert "aten.index_select.default" in ops(nodes)
    assert not ops(nodes) & {"aten.index.Tensor", "aten.index_put.default"}


@pytest.mark.parametrize("idx", [
    lambda: [None, torch.tensor(1)],  # one 0-d index after a slice
    lambda: [None, torch.tensor(1), torch.tensor(-2)],  # two adjacent 0-d indices
    lambda: [torch.tensor(2), None, torch.tensor(0)],  # 0-d indices separated by a slice
])
def test_index_of_0d_indices_called_directly(idx):
    x = torch.randn(3, 4, 5, dtype=torch.float64, requires_grad=True)
    want = torch.ops.aten.index.Tensor(x, idx())
    with cpu_as_hpu(indexing.install_index) as calls:
        got = torch.ops.aten.index.Tensor(x, idx())
        torch.autograd.gradcheck(lambda x: torch.ops.aten.index.Tensor(x, idx()), (x,))
    assert calls["index"] >= 1
    assert got.shape == want.shape and torch.equal(got, want)


def test_index_double_backward():
    x = torch.randn(5, 4, 4, dtype=torch.float64, requires_grad=True)
    f = lambda x: (_vech(x) ** 3).sum()  # noqa: E731
    with cpu_as_hpu(indexing.install_index):
        assert torch.autograd.gradgradcheck(f, (x,))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_vech_and_vec_to_sym_as_in_spd_learn(dtype):
    torch.manual_seed(0)
    v = torch.randn(8, 21, dtype=dtype)
    f = lambda v: _vech_scaled(_vec_to_sym(v))  # noqa: E731
    want = train_step(f, v)
    with cpu_as_hpu(hpu_ops._install) as calls:
        got = train_step(f, v)
    assert calls["mask_fill"] == SCALAR_MASK_PUT and calls["index"] >= 3
    assert_same(got, want, tol=0)


MASK_PUTS = {
    "trailing mask": lambda m, x: m.__setitem__((Ellipsis, x[0] > 0), 2.5),
    "middle mask": lambda m, x: m.__setitem__((slice(None), x[0, :, 0] > 0), -1.0),
    "full mask": lambda m, x: m.__setitem__(x > 0, 0.5),
    "0-d tensor value": lambda m, x: m.__setitem__((Ellipsis, x[0] > 0), torch.tensor(3.0)),
}


@pytest.mark.parametrize("name", sorted(MASK_PUTS))
def test_mask_put_of_one_value_is_where(name):
    torch.manual_seed(0)
    x = torch.randn(5, 4, 4)
    want = torch.ones(5, 4, 4)
    MASK_PUTS[name](want, x)
    got = torch.ones(5, 4, 4)
    with cpu_as_hpu(indexing.install_mask_fill) as calls:
        MASK_PUTS[name](got, x)
    assert calls["mask_fill"] == (1 if "tensor" in name else SCALAR_MASK_PUT)
    assert_same([got], [want], tol=0)


def test_mask_put_gradient_and_other_puts_keep_the_native_op():
    torch.manual_seed(0)
    x = torch.randn(5, 4, 4)

    def f(x):
        y = x * 1
        y[..., x[0] > 0] = 0.0
        return y

    want = train_step(f, x)
    with cpu_as_hpu(indexing.install_mask_fill) as calls:
        got = train_step(f, x)
    assert calls["mask_fill"] == SCALAR_MASK_PUT
    assert_same(got, want, tol=0)
    with cpu_as_hpu(indexing.install_mask_fill) as calls:
        y, z = torch.zeros(5, 4, 4), torch.zeros(5, 4, 4)
        y[..., x[0] > 0] = torch.arange(float((x[0] > 0).sum()))  # several values
        y[:, torch.tensor([0, 2])] = 1.0  # integer index
        y.index_put_((torch.tensor([0, 0]),), torch.tensor(1.0), accumulate=True)
    assert not calls["mask_fill"]
    z[..., x[0] > 0] = torch.arange(float((x[0] > 0).sum()))
    z[:, torch.tensor([0, 2])] = 1.0
    z.index_put_((torch.tensor([0, 0]),), torch.tensor(1.0), accumulate=True)
    assert_same([y], [z], tol=0)

# ---- advanced indexing (aten::index.Tensor) as index_select / device-moved indexing


@pytest.mark.usefixtures("cpu_indices_off_device")
@pytest.mark.parametrize("mode", ["eager", "compiled"])
def test_index_contiguous_triu_matches_torch_and_compiles(mode):
    """SPDNet / sym_to_upper pattern: (32, 22, 22) indexed at trailing [..., i0, i1]."""
    torch.manual_seed(0)
    ix = torch.triu_indices(22, 22)
    f = lambda x: x[..., ix[0], ix[1]]  # noqa: E731
    x = torch.randn(32, 22, 22, dtype=torch.float64)
    x2 = torch.randn(5, 22, 22, dtype=torch.float64)
    want, want2, nodes = train_step(f, x), train_step(f, x2), []  # native: before the rewrite is installed
    with cpu_as_hpu(indexing.install_index) as calls:
        g = f if mode == "eager" else compiled(f, nodes, fullgraph=True, dynamic=True)
        got = train_step(g, x)
        if mode == "compiled":
            assert_same(train_step(g, x2), want2)
    assert_same(got, want)
    assert calls["index"] > 0
    if mode == "compiled":
        assert "aten.index_select.default" in ops(nodes)
        assert not any("aten.index." in op for op in ops(nodes)), ops(nodes)


@pytest.mark.usefixtures("cpu_indices_off_device")
@pytest.mark.parametrize("shape, make_call", [
    ((3, 4, 5, 6), lambda: (lambda t: t[:, torch.tensor([0, 2, -1]), torch.tensor([1, -2, 3]), :])),
    ((3, 4, 5), lambda: (lambda t: torch.ops.aten.index.Tensor(t, [None, torch.tensor(2), None]))),
    ((4, 5), lambda: (lambda t: torch.ops.aten.index.Tensor(t, [torch.tensor(1), torch.tensor(-1)]))),
    ((3, 4, 5), lambda: (lambda t: t[:, torch.tensor([0, -1]), torch.tensor(2)])),
    ((3, 4, 5), lambda: (lambda t: t[:, torch.tensor([[0], [-1]]), torch.tensor([1, 2, -3])])),
    ((3, 4, 5), lambda: (lambda t: t[:, torch.tensor([1, 1, 3, 1], dtype=torch.int32)])),
    ((3, 4, 5), lambda: (lambda t: t[:, torch.empty(0, dtype=torch.long)])),
    ((0, 4, 5), lambda: (lambda t: t[:, torch.tensor([0, 2])])),
    ((3, 4, 0), lambda: (lambda t: t[:, torch.tensor([0, 2])])),
    ((3, 0, 5), lambda: (lambda t: t[:, torch.empty(0, dtype=torch.long)])),
    ((4, 5, 6), lambda: (lambda t: t[torch.tensor([-1, 0, 2, -1]), :, torch.tensor([1, -2, 4, 1])])),
    ((2, 3, 4, 5), lambda: (lambda t: t[:, torch.tensor([[0, 1], [1, -1]]), :, torch.tensor([1, -3])])),
])
def test_index_integer_patterns_eager_and_grad(shape, make_call):
    torch.manual_seed(0)
    x = torch.randn(*shape, dtype=torch.float64)
    f = make_call()
    want = train_step(f, x)
    with cpu_as_hpu(indexing.install_index) as calls:
        got = train_step(f, x)
    assert_same(got, want)
    assert calls["index"] == 1


@pytest.mark.usefixtures("cpu_indices_off_device")
def test_index_non_contiguous_self_and_compiled_non_contiguous_indices():
    torch.manual_seed(0)
    x = torch.randn(5, 3, 6, dtype=torch.float64).permute(1, 2, 0)
    i0 = torch.tensor([-1, 0, 1, 0])
    i1 = torch.tensor([2, -1, 0, 2])
    f = lambda t: t[i0, :, i1]  # noqa: E731
    want, nodes = train_step(f, x), []
    with cpu_as_hpu(indexing.install_index) as calls:
        eager = train_step(f, x)
        got = train_step(compiled(f, nodes, fullgraph=True, dynamic=True), x)
    assert_same(eager, want)
    assert_same(got, want)
    assert calls["index"] >= 2
    assert not any("aten.index." in op for op in ops(nodes)), ops(nodes)


@pytest.mark.usefixtures("cpu_indices_off_device")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16, torch.float16, torch.int64, torch.bool])
def test_index_dtypes_and_autocast(dtype):
    torch.manual_seed(0)
    if dtype.is_floating_point:
        x = torch.randn(4, 5, 5, dtype=dtype)
    elif dtype == torch.int64:
        x = torch.arange(100, dtype=dtype).reshape(4, 5, 5)
    else:
        x = torch.arange(100).reshape(4, 5, 5) % 2 == 0
    ix = torch.triu_indices(5, 5)
    want = x[..., ix[0], ix[1]]
    with cpu_as_hpu(indexing.install_index) as calls:
        got = x[..., ix[0], ix[1]]
        if dtype == torch.float32:
            with torch.autocast("cpu", dtype=torch.bfloat16):
                auto = x[..., ix[0], ix[1]]
            assert auto.dtype == torch.float32 and torch.equal(auto, want)
    assert got.dtype == want.dtype and torch.equal(got, want) and calls["index"] >= 1


def test_index_bool_masks_and_backend_kernel():
    torch.manual_seed(0)
    x = torch.randn(6, 5, 5, dtype=torch.float64)
    mask2d = torch.ones(5, 5, dtype=torch.bool).triu()
    empty_mask = torch.zeros(5, 5, dtype=torch.bool)
    mask1d = torch.tensor([True, False, True, False, True, False])
    want_2d = train_step(lambda t: t[:, mask2d], x)
    want_empty = train_step(lambda t: t[:, empty_mask], x)
    want_1d = train_step(lambda t: t[mask1d], x)
    with cpu_as_hpu(indexing.install_index) as calls:
        got_2d = train_step(lambda t: t[:, mask2d], x)
        got_empty = train_step(lambda t: t[:, empty_mask], x)
        assert calls["index"] == 2
        got_1d = train_step(lambda t: t[mask1d], x)
        assert calls["index"] == 2  # same-device bool mask without leading None keeps native op
    assert_same(got_2d, want_2d)
    assert_same(got_empty, want_empty)
    assert_same(got_1d, want_1d)

    # Below-autograd (inference_mode) kernel and off-device classification
    op = torch.ops.aten.index.Tensor
    backend = indexing._index_kernel("cpu", torch._C.DispatchKey.CompositeExplicitAutogradNonFunctional)
    ix = torch.triu_indices(5, 5)
    with torch.inference_mode():
        torch.testing.assert_close(backend(x, [None, ix[0], ix[1]]), op(x, [None, ix[0], ix[1]]), atol=0, rtol=0)
    xm = x.to("meta")  # meta in the role of the device, the CPU masks off it
    assert indexing._plan(xm, [torch.ones(6, dtype=torch.bool)]) == "move"
    assert indexing._plan(xm, [None, torch.tensor(True)]) == "move"


@pytest.mark.usefixtures("cpu_indices_off_device")
@pytest.mark.parametrize("bad_idx", [
    lambda: (torch.tensor([0.0, 1.0]),),
    lambda: (torch.tensor([0, 1], dtype=torch.int8),),
    lambda: (torch.tensor([0, 1], dtype=torch.int16),),
    lambda: (slice(None), torch.tensor([0, 1]), torch.tensor([0, 1, 2])),
    lambda: (torch.tensor([0]), torch.tensor([0]), torch.tensor([0]), torch.tensor([0])),
    lambda: (slice(None), torch.ones(2, 2, dtype=torch.bool)),
])
def test_index_invalid_inputs_reach_aten(bad_idx):
    x = torch.randn(3, 4, 5)
    idx = bad_idx()
    with pytest.raises(Exception) as want:
        x[idx]
    with cpu_as_hpu(indexing.install_index) as calls, pytest.raises(type(want.value)):
        x[idx]
    assert not calls["index"]


@pytest.mark.usefixtures("cpu_indices_off_device")
def test_index_elsewhere_stays_native():
    x = torch.randn(4, 5, 5)
    ix = torch.triu_indices(5, 5)
    nodes = []
    with cpu_as_hpu(lambda device: indexing.install_index("hpu")) as calls:
        train_step(compiled(lambda t: t[..., ix[0], ix[1]], nodes), x)
    assert "aten.index.Tensor" in ops(nodes) and not calls["index"]


# ---- im2col / col2im (F.unfold / F.fold): 4-D non-overlapping fast path and index_select / index_add general path


SPD_PATCH_SHAPES = [
    ((32, 16, 513, 1), (128, 1), (128, 1)),  # SSGCNet (513 % 128 = 1 remainder)
    ((32, 20, 514, 1), (257, 1), (257, 1)),  # MAtt (514 % 257 = 0 exact)
    ((32, 20, 513, 1), (85, 1), (85, 1)),    # GyroAtt / SPDMetaFormer / CorAtt (513 % 85 = 3 remainder)
]


@pytest.mark.parametrize("shape, kernel_size, stride", SPD_PATCH_SHAPES)
def test_im2col_col2im_spd_model_shapes_tier1(shape, kernel_size, stride):
    """All 5 affected SPD model PatchEmbeddingLayer shapes hit Tier 1 and match F.unfold / F.fold fwd+bwd."""
    torch.manual_seed(0)
    x = torch.randn(*shape, dtype=torch.float64)
    unfold_fn = lambda t: F.unfold(t, kernel_size, stride=stride)  # noqa: E731
    want_unfold = train_step(unfold_fn, x)
    with cpu_as_hpu(im2col.install) as calls:
        got_unfold = train_step(unfold_fn, x)
        assert calls["im2col"] == 1
        out_shape = want_unfold[0].shape
    assert_same(got_unfold, want_unfold)

    y = torch.randn(*out_shape, dtype=torch.float64)
    fold_fn = lambda t: F.fold(t, shape[-2:], kernel_size, stride=stride)  # noqa: E731
    want_fold = train_step(fold_fn, y)
    with cpu_as_hpu(im2col.install) as calls:
        got_fold = train_step(fold_fn, y)
        assert calls["col2im"] == 1
    assert_same(got_fold, want_fold)


@pytest.mark.parametrize("shape, kernel_size, dilation, padding, stride", [
    ((2, 3, 1, 17), (1, 5), (1, 1), (0, 0), (1, 5)),   # Tier 1 horizontal (H==1, kh==1, non-divisible)
    ((2, 3, 1, 12), (1, 4), (1, 1), (0, 0), (1, 4)),   # Tier 1 horizontal exact
    ((4, 3, 10, 12), (3, 5), (1, 1), (0, 0), (1, 1)),  # Tier 2 overlapping Issue #14 pattern
    ((2, 4, 15, 19), (3, 4), (2, 2), (1, 2), (2, 3)),  # Tier 2 dilation + asymmetric pad + stride
    ((2, 3, 11, 13), (3, 5), (1, 1), (1, 1), (4, 4)),  # Tier 2 non-divisible stride
    ((2, 3, 5, 6), (2, 3), (1, 1), (3, 4), (1, 1)),    # Tier 2 padding >= kernel_size
    ((0, 3, 7, 9), (3, 4), (1, 1), (1, 1), (2, 2)),    # zero batch size N=0 (Tier 2)
    ((0, 3, 9, 1), (4, 1), (1, 1), (0, 0), (4, 1)),    # zero batch size N=0 (Tier 1)
])
def test_im2col_col2im_general_shapes_and_orientations(shape, kernel_size, dilation, padding, stride):
    torch.manual_seed(0)
    x = torch.randn(*shape, dtype=torch.float64)
    kw_args = dict(kernel_size=kernel_size, dilation=dilation, padding=padding, stride=stride)
    unfold_fn = lambda t: F.unfold(t, **kw_args)  # noqa: E731
    want_u = train_step(unfold_fn, x)
    with cpu_as_hpu(im2col.install) as calls:
        got_u = train_step(unfold_fn, x)
        assert calls["im2col"] == 1
    assert_same(got_u, want_u)

    y = torch.randn(*want_u[0].shape, dtype=torch.float64)
    fold_fn = lambda t: F.fold(t, shape[-2:], **kw_args)  # noqa: E731
    want_f = train_step(fold_fn, y)
    with cpu_as_hpu(im2col.install) as calls:
        got_f = train_step(fold_fn, y)
        assert calls["col2im"] == 1
    assert_same(got_f, want_f)


def test_im2col_3d_col2im_2d_and_non_contiguous():
    torch.manual_seed(0)
    x3d = torch.randn(3, 8, 6, dtype=torch.float64).transpose(1, 2)  # non-contiguous 3D
    y2d = torch.randn(12, 12, dtype=torch.float64).t()               # non-contiguous 2D (3*2*2, 3*4)
    im_op = lambda t: torch.ops.aten.im2col.default(t, [2, 2], [1, 1], [0, 0], [2, 2])  # noqa: E731
    col_op = lambda t: torch.ops.aten.col2im.default(t, [6, 8], [2, 2], [1, 1], [0, 0], [2, 2])  # noqa: E731
    want_im, want_col = train_step(im_op, x3d), train_step(col_op, y2d)
    with cpu_as_hpu(im2col.install) as calls:
        got_im, got_col = train_step(im_op, x3d), train_step(col_op, y2d)
        assert calls["im2col"] == 1 and calls["col2im"] == 1
    assert_same(got_im, want_im)
    assert_same(got_col, want_col)

    # Non-contiguous 4D Tier 1
    x4d = torch.randn(2, 1, 11, 4, dtype=torch.float64).permute(0, 3, 2, 1)  # (2, 4, 11, 1) non-contiguous
    u4 = lambda t: F.unfold(t, (5, 1), stride=(5, 1))  # noqa: E731
    want_u4 = train_step(u4, x4d)
    with cpu_as_hpu(im2col.install) as calls:
        got_u4 = train_step(u4, x4d)
        assert calls["im2col"] == 1
    assert_same(got_u4, want_u4)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16, torch.float16, torch.bool,
                                   torch.complex64])
def test_im2col_col2im_dtypes_and_autocast(dtype):
    torch.manual_seed(0)
    if dtype == torch.bool:
        x = torch.randint(0, 2, (2, 3, 7, 9), dtype=dtype)
        y = torch.randint(0, 2, (2, 3 * 6, 24), dtype=dtype)
    else:
        x = torch.randn(2, 3, 7, 9, dtype=dtype)
        y = torch.randn(2, 3 * 6, 24, dtype=dtype)
    # Tier 2
    want_im = torch.ops.aten.im2col.default(x, [2, 3], [1, 1], [0, 0], [1, 2])
    want_col = torch.ops.aten.col2im.default(y, [7, 9], [2, 3], [1, 1], [0, 0], [1, 2])
    with cpu_as_hpu(im2col.install) as calls:
        got_im = torch.ops.aten.im2col.default(x, [2, 3], [1, 1], [0, 0], [1, 2])
        got_col = torch.ops.aten.col2im.default(y, [7, 9], [2, 3], [1, 1], [0, 0], [1, 2])
        if dtype == torch.float32:
            with torch.autocast("cpu", dtype=torch.bfloat16):
                auto_im = F.unfold(x, (2, 3), stride=(1, 2))
                auto_col = F.fold(y, (7, 9), (2, 3), stride=(1, 2))
            assert auto_im.dtype == torch.float32 and auto_col.dtype == torch.float32
            torch.testing.assert_close(auto_im, want_im, atol=0, rtol=0)
            torch.testing.assert_close(auto_col, want_col, atol=1e-5, rtol=1e-5)
    assert got_im.dtype == dtype and got_col.dtype == dtype
    assert calls["im2col"] >= 1 and calls["col2im"] >= 1
    tol = 2e-2 if dtype in (torch.float16, torch.bfloat16) else (1e-5 if dtype in (torch.float32, torch.complex64) else 0)
    torch.testing.assert_close(got_im, want_im, atol=tol, rtol=tol)
    torch.testing.assert_close(got_col, want_col, atol=tol, rtol=tol)


@pytest.mark.parametrize("tier, shape, kernel, stride", [
    ("tier1", (4, 16, 513, 1), (128, 1), (128, 1)),
    ("tier2", (4, 4, 10, 12), (3, 5), (1, 1)),
])
def test_im2col_col2im_compiled_fullgraph_dynamic(tier, shape, kernel, stride):
    torch.manual_seed(0)
    unfold_fn = lambda t: F.unfold(t, kernel, stride=stride)  # noqa: E731
    fold_fn = lambda t: F.fold(t, shape[-2:], kernel, stride=stride)  # noqa: E731

    x1 = torch.randn(*shape, dtype=torch.float64)
    x2 = torch.randn(8, *shape[1:], dtype=torch.float64)
    y_shape = unfold_fn(x1).shape
    y1 = torch.randn(*y_shape, dtype=torch.float64)
    y2 = torch.randn(8, *y_shape[1:], dtype=torch.float64)

    want_u1, want_u2 = train_step(unfold_fn, x1), train_step(unfold_fn, x2)
    want_f1, want_f2 = train_step(fold_fn, y1), train_step(fold_fn, y2)

    nodes_u, nodes_f = [], []
    with cpu_as_hpu(im2col.install) as calls:
        gu = compiled(unfold_fn, nodes_u, fullgraph=True, dynamic=True)
        gf = compiled(fold_fn, nodes_f, fullgraph=True, dynamic=True)
        got_u1, got_u2 = train_step(gu, x1), train_step(gu, x2)
        got_f1, got_f2 = train_step(gf, y1), train_step(gf, y2)
        assert calls["im2col"] >= 1 and calls["col2im"] >= 1
    assert_same(got_u1, want_u1)
    assert_same(got_u2, want_u2)
    assert_same(got_f1, want_f1)
    assert_same(got_f2, want_f2)
    assert not any("im2col" in op or "col2im" in op for op in ops(nodes_u)), ops(nodes_u)
    assert not any("im2col" in op or "col2im" in op for op in ops(nodes_f)), ops(nodes_f)


def test_im2col_col2im_backend_and_out_kernels_below_autograd():
    """Backend (.default) and .out kernels (inference_mode / below autograd on HPU), tested directly on CPU."""
    im_def, im_out = torch.ops.aten.im2col.default, torch.ops.aten.im2col.out
    col_def, col_out = torch.ops.aten.col2im.default, torch.ops.aten.col2im.out
    im_be = im2col._im2col_kernel("cpu", im_def)
    im_out_be = im2col._im2col_out_kernel("cpu", im_out)
    col_be = im2col._col2im_kernel("cpu", col_def)
    col_out_be = im2col._col2im_out_kernel("cpu", col_out)

    x = torch.randn(2, 3, 7, 9, dtype=torch.float64)
    y = torch.randn(2, 18, 24, dtype=torch.float64)
    with torch.inference_mode():
        want_im = im_def(x, [2, 3], [1, 1], [0, 0], [1, 2])
        torch.testing.assert_close(im_be(x, [2, 3], [1, 1], [0, 0], [1, 2]), want_im, atol=0, rtol=0)
        out_buf = torch.empty(1, dtype=torch.float64)
        ret = im_out_be(x, [2, 3], [1, 1], [0, 0], [1, 2], out=out_buf)
        assert ret is out_buf and out_buf.shape == want_im.shape
        torch.testing.assert_close(out_buf, want_im, atol=0, rtol=0)

        want_col = col_def(y, [7, 9], [2, 3], [1, 1], [0, 0], [1, 2])
        torch.testing.assert_close(col_be(y, [7, 9], [2, 3], [1, 1], [0, 0], [1, 2]), want_col, atol=0, rtol=0)
        out_col = torch.empty(1, dtype=torch.float64)
        ret_c = col_out_be(y, [7, 9], [2, 3], [1, 1], [0, 0], [1, 2], out=out_col)
        assert ret_c is out_col and out_col.shape == want_col.shape
        torch.testing.assert_close(out_col, want_col, atol=0, rtol=0)

        # Fallback path inside .out when dtype mismatches
        with pytest.raises(RuntimeError):
            im_out_be(x, [2, 3], [1, 1], [0, 0], [1, 2], out=torch.empty(1, dtype=torch.float32))
        with pytest.raises(RuntimeError):
            im_be(x.int(), [2, 3], [1, 1], [0, 0], [1, 2])


@pytest.mark.parametrize("bad_call", [
    lambda: F.unfold(torch.randint(0, 5, (2, 3, 6, 6), dtype=torch.int32), (3, 3)),
    lambda: F.unfold(torch.randint(0, 5, (2, 3, 6, 6), dtype=torch.int64), (3, 3)),
    lambda: F.fold(torch.randint(0, 5, (2, 27, 16), dtype=torch.int32), (6, 6), (3, 3)),
    lambda: F.unfold(torch.randn(4, 6), (2, 2)),
    lambda: torch.ops.aten.im2col.default(torch.randn(4, 6), [2, 2], [1, 1], [0, 0], [1, 1]),
    lambda: F.fold(torch.randn(12), (4, 4), (2, 2)),
    lambda: F.unfold(torch.randn(2, 3, 6, 6), (0, 3)),
    lambda: F.unfold(torch.randn(2, 3, 6, 6), (3, 3), padding=(-1, 0)),
    lambda: F.unfold(torch.randn(2, 3, 6, 6), (3, 3), dilation=(0, 1)),
    lambda: F.unfold(torch.randn(2, 3, 6, 6), (3, 3), stride=(1, 0)),
    lambda: F.unfold(torch.randn(2, 0, 6, 6), (3, 3)),
    lambda: F.unfold(torch.randn(2, 3, 2, 2), (3, 3)),
    lambda: F.fold(torch.randn(2, 27, 15), (6, 6), (3, 3)),   # mismatched L (15 != 16)
    lambda: F.fold(torch.randn(2, 26, 16), (6, 6), (3, 3)),   # non-divisible C*kh*kw
    lambda: F.fold(torch.randn(2, 0, 16), (6, 6), (3, 3)),    # C == 0
    lambda: F.fold(torch.randn(2, 27, 16), (2, 2), (3, 3)),   # oh <= 0
])
def test_im2col_col2im_invalid_inputs_reach_aten(bad_call):
    with pytest.raises(Exception) as want:
        bad_call()
    with cpu_as_hpu(im2col.install) as calls, pytest.raises(type(want.value)):
        bad_call()
    assert not calls["im2col"] and not calls["col2im"]


def test_im2col_col2im_elsewhere_stays_native():
    x = torch.randn(2, 3, 7, 9)
    y = torch.randn(2, 12, 48)
    nodes = []
    with cpu_as_hpu(lambda device: im2col.install("hpu")) as calls:
        train_step(compiled(lambda t: F.unfold(t, (2, 2)), nodes), x)
        train_step(compiled(lambda t: F.fold(t, (7, 9), (2, 2)), nodes), y)
    assert {"aten.im2col.default", "aten.col2im.default"} <= ops(nodes)
    assert not calls["im2col"] and not calls["col2im"]


# ---- installing and undoing


def test_undo_restores_torch():
    aten = torch.ops.aten
    touched = [aten.upsample_linear1d.vec, aten.upsample_nearest1d.vec, aten.upsample_nearest1d.default,
               aten._upsample_nearest_exact1d.default, aten.conv2d.padding, aten.glu.default, aten.renorm.default]
    before = [dict(op.py_kernels) for op in touched]
    reduce = torch.Tensor._reduce_ex_internal
    undo = hpu_ops._install("cpu")
    assert torch._C._dispatch_has_kernel_for_dispatch_key("aten::glu", "AutogradCPU")
    assert torch.Tensor._reduce_ex_internal is not reduce
    undo()
    assert [dict(op.py_kernels) for op in touched] == before
    assert torch.Tensor._reduce_ex_internal is reduce
    for name in ("aten::glu", "aten::renorm", "aten::upsample_nearest1d", "aten::_upsample_nearest_exact1d",
                 "aten::embedding_dense_backward", "aten::convolution", "aten::convolution_backward", "aten::roll",
                 "aten::upsample_linear1d", "aten::index.Tensor", "aten::index_put_", "aten::im2col",
                 "aten::col2im"):
        assert not torch._C._dispatch_has_kernel_for_dispatch_key(name, "AutogradCPU")
    for name in ("aten::index_put_", "aten::index_put", "aten::convolution"):
        assert not torch._C._dispatch_has_kernel_for_dispatch_key(name, "CPU")


def test_no_two_fixes_register_the_same_op_and_key(monkeypatch):
    """``register_kernels`` leaves a taken key alone, so a second fix of the same op and key would silently do nothing.
    Each fix is installed alone (then undone) and the keys it asks for are recorded."""
    asked, current = {}, [None]
    impl, py_impl = torch.library.Library.impl, torch._ops.OpOverload.py_impl

    def record_impl(self, op, fn, key="", **kwargs):
        name = op if isinstance(op, str) else op.name()
        asked.setdefault((name, key), []).append(current[0])
        return impl(self, op, fn, key, **kwargs)

    def record_py_impl(self, key):
        asked.setdefault((self.name(), str(key)), []).append(current[0])
        return py_impl(self, key)

    monkeypatch.setattr(torch.library.Library, "impl", record_impl)
    monkeypatch.setattr(torch._ops.OpOverload, "py_impl", record_py_impl)
    for fix in hpu_ops.FIXES:
        current[0] = f"{fix.__module__}.{fix.__name__}"
        fix("hpu")()  # "hpu" is not a CPU tensor's device: every key is free, nothing runs
    duplicates = {k: v for k, v in asked.items() if len(set(v)) > 1}
    assert not duplicates, duplicates
    assert ("aten::index.Tensor", "AutogradHPU") in asked and ("aten::index.Tensor", "HPU") in asked


@pytest.mark.parametrize("op", ["scaled_dot_product_attention", "_scaled_dot_product_fused_attention_overrideable"])
def test_attention_check_reads_the_fused_kernel_from_the_dispatcher(op):
    """Bridges 1.21/1.22 register their fused kernel for the first op at the device's autograd key when the setting is
    on; 1.23+ always register one for the second (habana_eager/ops/fused_sdpa.cpp, ``AutogradHPU``)."""
    assert not hpu_ops._attention_fused("cpu")
    library = torch.library.Library("aten", "IMPL")
    library.impl(op, lambda *args, **kwargs: None, "AutogradCPU")
    try:
        assert hpu_ops._attention_fused("cpu")
    finally:
        library._destroy()
    assert not hpu_ops._attention_fused("cpu")


def test_enable_is_a_no_op_off_gaudi(monkeypatch):
    if importlib.util.find_spec("habana_frameworks"):
        pytest.skip("Gaudi stack present")
    monkeypatch.delenv("PT_HPU_DISABLE_pass_weight_permutation", raising=False)
    hpu_ops.enable()
    assert hpu_ops._STATE.get("undo") is None and "PT_HPU_DISABLE_pass_weight_permutation" not in os.environ


@pytest.mark.parametrize("row", [[float("inf"), 100.0, 0.0], [3e38, 3e38, 0.0]])
def test_logsumexp_keeps_aten_gradient_with_every_fix_in_lazy_mode(monkeypatch, row):
    """Lazy-mode ``install_logsumexp`` uses ``_LseFn`` with ATen-matching non-finite/overflow gradients; eager mode
    leaves ``aten::logsumexp`` native and unregistered."""
    x = torch.tensor([row], requires_grad=True)
    (want,) = torch.autograd.grad(torch.logsumexp(x, -1).sum(), x)

    monkeypatch.setenv("PT_HPU_LAZY_MODE", "1")
    monkeypatch.setattr(hpu_ops.small, "_lazy", lambda: True)
    with cpu_as_hpu(hpu_ops._install) as calls:
        (got_lazy,) = torch.autograd.grad(torch.logsumexp(x, -1).sum(), x)
    assert calls.get("logsumexp") == 1
    torch.testing.assert_close(got_lazy, want, equal_nan=True)

    monkeypatch.setenv("PT_HPU_LAZY_MODE", "0")
    monkeypatch.setattr(hpu_ops.small, "_lazy", lambda: False)
    with cpu_as_hpu(hpu_ops._install) as calls:
        assert not torch._C._dispatch_has_kernel_for_dispatch_key("aten::logsumexp", "AutogradCPU")
        (got_eager,) = torch.autograd.grad(torch.logsumexp(x, -1).sum(), x)
    assert not calls.get("logsumexp")
    torch.testing.assert_close(got_eager, want, equal_nan=True)


# ---- Tensor.unfold: as_strided with ATen's sizes and strides (lazy mode has no HPU kernel; the CPU stands in)


@pytest.fixture
def unfold_key_free(monkeypatch):
    """The CPU key of ``aten::unfold`` counts as free, as the HPU key is in Habana's lazy mode, so ``register_kernels``
    puts the rewrite there (over the CPU kernel, restored by the undo)."""
    has = torch._C._dispatch_has_kernel_for_dispatch_key
    monkeypatch.setattr(torch._C, "_dispatch_has_kernel_for_dispatch_key",
                        lambda op, key: False if (op, key) == ("aten::unfold", "CPU") else has(op, key))


def _unfold_cases():
    base = torch.randn(4, 7, 13, dtype=torch.float64)
    return [
        ("contiguous", lambda: base.clone(), 2, 4, 3),
        ("step > size", lambda: base.clone(), 2, 3, 5),
        ("size == dim", lambda: base.clone(), 1, 7, 1),
        ("size 1", lambda: base.clone(), 0, 1, 1),
        ("negative dim", lambda: base.clone(), -1, 5, 2),
        ("negative dim -3", lambda: base.clone(), -3, 2, 2),
        ("non-contiguous", lambda: base.clone().transpose(0, 2), 1, 3, 2),
        ("offset view", lambda: base.clone()[1:, 2:, 3:], 2, 4, 3),
        ("strided slice", lambda: base.clone()[:, ::2, 1::3], 1, 2, 1),
        ("broadcast stride 0", lambda: torch.randn(1, 5, dtype=torch.float64).expand(3, 5), 0, 2, 1),
        ("1-d", lambda: torch.arange(11.0, dtype=torch.float64), 0, 4, 4),
        ("0-d", lambda: torch.tensor(3.0, dtype=torch.float64), 0, 1, 1),
        ("0-d dim -1", lambda: torch.tensor(3.0, dtype=torch.float64), -1, 0, 2),
        ("empty dim", lambda: torch.randn(3, 0, 4, dtype=torch.float64), 2, 2, 1),
        ("unfold twice", lambda: base.clone().unfold(1, 3, 2), 2, 4, 3),
    ]


@pytest.mark.parametrize("name,make,dim,size,step", _unfold_cases(), ids=[c[0] for c in _unfold_cases()])
def test_unfold_is_the_same_view(unfold_key_free, name, make, dim, size, step):
    """Sizes, strides, storage offset, values and aliasing equal ATen's; a write into the input shows in the view."""
    x = make()
    want = x.unfold(dim, size, step)
    with cpu_as_hpu(composite.install_unfold) as calls:
        got = x.unfold(dim, size, step)
    assert calls["unfold"] == 1
    assert (got.shape, got.stride(), got.storage_offset()) == (want.shape, want.stride(), want.storage_offset())
    assert torch._C._is_alias_of(got, x) and got._base is (x._base if x._base is not None else x)
    assert torch.equal(got, want)
    if x.numel() and not (x.stride() and 0 in x.stride()):
        x.add_(1)
        assert torch.equal(got, x.unfold(dim, size, step))


@pytest.mark.parametrize("dim,size,step,match", [
    (3, 2, 1, "out of range"), (2, 14, 1, "maximum size for tensor at dimension 2 is 13 but size is 14"),
    (2, 2, 0, "step is 0 but must be > 0"), (2, -1, 1, "size is -1 but must be >= 0"),
])
def test_unfold_raises_aten_errors(unfold_key_free, dim, size, step, match):
    x = torch.randn(4, 7, 13)
    with cpu_as_hpu(composite.install_unfold) as calls:
        with pytest.raises((RuntimeError, IndexError), match=match):
            x.unfold(dim, size, step)
    assert calls["unfold"] == 1  # the error is ATen's, raised from the rewrite's meta call


def _write_through(t, overlapping=True):
    """Writes through a view of an unfold of ``t`` (scaled first, so the graph input is not written): a copy instead
    of a view would leave ``c`` unchanged and the output and its gradient would differ from ATen's."""
    c = t * 1
    c.unfold(-1, 3, 3).select(-1, 0).mul_(3)
    if overlapping:  # torch.compile's functionalization bans this write, with ATen's kernel too
        c.unfold(-1, 3, 1)[..., 1].add_(c.detach()[..., :1])
    return c.sin()


def test_unfold_writes_through_the_view_in_training(unfold_key_free):
    x = torch.randn(3, 4, 11, dtype=torch.float64)
    want = train_step(_write_through, x)
    with cpu_as_hpu(composite.install_unfold) as calls:
        got = train_step(_write_through, x)
    assert calls["unfold"] == 2
    assert_same(got, want)
    assert not torch.equal(want[0], x.sin())  # the writes changed the output


def test_unfold_gradients_and_double_backward(unfold_key_free):
    """Torch's autograd kernel stays above the rewrite: gradients (unfold_backward), second order, and gradcheck."""
    x = torch.randn(3, 10, 9, dtype=torch.float64)
    f = lambda t: t.unfold(1, 4, 3).unfold(2, 2, 1).sin()  # noqa: E731  (overlapping windows: a gradient sum)
    want = train_step(f, x)
    xg = x.clone().requires_grad_()
    (g,) = torch.autograd.grad(f(xg).square().sum(), xg, create_graph=True)
    (want_gg,) = torch.autograd.grad(g.square().sum(), xg)
    with cpu_as_hpu(composite.install_unfold) as calls:
        got = train_step(f, x)
        xg = x.clone().requires_grad_()
        (g,) = torch.autograd.grad(f(xg).square().sum(), xg, create_graph=True)
        (got_gg,) = torch.autograd.grad(g.square().sum(), xg)
        assert torch.autograd.gradcheck(lambda t: t.unfold(-1, 3, 2), (torch.randn(2, 7, dtype=torch.float64,
                                                                                    requires_grad=True),))
    assert calls["unfold"] > 0
    assert_same(got, want)
    assert_same([got_gg], [want_gg])


def test_unfold_in_place_through_the_view_and_inference_mode(unfold_key_free):
    x = torch.zeros(2, 12)
    with cpu_as_hpu(composite.install_unfold) as calls:
        x.unfold(1, 3, 3).select(-1, 0).fill_(1)  # non-overlapping windows: writes reach the input
        with torch.inference_mode():
            y = torch.arange(10.0)
            u = y.unfold(0, 2, 2)
        a = torch.randn(5, 6, requires_grad=True)
        v = a.clone().unfold(1, 2, 2)
        v.mul_(2)  # an in-place op on the view of a graph tensor: autograd rebases it, as for ATen's view
        v.sum().backward()
    assert calls["unfold"] == 3
    assert torch.equal(x[:, ::3], torch.ones(2, 4)) and x.sum() == 8
    assert torch.equal(u, torch.arange(10.0).reshape(5, 2)) and torch._C._is_alias_of(u, y)
    assert torch.equal(a.grad, torch.full((5, 6), 2.0))


def test_unfold_compiles(unfold_key_free):
    """Under torch.compile the tracers see unfold's meta kernel; the CPU stand-in's aot graph then runs the rewrite.
    The compiled results and gradients equal eager's, also with writes through the view."""
    f = lambda t: (t.unfold(2, 4, 3) * 2).sum(-1)  # noqa: E731
    x = torch.randn(2, 3, 17, dtype=torch.float64)
    with cpu_as_hpu(composite.install_unfold) as calls:
        got = train_step(compiled(f, [], fullgraph=True), x)
        # 18 = 6 windows of 3: torch 2.6's functionalization replays a write through an unfold with unfold_backward,
        # which zeroes the elements no window covers (with ATen's kernel too), so the compiled write needs full cover
        x18 = torch.randn(2, 3, 18, dtype=torch.float64)
        written = train_step(compiled(lambda t: _write_through(t, False), [], fullgraph=True), x18)
    assert calls["unfold"] > 0  # the CPU aot graph runs aten ops, so the rewrite (in hpu_backend's place) runs
    assert_same(got, train_step(f, x))
    assert_same(written, train_step(lambda t: _write_through(t, False), x18))


def test_one_channel_conv_weight_grad_takes_the_unfold_rewrite(unfold_key_free):
    """TensorCSPNet's temporal block (Conv2d(1, c, (k, w * d^2), stride (1, d^2))): its weight gradient from hpu-ops'
    contraction runs both of its unfolds through the rewrite, and equals the CPU's."""
    conv2d = nn.Conv2d(1, 4, (3, 2 * 9), stride=(1, 9)).double()
    x = torch.randn(5, 1, 3, 9 * 6, dtype=torch.float64)
    (want,) = torch.autograd.grad(conv2d(x).square().sum(), conv2d.weight)
    with cpu_as_hpu(lambda d: hpu_ops._dispatch.chain([conv.install_weight_grad(d), composite.install_unfold(d)])) \
            as calls:
        (got,) = torch.autograd.grad(conv2d(x).square().sum(), conv2d.weight)
    assert calls["conv_weight_grad"] == 1 and calls["unfold"] == 2
    assert_same([got], [want])


def test_unfold_elsewhere_and_eager_hpu_keep_the_native_kernel():
    """Where the backend key has a kernel (the CPU here, Habana's eager stack on Gaudi) nothing is registered."""
    with cpu_as_hpu(composite.install_unfold) as calls:
        torch.randn(3, 8).unfold(1, 2, 2)
    assert not calls.get("unfold")


# ---- unfold_backward (Tensor.unfold's gradient): no Habana stack has a kernel; pads, reshapes and one sum (#43)


@pytest.fixture
def unfold_backward_key_free(monkeypatch):
    """The CPU key of ``aten::unfold_backward`` counts as free, as the HPU key is on Gaudi (eager and lazy), so
    ``register_kernels`` puts the rewrite there (over the CPU kernel, restored by the undo)."""
    has = torch._C._dispatch_has_kernel_for_dispatch_key
    monkeypatch.setattr(torch._C, "_dispatch_has_kernel_for_dispatch_key",
                        lambda op, key: False if (op, key) == ("aten::unfold_backward", "CPU") else has(op, key))


_UNFOLD_BACKWARD_CASES = [  # input shape, dim, size, step
    ((16, 52, 64), 1, 8, 1),  # #43's four: MetaNeuromotorHand's, overlapping
    ((16, 52, 64), 1, 4, 4),  # non-overlapping, step == size
    ((16, 8, 64), -1, 4, 2),  # negative dim, overlapping
    ((32, 128), 1, 16, 8),  # 2-D
    ((5, 13), 1, 3, 5),  # step > size: gaps between windows
    ((4, 12), -1, 3, 5),  # step > size: an uncovered tail
    ((4, 11), 1, 4, 4),  # step == size, an uncovered tail
    ((3, 17, 2), 1, 5, 2),  # step does not divide size
    ((3, 18, 2), 1, 5, 3),  # ... and a last window short of the end
    ((2, 9, 3), -2, 9, 1),  # size == length: one window
    ((2, 40, 3), 1, 30, 1),  # few long windows: the skew over the windows
    ((3, 41), 1, 30, 4),  # ... with step > 1 and an uncovered tail
    ((16, 64), -1, 4, 4),  # step == size, no tail, inner 1: a copy, never a view of the gradient
    ((2, 9, 3), 0, 1, 1),  # size 1
    ((2, 9, 3), 1, 0, 2),  # size 0: no gradient
    ((7,), 0, 3, 2),  # 1-d
    ((1, 1), 1, 1, 3),
    ((3, 0, 5), 2, 2, 1),  # empty
    ((), 0, 1, 1),  # 0-d
    ((), -1, 0, 4),
]


@pytest.mark.parametrize("shape,dim,size,step", _UNFOLD_BACKWARD_CASES, ids=[str(c) for c in _UNFOLD_BACKWARD_CASES])
@pytest.mark.parametrize("dtype", [torch.float64, torch.int32, torch.bool])
def test_unfold_backward_equals_aten(unfold_backward_key_free, shape, dim, size, step, dtype):
    """Values, dtype, shape and strides equal ATen's CPU kernel (booleans add as ``or``), for a contiguous and a
    non-contiguous gradient; the result is a new tensor, never a view of the gradient."""
    windows = torch.empty(shape).unfold(dim, size, step).shape
    grad = torch.randn(windows, dtype=torch.float64)
    grad = grad > 0 if dtype == torch.bool else (grad * 5).to(dtype)
    for g in (grad, grad.mT.contiguous().mT if grad.dim() > 1 else grad):
        want = torch.ops.aten.unfold_backward(g, list(shape), dim, size, step)
        with cpu_as_hpu(composite.install_unfold_backward) as calls:
            got = torch.ops.aten.unfold_backward(g, list(shape), dim, size, step)
        assert calls["unfold_backward"] == 1
        assert got.dtype == want.dtype and got.shape == want.shape and got.stride() == want.stride()
        assert not torch._C._is_alias_of(got, g)
        if dtype == torch.float64:
            assert_same([got], [want])
        else:
            assert torch.equal(got, want)


def test_unfold_backward_float32_and_16_bit_within_rounding(unfold_backward_key_free):
    """The overlapping sums of float32 and bfloat16 gradients, against float64."""
    for dtype, tol in ((torch.float32, 1e-6), (torch.bfloat16, 1e-2)):
        grad = torch.randn(16, 45, 64, 8, dtype=torch.float64)
        want = torch.ops.aten.unfold_backward(grad, [16, 52, 64], 1, 8, 1)
        with cpu_as_hpu(composite.install_unfold_backward):
            got = torch.ops.aten.unfold_backward(grad.to(dtype), [16, 52, 64], 1, 8, 1)
        assert got.dtype == dtype
        assert ((got.double() - want).norm() / want.norm()).item() < tol


@pytest.mark.parametrize("shape,dim,size,step", _UNFOLD_BACKWARD_CASES[:9], ids=[str(c) for c in
                                                                                 _UNFOLD_BACKWARD_CASES[:9]])
def test_unfold_backward_gradients_and_double_backward(unfold_backward_key_free, shape, dim, size, step):
    """Input gradients and second order through Tensor.unfold equal the native ones; torch's autograd kernel stays
    above the rewrite, so the derivative of unfold_backward is unfold."""
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float64)
    f = lambda t: t.unfold(dim, size, step).sin()  # noqa: E731

    def second(t):
        t = t.clone().requires_grad_()
        (g,) = torch.autograd.grad(f(t).square().sum(), t, create_graph=True)
        return torch.autograd.grad(g.square().sum(), t)[0]

    want, want_gg = train_step(f, x), second(x)
    with cpu_as_hpu(composite.install_unfold_backward) as calls:
        got, got_gg = train_step(f, x), second(x)
    assert calls["unfold_backward"] == 3  # one backward, then the double backward's first order and its second
    assert_same(got, want)
    assert_same([got_gg], [want_gg])


def test_unfold_backward_gradcheck_and_inference_mode(unfold_backward_key_free):
    with cpu_as_hpu(composite.install_unfold_backward) as calls:
        for dim, size, step in ((1, 3, 2), (-1, 2, 3), (0, 2, 2)):
            x = torch.randn(4, 7, dtype=torch.float64, requires_grad=True)
            assert torch.autograd.gradcheck(lambda t: t.unfold(dim, size, step), (x,))
            assert torch.autograd.gradgradcheck(lambda t: t.unfold(dim, size, step).square(), (x,))
        with torch.inference_mode():
            got = torch.ops.aten.unfold_backward(torch.ones(2, 4, 3), [2, 10], 1, 3, 2)
    assert calls["unfold_backward"] > 0
    assert torch.equal(got, torch.ops.aten.unfold_backward(torch.ones(2, 4, 3), [2, 10], 1, 3, 2))


@pytest.mark.parametrize("shape,dim,size,step", [
    ((4, 5, 13, 3), 2, 8, 4), ((2, 3, 9, 2, 3, 2), -4, 5, 2), ((2, 3, 9, 2, 3, 2), 0, 1, 3),
    ((2, 3, 10, 2, 3, 2), 2, 3, 3),  # step == size with a cut tail
    ((2, 3, 40, 2, 3, 2), 2, 30, 1),  # the skew over the windows
])
def test_unfold_backward_runs_ops_of_at_most_five_dims(shape, dim, size, step):
    """Gaudi's lazy ``constant_pad_nd`` falls back to the CPU above 5 dims (measured: the second unfold of conv2d
    patches, a 4-D input); the rewrite merges the dims around ``dim``, so no kernel it runs (views aside) sees more
    than 5, for a contiguous gradient (a non-contiguous one is first copied at its own rank)."""
    from torch.utils._python_dispatch import TorchDispatchMode

    class Ranks(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            out = func(*args, **(kwargs or {}))
            if not func.is_view:  # kernels, not views
                ranks.extend(t.dim() for t in [*args, out] if isinstance(t, torch.Tensor))
            return out

    ranks = []
    grad = torch.randn(torch.empty(shape).unfold(dim, size, step).shape, dtype=torch.float64)
    with Ranks():
        got = composite.unfold_backward(grad, list(shape), dim, size, step)
    assert ranks and max(ranks) <= 5, ranks
    assert_same([got], [torch.ops.aten.unfold_backward(grad, list(shape), dim, size, step)])


@pytest.mark.parametrize("shape,dim,size,step", [((16, 52, 64), 1, 8, 1), ((64, 3000), 1, 2500, 1),
                                                 ((2, 1000, 4), 1, 1000, 1), ((32, 128), 1, 16, 8)])
def test_unfold_backward_buffers_stay_near_the_gradient_size(shape, dim, size, step):
    """The skew over the blocks alone needs ``k * (n + k)`` blocks: 1001x the gradient at size == length 1000; the
    rewrite takes the smaller of its two skews, so no tensor it makes is above 1.25x the gradient here."""
    from torch.utils._python_dispatch import TorchDispatchMode

    class Peak(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            out = func(*args, **(kwargs or {}))
            sizes.append(out.numel() if isinstance(out, torch.Tensor) else 0)
            return out

    sizes = []
    grad = torch.randn(torch.empty(shape).unfold(dim, size, step).shape)
    with Peak():
        composite.unfold_backward(grad, list(shape), dim, size, step)
    assert max(sizes) <= 1.25 * grad.numel(), (max(sizes), grad.numel())


def test_unfold_backward_twice_leaves_the_upstream_gradient_alone(unfold_backward_key_free):
    """Autograd accumulates the second backward into ``x.grad`` in place: a result that aliased the incoming
    gradient (step == size, no tail) would change the caller's ``g``."""
    x = torch.randn(16, 64, requires_grad=True)
    g = torch.ones(16, 16, 4)
    with cpu_as_hpu(composite.install_unfold_backward) as calls:
        for _ in range(2):
            x.unfold(-1, 4, 4).backward(g)
    assert calls["unfold_backward"] == 2
    assert torch.equal(g, torch.ones(16, 16, 4)) and torch.equal(x.grad, torch.full((16, 64), 2.0))


@pytest.mark.parametrize("dim,step,error", [(-3, 2, IndexError), (2, 2, IndexError), (1, 0, RuntimeError)])
def test_unfold_backward_rejects_bad_arguments(dim, step, error):
    """An out-of-range dim raises ATen's IndexError; step 0 (which ``Tensor.unfold`` rejects) raises."""
    with pytest.raises(error):
        composite.unfold_backward(torch.ones(3, 5, 4), [3, 12], dim, 4, step)


def _layouts(g):
    """``g`` as contiguous, a slice with a storage offset, reversed-dims permuted, and expanded (stride 0)."""
    big = torch.zeros([s + 1 for s in g.shape], dtype=g.dtype)
    inner = tuple(slice(1, None) for _ in g.shape)
    big[inner] = g
    rev = list(range(g.dim()))[::-1]
    out = {"contiguous": g, "offset": big[inner], "permuted": g.permute(rev).contiguous().permute(rev)}
    if g.numel():
        out["expanded"] = g.flatten()[:1].reshape([1] * g.dim()).expand(g.shape)
    return out


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.complex64, torch.int8, torch.uint8,
                                   torch.int64])
@pytest.mark.parametrize("shape,dim,size,step", [((3, 7), 1, 7, 1), ((1, 7), 1, 3, 1), ((2, 5), 1, 5, 9),
                                                 ((4, 6, 3), 1, 6, 100), ((2, 3, 4, 5, 6, 7), 3, 3, 2),
                                                 ((2, 8), 1, 4, 2), ((2, 40), 1, 30, 1)])
def test_unfold_backward_dtypes_and_layouts_equal_aten(unfold_backward_key_free, dtype, shape, dim, size, step):
    """Every dtype (integer-valued, so 16-bit sums are exact in any order; int8/uint8 wrap as ATen's) and every
    gradient layout autograd can hand over: an offset slice, a permuted one, a stride-0 expand."""
    torch.manual_seed(0)
    scale = 2 if dtype in (torch.float16, torch.bfloat16) else 60  # 16-bit: sums of up to 30 stay exact
    g = (torch.randn(torch.empty(shape).unfold(dim, size, step).shape, dtype=torch.float64) * scale).round()
    g = torch.complex(g, -g).to(dtype) if dtype.is_complex else g.to(dtype)
    for name, v in _layouts(g).items():
        want = torch.ops.aten.unfold_backward(v, list(shape), dim, size, step)
        with cpu_as_hpu(composite.install_unfold_backward) as calls:
            got = torch.ops.aten.unfold_backward(v, list(shape), dim, size, step)
        assert calls["unfold_backward"] == 1
        assert got.dtype == want.dtype and got.stride() == want.stride(), name
        assert not torch._C._is_alias_of(got, v), name
        assert torch.equal(got, want), name


@pytest.mark.parametrize("shape,dim,size,step", [((2, 8), 1, 4, 4), ((2, 8), 1, 3, 5), ((2, 8), 1, 4, 2),
                                                 ((2, 40), 1, 30, 1)])
def test_unfold_backward_keeps_aten_signs_infs_and_nans(unfold_backward_key_free, shape, dim, size, step):
    """-0.0 stays -0.0 where ATen keeps it (no overlap) and turns +0.0 where ATen adds; inf + -inf is nan in the
    same places."""
    windows = torch.empty(shape).unfold(dim, size, step).shape
    g = torch.full(windows, -0.0)
    g.view(-1)[:: 3] = float("inf")
    g.view(-1)[1:: 5] = float("-inf")
    g.view(-1)[2:: 7] = float("nan")
    for v in (torch.full(windows, -0.0), g):
        want = torch.ops.aten.unfold_backward(v, list(shape), dim, size, step)
        with cpu_as_hpu(composite.install_unfold_backward):
            got = torch.ops.aten.unfold_backward(v, list(shape), dim, size, step)
        assert torch.equal(got.isnan(), want.isnan())
        assert torch.equal(got.nan_to_num(), want.nan_to_num())
        assert torch.equal(got.signbit() & ~got.isnan(), want.signbit() & ~want.isnan())


@pytest.mark.parametrize("shape,dim,size,step", [((4, 5), 1, 1, 10**7), ((4, 5), 1, 5, 10**7),
                                                 ((3, 100, 2), 1, 99, 10**6)])
def test_unfold_backward_a_step_past_the_end_pads_to_the_end_only(shape, dim, size, step):
    """One window and a step beyond the input (``Tensor.unfold`` allows it): the window is padded to the input's
    end, not to ``step`` (which made 2,000,000x the result: 160 MB for 20 floats at step 1e7)."""
    from torch.utils._python_dispatch import TorchDispatchMode

    class Peak(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            out = func(*args, **(kwargs or {}))
            sizes.append(out.numel() if isinstance(out, torch.Tensor) else 0)
            return out

    sizes = []
    grad = torch.randn(torch.empty(shape).unfold(dim, size, step).shape, dtype=torch.float64)
    with Peak():
        got = composite.unfold_backward(grad, list(shape), dim, size, step)
    assert max(sizes) <= torch.Size(shape).numel()
    assert torch.equal(got, torch.ops.aten.unfold_backward(grad, list(shape), dim, size, step))


def test_unfold_backward_under_torch_func(unfold_backward_key_free):
    """vmap (torch's per-sample fallback calls the rewrite once per sample, batch dim anywhere), jacrev and hessian
    of unfold equal the native ones."""
    torch.manual_seed(0)
    G = torch.randn(5, 3, 4, 4, dtype=torch.float64)
    x = torch.randn(3, 10, dtype=torch.float64)
    cases = [lambda: torch.vmap(lambda g: torch.ops.aten.unfold_backward(g, [3, 10], 1, 4, 2))(G),
             lambda: torch.vmap(lambda g: torch.ops.aten.unfold_backward(g, [3, 10], -1, 4, 2), in_dims=3)(
                 G.movedim(0, 3)),
             lambda: torch.func.jacrev(lambda t: t.unfold(1, 4, 2).sin())(x),
             lambda: torch.func.hessian(lambda t: t.unfold(1, 4, 2).sin().sum())(x)]
    for case in cases:
        want = case()
        with cpu_as_hpu(composite.install_unfold_backward) as calls:
            got = case()
        assert calls["unfold_backward"] > 0
        torch.testing.assert_close(got, want, rtol=0, atol=1e-12)


def test_unfold_backward_functionalized_write_through_unfold(unfold_backward_key_free):
    """Functionalization replays a write through an unfold view as ``unfold_backward`` (also under compile): the
    rewrite gives what torch's own kernel gives there. Both zero the input positions no window covers (torch 2.6
    functionalization bug, not the rewrite's), so the reference is the functionalized native op, not eager."""
    def write(t):
        t.unfold(1, 3, 3).mul_(2)
        t.unfold(1, 2, 4)[:, 1].add_(1)
        return t * 1

    x = torch.arange(30.0, dtype=torch.float64).reshape(3, 10)
    want = torch.func.functionalize(write)(x.clone())
    with cpu_as_hpu(composite.install_unfold_backward) as calls:
        got = torch.func.functionalize(write)(x.clone())
        torch._dynamo.reset()
        got_c = torch.compile(write, backend="aot_eager", dynamic=True)(x.clone())
    assert calls["unfold_backward"] >= 2
    assert torch.equal(got, want) and torch.equal(got_c, want)


def test_unfold_backward_compiled_dynamic_sees_concrete_shapes(unfold_backward_key_free):
    """Under ``torch.compile(dynamic=True)`` the op stays one graph node run eagerly: the rewrite sees plain ints, one
    graph serves every length, values equal eager."""
    from torch._dynamo.utils import counters

    torch._dynamo.reset()
    counters.clear()
    f = lambda t: t.unfold(1, 4, 2).sin().square().sum()  # noqa: E731
    cf = torch.compile(f, backend="aot_eager", dynamic=True)
    with cpu_as_hpu(composite.install_unfold_backward) as calls:
        for length in (10, 11, 17, 64, 65):
            x = torch.randn(2, length, 3, dtype=torch.float64, requires_grad=True)
            f(x).backward()
            want, x.grad = x.grad, None
            cf(x).backward()
            torch.testing.assert_close(x.grad, want, rtol=0, atol=1e-12)
    assert calls["unfold_backward"] == 10
    assert not counters["graph_break"]


def test_unfold_backward_above_the_element_limit_runs_aten_on_a_cpu_copy(unfold_backward_key_free, monkeypatch):
    """A gradient, result or buffer above ``UNFOLD_BACKWARD_MAX_ELEMENTS`` (2^31 - 1: a pad of 1.1e9 x 2 elements
    stopped Gaudi2) takes ATen's kernel on a CPU copy, a named route strict mode reports; at the limit the rewrite runs.
    The limit is lowered here (the stand-in's CPU key is the rewrite, which computes inside the route)."""
    grad = torch.randn(3, 9, 4, dtype=torch.float64)  # (3, 12) unfolded at size 4, step 1: block-skew buffer 3 * 4 * 13
    want = torch.ops.aten.unfold_backward(grad, [3, 12], 1, 4, 1)
    elements = composite.unfold_backward_elements(grad, [3, 12], 1, 4, 1)
    assert elements == 3 * 4 * (9 + 3 + 1)  # outer * k * (rows + 1) * step * inner
    with cpu_as_hpu(composite.install_unfold_backward) as calls:
        monkeypatch.setattr(composite, "UNFOLD_BACKWARD_MAX_ELEMENTS", elements)
        at_limit = torch.ops.aten.unfold_backward(grad, [3, 12], 1, 4, 1)
        assert calls["unfold_backward"] == 1 and not calls["unfold_backward_cpu"]
        monkeypatch.setattr(composite, "UNFOLD_BACKWARD_MAX_ELEMENTS", elements - 1)
        with hpu_ops.strict("report", watch_cpu_ops=False):
            routed = torch.ops.aten.unfold_backward(grad, [3, 12], 1, 4, 1)
            rows = hpu_ops.strict_report()
    assert calls["unfold_backward_cpu"] == 1 and calls["unfold_backward"] == 2  # the route's inner call computes
    assert [(r[0], r[1]) for r in rows] == [("hpu_ops route", "aten::unfold_backward")], rows
    assert_same([at_limit, routed], [want, want])


@pytest.mark.parametrize("length,routed", [(10**9, False), (11 * 10**8, True)])
def test_unfold_backward_element_limit_at_the_crash_on_gaudi(length, routed):
    """The review's measured case, ``(1, L - 1, 2)`` into ``[1, L]`` at size 2, step 1: its pad of ``2 * (L + 1)``
    elements was right on Gaudi2 at L = 1.0e9 and stopped the card at 1.1e9 (3 of 3 runs). Shapes only, no tensor."""
    plan = composite._unfold_backward_plan(length - 1, [1, length], 1, 2, 1)
    assert plan[0] == "blocks" and plan[-1] == 2 * (length + 1)
    assert (plan[-1] > composite.UNFOLD_BACKWARD_MAX_ELEMENTS) == routed


def test_unfold_backward_elsewhere_keeps_the_native_kernel():
    """Where the backend key has a kernel (the CPU here) nothing is registered."""
    x = torch.randn(3, 8, requires_grad=True)
    with cpu_as_hpu(composite.install_unfold_backward) as calls:
        x.unfold(1, 2, 2).sum().backward()
    assert not calls.get("unfold_backward")


def test_scatter_value_bool_out_of_place_inplace_and_float_autograd():
    idx = torch.tensor([[0, 2, 4]], dtype=torch.int64)
    xf = torch.randn(2, 4, requires_grad=True)
    idxf = torch.tensor([[0, 2], [1, 3]], dtype=torch.int64)
    want_yf = xf.scatter(1, idxf, 5.0)
    want_yf.sum().backward()
    want_grad = xf.grad.clone()
    xf.grad = None

    with cpu_as_hpu(composite.install_scatter_bool) as calls:
        got = torch.zeros((1, 6), dtype=torch.bool).scatter(1, idx, True)
        got_in = torch.ones((1, 6), dtype=torch.bool)
        got_in.scatter_(1, idx, False)
        compiled = torch.compile(lambda a, i: a.scatter(1, i, True), backend="aot_eager")
        got_compiled = compiled(torch.zeros((1, 6), dtype=torch.bool), idx)
        got_yf = xf.scatter(1, idxf, 5.0)
        got_yf.sum().backward()
        got_grad = xf.grad.clone()

    assert calls["scatter_bool"] >= 2
    assert torch.equal(got, torch.tensor([[True, False, True, False, True, False]]))
    assert torch.equal(got_in, torch.tensor([[False, True, False, True, False, True]]))
    assert torch.equal(got_compiled, got)
    torch.testing.assert_close(got_yf, want_yf, rtol=0, atol=0)
def test_gather_int64_bit_exact_and_edge_cases():
    x = torch.tensor(
        [[0, -1, -(1 << 62), (1 << 62) + 7], [42, 99, -5, 1 << 50]],
        dtype=torch.int64,
    ).t()
    idx = torch.tensor([[1, 0], [0, 1], [1, 1]], dtype=torch.int64)
    want = x.gather(-1, idx)
    want_0d = torch.tensor(77, dtype=torch.int64).gather(0, torch.tensor(0))
    want_empty = torch.arange(5, dtype=torch.int64).gather(0, torch.empty((0,), dtype=torch.int64))

    with cpu_as_hpu(composite.install_gather) as calls:
        got = x.gather(-1, idx)
        got_0d = torch.tensor(77, dtype=torch.int64).gather(0, torch.tensor(0))
        got_empty = torch.arange(5, dtype=torch.int64).gather(0, torch.empty((0,), dtype=torch.int64))
        compiled = torch.compile(lambda a, i: a.gather(-1, i), backend="aot_eager")
        got_compiled = compiled(x, idx)

    assert calls["gather_int64"] >= 3
    assert torch.equal(got, want)
    assert torch.equal(got_0d, want_0d)
    assert torch.equal(got_empty, want_empty)
    assert torch.equal(got_compiled, want)


def test_gather_preserves_float_autograd_and_native_passthrough():
    x = torch.randn(3, 4, requires_grad=True)
    idx = torch.tensor([[0, 2], [3, 1], [1, 0]], dtype=torch.int64)
    want_y = x.gather(1, idx)
    want_y.sum().backward()
    want_grad = x.grad.clone()
    x.grad = None

    with cpu_as_hpu(composite.install_gather) as calls:
        got_y = x.gather(1, idx)
        got_y.sum().backward()
        got_grad = x.grad.clone()

    assert not calls.get("gather_int64")
    torch.testing.assert_close(got_y, want_y, rtol=0, atol=0)


    torch.testing.assert_close(got_grad, want_grad, rtol=0, atol=0)
