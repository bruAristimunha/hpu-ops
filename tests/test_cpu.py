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
from hpu_ops import autocast, composite, conv, index, serialization, upsample


@contextlib.contextmanager
def cpu_as_hpu(install):
    """The rewrite ``install`` with "cpu" in the role of "hpu", undone afterwards; yields the rewrite call counts."""
    undo = install("cpu")
    hpu_ops._STATE["calls"].clear()
    try:
        yield hpu_ops._STATE["calls"]
    finally:
        undo()


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


def test_glu_elsewhere_stays_glu():
    nodes = []
    with cpu_as_hpu(lambda device: composite.install_glu("hpu")) as calls:
        train_step(compiled(lambda x: F.glu(x, 1), nodes), torch.randn(4, 6, 8))
    assert "aten.glu.default" in ops(nodes) and not calls["glu"]


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


# ---- torch.save: tensor-level host copies


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


# ---- advanced indexing (aten::index.Tensor) as index_select / device-moved indexing


@pytest.mark.parametrize("mode", ["eager", "compiled"])
def test_index_contiguous_triu_matches_torch_and_compiles(mode):
    """SPDNet / sym_to_upper pattern: (32, 22, 22) indexed at trailing [..., i0, i1]."""
    torch.manual_seed(0)
    ix = torch.triu_indices(22, 22)
    f = lambda x: x[..., ix[0], ix[1]]  # noqa: E731
    x = torch.randn(32, 22, 22, dtype=torch.float64)
    want, nodes = train_step(f, x), []
    with cpu_as_hpu(index.install) as calls:
        g = f if mode == "eager" else compiled(f, nodes, fullgraph=True, dynamic=True)
        got = train_step(g, x)
        if mode == "compiled":
            x2 = torch.randn(5, 22, 22, dtype=torch.float64)
            assert_same(train_step(g, x2), train_step(f, x2))
    assert_same(got, want)
    assert calls["index"] > 0
    if mode == "compiled":
        assert "aten.index_select.default" in ops(nodes)
        assert not any("aten.index." in op for op in ops(nodes)), ops(nodes)


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
    with cpu_as_hpu(index.install) as calls:
        got = train_step(f, x)
    assert_same(got, want)
    assert calls["index"] == 1


def test_index_non_contiguous_self_and_compiled_non_contiguous_indices():
    torch.manual_seed(0)
    x = torch.randn(5, 3, 6, dtype=torch.float64).permute(1, 2, 0)
    i0 = torch.tensor([-1, 0, 1, 0])
    i1 = torch.tensor([2, -1, 0, 2])
    f = lambda t: t[i0, :, i1]  # noqa: E731
    want, nodes = train_step(f, x), []
    with cpu_as_hpu(index.install) as calls:
        eager = train_step(f, x)
        got = train_step(compiled(f, nodes, fullgraph=True, dynamic=True), x)
    assert_same(eager, want)
    assert_same(got, want)
    assert calls["index"] >= 2
    assert not any("aten.index." in op for op in ops(nodes)), ops(nodes)


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
    with cpu_as_hpu(index.install) as calls:
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
    with cpu_as_hpu(index.install) as calls:
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
    backend = index._device_kernel("cpu", op, torch._C.DispatchKey.CompositeExplicitAutogradNonFunctional)
    ix = torch.triu_indices(5, 5)
    with torch.inference_mode():
        torch.testing.assert_close(backend(x, [None, ix[0], ix[1]]), op(x, [None, ix[0], ix[1]]), atol=0, rtol=0)
    assert index._classify(x, [torch.ones(6, dtype=torch.bool, device="meta")]) == "move"
    assert index._classify(x, [None, torch.tensor(True, device="meta")]) == "move"


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
    with cpu_as_hpu(index.install) as calls, pytest.raises(type(want.value)):
        x[idx]
    assert not calls["index"]


def test_index_elsewhere_stays_native():
    x = torch.randn(4, 5, 5)
    ix = torch.triu_indices(5, 5)
    nodes = []
    with cpu_as_hpu(lambda device: index.install("hpu")) as calls:
        train_step(compiled(lambda t: t[..., ix[0], ix[1]], nodes), x)
    assert "aten.index.Tensor" in ops(nodes) and not calls["index"]


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
                 "aten::embedding_dense_backward", "aten::convolution_backward", "aten::roll",
                 "aten::upsample_linear1d", "aten::index.Tensor"):
        assert not torch._C._dispatch_has_kernel_for_dispatch_key(name, "AutogradCPU")


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
