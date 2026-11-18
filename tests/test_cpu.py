"""The rewrites checked on the CPU standing in for HPU: what compiled graphs contain, that eager and compiled results
match the native ops, and that other devices and symbolic shapes keep torch's own path."""
import contextlib
import importlib.util
import os

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from functorch.compile import make_boxed_func
from torch._dynamo.backends.common import aot_autograd

import hpu_ops
from hpu_ops import composite, conv, serialization, upsample


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
    for name in ("aten::glu", "aten::renorm", "aten::upsample_nearest1d", "aten::_upsample_nearest_exact1d"):
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
