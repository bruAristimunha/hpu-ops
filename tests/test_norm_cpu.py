"""CPU tests of hpu_ops.norm (#42), "cpu" standing in for "hpu": under torch.compile with a symbolic size,
``group_norm`` traces to a composite that takes no size as an int argument (Habana's backend fails on
``native_group_norm``'s symbolic batch); static shapes, eager calls and other devices keep the native op."""
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import hpu_ops
from hpu_ops import norm

from test_cpu import assert_same, compiled, cpu_as_hpu, ops, train_step

NATIVE = "aten.native_group_norm.default"


def _symbolic_sizes(nodes):
    """The ``native_group_norm`` nodes that get a size as a symbolic int (what Habana converts with IntImplicit)."""
    return [n for n in nodes if str(n.target) == NATIVE
            and any(isinstance(a, torch.fx.Node) and isinstance(a.meta.get("val"), torch.SymInt) for a in n.args)]


def _module(groups, channels, affine=True, dtype=torch.float64):
    torch.manual_seed(0)
    m = nn.GroupNorm(groups, channels, affine=affine).to(dtype)
    if affine:
        with torch.no_grad():
            m.weight.uniform_(0.5, 1.5)
            m.bias.uniform_(-0.5, 0.5)
    return m


@pytest.mark.parametrize("order", [(16, 128), (128, 16)])
def test_second_batch_size_traces_the_composite(order):
    """The issue's steps 1-4: default compile, a second batch size recompiles with a symbolic batch."""
    m = _module(4, 16)
    nodes = []
    with cpu_as_hpu(norm.install_group_norm) as calls:
        g = compiled(m, nodes)
        for batch in order:
            x = torch.randn(batch, 16, 100, dtype=torch.float64)
            m.zero_grad()
            got = train_step(g, x) + [m.weight.grad.clone(), m.bias.grad.clone()]
            m.zero_grad()
            want = train_step(m, x) + [m.weight.grad.clone(), m.bias.grad.clone()]
            assert_same(got, want, tol=1e-10)
    assert calls["group_norm"] and NATIVE in ops(nodes)  # the first, static graph keeps the native op
    assert not _symbolic_sizes(nodes), ops(nodes)


def test_static_compile_keeps_the_native_op_for_each_batch_size():
    """Step 5: dynamic=False recompiles per batch size, each graph with Gaudi's kernel."""
    m = _module(4, 16)
    nodes = []
    with cpu_as_hpu(norm.install_group_norm) as calls:
        g = compiled(m, nodes, dynamic=False)
        for batch in (16, 128):
            x = torch.randn(batch, 16, 100, dtype=torch.float64)
            assert_same(train_step(g, x), train_step(m, x), tol=1e-10)
    assert not calls["group_norm"] and NATIVE in ops(nodes)


@pytest.mark.parametrize("shape, groups, affine", [
    ((5, 8), 4, True),  # (N, C): one inner element
    ((1, 6, 7), 3, True),  # batch 1
    ((3, 6, 7), 6, False),  # one channel per group, no affine
    ((3, 8, 2, 5), 1, True),  # 4-D, one group
    ((2, 4, 3, 2, 2), 2, True),  # 5-D
])
def test_symbolic_shapes_match_eager(shape, groups, affine):
    m = _module(groups, shape[1], affine)
    x = torch.randn(shape, dtype=torch.float64)
    nodes = []
    with cpu_as_hpu(norm.install_group_norm) as calls:
        got = train_step(compiled(m, nodes, dynamic=True, fullgraph=True), x)
    assert_same(got, train_step(m, x), tol=1e-10)
    assert calls["group_norm"] and NATIVE not in ops(nodes), ops(nodes)


def test_noncontiguous_input_and_symbolic_length():
    """A transposed input, every size symbolic (a varying length too)."""
    f = lambda x, w, b: F.group_norm(x.transpose(1, 2), 2, w, b)  # noqa: E731
    w, b = torch.rand(6, dtype=torch.float64) + 0.5, torch.randn(6, dtype=torch.float64)
    nodes = []
    with cpu_as_hpu(norm.install_group_norm) as calls:
        g = compiled(f, nodes, dynamic=True, fullgraph=True)
        for batch, length in ((3, 11), (5, 4)):
            x = torch.randn(batch, length, 6, dtype=torch.float64)
            assert_same(train_step(g, x, w, b), train_step(f, x, w, b), tol=1e-10)
    assert calls["group_norm"] and not _symbolic_sizes(nodes)


@pytest.mark.parametrize("dtype, ulps", [(torch.bfloat16, 0), (torch.float16, 1)])
def test_16_bit_inputs_compute_in_float32(dtype, ulps):
    """Against torch's own 16-bit kernel: bfloat16 bit for bit, float16 within one ulp (one rounding of the
    float32 result)."""
    x = torch.randn(4, 8, 33).to(dtype) * 3 + 1
    w, b = torch.rand(8).to(dtype) + 0.5, torch.randn(8).to(dtype)
    want = F.group_norm(x, 4, w, b)
    with cpu_as_hpu(norm.install_group_norm) as calls:
        got = compiled(lambda x: F.group_norm(x, 4, w, b), [], dynamic=True)(x)
    assert got.dtype == dtype and calls["group_norm"]
    ulp = torch.finfo(dtype).eps * want.float().abs().clamp_min(1)
    assert bool(((got.float() - want.float()).abs() <= ulps * ulp).all())


@pytest.mark.parametrize("which", ["weight", "bias"])
def test_weight_only_or_bias_only(which):
    x = torch.randn(3, 6, 5, dtype=torch.float64)
    kw = {which: torch.rand(6, dtype=torch.float64) + 0.5}
    f = lambda x: F.group_norm(x, 3, **kw)  # noqa: E731
    with cpu_as_hpu(norm.install_group_norm) as calls:
        got = train_step(compiled(f, [], dynamic=True), x)
    assert_same(got, train_step(f, x), tol=1e-10)
    assert calls["group_norm"]


def test_batch_only_symbolic_training_graph_has_no_symbolic_scalars():
    """With only the batch symbolic (the issue's case), no op of the forward or backward graph takes a symbolic
    scalar argument, the kind of value Habana's backend failed on."""
    m = _module(4, 16)
    nodes = []
    with cpu_as_hpu(norm.install_group_norm):
        g = compiled(m, nodes)
        for batch in (16, 128):
            train_step(g, torch.randn(batch, 16, 10, dtype=torch.float64))

    def symbolic_scalar(a):
        return isinstance(a, (torch.SymInt, torch.SymFloat)) or (
            isinstance(a, torch.fx.Node) and isinstance(a.meta.get("val"), (torch.SymInt, torch.SymFloat)))

    bad = [n for n in nodes if str(n.target).startswith("aten.") and "sym_size" not in str(n.target)
           and str(n.target) not in ("aten.view.default", "aten.reshape.default", "aten.expand.default",
                                     "aten._unsafe_view.default")
           and any(symbolic_scalar(a) for a in n.args)]
    assert not bad, [(str(n.target), n.args) for n in bad]


def test_wrong_weight_size_raises_torchs_error():
    """A weight that is not one value per channel: torch's own error, not a broadcast."""
    f = lambda x: F.group_norm(x, 2, weight=torch.ones(1))  # noqa: E731
    with cpu_as_hpu(norm.install_group_norm) as calls, pytest.raises(Exception, match="weight"):
        compiled(f, [], dynamic=True)(torch.randn(3, 4, 5))
    assert not calls["group_norm"]


def test_inference_mode_and_extreme_values():
    """Under no_grad, inputs offset by 1e4 with a small spread, and eps large against the variance: as close to the
    exact result (group norm does not see the offset) as ATen's kernel, whose own error there is about 1e-9."""
    centered = torch.randn(3, 4, 50, dtype=torch.float64) * 1e-3
    x = centered + 1e4
    for eps in (1e-5, 1.0):
        f = lambda x: F.group_norm(x, 2, eps=eps)  # noqa: E731,B023
        with cpu_as_hpu(norm.install_group_norm) as calls, torch.no_grad():
            got = compiled(f, [], dynamic=True)(x)
        exact = f(centered)
        assert float((got - exact).abs().max()) <= max(1e-9, 4 * float((f(x) - exact).abs().max()))
        assert calls["group_norm"]


def test_invalid_groups_raise_torchs_error():
    """Channels not divisible by the groups: torch's own error, as without the rewrite."""
    f = lambda x: F.group_norm(x, 3)  # noqa: E731
    with cpu_as_hpu(norm.install_group_norm) as calls, pytest.raises(Exception, match="divisible"):
        compiled(f, [], dynamic=True)(torch.randn(2, 4, 5))
    assert not calls["group_norm"]


def test_eager_and_other_devices_keep_the_native_op():
    m = _module(4, 16)
    x = torch.randn(8, 16, 10, dtype=torch.float64)
    nodes = []
    with cpu_as_hpu(norm.install_group_norm) as calls:
        m(x)
        assert not calls["group_norm"]  # eager calls never read the Python kernels
    undo = norm.install_group_norm("xpu")  # tensors of another device than the one installed for
    try:
        hpu_ops._STATE["calls"].clear()
        compiled(m, nodes, dynamic=True)(x)
    finally:
        undo()
    assert not hpu_ops._STATE["calls"]["group_norm"] and _symbolic_sizes(nodes)


def test_undo_restores_torchs_kernel():
    op = torch.ops.aten.group_norm.default
    before = dict(op.py_kernels)
    norm.install_group_norm("cpu")()
    assert op.py_kernels == before
    nodes = []
    compiled(lambda x: F.group_norm(x, 2), nodes, dynamic=True)(torch.randn(3, 4, 5))
    assert _symbolic_sizes(nodes)


def test_enable_installs_it():
    assert norm.install_group_norm in hpu_ops.FIXES


# ---- round 2 (adversarial)


@pytest.mark.parametrize("fmt, shape", [(torch.channels_last, (3, 8, 4, 5)), (torch.channels_last_3d, (2, 4, 3, 2, 5))])
def test_memory_format_is_kept(fmt, shape):
    x = torch.randn(shape, dtype=torch.float64).to(memory_format=fmt)
    f = lambda x: F.group_norm(x, 4)  # noqa: E731
    with cpu_as_hpu(norm.install_group_norm) as calls:
        got = compiled(f, [], dynamic=True)(x)
    want = f(x)
    assert calls["group_norm"] and got.stride() == want.stride()
    assert_same([got], [want], tol=1e-12)


@pytest.mark.parametrize("dtypes", [(torch.float32, torch.float64), (torch.float64, torch.float32),
                                    (torch.float32, torch.bfloat16)])
def test_mixed_dtypes_are_not_taken(dtypes):
    """Input and weight dtypes ATen's kernel rejects reach it (and its error); the composite would promote."""
    x, w = torch.randn(3, 4, 5, dtype=dtypes[0]), torch.ones(4, dtype=dtypes[1])
    with cpu_as_hpu(norm.install_group_norm) as calls, pytest.raises(Exception, match="dtype"):
        compiled(lambda x: F.group_norm(x, 2, w), [], dynamic=True)(x)
    assert not calls["group_norm"]


def test_16_bit_input_with_float32_parameters():
    x = torch.randn(3, 4, 9).bfloat16()
    w, b = torch.rand(4) + 0.5, torch.randn(4)
    f = lambda x: F.group_norm(x, 2, w, b)  # noqa: E731
    with cpu_as_hpu(norm.install_group_norm) as calls:
        got = compiled(f, [], dynamic=True)(x)
    assert calls["group_norm"] and got.dtype == torch.bfloat16
    torch.testing.assert_close(got, f(x), atol=0, rtol=0)


@pytest.mark.parametrize("view", [lambda x: x[1:], lambda x: x[..., 2:], lambda x: x[..., ::2],
                                  lambda x: x[:1].expand(4, -1, -1)])
def test_views_and_expanded_inputs(view):
    base = torch.randn(5, 6, 12, dtype=torch.float64)
    m = _module(3, 6)
    f = lambda x: m(view(x))  # noqa: E731
    with cpu_as_hpu(norm.install_group_norm) as calls:
        got = train_step(compiled(f, [], dynamic=True), base)
    assert_same(got, train_step(f, base), tol=1e-10)
    assert calls["group_norm"]


def test_weight_grad_only_with_frozen_input():
    m = _module(2, 4)
    x = torch.randn(3, 4, 7, dtype=torch.float64)
    with cpu_as_hpu(norm.install_group_norm):
        compiled(m, [], dynamic=True)(x).square().sum().backward()
    got = [m.weight.grad.clone(), m.bias.grad.clone()]
    m.zero_grad()
    m(x).square().sum().backward()
    assert_same(got, [m.weight.grad, m.bias.grad], tol=1e-10)


def test_zero_length_inner_dim():
    """An empty group: output and input gradient empty, as eager; the weight gradient is the sum over nothing, 0
    (ATen's CPU kernel gives NaN there)."""
    m = _module(4, 8)
    x = torch.randn(3, 8, 0, dtype=torch.float64)
    with cpu_as_hpu(norm.install_group_norm):
        out, gx = train_step(compiled(m, [], dynamic=True), x)
    assert out.shape == (3, 8, 0) and gx.shape == (3, 8, 0)
    assert bool((m.weight.grad == 0).all())


def test_symbolic_channels_train():
    """A symbolic channel count trains (torch 2.6's native_group_norm_backward decomposition fails on it)."""
    f = lambda x: F.group_norm(x, 2)  # noqa: E731
    with cpu_as_hpu(norm.install_group_norm) as calls:
        g = compiled(f, [], dynamic=True)
        for c in (4, 6):
            x = torch.randn(3, c, 5, dtype=torch.float64)
            assert_same(train_step(g, x), train_step(f, x), tol=1e-10)
    assert calls["group_norm"]


@pytest.mark.parametrize("dynamic_dim", [0, 2])
def test_with_every_fix_installed(dynamic_dim):
    """conv, GroupNorm, linear upsampling and GroupNorm(C, C) with all of hpu-ops' rewrites, one dim dynamic."""
    torch.manual_seed(0)
    m = nn.Sequential(nn.Conv1d(3, 8, 3, padding=1), nn.GroupNorm(4, 8), nn.Upsample(scale_factor=2, mode="linear"),
                      nn.GroupNorm(8, 8), nn.Conv1d(8, 2, 1)).double()
    x = torch.randn(4, 3, 16, dtype=torch.float64)
    torch._dynamo.reset()
    undo = hpu_ops._install("cpu")
    try:
        hpu_ops._STATE["calls"].clear()
        g = torch.compile(m, backend="aot_eager")
        torch._dynamo.mark_dynamic(x, dynamic_dim)
        g(x).square().sum().backward()
        got = [p.grad.clone() for p in m.parameters()]
        assert hpu_ops._STATE["calls"]["group_norm"]
    finally:
        undo()
    m.zero_grad()
    m(x).square().sum().backward()
    assert_same(got, [p.grad for p in m.parameters()], tol=1e-10)


def test_empty_batch_with_another_symbolic_size_compiles():
    """An empty batch has no ``-1`` to infer in the group reshape (review of #49)."""
    x = torch.randn(0, 4, 7)
    torch._dynamo.mark_dynamic(x, 2)
    torch._dynamo.reset()
    with cpu_as_hpu(norm.install_group_norm):
        out = torch.compile(lambda t: torch.nn.functional.group_norm(t, 2), backend="aot_eager")(x)
    torch._dynamo.reset()
    assert out.shape == (0, 4, 7)
