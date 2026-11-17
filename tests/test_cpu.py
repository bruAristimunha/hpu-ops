"""The upsampling override, checked on the CPU standing in for HPU: what compiled graphs contain, and that
symbolic shapes keep PyTorch's decomposition instead of crashing."""
import pytest
import torch
import torch.nn.functional as F
from functorch.compile import make_boxed_func
from torch._dynamo.backends.common import aot_autograd

import hpu_ops


@pytest.fixture
def cpu_as_hpu():
    """The override with "cpu" in the role of "hpu", undone afterwards."""
    undo = hpu_ops._override_upsample_linear1d_vec("cpu")
    hpu_ops._STATE["native_calls"] = 0
    yield
    undo()


def recording_backend(ops):
    def record(gm, example_inputs):
        ops.update(str(n.target) for n in gm.graph.nodes if n.op == "call_function")
        return make_boxed_func(gm.forward)

    return aot_autograd(fw_compiler=record, bw_compiler=record)


def upsample(x):
    return F.interpolate(x, scale_factor=2.0, mode="linear")


def train_step(f, x):
    x = x.detach().requires_grad_()
    out = f(x)
    out.square().sum().backward()
    return out.detach(), x.grad


@pytest.mark.parametrize("align_corners", [False, True])
@pytest.mark.parametrize("scale", [2.0, 256.0, 2.7])
def test_compiled_graph_keeps_the_native_op(cpu_as_hpu, scale, align_corners):
    ops = set()
    f = lambda x: F.interpolate(x, scale_factor=scale, mode="linear", align_corners=align_corners)  # noqa: E731
    x = torch.randn(8, 5, 3, dtype=torch.float64)
    torch._dynamo.reset()
    got = train_step(torch.compile(f, backend=recording_backend(ops)), x)
    for a, b in zip(got, train_step(f, x)):
        torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-12)
    assert "aten.upsample_linear1d.default" in ops and "aten.upsample_linear1d_backward.default" in ops
    assert not any("index_put" in op for op in ops), ops


def test_new_batch_size_recompiles_through_the_decomposition(cpu_as_hpu):
    """A second shape marks the batch dimension dynamic: torch's decomposition, not a crash."""
    ops = set()
    torch._dynamo.reset()
    compiled = torch.compile(upsample, backend=recording_backend(ops))
    for batch in (8, 5, 3):
        x = torch.randn(batch, 5, 3, dtype=torch.float64)
        for a, b in zip(train_step(compiled, x), train_step(upsample, x)):
            torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-12)


def test_other_devices_and_undo_keep_the_decomposition():
    ops = set()
    undo = hpu_ops._override_upsample_linear1d_vec("hpu")  # CPU tensors are not "hpu"
    try:
        torch._dynamo.reset()
        train_step(torch.compile(upsample, backend=recording_backend(ops)), torch.randn(8, 5, 3))
    finally:
        undo()
    assert any("index" in op for op in ops) and "aten.upsample_linear1d.default" not in ops, ops
