"""CPU stand-in tests for hpu_ops.assert_async (#96): ``aten::_assert_async`` as a host check matches
ATen's CPU kernel (pass, fail with the message, ambiguous inputs), the kernel is reached eagerly and
in inference mode, and compiled graphs keep the node and check it when they run."""
import pytest
import torch
from test_cpu import compiled, cpu_as_hpu, ops

import hpu_ops
from hpu_ops import assert_async

MSG = torch.ops.aten._assert_async.msg
DEFAULT = torch.ops.aten._assert_async.default


def _native(fn):
    try:
        fn()
    except RuntimeError as error:
        return type(error), str(error)
    return None, None


def _cases():
    return [
        (torch.tensor(True), "pass"),
        (torch.tensor([True]), "pass"),
        (torch.tensor([[True]]), "pass"),
        (torch.tensor(1), "pass"),
        (torch.tensor(1.5), "pass"),
        (torch.tensor(float("nan")), "pass"),
        (torch.tensor(1 + 1j), "pass"),
        (torch.tensor(False), "zero"),
        (torch.tensor(0), "zero"),
        (torch.tensor(0.0), "zero"),
        (torch.tensor(-0.0), "zero"),
        (torch.tensor(0j), "zero"),
        (torch.tensor([True, False]), "many"),
        (torch.tensor([]), "none"),
    ]


def test_assert_async_matches_aten():
    for tensor, kind in _cases():
        for call, args in ((MSG, (tensor, "guard failed")), (DEFAULT, (tensor,))):
            want = _native(lambda: call(*args))
            with cpu_as_hpu(assert_async.install_assert_async) as calls:
                got = _native(lambda: call(*args))
            assert got == want, (tensor, call, got, want)
            assert calls["assert_async"] == 1, (tensor, call, calls)


def test_assert_async_failing_raises_with_message():
    with cpu_as_hpu(assert_async.install_assert_async) as calls:
        with pytest.raises(RuntimeError, match="must be positive"):
            torch._assert_async(torch.tensor(False), "must be positive")
        with pytest.raises(RuntimeError, match="single nonzero value"):
            torch._assert_async(torch.tensor(False))
        with pytest.raises(RuntimeError, match="Assertion is failed"):
            MSG(torch.tensor(False), "")
    assert calls["assert_async"] == 3


def test_assert_async_passing_and_kernel_reached():
    with cpu_as_hpu(assert_async.install_assert_async) as calls:
        torch._assert_async(torch.tensor(True), "unused")
        torch._assert_async((torch.tensor([1.0, 2.0]) > 0).all(), "must be positive")
        assert calls["assert_async"] == 2
        calls.clear()
        with torch.inference_mode():
            torch._assert_async(torch.tensor(True), "unused")
        with torch.no_grad():
            torch._assert_async(torch.tensor(True), "unused")
    assert calls["assert_async"] == 2


def test_assert_async_compiled_keeps_node_and_checks_at_run_time():
    def f(x):
        torch._assert_async((x > 0).all(), "must be positive")
        return x * 2

    nodes = []
    with cpu_as_hpu(assert_async.install_assert_async):
        graph = compiled(f, nodes, fullgraph=True)
        torch.testing.assert_close(graph(torch.tensor([1.0, 2.0])), torch.tensor([2.0, 4.0]))
        assert "aten._assert_async.msg" in ops(nodes), ops(nodes)
        with pytest.raises(RuntimeError, match="must be positive"):
            graph(torch.tensor([1.0, -1.0]))


def test_assert_async_traced_is_a_noop():
    """A fake tensor computed from an input carries no value to read: ATen's Meta kernel for this
    op returns without checking, and no host read is counted."""
    from torch._subclasses.fake_tensor import FakeTensorMode

    with cpu_as_hpu(assert_async.install_assert_async) as calls, FakeTensorMode():
        x = torch.randn(3)
        torch._assert_async((x > 0).all(), "never raised while tracing")
    assert calls["assert_async"] == 0


def test_assert_async_host_read_is_one_strict_host_sync_row():
    with cpu_as_hpu(assert_async.install_assert_async):
        with hpu_ops.strict("report", watch_cpu_ops=False):
            torch._assert_async(torch.tensor(True), "unused")
        with pytest.raises(hpu_ops.CPUExecutionError, match="host sync"):
            with hpu_ops.strict(watch_cpu_ops=False):
                torch._assert_async(torch.tensor(True), "unused")
    rows = hpu_ops.strict_report()
    assert [(row[0], row[3]) for row in rows] == [("host sync", 1)], rows
