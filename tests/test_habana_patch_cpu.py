"""The vendor backend patches (#63 output_strides, #121 as_strided decompositions) on CPU."""
import pytest
import torch
from functorch.compile import make_boxed_func
from torch._decomp import get_decompositions
from torch._dynamo.backends.common import aot_autograd

import hpu_ops
from hpu_ops import habana_patch


class _M(torch.nn.Module):
    def forward(self, x, y):
        return x @ y


def _graph_with_missing():
    gm = torch.fx.symbolic_trace(_M())
    nodes = list(gm.graph.nodes)
    ph = [n for n in nodes if n.op == "placeholder"]
    ph[0].meta["val"] = torch.randn(4, 4).t()  # non-contiguous fake value
    ph[1].meta["val"] = torch.randn(4, 4)
    ph[1].meta["output_strides"] = [(999,)]  # vendor-computed: must not be touched
    return gm, ph


def test_fills_only_missing_from_fake_value():
    gm, ph = _graph_with_missing()
    assert habana_patch.fill_missing_output_strides(gm) == 1
    assert ph[0].meta["output_strides"] == [tuple(ph[0].meta["val"].stride())]
    assert ph[1].meta["output_strides"] == [(999,)]


def test_no_fake_value_no_fill():
    gm = torch.fx.symbolic_trace(_M())
    assert habana_patch.fill_missing_output_strides(gm) == 0


def test_patch_recipe_compiler_noop_off_habana():
    assert habana_patch.patch_recipe_compiler() is False  # no habana_frameworks here


def _as_strided_table():
    aten = torch.ops.aten
    return get_decompositions([aten.unfold.default, aten.diagonal.default])


def _compile_with_table(fn, table, nodes, **options):
    def record(gm, example_inputs):
        nodes.extend(n for n in gm.graph.nodes if n.op == "call_function")
        return make_boxed_func(gm.forward)

    torch._dynamo.reset()
    backend = aot_autograd(fw_compiler=record, bw_compiler=record, decompositions=table)
    return torch.compile(fn, backend=backend, **options)


def _op_names(nodes):
    return [str(n.target) for n in nodes]


def test_patch_as_strided_decompositions_noop_off_habana():
    undo = habana_patch.patch_as_strided_decompositions()
    undo()
    undo()  # idempotent


def test_patch_as_strided_decompositions_idempotent_and_undoable():
    aten = torch.ops.aten
    table = _as_strided_table()
    orig_unfold = table[aten.unfold.default]
    orig_diag = table[aten.diagonal.default]

    undo1 = habana_patch.patch_as_strided_decompositions(table=table, device_type="cpu")
    patched_unfold = table[aten.unfold.default]
    patched_diag = table[aten.diagonal.default]
    assert patched_unfold is not orig_unfold
    assert patched_diag is not orig_diag

    # Second call is a no-op and its undo must not disturb the first install
    undo2 = habana_patch.patch_as_strided_decompositions(table=table, device_type="cpu")
    assert table[aten.unfold.default] is patched_unfold
    assert table[aten.diagonal.default] is patched_diag
    undo2()
    assert table[aten.unfold.default] is patched_unfold
    assert table[aten.diagonal.default] is patched_diag

    undo1()
    assert table[aten.unfold.default] is orig_unfold
    assert table[aten.diagonal.default] is orig_diag
    undo1()  # idempotent


def test_unfold_decomp_contiguous_zero_offset_stays_zero_copy():
    table = _as_strided_table()
    undo = habana_patch.patch_as_strided_decompositions(table=table, device_type="cpu")
    hpu_ops._STATE["calls"].clear()
    try:
        x = torch.randn(4, 32, 16)
        fn = lambda a: torch.nan_to_num(a.unfold(-1, 8, 8), nan=0.0)  # noqa: E731
        nodes = []
        cf = _compile_with_table(fn, table, nodes)
        got = cf(x)
        want = fn(x)
    finally:
        undo()
    torch.testing.assert_close(got, want, rtol=0, atol=0)
    assert not hpu_ops._STATE["calls"]["unfold_decomp"]
    assert "aten.clone.default" not in _op_names(nodes), _op_names(nodes)
    assert "aten.as_strided.default" in _op_names(nodes)


def test_unfold_decomp_other_device_untouched():
    table = _as_strided_table()
    undo = habana_patch.patch_as_strided_decompositions(table=table, device_type="hpu")
    hpu_ops._STATE["calls"].clear()
    try:
        x = torch.randn(4, 32, 16)
        fn = lambda a: torch.nan_to_num(a.transpose(1, 2).unfold(-1, 8, 8), nan=0.0)  # noqa: E731
        nodes = []
        cf = _compile_with_table(fn, table, nodes)
        got = cf(x)
        want = fn(x)
    finally:
        undo()
    torch.testing.assert_close(got, want, rtol=0, atol=0)
    assert not hpu_ops._STATE["calls"]["unfold_decomp"]
    assert "aten.clone.default" not in _op_names(nodes), _op_names(nodes)


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("name,fn", [
    ("trans_unfold_nonov_nan2num", lambda x: torch.nan_to_num(x.transpose(1, 2).unfold(-1, 8, 8), nan=0.0)),
    ("trans_unfold_ov_nan2num", lambda x: torch.nan_to_num(x.transpose(1, 2).unfold(-1, 8, 4), nan=0.0)),
    ("slice_step2_unfold", lambda x: torch.relu(x[:, ::2, :].unfold(1, 4, 2))),
    ("narrow_unfold", lambda x: torch.relu(x[:, :, :12].unfold(1, 8, 4))),
    ("offset_slice_unfold", lambda x: torch.relu(x[2:].unfold(-1, 8, 8))),
    ("unfold_twice", lambda x: torch.relu(x.unfold(1, 8, 8).unfold(2, 4, 4))),
    ("trans_diagonal", lambda x: torch.relu(x.transpose(1, 2).diagonal(0, 1, 2))),
    ("offset_diagonal", lambda x: torch.relu(x[2:].diagonal(0, 1, 2))),
])
def test_as_strided_decomps_clone_noncontig_and_match_eager(name, fn, dtype, dynamic):
    torch.manual_seed(0)
    x = torch.randn(6, 32, 16, dtype=dtype)
    x[0, 3, 2] = float("nan")
    x[1, 5, 4] = float("inf")
    x[2, 7, 6] = float("-inf")
    table = _as_strided_table()
    undo = habana_patch.patch_as_strided_decompositions(table=table, device_type="cpu")
    hpu_ops._STATE["calls"].clear()
    try:
        nodes = []
        cf = _compile_with_table(fn, table, nodes, dynamic=dynamic)
        want = fn(x)
        got = cf(x)
    finally:
        undo()
    torch.testing.assert_close(got, want, rtol=0, atol=0, equal_nan=True)
    ops = _op_names(nodes)
    assert "aten.clone.default" in ops, f"{name}: expected clone before as_strided, got {ops}"
    assert "aten.as_strided.default" in ops
    expected_key = "diagonal_decomp" if "diagonal" in name else "unfold_decomp"
    assert hpu_ops._STATE["calls"][expected_key] >= 1


@pytest.mark.parametrize("step", [8, 4])
def test_unfold_decomp_grad_forward_and_backward(step):
    torch.manual_seed(0)
    x = torch.randn(4, 32, 16, dtype=torch.float64)
    fn = lambda a: torch.nan_to_num(a.transpose(1, 2).unfold(-1, 8, step), nan=0.0)  # noqa: E731
    table = _as_strided_table()
    undo = habana_patch.patch_as_strided_decompositions(table=table, device_type="cpu")
    try:
        nodes = []
        cf = _compile_with_table(fn, table, nodes)
        x_ref = x.clone().requires_grad_(True)
        y_ref = fn(x_ref)
        y_ref.square().sum().backward()

        x_got = x.clone().requires_grad_(True)
        y_got = cf(x_got)
        y_got.square().sum().backward()
    finally:
        undo()
    torch.testing.assert_close(y_got, y_ref, rtol=0, atol=1e-12)
    torch.testing.assert_close(x_got.grad, x_ref.grad, rtol=0, atol=1e-12)


def test_unfold_decomp_preserves_inplace_functionalization_and_leaf_view():
    torch.manual_seed(0)
    x = torch.randn(4, 32, 16)
    x[0, 2, 1] = float("nan")
    table = _as_strided_table()
    undo = habana_patch.patch_as_strided_decompositions(table=table, device_type="cpu")
    try:
        def inplace_fn(a):
            y = a.clone()
            y.transpose(1, 2).unfold(-1, 8, 8).nan_to_num_(nan=0.0)
            return y

        leaf_fn = lambda a: a.transpose(1, 2).unfold(-1, 8, 8)  # noqa: E731

        nodes_in, nodes_leaf = [], []
        cf_in = _compile_with_table(inplace_fn, table, nodes_in)
        cf_leaf = _compile_with_table(leaf_fn, table, nodes_leaf)
        got_in = cf_in(x)
        want_in = inplace_fn(x)
        got_leaf = cf_leaf(x)
        want_leaf = leaf_fn(x)
    finally:
        undo()
    torch.testing.assert_close(got_in, want_in, rtol=0, atol=0)
    torch.testing.assert_close(got_leaf, want_leaf, rtol=0, atol=0, equal_nan=True)
    assert got_leaf.data_ptr() == x.data_ptr()


@pytest.mark.parametrize("ctx", [torch.no_grad, torch.inference_mode])
def test_unfold_decomp_no_grad_and_inference_mode(ctx):
    torch.manual_seed(0)
    x = torch.randn(4, 32, 16)
    fn = lambda a: torch.nan_to_num(a.transpose(1, 2).unfold(-1, 8, 8), nan=0.0)  # noqa: E731
    table = _as_strided_table()
    undo = habana_patch.patch_as_strided_decompositions(table=table, device_type="cpu")
    try:
        nodes = []
        cf = _compile_with_table(fn, table, nodes)
        with ctx():
            want = fn(x)
            got = cf(x)
    finally:
        undo()
    torch.testing.assert_close(got, want, rtol=0, atol=0)
    assert "aten.clone.default" in _op_names(nodes)
