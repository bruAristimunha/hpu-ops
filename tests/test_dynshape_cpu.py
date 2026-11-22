"""CPU tests of hpu_ops.dynshape (#42), "cpu" standing in for "hpu": under torch.compile with symbolic sizes,
``aten.expand.default`` resolves ``-1`` dims and ``int64`` ``aten.constant_pad_nd.default`` rewrites to
``narrow`` + ``torch.full`` + ``torch.cat``; static ``expand``, eager calls, non-int64 dtypes, and other devices
keep torch's own path."""
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import hpu_ops
from hpu_ops import dynshape

from test_cpu import assert_same, compiled, cpu_as_hpu, ops, train_step

EXPAND_OP = "aten.expand.default"
PAD_OP = "aten.constant_pad_nd.default"


def _expand_has_minus_one(nodes):
    return [n for n in nodes if str(n.target) == EXPAND_OP
            and any(isinstance(a, int) and a == -1 for a in n.args[1])]


class _ParamExpandCat(nn.Module):
    """Minimal replica of NeuroRVQ / DIVER1 / LaBraM token expansion: param[1, 1, D].expand(B * K, -1, -1) + cat."""

    def __init__(self, k=4, d=16):
        super().__init__()
        self.k = k
        self.tok = nn.Parameter(torch.randn(1, 1, d, dtype=torch.float64))

    def forward(self, x):
        b = x.shape[0]
        expanded = self.tok.expand(b * self.k, -1, -1)
        return torch.cat([expanded, x.reshape(b * self.k, 9, -1)], dim=1)


@pytest.mark.parametrize("order", [(16, 128), (128, 16)])
@pytest.mark.parametrize("mode", ["train", "no_grad", "inference_mode"])
def test_expand_param_cat_modes(order, mode):
    torch.manual_seed(0)
    m = _ParamExpandCat()
    nodes = []
    with cpu_as_hpu(dynshape.install_expand) as calls:
        g = compiled(m, nodes)
        for batch in order:
            x = torch.randn(batch, 4, 9, 16, dtype=torch.float64)
            if mode == "train":
                m.zero_grad()
                got = train_step(g, x) + [m.tok.grad.clone()]
                m.zero_grad()
                want = train_step(m, x) + [m.tok.grad.clone()]
                assert_same(got, want, tol=1e-12)
            else:
                ctx = torch.no_grad() if mode == "no_grad" else torch.inference_mode()
                with ctx:
                    assert_same([g(x)], [m(x)], tol=1e-12)
    assert calls["expand"]
    # First batch compiles statically (keeps -1); second batch recompiles symbolically (no -1 in expand).
    expand_nodes = [n for n in nodes if str(n.target) == EXPAND_OP]
    assert len(expand_nodes) >= 2
    assert _expand_has_minus_one(expand_nodes[:1])
    assert not _expand_has_minus_one(expand_nodes[1:])


def test_expand_sym_self_expand():
    """SymSelfExpand: ``self`` itself has a symbolic dim while ``size`` contains ``-1`` literals."""
    f = lambda x: x.expand(-1, 4, -1)  # noqa: E731
    nodes = []
    with cpu_as_hpu(dynshape.install_expand) as calls:
        g = compiled(f, nodes, dynamic=True, fullgraph=True)
        for batch in (16, 128):
            x = torch.randn(batch, 1, 12, dtype=torch.float64)
            assert_same(train_step(g, x), train_step(f, x), tol=1e-12)
    assert calls["expand"]
    assert EXPAND_OP in ops(nodes) and not _expand_has_minus_one(nodes)


def test_expand_static_keeps_minus_one():
    m = _ParamExpandCat()
    nodes = []
    with cpu_as_hpu(dynshape.install_expand) as calls:
        g = compiled(m, nodes, dynamic=False)
        for batch in (16, 128):
            x = torch.randn(batch, 4, 9, 16, dtype=torch.float64)
            assert_same(train_step(g, x), train_step(m, x), tol=1e-12)
    assert not calls["expand"]
    assert _expand_has_minus_one(nodes)


def test_expand_zero_dim_tensor():
    def f(s, x):
        return s.expand(x.shape[0], 3)

    s = torch.tensor(2.5, dtype=torch.float64)
    nodes = []
    with cpu_as_hpu(dynshape.install_expand) as calls:
        g = compiled(f, nodes, dynamic=True)
        for batch in (4, 8):
            x = torch.randn(batch, 2, dtype=torch.float64)
            assert_same([g(s, x)], [f(s, x)], tol=1e-12)
    assert not calls["expand"]


@pytest.mark.parametrize("bad_fn", [
    lambda x: x.expand(-1),  # too few dimensions
    lambda x: x.expand(-1, x.shape[0], 1, 3),  # leading -1 at i < offset
    lambda x: x.expand(x.shape[0], -1, 5),  # non-singleton mismatch on dim 2
])
def test_expand_preserves_aten_errors(bad_fn):
    x = torch.randn(4, 1, 3)
    with cpu_as_hpu(dynshape.install_expand) as calls, pytest.raises(Exception):
        compiled(bad_fn, [], dynamic=True)(x)
    assert not calls["expand"]


def test_expand_zero_dim_error_preserved():
    s = torch.tensor(2.5, dtype=torch.float64)
    x = torch.randn(4, 2, dtype=torch.float64)
    with cpu_as_hpu(dynshape.install_expand) as calls, pytest.raises(Exception):
        compiled(lambda a, b: a.expand(b.shape[0], -1), [], dynamic=True)(s, x)
    assert not calls["expand"]


def test_expand_eager_and_other_device_untouched():
    m = _ParamExpandCat()
    x = torch.randn(4, 4, 9, 16, dtype=torch.float64)
    with cpu_as_hpu(dynshape.install_expand) as calls:
        m(x)
        assert not calls["expand"]
    undo = dynshape.install_expand("xpu")
    nodes = []
    try:
        hpu_ops._STATE["calls"].clear()
        compiled(m, nodes, dynamic=True)(x)
    finally:
        undo()
    assert not hpu_ops._STATE["calls"]["expand"]
    assert _expand_has_minus_one(nodes)


# ---- constant_pad_nd on int64


@pytest.mark.parametrize("shape, pad, value", [
    ((9,), (1, 0), 0.0),  # NeuroRVQ spatial_ix shape + float value
    ((5,), (2, 3), -1),
    ((5,), (1, 2), -1.9),  # float truncation toward zero matching ATen
    ((2, 3), (1, 2, 3, 4), 7),  # 2D
    ((2, 3, 4), (1, 0, 0, 2, 1, 1), -5),  # 3D
    ((5,), (-2, -1), 0),  # crop both ends
    ((5,), (-4, 4), 9),  # mixed crop left + pad right
    ((3, 5), (2, -1, -1, 3), 4),  # mixed 2D pad and crop
    ((5,), (), 0),  # empty pad
    ((5,), (0, 0), 0),  # zero pad
    ((0,), (1, 2), 3),  # 0-element 1D
    ((2, 0), (1, 0), 5),  # 0-element 2D (pad empty dim)
    ((0, 2), (1, 0), 5),  # 0-element 2D (pad non-empty dim)
])
def test_pad_int64_matches_aten(shape, pad, value):
    x = torch.arange(int(torch.tensor(shape).prod().item()), dtype=torch.int64).reshape(shape)
    want = torch.ops.aten.constant_pad_nd.default(x, pad, value)
    nodes = []
    f = lambda t: torch.ops.aten.constant_pad_nd.default(t, pad, value)  # noqa: E731
    with cpu_as_hpu(dynshape.install_pad_int64) as calls:
        got = compiled(f, nodes)(x)
    assert calls["constant_pad_int64"]
    assert PAD_OP not in ops(nodes), ops(nodes)
    assert torch.equal(got, want)
    assert got.dtype == torch.int64
    if x.numel() > 0:
        assert got.data_ptr() != x.data_ptr()


def test_pad_int64_dynamic_batch():
    f = lambda x: F.pad(x, (1, 0), value=0.0)  # noqa: E731
    nodes = []
    with cpu_as_hpu(dynshape.install_pad_int64) as calls:
        g = compiled(f, nodes)
        for batch in (16, 128):
            x = torch.arange(batch * 9, dtype=torch.int64).reshape(batch, 9)
            assert torch.equal(g(x), f(x))
    assert calls["constant_pad_int64"]
    assert PAD_OP not in ops(nodes), ops(nodes)


def test_pad_int64_noncontiguous_zero_pad():
    x = torch.arange(6, dtype=torch.int64).reshape(2, 3).t()
    want = torch.ops.aten.constant_pad_nd.default(x, (0, 0), 0)
    nodes = []
    with cpu_as_hpu(dynshape.install_pad_int64) as calls:
        got = compiled(lambda t: F.pad(t, (0, 0), value=0), nodes)(x)
    assert calls["constant_pad_int64"] and PAD_OP not in ops(nodes)
    assert torch.equal(got, want) and got.stride() == want.stride()


@pytest.mark.parametrize("shape, pad", [
    ((2, 3), (1,)),  # odd pad length
    ((2, 3), (1, 0, 1, 0, 1, 0)),  # too many dims
    ((5,), (-6, 10)),  # negative crop past dim length
    ((5,), (-3, -3)),  # resulting size negative
])
def test_pad_int64_preserves_aten_errors(shape, pad):
    x = torch.zeros(shape, dtype=torch.int64)
    with cpu_as_hpu(dynshape.install_pad_int64) as calls, pytest.raises(Exception):
        compiled(lambda t: torch.ops.aten.constant_pad_nd.default(t, pad, 0), [])(x)
    assert not calls["constant_pad_int64"]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.int32])
def test_pad_non_int64_untouched(dtype):
    x = torch.ones(3, 4, dtype=dtype)
    nodes = []
    with cpu_as_hpu(dynshape.install_pad_int64) as calls:
        got = compiled(lambda t: F.pad(t, (1, 0), value=0.0), nodes)(x)
    assert not calls["constant_pad_int64"]
    assert PAD_OP in ops(nodes)
    assert torch.equal(got, F.pad(x, (1, 0), value=0.0))


def test_pad_eager_and_other_device_untouched():
    x = torch.arange(5, dtype=torch.int64)
    with cpu_as_hpu(dynshape.install_pad_int64) as calls:
        F.pad(x, (1, 0), value=0.0)
        assert not calls["constant_pad_int64"]
    undo = dynshape.install_pad_int64("xpu")
    nodes = []
    try:
        hpu_ops._STATE["calls"].clear()
        compiled(lambda t: F.pad(t, (1, 0), value=0.0), nodes)(x)
    finally:
        undo()
    assert not hpu_ops._STATE["calls"]["constant_pad_int64"]
    assert PAD_OP in ops(nodes)


def test_undo_and_fixes_registration():
    assert dynshape.install_expand in hpu_ops.FIXES
    assert dynshape.install_pad_int64 in hpu_ops.FIXES
    for op, install in (
        (torch.ops.aten.expand.default, dynshape.install_expand),
        (torch.ops.aten.constant_pad_nd.default, dynshape.install_pad_int64),
    ):
        before = dict(op.py_kernels)
        install("cpu")()
        assert op.py_kernels == before
