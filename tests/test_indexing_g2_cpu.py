"""CPU unit tests for G2 (host-index LRU cache, host-skip, weakref liveness, inference_mode/vmap safety) and
AOT FunctionalTensor tracing in ``hpu_ops._dispatch.tracing`` / ``hpu_ops.indexing``."""
import gc
import threading

import pytest
import torch
from test_cpu import (  # noqa: F401
    assert_same,
    compiled,
    cpu_as_hpu,
    cpu_indices_off_device,
    ops,
    train_step,
)
from torch.fx.experimental.proxy_tensor import make_fx
from torch.utils._python_dispatch import TorchDispatchMode

import hpu_ops
from hpu_ops import _dispatch, _strict, indexing

aten = torch.ops.aten


@pytest.fixture(autouse=True)
def clean_cache_and_strict():
    indexing._clear_cache()
    hpu_ops.strict_reset()
    yield
    indexing._clear_cache()
    hpu_ops.strict_reset()


class _RecordOps(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.ops = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.ops.append(str(func))
        return func(*args, **(kwargs or {}))


def test_g2_cache_hit_skips_host_check_ops_and_route(monkeypatch):
    """Repeated ``x[..., ix[0], ix[1]]`` with fresh temporary views hits cache on 2nd+ calls: 0 ``_in_range``,
    0 ``_strict.route``, and 0 host range/linearization/copy ops."""
    ix = torch.triu_indices(6, 6)
    x = torch.randn(4, 6, 6)
    want = x[..., ix[0], ix[1]]

    in_range_calls = 0
    orig_in_range = indexing._in_range

    def count_in_range(t, size):
        nonlocal in_range_calls
        in_range_calls += 1
        return orig_in_range(t, size)

    monkeypatch.setattr(indexing, "_in_range", count_in_range)

    with cpu_as_hpu(indexing.install_index) as calls:
        with _RecordOps() as rec1:
            got1 = x[..., ix[0], ix[1]]
        assert in_range_calls == 2
        assert any("ge.Scalar" in op for op in rec1.ops)

        in_range_calls = 0
        with _RecordOps() as rec2:
            got2 = x[..., ix[0], ix[1]]
        with _RecordOps() as rec3:
            got3 = x[..., ix[0], ix[1]]

    assert torch.equal(got1, want) and torch.equal(got2, want) and torch.equal(got3, want)
    assert calls["index"] == 3
    assert in_range_calls == 0
    host_ops = {"aten.ge.Scalar", "aten.lt.Scalar", "aten.bitwise_and.Tensor", "aten.all.default",
                "aten.add.Tensor", "aten.mul.Tensor", "aten.where.self", "aten._to_copy.default"}
    assert not (set(rec2.ops) & host_ops), rec2.ops
    assert not (set(rec3.ops) & host_ops), rec3.ops

    # On a non-CPU device ("meta"), verify _strict.route(HOST_INDEX, ...) and _to_copy fire on miss and are 0 on hit
    route_calls = 0
    orig_route = _strict.route

    def count_route(source, *a, **kw):
        nonlocal route_calls
        route_calls += 1
        return orig_route(source, *a, **kw)

    monkeypatch.setattr(_strict, "route", count_route)
    xm = torch.empty(4, 6, 6, device="meta")
    meta_kernel = indexing._index_kernel("meta", torch._C.DispatchKey.Meta)
    with hpu_ops.strict("report"):
        with _RecordOps() as rec_m1:
            _ = meta_kernel(xm, [None, ix[0], ix[1]])
        assert route_calls == 4  # _host_index_route + 2 _in_range + index_select_flat route on miss
        assert "aten._to_copy.default" in rec_m1.ops
        route_calls = 0
        with _RecordOps() as rec_m2:
            _ = meta_kernel(xm, [None, ix[0], ix[1]])
            _ = meta_kernel(xm, [None, ix[0], ix[1]])
        assert route_calls == 0
        assert not (set(rec_m2.ops) & host_ops), rec_m2.ops
        assert len(indexing._INDEX_CACHE) == 1


def test_g2_cache_batch_vs_dim_size_and_oob_not_cached(monkeypatch):
    ix = torch.triu_indices(5, 5)
    x32 = torch.randn(32, 5, 5)
    x5 = torch.randn(5, 5, 5)
    x4d = torch.randn(2, 3, 5, 5)
    x_diff = torch.randn(4, 6, 6)

    in_range_calls = 0
    orig_in_range = indexing._in_range

    def count_in_range(t, size):
        nonlocal in_range_calls
        in_range_calls += 1
        return orig_in_range(t, size)

    monkeypatch.setattr(indexing, "_in_range", count_in_range)

    with cpu_as_hpu(indexing.install_index):
        assert torch.equal(x32[..., ix[0], ix[1]], indexing.index_select_flat(x32, [None, ix[0], ix[1]]))
        # Reset and measure via actual op dispatch
    indexing._clear_cache()
    in_range_calls = 0

    with cpu_as_hpu(indexing.install_index):
        got32 = x32[..., ix[0], ix[1]]
        assert in_range_calls == 2
        in_range_calls = 0
        got5 = x5[..., ix[0], ix[1]]
        got4d = x4d[..., ix[0], ix[1]]
        assert in_range_calls == 0  # batch dims changed -> still cache hit!

        # Different indexed dim size (6, 6) -> cache miss and checks range
        got_diff = x_diff[..., ix[0], ix[1]]
        assert in_range_calls == 2

        # OOB index -> raises IndexError and is NOT cached
        bad = torch.tensor([0, 6])
        with pytest.raises(IndexError):
            _ = x_diff[:, bad, torch.tensor([0, 1])]
        assert len(indexing._INDEX_CACHE) == 2  # only (5, 5) and (6, 6)

    assert torch.equal(got32, x32[..., ix[0], ix[1]])
    assert torch.equal(got5, x5[..., ix[0], ix[1]])
    assert torch.equal(got4d, x4d[..., ix[0], ix[1]])
    assert torch.equal(got_diff, x_diff[..., ix[0], ix[1]])


def test_g2_cache_version_mutation_and_view_metadata():
    x = torch.randn(3, 5, 5)
    ix = torch.tensor([[0, 1, 2], [1, 2, 3]])
    with cpu_as_hpu(indexing.install_index):
        got1 = x[:, ix[0], ix[1]]
        ix[0, 1] = 4
        got2 = x[:, ix[0], ix[1]]
        ix.add_(0)  # bumps _version
        got3 = x[:, ix[0], ix[1]]
        assert len(indexing._INDEX_CACHE) >= 2
    assert torch.equal(got1, x[:, torch.tensor([0, 1, 2]), torch.tensor([1, 2, 3])])
    assert torch.equal(got2, x[:, ix[0], ix[1]])
    assert torch.equal(got3, x[:, ix[0], ix[1]])

    # Different views of the same storage (shape, stride, dtype) must miss and produce distinct results
    indexing._clear_cache()
    base = torch.arange(8, dtype=torch.int64)
    v_a = base[:4]
    v_b = base[:6]
    v_strided = base[:8:2]
    v_i32 = base.view(torch.int32)[:4]
    x8 = torch.randn(2, 8)
    with cpu_as_hpu(indexing.install_index):
        ga = x8[:, v_a]
        gb = x8[:, v_b]
        gs = x8[:, v_strided]
        gi = x8[:, v_i32]
        assert len(indexing._INDEX_CACHE) == 4
    assert torch.equal(ga, x8[:, v_a])
    assert torch.equal(gb, x8[:, v_b])
    assert torch.equal(gs, x8[:, v_strided])
    assert torch.equal(gi, x8[:, v_i32])


def test_g2_cache_weakref_del_and_data_ptr_reuse_safety():
    x = torch.randn(4, 6)
    idx1 = torch.tensor([0, 1, 2, 3])
    with cpu_as_hpu(indexing.install_index):
        got1 = x[:, idx1]
        assert len(indexing._INDEX_CACHE) == 1
        del idx1
        gc.collect()
        assert len(indexing._DEAD_KEYS) >= 1

        # Repeatedly allocate/free same-size tensors with different values to force allocator address reuse
        for val in range(5):
            idx2 = torch.tensor([5 - val, 4 - (val % 4), 1, 0])
            got2 = x[:, idx2]
            assert torch.equal(got2, x.index_select(1, idx2))
            del idx2
            gc.collect()
    assert torch.equal(got1, x[:, torch.tensor([0, 1, 2, 3])])


def test_g2_inference_mode_to_train_backward_and_expanded():
    """First call in ``inference_mode()`` with a normal index tensor must NOT poison subsequent training
    ``.backward()`` (no 'Inference tensors cannot be saved for backward')."""
    ix = torch.triu_indices(5, 5)
    x_inf = torch.randn(4, 5, 5)
    x_tr = torch.randn(4, 5, 5)
    want = train_step(lambda t: t[..., ix[0], ix[1]], x_tr)
    backend = indexing._index_kernel("cpu", torch._C.DispatchKey.CPU)

    with cpu_as_hpu(indexing.install_index):
        undo_be = _dispatch.override_kernels("cpu", [(aten.index.Tensor, backend)])
        try:
            with torch.inference_mode():
                inf_out = x_inf[..., ix[0], ix[1]]
            assert len(indexing._INDEX_CACHE) == 1
            # Verify cached lin_flat is not an inference tensor
            cached_lin_flat = next(iter(indexing._INDEX_CACHE.values()))[0]
            assert not cached_lin_flat.is_inference()
            got = train_step(lambda t: t[..., ix[0], ix[1]], x_tr)
        finally:
            undo_be()
    assert torch.equal(inf_out, x_inf[..., ix[0], ix[1]])
    assert_same(got, want, tol=0)

    # Also test stride-0 expanded tensor across inference_mode -> train
    indexing._clear_cache()
    e_inf = torch.randn(3, 1, 5).expand(3, 5, 5)
    e_tr = torch.randn(3, 1, 5, requires_grad=True)
    with cpu_as_hpu(indexing.install_index):
        undo_be = _dispatch.override_kernels("cpu", [(aten.index.Tensor, backend)])
        try:
            with torch.inference_mode():
                _ = e_inf[..., ix[0], ix[1]]
            e_exp = e_tr.expand(3, 5, 5)
            out = e_exp[..., ix[0], ix[1]]
            out.square().sum().backward()
            assert e_tr.grad is not None
        finally:
            undo_be()


def test_g2_is_inference_index_tensor_not_cached_and_safe_on_mutation():
    x = torch.randn(3, 5, 5)
    backend = indexing._index_kernel("cpu", torch._C.DispatchKey.CPU)
    with cpu_as_hpu(indexing.install_index):
        undo_be = _dispatch.override_kernels("cpu", [(aten.index.Tensor, backend)])
        try:
            with torch.inference_mode():
                ix = torch.triu_indices(5, 5)
                assert ix.is_inference()
                got1 = x[..., ix[0], ix[1]]
                assert len(indexing._INDEX_CACHE) == 0
                ix[0, 0] = 4
                got2 = x[..., ix[0], ix[1]]
                assert len(indexing._INDEX_CACHE) == 0
        finally:
            undo_be()
    with torch.inference_mode():
        want2 = x[..., ix[0], ix[1]]
    assert torch.equal(got2, want2)
    assert got1.shape == (3, 15)


def test_g2_vmap_grad_jvp_do_not_crash():
    x = torch.randn(3, 4, 5)
    idx_2d = torch.tensor([[0, 2, 1], [3, 1, 0]])
    want_vmap = torch.vmap(lambda i: x[:, i])(idx_2d)
    with cpu_as_hpu(indexing.install_index):
        got_vmap = torch.vmap(lambda i: x[:, i])(idx_2d)

        # Direct BatchedTensor inside vmap is rejected by _cacheable_cpu and _cache_lookup without reading data_ptr()
        def check_batched(i):
            assert not indexing._cacheable_cpu(i)
            assert indexing._cache_lookup(x, [None, i]) is None
            return x[:, i]

        _ = torch.vmap(check_batched)(idx_2d)
    assert torch.equal(got_vmap, want_vmap)

    # vmap(grad) with expanded input
    ix = torch.triu_indices(4, 4)
    leaf = torch.randn(2, 3, 1, 4)

    def f_expanded(t):
        return t.expand(3, 4, 4)[..., ix[0], ix[1]].square().sum()

    want_vg = torch.vmap(torch.func.grad(f_expanded))(leaf)
    with cpu_as_hpu(indexing.install_index):
        got_vg = torch.vmap(torch.func.grad(f_expanded))(leaf)
    torch.testing.assert_close(got_vg, want_vg, atol=1e-12, rtol=0)

    # jvp
    primal = torch.randn(3, 4, 4)
    tangent = torch.randn(3, 4, 4)
    want_jvp = torch.autograd.functional.jvp(lambda t: t[..., ix[0], ix[1]], (primal,), (tangent,))
    with cpu_as_hpu(indexing.install_index):
        got_jvp = torch.autograd.functional.jvp(lambda t: t[..., ix[0], ix[1]], (primal,), (tangent,))
    assert_same(got_jvp, want_jvp, tol=0)


@pytest.mark.usefixtures("cpu_indices_off_device")
@pytest.mark.parametrize("dynamic", [False, True])
def test_aot_functional_tensor_tracing_zero_cache_and_strict_clean(dynamic):
    """Under ``torch.compile`` (Dynamo + AOT FunctionalTensor), ``_dispatch.tracing`` returns True:
    - ``_INDEX_CACHE`` stays empty (0 entries)
    - FX graph contains ``aten.index_select.default`` and zero CPU range-check ops
    - ``hpu_ops.strict("raise")`` does NOT fail compile."""
    ix = torch.triu_indices(5, 5)
    x = torch.randn(4, 5, 5)

    def f(t):
        return t[..., ix[0], ix[1]]

    want = train_step(f, x)
    nodes = []
    with cpu_as_hpu(indexing.install_index):
        with hpu_ops.strict("raise", per_op=False):
            g = compiled(f, nodes, fullgraph=True, dynamic=dynamic)
            got = train_step(g, x)
        assert len(indexing._INDEX_CACHE) == 0
    assert_same(got, want, tol=0)
    node_ops = ops(nodes)
    assert "aten.index_select.default" in node_ops
    assert not (node_ops & {"aten.index.Tensor", "aten.ge.Tensor", "aten.lt.Tensor", "aten.all.default"})


def test_dispatch_tracing_predicates():
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch._subclasses.functional_tensor import (
        FunctionalTensor,
        FunctionalTensorMode,
    )

    t = torch.randn(3, 3)
    assert _dispatch.tracing(t) is False
    with FakeTensorMode():
        ft = torch.empty(3, 3)
        assert _dispatch.tracing(ft) is True
    with FunctionalTensorMode():
        fnt = FunctionalTensor.to_functional(t)
        assert _dispatch.tracing(fnt) is True

    # torch.func.functionalize sets torch._is_functional_tensor(t) == True
    seen_functional = []

    def fn(a):
        seen_functional.append((torch._is_functional_tensor(a), _dispatch.tracing(a)))
        return a + 1

    torch.func.functionalize(fn)(t)
    assert seen_functional == [(True, True)]

    # make_fx sets PROXY dispatch mode on real tensors
    seen_proxy = []

    def fn_proxy(a):
        seen_proxy.append(_dispatch.tracing(a))
        return a + 1

    make_fx(fn_proxy)(t)
    assert seen_proxy == [True]


def test_g2_lru_byte_bounds_oversized_threads_and_undo(monkeypatch):
    x = torch.randn(2, 10)
    # 1. Exceed INDEX_CACHE_SIZE -> touched oldest stays, untouched oldest evicted
    monkeypatch.setattr(indexing, "INDEX_CACHE_SIZE", 4)
    idxs = [torch.tensor([i, (i + 1) % 10]) for i in range(5)]
    with cpu_as_hpu(indexing.install_index):
        for idx in idxs[:4]:
            _ = x[:, idx]
        assert len(indexing._INDEX_CACHE) == 4
        # Touch idxs[0] so idxs[1] becomes LRU
        _ = x[:, idxs[0]]
        _ = x[:, idxs[4]]
        assert len(indexing._INDEX_CACHE) == 4
        cached_ptrs = {k[-1][0][0] for k in indexing._INDEX_CACHE}
        assert idxs[0].data_ptr() in cached_ptrs
        assert idxs[1].data_ptr() not in cached_ptrs

    # 2. Byte bounds
    indexing._clear_cache()
    monkeypatch.setattr(indexing, "INDEX_CACHE_SIZE", 128)
    monkeypatch.setattr(indexing, "INDEX_CACHE_BYTES", 64)  # 4 int64 indices = 32 bytes each -> max 2 entries
    idxs_b = [torch.tensor([0, 1, 2, i]) for i in range(5)]
    with cpu_as_hpu(indexing.install_index):
        for idx in idxs_b:
            _ = x[:, idx]
            assert indexing._INDEX_BYTES <= 64
        assert len(indexing._INDEX_CACHE) == 2

    # 3. Oversized entry skipped
    indexing._clear_cache()
    monkeypatch.setattr(indexing, "INDEX_CACHE_ENTRY_MAX_BYTES", 16)
    big_idx = torch.tensor([0, 1, 2, 3])  # 32 bytes > 16
    with cpu_as_hpu(indexing.install_index):
        got = x[:, big_idx]
        assert len(indexing._INDEX_CACHE) == 0
    assert torch.equal(got, x[:, big_idx])

    # 4. 4-thread concurrent stress test
    monkeypatch.undo()
    indexing._clear_cache()
    xt = torch.randn(4, 16, 16)
    shared_ix = torch.triu_indices(16, 16)
    errors = []

    def worker(tid):
        try:
            local_ix = torch.tensor([[tid % 16, (tid + 1) % 16], [1, 2]])
            for step in range(20):
                out_s = xt[..., shared_ix[0], shared_ix[1]]
                assert out_s.shape == (4, 136)
                out_l = xt[:, local_ix[0], local_ix[1]]
                assert out_l.shape == (4, 2)
                if step % 5 == 0:
                    local_ix[0, 0] = (local_ix[0, 0] + 1) % 16
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    with cpu_as_hpu(indexing.install_index):
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
        assert len(indexing._INDEX_CACHE) > 0
    # undo() from cpu_as_hpu clears _INDEX_CACHE and _DEAD_KEYS
    assert len(indexing._INDEX_CACHE) == 0
    assert len(indexing._DEAD_KEYS) == 0
    assert indexing._INDEX_BYTES == 0


def test_mask_fill_calls_value_once(monkeypatch):
    """``install_mask_fill`` converts cross-dtype value once per ``index_put_``, not twice."""
    calls_to_value = 0
    orig_value = indexing._value

    def count_value(self, values):
        nonlocal calls_to_value
        calls_to_value += 1
        return orig_value(self, values)

    monkeypatch.setattr(indexing, "_value", count_value)
    x = torch.zeros(3, 4, dtype=torch.float32)
    mask = torch.tensor([True, False, True, False])
    val = torch.tensor(2.5, dtype=torch.float64)
    with cpu_as_hpu(indexing.install_mask_fill) as calls:
        x[:, mask] = val
    assert calls["mask_fill"] == 1
    assert calls_to_value == 1


@pytest.mark.usefixtures("cpu_indices_off_device")
def test_compile_eager_backend_inside_fwad_dual_level_indexing():
    """Bug 4: ``torch.compile(..., backend="eager")`` lowers ``fwAD.dual_level()`` to C++
    ``_enter_dual_level()`` and executes with ``_len_torch_dispatch_stack() == 0`` and
    ``fwAD._current_level == -1``; 0-arg ``forward_ad_level()`` must detect C++ dual level 0 so expanded
    indexing materializes instead of hitting ``_ExpandedSelect``."""
    import torch.autograd.forward_ad as fwAD

    ix = torch.triu_indices(4, 4)
    x = torch.randn(2, 1, 4)
    t = torch.randn_like(x)

    def run_expanded(a, ta):
        with fwAD.dual_level():
            d = fwAD.make_dual(a, ta)
            out = d.expand(2, 4, 4)[..., ix[0], ix[1]]
            return out.detach(), fwAD.unpack_dual(out).tangent

    def run_plain(a):
        return a.expand(2, 4, 4)[..., ix[0], ix[1]]

    want_p, want_t = run_expanded(x, t)
    with fwAD.dual_level():
        want_plain = run_plain(x)

    torch._dynamo.reset()
    compiled_expanded = torch.compile(run_expanded, backend="eager", fullgraph=True)
    compiled_plain = torch.compile(run_plain, backend="eager", fullgraph=True)

    with cpu_as_hpu(indexing.install_index):
        got_p, got_t = compiled_expanded(x, t)
        assert len(indexing._INDEX_CACHE) == 0
        with fwAD.dual_level():
            got_plain = compiled_plain(x)
        assert len(indexing._INDEX_CACHE) == 0

    torch.testing.assert_close(got_p, want_p, rtol=0, atol=0)
    torch.testing.assert_close(got_t, want_t, rtol=0, atol=0)
    torch.testing.assert_close(got_plain, want_plain, rtol=0, atol=0)
