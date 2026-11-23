"""CPU tests of hpu_ops.small, the bincount rewrite (issue #21): "cpu" stands in for "hpu"; every result is compared
with ATen's own CPU kernel, run with the rewrite undone."""
import contextlib

import pytest
import torch

import hpu_ops
from hpu_ops import small

inf, nan = float("inf"), float("nan")


@contextlib.contextmanager
def cpu_as_hpu(install):
    undo = install("cpu")
    hpu_ops._STATE["calls"].clear()
    try:
        yield hpu_ops._STATE["calls"]
    finally:
        undo()


def same(a, b, exact=False):
    assert a.dtype == b.dtype and a.shape == b.shape, (a.dtype, b.dtype, a.shape, b.shape)
    if exact:
        assert torch.equal(a, b), (a, b)
    else:
        torch.testing.assert_close(a, b, equal_nan=True, rtol=2e-6, atol=2e-6)


# ---- cumsum (measured, not rewritten) -----------------------------------------------------------------------------

def test_triangular_matmul_cumsum_breaks_on_inf_so_cumsum_stays_native():
    """Why cumsum keeps Habana's kernel: the fast rewrite (x @ triu(ones)) multiplies inf by 0."""
    x = torch.tensor([[1.0, inf]])
    assert torch.isnan((x @ torch.ones(2, 2).triu())[0, 0])
    assert torch.equal(torch.cumsum(x, -1), torch.tensor([[1.0, inf]]))
    with cpu_as_hpu(hpu_ops._install) as calls:
        torch.cumsum(torch.randn(64, 22, 22), -1)
        torch.logsumexp(torch.randn(64, 22, 22), -1)
    # cumsum and logsumexp keep Habana's kernels (measured, see the README); logsumexp's inner amax goes through
    # extrema's NaN fix, which is not a rewrite of logsumexp
    assert not {"cumsum", "logsumexp"} & set(calls), calls


# ---- bincount -------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("n,bins,dtype,minlength", [(1 << 16, 64, torch.int64, 0), (1 << 17, 256, torch.int64, 0),
                                                    (409600, 3, torch.int32, 0), (1 << 16, 1, torch.int64, 0),
                                                    (1 << 16, 10, torch.uint8, 0), (1 << 16, 10, torch.int16, 200),
                                                    (1 << 16, 64, torch.int8, 256), (1 << 17, 257, torch.int64, 0),
                                                    (1 << 17, 513, torch.int64, 0), (1 << 17, 1000, torch.int64, 0),
                                                    (1 << 17, 4096, torch.int32, 0), (1 << 17, 10, torch.int64, 4000)])
def test_bincount_large_inputs_match_aten_exactly(n, bins, dtype, minlength):
    torch.manual_seed(0)
    x = torch.randint(0, bins, (n,)).to(dtype)
    x[: n // 2] = bins - 1  # a heavy bin: half the input in one count
    with cpu_as_hpu(small.install_bincount) as calls:
        out = torch.bincount(x, minlength=minlength)
    assert calls["bincount"] == 1
    same(out, torch.bincount(x, minlength=minlength), exact=True)


def test_bincount_chunks_count_each_boundary_value_once():
    for bins in (257, 511, 512, 513, 768, 769, 1000, 4095, 4096):
        x = torch.arange(bins).repeat((1 << 17) // bins + 1)
        with cpu_as_hpu(small.install_bincount) as calls:
            out = torch.bincount(x)
        assert calls["bincount"] == 1
        same(out, torch.bincount(x), exact=True)


def test_bincount_negative_input_raises_atens_error():
    x = torch.randint(0, 10, (1 << 16,))
    x[5] = -1
    with cpu_as_hpu(small.install_bincount), pytest.raises(RuntimeError, match="non-negative"):
        torch.bincount(x)


@pytest.mark.parametrize("x,kw", [(torch.randint(0, 10, (4096,)), {}), (torch.randint(0, 4097, (1 << 16,)), {}),
                                  (torch.randint(0, 10, (1 << 16,)), {"minlength": 5000}),
                                  (torch.randint(0, 10, (1 << 16,)), {"weights": torch.rand(1 << 16)}),
                                  (torch.randint(0, 10, (1 << 16,)), {"weights": torch.rand(1 << 16).double()}),
                                  (torch.zeros(0, dtype=torch.long), {"minlength": 4}),
                                  (torch.zeros(0, dtype=torch.long), {})])
def test_bincount_keeps_atens_kernel_outside_its_cases(x, kw):
    ref = torch.bincount(x, **kw)
    with cpu_as_hpu(small.install_bincount) as calls:
        out = torch.bincount(x, **kw)
    assert calls["bincount"] == 0
    same(out, ref, exact=True)


def test_install_undo_restores_atens_kernel():
    undo = small.install_bincount("cpu")
    assert torch._C._dispatch_has_kernel_for_dispatch_key("aten::bincount", "AutogradCPU")
    undo()
    assert not torch._C._dispatch_has_kernel_for_dispatch_key("aten::bincount", "AutogradCPU")
    hpu_ops._STATE["calls"].clear()
    torch.bincount(torch.randint(0, 4, (1 << 16,)))
    assert sum(hpu_ops._STATE["calls"].values()) == 0


# ---- adversarial pass (round 2) -------------------------------------------------------------------------------------

@pytest.mark.parametrize("x,kw,taken", [
    (torch.full((1 << 16,), 255, dtype=torch.uint8), {}, 1),  # 256 bins, every value in the last one
    (torch.full((1 << 16,), 4095), {}, 1),  # 4096 bins exactly
    (torch.full((1 << 16,), 4096), {}, 0),  # 4097 bins: after the read, Habana's kernel
    (torch.zeros(1 << 16, dtype=torch.long), {"minlength": 4096}, 1),
    (torch.randint(0, 50, (1 << 17,))[::2], {}, 1),  # a strided view
    (torch.randint(0, 50, (2, 1 << 16))[1], {}, 1),  # an offset view
    (torch.randint(0, 50, (1 << 16,)).expand(2, -1)[0], {}, 1),
])
def test_bincount_adversarial_inputs_match_aten(x, kw, taken):
    ref = torch.bincount(x, **kw)
    with cpu_as_hpu(small.install_bincount) as calls:
        out, again = torch.bincount(x, **kw), torch.bincount(x, **kw)
    assert calls["bincount"] == 2 * taken
    same(out, ref, exact=True)
    same(again, out, exact=True)


def test_bincount_int8_negative_raises_and_bounds_of_the_size_range():
    x = torch.randint(0, 10, (1 << 16,), dtype=torch.int8)
    x[-1] = -128
    with cpu_as_hpu(small.install_bincount), pytest.raises(RuntimeError, match="non-negative"):
        torch.bincount(x)
    assert small._takes(torch.zeros((1 << 24) - 1, dtype=torch.uint8))
    assert not small._takes(torch.zeros(1 << 24, dtype=torch.uint8))  # float32 counts are exact below 2**24 only
    assert not small._takes(torch.zeros((1 << 16) - 1, dtype=torch.uint8))


def test_bincount_traced_and_0d_and_deterministic_keep_atens_behaviour():
    x = torch.randint(0, 7, (1 << 16,))
    with cpu_as_hpu(small.install_bincount) as calls:
        compiled = torch.compile(lambda t: torch.bincount(t), backend="eager")(x)
        with pytest.raises(RuntimeError):
            torch.bincount(torch.tensor(3))
        torch.use_deterministic_algorithms(True)
        try:
            det = torch.bincount(x)
        finally:
            torch.use_deterministic_algorithms(False)
    same(compiled, torch.bincount(x), exact=True)
    same(det, torch.bincount(x), exact=True)
    assert calls["bincount"] >= 1


# ---- transforms and tracers decline (coordinator review) ------------------------------------------------------------

def _counts_input():
    return torch.randint(0, 9, (1 << 16,))


def test_bincount_under_jvp_and_forward_ad_keeps_atens_kernel():
    x = _counts_input()
    ref = torch.bincount(x)
    with cpu_as_hpu(small.install_bincount) as calls:
        out, tangent = torch.func.jvp(lambda t: torch.bincount(x) + 0 * t.sum(), (torch.ones(3),), (torch.ones(3),))
        with torch.autograd.forward_ad.dual_level():
            fw = torch.bincount(x)
    assert calls["bincount"] == 0
    same(out.long(), ref, exact=True)  # the 0 * t term makes the jvp output float
    same(fw, ref, exact=True)


def test_bincount_under_vmap_is_exact():
    """bincount has no vmap batching rule: torch's fallback calls it once per sample on plain tensors, outside the
    transform (no functorch layer is visible there), so the rewrite may take those calls; the result is ATen's."""
    xs = torch.stack([_counts_input(), _counts_input()])
    ref = torch.stack([torch.bincount(x, minlength=9) for x in xs])
    with cpu_as_hpu(small.install_bincount) as calls, pytest.warns(UserWarning, match="batching rule"):
        out = torch.vmap(lambda t: torch.bincount(t, minlength=9))(xs)
    assert calls["bincount"] in (0, 2)
    same(out, ref, exact=True)


@pytest.mark.parametrize("mode", ["real", "fake", "symbolic"])
def test_bincount_under_make_fx_keeps_atens_kernel(mode):
    from torch.fx.experimental.proxy_tensor import make_fx

    x = _counts_input()

    def f(t):
        return torch.bincount(t, minlength=9)

    def trace():
        try:
            return make_fx(f, tracing_mode=mode)(x), None
        except Exception as e:  # noqa: BLE001  (data-dependent output: torch's own fake tracing refuses)
            return None, type(e)

    native_gm, native_err = trace()
    with cpu_as_hpu(small.install_bincount) as calls:
        gm, err = trace()
    assert calls["bincount"] == 0
    assert err is native_err  # the same outcome as torch's own kernel (fake / symbolic: DynamicOutputShapeException)
    if gm is not None:
        assert "bincount" in str(gm.graph) and "histc" not in str(gm.graph)
        same(gm(x), torch.bincount(x, minlength=9), exact=True)


@pytest.mark.parametrize("backend", ["eager", "aot_eager"])
def test_bincount_under_compile_keeps_atens_kernel_while_tracing(backend):
    x = _counts_input()
    torch._dynamo.reset()
    with cpu_as_hpu(small.install_bincount) as calls:
        out = torch.compile(lambda t: torch.bincount(t) + 1, backend=backend, fullgraph=False)(x)
    same(out, torch.bincount(x) + 1, exact=True)
    assert calls["bincount"] <= 1  # a graph break may run the call eagerly (then taken); traced calls are not


# ---- every histc interval the chunker generates ---------------------------------------------------------------------

def _chunk_intervals(bins):
    c = small.BINCOUNT_CHUNK
    if bins <= c:
        return [(0, bins - 1, bins)]
    starts = list(range(0, bins - c, c)) + [bins - c]
    return [(a, a + c - 1, c) for a in starts]


def test_every_chunk_interval_maps_every_integer_to_its_own_bin_exactly():
    """For every bin count 1..4096: each histc interval [lo, hi] with n bins puts each integer lo..hi in bin v - lo,
    the closed last edge hi in the last bin, and lo - 1 / hi + 1 outside, as CPU float32 histc; and the bin edges
    lo + k (hi - lo) / (n - 1) ... computed as histc does are exact in float32 there."""
    seen = set()
    for bins in range(1, small.BINCOUNT_MAX_BINS + 1):
        seen.update(_chunk_intervals(bins))
    for lo, hi, n in sorted(seen):
        vals = torch.arange(lo - 1, hi + 2, dtype=torch.float32)
        got = torch.histc(vals, n, lo, hi) if n > 1 else torch.histc(vals[1:-1], 1, lo, hi)
        assert torch.equal(got, torch.ones(n)), (lo, hi, n)
        # bin index of v as float32 arithmetic: (v - lo) * n / (hi - lo), floored, last edge clamped
        v = torch.arange(lo, hi + 1, dtype=torch.float32)
        idx = ((v - lo) * (n / max(hi - lo, 1))).floor().clamp(max=n - 1) if n > 1 else torch.zeros(1)
        assert torch.equal(idx.long(), torch.arange(n)), (lo, hi, n)
    assert len(seen) > 256


@pytest.mark.parametrize("hi", [(1 << 24) - 1, (1 << 24) - 2, (1 << 23) + 1, 4095])
def test_large_integer_values_stay_exact_in_float32_and_fall_outside_chunks(hi):
    """Values up to 2**24 - 1 convert to float32 exactly; values past the bins never land in a chunk (the rewrite
    declines past 4096 bins, but a value 2**24 - 1 in an input with minlength must not alias a small bin)."""
    v = torch.tensor([hi], dtype=torch.int64)
    assert int(v.float().item()) == hi
    for lo, top, n in _chunk_intervals(4096):
        assert torch.histc(v.float(), n, lo, top).sum() == (1 if lo <= hi <= top else 0)


# ---- expanded bincount (< 65536 high-bin + compile uint8) ----------------------------------------------------------

@pytest.mark.parametrize("lazy,n,minlength,taken", [
    (False, 4096, 255, 0),
    (False, 4096, 256, 1),
    (False, 40960, 128, 0),
    (False, 40960, 512, 1),
    (True, 4096, 127, 0),
    (True, 4096, 128, 1),
    (True, 40960, 128, 1),
    (True, 40960, 256, 1),
])
def test_bincount_small_input_high_minlength_threshold_per_mode(monkeypatch, lazy, n, minlength, taken):
    monkeypatch.setattr(small, "_lazy", lambda: lazy)
    torch.manual_seed(0)
    x = torch.randint(0, 64, (n,), dtype=torch.int32)
    ref = torch.bincount(x, minlength=minlength)
    with cpu_as_hpu(small.install_bincount) as calls:
        out = torch.bincount(x, minlength=minlength)
    assert calls["bincount"] == taken
    same(out, ref, exact=True)


def test_bincount_small_low_minlength_incurs_zero_host_reads(monkeypatch):
    """< 256 (eager) / < 128 (lazy) bins with < 65536 elements decline before any aminmax host read."""
    for lazy, ml in [(False, 64), (False, 255), (True, 64), (True, 127)]:
        monkeypatch.setattr(small, "_lazy", lambda: lazy)
        x = torch.randint(0, 64, (4096,), dtype=torch.int64)
        with cpu_as_hpu(small.install_bincount) as calls, hpu_ops.strict("raise", allow=("cpu op",)):
            out = torch.bincount(x, minlength=ml)
        assert calls["bincount"] == 0
        same(out, torch.bincount(x, minlength=ml), exact=True)


def test_bincount_uint8_high_minlength_skips_host_read():
    for ml in (256, 1000):
        x = torch.randint(0, 256, (4096,), dtype=torch.uint8)
        ref = torch.bincount(x, minlength=ml)
        with cpu_as_hpu(small.install_bincount) as calls, hpu_ops.strict("raise", allow=("cpu op",)):
            out = torch.bincount(x, minlength=ml)
        assert calls["bincount"] == 1
        same(out, ref, exact=True)


def test_bincount_padded_tail_when_active_bins_below_minlength():
    x = torch.randint(0, 16, (4096,), dtype=torch.int32)
    ref = torch.bincount(x, minlength=512)
    with cpu_as_hpu(small.install_bincount) as calls:
        out = torch.bincount(x, minlength=512)
    assert calls["bincount"] == 1
    same(out, ref, exact=True)


def test_bincount_small_high_minlength_negative_raises():
    x = torch.randint(0, 10, (4096,), dtype=torch.int8)
    x[7] = -1
    with cpu_as_hpu(small.install_bincount), pytest.raises(RuntimeError, match="non-negative"):
        torch.bincount(x, minlength=256)


@pytest.mark.parametrize("n,minlength", [(0, 256), (4096, 256), (40960, 512), (40960, 1000)])
def test_bincount_compile_uint8_static_minlength_no_graph_break(n, minlength):
    import warnings

    x = torch.randint(0, 256, (n,), dtype=torch.uint8) if n else torch.empty(0, dtype=torch.uint8)
    ref = torch.bincount(x, minlength=minlength)
    torch._dynamo.reset()
    with warnings.catch_warnings(record=True) as rec, cpu_as_hpu(small.install_bincount) as calls:
        warnings.simplefilter("always")
        fn = torch.compile(lambda t: torch.bincount(t, minlength=minlength), backend="eager", fullgraph=True)
        out = fn(x)
    assert calls["bincount"] >= 1
    assert not [w for w in rec if "get_dynamic_layer_stack_depth" in str(w.message)
                or "peek_interpreter_stack" in str(w.message)]
    same(out, ref, exact=True)


# ---- scoped lazy logsumexp ------------------------------------------------------------------------------------------

def test_logsumexp_stays_native_in_eager_and_above_max_numel(monkeypatch):
    x = torch.randn(64, 22, 22)
    monkeypatch.setattr(small, "_lazy", lambda: False)
    with cpu_as_hpu(small.install_logsumexp) as calls:
        y_eager = torch.logsumexp(x, -1)
    assert calls["logsumexp"] == 0
    same(y_eager, torch.logsumexp(x, -1))

    monkeypatch.setattr(small, "_lazy", lambda: True)
    big = torch.randn(small.LOGSUMEXP_MAX_NUMEL + 1)
    with cpu_as_hpu(small.install_logsumexp) as calls:
        y_big = torch.logsumexp(big, 0)
    assert calls["logsumexp"] == 0
    same(y_big, torch.logsumexp(big, 0))


@pytest.mark.parametrize("vals,gval", [
    ([-inf, -inf, -inf], 1.0),
    ([-inf, 2.0, -inf], 1.0),
    ([inf, 100.0, 0.0], 1.0),
    ([inf, 100.0, 0.0], 0.0),
    ([inf, -inf], 1.0),
    ([inf, inf], 1.0),
    ([nan, 1.0], 1.0),
    ([nan, 1.0], 0.0),
    ([3e38, 3e38, 0.0], 1.0),
])
def test_logsumexp_lazy_nonfinite_forward_and_backward_match_aten(monkeypatch, vals, gval):
    monkeypatch.setattr(small, "_lazy", lambda: True)
    x_ref = torch.tensor(vals, dtype=torch.float32, requires_grad=True)
    y_ref = torch.logsumexp(x_ref, 0)
    (g_ref,) = torch.autograd.grad(y_ref, x_ref, torch.tensor(gval))

    x = torch.tensor(vals, dtype=torch.float32, requires_grad=True)
    with cpu_as_hpu(small.install_logsumexp) as calls:
        y = torch.logsumexp(x, 0)
        (g,) = torch.autograd.grad(y, x, torch.tensor(gval))
    assert calls["logsumexp"] == 1
    same(y, y_ref)
    same(g, g_ref)


@pytest.mark.parametrize("keepdim", [False, True])
@pytest.mark.parametrize("shape,dim", [
    ((64, 22, 22), -1),
    ((64, 22, 22), 0),
    ((6, 4, 5), (0, 2)),
    ((4, 3, 2), (0, 1, 2)),
    ((22,), 0),
])
def test_logsumexp_lazy_shapes_dims_inference_and_double_backward(monkeypatch, shape, dim, keepdim):
    monkeypatch.setattr(small, "_lazy", lambda: True)
    torch.manual_seed(0)
    x_ref = torch.randn(*shape, dtype=torch.float32, requires_grad=True)
    y_ref = torch.logsumexp(x_ref, dim, keepdim=keepdim)
    (g1_ref,) = torch.autograd.grad(y_ref.sum(), x_ref, create_graph=True)
    (g2_ref,) = torch.autograd.grad((g1_ref ** 2).sum(), x_ref)

    x = x_ref.detach().clone().requires_grad_()
    with cpu_as_hpu(small.install_logsumexp) as calls:
        y = torch.logsumexp(x, dim, keepdim=keepdim)
        (g1,) = torch.autograd.grad(y.sum(), x, create_graph=True)
        (g2,) = torch.autograd.grad((g1 ** 2).sum(), x)
        with torch.no_grad():
            y_ng = torch.logsumexp(x.detach(), dim, keepdim=keepdim)
        with torch.inference_mode():
            y_im = torch.logsumexp(x.detach(), dim, keepdim=keepdim)
    assert calls["logsumexp"] == 3
    same(y, y_ref)
    same(g1, g1_ref)
    same(g2, g2_ref)
    same(y_ng, y_ref.detach())
    same(y_im, y_ref.detach())


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.int32, torch.bool])
def test_logsumexp_lazy_dtypes_match_aten(monkeypatch, dtype):
    monkeypatch.setattr(small, "_lazy", lambda: True)
    if dtype == torch.bool:
        x = torch.tensor([[True, False], [False, False]])
    elif dtype == torch.int32:
        x = torch.tensor([[1, -2, 3], [0, 2, -1]], dtype=dtype)
    else:
        x = torch.randn(4, 5, dtype=dtype)
    ref = torch.logsumexp(x, -1)
    with cpu_as_hpu(small.install_logsumexp) as calls:
        out = torch.logsumexp(x, -1)
    assert calls["logsumexp"] == 1
    same(out, ref)


@pytest.mark.parametrize("shape,dim", [((0,), 0), ((0, 3), 0), ((3, 0), 1), ((0, 3), 1)])
def test_logsumexp_lazy_empty_dims_match_aten(monkeypatch, shape, dim):
    monkeypatch.setattr(small, "_lazy", lambda: True)
    x_ref = torch.empty(shape, dtype=torch.float32, requires_grad=True)
    y_ref = torch.logsumexp(x_ref, dim)
    (g_ref,) = torch.autograd.grad(y_ref.sum(), x_ref)

    x = torch.empty(shape, dtype=torch.float32, requires_grad=True)
    with cpu_as_hpu(small.install_logsumexp) as calls:
        y = torch.logsumexp(x, dim)
        (g,) = torch.autograd.grad(y.sum(), x)
    assert calls["logsumexp"] == 1
    same(y, y_ref, exact=True)
    same(g, g_ref, exact=True)


def test_logsumexp_lazy_errors_0d_and_undo(monkeypatch):
    monkeypatch.setattr(small, "_lazy", lambda: True)
    with cpu_as_hpu(small.install_logsumexp) as calls:
        same(torch.logsumexp(torch.tensor(2.5), ()), torch.tensor(2.5))
        assert calls["logsumexp"] == 0
        with pytest.raises(RuntimeError):
            torch.logsumexp(torch.zeros(2, 2), ())
        with pytest.raises(RuntimeError):
            torch.logsumexp(torch.zeros(2, 2), (0, 0))
        with pytest.raises(IndexError):
            torch.logsumexp(torch.zeros(2, 2), 2)
    undo = small.install_logsumexp("cpu")
    assert torch._C._dispatch_has_kernel_for_dispatch_key("aten::logsumexp", "AutogradCPU")
    assert torch._C._dispatch_has_kernel_for_dispatch_key("aten::logsumexp", "CPU")
    undo()
    assert not torch._C._dispatch_has_kernel_for_dispatch_key("aten::logsumexp", "AutogradCPU")
    assert not torch._C._dispatch_has_kernel_for_dispatch_key("aten::logsumexp", "CPU")


def test_logsumexp_lazy_transforms_and_compile_decline(monkeypatch):
    from torch.fx.experimental.proxy_tensor import make_fx

    monkeypatch.setattr(small, "_lazy", lambda: True)
    x = torch.randn(4, 5, requires_grad=True)
    with cpu_as_hpu(small.install_logsumexp) as calls:
        torch.func.jvp(lambda t: torch.logsumexp(t, -1), (x,), (torch.ones_like(x),))
        with torch.autograd.forward_ad.dual_level():
            torch.logsumexp(torch.autograd.forward_ad.make_dual(x, torch.ones_like(x)), -1)
        for mode in ("fake", "symbolic"):
            gm = make_fx(lambda t: torch.logsumexp(t, -1), tracing_mode=mode)(x.detach())
            assert "logsumexp" in str(gm.graph) and "amax" not in str(gm.graph)
        assert calls["logsumexp"] == 0
        gm_real = make_fx(lambda t: torch.logsumexp(t, -1), tracing_mode="real")(x.detach())
        assert "logsumexp" in str(gm_real.graph) and "amax" not in str(gm_real.graph)


# ---- scoped lazy index_add ------------------------------------------------------------------------------------------

def test_index_add_scope_declines_eager_grad_dim_and_small(monkeypatch):
    rows = small.INDEX_ADD_MIN_ROWS
    dst = torch.randn(16, 8)  # taken in lazy no_grad (2560 source rows per row): every decline below is the scope's
    idx = torch.randint(0, 16, (rows,))
    src = torch.randn(rows, 8)

    monkeypatch.setattr(small, "_lazy", lambda: False)
    with cpu_as_hpu(small.install_index_add) as calls, torch.no_grad():
        same(dst.index_add(0, idx, src), dst.index_add(0, idx, src))
    assert calls["index_add"] == 0

    monkeypatch.setattr(small, "_lazy", lambda: True)
    with cpu_as_hpu(small.install_index_add) as calls:
        # grad_enabled=True keeps native autograd
        dg = dst.clone().requires_grad_()
        sg = src.clone().requires_grad_()
        yg = dg.index_add(0, idx, sg, alpha=2.0)
        gd, gs = torch.autograd.grad(yg.sum(), (dg, sg))
        with torch.no_grad():
            # dim != 0 and source.shape[0] < INDEX_ADD_MIN_ROWS stay native
            dst2 = torch.randn(8, 64)
            idx2 = torch.randint(0, 64, (64,))
            src2 = torch.randn(8, 64)
            dst2.index_add(1, idx2, src2)
            dst.index_add(0, idx[:rows - 1], src[:rows - 1])
    assert calls["index_add"] == 0
    ref_d = dst.clone().requires_grad_()
    ref_s = src.clone().requires_grad_()
    ref_gd, ref_gs = torch.autograd.grad(ref_d.index_add(0, idx, ref_s, alpha=2.0).sum(), (ref_d, ref_s))
    same(gd, ref_gd)
    same(gs, ref_gs)


@pytest.mark.parametrize("shape_dst,dim,alpha,idx_dtype", [
    ((16,), 0, 1.0, torch.int64),
    ((16, 22), 0, 1.0, torch.int64),
    ((16, 22), -2, 2.5, torch.int32),
    ((16, 4, 3), 0, 1.0, torch.int64),
    ((256, 128), 0, 1.0, torch.int64),  # wide rows: taken at 160 source rows per row
])
def test_index_add_lazy_nograd_and_inference_match_aten(monkeypatch, shape_dst, dim, alpha, idx_dtype):
    monkeypatch.setattr(small, "_lazy", lambda: True)
    torch.manual_seed(0)
    rows = small.INDEX_ADD_MIN_ROWS
    dst = torch.randn(shape_dst)
    idx = torch.randint(0, shape_dst[0], (rows,), dtype=idx_dtype)
    src = torch.randn((rows,) + shape_dst[1:])
    src[0] = inf
    src[1] = -inf
    src[2] = nan
    ref = dst.index_add(dim, idx, src, alpha=alpha)
    with cpu_as_hpu(small.install_index_add) as calls:
        with torch.no_grad():
            out_ng = dst.index_add(dim, idx, src, alpha=alpha)
        with torch.inference_mode():
            out_im = dst.index_add(dim, idx, src, alpha=alpha)
    assert calls["index_add"] == 2
    torch.testing.assert_close(out_ng, ref, equal_nan=True, rtol=1e-4, atol=1e-3)
    torch.testing.assert_close(out_im, ref, equal_nan=True, rtol=1e-4, atol=1e-3)


def test_index_add_errors_and_undo(monkeypatch):
    monkeypatch.setattr(small, "_lazy", lambda: True)
    rows = small.INDEX_ADD_MIN_ROWS
    dst = torch.zeros(16, 4)
    idx = torch.zeros(rows, dtype=torch.int64)
    src_f16 = torch.ones(rows, 4, dtype=torch.float16)
    with cpu_as_hpu(small.install_index_add) as calls, torch.no_grad():
        with pytest.raises(RuntimeError):
            dst.index_add(0, idx, src_f16)
    assert calls["index_add"] == 0
    undo = small.install_index_add("cpu")
    assert torch._C._dispatch_has_kernel_for_dispatch_key("aten::index_add", "AutogradCPU")
    undo()
    assert not torch._C._dispatch_has_kernel_for_dispatch_key("aten::index_add", "AutogradCPU")


@pytest.mark.parametrize("shape,src_rows,wins", [
    ((16, 22), 40960, True),  # 2560 source rows per row
    ((16, 22), 40959, False),  # below INDEX_ADD_MIN_ROWS
    ((32, 22), 1280 * 32, True),  # at INDEX_ADD_MIN_PER_ROW
    ((32, 22), 1280 * 32 - 1, False),
    ((64, 22), 40960, False),  # 640 per row in narrow rows: Habana's kernel is as fast
    ((64,), 1280 * 64, True),
    ((256, 128), 40960, True),  # wide rows win at 160 per row
    ((256, 127), 40960, False),
    ((16, 4, 32), 40960, True),  # 128 elements per row
    ((257, 128), 1 << 22, False),  # past INDEX_ADD_MAX_SELF_ROWS, whatever the contention or width
    ((1024, 22), 1280 * 1024, False),
    ((0, 22), 1 << 20, False),  # empty: remainder by 0 rows
    ((16, 0), 1 << 20, False),
])
def test_index_add_scope_rows_contention_and_width(shape, src_rows, wins):
    """Taken only where it was measured to win on Gaudi2 lazy, and never into more than 256 rows: Habana's lazy
    accumulating index_put into more than 512 rows misreads an index made by an add in the same graph."""
    assert small._index_add_wins(shape, src_rows) is wins


def test_index_add_past_the_row_cap_stays_native(monkeypatch):
    monkeypatch.setattr(small, "_lazy", lambda: True)
    rows = small.INDEX_ADD_MIN_ROWS
    dst = torch.randn(small.INDEX_ADD_MAX_SELF_ROWS + 1, 128)
    idx = torch.randint(0, dst.shape[0], (rows,))
    src = torch.randn(rows, 128)
    ref = dst.index_add(0, idx, src)
    with cpu_as_hpu(small.install_index_add) as calls, torch.no_grad():
        out = dst.index_add(0, idx, src)
    assert calls["index_add"] == 0
    same(out, ref, exact=True)


@pytest.mark.parametrize("idx_dtype", [torch.int64, torch.int32])
@pytest.mark.parametrize("bad", [[-1], [-16], [-17], [16], [1 << 20], [-(1 << 20)], [-1, -17, 16, 1 << 20]])
def test_index_add_out_of_range_indices_as_habanas_kernel(monkeypatch, bad, idx_dtype):
    """ATen raises for an index out of range, which needs a host read. Habana's own index_add (measured on Gaudi2,
    eager and lazy) wraps [-n, -1] and drops the others; its index_put(accumulate=True) stops the device on an index
    of -1 ("Compute or dma timeout"). The rewrite does what Habana's index_add does, and no index out of range
    reaches index_put (the CPU's index_put raises for one)."""
    monkeypatch.setattr(small, "_lazy", lambda: True)
    rows, n = small.INDEX_ADD_MIN_ROWS, 16
    dst = torch.full((n, 8), 0.5)
    idx = torch.zeros(rows, dtype=idx_dtype)
    idx[:len(bad)] = torch.tensor(bad)
    src = torch.ones(rows, 8)
    src[:4] = torch.tensor([2.0, 3.0, 5.0, 7.0]).unsqueeze(1)  # sums stay exact in float32, in any order
    wrapped = torch.where(idx < 0, idx + n, idx)
    keep = (wrapped >= 0) & (wrapped < n)
    ref = dst.index_add(0, wrapped[keep], src[keep])
    with cpu_as_hpu(small.install_index_add) as calls, torch.no_grad():
        out = dst.index_add(0, idx, src)
    assert calls["index_add"] == 1
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


@pytest.mark.parametrize("dtype,alpha", [(torch.int64, 0.5), (torch.int64, 1.5), (torch.int32, 1.5), (torch.int32, 2),
                                         (torch.int16, 1), (torch.uint8, 1), (torch.bool, 1), (torch.float16, 1.0),
                                         (torch.bfloat16, 0.5), (torch.float64, 0.5)])
def test_index_add_keeps_atens_kernel_off_float32(monkeypatch, dtype, alpha):
    """Only float32 is taken. ATen casts ``alpha`` to ``self``'s dtype (int: 1.5 adds 1, 0.5 adds 0) where
    ``source * alpha`` promotes to float, a bool ``self`` accumulates as a logical or, and ATen sums the half dtypes
    in float32 (40960 ones give 40960, not bfloat16's 256)."""
    monkeypatch.setattr(small, "_lazy", lambda: True)
    rows = small.INDEX_ADD_MIN_ROWS
    dst = torch.zeros(10, 4, dtype=dtype)
    idx = torch.zeros(rows, dtype=torch.int64)
    src = torch.ones(rows, 4, dtype=dtype)
    with torch.no_grad():
        ref = dst.index_add(0, idx, src, alpha=alpha)
    with cpu_as_hpu(small.install_index_add) as calls, torch.no_grad():
        out = dst.index_add(0, idx, src, alpha=alpha)
    assert calls["index_add"] == 0
    same(out, ref, exact=True)


def test_index_add_noncontiguous_taken_and_out_variants_native(monkeypatch):
    """The taken path accepts any strides (transposed self, sliced source); ``out=`` and ``index_add_`` are never
    registered, so they keep the native kernel."""
    monkeypatch.setattr(small, "_lazy", lambda: True)
    rows = small.INDEX_ADD_MIN_ROWS
    dst = torch.randn(32, 16).t()  # (16, 32), non-contiguous
    src = torch.ones(2 * rows, 32)[::2]  # non-contiguous; ones keep every sum exact in float32
    idx = torch.randint(0, 16, (rows,), generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        ref = dst.index_add(0, idx, src)
    with cpu_as_hpu(small.install_index_add) as calls, torch.no_grad():
        out = dst.index_add(0, idx, src)
    assert calls["index_add"] == 1
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
    with cpu_as_hpu(small.install_index_add) as calls, torch.no_grad():
        buf = dst.clone()
        torch.index_add(dst, 0, idx, src, out=buf)
        dst.clone().index_add_(0, idx, src)
    assert calls["index_add"] == 0


# ---- trace stays native ----------------------------------------------------------------------------------------------

def test_trace_stays_native_in_both_modes(monkeypatch):
    for lazy in (False, True):
        monkeypatch.setattr(small, "_lazy", lambda: lazy)
        with cpu_as_hpu(hpu_ops._install) as calls, torch.no_grad():
            torch.trace(torch.randn(64, 64))
        assert "trace" not in calls
