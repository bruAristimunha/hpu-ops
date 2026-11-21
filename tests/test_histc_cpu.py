"""CPU tests of hpu_ops.small's histc route: "cpu" stands in for "hpu". Gaudi's kernel aborts on a bound given as a
Python float and does not compile past 300 bins; those calls are taken and must equal ATen's CPU kernel bit for bit
(bins, the closed last bin, NaN / inf / out-of-range values skipped, equal bounds taking the data's range, ATen's
errors). Calls Gaudi's kernel takes keep it."""
import math

import pytest
import torch

import hpu_ops
from hpu_ops import small

from test_small_cpu import cpu_as_hpu, same

inf, nan = math.inf, math.nan


def _native(x, *args):
    """ATen's CPU kernel with one thread: its counts then do not depend on how the input is split across threads."""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        return torch.histc(x, *args)
    except RuntimeError as e:
        return e
    finally:
        torch.set_num_threads(threads)


def _check(x, *args, taken=1):
    ref = _native(x, *args)
    with cpu_as_hpu(small.install_histc) as calls:
        if isinstance(ref, RuntimeError):
            with pytest.raises(RuntimeError, match=str(ref).split(" ")[0]):  # the same kind of ATen error
                torch.histc(x, *args)
            return
        out = torch.histc(x, *args)
    assert calls["histc"] == taken, (args, calls)
    same(out, ref, exact=True)


def _data(n, lo, hi, dtype, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(n, generator=g, dtype=torch.float64) * (hi - lo) * 1.2 + lo - 0.1 * (hi - lo)
    return x.to(dtype)


BOUNDS = [(-3.0, 3.0, 64), (0.1, 0.7, 1000), (-1e-3, 2e-3, 7), (-0.5, 63.5, 64), (0.0, 1.0, 4096), (-0.37, 2.9, 2),
          (0.0, 1.0, 1), (1e6, 1e6 + 1.0, 300), (-1e30, 1e30, 5), (0, 63, 400), (0, 1, 301)]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("lo,hi,bins", BOUNDS)
def test_histc_float_bounds_and_many_bins_match_aten_bit_for_bit(lo, hi, bins, dtype):
    if dtype == torch.float16 and max(abs(lo), abs(hi)) > 6e4:
        pytest.skip("out of float16's range (ATen raises; tested below)")
    x = _data(200, lo, hi, dtype)  # few per bin: ATen's 16-bit kernel counts in the 16-bit dtype (see below)
    edges = small.histc_edges(float(lo), float(hi), bins, dtype)[0][1:-1]
    up = torch.nextafter(edges.float(), torch.tensor(inf)).to(dtype)
    down = torch.nextafter(edges.float(), torch.tensor(-inf)).to(dtype)
    x = torch.cat([x, edges, up, down, torch.tensor([lo, hi, nan, inf, -inf], dtype=dtype)])
    _check(x, bins, lo, hi)


def test_histc_every_value_on_an_edge_float32():
    lo, hi, bins = -0.37, 2.9, 777
    edges = small.histc_edges(lo, hi, bins, torch.float32)[0][1:-1]
    x = torch.cat([edges, torch.nextafter(edges, torch.tensor(inf)), torch.nextafter(edges, torch.tensor(-inf))])
    _check(x, bins, lo, hi)


@pytest.mark.parametrize("lo,hi,bins", [(-3.0, 3.0, 64), (0.1, 0.7, 1000), (-0.37, 2.9, 2)])
def test_histc_float64_counts_the_float32_values_the_card_holds(lo, hi, bins):
    x = _data(5000, lo, hi, torch.float64).float().double()  # what a float64 HPU tensor holds
    edges = small.histc_edges(lo, hi, bins, torch.float64)[0][1:-1]
    x = torch.cat([x, edges.double(), torch.nextafter(edges, torch.tensor(-inf)).double()])
    _check(x, bins, lo, hi)


@pytest.mark.parametrize("x", [
    torch.tensor([0.1, 0.2, 0.7, 2.5]), torch.full((5,), 2.5), torch.tensor([nan, 1.0]), torch.tensor([0.0, inf]),
    torch.tensor([-inf, 1.0]), torch.empty(0), torch.tensor([3.25]), torch.randn(3, 4, 5)])
@pytest.mark.parametrize("lo,hi", [(0.0, 0.0), (2.5, 2.5), (1.5, 1.5)])
def test_histc_equal_float_bounds_take_the_datas_range(x, lo, hi):
    _check(x, 10, lo, hi)


@pytest.mark.parametrize("args", [(0, 0.0, 1.0), (-2, 0.0, 1.0), (4, 2.5, 1.0), (4, nan, 1.0), (4, 0.0, inf),
                                  (4, 0.0, 1e300)])
def test_histc_invalid_arguments_raise_atens_errors(args):
    ref = _native(torch.rand(10), *args)
    assert isinstance(ref, RuntimeError)
    with cpu_as_hpu(small.install_histc), pytest.raises(RuntimeError):
        torch.histc(torch.rand(10), *args)


@pytest.mark.parametrize("args", [(64, 0, 63), (64, -1, 63), (300, 0, 1), (64, 0, 0), (64, 5, 5), (10, False, True)])
def test_histc_python_int_bounds_up_to_300_bins_keep_gaudis_kernel(args):
    _check(torch.rand(500) * 70 - 3, *args, taken=0)


def test_histc_views_and_shapes():
    base = torch.rand(4, 300) * 6 - 3
    for x in (base[1:], base[:, 1::3], base.t(), base[2, 5:], base.reshape(-1)[4:].view(4, -1)):
        _check(x, 64, -3.0, 3.0)
    _check(torch.tensor(1.5), 4, 1.0, 2.0)  # 0-d
    _check(torch.rand(1, 1, 1), 3, 0.0, 1.0)


@pytest.mark.parametrize("dtype,n,bins,value,want", [(torch.bfloat16, 4096, 4, 0.25, 256),
                                                     (torch.float16, 4096, 4, 0.25, 2048),
                                                     (torch.bfloat16, 70000, 4, 0.3, 256),
                                                     (torch.float32, 16777218, 1, 0.5, 16777216)])
def test_histc_counts_stop_where_atens_one_thread_kernel_stops(dtype, n, bins, value, want):
    """ATen's CPU kernel adds 1 per value in the input's dtype, so a bin stops at 2**p (p significand bits); the
    route caps its exact counts there (Codex review #5)."""
    x = torch.full((n,), value, dtype=dtype)
    ref = _native(x, bins, 0.0, 1.0)
    with cpu_as_hpu(small.install_histc) as calls:
        out = torch.histc(x, bins, 0.0, 1.0)
    assert calls["histc"] == 1 and float(out.max()) == want
    same(out, ref, exact=True)


def test_histc_edges_are_cached_and_reused():
    small._EDGE_CACHE.clear()
    x = torch.rand(100)
    with cpu_as_hpu(small.install_histc) as calls:
        a = torch.histc(x, 50, 0.0, 1.0)
        b = torch.histc(x, 50, 0.0, 1.0)
        assert len(small._EDGE_CACHE) == 1
    assert calls["histc"] == 2 and not small._EDGE_CACHE  # the undo releases the edge tables
    same(a, b, exact=True)


@pytest.mark.parametrize("backend", ["eager", "aot_eager"])
def test_histc_compiled_with_float_bounds(backend):
    x = torch.rand(1000) * 6 - 3
    f = lambda t: torch.histc(t * 1, 64, -3.0, 3.0)  # noqa: E731
    ref = f(x)
    torch._dynamo.reset()
    with cpu_as_hpu(small.install_histc) as calls:
        out = torch.compile(f, backend=backend, fullgraph=True)(x)
    torch._dynamo.reset()
    same(out, ref, exact=True)
    assert calls["histc"] >= 1


def test_histc_compiled_equal_bounds_keep_atens_kernel():
    x = torch.rand(1000)
    f = lambda t: torch.histc(t, 10, 0.0, 0.0)  # noqa: E731
    def run():
        torch._dynamo.reset()
        try:
            return torch.compile(f, backend="aot_eager")(x)
        finally:
            torch._dynamo.reset()

    ref = run()  # ATen's own histc reads the data's range on the host: dynamo breaks the graph around it
    with cpu_as_hpu(small.install_histc) as calls:
        out = run()
    same(out, ref, exact=True)
    same(out, torch.histc(x, 10, 0.0, 0.0), exact=True)  # traced calls are declined; after a graph break, taken


def test_histc_no_grad_and_requires_grad_input():
    x = (torch.rand(100) * 2).requires_grad_()
    ref = torch.histc(x.detach(), 8, 0.5, 1.5)
    with cpu_as_hpu(small.install_histc) as calls:
        with torch.no_grad():
            a = torch.histc(x, 8, 0.5, 1.5)
        b = torch.histc(x, 8, 0.5, 1.5)
    assert calls["histc"] == 2 and not b.requires_grad
    same(a, ref, exact=True)
    same(b, ref, exact=True)


def test_histc_undo_restores_aten():
    undo = small.install_histc("cpu")
    assert torch._C._dispatch_has_kernel_for_dispatch_key("aten::histc", "AutogradCPU")
    undo()
    assert not torch._C._dispatch_has_kernel_for_dispatch_key("aten::histc", "AutogradCPU")
    hpu_ops._STATE["calls"].clear()
    torch.histc(torch.rand(10), 4, 0.0, 1.0)
    assert not hpu_ops._STATE["calls"]["histc"]


def test_bincount_one_bin_still_counts_every_zero():
    x = torch.zeros(1 << 16, dtype=torch.long)
    with cpu_as_hpu(small.install_bincount) as calls:
        out = torch.bincount(x)
    assert calls["bincount"] == 1
    same(out, torch.bincount(x), exact=True)


def _clamped_reference(x, bins, lo, hi):
    """ATen's positions with those past the last bin put in it, counted exactly and rounded once."""
    L, R = torch.tensor(lo, dtype=x.dtype), torch.tensor(hi, dtype=x.dtype)
    inside = x[(x >= L) & (x <= R)]
    pos = ((inside - L) * torch.tensor(bins, dtype=x.dtype) / (R - L)).trunc().long().clamp(max=bins - 1)
    return small._cap(torch.bincount(pos, minlength=bins).float(), x.dtype)


def test_histc_16_bit_positions_past_the_end_go_to_the_last_bin():
    """In bfloat16 with 500 bins over [-3, 3], ATen's position of 3.0 is 502, not 500: its CPU kernel then writes past
    the histogram (undefined behaviour; its counts sum to more than the values in range). The route puts such values
    in the last bin."""
    x = torch.cat([_data(3000, -3.0, 3.0, torch.bfloat16), torch.tensor([3.0, 2.9844], dtype=torch.bfloat16)])
    with cpu_as_hpu(small.install_histc) as calls:
        out = torch.histc(x, 500, -3.0, 3.0)
    assert calls["histc"] == 1
    same(out, _clamped_reference(x, 500, -3.0, 3.0), exact=True)


# ---- adversarial pass -------------------------------------------------------------------------------------------

@pytest.mark.parametrize("args", [(64, 0.0, 63), (64, 0, 63.0), (64, -0.0, 1.0), (3, 1e-45, 2e-45), (5, -1.0, -0.5),
                                  (100000, 0.0, 1.0), (2, 0.5, 0.5000001)])
def test_histc_adversarial_bounds(args):
    x = torch.cat([torch.rand(500) * 2 - 1, torch.tensor([1e-45, 2e-45, -0.0, 0.0, 0.5, 0.5000001])])
    _check(x, *args)


def test_histc_two_calls_and_interleaved_dtypes_share_no_state_wrongly():
    small._EDGE_CACHE.clear()
    xs = [_data(300, -1.0, 1.0, d) for d in (torch.float32, torch.bfloat16, torch.float16, torch.float32)]
    refs = [torch.histc(x, 50, -1.0, 1.0) for x in xs]
    with cpu_as_hpu(small.install_histc):
        outs = [torch.histc(x, 50, -1.0, 1.0) for x in xs]
        assert len(small._EDGE_CACHE) == 3
    for o, r in zip(outs, refs):
        same(o, r, exact=True)


def test_histc_non_float_inputs_keep_atens_kernel():
    with cpu_as_hpu(small.install_histc) as calls, pytest.raises(RuntimeError, match="not implemented"):
        torch.histc(torch.arange(5), 4, 0.0, 4.0)
    assert calls["histc"] == 0




# ---- Codex review (PR #40): each exact input, against ATen's one-thread CPU kernel -------------------------------

def test_review1_tiny_range_does_not_overflow_the_position():
    x = torch.tensor([2e-38])
    _check(x, 400, 0.0, 1e-37)  # was bin 398 (an overflowing 400 / (high - low)), ATen: 80


def test_review2_float64_equal_bounds_widened_by_one():
    _check(torch.tensor([2.0 ** 27], dtype=torch.float64), 400, 0.0, 0.0)  # was bin 1, ATen: 200


@pytest.mark.parametrize("bins", [8, 400])
def test_review4_inference_mode_and_out_calls_take_the_route(bins):
    x = torch.rand(300)
    ref = _native(x, bins, 0.0, 1.0)
    with cpu_as_hpu(small.install_histc) as calls:
        with torch.inference_mode():
            a = torch.histc(x, bins, 0.0, 1.0)
        out = torch.empty(0)
        torch.histc(x, bins, 0.0, 1.0, out=out)
        with torch.inference_mode():
            out2 = torch.empty(3)
            torch.histc(x, bins, 0.0, 1.0, out=out2)
    # inference mode + out= reaches the backend histc.out, which stays Habana's: the route counts through it
    assert calls["histc"] == 2
    for got in (a, out, out2):
        same(got, ref, exact=True)


def _histc_float_bound_calls(gm):
    return [n for n in gm.graph.nodes if "histc" in str(n.target)
            and any(isinstance(a, float) for a in n.args[2:])]


@pytest.mark.parametrize("dynamic", [False, True])
def test_review4_compiled_graphs_hold_no_float_bound_histc(dynamic):
    graphs = []

    def backend(gm, inputs):
        from torch._functorch.aot_autograd import aot_module_simplified

        def keep(g, _):
            graphs.append(g)
            return g

        return aot_module_simplified(gm, inputs, fw_compiler=keep)

    f = lambda t, b: torch.histc(t * 1, b, 0.0, 1.0)  # noqa: E731
    torch._dynamo.reset()
    with cpu_as_hpu(small.install_histc):
        g = torch.compile(f, backend=backend, dynamic=dynamic, fullgraph=True)
        for n in (300, 301):
            x = torch.rand(n)
            same(g(x, 400), _native(x, 400, 0.0, 1.0), exact=True)
    torch._dynamo.reset()
    assert graphs and not any(_histc_float_bound_calls(gm) for gm in graphs)


def test_review4_compiled_equal_float_bounds_hold_no_float_bound_histc():
    graphs = []

    def backend(gm, inputs):
        from torch._functorch.aot_autograd import aot_module_simplified

        def keep(g, _):
            graphs.append(g)
            return g

        return aot_module_simplified(gm, inputs, fw_compiler=keep)

    x = torch.rand(1000) * 5
    torch._dynamo.reset()
    with cpu_as_hpu(small.install_histc):
        out = torch.compile(lambda t: torch.histc(t + 0, 64, 0.0, 0.0), backend=backend, fullgraph=True)(x)
    torch._dynamo.reset()
    assert graphs and not any(_histc_float_bound_calls(gm) for gm in graphs)
    assert out.sum() == 1000 and out.shape == (64,)  # the card's own arithmetic there: not ATen's at the edges


def test_review5_one_bin_float32_saturates_as_aten():
    x = torch.full((16777218,), 0.5)
    _check(x, 1, 0.0, 1.0)


def test_review6_float64_offset_views_are_read_through_an_offset_free_view():
    base = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float64)
    for v in (base[1:2], base[2:], base.view(2, 2)[1], base[1::2]):
        same(small._card_values(v, traced=False), v.float().reshape(-1), exact=True)
        _check(v, 4, 0.0, 4.0)


def test_review7_default_float64_does_not_move_the_bounds():
    small._EDGE_CACHE.clear()
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        x = torch.tensor([1.0])  # float64 under the default
        ref = _native(x.float().double(), 4, 1 + 2 ** -26, 2.0)
        with cpu_as_hpu(small.install_histc):
            out = torch.histc(x, 4, 1 + 2 ** -26, 2.0)
    finally:
        torch.set_default_dtype(old)
    same(out, ref, exact=True)
    x64 = torch.tensor([1.0], dtype=torch.float64)
    _check(x64, 4, 1 + 2 ** -26, 2.0)


def test_review8_bfloat16_unreachable_last_threshold():
    _check(torch.tensor([1.0078125], dtype=torch.bfloat16), 514, 1.0, 1.0078125)  # ATen: bin 512


@pytest.mark.parametrize("x,args", [(torch.tensor([65504.0], dtype=torch.float16), (4, 0.0, 0.0)),
                                    (torch.empty(0), (4, 0.0, 1e300)),
                                    (torch.tensor([1.0], dtype=torch.bfloat16), (4, 0.0, 3.39e38))])
def test_review9_bound_overflow_raises_as_aten(x, args):
    assert isinstance(_native(x, *args), RuntimeError)
    with cpu_as_hpu(small.install_histc), pytest.raises(RuntimeError, match="without overflow"):
        torch.histc(x, *args)


def test_review16_edge_cache_stays_bounded():
    small._EDGE_CACHE.clear()
    with cpu_as_hpu(small.install_histc):
        for k in range(small.EDGE_CACHE_SIZE + 5):
            torch.histc(torch.rand(10), 2, 0.0, 1.0 + k)
        assert len(small._EDGE_CACHE) == small.EDGE_CACHE_SIZE
    assert not small._EDGE_CACHE
