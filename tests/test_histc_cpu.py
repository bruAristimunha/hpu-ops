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
    try:
        return torch.histc(x, *args)
    except RuntimeError as e:
        return e


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


def test_histc_sixteen_bit_counts_are_exact_counts_rounded_once():
    """ATen's CPU kernel accumulates 16-bit counts in the 16-bit dtype (bfloat16: 1001 counts give 1000, 70000 give
    2560); the route counts in float32 and rounds once, which this pins down."""
    x = torch.full((70000,), 0.3, dtype=torch.bfloat16)
    with cpu_as_hpu(small.install_histc):
        out = torch.histc(x, 4, 0.0, 1.0)
    assert out.dtype == torch.bfloat16 and out[1].item() == torch.tensor(70000.0).bfloat16().item()


def test_histc_edges_are_cached_and_reused():
    small._EDGE_CACHE.clear()
    x = torch.rand(100)
    with cpu_as_hpu(small.install_histc) as calls:
        a = torch.histc(x, 50, 0.0, 1.0)
        b = torch.histc(x, 50, 0.0, 1.0)
    assert calls["histc"] == 2 and len(small._EDGE_CACHE) == 1
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


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16, torch.float64])
@pytest.mark.parametrize("lo,hi,bins", [(-3.0, 3.0, 64), (0.1, 0.7, 1000), (-0.37, 2.9, 31), (0.0, 1.0, 4096),
                                        (-1e-3, 2e-3, 7), (1e4, 1e4 + 3.0, 200)])
def test_histc_estimate_is_within_the_correction_steps(lo, hi, bins, dtype):
    """The float32 estimate against ATen's own positions (16-bit: ATen rounds in the 16-bit dtype), at and next to
    every edge; where the bound would need more than HISTC_MAX_STEPS the route uses bucketize instead."""
    edges, low, high = small.histc_edges(lo, hi, bins, dtype)
    inner = edges[1:-1]
    cdtype = torch.float32 if dtype == torch.float64 else dtype
    x = torch.cat([_data(20000, lo, hi, cdtype), inner, torch.nextafter(inner.float(), torch.tensor(-inf)).to(cdtype)])
    x = x[(x >= low) & (x <= high)]
    truth = torch.bucketize(x, inner, right=True)
    est = small._estimate(x, bins, low, high)
    assert int((est - truth).abs().max()) <= small._correction_steps(bins, dtype)


@pytest.mark.parametrize("dtype,bins", [(torch.float32, 257), (torch.float32, 1000), (torch.float32, 4096),
                                        (torch.float64, 300)])
def test_histc_corrects_an_estimate_off_by_the_bound(dtype, bins, monkeypatch):
    """Gaudi's division is not IEEE: the stand-in's estimate is pushed off by up to the step bound, the result must
    still be ATen's."""
    steps = small._correction_steps(bins, dtype)
    assert steps <= small.HISTC_MAX_STEPS and bins > small.HISTC_BUCKETIZE_MAX_BINS  # the correction path
    g = torch.Generator().manual_seed(1)
    lo, hi = -0.37, 2.9
    cdtype = torch.float32 if dtype == torch.float64 else dtype
    edges = small.histc_edges(lo, hi, bins, dtype)[0][1:-1]

    called = []

    def off(flat, bins_, low, high):  # ATen's position, pushed off by up to the bound
        called.append(1)
        assert bool(((flat >= low) & (flat <= high)).all())  # only in-range values reach the estimate
        truth = torch.bucketize(flat, edges, right=True)
        noise = torch.randint(-steps, steps + 1, truth.shape, generator=g)
        return (truth + noise).clamp(0, bins_ - 1)

    monkeypatch.setattr(small, "_estimate", off)
    small._EDGE_CACHE.clear()
    x = torch.cat([_data(3000, lo, hi, cdtype), edges, torch.nextafter(edges.float(), torch.tensor(-inf)).to(cdtype),
                   torch.tensor([nan, inf, -inf, lo, hi], dtype=cdtype)]).to(dtype)
    _check(x, bins, lo, hi)
    assert called


def _clamped_reference(x, bins, lo, hi):
    """ATen's positions with those past the last bin put in it, counted exactly and rounded once."""
    L, R = torch.tensor(lo, dtype=x.dtype), torch.tensor(hi, dtype=x.dtype)
    inside = x[(x >= L) & (x <= R)]
    pos = ((inside - L) * torch.tensor(bins, dtype=x.dtype) / (R - L)).trunc().long().clamp(max=bins - 1)
    return torch.bincount(pos, minlength=bins).float().to(x.dtype)


def test_histc_many_bins_in_16_bit_use_bucketize_and_keep_positions_past_the_end_in_the_last_bin():
    """In bfloat16 with 500 bins over [-3, 3], ATen's position of 3.0 is 502, not 500: its CPU kernel then writes past
    the histogram (undefined behaviour; its counts sum to more than the values in range). The route puts such values
    in the last bin."""
    assert small._correction_steps(4096, torch.bfloat16) > small.HISTC_MAX_STEPS
    assert small._correction_steps(4096, torch.float32) == 1
    x = torch.cat([_data(3000, -3.0, 3.0, torch.bfloat16), torch.tensor([3.0, 2.9844], dtype=torch.bfloat16)])
    with cpu_as_hpu(small.install_histc) as calls:
        out = torch.histc(x, 500, -3.0, 3.0)
    assert calls["histc"] == 1
    same(out, _clamped_reference(x, 500, -3.0, 3.0), exact=True)
