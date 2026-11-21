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
    assert sum(calls.values()) == 0  # cumsum and logsumexp keep Habana's kernels (measured, see the README)


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
