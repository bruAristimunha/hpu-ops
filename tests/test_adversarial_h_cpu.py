"""Adversarial round 2 of the indexing rewrites (install_index, install_mask_fill), on the CPU standing in for HPU.
The first tests are the round-2 breaks (mask puts that differ from ATen's index_put_), fixed by keeping the native op
for a value of another dtype and for a destination whose elements overlap; the rest widen the coverage."""
import contextlib
import warnings

import pytest
import torch
import torch.nn.functional as F

from hpu_ops import indexing
from test_cpu import SCALAR_MASK_PUT, assert_same, cpu_as_hpu, cpu_indices_off_device, train_step  # noqa: F401

MASK = torch.tensor([True, False, True, False])


# ---- round-2 breaks: mask_fill differed from ATen's index_put_; such puts now keep the native op


def test_mask_put_of_a_value_of_another_dtype_with_a_gradient_keeps_the_native_op():
    def run():
        v = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
        x = torch.zeros(3, 4)
        x[:, MASK] = v
        (x * torch.arange(12.0).reshape(3, 4)).sum().backward()
        return x.detach(), v.grad

    want = run()
    with cpu_as_hpu(indexing.install_mask_fill) as calls:
        got = run()
    assert not calls["mask_fill"]  # the host conversion would drop the value's gradient
    assert torch.equal(got[0], want[0]) and torch.equal(got[1], want[1])


@pytest.mark.parametrize("dtype, value, rewritten", [
    (torch.float16, torch.tensor(1e6), False),  # overflow: ATen raises
    (torch.int8, torch.tensor(300), False),  # overflow: ATen raises
    (torch.float32, torch.tensor(1e300, dtype=torch.float64), False),  # overflow: ATen raises
    (torch.int32, torch.tensor(float("nan")), False),  # ATen raises
    (torch.float32, torch.tensor(1 + 0j), True),  # complex without imaginary part: ATen writes 1
    (torch.int64, torch.tensor(1.5), True),  # float into int: ATen truncates
    (torch.uint8, torch.tensor(-1), True),  # ATen wraps to 255
    (torch.float32, torch.tensor(True), True),
    (torch.float32, torch.tensor(1.25, dtype=torch.float64), True),  # lazy Gaudi's native op rejects this CPU value
    (torch.float32, torch.tensor(float("inf"), dtype=torch.float64), True),
])
def test_mask_put_of_a_value_of_another_dtype_converts_as_aten(dtype, value, rewritten):
    def put():
        x = torch.zeros(3, 4, dtype=dtype)
        x[:, MASK] = value
        return x

    try:
        want = put()
    except RuntimeError as e:
        want = e
    with cpu_as_hpu(indexing.install_mask_fill) as calls:
        try:
            got = put()
        except RuntimeError as e:
            got = e
    assert calls["mask_fill"] == rewritten
    if isinstance(want, RuntimeError):
        assert isinstance(got, RuntimeError) and str(got) == str(want)
    else:
        assert torch.equal(got, want)


@pytest.mark.parametrize("stride", [(1, 1), (0, 1), (2, 1)])
def test_mask_put_into_an_overlapping_or_gapped_view_writes_like_aten(stride):
    def put():
        b = torch.zeros(12)
        b.as_strided((3, 4), stride)[:, MASK] = 1.0
        return b

    want = put()
    with cpu_as_hpu(indexing.install_mask_fill) as calls:
        got = put()
    assert not calls["mask_fill"]
    assert torch.equal(got, want)


def test_mask_put_of_the_same_dtype_into_a_dense_tensor_is_still_rewritten():
    # spd_learn's case: torch.ones_like(v)[..., mask] = sqrt(2), the Python float wrapped in the tensor's dtype
    for make in (lambda: torch.ones(5, 6), lambda: torch.ones(6, 5).t(), lambda: torch.ones(5, 6, dtype=torch.float64)):
        want, got = make(), make()
        want[..., torch.arange(6) % 2 == 0] = 2 ** 0.5
        with cpu_as_hpu(indexing.install_mask_fill) as calls:
            got[..., torch.arange(6) % 2 == 0] = 2 ** 0.5
        # a Python number reaches index_put_ through torch 2.14, masked_fill_ (left native) from 2.15
        assert calls["mask_fill"] == SCALAR_MASK_PUT and torch.equal(got, want)


def test_index_range_is_read_on_the_host_only_for_real_cpu_index_tensors():
    # A CPU index tensor is checked on the host (out of range: ATen's error). One on the device, or fake while
    # torch.compile traces, is not read (None): the rewrite then turns an index out of range into -1, which
    # index_select rejects.
    from torch._subclasses.fake_tensor import FakeTensorMode

    assert indexing._in_range(torch.tensor([0, -4, 3]), 4) is True
    assert indexing._in_range(torch.tensor([4]), 4) is False
    assert indexing._in_range(torch.empty(3, dtype=torch.long, device="meta"), 4) is None
    with FakeTensorMode():
        assert indexing._in_range(torch.tensor([0, 1]), 4) is None


@pytest.mark.parametrize("bad", [torch.tensor([0, 4]), torch.tensor([-5, 0])])
def test_index_out_of_range_unchecked_raises(bad):
    x = torch.randn(3, 4, 4)
    with pytest.raises(RuntimeError, match="out of"):  # index_select's error, where ATen's index raises IndexError
        indexing.index_select_flat(x, [None, bad, torch.tensor([0, 1])], checked=False)
    got = indexing.index_select_flat(x, [None, torch.tensor([-4, 3]), torch.tensor([0, -1])], checked=False)
    assert torch.equal(got, x[:, torch.tensor([-4, 3]), torch.tensor([0, -1])])


# ---- coverage that passes


@pytest.mark.usefixtures("cpu_indices_off_device")
@pytest.mark.parametrize("dtype", [torch.int64, torch.int32, torch.int8, torch.uint8, torch.bool, torch.complex64])
def test_index_rewrite_of_integer_bool_and_complex_data(dtype):
    torch.manual_seed(0)
    x = (torch.randn(5, 6, 6) * 50).to(dtype) if dtype != torch.complex64 else torch.randn(5, 6, 6, dtype=dtype)
    r, c = torch.triu_indices(6, 6)
    with cpu_as_hpu(indexing.install_index) as calls:
        got = x[..., r, c]
    assert calls["index"] == 1
    assert got.dtype == dtype and torch.equal(got, x[..., r, c])


@pytest.mark.parametrize("transform", ["jacrev", "hessian", "vmap_grad", "vmap_over_index", "jvp_mask"])
def test_functorch_transforms_through_the_rewrites(transform):
    torch.manual_seed(0)
    x = torch.randn(2, 4, 4, dtype=torch.float64)
    r, c = torch.triu_indices(4, 4)

    def vech(m):
        return m[..., r, c]

    def masked(m):
        y = m * 1
        y[..., MASK] = 0.5
        return y

    fs = {
        "jacrev": lambda: torch.func.jacrev(vech)(x),
        "hessian": lambda: torch.func.hessian(lambda m: (vech(m) ** 3).sum())(x),
        "vmap_grad": lambda: torch.vmap(torch.func.grad(lambda m: (vech(m) ** 2).sum()))(x),
        "vmap_over_index": lambda: torch.vmap(lambda i: x[:, i])(torch.tensor([[0, 1], [2, 3]])),
        "jvp_mask": lambda: torch.func.jvp(masked, (x,), (torch.ones_like(x),))[1],
    }
    want = fs[transform]()
    with cpu_as_hpu(indexing.install_index), cpu_as_hpu(indexing.install_mask_fill):
        got = fs[transform]()
    assert torch.equal(got, want)


def test_mask_put_gradient_of_the_value():
    torch.manual_seed(0)
    x, v = torch.randn(32, 6, 6), torch.randn((), requires_grad=True)

    def f(x):
        y = x * 1
        y[..., torch.arange(6) % 2 == 0] = v
        return y

    want = train_step(f, x)
    gv_want, v.grad = v.grad.clone(), None
    with cpu_as_hpu(indexing.install_mask_fill) as calls:
        got = train_step(f, x)
    assert calls["mask_fill"] == 1
    assert_same(got, want, tol=0)
    # the value's gradient is a sum of 576 elements: ATen's index_put_ backward sums the selected ones, where's the
    # masked full tensor, in another order (bit-equal on macOS, 1 ulp apart on Linux)
    torch.testing.assert_close(v.grad, gv_want, rtol=2e-6, atol=0)


def test_duplicate_indices_accumulate_in_the_backward():
    x = torch.randn(512, 6, 6, requires_grad=True)
    idx = torch.tensor([0, 0, 0, 1, 2, 2])
    w = torch.linspace(0.5, 1.5, 512 * 6).reshape(512, 6)
    (x[..., idx, idx] * w).sum().backward()
    want, x.grad = x.grad.clone(), None
    with cpu_as_hpu(indexing.install_index) as calls:
        (x[..., idx, idx] * w).sum().backward()
    assert calls["index"] == 1
    assert torch.allclose(x.grad, want, rtol=1e-6, atol=0)


# ---- composed with PR #24's index kernel: every integer index, masks after a slice, compiled graphs


COMPOSED = {
    "None and Ellipsis": lambda x: x[None, ..., torch.tensor([0, -1]), None],
    "leading, int32, broadcast": lambda x: x[torch.tensor([[0], [-1]], dtype=torch.int32), :, torch.tensor([1, 2, 0])],
    "mask after a slice": lambda x: x[:, torch.tensor([True, False, True, True])],
    "2-d mask after a slice": lambda x: x[:, torch.eye(4, dtype=torch.bool)],
    "transposed self": lambda x: x.transpose(0, 2)[..., torch.tensor([2, 0]), torch.tensor([1, -1])],
}


@pytest.mark.parametrize("mode", ["eager", "compiled"])
@pytest.mark.parametrize("name", sorted(COMPOSED))
def test_composed_index_forms_eager_and_compiled(name, mode):
    from test_cpu import compiled

    torch.manual_seed(0)
    x = torch.randn(3, 4, 4, dtype=torch.float64)
    f = COMPOSED[name]
    want = train_step(f, x)
    with cpu_as_hpu(indexing.install_index) as calls:
        got = train_step(f if mode == "eager" else compiled(f, []), x)
    assert calls["index"] >= 1
    assert_same(got, want, tol=0)


def test_compiled_index_out_of_range_still_raises():
    from test_cpu import compiled

    x, cols = torch.randn(3, 4, 4), torch.tensor([0, 5])  # 5 is out of range, linear index 5 is not
    with pytest.raises(IndexError):
        x[:, torch.tensor([0, 1]), cols]
    with cpu_as_hpu(indexing.install_index), pytest.raises(Exception, match="out of"):
        compiled(lambda x, c: x[:, torch.tensor([0, 1]), c], [])(x, cols)
    with cpu_as_hpu(indexing.install_index) as calls, pytest.raises(IndexError):
        x[:, torch.tensor([0, 1]), cols]  # eager, CPU index tensors: checked on the host, ATen's own error
    assert not calls["index"]


def test_indices_on_the_cpu_of_a_device_tensor_are_moved_and_other_devices_reach_aten():
    x = torch.randn(4, 5, device="meta")  # meta in the role of the device, the CPU index tensors off it
    mask = torch.tensor([True, False, True, False])
    assert indexing._plan(x, [mask, torch.tensor([0, 1])]) == "move"
    assert indexing._plan(x, [torch.tensor([0, 1])]) == "int"  # leading, but on the CPU: Gaudi falls back
    y = torch.randn(4, 5)
    assert indexing._plan(y, [mask.to("meta"), torch.tensor([0, 1])]) is None  # a third device: ATen's error


# ---- Codex adversarial review (REVISION.md): one test per item


class _Record(torch.utils._python_dispatch.TorchDispatchMode):
    """The aten ops that run, with their arguments and outputs."""

    def __init__(self):
        super().__init__()
        self.calls = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        self.calls.append((func, args, out))
        return out


def test_review_1_mask_put_keeps_the_forward_tangent_of_a_value_of_another_dtype():
    mask = torch.tensor([True, False, True])

    def f(v):
        z = torch.zeros(2, 3)
        z[:, mask] = v
        return z

    primal, tangent = torch.tensor(1.0, dtype=torch.float64), torch.tensor(4.0, dtype=torch.float64)
    want = torch.func.jvp(f, (primal,), (tangent,))
    with cpu_as_hpu(indexing.install_mask_fill) as calls:
        got = torch.func.jvp(f, (primal,), (tangent,))
    assert calls["mask_fill"] == 1
    assert torch.equal(got[0], want[0]) and torch.equal(got[1], want[1])  # [[4, 0, 4], [4, 0, 4]]


def test_review_2_int32_negative_index_of_a_dim_past_2_31_wraps_in_int64():
    size = 2 ** 31 + 10
    assert indexing._in_range(torch.tensor([-1, -size], dtype=torch.int64), size)
    assert indexing._in_range(torch.tensor([-1], dtype=torch.int32), size) is True
    x = torch.empty(2, size, device="meta")  # meta in the role of the device: no memory
    with _Record() as rec:
        out = indexing.index_select_flat(x, [None, torch.tensor([-1, 0], dtype=torch.int32)])
    moved = [a[0] for f, a, _ in rec.calls if f == torch.ops.aten._to_copy.default and a[0].device.type == "cpu"
             and a[0].dtype == torch.int64]
    assert out.shape == (2, 2) and moved[-1].tolist() == [size - 1, 0]


def test_review_3_mask_put_with_an_explicit_trailing_none_keeps_atens_alignment():
    m = torch.tensor([True, False])
    for shape in ((2, 2), (2, 3)):
        results = []
        for install in (None, indexing.install_mask_fill):
            z = torch.zeros(shape)
            ctx = cpu_as_hpu(install) if install else contextlib.nullcontext({})
            with ctx as calls:
                try:
                    torch.ops.aten.index_put_(z, [m, None], torch.tensor(1.0))
                    results.append(z)
                except (IndexError, RuntimeError) as e:
                    results.append(type(e))
            assert not calls.get("mask_fill")
        assert (results[0] == results[1]) if isinstance(results[0], type) else torch.equal(*results)


def test_review_4_byte_masks_reach_atens_error():
    z = torch.zeros(2, 3)
    m = torch.tensor([1, 0, 1], dtype=torch.uint8)
    with pytest.raises(Exception) as want:
        torch.ops.aten.index_put_(z.clone(), [None, m], torch.tensor(1.0))
    with cpu_as_hpu(indexing.install_mask_fill) as calls, pytest.raises(type(want.value)):
        torch.ops.aten.index_put_(z.clone(), [None, m], torch.tensor(1.0))
    assert not calls["mask_fill"]


@pytest.mark.parametrize("put", [
    lambda b: b[1:].__setitem__(b[:-1], True),  # the review's case, a leading mask
    lambda b: b.view(2, 4)[:, 1:].__setitem__((slice(None), b[:3]), True),  # a mask after a slice
])
def test_review_5_a_mask_that_aliases_the_destination_keeps_the_native_op(put):
    def run():
        b = torch.tensor([True, False, True, True, False, True, False, False])
        try:
            put(b)
        except RuntimeError as e:
            return str(e)
        return b

    want = run()
    with cpu_as_hpu(indexing.install_mask_fill) as calls:
        got = run()
    assert not calls["mask_fill"]
    assert got == want if isinstance(want, str) else torch.equal(got, want)


def test_review_6_unchecked_index_out_of_its_own_dim_never_reads_a_neighbour():
    x = torch.arange(6.0).reshape(1, 2, 3)  # [None, [0], [3]]: linear index 3 would be row 1, column 0
    with pytest.raises(RuntimeError, match="out of"):
        indexing.index_select_flat(x, [None, torch.tensor([0]), torch.tensor([3])], checked=False)
    with pytest.raises(RuntimeError, match="out of"):
        indexing.index_select_flat(x, [None, torch.tensor([-3]), torch.tensor([0])], checked=False)


def test_review_7_index_tensors_on_a_third_device_keep_the_native_op():
    x = torch.randn(4, 5, requires_grad=True)
    idx = [None, torch.tensor([0, 1], device="meta")]
    assert indexing._plan(x, idx) is None
    with cpu_as_hpu(indexing.install_index) as calls:
        try:
            torch.ops.aten.index.Tensor(x, idx)
        except Exception:  # noqa: BLE001  (whatever ATen does with it)
            pass
    assert not calls["index"]


def test_review_8_separated_indices_copy_no_more_than_the_result():
    x = torch.randn(64, 128, 64, requires_grad=True)
    i = torch.tensor([0, 5, 63])
    with cpu_as_hpu(indexing.install_index) as calls, _Record() as rec:
        out = x[i, :, i]
    assert calls["index"] == 1 and torch.equal(out, x.detach()[i, :, i])
    big = [f for f, _, o in rec.calls if isinstance(o, torch.Tensor) and not o._is_view() and o.numel() >= x.numel() // 4]
    assert not big, big  # no permuted copy of x (2 MB) as before; the result is 3 x 128
    # and the backward is right
    want = torch.autograd.grad((x[i, :, i] ** 2).sum(), x)[0]
    with cpu_as_hpu(indexing.install_index):
        got = torch.autograd.grad((x[i, :, i] ** 2).sum(), x)[0]
    assert torch.equal(got, want)


def test_review_8_leading_mask_put_is_still_rewritten():
    # Not as the review asked: keeping leading mask puts native brought Habana's lazy launch failure back in SPNN and
    # DeepGeoCCA (Gaudi2, measured: the puts of shapes (32, 12) and (1, 16) in one graph with the index rewrites)
    x = torch.randn(64, 128, 64)
    want = x.clone()
    want[want[:, 0, 0] > 0] = 0.0
    with cpu_as_hpu(indexing.install_mask_fill) as calls:
        x[x[:, 0, 0] > 0] = 0.0
    assert calls["mask_fill"] == SCALAR_MASK_PUT and torch.equal(x, want)  # torch 2.15+: masked_fill_, native


@pytest.mark.parametrize("idx", [
    lambda: [None, torch.tensor([0, 2]), None, torch.tensor([1, 0])],
    lambda: [None, torch.tensor([[0], [2]]), None, torch.tensor([1, 0, 3])],
    lambda: [torch.tensor([1]), None, None, torch.tensor([-1])],
    lambda: [None, torch.tensor(1), None, torch.tensor(-2)],
])
def test_review_13_separated_results_have_atens_values_shape_and_strides(idx):
    x = torch.randn(3, 4, 5, 6, dtype=torch.float64, requires_grad=True)
    want = torch.ops.aten.index.Tensor(x, idx())
    with cpu_as_hpu(indexing.install_index) as calls:
        got = torch.ops.aten.index.Tensor(x, idx())
        torch.autograd.gradcheck(lambda x: torch.ops.aten.index.Tensor(x, idx()), (x,))
    assert calls["index"] >= 1
    assert got.shape == want.shape and got.stride() == want.stride() and torch.equal(got, want)


def _import_gaudi_script(name, monkeypatch):
    """A Gaudi script imported on the CPU, with habana_frameworks stubbed (only its pure helpers are called)."""
    import importlib
    import sys
    import types

    for mod in ("habana_frameworks", "habana_frameworks.torch", "habana_frameworks.torch.core",
                "habana_frameworks.torch.hpu", "habana_frameworks.torch.hpu.metrics"):
        stub = types.ModuleType(mod)
        stub.metric_global = lambda *a: None
        monkeypatch.setitem(sys.modules, mod, stub)
    monkeypatch.delitem(sys.modules, name, raising=False)
    return importlib.import_module(name)


def test_review_10_model_parity_fails_a_missing_gradient(monkeypatch):
    mp = _import_gaudi_script("model_parity", monkeypatch)
    g64 = {"w": torch.ones(3), "b": torch.ones(2)}
    eh = mp.rel_errs({"w": torch.ones(3), "b": None}, g64, 1e-3)  # b missing on the HPU
    e32 = mp.rel_errs({"w": torch.ones(3), "b": None}, g64, 1e-3)  # and on CPU float32 too
    assert eh["b"] == float("inf") and e32["b"] == float("inf")
    _, failed = mp.grad_failures(eh, e32)
    assert failed == ["b"]
    _, failed = mp.grad_failures({"w": float("nan")}, {"w": 0.0})
    assert failed == ["w"]


def test_review_11_probes_fail_a_nan_gradient(monkeypatch):
    ai = _import_gaudi_script("adversarial_index", monkeypatch)
    ph = _import_gaudi_script("probe_adversarial_h", monkeypatch)
    nan, ref = torch.tensor([1.0, float("nan")]), torch.tensor([1.0, 2.0])
    for err in (ai.err, ph.err):
        e = err(nan, ref)
        assert not e <= 1e-5  # NaN fails
        assert max(0.0, e) == 0.0  # why the probes no longer take max() over a list with a NaN in it
    assert ph.err(torch.tensor([float("nan")]), torch.tensor([float("nan")])) == 0.0  # the same NaN as the reference


def test_compiled_mask_put_of_an_overflowing_value_keeps_atens_error():
    # Gaudi2 compiled: a fake full() skipped ATen's overflow check, and the rewrite wrote inf
    from test_cpu import compiled

    def put(x):
        y = x * 1
        y[:, MASK] = torch.tensor(1e6)
        return y

    x = torch.zeros(3, 4, dtype=torch.float16)
    with pytest.raises(RuntimeError, match="overflow"):
        put(x)
    with cpu_as_hpu(indexing.install_mask_fill) as calls, pytest.raises(Exception, match="overflow"):
        compiled(put, [])(x)
    assert not calls["mask_fill"]


def test_mask_and_value_on_another_device_never_alias(monkeypatch):
    # Gaudi2 lazy: _is_alias_of raised on HPU tensors, so spd_learn's CPU mask put was declined and SPNN and
    # DeepGeoCCA hit Habana's "Sections' validation failure" again
    def no_alias(a, b):
        raise RuntimeError("no storage")

    monkeypatch.setattr(torch._C, "_is_alias_of", no_alias)
    x = torch.ones(2, 3, device="meta")
    assert not indexing._shares_memory(x, torch.tensor([True, False, True]))
    assert not indexing._shares_memory(x, torch.tensor(2 ** 0.5))
    y = torch.ones(2, 3)
    assert indexing._shares_memory(y, y[0])  # storage comparison when _is_alias_of cannot tell


# ---- Codex adversarial review of PR #24 (REVISION-B.md): one test per item, against the native op


@pytest.mark.usefixtures("cpu_indices_off_device")
@pytest.mark.parametrize("idx", [
    lambda: (torch.tensor([0]), torch.tensor([3])),  # linear index 3 is x[1, 0]
    lambda: (torch.tensor([1]), torch.tensor([-4])),  # linear index 2 is x[0, 2]
    lambda: (torch.tensor([3]),),
    lambda: (slice(None), torch.tensor([0]), torch.tensor([3])),
])
def test_b1_index_out_of_its_own_dim_raises_atens_index_error(idx):
    x = torch.arange(6.0).reshape(2, 3) if len(idx()) < 3 else torch.arange(12.0).reshape(2, 2, 3)
    with pytest.raises(IndexError) as want:
        x[idx()]
    with cpu_as_hpu(indexing.install_index) as calls, pytest.raises(IndexError) as got:
        x[idx()]
    assert str(got.value) == str(want.value) and not calls["index"]


@pytest.mark.parametrize("call", [
    lambda x: F.unfold(x, (2, 1), stride=(2, 1)),  # vertical fast path
    lambda x: F.unfold(x.transpose(2, 3), (1, 2), stride=(1, 2)),  # horizontal fast path
    lambda x: F.fold(x.reshape(2, 6, 2), (4, 1), (2, 1), stride=(2, 1)),
])
def test_b2_unfold_and_fold_never_return_a_view_of_the_input(call):
    from hpu_ops import im2col

    x = torch.randn(2, 3, 4, 1)
    want = call(x.clone())
    with cpu_as_hpu(im2col.install) as calls:
        before = x.clone()
        out = call(x)
        out.zero_()
    assert calls["im2col"] + calls["col2im"] == 1
    assert torch.equal(x, before) and out.is_contiguous() and want.shape == out.shape


def test_b2_out_aliasing_the_input_raises_before_writing():
    from hpu_ops import im2col

    x = torch.randn(2, 3, 4, 4)
    before = x.clone()
    with cpu_as_hpu(im2col.install), torch.inference_mode(False):
        kernel = im2col._im2col_out_kernel("cpu", torch.ops.aten.im2col.out)
        with pytest.raises(RuntimeError, match="single memory location"):
            kernel(x, [2, 2], [1, 1], [0, 0], [1, 1], out=x)
    assert x.shape == before.shape and torch.equal(x, before)


def test_b3_padded_fold_is_contiguous_and_holds_no_padded_storage():
    from hpu_ops import im2col

    y = torch.randn(2, 12, 25)
    want = F.fold(y, (4, 4), (2, 2), padding=1)
    with cpu_as_hpu(im2col.install) as calls:
        got = F.fold(y, (4, 4), (2, 2), padding=1)
    assert calls["col2im"] == 1 and got.is_contiguous() and got.view(-1).numel() == want.numel()
    assert got.untyped_storage().nbytes() == got.numel() * got.element_size()
    torch.testing.assert_close(got, want)


def test_b4_separated_indices_of_an_expanded_input_copy_no_more_than_the_result():
    base = torch.randn(1, 8, 8, requires_grad=True)
    i = torch.tensor([0, 3, 7])

    def f(b):
        return b.expand(64, 64, 8, 8)[:, i, :, i]  # 4 MiB expanded, 256 B of data

    want = train_step(f, base)
    with cpu_as_hpu(indexing.install_index) as calls:
        with _Record() as rec:  # the forward; the backward of an expand has the expanded size, natively too
            f(base)
        got = train_step(f, base)
    big = [fn for fn, _, o in rec.calls if isinstance(o, torch.Tensor) and not o._is_view() and o.numel() > 64 * 3 * 64]
    assert calls["index"] == 2 and not big, big
    assert_same(got, want, tol=0)


def test_b5_empty_batch_builds_no_index_and_large_calls_keep_the_native_op(monkeypatch):
    from hpu_ops import im2col

    x = torch.randn(0, 3, 9, 9, requires_grad=True)
    want = train_step(lambda t: F.unfold(t, (3, 3), padding=1), x)
    with cpu_as_hpu(im2col.install) as calls, _Record() as rec:
        got = train_step(lambda t: F.unfold(t, (3, 3), padding=1), x)
    assert calls["im2col"] == 1 and not [f for f, _, _ in rec.calls if f == torch.ops.aten.arange.start_step]
    assert_same(got, want, tol=0)
    monkeypatch.setattr(im2col, "MAX_INDEX", 8)
    y = torch.randn(2, 3, 9, 9)
    with cpu_as_hpu(im2col.install) as calls:
        torch.testing.assert_close(F.unfold(y, (3, 3)), F.unfold(y.clone(), (3, 3)))
    assert not calls["im2col"]


def test_b5_unfold_index_is_built_on_the_device_while_tracing():
    from hpu_ops import im2col

    idx = im2col._flat_indices(9, 9, 3, 3, 1, 1, 0, 0, 1, 1, torch.device("meta"), tracing=True)[0]
    assert idx.device.type == "meta"
    idx = im2col._flat_indices(9, 9, 3, 3, 1, 1, 0, 0, 1, 1, torch.device("meta"), tracing=False)[0]
    assert idx.device.type == "meta"  # built on the host, then moved once


def test_b6_byte_masks_off_the_device_are_moved_and_compared_to_native(monkeypatch):
    xm = torch.randn(4, 5, device="meta")
    assert indexing._plan(xm, [torch.tensor([1, 0, 1, 0], dtype=torch.uint8)]) == "move"
    monkeypatch.setattr(indexing, "_off_device", lambda self, t: True)
    x = torch.randn(4, 5)
    for mask in (torch.tensor([1, 0, 1, 1], dtype=torch.uint8), torch.tensor([True, False, True, True])):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # ATen's byte-mask deprecation
            want = train_step(lambda t: t[mask], x)
            with cpu_as_hpu(indexing.install_index) as calls:
                got = train_step(lambda t: t[mask], x)
        assert calls["index"] == 1
        assert_same(got, want, tol=0)
