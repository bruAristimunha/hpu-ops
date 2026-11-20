"""Adversarial round 2 of the indexing rewrites (install_index, install_mask_fill), on the CPU standing in for HPU.
The first tests are the round-2 breaks (mask puts that differ from ATen's index_put_), fixed by keeping the native op
for a value of another dtype and for a destination whose elements overlap; the rest widen the coverage."""
import pytest
import torch

from hpu_ops import indexing
from test_cpu import assert_same, cpu_as_hpu, train_step

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
        assert calls["mask_fill"] == 1 and torch.equal(got, want)


def test_index_rewrite_under_tracing_depends_on_the_index_device():
    # While torch.compile traces, the values of a CPU index tensor cannot be checked: the native op takes it. An index
    # tensor on the device is never read on the host, traced or not, so the rewrite is traced (measured right on
    # Gaudi2, negative indices too).
    from torch._subclasses.fake_tensor import FakeTensorMode

    assert indexing._in_range(torch.empty(3, dtype=torch.long, device="meta"), 4)
    with FakeTensorMode():
        assert not indexing._in_range(torch.tensor([0, 1]), 4)


# ---- coverage that passes


@pytest.mark.parametrize("dtype", [torch.int64, torch.int32, torch.int8, torch.uint8, torch.bool, torch.complex64])
def test_index_rewrite_of_integer_bool_and_complex_data(dtype):
    torch.manual_seed(0)
    x = (torch.randn(5, 6, 6) * 50).to(dtype) if dtype != torch.complex64 else torch.randn(5, 6, 6, dtype=dtype)
    r, c = torch.triu_indices(6, 6)
    with cpu_as_hpu(indexing.install_index) as calls:
        got = x[..., r, c]
    assert calls["index_select_flat"] == 1
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
    assert torch.equal(v.grad, gv_want)


def test_duplicate_indices_accumulate_in_the_backward():
    x = torch.randn(512, 6, 6, requires_grad=True)
    idx = torch.tensor([0, 0, 0, 1, 2, 2])
    w = torch.linspace(0.5, 1.5, 512 * 6).reshape(512, 6)
    (x[..., idx, idx] * w).sum().backward()
    want, x.grad = x.grad.clone(), None
    with cpu_as_hpu(indexing.install_index) as calls:
        (x[..., idx, idx] * w).sum().backward()
    assert calls["index_select_flat"] == 1
    assert torch.allclose(x.grad, want, rtol=1e-6, atol=0)
