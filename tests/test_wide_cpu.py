"""CPU tests of hpu_ops.wide, the checks at the copy of int64 and float64 data to the card: "cpu" stands in for "hpu"
(ATen's CPU has no copy_ kernel at the CPU key either, so the check sits where it sits on eager Gaudi)."""
import warnings

import pytest
import torch

from hpu_ops import wide

from test_small_cpu import cpu_as_hpu


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    wide._WARNED.clear()
    monkeypatch.delenv("HPU_OPS_INT64_CHECK", raising=False)
    monkeypatch.delenv("HPU_OPS_COPY_CHECKS", raising=False)
    yield
    wide._WARNED.clear()


def _copy(src, dtype=None):
    dst = torch.empty(src.shape, dtype=dtype or src.dtype)
    return dst.copy_(src)


@pytest.mark.parametrize("value", [2 ** 31, -(2 ** 31) - 1, 2 ** 40 + 1])
def test_int64_outside_int32_warns_once_and_copies_exactly(value):
    src = torch.tensor([1, value, 3])
    with cpu_as_hpu(wide.install_copy_checks):
        with pytest.warns(UserWarning, match="outside the int32 range"):
            out = _copy(src)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            again = src.to(torch.int64, copy=True) + 0
            _copy(torch.tensor([2 ** 50]))  # warned once: no scan, no second warning
    assert torch.equal(out, src) and torch.equal(again, src)


@pytest.mark.parametrize("src", [torch.tensor([2 ** 31 - 1, -(2 ** 31)]), torch.zeros(0, dtype=torch.long),
                                 torch.arange(10), torch.tensor([2 ** 40], dtype=torch.float64).long().int()])
def test_int64_within_int32_and_other_dtypes_do_not_warn(src):
    with cpu_as_hpu(wide.install_copy_checks), warnings.catch_warnings():
        warnings.simplefilter("error")
        out = _copy(src)
    assert torch.equal(out, src)


def test_int64_check_raise_and_off(monkeypatch):
    src = torch.tensor([2 ** 33])
    monkeypatch.setenv("HPU_OPS_INT64_CHECK", "raise")
    with cpu_as_hpu(wide.install_copy_checks), pytest.raises(RuntimeError, match="outside the int32 range"):
        _copy(src)
    monkeypatch.setenv("HPU_OPS_INT64_CHECK", "0")
    with cpu_as_hpu(wide.install_copy_checks), warnings.catch_warnings():
        warnings.simplefilter("error")
        assert torch.equal(_copy(src), src)


def test_int64_check_reads_views_and_casts():
    base = torch.tensor([[0, 2 ** 35], [1, 2]])
    with cpu_as_hpu(wide.install_copy_checks), pytest.warns(UserWarning, match=r"\[2, 34359738368\]"):
        out = _copy(base.t()[1])  # a strided view holding the large value
    assert torch.equal(out, base.t()[1])


def test_float64_destination_warns_once():
    with cpu_as_hpu(wide.install_copy_checks):
        with pytest.warns(UserWarning, match="twice the offset"):
            out = _copy(torch.arange(4.0), torch.float64)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _copy(torch.arange(4.0), torch.float64)
            _copy(torch.arange(4.0))  # float32: never
    assert torch.equal(out, torch.arange(4.0, dtype=torch.float64))


def test_copy_with_a_python_scalar_and_through_to_and_compile():
    with cpu_as_hpu(wide.install_copy_checks):
        x = torch.zeros(3).fill_(2.0)
        y = torch.zeros(3)
        y.copy_(torch.tensor(5.0))
        z = torch.compile(lambda t: t.to(torch.float16) + 1, backend="aot_eager")(torch.arange(3.0))
    assert torch.equal(x, torch.full((3,), 2.0)) and torch.equal(y, torch.full((3,), 5.0))
    assert torch.equal(z, torch.arange(3.0).half() + 1)


def test_undo_removes_the_kernel():
    undo = wide.install_copy_checks("cpu")
    assert torch._C._dispatch_has_kernel_for_dispatch_key("aten::copy_", "CPU")
    undo()
    assert not torch._C._dispatch_has_kernel_for_dispatch_key("aten::copy_", "CPU")


def test_int64_cast_to_another_dtype_and_card_sources_are_not_scanned():
    big = torch.tensor([2 ** 40])
    with cpu_as_hpu(wide.install_copy_checks), warnings.catch_warnings():
        warnings.simplefilter("error")
        out = _copy(big, torch.float32)  # a cast: the card gets float32
    assert out.item() == 2.0 ** 40


def test_copy_under_inference_mode_and_no_grad_still_checks():
    with cpu_as_hpu(wide.install_copy_checks), torch.inference_mode():
        with pytest.warns(UserWarning, match="outside the int32 range"):
            _copy(torch.tensor([-(2 ** 40)]))


# ---- Codex review (PR #40) ----------------------------------------------------------------------------------------

def test_review3_another_threads_scan_does_not_skip_this_threads_check(monkeypatch):
    """Thread A is inside its own scan (its aminmax held on an event) while thread B copies 2**40 in raise mode."""
    import threading

    monkeypatch.setenv("HPU_OPS_INT64_CHECK", "raise")
    scanning, release = threading.Event(), threading.Event()
    aminmax = torch.aminmax

    def held(t, *args, **kwargs):
        if threading.current_thread().name == "scanner":
            scanning.set()
            release.wait(10)
        return aminmax(t, *args, **kwargs)

    monkeypatch.setattr(torch, "aminmax", held)
    with cpu_as_hpu(wide.install_copy_checks):
        a = threading.Thread(target=lambda: _copy(torch.zeros(1000, dtype=torch.long)), name="scanner")
        a.start()
        assert scanning.wait(10)
        try:
            with pytest.raises(RuntimeError, match="outside the int32 range"):
                _copy(torch.tensor([2 ** 40]))
        finally:
            release.set()
            a.join()


def test_review10_named_tensor_copies_need_the_checks_off(monkeypatch):
    named = torch.tensor([7.0, 13.0], names=("N",))
    with cpu_as_hpu(wide.install_copy_checks), pytest.raises(RuntimeError, match="Named tensors"):
        torch.empty(2).copy_(named)  # a limit of Python kernels, documented
    monkeypatch.setenv("HPU_OPS_COPY_CHECKS", "0")
    with cpu_as_hpu(wide.install_copy_checks):
        out = torch.empty(2).copy_(named)
    assert out.tolist() == [7.0, 13.0]


def test_review15_empty_broadcast_passes_and_a_python_int_is_checked(monkeypatch):
    monkeypatch.setenv("HPU_OPS_INT64_CHECK", "raise")
    with cpu_as_hpu(wide.install_copy_checks):
        torch.empty(0, dtype=torch.long).copy_(torch.tensor([2 ** 40]))
        with pytest.raises(RuntimeError, match="outside the int32 range"):
            torch.empty(1, dtype=torch.long).copy_(2 ** 40)
        assert torch.empty(1, dtype=torch.long).copy_(5).item() == 5


def test_review14_an_empty_float64_copy_does_not_use_up_the_warning():
    with cpu_as_hpu(wide.install_copy_checks):
        torch.empty(0, dtype=torch.float64).copy_(torch.empty(0))
        with pytest.warns(UserWarning, match="twice the offset"):
            _copy(torch.arange(3.0), torch.float64)


def test_rereview_float_sources_copied_into_int64_are_scanned(monkeypatch):
    monkeypatch.setenv("HPU_OPS_INT64_CHECK", "raise")
    with cpu_as_hpu(wide.install_copy_checks):
        with pytest.raises(RuntimeError, match="outside the int32 range"):
            torch.empty(2, dtype=torch.long).copy_(torch.tensor([1.5, 3e9]))
        out = torch.empty(3, dtype=torch.long).copy_(torch.tensor([1.9, -2.9, float("nan")]))
    assert out[:2].tolist() == [1, -2]
