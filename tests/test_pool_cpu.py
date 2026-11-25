"""CPU tests of the wide-kernel max_pool1d composite, "cpu" standing in for "hpu": Habana's kernel
fails its graph compile when the kernel is wider than the input (ceil_mode, kernel 2 over length 1,
GCNsNet's last pooling); ATen returns the window max."""
import contextlib

import torch
import torch.nn.functional as F


def _null():
    return contextlib.nullcontext()

import hpu_ops
from hpu_ops import composite

from test_cpu import assert_same, cpu_as_hpu, train_step


def _check(shape, kernel, stride=None, padding=0, ceil_mode=True):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float64)
    with cpu_as_hpu(composite.install_max_pool1d) as calls:
        got = train_step(lambda t: F.max_pool1d(t, kernel, stride, padding, 1, ceil_mode), x)
    want = train_step(lambda t: F.max_pool1d(t, kernel, stride, padding, 1, ceil_mode), x)
    assert_same(got, want, tol=1e-12)
    return calls


def test_length1_kernel2():
    calls = _check((4, 512, 1), 2)
    assert calls["max_pool1d_wide"]


def test_wider_kernel_and_stride_windows():
    _check((2, 8, 3), 5)  # one window covering all three
    _check((2, 8, 3), 4, stride=2)  # windows [0:3], [2:3] past padding
    _check((2, 8, 4), 3, stride=1, padding=1)


def test_narrow_kernel_keeps_native():
    calls = _check((2, 8, 6), 2)
    assert not calls["max_pool1d_wide"]


def test_tie_and_nan_gradients_match_native():
    for values in ([5.0, 5.0], [float("nan"), 1.0]):
        grads = []
        for install in (True, False):
            x = torch.tensor([[values]], dtype=torch.float64, requires_grad=True)
            ctx = cpu_as_hpu(composite.install_max_pool1d) if install else _null()
            with ctx:
                F.max_pool1d(x, 3, ceil_mode=True).sum().backward()
            grads.append(x.grad)
        torch.testing.assert_close(grads[0], grads[1], atol=1e-12, rtol=1e-12, equal_nan=True)


def test_invalid_padding_declines():
    x = torch.randn(1, 1, 1)
    with cpu_as_hpu(composite.install_max_pool1d) as calls:
        try:
            F.max_pool1d(x, 2, padding=5, ceil_mode=True)
        except RuntimeError:
            pass
        assert not calls["max_pool1d_wide"]
