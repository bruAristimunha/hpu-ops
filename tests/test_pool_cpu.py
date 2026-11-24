"""CPU tests of the wide-kernel max_pool1d composite, "cpu" standing in for "hpu": Habana's kernel
fails its graph compile when the kernel is wider than the input (ceil_mode, kernel 2 over length 1,
GCNsNet's last pooling); ATen returns the window max."""
import torch
import torch.nn.functional as F

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
