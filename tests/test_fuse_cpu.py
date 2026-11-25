"""hpu_ops.fuse on the CPU: Sequential Conv+BatchNorm folds to the same eval function."""
import copy

import pytest
import torch

import hpu_ops

from test_cpu import assert_same


def _model(conv_bias=True, affine=True):
    torch.manual_seed(0)
    m = torch.nn.Sequential(
        torch.nn.Conv2d(3, 8, 3, padding=1, bias=conv_bias),
        torch.nn.BatchNorm2d(8, affine=affine),
        torch.nn.ReLU(),
    ).double()
    with torch.no_grad():
        m[1].running_mean.uniform_(-1, 1)
        m[1].running_var.uniform_(0.5, 2)
        if affine:
            m[1].weight.uniform_(0.5, 1.5)
            m[1].bias.uniform_(-0.5, 0.5)
    return m.eval()


def _check(m):
    x = torch.randn(2, 3, 10, 10, dtype=torch.float64)
    want = copy.deepcopy(m)(x)
    got = hpu_ops.fuse(m)(x)
    assert_same([got], [want], tol=1e-12)


def test_fold_parity():
    _check(_model())


def test_fold_no_conv_bias_no_affine():
    _check(_model(conv_bias=False, affine=False))


def test_training_refused_and_nonsequential_untouched():
    m = _model()
    m.train()
    with pytest.raises(RuntimeError, match="eval mode"):
        hpu_ops.fuse(m)

    class Attr(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 4, 1)
            self.bn = torch.nn.BatchNorm2d(4)

        def forward(self, x):
            return self.bn(self.conv(x))

    a = Attr().eval()
    w = a.conv.weight.clone()
    hpu_ops.fuse(a)
    assert torch.equal(a.conv.weight, w)
