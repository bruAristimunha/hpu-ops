"""CPU tests of the instance_norm composite, "cpu" standing in for "hpu": Habana's inference
recipe fails on EpiNT's instance norm, so the composite computes it (traced, autograd and
backend keys); a training call that must update running statistics declines to ATen, which
owns the in-place buffer update."""
import pytest
import torch
import torch.nn.functional as F

import hpu_ops
from hpu_ops import norm

from test_cpu import assert_same, compiled, cpu_as_hpu


def _module():
    torch.manual_seed(0)
    m = torch.nn.InstanceNorm1d(16, affine=True).double()
    with torch.no_grad():
        m.weight.uniform_(0.5, 1.5)
        m.bias.uniform_(-0.5, 0.5)
    return m


def test_inference_and_training_parity():
    m = _module()
    torch.manual_seed(1)
    x = torch.randn(4, 16, 100, dtype=torch.float64)
    with cpu_as_hpu(norm.install_instance_norm) as calls:
        m.eval()
        with torch.inference_mode():
            got = m(x)
        assert calls["instance_norm"]
        m.train()
        xg = x.clone().requires_grad_()
        out = m(xg)
        out.square().mean().backward()
        got_train = [out.detach(), xg.grad, m.weight.grad.clone(), m.bias.grad.clone()]
    m.eval()
    with torch.inference_mode():
        want = m(x)
    m.train()
    m.zero_grad()
    xw = x.clone().requires_grad_()
    outw = m(xw)
    outw.square().mean().backward()
    want_train = [outw.detach(), xw.grad, m.weight.grad.clone(), m.bias.grad.clone()]
    assert_same([got], [want], tol=1e-12)
    assert_same(got_train, want_train, tol=1e-10)


def test_compiled_inference_traces_composite():
    m = _module()
    x = torch.randn(4, 16, 100, dtype=torch.float64)
    nodes = []
    with cpu_as_hpu(norm.install_instance_norm) as calls:
        g = compiled(m.eval(), nodes)
        with torch.inference_mode():
            got = g(x)
        assert calls["instance_norm"]
    with torch.inference_mode():
        want = m(x)
    assert_same([got], [want], tol=1e-12)


def test_train_running_stats_update_stays_native():
    m = torch.nn.InstanceNorm1d(4, affine=False, track_running_stats=True, momentum=0.5).double()
    x = torch.randn(2, 4, 3, dtype=torch.float64)
    with cpu_as_hpu(norm.install_instance_norm) as calls:
        m.train()
        m(x)
        assert not calls["instance_norm"]
    assert not torch.equal(m.running_mean, torch.zeros(4, dtype=torch.float64))


def test_bogus_running_stats_decline():
    x = torch.randn(2, 4, 10, dtype=torch.float64)
    with cpu_as_hpu(norm.install_instance_norm) as calls:
        with torch.inference_mode(), pytest.raises(RuntimeError, match="should contain"):
            F.instance_norm(x, torch.zeros(99), torch.ones(99), None, None, True, 0.1, 1e-5)
        assert not calls["instance_norm"]


def test_running_stats_path():
    torch.manual_seed(0)
    x = torch.randn(2, 8, 50, dtype=torch.float64)
    mean = torch.randn(8, dtype=torch.float64)
    var = torch.rand(8, dtype=torch.float64) + 0.5
    with cpu_as_hpu(norm.install_instance_norm):
        with torch.inference_mode():
            got = F.instance_norm(x, mean, var, None, None, False, 0.1, 1e-5)
    with torch.inference_mode():
        want = F.instance_norm(x, mean, var, None, None, False, 0.1, 1e-5)
    assert_same([got], [want], tol=1e-12)
