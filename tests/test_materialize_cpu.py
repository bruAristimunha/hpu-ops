"""hpu_ops.materialize on the CPU: lazy parameters allocated, plain models untouched (#68)."""
import pytest
import torch
from torch.nn.parameter import UninitializedParameter

import hpu_ops


def _uninit(model):
    return [n for n, p in model.named_parameters() if isinstance(p, UninitializedParameter)]


def test_lazy_linear_allocated():
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.LazyLinear(4))
    assert _uninit(model)
    out = hpu_ops.materialize(model, torch.randn(2, 3, 5))
    assert out is model
    assert _uninit(model) == []
    assert model[1].weight.shape == (4, 15)
    assert model(torch.randn(2, 3, 5)).shape == (2, 4)


def test_plain_model_no_forward():
    model = torch.nn.Linear(3, 2)
    weight = model.weight
    assert hpu_ops.materialize(model, torch.randn(1, 3)) is model
    assert model.weight is weight


def test_compile_materializes_and_wraps():
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.LazyLinear(4))
    with pytest.raises(Exception, match="hpu_backend"):
        hpu_ops.compile(model, torch.randn(2, 3, 5), device="cpu")
    assert _uninit(model) == []
    assert model[1].weight.shape == (4, 15)


def test_segmented_is_noop_off_gaudi():
    model = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.ReLU())
    out, undo = hpu_ops.segmented(model)
    assert out is model
    undo()  # no hooks to remove, no habana stack to mark steps on
    assert model(torch.randn(1, 3)).shape == (1, 2)


def test_segmented_deep_hooks_every_submodule(monkeypatch):
    import sys
    import types

    marks = []
    fake = types.ModuleType("habana_frameworks.torch.core")
    fake.mark_step = lambda: marks.append(1)
    for name in ("habana_frameworks", "habana_frameworks.torch"):
        pkg = types.ModuleType(name)
        pkg.__path__ = []
        monkeypatch.setitem(sys.modules, name, pkg)
    monkeypatch.setitem(sys.modules, "habana_frameworks.torch.core", fake)
    monkeypatch.setattr(hpu_ops, "_lazy", lambda: True)
    model = torch.nn.Sequential(torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.ReLU()),
                                torch.nn.Linear(2, 1))
    out, undo = hpu_ops.segmented(model, deep=True)
    assert out is model
    model(torch.randn(1, 3))
    assert len(marks) == 4  # inner Sequential, its Linear and ReLU, and the outer Linear
    undo()
    marks.clear()
    model(torch.randn(1, 3))
    assert marks == []


def test_mixed_train_eval_modes_preserved():
    model = torch.nn.Sequential(torch.nn.BatchNorm1d(4), torch.nn.LazyLinear(2))
    model.train()
    model[0].eval()
    assert model.training and not model[0].training
    hpu_ops.materialize(model, torch.randn(2, 4))
    assert model.training and not model[0].training
    assert model[1].weight.shape == (2, 4)


def test_lazy_batchnorm_buffers_allocated():
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.LazyBatchNorm1d())
    hpu_ops.materialize(model, torch.randn(2, 3, 5))
    assert model[1].running_mean.shape == (15,)


def test_unreached_lazy_module_raises():
    class Partial(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.used = torch.nn.LazyLinear(2)
            self.unused = torch.nn.LazyLinear(2)

        def forward(self, x):
            return self.used(x)

    with pytest.raises(RuntimeError, match="unused"):
        hpu_ops.materialize(Partial(), torch.randn(1, 3))
