"""CPU tests: ``linalg._decline_traced`` declines a call only while a tracer sees it on ``hpu`` (so a
``torch.compile`` graph keeps the aten op and the TPC kernel runs at run time, outside Habana's recipe); the CPU
stand-in and eager calls are taken as before. The recipe failure itself is Gaudi-only (#60, gated on Voyager)."""
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from hpu_ops import linalg


def _always(*args, **kwargs):
    return True


def test_hpu_traced_declines_eager_takes():
    takes = linalg._decline_traced("hpu", _always)
    A = torch.eye(4)
    assert takes(A)
    with FakeTensorMode():
        fake = torch.empty(4, 4, device="cpu")
        assert not takes(fake)


def test_cpu_standin_never_declines():
    takes = linalg._decline_traced("cpu", _always)
    assert takes(torch.eye(4))
    with FakeTensorMode():
        assert takes(torch.empty(4, 4, device="cpu"))


def test_matrix_at_and_inner_predicate():
    never = linalg._decline_traced("hpu", lambda *a, **k: False, matrix_at=1)
    assert not never(torch.eye(4), torch.eye(4))
    takes = linalg._decline_traced("hpu", _always, matrix_at=1)
    assert takes(torch.eye(4), torch.eye(4))
    with FakeTensorMode():
        assert not takes(torch.empty(4, 4), torch.empty(4, 4))
