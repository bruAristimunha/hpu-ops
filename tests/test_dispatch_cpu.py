"""Undo safety in hpu_ops._dispatch: an undo runs at most once, so a stale undo cannot
remove a later install of the same op."""
import torch

from hpu_ops import composite


def _has_renorm():
    return torch._C._dispatch_has_kernel_for_dispatch_key("aten::renorm", "AutogradCPU")


def test_double_and_stale_undo():
    undo = composite.install_renorm("cpu")
    assert _has_renorm()
    undo()
    undo()  # no second destroy, no raise
    assert not _has_renorm()
    new_undo = composite.install_renorm("cpu")
    try:
        assert _has_renorm()
        undo()  # stale: the new install stays
        assert _has_renorm()
    finally:
        new_undo()
    assert not _has_renorm()
