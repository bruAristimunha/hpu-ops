"""``aten::_assert_async`` on the device as a host check (#96).

Gaudi has no kernel for ``_assert_async`` (bridge 1.21.4: ``_dispatch_has_kernel_for_dispatch_key``
is False at ``HPU`` and ``AutogradHPU`` for both overloads), so every call -- Dynamo's guard asserts,
``torch._assert_async`` in user code, quantile's traced range check -- ran through Habana's CPU
fallback (one bridge fallback per call).

The check itself is a single host read, so the kernel here reads the one element on the host and
raises exactly as ATen's CPU kernel does. It is registered at the device's backend key only, below
autograd: the op returns nothing and needs no derivative, and a kernel at the autograd key makes
AOT autograd drop the node from compiled graphs (measured on the CPU stand-in: with an
``AutogradCPU`` kernel the forward graph keeps only the ops around the assert; with the backend
kernel alone the ``aten._assert_async.msg`` node stays and the compiled callable still raises at
run time). Eager, ``no_grad`` and inference calls all reach the backend kernel.

Under tracing (fake or functional tensors, ``torch.compile`` capture, a proxy mode) the kernel is a
no-op, which is ATen's own Meta kernel for this op: a host read there is a data-dependent error
(``aten._local_scalar_dense`` on a fake tensor), and the graph keeps the node for the backend to
run. The host read is wrapped as one strict-mode ``host sync`` row (``_strict.host_read``): a rare
assert costs one sync, never a bridge fallback.
"""
import torch

from . import _strict
from ._dispatch import count, override_kernels, tracing

aten = torch.ops.aten
_DEFAULT_MSG = "Expected Tensor with single nonzero value, but got zero"
_EMPTY_MSG = "Assertion is failed"


def _check(self, message):
    """ATen's ``_assert_async`` check on a concrete device tensor: ``message`` on a zero element."""
    count("assert_async")
    if self.numel() != 1:
        if self.numel() == 0:
            raise RuntimeError("Boolean value of Tensor with no values is ambiguous")
        raise RuntimeError("Boolean value of Tensor with more than one value is ambiguous")
    with _strict.host_read("aten::_assert_async.msg", self):
        value = self.item()
    if not value:
        raise RuntimeError(message)


def _msg_kernel(self, assert_msg):
    if tracing(self):
        return None
    _check(self, assert_msg if assert_msg else _EMPTY_MSG)
    return None


def _default_kernel(self):
    if tracing(self):
        return None
    _check(self, _DEFAULT_MSG)
    return None


def install_assert_async(device_type):
    """``_assert_async`` (both overloads) on ``device_type`` as a host check, at the backend key.
    Returns an undo. The backend key is overridden, as in ``small.install_scalar_dense``: on the
    CPU stand-in ATen's kernel already sits there, and inference-mode calls must still reach this
    kernel; on Gaudi the key is empty."""
    return override_kernels(device_type, [
        (aten._assert_async.msg, _msg_kernel),
        (aten._assert_async.default, _default_kernel),
    ])
