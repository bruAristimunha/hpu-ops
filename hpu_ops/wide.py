"""Checks for the 8-byte dtypes at the copy to the card, eager mode (Gaudi2, bridge 1.21.4; README, "float64").

Neither bug can be fixed from the dispatcher at a cost a model would accept; both give wrong values without an error,
so the copy that puts such data on the card warns once (``repro/float64_storage_offset_eager.py``,
``repro/probe_offset8.py``, ``repro/probe_f64_int64.py``):

- float64. The card holds float32 values in an 8-byte-per-element allocation. In eager mode every op on a
  *contiguous* float64 view with a storage offset (``x[1:]``, ``x[i]``, ``narrow``, ``split``, ``unbind``,
  ``as_strided`` with an offset) reads and writes at twice the offset: ``h[1:] + 0`` of ``[1, ..., 6]`` gives
  ``[3, 4, 5, 6, 0]``, ``h[1:4].copy_(v)`` writes elements 2 to 4, the gradient of a slice is zeros, ``clone`` and
  ``copy_`` read wrong too. Non-contiguous views (``x[:, 1:]``, ``x[1::2]``, a transposed row), ``.cpu()``, lazy mode
  and the other dtypes (int64 included) are right. A fix would have to give every op on such a view a copy made
  through a strided read, and route every in-place write back the same way; at view creation a copy breaks view
  semantics. float64 on the card is float32 anyway: hpu-ops warns and the README says to keep float64 on the CPU.
- int64. Eager kernels compute in 32 bits: arithmetic, ``sum`` / ``cumsum`` / ``max``, ``sort``, ``where``,
  ``index_select``, shifts, ``>`` read the low 32 bits of each value (``(2**40 + 1) + 1`` gives 2). Copies, ``clone``,
  views, ``cat``, ``==`` and ``.float()`` keep all 64 bits. Lazy mode raises at the copy instead ("Error when
  trying to cast Long to Int"). In both modes a result past the int32 range from in-range inputs wraps or saturates
  (``2**31 - 1 + 1`` gives ``2**31 - 1``, ``2**20 * 2**20`` gives 0) and float32 to int64 saturates at the int32
  bounds: no copy-time check can see that. Here a CPU int64 tensor copied into an int64 card tensor is scanned on
  the host (one ``aminmax``) until the first out-of-range one is found; ``HPU_OPS_INT64_CHECK=raise`` raises instead
  of warning, ``=0`` skips the scan.

Registered at the ``HPU`` key of ``copy_``, where eager Habana has no kernel (``copy_`` resolves to ATen's
composite there, which the kernel calls); lazy mode registers its own ``copy_``, so nothing is installed there.
"""
import os
import warnings

import torch
from torch._C import DispatchKey

from ._dispatch import register_kernels

aten = torch.ops.aten
INT32_MIN, INT32_MAX = -(1 << 31), (1 << 31) - 1
_WARNED = set()
_SCANNING = []

FLOAT64_WARNING = (
    "A float64 tensor was copied to the HPU. On Gaudi2 (bridge 1.21.4) it holds float32 values, and in eager mode an "
    "op on a contiguous float64 view with a storage offset (x[1:], x[i], narrow, split, unbind) reads and writes at "
    "twice the offset: wrong values and gradients without an error. Keep float64 work on the CPU, or use float32 "
    "(lazy mode reads such views right). See hpu-ops' README, \"float64\".")
INT64_WARNING = (
    "An int64 tensor with values in [{lo}, {hi}], outside the int32 range, was copied to the HPU. Gaudi2 (bridge "
    "1.21.4) computes int64 in 32 bits: in eager mode arithmetic, reductions, comparisons other than ==, sort and "
    "indexing read the low 32 bits of each value (copies, views, cat, == and .float() keep all 64). Lazy mode "
    "raises on such values. In both modes results past the int32 range wrap or saturate. Set "
    "HPU_OPS_INT64_CHECK=raise to raise here, HPU_OPS_INT64_CHECK=0 to skip this check. See hpu-ops' README, "
    "\"float64\".")


def _warn_once(key, message):
    if key not in _WARNED:
        _WARNED.add(key)
        warnings.warn(message, stacklevel=3)


def check_int64_source(src):
    """Warn once (or raise, ``HPU_OPS_INT64_CHECK=raise``) when ``src``, a CPU int64 tensor about to be copied to the
    card, holds a value outside the int32 range. After the first warning no tensor is scanned."""
    setting = os.environ.get("HPU_OPS_INT64_CHECK", "warn")
    if setting == "0" or src.numel() == 0 or (setting != "raise" and "int64" in _WARNED):
        return
    if _SCANNING:  # the scan's own copies (a strided view made contiguous) are not checked
        return
    _SCANNING.append(True)
    try:
        lo, hi = (int(v) for v in torch.aminmax(src))
    finally:
        _SCANNING.pop()
    if lo < INT32_MIN or hi > INT32_MAX:
        message = INT64_WARNING.format(lo=lo, hi=hi)
        if setting == "raise":
            raise RuntimeError(message)
        _warn_once("int64", message)


def _plain(t):
    from torch._subclasses.fake_tensor import is_fake

    return not is_fake(t) and not torch._is_functional_tensor(t) and t.layout == torch.strided


def copy_checks(self, src, device_type):
    """The checks of a ``self.copy_(src)`` whose ``self`` is on ``device_type``."""
    if (self.dtype == torch.int64 and isinstance(src, torch.Tensor) and src.dtype == torch.int64
            and src.device.type == "cpu" and _plain(src)):
        check_int64_source(src)
    if self.dtype == torch.float64:
        _warn_once("float64", FLOAT64_WARNING)


def install_copy_checks(device_type):
    """The int64 and float64 checks at ``copy_`` to ``device_type`` where the device has no ``copy_`` kernel of its
    own (Habana's eager mode). Returns an undo."""
    op = aten.copy_.default

    def kernel(self, src, non_blocking=False):
        if self.device.type == device_type:
            copy_checks(self, src, device_type)
        return op._op_dk(DispatchKey.CompositeExplicitAutograd, self, src, non_blocking)

    return register_kernels(device_type, [(op, "{}", kernel)])
