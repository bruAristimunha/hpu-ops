"""Two rewrites of advanced indexing that Gaudi's lazy mode gets wrong (bridge 1.21.4; eager mode is right).

- ``index`` whose indexed dims do not lead, e.g. ``X[..., rows, cols]``: its backward,
  ``_index_put_impl_(zeros, indices, grad, accumulate=True)`` with a ``None`` before the first index tensor or between
  two of them, returns wrong sums in lazy mode from batch 2 to 8 on (relative error 0.45 to 1 for
  ``X[..., triu_indices]``, batch 32, n = 12 or 22). That is spd_learn's ``vech`` (``sym_to_upper``) and the reads of
  ``vec_to_sym``: in lazy mode every gradient before them was wrong in SPDResNet, GyroSPD, CorNet, SPNN, DeepGeoCCA,
  SPDNetAE, EEGSPDNet, MAtt and CorAtt. Such an index of integer indices is rewritten at ``AutogradHPU`` as one
  ``index_select`` on the indexed dims flattened into one (after a permute that brings them together first when a
  slice separates them), with the linear index of the broadcast indices; its backward is ``index_add``, which is right.
  Index tensors on the CPU (spd_learn's ``triu_indices``) are checked against the sizes on the host, and an index out of
  range goes to ATen's kernel, which raises; an index tensor already on the device is wrapped for negative values but
  not checked against the sizes (that would need a host sync). Eager and lazy mode only: while ``torch.compile`` traces,
  the index tensors are fake, their values cannot be checked, and the traced graph keeps ATen's ``index`` and its
  accumulating ``index_put`` backward (hpu_backend does not run the lazy-mode kernel this works around).
  Two other forms were measured right on ``vech`` and SPDResNet but not everywhere: a leading-index ``index`` of a
  permuted view (CorNet's gradient stayed 0.44 off), and rewriting the backward's ``_index_put_impl_`` through a
  permuted view (SPNN's launch failed, see the next item).
- A boolean-mask put of one value, ``m[..., mask] = c`` (vech's sqrt(2) scale): in one lazy graph with index puts into
  a fresh matrix and index reads of it (``vech(vec_to_sym(v))``, SPNN's SPDLinear) Habana's launch fails, "synLaunch
  failed ... Sections' validation failure" (``repro/lazy_section_validation.py``, without hpu-ops). It is rewritten at
  ``AutogradHPU`` as ``self.copy_(where(mask, c, self))``: equal values, and no ``nonzero`` and host sync.

``index_put_`` with integer indices, and calls below autograd (inference mode), keep the native kernels.
"""
import torch
from torch._C import DispatchKey

from ._dispatch import count, register_kernels

aten = torch.ops.aten
INTEGER = (torch.int64, torch.int32)
MASK = (torch.bool, torch.uint8)


def _positions(indices):
    return [i for i, t in enumerate(indices) if t is not None]


def _in_range(t, size):
    """True for an index tensor on the device (not checked), or on the CPU with every value in [-size, size). False
    when the values cannot be read (a fake tensor while ``torch.compile`` traces): the native op takes those."""
    if t.device.type != "cpu" or t.numel() == 0:
        return True
    try:
        return bool(((t >= -size) & (t < size)).all())
    except Exception:  # noqa: BLE001  (data-dependent under tracing)
        return False


def _index_takes(self, indices):
    pos = _positions(indices)
    if not pos or len(indices) > self.dim() or any(indices[i].dtype not in INTEGER for i in pos):
        return False  # bool and byte masks cover several dims; no tensor index or too many reach the native op
    if pos == list(range(len(pos))):
        return False  # leading indices: the native op is right
    if len({indices[i].device for i in pos}) > 1:
        return False  # index tensors on two devices: ATen's rules
    try:
        torch.broadcast_shapes(*(indices[i].shape for i in pos))
    except RuntimeError:
        return False  # ATen's error
    return all(_in_range(indices[i], self.shape[i]) for i in pos)


def index_select_flat(self, indices):
    """``self[indices]`` as ``index_select`` of the indexed dims flattened, with the linear index. ATen puts the broadcast
    index dims in place of the indexed dims when these are adjacent, else first: a permute first brings them to the
    front in that case, which is where the result then has them."""
    pos = _positions(indices)
    if pos != list(range(pos[0], pos[-1] + 1)):
        self = self.permute(pos + [d for d in range(self.dim()) if d not in pos])
        pos = list(range(len(pos)))
        indices = [indices[i] for i in _positions(indices)]
    else:
        indices = [indices[i] for i in pos]
    start, sizes = pos[0], self.shape[pos[0]:pos[-1] + 1]
    lin = None
    for t, size in zip(torch.broadcast_tensors(*indices), sizes):
        t = torch.where(t < 0, t + size, t).long()
        lin = t if lin is None else lin * size + t
    flat = self.flatten(start, pos[-1]) if len(pos) > 1 else self
    out = flat.index_select(start, lin.reshape(-1).to(self.device))
    if lin.dim() == 0:
        return out.squeeze(start)  # 0-d indices (direct aten calls): the indexed dims go away
    return out.unflatten(start, lin.shape) if lin.dim() != 1 else out


def _mask_takes(self, indices, values, accumulate=False):
    pos = _positions(indices)
    if accumulate or len(pos) != 1 or values.numel() != 1 or values.dim() > 1 or self.is_complex():
        return False
    mask, p = indices[pos[0]], pos[0]
    if mask.dtype not in MASK or len(indices) > self.dim():
        return False
    return mask.dim() > 0 and tuple(mask.shape) == tuple(self.shape[p:p + mask.dim()])


def mask_fill(self, indices, values, accumulate=False):
    """``self[..., mask, ...] = value`` as ``self.copy_(where(mask, value, self))``; returns self."""
    p = _positions(indices)[0]
    mask = indices[p]
    mask = mask.to(self.device, torch.bool).reshape((1,) * p + tuple(mask.shape)
                                                    + (1,) * (self.dim() - p - mask.dim()))
    return self.copy_(torch.where(mask, values.reshape(()).to(self.device, self.dtype), self))


def install_index(device_type):
    """``index`` with integer indices that do not lead, on ``device_type``, as one ``index_select``. Returns an undo."""
    op = aten.index.Tensor

    def kernel(self, indices):
        if _index_takes(self, indices):
            count("index_select_flat")
            return index_select_flat(self, indices)
        return op._op_dk(DispatchKey.Autograd, self, indices)

    return register_kernels(device_type, [(op, "Autograd{}", kernel)])


def install_mask_fill(device_type):
    """``index_put_`` of one value through one boolean mask, on ``device_type``, as ``where``. Returns an undo."""
    op = aten.index_put_.default

    def kernel(self, indices, values, accumulate=False):
        if _mask_takes(self, indices, values, accumulate):
            count("mask_fill")
            return mask_fill(self, indices, values, accumulate)
        return op._op_dk(DispatchKey.Autograd, self, indices, values, accumulate)

    return register_kernels(device_type, [(op, "Autograd{}", kernel)])
