"""Advanced indexing (``aten::index.Tensor``) on HPU.

Habana's eager ``aten::index.Tensor_out`` kernel falls back to the CPU whenever any index tensor resides on the CPU
(e.g. ``torch.triu_indices`` in ``spd_learn``'s ``sym_to_upper``), and Habana's lazy ``aten::index.Tensor`` backward
drops gradient entries. Habana's native indexing also miscomputes gradients for non-contiguous negative integer indices
and returns transposed shapes when a boolean mask follows leading ``None`` slices.

Integer-tensor indexing (contiguous or permuted to front when non-contiguous) is rewritten via normalized linear
indices and ``index_select``, which runs on-card in eager, lazy and compiled modes with exact gradients. Boolean masks
after leading ``None`` slices are expanded via ``nonzero`` into the same ``index_select`` path; other valid off-device
indices are moved to ``self.device`` before delegating to the native op. Invalid inputs pass through to ATen so native
exceptions are preserved.
"""
import math

import torch
from torch._C import DispatchKey

from ._dispatch import count, register_kernels

aten = torch.ops.aten
INT_INDEX_DTYPES = frozenset((torch.int32, torch.int64))
VALID_INDEX_DTYPES = frozenset((torch.int32, torch.int64, torch.bool))


def _classify(self, indices):
    if not indices or len(indices) > self.ndim:
        return None
    non_none = [(k, t) for k, t in enumerate(indices) if t is not None]
    if not non_none:
        return None
    dtypes = {t.dtype for _, t in non_none}
    if not dtypes <= VALID_INDEX_DTYPES:
        return None
    if dtypes <= INT_INDEX_DTYPES:
        try:
            torch.broadcast_shapes(*(t.shape for _, t in non_none))
        except RuntimeError:
            return None
        return "int"
    dim = 0
    for t in indices:
        if t is None:
            dim += 1
        elif t.dtype == torch.bool:
            if t.ndim > 0:
                if dim + t.ndim > self.ndim or tuple(t.shape) != tuple(self.shape[dim : dim + t.ndim]):
                    return None
                dim += t.ndim
        else:
            dim += 1
    if dim > self.ndim:
        return None
    if len(non_none) == 1 and non_none[0][1].dtype == torch.bool and non_none[0][1].ndim > 0:
        pos = non_none[0][0]
        if any(idx is None for idx in indices[:pos]):
            return "bool_expand"
    if any(t.device != self.device for _, t in non_none):
        return "move"
    return None


def _index_by_select(self, indices):
    non_none = [k for k, t in enumerate(indices) if t is not None]
    if non_none[-1] - non_none[0] + 1 == len(non_none):
        first, last = non_none[0], non_none[-1]
        x = self
        prefix_shape = self.shape[:first]
        indexed_shape = self.shape[first : last + 1]
        suffix_shape = self.shape[last + 1 :]
        tensors = [indices[k] for k in range(first, last + 1)]
    else:
        order = non_none + [d for d in range(self.ndim) if d not in non_none]
        x = self.permute(order)
        prefix_shape = ()
        indexed_shape = tuple(self.shape[d] for d in non_none)
        suffix_shape = tuple(self.shape[d] for d in range(self.ndim) if d not in non_none)
        tensors = [indices[d] for d in non_none]
    bcast_shape = torch.broadcast_shapes(*(t.shape for t in tensors))
    compute_device = (torch.device("cpu")
                      if not torch.compiler.is_compiling() and all(t.device.type == "cpu" for t in tensors)
                      else self.device)
    with torch.autocast(self.device.type, enabled=False):
        flat_idx = None
        stride = 1
        for t, dim_size in zip(reversed(tensors), reversed(indexed_shape)):
            idx = t.to(device=compute_device, dtype=torch.int64)
            idx = torch.where(idx < 0, idx + dim_size, idx)
            term = idx if stride == 1 else idx * stride
            flat_idx = term if flat_idx is None else flat_idx + term
            stride *= dim_size
        flat_idx = torch.broadcast_to(flat_idx, bcast_shape).reshape(-1).to(self.device)
    prefix = math.prod(prefix_shape)
    middle = math.prod(indexed_shape)
    suffix = math.prod(suffix_shape)
    return x.reshape((prefix, middle, suffix)).index_select(1, flat_idx).reshape((*prefix_shape, *bcast_shape,
                                                                                 *suffix_shape))


def _expand_bool_indices(self, indices):
    expanded = []
    for t in indices:
        if t is not None and t.dtype == torch.bool and t.ndim > 0:
            expanded.extend(t.to(self.device).nonzero(as_tuple=True))
        else:
            expanded.append(t)
    return expanded


def _device_kernel(device_type, op, fallback):
    def kernel(self, indices):
        if self.device.type != device_type:
            return op._op_dk(fallback, self, indices)
        kind = _classify(self, indices)
        if kind == "int":
            count("index")
            return _index_by_select(self, indices)
        if kind == "bool_expand":
            count("index")
            return _index_by_select(self, _expand_bool_indices(self, indices))
        if kind == "move":
            count("index")
            moved = [t.to(self.device) if t is not None else None for t in indices]
            return op._op_dk(fallback, self, moved)
        return op._op_dk(fallback, self, indices)

    return kernel


def install(device_type):
    """Rewrite ``aten::index.Tensor`` on ``device_type`` to stay on-card. Returns an undo."""
    op = aten.index.Tensor
    below = DispatchKey.CompositeExplicitAutogradNonFunctional
    return register_kernels(device_type, [
        (op, "Autograd{}", _device_kernel(device_type, op, DispatchKey.Autograd)),
        (op, "{}", _device_kernel(device_type, op, below)),
    ])
