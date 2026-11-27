"""Patches for Habana backend bugs that cannot be reached from the dispatcher (#63, #121).

- ``recipe_compiler.get_input_symbolic`` (#63, dynamic compile only) indexes
  ``input_node.meta["output_strides"]`` on every placeholder, but
  ``fill_propagated_tensor_metadata_to_node`` returns early on ``get_attr`` nodes without
  setting it, and ``fuse_partitions`` promotes such a node (a lifted ``_tensor_constant``)
  to a partition placeholder: dynamic-compiled inference then dies with
  ``KeyError: 'output_strides'`` (TensorCSPNet/GyroAtt, repro/habana_inference_output_strides.py).
  The wrapper below fills the missing entries from the placeholder's fake value - exactly
  what the helper would have written for a non-``get_attr`` node - before the vendor
  function reads them. It changes no strides the vendor computed itself.
- ``hpu_backend_decompositions_common`` (#121): Habana's ``hpu_backend`` decomposes
  ``aten.unfold.default`` (via PyTorch's builtin decomp) and ``aten.diagonal.default``
  (custom decomp) into ``self.as_strided(...)`` using PyTorch's storage strides/offset.
  Inside Synapse, upstream view ops (``transpose``, ``permute``, ``slice``, ``narrow``,
  a prior ``unfold``) materialize dense logical buffers, so Synapse's ``strided_view``
  reads the dense buffer with PyTorch's pre-materialization strides/offset: non-contiguous
  inputs silently scramble output elements (``rel_l2 = 1.414, cos ~ 0``) or crash with
  ``synNodeCreateWithId failed for node: strided_view with synStatus 1`` (slice/narrow/offset).
  Cloning non-contiguous or non-zero-offset HPU inputs to a contiguous zero-offset base
  inside those decompositions fixes compiled forward, backward, in-place functionalization,
  ``no_grad``, ``inference_mode``, and dynamic shapes while leaving contiguous zero-offset
  inputs zero-overhead.
"""
import functools

import torch
import torch.utils._pytree as pytree

from ._dispatch import _once, count

_PATCHED = False
_AS_STRIDED_DECOMP_OPS = (
    (torch.ops.aten.unfold.default, "unfold_decomp"),
    (torch.ops.aten.diagonal.default, "diagonal_decomp"),
)


def fill_missing_output_strides(graph_module):
    """Fill ``meta["output_strides"]`` on placeholders that lack it, from ``meta["val"]``.
    Returns the number filled."""
    filled = 0
    for node in graph_module.graph.nodes:
        if node.op == "placeholder" and "output_strides" not in node.meta:
            leaves = [t for t in pytree.tree_leaves(node.meta.get("val"))
                      if isinstance(t, torch.Tensor)]
            if leaves:
                node.meta["output_strides"] = [tuple(t.stride()) for t in leaves]
                filled += 1
    return filled


def patch_recipe_compiler():
    """Wrap Habana's ``get_input_symbolic`` with ``fill_missing_output_strides``.
    Idempotent; a no-op off a Habana stack. Returns True when the wrapper is in place."""
    global _PATCHED
    if _PATCHED:
        return True
    try:
        from habana_frameworks.torch.dynamo.compile_backend import recipe_compiler
    except ImportError:
        return False
    original = recipe_compiler.get_input_symbolic
    if getattr(original, "_hpu_ops_strides_patch", False):
        _PATCHED = True
        return True

    def get_input_symbolic(graph_module, inputs):
        fill_missing_output_strides(graph_module)
        return original(graph_module, inputs)

    get_input_symbolic._hpu_ops_strides_patch = True
    recipe_compiler.get_input_symbolic = get_input_symbolic
    _PATCHED = True
    return True


def _needs_contiguous_base(tensor):
    try:
        return not tensor.is_contiguous() or tensor.storage_offset() != 0
    except Exception:  # noqa: BLE001  (unbacked SymInt guard: materialize conservatively)
        return True


def wrap_as_strided_decomp(original, name, device_type="hpu"):
    """Wrap an ``as_strided``-based decomposition so non-contiguous or non-zero-offset tensors
    on ``device_type`` are cloned to a contiguous, zero-offset base before ``as_strided``."""

    @functools.wraps(original)
    def wrapped(self, *args, **kwargs):
        if (isinstance(self, torch.Tensor) and self.device.type == device_type
                and _needs_contiguous_base(self)):
            count(name)
            self = self.clone(memory_format=torch.contiguous_format)
        return original(self, *args, **kwargs)

    wrapped._hpu_ops_as_strided_patch = True
    return wrapped


def patch_as_strided_decompositions(table=None, device_type="hpu"):
    """Patch Habana's ``as_strided``-based compile decompositions (``aten.unfold.default`` and
    ``aten.diagonal.default``) in place so non-contiguous or offset inputs are materialized
    before ``as_strided`` (#121).

    Returns an idempotent undo callable. When ``table`` is omitted, patches Habana's
    ``hpu_backend_decompositions_common`` in place (a no-op off a Habana stack).
    """
    if table is None:
        try:
            from habana_frameworks.torch.dynamo.compile_backend import decomposition
        except ImportError:
            return _once(lambda: None)
        table = getattr(decomposition, "hpu_backend_decompositions_common", None)
        if not isinstance(table, dict):
            return _once(lambda: None)

    originals = {}
    installed = {}
    for op, name in _AS_STRIDED_DECOMP_OPS:
        fn = table.get(op)
        if fn is None or getattr(fn, "_hpu_ops_as_strided_patch", False):
            continue
        wrapped = wrap_as_strided_decomp(fn, name, device_type=device_type)
        originals[op] = fn
        installed[op] = wrapped
        table[op] = wrapped

    def undo():
        for op, fn in originals.items():
            if table.get(op) is installed[op]:
                table[op] = fn

    return _once(undo)
