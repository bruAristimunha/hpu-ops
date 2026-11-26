"""Patches for Habana backend bugs that cannot be reached from the dispatcher (#63).

``recipe_compiler.get_input_symbolic`` (dynamic compile only) indexes
``input_node.meta["output_strides"]`` on every placeholder, but
``fill_propagated_tensor_metadata_to_node`` returns early on ``get_attr`` nodes without
setting it, and ``fuse_partitions`` promotes such a node (a lifted ``_tensor_constant``)
to a partition placeholder: dynamic-compiled inference then dies with
``KeyError: 'output_strides'`` (TensorCSPNet/GyroAtt, repro/habana_inference_output_strides.py).
The wrapper below fills the missing entries from the placeholder's fake value - exactly
what the helper would have written for a non-``get_attr`` node - before the vendor
function reads them. It changes no strides the vendor computed itself.
"""
import torch
import torch.utils._pytree as pytree

_PATCHED = False


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
