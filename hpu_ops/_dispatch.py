"""How hpu-ops reaches the dispatcher, each way with an undo.

- Replacing torch's Python kernels of an op (``py_kernels``). Only the Python dispatcher reads them, i.e. while
  ``torch.compile`` traces a graph; eager calls never see them.
- Registering Python kernels at a device's own dispatch keys (``HPU``, ``AutogradHPU``) with ``torch.library``. Eager
  calls and traced graphs both go through these.

The rewrites read the device type they apply to as an argument, so the tests can run them with "cpu" in the role of
"hpu".
"""
import collections
import functools

import torch
from torch._C import DispatchKey

_STATE = {"calls": collections.Counter()}


def count(name):
    """Record that the rewrite ``name`` ran (the Gaudi checks read the counts)."""
    _STATE["calls"][name] += 1


def static(*sizes):
    """True when every size is a plain int, i.e. none is symbolic under ``torch.compile``."""
    return all(isinstance(size, int) for size in sizes)


def below_autograd():
    """True where autograd's keys are excluded: below an autograd kernel (backend kernels, ``__torch_dispatch__``
    modes, hence the tracers' ``decompose()`` calls) and in inference mode."""
    return torch._C._dispatch_tls_is_dispatch_key_excluded(DispatchKey.AutogradFunctionality)


def chain(undos):
    """One undo that runs ``undos`` in reverse order."""

    def undo():
        for step in reversed(undos):
            step()

    return undo


def replace_py_kernels(op, keys, make_kernel):
    """Swap ``op``'s Python kernels at ``keys`` for ``make_kernel(original)``. ``original`` is the kernel replaced, or
    the C++ kernel at that key when torch registers none in Python. Returns an undo."""
    originals = {key: op.py_kernels.pop(key, None) for key in keys}
    for key, original in originals.items():
        op.py_impl(key)(make_kernel(original or functools.partial(op._op_dk, key)))  # py_impl clears the cache

    def undo():
        for key, original in originals.items():
            op.py_kernels.pop(key, None)
            if original is not None:
                op.py_kernels[key] = original
        op._dispatch_cache.clear()

    return undo


def register_kernels(device_type, kernels):
    """Register ``kernels``, a list of ``(op, key, kernel)`` with ``key`` "{}" (the device's backend key) or
    "Autograd{}", for ``device_type``. A key where a kernel is already registered for that very key is left alone, so
    Habana's own kernels (and, in the tests, the CPU kernels) are never overridden. Returns an undo."""
    library = torch.library.Library("aten", "IMPL")
    touched = []
    for op, key, kernel in kernels:
        key = key.format(device_type.upper())
        if torch._C._dispatch_has_kernel_for_dispatch_key(op.name(), key):
            continue
        library.impl(op, kernel, key)
        touched.append(op)
    for op in touched:  # the Python dispatcher caches where each key resolved to
        op._dispatch_cache.clear()

    def undo():
        library._destroy()
        for op in touched:
            op._dispatch_cache.clear()

    return undo
