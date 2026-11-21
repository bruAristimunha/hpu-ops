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


def tracing(t):
    """Whether a tracer sees ``t``: ``torch.compile`` (Dynamo, where ``is_compiling()``; AOT autograd's fake tensors
    and the ``FunctionalTensor`` wrappers around them, where it is False), a fake mode, or ``make_fx`` (a proxy mode,
    also on real tensors). ``linalg._tracing``'s test, for the modules that cannot import linalg."""
    from torch._subclasses.fake_tensor import is_fake
    from torch._subclasses.functional_tensor import FunctionalTensor

    if torch.compiler.is_compiling() or is_fake(t) or (isinstance(t, FunctionalTensor) and is_fake(t.elem)):
        return True
    if torch._guards.detect_fake_mode() is not None:
        return True
    return torch._C._get_dispatch_mode(torch._C._TorchDispatchModeKey.PROXY) is not None


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


def register_paired(device_type, groups):
    """``register_kernels`` for ``groups``, lists of ``(op, key, kernel)`` whose first entry is the autograd kernel and
    the others the backend kernels below it. A backend kernel is registered only together with its autograd kernel:
    if the autograd key of a group already has a kernel on this bridge, none of the group is, as ours at the
    backend key under Habana's autograd kernel would split the op between the two (inference mode ours, training
    Habana's). A taken backend key alone only skips that key (the CPU stand-in's case: the CPU kernels stay, our
    autograd kernel computes by itself). Returns an undo."""
    undos = []
    for group in groups:
        op, key, _ = group[0]
        assert key.startswith("Autograd"), key
        if torch._C._dispatch_has_kernel_for_dispatch_key(op.name(), key.format(device_type.upper())):
            continue
        undos.append(register_kernels(device_type, group))
    return chain(undos)


def override_kernels(device_type, kernels):
    """Register ``kernels``, a list of ``(op, kernel)``, at ``device_type``'s backend key even where a kernel is
    registered there (Habana's lazy kernels, the structured default in eager). The kernel replaced is not callable
    any more: the kernels reach the device through other ops (``extrema`` through the ops' ``.out`` overloads). The
    undo restores the kernels replaced (the dispatcher keeps them under the newer registration)."""
    import warnings

    library = torch.library.Library("aten", "IMPL")
    key = device_type.upper()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*[Oo]verrid.*")
        for op, kernel in kernels:
            library.impl(op, kernel, key)
    for op, _ in kernels:
        op._dispatch_cache.clear()

    def undo():
        library._destroy()
        for op, _ in kernels:
            op._dispatch_cache.clear()

    return undo
