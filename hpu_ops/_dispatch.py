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
from torch._subclasses.fake_tensor import is_fake

REDUCED = (torch.float16, torch.bfloat16)
_STATE = {"calls": collections.Counter()}
_DUAL_PROBE = torch.empty(0)


def forward_ad_level(*tensors):
    """Active forward-mode AD dual level (>= 0), or -1 if none.

    ``torch.compile`` lowers ``fwAD.dual_level()`` and ``torch.func.jvp`` to C++
    ``torch._C._enter_dual_level()`` without updating Python's ``fwAD._current_level`` (which stays ``-1``).
    Check passed tensors for a level-0 dual tangent first; otherwise probe C++ level 0 (popping active
    dispatch modes when present so tracers do not record the probe). Skip probing only during Dynamo bytecode
    tracing (where ``is_compiling()`` is True and no dispatch modes are active).
    """
    has_modes = (
        torch._C._len_torch_dispatch_stack() > 0
        or torch._ops._len_torch_dispatch_stack_pre_dispatch() > 0
    )
    if torch.compiler.is_compiling() and not has_modes:
        return getattr(torch.autograd.forward_ad, "_current_level", -1)
    if torch.is_inference_mode_enabled():
        return -1
    lvl = getattr(torch.autograd.forward_ad, "_current_level", -1)
    if lvl >= 0:
        return lvl
    if (not below_autograd()
            and any(isinstance(t, torch.Tensor) and not t.is_inference()
                    and torch._VF._unpack_dual(t, 0)[1] is not None for t in tensors)):
        return 0
    if has_modes:
        from torch.utils._python_dispatch import _disable_current_modes

        with _disable_current_modes():
            try:
                torch._VF._make_dual(_DUAL_PROBE, _DUAL_PROBE, 0)
                return 0
            except RuntimeError:
                return -1
    try:
        torch._VF._make_dual(_DUAL_PROBE, _DUAL_PROBE, 0)
        return 0
    except RuntimeError:
        return -1


def count(name):
    """Record that the rewrite ``name`` ran (the Gaudi checks read the counts)."""
    _STATE["calls"][name] += 1


def static(*sizes):
    """True when every size is a plain int, i.e. none is symbolic under ``torch.compile``."""
    return all(isinstance(size, int) for size in sizes)


def tracing(*tensors):
    """True when ``torch.compile`` (Dynamo, or AOT autograd's fake and functional tensors, where
    ``torch.compiler.is_compiling()`` is False), a fake mode, or ``make_fx`` (a proxy mode, also on real tensors)
    is active on ``tensors``."""
    if (torch._C._len_torch_dispatch_stack() == 0 and not torch.compiler.is_compiling()
            and all(type(t) is torch.Tensor and not torch._is_functional_tensor(t) for t in tensors)):
        return False
    from torch._subclasses.fake_tensor import FakeTensor, is_fake
    from torch._subclasses.functional_tensor import FunctionalTensor

    if (torch.compiler.is_compiling() or torch._guards.detect_fake_mode() is not None
            or torch._C._get_dispatch_mode(torch._C._TorchDispatchModeKey.PROXY) is not None):
        return True
    return any(isinstance(t, FakeTensor) or is_fake(t) or torch._is_functional_tensor(t)
               or isinstance(t, FunctionalTensor) for t in tensors)


def concrete(t):
    """True for a real tensor outside any tracer."""
    return not is_fake(t) and torch._guards.detect_fake_mode() is None and not torch._is_functional_tensor(t)


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
