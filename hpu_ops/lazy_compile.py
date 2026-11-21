"""``torch.compile(backend="hpu_backend")`` in lazy mode (``PT_HPU_LAZY_MODE=1``): one warning, then uncompiled.

Habana's support matrix for 1.21.4 lists ``torch.compile(backend="hpu_backend")`` under "Eager + torch.compile
(PT_HPU_LAZY_MODE=0)" only. On bridge 1.21.4, ``habana_frameworks.torch.core`` imports the backend's meta
registrations only outside lazy mode, and lazy mode loads the lazy plugin, so the recipe compiler's library does not
link (``_recipe_compiler_C.so: undefined symbol habana::graph::GraphStorage::get``) and
``torch.ops.hpu.weight_permutation``, which the backend's partitioner reads, is not registered (``AttributeError``). A
captured graph fails deep in the backend (SPDNet, with or without hpu_ops); a model whose frames dynamo does not
capture runs uncompiled (a Linear MLP: no graph reached the backend).

``enable()`` in lazy mode makes ``torch.compile(model, backend="hpu_backend")`` warn once (``MESSAGE``) and return
``model`` itself, which then runs in lazy mode as if never compiled (``nn.Module.compile`` too: its call stays
uncompiled). A route that reaches the backend by its name some other way (``torch._dynamo.optimize("hpu_backend")``,
a ``torch.compile`` bound before ``enable()``) gets a pass-through backend: dynamo still traces, and the captured
graph runs op by op, in lazy mode, after the same one warning. With ``HPU_OPS_LAZY_COMPILE=raise`` every lookup of
the name raises ``RAISE_MESSAGE`` as a ``RuntimeError`` instead. Not covered: a function compiled before
``enable()`` and the backend passed as a function, not by name. ``torch._dynamo.list_backends()`` still lists the name."""
import functools
import os
import warnings

import torch

NAME = "hpu_backend"
_FACT = ("torch.compile(backend='hpu_backend') is not supported in lazy mode (PT_HPU_LAZY_MODE=1): Habana's support "
         "matrix lists it for eager mode only. Run with PT_HPU_LAZY_MODE=0 to compile")
MESSAGE = _FACT + "; in lazy mode hpu_ops runs the model uncompiled (HPU_OPS_LAZY_COMPILE=raise makes this an error)."
RAISE_MESSAGE = _FACT + ", or unset HPU_OPS_LAZY_COMPILE=raise to run it uncompiled in lazy mode."
_WARNED = set()


def _warn_once():
    if not _WARNED:
        _WARNED.add(NAME)
        warnings.warn(MESSAGE, stacklevel=3)


class _Refused:
    """Stands where torch's registry keeps a backend's entry point: loading it raises the message."""

    def load(self):
        raise RuntimeError(RAISE_MESSAGE)


def _pass_through(gm, example_inputs):
    """A dynamo backend that compiles nothing: the captured graph runs op by op."""
    _warn_once()
    return gm.forward


def _guard_registry(registry, name, raising):
    """The registry entry of ``name``: the refusal (``raising``) or the pass-through. Returns an undo, which leaves a
    later registration of ``name`` in place."""
    registry._lazy_import()  # entry points read now (cached), so a later lookup does not overwrite the guard
    saved = [(table, name in table, table.get(name)) for table in (registry._BACKENDS, registry._COMPILER_FNS)]
    if raising:
        guard = _Refused()
        registry._BACKENDS[name] = guard
        registry._COMPILER_FNS.pop(name, None)  # a cached compiler would be returned without loading the entry point
    else:
        guard = functools.partial(_pass_through)  # its own object, so the undo can tell it from a later one
        guard._tags = ()
        registry._BACKENDS.setdefault(name, None)
        registry._COMPILER_FNS[name] = guard

    def in_place():
        if raising:
            return registry._BACKENDS.get(name) is guard and name not in registry._COMPILER_FNS
        return registry._COMPILER_FNS.get(name) is guard

    def undo():
        if not in_place():
            return  # registered again since the guard: that registration stands
        for table, had, value in saved:
            if had:
                table[name] = value
            else:
                table.pop(name, None)
    return undo


def _guard_compile(name):
    """``torch.compile`` returning its model uncompiled (one warning) for ``backend=name``. Returns an undo."""
    original = torch.compile

    @functools.wraps(original)
    def compile_(model=None, **kwargs):
        if kwargs.get("backend") != name:
            return original(model, **kwargs)
        _warn_once()
        return (lambda m: m) if model is None else model  # also as a decorator: @torch.compile(backend=...)

    torch.compile = compile_

    def undo():
        if torch.compile is compile_:
            torch.compile = original
    return undo


def install(name=NAME):
    """Lazy mode's handling of the backend ``name``, per ``HPU_OPS_LAZY_COMPILE`` (``raise``: an error at the call;
    else: one warning, uncompiled). Returns an undo."""
    from torch._dynamo.backends import registry  # here: importing dynamo costs about 1 s, paid only in lazy mode

    if os.environ.get("HPU_OPS_LAZY_COMPILE", "").lower() == "raise":
        return _guard_registry(registry, name, raising=True)
    undo_registry = _guard_registry(registry, name, raising=False)
    undo_compile = _guard_compile(name)

    def undo():
        undo_compile()
        undo_registry()
    return undo
