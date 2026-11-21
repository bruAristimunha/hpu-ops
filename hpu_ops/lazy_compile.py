"""``torch.compile(backend="hpu_backend")`` in lazy mode (``PT_HPU_LAZY_MODE=1``) refused at the call, in one line.

Habana's support matrix for 1.21.4 lists ``torch.compile(backend="hpu_backend")`` under "Eager + torch.compile
(PT_HPU_LAZY_MODE=0)" only. On bridge 1.21.4, ``habana_frameworks.torch.core`` imports the backend's meta
registrations only outside lazy mode, and lazy mode loads the lazy plugin, so the recipe compiler's library does not
link (``_recipe_compiler_C.so: undefined symbol habana::graph::GraphStorage::get``) and
``torch.ops.hpu.weight_permutation``, which the backend's partitioner reads, is not registered (``AttributeError``). A
captured graph fails deep in the backend (SPDNet, with or without hpu_ops); a model whose frames dynamo does not
capture runs uncompiled (a Linear MLP: no graph reached the backend).

``enable()`` in lazy mode makes every lookup of the name ``hpu_backend`` after it (``torch.compile(...,
backend="hpu_backend")``, ``torch._dynamo.optimize("hpu_backend")``) raise ``MESSAGE`` at that call. Not covered: a
function compiled before ``enable()`` (it holds the backend already) and the backend passed as a function, not by
name. ``torch._dynamo.list_backends()`` still lists the name."""
NAME = "hpu_backend"
MESSAGE = ("torch.compile(backend='hpu_backend') is not supported in lazy mode (PT_HPU_LAZY_MODE=1) on Habana's "
           "bridge: run with PT_HPU_LAZY_MODE=0 (eager + torch.compile), or drop torch.compile in lazy mode.")


class _Refused:
    """Stands where torch's registry keeps a backend's entry point: loading it raises the message."""

    def load(self):
        raise RuntimeError(MESSAGE)


def install(name=NAME):
    """Make a lookup of the backend ``name`` raise ``MESSAGE``. Returns an undo that restores the registry entries,
    unless the registry changed the name since (then it is left as it is)."""
    from torch._dynamo.backends import registry  # here: importing dynamo costs about 1 s, paid only in lazy mode

    registry._lazy_import()  # entry points read now (cached), so a later lookup does not overwrite the guard
    saved = [(table, name in table, table.get(name)) for table in (registry._BACKENDS, registry._COMPILER_FNS)]
    guard = _Refused()
    registry._BACKENDS[name] = guard
    registry._COMPILER_FNS.pop(name, None)  # a cached compiler would be returned without loading the entry point

    def undo():
        if registry._BACKENDS.get(name) is not guard or name in registry._COMPILER_FNS:
            return  # registered again since the guard: that registration stands
        for table, had, value in saved:
            if had:
                table[name] = value
            else:
                table.pop(name, None)
    return undo
