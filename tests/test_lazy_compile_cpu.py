"""``torch.compile(backend="hpu_backend")`` in lazy mode (``hpu_ops.lazy_compile``), on the CPU: a backend registered
under a test name stands in for Habana's, and a stub Gaudi stack for ``enable()``. Default: one warning, the model
returned uncompiled; ``HPU_OPS_LAZY_COMPILE=raise``: an error at the call."""
import sys
import types
import warnings

import pytest
import torch
from torch._dynamo.backends import registry

import hpu_ops
from hpu_ops import lazy_compile

NAME = "hpu_ops_test_backend"


@pytest.fixture(autouse=True)
def fresh_warning(monkeypatch):
    """Each test sees the once-per-process warning anew; the mode is the default unless a test sets it."""
    monkeypatch.setattr(lazy_compile, "_WARNED", set())
    monkeypatch.delenv("HPU_OPS_LAZY_COMPILE", raising=False)


@pytest.fixture
def raise_mode(monkeypatch):
    monkeypatch.setenv("HPU_OPS_LAZY_COMPILE", "raise")


@pytest.fixture
def backend():
    """A working backend registered as ``NAME`` (as Habana registers ``hpu_backend``), removed afterwards; yields the
    graphs it compiled."""
    calls = []

    def compiler(gm, example_inputs):
        calls.append(gm)
        return gm.forward
    registry.register_backend(compiler, name=NAME)
    yield calls
    if hpu_ops._STATE.get("undo"):  # an enable() of the test first, so its undo cannot put NAME back afterwards
        hpu_ops._STATE["undo"]()
    registry._COMPILER_FNS.pop(NAME, None)
    registry._BACKENDS.pop(NAME, None)
    torch._dynamo.reset()


def _messages(caught):
    return [str(w.message) for w in caught if "lazy mode" in str(w.message)]


# ---- default: one warning, uncompiled


def test_compile_returns_the_model_itself_and_warns_once(backend):
    model, fn = torch.nn.Linear(3, 2), (lambda x: x * 2)
    undo = lazy_compile.install(NAME)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert torch.compile(model, backend=NAME) is model
            assert torch.compile(fn, backend=NAME, dynamic=False) is fn
            assert torch.compile(backend=NAME)(fn) is fn  # the decorator form
        assert _messages(caught) == [lazy_compile.MESSAGE]
        assert "\n" not in lazy_compile.MESSAGE and "PT_HPU_LAZY_MODE=0" in lazy_compile.MESSAGE
        x = torch.randn(4, 3)
        torch.testing.assert_close(torch.compile(model, backend=NAME)(x), model(x))
        assert not backend, "nothing reached the backend"
    finally:
        undo()
    assert torch.compile is not None and torch.compile(model, backend=NAME) is not model  # undone: compiles again
    torch.compile(model, backend=NAME)(torch.randn(4, 3))
    assert backend


def test_module_compile_method_runs_uncompiled(backend):
    model = torch.nn.Linear(3, 2)
    undo = lazy_compile.install(NAME)
    try:
        with pytest.warns(UserWarning, match="lazy mode"):
            model.compile(backend=NAME)
        model(torch.randn(4, 3))
        assert not backend
    finally:
        undo()


def test_other_backends_compile_as_before(backend):
    undo = lazy_compile.install(NAME)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = torch.compile(torch.nn.Linear(3, 2), backend="eager")
            assert isinstance(out, torch._dynamo.eval_frame.OptimizedModule)
            out(torch.randn(4, 3))
        assert not _messages(caught)
    finally:
        undo()


def test_other_routes_to_the_name_get_the_pass_through(backend):
    """``torch._dynamo.optimize`` (and a torch.compile bound before enable) look the name up: dynamo traces, the graph
    runs op by op, one warning, Habana's backend never called."""
    undo = lazy_compile.install(NAME)
    try:
        fn = torch._dynamo.optimize(NAME)(lambda x: torch.sin(x) + 1)
        x = torch.randn(5)
        with pytest.warns(UserWarning, match="lazy mode"):
            torch.testing.assert_close(fn(x), torch.sin(x) + 1)
        assert not backend
    finally:
        undo()
    assert registry.lookup_backend(NAME) is not None and registry._COMPILER_FNS[NAME].__name__ == "compiler"


# ---- HPU_OPS_LAZY_COMPILE=raise


def test_raise_mode_raises_one_line_at_torch_compile_and_undo_restores(backend, raise_mode):
    undo = lazy_compile.install(NAME)
    try:
        with pytest.raises(RuntimeError, match="PT_HPU_LAZY_MODE=0") as info:
            torch.compile(torch.nn.Linear(3, 2), backend=NAME)  # at the call, before any graph is captured
        assert "\n" not in str(info.value) and str(info.value) == lazy_compile.RAISE_MESSAGE
        with pytest.raises(RuntimeError, match="lazy mode"):
            torch._dynamo.optimize(NAME)
        assert NAME in registry.list_backends()  # still listed: torch's listing does not load entry points
    finally:
        undo()
    out = torch.compile(torch.nn.Linear(3, 2), backend=NAME)(torch.randn(4, 3))
    assert out.shape == (4, 2) and backend, "the original backend compiles again after the undo"


def test_raise_mode_survives_a_later_entry_point_discovery(backend, raise_mode):
    """torch fills its table from entry points once (cached); the guard runs that first, so it is not overwritten."""
    undo = lazy_compile.install(NAME)
    try:
        registry._lazy_import()
        registry._discover_entrypoint_backends()
        with pytest.raises(RuntimeError, match="lazy mode"):
            registry.lookup_backend(NAME)
    finally:
        undo()


# ---- both modes: the registry's undo


@pytest.mark.parametrize("mode", ["warn", "raise"])
def test_guard_of_an_unregistered_name_is_undone_to_nothing(monkeypatch, mode):
    if mode == "raise":
        monkeypatch.setenv("HPU_OPS_LAZY_COMPILE", "raise")
    undo = lazy_compile.install(NAME)
    assert NAME in registry._BACKENDS
    undo()
    assert NAME not in registry._BACKENDS and NAME not in registry._COMPILER_FNS


@pytest.mark.parametrize("mode", ["warn", "raise"])
def test_a_backend_registered_after_the_guard_is_kept_by_the_undo(monkeypatch, mode):
    """Registered after install() (a late import of its package), the backend replaces the guard; the undo leaves it."""
    if mode == "raise":
        monkeypatch.setenv("HPU_OPS_LAZY_COMPILE", "raise")
    undo = lazy_compile.install(NAME)
    try:
        def late(gm, example_inputs):
            return gm.forward
        registry._COMPILER_FNS.pop(NAME, None)  # register_backend refuses a name already in the table
        registry.register_backend(late, name=NAME)
        undo()
        assert registry.lookup_backend(NAME) is late
    finally:
        registry._COMPILER_FNS.pop(NAME, None)
        registry._BACKENDS.pop(NAME, None)


# ---- enable() on a stub Gaudi stack


@pytest.fixture
def stub_gaudi(monkeypatch):
    """A Gaudi stack of stubs: ``habana_frameworks.torch.core`` imports and ``is_lazy()`` reads ``state["lazy"]``;
    the kernel path and the rewrites are no-ops, so only enable()'s own logic runs."""
    state = {"lazy": True}
    internal = types.ModuleType("habana_frameworks.torch.utils.internal")
    internal.is_lazy = lambda: state["lazy"]
    for name in ("habana_frameworks", "habana_frameworks.torch", "habana_frameworks.torch.core",
                 "habana_frameworks.torch.utils"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "habana_frameworks.torch.utils.internal", internal)
    monkeypatch.setattr(hpu_ops.tpc, "install_kernel_path", lambda: None)
    monkeypatch.setattr(hpu_ops, "_install", lambda device_type: lambda: None)
    monkeypatch.setattr(hpu_ops, "_attention_fused", lambda device_type="hpu": True)
    monkeypatch.setattr(lazy_compile, "NAME", NAME)
    for p in hpu_ops.BROKEN_PASSES:
        monkeypatch.setenv(f"PT_HPU_DISABLE_{p}", "True")
    yield state
    if hpu_ops._STATE.get("undo"):
        hpu_ops._STATE["undo"]()


@pytest.mark.parametrize("lazy", [True, False])
def test_enable_handles_compile_in_lazy_mode_only(backend, stub_gaudi, lazy):
    stub_gaudi["lazy"] = lazy
    model = torch.nn.Linear(3, 2)
    hpu_ops.enable()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        compiled = torch.compile(model, backend=NAME)
        compiled(torch.randn(4, 3))
    if lazy:
        assert compiled is model and not backend and len(_messages(caught)) == 1
    else:
        assert compiled is not model and backend and not _messages(caught)
    hpu_ops._STATE["undo"]()
    backend.clear()
    torch._dynamo.reset()  # else the eager case's cached graph is reused without the backend
    torch.compile(model, backend=NAME)(torch.randn(4, 3))  # undone: the backend compiles again
    assert backend


def test_enable_in_raise_mode_raises(backend, stub_gaudi, raise_mode):
    hpu_ops.enable()
    with pytest.raises(RuntimeError, match="PT_HPU_LAZY_MODE=0"):
        torch.compile(torch.nn.Linear(3, 2), backend=NAME)


def test_enable_undo_enable_in_lazy_mode_guards_again(backend, stub_gaudi):
    model = torch.nn.Linear(3, 2)
    hpu_ops.enable()
    first = hpu_ops._STATE["undo"]
    hpu_ops.enable()  # idempotent: one guard, one undo
    assert hpu_ops._STATE["undo"] is first
    first()
    assert torch.compile(model, backend=NAME) is not model
    hpu_ops.enable()
    with pytest.warns(UserWarning, match="lazy mode"):
        assert torch.compile(model, backend=NAME) is model


def test_enable_rolls_back_the_rewrites_when_the_guard_fails(stub_gaudi, monkeypatch):
    undone = []
    monkeypatch.setattr(hpu_ops, "_install", lambda device_type: lambda: undone.append(device_type))

    def broken(name):
        raise ImportError("a broken optional backend")
    monkeypatch.setattr(lazy_compile, "install", broken)
    with pytest.raises(ImportError):
        hpu_ops.enable()
    assert undone == ["hpu"] and not hpu_ops._STATE.get("undo")


@pytest.mark.parametrize(("env", "lazy"), [("1", True), ("0", False), (None, False)])
def test_lazy_mode_from_the_environment_without_habanas_is_lazy(monkeypatch, env, lazy):
    """A bridge without ``habana_frameworks.torch.utils.internal``: the setting itself, eager by default (1.21)."""
    monkeypatch.setitem(sys.modules, "habana_frameworks.torch.utils.internal", None)  # import raises ImportError
    if env is None:
        monkeypatch.delenv("PT_HPU_LAZY_MODE", raising=False)
    else:
        monkeypatch.setenv("PT_HPU_LAZY_MODE", env)
    assert hpu_ops._lazy() is lazy


def test_no_registry_entry_leaks():
    """Runs after the tests above (file order): none left the test backend's name in torch's registry, nor
    torch.compile patched."""
    assert NAME not in registry._BACKENDS and NAME not in registry._COMPILER_FNS
    assert torch.compile.__module__ == "torch"
