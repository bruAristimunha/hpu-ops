"""``torch.compile(backend="hpu_backend")`` in lazy mode refused at the call (``hpu_ops.lazy_compile``), on the CPU:
a backend registered under a test name stands in for Habana's, and a stub Gaudi stack for ``enable()``."""
import sys
import types

import pytest
import torch
from torch._dynamo.backends import registry

import hpu_ops
from hpu_ops import lazy_compile

NAME = "hpu_ops_test_backend"


@pytest.fixture
def backend():
    """A working backend registered as ``NAME`` (as Habana registers ``hpu_backend``), removed afterwards."""
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


def test_guard_raises_one_line_at_torch_compile_and_undo_restores(backend):
    undo = lazy_compile.install(NAME)
    try:
        with pytest.raises(RuntimeError, match="PT_HPU_LAZY_MODE=0") as info:
            torch.compile(torch.nn.Linear(3, 2), backend=NAME)  # at the call, before any graph is captured
        assert "\n" not in str(info.value) and "lazy mode" in str(info.value)
        assert NAME in registry.list_backends()  # still listed: torch's listing does not load entry points
    finally:
        undo()
    out = torch.compile(torch.nn.Linear(3, 2), backend=NAME)(torch.randn(4, 3))
    assert out.shape == (4, 2) and backend, "the original backend compiles again after the undo"


def test_guard_of_an_unregistered_name_is_undone_to_nothing():
    undo = lazy_compile.install(NAME)
    assert NAME in registry._BACKENDS
    undo()
    assert NAME not in registry._BACKENDS and NAME not in registry._COMPILER_FNS


def test_guard_survives_a_later_entry_point_discovery(backend):
    """torch fills its table from entry points once (cached); the guard runs that first, so it is not overwritten."""
    undo = lazy_compile.install(NAME)
    try:
        registry._lazy_import()
        registry._discover_entrypoint_backends()
        with pytest.raises(RuntimeError, match="lazy mode"):
            registry.lookup_backend(NAME)
    finally:
        undo()


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
def test_enable_guards_compile_in_lazy_mode_only(backend, stub_gaudi, lazy):
    stub_gaudi["lazy"] = lazy
    hpu_ops.enable()
    if lazy:
        with pytest.raises(RuntimeError, match="PT_HPU_LAZY_MODE=0"):
            torch.compile(torch.nn.Linear(3, 2), backend=NAME)
    else:
        torch.compile(torch.nn.Linear(3, 2), backend=NAME)(torch.randn(4, 3))
        assert backend
    hpu_ops._STATE["undo"]()
    torch.compile(torch.nn.Linear(3, 2), backend=NAME)(torch.randn(4, 3))  # undone: the backend compiles again
    assert backend


def test_enable_undo_enable_in_lazy_mode_guards_again(backend, stub_gaudi):
    hpu_ops.enable()
    first = hpu_ops._STATE["undo"]
    hpu_ops.enable()  # idempotent: one guard, one undo
    assert hpu_ops._STATE["undo"] is first
    first()
    torch.compile(torch.nn.Linear(3, 2), backend=NAME)(torch.randn(4, 3))
    hpu_ops.enable()
    with pytest.raises(RuntimeError, match="lazy mode"):
        torch.compile(torch.nn.Linear(3, 2), backend=NAME)


@pytest.mark.parametrize(("env", "lazy"), [("1", True), ("0", False), (None, False)])
def test_lazy_mode_from_the_environment_without_habanas_is_lazy(monkeypatch, env, lazy):
    """A bridge without ``habana_frameworks.torch.utils.internal``: the setting itself, eager by default (1.21)."""
    monkeypatch.setitem(sys.modules, "habana_frameworks.torch.utils.internal", None)  # import raises ImportError
    if env is None:
        monkeypatch.delenv("PT_HPU_LAZY_MODE", raising=False)
    else:
        monkeypatch.setenv("PT_HPU_LAZY_MODE", env)
    assert hpu_ops._lazy() is lazy


def test_guard_covers_dynamo_optimize_by_name(backend):
    undo = lazy_compile.install(NAME)
    try:
        with pytest.raises(RuntimeError, match="lazy mode"):
            torch._dynamo.optimize(NAME)
    finally:
        undo()


def test_a_backend_registered_after_the_guard_is_kept_by_the_undo():
    """Registered after install() (a late import of its package), the backend replaces the guard; the undo leaves it."""
    undo = lazy_compile.install(NAME)
    try:
        def late(gm, example_inputs):
            return gm.forward
        registry.register_backend(late, name=NAME)
        undo()
        assert registry.lookup_backend(NAME) is late
    finally:
        registry._COMPILER_FNS.pop(NAME, None)
        registry._BACKENDS.pop(NAME, None)


def test_enable_rolls_back_the_rewrites_when_the_guard_fails(stub_gaudi, monkeypatch):
    undone = []
    monkeypatch.setattr(hpu_ops, "_install", lambda device_type: lambda: undone.append(device_type))

    def broken(name):
        raise ImportError("a broken optional backend")
    monkeypatch.setattr(lazy_compile, "install", broken)
    with pytest.raises(ImportError):
        hpu_ops.enable()
    assert undone == ["hpu"] and not hpu_ops._STATE.get("undo")


def test_no_registry_entry_leaks():
    """Runs after the tests above (file order): none left the test backend's name in torch's registry."""
    assert NAME not in registry._BACKENDS and NAME not in registry._COMPILER_FNS
