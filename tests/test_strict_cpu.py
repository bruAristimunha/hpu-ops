"""Strict mode on the CPU: the bridge's fallback metric and hpu_complex's counters are stand-ins (monkeypatched), and
CPU tensors take the device's role where a test patches ``_is_hpu``."""
import io
import os

import pytest
import torch

import hpu_ops
from torch.utils._python_dispatch import _get_current_dispatch_mode_stack
from hpu_ops import _strict, linalg, serialization

aten = torch.ops.aten
HERE = os.path.basename(__file__)
pytestmark = pytest.mark.filterwarnings("ignore:Warning only once")


@pytest.fixture(autouse=True)
def clean():
    hpu_ops.strict_reset()
    yield
    hpu_ops.strict_reset()
    assert not _strict.active()


class FakeBridge:
    """The bridge's ``cpu_fallback`` metric: ``fall(op)`` counts one fallback of ``op``."""

    def __init__(self):
        self.ops = {}

    def fall(self, op, n=1):
        self.ops[op] = self.ops.get(op, 0) + n

    def stats(self):
        return sum(self.ops.values()), dict(self.ops)


@pytest.fixture
def bridge(monkeypatch):
    fake = FakeBridge()
    monkeypatch.setattr(_strict, "_bridge_stats", fake.stats)
    return fake


def test_route_raises_with_op_shape_dtype_and_caller():
    A = torch.randn(3, 4, 4)
    with pytest.raises(hpu_ops.CPUExecutionError) as info:
        with hpu_ops.strict():
            linalg._on_cpu(aten.linalg_inv.default, (A,), {})
    message = str(info.value)
    assert "aten::linalg_inv" in message and "(3, 4, 4) float32" in message and "hpu_ops route" in message
    assert HERE in message  # the caller's frame, not hpu_ops' or torch's
    assert isinstance(info.value, RuntimeError)


def test_out_route_raises():
    A, B = torch.randn(4, 4).triu() + 4 * torch.eye(4), torch.randn(4, 2)
    out = torch.empty(4, 2)
    with pytest.raises(hpu_ops.CPUExecutionError, match="linalg_solve_triangular"):
        with hpu_ops.strict():
            linalg._out_on_cpu(aten.linalg_solve_triangular.out, (A, B), dict(upper=True), dict(out=out))


def test_report_counts_and_formats():
    A = torch.randn(2, 3, 3)
    with hpu_ops.strict("report"):
        for _ in range(3):
            linalg._on_cpu(aten.linalg_inv.default, (A,), {})
    rows = hpu_ops.strict_report()
    assert len(rows) == 1
    source, op, sig, n, where = rows[0]
    assert (source, op, sig, n) == ("hpu_ops route", "aten::linalg_inv", "(2, 3, 3) float32 cpu", 3)
    assert HERE in where
    table = hpu_ops.strict_format_report()
    assert "aten::linalg_inv" in table and "3 CPU executions" in table


def test_route_hides_its_own_cpu_ops():
    """The CPU ops inside a named route are that route, not ``cpu op`` rows of their own."""
    A = torch.randn(2, 3, 3)
    with hpu_ops.strict("report"):
        linalg._on_cpu(aten.linalg_inv.default, (A,), {})
    assert [r[0] for r in hpu_ops.strict_report()] == ["hpu_ops route"]


def test_cpu_ops_are_rows_unless_dropped():
    x = torch.randn(5)
    with hpu_ops.strict("report"):
        x * 2
    assert any(r[0] == "cpu op" and r[1] == "aten::mul.Tensor" for r in hpu_ops.strict_report())
    hpu_ops.strict_reset()
    with hpu_ops.strict("report", watch_cpu_ops=False):
        x * 2
    assert hpu_ops.strict_report() == []
    with hpu_ops.strict("raise", allow=("cpu op",)):
        x * 2  # allowed: no error


def test_allow_and_off():
    A = torch.randn(2, 3, 3)
    with hpu_ops.strict("raise", allow=("hpu_ops route", "cpu op")):
        linalg._on_cpu(aten.linalg_inv.default, (A,), {})
    linalg._on_cpu(aten.linalg_inv.default, (A,), {})  # off: nothing
    assert hpu_ops.strict_report() == []
    with pytest.raises(ValueError):
        with hpu_ops.strict("loud"):
            pass
    with pytest.raises(ValueError):
        with hpu_ops.strict(allow=("gpu",)):
            pass


def test_bridge_fallback_attributed_per_op(bridge):
    """An op during which the bridge's count rises is named, with the bridge's op, the tensors and the caller; a
    count that rose between ops is a per-step row."""
    x = torch.randn(4, 2)
    with hpu_ops.strict("report", watch_cpu_ops=False):
        bridge.fall("aten::before")  # before the op: not this op's (the block's baseline is earlier: step check)
        with torch.library._scoped_library("aten", "IMPL") as lib:
            lib.impl("special_zeta", lambda a, b: (bridge.fall("aten::special_zeta"), a + b)[1], "CPU")
            torch.special.zeta(x, x)
    rows = {(r[0], r[1]): r for r in hpu_ops.strict_report()}
    zeta = rows[("bridge fallback", "aten::special_zeta")]
    assert zeta[3] == 1 and zeta[2] == "(4, 2) float32 cpu, (4, 2) float32 cpu" and HERE in zeta[4]
    assert rows[("bridge fallback", "aten::before")][4] == "(per step)"


def test_bridge_fallback_raises(bridge):
    x = torch.randn(3)
    with torch.library._scoped_library("aten", "IMPL") as lib:
        lib.impl("special_zeta", lambda a, b: (bridge.fall("aten::special_zeta"), a + b)[1], "CPU")
        with pytest.raises(hpu_ops.CPUExecutionError, match="special_zeta.*bridge fallback") as info:
            with hpu_ops.strict(allow=("cpu op",)):
                torch.special.zeta(x, x)
    assert HERE in str(info.value)


def test_step_check_without_the_watch(bridge):
    """``per_op=False`` (compiled steps): the bridge's fallbacks are read when the block ends, named by op."""
    with hpu_ops.strict("report", per_op=False):
        bridge.fall("aten::linalg_eigh", 3)
    assert hpu_ops.strict_report()[0][:4] == ("bridge fallback", "aten::linalg_eigh", "", 3)
    hpu_ops.strict_reset()
    with pytest.raises(hpu_ops.CPUExecutionError, match="linalg_eigh"):
        with hpu_ops.strict(per_op=False):
            bridge.fall("aten::linalg_eigh")


def test_check_mid_block(bridge):
    with hpu_ops.strict("report", per_op=False):
        bridge.fall("aten::a")
        hpu_ops.strict_check()
        bridge.fall("aten::a")
    assert hpu_ops.strict_report()[0][3] == 2  # not counted twice


def test_hpu_complex_counter(monkeypatch):
    calls = {"n": 0}

    def resolve():
        _strict._COMPLEX.update(cpu_fallback_calls=lambda: calls["n"], is_host_backed=lambda t: False)

    monkeypatch.setattr(_strict, "_resolve_complex", resolve)
    x = torch.randn(3, dtype=torch.complex64)  # the counter is read around ops with a complex tensor
    with torch.library._scoped_library("aten", "IMPL") as lib:
        lib.impl("special_zeta", lambda a, b: (calls.update(n=calls["n"] + 1), a + b)[1], "CPU")
        with hpu_ops.strict("report", watch_cpu_ops=False):
            torch.special.zeta(x, x)
    assert ("hpu_complex", "aten::special_zeta") in {(r[0], r[1]) for r in hpu_ops.strict_report()}
    _strict._COMPLEX.clear()


def test_host_sync_and_copy_of_device_tensors(monkeypatch):
    """``.item()`` of a device tensor is a host sync; a device tensor copied to the CPU is a host copy (CPU tensors in
    the device's role)."""
    monkeypatch.setattr(_strict, "_is_hpu", lambda t: True)
    x = torch.randn(4)
    with hpu_ops.strict("report", watch_cpu_ops=False):
        x.sum().item()
    assert [r[:2] for r in hpu_ops.strict_report()] == [("host sync", "aten::_local_scalar_dense")]
    with pytest.raises(hpu_ops.CPUExecutionError, match="host sync"):
        with hpu_ops.strict(watch_cpu_ops=False):
            bool(x.sum() > 0)


def test_complex64_small_matmul(monkeypatch):
    monkeypatch.setattr(_strict, "_is_hpu", lambda t: True)
    z = torch.randn(2, 3, 3, dtype=torch.complex64)
    with hpu_ops.strict("report", watch_cpu_ops=False):
        torch.bmm(z, z)
    assert ("hpu_complex", "aten::bmm") in {(r[0], r[1]) for r in hpu_ops.strict_report()}


def test_det_double_backward_host_read():
    A = torch.randn(2, 3, 3, dtype=torch.float64)
    A[0, 0] = A[0, 1]  # singular
    det = torch.linalg.det(A)
    with pytest.raises(hpu_ops.CPUExecutionError, match="singular check"):
        with hpu_ops.strict(watch_cpu_ops=False), torch.enable_grad():
            # inv_ex: the card's inverse does not raise on a singular matrix (CPU torch's inv does from 2.15)
            linalg._det_backward(A, det, torch.ones_like(det), lambda M: torch.linalg.inv_ex(M)[0])


def test_save_is_allowed():
    undo = serialization.install("cpu")
    try:
        with hpu_ops.strict(watch_cpu_ops=False):
            torch.save(torch.randn(3), io.BytesIO())
    finally:
        undo()


def test_env(monkeypatch):
    monkeypatch.setenv("HPU_OPS_STRICT", "report")
    monkeypatch.setenv("HPU_OPS_STRICT_ALLOW", "cpu op")
    monkeypatch.setattr(_strict, "_STATE", {**_strict._STATE, "atexit": True})  # no table at the tests' exit
    try:
        assert _strict.from_env() == "report"
        assert _strict._config().level == "report" and _strict._config().allow == {"cpu op"}
        linalg._on_cpu(aten.linalg_inv.default, (torch.randn(2, 2) + 2 * torch.eye(2),), {})
        assert hpu_ops.strict_report()[0][1] == "aten::linalg_inv"
        assert _strict.from_env() == "report"  # once per process: no second watch
        assert sum(isinstance(m, _strict._Watch) for m in _get_current_dispatch_mode_stack()) == 1
    finally:
        _strict._STATE.pop("env_config")
        _strict._STATE.pop("env_block").exit(False)
    monkeypatch.setenv("HPU_OPS_STRICT", "0")
    assert _strict.from_env() is None


def test_compiled_code_runs_eagerly_in_a_block_and_compiles_after():
    """Inside a per-op block compiled code runs eagerly (force_eager): no frame is skipped for good (a plain dispatch
    mode: 0 graphs over 1 strict + 3 plain steps) and nothing recompiles at every call (an infra mode: the guards see
    the mode's dispatch keys)."""
    compiles = {"n": 0}

    def backend(gm, example_inputs):
        compiles["n"] += 1
        return gm.forward

    class Net(torch.nn.Module):  # a class of its own: Dynamo's skips are per code object
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(8, 8)

        def forward(self, x):
            return torch.relu(self.lin(x))

    torch._dynamo.reset()
    model = torch.compile(Net(), backend=backend)
    x = torch.randn(4, 8)
    with hpu_ops.strict("report", watch_cpu_ops=False):
        for _ in range(4):
            model(x)
    assert compiles["n"] == 0
    for _ in range(3):
        model(x)
    assert compiles["n"] == 1
    model(torch.randn(5, 8))  # a new shape after the block compiles too
    assert compiles["n"] == 2


def test_nested_blocks_count_once():
    x = torch.randn(5)
    with hpu_ops.strict("report"), hpu_ops.strict("report"):
        x * 2
    assert [r[3] for r in hpu_ops.strict_report() if r[1] == "aten::mul.Tensor"] == [1]


def test_inner_block_keeps_outer_fallbacks(bridge):
    with hpu_ops.strict("report", per_op=False):
        bridge.fall("aten::a", 3)
        with hpu_ops.strict("report", per_op=False):
            pass
    assert sum(r[3] for r in hpu_ops.strict_report()) == 3


def test_a_raised_fallback_is_not_raised_again(bridge):
    x = torch.randn(3)
    with hpu_ops.strict(allow=("cpu op",)):
        bridge.fall("aten::a")
        with pytest.raises(hpu_ops.CPUExecutionError):
            x + 1
        x + 1  # the same fallback does not raise twice


def test_host_to_device_copy_is_not_a_cpu_op(monkeypatch):
    monkeypatch.setattr(_strict, "_is_hpu", lambda t: t.dtype == torch.float64)  # float64 plays the card
    x = torch.randn(4)
    with hpu_ops.strict():
        x.to(torch.float64)  # the "copy to the card"


def test_host_read_is_one_row(monkeypatch):
    """hpu-ops' own read is one ``host sync`` row, not a second one for the ``_local_scalar_dense`` inside."""
    monkeypatch.setattr(_strict, "_is_hpu", lambda t: True)
    x = torch.randn(4)
    with hpu_ops.strict("report", watch_cpu_ops=False):
        with _strict.host_read("test: read", x):
            bool(x.sum() > 0)
    assert [(r[0], r[1], r[3]) for r in hpu_ops.strict_report()] == [("host sync", "test: read", 1)]


def test_forked_child_is_off():
    with hpu_ops.strict():
        _strict._reset_in_child()
        assert not _strict.active()
    assert not _strict.active()


def test_the_watch_is_not_a_foreign_mode():
    """Rewrites that decline under a tracer's dispatch mode (bincount) take the same route under the watch."""
    from torch.utils._python_dispatch import TorchDispatchMode

    with hpu_ops.strict("report"):
        assert not _strict.foreign_modes()
        with TorchDispatchMode():
            assert _strict.foreign_modes()


def test_save_frames_are_skipped():
    with hpu_ops.strict(watch_cpu_ops=False):
        torch.save(torch.randn(3), io.BytesIO())


def test_threads_have_their_own_blocks():
    import threading

    seen = {}

    def other():
        seen["active"] = _strict.active()
        with hpu_ops.strict("report", watch_cpu_ops=False):
            seen["inner"] = _strict.active()

    with hpu_ops.strict():
        thread = threading.Thread(target=other)
        thread.start()
        thread.join()
    assert seen == {"active": False, "inner": True}
    assert not _strict.active()


def test_inner_block_settings_apply():
    x = torch.randn(5)
    with hpu_ops.strict("raise"):
        with hpu_ops.strict("raise", watch_cpu_ops=False):
            x * 2  # no cpu op row inside the inner block
        with pytest.raises(hpu_ops.CPUExecutionError):
            x * 2


def test_views_and_metadata_are_not_cpu_ops():
    x = torch.randn(4, 3)
    with hpu_ops.strict():
        x.t()
        x.view(12)
        x.detach()
        torch.empty(3, 3)


def test_higher_order_ops_run_and_their_bodies_are_watched():
    x = torch.randn(3)
    with hpu_ops.strict("report", watch_cpu_ops=False):
        out = torch.cond(x.sum() > 0, lambda v: v + 1, lambda v: v - 1, (x,))
    assert torch.equal(out, x + 1 if x.sum() > 0 else x - 1)
    with hpu_ops.strict("report"):
        torch.cond(x.sum() > 0, lambda v: v * 2, lambda v: v * 3, (x,))
        n = torch._higher_order_ops.while_loop(lambda i: i < 3, lambda i: (i + 1,), (torch.tensor(0),))[0]
    assert int(n) == 3
    ops = {r[1] for r in hpu_ops.strict_report() if r[0] == "cpu op"}
    assert "aten::mul.Tensor" in ops  # the branch's op, inside cond


def test_fake_tensors_are_not_cpu_ops():
    from torch._subclasses.fake_tensor import FakeTensorMode

    with hpu_ops.strict(), FakeTensorMode():
        torch.randn(4) * torch.zeros(4)


def test_per_op_env_is_case_insensitive(monkeypatch):
    monkeypatch.setenv("HPU_OPS_STRICT", "report")
    monkeypatch.setenv("HPU_OPS_STRICT_PER_OP", "OFF")
    monkeypatch.setattr(_strict, "_STATE", {**_strict._STATE, "atexit": True})
    try:
        assert _strict.from_env() == "report"
        assert not any(isinstance(m, _strict._Watch) for m in _get_current_dispatch_mode_stack())
    finally:
        _strict._STATE.pop("env_config")
        _strict._STATE.pop("env_block").exit(False)


def _counting_backend(stats):
    def backend(gm, example_inputs):
        stats["compiles"] += 1

        def run(*args):
            stats["runs"] += 1
            return gm.forward(*args)

        return run

    return backend


def test_a_block_runs_compiled_code_eagerly_on_its_own_thread_only():
    """Round 3, N1: the force_eager stance is per thread. Another thread compiles while a block is open, and blocks
    that exit out of order leave compilation on."""
    import threading

    stats = {"compiles": 0, "runs": 0}
    fn = torch.compile(lambda x: torch.relu(x) * 2, backend=_counting_backend(stats))
    entered, release = threading.Event(), threading.Event()

    def holder():
        with hpu_ops.strict("report", watch_cpu_ops=False):
            entered.set()
            release.wait()

    torch._dynamo.reset()
    thread = threading.Thread(target=holder)
    thread.start()
    assert entered.wait(30)
    fn(torch.randn(4))  # this thread is in no block: compiled
    assert stats == {"compiles": 1, "runs": 1}
    with hpu_ops.strict("report", watch_cpu_ops=False):
        fn(torch.randn(4))  # this one is: eager
    assert stats["runs"] == 1
    release.set()
    thread.join()
    fn(torch.randn(4))
    assert stats["runs"] == 2


def test_exit_removes_the_watch_not_a_mode_entered_in_the_block():
    """Round 3, N2: a mode entered inside the block and left on stays; the watch goes."""
    from torch.utils._python_dispatch import TorchDispatchMode

    class Other(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            return func(*args, **(kwargs or {}))

    other = Other()
    with hpu_ops.strict("report"):
        other.__enter__()  # left on, as hpu_complex.enable(enter_mode=True) does
    stack = _get_current_dispatch_mode_stack()
    try:
        assert stack == [other]
        assert not _strict.active()
    finally:
        other.__exit__(None, None, None)
    assert _get_current_dispatch_mode_stack() == []


def test_exit_keeps_an_unreferenced_mode_alive():
    """Round 3, N2: the stack's reference is the only one to a mode entered with ``Mode().__enter__()``; taking the
    watch out from under it must not free it (a use-after-free crashed the process)."""
    from torch.utils._python_dispatch import TorchDispatchMode, _pop_mode

    class Other(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            return func(*args, **(kwargs or {}))

    with hpu_ops.strict("report", watch_cpu_ops=False):
        Other().__enter__()
    stack = _get_current_dispatch_mode_stack()
    assert [type(m).__name__ for m in stack] == ["Other"]
    torch.randn(3) + 1  # dispatches through it
    _pop_mode()
    del stack


def test_the_watch_carries_the_innermost_settings():
    """Round 3, N3: autograd's threads read the watch's settings, which follow the innermost block."""
    with hpu_ops.strict("raise"):
        outer = _strict._watch_on_stack().config
        with hpu_ops.strict("report", allow=("host sync",)):
            inner = _strict._watch_on_stack().config
            assert inner.level == "report" and inner.allow == {"host sync"}
        assert _strict._watch_on_stack().config is outer


def test_dataloader_worker_loop_is_off(monkeypatch):
    """Round 3, N5: a DataLoader worker (forked or spawned) computes on the CPU by design."""
    import torch.utils.data._utils.worker as worker

    with hpu_ops.strict():
        monkeypatch.setattr(worker, "_worker_info", object())
        torch.randn(5) * 2  # no cpu op raised in the worker loop
        monkeypatch.setattr(worker, "_worker_info", None)
        with pytest.raises(hpu_ops.CPUExecutionError):
            torch.randn(5) * 2


def test_fork_reset_removes_the_watch_under_another_mode():
    """Round 3, N6: the watch goes from a forked child's stack even below another mode."""
    from torch.utils._python_dispatch import TorchDispatchMode

    class Other(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            return func(*args, **(kwargs or {}))

    with hpu_ops.strict():
        with Other() as other:
            _strict._reset_in_child()  # what os.register_at_fork runs in the child
            assert _get_current_dispatch_mode_stack() == [other]
            assert not _strict.active()
            assert not getattr(_strict._TLS, "eager", 0)
            torch.randn(5) * 2


def test_copies_are_cpu_ops_allocations_and_inplace_views_are_not():
    """Round 3, N7."""
    x = torch.randn(4, 3)
    with hpu_ops.strict("report"):
        x.clone()
        x.t().contiguous()
    assert {r[1] for r in hpu_ops.strict_report()} >= {"aten::clone"}
    with hpu_ops.strict():
        torch.empty_like(x)
        x.new_empty(3)
        y = x.clone() if False else torch.empty(4, 3)
        y.unsqueeze_(0)
        y.t_() if y.dim() == 2 else None


def test_off_routes_are_free():
    """Round 3, N9: strict mode off, a route is the shared no-op context."""
    assert _strict.route("x", torch.randn(2)) is _strict._NULL
    assert _strict.quiet() is _strict._NULL


def test_a_block_inside_compiled_code_runs_eagerly():
    """Round 3, N10: no Dynamo internal error; the block's frame runs eagerly."""

    @torch.compile(backend="eager")
    def f(x):
        with hpu_ops.strict("report", watch_cpu_ops=False):
            return x + 1

    torch._dynamo.reset()
    assert torch.equal(f(torch.ones(3)), torch.full((3,), 2.0))
