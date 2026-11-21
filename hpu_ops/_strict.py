"""Strict mode: every CPU execution of an HPU step that hpu-ops can see, raised or counted.

``HPU_OPS_STRICT=1`` (read by ``enable()``) or ``with hpu_ops.strict():`` raises ``CPUExecutionError`` with the op,
the shapes, dtypes and devices of its tensors, and the first Python frame of the caller outside torch, Habana,
hpu-ops and hpu-complex. ``HPU_OPS_STRICT=report`` (or ``strict("report")``) counts instead and prints one table at
exit (``report()`` returns the rows). The sources:

- ``bridge fallback``: Habana's CPU fallback, from the bridge's ``cpu_fallback`` metric read around every op.
- ``hpu_ops route``: hpu-ops' own CPU routes (``linalg._on_cpu`` / ``_out_on_cpu``, the singular det double backward's
  SVD, im2col's declined calls) and index tensors checked and built on the host.
- ``hpu_complex``: its CPU fallback (``torch.ops.hpu_complex.cpu_fallback_calls``), complex128 and host-backed
  float64 computed on the CPU in host memory, complex64 mm/bmm below its CPU threshold.
- ``host sync``: a device value read on the host (``.item()``, ``bool(t)``, ``.tolist()``), in hpu-ops or in the
  caller's code.
- ``host copy``: a device tensor copied to the CPU (``.cpu()``, ``.to("cpu")``), outside hpu-ops' named routes.
- ``cpu op``: an op computing CPU tensors of more than one element inside the step (the caller's or a library's);
  views, metadata ops and copies of CPU tensors to the card are not.

Per-op attribution is a ``TorchDispatchMode`` (``per_op=True``, the default), pushed on the thread's mode stack
without ``TorchDispatchMode.__enter__``'s process-wide flags. A dispatch mode and torch.compile do not mix (Dynamo
skips the frames for good, or, as an infra mode, recompiles at every call: its guards see the mode's dispatch
keys), so inside a per-op block torch.compile'd code runs eagerly on that thread (Dynamo's ``force_eager`` stance,
per thread; process-wide on a torch without the hook, counted over the open blocks): the check is of the eager
program. ``per_op=False`` (``HPU_OPS_STRICT_PER_OP=0``) leaves compilation alone and reads the bridge's and
hpu-complex's counters when the block ends (or at ``check()``), named by op, without a stack frame.
``HPU_OPS_STRICT_ALLOW`` (or ``allow=``) lists sources, comma-separated, that are not raised nor counted, e.g.
``host sync,cpu op``. Strict mode is per thread (autograd's threads follow the innermost block's settings through the
watch), off in DataLoader workers (forked or spawned: their worker loop) and while ``torch.save`` runs.
"""
import atexit
import collections
import contextlib
import os
import sys
import threading

import torch
from torch._subclasses.fake_tensor import is_fake
from torch.utils._python_dispatch import TorchDispatchMode, _get_current_dispatch_mode_stack

__all__ = ["CPUExecutionError", "strict", "record", "route", "host_read", "allowed", "check", "report", "reset"]

ENV, ENV_PER_OP, ENV_ALLOW = "HPU_OPS_STRICT", "HPU_OPS_STRICT_PER_OP", "HPU_OPS_STRICT_ALLOW"
SOURCES = ("bridge fallback", "hpu_ops route", "hpu_complex", "host sync", "host copy", "cpu op")
COMPLEX_MATMUL_CPU_BELOW = 1 << 22  # hpu_complex's default threshold, tracked through set_matmul_cpu_threshold
OFF = ("", "0", "false", "off", "no")
aten = torch.ops.aten


class CPUExecutionError(RuntimeError):
    """An HPU step ran something on the CPU (strict mode)."""


class _Config:
    """One strict block's settings."""

    def __init__(self, level, allow=(), watch_cpu_ops=True):
        self.level, self.allow, self.watch_cpu_ops = level, frozenset(allow), watch_cpu_ops


# Counts and the counters' baseline are process-wide (the bridge's metric is); the settings are per thread.
_STATE = {"counts": collections.Counter(), "where": {}, "baseline": None, "atexit": False,
          "matmul_cpu_below": COMPLEX_MATMUL_CPU_BELOW, "generation": 0}  # generation: forks since import
_TLS = threading.local()
_LOCK = threading.Lock()
_ACTIVE = {"n": 0}  # open strict blocks, every thread: off, the hooks in hpu-ops' CPU routes return at once
_GLOBAL_EAGER = {"n": 0}  # the process-wide force_eager stance's blocks, on a torch without the per-thread hook
_EAGER_HOOK = {}
_HOPS = {"seen": -1}


def _worker_info():
    """The DataLoader worker this process runs, if any (its loop computes on the CPU by design)."""
    worker = sys.modules.get("torch.utils.data._utils.worker")
    return getattr(worker, "_worker_info", None) if worker is not None else None


def _watch_on_stack():
    for mode in reversed(_get_current_dispatch_mode_stack()):
        if isinstance(mode, _Watch):
            return mode
    return None


def _config():
    """The settings in effect on this thread: its innermost block's, else (autograd's threads, the watch's own
    handler) the watch's, which follows the innermost block of the thread that owns it. None in a DataLoader worker."""
    if not _ACTIVE["n"] or _worker_info() is not None:
        return None
    blocks = getattr(_TLS, "blocks", None)
    if blocks:
        return blocks[-1]
    watch = getattr(_TLS, "watch", None) or _watch_on_stack()
    return watch.config if watch is not None else None


def active():
    """Whether strict mode is on (raise or report) on this thread."""
    return _config() is not None


def _push_watch(watch):
    from torch.utils._python_dispatch import _push_mode

    _push_mode(watch)


def _remove_watch(watch):
    """``watch`` off this thread's mode stack wherever it sits; the modes above it stay, in order (a mode entered in
    the block and left on, as ``hpu_complex.enable(enter_mode=True)`` does). ``TorchDispatchMode.__exit__`` would pop
    the top one instead."""
    from torch.utils._python_dispatch import _pop_mode, _push_mode

    stack = _get_current_dispatch_mode_stack()  # holds a reference to each mode: _pop_mode returns the stack's own,
    at = next((i for i, m in enumerate(stack) if m is watch), None)  # freed with it when nothing else holds one
    if at is None:
        return
    above = [m for m in stack[at + 1:] if getattr(m, "_mode_key", None) is None]  # infra modes have their own slots
    for _ in range(len(above) + 1):
        _pop_mode()
    for mode in above:
        _push_mode(mode)


def _reset_in_child():
    """A forked child (a DataLoader worker) computes on the CPU by design: strict mode off, every watch off its mode
    stack, Dynamo's stance back."""
    for mode in list(_get_current_dispatch_mode_stack()):
        if isinstance(mode, _Watch):
            _remove_watch(mode)
    _TLS.blocks, _TLS.watch, _TLS.eager, _TLS.depth = [], None, 0, 0
    _ACTIVE["n"] = 0
    _STATE["generation"] += 1  # the blocks open at the fork undo nothing when they exit in the child
    _STATE.pop("env_config", None)
    if _GLOBAL_EAGER["n"]:
        _GLOBAL_EAGER["n"] = 0
        _GLOBAL_EAGER.pop("cm").__exit__(None, None, None)


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_in_child)


# ---- where a CPU execution comes from ----------------------------------------------------------------------------

def _skip_dirs():
    import hpu_ops

    dirs = [os.path.dirname(torch.__file__), os.path.dirname(hpu_ops.__file__)]
    for name in ("habana_frameworks", "hpu_complex"):
        module = sys.modules.get(name)
        paths = list(getattr(module, "__path__", None) or [])  # habana_frameworks is a namespace package
        if not paths and getattr(module, "__file__", None):
            paths = [os.path.dirname(module.__file__)]
        dirs.extend(paths)
    return tuple(d + os.sep for d in dirs)


_STDLIB = os.path.dirname(contextlib.__file__)


def _stdlib(name):
    """A file of the stdlib (its subpackages too, not site-packages below it, as in conda)."""
    return name.startswith(_STDLIB + os.sep) and "site-packages" not in name and "dist-packages" not in name


def _caller():
    """``file:line in function`` of the first frame outside torch, Habana, hpu-ops, hpu-complex, the stdlib and FX's
    generated code; ``None`` inside ``torch.save``."""
    skip, found = _skip_dirs(), None
    frame = sys._getframe(2)
    while frame is not None:
        name = frame.f_code.co_filename
        if name.endswith(os.path.join("torch", "serialization.py")):
            return None  # saving is I/O (hpu-complex's complex128 save reduces through .cpu())
        if found is None and not name.startswith(skip) and not _stdlib(name) \
                and not name.startswith(("<frozen", "<eval_with_key")):
            found = f"{name}:{frame.f_lineno} in {frame.f_code.co_name}"
        frame = frame.f_back
    return found or "(no Python caller: autograd's backward)"


def _signature(tensors):
    parts = []
    for t in tensors:
        if isinstance(t, torch.Tensor):
            parts.append(f"{tuple(t.shape)} {str(t.dtype).replace('torch.', '')} {t.device.type}")
    return ", ".join(parts)


def _fake(t):
    """A tensor no data is behind: fake, or a functional wrapper of one (AOT autograd's)."""
    if is_fake(t):
        return True
    if torch._is_functional_tensor(t):
        return True
    elem = getattr(t, "elem", None)
    return isinstance(elem, torch.Tensor) and is_fake(elem)


def _tracing(tensors):
    if torch.compiler.is_compiling() or torch._guards.detect_fake_mode() is not None:
        return True
    return any(isinstance(t, torch.Tensor) and _fake(t) for t in tensors)


def record(source, op, tensors=(), detail="", n=1, where=None, config=None):
    """One CPU execution of ``op`` from ``source``: raised in strict mode, counted in report mode, else nothing."""
    config = config or _config()
    if config is None or source in config.allow or getattr(_TLS, "allowed", 0):
        return
    if where is None:
        where = _caller()
        if where is None:
            return
    if _tracing(tensors):
        detail = (detail + "; " if detail else "") + "while a graph is traced: in the graph, every step"
    sig = _signature(tensors)
    if config.level == "raise":
        raise CPUExecutionError(f"hpu_ops strict: {op} runs on the CPU [{source}]"
                                + (f" ({sig})" if sig else "") + (f": {detail}" if detail else "")
                                + f"\n  called from {where}")
    key = (source, str(op), sig)
    _STATE["counts"][key] += n
    _STATE["where"].setdefault(key, where)


@contextlib.contextmanager
def _depth():
    _TLS.depth = getattr(_TLS, "depth", 0) + 1
    try:
        yield
    finally:
        _TLS.depth -= 1


_NULL = contextlib.nullcontext()


def _op_name(op):
    name = getattr(op, "name", None)
    return name() if callable(name) else str(op)


def route(op, *tensors, detail="", source="hpu_ops route", leaves=None):
    """A named CPU route of hpu-ops around its body: recorded once; the copies, syncs and CPU ops inside are not
    recorded again. ``leaves``: a pytree whose tensors are the route's, flattened only when strict mode is on. A
    no-op context when strict mode is off."""
    if not _ACTIVE["n"] or _config() is None:
        return _NULL
    if getattr(_TLS, "depth", 0):  # inside another route: part of it, not recorded again
        return _depth()
    if leaves is not None:
        import torch.utils._pytree as pytree

        tensors = tensors + tuple(pytree.tree_leaves(leaves))
    record(source, _op_name(op), tensors, detail() if callable(detail) else detail)
    return _depth()


def host_read(op, tensor):
    """A context around hpu-ops' own read of ``tensor`` (on the device) on the host (``bool``, ``item``,
    ``tolist``): one ``host sync`` row, not a second one for the ``_local_scalar_dense`` inside."""
    return route(op, tensor, detail="a device value read on the host", source="host sync")


def quiet():
    """CPU work inside is part of a route recorded elsewhere: not recorded."""
    return _depth() if _ACTIVE["n"] and _config() is not None else _NULL


@contextlib.contextmanager
def allowed():
    """CPU executions inside are neither raised nor counted (``torch.save``'s host copy)."""
    _TLS.allowed = getattr(_TLS, "allowed", 0) + 1
    try:
        yield
    finally:
        _TLS.allowed -= 1


def foreign_modes():
    """Whether a dispatch mode other than strict mode's watch is active (a tracer's), for rewrites that read values
    on the host only outside every tracer: the watch must not change which route they take."""
    return any(not isinstance(m, _Watch) for m in _get_current_dispatch_mode_stack())


# ---- what the bridge and hpu-complex count ----------------------------------------------------------------------

_METRIC = {}


def _bridge_stats():
    """``(total, {op: count})`` of Habana's CPU fallback so far; ``(0, {})`` off a Gaudi stack."""
    if "m" not in _METRIC:
        try:
            from habana_frameworks.torch.hpu.metrics import metric_global

            _METRIC["m"] = metric_global("cpu_fallback")
        except Exception:  # noqa: BLE001  (no Habana, or no such metric on this bridge)
            _METRIC["m"] = None
    metric = _METRIC["m"]
    if metric is None:
        return 0, {}
    stats = dict(metric.stats())
    return stats.get("TotalNumber", 0), dict(stats.get("FallbackOps", {}) or {})


_COMPLEX = {}


def _resolve_complex():
    """hpu_complex's counters (looked up again while missing: its extension may load after strict mode starts), and
    its matmul threshold, followed through its Python setter from then on (the extension has no getter)."""
    if _COMPLEX.get("cpu_fallback_calls") is not None:
        return
    ns = torch.ops.hpu_complex
    for name in ("cpu_fallback_calls", "is_host_backed"):
        try:
            _COMPLEX[name] = getattr(ns, name)
        except (AttributeError, RuntimeError):
            _COMPLEX[name] = None
    module = sys.modules.get("hpu_complex")
    setter = getattr(module, "set_matmul_cpu_threshold", None)
    if setter is not None and not getattr(setter, "_hpu_ops_strict", False):
        def set_matmul_cpu_threshold(macs):
            setter(macs)
            _STATE["matmul_cpu_below"] = int(macs)

        set_matmul_cpu_threshold._hpu_ops_strict = True
        set_matmul_cpu_threshold.__doc__ = setter.__doc__
        module.set_matmul_cpu_threshold = set_matmul_cpu_threshold


def _complex_calls():
    """hpu_complex's CPU-fallback call count (0 when its extension is not loaded)."""
    op = _COMPLEX.get("cpu_fallback_calls")
    if op is None:
        _resolve_complex()
        op = _COMPLEX.get("cpu_fallback_calls")
    return int(op()) if op is not None else 0


def _host_backed(t):
    op = _COMPLEX.get("is_host_backed")
    return bool(op(t)) if op is not None else False


def _snapshot():
    return _bridge_stats(), _complex_calls()


def check(config=None):
    """Record (or raise on) the bridge fallbacks and hpu-complex fallbacks since the last ``check()``, the last op the
    watch saw, or the start of the strict block, named by op, without a frame: what ran outside any op the watch saw
    (a compiled graph's recipes, another thread) or every fallback with ``per_op=False``."""
    config = config or _config()
    if config is None:
        return
    (total0, ops0), calls0 = _STATE["baseline"] or ((0, {}), 0)
    (total, ops), calls = _snapshot()
    _STATE["baseline"] = ((total, ops), calls)  # before recording: a raise must not report these again
    if total != total0:
        for op, value in ops.items():
            if value - ops0.get(op, 0):
                record("bridge fallback", op, (), "counted per step (not attributed to a call)",
                       n=value - ops0.get(op, 0), where="(per step)", config=config)
    if calls != calls0:
        record("hpu_complex", "hpu_complex cpu_fallback", (), "counted per step (not attributed to a call)",
               n=calls - calls0, where="(per step)", config=config)


# ---- the per-op watch --------------------------------------------------------------------------------------------

_COPIES = {aten._to_copy.default, aten.copy_.default, aten.to.device, aten.to.dtype_layout, aten.to.dtype,
           aten.to.other}
_MATMUL = {aten.mm.default, aten.bmm.default, aten.mm.out, aten.bmm.out}
_METADATA = {aten.detach.default, aten.alias.default, aten.lift_fresh.default, aten.empty.memory_format,
             aten.empty_strided.default, aten.empty_like.default, aten.new_empty.default,
             aten.new_empty_strided.default, aten._unsafe_view.default, aten.resize_.default,
             aten.set_.source_Storage_storage_offset, aten.set_.source_Storage, aten.set_.source_Tensor,
             aten.set_.default}  # allocations and metadata; copies (clone, contiguous) are CPU work
_COMPLEX_DTYPES = (torch.complex32, torch.complex64, torch.complex128)
_MAKES_COMPLEX = {}  # func -> whether it can make a complex tensor from real ones (hpu_complex's fallback then)


def _makes_complex(func):
    known = _MAKES_COMPLEX.get(func)
    if known is None:
        name = _op_name(func)
        known = "fft" in name or "stft" in name or any(
            name.startswith(p) for p in ("aten::complex", "aten::polar", "aten::view_as_complex"))
        _MAKES_COMPLEX[func] = known
    return known


def _tensors(args, kwargs):
    out = []
    for x in (*args, *kwargs.values()):
        if isinstance(x, torch.Tensor):
            out.append(x)
        elif isinstance(x, (list, tuple)):
            out.extend(t for t in x if isinstance(t, torch.Tensor))
    return out


def _is_hpu(t):
    return t.device.type == "hpu"


def _outputs(out):
    return [t for t in (out if isinstance(out, (list, tuple)) else (out,)) if isinstance(t, torch.Tensor)]


def _classify(func, args, tensors, outs):
    """The ``(source, detail)`` of a CPU execution this op is, or None."""
    if func is aten._local_scalar_dense.default:
        if _is_hpu(args[0]):
            return "host sync", "a device value read on the host (.item(), bool(), int(), float())"
        return None
    if func in _COPIES:
        if func is aten.copy_.default:
            dst, src = args[0], args[1]
            if dst.device.type == "cpu" and _is_hpu(src):
                return "host copy", "copy_ of a device tensor into a CPU tensor"
            if _is_hpu(dst):
                return None  # a host-to-device copy: the step's input, not CPU compute
        elif tensors and _is_hpu(tensors[0]) and outs and outs[0].device.type == "cpu":
            return "host copy", "a device tensor copied to the CPU (.cpu(), .to('cpu'), .tolist(), .numpy())"
        elif outs and _is_hpu(outs[0]):
            return None
    both = tensors + outs
    if any(_is_hpu(t) and (t.is_complex() or t.dtype == torch.float64) and _host_backed(t) for t in both):
        return "hpu_complex", "complex128 (or a float64 view of it) computed on the CPU in host memory"
    if func in _MATMUL and len(tensors) >= 2 and all(_is_hpu(t) and t.dtype == torch.complex64 for t in tensors[:2]):
        a, b = tensors[0], tensors[1]
        if a.numel() * b.shape[-1] < _STATE["matmul_cpu_below"]:
            return "hpu_complex", "complex64 mm/bmm below hpu_complex's CPU threshold (set_matmul_cpu_threshold(0))"
    if func.is_view or func in _METADATA or torch.Tag.inplace_view in func.tags:
        return None
    if both and all(t.device.type == "cpu" for t in both) and any(t.numel() > 1 for t in both):
        return "cpu op", "an op on CPU tensors inside the step"
    return None


class _Watch(TorchDispatchMode):
    """Reads the bridge's fallback total around each op, and hpu-complex's fallback count around ops that have or
    can make a complex tensor."""

    def __init__(self, config):
        super().__init__()
        self.config = config

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func is aten._local_scalar_dense.default and not isinstance(args[0], torch.Tensor):
            return args[0]  # a wrapped number (z * 2j inside hpu_complex) reaches the mode as its Python value
        if _worker_info() is not None:  # a DataLoader worker (spawned: its enable() ran the environment form)
            return func(*args, **kwargs)
        if len(torch._ops._higher_order_ops) != _HOPS["seen"]:
            _passthrough_hops()
        tensors = _tensors(args, kwargs)
        config = _config() or self.config
        if getattr(_TLS, "allowed", 0) or _tracing(tensors):
            return func(*args, **kwargs)
        complex_in = any(t.is_complex() for t in tensors) or _makes_complex(func) or any(
            v in _COMPLEX_DTYPES for v in (*args, *kwargs.values()) if isinstance(v, torch.dtype))
        base = _STATE["baseline"] or ((0, {}), 0)
        (total0, ops0), calls0 = _bridge_stats(), (_complex_calls() if complex_in else base[1])
        if base[0][0] != total0 or base[1] != calls0:
            check(config)  # counted since the last op, outside any op this watch saw (another thread, a graph)
        outer, _TLS.watch = getattr(_TLS, "watch", None), self  # hpu-ops' routes below run with the mode popped
        try:
            out = func(*args, **kwargs)
        finally:
            _TLS.watch = outer
        (total, ops), calls = _bridge_stats(), (_complex_calls() if complex_in else calls0)
        _STATE["baseline"] = ((total, ops), calls)  # before recording: a raise must not report these again
        name = _op_name(func)
        if total != total0:
            for op, value in ops.items():
                if value - ops0.get(op, 0):
                    record("bridge fallback", op, tensors, f"under {name}" if op != name else "",
                           n=value - ops0.get(op, 0), config=config)
        if calls != calls0:
            record("hpu_complex", name, tensors, "hpu_complex's CPU fallback", n=calls - calls0, config=config)
        if getattr(_TLS, "depth", 0) == 0:
            outs = _outputs(out)
            if not any(_fake(t) for t in outs):
                found = _classify(func, args, tensors, outs)
                if found is not None and (found[0] != "cpu op" or config.watch_cpu_ops):
                    record(found[0], name, tensors, found[1], config=config)
        return out


def _cond_eager(pred, true_fn, false_fn, operands):
    return true_fn(*operands) if pred else false_fn(*operands)  # cond_op_dense, which asserts that no mode is on


def _while_loop_eager(cond_fn, body_fn, carried_inputs, additional_inputs):
    carried = carried_inputs  # while_loop_dense, which runs below every mode
    while cond_fn(*carried, *additional_inputs):
        carried = body_fn(*carried, *additional_inputs)
    return carried


_HOP_EAGER = {"cond": _cond_eager, "while_loop": _while_loop_eager}


def _passthrough_hops():
    """Higher-order ops need a rule per mode. ``cond`` and ``while_loop`` run their eager semantics with the watch
    pushed back (HOP dispatch pops it), so the ops of their bodies and the predicate's host read are watched; other
    HOPs run as without the watch (their bodies unwatched). Looked up again when the registry grows (HOPs are
    created on import)."""
    registry = torch._ops._higher_order_ops
    for name, hop in list(registry.items()):
        if _Watch not in hop.python_key_table:
            def rule(mode, *args, _hop=hop, _eager=_HOP_EAGER.get(name), **kwargs):
                if _eager is None:
                    return _hop(*args, **kwargs)
                _push_watch(mode)
                try:
                    return _eager(*args, **kwargs)
                finally:
                    _remove_watch(mode)

            hop.py_impl(_Watch)(rule)
    _HOPS["seen"] = len(registry)


def _install_eager_hook():
    """Dynamo's ``force_eager`` stance for the threads in a per-op block only: Dynamo reads its stance through
    ``eval_frame._callback_from_stance`` at every call of compiled code; this wraps it with the thread's flag. False
    when this torch has no such function (the process-wide stance is used then)."""
    if "installed" not in _EAGER_HOOK:
        try:
            import torch._dynamo.eval_frame as eval_frame

            original = eval_frame._callback_from_stance
        except (ImportError, AttributeError):
            _EAGER_HOOK["installed"] = False
            return False

        def _callback_from_stance(callback):
            if getattr(_TLS, "eager", 0):
                return None  # Dynamo's own "force_eager", for this thread
            return original(callback)

        _callback_from_stance._hpu_ops_strict = True
        eval_frame._callback_from_stance = _callback_from_stance
        _EAGER_HOOK.update(installed=True, original=original)
    return _EAGER_HOOK["installed"]


def _eager_on():
    if _install_eager_hook():
        _TLS.eager = getattr(_TLS, "eager", 0) + 1
        return
    with _LOCK:  # no hook: the process-wide stance, set by the first open block of any thread, undone by the last
        if not _GLOBAL_EAGER["n"]:
            _GLOBAL_EAGER["cm"] = torch.compiler.set_stance("force_eager")
        _GLOBAL_EAGER["n"] += 1


def _eager_off():
    if _EAGER_HOOK.get("installed"):
        _TLS.eager = max(0, getattr(_TLS, "eager", 0) - 1)
        return
    with _LOCK:
        _GLOBAL_EAGER["n"] -= 1
        if not _GLOBAL_EAGER["n"]:
            _GLOBAL_EAGER.pop("cm").__exit__(None, None, None)


class _Block:
    """One strict block: its settings pushed on this thread, the watch (the outermost per-op block's), the counters'
    baseline, and under the watch ``set_stance("force_eager")`` (see the module doc)."""

    def __init__(self, config, per_op):
        self.config, self.per_op = config, per_op
        self.watch = self.outer_watch = self.outer_config = None
        self.generation = _STATE["generation"]

    def enter(self):
        _resolve_complex()
        check()  # an outer block's fallbacks so far are the outer block's
        with _LOCK:
            _ACTIVE["n"] += 1
        _TLS.blocks = getattr(_TLS, "blocks", []) + [self.config]
        _STATE["baseline"] = _snapshot()
        outer = _watch_on_stack()
        if outer is not None:  # an outer block's watch: autograd's threads read its settings, so it takes these
            self.outer_watch, self.outer_config = outer, outer.config
            outer.config = self.config
        elif self.per_op:
            _passthrough_hops()
            _eager_on()
            self.watch = _Watch(self.config)
            _push_watch(self.watch)
        return self

    def exit(self, failed):
        try:
            if not failed:
                check(self.config)
        finally:
            _TLS.blocks = getattr(_TLS, "blocks", [])[:-1]  # the baseline stays: an outer block does not count again
            if self.generation == _STATE["generation"]:  # else a fork reset strict mode in this (child) process
                if self.watch is not None:
                    _remove_watch(self.watch)
                    _eager_off()
                if self.outer_watch is not None:
                    self.outer_watch.config = self.outer_config
                with _LOCK:
                    _ACTIVE["n"] = max(0, _ACTIVE["n"] - 1)


@contextlib.contextmanager
def strict(level="raise", per_op=True, allow=(), watch_cpu_ops=True):
    """Raise (``level="raise"``) or count (``"report"``) every CPU execution inside the block (see the module doc).
    ``per_op=False``: no dispatch mode (compiled code compiles), the bridge's and hpu-complex's fallbacks are checked
    when the block ends. ``allow``: sources neither raised nor counted. ``watch_cpu_ops=False`` drops the ``cpu op``
    source. Nested blocks share the outermost per-op block's watch; the innermost block's settings apply."""
    if torch.compiler.is_compiling():  # Dynamo breaks the graph here and runs the block eagerly
        raise RuntimeError("hpu_ops.strict() inside a torch.compile'd function: open the block around the call")
    if level not in ("raise", "report"):
        raise ValueError(f"strict level {level!r}: 'raise' or 'report'")
    unknown = set(allow) - set(SOURCES)
    if unknown:
        raise ValueError(f"unknown strict sources {sorted(unknown)}; known: {SOURCES}")
    block = _without_dynamo(_Block(_Config(level, allow, watch_cpu_ops), per_op).enter)
    failed = True
    try:
        yield
        failed = False
    finally:
        _without_dynamo(block.exit, failed)


def _without_dynamo(fn, *args):
    """``fn(*args)`` with Dynamo off for its frames: a block opened in code that torch.compile runs (eagerly, after
    the graph break at ``strict()``) must not be traced. Without Dynamo loaded nothing traces them."""
    dynamo = sys.modules.get("torch._dynamo")
    return fn(*args) if dynamo is None else dynamo.disable(fn)(*args)


def report():
    """The counted rows, ``[(source, op, tensors, count, first caller)]``, most frequent first."""
    rows = [(*key, n, _STATE["where"].get(key, "?")) for key, n in _STATE["counts"].items()]
    return sorted(rows, key=lambda r: (-r[3], r[0], r[1]))


def reset():
    _STATE["counts"].clear()
    _STATE["where"].clear()


def format_report(rows=None):
    rows = report() if rows is None else rows
    if not rows:
        return "hpu_ops strict report: no CPU execution seen"
    lines = [f"hpu_ops strict report: {sum(r[3] for r in rows)} CPU executions, {len(rows)} kinds",
             f"{'count':>7}  {'source':<15}  {'op':<40}  tensors  |  first caller"]
    for source, op, sig, n, where in rows:
        lines.append(f"{n:>7}  {source:<15}  {op:<40}  {sig or '-'}  |  {where}")
    return "\n".join(lines)


def _at_exit():
    config = _STATE.get("env_config")
    if config is None:
        return
    if config.level == "raise":  # per_op=False: the fallbacks no op was there to raise on, all of them, as a table
        before = len(_STATE["counts"])
        check(_Config("report", config.allow, config.watch_cpu_ops))
        if len(_STATE["counts"]) != before:
            print("hpu_ops strict (HPU_OPS_STRICT=1): CPU executions seen after the last op\n" + format_report(),
                  file=sys.stderr, flush=True)
        return
    check(config)
    print(format_report(), file=sys.stderr, flush=True)


def from_env():
    """``HPU_OPS_STRICT`` read by ``enable()``: ``1``/``raise`` raises, ``report`` counts and prints at exit, for the
    main thread (autograd's threads follow its watch). Once per process; a DataLoader worker runs its loop outside
    it. Returns the level or None."""
    if "env_config" in _STATE:
        return _STATE["env_config"].level
    value = os.environ.get(ENV, "").strip().lower()
    if value in OFF:
        return None
    level = "report" if value == "report" else "raise"
    allow = tuple(s.strip() for s in os.environ.get(ENV_ALLOW, "").split(",") if s.strip())
    unknown = set(allow) - set(SOURCES)
    if unknown:
        raise ValueError(f"{ENV_ALLOW}: unknown sources {sorted(unknown)}; known: {SOURCES}")
    per_op = os.environ.get(ENV_PER_OP, "1").strip().lower() not in OFF
    block = _Block(_Config(level, allow), per_op).enter()  # for the process: never exited
    _STATE.update(env_config=block.config, env_block=block)
    if not _STATE["atexit"]:
        atexit.register(_at_exit)
        _STATE["atexit"] = True
    return level
