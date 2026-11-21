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
  copies of CPU tensors to the card are not.

Per-op attribution is a ``TorchDispatchMode`` (``per_op=True``, the default), declared an infra mode so that
torch.compile still compiles under it (Dynamo skips frames, for good, under other modes); the ops a compiled graph
runs inside Habana's recipes are seen through the bridge's counter only, at the next op or when the block ends.
``per_op=False`` (``HPU_OPS_STRICT_PER_OP=0``) reads the counters only when the block ends (or at ``check()``), named
by op, without a stack frame. ``HPU_OPS_STRICT_ALLOW`` (or ``allow=``) lists sources, comma-separated, that are not
raised nor counted, e.g. ``host sync,cpu op``. Strict mode is off in forked children (DataLoader workers) and while
``torch.save`` runs.
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
aten = torch.ops.aten


class CPUExecutionError(RuntimeError):
    """An HPU step ran something on the CPU (strict mode)."""


_STATE = {"level": None, "allow": frozenset(), "counts": collections.Counter(), "where": {}, "baseline": None,
          "atexit": False, "watches": 0, "matmul_cpu_below": COMPLEX_MATMUL_CPU_BELOW}
_TLS = threading.local()


def active():
    """Whether strict mode is on (raise or report)."""
    return _STATE["level"] is not None


def _reset_in_child():  # a forked DataLoader worker computes on the CPU by design
    _STATE.update(level=None, baseline=None, watches=0)


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


_STDLIB = os.path.dirname(contextlib.__file__)  # its own files only: site-packages may sit below it (conda)


def _frames():
    frame = sys._getframe(2)
    while frame is not None:
        yield frame
        frame = frame.f_back


def _caller():
    """``file:line in function`` of the first frame outside torch, Habana, hpu-ops, hpu-complex, the stdlib and FX's
    generated code; ``None`` inside ``torch.save``."""
    skip, found = _skip_dirs(), None
    for frame in _frames():
        name = frame.f_code.co_filename
        if name.endswith(os.path.join("torch", "serialization.py")):
            return None  # saving is I/O (hpu-complex's complex128 save reduces through .cpu())
        if found is None and not name.startswith(skip) and os.path.dirname(name) != _STDLIB \
                and not name.startswith(("<frozen", "<eval_with_key")):
            found = f"{name}:{frame.f_lineno} in {frame.f_code.co_name}"
    return found or "(no Python caller: autograd's backward)"


def _signature(tensors):
    parts = []
    for t in tensors:
        if isinstance(t, torch.Tensor):
            parts.append(f"{tuple(t.shape)} {str(t.dtype).replace('torch.', '')} {t.device.type}")
    return ", ".join(parts)


def _tracing(tensors):
    if torch.compiler.is_compiling():
        return True
    return any(isinstance(t, torch.Tensor) and (is_fake(t) or torch._is_functional_tensor(t)) for t in tensors)


def record(source, op, tensors=(), detail="", n=1, where=None):
    """One CPU execution of ``op`` from ``source``: raised in strict mode, counted in report mode, else nothing."""
    if _STATE["level"] is None or source in _STATE["allow"] or getattr(_TLS, "allowed", 0):
        return
    if where is None:
        where = _caller()
        if where is None:
            return
    if _tracing(tensors):
        detail = (detail + "; " if detail else "") + "while a graph is traced: once per compile, not per step"
    sig = _signature(tensors)
    if _STATE["level"] == "raise":
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
    if _STATE["level"] is None:
        return _NULL
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
    return _depth() if _STATE["level"] is not None else _NULL


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
    its matmul threshold, followed through its Python setter."""
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


def check():
    """Record (or raise on) the bridge fallbacks and hpu-complex fallbacks since the last ``check()``, the last op the
    watch saw, or the start of the strict block, named by op, without a frame: what ran outside any op the watch saw
    (a compiled graph's recipes, another thread) or every fallback with ``per_op=False``."""
    if _STATE["level"] is None:
        return
    (total0, ops0), calls0 = _STATE["baseline"] or ((0, {}), 0)
    (total, ops), calls = _snapshot()
    _STATE["baseline"] = ((total, ops), calls)  # before recording: a raise must not report these again
    if total != total0:
        for op, value in ops.items():
            if value - ops0.get(op, 0):
                record("bridge fallback", op, (), "counted per step (not attributed to a call)",
                       n=value - ops0.get(op, 0), where="(per step)")
    if calls != calls0:
        record("hpu_complex", "hpu_complex cpu_fallback", (), "counted per step (not attributed to a call)",
               n=calls - calls0, where="(per step)")


# ---- the per-op watch --------------------------------------------------------------------------------------------

_COPIES = {aten._to_copy.default, aten.copy_.default, aten.to.device, aten.to.dtype_layout, aten.to.dtype,
           aten.to.other}
_MATMUL = {aten.mm.default, aten.bmm.default, aten.mm.out, aten.bmm.out}


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


def _classify(func, args, tensors, out):
    """The ``(source, detail)`` of a CPU execution this op is, or None."""
    if func is aten._local_scalar_dense.default:
        if _is_hpu(args[0]):
            return "host sync", "a device value read on the host (.item(), bool(), int(), float())"
        return None
    outs = _outputs(out)
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
    if both and all(t.device.type == "cpu" for t in both) and any(t.numel() > 1 for t in both):
        return "cpu op", "an op on CPU tensors inside the step"
    return None


class _Watch(TorchDispatchMode):
    """Reads the bridge's fallback total around each op, and hpu-complex's fallback count around ops with a complex
    tensor."""

    def __init__(self, watch_cpu_ops=True):
        super().__init__()
        self.watch_cpu_ops = watch_cpu_ops

    @classmethod
    def is_infra_mode(cls):
        return True  # else Dynamo skips (for good) every frame it would compile under the watch

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func is aten._local_scalar_dense.default and not isinstance(args[0], torch.Tensor):
            return args[0]  # a wrapped number (z * 2j inside hpu_complex) reaches the mode as its Python value
        tensors = _tensors(args, kwargs)
        if _STATE["level"] is None or getattr(_TLS, "allowed", 0) or _tracing(tensors):
            return func(*args, **kwargs)
        complex_in = any(t.is_complex() for t in tensors)  # hpu_complex's fallback takes complex tensors only
        base = _STATE["baseline"] or ((0, {}), 0)
        (total0, ops0), calls0 = _bridge_stats(), (_complex_calls() if complex_in else base[1])
        if base[0][0] != total0 or base[1] != calls0:
            check()  # counted since the last op, outside any op this watch saw (another thread, a compiled graph)
        out = func(*args, **kwargs)
        (total, ops), calls = _bridge_stats(), (_complex_calls() if complex_in else calls0)
        _STATE["baseline"] = ((total, ops), calls)  # before recording: a raise must not report these again
        name = _op_name(func)
        if total != total0:
            for op, value in ops.items():
                if value - ops0.get(op, 0):
                    record("bridge fallback", op, tensors, f"under {name}" if op != name else "",
                           n=value - ops0.get(op, 0))
        if calls != calls0:
            record("hpu_complex", name, tensors, "hpu_complex's CPU fallback", n=calls - calls0)
        if getattr(_TLS, "depth", 0) == 0:
            found = _classify(func, args, tensors, out)
            if found is not None and (found[0] != "cpu op" or self.watch_cpu_ops):
                record(found[0], name, tensors, found[1])
        return out


@contextlib.contextmanager
def strict(level="raise", per_op=True, allow=(), watch_cpu_ops=True):
    """Raise (``level="raise"``) or count (``"report"``) every CPU execution inside the block (see the module doc).
    ``per_op=False``: the bridge's and hpu-complex's fallbacks are checked when the block ends only. ``allow``:
    sources neither raised nor counted. ``watch_cpu_ops=False`` drops the ``cpu op`` source. Nested blocks share one
    watch; the inner block's level and ``allow`` apply inside it."""
    if level not in ("raise", "report"):
        raise ValueError(f"strict level {level!r}: 'raise' or 'report'")
    unknown = set(allow) - set(SOURCES)
    if unknown:
        raise ValueError(f"unknown strict sources {sorted(unknown)}; known: {SOURCES}")
    _resolve_complex()
    check()  # an outer block's fallbacks so far are the outer block's
    saved = dict(level=_STATE["level"], allow=_STATE["allow"])
    _STATE.update(level=level, allow=frozenset(allow), baseline=_snapshot())
    watch = _Watch(watch_cpu_ops) if per_op and not _STATE["watches"] else _NULL
    _STATE["watches"] += watch is not _NULL
    try:
        with watch:
            yield
        check()
    finally:
        _STATE["watches"] -= watch is not _NULL
        _STATE.update(saved)  # the baseline stays: an outer block does not count the inner one's fallbacks again


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
    try:
        check()
    except CPUExecutionError as e:  # raise mode, per_op=False: the fallbacks no op was there to raise on
        print(str(e), file=sys.stderr, flush=True)
        return
    if _STATE["level"] == "report":
        print(format_report(), file=sys.stderr, flush=True)


def from_env():
    """``HPU_OPS_STRICT`` read by ``enable()``: ``1``/``raise`` raises, ``report`` counts and prints at exit, for the
    whole process (the main thread's watch; autograd's threads inherit it). Once per process. Returns the level or
    None."""
    if "env_context" in _STATE:
        return _STATE["env_level"]
    value = os.environ.get(ENV, "").strip().lower()
    if value in ("", "0", "false", "off"):
        return None
    level = "report" if value == "report" else "raise"
    allow = tuple(s.strip() for s in os.environ.get(ENV_ALLOW, "").split(",") if s.strip())
    per_op = os.environ.get(ENV_PER_OP, "1").strip() not in ("0", "false", "off")
    context = strict(level, per_op=per_op, allow=allow)
    context.__enter__()  # for the process: never exited
    _STATE.update(env_context=context, env_level=level)
    if not _STATE["atexit"]:
        atexit.register(_at_exit)
        _STATE["atexit"] = True
    return level
