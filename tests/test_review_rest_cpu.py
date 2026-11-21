"""The non-linalg findings of the max-effort review of main (``gate/code-review-hpu-ops.txt``), each with the review's
exact input, on the CPU standing in for HPU. Each test fails on main (bda890b) and passes with the fix."""
import os
import pathlib
import subprocess
import sys

import pytest
from test_extrema_cpu import card  # noqa: F401  (the extrema fixture)
import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
aten = torch.ops.aten
NAN, INF = float("nan"), float("inf")


def _run(code, env=None):
    """``code`` in a fresh interpreter with this checkout first on the path; returns its stdout."""
    environ = dict(os.environ, PYTHONPATH=os.pathsep.join([str(ROOT), str(ROOT / "tests")]), **(env or {}))
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=environ, timeout=300)
    assert done.returncode == 0, done.stderr[-2000:]
    return done.stdout.strip()


# ---- (2) quantile's import-time probes read the default dtype and device


PROBES = "import hpu_ops.quantile as q; print(q._ranks_in_float64(), q._length_limited())"


@pytest.mark.parametrize("setup", ["torch.set_default_dtype(torch.bfloat16)", "torch.set_default_dtype(torch.float16)",
                                   "torch.set_default_dtype(torch.float64)", "torch.set_default_device('meta')"])
def test_import_under_another_default_dtype_or_device(setup):
    """``torch.set_default_dtype(torch.bfloat16); import hpu_ops`` raised (quantile rejects bf16), and the meta default
    device raised on ``item()``. The probes give the default process's answers."""
    want = _run(PROBES)
    assert _run(f"import torch; {setup}; import hpu_ops; {PROBES}") == want


# ---- (5) compiled inference_mode graphs left padding="same" and unbatched large-kernel convolutions whole


def _conv_widths(layer, x, context):
    import torch.nn as nn  # noqa: F401
    from test_cpu import compiled, cpu_as_hpu
    from hpu_ops import _dispatch, conv

    nodes = []
    with cpu_as_hpu(lambda d: _dispatch.chain([conv.install(d), conv.install_large_kernel(d)])), context():
        got = compiled(layer, nodes, dynamic=False)(x)
    widths = [tuple(n.args[1].meta["val"].shape[2:]) for n in nodes if n.target is aten.convolution.default]
    return got, widths


def _conv_cases():
    import torch.nn as nn

    torch.manual_seed(0)
    return {
        "conv1d k351 same": (nn.Conv1d(1, 2, 351, padding="same"), torch.randn(2, 1, 400)),
        "conv1d k350 same (asymmetric)": (nn.Conv1d(1, 2, 350, padding="same"), torch.randn(2, 1, 400)),
        "conv1d k300 same dilation 2": (nn.Conv1d(2, 2, 300, padding="same", dilation=2), torch.randn(1, 2, 700)),
        "conv2d (1,301) same": (nn.Conv2d(1, 4, (1, 301), padding="same"), torch.randn(2, 1, 3, 400)),
        "conv1d k351 unbatched": (nn.Conv1d(1, 2, 351), torch.randn(1, 400)),
        "conv2d (351,1) unbatched": (nn.Conv2d(1, 4, (351, 1)), torch.randn(1, 400, 3)),
        "conv1d k351 same unbatched": (nn.Conv1d(1, 2, 351, padding="same"), torch.randn(1, 400)),
    }


@pytest.mark.parametrize("name", list(_conv_cases()))
@pytest.mark.parametrize("context", [torch.inference_mode, torch.no_grad])
def test_compiled_inference_mode_splits_same_and_unbatched_large_kernels(monkeypatch, name, context):
    """The review's inputs: ``nn.Conv1d(1, 2, 351, padding='same')`` on (2, 1, 400), ``nn.Conv2d(1, 4, (1, 301),
    padding='same')``, ``nn.Conv1d(1, 2, 351)`` on an unbatched (1, 400): compiled under ``inference_mode()`` the graph
    kept one 351-wide convolution (Gaudi2's MME rejects it). Every graph convolution is now at most 256 wide and the
    values are the layer's."""
    layer, x = _conv_cases()[name]
    with torch.no_grad():
        want = layer(x)
    got, widths = _conv_widths(layer, x, context)
    assert widths and all(max(w) <= 256 for w in widths), widths
    torch.testing.assert_close(got, want, atol=1e-4, rtol=1e-4)


# ---- (6) compiled inference_mode graphs kept Gaudi's drifting linear upsampling kernel


@pytest.mark.parametrize("length, size", [(9920, 10000), (1000, 1500), (1000, 2000)])
@pytest.mark.parametrize("context", [torch.inference_mode, torch.no_grad])
def test_compiled_inference_mode_linear_upsampling_takes_the_index_path(length, size, context):
    """The review's input: ``F.interpolate(x, size=10000, mode='linear')`` of a 9920-sample signal (NeuroPose),
    compiled under ``inference_mode()``: the graph was exactly ``{aten.upsample_linear1d.default}``. Now the explicit
    index path, as eager and compiled ``no_grad()``; a power-of-two factor keeps the native op."""
    import torch.nn.functional as F
    from test_cpu import compiled, cpu_as_hpu, ops
    from hpu_ops import _dispatch, upsample

    x = torch.randn(2, 3, length)
    f = lambda t: F.interpolate(t, size=size, mode="linear")  # noqa: E731
    nodes = []
    both = lambda d: _dispatch.chain([upsample.install_linear(d), upsample.install_linear_forward(d)])  # noqa: E731
    with cpu_as_hpu(both) as calls, context():
        got = compiled(f, nodes, dynamic=False)(x)
        counts = dict(calls)
    want, cpu = f(x.double()), f(x)  # the index ops' error against float64 is the CPU float32 kernel's
    assert (got.double() - want).abs().max() <= 2 * (cpu.double() - want).abs().max() + 1e-6
    exact = size % length == 0 and (size // length) & (size // length - 1) == 0
    assert ("aten.upsample_linear1d.default" in ops(nodes)) == exact, ops(nodes)
    assert bool(counts.get("linear_index")) == (not exact), counts


def test_linear_upsampling_without_the_forward_fix_keeps_the_native_op_traced():
    """``install_linear`` alone (the forward fix not installed, or undone) keeps the native op in the graph."""
    import torch.nn.functional as F
    from test_cpu import compiled, cpu_as_hpu, ops
    from hpu_ops import upsample

    undo = upsample.install_linear_forward("cpu")
    undo()
    nodes = []
    with cpu_as_hpu(upsample.install_linear), torch.inference_mode():
        compiled(lambda t: F.interpolate(t, size=1500, mode="linear"), nodes, dynamic=False)(torch.randn(1, 1, 1000))
    assert "aten.upsample_linear1d.default" in ops(nodes)


# ---- (9) compiled inference_mode graphs of the max/min family held the bare ops


class _GaudiDropsNaN(torch.utils._python_dispatch.TorchDispatchMode):
    """A graph's family ops as Gaudi computes them (measured, bridge 1.21.4): NaN read as -inf by the max ops, +inf by
    the min ops, 0 by the inf-norms; maximum / minimum / clamp return the first operand where one is NaN. The graph's
    other ops are torch's."""

    LARGEST = {aten.amax.default, aten.max.default, aten.max.dim, aten.argmax.default, aten.median.default,
               aten.median.dim, aten.aminmax.default}
    SMALLEST = {aten.amin.default, aten.min.default, aten.min.dim, aten.argmin.default}

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        x, rest = args[0], args[1:]
        if not (isinstance(x, torch.Tensor) and x.is_floating_point()):
            return func(*args, **kwargs)
        if func in self.LARGEST or func in self.SMALLEST:
            return func(torch.where(torch.isnan(x), -INF if func in self.LARGEST else INF, x), *rest, **kwargs)
        if func in (aten.maximum.default, aten.minimum.default, aten.fmax.default, aten.fmin.default):
            a, b = args
            return torch.where(torch.isnan(a) | torch.isnan(b), a, func(a, b))
        if func is aten.linalg_vector_norm.default:
            return func(torch.where(torch.isnan(x), 0.0, x), *rest, **kwargs)
        if func in (aten.clamp.Tensor, aten.clamp_min.Tensor, aten.clamp_max.Tensor):
            return func(torch.where(torch.isnan(x), 0.0, x),
                        *[torch.where(torch.isnan(b), 0.0, b) if isinstance(b, torch.Tensor) else b for b in rest])
        return func(*args, **kwargs)


INFERENCE_CASES = {
    "amax": lambda t: t.amax(-1), "amin": lambda t: t.amin(0), "max()": lambda t: t.max(), "min()": lambda t: t.min(),
    "median()": lambda t: t.median(), "max(dim)": lambda t: t.max(-1).values, "min(dim) indices": lambda t: t.min(-1)[1],
    "argmax": lambda t: t.argmax(-1), "argmin": lambda t: t.argmin(-1),
    "aminmax": lambda t: torch.aminmax(t, dim=-1).max,
    "maximum": lambda t: torch.maximum(t, t.flip(0)), "minimum": lambda t: torch.minimum(t, t.flip(0)),
    "fmax": lambda t: torch.fmax(t, t.flip(0)), "clamp(t, t)": lambda t: t.clamp(t.flip(0) - 1, t.flip(0) + 1),
    "clamp_(t)": lambda t: t.clone().clamp_(min=t.flip(0)),
    "norm(inf, -1)": lambda t: t.norm(INF, -1), "vector_norm(inf)": lambda t: torch.linalg.vector_norm(t, INF, -1),
}


@pytest.mark.parametrize("name", list(INFERENCE_CASES))
@pytest.mark.parametrize("context", [torch.inference_mode, torch.no_grad])
def test_compiled_inference_mode_graphs_carry_the_nan_fix(monkeypatch, name, context):
    """The review's case: ``torch.compile(lambda t: t.amax(-1))`` under ``inference_mode()`` on rows with NaN traced
    ``{aten.amax.default}`` alone. The graph, run with Gaudi's NaN-dropping kernels and no hpu-ops in the dispatcher
    (as the card's graph compiler runs it), now gives torch's values."""
    from functorch.compile import make_boxed_func
    from torch._dynamo.backends.common import aot_autograd

    import test_extrema_cpu as T
    from hpu_ops import extrema

    f, x = INFERENCE_CASES[name], T.rows(n=8, dtype=torch.float32)
    want = f(x)
    graphs = []

    def record(gm, example_inputs):
        graphs.append(gm)
        return make_boxed_func(gm.forward)

    monkeypatch.setattr(extrema, "native", T.gaudi)
    monkeypatch.setattr(extrema, "native_out", T.gaudi_out)
    undo = extrema.install("cpu")
    try:
        torch._dynamo.reset()
        with context():
            torch.compile(f, backend=aot_autograd(fw_compiler=record), dynamic=False, fullgraph=True)(x)
    finally:
        undo()
    assert len(graphs) == 1
    with context(), _GaudiDropsNaN():
        got = graphs[0](x)
    got = got[0] if isinstance(got, (tuple, list)) else got
    T.same(got, want)


# ---- (7) _Whole / _InfNorm were old-style Functions: torch.func transforms and forward-mode AD raised


def _fwd(f, t):
    import torch.autograd.forward_ad as fwAD

    with fwAD.dual_level():
        return fwAD.unpack_dual(f(fwAD.make_dual(t, torch.arange(1.0, t.numel() + 1).reshape(t.shape)))).tangent


TRANSFORMS = {
    "func.grad(t.max())": lambda t: torch.func.grad(lambda r: r.max())(t),
    "func.grad(t.min())": lambda t: torch.func.grad(lambda r: r.min())(t),
    "func.grad(t.median())": lambda t: torch.func.grad(lambda r: r.median())(t),
    "func.grad(vector_norm(t, inf))": lambda t: torch.func.grad(lambda r: torch.linalg.vector_norm(r, INF))(t),
    "func.grad(t.norm(inf))": lambda t: torch.func.grad(lambda r: r.norm(INF))(t),
    "func.vjp(t.max())": lambda t: torch.func.vjp(lambda r: r.max(), t)[1](torch.tensor(1.0, dtype=t.dtype))[0],
    "func.jacrev(t.max())": lambda t: torch.func.jacrev(lambda r: r.max())(t),
    "vmap(grad(r.max()))": lambda t: torch.func.vmap(torch.func.grad(lambda r: r.max()))(t),
    "vmap(grad(r.norm(-inf)))": lambda t: torch.func.vmap(torch.func.grad(lambda r: r.norm(-INF)))(t),
    "forward AD max()": lambda t: _fwd(lambda d: d.max(), t.clone().requires_grad_()),
    "forward AD median()": lambda t: _fwd(lambda d: d.median(), t.clone().requires_grad_()),
    "forward AD vector_norm(inf, -1)": lambda t: _fwd(lambda d: torch.linalg.vector_norm(d, INF, -1),
                                                      t.clone().requires_grad_()),
    "forward AD norm(-inf, 0, keepdim)": lambda t: _fwd(lambda d: d.norm(-INF, 0, True), t.clone().requires_grad_()),
    "forward AD vector_norm(inf) of all": lambda t: _fwd(lambda d: torch.linalg.vector_norm(d, INF),
                                                         t.clone().requires_grad_()),
}


@pytest.mark.parametrize("name", list(TRANSFORMS))
@pytest.mark.parametrize("nan", [True, False])
def test_functorch_and_forward_ad_through_whole_reductions_and_inf_norms(card, name, nan):
    """The review's input ``x = [[1., 3., 2.], [.5, 4., -1.]]``: ``torch.func.grad(lambda t: t.max())(x)`` raised
    "must override the setup_context staticmethod", forward-mode AD "You must implement the jvp function". Now
    torch's values, with and without a NaN."""
    from test_extrema_cpu import _reference, same

    x = torch.tensor([[1.0, 3.0, 2.0], [0.5, NAN if nan else 4.0, -1.0]], dtype=torch.float64)
    f = TRANSFORMS[name]
    same(f(x), _reference(f, x))


# ---- (12) a CPU 0-d first operand placed the backend kernel's out on the CPU


OUT_AS_TORCH = {aten.maximum.out: torch.maximum, aten.minimum.out: torch.minimum, aten.fmax.out: torch.fmax,
                aten.fmin.out: torch.fmin, aten.clamp.Tensor_out: torch.clamp, aten.clamp_min.Tensor_out: torch.clamp_min,
                aten.clamp_max.Tensor_out: torch.clamp_max}


def _device_out(op, dev):
    """A device's ``.out`` kernel as TensorIterator runs it: a CPU 0-d operand is taken as a scalar, ``out`` must be on
    the device (Meta's own ``.out`` kernels are Python references that reject the CPU operand, unlike Gaudi's)."""

    def kernel(*args, out, **kwargs):
        r = OUT_AS_TORCH[op](*args, **kwargs)
        if out.device != r.device:
            raise RuntimeError(f"Expected out tensor to have device {r.device}, but got {out.device} instead")
        return out.copy_(r)

    return kernel


@pytest.fixture
def meta_kernels(monkeypatch):
    """The backend layer's kernels for "meta" standing in for HPU, as the dispatcher calls them for a CPU 0-d operand
    with a device tensor."""
    from hpu_ops import extrema

    monkeypatch.setattr(extrema, "native_out", _device_out)
    return dict(extrema._backend_kernels("meta"))


ZERO_D_FIRST = {  # the review's calls, as the dispatcher hands them to the backend kernels
    "maximum(cpu0d, d)": (aten.maximum.default, lambda c, d: (c, d)),
    "minimum(cpu0d, d)": (aten.minimum.default, lambda c, d: (c, d)),
    "fmax(cpu0d, d)": (aten.fmax.default, lambda c, d: (c, d)), "fmin(cpu0d, d)": (aten.fmin.default, lambda c, d: (c, d)),
    "clamp(cpu0d, min=d)": (aten.clamp.Tensor, lambda c, d: (c, d, None)),
    "clamp(cpu0d, d, d)": (aten.clamp.Tensor, lambda c, d: (c, d, d)),
    "clamp_min(cpu0d, d)": (aten.clamp_min.Tensor, lambda c, d: (c, d)),
    "clamp_max(cpu0d, d)": (aten.clamp_max.Tensor, lambda c, d: (c, d)),
    "clamp(d, min=cpu0d)": (aten.clamp.Tensor, lambda c, d: (d, c, None)),
    "maximum(d, cpu0d)": (aten.maximum.default, lambda c, d: (d, c)),
}


@pytest.mark.parametrize("name", list(ZERO_D_FIRST))
def test_a_cpu_0d_first_operand_gives_a_device_result(meta_kernels, name):
    """``torch.maximum(torch.tensor(0.), x_hpu)``: torch dispatches it to the device, where the backend kernel
    allocated ``out`` like the first operand, on the CPU ("Attempting to copy from device meta to device cpu")."""
    op, args = ZERO_D_FIRST[name]
    got = meta_kernels[op](*args(torch.tensor(0.5), torch.ones(3, device="meta")))
    assert got.device.type == "meta" and got.shape == (3,) and got.dtype == torch.float32


# ---- cut items: clamp_ of an integer tensor with float bounds, the opt-out cancelling the exact sort


INPLACE = {
    "arange(5).clamp_(min=full(1.5))": lambda: torch.arange(5).clamp_(min=torch.full((5,), 1.5)),
    "arange(5, int32).clamp_(min=full(1.5))": lambda: torch.arange(5, dtype=torch.int32).clamp_(
        min=torch.full((5,), 1.5)),
    "arange(5).clamp_min_(full(1.5))": lambda: torch.arange(5).clamp_min_(torch.full((5,), 1.5)),
    "arange(5).clamp_max_(full(2.5))": lambda: torch.arange(5).clamp_max_(torch.full((5,), 2.5)),
    "arange(5).clamp_(min=tensor(1.5))": lambda: torch.arange(5).clamp_(min=torch.tensor(1.5)),
}


@pytest.mark.parametrize("name", list(INPLACE))
@pytest.mark.parametrize("context", [torch.enable_grad, torch.no_grad, torch.inference_mode])
def test_inplace_clamp_of_an_integer_tensor_with_float_bounds_raises_as_torch(card, name, context):
    """torch raises "result type Float can't be cast to the desired output type Long"; the rewrite truncated."""
    from test_extrema_cpu import _reference

    with context():
        with pytest.raises(RuntimeError, match="can't be cast to the desired output type") as want:
            _reference(INPLACE[name])
        with pytest.raises(RuntimeError, match="can't be cast to the desired output type") as got:
            INPLACE[name]()
    assert str(got.value) == str(want.value)


def test_inplace_clamp_that_torch_allows_still_runs(card):
    from test_extrema_cpu import _reference, same

    for f in (lambda: torch.arange(5.0).clamp_(min=torch.full((5,), 1.5)),
              lambda: torch.arange(5.0).clamp_(min=torch.full((5,), 1.5, dtype=torch.float64)),
              lambda: torch.arange(5).clamp_(min=torch.full((5,), 2))):
        same(f(), _reference(f))


def test_nan_propagation_opt_out_keeps_the_exact_sort(monkeypatch):
    """``HPU_OPS_NAN_PROPAGATION=0`` with ``HPU_OPS_EXACT_SORT=1``: the sort rewrite is registered (it was not) and
    the max/min family stays native."""
    from test_cpu import cpu_as_hpu
    from hpu_ops import extrema

    monkeypatch.setenv("HPU_OPS_NAN_PROPAGATION", "0")
    monkeypatch.setenv("HPU_OPS_EXACT_SORT", "1")
    x = torch.tensor([3.0, INF, 1.0, NAN, 2.0])
    with cpu_as_hpu(extrema.install) as calls:
        assert torch._C._dispatch_has_kernel_for_dispatch_key("aten::sort", "AutogradCPU")
        torch.sort(x), x.amax()
        assert not extrema._BACKEND
    assert calls["sort"] and not calls["amax"]
    assert not torch._C._dispatch_has_kernel_for_dispatch_key("aten::sort", "AutogradCPU")
    monkeypatch.setenv("HPU_OPS_EXACT_SORT", "0")
    with cpu_as_hpu(extrema.install) as calls:
        torch.sort(x)
        assert not torch._C._dispatch_has_kernel_for_dispatch_key("aten::sort", "AutogradCPU")
    assert not calls


# ---- (15) the gradient with respect to an expanded (stride-0) tensor itself landed at index 0


def _index_rewrite(monkeypatch, off=False):
    from test_cpu import cpu_as_hpu
    from hpu_ops import indexing

    if off:
        monkeypatch.setattr(indexing, "_off_device", lambda self, t: True)
    return cpu_as_hpu(indexing.install_index)


def _expanded_cases():
    from torch.func import grad, jvp

    idx = torch.tensor([1, 3, 3])
    w = torch.tensor([1.0, 10.0, 100.0])

    def leaf_grad():  # the review's input
        leaf = torch.zeros(3, 1).expand(3, 4).requires_grad_()
        leaf[:, idx].sum().backward()
        return leaf.grad

    def leaf_weighted():
        leaf = torch.zeros(3, 1).expand(3, 4).requires_grad_()
        (leaf[:, idx] * w).sum().backward()
        return leaf.grad

    def retained():
        base = torch.randn(3, 1, requires_grad=True)
        e = base.expand(3, 4)
        e.retain_grad()
        (e[:, idx] * w).sum().backward()
        return torch.cat([e.grad.flatten(), base.grad.flatten()])

    def autograd_grad():
        e = torch.randn(1, 4).expand(5, 4).requires_grad_()
        return torch.autograd.grad((e[torch.tensor([0, 4, 4])][:, idx] * w).sum(), e)[0]

    def parameter():
        p = torch.nn.Parameter(torch.randn(3, 1).expand(3, 4))
        (p[:, idx] * w).sum().backward()
        return p.grad

    def mask():
        e = torch.zeros(3, 1).expand(3, 4).requires_grad_()
        (e[:, torch.tensor([False, True, False, True])] * torch.tensor([1.0, 10.0])).sum().backward()
        return e.grad

    def func_grad():
        return grad(lambda t: (t[:, idx] * w).sum())(torch.randn(3, 1).expand(3, 4))

    def func_jvp():  # an expanded primal cannot be made dual (torch raises); expanded inside the function
        return jvp(lambda b: b.expand(5, 4)[torch.tensor([0, 4])][:, idx], (torch.randn(1, 4),),
                   (torch.arange(4.0).reshape(1, 4),))[1]

    def forward_values():  # no gradient: the shortcut stays, the values are the same
        return torch.randn(3, 1).expand(3, 4)[:, idx]

    return {f.__name__: f for f in (leaf_grad, leaf_weighted, retained, autograd_grad, parameter, mask, func_grad,
                                     func_jvp, forward_values)}


@pytest.mark.parametrize("name", list(_expanded_cases()))
@pytest.mark.parametrize("off", [False, True])
def test_gradient_of_an_expanded_tensor_lands_at_its_indices(monkeypatch, name, off):
    """``leaf = torch.zeros(3, 1).expand(3, 4).requires_grad_(); leaf[:, torch.tensor([1, 3, 3])].sum().backward()``
    gave ``leaf.grad`` rows [3, 0, 0, 0] instead of [0, 1, 0, 2]."""
    f = _expanded_cases()[name]
    torch.manual_seed(0)
    want = f()
    torch.manual_seed(0)
    with _index_rewrite(monkeypatch, off) as calls:
        got = f()
    assert calls["index"], calls
    torch.testing.assert_close(got, want, atol=0, rtol=0)


def test_expanded_index_without_a_gradient_keeps_the_one_element_shortcut(monkeypatch):
    """Without a gradient the stride-0 dim is still read from its one element (no copy of the expanded dims)."""
    from hpu_ops import indexing

    seen = []
    real = torch.Tensor.index_select
    monkeypatch.setattr(torch.Tensor, "index_select", lambda self, *a: seen.append(self.shape) or real(self, *a))
    x = torch.randn(3, 1).expand(3, 4)
    with _index_rewrite(monkeypatch):
        torch.testing.assert_close(indexing.index_select_flat(x, [None, torch.tensor([1, 3, 3])]),
                                   x[:, torch.tensor([1, 3, 3])])
        indexing.index_select_flat(x.clone().requires_grad_(), [None, torch.tensor([1, 3])])
    assert seen[0] == (3, 1), seen


# ---- efficiency: the tracing checks missed AOT's FunctionalTensor (host index build + copy every compiled step)


def test_compiled_index_and_unfold_build_their_indices_in_the_graph(monkeypatch):
    """The review's probe: ``x[:, idx]`` with a CPU index tensor (spd_learn's triu_indices) and ``F.unfold`` under
    ``torch.compile``: in AOT's functional pass ``indexing._tracing`` / ``im2col._tracing`` answered False, so the
    linear index was built on the host."""
    import torch.nn.functional as F
    from functorch.compile import make_boxed_func
    from torch._dynamo.backends.common import aot_autograd
    from torch._subclasses.functional_tensor import FunctionalTensor

    from hpu_ops import im2col, indexing

    answers = []
    for module in (indexing, im2col):
        real = module._tracing
        monkeypatch.setattr(module, "_tracing", lambda t, real=real: answers.append(
            (type(t) is FunctionalTensor, real(t))) or real(t))
    idx = torch.tensor([0, 2])

    def f(x):
        return x[:, idx].sum() + F.unfold(x.unsqueeze(1), (2, 2), padding=1).sum()

    backend = aot_autograd(fw_compiler=lambda gm, _: make_boxed_func(gm.forward),
                           bw_compiler=lambda gm, _: make_boxed_func(gm.forward))
    monkeypatch.setattr(indexing, "_off_device", lambda self, t: True)
    from test_cpu import cpu_as_hpu

    with cpu_as_hpu(lambda d: __import__("hpu_ops")._dispatch.chain([indexing.install_index(d), im2col.install(d)])):
        torch._dynamo.reset()
        torch.compile(f, backend=backend)(torch.randn(3, 4, requires_grad=True)).backward()
    assert any(functional for functional, _ in answers), answers
    assert all(traced for _, traced in answers), answers


def test_expanded_index_double_backward_and_compiled_gradients(monkeypatch):
    from test_cpu import compiled

    idx = torch.tensor([1, 3, 3])
    w = torch.tensor([1.0, 10.0, 100.0], dtype=torch.float64)

    def second(e):
        (g,) = torch.autograd.grad((e[:, idx] ** 3 * w).sum(), e, create_graph=True)
        (gg,) = torch.autograd.grad((g * torch.arange(12.0, dtype=e.dtype).reshape(3, 4)).sum(), e)
        return g.detach(), gg

    def leaf():
        return torch.linspace(-1, 1, 3, dtype=torch.float64).reshape(3, 1).expand(3, 4).requires_grad_()

    want = second(leaf())
    with _index_rewrite(monkeypatch, off=True) as calls:
        got = second(leaf())
    assert calls["index"]
    for g, x in zip(got, want):
        torch.testing.assert_close(g, x, atol=1e-12, rtol=0)

    def step(e):
        return (e[:, idx] * w).sum()

    e = leaf()
    step(e).backward()
    want = e.grad
    with _index_rewrite(monkeypatch, off=True) as calls:
        e = leaf()
        compiled(step, [], dynamic=False)(e).backward()
    assert calls["index"]
    torch.testing.assert_close(e.grad, want, atol=0, rtol=0)


# ---- cut item: enable() under -W error installed nothing when the kernel-path step warned


ENABLE_WERROR = """
import os, pathlib, sys, types, warnings
import torch
for name in ("habana_frameworks", "habana_frameworks.torch", "habana_frameworks.torch.core"):
    sys.modules[name] = types.ModuleType(name)
import hpu_ops
from hpu_ops import tpc
if sys.argv[1] == "late":
    tpc._initialized = lambda: True  # the HPU was used before enable()
else:
    tpc.LIB = pathlib.Path("/nonexistent/liblinalg_tpc.so")
warnings.simplefilter("error")
try:
    hpu_ops.enable()
    raised = "none"
except UserWarning as error:
    raised = "kernels" if "TPC linalg kernels" in str(error) else "other"
print(raised, "undo" in hpu_ops._STATE, os.environ.get("PT_HPU_DISABLE_pass_weight_permutation"),
      torch._C._dispatch_has_kernel_for_dispatch_key("aten::glu", "AutogradHPU"))
"""


@pytest.mark.parametrize("case", ["late", "missing"])
def test_enable_under_warnings_as_errors_installs_the_fixes_first(case):
    """``python -W error`` with ``tpc.install_kernel_path()`` warning (the HPU used before ``enable()``, or the
    kernel library missing): ``enable()`` raised before any fix was installed. Now the warning is raised last."""
    code = ENABLE_WERROR.replace('sys.argv[1]', repr(case))
    assert _run(code) == "kernels True True True"
