"""The NaN rewrites of the max/min family (``hpu_ops.extrema``) on the CPU standing in for HPU, with a stand-in for
Gaudi's kernels that drops NaN as the card does (measured, bridge 1.21.4): results, indices and gradients must be
the CPU's own, which propagates NaN."""
import pytest
import torch
import torch.nn.functional as F

import hpu_ops
from hpu_ops import extrema
from functorch.compile import make_boxed_func
from torch._dynamo.backends.common import aot_autograd

from test_cpu import cpu_as_hpu, ops

aten = torch.ops.aten
NAN, INF = float("nan"), float("inf")
LARGEST = {aten.amax.default, aten.max.default, aten.max.dim, aten.argmax.default, aten.median.default,
           aten.median.dim, aten.max_pool2d_with_indices.default, aten.max_pool3d_with_indices.default}
SMALLEST = {aten.amin.default, aten.min.default, aten.min.dim, aten.argmin.default}
FLOATS = (torch.float64, torch.float32, torch.bfloat16, torch.float16)


REAL = extrema.native  # the rewrite's view of the native kernels; the stand-in wraps it
_CURRENT = {}
POOLS_ND = {aten.max_pool2d_with_indices.default: 2, aten.max_pool3d_with_indices.default: 3}
POOL_BACKWARDS = {aten.max_pool2d_with_indices_backward.default: 2, aten.max_pool3d_with_indices_backward.default: 3}


def _pool_geometry(x, args, nd, out_shape):
    kernel, stride, padding, dilation = args[:4]
    kernel = extrema._per_dim(kernel, nd)
    stride = extrema._per_dim(stride, nd, kernel)
    padding, dilation = extrema._per_dim(padding, nd), extrema._per_dim(dilation, nd)
    starts = []
    for j in range(nd):
        o = torch.arange(out_shape[j]).reshape([-1 if t == j else 1 for t in range(nd)])
        starts.append(o * stride[j] - padding[j])
    return kernel, dilation, starts


def to_local(gidx, x, args, nd):
    """torch's flat plane indices as Gaudi's window-local ones (kernel coordinates, row-major; measured)."""
    kernel, dilation, starts = _pool_geometry(x, args, nd, gidx.shape[-nd:])
    rem, coords = gidx, []
    for size in reversed(x.shape[-nd:]):
        coords.append(rem % size)
        rem = rem // size
    coords.reverse()
    local = torch.zeros_like(gidx)
    for j in range(nd):
        local = local * kernel[j] + (coords[j] - starts[j]) // dilation[j]
    return local


def to_global(local, x, args, nd):
    kernel, dilation, starts = _pool_geometry(x, args, nd, local.shape[-nd:])
    flat, rem = torch.zeros_like(local), local
    coords = []
    for j in reversed(range(nd)):
        coords.append(starts[j] + (rem % kernel[j]) * dilation[j])
        rem = rem // kernel[j]
    coords.reverse()
    for j, size in enumerate(x.shape[-nd:]):
        flat = flat * size + coords[j]
    return flat


REAL_OUT = extrema.native_out
INT64_MIN = torch.iinfo(torch.int64).min
OUT_LARGEST = {aten.amax.out, aten.max.dim_max, aten.argmax.out, aten.median.dim_values,
               aten.max_pool2d_with_indices.out, aten.max_pool3d_with_indices.out}
OUT_SMALLEST = {aten.amin.out, aten.min.dim_min, aten.argmin.out}
# 16-bit rows of the fill (-inf for max, +inf for min) get an index out of range: n eager, INT64_MIN lazy (measured)
OUT_OF_RANGE = {aten.max.dim_max: "garbage", aten.argmin.out: "garbage", aten.min.dim_min: "n", aten.argmax.out: "n"}
POOL_OUTS = {aten.max_pool2d_with_indices.out: 2, aten.max_pool3d_with_indices.out: 3}


def drop(x, fill):
    return torch.where(torch.isnan(x), fill, x) if isinstance(x, torch.Tensor) and x.is_floating_point() else x


def gaudi_out(op, device_type):
    """Gaudi's ``.out`` kernels as measured: NaN dropped (read as -inf by the max ops, +inf by the min ops),
    ``maximum`` / ``fmax`` and their min twins return the first operand where one is NaN, clamp ignores a NaN input
    or bound, max_pool returns window-local indices, and in 16-bit dtypes argmax / max(dim) of a row of -inf (+inf:
    min) return an index out of range (n, or INT64_MIN as lazy mode)."""
    real = REAL_OUT(op, device_type)

    def fn(*args, **kwargs):
        x, rest = args[0], args[1:]
        if not x.is_floating_point():
            return real(*args, **kwargs)
        if op in POOL_OUTS:
            real(drop(x, -INF), *rest, **kwargs)
            kwargs["indices"].copy_(to_local(kwargs["indices"], x, rest, POOL_OUTS[op]))
            return kwargs["out"], kwargs["indices"]
        if op in OUT_LARGEST or op in OUT_SMALLEST:
            fill = -INF if op in OUT_LARGEST else INF
            out = real(drop(x, fill), *rest, **kwargs)
            if op in OUT_OF_RANGE and x.dtype in extrema.REDUCED and x.dim() and rest and rest[0] is not None:
                d, keep = rest[0], len(rest) > 1 and rest[1]
                bad = x.shape[d] if OUT_OF_RANGE[op] == "n" else INT64_MIN
                i = out[1] if isinstance(out, tuple) else out
                i.copy_(torch.where((drop(x, fill) == fill).all(d, keepdim=keep), bad, i))
            return out
        if op is aten.aminmax.out:
            lo, hi = kwargs["min"], kwargs["max"]
            tmp = torch.empty_like(hi)
            real(drop(x, INF), dim=kwargs.get("dim"), keepdim=kwargs.get("keepdim", False), min=lo, max=tmp)
            real(drop(x, -INF), dim=kwargs.get("dim"), keepdim=kwargs.get("keepdim", False), min=tmp, max=hi)
            return lo, hi
        if op in (aten.maximum.out, aten.minimum.out, aten.fmax.out, aten.fmin.out):
            a, b = args
            out = real(a, b, **kwargs)
            return out.copy_(torch.where(torch.isnan(a) | torch.isnan(b), a, out))
        if op in (aten.linalg_vector_norm.out, aten.norm.out):  # NaN read as 0
            return real(drop(x, 0.0), *rest, **kwargs)
        if op in (aten.clamp.Tensor_out, aten.clamp_min.Tensor_out, aten.clamp_max.Tensor_out):
            lows = op is not aten.clamp_max.Tensor_out
            bounds = [drop(b, -INF if k == 0 and lows else INF) for k, b in enumerate(rest)]
            return real(drop(x, 0.0), *bounds, **kwargs)
        return real(*args, **kwargs)

    return fn


def gaudi(op):
    """The functional ops hpu-ops rewrites at the autograd key only (the sorts), as Gaudi computes them: NaN read as
    +-inf. Every other op is torch's own (it reaches the backend layer,
    which calls ``gaudi_out``)."""
    real = REAL(op)

    def fn(*args, **kwargs):
        x, rest = args[0], args[1:]
        if not (isinstance(x, torch.Tensor) and x.is_floating_point()):
            return real(*args, **kwargs)
        if op in (aten.sort.default, aten.sort.stable):
            desc = kwargs.get("descending", rest[1] if len(rest) > 1 else False)
            return real(drop(x, -INF if desc else INF), *rest, **kwargs)
        return real(*args, **kwargs)

    return fn


def _pool_backward_reads_local_indices():
    """Gaudi's max_pool backward reads its window-local indices; torch's CPU one reads flat plane indices."""
    library = torch.library.Library("aten", "IMPL")
    for op, out_op, nd in ((aten.max_pool2d_with_indices_backward.default,
                            aten.max_pool2d_with_indices_backward.grad_input, 2),
                           (aten.max_pool3d_with_indices_backward.default,
                            aten.max_pool3d_with_indices_backward.grad_input, 3)):
        def kernel(grad, x, *rest, out_op=out_op, nd=nd):
            *args, local = rest
            out = torch.empty_like(x)
            return out_op._op_dk(torch._C.DispatchKey.CPU, grad, x, *args, to_global(local, x, args, nd),
                                 grad_input=out)

        library.impl(op, kernel, "CPU")
    return library._destroy


def _install_card():
    undos = [extrema.install("cpu", exact_sort=True), _pool_backward_reads_local_indices()]
    return lambda: [u() for u in reversed(undos)]


def compiled(f, nodes, **options):
    """``torch.compile`` through aot_autograd, collecting graph nodes into ``nodes``. The graphs run below autograd,
    as a device graph compiler runs its ops: run through the dispatcher, a graph's native op would reach the rewrite
    (and the stand-in) a second time. On the card that is idempotent (a fixed result fixes to itself); the stand-in's
    index conversion is not."""

    def record(gm, example_inputs):
        nodes.extend(n for n in gm.graph.nodes if n.op == "call_function")

        def run(*args):
            with torch._C._AutoDispatchBelowAutograd():
                return gm.forward(*args)

        return make_boxed_func(run)

    torch._dynamo.reset()
    return torch.compile(f, backend=aot_autograd(fw_compiler=record, bw_compiler=record), **options)


@pytest.fixture
def card(monkeypatch):
    """extrema installed for "cpu", over the NaN-dropping stand-in of Gaudi's kernels; yields the call counts."""
    monkeypatch.setattr(extrema, "native", gaudi)
    monkeypatch.setattr(extrema, "native_out", gaudi_out)
    _CURRENT["undo"] = _install_card()
    hpu_ops._STATE["calls"].clear()
    try:
        yield hpu_ops._STATE["calls"]
    finally:
        _CURRENT.pop("undo")()


def _reference(f, *args):
    """``f`` with the rewrite off: torch's CPU kernels, which propagate NaN."""
    _CURRENT.pop("undo")()
    try:
        return f(*args)
    finally:
        _CURRENT["undo"] = _install_card()


def rows(n=7, dtype=torch.float64, seed=0):
    """NaN first / middle / last, NaN with +inf and -inf, all NaN, finite, all -inf, ties, -0.0 and 0.0."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(10, n, generator=g, dtype=torch.float64)
    x[0, 0] = NAN
    x[1, n // 2] = NAN
    x[2, -1] = NAN
    x[3, 0], x[3, -1] = INF, NAN
    x[4, -1], x[4, 0] = -INF, NAN
    x[5] = NAN
    x[7] = -INF
    x[8] = 1.5
    x[9, 0], x[9, 1], x[9, 2:] = -0.0, 0.0, -1.0
    return x.to(dtype)


def same(got, want):
    if isinstance(want, (tuple, list)):
        assert len(got) == len(want)
        for g, w in zip(got, want):
            same(g, w)
        return
    if want is None:
        assert got is None
        return
    assert got.dtype == want.dtype and got.shape == want.shape, (got.dtype, want.dtype, got.shape, want.shape)
    if want.is_floating_point():
        assert torch.equal(torch.isnan(got), torch.isnan(want)), (got, want)
        torch.testing.assert_close(got, want, equal_nan=True, atol=0, rtol=0)
    else:
        assert torch.equal(got, want), (got, want)


def grads(f, *xs):
    """Output and input gradients of a weighted sum (weights 1..k, so a gradient's destination shows)."""
    xs = [x.detach().requires_grad_() for x in xs]
    out = f(*xs)
    outs = out if isinstance(out, (tuple, list)) else (out,)
    loss = sum((o * torch.arange(1, o.numel() + 1, dtype=o.dtype).reshape(o.shape)).sum()
               for o in outs if o.is_floating_point())
    loss.backward()
    return [o.detach() for o in outs] + [x.grad for x in xs]


REDUCTIONS = {
    "amax_last": lambda x: x.amax(-1), "amax_dim0": lambda x: x.amax(0), "amax_all": lambda x: x.amax(),
    "amax_dims": lambda x: x.reshape(2, 5, -1).amax((0, 2)), "amax_keep": lambda x: x.amax(-1, keepdim=True),
    "amin_last": lambda x: x.amin(-1), "amin_all": lambda x: x.amin(),
    "max_all": lambda x: x.max(), "min_all": lambda x: x.min(), "median_all": lambda x: x.median(),
    "max_dim": lambda x: tuple(x.max(-1)), "max_dim0_keep": lambda x: tuple(x.max(0, keepdim=True)),
    "min_dim": lambda x: tuple(x.min(-1)), "median_dim": lambda x: tuple(x.median(-1)),
    "argmax": lambda x: x.argmax(-1), "argmin": lambda x: x.argmin(-1), "argmax_all": lambda x: x.argmax(),
    "argmax_all_keep": lambda x: x.argmax(keepdim=True), "argmin_dim0": lambda x: x.argmin(0, keepdim=True),
    "aminmax": lambda x: tuple(torch.aminmax(x, dim=-1)), "aminmax_all": lambda x: tuple(torch.aminmax(x)),
    "norm_inf": lambda x: x.norm(INF, -1), "norm_ninf": lambda x: x.norm(-INF, -1),
    "norm_inf_all": lambda x: x.norm(INF), "vnorm_inf": lambda x: torch.linalg.vector_norm(x, INF, -1),
    "vnorm_inf_all": lambda x: torch.linalg.vector_norm(x, INF),
    "vnorm_ninf_keep": lambda x: torch.linalg.vector_norm(x, -INF, (0, 1), keepdim=True),
}
NO_GRAD = {"aminmax", "aminmax_all", "argmax", "argmin", "argmax_all", "argmax_all_keep", "argmin_dim0"}


@pytest.mark.parametrize("dtype", FLOATS)
@pytest.mark.parametrize("name", sorted(REDUCTIONS))
def test_reductions_propagate_nan_as_torch(card, name, dtype):
    f, x = REDUCTIONS[name], rows(dtype=dtype)
    same(f(x), _reference(f, x))
    assert sum(card.values()) > 0


@pytest.mark.parametrize("name", sorted(set(REDUCTIONS) - NO_GRAD))
def test_reduction_gradients_are_torch_s(card, name):
    """Torch's own gradients at NaN: amax's NaN slice is all NaN, max() / median() split it among the NaNs, max(dim)
    sends it to the first NaN, the inf-norms give 0."""
    f, x = REDUCTIONS[name], rows()
    same(grads(f, x), _reference(grads, f, x))


def test_aminmax_has_no_derivative_as_torch(card):
    x = rows().requires_grad_()
    with pytest.raises(RuntimeError, match="not implemented"):
        sum(torch.aminmax(x, dim=-1)).sum().backward()


def test_the_stand_in_drops_nan_like_the_card(monkeypatch):
    """Without the rewrite, the stand-in reproduces the card's results (so the tests above test something)."""
    x = rows(dtype=torch.float32)
    amax = gaudi_out(aten.amax.out, "cpu")(x, [-1], False, out=torch.empty(10))
    assert not torch.isnan(amax).any() and amax[0] == x[0, 1:].max()
    xb = rows(dtype=torch.bfloat16)
    v, i = gaudi_out(aten.max.dim_max, "cpu")(xb, -1, False, max=torch.empty(10, dtype=xb.dtype),
                                              max_values=torch.empty(10, dtype=torch.int64))
    assert not torch.isnan(v).any() and i[7] == INT64_MIN  # garbage for the row of -inf, as Gaudi lazy
    i = gaudi_out(aten.argmax.out, "cpu")(xb, -1, False, out=torch.empty(10, dtype=torch.int64))
    assert i[7] == 7  # n, as Gaudi eager
    out = gaudi_out(aten.maximum.out, "cpu")(torch.ones(2), torch.tensor([NAN, 1.0]), out=torch.empty(2))
    assert not torch.isnan(out).any()


@pytest.mark.parametrize("dtype", FLOATS)
def test_argmax_of_16_bit_rows_of_inf_is_in_range(card, dtype):
    x = torch.tensor([[-INF] * 4, [INF] * 4, [1.0, 3.0, 3.0, 0.0], [NAN] * 4]).to(dtype)
    same(x.argmax(-1), torch.tensor([0, 0, 1, 0]))
    same(x.argmin(-1), torch.tensor([0, 0, 3, 0]))
    same(tuple(x.max(-1)), (torch.tensor([-INF, INF, 3.0, NAN]).to(dtype), torch.tensor([0, 0, 1, 0])))


BINARY = {
    "maximum": torch.maximum, "minimum": torch.minimum, "max_other": torch.max, "min_other": torch.min,
    "fmax": torch.fmax, "fmin": torch.fmin,
    "clamp_t": lambda a, b: a.clamp(b, b + 1), "clamp_min_t": lambda a, b: a.clamp(min=b),
    "clamp_max_t": lambda a, b: a.clamp(max=b), "clamp_min_fn": torch.clamp_min, "clamp_max_fn": torch.clamp_max,
}


def pair(dtype=torch.float64):
    a = rows(dtype=dtype)
    b = rows(seed=1, dtype=dtype).flip(-1)
    b[8] = NAN
    b[9, 0], b[9, 1] = 0.0, -0.0
    return a, b


@pytest.mark.parametrize("dtype", FLOATS)
@pytest.mark.parametrize("name", sorted(BINARY))
def test_binary_ops_propagate_nan_as_torch(card, name, dtype):
    a, b = pair(dtype)
    same(BINARY[name](a, b), _reference(BINARY[name], a, b))
    assert sum(card.values()) > 0


@pytest.mark.parametrize("name", sorted(BINARY))
def test_binary_gradients_are_torch_s(card, name):
    a, b = pair()
    same(grads(BINARY[name], a, b), _reference(grads, BINARY[name], a, b))


def test_binary_broadcasts_and_promotes(card):
    a = torch.tensor([[NAN, 1.0, 2.0]], dtype=torch.float32)
    b = torch.tensor([[1.0], [NAN]], dtype=torch.float64)
    same(torch.maximum(a, b), _reference(torch.maximum, a, b))
    same(torch.fmin(a, torch.tensor(NAN)), _reference(torch.fmin, a, torch.tensor(NAN)))


POOLS = {
    # the 1-D op through its indices form: the CPU's no-grad max_pool1d has its own kernel, Gaudi's reaches
    # max_pool2d_with_indices (measured)
    "pool1d": lambda x: F.max_pool1d(x, 2, return_indices=True)[0],
    "pool1d_k3s1p1": lambda x: F.max_pool1d(x, 3, 1, 1, return_indices=True)[0],
    "pool2d": lambda x: F.max_pool2d(x.reshape(2, 5, 2, -1), 2), "pool2d_ceil": lambda x: F.max_pool2d(
        x.reshape(2, 5, 2, -1), (2, 3), ceil_mode=True),
    "pool2d_dil": lambda x: F.max_pool2d(x.reshape(1, 10, 2, -1), (1, 2), dilation=(1, 2)),
    "pool2d_unbatched": lambda x: F.max_pool2d(x.reshape(10, 2, -1), 2),
    "pool3d": lambda x: F.max_pool3d(x.reshape(1, 10, 2, 2, -1), 2, padding=(1, 0, 0)),
}


@pytest.mark.parametrize("dtype", FLOATS)
@pytest.mark.parametrize("name", sorted(POOLS))
def test_max_pool_propagates_nan_as_torch(card, name, dtype):
    x = rows(n=8, dtype=dtype)
    same(POOLS[name](x), _reference(POOLS[name], x))
    assert card["max_pool2d"] + card["max_pool3d"] > 0


@pytest.mark.parametrize("name", sorted(POOLS))
def test_max_pool_gradients_go_to_the_nan_as_torch(card, name):
    """torch sends a NaN window's gradient to its last NaN, the other windows' to their max."""
    x = rows(n=8)
    x[6, 1] = NAN  # two NaNs in one window
    x[6, 0] = NAN
    same(grads(POOLS[name], x), _reference(grads, POOLS[name], x))


@pytest.mark.parametrize("name", sorted(POOLS))
def test_max_pool_indices_are_gaudi_s_form_of_torch_s(card, name):
    """The indices returned keep Gaudi's window-local form; they point to torch's pick, the last NaN of a NaN
    window."""
    x = rows(n=8)
    x[6, 0] = x[6, 1] = NAN
    nodes = {}

    def spy(t):
        out = POOLS[name](t)
        return out

    import hpu_ops.extrema as ex

    seen = []
    real_pool = ex._pool

    def record(x_, args, fn, nd):
        out, idx = real_pool(x_, args, fn, nd)
        seen.append((x_, args, nd, idx))
        return out, idx

    ex._pool, saved = record, real_pool
    try:
        spy(x)
    finally:
        ex._pool = saved
    (x_, args, nd, idx), = seen
    op = aten.max_pool2d_with_indices.default if nd == 2 else aten.max_pool3d_with_indices.default
    _, want = _reference(lambda t: REAL(op)(t, *args), x_)
    assert torch.equal(idx, to_local(want, x_, args, nd))
    del nodes


# ---- edges


def test_integer_and_bool_dtypes_keep_the_native_op(card):
    for x in (torch.arange(12).reshape(3, 4), torch.tensor([[True, False]]), torch.arange(6, dtype=torch.uint8)):
        for f in (lambda t: t.amax(-1), lambda t: tuple(t.max(-1)), lambda t: torch.maximum(t, t.flip(-1))):
            same(f(x), _reference(f, x))
        if x.dtype != torch.bool:
            same(x.argmax(-1), _reference(lambda t: t.argmax(-1), x))
    assert sum(card.values()) == 0


def test_empty_and_zero_dim_inputs(card):
    with pytest.raises((RuntimeError, IndexError)):
        torch.empty(3, 0).amax(-1)
    with pytest.raises((RuntimeError, IndexError)):
        torch.empty(3, 0).max(-1)
    with pytest.raises((RuntimeError, IndexError)):
        torch.empty(0).argmax()
    same(torch.empty(0, 4).amax(-1), torch.empty(0))
    same(tuple(torch.empty(0, 4).max(-1)), (torch.empty(0), torch.empty(0, dtype=torch.int64)))
    same(torch.empty(0, 4).argmax(-1), torch.empty(0, dtype=torch.int64))
    for v in (NAN, 2.0, -INF):
        s = torch.tensor(v)
        same(s.amax(), _reference(lambda t: t.amax(), s))
        same(s.max(), _reference(lambda t: t.max(), s))
        same(tuple(s.max(0)), _reference(lambda t: tuple(t.max(0)), s))
        same(s.argmax(), _reference(lambda t: t.argmax(), s))
        same(s.norm(INF), _reference(lambda t: t.norm(INF), s))


def test_non_contiguous_and_offset_views(card):
    x = rows(n=12, dtype=torch.float32)
    for v in (x.t(), x[:, 1::3], x[2:, 3:]):
        for f in (lambda t: t.amax(0), lambda t: tuple(t.max(-1)), lambda t: t.argmin(0), lambda t: F.max_pool1d(
                t.unsqueeze(0), 2, return_indices=True)[0]):
            same(f(v), _reference(f, v))


INFERENCE = {**REDUCTIONS,
             "maximum": lambda x: torch.maximum(x, x.flip(0)), "fmin": lambda x: torch.fmin(x, x.flip(0)),
             "clamp_t": lambda x: x.clamp(x.flip(0) - 1, x.flip(0) + 1),
             "clamp_inplace": lambda x: x.clone().clamp_(min=x.flip(0)),
             "clamp_max_inplace": lambda x: x.clone().clamp_max_(x.flip(-1)),
             "pool1d": lambda x: F.max_pool1d(x.unsqueeze(0), 2, return_indices=True)[0],
             "pool3d": lambda x: F.max_pool3d(x.reshape(1, 10, 1, 1, -1), (1, 1, 2))}


@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16))
@pytest.mark.parametrize("name", sorted(INFERENCE))
def test_inference_mode_propagates_nan(card, name, dtype):
    """Evaluation loops (Lightning validate / test / predict) run under inference mode, below autograd: the backend
    layer fixes them too."""
    x, f = rows(n=8, dtype=dtype), INFERENCE[name]
    with torch.inference_mode():
        got = f(x)
    same(got, _reference(f, x))
    assert sum(card.values()) > 0


OUT_CALLS = {
    "amax": lambda x: torch.amax(x, -1, out=torch.empty(0, dtype=x.dtype)),
    "amin_keep": lambda x: torch.amin(x, 0, keepdim=True, out=torch.empty(1, 8, dtype=x.dtype)),
    "max_dim": lambda x: tuple(torch.max(x, -1, out=(torch.empty(0, dtype=x.dtype), torch.empty(0, dtype=torch.long)))),
    "min_dim": lambda x: tuple(torch.min(x, 0, out=(torch.empty(8, dtype=x.dtype), torch.empty(8, dtype=torch.long)))),
    "median_dim": lambda x: tuple(torch.median(x, -1, out=(torch.empty(0, dtype=x.dtype),
                                                          torch.empty(0, dtype=torch.long)))),
    "argmax": lambda x: torch.argmax(x, -1, out=torch.empty(0, dtype=torch.long)),
    "argmin": lambda x: torch.argmin(x, out=torch.empty(0, dtype=torch.long)),
    "aminmax": lambda x: tuple(torch.aminmax(x, dim=-1, out=(torch.empty(0, dtype=x.dtype),
                                                             torch.empty(0, dtype=x.dtype)))),
    "maximum": lambda x: torch.maximum(x, x.flip(0), out=torch.empty(0, dtype=x.dtype)),
    "fmax_to_double": lambda x: torch.fmax(x, x.flip(0), out=torch.empty(0, dtype=torch.float64)),
    "clamp_t": lambda x: torch.clamp(x, x.flip(0) - 1, x.flip(0) + 1, out=torch.empty(0, dtype=x.dtype)),
    "clamp_min_t": lambda x: torch.clamp_min(x, x.flip(0), out=torch.empty(0, dtype=x.dtype)),
    "pool": lambda x: tuple(torch.ops.aten.max_pool2d_with_indices.out(
        x.reshape(1, 10, 1, 8), [1, 2], [], [0, 0], [1, 1], False, out=torch.empty(0, dtype=x.dtype),
        indices=torch.empty(0, dtype=torch.long)))[:1],
    "vnorm_inf": lambda x: torch.linalg.vector_norm(x, INF, -1, out=torch.empty(0, dtype=x.dtype)),
}


@pytest.mark.parametrize("name", sorted(OUT_CALLS))
def test_out_overloads_propagate_nan(card, name):
    x, f = rows(n=8, dtype=torch.float32), OUT_CALLS[name]
    same(f(x), _reference(f, x))


def test_out_keeps_torch_s_errors(card):
    x = rows(dtype=torch.float32)
    for f in (lambda: torch.amax(x, -1, out=torch.empty(10, dtype=torch.long)),  # float into long
              lambda: torch.amax(x.requires_grad_(), -1, out=torch.empty(10))):  # out= with autograd
        with pytest.raises(RuntimeError):
            _reference(f)
        with pytest.raises(RuntimeError):
            f()


def test_compiled_inplace_clamp_runs_the_rewrite(card):
    nodes = []
    f = lambda t, lo: t.clone().clamp_(min=lo)  # noqa: E731
    x, lo = torch.tensor([1.0, NAN, 3.0]), torch.tensor([NAN, 0.0, 2.0])
    same(compiled(f, nodes, dynamic=False)(x, lo), _reference(f, x, lo))
    assert "aten.isnan.default" in ops(nodes), ops(nodes)


@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16))
def test_clamp_with_one_nan_bound_has_torch_s_gradient(card, dtype):
    """clamp(5, min=nan, max=1): NaN, and the max bound's gradient 1, as torch's comparisons give."""
    args = (torch.tensor([5.0, 0.0, NAN, -3.0], dtype=dtype), torch.tensor([NAN, 1.0, 0.0, -1.0], dtype=dtype),
            torch.tensor([1.0, NAN, 2.0, NAN], dtype=dtype))
    for f in (torch.clamp, lambda t, lo, hi: torch.clamp_min(t, lo), lambda t, lo, hi: torch.clamp_max(t, hi)):
        same(grads(f, *args), _reference(grads, f, *args))


@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16))
@pytest.mark.parametrize("name", sorted(set(REDUCTIONS) - NO_GRAD))
def test_reduction_gradients_in_float32_and_bfloat16(card, name, dtype):
    f, x = REDUCTIONS[name], rows(dtype=dtype)
    same(grads(f, x), _reference(grads, f, x))


MULTI = {"max_dim": lambda x: tuple(x.max(-1)), "min_dim": lambda x: tuple(x.min(-1)),
         "median_dim": lambda x: tuple(x.median(-1)), "argmax": lambda x: x.argmax(-1), "argmin": lambda x: x.argmin(0),
         "amax": lambda x: x.amax(-1), "pool": lambda x: F.max_pool1d(x.unsqueeze(0), 4, return_indices=True)[0]}


@pytest.mark.parametrize("name", sorted(MULTI))
def test_rows_with_several_nans(card, name):
    """torch's index of the first NaN, and the gradients, with two or three NaNs in a row."""
    x = torch.tensor([[1.0, NAN, 2.0, NAN, 0.0, 5.0, NAN, 1.0], [NAN, NAN, 1.0, 2.0, 3.0, NAN, NAN, NAN],
                      [INF, 0.0, NAN, -INF, NAN, 1.0, 2.0, 3.0]], dtype=torch.float64)
    same(MULTI[name](x), _reference(MULTI[name], x))
    if name not in ("argmax", "argmin"):
        f = (lambda t: MULTI[name](t)[0]) if isinstance(MULTI[name](x), tuple) else MULTI[name]
        same(grads(f, x), _reference(grads, f, x))


def test_garbage_indices_of_16_bit_rows_are_fixed(card):
    """Lazy mode returns INT64_MIN (negative) for a 16-bit row of -inf; eager returns n: both become torch's 0."""
    x = torch.tensor([[-INF] * 3, [1.0, 2.0, -INF]], dtype=torch.bfloat16)
    same(tuple(x.max(-1)), _reference(lambda t: tuple(t.max(-1)), x))
    same((-x).argmin(-1), _reference(lambda t: (-t).argmin(-1), x))


def test_vector_norm_with_a_wider_dtype(card):
    x = rows(dtype=torch.float32)
    f = lambda t: torch.linalg.vector_norm(t, INF, -1, dtype=torch.float64)  # noqa: E731
    same(f(x), _reference(f, x))
    same(grads(f, x), _reference(grads, f, x))


def test_other_devices_keep_the_native_op():
    x = rows(dtype=torch.float32)
    with cpu_as_hpu(lambda d: extrema.install("cuda")) as calls:
        x.amax(-1)
    assert not calls["amax"]


def test_no_grad_and_requires_grad_agree(card):
    x = rows(dtype=torch.float32)
    for f in (lambda t: t.amax(-1), lambda t: t.max(), lambda t: F.max_pool2d(t.reshape(2, 5, 7), (1, 2))):
        with torch.no_grad():
            a = f(x)
        same(f(x.clone().requires_grad_()).detach(), a)


@pytest.mark.parametrize("name", ["amax_last", "max_dim", "max_all", "norm_inf"])
def test_double_backward(card, name):
    """The gradient of the gradient (weights times the input's first gradient) as torch's."""
    f = REDUCTIONS[name]
    f = (lambda t: REDUCTIONS[name](t)[0]) if name == "max_dim" else f

    def second(x):  # the double backward with respect to the output's weights: J v
        x = x.detach().requires_grad_()
        out = f(x)
        w = torch.ones_like(out).requires_grad_()
        (g,) = torch.autograd.grad((out * w).sum(), x, create_graph=True)
        v = torch.arange(x.numel(), dtype=x.dtype).reshape(x.shape)
        (gw,) = torch.autograd.grad((g.nan_to_num(0.0) * v).sum(), w)
        return g.detach(), gw

    x = rows()
    x[8, 0] = 2.0  # break the tie of row 8
    same(second(x), _reference(second, x))


@pytest.mark.parametrize("name", ["amax_last", "max_dim", "argmax", "aminmax"])
def test_compiled_graphs_run_the_rewrite(card, name):
    f, x, nodes = REDUCTIONS[name], rows(dtype=torch.float32), []
    got = compiled(f, nodes, dynamic=False)(x)
    same(got, _reference(f, x))
    assert "aten.isnan.default" in ops(nodes), ops(nodes)


def test_compiled_max_pool_and_maximum_with_gradients(card):
    x, nodes = rows(n=8), []
    f = lambda t: torch.maximum(F.max_pool1d(t.unsqueeze(0), 2), t.unsqueeze(0)[..., ::2].flip(-1))  # noqa: E731
    got = grads(compiled(f, nodes, dynamic=False), x)
    same(got, _reference(grads, f, x))
    assert "aten.isnan.default" in ops(nodes)


def test_undo_restores_the_native_op(monkeypatch):
    """After the undo the stand-in (dropping NaN) is what runs: the rewrite is gone."""
    monkeypatch.setattr(extrema, "native", gaudi)
    monkeypatch.setattr(extrema, "native_out", gaudi_out)
    undo = extrema.install("cpu")
    x = rows(dtype=torch.float32)
    with torch.inference_mode():
        assert torch.isnan(x.amax(-1)[0])
    undo()
    assert not extrema._BACKEND
    assert torch.isnan(x.amax(-1)[0])  # torch's CPU kernel itself, no rewrite (nor stand-in) in between


# ---- adversarial round


@pytest.mark.parametrize("f", [lambda t: t.amax(5), lambda t: t.amax((0, 0)), lambda t: t.amin(-3),
                               lambda t: t.max(2), lambda t: t.argmax(2), lambda t: t.norm(INF, 4),
                               lambda t: torch.linalg.vector_norm(t, INF, (1, 1)), lambda t: torch.aminmax(t, dim=3),
                               lambda t: t.median(7)])
def test_bad_dims_raise_torch_s_error(card, f):
    x = rows()
    with pytest.raises(Exception) as want:
        _reference(f, x)
    with pytest.raises(type(want.value)):
        f(x)
    with pytest.raises(type(want.value)):
        f(x.clone().requires_grad_())


def test_mixed_dtypes_and_broadcast_gradients(card):
    a = torch.tensor([[NAN, 1.0, 2.0, 2.0]], dtype=torch.float32)
    b = torch.tensor([[1.0], [NAN], [2.0]], dtype=torch.float64)
    for f in (torch.maximum, torch.minimum, torch.fmax, lambda u, v: u.clamp(v, v + 1)):
        same(grads(f, a, b), _reference(grads, f, a, b))
    s = torch.tensor(NAN, dtype=torch.float64)
    same(grads(torch.maximum, a, s), _reference(grads, torch.maximum, a, s))


def test_one_operand_without_grad(card):
    a, b = pair()
    f = torch.maximum
    xa = a.clone().requires_grad_()
    torch.maximum(xa, b).sum().backward()
    ra = a.clone().requires_grad_()
    _reference(lambda u: torch.maximum(u, b).sum().backward(), ra)
    same(xa.grad, ra.grad)
    del f


@pytest.mark.parametrize("name", ["amax_last", "max_dim", "argmax", "norm_inf"])
def test_dynamic_shapes_compile(card, name):
    f, nodes = REDUCTIONS[name], []
    g = compiled(f, nodes, dynamic=True)
    for n in (7, 9):
        x = rows(n=n, dtype=torch.float32)
        same(g(x), _reference(f, x))


def test_dynamic_shapes_max_pool(card):
    f, nodes = POOLS["pool2d"], []
    g = compiled(lambda t: grads(f, t), nodes, dynamic=True)
    for n in (8, 12):
        x = rows(n=n)
        same(g(x), _reference(grads, f, x))


def test_nan_and_inf_in_one_window_and_all_nan_planes(card):
    x = torch.tensor([[[[NAN, INF], [-INF, NAN]], [[NAN, NAN], [NAN, NAN]], [[INF, -INF], [0.0, -0.0]]]],
                     dtype=torch.float64)
    for f in (lambda t: F.max_pool2d(t, 2), lambda t: F.max_pool2d(t, 1), lambda t: F.max_pool2d(t, 2, 1, 1)):
        same(grads(f, x), _reference(grads, f, x))


def test_keepdim_all_dims_and_amax_of_empty_dim_list(card):
    x = rows(dtype=torch.float32)
    for f in (lambda t: t.amax(dim=(), keepdim=True), lambda t: t.amax(dim=[0, 1], keepdim=True),
              lambda t: t.amin(dim=[]), lambda t: torch.linalg.vector_norm(t, -INF, keepdim=True)):
        same(f(x), _reference(f, x))
        same(grads(f, x.double()), _reference(grads, f, x.double()))


def test_compiled_max_pool_indices(card):
    for seed in range(4):
        x = rows(n=8, seed=seed).reshape(1, 10, 2, 4)
        f = lambda t: F.max_pool2d(t, 2, return_indices=True)  # noqa: E731
        same(compiled(f, [], dynamic=False)(x), f(x))


def test_a_graph_run_through_the_rewrite_again_gives_the_same():
    """A compiled graph whose ops go through the dispatcher reaches the rewrite again with the fixed results: fixing
    them again changes nothing (here over torch's own CPU kernels, which already propagate NaN). Not the pools: their
    NaN-window index is Gaudi's window-local form, which torch's CPU backward does not read (the card's compiled run
    is in ``repro/probe_extrema_grads.py --compile``)."""
    from test_cpu import compiled as redispatching

    names = ["amax_last", "amax_all", "max_all", "max_dim", "median_dim", "norm_inf", "vnorm_inf_all"]
    cases = [REDUCTIONS[n] for n in names]
    x = rows(n=8)
    wants = [grads(f, x) for f in cases] + [grads(torch.maximum, x, x.flip(0))]
    with cpu_as_hpu(extrema.install) as calls:
        gots = [grads(redispatching(f, [], dynamic=False), x) for f in cases]
        gots.append(grads(redispatching(torch.maximum, [], dynamic=False), x, x.flip(0)))
        assert sum(calls.values()) > 0
    for got, want in zip(gots, wants):
        same(got, want)


# ---- sort / topk / kthvalue


SORTS = {
    "sort": lambda x: tuple(x.sort(-1)), "sort_desc": lambda x: tuple(x.sort(-1, descending=True)),
    "sort_stable": lambda x: tuple(x.sort(dim=-1, stable=True)),
    "sort_stable_desc_dim0": lambda x: tuple(x.sort(dim=0, stable=True, descending=True)),
    "argsort": lambda x: x.argsort(-1),
    "argsort_stable_desc": lambda x: x.argsort(dim=-1, descending=True, stable=True),
    "msort": lambda x: x.msort(),
    "topk3": lambda x: tuple(x.topk(3, -1)), "topk3_small": lambda x: tuple(x.topk(3, -1, largest=False)),
    "topk_all": lambda x: tuple(x.topk(x.shape[-1], -1)), "topk0": lambda x: tuple(x.topk(0, -1)),
    "topk2_dim0": lambda x: tuple(x.topk(2, 0)),
    "kthvalue2": lambda x: tuple(x.kthvalue(2, -1)), "kthvalue_last_keep": lambda x: tuple(
        x.kthvalue(x.shape[-1], -1, keepdim=True)),
}
STABLE = {"sort_stable", "sort_stable_desc_dim0", "argsort_stable_desc"}


def same_selection(got, want, x, dim, exact_index):
    """Values exactly torch's; indices torch's when the sort is stable, else pointing to those values."""
    if not isinstance(want, tuple) and want.is_floating_point():  # values only (msort)
        same(got, want)
        return
    got = got if isinstance(got, tuple) else (None, got)
    want = want if isinstance(want, tuple) else (None, want)
    if want[0] is not None:
        same(got[0], want[0])
    if exact_index:
        same(got[1], want[1])
    else:
        assert got[1].dtype == torch.int64 and got[1].shape == want[1].shape
        gi, wi = (got[1], want[1]) if got[1].dim() == x.dim() else (got[1].unsqueeze(dim), want[1].unsqueeze(dim))
        same(x.gather(dim, gi), x.gather(dim, wi))


@pytest.mark.parametrize("dtype", FLOATS)
@pytest.mark.parametrize("name", sorted(SORTS))
def test_sorts_order_nan_as_torch(card, name, dtype):
    x = rows(dtype=dtype)
    x[6, 2] = INF  # a real +inf next to NaNs
    x[6, 0] = NAN
    f = SORTS[name]
    dim = 0 if "dim0" in name else -1
    if name == "msort":
        dim = 0
    same_selection(f(x), _reference(f, x), x, dim, name in STABLE)
    assert card["sort"] + card["topk"] + card["kthvalue"] > 0


@pytest.mark.parametrize("name", sorted(set(SORTS) - {"argsort", "argsort_stable_desc"}))
def test_sort_gradients_are_torch_s(card, name):
    x = rows()
    x[6, 2], x[6, 0] = INF, NAN
    f = lambda t: SORTS[name](t)[0] if name != "msort" else SORTS[name](t)  # noqa: E731
    got, want = grads(f, x), _reference(grads, f, x)
    same(got[0], want[0])
    if name in STABLE or name.startswith("sort") or name == "msort":
        same(got[1], want[1])


def test_sort_without_nan_matches_including_ties_and_infs(card):
    g = torch.Generator().manual_seed(0)
    x = torch.randn(4, 1200, generator=g)
    x[:, ::7], x[:, 3::11], x[:, 5::13] = INF, -INF, 0.5
    x[:, 6::17] = -0.0
    same_selection(tuple(x.sort(-1)), _reference(lambda t: tuple(t.sort(-1)), x), x, -1, False)
    same(tuple(x.sort(dim=-1, stable=True)), _reference(lambda t: tuple(t.sort(dim=-1, stable=True)), x))


@pytest.mark.parametrize("f", [lambda t: t.topk(11, -1), lambda t: t.topk(-1, -1), lambda t: t.kthvalue(0, -1),
                               lambda t: t.kthvalue(11, -1), lambda t: t.sort(2), lambda t: t.topk(2, 3)])
def test_sort_arguments_torch_rejects_raise(card, f):
    x = rows()
    with pytest.raises(Exception) as want:
        _reference(f, x)
    with pytest.raises(type(want.value)):
        f(x)


def test_sorts_of_integers_scalars_and_empty_keep_the_native_op(card):
    for x in (torch.tensor([3, 1, 2]), torch.tensor(2.0), torch.empty(0), torch.empty(3, 0)):
        for f in (lambda t: tuple(t.sort(-1)), lambda t: tuple(t.topk(0, -1)), lambda t: t.argsort(-1)):
            same(f(x), _reference(f, x))
    assert card["sort"] == card["topk"] == 0


def test_the_sort_family_is_opt_in(monkeypatch):
    monkeypatch.delenv("HPU_OPS_EXACT_SORT", raising=False)
    x = rows()
    with cpu_as_hpu(extrema.install) as calls:
        x.sort(-1), x.topk(2, -1), x.amax(-1)
    assert calls["amax"] and not calls["sort"] and not calls["topk"]
    monkeypatch.setenv("HPU_OPS_EXACT_SORT", "1")
    with cpu_as_hpu(extrema.install) as calls:
        x.sort(-1)
    assert calls["sort"]


def test_the_sort_family_keeps_the_native_op_under_compile(card):
    nodes = []
    f = lambda t: t.sort(-1).values  # noqa: E731
    compiled(f, nodes, dynamic=False)(rows(dtype=torch.float32))
    assert "aten.sort.stable" not in ops(nodes) and card["sort"] == 0
