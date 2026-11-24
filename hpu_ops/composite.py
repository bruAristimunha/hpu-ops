"""HPU composites for ops whose Gaudi kernel fails under ``torch.compile`` (``glu``), is missing (``renorm``,
``Tensor.unfold`` in lazy mode, ``unfold_backward``), is wrong (``embedding_dense_backward`` of a 0-d index, ``roll`` of
a strided input by a multiple of the size) or rejects an inferred argument (``one_hot`` with ``num_classes=-1``).

Each is registered at ``AutogradHPU``, so eager calls and traced graphs both use it and autograd differentiates the
composite; ``renorm`` and ``one_hot`` are also registered at ``HPU`` for calls below autograd (inference mode),
and ``glu`` is as well in lazy mode. ``unfold`` and
``unfold_backward`` are registered at ``HPU`` only, below torch's own autograd kernels, which keep their formulas. Cases a composite does
not take (odd ``glu`` halves, invalid ``renorm`` arguments) go to the op's own kernel, which raises its usual error;
so do embedding gradients of indices with dimensions, which Gaudi gets right. The composites run with autocast off and
return the input dtype, as the native ops do.
"""
import math
import threading

import torch
from torch._C import DispatchKey

from . import _strict
from ._dispatch import REDUCED, chain, count, override_kernels, register_kernels, static, tracing

aten = torch.ops.aten
# the most elements of any tensor unfold_backward's rewrite makes or takes on the card: a pad of 1.1e9 x 2 elements
# stopped Gaudi2 ("Synapse detected a device critical error", 3 of 3 runs; 1.0e9 x 2 was right); larger calls run
# ATen's kernel on a CPU copy
UNFOLD_BACKWARD_MAX_ELEMENTS = 2**31 - 1
_ROUTE = threading.local()


def _opmath(dtype):
    return torch.float32 if dtype in REDUCED else dtype


def glu(input, dim=-1):
    """``a * sigmoid(b)`` for the two halves ``a, b`` of ``input`` along ``dim``; float32 math for 16-bit inputs."""
    half = input.shape[dim] // 2
    a, b = input.narrow(dim, 0, half), input.narrow(dim, half, half)
    dtype = _opmath(input.dtype)
    with torch.autocast(input.device.type, enabled=False):
        return (a.to(dtype) * torch.sigmoid(b.to(dtype))).to(input.dtype)


def _glu_takes(input, dim=-1):
    return input.is_floating_point() and input.dim() > 0 and input.shape[dim] % 2 == 0


def renorm(input, p, dim, maxnorm):
    """Each slice of ``input`` along ``dim`` scaled to a ``p``-norm of at most ``maxnorm``, as ``aten.renorm`` does
    it: by ``maxnorm / (norm + 1e-7)`` where ``norm > maxnorm``, else 1, with the norm in float32 for 16-bit inputs."""
    dim = dim % input.dim()
    others = [d for d in range(input.dim()) if d != dim]
    with torch.autocast(input.device.type, enabled=False):
        norm = torch.linalg.vector_norm(input, p, others, keepdim=True, dtype=_opmath(input.dtype))
        limit = torch.full_like(norm, maxnorm)
        factor = torch.where(norm > limit, limit / (norm + 1e-7), 1.0)
        return input * factor.to(input.dtype)


def _renorm_takes(input, p, dim, maxnorm):
    real = not isinstance(p, complex) and not isinstance(maxnorm, complex)
    shape = input.is_floating_point() and input.dim() > 1 and -input.dim() <= dim < input.dim()
    return real and shape and p > 0 and maxnorm >= 0


def embedding_dense_backward_one(grad_output, indices, num_weights, padding_idx, scale_grad_by_freq):
    """The weight gradient of a 0-d index's embedding as that of the one-element index, the shapes ATen's own kernels
    flatten to. Gaudi's kernel sends a 0-d index's gradient to row 0 (BIOT's channel tokens: every channel's gradient
    to the first token)."""
    return aten.embedding_dense_backward.default._op_dk(DispatchKey.Autograd, grad_output.reshape(1, -1),
                                                        indices.reshape(1), num_weights, padding_idx,
                                                        scale_grad_by_freq)


def _embedding_backward_takes(grad_output, indices, *args, **kwargs):
    return indices.dim() == 0


def roll_unshifted(input, shifts, dims=()):
    """A roll whose every shift is a multiple of its dim's size, as the copy it is: contiguous, as roll's result.
    Gaudi's eager kernel reads a strided input of such a roll wrongly (EMG2QwertyNet: a stack's gradient slice rolled
    back by 0)."""
    return input.clone(memory_format=torch.contiguous_format)


def _roll_takes(input, shifts, dims=()):
    if len(shifts) == 0 or len(shifts) != max(len(dims), 1) or not all(-input.dim() <= d < max(input.dim(), 1)
                                                                       for d in dims):
        return False  # what ATen rejects reaches it
    if all(isinstance(shift, int) and shift == 0 for shift in shifts):
        return True  # a copy at any size, symbolic too
    sizes = [input.numel()] if len(dims) == 0 else [input.shape[d] if input.dim() else 1 for d in dims]
    return static(*sizes, *shifts) and all(size == 0 or shift % size == 0 for size, shift in zip(sizes, shifts))


def unfold(input, dimension, size, step):
    """``Tensor.unfold`` as the view ATen makes of it: ``as_strided`` with the sizes and strides that ATen's own kernel
    computes, read from it on a meta tensor of ``input``'s sizes and strides (which also raises ATen's errors). The
    result is a view of ``input`` at its storage offset, as ``unfold``'s. Habana's lazy stack has no ``unfold``
    kernel: the bridge ran it on the CPU and returned a copy, not a view (``repro/tensor_unfold_lazy.py``)."""
    meta = torch.empty_strided(input.shape, input.stride(), dtype=input.dtype, device="meta")
    view = meta.unfold(dimension, size, step)
    return input.as_strided(view.shape, view.stride(), input.storage_offset())


def _pad_dims(t, pads):
    """``t`` with ``pads[d]`` zeros after its end along each dim ``d`` (one ``constant_pad_nd``). The copy keeps a
    permuted input's layout (CPU and Gaudi alike), so a reshape after it may copy again; building it from zeros instead
    was within noise on Gaudi2 (eager and lazy, #43's shapes)."""
    first = min(pads)
    pad = []
    for d in range(t.dim() - 1, first - 1, -1):
        pad += [0, pads.get(d, 0)]
    return aten.constant_pad_nd.default(t, pad, 0)


def _unfold_backward_plan(n, input_sizes, dim, size, step):
    """How ``unfold_backward`` sums ``n`` windows of ``size`` at ``step`` along ``input_sizes[dim]`` (``dim`` >= 0,
    ``size`` > 0): ``(branch, step, blocks, rows, elements)``, with ``branch`` "pad" (no overlap), "blocks" or
    "windows" (the two skews, the smaller buffer), the step it uses, the zero blocks of an uncovered tail, the rows of
    the block skew, and the most elements of any tensor the branch makes (the gradient and the result included)."""
    length = input_sizes[dim]
    outer, inner = math.prod(input_sizes[:dim]), math.prod(input_sizes[dim + 1:])
    k = -(-size // step)
    blocks = max(-(-length // step) - (n + k - 1), 0)  # e: zero blocks up to the input's end
    base = max(outer * n * size * inner, outer * length * inner)
    if k == 1:
        step = min(step, length)  # step > length: one window (n == 1), padded to the input's end only
        return "pad", step, blocks, 0, max(base, outer * (n + blocks) * step * inner)
    rows = n + k - 1 + blocks
    by_blocks, by_windows = k * (rows + 1) * step, n * (length + step)
    if by_blocks <= by_windows:
        return "blocks", step, blocks, rows, max(base, outer * by_blocks * inner)
    return "windows", step, blocks, rows, max(base, outer * by_windows * inner)


def unfold_backward_elements(grad, input_sizes, dim, size, step):
    """The most elements of any tensor ``unfold_backward`` makes or takes for these arguments (0 where it makes none
    beyond the gradient and result: 0-d inputs, size 0, empty gradients, invalid arguments, which it handles or
    raises itself)."""
    ndim = len(input_sizes)
    if ndim == 0 or size <= 0 or step <= 0 or grad.numel() == 0 or not -ndim <= dim < ndim:
        return 0
    return _unfold_backward_plan(grad.shape[dim % ndim], input_sizes, dim % ndim, size, step)[-1]


def unfold_backward(grad, input_sizes, dim, size, step):
    """The gradient of ``Tensor.unfold`` (``aten.unfold_backward``) as one or two pads and one sum, no index ops.

    Window ``i``, offset ``j`` goes to position ``i * step + j``. Overlaps are added by a skew: a padded buffer,
    flattened and read again with rows one shift shorter, so each row lands shifted, then one sum over the rows. Two
    skews, whichever buffer is smaller (both about the gradient's size at #43's shapes):

    - over the ``k = ceil(size / step)`` blocks of ``step`` offsets in a window (``j = q * step + r`` goes to block
      ``i + q``): the windows padded to ``k * step`` offsets, moved next to their blocks, padded from ``n`` to ``n +
      k + e`` blocks (``e`` reaches an uncovered tail); buffer ``k * (n + k + e) * step``, large when ``n << k``;
    - over the ``n`` windows: each padded to ``L + step`` and read again in rows of ``L``, so window ``i`` lands
      shifted by ``i * step``; buffer ``n * (L + step)``, large when ``L >> size``.

    Non-overlapping windows (``step >= size``) need no sum: one pad of each window to ``step`` and of the tail. The
    dims before and after ``dim`` are merged, so every pad, copy and sum sees at most 5 dims (Gaudi's lazy
    ``constant_pad_nd`` falls back above 5); a non-contiguous gradient is copied at its own rank first. The result is
    a new contiguous tensor, as ATen's; booleans add as ``or``, as ATen's CPU kernel. ``dim`` is checked as ATen
    does; ``step`` must be positive (``Tensor.unfold`` rejects 0, so autograd never passes it). The derivative stays
    torch's autograd formula (``unfold``)."""
    ndim = len(input_sizes)
    dim = torch._prims_common.canonicalize_dim(ndim, dim)  # ATen's IndexError
    if step <= 0:
        raise RuntimeError(f"unfold_backward: step is {step} but must be > 0")
    if ndim == 0:  # a 0-d input unfolds to (size,) with size 0 or 1
        return grad.any().reshape(()) if grad.dtype == torch.bool else grad.sum(dtype=grad.dtype).reshape(())
    length = input_sizes[dim]
    if size == 0 or grad.numel() == 0:
        return grad.new_zeros(input_sizes)
    n = grad.shape[dim]
    branch, step, blocks, rows, _ = _unfold_backward_plan(n, input_sizes, dim, size, step)
    k = -(-size // step)
    outer, inner = math.prod(input_sizes[:dim]), math.prod(input_sizes[dim + 1:])
    g = grad.reshape(outer, n, inner, size).transpose(2, 3)  # (outer, n, size, inner), a view of a contiguous grad
    if branch == "pad":
        if blocks or step > size:
            g = _pad_dims(g, {1: blocks, 2: step - size})
        else:  # step == size and no tail: only a copy, never a view of ``grad``
            g = g.clone(memory_format=torch.contiguous_format)
        g = g.reshape(outer, (n + blocks) * step, inner).narrow(1, 0, length).contiguous()
        return g.reshape(input_sizes)
    if branch == "blocks":  # skew over the blocks
        if size < k * step:
            g = _pad_dims(g, {2: k * step - size})
        g = g.reshape(outer, n, k, step, inner).transpose(1, 2)  # (outer, k, n, step, inner), a view
        g = _pad_dims(g, {2: k + blocks})  # (outer, k, rows + 1, step, inner)
        g = g.reshape(outer, k * (rows + 1) * step * inner).narrow(1, 0, k * rows * step * inner)
        g = g.reshape(outer, k, rows * step, inner).narrow(2, 0, length)  # row q shifted by q blocks
    else:  # skew over the windows
        g = _pad_dims(g, {2: length + step - size})  # (outer, n, length + step, inner)
        g = g.reshape(outer, n * (length + step) * inner).narrow(1, 0, n * length * inner)
        g = g.reshape(outer, n, length, inner)  # window i shifted by i * step
    g = g.any(1) if g.dtype == torch.bool else g.sum(1, dtype=g.dtype)
    return g.reshape(input_sizes)


def _kernel(op, name, composite, takes, fallback):
    """``composite`` for the arguments ``takes`` accepts, else ``op``'s kernel at the ``fallback`` key."""

    def kernel(*args, **kwargs):
        if takes(*args, **kwargs):
            count(name)
            return composite(*args, **kwargs)
        return op._op_dk(fallback, *args, **kwargs)

    return kernel


def install_glu(device_type):
    """``glu`` on ``device_type`` as ``a * sigmoid(b)``, at the autograd and backend keys. Returns an undo.

    The backend registration replaces Habana's lazy kernel, whose graph for a permuted intermediate is
    rejected in inference mode (BrainModule's (32, 640, 1000) GLU along dim 1); cases the composite
    declines go to the structured kernel below it. It is registered only in lazy mode: the eager kernel
    computes the same call, so eager inference keeps Habana's path."""
    from . import tpc

    op = aten.glu.default
    below = DispatchKey.CompositeExplicitAutogradNonFunctional
    undos = [register_kernels(device_type, [
        (op, "Autograd{}", _kernel(op, "glu", glu, _glu_takes, DispatchKey.Autograd))])]
    if tpc.lazy_mode():
        undos.append(override_kernels(device_type, [
            (op, _kernel(op, "glu", glu, _glu_takes, below))]))
    return chain(undos)


def install_renorm(device_type):
    """``renorm`` on ``device_type`` as norm, scale factor and multiply. Returns an undo."""
    op = aten.renorm.default
    below = DispatchKey.CompositeExplicitAutogradNonFunctional  # what the HPU key resolved to: renorm.out, on the CPU
    return register_kernels(device_type, [
        (op, "Autograd{}", _kernel(op, "renorm", renorm, _renorm_takes, DispatchKey.Autograd)),
        (op, "{}", _kernel(op, "renorm", renorm, _renorm_takes, below)),
    ])


def install_embedding(device_type):
    """The embedding gradient of a 0-d index on ``device_type`` through the one-element index. Returns an undo."""
    op = aten.embedding_dense_backward.default
    return register_kernels(device_type, [(op, "Autograd{}", _kernel(
        op, "embedding_0d", embedding_dense_backward_one, _embedding_backward_takes, DispatchKey.Autograd))])


def install_roll(device_type):
    """Rolls by a multiple of the size on ``device_type`` as copies. Returns an undo. Below autograd (inference mode)
    they keep Habana's kernel, which hpu-ops does not override."""
    op = aten.roll.default
    return register_kernels(device_type, [(op, "Autograd{}", _kernel(
        op, "roll_zero", roll_unshifted, _roll_takes, DispatchKey.Autograd))])


def one_hot(input, num_classes=-1):
    """``aten.one_hot`` as an equality against ``arange``: Habana's HPU kernel rejects an inferred count
    ("Number of classes cannot be -1", GCNsNet's graph coarsening, eager, lazy and inference), and ATen's
    composite — which infers it — reaches that kernel, so delegating to it recurses once the backend kernel
    is replaced. Inferring or validating the count reads the input's min/max on the host, as ATen's composite
    does on CPU, with its errors (Habana's explicit-count kernel silently zero-fills out-of-range classes)."""
    if input.dtype != torch.int64:
        raise RuntimeError("one_hot is only applicable to index tensor of type LongTensor.")
    if input.numel():
        lo, hi = torch.stack([input.min(), input.max()]).tolist()
        if lo < 0:
            raise RuntimeError("Class values must be non-negative.")
        if num_classes < 0:
            num_classes = hi + 1
        elif hi >= num_classes:
            raise RuntimeError("Class values must be smaller than num_classes.")
    elif num_classes < 0:
        raise RuntimeError("Can not infer total number of classes from empty tensor.")
    return (torch.arange(num_classes, device=input.device) == input.unsqueeze(-1)).to(torch.int64)


def _one_hot_takes(input, num_classes=-1):
    return num_classes < 0 and not tracing(input)  # traced: the inferred count is data-dependent either way


def install_one_hot(device_type):
    """``one_hot`` with an inferred class count on ``device_type`` as an equality against ``arange``. Returns
    an undo. The backend kernel computes the equality for explicit counts too (Habana's kernel there is
    replaced, as in ``extrema``, and cannot be called any more); at the autograd key explicit counts go to
    ATen's composite, which computes them without reaching the replaced kernel."""
    op = aten.one_hot.default

    def backend(input, num_classes=-1):
        count("one_hot")
        return one_hot(input, num_classes)

    return chain([
        register_kernels(device_type, [
            # no Autograd kernel exists to decline to (the output is integral): explicit counts go to
            # ATen's composite, which computes them without reaching the replaced backend kernel
            (op, "Autograd{}", _kernel(op, "one_hot", one_hot, _one_hot_takes,
                                        DispatchKey.CompositeImplicitAutograd))]),
        override_kernels(device_type, [(op, backend)]),
    ])


def install_unfold(device_type):
    """``Tensor.unfold`` on ``device_type`` as ``as_strided``, at the backend key, below torch's own autograd kernel
    (whose backward is ``unfold_backward``). Habana's eager stack has its own kernel there, which is left alone, so
    only lazy mode takes it. Returns an undo."""
    op = aten.unfold.default

    def kernel(input, dimension, size, step):
        count("unfold")
        return unfold(input, dimension, size, step)

    return register_kernels(device_type, [(op, "{}", kernel)])


def _unfold_backward_on_cpu(op, grad, input_sizes, dim, size, step):
    """ATen's kernel on a CPU copy of ``grad``, the result moved back (a named hpu-ops CPU route). On the CPU stand-in
    the CPU key is the rewrite itself: inside the route it computes instead of routing again."""
    with _strict.route(op, grad, detail="unfold_backward above 2^31 - 1 elements on a CPU copy"):
        _ROUTE.depth = getattr(_ROUTE, "depth", 0) + 1
        try:
            return op(grad.cpu(), input_sizes, dim, size, step).to(grad.device)
        finally:
            _ROUTE.depth -= 1


def install_unfold_backward(device_type):
    """``aten.unfold_backward`` (the gradient of ``Tensor.unfold``) on ``device_type`` as pads, reshapes and one sum
    (``unfold_backward`` above), at the backend key, below torch's autograd kernel (whose derivative is ``unfold``).
    No Habana stack registers a kernel there: every call ran on the CPU, eager and lazy (#43). Calls whose gradient,
    result or buffer would pass ``UNFOLD_BACKWARD_MAX_ELEMENTS`` run ATen's kernel on a CPU copy instead (a named CPU
    route, reported by strict mode). Returns an undo."""
    op = aten.unfold_backward.default

    def kernel(grad_in, input_sizes, dim, size, step):
        if getattr(_ROUTE, "depth", 0) == 0 and unfold_backward_elements(
                grad_in, input_sizes, dim, size, step) > UNFOLD_BACKWARD_MAX_ELEMENTS:
            count("unfold_backward_cpu")
            return _unfold_backward_on_cpu(op, grad_in, input_sizes, dim, size, step)
        count("unfold_backward")
        return unfold_backward(grad_in, input_sizes, dim, size, step)

    return register_kernels(device_type, [(op, "{}", kernel)])
