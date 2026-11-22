"""hpu_ops: what Intel Gaudi's PyTorch stack lacks or gets wrong, fixed at the root or with Habana's own settings.

Model code and torch stay unmodified. Only HPU tensors are rewritten; other devices run as before. The rewrites live
in the dispatcher (see ``_dispatch``), except ``torch.save``'s, which sits where Habana puts its own in lazy mode,
at the tensor's pickling:

- 1-D upsampling (``upsample``). Under ``torch.compile``, linear keeps Gaudi's native op instead of torch's
  gather/scatter decomposition (CodeBrain: 56.9 to 3.4 ms forward + backward). Linear's forward by a factor other
  than a power of two comes from explicit indices, eager and compiled: Gaudi's kernel drifts with the length there
  (NeuroPose: 34% off). Nearest by an integer factor repeats each sample, eager and compiled: Gaudi's nearest kernel
  fails after a BatchNorm1d (USleep).
- ``padding="valid"`` convolutions under compile (``conv``): passed as full zero lists, which Habana's compiled
  convolution needs (ATCNet, EEGInceptionERP, EEGITNet).
- The weight gradient of one-channel convolutions with a stride (``conv``): Gaudi's compiled
  ``convolution_backward`` returns about half of it plus an unrelated part (EEGDINO, CodeBrain, LaBraM).
- Large-kernel convolutions that fail natively (``conv``): Gaudi2's MME AGU rejects most spatial kernel sizes
  ``> 256`` (``setAssociatedDimAndSize`` in ``STaRNet(64)`` and ``TensorCSPNet(22, dims=(22, 11, 11, 11))``), and in
  lazy mode a kernel of 256 over 1000 fails too while kernels 129..255 can get a wrong input gradient. Those are split
  into ``<= 256`` slices (``<= 128`` in lazy mode); the one class measured to run natively (one input channel, wide
  only in its last dim, e.g. TensorCSPNet's default ``(4, 4356)`` stride ``(1, 484)``) keeps the native op.
- ``group_norm`` with a symbolic size under compile (``norm``, #42) as a reshape into groups, ``var_mean`` and the
  affine terms: with the default dynamic shapes a second batch size traced ``native_group_norm`` with the batch as
  a SymInt argument, and Habana's backend failed ("aten::IntImplicit isn't registered in KernelRegistry!":
  FNIRSCNN1D, SAMBA, SignalJEPA_PostLocal now compile). Static shapes and eager keep Gaudi's kernel.
- Symbolic ``expand`` and ``int64`` ``constant_pad_nd`` under compile (``dynshape``, #42): resolving ``-1`` dims in
  symbolic ``aten.expand.default`` fixes Habana's ``BroadcastOperator`` crash, and rewriting ``int64``
  ``aten.constant_pad_nd.default`` avoids ``hpu.constant_pad_nd_ds`` ``synStatus 26`` (``NeuroRVQ``,
  ``NeuroRVQTokenizer``, ``DIVER1``, and ``LaBraM`` now compile and run across ``16 <-> 128``). Static ``expand``,
  non-``int64`` dtypes, and eager calls keep Gaudi's path.
- ``glu`` as ``a * sigmoid(b)`` (``composite``): Gaudi's compiled glu fails (BrainModule).
- ``renorm`` as norm, scale and multiply (``composite``): it has no HPU kernel and ran on the CPU (SSTDPN).
- The ``embedding`` gradient of a 0-d index through the one-element index (``composite``): Gaudi's
  ``embedding_dense_backward`` sends it to row 0, eager and compiled (BIOT's channel tokens).
- ``roll`` by a multiple of the size as a copy (``composite``): Gaudi's eager kernel reads a strided input of such a
  roll wrongly (EMG2QwertyNet's input gradient).
- ``Tensor.unfold`` as ``as_strided`` with ATen's sizes and strides (``composite``): in lazy mode it had no HPU kernel,
  ran on the CPU and returned a copy, not a view. hpu-ops' own one-channel conv weight gradient uses it (TensorCSPNet:
  2 fallbacks per lazy step). Eager keeps Habana's kernel.
- Advanced indexing by integer tensors, ``X[..., rows, cols]``, as one ``index_select`` of the indexed dims
  flattened (``indexing``): with CPU index tensors Gaudi's eager kernel ran on the CPU, and in lazy mode its
  backward, the accumulating ``_index_put_impl_`` with indices that do not lead, returns wrong sums from batch 2 to
  8 on; ``index_select``'s backward is right (spd_learn's ``vech`` and ``vec_to_sym``: in lazy mode every gradient
  before them was wrong in 9 SPD models). ``X[:, mask]`` through its ``nonzero`` into the same ``index_select``.
  A boolean-mask put of one value as ``where`` (``indexing``): in lazy mode, in one graph with index puts and
  reads, Habana's launch fails (SPNN).
- ``F.unfold`` and ``F.fold`` (``im2col``): ``aten::im2col`` and ``aten::col2im`` have no HPU kernel and ran on the
  CPU, eager and lazy (SSGCNet, MAtt, GyroAtt, SPDMetaFormer, CorAtt). Non-overlapping 4-D patches as reshapes,
  others as ``index_select`` and ``index_add``.
- ``linalg.solve_triangular`` and ``cholesky_solve`` of float32 matrices with n <= 64 run through a TPC substitution
  kernel (``tpc.tri_solve``, two substitutions for ``cholesky_solve``): they have no HPU kernel and ran on the CPU.
  Substitution keeps CPU float32's accuracy on ill-conditioned triangles. Cholesky's and QR's backward use them.
- ``linalg.inv``, ``det`` and ``slogdet`` from the masked LU and triangular matmuls (``linalg``), as ``linalg.solve``.
- On Gaudi2 those composites were measured 3-25x slower per call than Habana's CPU fallback (one launch per op), so
  ``linalg.solve``, ``inv``, ``det`` and ``slogdet`` of float32 matrices
  with n <= 64 run through a TPC Gauss-Jordan kernel with partial pivoting (``tpc``), one matrix per index-space
  member. Eager, batch 32, n = 22, mean ms per call in two processes: solve 2.9 / 1.3 against 6.1 / 2.5 for the
  fallback, inv 2.0 / 1.2 against 4.3 / 2.0, det 0.67 / 0.69 against 2.7 / 1.2. Lazy, per call, within the card's
  noise of the fallback (slogdet slower); the card's noise is about 2x between processes
  (``tests/linalg_percall.py``). Habana reads the kernel path when the device initialises, so ``enable()``
  must run before the first HPU op, else the kernel is off (one warning). The kernels
  are off too on a device that is not a Gaudi2 and with ``HPU_OPS_TPC=0`` in the environment. Calls the kernel does
  not take (n > 64, float64, kernel off): ``inv`` runs through hpu-ops' own CPU route, which is not Habana's
  fallback; ``solve``, ``det`` and ``slogdet`` register only at the autograd key and reach Habana's own kernel and
  its CPU fallback (bridge-counted).
- ``linalg.eigh`` and ``eigvalsh`` of float32 matrices with n <= 32 run through a TPC one-sided Jacobi kernel
  (``tpc``, ``eigh``): they fell back to the CPU in 21 of 21 SPD models. Eager, batch 32, n = 22: 0.81 / 2.6 ms
  in two processes against 3.9 / 2.3 for the fallback (lazy 0.54 / 1.0 against 2.4 / 2.7); batch 1024 3.7 ms against
  52 ms (older run); within about
  3.8x of CPU float32 accuracy, also on repeated eigenvalues, condition 1e6 and inputs scaled by 1e30.
  Without the kernel, or past n = 32, the call is declined and runs through hpu-ops' own CPU route, not Habana's
  fallback: the parallel-Jacobi composite of ``eigh`` was measured slower than the fallback on Gaudi2, so it only
  serves the CPU stand-in.
- ``quantile`` / ``nanquantile`` as sort, gather and lerp (``quantile``): ATen's composite resizes an empty ``out``
  tensor, which aborts lazy mode ("Trying to resize tensor with non-existing TensorImpl").
- NaN through the max/min family (``extrema``): Gaudi's ``amax`` / ``amin``, ``max`` / ``min`` (whole, dim, binary),
  ``median``, ``aminmax``, ``argmax`` / ``argmin``, ``maximum`` / ``minimum``, ``fmax`` / ``fmin``, ``clamp`` with
  tensor bounds, the inf-norms and ``max_pool2d`` / ``max_pool3d`` drop NaN, eager, lazy and compiled
  (``amax([nan, 1, 2])`` is ``-inf``). Each runs Gaudi's kernel and puts NaN where torch has it, with torch's
  indices (the first NaN) and gradients, in inference mode too (at the ``HPU`` key, through the ops' ``.out``
  overloads), for ``out=`` calls and in-place ``clamp_``; in 16-bit dtypes ``argmax`` / ``max(dim)`` of a row of ``-inf`` no longer
  return the out-of-range index ``n``. ``softmax``, ``log_softmax``, ``logsumexp``, ``cummax`` and the sums were
  right and stay native. ``sort`` / ``argsort`` / ``topk`` / ``kthvalue`` / ``msort`` in torch's order with NaN, and
  with indices in range where Gaudi's float sort returns out-of-range ones (NaN, and any real infinity at an n that
  is not a power of two), eager, lazy and compiled; ``HPU_OPS_EXACT_SORT=0`` keeps Gaudi's sort. hpu-ops' own
  eigenvalue / singular-value sorts keep Gaudi's sort, so the SPD steps keep main's cost.
- ``histc`` with a bound given as a Python float, or more than 300 bins (``small``): Gaudi's kernel reads its bounds
  as ints and aborts the process on a float one, and does not compile from about 350 bins. The CPU kernel's bin
  edges are found on the host, the card only compares (bucketize) and counts; bit for bit ATen's one-thread result,
  eager, lazy, compiled and in inference mode.
- ``bincount`` of 1-D integer inputs of 65536 to 2**24 elements with at most 4096 bins and no weights through one
  ``aminmax`` read and ``histc`` in chunks of 256 bins (``small``, #21): Gaudi's kernel took 4.7-427 ms at 409600
  elements (64 to 4096 bins), 2-43x longer.
- Warnings for the 8-byte dtypes in eager mode (``wide``), at the copy to the card: float64 views with a storage
  offset read and write at twice the offset, int64 values past int32 are computed on their low 32 bits. Neither is
  fixable at the dispatcher at a cost a model would accept (see the README, "float64").
- ``torch.save`` through a tensor-level host copy (``serialization``): Habana's storage copy fails on conv weights
  that an optimizer step left in a permuted layout. torch pickles XLA tensors this way, and Habana HPU tensors in
  lazy mode.
- Three Habana compile passes break valid graphs. ``enable()`` turns them off through Habana's own
  ``PT_HPU_DISABLE_<pass>`` switch, read at every compile:
  - ``pass_remove_unnecessary_bmm_view`` strips einsum's views (BIOT: "Can not compile graph"). Unfixed through
    1.24.2; Intel uses the same switch in vllm-gaudi PR #1102.
  - ``pass_batch_as_strided_groups`` puts a node before its input (LaBraM: "used before defined"). Fixed in 1.22.
  - ``pass_weight_permutation`` inserts a weight permutation before 2-D and 3-D convolutions; some graphs then fail
    ("create tensor invalid permutation 4 2": FBCNet, FBMSNet, at times EEGNet and Deep4Net). Off, CBraMod steps
    2% faster and CodeBrain the same.

Strict mode (``_strict``): ``HPU_OPS_STRICT=1`` or ``with hpu_ops.strict():`` raises ``CPUExecutionError`` on every CPU
execution an HPU step reaches that hpu-ops can see (Habana's CPU fallback per op, hpu-ops' own CPU routes, hpu-complex's
CPU and host-memory compute, host syncs and copies); ``HPU_OPS_STRICT=report`` counts them and prints a table at exit.

In lazy mode (``PT_HPU_LAZY_MODE=1``), ``torch.compile(model, backend="hpu_backend")`` called after ``enable()`` warns
once and returns ``model`` uncompiled (``lazy_compile``; ``HPU_OPS_LAZY_COMPILE=raise``: a one-line error instead):
Habana supports the backend in eager mode only (1.21.4 support matrix), and in lazy mode it fails deep inside (an
undefined symbol in the recipe compiler, ``torch.ops.hpu.weight_permutation`` missing). Run with
``PT_HPU_LAZY_MODE=0`` to compile.

Masked attention needs a Habana setting that is read when its library loads, so it must be exported before Python
starts. The setting is per bridge version, listed in the README; ``enable()`` warns when it did not take effect.
"""
import os
import warnings

import torch

from . import (composite, conv, dynshape, eigh, extrema, im2col, indexing, lazy_compile, linalg, norm,  # noqa: F401
               quantile, serialization, small, tpc, upsample, wide)
from ._dispatch import _STATE, chain
from ._strict import CPUExecutionError, strict  # noqa: F401
from ._strict import check as strict_check, format_report as strict_format_report  # noqa: F401
from ._strict import report as strict_report, reset as strict_reset  # noqa: F401

__all__ = ["enable", "strict", "CPUExecutionError", "strict_check", "strict_report", "strict_reset",
           "strict_format_report"]

BROKEN_PASSES = ("pass_remove_unnecessary_bmm_view", "pass_batch_as_strided_groups", "pass_weight_permutation")
FIXES = (upsample.install_linear, upsample.install_linear_forward, upsample.install_nearest, conv.install,
         conv.install_weight_grad, conv.install_large_kernel, composite.install_glu, composite.install_renorm,
         composite.install_embedding, composite.install_roll, composite.install_unfold,
         composite.install_unfold_backward, norm.install_group_norm, dynshape.install_expand,
         dynshape.install_pad_int64, indexing.install_index,
         indexing.install_mask_fill, im2col.install,
         linalg.install_solve_triangular, linalg.install_cholesky_solve,
         linalg.install_solve, linalg.install_inv_det, eigh.install_eigh,
         linalg.install_svd, linalg.install_pinv, linalg.install_lstsq, linalg.install_matrix_rank,
         linalg.install_matrix_exp, linalg.install_householder_product, linalg.install_lu, quantile.install,
         extrema.install, small.install_bincount, small.install_histc,
         wide.install_copy_checks, serialization.install)
ATTENTION_HINT = ("Gaudi's fused attention kernel is not registered, so F.scaled_dot_product_attention with a mask "
                  "runs op by op (about 5x slower, with a larger peak memory). If the model uses attention, export "
                  "the setting for your bridge before Python starts: PT_HPU_USE_OVERRIDE_ATEN_SDPA=1 (1.21) or "
                  "PT_HPU_USE_ATEN_SDPA=0 (1.22).")


def _install(device_type):
    """Every rewrite, for tensors of ``device_type`` (the tests pass "cpu" in the role of "hpu"). Returns an undo."""
    return chain([fix(device_type) for fix in FIXES])


def _attention_fused(device_type="hpu"):
    """Whether Habana's attention setting took effect, read from the dispatcher, without touching the device. With
    the setting, bridges 1.21 and 1.22 register their fused kernel for ``scaled_dot_product_attention`` at the
    device's autograd key when the library loads. Bridges 1.23+ always register the overrideable backend's kernel,
    which ``sdpa_kernel([SDPBackend.OVERRIDEABLE])`` picks per call, so there the answer is always True."""
    key = f"Autograd{device_type.upper()}"
    return any(torch._C._dispatch_has_kernel_for_dispatch_key(f"aten::{op}", key)
               for op in ("scaled_dot_product_attention", "_scaled_dot_product_fused_attention_overrideable"))


def _strict_from_env():
    from . import _strict

    if _strict.from_env() == "report":
        warnings.warn("hpu_ops: HPU_OPS_STRICT=report counts every CPU execution and prints a table at exit",
                      stacklevel=3)


def _lazy():
    """Whether Habana's bridge runs in lazy mode (read after ``habana_frameworks.torch.core`` loads): its own
    ``is_lazy``; on a bridge without it, ``PT_HPU_LAZY_MODE == "1"`` (unset counts as eager, 1.21.4's default:
    its bridge configuration prints ``PT_HPU_LAZY_MODE = 0`` when the variable is unset)."""
    try:
        from habana_frameworks.torch.utils.internal import is_lazy
    except ImportError:
        return os.environ.get("PT_HPU_LAZY_MODE", "0") == "1"
    return is_lazy()


def enable():
    """Install the rewrites and turn off the broken passes. Idempotent; a no-op off a Gaudi stack. With
    ``HPU_OPS_STRICT=1`` (or ``report``) in the environment, strict mode for the whole process (``hpu_ops._strict``)."""
    if _STATE.get("undo"):
        return
    try:
        import habana_frameworks.torch.core  # noqa: F401  (Habana's kernels first: they are never overridden)
    except ImportError:
        return
    with warnings.catch_warnings(record=True) as path_warnings:  # warned after the fixes are in (-W error)
        warnings.simplefilter("always")
        tpc.install_kernel_path()  # before the first HPU op: Habana reads the kernel path when the device initialises
    for name in BROKEN_PASSES:
        if not os.environ.get(f"PT_HPU_DISABLE_{name}"):  # unset or empty; an explicit user setting wins
            os.environ[f"PT_HPU_DISABLE_{name}"] = "True"
    undo = _install("hpu")
    if _lazy():  # Habana's compile backend does not run in lazy mode: uncompiled with one warning (or raise)
        try:
            undo = chain([undo, lazy_compile.install(lazy_compile.NAME)])
        except Exception:
            undo()  # no half-enabled state: a later enable() starts clean
            raise

    def undo_once():  # clears the state, so enable() after the undo installs again
        if _STATE.get("undo") is undo_once:
            del _STATE["undo"]
            undo()

    _STATE["undo"] = undo_once
    _strict_from_env()
    for caught in path_warnings:  # last, as the attention hint: under -W error the fixes are already in
        warnings.warn(caught.message, stacklevel=2)
    if not _attention_fused():
        warnings.warn(ATTENTION_HINT, stacklevel=2)
