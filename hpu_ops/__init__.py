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
- ``glu`` as ``a * sigmoid(b)`` (``composite``): Gaudi's compiled glu fails (BrainModule).
- ``renorm`` as norm, scale and multiply (``composite``): it has no HPU kernel and ran on the CPU (SSTDPN).
- The ``embedding`` gradient of a 0-d index through the one-element index (``composite``): Gaudi's
  ``embedding_dense_backward`` sends it to row 0, eager and compiled (BIOT's channel tokens).
- ``roll`` by a multiple of the size as a copy (``composite``): Gaudi's eager kernel reads a strided input of such a
  roll wrongly (EMG2QwertyNet's input gradient).
- Advanced indexing whose indexed dims do not lead, ``X[..., rows, cols]``, as one ``index_select`` of the indexed
  dims flattened (``indexing``): its backward, Gaudi's accumulating ``_index_put_impl_`` with such indices, returns
  wrong sums in lazy mode from batch 2 to 8 on; ``index_select``'s backward is right (spd_learn's ``vech`` and
  ``vec_to_sym``: in lazy mode every gradient before them was wrong in 9 SPD models). A boolean-mask put of one value
  as ``where`` (``indexing``): in lazy mode, in one graph with index puts and reads, Habana's launch fails (SPNN).
- ``linalg.solve_triangular`` and ``cholesky_solve`` (``linalg.install_solve_triangular``, ``install_cholesky_solve``)
  are NOT installed by ``enable()``: pivoted Gauss-Jordan on a triangle loses accuracy (no correct digits at n = 32
  when ill-conditioned) and ``cholesky_solve`` squares the condition number. They run through Habana's fallback, as on
  ``origin/main``, until a substitution kernel lands. Cholesky's and QR's backward use them, so they are unchanged too.
- ``linalg.inv``, ``det`` and ``slogdet`` from the masked LU and triangular matmuls (``linalg``), as ``linalg.solve``.
- On Gaudi2 those composites were measured 3-25x slower per call than Habana's CPU fallback (one launch per op), so
  ``linalg.solve``, ``inv``, ``det`` and ``slogdet`` of float32 matrices
  with n <= 32 run through a TPC Gauss-Jordan kernel with partial pivoting (``tpc``), one matrix per index-space
  member. Eager, batch 32, n = 22: solve 2.7 ms against about 4.5 ms for the fallback, inv 1.8 against 4.6, det 0.82
  against 2.7 (lazy: 0.57, 0.48, 0.51 against about 2.2, 2.2, 1.4). Habana reads the kernel path when the device
  initialises, so ``enable()`` must run before the first HPU op, else the kernel is off (one warning). The kernels
  are off too on a device that is not a Gaudi2 and with ``HPU_OPS_TPC=0`` in the environment. Calls the kernel does
  not take (n > 32, kernel off) run through hpu-ops' own CPU route, which is not Habana's fallback.
- ``linalg.eigh`` and ``eigvalsh`` of float32 matrices with n <= 32 run through a TPC one-sided Jacobi kernel
  (``tpc``, ``eigh``): they fell back to the CPU in 21 of 21 SPD models. Eager, batch 32, n = 22: 1.27 ms
  against about 3.7 ms for the fallback (lazy 0.80 against about 1.4); batch 1024 3.7 ms against 52 ms; within about
  3.8x of CPU float32 accuracy, also on repeated eigenvalues, condition 1e6 and inputs scaled by 1e30.
  Without the kernel, or past n = 32, the call is declined and runs through hpu-ops' own CPU route, not Habana's
  fallback: the parallel-Jacobi composite of ``eigh`` was measured slower than the fallback on Gaudi2, so it only
  serves the CPU stand-in.
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

Masked attention needs a Habana setting that is read when its library loads, so it must be exported before Python
starts. The setting is per bridge version, listed in the README; ``enable()`` warns when it did not take effect.
"""
import os
import warnings

import torch

from . import composite, conv, eigh, indexing, linalg, serialization, tpc, upsample  # noqa: F401
from ._dispatch import _STATE, chain

__all__ = ["enable"]

BROKEN_PASSES = ("pass_remove_unnecessary_bmm_view", "pass_batch_as_strided_groups", "pass_weight_permutation")
FIXES = (upsample.install_linear, upsample.install_linear_forward, upsample.install_nearest, conv.install,
         conv.install_weight_grad, composite.install_glu, composite.install_renorm, composite.install_embedding,
         composite.install_roll, indexing.install_index, indexing.install_mask_fill, linalg.install_solve,
         linalg.install_inv_det, eigh.install_eigh, serialization.install)
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


def enable():
    """Install the rewrites and turn off the broken passes. Idempotent; a no-op off a Gaudi stack."""
    if _STATE.get("undo"):
        return
    try:
        import habana_frameworks.torch.core  # noqa: F401  (Habana's kernels first: they are never overridden)
    except ImportError:
        return
    tpc.install_kernel_path()  # before the first HPU op: Habana reads the kernel path when the device initialises
    for name in BROKEN_PASSES:
        if not os.environ.get(f"PT_HPU_DISABLE_{name}"):  # unset or empty; an explicit user setting wins
            os.environ[f"PT_HPU_DISABLE_{name}"] = "True"
    undo = _install("hpu")

    def undo_once():  # clears the state, so enable() after the undo installs again
        if _STATE.get("undo") is undo_once:
            del _STATE["undo"]
            undo()

    _STATE["undo"] = undo_once
    if not _attention_fused():  # last: under -W error the fixes are already in
        warnings.warn(ATTENTION_HINT, stacklevel=2)
