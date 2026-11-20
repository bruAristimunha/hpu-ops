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
- ``linalg.solve_triangular`` as a matmul with a block-recursive inverse (``linalg``): it has no HPU kernel and ran
  on the CPU; float64 stays on the CPU, as Gaudi2 computes it at float32 precision (hpu-ops#16). Cholesky's and QR's
  backward use it.
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

from . import composite, conv, linalg, serialization, upsample
from ._dispatch import _STATE, chain

__all__ = ["enable"]

BROKEN_PASSES = ("pass_remove_unnecessary_bmm_view", "pass_batch_as_strided_groups", "pass_weight_permutation")
FIXES = (upsample.install_linear, upsample.install_linear_forward, upsample.install_nearest, conv.install,
         conv.install_weight_grad, composite.install_glu, composite.install_renorm, composite.install_embedding,
         composite.install_roll, linalg.install_solve_triangular, serialization.install)
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
    for name in BROKEN_PASSES:
        if not os.environ.get(f"PT_HPU_DISABLE_{name}"):  # unset or empty; an explicit user setting wins
            os.environ[f"PT_HPU_DISABLE_{name}"] = "True"
    _STATE["undo"] = _install("hpu")
    if not _attention_fused():  # last: under -W error the fixes are already in
        warnings.warn(ATTENTION_HINT, stacklevel=2)
