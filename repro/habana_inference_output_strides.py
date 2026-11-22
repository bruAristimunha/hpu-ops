"""Habana bug (Gaudi2, bridge 1.21.4, PT_HPU_LAZY_MODE=0): ``torch.compile(..., dynamic=True)`` under
``torch.inference_mode()`` raises ``KeyError: 'output_strides'`` when an eager-placed op (``linalg.eigh``) is
followed by a ``get_attr`` constant tensor consumed in a compiled partition (e.g. ``spd_learn`` ``GyroAtt``,
``MAtt``, and ``TensorCSPNet`` via ``ReEig``).

    PT_HPU_LAZY_MODE=0 python3 repro/habana_inference_output_strides.py

Root cause in ``habana_frameworks/torch/dynamo/compile_backend/``:
- ``_helpers/helpers.py:204-205`` (``fill_propagated_tensor_metadata_to_node``) returns early on
  ``node.op == "get_attr"`` without populating ``node.meta["output_strides"]``.
- ``passes.py:1172`` (``fuse_partitions``) turns ``_tensor_constant0`` into a ``placeholder`` of the compiled
  partition after eager ``_linalg_eigh``, preserving its ``output_strides``-less metadata.
- ``recipe_compiler.py:133`` (``get_input_symbolic``, under ``dynamic=True``) indexes
  ``input_node.meta["output_strides"]`` unconditionally and raises ``KeyError: 'output_strides'``.
"""
import habana_frameworks.torch.core  # noqa: F401
import habana_frameworks.torch.hpu as hthpu
import torch
import torch.nn as nn


class ReproOutputStrides(nn.Module):
    def forward(self, x):
        w, v = torch.linalg.eigh(x)
        t = torch.tensor(1e-4, device=x.device, dtype=x.dtype)
        return w.clamp(min=t) @ v.mT


if __name__ == "__main__":
    torch.manual_seed(0)
    m = ReproOutputStrides().eval().to("hpu")
    c = torch.compile(m, backend="hpu_backend", dynamic=True)
    a = torch.randn(4, 6, 6, device="hpu")
    a = a @ a.mT + torch.eye(6, device="hpu")
    with torch.inference_mode():
        try:
            out = c(a)
            hthpu.synchronize()
            print(f"OK shape={tuple(out.shape)}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"REPRODUCED {type(e).__name__}: {e}", flush=True)
