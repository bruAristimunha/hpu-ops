"""Habana bug (Gaudi2, bridge 1.21.4, PT_HPU_LAZY_MODE=0), no plugin: the training step of two LaBraM transformer
blocks compiled with the default dynamic shapes crashes at the second batch size, in an op Habana launches eagerly:

    RuntimeError: [Rank:0] FATAL ERROR :: MODULE:PT_EAGER HabanaLaunchOpPT Run returned exception....
    Reshape doesnt support change in number of elements: [128, 10, 200] Size of output: [1280, 800]
    [Rank:0] Habana exception raised from AllocateAndAddSynapseNode at tensor_shape_kernels.cpp:536

    PT_HPU_LAZY_MODE=0 python3 repro/issue42_backward_reshape_dynamic.py     (needs braindecode)

The blocks are braindecode's (``NeuroRVQ(...).blocks[:2]``: pre-norm, qkv attention with a q/v bias and q/k LayerNorm,
GELU MLP 200 -> 800 -> 200, layer scale); one block alone trains. They hold no group norm, no ``expand`` and no int64
pad, so none of hpu-ops' dynamic-shape rewrites is involved, and hpu-ops is not imported. The full models fail the
same way with and without hpu-ops (the plugin's rewrites written into the model source instead): NeuroRVQ
(``[512, 10, 200]`` to ``[5120, 800]``), LaBraM (``[128, 641, 200]`` to ``[82048, 800]``), and DIVER1 (``The size of
tensor a (1536) must match the size of tensor b (12)``); their inference at batch 16 then 128 is right. Habana's three
passes that hpu-ops turns off are turned off here too (their own switches), as with ``hpu_ops.enable()``.

Both settings that avoid the Conv2d bug of ``issue42_conv2d_dynamic_recipe.py`` avoid this one:
``PT_HPU_FORCE_STATIC_COMPILE=1`` and ``torch.compile(..., dynamic=False)``. Measured with this script:

    default compile    batch 16: ok   batch 128: RuntimeError ... Reshape doesnt support change in number of elements:
                                                 [128, 10, 200] Size of output: [1280, 800]
    FORCE_STATIC=1     batch 16: ok   batch 128: ok
    dynamic=False      batch 16: ok   batch 128: ok
"""
import os
import subprocess
import sys

for _name in ("pass_remove_unnecessary_bmm_view", "pass_batch_as_strided_groups", "pass_weight_permutation"):
    os.environ.setdefault(f"PT_HPU_DISABLE_{_name}", "True")

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402


def train_steps(**compile_kwargs):
    import habana_frameworks.torch.core  # noqa: F401
    import habana_frameworks.torch.hpu as hthpu
    from braindecode.models import NeuroRVQ

    torch.manual_seed(0)
    chs_info = [{"ch_name": name, "kind": "eeg"} for name in ("F3", "F4", "Cz")]
    model = NeuroRVQ(n_chans=3, n_outputs=2, n_times=600, sfreq=200.0, chs_info=chs_info)
    blocks = nn.Sequential(*model.blocks[:2]).train().to("hpu")
    for m in blocks.modules():
        if isinstance(m, nn.Dropout):
            m.p = 0.0  # deterministic; the crash does not need dropout
    compiled = torch.compile(blocks, backend="hpu_backend", **compile_kwargs)
    out = []
    for b in (16, 128):
        try:
            compiled(torch.randn(b, 10, 200, device="hpu")).square().mean().backward()
            hthpu.synchronize()
            out.append(f"batch {b}: ok")
        except RuntimeError as e:
            lines = str(e).splitlines()
            out.append(f"batch {b}: RuntimeError ... {next((ln for ln in lines if 'Reshape' in ln), lines[0])[:110]}")
    return "   ".join(out)


if __name__ == "__main__":
    if len(sys.argv) > 1:  # one setting per process: PT_HPU_FORCE_STATIC_COMPILE is read when Habana loads
        print(f"{sys.argv[1]:17s}  {train_steps(**({'dynamic': False} if sys.argv[1] == 'dynamic=False' else {}))}",
              flush=True)
        sys.exit()
    for setting, env in (("default compile", {}), ("FORCE_STATIC=1", {"PT_HPU_FORCE_STATIC_COMPILE": "1"}),
                         ("dynamic=False", {})):
        subprocess.run([sys.executable, __file__, setting], env={**os.environ, **env}, check=False)
