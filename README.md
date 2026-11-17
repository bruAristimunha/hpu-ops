# hpu-ops

Fixes for what Intel Gaudi's PyTorch stack lacks or gets wrong. Each one is made at its root, or with Habana's own switch where one exists. It is the sibling of [hpu-complex](https://github.com/bruAristimunha/hpu-complex), which adds complex tensors and `torch.fft`.

Libraries stay device-agnostic. When something works on CUDA, CPU and AMD but not on Gaudi, it is fixed here, not in the library.

## Use

Export the attention setting for your Habana bridge before Python starts. It is read when Habana's library loads.

| bridge | fused attention with masks |
|---|---|
| 1.21 | `export PT_HPU_USE_OVERRIDE_ATEN_SDPA=1` |
| 1.22 | `export PT_HPU_USE_ATEN_SDPA=0` |
| 1.23+ (torch ≥ 2.8) | `with torch.nn.attention.sdpa_kernel([SDPBackend.OVERRIDEABLE]):` |

Then, before building or compiling the model:

```python
import hpu_ops
hpu_ops.enable()
```

## What it fixes

| gap on Gaudi | root cause | fix |
|---|---|---|
| Masked attention runs op by op (REVE shape: 27.1 ms, 63.6 GiB peak, eager). | The math path is the default on HPU. | Habana's setting above. Every mask (boolean, additive, none) then runs `hpu.sdpa_fwd` in eager mode, in 4.9 ms with a 9.8 GiB peak. Compiled time drops from 12.3 to 4.9 ms. |
| Compiled linear upsampling is 20x slower than eager (CodeBrain). | `torch.compile` traces `upsample_linear1d.vec` through PyTorch's Python decomposition, i.e. gather/scatter index ops, and their scatter backward is slow on Gaudi. Unfixed in bridge 1.24.2. | `enable()` replaces that decomposition, the way Habana's `override_composite_ops` does for other ops. HPU tensors with static shapes get the native op, which Habana compiles with its own kernels: forward + backward goes from 56.9 to 3.4 ms (2.8 ms eager). |
| einsum graphs fail to compile (BIOT: `Can not compile graph`). | `pass_remove_unnecessary_bmm_view` removes views that are not pure batch merges. Unfixed in bridge 1.24.2. | `enable()` sets Habana's `PT_HPU_DISABLE_pass_remove_unnecessary_bmm_view`, Intel's own workaround (vllm-gaudi #1102). |
| LaBraM fails to compile (`used before defined`). | `batch_as_strided` puts a `getitem` before the node it reads. Fixed in bridge 1.22. | `enable()` sets `PT_HPU_DISABLE_pass_batch_as_strided_groups`. |

The pass switches are read at every compile. A non-empty setting in your environment wins.

## Measured on Gaudi2 (habana 1.21.4 / torch 2.6.0)

One card, seconds per training step on synthetic inputs (3 micro-batches of 100 samples; REVE 6 of 50).

- The plugin route column uses the attention setting plus `hpu_ops.enable()`.
- The last column patches braindecode for each gap instead.
- Both columns share two patches: the masked group norm of braindecode#1575, and real-DFT patch spectra for CBraMod and CodeBrain, which hpu-complex covers without patches.

| model | plugin route | patched braindecode |
|---|---|---|
| CodeBrain, compiled | 3.240 | 3.228 |
| REVE, eager | 3.215 | 3.228 |
| REVE, compiled | 1.990 | 1.996 |
| CBraMod, compiled | 0.858 | 0.955 |
| LaBraM, compiled | 1.808* | 1.820 |
| BIOT, compiled | 0.864 | 0.924 |

\* Measured with an earlier corrected `batch_as_strided` pass in place of the switch. Without the attention setting, the two measured the same (1.820 s/step).

Turning the two passes off costs at most 1.7% of a step:

| model | pass on | pass off |
|---|---|---|
| STEEGFormer | 0.973 | 0.978 |
| EEGPT | 0.618 | 0.629 |

CBraMod, REVE and BIOT show no change. Habana's scatter-add flags (`PT_HPU_USE_UNSORTED_SCATTER_ADD`, `PT_HPU_SORT_INDEX_IN_SCATTER_ADD`) do not help compiled upsampling: it stays at 57–64 ms.

## Tests

```bash
python -m pytest tests/test_cpu.py   # the CPU stands in for HPU
python tests/run_on_hpu.py           # on a Gaudi card, with the attention setting exported; exits 1 on any failure
```

`test_cpu.py` checks three things:
- compiled graphs contain the native op and no `index_put`;
- results match eager;
- a new batch size recompiles through PyTorch's decomposition instead of crashing.

`run_on_hpu.py` compares outputs and input gradients against torch on the CPU, eager and compiled. It covers:
- SDPA with boolean, additive and no mask;
- `nn.MultiheadAttention`;
- CodeBrain's upsampling at two batch sizes, and long upsampling;
- braindecode's BIOT attention block.

It also checks that eager attention runs `hpu.sdpa_fwd` and that compiled upsampling took the native op.

## Limits

- Only 1-D linear upsampling is rerouted. Nearest, nearest-exact, and 2-D and 3-D modes keep PyTorch's decomposition.
- Shapes that `torch.compile` treats as symbolic also keep the decomposition. The native backward has no meta kernel for them. After a new batch size, dynamo marks dimensions dynamic, so those graphs fall back.
- On bridge 1.21, Habana's attention override keeps the full attention-probability tensor for the backward (no recompute).
- An attention mask gets no gradient; key-padding masks need none.
- Fully masked query rows (a sample with no real key) may differ from torch's zeros.
