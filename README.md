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

On 1.21 and 1.22, `enable()` warns when the setting did not take effect: Habana's fused kernel is then not registered in the dispatcher.

Compile with `dynamic=False`, or keep one batch size (`drop_last=True`). A new batch size otherwise makes the shapes symbolic, and linear upsampling falls back to the slow decomposition, with a warning.

## What it fixes

Only HPU tensors are rewritten; other devices run the original ops. The rewrites sit in PyTorch's dispatcher, in one of two ways:
- **compile** — torch's Python kernel of the op is replaced. Only `torch.compile`'s tracers read it, so eager calls are unchanged.
- **eager + compile** — a Python kernel is registered at the `AutogradHPU` key (and `HPU`, for calls below autograd such as inference mode). Eager calls and traced graphs both use it. A key where Habana registers its own kernel is left alone.

| gap on Gaudi | root cause | fix | result on Gaudi2 |
|---|---|---|---|
| Masked attention runs op by op (REVE shape: 27.1 ms, 63.6 GiB peak, eager). | The math path is the default on HPU. | Habana's setting above. Every mask (boolean, additive, none) then runs `hpu.sdpa_fwd`. | 4.9 ms eager with a 9.8 GiB peak; compiled, 12.3 to 4.9 ms. |
| Compiled linear upsampling is 20x slower than eager (CodeBrain). | `torch.compile` traces `upsample_linear1d.vec` through torch's decomposition, i.e. gather/scatter index ops; their scatter backward is slow on Gaudi. Unfixed in bridge 1.24.2. | compile: HPU tensors with static shapes keep the native op, which Habana compiles with its own kernels. | Forward + backward 56.9 to 3.4 ms (2.8 ms eager). |
| Nearest upsampling fails after a BatchNorm1d, eager and compiled (USleep: `layout.h:75 permute`). Compiled, torch's decomposition leaves `_unsafe_index` ops that Habana runs eagerly. | Gaudi's `upsample_nearest1d` kernel cannot take a BatchNorm1d output, with or without autograd. torch decomposes `.vec` and `.default` into index ops. | eager + compile: an integer factor `s` repeats each sample `s` times (`repeat_interleave` along the last dim), for nearest and nearest-exact. Other factors keep the native op, under compile too. | USleep trains, eager and compiled (7.6–8.7 ms/step). SeizureTransformer compiled: 43.0 to 8.6 ms/step. |
| `padding="valid"` convolutions fail to compile (ATCNet, EEGInceptionERP, EEGITNet: `Can not compile graph`). | For `"valid"`, ATen's `conv{n}d.padding` passes the one-element list `[0]` as padding and output padding. ATen broadcasts it; Habana's compiled convolution reads a garbage second value. | compile: HPU `"valid"` convolutions go through `conv{n}d.default` with `n` zeros. | All three compile: 2.2–7.1 ms/step, against 22–152 ms eager. |
| `glu` fails to compile after a convolution (BrainModule: `glu_fwd_f32 GLUE_INCOMPATIBLE_OUTPUT_SIZE`). | Gaudi's compiled `glu` kernel. | eager + compile: `a * sigmoid(b)` on the two halves, differentiated by autograd. | BrainModule compiles: 7.2–7.3 ms/step, against 19–25 ms eager. |
| `renorm` runs on the CPU at every step (SSTDPN: `aten::renorm.out` CPU fallback). | No HPU kernel. | eager + compile: norm, scale factor `maxnorm / (norm + 1e-7)` where the norm exceeds `maxnorm`, and multiply, as ATen computes it. | No CPU fallback left: eager, compiled, and in inference mode. |
| einsum graphs fail to compile (BIOT: `Can not compile graph`). | `pass_remove_unnecessary_bmm_view` removes views that are not pure batch merges. Unfixed in bridge 1.24.2. | `enable()` sets Habana's `PT_HPU_DISABLE_pass_remove_unnecessary_bmm_view`, Intel's own workaround (vllm-gaudi #1102). | BIOT compiles. |
| LaBraM fails to compile (`used before defined`). | `batch_as_strided` puts a `getitem` before the node it reads. Fixed in bridge 1.22. | `enable()` sets `PT_HPU_DISABLE_pass_batch_as_strided_groups`. | LaBraM compiles. |
| FBCNet and FBMSNet fail to compile (`create tensor invalid permutation 4 2`), EEGNet and Deep4Net at times. | `pass_weight_permutation` puts an `hpu.weight_permutation` before every 2-D and 3-D convolution weight. The backward of some graphs then fails to create a tensor. | `enable()` sets `PT_HPU_DISABLE_pass_weight_permutation`. | All four compile (EEGNet and Deep4Net: 8 of 8 times). CBraMod steps 2.2% faster, CodeBrain the same. |

The pass switches are read at every compile. A non-empty setting in your environment wins (`PT_HPU_DISABLE_<pass>=0` keeps a pass on).

## Measured on Gaudi2 (habana 1.21.4 / torch 2.6.0)

### braindecode models

One card, with the model-zoo scan's setup: 8 channels × 1000 samples, batch 4, and the sum of the outputs' means as the loss. Each cell is the median ms per training step over 5 steps, for two runs. Both hpu-ops versions run with hpu-complex and the attention setting.

| model | gap | 0.1 | 0.2, eager | 0.2, compiled |
|---|---|---|---|---|
| FBCNet | weight permutation | compile fails | 5.0, 8.8 | 1.8, 1.9 |
| FBMSNet | weight permutation | compile fails | 6.5, 8.5 | 2.4, 2.8 |
| EEGNet | weight permutation | compile fails at times¹ | 10.9, 10.0 | 1.6, 1.8 |
| Deep4Net | weight permutation | compile fails at times¹ | 13.1, 10.9 | 1.8, 2.1 |
| ATCNet | `padding="valid"` | compile fails | 152, 110 | 7.1, 6.8 |
| EEGInceptionERP | `padding="valid"` | compile fails | 21.8, 28.3 | 2.2, 2.7 |
| EEGITNet | `padding="valid"` | compile fails | 45.2, 26.5 | 2.9, 3.0 |
| BrainModule | `glu` | compile fails | 24.6, 18.6 | 7.3, 7.2 |
| USleep | nearest upsampling | eager and compile fail | 53.4, 66.9 | 7.6, 8.7 |
| SSTDPN | `renorm` | `renorm.out` on the CPU at every step | 11.2, 15.9 | 2.1, 3.4 |
| SeizureTransformer | nearest upsampling | compiled 43.0, with eager index ops | 109 | 8.6 |

¹ Without plugins, EEGNet failed 4 of 17 compiles; with 0.1, Deep4Net failed 1 of 3. With 0.2, 8 of 8 compiles of each passed.

Other models that reach a rewritten op still compile at their earlier step times: CTNet (`padding="valid"`, 5.9 ms) and CodeBrain (linear upsampling, 22.9 ms). MetaNeuromotorHand's eager step (`nn.GLU`) passes. NeuroPose upsamples in 2-D, which 0.2 leaves to torch: its compiled graph keeps three eager index ops, as with 0.1, and runs slower than eager (133 against 55 ms).

### Plugin route against patched braindecode (0.1)

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

### Cost of the pass switches

Turning the 0.1 passes off costs at most 1.7% of a step. Turning `pass_weight_permutation` off costs nothing measurable, and makes CBraMod faster.

| model, pass | pass on | pass off |
|---|---|---|
| STEEGFormer, the 0.1 passes | 0.973 | 0.978 |
| EEGPT, the 0.1 passes | 0.618 | 0.629 |
| CBraMod, weight permutation | 0.884, 0.879, 0.884 | 0.862, 0.865, 0.860 |
| CodeBrain, weight permutation | 3.230, 3.237 | 3.228, 3.242 |
| EEGNet, weight permutation (ms) | 2.44, 2.46, 5.46 | 2.38, 2.36, 2.47 |
| Deep4Net, weight permutation (ms) | 6.87, 2.90, 2.87 | 2.81, 2.90, 3.04 |

- Seconds per step unless marked; one value per run.
- CBraMod and CodeBrain are compiled production steps, as in the table above, with hpu-ops 0.1 and the attention setting.
- EEGNet and Deep4Net are a compiled SGD loop at batch 64, as the median of 200 synchronized steps.
- The switch changes the losses by up to 0.06% (CBraMod, step 3: 37.6417 with the pass, 37.6201 without).

CBraMod, REVE and BIOT show no change from the 0.1 passes. Habana's scatter-add flags (`PT_HPU_USE_UNSORTED_SCATTER_ADD`, `PT_HPU_SORT_INDEX_IN_SCATTER_ADD`) do not help compiled linear upsampling: it stays at 57–64 ms.

## Tests

```bash
python -m pytest tests/test_cpu.py   # the CPU stands in for HPU
python tests/run_on_hpu.py           # on a Gaudi card, with the attention setting exported; exits 1 on any failure
```

`test_cpu.py` installs each rewrite with the CPU in the role of HPU and checks:
- what compiled graphs contain: the native linear op and no `index_put`; no index ops for nearest by an integer factor, and the native nearest op for other factors; full zero padding lists for `"valid"`; no `glu` and no `renorm`;
- that eager and compiled results match the native ops, with 16-bit and non-contiguous inputs too;
- that symbolic shapes (a new batch size) and other devices keep torch's own path, and that undoing restores torch;
- that the attention check reads the fused kernel's registration, and that symbolic linear upsampling warns while `dynamic=False` keeps the native op.

CI runs `test_cpu.py` on torch 2.6.0 (Gaudi's) and the latest CPU release.

`run_on_hpu.py` compares outputs and input gradients against torch on the CPU, eager and compiled. It covers:
- SDPA with boolean, additive and no mask, and `nn.MultiheadAttention`;
- CodeBrain's linear upsampling at two batch sizes, and long upsampling;
- nearest upsampling after a BatchNorm1d, with gradients, under `no_grad` and under `inference_mode`; nearest-exact; a non-integer factor;
- `padding="valid"` 2-D depthwise and 1-D convolutions;
- `glu` after a convolution;
- `renorm`, also in inference mode;
- braindecode's BIOT attention block.

It also checks:
- that eager attention runs `hpu.sdpa_fwd`, and that each rewrite ran;
- that `renorm` leaves Habana's CPU-fallback counter at zero, eager, compiled and in inference mode;
- that braindecode's FBCNet compiles (the weight-permutation switch), and that its output and parameter gradients match eager mode.

## Limits

- 2-D and 3-D upsampling keep torch's decomposition under compile, and Habana runs its index ops eagerly (NeuroPose).
- Nearest upsampling by a non-integer factor keeps Gaudi's kernel, so it still fails after a BatchNorm1d. So do integer factors whose float32 index arithmetic is not exact, such as 41: there the native kernels pick the previous sample at some positions, and a repeat would differ from them.
- Shapes that `torch.compile` treats as symbolic keep torch's decomposition for linear upsampling (with a warning) and non-integer nearest upsampling, because the native backward has no meta kernel for them. After a new batch size, dynamo marks dimensions dynamic; with `dynamic=False` it recompiles with static shapes instead. Integer-factor nearest upsampling repeats with a symbolic batch too.
- `glu` is rewritten at the autograd key only. Under `torch.inference_mode`, Gaudi's eager kernel runs (it works), and a graph compiled in inference mode keeps it.
- `renorm_` and `renorm.out` still fall back to the CPU; the functional `renorm` does not.
- `torch.save` of conv2d weights fails after an optimizer step on HPU (`Number of dims in tensor don't match in permute`), with or without compile, and with `pass_weight_permutation` on or off. Fresh weights, and weights after a forward and backward only, save fine. The `.cpu()` and `copy.deepcopy` copies are correct, so save `{k: v.cpu() for k, v in model.state_dict().items()}`.
- On bridge 1.21, Habana's attention override keeps the full attention-probability tensor for the backward (no recompute).
- An attention mask gets no gradient; key-padding masks need none.
- Fully masked query rows (a sample with no real key) may differ from torch's zeros.
