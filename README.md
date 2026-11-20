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

On 1.21 and 1.22, `enable()` warns when the setting did not take effect: Habana's fused kernel is then not registered in the dispatcher. On 1.23+ the kernel is always registered and chosen per call, so `enable()` cannot tell whether your attention runs inside `sdpa_kernel`.

Compile with `dynamic=False`, or keep one batch size (`drop_last=True`). A new batch size otherwise makes the shapes symbolic, and linear upsampling falls back to the slow decomposition, with a warning.

### bfloat16 autocast with CUDA's op lists (optional)

Under `torch.autocast("hpu", torch.bfloat16)`, Habana's default lists put norms, softmax, `mean` and `mul` in bfloat16, even for float32 inputs. CUDA keeps them in float32. The lists are the same in bridges 1.21 to 1.24. Habana reads its two lists from files when its library loads. For torch's CUDA lists (`ATen/autocast_mode.h`), export them before Python starts. CUDA's promote ops that Habana's fixed promote list lacks (`atan2`, `bilinear`, `cross`, `dot`, `vdot`, `grid_sampler`, `index_put`, `tensordot`, `scatter_add`) join the float32 list: float32 is CUDA's result for mixed inputs, and mixed inputs would otherwise fail. So does `binary_cross_entropy`, which CUDA refuses under autocast.

```bash
eval "$(python -m hpu_ops.autocast)"
```

Output dtypes from float32 inputs under bfloat16 autocast, on Gaudi2 (bridge 1.21.4):

| op | Habana's lists | CUDA's lists |
|---|---|---|
| `linear`, `matmul`, convolutions, SDPA | bfloat16 | bfloat16 |
| `layer_norm`, `group_norm`, `softmax`, `log_softmax` | bfloat16 | float32 |
| `batch_norm`, `mean`, `mul` | bfloat16 | float32 (the input's) |
| `sum`, `exp`, `div`, `interpolate` | float32 | float32 |

Compiled production steps (bfloat16 autocast, one card) take 2.2% longer for EEGPT (0.629 to 0.642 s) and 0.9% longer for LaBraM (1.813 to 1.829 s), each the mean of two interleaved runs.

## What it fixes

Only HPU tensors are rewritten; other devices run the original ops. The rewrites sit in PyTorch's dispatcher, in one of two ways:
- **compile** — torch's Python kernel of the op is replaced. Only `torch.compile`'s tracers read it, so eager calls are unchanged.
- **eager + compile** — a Python kernel is registered at the `AutogradHPU` key (and `HPU`, for calls below autograd such as inference mode). Eager calls and traced graphs both use it. A key where Habana registers its own kernel is left alone.

One fix is outside the dispatcher: `torch.save` pickles HPU tensors through a tensor-level host copy. It hooks the tensor's pickling (`Tensor._reduce_ex_internal`), where Habana hooks its own in lazy mode.

| gap on Gaudi | root cause | fix | result on Gaudi2 |
|---|---|---|---|
| Masked attention runs op by op (REVE shape: 27.1 ms, 63.6 GiB peak, eager). | The math path is the default on HPU. | Habana's setting above. Every mask (boolean, additive, none) then runs `hpu.sdpa_fwd`. | 4.9 ms eager with a 9.8 GiB peak; compiled, 12.3 to 4.9 ms. |
| Compiled linear upsampling is 20x slower than eager (CodeBrain). | `torch.compile` traces `upsample_linear1d.vec` through torch's decomposition, i.e. gather/scatter index ops; their scatter backward is slow on Gaudi. Unfixed in bridge 1.24.2. | compile: HPU tensors with static shapes keep the native op, which Habana compiles with its own kernels. | Forward + backward 56.9 to 3.4 ms (2.8 ms eager). |
| Nearest upsampling fails after a BatchNorm1d, eager and compiled (USleep: `layout.h:75 permute`). Compiled, torch's decomposition leaves `_unsafe_index` ops that Habana runs eagerly. | Gaudi's `upsample_nearest1d` kernel cannot take a BatchNorm1d output, with or without autograd. torch decomposes `.vec` and `.default` into index ops. | eager + compile: an integer factor `s` repeats each sample `s` times (`repeat_interleave` along the last dim), for nearest and nearest-exact. Other factors keep the native op, under compile too. | USleep trains, eager and compiled (7.6–8.7 ms/step). SeizureTransformer compiled: 43.0 to 8.6 ms/step. |
| `padding="valid"` convolutions fail to compile (ATCNet, EEGInceptionERP, EEGITNet: `Can not compile graph`). | For `"valid"`, ATen's `conv{n}d.padding` passes the one-element list `[0]` as padding and output padding. ATen broadcasts it; Habana's compiled convolution reads a garbage second value. | compile: HPU `"valid"` convolutions go through `conv{n}d.default` with `n` zeros. | All three compile: 2.2–7.1 ms/step, against 22–152 ms eager. |
| The weight gradient of a convolution with one input channel and a stride is wrong, compiled: about half of it plus an unrelated part (the first convolutions of EEGDINO, CodeBrain, LaBraM; scale 0.48-0.52, cosine 0.50-0.62 on random data). | Gaudi's compiled `convolution_backward`; eager is right at these shapes but wrong at others, e.g. `Conv2d(1, 25, (1, 3), stride (1, 2))`. | eager + compile: that weight gradient is one contraction of the output gradient with the input's strided patches (`Tensor.unfold`), at the `AutogradHPU` key of `convolution_backward`; the input and bias gradients stay native. | The weight gradients match the CPU, eager and compiled. |
| `glu` fails to compile after a convolution (BrainModule: `glu_fwd_f32 GLUE_INCOMPATIBLE_OUTPUT_SIZE`). | Gaudi's compiled `glu` kernel. | eager + compile: `a * sigmoid(b)` on the two halves, differentiated by autograd. | BrainModule compiles: 7.2–7.3 ms/step, against 19–25 ms eager. |
| `renorm` runs on the CPU at every step (SSTDPN: `aten::renorm.out` CPU fallback). | No HPU kernel. | eager + compile: norm, scale factor `maxnorm / (norm + 1e-7)` where the norm exceeds `maxnorm`, and multiply, as ATen computes it. | No CPU fallback left: eager, compiled, and in inference mode. |
| `nn.Embedding` of a 0-d index trains the wrong row, eager and compiled: every gradient goes to row 0 (BIOT: only the first channel token learns). | Gaudi's `embedding_dense_backward` with a 0-d index; a 1-element index is right, and so is the forward. | eager + compile: the backward of a 0-d index runs as that of the 1-element index, the shapes ATen's own kernels flatten to. The forward is unchanged. | BIOT's channel tokens get their gradients, equal to the CPU's. |
| A roll of a strided tensor by a multiple of its size (0, the size) reads the wrong elements, eager: 1.4 off. EMG2QwertyNet's rotation-invariant MLP rolls each slice of a stack's gradient back by 0: input gradient 1.2 off, and its compiled graph fails. | Gaudi's `roll` kernel; contiguous inputs and other shifts are right. | eager + compile: such a roll is a copy (`clone`), at the `AutogradHPU` key of `roll`. | EMG2QwertyNet matches the CPU, eager and compiled (6.3e-07). |
| Linear 1-D upsampling drifts with the length unless the factor is a power of two without `align_corners`, eager and compiled (NeuroPose's last resize, 9920 to 10000 samples: 34% off). | Gaudi's `upsample_linear1d` forward; its backward is also less exact than the CPU's at some sizes. | eager + compile: explicit source indices and weights, computed as ATen's CPU kernel computes them, differentiated by autograd. Powers of two keep the native op. | Output and input gradient as close to float64 as the CPU's float32 kernel; NeuroPose matches the CPU (1.2e-06). |
| `torch.save` fails after an optimizer step, eager and compiled (`Number of dims in tensor don't match in permute`). | Synapse writes a conv weight updated in place in a permuted layout, and records the permutation on its storage. `torch.save` copies storages to the host as flat byte tensors, and Habana's device-to-host copy cannot undo a 4-D permutation on them (`habana_eager/ops/copy_from.cpp`). | eager + compile: HPU tensors are pickled through a tensor-level host copy, which undoes the permutation, the way torch pickles XLA tensors (and Habana its tensors in lazy mode). They load back on the HPU. | Checkpoints of conv nets save after eager and compiled steps, and load on the card with the values of the same step run on the CPU. |
| `GBWAtt` fails in lazy mode (`synLaunch failed. synStatus 30 [Sections' validation failure]`). | `aten::_index_put_impl_` has no HPU kernel, so `s[s <= threshold] = 0` inside `matrix_sqrt.backward` falls back to the CPU mid-backward, flushing a partial DSD backward graph containing `aten::index.Tensor` (`Vech`) + broadcast matmul with a missing `SHAPE_TENSOR`. | eager + compile + lazy: boolean-mask scalar `index_put_` / `index_put` via `torch.where` at `AutogradHPU` and `HPU`; trailing 2-D integer `index.Tensor` (`X[..., ix0, ix1]`) via flat `index_select` at `AutogradHPU`. | `GBWAtt` trains in lazy mode (912 ms/step vs 2174–2225 ms eager), matching eager loss (`1.44817` vs `1.44816`) with zero `index_put_` or `index.Tensor_out` CPU fallbacks. |
| Large-kernel convolutions (`> 256`) fail graph compile (`gaudi2_agu_config::setAssociatedDimAndSize: size - 1 <= uint8 max`) in `STaRNet(64)` (`conv_spatial_fusion` `(351, 1)`) and `TensorCSPNet(22, dims=(22, 11, 11, 11))` (`temporal_block` `(4, 1089)`). | Gaudi2's MME AGU rejects a single `Conv` node whose spatial kernel dimension exceeds 256 (odd or `stride <= 255`). | eager + compile + lazy: `aten.convolution.default` at `AutogradHPU` and `HPU` splits any spatial kernel dim `> 256` into `<= 256` chunks with exact input slicing and `F.pad`, summing partial outputs (16-bit inputs: each chunk is convolved in 16-bit, the chunks are summed in `float32`, so the output is rounded twice, about twice the native op's error). | `STaRNet(64)` trains eager (279.1 ms/step), lazy (127.1 ms/step), and compiled (max\|err\| 6.4e-07); `TensorCSPNet(22, k=11)` trains eager (255.3 ms/step), lazy (130.3 ms/step), and compiled, with exact loss parity. |
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
- that the attention check reads the fused kernel's registration, and that symbolic linear upsampling warns while `dynamic=False` keeps the native op;
- that a checkpoint after an optimizer step (state dicts, a `Parameter`, momentum) goes through the host copies and loads with `weights_only=True`, on its device or where `map_location` says; that hooks still warn and `skip_data` raises;
- that boolean-mask scalar `index_put_` / `index_put` (including inside `autograd.Function.backward()` and `inference_mode()`) and trailing 2-D integer `index.Tensor` match the native ops, eager and compiled, while unsupported or invalid inputs fall through to ATen;
- that large-kernel convolutions (`> 256`) match native 1-D/2-D/3-D convolutions across eager, `torch.compile`, `inference_mode()`, `channels_last`, `bfloat16`/`autocast`, `groups`, `Conv1d`/`Conv3d`, dynamic batch shapes, and coexistence with `valid_conv` and `conv_weight_grad`, while invalid/transposed/symbolic-spatial inputs fall through to ATen;
- that the autocast files hold the installed torch's CUDA lists, and that the installed package ships them.

CI runs `test_cpu.py` on torch 2.6.0 (Gaudi's) and the latest CPU release.

`run_on_hpu.py` compares outputs and input gradients against torch on the CPU, eager and compiled. It covers:
- SDPA with boolean, additive and no mask, and `nn.MultiheadAttention`;
- SDPA under the attention setting at braindecode's shapes (`sdpa_checks.py`, also standalone): q/k/v as strided views and as copies, bfloat16, dropout statistics with a backward on the forward's draws, and fully masked rows (recorded); a known Habana bug with sliced views is probed in separate processes and not counted;
- CodeBrain's linear upsampling at two batch sizes, and long upsampling;
- nearest upsampling after a BatchNorm1d, with gradients, under `no_grad` and under `inference_mode`; nearest-exact; a non-integer factor;
- `padding="valid"` 2-D depthwise and 1-D convolutions;
- `glu` after a convolution;
- `renorm`, also in inference mode;
- braindecode's BIOT attention block.

It also checks:
- that eager attention runs `hpu.sdpa_fwd`, and that each rewrite ran;
- that `renorm` leaves Habana's CPU-fallback counter at zero, eager, compiled and in inference mode;
- that the embedding of 0-d indices (BIOT's channel tokens) gets the CPU's weight gradients;
- that one-channel strided convolutions (EEGDINO, CodeBrain, LaBraM) get the CPU's weight gradients, eager and compiled;
- that rolls of strided slices by a multiple of the size, and the input gradient of a stack of rolls by -1, 0, +1, match the CPU, eager and compiled;
- that linear upsampling by other than a power of two is as close to float64 as the CPU's float32 kernel, output and input gradient, eager and compiled, and that powers of two keep the native op;
- that `torch.save` works after an eager and after a compiled optimizer step, and loads, on the card, the values of the same step run on the CPU;
- that boolean-mask scalar `index_put_` (including inside `autograd.Function.backward()` and `inference_mode()`) and trailing 2-D integer `index.Tensor` (`Vech`) match the CPU, eager and compiled, with zero CPU fallbacks;
- that large-kernel convolutions (`(351, 1)` and `(4, 1089)` stride `(1, 121)`) match the CPU, eager, compiled, and in `inference_mode()`, via `split_conv`;
- with `hpu_ops.autocast`'s lists exported, CUDA's output dtypes under bfloat16 autocast (without them, Habana's are printed);
- that braindecode's FBCNet compiles (the weight-permutation switch), and that its output and parameter gradients match eager mode.

## Limits

- 2-D and 3-D upsampling keep torch's decomposition under compile, and Habana runs its index ops eagerly (NeuroPose).
- Nearest upsampling by a non-integer factor keeps Gaudi's kernel, so it still fails after a BatchNorm1d. So do integer factors whose float32 index arithmetic is not exact, such as 41: there the native kernels pick the previous sample at some positions, and a repeat would differ from them.
- Shapes that `torch.compile` treats as symbolic keep torch's decomposition for linear upsampling (with a warning) and non-integer nearest upsampling, because the native backward has no meta kernel for them. After a new batch size, dynamo marks dimensions dynamic; with `dynamic=False` it recompiles with static shapes instead. Integer-factor nearest upsampling repeats with a symbolic batch too.
- `glu` is rewritten at the autograd key only. Under `torch.inference_mode`, Gaudi's eager kernel runs (it works), and a graph compiled in inference mode keeps it.
- `renorm_` and `renorm.out` still fall back to the CPU; the functional `renorm` does not.
- `torch.save` saves each HPU tensor through its own host copy, as torch does for XLA tensors: two HPU views of one storage are saved as two copies, and no longer share it after loading. `torch.load` puts the tensors back on the HPU, or where `map_location` says. Under `torch.serialization.skip_data`, saving HPU tensors raises, as for XLA. Habana's `PT_HPU_ENABLE_SYNAPSE_OUTPUT_PERMUTE=0` also avoids the failure, but compiled convolution training steps take 23–43% longer (EEGNet, Deep4Net, batch 64).
- On bridge 1.21, Habana's attention override keeps the full attention-probability tensor for the backward (no recompute).
- An attention mask gets no gradient; key-padding masks need none, but a learned bias passed as the mask never trains (DIVER1's channel bias). Without the attention setting, the math path gives it its gradient.
- On bridge 1.21.4, in eager mode, Habana's fused attention returns wrong values for a call whose contiguous q, k and v have as many elements as an earlier call's but another shape. Axial attention over (batch x channels, patches) and (batch x patches, channels) tokens always has that (AXON: 36% off). Views of one projection, compiled graphs and the math path are right.
- `roll` by a multiple of the size keeps Gaudi's kernel below autograd (inference mode): hpu-ops does not override Habana's `HPU` kernel.
- Boolean-mask `index_put_` / `index_put` only rewrites single-mask boolean indexing with a 0-d scalar value; non-scalar value tensors and non-boolean indices keep ATen. Trailing 2-D integer `index.Tensor` is rewritten at `AutogradHPU` only, keeping Habana's `HPU` kernel below autograd (`inference_mode()`).
- Large-kernel convolution splitting requires static spatial, kernel, stride, padding, and dilation sizes (dynamic spatial sizes fall back to the native op); under `torch.compile` inside `torch.inference_mode()`, `AutogradHPU` is excluded and `ProxyTorchDispatchMode` sits above `HPU`, so compiled `inference_mode()` graphs keep the unsplit `aten.convolution.default`. Separately, 1-D convolutions with `kW in {255, 256}` and long inputs (e.g. `Win = 1000`, `Wout = 745..746`) still hit `setAssociatedDimAndSize` in backward `DEDX` even with `kW <= 256`.
- Fully masked query rows (a sample with no real key) may differ from torch's zeros.
