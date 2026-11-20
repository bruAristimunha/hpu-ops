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

### float64

Gaudi2 stores float64 as float32 when a tensor is copied to it: `1 + 2**-40` comes back as `1`. No result computed on the card can be float64-exact. For float64 linalg, hpu-ops computes on the CPU, over the values the card holds, and warns once per op (hpu-ops#16). The kernels above take float32 only.

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
| einsum graphs fail to compile (BIOT: `Can not compile graph`). | `pass_remove_unnecessary_bmm_view` removes views that are not pure batch merges. Unfixed in bridge 1.24.2. | `enable()` sets Habana's `PT_HPU_DISABLE_pass_remove_unnecessary_bmm_view`, Intel's own workaround (vllm-gaudi #1102). | BIOT compiles. |
| LaBraM fails to compile (`used before defined`). | `batch_as_strided` puts a `getitem` before the node it reads. Fixed in bridge 1.22. | `enable()` sets `PT_HPU_DISABLE_pass_batch_as_strided_groups`. | LaBraM compiles. |
| FBCNet and FBMSNet fail to compile (`create tensor invalid permutation 4 2`), EEGNet and Deep4Net at times. | `pass_weight_permutation` puts an `hpu.weight_permutation` before every 2-D and 3-D convolution weight. The backward of some graphs then fails to create a tensor. | `enable()` sets `PT_HPU_DISABLE_pass_weight_permutation`. | All four compile (EEGNet and Deep4Net: 8 of 8 times). CBraMod steps 2.2% faster, CodeBrain the same. |
| `linalg.solve_triangular` and `cholesky_solve` run on the CPU, and so do the backward passes of `cholesky` and `qr` (#12). | No HPU kernel; Habana's CPU fallback. The backward of Cholesky and QR is made of these two solves. | A TPC kernel (below) for float32 matrices with n <= 32. The backward passes of `cholesky` and `qr` reach it through the two solves. | `solve_triangular`, n = 22, batch 32: 0.49 ms lazy, 0.46 ms eager, against about 1.4 and 1.5 ms for the fallback. |
| `linalg.solve`, `solve_ex`, `det` and `slogdet` run on the CPU (#10). | No HPU kernel. Rebuilt from torch ops, they were measured 3 to 25 times slower than the fallback per call: one launch per op. | A TPC Gauss-Jordan kernel with partial pivoting, n <= 32, right-hand sides in chunks of 32. Gradients are torch's formulas; double backward works. | n = 22, batch 32, lazy / eager: `solve` 0.57 / 2.7 ms against about 2.2 / 4.5; `det` 0.51 / 0.82 against about 1.4 / 2.7. `solve_ex`, eager: 0.46 ms against about 4.5. |
| `linalg.inv` runs on the CPU (#11). | No HPU kernel. | The same Gauss-Jordan kernel. | n = 22, batch 32, lazy / eager: 0.48 / 1.8 ms against about 2.2 / 4.6. |
| `linalg.eigh` and `eigvalsh` run on the CPU (#8; 21 of 21 SPD models). | No HPU kernel. The parallel-Jacobi composite was measured slower than the fallback, so it only serves the CPU stand-in. | A TPC one-sided Jacobi kernel, n <= 32. | n = 22, batch 32, lazy / eager: 0.80 / 1.27 ms against about 1.4 / 3.7 (the raw kernel takes 0.19 ms). Batch 1024: 3.7 ms against 52 ms. Training steps: see below. |

The pass switches are read at every compile. A non-empty setting in your environment wins (`PT_HPU_DISABLE_<pass>=0` keeps a pass on).

### Linalg kernels on the TPC

The linalg rows above run as single TPC kernels, one matrix per index-space member. They live in `hpu_ops/tpc/`. The library, `liblinalg_tpc.so`, is prebuilt. To rebuild it inside the Habana image, run `hpu_ops/tpc/build.sh`.

- **Kernels:** Gauss-Jordan with partial pivoting for `solve`, `inv`, `det`, `slogdet`, `solve_triangular` and `cholesky_solve`; one-sided Jacobi for `eigh` and `eigvalsh`.
- **Limit:** float32 matrices with n <= 32 on a Gaudi2. Calls the kernels do not take (n > 32, float64, kernels off) run through hpu-ops' own CPU route (`hpu_ops.linalg._on_cpu`: the tensors are copied to the CPU, the op runs there, the result is copied back). That is not Habana's fallback, and its cost is not the fallback's. Float64: see "float64".
- **Order:** call `hpu_ops.enable()` before the first HPU op. Habana reads the kernel path when the device initialises; later, the kernels are off, with one warning, and the calls take the CPU route above.
- **Opt-out:** `HPU_OPS_TPC=0` in the environment keeps the kernels off. They are also off on a device that is not a Gaudi2, and when `liblinalg_tpc.so` is missing (one warning).
- **Transforms:** under `torch.func` transforms and forward-mode AD, the transformed call goes through torch's own derivative rules (`jacfwd`, `jvp`, `hessian` and forward AD use torch's kernel; for `vmap`, `grad` and `jacrev` the inner calls may still use the TPC kernel; results match).
- **Singular triangles:** `solve_triangular` and `cholesky_solve` of a triangle with a zero diagonal entry (or a zero pivot) return NaN for that matrix, as torch's give non-finite values. `det` of a singular matrix is 0 on the card and its gradient equals torch's (measured on Gaudi2). `slogdet` of a singular matrix gives `logabsdet = -inf`, as torch does, but its gradient is finite where torch's is non-finite, so a finiteness check on that gradient does not flag a singular input.
- **Lazy mode:** works. Each kernel call ends its lazy graph.
- **Accuracy:** within 10x of CPU float32 against float64 (`eigh` within about 3.8x), also with dense repeated eigenvalues, condition 1e6 and inputs scaled by 1e30. Measured at n = 16 and 22. The gate runs 130 cases, including n = 1, 3, 21, 31 and 32 for the eight kernel ops and an indefinite `eigh` and `eigvalsh` at n = 3, 22 and 32: 130 of 130 pass in eager mode and 129 of 130 in lazy mode, measured before the 1x1 change below. The one lazy failure is the backward of `qr` at n = 22 (error 6.11e-6 against a bar of 6.05e-6), the known QR gap below. `eigh` and `eigvalsh` at n = 1 do not use the kernel (its graph does not compile at n = 1 in lazy mode): they return the entry and 1.

## Measured on Gaudi2 (habana 1.21.4 / torch 2.6.0)

### SPD models: training steps with the linalg kernels

Same pod, batch 32, ms per training step, `origin/main` to this branch.

| model | lazy | eager |
|---|---|---|
| SPDNet | 24.7 to 20.0 | 66.8 to 60.5 |
| TensorCSPNet | 344.7 to 175.7 | 367.2 to 200.0 |
| EEGSPDNet | 65.6 to 61.0 | 220.7 to 226.2 |
| TSMNet | 38.1 to 48.0 | 159.9 to 175.1 |

TSMNet got slower: its `solve` has n = 40, above the kernel limit, so it takes hpu-ops' own CPU route, not Habana's fallback.

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
- that a checkpoint after an optimizer step (state dicts, a `Parameter`, momentum) goes through the host copies and loads with `weights_only=True`, on its device or where `map_location` says; that hooks still warn and `skip_data` raises.
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
- with `hpu_ops.autocast`'s lists exported, CUDA's output dtypes under bfloat16 autocast (without them, Habana's are printed);
- that braindecode's FBCNet compiles (the weight-permutation switch), and that its output and parameter gradients match eager mode.

## Known gaps

- TSMNet's `solve` has n = 40, above the kernel limit of 32. It runs through hpu-ops' own CPU route, not Habana's fallback, and its training step is slower than on `origin/main` (table above). The limit is being raised to 64.
- `info`, the index of a singular pivot, follows the kernel's own elimination order and can differ from LAPACK's.
- The backward of `qr` at n = 16 and, in lazy mode, at n = 22 (6.11e-6 against a 6.05e-6 bar) is less accurate than CPU float32. A substitution kernel is being written for it.
- float64 linalg runs on the CPU (see "float64").
- `slogdet` of a singular matrix: `logabsdet = -inf` as in torch, but the gradient is finite where torch's is non-finite (measured on Gaudi2).
- Calls the kernels decline (n > 32, kernels off, another Gaudi generation) take hpu-ops' own CPU route, which is not Habana's fallback. Its per-call cost against the fallback is not measured.

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
- Fully masked query rows (a sample with no real key) may differ from torch's zeros.
