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

`torch.compile(backend="hpu_backend")` needs eager mode (`PT_HPU_LAZY_MODE=0`, the default since 1.21): Habana's 1.21.4 support matrix lists it under "Eager + torch.compile" only. In lazy mode (`PT_HPU_LAZY_MODE=1`), after `enable()`, `torch.compile(model, backend="hpu_backend")` warns once, saying so, and returns `model` itself, which runs in lazy mode as if not compiled (SPDNet trains with the uncompiled lazy loss). `torch._dynamo.optimize("hpu_backend")` and a `torch.compile` bound before `enable()` get a pass-through backend: dynamo traces, the graph runs op by op in lazy mode. `HPU_OPS_LAZY_COMPILE=raise` makes it a `RuntimeError` at the call instead. A function compiled before `enable()`, or the backend passed as a function, is not covered. Without hpu-ops, Habana's backend fails deep inside on the first graph it receives (SPDNet: `_recipe_compiler_C.so: undefined symbol`, since lazy mode does not load the eager plugin that library links against; `torch.ops.hpu.weight_permutation`, read by the backend's partitioner, is not registered either), and a model whose frames dynamo does not capture runs uncompiled without notice (a Linear MLP). To compare the modes, see "Choosing a mode" below.

Compile with `dynamic=False`, or keep one batch size (`drop_last=True`). A new batch size otherwise makes the shapes symbolic, and linear upsampling falls back to the slow decomposition, with a warning.

### NaN propagation (on by default)

`enable()` makes Gaudi's max/min ops propagate NaN as torch does (see `extrema` in the table below). Measured cost,
eager training steps against origin/main (3 alternating rounds, medians): TensorCSPNet +7%, GBWAtt +9%, EEGSPDNet
+8%, SPDNet and TSMNet within noise; lazy within noise. `export HPU_OPS_NAN_PROPAGATION=0` before `enable()` keeps
Gaudi's NaN-dropping ops. The sort family is exact too (`HPU_OPS_EXACT_SORT=0` keeps Gaudi's sort); hpu-ops' own
eigenvalue sorts keep Gaudi's sort, so SPD steps are unchanged.

### float64

Gaudi2 stores float64 as float32 when a tensor is copied to it: `1 + 2**-40` comes back as `1`. No result computed on the card can be float64-exact. For float64 linalg, hpu-ops computes on the CPU, over the values the card holds, and warns once per op (hpu-ops#16). The kernels above take float32 only.

What the card holds (`tests/probe_float64.py`, bridge 1.21.4, eager and lazy):

- A float64 HPU tensor reports 8 bytes per element and allocates them, but its data is float32: viewed as `int32` on the card, the words are the float32 bit patterns of the values, packed 4 bytes apart. Random float64 data comes back equal to its float32 rounding (no bit of 4096 kept). Values out of the float32 range become `inf` / `0` in eager mode and raise in lazy mode. float64 arithmetic on the card is float32 arithmetic (`(1 + 2**-40) - 1 == 0`).
- 64-bit bit patterns cannot be carried in a float64 tensor either: an `int32` tensor of the pairs, viewed as float64 on the card, reads each 32-bit word as one float32 (eager) or fails the copy (lazy).
- So a float64 input is already rounded when the op sees it, and a float64 output is rounded when it is written. Mixed-precision refinement cannot recover the lost bits, and hpu-ops does not emulate float64: today's float64 linalg on HPU tensors returns float32-level accuracy against the original float64 data (solve at n = 22: 3.3e-7 relative at condition 1e2, 1.3e-3 at 1e6; `det` of a matrix with determinant 1e-66 comes back 0). Keep float64 work on the CPU.
- Eager-mode bugs of the 8-byte dtypes (`repro/float64_storage_offset_eager.py`, `repro/probe_offset8.py`, `repro/probe_f64_int64.py`; bridge 1.21.4), not fixable from the dispatcher at a cost a model would accept, so `enable()` warns once at the copy that puts such data on the card (`wide`, eager only):
  - float64: an op on a *contiguous* float64 view with a storage offset (`x[1:]`, `x[i]`, `narrow`, `split`, `unbind`, `as_strided` with an offset) reads and writes at twice the offset: `h[1:] + 0` of `[1, ..., 6]` gives `[3, 4, 5, 6, 0]`, `h[1:4].copy_(v)` writes elements 2 to 4, the gradient of `h[3:9]` is zeros, `clone` / `copy_` / `.to()` / `matmul` / `sum` / `index_select` of such a view read wrong. Non-contiguous views (`x[:, 1:]`, `x[1::2]`, a transposed row), `.cpu()`, lazy mode, `torch.compile` (`hpu_backend`) and every other dtype (int64, int32, bfloat16, ... checked) are right. The first float64 copy to the card warns. A fix would give every op on such a view a copy made through a strided read (measured right: an expanded or `index_select` read of the base) and send every in-place write back the same way; a copy at view creation breaks view semantics.
  - int64: eager kernels compute in 32 bits. Arithmetic, `sum` / `cumsum` / `max`, `sort`, `where`, `index_select`, shifts, `>` read the low 32 bits of each value (`(2**40 + 1) + 1` gives `2`); copies, `clone`, views, `cat`, `==` and `.float()` keep all 64; `gather` of int64 data (any values, a known trap) and `arange` past 2^31 fail to compile. Lazy mode raises at the copy ("Error when trying to cast Long to Int"). In both modes a result past the int32 range from in-range inputs wraps or saturates (`(2**31 - 1) + 1` gives `2**31 - 1`, `2**20 * 2**20` gives `0`) and float32 to int64 saturates at the int32 bounds, with no error; no copy-time check sees that. In eager mode each CPU int64 tensor copied into an int64 tensor on the card is scanned on the host (one `aminmax`, +32-44 us for 32 labels; a Python int or a CPU float tensor copied in is checked too; the scan is per thread) until one is out of the int32 range, which warns once; `HPU_OPS_INT64_CHECK=raise` raises instead, `=0` skips the scan.
- float32 matmul on the MME is IEEE float32: products keep all 23 mantissa bits, accumulation rounds at 2^-24, errors against float64 equal the CPU's (random 64 x 64: 2.5e-7 vs 3.2e-7), and neither `torch.set_float32_matmul_precision` nor `GAUDI2_FORCE_MME_FP32_IEEE=1` changes them.

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

### Strict mode: every CPU execution, raised or counted

To check that a training or inference step runs entirely on the card:

```python
with hpu_ops.strict():            # raises hpu_ops.CPUExecutionError at the first CPU execution
    step()
with hpu_ops.strict("report"):    # counts instead
    step()
print(hpu_ops.strict_format_report())
```

or, for the whole process, `export HPU_OPS_STRICT=1` (raise) or `HPU_OPS_STRICT=report` (a table on stderr at
exit), read by `enable()`. The error names the op, the shapes, dtypes and devices of its tensors, and the first
Python frame outside torch, Habana, hpu-ops and hpu-complex, e.g.

```
hpu_ops strict: aten::_cummin_helper runs on the CPU [bridge fallback] ((8, 4, 4) float32 hpu): under aten::cummin
  called from train.py:92 in step
```

What it sees (source column of the report):

| source | what |
|---|---|
| `bridge fallback` | Habana's CPU fallback: the bridge's `cpu_fallback` metric, read around every op |
| `hpu_ops route` | hpu-ops' own CPU routes: declined linalg calls (`_on_cpu`, float64, n past the kernels, `out=` dtype errors), the singular `det` double backward's SVD, declined `im2col`/`col2im`, index tensors built on the host |
| `hpu_complex` | hpu-complex's CPU fallback, complex128 (and host-backed float64) computed on the CPU in host memory, complex64 `mm`/`bmm` below its CPU threshold |
| `host sync` | a device value read on the host: `.item()`, `bool(t)`, `int(t)`, and hpu-ops' own reads (`lstsq`'s rank check, `matrix_exp`'s scaling counts, `bincount`'s range, `quantile`'s q check, `det`'s singular check in double backward) |
| `host copy` | a device tensor copied to the CPU (`.cpu()`, `.to("cpu")`, `.tolist()`) |
| `cpu op` | an op on CPU tensors of more than one element inside the block (a library building indices on the CPU, say) |

`allow=("host sync", "cpu op")` (or `HPU_OPS_STRICT_ALLOW="host sync,cpu op"`) leaves sources out. `torch.save`'s
host copy is never counted.

The per-op attribution is a `TorchDispatchMode`, pushed on the thread's mode stack. It costs about 10-25 us per op
in eager mode and 45-60 us in lazy mode (bridge 1.21.4, `tests/strict_probe.py`), so time steps outside it. A
dispatch mode and torch.compile do not mix: under an ordinary mode Dynamo skips the frame and keeps skipping it after
the block; under an infra mode it recompiles at every call (its guards see the mode's dispatch keys). So inside a
per-op block, compiled code runs eagerly on that thread (Dynamo's `force_eager` stance, for the block's thread only;
on a torch without the hook hpu-ops wraps, `eval_frame._callback_from_stance`, the stance is process-wide while any
block is open): a per-op block checks the eager program. To check the compiled program, use
`strict(per_op=False)` (`HPU_OPS_STRICT_PER_OP=0`): no mode, the bridge's and hpu-complex's counters read when the
block ends (at exit for the environment form, or at `hpu_ops.strict_check()`), named by op, without a frame;
hpu-ops' own routes are still raised where they run, marked when that is while a graph is traced (they are then
nodes of the graph). What strict mode cannot see:

- ops inside C++ kernels (hpu-complex's own copies, Habana's) other than through their counters, and host syncs
  inside an op's kernel (`nonzero`, `torch.equal`);
- ops under hpu-complex's `complex_mode()` (it disables the modes below it), and the bodies of higher-order ops
  other than `torch.cond` and `while_loop`;
- complex64 `mm`/`bmm` against a `set_matmul_cpu_threshold` value set before the first strict block (the extension
  has no getter: strict mode assumes 2^22 until it sees the setter called).

Strict mode is per thread (autograd's threads follow the innermost block's settings), off in DataLoader worker
loops (forked or spawned) and while `torch.save` runs; a block opened inside a torch.compile'd function breaks the
graph there and runs eagerly; `torch.export` inside a per-op block is not supported. Leaving a block takes its watch
off the mode stack wherever it is (a mode entered inside the block and left on stays). The inventory of the 21 SPD
models and the braindecode zoo under strict mode is in `INVENTORY.md` (`tests/strict_sweep.py`,
`tests/strict_inventory.py`).

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
| `Tensor.unfold` runs on the CPU in lazy mode and returns a copy, not a view (TensorCSPNet: 2 fallbacks per step, from hpu-ops' own one-channel conv weight gradient). | Habana's lazy stack has no `aten::unfold` kernel; the bridge's CPU fallback warns that a view op cannot fall back (`repro/tensor_unfold_lazy.py`). Eager has a kernel. | lazy: at the `HPU` key, `as_strided` with the sizes and strides ATen's own kernel computes (read from it on a meta tensor, which also raises ATen's errors): the same view at the same storage offset. torch's autograd kernel stays above it. Eager keeps Habana's kernel. | No `unfold` fallback; values exact, writes into the input show through the view, gradients and double backward match the CPU (adversarial probe, eager and lazy). TensorCSPNet lazy: 2 to 0 fallbacks per step, 167.9 to 164.8 ms (3 alternating rounds; within noise). `unfold_backward` still falls back, eager and lazy (not on TensorCSPNet's step). |
| In lazy mode the gradient of `X[..., rows, cols]` (spd_learn's `vech` and `vec_to_sym`) is wrong from batch 2 to 8 on: 0.45 to 1 relative error; every gradient before it was wrong in SPDResNet, GyroSPD, CorNet, SPNN, DeepGeoCCA, SPDNetAE, EEGSPDNet, MAtt and CorAtt (up to 1.19), eager right. Once that gradient is right, SPNN's lazy graph fails to launch (`Sections' validation failure`). | Gaudi's lazy accumulating `_index_put_impl_` (the backward of `index`) with a `None` before or between the index tensors (`repro/index_put_lazy.py`); and, in one lazy graph with index puts and reads, a boolean-mask put (`repro/lazy_section_validation.py`). Both without hpu-ops. | eager + compile, at `AutogradHPU` and `HPU`: an `index` of integer indices (int32 or int64, broadcast, negative) with an index tensor on the CPU, or that do not lead, is one `index_select` of the indexed span flattened (a slice between two indices joins the linear index as an `arange`, so the input is not copied), with the linear index (backward: `index_add`), built on the host for CPU index tensors and moved once; leading device indices keep Habana's kernel; one boolean mask after a slice goes through its `nonzero` into the same `index_select`; other masks with CPU index tensors are moved to the device; eager + compile: a boolean-mask put of one value, `m[..., mask] = c`, into a dense tensor is `copy_(where(mask, value, self))`. | Lazy: the 9 models within the parity bar (worst 3.1e-4), 20 of 20 lazy models ok, steps as fast or faster (SPDResNet 47.1 to 37.2 ms). Eager: the same gradients, and `aten::index.Tensor_out` no longer falls back to the CPU (SPNN 326 to 240 ms per step). |
| Linear 1-D upsampling drifts with the length unless the factor is a power of two without `align_corners`, eager and compiled (NeuroPose's last resize, 9920 to 10000 samples: 34% off). | Gaudi's `upsample_linear1d` forward; its backward is also less exact than the CPU's at some sizes. | eager + compile: explicit source indices and weights, computed as ATen's CPU kernel computes them, differentiated by autograd. Powers of two keep the native op. | Output and input gradient as close to float64 as the CPU's float32 kernel; NeuroPose matches the CPU (1.2e-06). |
| `torch.save` fails after an optimizer step, eager and compiled (`Number of dims in tensor don't match in permute`). | Synapse writes a conv weight updated in place in a permuted layout, and records the permutation on its storage. `torch.save` copies storages to the host as flat byte tensors, and Habana's device-to-host copy cannot undo a 4-D permutation on them (`habana_eager/ops/copy_from.cpp`). | eager + compile: HPU tensors are pickled through a tensor-level host copy, which undoes the permutation, the way torch pickles XLA tensors (and Habana its tensors in lazy mode). They load back on the HPU. | Checkpoints of conv nets save after eager and compiled steps, and load on the card with the values of the same step run on the CPU. |
| Large-kernel convolutions fail graph compile (`gaudi2_agu_config::setAssociatedDimAndSize: size - 1 <= uint8 max`) in `STaRNet(64)` (`conv_spatial_fusion` `(351, 1)`) and `TensorCSPNet(22, dims=(22, 11, 11, 11))` (`temporal_block` `(4, 1089)`, stride `(1, 121)`, eager only). | Gaudi2's MME AGU rejects most `Conv` nodes with a spatial kernel dimension above 256. Not all: a card sweep of the native op (forward and backward against CPU float64, one case per process; [table below](#large-kernel-convolutions-which-run-natively)) finds one class that runs, one input channel with the kernel wide only in its last dim, and there the result depends on the mode: eager runs every even width from 258 to 6000 at strides >= 32 and fails every odd width tried (257, 351, 1089, 1091, 1501); lazy runs every width from 257 to 6000 at strides >= 8, and stops the card at stride 1 when the input is as wide as the kernel. In lazy mode a native conv1d (4, 8, 1000) x (16, 8, 256) fails as well, and kernels 129..255 give a wrong input gradient (relative error 1.3 to 1.5 at k = 144 over 260, 240 over 300, 255 over 257) when the input is longer than 256 but shorter than twice the kernel. | eager + compile + lazy: `aten.convolution.default` at `AutogradHPU` and `HPU` splits spatial kernel dims `> 256` (lazy mode: `> 128`) into chunks of at most that size with exact input slicing and `F.pad`, summing partial outputs (16-bit inputs: each chunk is convolved in 16-bit, the chunks are summed in `float32`, so the output is rounded twice). It leaves to the native op the measured class (`conv.NATIVE_WIDE`): 2-D float32, groups 1, one input channel, no padding or dilation, `kH <= 22`, only `kW` above the bound; eager: `kW` even and stride `>= 32`; lazy: `kW >= 257` and stride `>= 8`. TensorCSPNet's default `temporal_block` (`(4, 4356)`, stride `(1, 484)`) stays native in both modes, its `dims=(22, 11, 11, 11)` one in lazy mode. Compiled graphs (not measured) split as before. | Bridge 1.21.4, 20 synced steps after 3 warm-up, each run in its own process. Without the fix `STaRNet(64)` fails in eager, lazy and compile; with it 266.0 ms/step eager, 92.0 lazy, 136.4 compiled. `TensorCSPNet(22, k=11)` fails eager without it; with it 140.0 eager (split), 35.3 lazy (native), 57.9 compiled (split). First loss the same in every mode (`1.39551`, `1.47586`). Models whose convolutions run natively are within noise of main, eager and lazy (medians of alternating rounds, this fix on top of main against main): default-dims `TensorCSPNet(22)` 197.5 / 200.0 ms eager, 162.8 / 161.7 lazy (5 rounds), `STaRNet(22)` 100.3 / 122.5 eager, 54.4 / 50.7 lazy, `SPDNet` 34.2 / 44.8 eager (5 rounds), 17.1 / 16.5 lazy; per-process spread 10-30%. Before this change the default-dims TensorCSPNet split ran 405-485 ms/step lazy with 70 `unfold` fallbacks; now 2, as on main. |
| `torch.quantile` / `torch.nanquantile` abort lazy mode (`FATAL ERROR :: MODULE:PT_BRIDGE ... Trying to resize tensor with non-existing TensorImpl`, from `resize_impl_hpu_`). Eager, a slice with a NaN gives a number (`quantile([3, nan, 1, 2], 0.5)`: 2.5, torch: nan; `nanquantile`: 2.5, torch: 2). | ATen's composite writes into an empty `out` tensor that it resizes; Habana's lazy resize of it fails in the accumulation thread. Gaudi's `sort` returns NaN as `+inf`, which the composite then reads as a value. | eager + compile + lazy: `quantile` / `nanquantile` (float and tensor `q`) at `AutogradHPU` and `HPU` as ATen's own steps (sort, ranks, the five interpolations, `gather`, `lerp`) into a fresh tensor; NaN slices found in the input. Arguments ATen rejects go to ATen. | Bridge 1.21.4, each case in its own process, against CPU float64. Lazy without the fix: FATAL for float `q`, tensor `q`, `dim` + `keepdim` + midpoint, and NaN inputs (the backward of a `requires_grad` input ran). With it, all run, error <= 1.1e-7, no CPU fallback, NaNs where torch has them. Eager: native 0.0106 off with a NaN, the rewrite 3.7e-9. Cost, `quantile(x, 0.9, dim=1)` of an 8x300 tensor, 30 synced calls: lazy 0.93 / 1.94 ms (native fails), eager 3.41 / 3.08 ms against native 0.81 / 1.86 ms (two processes). |
| The max/min family drops NaN, eager, lazy and compiled: `amax([nan, 1, 2])` is `-inf`, `max(-1)` returns a finite value and not the first NaN's index, `maximum(1, nan)` is 1, `max_pool` keeps a window's finite max, `norm(x, inf)` is finite, `sort` returns NaN as `+inf`. A NaN then never reaches the loss: NaN checks do not fire. In 16-bit dtypes `argmax` / `max(dim)` of a row of `-inf` return the out-of-range index n (lazy: garbage such as INT64_MIN). `softmax`, `log_softmax`, `logsumexp`, `cummax`, the sums and `relu`/`hardtanh` are right. (`repro/probe_reduction_nan.py`, `repro/maxmin_drop_nan.py`.) | Gaudi's TPC max/min kernels ignore NaN (maxNum semantics); `argmax` starts from an index past the end. Its float `sort` reads NaN as `+inf` with an out-of-range index; its non-stable sort pads rows to a power of two with `+inf` (descending `-inf`), so a real infinity at any other n gets an out-of-range index (n = 3 to 5000); its stable sort falls back to the CPU from ~1000 elements or along a non-last dim. | eager + compile + lazy, inference mode too (`extrema`): at the `HPU` key, for every call, Gaudi's op through its `.out` overload (Habana's lazy kernels of the functional ops are replaced), then NaN where a reduced slice, window or operand holds one (`isnan().any()`, `where`), the first NaN's index (`argmax` of the NaN mask), values read at that index (`gather`); torch's own autograd formulas over that result give torch's gradients (amax: NaN; `max(dim)`: the first NaN; `maximum`: both operands; `clamp`: torch's comparisons); `max()`/`median()` and the inf-norms keep their own backward (torch's syncs; Gaudi's `sign(NaN)` is NaN). At `AutogradHPU`, what `torch.compile` traces: the same fixes as a composite. `amax`/`amin`, `max`/`min` (whole, dim), `median`, `aminmax`, `argmax`/`argmin`, `maximum`/`minimum`, `fmax`/`fmin`, `clamp`/`clamp_min`/`clamp_max` with tensor bounds (in place too), `norm`/`vector_norm` at `+-inf`, `max_pool2d`/`3d` (1-D reaches 2-D): a second pool of NaN codes gives each window's last NaN (torch's pick) as Gaudi's window-local index, so Gaudi's own backward sends its gradient there. By default (`HPU_OPS_EXACT_SORT=0` opts out), eager, lazy and compiled: `sort`/`argsort`/`msort`/`topk`/`kthvalue` as two sorts of finite keys (`nan_to_num(x)`, then its positions shifted per class, -inf/+inf/NaN), along the last dim; hpu-ops' own eigenvalue sorts keep Gaudi's sort. `out=` overloads: the functional result copied out (with autograd on). Integer dtypes keep the native result; adaptive and fractional max_pool are right natively (probed). | Bridge 1.21.4: every fixed op matches torch on the CPU (NaN positions, first-NaN indices, values) in float32/bfloat16/float16 at n = 3 to 65537, eager and lazy, compiled at n = 3, 1000; gradients 0 differences in float32 and bfloat16, eager, lazy and compiled. No CPU fallback. Per call against the native op (`tests/bench_extrema.py`, 32x22x1000 unless noted, 3 alternating processes, medians; the earlier numbers here timed the native op in both arms, as the bench dropped the install's undo), eager: forward `amax()` / `max()` +0.1-0.2 ms, `amax(-1)` +0.5, `max(-1)` / `argmax` / `maximum` / `clamp` / `norm(inf)` +1.4-1.7, pools +5-8 (native 0.15-0.2); forward + backward within noise to +1.5, pools +4.5-8.7; lazy: forward +1.8-4.6, pools +5-8, forward + backward `amax` within noise, the rest +2.5-4.3, `norm(inf)` +8.9, pools +6.7-11.7. No CPU fallback. SPD training steps within the per-process noise of the rewrite off, eager and lazy (`repro/run_steps_extrema_ab.sh`); the exact sort is on by default: the SPD steps make no user-facing sort, and hpu-ops' own eigenvalue sorts skip it. Not fixed: max_pool's window-local indices (MaxUnpool), scalar `clamp` of `+-inf` in float32 (`+-FLT_MAX`) and with a NaN scalar bound (FLT_MAX / inf; torch: `x` for one NaN bound, NaN for two), the sorts in inference mode and with `out=`, `out=` in inference mode. |
| Large-kernel convolutions fail graph compile (`gaudi2_agu_config::setAssociatedDimAndSize: size - 1 <= uint8 max`) in `STaRNet(64)` (`conv_spatial_fusion` `(351, 1)`) and `TensorCSPNet(22, dims=(22, 11, 11, 11))` (`temporal_block` `(4, 1089)`, stride `(1, 121)`, eager only). | Gaudi2's MME AGU rejects most `Conv` nodes with a spatial kernel dimension above 256. Not all: a card sweep of the native op (forward and backward against CPU float64, one case per process; [table below](#large-kernel-convolutions-which-run-natively)) finds one class that runs, one input channel with the kernel wide only in its last dim, and there the result depends on the mode: eager runs every even width from 258 to 6000 at strides >= 32 and fails every odd width tried (257, 351, 1089, 1091, 1501); lazy runs every width from 257 to 6000 at strides >= 8, and stops the card at stride 1 when the input is as wide as the kernel. In lazy mode a native conv1d (4, 8, 1000) x (16, 8, 256) fails as well, and kernels 129..255 give a wrong input gradient (relative error 1.3 to 1.5 at k = 144 over 260, 240 over 300, 255 over 257) when the input is longer than 256 but shorter than twice the kernel. | eager + compile + lazy: `aten.convolution.default` at `AutogradHPU` and `HPU` splits spatial kernel dims `> 256` (lazy mode: `> 128`) into chunks of at most that size with exact input slicing and `F.pad`, summing partial outputs (16-bit inputs: the chunks are convolved and summed in `float32`, the bias added once after the sum, then rounded once, as ATen's CPU convolution; lazy mode pads a 2-D kernel's shorter last chunk with zero taps to the full chunk, since native lazy (44, 44) over (54, 54), (44, 128) over (54, 138) and (54, 54) over (64, 64) stop the card; a chunk wholly in the padding is kept, so `0 * inf` is NaN as natively). It leaves to the native op the measured class (`conv.NATIVE_WIDE`): 2-D float32, groups 1, one input channel, no padding or dilation, `kH <= 22`, only `kW` above the bound and no wider than measured; eager: `kW` even up to 6002 and stride `>= 32`, or `kW` even from 512 to 2000 with `kH <= 4` and stride `>= 8`; lazy: `kW` from 257 to 6000 and stride `>= 8`. TensorCSPNet's default `temporal_block` (`(4, 4356)`, stride `(1, 484)`) stays native in both modes, its `dims=(22, 11, 11, 11)` one in lazy mode. Compiled graphs (not measured) split as before, `inference_mode()` ones included. | Bridge 1.21.4, 20 synced steps after 3 warm-up, each run in its own process. Without the fix `STaRNet(64)` fails in eager, lazy and compile; with it 266.0 ms/step eager, 92.0 lazy, 136.4 compiled. `TensorCSPNet(22, k=11)` fails eager without it; with it 140.0 eager (split), 35.3 lazy (native), 57.9 compiled (split). First loss the same in every mode (`1.39551`, `1.47586`). Models whose convolutions run natively are within noise of main, eager and lazy (medians of alternating rounds, this fix on top of main against main): default-dims `TensorCSPNet(22)` 197.5 / 200.0 ms eager, 162.8 / 161.7 lazy (5 rounds), `STaRNet(22)` 100.3 / 122.5 eager, 54.4 / 50.7 lazy, `SPDNet` 34.2 / 44.8 eager (5 rounds), 17.1 / 16.5 lazy; per-process spread 10-30%. Before this change the default-dims TensorCSPNet split ran 405-485 ms/step lazy with 70 `unfold` fallbacks; now 2, as on main. |
| `torch.quantile` / `torch.nanquantile` abort lazy mode (`FATAL ERROR :: MODULE:PT_BRIDGE ... Trying to resize tensor with non-existing TensorImpl`, from `resize_impl_hpu_`). Eager, a slice with a NaN gives a number (`quantile([3, nan, 1, 2], 0.5)`: 2.5, torch: nan; `nanquantile`: 2.5, torch: 2). | ATen's composite writes into an empty `out` tensor that it resizes; Habana's lazy resize of it fails in the accumulation thread. Gaudi's `sort` returns NaN as `+inf`, which the composite then reads as a value. | eager + compile + lazy: `quantile` / `nanquantile` (float and tensor `q`) at `AutogradHPU` and `HPU` as ATen's own steps (sort, ranks, the five interpolations, `gather`, `lerp`) into a fresh tensor; the sort never sees a NaN (torch's order is rebuilt from two NaN-free sorts) and the values are gathered from the input, so the gradient reaches the elements torch's does and the backward buffer is input-sized, not one per `q`. `out=` overloads copy the fresh result. Arguments ATen rejects go to ATen; a slice past 2^24 raises ATen's error; a traced tensor `q` is range-checked at run time (`_assert_async`). | Bridge 1.21.4, each case in its own process, against CPU float64. Lazy without the fix: FATAL for float `q`, tensor `q`, `dim` + `keepdim` + midpoint, and NaN inputs (the backward of a `requires_grad` input ran). With it, all run, error <= 1.1e-7, no CPU fallback, NaNs where torch has them. Eager: native 0.0106 off with a NaN, the rewrite 3.7e-9. Cost, `quantile(x, 0.9, dim=1)` of an 8x300 tensor, 30 synced calls: lazy 0.93 / 1.94 ms (native fails), eager 3.41 / 3.08 ms against native 0.81 / 1.86 ms (two processes). After the #31 review (two NaN-free sorts, values gathered from the input; `repro/probe_review31.py perf`, alternating processes): eager 6.0 / 6.8 / 8.7 ms against the previous rewrite's 3.6 / 5.3 / 3.7 ms (forward + backward of 3 quantiles 13.5 / 14.2 / 17.6 against 9.6 / 12.7 / 12.5), lazy 1.6 / 1.2 / 1.4 against 2.0 / 1.1 / 0.75 (forward + backward 4.0 / 3.5 / 2.5 against 3.7 / 4.6 / 3.7); no CPU fallback. Backward of 257 quantiles of 64x4096: peak 11.1-13.0 MB against native 9.8-11.6 MB. A traced tensor `q` outside [0, 1] gives NaN on Gaudi (it has no `_assert_async` kernel; the check as an assert cost one CPU fallback per call, 1.02 ms against 0.12). |
| `torch.bincount` of large inputs is slower than one CPU core, and grows steeply with the bins (#21): 409600 elements with 64 / 1000 / 4096 bins take 7.4 / 230 / 427 ms eager (CPU core: 0.8 ms). A negative value is counted instead of raising. | Gaudi's kernel; `histc` of the same data takes ~0.1 ms. | eager + lazy: `bincount` of 1-D integer inputs of 65536 to 2**24 elements, no weights, at most 4096 bins, at `AutogradHPU` as one `aminmax` read (Habana's kernel syncs the host too) and `histc` over integer bounds in chunks of 256 bins, int64 counts; a negative value raises ATen's error. Smaller inputs, weights, more bins, traced graphs keep Habana's kernel. | Bridge 1.21.4, through the dispatcher, alternating install/undo in one process, isolated synced calls / a batch of 30: 65536 x 64 bins eager 1.04x / 1.11x, lazy 1.89x / 1.98x; 409600 x 64 eager 2.0x / 2.7x, lazy 5.5x / 6.2x; 409600 x 1000 eager 32x / 43x; 409600 x 4096 eager 24x / 26x. Exact counts in every case, 0 CPU fallbacks. At 4096 elements (not taken) and 16384 the Python kernel in the path costs up to ~30 us per lazy isolated call (0.94x); other rows 0.84-1.20x, within the eager noise. |
| `torch.histc` with a bound given as a Python float (any value: `histc(x, 64, 0.0, 63.0)` too) fails in eager and lazy mode (`expected int` from `habana::FillHistcParams`; eager: in the launch thread, the device stops and the process dies), compiled graphs abort (`HabanaLaunchOpPT`), and from about 350 bins its graph does not compile, whatever the bounds (`repro/histc_float_bounds.py`, `repro/probe_histc_scope.py`). | Habana's kernel reads the bounds as ints. Its float32 division is not IEEE (154 of 2^20 truncations of `(x - L) * bins / (R - L)` differ from the CPU's), so the CPU's bins cannot be computed on the card. | eager + compile + lazy, inference mode too (`histc` at `AutogradHPU` and, overriding Habana's, at `HPU`; `histc.out` at `AutogradHPU`), for float32 / bfloat16 / float16 / float64 inputs: calls with a float bound or more than 300 bins. Each bin's lower edge is found on the host with ATen's CPU arithmetic (bisection over the dtype's bit patterns; `inf` for a bin no value reaches; cached per bounds, bins, dtype and device, 64 tables, released by `undo()`); on the card one `bucketize` against those edges, values outside `[min, max]` and NaN not counted, the last bin closed; counts through Habana's `histc.out` with int bounds in chunks of 256 bins, or `scatter_add_` above 256, capped as ATen's one-thread kernel caps them. Equal bounds take the data's range (one host read, as ATen's `.item()`), NaN / inf there and bounds past the dtype's range raise ATen's errors; in compiled graphs equal bounds are read on the card (see Known gaps). float64: the float32 values the card holds, read through an offset-free view (eager float64 views with an offset read wrong, see "float64"), against float64 edges rounded up. Calls with int bounds and at most 300 bins keep Habana's kernel. | Bridge 1.21.4, against ATen's CPU kernel with one thread on the values the card holds (values at, above and below every edge, NaN, inf, views, float64 offset views, 1 to 4096 bins, the four dtypes, inference mode, `out=`, and the Codex review's inputs: a range of 1e-37, float64 2^27 with equal bounds, a bfloat16 bin no value reaches, 16-bit count saturation, bound overflow; `tests/check_histc_hpu.py`, one compiled graph per case): bit for bit in eager, lazy and compiled (`hpu_backend`), 0 CPU fallbacks; natively each such call failed. Cost, 32x22x1000 float32, median of 3 processes, isolated / batch of 30: 64 bins eager 2.4 / 0.90 ms (native with int bounds 1.6 / 0.46), lazy 1.9 / 3.4 (0.73 / 1.6); 256 bins eager 4.5 / 2.8, lazy 4.1 / 4.3; 1000 bins eager 7.2 / 5.5, lazy 6.9 / 4.1; 4096 bins eager 21 / 19, lazy 20 / 4.7 (Gaudi's `bucketize` is linear in the edges); equal float bounds eager 5.0 / 4.0, lazy 2.1 / 1.8. |
| einsum graphs fail to compile (BIOT: `Can not compile graph`). | `pass_remove_unnecessary_bmm_view` removes views that are not pure batch merges. Unfixed in bridge 1.24.2. | `enable()` sets Habana's `PT_HPU_DISABLE_pass_remove_unnecessary_bmm_view`, Intel's own workaround (vllm-gaudi #1102). | BIOT compiles. |
| LaBraM fails to compile (`used before defined`). | `batch_as_strided` puts a `getitem` before the node it reads. Fixed in bridge 1.22. | `enable()` sets `PT_HPU_DISABLE_pass_batch_as_strided_groups`. | LaBraM compiles. |
| FBCNet and FBMSNet fail to compile (`create tensor invalid permutation 4 2`), EEGNet and Deep4Net at times. | `pass_weight_permutation` puts an `hpu.weight_permutation` before every 2-D and 3-D convolution weight. The backward of some graphs then fails to create a tensor. | `enable()` sets `PT_HPU_DISABLE_pass_weight_permutation`. | All four compile (EEGNet and Deep4Net: 8 of 8 times). CBraMod steps 2.2% faster, CodeBrain the same. |
| `linalg.solve`, `solve_ex`, `det` and `slogdet` run on the CPU (#10). | No HPU kernel. Rebuilt from torch ops, they were measured 3 to 25 times slower than the fallback per call: one launch per op. | A TPC Gauss-Jordan kernel with partial pivoting, n <= 64, right-hand sides in chunks of 64. Gradients are torch's formulas; double backward works. | n = 22, batch 32, mean ms per call in two processes (`tests/linalg_percall.py`), eager: `solve` 2.9 / 1.3 against the fallback's 6.1 / 2.5; `det` 0.67 / 0.69 against 2.7 / 1.2; `slogdet` 2.8 / 0.84 against 2.7 / 1.3. Lazy: `solve` 0.97 / 1.36 against 1.22 / 1.38; `det` 0.53 / 0.75 against 0.68 / 0.75; `slogdet` 2.3 / 2.9 against 1.1 / 0.8 (slower: its first round after warm-up took 5-7 ms). In lazy mode the per-call gain is within the card's noise; the gain shows in the training steps below. |
| `linalg.inv` runs on the CPU (#11). | No HPU kernel. | The same Gauss-Jordan kernel. | n = 22, batch 32, same runs, eager: 1.96 / 1.22 ms against 4.33 / 2.02; lazy: 1.14 / 1.24 against 1.14 / 1.52. |
| `linalg.eigh` and `eigvalsh` run on the CPU (#8; 21 of 21 SPD models). | No HPU kernel. The parallel-Jacobi composite was measured slower than the fallback, so it only serves the CPU stand-in. | A TPC one-sided Jacobi kernel, n <= 32. | n = 22, batch 32, same runs, eager: 0.81 / 2.64 ms against 3.92 / 2.25 (the second process was slower than the fallback); lazy: 0.54 / 1.00 against 2.42 / 2.69. Batch 1024 (older measurement): 3.7 ms against 52 ms. Training steps: see below. |
| `linalg.svd`, `svdvals`, `pinv`, `lstsq`, `matrix_norm` (2, -2, nuc) and `matrix_rank` run on the CPU (#9). | No HPU kernel (`_linalg_svd.U`, `linalg_lstsq.out` fall back). | A TPC one-sided Jacobi SVD kernel, m, n <= 32 (thin, full-U and values-only variants); `pinv` and full-rank `lstsq` (min-norm through `pinv`, rank-deficient declined) and the norms and rank from it. Gradients are torch's formulas. | n = 22, batch 32, same process, eager / lazy, against Habana's fallback: `svd` 0.80 / 2.16 ms against 3.90 / 5.14; `svdvals` 0.38 / 1.05 against 1.50 / 1.58; `pinv` 1.05 / 1.55 against 6.49 / 5.07; `lstsq` 2.05 / 1.54 against 2.74 / 1.61. `matrix_norm(ord='nuc')` is at the accuracy bar: 1.33e-6 relative error against 1.31e-6 for 10x CPU float32. `atol`/`rtol` follow torch (`atol <= 0` keeps the default `rtol`). Any nonzero matrix is scaled by 1/max|a| before the rotations, also a tiny one: singular values within 3e-7 down to max|a| = 1e-35, 1.4e-3 at 1e-36 to 3e-37 and 2e-2 at 1e-37 (measured; before review round 3 they were all 0 below 1e-36). Known limits: the card flushes float32 subnormals to zero, so a matrix of entries all below 1.2e-38 has singular values 0 and rank 0 (torch: the true values); a matrix with NaN or inf gives non-finite values where torch's `svd` and `pinv` raise `LinAlgError` (a check would cost a host sync per call; torch's `svdvals` returns NaN too); the SVD backward with exactly repeated singular values sets the 1/(s_i² − s_j²) terms to 0, so its gradient is finite where torch's is inf/NaN (a loss of S alone is unaffected); at condition 1e5 `svdvals` is 13x CPU float32 (1.55e-6). Under `torch.compile`, `lstsq` is taken only with `driver='gels'` (rank and residuals of the other drivers depend on the values). |
| `linalg.matrix_exp` and `householder_product` run on the CPU (#13): the `matrix_exp` and `householder` maps of `orthogonal` parametrizations. | No HPU kernel. | TPC kernels: Padé-13 scaling and squaring (and its adjoint Fréchet derivative for the backward), and the product of reflectors, m <= 64. The compact-WY composite was measured slower and is a knob only. Each matrix is scaled by its own 2^-s, s up to 126, which covers float32's range (16 before review round 3: a 1-norm past 3.5e5 gave NaN). The first 16 squarings are always run, masked; more run only for the matrices that need them, their count read back from the output by one scalar load. NaN or inf input gives non-finite output, as torch's. | n = 22, batch 32, eager / lazy: `matrix_exp` 0.33 / 0.46 ms against 1.79 / 1.63; with backward 1.49 / 2.87 against 10.7 / 8.4; `householder_product` 0.27 / 0.40 against 2.35 / 0.56. An `orthogonal(Linear(22, 22))` forward and backward: matrix_exp map 1.61 / 2.77 ms against 5.94 / 3.06, householder map 9.8 / 5.3 against 80 / 21. |
| `linalg.lu_factor`, `lu_factor_ex`, `lu_solve` and `lu_unpack` run on the CPU in eager mode (#10). | No HPU kernel. | TPC kernels, n <= 32: LU with partial pivoting (LAPACK's packed factors, pivots and info), pivots to permutation, and getrs (row swaps and two substitutions). Adjoint and right-side `lu_solve` use the factors directly: two triangular solves (the substitution kernel) and a matmul with P, never A = P L U rebuilt; the LU backward is torch's P L^-H M U^-H by the same two solves. A multiplier that flushes below 2^-126 is carried scaled against the pivot row, so exactly dependent rows still give a zero pivot. Installed in eager mode only: in lazy mode, and for `linalg.lu` in both modes, they were measured at parity or slower than the fallback, so Habana's ops stay. `HPU_OPS_LU_SLOW=1` installs them all anyway (no fallbacks, at that speed). The pivot is the first largest candidate, LAPACK's rule, also on exact ties; pivots of magnitude 2^64 and above are scaled by 2^-64 before the reciprocal, so 1/pivot is not flushed to zero past 2^126. Pivots are trusted: out-of-range pivots give finite garbage on the card, not torch's error. NaN input gives non-finite factors, as torch's; the pivots may then differ. | n = 22, batch 32, eager, same process (each includes the `lu_factor`): `lu_factor_ex` 0.27–0.46 ms against 0.69–2.65 (2.6x to 6.9x: the fallback's time varies between jobs); `lu_solve` 2.28 against 6.50; `lu_unpack` 2.87 against 4.54. Not installed by default: lazy `lu_factor_ex` 0.94–0.98x, lazy `lu_solve` 0.96x, lazy `lu_unpack` 0.91x, `linalg.lu` 0.87x eager and 0.77x lazy (medians). |

The pass switches are read at every compile. A non-empty setting in your environment wins (`PT_HPU_DISABLE_<pass>=0` keeps a pass on).

### Large-kernel convolutions: which run natively

Native `F.conv2d` forward and backward (output and both gradients against CPU float64, relative error < 1e-3; "no x grad": weight gradient only) on Gaudi2, bridge 1.21.4, one case per process: `bash repro/sweep_conv_native.sh "B C H W O KH KW SH SW [XGRAD]"` (`PT_HPU_LAZY_MODE=0|1`), raw results in `repro/sweep_conv_native_results.jsonl`, this table from `python3 repro/sweep_conv_native_table.py repro/sweep_conv_native_results.jsonl`. Strides are along W unless written `sH,sW`. A failure is `setAssociatedDimAndSize` (eager) or a lazy graph compile failure; "died" is a device critical error that stopped the card. `conv.install_large_kernel` leaves to the native op only the cells this table backs (see `conv.NATIVE_WIDE`) and splits the rest.

| x (B, C, H, W) | w (O, C, kH, kW) | eager runs | eager fails | lazy runs | lazy fails |
|---|---|---|---|---|---|
| 8, 1, 4, 257 | 16, 1, 4, 257 |  | 1, 8, 32, 121, 256, 257, 484 | 8, 32, 121 | 1 (died) |
| 8, 1, 4, 514 | 16, 1, 4, 257 |  | 1, 8, 32, 121 | 1, 8, 32, 121 |  |
| 8, 1, 4, 258 | 16, 1, 4, 258 | 32 |  | 32 |  |
| 8, 1, 4, 300 | 16, 1, 4, 300 | 32 |  | 32 |  |
| 8, 1, 4, 351 | 16, 1, 4, 351 |  | 1, 8, 32, 121, 256, 257, 484 | 8, 32, 121 | 1 (died) |
| 8, 1, 4, 702 | 16, 1, 4, 351 |  | 1, 8, 32, 121 | 1, 8, 32, 121 |  |
| 8, 1, 4, 384 | 16, 1, 4, 384 | 32 |  | 32 |  |
| 8, 1, 4, 512 | 16, 1, 4, 512 | 8, 32, 121 | 1 | 8, 32, 121 | 1 (died) |
| 8, 1, 4, 1024 | 16, 1, 4, 512 | 8, 32, 121 | 1 | 1, 8, 32, 121 |  |
| 8, 1, 4, 600 | 16, 1, 4, 600 | 8, 32, 121, 600 | 1 | 8, 32, 121 | 1 (died) |
| 8, 1, 4, 1200 | 16, 1, 4, 600 | 8, 32, 121 | 1 | 1, 8, 32, 121 |  |
| 8, 1, 4, 800 | 16, 1, 4, 800 | 8, 32, 121 | 1 | 8, 32, 121 | 1 (died) |
| 8, 1, 4, 1600 | 16, 1, 4, 800 | 8, 32, 121 | 1 | 1, 8, 32, 121 |  |
| 8, 1, 4, 1024 | 16, 1, 4, 1024 | 8, 32, 121 | 1 | 8, 32, 121 | 1 (died) |
| 8, 1, 4, 2048 | 16, 1, 4, 1024 | 8, 32, 121 | 1 | 1, 8, 32, 121 |  |
| 8, 1, 4, 1088 | 16, 1, 4, 1088 | 32 |  | 32 |  |
| 8, 1, 4, 1089 | 16, 1, 4, 1089 |  | 1, 8, 32, 64, 121, 256, 257, 484, 1089 | 8, 32, 64, 121, 484, 1089 | 1 (died) |
| 8, 1, 22, 1089 | 16, 1, 22, 1089 |  | 121 | 121 |  |
| 32, 1, 4, 1089 | 16, 1, 4, 1089 |  | 121 | 121 |  |
| 8, 1, 4, 1089 (no x grad) | 16, 1, 4, 1089 |  | 121 | 121 |  |
| 32, 1, 4, 1089 (no x grad) | 16, 1, 4, 1089 |  | 121 | 121 |  |
| 8, 1, 4, 2178 | 16, 1, 4, 1089 |  | 1, 8, 32, 121 | 1, 8, 32, 121 |  |
| 8, 1, 4, 1090 | 16, 1, 4, 1090 | 32 |  | 32 |  |
| 8, 1, 4, 1091 | 16, 1, 4, 1091 |  | 32 | 32 |  |
| 8, 1, 4, 1200 | 16, 1, 4, 1200 | 8, 32, 121 | 1 | 8, 32, 121 | 1 (died) |
| 8, 1, 4, 2400 | 16, 1, 4, 1200 | 8, 32, 121 | 1 | 1, 8, 32, 121 |  |
| 8, 1, 4, 1500 | 16, 1, 4, 1500 | 8, 32, 64, 121, 484, 1500 | 1 | 8, 32, 64, 121, 484, 1500 | 1 (died) |
| 8, 1, 22, 1500 | 16, 1, 22, 1500 | 32 |  | 32 |  |
| 32, 1, 4, 1500 | 16, 1, 4, 1500 | 32 |  | 32 |  |
| 8, 1, 4, 1500 | 64, 1, 4, 1500 | 32 |  | 32 |  |
| 8, 1, 4, 3000 | 16, 1, 4, 1500 | 8, 32, 121 | 1 | 1, 8, 32, 121 |  |
| 8, 1, 4, 1501 | 16, 1, 4, 1501 |  | 32 | 32 |  |
| 8, 1, 4, 2000 | 16, 1, 4, 2000 | 32, 64, 121, 484, 2000 |  | 32, 64, 121, 484, 2000 |  |
| 8, 1, 4, 2048 | 16, 1, 4, 2048 | 32, 64, 121, 484, 2048 |  | 32, 64, 121, 484, 2048 |  |
| 8, 1, 4, 4096 | 16, 1, 4, 2048 | 64, 484 |  | 64, 484 |  |
| 8, 1, 4, 2500 | 16, 1, 4, 2500 | 32, 64, 121, 484, 2500 |  | 32, 64, 121, 484, 2500 |  |
| 8, 1, 4, 3000 | 16, 1, 4, 3000 | 32, 64, 121, 484, 3000 |  | 32, 64, 121, 484, 3000 |  |
| 8, 1, 4, 4095 | 16, 1, 4, 4095 | 4095 |  |  |  |
| 8, 1, 4, 4096 | 16, 1, 4, 4096 | 4096 |  |  |  |
| 8, 1, 4, 4097 | 16, 1, 4, 4097 | 4097 |  |  |  |
| 8, 1, 4, 4356 | 16, 1, 4, 4356 | 32, 64, 120, 121, 256, 257, 484, 4356 | 1, 2, 8, 16 | 8, 16, 32, 64, 121, 484, 4356 |  |
| 32, 1, 4, 4356 | 16, 1, 1, 4356 | 484 |  | 484 |  |
| 8, 1, 22, 4356 | 16, 1, 22, 4356 | 484 |  | 484 |  |
| 8, 1, 4, 4356 | 64, 1, 4, 4356 | 484 |  | 484 |  |
| 32, 1, 4, 4356 | 16, 1, 4, 4356 | 484 |  | 484 |  |
| 32, 1, 4, 4356 (no x grad) | 16, 1, 4, 4356 | 484 |  | 484 |  |
| 8, 1, 4, 5000 | 16, 1, 4, 4356 | 484 |  |  |  |
| 8, 1, 4, 8712 | 16, 1, 4, 4356 | 484 | 1 | 1 |  |
| 8, 1, 4, 4358 | 16, 1, 4, 4358 | 32 |  | 32 |  |
| 8, 1, 4, 6000 | 16, 1, 4, 6000 | 32, 64, 121, 484, 6000 |  | 32, 64, 121, 484, 6000 |  |
| 8, 1, 4351, 2 | 8, 1, 4351, 1 |  | 4351,1 |  |  |
| 8, 1, 351, 1000 | 8, 1, 351, 1 |  | 351,1 |  |  |
| 32, 1, 351, 1000 | 8, 1, 351, 1 |  | 1 |  | 1 |
| 32, 1, 351, 1000 (no x grad) | 8, 1, 351, 1 |  | 1 |  | 1 |
| 8, 1, 4356, 4 | 16, 1, 4356, 4 |  | 484,1 |  | 484,1 |
| 8, 4, 4, 4356 | 16, 4, 4, 4356 |  | 484 |  | 484 |

### Linalg kernels on the TPC

The linalg rows above run as single TPC kernels, one matrix per index-space member. They live in `hpu_ops/tpc/`. The library, `liblinalg_tpc.so`, is prebuilt. To rebuild it inside the Habana image, run `hpu_ops/tpc/build.sh`.

- **Kernels:** Gauss-Jordan with partial pivoting for `solve`, `inv`, `det` and `slogdet` (n <= 64 with the rows in local memory, 64 < n <= 512 an in-place inverse, below; LU-style elimination with the pivot row left unnormalized, so exact duplicate rows cancel exactly and report a zero pivot; B is divided by the pivots only at the end; multipliers below 2^-126 are carried scaled, so they are not flushed); forward and back substitution for `solve_triangular` and `cholesky_solve` (two substitutions, never L·Lᴴ; each product subtracted from the right-hand side in turn, in LAPACK's order, never summed first); one-sided Jacobi for `eigh` and `eigvalsh`; one-sided Jacobi SVD for the SVD family; Padé-13 for `matrix_exp`; reflectors for `householder_product`; LU, pivots-to-permutation and getrs for the LU ops (eager mode by default, see the LU row). The SVD, `matrix_exp` and LU kernels take n <= 32 (`householder_product` m <= 64). The Gauss-Jordan, substitution and `matrix_exp` kernels use no LUT-based special function (the reciprocal is Newton steps from a bit-level guess), which leaves the 80 KB of local memory for n <= 64; the eigh, SVD and LU kernels use the hardware rsqrt, reciprocal or lane reductions (16 KB of local memory, enough for n <= 32).
- **Past n = 64:** eager mode only: in lazy mode (and under `torch.compile`) these calls keep main's route, Habana's fallback, since EEGSPDNet's lazy step was measured slower with the kernel (73.2 against 63.0 ms median). In eager mode (`lazy_mode()` false, and not traced: `torch.compile`, AOT autograd and `make_fx` keep main's route), `solve`, `solve_ex`, `inv`, `det` and `slogdet` of float32 matrices with 64 < n <= 512 run on `gj_large`, an in-place Gauss-Jordan inverse with partial pivoting over the whole height, applied to A^T: torch's CPU kernels factor the row-major A as LAPACK's column-major A^T, so the pivot order is theirs (on A, Wilkinson's well-conditioned matrix overflowed at n = 130; torch overflows on its transpose, and so does the kernel). Each pivot row is stored normalized by the power of two of its pivot (exact), so the quotients of rows already used stay bounded (implicit: the pivot row is found while the previous step's pass runs, no row moves). One matrix per TPC; the first rows stay in local memory, the others are re-read from global memory on each step (one ASO per step), one ELF per ceil(n / 64). `solve` is a matmul with the inverse (plus the residual step), and its backward reuses it (matmuls only, no kernel). A NaN in a matrix gives NaN outputs, as LAPACK's. The elimination is LU-style, as for n <= 64 (the pivot rows are divided by the pivots only at the end, by torch), so entries / pivot below 2^-126 and pivots up to the float32 maximum stay right; `solve` takes one residual step `X += A^-1 (B - A X)` on the MME. A subnormal pivot counts as zero (the card flushes it): `info` is set and that row of the inverse is 0, where torch on the CPU returns huge or non-finite values. Past n = 256 the kernel takes only batches of 8 or more (`tpc.large_pays`): batch 1 at n = 384 and batch 1 to 4 at n = 512 were slower than Habana's CPU fallback. Per call, mean of synced calls, eager / lazy, kernel against Habana's fallback: batch 32, n = 220, `solve` (3 right-hand sides) 8.2 / 8.7 ms against 17.8 / 13.9, `inv` 7.3 / 8.3 against 26.2 / 22.6, `det` 5.9 / 5.4 against 13.2 / 12.2; batch 32, n = 128, `solve` 5.7 / 3.4 against 11.4 / 8.9; batch 1, n = 220, 220 right-hand sides (EEGSPDNet's Cayley map), `solve` 4.6 / 3.8 against 5.2 / 3.5, `det` 3.1 / 4.7 against 2.0 / 1.8 (measured before the review fixes; not faster per isolated call at batch 1; EEGSPDNet's training step: see below).
- **Triangular solves:** `linalg.solve_triangular` and `cholesky_solve`, and so the backward passes of `cholesky` and `qr`, run on the substitution kernel. Batch 32, every variant (upper/lower, left/right, unit diagonal, condition up to 1e3 per triangle, 1e6 for `cholesky_solve`), n = 1 to 64: within 5.3x of CPU float32's error, eager and lazy, no fallback. Per call, eager: 0.35 to 0.44 ms against about 2 ms for Habana's fallback; lazy: 0.31 to 0.34 ms against 0.14 to 0.50 (the fallback is faster per isolated call at n <= 40 in lazy mode; the kernel needs a graph boundary). Each figure is the mean of 30 synced calls (`tests/tri_solve_probe.py`).
- **Backend kernels:** `solve`, `det` and `slogdet` register only at the autograd key. Below autograd Habana's own kernel runs. A call a tracer sees declined (n > 64 under `torch.compile`, AOT autograd or `make_fx`, or a symbolic shape) is recorded as the native op with a backward from `A` and the outputs only (`solve(A^H, g)`, `det A^-H`, `A^-H`), never from the LU factors: `hpu_backend` runs that node eagerly, through the autograd kernel, where the large route (or the now static shape) returns placeholder factors. Before, torch's traced derivative read them: `STaRNet(64)`'s n = 96 Cayley solve trained to NaN compiled.
- **Limit:** float32 matrices with n <= 64 (`eigh`, the SVD family, `matrix_exp` and the LU ops: 32) on a Gaudi2. Calls the kernels do not take (n > 64, float64, kernels off) of `inv`, `eigh`, `eigvalsh`, `solve_triangular` and `cholesky_solve` run through hpu-ops' own CPU route (`hpu_ops.linalg._on_cpu`: the tensors are copied to the CPU, the op runs there, the result is copied back), which is not Habana's fallback. `solve`, `solve_ex`, `det` and `slogdet` register only at the autograd key, so such calls reach Habana's own kernel and its CPU fallback, which the bridge counts (measured at n = 128: `_linalg_solve_ex`, `_linalg_det`, `_linalg_slogdet`). Float64: see "float64".
- **Value range:** the kernels work in the normal float32 range. The card flushes subnormals to zero, on input and on output, while Habana's CPU fallback keeps them: a subnormal entry counts as zero, so `solve` of a matrix whose entries are all subnormal (e.g. an SPD matrix scaled by 1e-38) raises the singular-matrix error that the CPU would not, a subnormal `det` is 0, and a subnormal solution of `solve_triangular` is 0 (all measured on Gaudi2, bridge 1.21.4). Huge pivots and diagonals (up to the float32 maximum) are right: the reciprocal is taken of the value scaled by 2^-64 when it is >= 2^64, so it never becomes subnormal.
- **Overloads:** `linalg.solve_triangular(..., out=)` and the legacy `torch.triangular_solve`, also with `out=` (its `solution`, with `transpose`; `cloned_coefficient` is a copy of `A` broadcast to the batch, differentiable as torch's) take the same route as `linalg.solve_triangular`. As in torch, an `out` of another dtype or one that requires grad raises, a wrong-sized `out` is resized to torch's layout with torch's warning, and tensors on two devices raise.
- **Order:** call `hpu_ops.enable()` before the first HPU op. Habana reads the kernel path when the device initialises; later, the kernels are off, with one warning, and the calls take the CPU route above.
- **Opt-out:** `HPU_OPS_TPC=0` in the environment keeps the kernels off. They are also off on a device that is not a Gaudi2, and when `liblinalg_tpc.so` is missing (one warning).
- **Transforms:** under `torch.func` transforms and forward-mode AD, every linalg rewrite (including the SVD family, `pinv`, `lstsq`, `householder_product`, `matrix_exp` and the LU ops) declines and the transformed call goes through torch's own derivative rules (`jacfwd`, `jvp`, `hessian` and forward AD use torch's kernel; for `vmap`, `grad` and `jacrev` the inner calls may still use the TPC kernel; results match).
- **Duplicate rows (differs from torch):** exact duplicate rows of general values (or power-of-two multiples) cancel exactly on the card, so `solve_ex`/`inv_ex` report `info != 0` and `solve`/`inv` raise the singular-matrix error, where torch on the CPU with macOS Accelerate leaves a rounding residue in LAPACK's multiplier and returns `info = 0` with a huge solution (measured at n = 2, 3, 5, 17, 63 with a random row duplicated; Linux MKL / OpenBLAS cancel exactly too, `info = 2`). Other rank-deficient matrices (e.g. an integer row that is the sum of two others) are detected or not as in torch: neither checks a tolerance.
- **Singular triangles:** `solve_triangular` and `cholesky_solve` of a triangle with a zero diagonal entry give non-finite values for that matrix, as torch's do (substitution divides by it). `det` of a singular matrix is 0 on the card and its gradient equals torch's (measured on Gaudi2), also under `create_graph` (torch's SVD formula, its SVD on a CPU copy: Gaudi has none). `slogdet` of a singular matrix gives `logabsdet = -inf`, as torch does, but its gradient is finite where torch's is non-finite, so a finiteness check on that gradient does not flag a singular input.
- **Lazy mode:** works. Each kernel call ends its lazy graph, and its outputs are copied: on bridge 1.21.4 a custom op's output read with `.cpu()` and no ATen op after it comes back wrong (`repro/lazy_custom_op_output_cpu.py`). The mode is Habana's own `is_lazy()`; bridge 1.21.4 runs eager when `PT_HPU_LAZY_MODE` is unset.
- **Accuracy:** within 10x of CPU float32 against float64 (`eigh` within about 3.8x), also with dense repeated eigenvalues, condition 1e6 and inputs scaled by 1e30. Measured at n = 16 and 22. The gate runs 130 cases, including n = 1, 3, 21, 31 and 32 for the eight kernel ops and an indefinite `eigh` and `eigvalsh` at n = 3, 22 and 32: 130 of 130 pass in eager mode and 129 of 130 in lazy mode, measured before the 1x1 change below. The one lazy failure is the backward of `qr` at n = 22 (error 6.11e-6 against a bar of 6.05e-6), the known QR gap below. `eigh` and `eigvalsh` at n = 1 do not use the kernel (its graph does not compile at n = 1 in lazy mode): they return the entry and 1.
- **Accuracy:** within 10x of CPU float32 against float64 (`eigh` within about 3.8x), also with dense repeated eigenvalues, condition 1e6 and inputs scaled by 1e30. Measured at n = 16, 22 and 64. The gate (`tests/linalg_on_hpu.py`) runs 173 cases, including n = 1, 3, 21, 31 and 32 for the eight kernel ops, the triangular solves on both sides, unit diagonal, ill-conditioned, 2-D, chunked (70 right-hand sides) and NaN in the unread triangle, and an indefinite `eigh` and `eigvalsh` at n = 3, 22 and 32: 173 of 173 pass in eager mode and 172 of 173 in lazy mode. The one lazy failure is the backward of `qr` at n = 22 (see Known gaps). `eigh` and `eigvalsh` at n = 1 do not use the kernel (its graph does not compile at n = 1 in lazy mode): they return the entry and 1.

## Measured on Gaudi2 (habana 1.21.4 / torch 2.6.0)

### SPD models: training steps with the linalg kernels

Same pod, batch 32, ms per training step, `origin/main` to this branch.

| model | lazy | eager |
|---|---|---|
| SPDNet | 24.0 to 18.9 | 64.0 to 59.6 |
| TensorCSPNet | 335.9 to 178.2 | 364.8 to 200.2 |
| EEGSPDNet | 86.5 to 73.7 | 234.3 to 199.5 |
| TSMNet | 60.1 to 51.2 | 165.6 to 166.7 |

Measured in one session on one pod (Gaudi2, bridge 1.21.4, BNCI2014_001), `tests/linalg_steps.py`: 20 steps after 5 warm-up steps, the four runs back to back on the current code. EEGSPDNet's `solve` (n = 220, batch 1) now runs on the in-place kernel: 0 linalg fallbacks per step (4 before: 2 `_linalg_solve_ex`, 2 `linalg_lu_solve`). Alternating A/B runs in one checkout (`tests/linalg_steps.py --models EEGSPDNet`, against `--baseline --large-maxn 64`, main's route past n = 64), after the review fixes, 3 rounds per mode, median ms per step: eager 189.5 against 205.4; lazy 73.2 against 63.0 (slower in lazy mode; the card's noise is about ±20 ms eager, ±8 ms lazy). GBWAtt fails in lazy mode on both (#20); eager, 2499.7 to 2544.5 ms. TSMNet (its `solve` has n = 40, now on the kernel, no fallback) is the same in eager mode in this run (165.6 against 166.7). In an earlier session, three alternating runs per side of 40 steps each gave 56.9 to 45.7 ms lazy and 167.5 to 156.7 eager, the branch faster in each of the 3 pairs; the card's noise is about ±15 ms per step in eager mode.

Model gradients against CPU float64 (`tests/model_parity.py --parts grads`, 21 models, batch 32): in eager mode 20 of 21 within `max(1e-3, 10x CPU float32)`, all at most 2.2x `origin/main`'s error (SPDNet's BiMap weight 5.9e-5, `origin/main` 8.2e-5); Green errors as on `origin/main` (hpu-complex #4). In lazy mode the same 9 models fail with the same errors on `origin/main` and on this branch with or without the `eigh` kernel: a lazy-mode gap outside these kernels.

### Choosing a mode: eager, eager + `torch.compile`, lazy

Same pod (Gaudi2, bridge 1.21.4, BNCI2014_001), batch 32, ms per training step (forward, backward, Adam; `torch.compile(model, backend="hpu_backend")` around the forward, Adam eager), two runs per mode, interleaved eager / compile / lazy (`tests/linalg_steps.py [--compile]`). Warm-up: the first five steps, in s, where compilation happens. 0 CPU fallbacks per step in every run.

| model | eager | eager + compile | lazy | lazy + `torch.compile` (runs uncompiled) | warm-up eager / compile / lazy (s) |
|---|---|---|---|---|---|
| SPDNet | 53.6, 43.6 | 17.5, 29.6 | **12.5, 15.3** | 18.3 | 0.3 / 2.1-3.0 / 1.8-2.1 |
| TSMNet | 164.5, 145.8 | 54.9, 50.7 | 49.8, 50.2 | 48.6 | 1.0 / 12.3-13.9 / 4.4-4.8 |
| TensorCSPNet | 220.0, 200.2 | **151.1, 164.7** | 174.9, 176.4 | 174.5 | 1.1 / 15.0-15.5 / 5.1-5.3 |

Compiling is 1.3-3x faster per step than plain eager for all three, and it costs 2-15 s of compilation. Against lazy: SPDNet trains faster lazy, TSMNet the same, TensorCSPNet about 10% faster compiled. Lazy compiles less (2-5 s) but Habana no longer develops it ("a legacy fallback"), and `torch.compile` cannot run there (after `enable()` it runs the model uncompiled, with one warning, see "Use"). The card's noise between runs is up to ±12 ms on SPDNet.

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

### Small ops slower than one CPU core (#21)

Measured per call on Gaudi2 (bridge 1.21.4) in one process against Habana's kernel, eager and lazy, isolated synced
calls and a batch of 30 calls (`tests/bench_small_ops2.py`, `tests/bench_small_dispatch.py`). Only `bincount` of large
inputs is rewritten (see the table above). The rest keeps Habana's kernel:

| op, size | best rewrite | eager iso / batch | lazy iso / batch | why not |
|---|---|---|---|---|
| `cumsum` 64x22x22, last dim | `x @ triu(ones)` | 1.03-1.22x / 1.42-1.45x | 1.07-1.15x / 1.06-1.08x | turns `[1, inf]` into `[nan, inf]` (`inf * 0`); the exact variants (a `where` select, or the matmul plus an inf/NaN code) 0.02-0.45x |
| `cumsum` 4096, 1-D | blocked matmuls | 0.86-1.19x / 0.36-0.49x | 0.32-0.55x / 0.19-0.35x | slower |
| `logsumexp` 64x22x22, one dim, fwd / fwd+bwd | ATen's composite in one graph | not faster (five launches against one) | fwd 1.24-1.61x / 1.79-1.87x, fwd+bwd 0.86-1.06x / 1.05-1.21x | eager not faster; lazy backward and calls past ~40k elements slower |
| `index_add` (64,22) <- (4096,22) | one-hot `addmm` | 1.21x / 0.86x | 1.66x / 1.22x | eager batch slower; `inf * 0` |
| `scatter_add` 64x22x22, last dim | one-hot `baddbmm` / `where` | 1.08-1.16x / 0.10-0.49x | 0.56-0.57x / 0.42-0.46x | slower |
| `bincount` 4096, 64 bins | a min/max read + `histc` (four exact variants) | 0.73-0.95x / 0.56-0.84x | 0.56-1.08x / 0.79-1.20x | the host read costs as much as Habana's kernel |

Habana bugs found on the way (bridge 1.21.4, eager and lazy unless noted): `amax` / `max` drop NaN (`[nan, 1, 2]` gives
`-inf`; `repro/amax_nan.py`); `bincount` counts negative values instead of raising (`repro/bincount_negative.py`);
`histc` with a float bound aborts the process in eager mode, and does not compile from about 350 bins
(`repro/histc_float_bounds.py`; now rewritten, see the table above).

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
- which large-kernel convolutions are split and which keep the native op, per mode (`tests/test_conv_routing.py`: the model shapes, the parity, stride, channel, orientation, padding, dilation, `kH`, dtype, conv1d and width-ceiling bounds of `conv.NATIVE_WIDE` with the real lazy flag, compiled graphs splitting as before, `inference_mode()` ones included, the lazy chunk shapes against those that stopped the card, 16-bit sums and the bias against the native op, wholly padded chunks with `0 * inf` and double backward, the output layout); the quantile review cases (`tests/test_quantile.py`: NaN rows and the `q` gradient, a genuine `+inf` in `nanquantile`, backward memory, traced `q` range, the 2^24 guard, compiled `inference_mode()`, `out=`, the result layout);
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
- that a boolean-mask put of one value inside `autograd.Function.backward()` followed by `vech`'s `X[..., rows, cols]` (GBWAtt's lazy synStatus 30, #20) matches the CPU with zero CPU fallbacks;
- that large-kernel convolutions (`(351, 1)` and `(4, 1089)` stride `(1, 121)`) match the CPU, eager, compiled, and in `inference_mode()`, via `split_conv`;
- with `hpu_ops.autocast`'s lists exported, CUDA's output dtypes under bfloat16 autocast (without them, Habana's are printed);
- that braindecode's FBCNet compiles (the weight-permutation switch), and that its output and parameter gradients match eager mode.

## Known gaps

- `info`, the index of a singular pivot, follows the kernel's own elimination order and can differ from LAPACK's.
- The backward of `qr` at n = 16 and 22 is 17-20x less accurate than CPU float32 (lazy, n = 22: 6.12e-6 against the 20x bar of 6.05e-6). The cause is Habana's native `linalg.qr` forward (130x CPU float32's error, `repro/habana_qr_n16_accuracy.py`): with CPU float32 QR factors the same backward on the card is within 0.7-1.2x of CPU float32.
- float64 linalg runs on the CPU (see "float64").
- Eager float64 views with a storage offset and eager int64 values past int32 stay wrong on the card (see "float64"); hpu-ops only warns, at the copy to the card. Data made on the card (`torch.zeros(..., dtype=torch.float64, device="hpu")`, int64 arithmetic that leaves the int32 range) is not checked.
- `histc` with int bounds and at most 300 bins keeps Habana's kernel, which is not ATen's at the bin edges: its float32 division puts values within an ulp of an edge in the neighbouring bin (4 of 64 bins differ on [-3, 3] with values at every edge), and 16-bit inputs are binned in float32 where ATen bins in the 16-bit dtype (bfloat16, 64 bins: 40 bins differ). With equal int bounds (`histc(x)`), NaN or inf in the data give counts where ATen raises. `histc` with `out=` inside `torch.inference_mode()` reaches Habana's `histc.out`, which the rewrite needs for its own counting: float bounds still fail there.
- `histc` through the rewrite equals ATen's CPU kernel run with one thread: ATen adds 1 per value in the input's dtype, so a bin stops at 2^p (256 in bfloat16, 2048 in float16, 2^24 in float32), and the rewrite caps its counts there. With several threads ATen sums per-thread histograms, and its counts then depend on the thread count. Where ATen's 16-bit position passes the last bin (bfloat16, 500 bins over [-3, 3]: 3.0 lands at 502) ATen writes outside the histogram (undefined behaviour); the rewrite counts those values in the last bin. Where a step of ATen's position overflows the dtype (float16, [0, 1000], 100 bins: `(x - L) * bins` passes 65504 from x = 655.5) its position is infinite and its write undefined (one thread: the value is dropped); the rewrite bins those values by the float64 position (700 in bin 70). A range whose width overflows the dtype (float16 [-40000, 40000], float32 [-3e38, 3e38]) counts nothing, as in ATen. In compiled graphs, equal float bounds take the data's range on the card and bin with the card's arithmetic, as Habana's own kernel does: not ATen's at the edges, and NaN or inf give counts where ATen raises.
- The copy checks (`wide`) are a Python kernel at eager `copy_`, which cannot take named tensors: a named tensor copied to the card raises ("Named tensors are currently unsupported in TorchScript") while they are installed. `export HPU_OPS_COPY_CHECKS=0` before `enable()` leaves `copy_` alone.
- `slogdet` of a singular matrix: `logabsdet = -inf` as in torch, but the gradient is finite where torch's is non-finite (measured on Gaudi2).
- In inference mode `solve` falls back (no backend kernel for `_linalg_solve_ex`), at any n. Under `torch.compile` a singular `solve` returns instead of raising, at any n.
- Habana's own `linalg.qr` gradient at n = 220 is wrong (relative error 0.25, with or without hpu-ops; 2.2e-6 at n = 128): `repro/habana_qr_n220_grad.py`.
- Calls the kernels decline (n > 64, kernels off, another Gaudi generation) take hpu-ops' own CPU route, which is not Habana's fallback. Its per-call cost against the fallback is not measured.

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
- `index` of integer indices, rewritten as `index_select`, checks CPU index tensors against the sizes on the host (an index out of range reaches ATen's `IndexError`). Index tensors on the device, and every index tensor while `torch.compile` traces, are not read on the host: an index out of range becomes -1, which `index_select` rejects with a `RuntimeError` (on the CPU; Gaudi's lazy kernel may not check it, as its native `index` does not). Its result is contiguous, where ATen's follows a non-contiguous input's layout. `index_put_` with integer indices that do not lead, called directly with `accumulate=True`, keeps Gaudi's lazy-mode kernel (wrong sums, see `repro/index_put_lazy.py`); so do calls in inference mode.
- Negative indices in an index that does not lead are wrong on Gaudi without hpu-ops, eager and compiled (`aten::index.Tensor_out`, 1.0 off, `tests/adversarial_index.py`). The rewrite makes them right in eager and lazy mode, and in compiled graphs, which hold the rewrite whether the index tensors are on the CPU or on the device (measured on Gaudi2, `hpu_backend`, CPU index tensors: 1.2e-07 against 1.0 native, no CPU fallback, `tests/adversarial_index.py`). An `index` called directly with a 0-d index tensor still has one CPU fallback compiled: its traced backward sorts one element (`aten::sort.values_stable`).
- The mask put is rewritten only with one bool mask that is the last index. Leading masks (`x[x < eps] = eps`) are rewritten too, as `where` plus a copy instead of the in-place `masked_fill_` (one extra copy of the destination): keeping them native brought the lazy launch failure back in SPNN and DeepGeoCCA. Byte masks, masks that share memory with the destination, and calls below autograd (inference mode) keep the native op; compiled graphs reach the rewrite through autograd before functionalization. A boolean-mask put of a value of another dtype is checked on the host with ATen's checked cast and converted with `to`, which keeps a forward-mode tangent. A value that ATen rejects (overflow, complex into real), a value on the device or with a gradient, and a put into a view whose elements overlap or leave gaps (expanded, `as_strided`) keep Gaudi's native op, and with it the lazy-mode launch failure of `repro/lazy_section_validation.py` in that graph shape. Gaudi's native put into an overlapping `as_strided` view differs from the CPU's (measured 1.07 eager, 0.18 lazy, with and without hpu-ops).
- `undo()` removes the kernels from the dispatcher, but functions already compiled with `torch.compile` keep the graphs they traced through them until dynamo recompiles (`torch._dynamo.reset()`); hpu-ops does not reset dynamo's caches.
- In lazy mode under `torch.inference_mode()` (below autograd), `index` and `F.unfold` keep Habana's own `HPU` kernels, which hpu-ops does not override: measured on Gaudi2 (`tests/index_im2col_modes.py`), `vech` and `X[:, mask]` then run on the card without a fallback, and `F.unfold` falls back to the CPU once per call (`aten::im2col`). `F.fold` is rewritten there too, and `torch.no_grad()` goes through every rewrite, eager and lazy.
- `F.fold` with overlapping windows sums through `index_add`; in float16 and bfloat16 that sum is exact where ATen's CPU kernel rounds per add. The gradient of a complex `index` holding `inf` gives `inf+nanj` through `index_select`'s backward, where ATen gives `inf+0j`.
- `F.unfold` and `F.fold` whose spatial index would exceed 2^24 elements (`im2col.MAX_INDEX`, `kh * kw * oh * ow`) keep the native op (a CPU fallback on Gaudi).
- Duplicate indices accumulate their gradients through `index_add`, which in bfloat16 sums in the order of the index; Gaudi's native accumulating put may round differently (not a defect of either).
- The native class of wide kernels (`conv.NATIVE_WIDE`) is empirical: it holds for the shapes in the sweep table (bridge 1.21.4); a new bridge may move it, rerun `repro/sweep_conv_native.sh`. Eager odd widths above 256 with one input channel are split although 4095 and 4097 ran at stride = width.
- Large-kernel convolution splitting requires static spatial, kernel, stride, padding, and dilation sizes (dynamic spatial sizes fall back to the native op); under `torch.compile` inside `torch.inference_mode()`, `AutogradHPU` is excluded and `ProxyTorchDispatchMode` sits above `HPU`, so there the split comes from `conv{1,2,3}d`'s Python composite kernel (a direct `aten.convolution` call in such a graph stays unsplit). Compiled, some split graphs still fail in Habana's graph compiler, as the native op does: conv1d k = 351 over 1000 (`Cannot add a node DEDW to bundle`, any chunking but 256 + 256, `repro/probe_compile_chunks.py`) and a (4, 1089) kernel over width 1500 (`setAssociatedDimAndSize`). In lazy mode a native conv2d (1, 1, 54, 54) x (1, 1, 44, 44), forward and backward, stops the card (`Synapse detected a device critical error ... Compute or dma timeout`; `repro/probe_conv2d_lazy.py 54 54 44 44`), as do (54, 138) x (44, 128) and (64, 64) x (54, 54), while (138, 54) x (128, 44), (110, 110) x (100, 100) and (138, 138) x (128, 128) run; the lazy split pads a 2-D kernel's last chunk to 128 so it never emits such a chunk (the (300, 300) kernel over 310 x 310 runs as nine (128, 128) chunks over (138, 138)). The split sums in another order than native: where a float32 partial sum overflows midway (past 3.4e38) a split output can be inf or NaN where native is finite or inf, and the reverse.
- Fully masked query rows (a sample with no real key) may differ from torch's zeros.
