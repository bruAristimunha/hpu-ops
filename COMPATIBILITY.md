# Habana compatibility: what does not work, why, and what hpu-ops does about it

Measured on Gaudi2 (HL-225), gaudi-docker 1.21.4, bridge 1.21.4, torch 2.6.0, SDSC Voyager.
`INVENTORY.md` holds the per-model census; this file holds the causes. "Fixed" means a rewrite in
hpu-ops (a composite or decomposition registered with the dispatcher) makes the stock call work;
no model code changes. Counts are the Braindecode zoo (104 models) unless said otherwise.

## Scoreboard (targeted re-runs on the current stack; full census recount pending)

| mode | proven | remaining |
|---|---|---|
| eager | 104/104 | — |
| compiled (`hpu_backend`, static) | 104/104 (EpiNT inference under `no_grad`, not `inference_mode`) | — |
| lazy | 99/104 stock recount; 103/104 with segmented; every model passes some lazy route | NeuroPose fails only in the batched census (passes isolated) |
| spd zoo (13/21 timed, synced) | lazy fastest for 11, compiled-static for SPDResNet + CorNet | TensorCSPNet/GraphCSPNet compiled slower (eigh-bound, graph breaks) |

Routing: `strict_sweep --auto` tries compiled (static), lazy, then segmented lazy per model in
subprocesses and records the first working mode (`auto_chosen`); on GCNsNet the segmented pass
also measured fastest in the latest run (85.1 vs lazy 114.4 vs compiled 129.0 ms/step).

Routing: `strict_sweep --auto` tries compiled (static), lazy, then segmented lazy per model in
subprocesses and records the first working mode (`auto_chosen`); on GCNsNet the segmented pass
also measured fastest in the latest run (85.1 vs lazy 114.4 vs compiled 129.0 ms/step).

## The incompatibilities

| # | What breaks | Root cause (what Habana 1.21.4 does) | Models hit | hpu-ops answer | Status |
|---|---|---|---|---|---|
| 1 | `max_pool1d` with kernel wider than the input (`ceil_mode`, e.g. kernel 2 over length 1) fails graph compile `synStatus 26` | Habana's pooling kernel has no recipe for a window wider than the signal; ATen returns the window max | GCNsNet (last pooling over length 1) | `composite.install_max_pool1d`: window maxima (`amax` per window) at the autograd, backend and traced keys; narrower kernels decline to torch's composite | fixed |
| 2 | `instance_norm` in inference fails graph compile `synStatus 26` (training passes) | Habana's inference recipe for the op is rejected by its own compiler; the training recipe is a different graph | EpiNT (eager/lazy inference) | `norm.install_instance_norm`: per-sample/channel `var_mean` composite at the traced, autograd and backend keys (float32 statistics for 16-bit inputs) | fixed eager/lazy; compiled inference still fails (row 9) |
| 3 | Lazy full-step graphs are rejected (`synStatus 26`) or hang | Habana's lazy fuser cannot compile very large joint graphs (BBL's is 855 nodes; autoregressive unrolls, CareSleepNet's inference graph hangs with no exception) | BBL, BrainTokenizer, VEMG2Pose, CareSleepNet (lazy) | one graph per top-level module: `mark_step` after each child (`strict_sweep --segment-topchild`); no hpu-ops kernel is missing — the limit is graph size | workaround in the sweep; vendor limit |
| 4 | `LazyLinear` / uninitialized parameters become `(0,)` shells on `.to("hpu")` | `HabanaParameterWrapper` rewrites every parameter on device placement and empties the original before `LazyModuleMixin` can materialize it (#68) | CBraMod, CSBrain, GCNsNet biases | `hpu_ops.materialize(model, example_input)`: one CPU forward allocates the parameters before the move; the sweep calls it for every model | fixed (helper); bridge bug |
| 5 | `group_norm` with a symbolic batch (second batch size under default `torch.compile`) raises `aten::IntImplicit isn't registered` | the batch reaches `native_group_norm` as a SymInt argument, which Habana converts with a TorchScript symbol that has no dispatcher schema (#42) | 8 Braindecode models with GroupNorm | `norm.install_group_norm`: traced reshape/`var_mean` composite that takes no size as an int argument; static shapes keep Gaudi's kernel | fixed |
| 6 | `glu` on a permuted intermediate is rejected in lazy inference | Habana's lazy GLU graph assumes the split dim's layout | BrainModule, FNIRS4D (lazy inference) | `composite.install_glu` at the backend key in lazy mode: `a * sigmoid(b)` (#70) | fixed |
| 7 | `one_hot` with an inferred class count (`num_classes=-1`) is rejected | Habana's kernel requires an explicit count; ATen's composite reaches the same kernel | GCNsNet graph coarsening | `composite.install_one_hot`: equality against `arange` (#67) | fixed |
| 8 | float64 is stored as float32; float64 `eigh` routes to the CPU | Gaudi2 has no float64 arithmetic | GBWAtt (spd_learn, 30 eigh/step) | `HPU_OPS_FLOAT64=card` (float32 precision, zero rows); a double-single Jacobi eigh prototypes at 3.4e-12 vs float64 on ill-conditioned spectra (#64) | precision decision open |
| 9 | Compiled `inference_mode` graphs fail `synStatus 26`, whole-model and per-child | Habana's recipe compiler rejects the inference-mode graph specifically; the same model compiles under `no_grad` | EpiNT compiled inference | run compiled inference under `torch.no_grad()` instead of `torch.inference_mode()` (measured passing; manual SDPA and per-child compile do not help) | workaround; vendor recipe |
| 10 | Default-dynamic compiled inference raises `KeyError: 'output_strides'` | Habana recipe compiler bug with dynamic shapes (#63) | TensorCSPNet, GyroAtt (spd_learn) | compile with `dynamic=False` (the census does) | vendor; workaround known |
| 11 | Traced composites returned `NotImplemented` outside inference, breaking compiled graphs | Dynamo casts the sentinel to Tensor in extrema's py kernels | SPD compiled graphs (#60) | traced decline in linalg (#62): graphs keep the aten ops, the runtime dispatches to the TPC kernels | fixed |
| 12 | `argmax.out` layout, lazy conv band 129..255, `unfold`, `renorm`, 0-d embedding grads, `roll` of strided inputs | assorted missing or wrong Habana kernels (each with a reproducer in `repro/`) | MEGGPT, EEGInceptionMI, and the op-level sweep | composites in `hpu_ops/composite.py` (#43, #65, #66 and neighbours) | fixed |
| 13 | Lazy graph segfault after `.item()` / `bool()` in a forward (stock Braindecode quantizers) | Habana's lazy compiler segfaults when the 8192-code k-means `initialize` loop is fused into the graph that produced its vectors; the loop as its own graph compiles | NeuroRVQTokenizer (lazy, stock) | `small.install_scalar_dense`: flush the pending lazy graph (`mark_step`) before Habana's scalar read, at the autograd and backend keys (backend covers `inference_mode` reads); the read itself still syncs - removing it needs the value mirrored in Python | fixed (crash); sync is inherent |
| 14 | `scaled_dot_product_attention` op-by-op (~5x slower), fused kernel wrong for some view layouts | bridge 1.21 registers no fused SDPA kernel at the HPU key without the override env setting (`AutogradHPU`/`HPU` both empty before `enable()`) | attention models (AXON, EpiNT, LaBraM, REVE layouts) | `composite.install_sdpa`: torch's math decomposition (float32 attention math for 16-bit inputs) registered only at keys Habana left empty; dropout and head-mismatch errors decline to native | fixed (card-verified eager+lazy, fwd+bwd) |
| 15 | `unique` / `unique_dim` fall back to the CPU | Habana registers no on-card kernel; the bridge copies the tensor to the CPU (data-dependent output shape) | BrainOmni, BrainTokenizer (`_kmeans`) | `composite.install_unique`: lexicographic row sort (stable, per column) + bool-mask selection, which Habana does execute on the card; inverse via `cumsum`, counts via `bincount`; traced calls decline (dynamic shape) | fixed (card-verified vs ATen) |
| 16 | `linalg.qr` backward is wrong (relative gradient error 0.25 at n=220, well-conditioned) | Habana's QR derivative formula/kernel is inaccurate at larger n (forward is correct) | spd_learn QR users, any training through QR | `linalg.install_qr`: Householder QR (LAPACK's reflector convention) as differentiable ops for gradient-needing inputs up to 300 rows/columns (64 in lazy mode, where the reflector graph compiles too slowly past that), so autograd supplies the exact backward; inference and larger inputs keep the native forward | fixed (card-verified vs float64) |
| 17 | Lazy large-graph `synStatus 26` / hangs inside one top-level child (row 3) | Habana's lazy fuser graph-size limit, below the top-child boundary | quantizer stacks, autoregressive unrolls | `hpu_ops.segmented(model, deep=True)`: `mark_step` after every submodule, not just top-level children | workaround (helper) |
| 18 | Lazy SDPA with dropout segfaults (`TransformerEncoderLayer` in train mode) | Habana's lazy attention kernel crashes when `dropout_p > 0`; eval and dropout=0 pass | CareSleepNet (lazy train) | `composite.install_sdpa` math path now applies `F.dropout` on the softmax weights instead of declining dropout calls | fixed (card-verified: layer + full CareSleepNet train step) |
| 19 | Compiled strided-convolution backward fails `synStatus 26` | Habana's compiled strided-conv backward recipe is rejected; forward and stride-1 compile fine | SEANet stacks (BrainTokenizer/BrainOmni encoders) | traced `conv{n}d` kernel: compile-mode strided calls compute the stride-1 convolution and subsample every strided dim (exact grads, card 1e-11) | fixed (op level; BrainTokenizer whole-graph still hits a multi-branch recipe interaction - lazy segmented is its fastest route) |

## Rules the fixes follow

- A rewrite only takes the calls Habana gets wrong; valid calls it declines reach the native or
  torch-composite kernel below it (never a second implementation of the same call).
- Traced (`CompositeImplicitAutograd`) registrations are what `torch.compile` bakes into graphs;
  autograd and backend registrations cover eager and lazy. An op with parameters must be fixed at
  the traced level, or autograd cannot differentiate the composite.
- Every number above was measured on the card with `hpu_ops.strict("report")` counting CPU
  executions per step; a model counts as unlocked only with zero rows in the mode claimed.
