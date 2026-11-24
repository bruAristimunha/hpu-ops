# Habana compatibility: what does not work, why, and what hpu-ops does about it

Measured on Gaudi2 (HL-225), gaudi-docker 1.21.4, bridge 1.21.4, torch 2.6.0, SDSC Voyager.
`INVENTORY.md` holds the per-model census; this file holds the causes. "Fixed" means a rewrite in
hpu-ops (a composite or decomposition registered with the dispatcher) makes the stock call work;
no model code changes. Counts are the Braindecode zoo (104 models) unless said otherwise.

## Scoreboard (targeted re-runs on the current stack; full census recount pending)

| mode | proven | remaining |
|---|---|---|
| eager | 104/104 | — |
| compiled (`hpu_backend`, static) | 103/104 | EpiNT compiled inference (vendor recipe) |
| lazy | 99/104 (4 of them segmented) | full recount pending; no known own-code failure |

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
| 9 | Whole-model compiled inference graphs fail `synStatus 26` even per-child compiled | Habana's recipe compiler rejects the inference graph; eager and lazy graphs of the same model compile | EpiNT compiled inference | none yet (manual SDPA and per-child compile both measured failing) | vendor |
| 10 | Default-dynamic compiled inference raises `KeyError: 'output_strides'` | Habana recipe compiler bug with dynamic shapes (#63) | TensorCSPNet, GyroAtt (spd_learn) | compile with `dynamic=False` (the census does) | vendor; workaround known |
| 11 | Traced composites returned `NotImplemented` outside inference, breaking compiled graphs | Dynamo casts the sentinel to Tensor in extrema's py kernels | SPD compiled graphs (#60) | traced decline in linalg (#62): graphs keep the aten ops, the runtime dispatches to the TPC kernels | fixed |
| 12 | `argmax.out` layout, lazy conv band 129..255, `unfold`, `renorm`, 0-d embedding grads, `roll` of strided inputs | assorted missing or wrong Habana kernels (each with a reproducer in `repro/`) | MEGGPT, EEGInceptionMI, and the op-level sweep | composites in `hpu_ops/composite.py` (#43, #65, #66 and neighbours) | fixed |

## Rules the fixes follow

- A rewrite only takes the calls Habana gets wrong; valid calls it declines reach the native or
  torch-composite kernel below it (never a second implementation of the same call).
- Traced (`CompositeImplicitAutograd`) registrations are what `torch.compile` bakes into graphs;
  autograd and backend registrations cover eager and lazy. An op with parameters must be fixed at
  the traced level, or autograd cannot differentiate the composite.
- Every number above was measured on the card with `hpu_ops.strict("report")` counting CPU
  executions per step; a model counts as unlocked only with zero rows in the mode claimed.
