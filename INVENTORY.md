# CPU executions of Gaudi steps: the inventory

Every CPU execution that hpu-ops' strict mode (`HPU_OPS_STRICT=report`, `hpu_ops.strict("report")`) saw in one
training step and one inference forward of each model, on one Gaudi2 card (gaudi-docker 1.21.4, bridge 1.21.4,
torch 2.6.0), hpu-ops at this branch, hpu-complex at origin/main (622e2c7), November 21, 2026.

- Models: the 21 `spd_diag.models.MODELS` (BNCI2014_001, 22 channels, real batch of 32, cross-entropy) and the 85
  braindecode models of `braindecode.models.util.models_mandatory_parameters` (braindecode a78ecc02 with
  braindecode-kernels' `patch_braindecode()`, random batch of 32 at each model's test signal parameters, mean-square
  loss). braindecode is not installed on the pod: its source and its missing pure-Python dependencies (skorch,
  torchinfo, tabulate, wfdb, linear_attention_transformer and its three helpers) were put on `PYTHONPATH` from an
  untracked `_ext/` directory of the worktree.
- Step: forward, backward, Adam. One warm-up step, then strict mode over 3 steps (counts are per step), then the mean
  of 5 synced steps outside strict mode (ms/step), then one `torch.inference_mode()` forward under strict mode.
- Modes: eager (`PT_HPU_LAZY_MODE=0`), lazy (`=1`), and eager + `torch.compile(backend="hpu_backend")`. Eager and
  lazy use the per-op watch (op, tensors and caller frame). Compiled steps use `strict(per_op=False)`: a dispatch mode
  changes what Dynamo does (see the README), so the compiled column holds what the counters see per step (bridge and
  hpu-complex fallbacks, named by op without a frame) and hpu-ops' own routes (in the graph, recorded at run time when
  a graph break runs them eagerly); host syncs, host copies, CPU ops and complex128 host compute of compiled steps are
  not seen. The first compiled step (trace) is recorded apart and not counted in the impact.
- Impact = the sum, over (model, mode) and over training step and inference forward, of the count per step: count
  x models. A model counts once per mode where the entry appears.
- Reproduce: `bash tests/strict_sweep.sh OUTDIR "spd:eager spd:lazy bd:eager bd:lazy"`, then
  `tests/strict_sweep.py --family spd|bd --compile --step-check`, then `python3 tests/strict_inventory.py OUTDIR`.
  The eager and lazy rows were taken with strict mode at 8e157c4, which counted views and host-to-device copies of
  CPU tensors as `cpu op`; `strict_inventory.py` drops the views and relabels the copies `h2d copy`, as 6e44366 records
  them.

## Root causes, by impact

| rank | id | CPU execution | source | impact | models eager / lazy / compile / inference | models | proposed fix route |
|---|---|---|---|---|---|---|---|
| 1 | G2 | Constant CPU index tensors (`triu_indices`) checked, linearized and copied to the card every call (spd_learn `sym_to_upper` / `vec_to_sym` through hpu_ops indexing) | hpu_ops route + cpu op + h2d copy | 1063 | 17 / 16 / 13 / 17 | DeepGeoCCA, EEGSPDNet, GBWAtt, Green, GyroAtt, GyroSPD +11 | dispatcher rewrite (hpu_ops indexing): cache the device copy and linear index of constant CPU index tensors (data_ptr, version) and skip the host range check; library alt: `triu_indices(device=X.device)` cached |
| 2 | G1 | spd_learn GBW metric validation: `torch.isfinite(x).all()` / symmetry `.any()` read on the host | host sync | 784 | 1 / 1 / 0 / 0 | GBWAtt | library (spd_learn): GBW `_require_spd` validation off the step (once, or opt-in), `torch._assert_async` on the card |
| 3 | G3 | `linalg.eigh` past the TPC kernel (n > 32) or float64: hpu_ops' own CPU route | hpu_ops route | 519 | 5 / 5 / 2 / 5 | EEGSPDNet, GBWAtt, GraphCSPNet, STaRNet, TensorCSPNet | TPC kernel: Jacobi eigh for 32 < n <= 128 (EEGSPDNet 55/110, GraphCSPNet/TensorCSPNet 36, STaRNet 48); float64 (GBWAtt 18x18): model change to float32, or a policy decision |
| 4 | G4 | spd_learn matrix_log clamp warning reads `s.min()` on the host (skipped on CUDA only) | host sync | 167 | 17 / 16 / 0 / 0 | CorAtt, DeepGeoCCA, EEGSPDNet, Green, GyroAtt, GyroSPD +11 | library (spd_learn): `warn_on_clamp`'s `s.min()` read skips CUDA only; test `s.device.type == 'cpu'`, or default the warning off |
| 5 | G5 | `_linalg_solve_ex` falls back: inference_mode (bimap, n 16..220) and lazy n > 64 | bridge fallback | 109 | 0 / 2 / 0 / 19 | CorAtt, CorNet, DeepGeoCCA, EEGSPDNet, GBWAtt, GraphCSPNet +13 | dispatcher rewrite: `_linalg_solve_ex` backend (HPU-key) kernel into the TPC Gauss-Jordan for inference_mode; TPC large-n path in lazy mode (n = 96..220) |
| 6 | G9 | complex128 computed on the CPU in host memory (hpu_complex) | hpu_complex | 67 | 1 / 0 / 0 / 1 | SensingDynamics | model change (SensingDynamics builds complex128 from float64); hpu_complex keeps complex128 in host memory by design |
| 7 | G14 | spd_learn model host reads (GraphCSPNet, CorNet) | host sync | 48 | 2 / 2 / 0 / 0 | CorNet, GraphCSPNet | library (spd_learn): GraphCSPNet graphcsp.py:354, CorNet cornet.py:124/127 host reads |
| 8 | G12 | braindecode host reads in forward | host sync | 36 | 4 / 3 / 0 / 0 | BaRISTA, BrainOmni, BrainTokenizer, CodeBrain | library (braindecode): CodeBrain codebrain.py:508, VQ init/expire codes (quantization.py:183/208), BaRISTA barista.py:531 host reads |
| 9 | G8 | `unfold_backward` falls back in eager and compiled steps | bridge fallback | 32 | 1 / 0 / 1 / 0 | MetaNeuromotorHand | dispatcher rewrite: eager `unfold_backward` as index_add / as_strided scatter (hpu_ops covers lazy unfold only) |
| 10 | G13 | braindecode constants built on the CPU and copied each step | h2d copy | 11 | 2 / 2 / 0 / 2 | LUNA, REVE | library (braindecode): REVE/LUNA build constants on the CPU and copy them each step; register buffers on the device |
| 11 | G7 | `im2col` falls back in lazy inference_mode | bridge fallback | 5 | 0 / 0 / 0 / 5 | CorAtt, GyroAtt, MAtt, SPDMetaFormer, SSGCNet | dispatcher rewrite: hpu_ops im2col also at the HPU key for lazy inference_mode (Habana's im2col falls back) |
| 12 | G16 | hpu_complex CPU fallback under compile (per step, op unattributed) | hpu_complex | 4 | 0 / 0 / 2 / 2 | LUNA, MetaNeuromotorHand | hpu_complex: its CPU fallback under compile (LUNA angle, MetaNeuromotorHand complex im2col), as G10/G11 |
| 13 | G10 | hpu_complex CPU fallback: `angle` | hpu_complex | 3 | 1 / 1 / 0 / 1 | LUNA | hpu_complex composite: `angle` as atan2(imag, real) on the card |
| 14 | G6 | `linalg_lu_solve.out` falls back in solve's lazy backward (n = 96..220) | bridge fallback | 3 | 0 / 2 / 0 / 0 | EEGSPDNet, STaRNet | composite of HPU ops: solve's lazy backward through the saved inverse (matmul) instead of `lu_solve` |
| 15 | G11 | hpu_complex CPU fallback: `im2col` of complex64 | hpu_complex | 2 | 1 / 0 / 0 / 1 | MetaNeuromotorHand | hpu_complex composite: `im2col` of complex64 through its float pair (view_as_real) |
| 16 | G15 | `unique_dim` falls back under compile | bridge fallback | 0 | 0 / 0 / 2 / 0 | BrainOmni, BrainTokenizer | Habana bug -> report: `unique_dim` has no HPU kernel under compile (trace only) |

Fix routes, for the controller: dispatcher rewrite (hpu-ops team): G2, G5, G7, G8; TPC kernel: G3 (n <= 128), G5's
lazy large path; composite of HPU ops: G6, G10, G11 (hpu-complex team); library change upstream: G1, G4, G14
(spd_learn), G12, G13 (braindecode), G9 (model dtype); Habana bug -> report: G15.

Notes on the largest entries:

- G2 is one line of spd_learn (`ixs = torch.triu_indices(n, n)` on the CPU, then `X[..., ixs[0], ixs[1]]`) in 17 of
  21 models, eager, lazy, compiled and in inference. hpu-ops' indexing rewrite takes CPU index tensors on purpose (its
  host range check and linear index avoid a device sync), and copies the result to the card at every call. Under
  torch.compile the same code is traced into the graph as CPU nodes (AOT's functional tensors are not recognised as
  tracing in `indexing._tracing` / `im2col._tracing`), so it runs every step there too.
- G1 is 392 host reads per GBWAtt step (validation of every SPD input of its attention), G4 one to 22 per step in 17
  models: `if numerical_config.warn_on_clamp and (not s.is_cuda or ...)` reads `s.min()` on every non-CUDA device.
- G3: float32 eigh above n = 32 (the TPC Jacobi kernel's limit) and float64 eigvalsh (GBWAtt computes its metric in
  float64: 51 + 30 calls per step) go through hpu-ops' CPU route, eager, lazy and compiled.
- G5: hpu-ops registers `solve` at the autograd key only (its derivative needs Habana's LU factors when traced), so
  inference_mode reaches Habana's kernel and its CPU fallback in 19 models; lazy training falls back for n > 64,
  where the large TPC path is eager-only.

## Not CPU executions, but blockers seen on the way (each a separate issue)

- Lazy aborts (`terminate called after throwing an instance of 'std::runtime_error'`): spd Green and braindecode
  SensingDynamics, also without strict mode (`--step-check`).
- Lazy errors: BrainTokenizer, EEGInceptionMI, VEMG2Pose (synapse graph compile failed); CBraMod, CSBrain (matmul
  shapes `32x3000 and 1x0` / `1783705536x0`: garbage sizes); hpu-complex refuses `torch.fft.rfft` under
  `inference_mode` in lazy mode (9 models: BIOT, BrainBERT, Brant, CodeBrain, EEGDINO, EMG2QwertyNet, LUNA, MAPA,
  TFMTokenizer) and rank >= 5 permuted complex64 (MetaNeuromotorHand).
- Compile errors (bridge 1.21.4): DeepGeoCCA (PT_DYNAMIC_SHAPE), EEGSPDNet and STaRNet ("dim idx is out of range",
  STaRNet's through hpu-ops' `_on_cpu` in the graph), GraphCSPNet ("Dynamic patching had failed"), and
  `KeyError('output_strides')` in Habana's recipe compiler for the inference_mode graph of GyroAtt, MAtt and
  TensorCSPNet; braindecode: 3 errors (below).
- Model or harness: GCNsNet ("Number of classes cannot be -1"), NeuroRVQTokenizer (its output has no grad: the
  harness's loss), EpiNT eager inference (synapse graph compile).

## Full table (every entry)

Sources: `bridge fallback` (Habana's CPU fallback), `hpu_ops route` (hpu-ops' own CPU code), `hpu_complex`,
`host sync` (a device value read on the host), `h2d copy` (a CPU tensor built in the step and copied to the card),
`cpu op` (ops computing on CPU tensors). `host sync`, `h2d copy` and `cpu op` entries are per caller line. A caller
`?` is autograd's backward (no Python frame).

| # | source | CPU op | impact | models eager / lazy / compile / inference | count per step (max) | example caller | proposed fix |
|---|---|---|---|---|---|---|---|
| 1 | host sync | `aten::_local_scalar_dense` | 784 | 1 / 1 / 0 / 0 | 392 | `spd_learn/functional/metrics/generalized_bures_wasserstein.py:31 in _require_spd` | G1: library (spd_learn): GBW `_require_spd` validation off the step (once, or opt-in), `torch._assert_async` on the card |
| 2 | cpu op | `aten::add.Tensor, aten::all, aten::bitwise_and.Tensor, aten::ge.Scalar, aten::lt.Scalar, aten::mul.Tensor, aten::where.s` | 624 | 17 / 16 / 0 / 0 | 8 | `spd_learn/functional/core.py:658 in sym_to_upper` | G2: dispatcher rewrite (hpu_ops indexing): cache the device copy and linear index of constant CPU index tensors (data_ptr, version) and skip the host range check; library alt: `triu_indices(device=X.device)` cached |
| 3 | hpu_ops route | `aten::_linalg_eigh` | 519 | 5 / 5 / 2 / 5 | 51 | `spd_learn/functional/autograd.py:36 in modeig_forward` | G3: TPC kernel: Jacobi eigh for 32 < n <= 128 (EEGSPDNet 55/110, GraphCSPNet/TensorCSPNet 36, STaRNet 48); float64 (GBWAtt 18x18): model change to float32, or a policy decision |
| 4 | host sync | `aten::_local_scalar_dense` | 167 | 17 / 16 / 0 / 0 | 22 | `spd_learn/functional/core.py:209 in forward` | G4: library (spd_learn): `warn_on_clamp`'s `s.min()` read skips CUDA only; test `s.device.type == 'cpu'`, or default the warning off |
| 5 | cpu op | `aten::add.Tensor, aten::all, aten::bitwise_and.Tensor, aten::ge.Scalar, aten::index.Tensor, aten::lt.Scalar, aten::mul.T` | 152 | 1 / 1 / 0 / 1 | 16 | `spd_learn/functional/core.py:731 in vec_to_sym` | G2: dispatcher rewrite (hpu_ops indexing): cache the device copy and linear index of constant CPU index tensors (data_ptr, version) and skip the host range check; library alt: `triu_indices(device=X.device)` cached |
| 6 | bridge fallback | `aten::_linalg_solve_ex.result` | 109 | 0 / 2 / 0 / 19 | 1 | `spd_learn/modules/bilinear.py:273 in forward` | G5: dispatcher rewrite: `_linalg_solve_ex` backend (HPU-key) kernel into the TPC Gauss-Jordan for inference_mode; TPC large-n path in lazy mode (n = 96..220) |
| 7 | hpu_ops route | `index: linear index on the host` | 68 | 17 / 16 / 0 / 17 | 4 | `spd_learn/functional/core.py:658 in sym_to_upper` | G2: dispatcher rewrite (hpu_ops indexing): cache the device copy and linear index of constant CPU index tensors (data_ptr, version) and skip the host range check; library alt: `triu_indices(device=X.device)` cached |
| 8 | cpu op | `aten::ne.Tensor` | 66 | 16 / 15 / 0 / 16 | 2 | `spd_learn/functional/core.py:662 in sym_to_upper` | G2: dispatcher rewrite (hpu_ops indexing): cache the device copy and linear index of constant CPU index tensors (data_ptr, version) and skip the host range check; library alt: `triu_indices(device=X.device)` cached |
| 9 | hpu_ops route | `index: CPU index tensors checked and linearized on the host` | 63 | 0 / 0 / 13 / 12 | 2 | `spd_learn/functional/core.py:658 in sym_to_upper` | G2: dispatcher rewrite (hpu_ops indexing): cache the device copy and linear index of constant CPU index tensors (data_ptr, version) and skip the host range check; library alt: `triu_indices(device=X.device)` cached |
| 10 | h2d copy | `aten::_to_copy` | 39 | 17 / 16 / 0 / 0 | 2 | `spd_learn/functional/core.py:658 in sym_to_upper` | G2: dispatcher rewrite (hpu_ops indexing): cache the device copy and linear index of constant CPU index tensors (data_ptr, version) and skip the host range check; library alt: `triu_indices(device=X.device)` cached |
| 11 | host sync | `aten::_local_scalar_dense` | 36 | 1 / 1 / 0 / 0 | 18 | `spd_learn/models/graphcsp.py:354 in forward_covariances` | G14: library (spd_learn): GraphCSPNet graphcsp.py:354, CorNet cornet.py:124/127 host reads |
| 12 | h2d copy | `aten::_to_copy` | 35 | 16 / 15 / 0 / 0 | 2 | `spd_learn/functional/core.py:664 in sym_to_upper` | G2: dispatcher rewrite (hpu_ops indexing): cache the device copy and linear index of constant CPU index tensors (data_ptr, version) and skip the host range check; library alt: `triu_indices(device=X.device)` cached |
| 13 | bridge fallback | `aten::unfold_backward` | 32 | 1 / 0 / 1 / 0 | 16 | `?` | G8: dispatcher rewrite: eager `unfold_backward` as index_add / as_strided scatter (hpu_ops covers lazy unfold only) |
| 14 | hpu_complex | `aten::mul.Tensor` | 22 | 1 / 0 / 0 / 1 | 10 | `braindecode/models/sensingdynamics.py:407 in forward` | G9: model change (SensingDynamics builds complex128 from float64); hpu_complex keeps complex128 in host memory by design |
| 15 | hpu_complex | `aten::add.Tensor` | 20 | 1 / 0 / 0 / 1 | 10 | `braindecode/models/sensingdynamics.py:409 in forward` | G9: model change (SensingDynamics builds complex128 from float64); hpu_complex keeps complex128 in host memory by design |
| 16 | host sync | `aten::_local_scalar_dense` | 16 | 1 / 1 / 0 / 0 | 8 | `braindecode/models/codebrain.py:508 in forward` | G12: library (braindecode): CodeBrain codebrain.py:508, VQ init/expire codes (quantization.py:183/208), BaRISTA barista.py:531 host reads |
| 17 | host sync | `aten::_local_scalar_dense` | 12 | 2 / 1 / 0 / 0 | 4 | `braindecode/modules/quantization.py:183 in init_embed_` | G12: library (braindecode): CodeBrain codebrain.py:508, VQ init/expire codes (quantization.py:183/208), BaRISTA barista.py:531 host reads |
| 18 | host sync | `aten::_local_scalar_dense` | 12 | 1 / 1 / 0 / 0 | 6 | `spd_learn/models/cornet.py:127 in _regularized_cov_to_corr` | G14: library (spd_learn): GraphCSPNet graphcsp.py:354, CorNet cornet.py:124/127 host reads |
| 19 | hpu_complex | `aten::pow.Tensor_Scalar` | 11 | 1 / 0 / 0 / 1 | 5 | `braindecode/models/sensingdynamics.py:406 in forward` | G9: model change (SensingDynamics builds complex128 from float64); hpu_complex keeps complex128 in host memory by design |
| 20 | h2d copy | `aten::_to_copy` | 8 | 1 / 1 / 0 / 0 | 4 | `spd_learn/functional/core.py:731 in vec_to_sym` | G2: dispatcher rewrite (hpu_ops indexing): cache the device copy and linear index of constant CPU index tensors (data_ptr, version) and skip the host range check; library alt: `triu_indices(device=X.device)` cached |
| 21 | cpu op | `aten::ne.Tensor` | 8 | 1 / 1 / 0 / 1 | 2 | `spd_learn/functional/core.py:722 in vec_to_sym` | G2: dispatcher rewrite (hpu_ops indexing): cache the device copy and linear index of constant CPU index tensors (data_ptr, version) and skip the host range check; library alt: `triu_indices(device=X.device)` cached |
| 22 | bridge fallback | `aten::im2col` | 5 | 0 / 0 / 0 / 5 | 0 | `spd_learn/modules/utils.py:68 in forward` | G7: dispatcher rewrite: hpu_ops im2col also at the HPU key for lazy inference_mode (Habana's im2col falls back) |
| 23 | hpu_complex | `hpu_complex cpu_fallback` | 4 | 0 / 0 / 2 / 2 | 1 | `(per step)` | G16: hpu_complex: its CPU fallback under compile (LUNA angle, MetaNeuromotorHand complex im2col), as G10/G11 |
| 24 | host sync | `aten::_local_scalar_dense` | 4 | 1 / 1 / 0 / 0 | 2 | `braindecode/models/barista.py:531 in forward` | G12: library (braindecode): CodeBrain codebrain.py:508, VQ init/expire codes (quantization.py:183/208), BaRISTA barista.py:531 host reads |
| 25 | hpu_complex | `aten::zeros_like` | 4 | 1 / 0 / 0 / 1 | 2 | `braindecode/models/sensingdynamics.py:400 in forward` | G9: model change (SensingDynamics builds complex128 from float64); hpu_complex keeps complex128 in host memory by design |
| 26 | h2d copy | `aten::_to_copy, aten::to.dtype_layout` | 4 | 1 / 1 / 0 / 1 | 1 | `braindecode/models/reve.py:509 in forward` | G13: library (braindecode): REVE/LUNA build constants on the CPU and copy them each step; register buffers on the device |
| 27 | h2d copy | `aten::_to_copy, aten::to.device` | 4 | 1 / 1 / 0 / 1 | 1 | `braindecode/models/reve.py:729 in _embed` | G13: library (braindecode): REVE/LUNA build constants on the CPU and copy them each step; register buffers on the device |
| 28 | host sync | `aten::_local_scalar_dense` | 4 | 1 / 0 / 0 / 0 | 4 | `braindecode/modules/quantization.py:208 in expire_codes_` | G12: library (braindecode): CodeBrain codebrain.py:508, VQ init/expire codes (quantization.py:183/208), BaRISTA barista.py:531 host reads |
| 29 | h2d copy | `aten::_to_copy, aten::to.device` | 3 | 1 / 1 / 0 / 1 | 1 | `braindecode/models/luna.py:402 in get_default_channel_locations` | G13: library (braindecode): REVE/LUNA build constants on the CPU and copy them each step; register buffers on the device |
| 30 | hpu_complex | `aten::angle` | 3 | 1 / 1 / 0 / 1 | 1 | `braindecode/models/luna.py:554 in forward` | G10: hpu_complex composite: `angle` as atan2(imag, real) on the card |
| 31 | bridge fallback | `aten::linalg_lu_solve.out` | 3 | 0 / 2 / 0 / 0 | 1 | `?` | G6: composite of HPU ops: solve's lazy backward through the saved inverse (matmul) instead of `lu_solve` |
| 32 | hpu_complex | `aten::im2col` | 2 | 1 / 0 / 0 / 1 | 1 | `braindecode/models/meta_neuromotor.py:996 in forward` | G11: hpu_complex composite: `im2col` of complex64 through its float pair (view_as_real) |
| 33 | hpu_complex | `aten::_to_copy` | 2 | 1 / 0 / 0 / 0 | 2 | `braindecode/models/sensingdynamics.py:397 in forward` | G9: model change (SensingDynamics builds complex128 from float64); hpu_complex keeps complex128 in host memory by design |
| 34 | hpu_complex | `aten::abs` | 2 | 1 / 0 / 0 / 1 | 1 | `braindecode/models/sensingdynamics.py:413 in forward` | G9: model change (SensingDynamics builds complex128 from float64); hpu_complex keeps complex128 in host memory by design |
| 35 | hpu_complex | `aten::div.Tensor` | 2 | 1 / 0 / 0 / 1 | 1 | `braindecode/models/sensingdynamics.py:412 in forward` | G9: model change (SensingDynamics builds complex128 from float64); hpu_complex keeps complex128 in host memory by design |
| 36 | hpu_complex | `aten::exp` | 2 | 1 / 0 / 0 / 1 | 1 | `braindecode/models/sensingdynamics.py:398 in forward` | G9: model change (SensingDynamics builds complex128 from float64); hpu_complex keeps complex128 in host memory by design |
| 37 | hpu_complex | `aten::square` | 1 | 0 / 0 / 0 / 1 | 0 | `braindecode/models/sensingdynamics.py:414 in forward` | G9: model change (SensingDynamics builds complex128 from float64); hpu_complex keeps complex128 in host memory by design |
| 38 | hpu_complex | `aten::to.dtype` | 1 | 0 / 0 / 0 / 1 | 0 | `braindecode/models/sensingdynamics.py:417 in forward` | G9: model change (SensingDynamics builds complex128 from float64); hpu_complex keeps complex128 in host memory by design |
| 39 | bridge fallback | `aten::unique_dim` | 0 | 0 / 0 / 2 / 0 | 0 | `(per step)` | G15: Habana bug -> report: `unique_dim` has no HPU kernel under compile (trace only) |


## Per model: ms per step (forward + backward + Adam, batch 32) and status

Mean of 5 synced steps after the strict-mode steps, outside strict mode. Eager and lazy in one process per family and
mode, compiled with `--step-check` (no dispatch mode). An error's stage says where it happened; `inference` means the
training step ran and was timed.

| family | model | eager (ms/step, status) | lazy (ms/step, status) | compile (ms/step, status) |
|---|---|---|---|---|
| bd | ATCNet | 440.48 | 79.94 | 198.79 |
| bd | AXON | 1337.35 | 210.03 | 661.94 |
| bd | AttentionBaseNet | 39.46 | 13.99 | 39.4 |
| bd | AttnSleep | 240.32 | 46.82 | 237.42 |
| bd | BDTCN | 81.52 | 21.95 | 22.5 |
| bd | BENDR | 1361.91 | 49.94 | 4386.49 |
| bd | BIOT | 177.63 | 45.22, error: inference RuntimeError("hpu_complex: torch.fft.rfft under torch.inference_mode i | 158.97 |
| bd | BaRISTA | 803.34 | 117.52 | 314.73 |
| bd | BrainBERT | 98.07 | 24.78, error: inference RuntimeError("hpu_complex: torch.fft.rfft under torch.inference_mode i | 104.79 |
| bd | BrainModule | 120.74 | 32.94 | 135.05 |
| bd | BrainOmni | 500.99 | 143.42 | 943.87 |
| bd | BrainTokenizer | 501.03 | error: warm-up RuntimeError('[Rank:0] FATAL ERROR :: MODULE:PT_BRIDGE Exception in La | 822.67 |
| bd | Brant | 179.73 | 48.99, error: inference RuntimeError("hpu_complex: torch.fft.rfft under torch.inference_mode i | 220.18 |
| bd | CBraMod | 1350.74 | error: warm-up RuntimeError('matrix 1 and matrix 2 shapes cannot be multiplied (32x30 | 2910.65 |
| bd | CSBrain | 1500.82 | error: warm-up RuntimeError('matrix 1 and matrix 2 shapes cannot be multiplied (32x30 | 4083.26 |
| bd | CTNet | 281.72 | 50.06 | 301.99 |
| bd | CodeBrain | 1579.58 | 195.3, error: inference RuntimeError("hpu_complex: torch.fft.rfft under torch.inference_mode i | 4259.81 |
| bd | ContraWR | 197.97 | 31.95 | 159.94 |
| bd | DANCE | 460.16 | 85.15 | 459.1 |
| bd | DGCNN | 17.58 | 9.27 | 24.08 |
| bd | DIVER1 | 1384.94 | 195.73 | 4242.53 |
| bd | Deep4Net | 93.28 | 19.77 | 39.3 |
| bd | DeepSleepNet | 219.89 | 41.65 | 222.64 |
| bd | DuIN | 376.91 | 82.57 | 538.02 |
| bd | EEGCLIP | 103.86 | 24.74 | 100.59 |
| bd | EEGConformer | 299.77 | 49.85 | 305.67 |
| bd | EEGDINO | 1305.42 | 112.18, error: inference RuntimeError("hpu_complex: torch.fft.rfft under torch.inference_mode i | 3620.1 |
| bd | EEGITNet | 200.22 | 32.34 | 76.42 |
| bd | EEGInceptionERP | 105.27 | 27.26 | 77.65 |
| bd | EEGInceptionMI | 243.18 | error: warm-up RuntimeError('[Rank:0] FATAL ERROR :: MODULE:PT_BRIDGE Exception in La | 242.79 |
| bd | EEGMiner | 76.34 | 17.51 | 65.03 |
| bd | EEGNeX | 76.98 | 17.84 | 58.52 |
| bd | EEGNet | 58.68 | 12.01 | 20.46 |
| bd | EEGPT | 220.05 | 58.94 | 219.73 |
| bd | EEGSimpleConv | 83.64 | 21.2 | 83.41 |
| bd | EEGSym | 342.38 | 49.58 | 336.92 |
| bd | EEGTCNet | 80.86 | 28.33 | 101.17 |
| bd | EMG2QwertyNet | 191.03 | 109.8, error: inference RuntimeError("hpu_complex: torch.fft.rfft under torch.inference_mode i | 191.47 |
| bd | EpiNT | 261.09, error: inference RuntimeError('Graph compile failed. synStatus=synStatus 26 [Generic fa | 48.65 | 282.55, error: inference RuntimeError('Graph compile failed. synStatus=synStatus 26 [Generic fa |
| bd | FBCNet | 38.96 | 14.19 | 40.89 |
| bd | FBLightConvNet | 38.08 | 8.44 | 25.98 |
| bd | FBMSNet | 43.33 | 12.95 | 43.61 |
| bd | GCNsNet | error: warm-up RuntimeError('Number of classes cannot be -1') | error: warm-up RuntimeError('Number of classes cannot be -1') | error: warm-up RuntimeError('Number of classes cannot be -1') |
| bd | Guetschel2026 | 120.78 | 37.97 | 81.05 |
| bd | IFNet | 74.51 | 13.29 | 59.57 |
| bd | LUNA | 521.52 | 103.65, error: inference RuntimeError("hpu_complex: torch.fft.rfft under torch.inference_mode i | 561.41 |
| bd | Labram | 933.8 | 113.02 | 3575.9 |
| bd | MAPA | 424.27 | 76.38, error: inference RuntimeError("hpu_complex: torch.fft.rfft under torch.inference_mode i | 476.93 |
| bd | MEDFormer | 657.08 | 108.55 | 823.47 |
| bd | MIRepNet | 278.36 | 51.29 | 300.71 |
| bd | MSCFormer | 258.21 | 48.59 | 277.88 |
| bd | MSVTNet | 199.8 | 40.47 | 219.33 |
| bd | MVPFormer | 174.32 | 49.49 | 236.23 |
| bd | MetaNeuromotorHand | 5653.45 | error: warm-up RuntimeError('hpu_complex: a permuted complex64 HPU tensor of rank 5 o | 5319.11 |
| bd | NeuroPose | 358.99 | 71.12 | 423.26 |
| bd | NeuroRVQ | 562.29 | 101.99 | 697.73 |
| bd | NeuroRVQTokenizer | error: warm-up RuntimeError('element 0 of tensors does not require grad and does not  | error: warm-up RuntimeError('element 0 of tensors does not require grad and does not  | error: warm-up RuntimeError('element 0 of tensors does not require grad and does not  |
| bd | PBT | 101.81 | 21.46 | 175.54 |
| bd | PopulationTransformer | 200.17 | 52.92 | 274.01 |
| bd | REVE | 398.67 | 105.79 | 360.27 |
| bd | SCCNet | 22.04 | 9.52 | 22.07 |
| bd | SPARCNet | 922.0 | 154.38 | 761.8 |
| bd | SSTDPN | 37.55 | 19.24 | 60.75 |
| bd | STEEGFormer | 322.32 | 54.77 | 261.0 |
| bd | SeizureTransformer | 618.58 | 120.45 | 778.94 |
| bd | SensingDynamics | 123.19 | crashed:  the process died while running this model | 140.09 |
| bd | ShallowFBCSPNet | 21.9 | 10.88 | 21.1 |
| bd | SignalJEPA | 318.42 | 62.98 | 323.02 |
| bd | SignalJEPA_Contextual | 338.71 | 61.16 | 341.78 |
| bd | SignalJEPA_PostLocal | 37.73 | 11.15 | 39.52 |
| bd | SignalJEPA_PreLocal | 37.94 | 11.58 | 40.18 |
| bd | SincShallowNet | 38.37 | 8.31 | 41.24 |
| bd | SleepFM | 302.62 | 68.87 | 324.32 |
| bd | SleepFMStager | 343.86 | 80.53 | 377.39 |
| bd | SleepStagerBlanco2020 | 79.94 | 22.31 | 99.96 |
| bd | SleepStagerChambon2018 | 39.85 | 10.02 | 39.73 |
| bd | SyncNet | 23.54 | 9.45 | 37.48 |
| bd | TCFormer | 218.5 | 45.38 | 236.63 |
| bd | TFMTokenizer | 577.88 | 88.12, error: inference RuntimeError("hpu_complex: torch.fft.rfft under torch.inference_mode i | 542.46 |
| bd | TIDNet | 80.23 | 23.46 | 78.16 |
| bd | TMSANet | 104.28 | 21.1 | 99.38 |
| bd | TSception | 39.22 | 11.69 | 40.83 |
| bd | USleep | 380.03 | 130.93 | 459.38 |
| bd | VEMG2Pose | 5141.15 | error: warm-up RuntimeError('[Rank:0] FATAL ERROR :: MODULE:PT_BRIDGE Exception in La | 6141.48 |
| bd | ZUNA | 444.64 | 145.4 | 760.26 |
| spd | CorAtt | 299.34 | 71.24 | 158.97 |
| spd | CorNet | 220.26 | 36.52 | 73.91 |
| spd | DeepGeoCCA | 258.69 | 75.49 | error: warm-up RuntimeError('[Rank:0] FATAL ERROR :: MODULE:PT_EAGER HabanaLaunchOpPT |
| spd | EEGSPDNet | 129.63 | 76.04 | error: warm-up RuntimeError('[Rank:0] FATAL ERROR :: MODULE:PT_EAGER HabanaLaunchOpPT |
| spd | GBWAtt | 2398.82 | 862.1 | 1334.52 |
| spd | GraphCSPNet | 718.73 | 583.4 | error: warm-up RuntimeError('synLaunch failed. synStatus=synStatus 35 [Dynamic patchi |
| spd | Green | 180.51 | crashed:  the process died while running this model | 158.67 |
| spd | GyroAtt | 162.73 | 50.37 | 116.96, error: inference KeyError('output_strides') |
| spd | GyroSPD | 143.19 | 47.68 | 136.8 |
| spd | MAtt | 141.27 | 51.49 | 81.8, error: inference KeyError('output_strides') |
| spd | PhaseSPDNet | 52.78 | 17.64 | 19.63 |
| spd | RCEEGNet | 74.15 | 21.28 | 44.46 |
| spd | SPDMetaFormer | 300.19 | 100.99 | 221.56 |
| spd | SPDNet | 40.46 | 19.75 | 17.99 |
| spd | SPDNetAE | 143.94 | 46.25 | 93.21 |
| spd | SPDResNet | 238.66 | 39.92 | 57.33 |
| spd | SPNN | 259.69 | 77.69 | 177.77 |
| spd | SSGCNet | 99.33 | 36.73 | 56.16 |
| spd | STaRNet | 138.46 | 53.94 | error: warm-up RuntimeError('[Rank:0] FATAL ERROR :: MODULE:PT_EAGER HabanaLaunchOpPT |
| spd | TSMNet | 155.99 | 36.13 | 66.9 |
| spd | TensorCSPNet | 300.78 | 170.16 | 247.59, error: inference KeyError('output_strides') |
