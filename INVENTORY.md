# CPU executions of Gaudi steps: the inventory

Every CPU execution that hpu-ops' strict mode (`HPU_OPS_STRICT=report`, `hpu_ops.strict("report")`) saw in one
training step and one inference forward of each model, on one Gaudi2 card (gaudi-docker 1.21.4, bridge 1.21.4,
torch 2.6.0), hpu-ops at 5c318d3, hpu-complex at c663754, braindecode at dafe8339 (the G12/G13 branch), stock
spd_learn from the pod, November 23, 2026.

- Models: the 21 `spd_diag.models.MODELS` (BNCI2014_001, 22 channels, real batch of 32, cross-entropy) and the
  braindecode models of `braindecode.models.util.models_mandatory_parameters` (104 census rows, braindecode-kernels'
  `patch_braindecode()`, random batch of 32 at each model's test signal parameters, mean-square loss). braindecode is
  not installed on the pod: its source, the plugins and its missing pure-Python dependencies were put on
  `PYTHONPATH` from the census staging directories.
- Step: forward, backward, Adam. One warm-up step, then strict mode over 3 steps (counts are per step), then the mean
  of 5 synced steps outside strict mode (ms/step), then one `torch.inference_mode()` forward under strict mode.
- Modes: eager (`PT_HPU_LAZY_MODE=0`), lazy (`=1`), and eager + `torch.compile(backend="hpu_backend")` (spd only;
  see the blocker note). Compiled steps use `strict(per_op=False)` as before.
- Impact = the sum, over (model, mode) and over training step and inference forward, of the count per step.
- Reproduce: `bash tests/strict_sweep.sh OUTDIR "spd:eager spd:lazy spd:compile"`, the bd family with
  `tests/strict_sweep.py --family bd` under the staging `PYTHONPATH` above, then
  `python3 tests/strict_inventory.py OUTDIR`.

## Root causes, by impact

Everything from the November 21 inventory is closed except the rows below. Closed, with the change that closed
each: G2's host linearization and copy (ptr cache #56, value cache #58 — training steps now measure zero; the
`triu_indices` cpu-op row is spd_learn building the CPU tensor itself), G3 float32 (jacobi_big #51), G5/G7/G6
(#56), G8 (#50), G10/G11/G16 (hpu-complex#18), G12/G13 (braindecode#1796). Green, which errored in the old
inventory, now trains (77.8 ms eager).

| rank | id | CPU execution | source | impact | models | status / owner |
|---|---|---|---|---|---|---|
| 1 | G1 | spd_learn GBW metric validation host reads (392/step) | host sync | 1176 | GBWAtt | fixed in spdlearn/spd_learn#66 (open, green; upstream merge) |
| 2 | G3-f64 | `linalg.eigh` of float64 inputs through hpu-ops' CPU route | hpu_ops route | 405 | GBWAtt | precision policy: Gaudi2 has no float64; GBWAtt's metric in float32, or accept the route (loud warning exists) |
| 3 | G4 | spd_learn matrix_log clamp warning host read | host sync | 252 | 17 models | fixed in spdlearn/spd_learn#66 |
| 4 | G2-residual | `triu_indices` built on the CPU by spd_learn each call; compile-trace and first-inference linearization | cpu op + hpu_ops route | ~190 | 17 models | library (cache the indices in spd_learn) or accept: no device sync, no per-step copy left |
| 5 | G14 | spd_learn GraphCSPNet/CorNet host reads | host sync | 72 | 2 models | fixed in spdlearn/spd_learn#66 |
| 6 | G9 | complex128 computed on the CPU in host memory (SensingDynamics' float64 filter response, by design) | hpu_complex | 64 | SensingDynamics | upstream float32-response precision decision |
| 7 | G15 | `unique_dim` falls back under compile | bridge fallback | trace only | BrainOmni, BrainTokenizer | Habana bug (reported with the #53 reproducers) |

## Not CPU executions, but blockers seen on the way (each a separate issue)

- Compiled spd graphs fail recipe compilation for 18/21 models (synStatus 26; fuser: `syn.broadcast` input rank
  larger than output rank). Bisected 2026-11-23: pre-#51 hpu-ops compiles SPDNet, #51 (e3e49e2) does not; single
  ops (eigh/solve/svd/cholesky/triangular/inv at 22) compile fine, forward-only also fails,
  `PT_HPU_FORCE_STATIC_COMPILE=1` does not help, plugins-off compiles. The compiled column therefore holds trace
  rows and errors, not steady-state steps. Needs graph-dump node bisection in the #51 composites.
- Compiled complex det/slogdet still reach the boxed CPU fallback (the dispatcher declines while tracing).
  A custom-op route was measured unsound inside the dispatcher (AOT also differentiates `_linalg_det` by
  formula on placeholder LU); a user-level `torch.compiler.disable` wrapper measures zero. Eager/lazy are zero.
- EpiNT eager inference: synapse graph compile failure (also in the old inventory).
- Lazy braindecode: 88 ok / 14 error / 2 crashed of 104. Errors are Habana bridge/graph failures (EEGInceptionMI,
  VEMG2Pose, BrainTokenizer, Brant, MEGGPT, BBL, NeuroPose, BrainModule, FNIRS4D, FNIRST, CareSleepNet, JungMAE)
  and garbage matmul shapes (CBraMod, CSBrain — also in the old inventory); GCNsNet and NeuroRVQTokenizer are
  harness/model errors as before. No lazy refusal from hpu-complex remains (the rfft/rank-5 refusals are lifted).
- Harness, not gaps: GCNsNet ("Number of classes cannot be -1"), NeuroRVQTokenizer (output has no grad).

## Full table (every entry)

Sources: `bridge fallback` (Habana's CPU fallback), `hpu_ops route` (hpu-ops' own CPU code), `hpu_complex`,
`host sync` (a device value read on the host), `h2d copy` (a CPU tensor built in the step and copied to the card),
`cpu op` (ops computing on CPU tensors). `host sync`, `h2d copy` and `cpu op` entries are per caller line.

| # | source | CPU op | impact | models eager / lazy / compile / inference | count per step (max) | example caller | proposed fix |
|---|---|---|---|---|---|---|---|
| 1 | host sync | `aten::_local_scalar_dense` | 1176 | 1 / 1 / 1 / 0 | 392 | `spd_learn/functional/metrics/generalized_bures_wasserstein.py:31 in _require_spd` |  |
| 2 | hpu_ops route | `aten::_linalg_eigh` | 405 | 1 / 1 / 1 / 1 | 51 | `spd_learn/functional/metrics/generalized_bures_wasserstein.py:40 in _require_spd` |  |
| 3 | host sync | `aten::_local_scalar_dense` | 252 | 17 / 17 / 17 / 0 | 22 | `spd_learn/functional/core.py:209 in forward` |  |
| 4 | cpu op | `aten::triu_indices` | 105 | 17 / 17 / 17 / 17 | 3 | `spd_learn/functional/core.py:654 in sym_to_upper` |  |
| 5 | cpu op | `aten::ne.Tensor` | 88 | 16 / 16 / 16 / 16 | 2 | `spd_learn/functional/core.py:662 in sym_to_upper` |  |
| 6 | host sync | `aten::_local_scalar_dense` | 54 | 1 / 1 / 1 / 0 | 18 | `spd_learn/models/graphcsp.py:354 in forward_covariances` |  |
| 7 | hpu_ops route | `index: CPU index tensors checked and linearized on the host` | 50 | 0 / 0 / 17 / 17 | 3 | `spd_learn/functional/core.py:658 in sym_to_upper` |  |
| 8 | hpu_complex | `aten::mul.Tensor` | 44 | 1 / 1 / 0 / 1 | 10 | `braindecode/models/sensingdynamics.py:415 in forward` |  |
| 9 | hpu_complex | `aten::add.Tensor` | 40 | 1 / 1 / 0 / 1 | 10 | `braindecode/models/sensingdynamics.py:417 in forward` |  |
| 10 | cpu op | `aten::index.Tensor` | 36 | 1 / 1 / 1 / 1 | 6 | `spd_learn/functional/core.py:731 in vec_to_sym` |  |
| 11 | hpu_complex | `aten::pow.Tensor_Scalar` | 22 | 1 / 1 / 0 / 1 | 5 | `braindecode/models/sensingdynamics.py:414 in forward` |  |
| 12 | host sync | `aten::_local_scalar_dense` | 18 | 1 / 1 / 1 / 0 | 6 | `spd_learn/models/cornet.py:127 in _regularized_cov_to_corr` |  |
| 13 | cpu op | `aten::ne.Tensor` | 12 | 1 / 1 / 1 / 1 | 2 | `spd_learn/functional/core.py:722 in vec_to_sym` |  |
| 14 | hpu_complex | `aten::zeros_like` | 8 | 1 / 1 / 0 / 1 | 2 | `braindecode/models/sensingdynamics.py:408 in forward` |  |
| 15 | hpu_complex | `aten::abs` | 4 | 1 / 1 / 0 / 1 | 1 | `braindecode/models/sensingdynamics.py:421 in forward` |  |
| 16 | hpu_complex | `aten::div.Tensor` | 4 | 1 / 1 / 0 / 1 | 1 | `braindecode/models/sensingdynamics.py:420 in forward` |  |
| 17 | hpu_complex | `aten::exp` | 4 | 1 / 1 / 0 / 1 | 1 | `braindecode/models/sensingdynamics.py:406 in forward` |  |
| 18 | hpu_complex | `aten::square` | 2 | 0 / 0 / 0 / 1 | 0 | `braindecode/models/sensingdynamics.py:422 in forward` |  |

| family | model | eager (ms/step, status) | lazy (ms/step, status) | compile (ms/step, status) |
|---|---|---|---|---|
| bd | ATCNet | 696.35 | 73.45 | not run |
| bd | AXON | 143.21 | 37.84 | not run |
| bd | AttentionBaseNet | 22.5 | 12.42 | not run |
| bd | AttnSleep | 236.64 | 51.76 | not run |
| bd | BBL | 283.91 | error: warm-up RuntimeError('[Rank:0] FATAL ERROR :: MODULE:PT_BRIDGE Exception in La | not run |
| bd | BDTCN | 79.92 | 31.77 | not run |
| bd | BENDR | 1223.9 | 37.15 | not run |
| bd | BIOT | 161.34 | 38.42 | not run |
| bd | BaRISTA | 583.21 | 119.43 | not run |
| bd | BrainBERT | 65.57 | 24.13 | not run |
| bd | BrainModule | 142.81 | 39.58, error: inference RuntimeError('Graph compile failed. synStatus=synStatus 26 [Generic fa | not run |
| bd | BrainOmni | 1041.35 | 156.01 | not run |
| bd | BrainTokenizer | 745.66 | error: warm-up RuntimeError('[Rank:0] FATAL ERROR :: MODULE:PT_BRIDGE Exception in La | not run |
| bd | Brant | 221.4 | error: warm-up RuntimeError('[Rank:0] FATAL ERROR :: MODULE:PT_BRIDGE Exception in La | not run |
| bd | CBraMod | 1295.04 | error: warm-up RuntimeError('matrix 1 and matrix 2 shapes cannot be multiplied (32x30 | not run |
| bd | CORTEG | 117.11 | 24.51 | not run |
| bd | CSBrain | 1542.58 | error: warm-up RuntimeError('matrix 1 and matrix 2 shapes cannot be multiplied (32x30 | not run |
| bd | CTNet | 213.47 | 47.12 | not run |
| bd | CareSleepNet | 595.55 | crashed:  the process died while running this model | not run |
| bd | CodeBrain | 1657.05 | 166.98 | not run |
| bd | ContraWR | 181.48 | 35.16 | not run |
| bd | DANCE | 118.09 | 33.85 | not run |
| bd | DGCNN | 21.41 | 10.0 | not run |
| bd | DIVER1 | 1452.04 | 183.89 | not run |
| bd | Deep4Net | 76.59 | 21.66 | not run |
| bd | DeepSOZ | 219.62 | 39.51 | not run |
| bd | DeepSleepNet | 164.11 | 46.01 | not run |
| bd | DuIN | 340.54 | 77.11 | not run |
| bd | EEGCLIP | 83.48 | 26.09 | not run |
| bd | EEGConformer | 259.94 | 55.75 | not run |
| bd | EEGDINO | 1178.27 | 95.59 | not run |
| bd | EEGITNet | 140.7 | 37.86 | not run |
| bd | EEGInceptionERP | 103.67 | 27.0 | not run |
| bd | EEGInceptionMI | 322.88 | error: warm-up RuntimeError('[Rank:0] FATAL ERROR :: MODULE:PT_BRIDGE Exception in La | not run |
| bd | EEGMiner | 58.09 | 21.64 | not run |
| bd | EEGNeX | 53.68 | 15.6 | not run |
| bd | EEGNet | 36.84 | 11.28 | not run |
| bd | EEGPT | 253.42 | 63.02 | not run |
| bd | EEGSimpleConv | 58.3 | 19.97 | not run |
| bd | EEGSym | 297.12 | 67.94 | not run |
| bd | EEGTCNet | 98.34 | 23.86 | not run |
| bd | EMG2QwertyNet | 191.78 | 113.27 | not run |
| bd | EpiNT | 323.8, error: inference RuntimeError('Graph compile failed. synStatus=synStatus 26 [Generic fa | 58.42 | not run |
| bd | FBCNet | 36.8 | 16.61 | not run |
| bd | FBLightConvNet | 21.42 | 9.62 | not run |
| bd | FBMSNet | 40.15 | 15.58 | not run |
| bd | FNIRS4D | 379.33 | 52.56, error: inference RuntimeError('Graph compile failed. synStatus=synStatus 26 [Generic fa | not run |
| bd | FNIRSCNN1D | 410.29 | 12.2 | not run |
| bd | FNIRSNet | 22.16 | 8.06 | not run |
| bd | FNIRSPreT | 361.81 | 96.96 | not run |
| bd | FNIRST | 360.93 | error: warm-up RuntimeError('[Rank:0] FATAL ERROR :: MODULE:PT_BRIDGE Exception in La | not run |
| bd | GCNsNet | error: warm-up RuntimeError('Number of classes cannot be -1') | error: warm-up RuntimeError('Number of classes cannot be -1') | not run |
| bd | GeoNIRS | 576.23 | 119.24 | not run |
| bd | Guetschel2026 | 120.65 | 33.84 | not run |
| bd | IFNet | 40.43 | 12.11 | not run |
| bd | JungMAE | 160.59 | crashed:  the process died while running this model | not run |
| bd | LUNA | 1031.52 | 106.71 | not run |
| bd | Labram | 902.13 | 111.44 | not run |
| bd | MAPA | 480.99 | 84.51 | not run |
| bd | MEDFormer | 724.48 | 123.82 | not run |
| bd | MEGGPT | 2755.35 | error: warm-up RuntimeError('[Rank:0] FATAL ERROR :: MODULE:PT_BRIDGE Exception in La | not run |
| bd | MEGXL | 378.44 | 74.89 | not run |
| bd | MIRepNet | 238.79 | 58.99 | not run |
| bd | MSCFormer | 202.42 | 43.99 | not run |
| bd | MSVTNet | 163.65 | 44.28 | not run |
| bd | MVPFormer | 223.71 | 52.96 | not run |
| bd | MetaNeuromotorHand | 4180.05 | 20266.04 | not run |
| bd | MtReNIRS | 297.59 | 63.61 | not run |
| bd | NICE | 38.54 | 10.51 | not run |
| bd | NeuroPose | 261.66 | error: warm-up RuntimeError('[Rank:0] FATAL ERROR :: MODULE:PT_BRIDGE Exception in La | not run |
| bd | NeuroRVQ | 1717.7 | 115.06 | not run |
| bd | NeuroRVQTokenizer | error: warm-up RuntimeError('element 0 of tensors does not require grad and does not  | error: warm-up RuntimeError('element 0 of tensors does not require grad and does not  | not run |
| bd | PBT | 97.74 | 25.13 | not run |
| bd | PopulationTransformer | 195.76 | 51.33 | not run |
| bd | Pratibha2 | 80.25 | 24.35 | not run |
| bd | REVE | 46.95 | 37.19 | not run |
| bd | SAMBA | 2980.06 | 219.38 | not run |
| bd | SCCNet | 17.88 | 5.39 | not run |
| bd | SPARCNet | 624.09 | 132.74 | not run |
| bd | SSTDPN | 53.63 | 30.02 | not run |
| bd | STEEGFormer | 257.05 | 59.93 | not run |
| bd | SeizureTransformer | 357.23 | 73.72 | not run |
| bd | SensingDynamics | 95.69 | 27.13 | not run |
| bd | ShallowFBCSPNet | 19.46 | 6.27 | not run |
| bd | SignalJEPA | 556.81 | 51.75 | not run |
| bd | SignalJEPA_Contextual | 597.05 | 51.5 | not run |
| bd | SignalJEPA_PostLocal | 310.55 | 13.13 | not run |
| bd | SignalJEPA_PreLocal | 324.71 | 9.85 | not run |
| bd | SincShallowNet | 35.9 | 7.4 | not run |
| bd | SleepFM | 332.53 | 2236.83 | not run |
| bd | SleepFMStager | 357.96 | 2623.72 | not run |
| bd | SleepStagerBlanco2020 | 96.51 | 26.6 | not run |
| bd | SleepStagerChambon2018 | 24.79 | 9.95 | not run |
| bd | SyncNet | 34.85 | 9.38 | not run |
| bd | TCCNet | 117.41 | 27.66 | not run |
| bd | TCFormer | 179.84 | 40.34 | not run |
| bd | TFMTokenizer | 1300.81 | 119.83 | not run |
| bd | TIDNet | 79.16 | 22.25 | not run |
| bd | TMSANet | 99.73 | 27.15 | not run |
| bd | TSception | 52.23 | 12.02 | not run |
| bd | TinyMyo | 94.19 | 23.35 | not run |
| bd | USleep | 558.29 | 86.96 | not run |
| bd | VEMG2Pose | 6178.21 | 16345.1, error: inference RuntimeError('Graph compile failed. synStatus=synStatus 26 [Generic fa | not run |
| bd | ZUNA | 102.96 | 41.83 | not run |
| spd | CorAtt | 98.35 | 71.14 | error: timed steps RuntimeError('[Rank:0] FATAL ERROR :: MODULE:PT_EAGER HabanaLaunchOpPT |
| spd | CorNet | 54.88 | 39.66 | error: timed steps RuntimeError('[Rank:0] FATAL ERROR :: MODULE:PT_EAGER HabanaLaunchOpPT |
| spd | DeepGeoCCA | 107.63 | 267.84 | error: timed steps BackendCompilerFailed('backend=\'hpu_backend\' raised:\nRuntimeError:  |
| spd | EEGSPDNet | 66.59 | 231.88 | error: timed steps RuntimeError('[Rank:0] FATAL ERROR :: MODULE:PT_EAGER HabanaLaunchOpPT |
| spd | GBWAtt | 1571.33 | 563.11 | error: timed steps RuntimeError('[Rank:0] FATAL ERROR :: MODULE:PT_EAGER HabanaLaunchOpPT |
| spd | GraphCSPNet | 367.46 | 312.05 | error: timed steps RuntimeError('[Rank:0] FATAL ERROR :: MODULE:PT_EAGER HabanaLaunchOpPT |
| spd | Green | 81.43 | 254.98 | 1221.91 |
| spd | GyroAtt | 80.67 | 70.84 | error: timed steps BackendCompilerFailed('backend=\'hpu_backend\' raised:\nRuntimeError:  |
| spd | GyroSPD | 90.53 | 78.43 | error: timed steps RuntimeError('[Rank:0] FATAL ERROR :: MODULE:PT_EAGER HabanaLaunchOpPT |
| spd | MAtt | 61.91 | 57.24 | error: timed steps RuntimeError('[Rank:0] FATAL ERROR :: MODULE:PT_EAGER HabanaLaunchOpPT |
| spd | PhaseSPDNet | 21.37 | 20.56 | error: timed steps RuntimeError('[Rank:0] FATAL ERROR :: MODULE:PT_EAGER HabanaLaunchOpPT |
| spd | RCEEGNet | 27.58 | 20.9 | error: timed steps BackendCompilerFailed('backend=\'hpu_backend\' raised:\nRuntimeError:  |
| spd | SPDMetaFormer | 110.97 | 85.57 | error: timed steps RuntimeError('[Rank:0] FATAL ERROR :: MODULE:PT_EAGER HabanaLaunchOpPT |
| spd | SPDNet | 23.58 | 22.44 | error: timed steps RuntimeError('[Rank:0] FATAL ERROR :: MODULE:PT_EAGER HabanaLaunchOpPT |
| spd | SPDNetAE | 64.8 | 66.21 | error: timed steps RuntimeError('[Rank:0] FATAL ERROR :: MODULE:PT_EAGER HabanaLaunchOpPT |
| spd | SPDResNet | 64.45 | 54.49 | error: timed steps RuntimeError('[Rank:0] FATAL ERROR :: MODULE:PT_EAGER HabanaLaunchOpPT |
| spd | SPNN | 111.39 | 92.43 | 2220.16 |
| spd | SSGCNet | 38.31 | 29.47 | error: timed steps RuntimeError('[Rank:0] FATAL ERROR :: MODULE:PT_EAGER HabanaLaunchOpPT |
| spd | STaRNet | 51.68 | 53.56 | 1159.36 |
| spd | TSMNet | 71.17 | 65.51 | error: timed steps RuntimeError('[Rank:0] FATAL ERROR :: MODULE:PT_EAGER HabanaLaunchOpPT |
| spd | TensorCSPNet | 126.27 | 113.86 | 2120.86, error: inference KeyError('output_strides') |
