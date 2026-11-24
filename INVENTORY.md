# CPU executions of Gaudi steps: the inventory

Every CPU execution that hpu-ops' strict mode (`HPU_OPS_STRICT=report`, `hpu_ops.strict("report")`) saw in one
training step and one inference forward of each model, on one Gaudi2 card (gaudi-docker 1.21.4, bridge 1.21.4,
torch 2.6.0), hpu-ops at 97065c2, hpu-complex at 754ff53, braindecode at dafe8339 (the G12/G13 branch),
spd_learn at 16b39ce, November 24, 2026. BrainModule's and FNIRS4D's lazy rows were re-measured at
hpu-ops 5bc0116 (the #70 GLU backend fix): both pass strict with zero rows. Targeted rows below were
re-measured at 7d9f8b7 (#72): CBraMod and CSBrain pass lazy after `hpu_ops.materialize`, BBL,
BrainTokenizer and VEMG2Pose pass lazy with `--segment-topchild`, and NeuroRVQTokenizer passes eager
and compiled after the sweep's grad-carrying target fix. At 88bf934 (#74) GCNsNet passes eager,
lazy and compiled (the wide-kernel `max_pool1d` composite), EpiNT passes eager and lazy (the
`instance_norm` composite; compiled inference passes under `no_grad`), and CareSleepNet passes
segmented lazy. Final totals: eager 104/104, compiled 104/104, lazy 99/104 in the stock recount
(103/104 with the segmented pass; NeuroPose, the fifth stock failure, passes isolated, so every
model has a passing lazy route). Causes and fixes: `COMPATIBILITY.md`.

- Models: the 21 `spd_diag.models.MODELS` (BNCI2014_001, 22 channels, real batch of 32, cross-entropy) and the
  braindecode models of `braindecode.models.util.models_mandatory_parameters` (104 census rows, braindecode-kernels'
  `patch_braindecode()`, random batch of 32 at each model's test signal parameters, mean-square loss). braindecode is
  not installed on the pod: its source, the plugins and its missing pure-Python dependencies were put on
  `PYTHONPATH` from the census staging directories.
- Step: forward, backward, Adam. One warm-up step, then strict mode over 3 steps (counts are per step), then the mean
  of 5 synced steps outside strict mode (ms/step), then one `torch.inference_mode()` forward under strict mode.
- Modes: eager (`PT_HPU_LAZY_MODE=0`), lazy (`=1`), and eager + `torch.compile(backend="hpu_backend", dynamic=False)`
  (both families; Habana's default-dynamic inference compile has its own bug, #63).
- Impact = the sum, over (model, mode) and over training step and inference forward, of the count per step.
- Reproduce: `bash tests/strict_sweep.sh OUTDIR "spd:eager spd:lazy spd:compile bd:eager bd:lazy bd:compile"`
  under the staging `PYTHONPATH` above (the compile runs used a `dynamic=False` copy of `strict_sweep.py`), then
  `python3 tests/strict_inventory.py OUTDIR`.

## Root causes, by impact

Everything from the November 23 inventory is closed except the rows below. Closed since then, with the change
that closed each: G1/G4/G14 (spd_learn#66, merged, with the LSM/AIRM syncs in spd_learn#68/#69), G2-residual
(spd_learn#67 device triangular indices, spd_learn#70 `vec_to_sym` strict indices), and two lazy-only rows from
the root hunt: EEGInceptionMI's split-conv backward graph (the lazy 129..255 band now splits only where the
native input gradient was measured wrong, #66) and MEGGPT's argmax (a same-shape reshape before `argmax.out`,
#65). GCNsNet's `one_hot` blocker is fixed (#67); after materialization it reaches a separate Habana
graph-compile failure in every mode.

| rank | id | CPU execution | source | impact | models | status / owner |
|---|---|---|---|---|---|---|
| 1 | G9 | complex128 computed on the CPU in host memory (SensingDynamics' float64 filter response, by design) | hpu_complex | 192 | SensingDynamics | upstream float32-response precision decision |
| 2 | G3-f64 | `linalg.eigh` of float64 inputs through hpu-ops' CPU route | hpu_ops route | 180 | GBWAtt | precision policy: Gaudi2 has no float64 and the float64 fixed-point mean is a tested design contract; accept the route, set `HPU_OPS_FLOAT64=card` (float32 precision), or build double-single spectral math (#64) |
| 3 | G15 | `unique_dim` falls back and `bincount` reads its value range under compile | bridge fallback + host sync | trace only | BrainOmni, BrainTokenizer | Habana bug (reported with the #53 reproducers); steady-state steps are zero |

## Not CPU executions, but blockers seen on the way (each a separate issue)

- Lazy braindecode: 92 ok / 11 error / 1 crashed in the original census; the full recount on the
  current stack gives 99/104 stock and 4/4 segmented (103/104; the fifth stock failure, NeuroPose,
  passes isolated). Habana's lazy fuser rejects the full-step graph
  (BBL, BrainTokenizer; VEMG2Pose at inference): the sweep's `--segment-topchild` mode (one
  `mark_step` after each top-level child) passes all three, and CareSleepNet segmented too (its
  unsegmented lazy inference hangs, even in a fresh process; training alone passes).
  `HabanaParameterWrapper` breaks `UninitializedParameter` materialization (#68, with a bare
  `nn.LazyLinear` repro): the sweep now calls `hpu_ops.materialize(model, X)` on the CPU first, and
  CBraMod and CSBrain pass. GCNsNet then reached its last pooling (kernel 2 over length 1), now a
  composite (#74). SyncNet's census crash was the knock-on from CareSleepNet's fatality in the same
  census process (it passes isolated), as do Brant, NeuroPose and FNIRST, whose census errors do
  not reproduce alone.
- EpiNT eager/compiled inference: synapse graph compile failure (also in the old inventory).
- Default-dynamic compiled inference: Habana's recipe compiler raises `KeyError('output_strides')` (#63,
  TensorCSPNet, GyroAtt; reproduces stock, no plugins). This census compiled with `dynamic=False`.
- Fixed harness gap: NeuroRVQTokenizer's census loss picked the target (the first tuple element, which
  has no grad by design); the sweep now trains on the first grad-carrying tensor, and eager and
  compiled pass (one host-sync row remains from the model's own `initialize`).

## Full table (every entry)

Sources: `bridge fallback` (Habana's CPU fallback), `hpu_ops route` (hpu-ops' own CPU code), `hpu_complex`,
`host sync` (a device value read on the host), `h2d copy` (a CPU tensor built in the step and copied to the card),
`cpu op` (ops computing on CPU tensors). `host sync`, `h2d copy` and `cpu op` entries are per caller line.

| # | source | CPU op | impact | models eager / lazy / compile / inference | count per step (max) | example caller | proposed fix |
|---|---|---|---|---|---|---|---|
| 1 | hpu_ops route | `aten::_linalg_eigh` | 180 | 1 / 1 / 1 / 1 | 20 | `spd_learn/functional/autograd.py:36 in modeig_forward` |  |
| 2 | hpu_complex | `aten::mul.Tensor` | 66 | 1 / 1 / 1 / 1 | 10 | `braindecode/models/sensingdynamics.py:415 in forward` |  |
| 3 | hpu_complex | `aten::add.Tensor` | 60 | 1 / 1 / 1 / 1 | 10 | `braindecode/models/sensingdynamics.py:417 in forward` |  |
| 4 | hpu_complex | `aten::pow.Tensor_Scalar` | 33 | 1 / 1 / 1 / 1 | 5 | `braindecode/models/sensingdynamics.py:414 in forward` |  |
| 5 | hpu_complex | `aten::zeros_like` | 12 | 1 / 1 / 1 / 1 | 2 | `braindecode/models/sensingdynamics.py:408 in forward` |  |
| 6 | hpu_complex | `aten::abs` | 6 | 1 / 1 / 1 / 1 | 1 | `braindecode/models/sensingdynamics.py:421 in forward` |  |
| 7 | hpu_complex | `aten::div.Tensor` | 6 | 1 / 1 / 1 / 1 | 1 | `braindecode/models/sensingdynamics.py:420 in forward` |  |
| 8 | hpu_complex | `aten::exp` | 6 | 1 / 1 / 1 / 1 | 1 | `braindecode/models/sensingdynamics.py:406 in forward` |  |
| 9 | hpu_complex | `aten::square` | 3 | 0 / 0 / 0 / 1 | 0 | `braindecode/models/sensingdynamics.py:422 in forward` |  |
| 10 | host sync | `bincount: value range` | 0 | 0 / 0 / 2 / 0 | 0 | `braindecode/modules/quantization.py:194 in _kmeans` |  |
| 11 | bridge fallback | `aten::unique_dim` | 0 | 0 / 0 / 2 / 0 | 0 | `braindecode/modules/quantization.py:184 in _kmeans` |  |

| family | model | eager (ms/step, status) | lazy (ms/step, status) | compile (ms/step, status) |
|---|---|---|---|---|
| bd | ATCNet | 100.9 | 67.12 | 2385.55 |
| bd | AXON | 36.03 | 27.22 | 727.95 |
| bd | AttentionBaseNet | 16.97 | 7.77 | 14.15 |
| bd | AttnSleep | 74.28 | 42.26 | 81.47 |
| bd | BBL | 44.62 | 47.47 (segmented, #72) | 50.15 |
| bd | BDTCN | 20.01 | 19.95 | 23.31 |
| bd | BENDR | 553.99 | 32.24 | 760.03 |
| bd | BIOT | 45.31 | 30.28 | 49.71 |
| bd | BaRISTA | 146.94 | 82.7 | 2555.1 |
| bd | BrainBERT | 21.66 | 19.09 | 41.27 |
| bd | BrainModule | 57.92 | 29.17 | 56.57 |
| bd | BrainOmni | 183.55 | 119.33 | 278.49 |
| bd | BrainTokenizer | 157.38 | 102.89 (segmented, #72) | 211.52 |
| bd | Brant | 55.03 | error: warm-up RuntimeError('[Rank:0] FATAL ERROR :: MODULE:PT_BRIDGE Exception in La | 92.18 |
| bd | CBraMod | 476.7 | 149.64 (materialized, #72) | 618.21 |
| bd | CORTEG | 24.93 | 21.12 | 38.61 |
| bd | CSBrain | 491.68 | 155.17 (materialized, #72) | 651.73 |
| bd | CTNet | 57.23 | 52.49 | 100.12 |
| bd | CareSleepNet | 155.88 | 122.52 (segmented, #74) | 344.16 |
| bd | CodeBrain | 549.34 | 174.37 | 684.99 |
| bd | ContraWR | 30.38 | 24.2 | 54.04 |
| bd | DANCE | 26.94 | 21.93 | 36.15 |
| bd | DGCNN | 6.71 | 5.62 | 8.6 |
| bd | DIVER1 | 553.88 | 141.72 | 675.21 |
| bd | Deep4Net | 16.13 | 14.34 | 716.04 |
| bd | DeepSOZ | 53.06 | 27.36 | 80.17 |
| bd | DeepSleepNet | 35.09 | 33.43 | 49.84 |
| bd | DuIN | 102.78 | 70.93 | 118.36 |
| bd | EEGCLIP | 23.15 | 18.24 | 42.71 |
| bd | EEGConformer | 62.41 | 44.16 | 105.68 |
| bd | EEGDINO | 479.6 | 89.71 | 611.7 |
| bd | EEGITNet | 35.6 | 31.07 | 930.85 |
| bd | EEGInceptionERP | 28.43 | 17.98 | 634.74 |
| bd | EEGInceptionMI | 103.31 | 62.58 | 102.37 |
| bd | EEGMiner | 17.18 | 20.01 | 25.67 |
| bd | EEGNeX | 12.82 | 16.92 | 18.43 |
| bd | EEGNet | 10.16 | 7.9 | 416.36 |
| bd | EEGPT | 114.48 | 60.39 | 112.68 |
| bd | EEGSimpleConv | 16.78 | 14.27 | 17.09 |
| bd | EEGSym | 85.45 | 50.0 | 132.89 |
| bd | EEGTCNet | 22.21 | 19.83 | 36.82 |
| bd | EMG2QwertyNet | 191.76 | 118.25 | 191.85 |
| bd | EpiNT | 110.31 (#74) | 45.93 (#74) | 118.6, error: inference RuntimeError('Graph compile failed. synStatus=synStatus 26 [Generic fa |
| bd | FBCNet | 6.16 | 8.76 | 10.34 |
| bd | FBLightConvNet | 5.82 | 6.97 | 8.49 |
| bd | FBMSNet | 13.83 | 11.35 | 14.9 |
| bd | FNIRS4D | 58.16 | 40.83 | 85.31 |
| bd | FNIRSCNN1D | 212.78 | 7.3 | 280.6 |
| bd | FNIRSNet | 6.3 | 5.2 | 6.35 |
| bd | FNIRSPreT | 76.21 | 76.36 | 122.83 |
| bd | FNIRST | 76.57 | error: warm-up RuntimeError('[Rank:0] FATAL ERROR :: MODULE:PT_BRIDGE Exception in La | 101.23 |
| bd | GCNsNet | 132.42 (#74) | 71.75 (#74) | 1088.84 (#74) |
| bd | GeoNIRS | 138.01 | 85.02 | 209.93 |
| bd | Guetschel2026 | 25.0 | 29.91 | 32.96 |
| bd | IFNet | 10.43 | 9.41 | 14.89 |
| bd | JungMAE | 42.93 | 36.24 | 63.32 |
| bd | LUNA | 415.29 | 89.77 | 557.33 |
| bd | Labram | 270.89 | 115.5 | 410.32 |
| bd | MAPA | 293.06 | 107.03 | 291.36 |
| bd | MEDFormer | 206.56 | 118.49 | 323.32 |
| bd | MEGGPT | 1000.66 | 294.22 | 1170.75 |
| bd | MEGXL | 114.02 | 60.27 | 216.27 |
| bd | MIRepNet | 77.48 | 39.31 | 105.87 |
| bd | MSCFormer | 57.81 | 45.62 | 108.72 |
| bd | MSVTNet | 44.51 | 40.84 | 75.67 |
| bd | MVPFormer | 54.66 | 43.46 | 85.62 |
| bd | MetaNeuromotorHand | 3575.47 | 4830.77 | 3789.1 |
| bd | MtReNIRS | 75.0 | 56.51 | 112.12 |
| bd | NICE | 10.03 | 8.16 | 16.15 |
| bd | NeuroPose | 131.98 | error: warm-up RuntimeError('[Rank:0] FATAL ERROR :: MODULE:PT_BRIDGE Exception in La | 146.43 |
| bd | NeuroRVQ | 743.22 | 97.17 | 1021.17 |
| bd | NeuroRVQTokenizer | 919.97 (#72) | error: warm-up RuntimeError('element 0 of tensors does not require grad and does not  | 2490.31 (#72) |
| bd | PBT | 26.29 | 21.46 | 35.66 |
| bd | PopulationTransformer | 43.72 | 43.42 | 89.78 |
| bd | Pratibha2 | 19.76 | 18.8 | 402.76 |
| bd | REVE | 15.17 | 22.79 | 20.51 |
| bd | SAMBA | 1061.56 | 199.03 | 1317.43 |
| bd | SCCNet | 5.05 | 7.5 | 7.76 |
| bd | SPARCNet | 164.34 | 94.58 | 245.87 |
| bd | SSTDPN | 9.68 | 13.55 | 15.39 |
| bd | STEEGFormer | 63.42 | 50.06 | 88.81 |
| bd | SeizureTransformer | 100.87 | 55.72 | 127.72 |
| bd | SensingDynamics | 22.11 | 18.89 | 34.3 |
| bd | ShallowFBCSPNet | 5.32 | 6.29 | 9.7 |
| bd | SignalJEPA | 208.56 | 62.57 | 320.41 |
| bd | SignalJEPA_Contextual | 211.64 | 62.26 | 316.57 |
| bd | SignalJEPA_PostLocal | 128.74 | 9.89 | 185.39 |
| bd | SignalJEPA_PreLocal | 156.29 | 7.58 | 193.75 |
| bd | SincShallowNet | 7.59 | 7.26 | 13.24 |
| bd | SleepFM | 77.9 | 410.88 | 143.29 |
| bd | SleepFMStager | 89.96 | 488.8 | 121.55 |
| bd | SleepStagerBlanco2020 | 16.83 | 17.8 | 32.99 |
| bd | SleepStagerChambon2018 | 7.39 | 6.4 | 12.88 |
| bd | SyncNet | 6.13 | crashed:  the process died while running this model | 10.61 |
| bd | TCCNet | 38.32 | 20.42 | 38.87 |
| bd | TCFormer | 44.48 | 41.1 | 83.23 |
| bd | TFMTokenizer | 622.61 | 85.52 | 797.75 |
| bd | TIDNet | 37.05 | 18.51 | 29.78 |
| bd | TMSANet | 23.04 | 18.3 | 38.8 |
| bd | TSception | 7.89 | 7.87 | 15.99 |
| bd | TinyMyo | 27.35 | 18.78 | 30.81 |
| bd | USleep | 204.96 | 96.97 | 114.98 |
| bd | VEMG2Pose | 1218.08 | 6435.48 (segmented, #72) | 1603.71 |
| bd | ZUNA | 17.66 | 26.75 | 22.66 |
| spd | CorAtt | 60.65 | 55.59 | 2006.46 |
| spd | CorNet | 35.54 | 37.74 | 1586.23 |
| spd | DeepGeoCCA | 81.62 | 217.05 | 1979.53 |
| spd | EEGSPDNet | 61.72 | 180.76 | 928.74 |
| spd | GBWAtt | 328.11 | 338.5 | 5216.19 |
| spd | GraphCSPNet | 159.26 | 136.43 | 2393.64 |
| spd | Green | 44.58 | 191.2 | 818.76 |
| spd | GyroAtt | 49.04 | 53.08 | 1137.02 |
| spd | GyroSPD | 50.18 | 53.51 | 1288.78 |
| spd | MAtt | 41.72 | 41.28 | 997.15 |
| spd | PhaseSPDNet | 18.8 | 17.16 | 402.06 |
| spd | RCEEGNet | 21.92 | 20.45 | 449.64 |
| spd | SPDMetaFormer | 70.63 | 73.42 | 5323.31 |
| spd | SPDNet | 14.39 | 16.28 | 274.66 |
| spd | SPDNetAE | 42.99 | 45.8 | 672.74 |
| spd | SPDResNet | 42.18 | 56.42 | 1514.91 |
| spd | SPNN | 59.59 | 61.78 | 995.53 |
| spd | SSGCNet | 29.27 | 31.72 | 843.41 |
| spd | STaRNet | 38.52 | 55.89 | 1293.3 |
| spd | TSMNet | 41.53 | 44.82 | 1254.73 |
| spd | TensorCSPNet | 82.49 | 78.11 | 3006.36 |
