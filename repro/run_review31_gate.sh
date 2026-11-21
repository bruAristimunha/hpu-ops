#!/bin/bash
# Gaudi gate of the #31 review fixes, one process per check, one at a time (one card). Logs into OUT.
#   bash repro/run_review31_gate.sh OUT [models] [ab] [quantile] [gate]
# models: the #19 models eager/lazy/compile; ab: default TensorCSPNet / STaRNet(22) / SPDNet against main
# (probe/main_ref, 3 alternating rounds); quantile: the #17 probe eager and lazy; gate: linalg_on_hpu.py and the
# 20-model gradient parity, eager and lazy.
cd "$(dirname "$0")/.." || exit 1
OUT=${1:-/tmp/tp_gate}; shift; mkdir -p "$OUT"
for s in ${*:-models ab quantile gate}; do
  case $s in
    models) bash repro/run_native_routing.sh models > "$OUT/models.txt" 2>&1; echo "models rc=$?" ;;
    ab) ROUNDS=3 bash repro/run_native_routing.sh ab probe/main_ref > "$OUT/ab.txt" 2>&1; echo "ab rc=$?" ;;
    quantile) bash repro/run_quantile.sh > "$OUT/quantile.txt" 2>&1; echo "quantile rc=$?" ;;
    gate) for m in 0 1; do PT_HPU_LAZY_MODE=$m bash repro/run_merge_gate.sh "$OUT/gate" linalg parity; done
          echo "gate done" ;;
  esac
done
