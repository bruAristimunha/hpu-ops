#!/bin/bash
# Gaudi gate of the extrema (NaN) fix, one process per check, one at a time (one card). Mode from PT_HPU_LAZY_MODE.
#   PT_HPU_LAZY_MODE=0|1 bash repro/run_extrema_gate.sh <outdir> [step...]   steps: probes linalg parity steps
# "steps" alternates origin/main's hpu_ops (a copy at $SRC/_main, first on PYTHONPATH) and this branch, 3 rounds.
OUT=${1:-/tmp/ex_gate}; shift
STEPS=${*:-probes linalg parity steps}
SRC=$(cd "$(dirname "$0")/.." && pwd)
M=${PT_HPU_LAZY_MODE:-0}
MODELS=SPDNet,PhaseSPDNet,SPDResNet,GyroSPD,CorNet,SPNN,DeepGeoCCA,SPDNetAE,TensorCSPNet,GraphCSPNet,TSMNet,EEGSPDNet,RCEEGNet,STaRNet,SSGCNet,MAtt,GBWAtt,GyroAtt,SPDMetaFormer,CorAtt
mkdir -p "$OUT"
for s in $STEPS; do
  case $s in
    probes)
      timeout 1800 python3 "$SRC/repro/probe_reduction_nan.py" --enable > "$OUT/map_$M.txt" 2>&1; echo "map rc=$? $(tail -1 $OUT/map_$M.txt)"
      timeout 900 python3 "$SRC/repro/probe_extrema_grads.py" > "$OUT/grads_$M.txt" 2>&1; echo "grads rc=$? $(tail -1 $OUT/grads_$M.txt)"
      if [ "$M" = 0 ]; then
        timeout 2400 python3 "$SRC/repro/probe_extrema_grads.py" --compile > "$OUT/grads_compile.txt" 2>&1
        echo "grads compile rc=$? $(tail -1 $OUT/grads_compile.txt)"
        timeout 2400 python3 "$SRC/repro/probe_reduction_nan.py" --enable --compile --sizes 3,1000 > "$OUT/map_compile.txt" 2>&1
        echo "map compile rc=$? $(tail -1 $OUT/map_compile.txt)"
      fi ;;
    linalg) timeout 1800 python3 "$SRC/tests/linalg_on_hpu.py" > "$OUT/linalg_$M.txt" 2>&1; echo "linalg rc=$? $(tail -1 $OUT/linalg_$M.txt)" ;;
    parity) timeout 3600 python3 "$SRC/tests/model_parity.py" --parts grads --models $MODELS > "$OUT/parity_$M.txt" 2>&1
            echo "parity rc=$? $(tail -1 $OUT/parity_$M.txt)" ;;
    steps)
      for r in 0 1 2; do
        for arm in main branch; do
          if [ $arm = main ]; then PP="$SRC/_main:$SRC/_main/tests:$PYTHONPATH"; else PP="$PYTHONPATH"; fi
          PYTHONPATH="$PP" timeout 2400 python3 "$SRC/tests/linalg_steps.py" --baseline > "$OUT/steps_${M}_${arm}_$r.txt" 2>&1
          echo "steps $arm round $r rc=$? $(grep -c '^RESULT' $OUT/steps_${M}_${arm}_$r.txt) results"
        done
      done ;;
  esac
done
