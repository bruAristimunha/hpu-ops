#!/bin/bash
# Gaudi gate after the merge with main, one process per check, one at a time (one card). Mode from PT_HPU_LAZY_MODE.
#   PT_HPU_LAZY_MODE=0|1 bash repro/run_merge_gate.sh <outdir> [step...]   steps: linalg parity steps repro
# Green is left out of the parity list: its device error contaminates the next model and crashes lazy runs.
OUT=${1:-/tmp/te_gate}; shift
STEPS=${*:-linalg parity steps repro}
SRC=$(cd "$(dirname "$0")/.." && pwd)
M=${PT_HPU_LAZY_MODE:-0}
MODELS=SPDNet,PhaseSPDNet,SPDResNet,GyroSPD,CorNet,SPNN,DeepGeoCCA,SPDNetAE,TensorCSPNet,GraphCSPNet,TSMNet,EEGSPDNet,RCEEGNet,STaRNet,SSGCNet,MAtt,GBWAtt,GyroAtt,SPDMetaFormer,CorAtt
mkdir -p "$OUT"
for s in $STEPS; do
  case $s in
    linalg) timeout 1800 python3 "$SRC/tests/linalg_on_hpu.py" > "$OUT/linalg_$M.txt" 2>&1; echo "linalg rc=$?" ;;
    parity) timeout 3600 python3 "$SRC/tests/model_parity.py" --parts grads --models $MODELS > "$OUT/parity_$M.txt" 2>&1
            echo "parity rc=$?" ;;
    steps) timeout 2400 python3 "$SRC/tests/linalg_steps.py" --baseline > "$OUT/steps_$M.txt" 2>&1; echo "steps rc=$?" ;;
    repro)
      for c in starnet64 tensorcsp22; do
        timeout 900 python3 "$SRC/repro/issue19_agu_set_associated_dim.py" --case $c --enable > "$OUT/i19_${c}_$M.txt" 2>&1
        echo "i19 $c rc=$?"
      done
      timeout 600 python3 "$SRC/repro/issue17_verbatim.py" > "$OUT/i17v_$M.txt" 2>&1; echo "i17 verbatim rc=$?"
      for c in scalar tensorq dim nan grad; do
        timeout 300 python3 "$SRC/repro/issue17_quantile_lazy.py" $c en 2>&1 | grep -o -E 'PROBE {.*' >> "$OUT/i17_$M.txt" \
          || echo "PROBE {\"case\": \"$c\", \"ok\": false, \"raised\": \"no output (abort)\"}" >> "$OUT/i17_$M.txt"
      done; echo "i17 probe done" ;;
  esac
done
