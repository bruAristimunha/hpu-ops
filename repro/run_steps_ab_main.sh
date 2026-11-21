#!/bin/bash
# Training steps of this checkout against another hpu-ops tree (origin/main), alternating processes.
#   PT_HPU_LAZY_MODE=0|1 bash repro/run_steps_ab_main.sh <main-tree> [rounds] [models]
# Both arms run this checkout's tests/linalg_steps.py --baseline; only the hpu_ops package on PYTHONPATH differs.
SRC=$(cd "$(dirname "$0")/.." && pwd)
MAIN=$1; ROUNDS=${2:-3}; MODELS=${3:-SPDNet,TSMNet,TensorCSPNet,EEGSPDNet,GBWAtt}
for r in $(seq 0 $((ROUNDS - 1))); do
  if [ $((r % 2)) = 0 ]; then ARMS="branch main"; else ARMS="main branch"; fi
  for arm in $ARMS; do
    if [ $arm = main ]; then P=$MAIN; else P=$SRC; fi
    PYTHONPATH=$P:$SRC/tests:$PYTHONPATH timeout 2400 python3 "$SRC/tests/linalg_steps.py" --baseline --models $MODELS \
      2>&1 | grep -E '^(RESULT|PLUGINS)' | sed "s/^/$arm $r /"
  done
done
