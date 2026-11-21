#!/bin/bash
# The rest of team t3's card gate, sequentially: the sort adversarial probe and the SVD per-call A/B (main vs branch,
# 3 alternating rounds) eager and lazy, then the 20-model gradient parity eager and lazy.
#   MAIN=/path/to/origin-main-src bash repro/run_t3_gate_tail.sh
SRC=$(cd "$(dirname "$0")/.." && pwd)
MAIN=${MAIN:?set MAIN}
for L in 0 1; do
  PT_HPU_LAZY_MODE=$L timeout 900 python3 "$SRC/repro/probe_exact_sort_adv.py" 2>&1 | grep -E '^ADV|^SUMMARY' | sed "s/^/L$L /"
  for r in 0 1 2; do for ROOT in $MAIN $SRC; do
    PT_HPU_LAZY_MODE=$L PYTHONPATH="$ROOT:$ROOT/tests:$PYTHONPATH" timeout 900 python3 "$SRC/repro/probe_svd_time.py" 2>&1 | grep '^TIME' | sed "s/^/L$L $r /"
  done; done
done
for L in 0 1; do
  PT_HPU_LAZY_MODE=$L timeout 5400 python3 "$SRC/tests/model_parity.py" --parts grads 2>&1 | grep -E '^RESULT|^SUMMARY' | sed "s/^/L$L /"
done
# extra rounds of the sort A/B for the models whose first 3 rounds spread past 5% (main vs on)
for L in 0 1; do
  PT_HPU_LAZY_MODE=$L ROUNDS=4 ARMS="main on" MAIN=$MAIN bash "$SRC/repro/run_steps_sort_ab.sh" --models SPDNet,TSMNet,GBWAtt | sed "s/^/L$L /"
done
