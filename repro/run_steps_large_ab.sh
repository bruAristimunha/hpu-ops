#!/bin/bash
# Train steps (tests/linalg_steps.py --baseline) with origin/main ($MAIN first on PYTHONPATH), this branch with the
# large route's trust check off (HPU_OPS_LARGE_CHECK=0) and on, alternating processes, $ROUNDS rounds (default 3); mode from
# PT_HPU_LAZY_MODE. Extra arguments go to linalg_steps.py (e.g. --models EEGSPDNet).
#   MAIN=/path/to/origin-main-src bash repro/run_steps_large_ab.sh --models EEGSPDNet
SRC=$(cd "$(dirname "$0")/.." && pwd)
MAIN=${MAIN:?set MAIN to an origin/main checkout}
for r in $(seq 0 $((${ROUNDS:-3} - 1))); do
  for arm in main off on; do
    if [ "$arm" = main ]; then ROOT=$MAIN; else ROOT=$SRC; fi
    if [ "$arm" = off ]; then E=0; else E=; fi
    env PYTHONPATH="$ROOT:$ROOT/tests:$PYTHONPATH" HPU_OPS_LARGE_CHECK=$E timeout 2400 \
      python3 "$ROOT/tests/linalg_steps.py" --baseline "$@" 2>&1 | grep -E '^RESULT' | sed "s/^/$arm $r /"
  done
done
