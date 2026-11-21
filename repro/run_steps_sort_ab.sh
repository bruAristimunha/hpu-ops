#!/bin/bash
# Train steps of the five SPD models (tests/linalg_steps.py --baseline), arms in alternating processes, 3 rounds:
#   main: origin/main's hpu_ops (a git archive at $MAIN first on PYTHONPATH; its sort family off, its default);
#   on: this branch (exact sort on by default); off: this branch with HPU_OPS_EXACT_SORT=0.
# Mode from PT_HPU_LAZY_MODE. One "arm round RESULT {json}" line per model.
#   MAIN=/path/to/origin-main-src bash repro/run_steps_sort_ab.sh
SRC=$(cd "$(dirname "$0")/.." && pwd)
MAIN=${MAIN:?set MAIN to an origin/main checkout}
ARMS=${ARMS:-main on off}
for r in 0 1 2; do
  for arm in $ARMS; do
    if [ "$arm" = main ]; then ROOT=$MAIN; else ROOT=$SRC; fi
    if [ "$arm" = off ]; then E=0; else E=; fi
    env PYTHONPATH="$ROOT:$ROOT/tests:$PYTHONPATH" HPU_OPS_EXACT_SORT=$E timeout 2400 \
      python3 "$ROOT/tests/linalg_steps.py" --baseline 2>&1 | grep -E '^RESULT|^PLUGINS' | sed "s/^/$arm $r /"
  done
done
