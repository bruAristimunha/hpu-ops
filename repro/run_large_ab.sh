#!/bin/bash
# repro/probe_large_trust.py in alternating processes, 3 rounds: origin/main's hpu_ops ($MAIN first on PYTHONPATH),
# this branch with the trust check off (HPU_OPS_LARGE_CHECK=0) and on. Eager. One "arm round LINE" per result.
#   MAIN=/path/to/origin-main-src bash repro/run_large_ab.sh
SRC=$(cd "$(dirname "$0")/.." && pwd)
MAIN=${MAIN:?set MAIN to an origin/main checkout}
for r in 0 1 2; do
  for arm in main off on; do
    if [ "$arm" = main ]; then ROOT=$MAIN; else ROOT=$SRC; fi
    if [ "$arm" = off ]; then E=0; else E=; fi
    env PYTHONPATH="$ROOT:$ROOT/tests:$PYTHONPATH" HPU_OPS_LARGE_CHECK=$E PT_HPU_LAZY_MODE=0 timeout 900 \
      python3 "$SRC/repro/probe_large_trust.py" 2>&1 | grep -E '^CHECK|^TIME' | sed "s/^/$arm $r /"
  done
done
