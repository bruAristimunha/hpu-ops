#!/bin/bash
# Training steps with hpu_ops.extrema off / without sort / on, alternating processes, 4 rounds. Mode from PT_HPU_LAZY_MODE.
SRC=$(cd "$(dirname "$0")/.." && pwd)
for r in 0 1 2 3; do
  for arm in off nosort on; do
    timeout 2400 python3 "$SRC/repro/steps_extrema_ab.py" $arm "$@" 2>&1 | grep '^RESULT' | sed "s/^/$arm $r /"
  done
done
