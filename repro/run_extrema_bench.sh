#!/bin/bash
# Per-call cost of hpu_ops.extrema on Gaudi: alternating processes, fix on / off, 3 rounds. Mode from PT_HPU_LAZY_MODE.
#   PT_HPU_LAZY_MODE=0|1 bash repro/run_extrema_bench.sh > bench.txt
SRC=$(cd "$(dirname "$0")/.." && pwd)
for r in 0 1 2; do
  for fix in on off; do
    timeout 900 python3 "$SRC/tests/bench_extrema.py" --fix $fix --round $r 2>&1 | grep -E '^BENCH|Error'
  done
done
