#!/bin/bash
# The extrema rewrite's per-call cost, this checkout against another (e.g. main), both with the fix on, alternating
# processes, 4 rounds in alternating order. Mode from PT_HPU_LAZY_MODE.   bash repro/run_extrema_ab_main.sh <other checkout> > ab.txt
SRC=$(cd "$(dirname "$0")/.." && pwd)
OTHER=$1
for r in 0 1 2 3; do
  arms="$SRC $OTHER"; [ $((r % 2)) = 1 ] && arms="$OTHER $SRC"  # the order alternates: no first-process bias
  for arm in $arms; do
    echo "ARM $arm round $r"
    PYTHONPATH="$arm:$arm/tests" timeout 900 python3 "$SRC/tests/bench_extrema.py" --fix on --round $r 2>&1 | grep -E '^BENCH|Error'
  done
done
