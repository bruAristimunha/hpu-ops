#!/bin/bash
# On the pod: the strict-mode sweep, one card process at a time, resumed until every model has a row.
#   bash tests/strict_sweep.sh OUTDIR "spd:eager spd:lazy spd:compile bd:eager bd:lazy bd:compile"
S=$(cd "$(dirname "$0")/.." && pwd)
OUT=$1; RUNS=$2; mkdir -p "$OUT"
export PYTHONPATH=$S/_ext:$S/_ext/deps:$PYTHONPATH
for run in $RUNS; do
  family=${run%%:*}; mode=${run##*:}
  lazy=0; flag=""; [ "$mode" = lazy ] && lazy=1; [ "$mode" = compile ] && flag=--compile
  for attempt in $(seq 1 12); do
    PT_HPU_LAZY_MODE=$lazy timeout ${CHUNK_TIMEOUT:-3000} python3 "$S/tests/strict_sweep.py" --family "$family" $flag \
      --resume --out "$OUT/${family}_${mode}.jsonl" > "$OUT/log_${family}_${mode}_$attempt.txt" 2>&1
    code=$?
    echo "$(date +%H:%M:%S) $family $mode attempt $attempt exit $code rows $(grep -vc started "$OUT/${family}_${mode}.jsonl")"
    [ $code = 0 ] && break
  done
done
