#!/bin/bash
# Run repro/probe_conv_native.py once per case, each in its own process (one card; sequential).
# usage: bash repro/sweep_conv_native.sh "B C H W O KH KW SH SW [XGRAD] [en]" ...   (mode from PT_HPU_LAZY_MODE)
cd "$(dirname "$0")/.." || exit 1
for case in "$@"; do
  # shellcheck disable=SC2086
  out=$(timeout 300 python3 repro/probe_conv_native.py $case 2>&1)
  echo "$out" | grep -o -E 'PROBE \{.*' \
    || echo "PROBE {\"case\": \"$case\", \"lazy\": \"$PT_HPU_LAZY_MODE\", \"ok\": false, \"raised\": \"no output (timeout/abort): $(echo "$out" | grep -m1 -o -E 'critical error|Killing|FATAL[^.]*' | head -1)\"}"
done
