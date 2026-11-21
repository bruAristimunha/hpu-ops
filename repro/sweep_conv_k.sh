#!/bin/bash
# Run repro/probe_conv_k_bwd.py once per case, each in its own process (one card; sequential).
# usage: bash repro/sweep_conv_k.sh "K L [en] [B C O]" "K L ..." ...
cd "$(dirname "$0")/.." || exit 1
for case in "$@"; do
  # shellcheck disable=SC2086
  timeout 300 python3 repro/probe_conv_k_bwd.py $case 2>&1 | grep -o -E 'PROBE {.*' || echo "PROBE {\"case\": \"$case\", \"ok\": false, \"raised\": \"no output (timeout/abort)\"}"
done
