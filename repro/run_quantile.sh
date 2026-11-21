#!/bin/bash
# hpu-ops #17 probe: each case in its own process, native then with hpu_ops.enable(), eager and lazy.
cd "$(dirname "$0")/.." || exit 1
for lz in ${LZ:-1 0}; do for c in ${CASES:-scalar tensorq dim nan nansort grad sort kth perf}; do for en in "" en; do
  PT_HPU_LAZY_MODE=$lz timeout 300 python3 repro/issue17_quantile_lazy.py $c $en 2>&1 | grep -o -E 'PROBE {.*' \
    || echo "PROBE {\"case\": \"$c\", \"lazy\": $lz, \"enable\": \"$en\", \"ok\": false, \"raised\": \"no output (abort)\"}"
done; done; done
