#!/bin/bash
# Run tests/model_steps.py once per case, each in its own process (one card; sequential). Lazy/eager comes from the
# caller's PT_HPU_LAZY_MODE unless a case starts with L= (L=1 or L=0).
# usage: bash repro/run_models.sh "--model STaRNet --ch 64" "L=1 --model TensorCSPNet --k resolve" ...
cd "$(dirname "$0")/.." || exit 1
for case in "$@"; do
  mode=$PT_HPU_LAZY_MODE
  case $case in L=*) mode=${case:2:1}; case=${case:4};; esac
  # shellcheck disable=SC2086
  PT_HPU_LAZY_MODE=$mode timeout 900 python3 tests/model_steps.py $case 2>&1 | grep -E '^RESULT' \
    || echo "RESULT {\"case\": \"$case\", \"lazy\": $mode, \"status\": \"no output (timeout/abort)\"}"
done
