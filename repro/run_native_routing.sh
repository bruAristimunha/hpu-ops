#!/bin/bash
# Gaudi2 gate for the large-kernel routing (#19 revision): which shapes split, the two failing models in every mode,
# then this branch against main on models whose convolutions run natively (alternating rounds, one process each).
#   [BRANCH=DIR] bash repro/run_native_routing.sh [route|models|ab] [MAIN_DIR]
# MAIN_DIR holds main's hpu_ops; BRANCH (default: this checkout) the hpu_ops under test, e.g. main + this branch's
# conv.py and quantile.py when the branch is behind main.
cd "$(dirname "$0")/.." || exit 1
part=${1:-all}; main=${2:-probe/main_ref}; rounds=${ROUNDS:-3}
if [ "$part" = route ] || [ "$part" = all ]; then
  for lazy in 0 1; do
    PT_HPU_LAZY_MODE=$lazy bash repro/sweep_conv_native.sh "32 1 4 4356 16 4 4356 1 484 1 en" \
      "32 1 4 1089 16 4 1089 1 121 1 en" "32 1 351 1000 8 351 1 1 1 1 en"
  done
fi
if [ "$part" = models ] || [ "$part" = all ]; then
  for args in "--model STaRNet --ch 64" "--model TensorCSPNet --k resolve"; do
    # shellcheck disable=SC2086
    PT_HPU_LAZY_MODE=0 timeout 900 python3 tests/model_steps.py $args 2>&1 | grep -E '^RESULT'
    PT_HPU_LAZY_MODE=1 timeout 900 python3 tests/model_steps.py $args 2>&1 | grep -E '^RESULT'
    PT_HPU_LAZY_MODE=0 timeout 900 python3 tests/model_steps.py $args --compile 2>&1 | grep -E '^RESULT'
  done
fi
if [ "$part" = ab ] || [ "$part" = all ]; then
  for lazy in 0 1; do
    for r in $(seq "$rounds"); do
      for code in branch main; do
        for args in "--model TensorCSPNet" "--model STaRNet" "--model SPDNet"; do
          pp=${BRANCH:+$BRANCH:}$PYTHONPATH; [ $code = main ] && pp=$main:$PYTHONPATH
          # shellcheck disable=SC2086
          PYTHONPATH=$pp PT_HPU_LAZY_MODE=$lazy timeout 900 python3 tests/model_steps.py $args 2>&1 \
            | grep -E '^(RESULT|PLUGINS)' | sed "s/^/AB code=$code round=$r /"
        done
      done
    done
  done
fi
