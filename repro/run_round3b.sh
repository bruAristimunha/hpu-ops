#!/bin/bash
# Round 3b card checks: split threshold 256 eager, 128 lazy and compiled.
cd "$(dirname "$0")/.." || exit 1
P=tests/probe_adversarial_e.py
run() { timeout 900 "$@" 2>&1 | grep -o -E 'PROBE {.*|RESULT {.*' || echo "PROBE {\"cmd\": \"$*\", \"ok\": false, \"raised\": \"no output (timeout/abort)\"}"; }
for c in "conv1d k=351 (STaRNet" "conv2d (4,1089)" "conv1d k=300 stride 3" "conv1d k=600 padding" "conv1d k=351 bf16" \
         "conv2d (300,300)" "conv1d k=257 depthwise" "conv1d k=300 non-contiguous" "conv1d k=351 inference" "conv1d k=300 batch 1"; do
  echo "== eager: $c"; CASE="$c" PT_HPU_LAZY_MODE=0 run python3 $P ops
done
echo "== compile probe"; PT_HPU_LAZY_MODE=0 run python3 $P compile
for c in '351 400 en' '600 1000 en' '1089 1500 en' '351 1000 en'; do
  echo "== compile conv $c"; PT_HPU_LAZY_MODE=0 run python3 repro/probe_compile_conv_k.py $c
done
for m in "--model STaRNet --ch 64 --compile" "--model TensorCSPNet --k resolve --compile" "--model STaRNet --ch 64" "--model TensorCSPNet --k resolve"; do
  echo "== eager $m"; PT_HPU_LAZY_MODE=0 run python3 tests/model_steps.py $m
done
echo "== run_on_hpu eager"; PT_HPU_LAZY_MODE=0 timeout 1500 python3 tests/run_on_hpu.py 2>&1 | grep -E 'FAIL|ERROR|failed|passed|took'
