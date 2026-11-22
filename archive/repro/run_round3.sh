#!/bin/bash
# Round 3 card checks after the mode-dependent split threshold (256 eager/compiled, 128 lazy).
cd "$(dirname "$0")/.." || exit 1
P=tests/probe_adversarial_e.py
run() { timeout 900 "$@" 2>/dev/null | grep -o -E 'PROBE {.*|RESULT {.*' || echo "PROBE {\"cmd\": \"$*\", \"ok\": false, \"raised\": \"no output (timeout/abort)\"}"; }
for c in "conv1d k=256 exactly" "conv1d k=351 (STaRNet" "conv2d (4,1089)" "conv1d k=300 stride 3" "conv1d k=600 padding" \
         "conv1d k=351 bf16" "conv2d (300,300)" "conv1d k=257 depthwise" "conv1d k=300 non-contiguous" "conv1d k=351 inference"; do
  echo "== eager: $c"; CASE="$c" PT_HPU_LAZY_MODE=0 run python3 $P ops
done
echo "== lazy conv2d (300,300) stderr tail"; CASE="conv2d (300,300)" PHASES=hpu_ops PT_HPU_LAZY_MODE=1 timeout 900 python3 $P ops 2>&1 | grep -v -i warn | tail -5
echo "== compile probe"; PT_HPU_LAZY_MODE=0 run python3 $P compile
echo "== perf eager"; PT_HPU_LAZY_MODE=0 run python3 $P perf
for m in "--model STaRNet --ch 64" "--model TensorCSPNet --k resolve" "--model STaRNet --ch 64 --compile" "--model TensorCSPNet --k resolve --compile"; do
  echo "== eager $m"; PT_HPU_LAZY_MODE=0 run python3 tests/model_steps.py $m
done
echo "== run_on_hpu eager"; PT_HPU_LAZY_MODE=0 timeout 1500 python3 tests/run_on_hpu.py 2>&1 | grep -E 'FAIL|ERROR|failed|passed'
echo "== run_on_hpu lazy (tail)"; PT_HPU_LAZY_MODE=1 timeout 1500 python3 tests/run_on_hpu.py 2>&1 | grep -v -i "warn" | tail -15
