#!/bin/bash
# Round-2 adversarial probe re-run, each lazy case in its own process (a failed lazy graph poisons the rest).
cd "$(dirname "$0")/.." || exit 1
P=tests/probe_adversarial_e.py
run() { timeout 600 "$@" 2>&1 | grep -o -E 'PROBE {.*' || echo "PROBE {\"cmd\": \"$*\", \"ok\": false, \"raised\": \"no output (timeout/abort)\"}"; }
echo "== eager ops"; PT_HPU_LAZY_MODE=0 run python3 $P ops
for c in "conv1d k=256 exactly" "conv1d k=351 (STaRNet" "conv2d (4,1089)" "conv1d k=257 empty" "conv1d k=300 batch 1" \
         "conv1d k=300 stride 3" "conv1d k=600 padding" "conv1d k=351 bf16" "conv2d (300,300)" "conv1d k=257 depthwise" \
         "conv1d k=300 non-contiguous" "conv1d k=351 inference" "bool index_put_ on a column-slice" "index_2d vech f32" \
         "index_2d vech host" "index_2d duplicate" "index_2d int32 indices broadcast" "index_2d 1x1" "index_2d empty index"; do
  echo "== lazy hpu_ops: $c"; CASE="$c" PHASES=hpu_ops PT_HPU_LAZY_MODE=1 run python3 $P ops
done
for v in fwd wgrad xgrad full; do echo "== convlazy 1d $v"; VARIANT=$v PT_HPU_LAZY_MODE=1 run python3 $P convlazy; done
echo "== convlazy 2d full"; VARIANT=full SHAPE=2d PT_HPU_LAZY_MODE=1 run python3 $P convlazy
echo "== perf eager"; PT_HPU_LAZY_MODE=0 run python3 $P perf
echo "== perf lazy hpu_ops"; PHASES=hpu_ops PT_HPU_LAZY_MODE=1 run python3 $P perf
echo "== perf lazy native"; PHASES=native PT_HPU_LAZY_MODE=1 run python3 $P perf
echo "== twice lazy"; PT_HPU_LAZY_MODE=1 run python3 $P twice
echo "== compile eager"; PT_HPU_LAZY_MODE=0 run python3 $P compile
