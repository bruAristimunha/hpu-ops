#!/bin/bash
# The #31 review probe (repro/probe_review31.py): every case in its own process, native then hpu_ops, eager and lazy.
#   [LZ="0 1"] [CASES="..."] bash repro/run_review31.sh
cd "$(dirname "$0")/.." || exit 1
ALL="q_grad_nan_row grad_to_nan nan_inf_grad out layout big_q_backward compiled_q_range compiled_infer compiled_perf perf
     conv_bf16_bias conv_fp16_cancel conv_bias_once conv_pad_inf conv_cl_weight conv_lazy_300"
for lz in ${LZ:-0 1}; do for c in ${CASES:-$ALL}; do for en in "" en; do
  [ "$c" = conv_lazy_300 ] && [ -z "$en" ] && [ "$lz" = 1 ] && continue  # native: stops the card (see the README)
  PT_HPU_LAZY_MODE=$lz timeout 600 python3 repro/probe_review31.py "$c" $en 2>&1 | grep -o -E 'PROBE \{.*' \
    || echo "PROBE {\"case\": \"$c\", \"lazy\": $lz, \"enable\": \"$en\", \"ok\": false, \"raised\": \"no output (abort)\"}"
done; done; done
