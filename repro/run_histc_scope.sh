#!/bin/bash
# One histc case per process, eager then lazy (repro/probe_histc_scope.py). Usage on the pod: bash repro/run_histc_scope.sh
D=$(cd "$(dirname "$0")" && pwd)
run() {  # retries while a crashed case still holds the device
  echo "> $*"
  for try in 1 2 3 4 5 6 7 8; do
    out=$(timeout 300 python "$D/probe_histc_scope.py" "$@" 2>&1); rc=$?
    echo "$out" | grep -q "Device acquire failed" || break
    sleep 15
  done
  echo "$out" | grep -E "^(OK|BAD|RAISE|GOT|arith|bucketize|   native|  diff)" || { echo "DIED $* (exit $rc)"; echo "$out" | grep -v -i warn | grep -E "Error|error|FATAL|assert" | head -3; }
  sleep 3
}
for lazy in 0 1; do
  export PT_HPU_LAZY_MODE=$lazy
  echo "== PT_HPU_LAZY_MODE=$lazy"
  run arith
  run bucketize
  for c in "64 0 63" "64 0.0 63.0" "64 -1 63" "64 0 0" "64 0.0 0.0" "64 5 5" "64 -0.5 63.5" "64 2.5 2.5" \
           "300 0 63" "350 0 63" "399 0 63" "400 0 63" "1000 0 63" "4096 -3 3"; do
    run native $c
  done
  for d in "64 -3 3 float32 nan" "64 -3 3 float32 inf" "64 0 0 float32 nan" "64 0 0 float32 inf" \
           "64 0 0 float32 unif" "64 0 0 float32 const" "64 -3 3 bfloat16 unif" "64 -3 3 bfloat16 int" \
           "64 -3 3 float16 unif" "64 -3 3 float64 unif" "64 -3.0 3.0 float64 unif"; do
    run native $d
  done
  for c in "64 0 63" "64 -0.5 63.5" "64 0.0 63.0" "400 0 63"; do run compile $c; done
done
