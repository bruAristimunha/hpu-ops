#!/usr/bin/env bash
# In the Gaudi pod: rebuild liblinalg_tpc.so from the synced sources into a temporary name, then move it in place (a
# half-written library is never on the kernel path). Prints the sha256.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/../hpu_ops/tpc" && pwd)"
build="$(mktemp -d)"
trap 'rm -rf "$build"' EXIT
cmake -S "$here" -B "$build" -DTPC_COMPILER_PATH=/usr/bin/tpc-clang -DTPC_INCLUDE_DIR=/usr/lib/habanatools/include >/dev/null
cmake --build "$build" -j"$(nproc)" > "$build/log" 2>&1 || { grep -E -A4 "error" "$build/log"; exit 1; }
cp "$build/liblinalg_tpc.so" "$here/liblinalg_tpc.so.tmp"
mv "$here/liblinalg_tpc.so.tmp" "$here/liblinalg_tpc.so"
sha256sum "$here/liblinalg_tpc.so"
