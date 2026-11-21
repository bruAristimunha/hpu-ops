#!/usr/bin/env bash
# Rebuilds liblinalg_tpc.so, the Gaudi2 TPC linalg kernels (gj_solve and gj_inv, Gauss-Jordan; jacobi_eigh and
# jacobi_eigh_refine, one-sided Jacobi; tri_solve_{lower,upper}[_unit], substitution; jacobi_svd[_full], jacobi_svdvals;
# householder_product; matrix_exp[_frechet]; lu_factor, lu_perm, lu_solve), and copies it next to this
# script.
# Run it inside the Gaudi image (it needs tpc-clang from habanalabs-tools 1.21.4, cmake, objcopy). The build happens
# in a temporary directory; the sha256 of the library is printed so a rebuild can be compared with the committed one.
# Override TPC_COMPILER_PATH and TPC_INCLUDE_DIR in the environment if the tools are not in their default places.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
build="$(mktemp -d)"
trap 'rm -rf "$build"' EXIT

cmake -S "$here" -B "$build" \
    -DTPC_COMPILER_PATH="${TPC_COMPILER_PATH:-/usr/bin/tpc-clang}" \
    -DTPC_INCLUDE_DIR="${TPC_INCLUDE_DIR:-/usr/lib/habanatools/include}"
cmake --build "$build" -j"$(nproc)"
cp "$build/liblinalg_tpc.so" "$here/liblinalg_tpc.so"
sha256sum "$here/liblinalg_tpc.so"
