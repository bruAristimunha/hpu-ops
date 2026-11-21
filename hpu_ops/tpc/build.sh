#!/usr/bin/env bash
# Rebuilds liblinalg_tpc.so, the Gaudi2 TPC linalg kernels (gj_solve and gj_inv, Gauss-Jordan; gj_large{2..8}, the
# in-place Gauss-Jordan inverse for 64 < n <= 512; jacobi_eigh and
# jacobi_eigh_refine, one-sided Jacobi; tri_solve_{lower,upper}[_unit], substitution; jacobi_svd[_full], jacobi_svdvals;
# householder_product; matrix_exp[_frechet]; lu_factor, lu_perm, lu_solve; the complex64 cgj_solve, cgj_inv,
# jacobi_ceigh[_refine], jacobi_csvd[vals]), and copies it next to this script.
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
# a new file renamed over the old one, never written in place: a running job has the old library mapped, and
# overwriting its pages (cp) crashed it with SIGBUS or SIGSEGV; the rename leaves the old inode to it (code review)
cp "$build/liblinalg_tpc.so" "$here/.liblinalg_tpc.so.new.$$"
mv -f "$here/.liblinalg_tpc.so.new.$$" "$here/liblinalg_tpc.so"
sha256sum "$here/liblinalg_tpc.so"
