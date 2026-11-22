"""The Gaudi2 TPC kernels behind ``linalg.solve``, ``inv``, ``det``, ``slogdet``, ``solve_triangular``,
``cholesky_solve`` and ``eigh``:

- ``torch.ops.linalg_tpc.gj_solve(A[b, n, n], B[b, n, m]) -> (X[b, n, m], D[b, n], S[b, 2])`` and
  ``gj_inv(A) -> (X, D, S)`` (B = identity, built in the kernel): Gauss-Jordan with partial pivoting for float32,
  n <= 64 (B columns chunked by 64). One matrix per index-space member, rows in local memory, no LUT-based special
  function (so the 80 KB of local memory are available). ``D`` holds the pivots (U's diagonal), ``S[:, 0]`` the sign
  (-1)^swaps of the row permutation, ``S[:, 1]`` the info: the index follows the kernel's own elimination order, can
  differ from LAPACK's, and is nonzero exactly when a pivot is exactly zero. The elimination is LU-style (a row minus
  ``a_ik / pivot`` times the unnormalized pivot row, the quotient rounded with one FMA correction; X = B / diag(D) at
  the end), so rows that are exact duplicates (or power-of-two multiples) cancel exactly and give a zero pivot,
  where LAPACK's rounded multipliers can leave a residue and report info 0 (other rank-deficient matrices, e.g.
  integer ones, are not detected, as in torch); B is never scaled by 1/pivot before the cancellation, and a
  multiplier below 2^-126 is carried scaled by 2^64 so it is not flushed when its product is normal. A zero pivot makes its elimination step a no-op and its row of X
  zero, so later pivots stay finite.
- ``torch.ops.linalg_tpc.gj_large(A[b, m, n]) -> (R, P, Q, D, S)`` for 64 < n <= ``LARGE_MAXN`` (512), m = n or, a
  tall panel, n < m <= ``TALL_MAXM`` (the blocked route's panels: the pivot rows chosen over all m rows): the in-place
  Gauss-Jordan inverse with partial pivoting (implicit: no row moves), one matrix per index-space member, the first
  rows in local memory and the others re-read from global memory on each step (one ASO per step). ``A^-1 = R[P, :][:,
  Q]``, ``D`` the pivots, ``S[:, 1]`` the info; one ELF per ``ceil(n / 64)``. ``gj_solve`` and ``gj_inv`` route n >
  ``MAXN`` to it on ``A^T`` (LAPACK's pivot order on the matrix torch's CPU kernels factor): ``inv`` is two gathers
  of ``R^T``, ``solve`` the matmul ``(R^T @ B[Q])[Q]``, the sign of ``det`` the parity of ``P``.
- ``torch.ops.linalg_tpc.tri_solve_{lower,upper}[_unit](A[b, n, n], B[b, n, m]) -> X[b, n, m]``: forward or back
  substitution without pivoting for float32, n <= 64 (B columns chunked by 64; ``left=False`` through ``.mT`` and the
  flipped triangle). The other triangle (and the diagonal for ``_unit``) is never read. Each ``X[i]`` is ``B[i]``
  minus the products one at a time in LAPACK's order (never a separate sum, which can overflow where the solution
  does not). Substitution is backward stable, so it keeps CPU float32's accuracy on ill-conditioned triangles, which pivoted Gauss-Jordan does not.

Both work in the normal float32 range: the card flushes subnormal inputs and results to zero (a subnormal entry
counts as zero, so an all-subnormal matrix is singular on the card), while a pivot or diagonal up to the float32
maximum is right (values >= 2^64 are scaled by 2^-64 before the reciprocal, which then never becomes subnormal).

``jacobi_eigh(A[b, n, n]) -> (W[b, n], Vt[b, n, n])`` is a one-sided Jacobi eigensolver of the same shape (float32,
n <= ``EIGH_MAXN``): ``W`` unsorted, the rows of ``Vt`` the eigenvectors; ``eigh`` sorts them. Batch 32, n = 22,
lazy: 0.54 / 1.0 ms per call in two processes against 2.4 / 2.7 for Habana's CPU fallback
(``tests/linalg_percall.py``; the raw kernel took 0.19 ms in an older run); batch 1024: 3.7 ms against 52 ms (older
run). Accuracy within about 3.8x of CPU float32's, also on repeated eigenvalues, condition 1e6 and inputs scaled by
1e30.

The torch-op composites of ``hpu_ops.linalg`` are correct but were measured 3-25x slower than Habana's CPU fallback
per call on Gaudi2 (one launch per op). The solve kernel, batch 32, n = 22, eager: 2.9 / 1.3 ms in two processes against
6.1 / 2.5 for the fallback; lazy 0.97 / 1.36 against 1.22 / 1.38, within the card's noise.

complex64 kernels (planar re/im rows, for hpu-complex) are behind ``hpu_ops.tpc.complex_linalg``: ``cgj_solve`` /
``cgj_inv`` (Gauss-Jordan, cabs1 pivoting, the determinant's phase), ``jacobi_ceigh`` + ``jacobi_ceigh_refine``
(Hermitian eigh) and ``jacobi_csvd`` / ``jacobi_csvdvals`` (one-sided SVD), n <= 32.

Habana reads ``GC_KERNEL_PATH`` when the device initialises, so ``install_kernel_path`` must run before the first HPU
op of the process; ``hpu_ops.enable()`` calls it. ``HPU_OPS_TPC=0`` in the environment keeps the kernels off.
"""
import contextlib
import functools
import math
import os
import pathlib
import time
import warnings

import torch

from ..extrema import internal_sort

MAXN = 64  # MAXN of gj_solve and tri_solve (main)
SMALL_MAXN = 32  # MAXN of jacobi_svd, matrix_exp, lu_factor, lu_perm and lu_solve
MAXHH = 64
LARGE_MAXN = 512  # NMAX of gj_large_f32_gaudi2.c: the square in-place kernel; past it the blocked route (any n)
EIGH_MAXN = 32  # MAXN of jacobi_eigh_f32_gaudi2.c
LIB = pathlib.Path(__file__).with_name("liblinalg_tpc.so")
DEFAULT_KERNELS = "/usr/lib/habanalabs/libtpc_kernels.so"
LATE_WARNING = ("hpu_ops: the HPU was used before hpu_ops.enable(); TPC linalg kernels are off for this process "
                "(call enable() before the first HPU op). The linalg ops then run through hpu-ops' own CPU route, "
                "not Habana's fallback.")
BUILD_WARNING = ("hpu_ops: building the TPC linalg extension failed ({error}); solve, inv, det, slogdet, the "
                 "triangular solves and eigh run through hpu-ops' own CPU route, not Habana's fallback.")
MISSING_WARNING = ("hpu_ops: {path} not found; TPC linalg kernels are off for this process. The linalg ops run "
                   "through hpu-ops' own CPU route, not Habana's fallback.")
OPT_OUT = "HPU_OPS_TPC"
_INSTALLED = False
_PATH_SET = False  # install_kernel_path put LIB on GC_KERNEL_PATH before the device initialised
_LATE_WARNED = False
_MISSING_WARNED = False
_DEVICE = None

# The registration strings are those measured on Gaudi2, plus gj_inv (A alone, identity right-hand side) and
# jacobi_eigh (A alone; W [batch, n], Vt [batch, n, n]). CPP: the
# eager and compile API (bridge 2), CPP_LAZY: the legacy API of lazy mode. S is [batch, 2]: sign and info.
CPP = r"""
#include <torch/extension.h>
#include "hpu_custom_op_pt2.h"

static habana::PartialOutputMetaDataVector solve_meta(const at::Stack& inputs) {
  auto a = inputs[0].toTensor(), b = inputs[1].toTensor();
  auto batch = a.size(0), n = a.size(1);
  return {habana::PartialOutputMetaData{b.scalar_type(), b.sizes().vec()},
          habana::PartialOutputMetaData{b.scalar_type(), {batch, n}},
          habana::PartialOutputMetaData{b.scalar_type(), {batch, 2}}};
}
static habana::PartialOutputMetaDataVector inv_meta(const at::Stack& inputs) {
  auto a = inputs[0].toTensor();
  auto batch = a.size(0), n = a.size(1);
  return {habana::PartialOutputMetaData{a.scalar_type(), a.sizes().vec()},
          habana::PartialOutputMetaData{a.scalar_type(), {batch, n}},
          habana::PartialOutputMetaData{a.scalar_type(), {batch, 2}}};
}
static habana::PartialOutputMetaDataVector svd_meta(const at::Stack& inputs) {
  auto a = inputs[0].toTensor();
  auto b = a.size(0), n = a.size(1), m = a.size(2);
  return {habana::PartialOutputMetaData{a.scalar_type(), {b, n, m}},
          habana::PartialOutputMetaData{a.scalar_type(), {b, n}},
          habana::PartialOutputMetaData{a.scalar_type(), {b, n, n}}};
}
static habana::PartialOutputMetaDataVector svd_full_meta(const at::Stack& inputs) {
  auto a = inputs[0].toTensor();
  auto b = a.size(0), n = a.size(1), m = a.size(2);
  return {habana::PartialOutputMetaData{a.scalar_type(), {b, m, m}},
          habana::PartialOutputMetaData{a.scalar_type(), {b, n}},
          habana::PartialOutputMetaData{a.scalar_type(), {b, n, n}}};
}
static habana::PartialOutputMetaDataVector svdvals_meta(const at::Stack& inputs) {
  auto a = inputs[0].toTensor();
  return {habana::PartialOutputMetaData{a.scalar_type(), {a.size(0), a.size(1)}}};
}
static habana::PartialOutputMetaDataVector hh_meta(const at::Stack& inputs) {
  auto a = inputs[0].toTensor();
  return {habana::PartialOutputMetaData{a.scalar_type(), a.sizes().vec()}};
}
static habana::PartialOutputMetaDataVector lu_meta(const at::Stack& inputs) {
  auto a = inputs[0].toTensor();
  return {habana::PartialOutputMetaData{a.scalar_type(), a.sizes().vec()},
          habana::PartialOutputMetaData{a.scalar_type(), {a.size(0), 3, a.size(1)}}};
}
static habana::PartialOutputMetaDataVector lu_solve_meta(const at::Stack& inputs) {
  auto b = inputs[2].toTensor();
  return {habana::PartialOutputMetaData{b.scalar_type(), b.sizes().vec()}};
}
static habana::PartialOutputMetaDataVector big_meta(const at::Stack& inputs) {
  auto u = inputs[0].toTensor();
  return {habana::PartialOutputMetaData{u.scalar_type(), u.sizes().vec()}};
}
static habana::PartialOutputMetaDataVector eigh_meta(const at::Stack& inputs) {
  auto a = inputs[0].toTensor();
  return {habana::PartialOutputMetaData{a.scalar_type(), {a.size(0), a.size(1)}},
          habana::PartialOutputMetaData{a.scalar_type(), a.sizes().vec()}};
}
static habana::PartialOutputMetaDataVector tri_solve_meta(const at::Stack& inputs) {
  auto b = inputs[1].toTensor();
  return {habana::PartialOutputMetaData{b.scalar_type(), b.sizes().vec()}};
}
static habana::PartialOutputMetaDataVector large_meta(const at::Stack& inputs) {
  auto a = inputs[0].toTensor();
  auto batch = a.size(0), m = a.size(1), n = a.size(2);  // m rows: n, or more for a tall panel
  return {habana::PartialOutputMetaData{a.scalar_type(), {batch, m, 64 * ((n + 63) / 64)}},
          habana::PartialOutputMetaData{a.scalar_type(), {batch, n}},
          habana::PartialOutputMetaData{a.scalar_type(), {batch, m}},
          habana::PartialOutputMetaData{a.scalar_type(), {batch, n}},
          habana::PartialOutputMetaData{a.scalar_type(), {batch, 2}}};
}
static habana::PartialOutputMetaDataVector csolve_meta(const at::Stack& inputs) {
  auto a = inputs[0].toTensor(), b = inputs[1].toTensor();
  auto batch = a.size(0), n = a.size(2);
  return {habana::PartialOutputMetaData{b.scalar_type(), b.sizes().vec()},
          habana::PartialOutputMetaData{b.scalar_type(), {batch, 3, n}},
          habana::PartialOutputMetaData{b.scalar_type(), {batch, 4}}};
}
static habana::PartialOutputMetaDataVector cinv_meta(const at::Stack& inputs) {
  auto a = inputs[0].toTensor();
  auto batch = a.size(0), n = a.size(2);
  return {habana::PartialOutputMetaData{a.scalar_type(), a.sizes().vec()},
          habana::PartialOutputMetaData{a.scalar_type(), {batch, 3, n}},
          habana::PartialOutputMetaData{a.scalar_type(), {batch, 4}}};
}
static habana::PartialOutputMetaDataVector ceigh_meta(const at::Stack& inputs) {
  auto a = inputs[0].toTensor();
  return {habana::PartialOutputMetaData{a.scalar_type(), a.sizes().vec()},
          habana::PartialOutputMetaData{a.scalar_type(), a.sizes().vec()}};
}
static habana::PartialOutputMetaDataVector ceigh_refine_meta(const at::Stack& inputs) {
  auto a = inputs[0].toTensor();
  return {habana::PartialOutputMetaData{a.scalar_type(), {a.size(0), a.size(2)}},
          habana::PartialOutputMetaData{a.scalar_type(), a.sizes().vec()}};
}
static habana::PartialOutputMetaDataVector csvd_meta(const at::Stack& inputs) {
  auto a = inputs[0].toTensor();
  auto b = a.size(0), k2 = a.size(1);
  return {habana::PartialOutputMetaData{a.scalar_type(), a.sizes().vec()},
          habana::PartialOutputMetaData{a.scalar_type(), {b, k2 / 2}},
          habana::PartialOutputMetaData{a.scalar_type(), {b, k2, k2 / 2}}};
}
static habana::PartialOutputMetaDataVector csvdvals_meta(const at::Stack& inputs) {
  auto a = inputs[0].toTensor();
  return {habana::PartialOutputMetaData{a.scalar_type(), {a.size(0), a.size(1) / 2}}};
}
static bool registered_complex = [] {
  habana::custom_op::registerUserCustomOp("linalg_tpc::jacobi_ceigh", "jacobi_ceigh_f32_gaudi2", ceigh_meta, nullptr);
  habana::custom_op::registerUserCustomOp("linalg_tpc::jacobi_ceigh_refine", "jacobi_ceigh_refine_f32_gaudi2",
                                          ceigh_refine_meta, nullptr);
  habana::custom_op::registerUserCustomOp("linalg_tpc::jacobi_csvd", "jacobi_csvd_f32_gaudi2", csvd_meta, nullptr);
  habana::custom_op::registerUserCustomOp("linalg_tpc::jacobi_csvdvals", "jacobi_csvdvals_f32_gaudi2", csvdvals_meta,
                                          nullptr);
  habana::custom_op::registerUserCustomOp("linalg_tpc::cgj_solve", "cgj_solve_f32_gaudi2", csolve_meta, nullptr);
  habana::custom_op::registerUserCustomOp("linalg_tpc::cgj_inv", "cgj_inv_f32_gaudi2", cinv_meta, nullptr);
  return true;
}();
static bool registered = [] {
  habana::custom_op::registerUserCustomOp("linalg_tpc::gj_large", "gj_large_f32_gaudi2", large_meta, nullptr);
  habana::custom_op::registerUserCustomOp("linalg_tpc::jacobi_eigh", "jacobi_eigh_f32_gaudi2", eigh_meta, nullptr);
  habana::custom_op::registerUserCustomOp("linalg_tpc::jacobi_big", "jacobi_big_f32_gaudi2", big_meta, nullptr);
  habana::custom_op::registerUserCustomOp("linalg_tpc::jacobi_big_refine", "jacobi_big_refine_f32_gaudi2", big_meta,
                                          nullptr);
  habana::custom_op::registerUserCustomOp("linalg_tpc::jacobi_eigh_refine", "jacobi_eigh_refine_f32_gaudi2",
                                          eigh_meta, nullptr);
  habana::custom_op::registerUserCustomOp("linalg_tpc::gj_solve", "gj_solve_f32_gaudi2", solve_meta, nullptr);
  habana::custom_op::registerUserCustomOp("linalg_tpc::gj_inv", "gj_inv_f32_gaudi2", inv_meta, nullptr);
  habana::custom_op::registerUserCustomOp("linalg_tpc::jacobi_svd", "jacobi_svd_f32_gaudi2", svd_meta, nullptr);
  habana::custom_op::registerUserCustomOp("linalg_tpc::jacobi_svd_full", "jacobi_svd_full_f32_gaudi2", svd_full_meta,
                                          nullptr);
  habana::custom_op::registerUserCustomOp("linalg_tpc::jacobi_svdvals", "jacobi_svdvals_f32_gaudi2", svdvals_meta, nullptr);
  habana::custom_op::registerUserCustomOp("linalg_tpc::householder_product", "householder_product_f32_gaudi2", hh_meta,
                                          nullptr);
  habana::custom_op::registerUserCustomOp("linalg_tpc::matrix_exp", "matrix_exp_f32_gaudi2", hh_meta, nullptr);
  habana::custom_op::registerUserCustomOp("linalg_tpc::matrix_exp_frechet", "matrix_exp_frechet_f32_gaudi2", hh_meta,
                                          nullptr);
  habana::custom_op::registerUserCustomOp("linalg_tpc::lu_factor", "lu_factor_f32_gaudi2", lu_meta, nullptr);
  habana::custom_op::registerUserCustomOp("linalg_tpc::lu_perm", "lu_perm_f32_gaudi2", svdvals_meta, nullptr);
  habana::custom_op::registerUserCustomOp("linalg_tpc::lu_solve", "lu_solve_f32_gaudi2", lu_solve_meta, nullptr);
  habana::custom_op::registerUserCustomOp("linalg_tpc::tri_solve_lower", "tri_solve_lower_f32_gaudi2", tri_solve_meta, nullptr);
  habana::custom_op::registerUserCustomOp("linalg_tpc::tri_solve_upper", "tri_solve_upper_f32_gaudi2", tri_solve_meta, nullptr);
  habana::custom_op::registerUserCustomOp("linalg_tpc::tri_solve_lower_unit", "tri_solve_lower_unit_f32_gaudi2", tri_solve_meta, nullptr);
  habana::custom_op::registerUserCustomOp("linalg_tpc::tri_solve_upper_unit", "tri_solve_upper_unit_f32_gaudi2", tri_solve_meta, nullptr);
  return true;
}();

std::tuple<at::Tensor, at::Tensor, at::Tensor> gj_solve(at::Tensor a, at::Tensor b) {
  std::vector<c10::IValue> inputs{a, b};
  auto desc = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor("linalg_tpc::gj_solve");
  auto out = desc.execute(inputs);
  return {out[0], out[1], out[2]};
}
std::tuple<at::Tensor, at::Tensor, at::Tensor> gj_inv(at::Tensor a) {
  std::vector<c10::IValue> inputs{a};
  auto desc = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor("linalg_tpc::gj_inv");
  auto out = desc.execute(inputs);
  return {out[0], out[1], out[2]};
}
std::tuple<at::Tensor, at::Tensor, at::Tensor> cgj_solve(at::Tensor a, at::Tensor b) {
  std::vector<c10::IValue> inputs{a, b};
  auto desc = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor("linalg_tpc::cgj_solve");
  auto out = desc.execute(inputs);
  return {out[0], out[1], out[2]};
}
std::tuple<at::Tensor, at::Tensor> jacobi_ceigh(at::Tensor a) {
  std::vector<c10::IValue> inputs{a};
  auto desc = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor("linalg_tpc::jacobi_ceigh");
  auto out = desc.execute(inputs);
  return {out[0], out[1]};
}
std::tuple<at::Tensor, at::Tensor> jacobi_ceigh_refine(at::Tensor h, at::Tensor v1) {
  std::vector<c10::IValue> inputs{h, v1};
  auto desc = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor("linalg_tpc::jacobi_ceigh_refine");
  auto out = desc.execute(inputs);
  return {out[0], out[1]};
}
std::tuple<at::Tensor, at::Tensor, at::Tensor> jacobi_csvd(at::Tensor a) {
  std::vector<c10::IValue> inputs{a};
  auto desc = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor("linalg_tpc::jacobi_csvd");
  auto out = desc.execute(inputs);
  return {out[0], out[1], out[2]};
}
at::Tensor jacobi_csvdvals(at::Tensor a) {
  std::vector<c10::IValue> inputs{a};
  auto desc = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor("linalg_tpc::jacobi_csvdvals");
  return desc.execute(inputs)[0];
}
std::tuple<at::Tensor, at::Tensor, at::Tensor> cgj_inv(at::Tensor a) {
  std::vector<c10::IValue> inputs{a};
  auto desc = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor("linalg_tpc::cgj_inv");
  auto out = desc.execute(inputs);
  return {out[0], out[1], out[2]};
}
using Large = std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>;
Large gj_large(at::Tensor a) {
  std::vector<c10::IValue> inputs{a};
  auto desc = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor("linalg_tpc::gj_large");
  auto out = desc.execute(inputs);
  return {out[0], out[1], out[2], out[3], out[4]};
}
std::tuple<at::Tensor, at::Tensor, at::Tensor> jacobi_svd(at::Tensor a) {
  std::vector<c10::IValue> inputs{a};
  auto desc = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor("linalg_tpc::jacobi_svd");
  auto out = desc.execute(inputs);
  return {out[0], out[1], out[2]};
}
std::tuple<at::Tensor, at::Tensor, at::Tensor> jacobi_svd_full(at::Tensor a) {
  std::vector<c10::IValue> inputs{a};
  auto desc = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor("linalg_tpc::jacobi_svd_full");
  auto out = desc.execute(inputs);
  return {out[0], out[1], out[2]};
}
at::Tensor jacobi_svdvals(at::Tensor a) {
  std::vector<c10::IValue> inputs{a};
  auto desc = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor("linalg_tpc::jacobi_svdvals");
  auto out = desc.execute(inputs);
  return out[0];
}
at::Tensor householder_product(at::Tensor a, at::Tensor tau) {
  std::vector<c10::IValue> inputs{a, tau};
  auto desc = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor("linalg_tpc::householder_product");
  auto out = desc.execute(inputs);
  return out[0];
}
at::Tensor matrix_exp(at::Tensor a) {
  std::vector<c10::IValue> inputs{a};
  auto desc = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor("linalg_tpc::matrix_exp");
  auto out = desc.execute(inputs);
  return out[0];
}
at::Tensor matrix_exp_frechet(at::Tensor a, at::Tensor g) {
  std::vector<c10::IValue> inputs{a, g};
  auto desc = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor("linalg_tpc::matrix_exp_frechet");
  auto out = desc.execute(inputs);
  return out[0];
}
std::tuple<at::Tensor, at::Tensor> lu_factor(at::Tensor a) {
  std::vector<c10::IValue> inputs{a};
  auto desc = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor("linalg_tpc::lu_factor");
  auto out = desc.execute(inputs);
  return {out[0], out[1]};
}
at::Tensor lu_solve(at::Tensor lu, at::Tensor p, at::Tensor b) {
  std::vector<c10::IValue> inputs{lu, p, b};
  auto desc = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor("linalg_tpc::lu_solve");
  auto out = desc.execute(inputs);
  return out[0];
}
at::Tensor lu_perm(at::Tensor p) {
  std::vector<c10::IValue> inputs{p};
  auto desc = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor("linalg_tpc::lu_perm");
  auto out = desc.execute(inputs);
  return out[0];
}
std::tuple<at::Tensor, at::Tensor> jacobi_eigh(at::Tensor a) {
  std::vector<c10::IValue> inputs{a};
  auto desc = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor("linalg_tpc::jacobi_eigh");
  auto out = desc.execute(inputs);
  return {out[0], out[1]};
}
static at::Tensor run_tri_solve(const char* op_name, at::Tensor a, at::Tensor b) {
  std::vector<c10::IValue> inputs{a, b};
  auto desc = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(op_name);
  return desc.execute(inputs)[0];
}
at::Tensor tri_solve_lower(at::Tensor a, at::Tensor b) { return run_tri_solve("linalg_tpc::tri_solve_lower", a, b); }
at::Tensor tri_solve_upper(at::Tensor a, at::Tensor b) { return run_tri_solve("linalg_tpc::tri_solve_upper", a, b); }
at::Tensor tri_solve_lower_unit(at::Tensor a, at::Tensor b) { return run_tri_solve("linalg_tpc::tri_solve_lower_unit", a, b); }
at::Tensor tri_solve_upper_unit(at::Tensor a, at::Tensor b) { return run_tri_solve("linalg_tpc::tri_solve_upper_unit", a, b); }
static at::Tensor tri_meta_tensor(at::Tensor, at::Tensor b) { return b.new_empty(b.sizes()); }
std::tuple<at::Tensor, at::Tensor> jacobi_eigh_refine(at::Tensor a, at::Tensor v1) {
  std::vector<c10::IValue> inputs{a, v1};
  auto desc = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor("linalg_tpc::jacobi_eigh_refine");
  auto out = desc.execute(inputs);
  return {out[0], out[1]};
}
static at::Tensor run_big(const char* name, at::Tensor s0) {
  std::vector<c10::IValue> inputs{s0};
  auto desc = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(name);
  auto out = desc.execute(inputs);
  return out[0];
}
at::Tensor jacobi_big(at::Tensor s0) { return run_big("linalg_tpc::jacobi_big", s0); }
at::Tensor jacobi_big_refine(at::Tensor s0) { return run_big("linalg_tpc::jacobi_big_refine", s0); }
TORCH_LIBRARY(linalg_tpc, m) {
  m.def("gj_large(Tensor a) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");
  m.def("gj_solve(Tensor a, Tensor b) -> (Tensor, Tensor, Tensor)");
  m.def("gj_inv(Tensor a) -> (Tensor, Tensor, Tensor)");
  m.def("cgj_solve(Tensor a, Tensor b) -> (Tensor, Tensor, Tensor)");
  m.def("cgj_inv(Tensor a) -> (Tensor, Tensor, Tensor)");
  m.def("jacobi_ceigh(Tensor a) -> (Tensor, Tensor)");
  m.def("jacobi_ceigh_refine(Tensor h, Tensor v1) -> (Tensor, Tensor)");
  m.def("jacobi_csvd(Tensor a) -> (Tensor, Tensor, Tensor)");
  m.def("jacobi_csvdvals(Tensor a) -> Tensor");
  m.def("jacobi_svd(Tensor a) -> (Tensor, Tensor, Tensor)");
  m.def("jacobi_svd_full(Tensor a) -> (Tensor, Tensor, Tensor)");
  m.def("jacobi_svdvals(Tensor a) -> Tensor");
  m.def("householder_product(Tensor a, Tensor tau) -> Tensor");
  m.def("matrix_exp(Tensor a) -> Tensor");
  m.def("matrix_exp_frechet(Tensor a, Tensor g) -> Tensor");
  m.def("lu_factor(Tensor a) -> (Tensor, Tensor)");
  m.def("lu_perm(Tensor p) -> Tensor");
  m.def("lu_solve(Tensor lu, Tensor p, Tensor b) -> Tensor");
  m.def("jacobi_eigh(Tensor a) -> (Tensor, Tensor)");
  m.def("tri_solve_lower(Tensor a, Tensor b) -> Tensor");
  m.def("tri_solve_upper(Tensor a, Tensor b) -> Tensor");
  m.def("tri_solve_lower_unit(Tensor a, Tensor b) -> Tensor");
  m.def("tri_solve_upper_unit(Tensor a, Tensor b) -> Tensor");
  m.def("jacobi_eigh_refine(Tensor a, Tensor v1) -> (Tensor, Tensor)");
  m.def("jacobi_big(Tensor s0) -> Tensor");
  m.def("jacobi_big_refine(Tensor s0) -> Tensor");
}
TORCH_LIBRARY_IMPL(linalg_tpc, HPU, m) {
  m.impl("gj_large", gj_large);
  m.impl("gj_solve", gj_solve);
  m.impl("gj_inv", gj_inv);
  m.impl("cgj_solve", cgj_solve);
  m.impl("cgj_inv", cgj_inv);
  m.impl("jacobi_ceigh", jacobi_ceigh);
  m.impl("jacobi_ceigh_refine", jacobi_ceigh_refine);
  m.impl("jacobi_csvd", jacobi_csvd);
  m.impl("jacobi_csvdvals", jacobi_csvdvals);
  m.impl("jacobi_svd", jacobi_svd);
  m.impl("jacobi_svd_full", jacobi_svd_full);
  m.impl("jacobi_svdvals", jacobi_svdvals);
  m.impl("householder_product", householder_product);
  m.impl("matrix_exp", matrix_exp);
  m.impl("matrix_exp_frechet", matrix_exp_frechet);
  m.impl("lu_factor", lu_factor);
  m.impl("lu_perm", lu_perm);
  m.impl("lu_solve", lu_solve);
  m.impl("jacobi_eigh", jacobi_eigh);
  m.impl("tri_solve_lower", tri_solve_lower);
  m.impl("tri_solve_upper", tri_solve_upper);
  m.impl("tri_solve_lower_unit", tri_solve_lower_unit);
  m.impl("tri_solve_upper_unit", tri_solve_upper_unit);
  m.impl("jacobi_eigh_refine", jacobi_eigh_refine);
  m.impl("jacobi_big", jacobi_big);
  m.impl("jacobi_big_refine", jacobi_big_refine);
}
TORCH_LIBRARY_IMPL(linalg_tpc, Meta, m) {
  m.impl("gj_large", [](at::Tensor a) {
    auto bt = a.size(0), m = a.size(1), n = a.size(2);
    return std::make_tuple(a.new_empty({bt, m, 64 * ((n + 63) / 64)}), a.new_empty({bt, n}), a.new_empty({bt, m}),
                           a.new_empty({bt, n}), a.new_empty({bt, 2}));
  });
  m.impl("gj_solve", [](at::Tensor a, at::Tensor b) {
    return std::make_tuple(b.new_empty(b.sizes()), b.new_empty({a.size(0), a.size(1)}), b.new_empty({a.size(0), 2}));
  });
  m.impl("gj_inv", [](at::Tensor a) {
    return std::make_tuple(a.new_empty(a.sizes()), a.new_empty({a.size(0), a.size(1)}), a.new_empty({a.size(0), 2}));
  });
  m.impl("cgj_solve", [](at::Tensor a, at::Tensor b) {
    return std::make_tuple(b.new_empty(b.sizes()), b.new_empty({a.size(0), 3, a.size(2)}), b.new_empty({a.size(0), 4}));
  });
  m.impl("jacobi_ceigh", [](at::Tensor a) { return std::make_tuple(a.new_empty(a.sizes()), a.new_empty(a.sizes())); });
  m.impl("jacobi_ceigh_refine", [](at::Tensor h, at::Tensor v1) {
    return std::make_tuple(h.new_empty({h.size(0), h.size(2)}), h.new_empty(h.sizes()));
  });
  m.impl("jacobi_csvd", [](at::Tensor a) {
    auto b = a.size(0), k2 = a.size(1);
    return std::make_tuple(a.new_empty(a.sizes()), a.new_empty({b, k2 / 2}), a.new_empty({b, k2, k2 / 2}));
  });
  m.impl("jacobi_csvdvals", [](at::Tensor a) { return a.new_empty({a.size(0), a.size(1) / 2}); });
  m.impl("cgj_inv", [](at::Tensor a) {
    return std::make_tuple(a.new_empty(a.sizes()), a.new_empty({a.size(0), 3, a.size(2)}), a.new_empty({a.size(0), 4}));
  });
  m.impl("jacobi_svd", [](at::Tensor a) {
    auto b = a.size(0), n = a.size(1), m = a.size(2);
    return std::make_tuple(a.new_empty({b, n, m}), a.new_empty({b, n}), a.new_empty({b, n, n}));
  });
  m.impl("jacobi_svd_full", [](at::Tensor a) {
    auto b = a.size(0), n = a.size(1), m = a.size(2);
    return std::make_tuple(a.new_empty({b, m, m}), a.new_empty({b, n}), a.new_empty({b, n, n}));
  });
  m.impl("jacobi_svdvals", [](at::Tensor a) {
    return a.new_empty({a.size(0), a.size(1)});
  });
  m.impl("householder_product", [](at::Tensor a, at::Tensor tau) {
    return a.new_empty(a.sizes());
  });
  m.impl("matrix_exp", [](at::Tensor a) {
    return a.new_empty(a.sizes());
  });
  m.impl("matrix_exp_frechet", [](at::Tensor a, at::Tensor g) {
    return a.new_empty(a.sizes());
  });
  m.impl("lu_factor", [](at::Tensor a) {
    return std::make_tuple(a.new_empty(a.sizes()), a.new_empty({a.size(0), 3, a.size(1)}));
  });
  m.impl("lu_perm", [](at::Tensor p) {
    return p.new_empty(p.sizes());
  });
  m.impl("lu_solve", [](at::Tensor lu, at::Tensor p, at::Tensor b) {
    return b.new_empty(b.sizes());
  });
  m.impl("jacobi_eigh", [](at::Tensor a) {
    return std::make_tuple(a.new_empty({a.size(0), a.size(1)}), a.new_empty(a.sizes()));
  });
  m.impl("tri_solve_lower", tri_meta_tensor);
  m.impl("tri_solve_upper", tri_meta_tensor);
  m.impl("tri_solve_lower_unit", tri_meta_tensor);
  m.impl("tri_solve_upper_unit", tri_meta_tensor);
  m.impl("jacobi_eigh_refine", [](at::Tensor a, at::Tensor v1) {
    return std::make_tuple(a.new_empty({a.size(0), a.size(1)}), a.new_empty(a.sizes()));
  });
  m.impl("jacobi_big", [](at::Tensor s0) { return s0.new_empty(s0.sizes()); });
  m.impl("jacobi_big_refine", [](at::Tensor s0) { return s0.new_empty(s0.sizes()); });
}
"""
CPP_LAZY = r"""
#include <torch/extension.h>
#include "hpu_custom_op.h"

static auto xshape_b = [](const at::Stack& s) { return s[1].toTensor().sizes().vec(); };
static auto xshape_a = [](const at::Stack& s) { return s[0].toTensor().sizes().vec(); };
static auto dshape = [](const at::Stack& s) {
  auto a = s[0].toTensor();
  return std::vector<int64_t>{a.size(0), a.size(1)};
};
static auto sshape = [](const at::Stack& s) { return std::vector<int64_t>{s[0].toTensor().size(0), 2}; };
static auto vhshape = [](const at::Stack& s) {
  auto a = s[0].toTensor();
  return std::vector<int64_t>{a.size(0), a.size(1), a.size(1)};
};
static auto utfullshape = [](const at::Stack& s) {
  auto a = s[0].toTensor();
  return std::vector<int64_t>{a.size(0), a.size(2), a.size(2)};
};

static bool registered_solve = [] {
  std::vector<habana::custom_op::InputDesc> in{{habana::custom_op::input_type::TENSOR, 0},
                                               {habana::custom_op::input_type::TENSOR, 1}};
  std::vector<habana::custom_op::OutputDesc> out{{0, c10::ScalarType::Float, xshape_b},
                                                 {1, c10::ScalarType::Float, dshape},
                                                 {2, c10::ScalarType::Float, sshape}};
  REGISTER_CUSTOM_OP_ATTRIBUTES("linalg_tpc::gj_solve", "gj_solve_f32_gaudi2", in, out, nullptr);
  return true;
}();
static auto rshape = [](const at::Stack& s) {
  auto a = s[0].toTensor();
  auto n = a.size(2);
  return std::vector<int64_t>{a.size(0), a.size(1), 64 * ((n + 63) / 64)};
};
static auto cshape = [](const at::Stack& s) {  // one per column of a (tall) panel
  auto a = s[0].toTensor();
  return std::vector<int64_t>{a.size(0), a.size(2)};
};
static bool registered_large = [] {
  std::vector<habana::custom_op::InputDesc> in{{habana::custom_op::input_type::TENSOR, 0}};
  std::vector<habana::custom_op::OutputDesc> out{{0, c10::ScalarType::Float, rshape},
                                                 {1, c10::ScalarType::Float, cshape},
                                                 {2, c10::ScalarType::Float, dshape},
                                                 {3, c10::ScalarType::Float, cshape},
                                                 {4, c10::ScalarType::Float, sshape}};
  REGISTER_CUSTOM_OP_ATTRIBUTES("linalg_tpc::gj_large", "gj_large_f32_gaudi2", in, out, nullptr);
  return true;
}();
static auto cdshape = [](const at::Stack& s) {
  auto a = s[0].toTensor();
  return std::vector<int64_t>{a.size(0), 3, a.size(2)};
};
static auto csshape = [](const at::Stack& s) { return std::vector<int64_t>{s[0].toTensor().size(0), 4}; };
static bool registered_complex = [] {
  std::vector<habana::custom_op::InputDesc> in2{{habana::custom_op::input_type::TENSOR, 0},
                                                {habana::custom_op::input_type::TENSOR, 1}};
  std::vector<habana::custom_op::OutputDesc> out2{{0, c10::ScalarType::Float, xshape_b},
                                                  {1, c10::ScalarType::Float, cdshape},
                                                  {2, c10::ScalarType::Float, csshape}};
  REGISTER_CUSTOM_OP_ATTRIBUTES("linalg_tpc::cgj_solve", "cgj_solve_f32_gaudi2", in2, out2, nullptr);
  std::vector<habana::custom_op::InputDesc> in1{{habana::custom_op::input_type::TENSOR, 0}};
  std::vector<habana::custom_op::OutputDesc> out1{{0, c10::ScalarType::Float, xshape_a},
                                                  {1, c10::ScalarType::Float, cdshape},
                                                  {2, c10::ScalarType::Float, csshape}};
  REGISTER_CUSTOM_OP_ATTRIBUTES("linalg_tpc::cgj_inv", "cgj_inv_f32_gaudi2", in1, out1, nullptr);
  return true;
}();
static auto cwshape = [](const at::Stack& s) {
  auto a = s[0].toTensor();
  return std::vector<int64_t>{a.size(0), a.size(2)};
};
static auto csshape_svd = [](const at::Stack& s) {
  auto a = s[0].toTensor();
  return std::vector<int64_t>{a.size(0), a.size(1) / 2};
};
static auto cjshape = [](const at::Stack& s) {
  auto a = s[0].toTensor();
  return std::vector<int64_t>{a.size(0), a.size(1), a.size(1) / 2};
};
static bool registered_cjacobi = [] {
  std::vector<habana::custom_op::InputDesc> in1{{habana::custom_op::input_type::TENSOR, 0}};
  std::vector<habana::custom_op::InputDesc> in2{{habana::custom_op::input_type::TENSOR, 0},
                                                {habana::custom_op::input_type::TENSOR, 1}};
  std::vector<habana::custom_op::OutputDesc> o_eigh{{0, c10::ScalarType::Float, xshape_a},
                                                    {1, c10::ScalarType::Float, xshape_a}};
  REGISTER_CUSTOM_OP_ATTRIBUTES("linalg_tpc::jacobi_ceigh", "jacobi_ceigh_f32_gaudi2", in1, o_eigh, nullptr);
  std::vector<habana::custom_op::OutputDesc> o_ref{{0, c10::ScalarType::Float, cwshape},
                                                   {1, c10::ScalarType::Float, xshape_a}};
  REGISTER_CUSTOM_OP_ATTRIBUTES("linalg_tpc::jacobi_ceigh_refine", "jacobi_ceigh_refine_f32_gaudi2", in2, o_ref, nullptr);
  std::vector<habana::custom_op::OutputDesc> o_svd{{0, c10::ScalarType::Float, xshape_a},
                                                   {1, c10::ScalarType::Float, csshape_svd},
                                                   {2, c10::ScalarType::Float, cjshape}};
  REGISTER_CUSTOM_OP_ATTRIBUTES("linalg_tpc::jacobi_csvd", "jacobi_csvd_f32_gaudi2", in1, o_svd, nullptr);
  std::vector<habana::custom_op::OutputDesc> o_vals{{0, c10::ScalarType::Float, csshape_svd}};
  REGISTER_CUSTOM_OP_ATTRIBUTES("linalg_tpc::jacobi_csvdvals", "jacobi_csvdvals_f32_gaudi2", in1, o_vals, nullptr);
  return true;
}();
static bool registered_inv = [] {
  std::vector<habana::custom_op::InputDesc> in{{habana::custom_op::input_type::TENSOR, 0}};
  std::vector<habana::custom_op::OutputDesc> out{{0, c10::ScalarType::Float, xshape_a},
                                                 {1, c10::ScalarType::Float, dshape},
                                                 {2, c10::ScalarType::Float, sshape}};
  REGISTER_CUSTOM_OP_ATTRIBUTES("linalg_tpc::gj_inv", "gj_inv_f32_gaudi2", in, out, nullptr);
  return true;
}();
static bool registered_svd = [] {
  std::vector<habana::custom_op::InputDesc> in{{habana::custom_op::input_type::TENSOR, 0}};
  std::vector<habana::custom_op::OutputDesc> out{{0, c10::ScalarType::Float, xshape_a},
                                                 {1, c10::ScalarType::Float, dshape},
                                                 {2, c10::ScalarType::Float, vhshape}};
  REGISTER_CUSTOM_OP_ATTRIBUTES("linalg_tpc::jacobi_svd", "jacobi_svd_f32_gaudi2", in, out, nullptr);
  return true;
}();
static bool registered_svd_full = [] {
  std::vector<habana::custom_op::InputDesc> in{{habana::custom_op::input_type::TENSOR, 0}};
  std::vector<habana::custom_op::OutputDesc> out{{0, c10::ScalarType::Float, utfullshape},
                                                 {1, c10::ScalarType::Float, dshape},
                                                 {2, c10::ScalarType::Float, vhshape}};
  REGISTER_CUSTOM_OP_ATTRIBUTES("linalg_tpc::jacobi_svd_full", "jacobi_svd_full_f32_gaudi2", in, out, nullptr);
  return true;
}();
static bool registered_svdvals = [] {
  std::vector<habana::custom_op::InputDesc> in{{habana::custom_op::input_type::TENSOR, 0}};
  std::vector<habana::custom_op::OutputDesc> out{{0, c10::ScalarType::Float, dshape}};
  REGISTER_CUSTOM_OP_ATTRIBUTES("linalg_tpc::jacobi_svdvals", "jacobi_svdvals_f32_gaudi2", in, out, nullptr);
  return true;
}();
static bool registered_hh = [] {
  std::vector<habana::custom_op::InputDesc> in{{habana::custom_op::input_type::TENSOR, 0},
                                               {habana::custom_op::input_type::TENSOR, 1}};
  std::vector<habana::custom_op::OutputDesc> out{{0, c10::ScalarType::Float, xshape_a}};
  REGISTER_CUSTOM_OP_ATTRIBUTES("linalg_tpc::householder_product", "householder_product_f32_gaudi2", in, out, nullptr);
  return true;
}();
static bool registered_expm = [] {
  std::vector<habana::custom_op::InputDesc> in{{habana::custom_op::input_type::TENSOR, 0}};
  std::vector<habana::custom_op::OutputDesc> out{{0, c10::ScalarType::Float, xshape_a}};
  REGISTER_CUSTOM_OP_ATTRIBUTES("linalg_tpc::matrix_exp", "matrix_exp_f32_gaudi2", in, out, nullptr);
  return true;
}();
static bool registered_expm_frechet = [] {
  std::vector<habana::custom_op::InputDesc> in{{habana::custom_op::input_type::TENSOR, 0},
                                               {habana::custom_op::input_type::TENSOR, 1}};
  std::vector<habana::custom_op::OutputDesc> out{{0, c10::ScalarType::Float, xshape_a}};
  REGISTER_CUSTOM_OP_ATTRIBUTES("linalg_tpc::matrix_exp_frechet", "matrix_exp_frechet_f32_gaudi2", in, out, nullptr);
  return true;
}();

static auto pvshape = [](const at::Stack& s) {
  auto a = s[0].toTensor();
  return std::vector<int64_t>{a.size(0), 3, a.size(1)};
};
static bool registered_lu = [] {
  std::vector<habana::custom_op::InputDesc> in{{habana::custom_op::input_type::TENSOR, 0}};
  std::vector<habana::custom_op::OutputDesc> out{{0, c10::ScalarType::Float, xshape_a},
                                                 {1, c10::ScalarType::Float, pvshape}};
  REGISTER_CUSTOM_OP_ATTRIBUTES("linalg_tpc::lu_factor", "lu_factor_f32_gaudi2", in, out, nullptr);
  return true;
}();
static bool registered_lu_perm = [] {
  std::vector<habana::custom_op::InputDesc> in{{habana::custom_op::input_type::TENSOR, 0}};
  std::vector<habana::custom_op::OutputDesc> out{{0, c10::ScalarType::Float, xshape_a}};
  REGISTER_CUSTOM_OP_ATTRIBUTES("linalg_tpc::lu_perm", "lu_perm_f32_gaudi2", in, out, nullptr);
  return true;
}();
static auto xshape_c = [](const at::Stack& s) { return s[2].toTensor().sizes().vec(); };
static bool registered_lu_solve = [] {
  std::vector<habana::custom_op::InputDesc> in{{habana::custom_op::input_type::TENSOR, 0},
                                               {habana::custom_op::input_type::TENSOR, 1},
                                               {habana::custom_op::input_type::TENSOR, 2}};
  std::vector<habana::custom_op::OutputDesc> out{{0, c10::ScalarType::Float, xshape_c}};
  REGISTER_CUSTOM_OP_ATTRIBUTES("linalg_tpc::lu_solve", "lu_solve_f32_gaudi2", in, out, nullptr);
  return true;
}();

static auto wshape = [](const at::Stack& s) {
  auto a = s[0].toTensor();
  return std::vector<int64_t>{a.size(0), a.size(1)};
};
static bool registered_eigh = [] {
  std::vector<habana::custom_op::InputDesc> in{{habana::custom_op::input_type::TENSOR, 0}};
  std::vector<habana::custom_op::OutputDesc> out{{0, c10::ScalarType::Float, wshape},
                                                 {1, c10::ScalarType::Float, xshape_a}};
  REGISTER_CUSTOM_OP_ATTRIBUTES("linalg_tpc::jacobi_eigh", "jacobi_eigh_f32_gaudi2", in, out, nullptr);
  return true;
}();
static bool registered_tri = [] {
  std::vector<habana::custom_op::InputDesc> in{{habana::custom_op::input_type::TENSOR, 0},
                                               {habana::custom_op::input_type::TENSOR, 1}};
  std::vector<habana::custom_op::OutputDesc> out{{0, c10::ScalarType::Float, xshape_b}};
  REGISTER_CUSTOM_OP_ATTRIBUTES("linalg_tpc::tri_solve_lower", "tri_solve_lower_f32_gaudi2", in, out, nullptr);
  REGISTER_CUSTOM_OP_ATTRIBUTES("linalg_tpc::tri_solve_upper", "tri_solve_upper_f32_gaudi2", in, out, nullptr);
  REGISTER_CUSTOM_OP_ATTRIBUTES("linalg_tpc::tri_solve_lower_unit", "tri_solve_lower_unit_f32_gaudi2", in, out, nullptr);
  REGISTER_CUSTOM_OP_ATTRIBUTES("linalg_tpc::tri_solve_upper_unit", "tri_solve_upper_unit_f32_gaudi2", in, out, nullptr);
  return true;
}();
static bool registered_big = [] {
  std::vector<habana::custom_op::InputDesc> in{{habana::custom_op::input_type::TENSOR, 0}};
  std::vector<habana::custom_op::OutputDesc> out{{0, c10::ScalarType::Float, xshape_a}};
  REGISTER_CUSTOM_OP_ATTRIBUTES("linalg_tpc::jacobi_big", "jacobi_big_f32_gaudi2", in, out, nullptr);
  REGISTER_CUSTOM_OP_ATTRIBUTES("linalg_tpc::jacobi_big_refine", "jacobi_big_refine_f32_gaudi2", in, out, nullptr);
  return true;
}();
static bool registered_eigh_refine = [] {
  std::vector<habana::custom_op::InputDesc> in{{habana::custom_op::input_type::TENSOR, 0},
                                               {habana::custom_op::input_type::TENSOR, 1}};
  std::vector<habana::custom_op::OutputDesc> out{{0, c10::ScalarType::Float, wshape},
                                                 {1, c10::ScalarType::Float, xshape_a}};
  REGISTER_CUSTOM_OP_ATTRIBUTES("linalg_tpc::jacobi_eigh_refine", "jacobi_eigh_refine_f32_gaudi2", in, out, nullptr);
  return true;
}();

std::tuple<at::Tensor, at::Tensor, at::Tensor> gj_solve(at::Tensor a, at::Tensor b) {
  auto op = habana::custom_op::HabanaCustomOpDescriptor::getCustomOpDescriptor("linalg_tpc::gj_solve");
  std::vector<c10::IValue> inputs{a, b};
  auto out = op.execute(inputs);
  return {out[0], out[1], out[2]};
}
std::tuple<at::Tensor, at::Tensor, at::Tensor> gj_inv(at::Tensor a) {
  auto op = habana::custom_op::HabanaCustomOpDescriptor::getCustomOpDescriptor("linalg_tpc::gj_inv");
  std::vector<c10::IValue> inputs{a};
  auto out = op.execute(inputs);
  return {out[0], out[1], out[2]};
}
std::tuple<at::Tensor, at::Tensor, at::Tensor> cgj_solve(at::Tensor a, at::Tensor b) {
  auto op = habana::custom_op::HabanaCustomOpDescriptor::getCustomOpDescriptor("linalg_tpc::cgj_solve");
  std::vector<c10::IValue> inputs{a, b};
  auto out = op.execute(inputs);
  return {out[0], out[1], out[2]};
}
std::tuple<at::Tensor, at::Tensor> jacobi_ceigh(at::Tensor a) {
  std::vector<c10::IValue> inputs{a};
  auto desc = habana::custom_op::HabanaCustomOpDescriptor::getCustomOpDescriptor("linalg_tpc::jacobi_ceigh");
  auto out = desc.execute(inputs);
  return {out[0], out[1]};
}
std::tuple<at::Tensor, at::Tensor> jacobi_ceigh_refine(at::Tensor h, at::Tensor v1) {
  std::vector<c10::IValue> inputs{h, v1};
  auto desc = habana::custom_op::HabanaCustomOpDescriptor::getCustomOpDescriptor("linalg_tpc::jacobi_ceigh_refine");
  auto out = desc.execute(inputs);
  return {out[0], out[1]};
}
std::tuple<at::Tensor, at::Tensor, at::Tensor> jacobi_csvd(at::Tensor a) {
  std::vector<c10::IValue> inputs{a};
  auto desc = habana::custom_op::HabanaCustomOpDescriptor::getCustomOpDescriptor("linalg_tpc::jacobi_csvd");
  auto out = desc.execute(inputs);
  return {out[0], out[1], out[2]};
}
at::Tensor jacobi_csvdvals(at::Tensor a) {
  std::vector<c10::IValue> inputs{a};
  auto desc = habana::custom_op::HabanaCustomOpDescriptor::getCustomOpDescriptor("linalg_tpc::jacobi_csvdvals");
  return desc.execute(inputs)[0];
}
std::tuple<at::Tensor, at::Tensor, at::Tensor> cgj_inv(at::Tensor a) {
  auto op = habana::custom_op::HabanaCustomOpDescriptor::getCustomOpDescriptor("linalg_tpc::cgj_inv");
  std::vector<c10::IValue> inputs{a};
  auto out = op.execute(inputs);
  return {out[0], out[1], out[2]};
}
using Large = std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>;
Large gj_large(at::Tensor a) {
  auto op = habana::custom_op::HabanaCustomOpDescriptor::getCustomOpDescriptor("linalg_tpc::gj_large");
  std::vector<c10::IValue> inputs{a};
  auto out = op.execute(inputs);
  return {out[0], out[1], out[2], out[3], out[4]};
}
std::tuple<at::Tensor, at::Tensor, at::Tensor> jacobi_svd(at::Tensor a) {
  auto op = habana::custom_op::HabanaCustomOpDescriptor::getCustomOpDescriptor("linalg_tpc::jacobi_svd");
  std::vector<c10::IValue> inputs{a};
  auto out = op.execute(inputs);
  return {out[0], out[1], out[2]};
}
std::tuple<at::Tensor, at::Tensor, at::Tensor> jacobi_svd_full(at::Tensor a) {
  auto op = habana::custom_op::HabanaCustomOpDescriptor::getCustomOpDescriptor("linalg_tpc::jacobi_svd_full");
  std::vector<c10::IValue> inputs{a};
  auto out = op.execute(inputs);
  return {out[0], out[1], out[2]};
}
at::Tensor jacobi_svdvals(at::Tensor a) {
  auto op = habana::custom_op::HabanaCustomOpDescriptor::getCustomOpDescriptor("linalg_tpc::jacobi_svdvals");
  std::vector<c10::IValue> inputs{a};
  auto out = op.execute(inputs);
  return out[0];
}
at::Tensor householder_product(at::Tensor a, at::Tensor tau) {
  auto op = habana::custom_op::HabanaCustomOpDescriptor::getCustomOpDescriptor("linalg_tpc::householder_product");
  std::vector<c10::IValue> inputs{a, tau};
  auto out = op.execute(inputs);
  return out[0];
}
at::Tensor matrix_exp(at::Tensor a) {
  auto op = habana::custom_op::HabanaCustomOpDescriptor::getCustomOpDescriptor("linalg_tpc::matrix_exp");
  std::vector<c10::IValue> inputs{a};
  auto out = op.execute(inputs);
  return out[0];
}
at::Tensor matrix_exp_frechet(at::Tensor a, at::Tensor g) {
  auto op = habana::custom_op::HabanaCustomOpDescriptor::getCustomOpDescriptor("linalg_tpc::matrix_exp_frechet");
  std::vector<c10::IValue> inputs{a, g};
  auto out = op.execute(inputs);
  return out[0];
}
std::tuple<at::Tensor, at::Tensor> lu_factor(at::Tensor a) {
  auto op = habana::custom_op::HabanaCustomOpDescriptor::getCustomOpDescriptor("linalg_tpc::lu_factor");
  std::vector<c10::IValue> inputs{a};
  auto out = op.execute(inputs);
  return {out[0], out[1]};
}
at::Tensor lu_solve(at::Tensor lu, at::Tensor p, at::Tensor b) {
  auto op = habana::custom_op::HabanaCustomOpDescriptor::getCustomOpDescriptor("linalg_tpc::lu_solve");
  std::vector<c10::IValue> inputs{lu, p, b};
  auto out = op.execute(inputs);
  return out[0];
}
at::Tensor lu_perm(at::Tensor p) {
  auto op = habana::custom_op::HabanaCustomOpDescriptor::getCustomOpDescriptor("linalg_tpc::lu_perm");
  std::vector<c10::IValue> inputs{p};
  auto out = op.execute(inputs);
  return out[0];
}
std::tuple<at::Tensor, at::Tensor> jacobi_eigh(at::Tensor a) {
  auto op = habana::custom_op::HabanaCustomOpDescriptor::getCustomOpDescriptor("linalg_tpc::jacobi_eigh");
  std::vector<c10::IValue> inputs{a};
  auto out = op.execute(inputs);
  return {out[0], out[1]};
}
static at::Tensor run_tri_solve(const char* op_name, at::Tensor a, at::Tensor b) {
  auto op = habana::custom_op::HabanaCustomOpDescriptor::getCustomOpDescriptor(op_name);
  std::vector<c10::IValue> inputs{a, b};
  return op.execute(inputs)[0];
}
at::Tensor tri_solve_lower(at::Tensor a, at::Tensor b) { return run_tri_solve("linalg_tpc::tri_solve_lower", a, b); }
at::Tensor tri_solve_upper(at::Tensor a, at::Tensor b) { return run_tri_solve("linalg_tpc::tri_solve_upper", a, b); }
at::Tensor tri_solve_lower_unit(at::Tensor a, at::Tensor b) { return run_tri_solve("linalg_tpc::tri_solve_lower_unit", a, b); }
at::Tensor tri_solve_upper_unit(at::Tensor a, at::Tensor b) { return run_tri_solve("linalg_tpc::tri_solve_upper_unit", a, b); }
std::tuple<at::Tensor, at::Tensor> jacobi_eigh_refine(at::Tensor a, at::Tensor v1) {
  auto op = habana::custom_op::HabanaCustomOpDescriptor::getCustomOpDescriptor("linalg_tpc::jacobi_eigh_refine");
  std::vector<c10::IValue> inputs{a, v1};
  auto out = op.execute(inputs);
  return {out[0], out[1]};
}
static at::Tensor run_big(const char* name, at::Tensor s0) {
  auto op = habana::custom_op::HabanaCustomOpDescriptor::getCustomOpDescriptor(name);
  std::vector<c10::IValue> inputs{s0};
  auto out = op.execute(inputs);
  return out[0];
}
at::Tensor jacobi_big(at::Tensor s0) { return run_big("linalg_tpc::jacobi_big", s0); }
at::Tensor jacobi_big_refine(at::Tensor s0) { return run_big("linalg_tpc::jacobi_big_refine", s0); }
TORCH_LIBRARY(linalg_tpc, m) {
  m.def("gj_large(Tensor a) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");
  m.def("gj_solve(Tensor a, Tensor b) -> (Tensor, Tensor, Tensor)");
  m.def("gj_inv(Tensor a) -> (Tensor, Tensor, Tensor)");
  m.def("cgj_solve(Tensor a, Tensor b) -> (Tensor, Tensor, Tensor)");
  m.def("cgj_inv(Tensor a) -> (Tensor, Tensor, Tensor)");
  m.def("jacobi_ceigh(Tensor a) -> (Tensor, Tensor)");
  m.def("jacobi_ceigh_refine(Tensor h, Tensor v1) -> (Tensor, Tensor)");
  m.def("jacobi_csvd(Tensor a) -> (Tensor, Tensor, Tensor)");
  m.def("jacobi_csvdvals(Tensor a) -> Tensor");
  m.def("jacobi_svd(Tensor a) -> (Tensor, Tensor, Tensor)");
  m.def("jacobi_svd_full(Tensor a) -> (Tensor, Tensor, Tensor)");
  m.def("jacobi_svdvals(Tensor a) -> Tensor");
  m.def("householder_product(Tensor a, Tensor tau) -> Tensor");
  m.def("matrix_exp(Tensor a) -> Tensor");
  m.def("matrix_exp_frechet(Tensor a, Tensor g) -> Tensor");
  m.def("lu_factor(Tensor a) -> (Tensor, Tensor)");
  m.def("lu_perm(Tensor p) -> Tensor");
  m.def("lu_solve(Tensor lu, Tensor p, Tensor b) -> Tensor");
  m.def("jacobi_eigh(Tensor a) -> (Tensor, Tensor)");
  m.def("tri_solve_lower(Tensor a, Tensor b) -> Tensor");
  m.def("tri_solve_upper(Tensor a, Tensor b) -> Tensor");
  m.def("tri_solve_lower_unit(Tensor a, Tensor b) -> Tensor");
  m.def("tri_solve_upper_unit(Tensor a, Tensor b) -> Tensor");
  m.def("jacobi_eigh_refine(Tensor a, Tensor v1) -> (Tensor, Tensor)");
  m.def("jacobi_big(Tensor s0) -> Tensor");
  m.def("jacobi_big_refine(Tensor s0) -> Tensor");
}
TORCH_LIBRARY_IMPL(linalg_tpc, HPU, m) {
  m.impl("gj_large", gj_large);
  m.impl("gj_solve", gj_solve);
  m.impl("gj_inv", gj_inv);
  m.impl("cgj_solve", cgj_solve);
  m.impl("cgj_inv", cgj_inv);
  m.impl("jacobi_ceigh", jacobi_ceigh);
  m.impl("jacobi_ceigh_refine", jacobi_ceigh_refine);
  m.impl("jacobi_csvd", jacobi_csvd);
  m.impl("jacobi_csvdvals", jacobi_csvdvals);
  m.impl("jacobi_svd", jacobi_svd);
  m.impl("jacobi_svd_full", jacobi_svd_full);
  m.impl("jacobi_svdvals", jacobi_svdvals);
  m.impl("householder_product", householder_product);
  m.impl("matrix_exp", matrix_exp);
  m.impl("matrix_exp_frechet", matrix_exp_frechet);
  m.impl("lu_factor", lu_factor);
  m.impl("lu_perm", lu_perm);
  m.impl("lu_solve", lu_solve);
  m.impl("jacobi_eigh", jacobi_eigh);
  m.impl("tri_solve_lower", tri_solve_lower);
  m.impl("tri_solve_upper", tri_solve_upper);
  m.impl("tri_solve_lower_unit", tri_solve_lower_unit);
  m.impl("tri_solve_upper_unit", tri_solve_upper_unit);
  m.impl("jacobi_eigh_refine", jacobi_eigh_refine);
  m.impl("jacobi_big", jacobi_big);
  m.impl("jacobi_big_refine", jacobi_big_refine);
}
"""


def _initialized():
    """Whether the HPU has been used in this process, by ``torch.hpu`` or, failing that, Habana's own module. Without
    either name the answer is False: the kernel path is then set, as it was in the measurement."""
    for module in (getattr(torch, "hpu", None), _habana_hpu()):
        check = getattr(module, "is_initialized", None)
        if check is not None:
            return bool(check())
    return False


def _habana_hpu():
    try:
        import habana_frameworks.torch.hpu as hthpu
    except ImportError:
        return None
    return hthpu


def install_kernel_path():
    """Put ``LIB`` first on ``GC_KERNEL_PATH`` (default ``DEFAULT_KERNELS`` when unset or empty, never duplicated) if
    the HPU is not initialized yet and return True; otherwise warn once and return False, as Habana reads the path
    when the device initialises: set later, the kernel's graph fails to compile. Also False, silently, with
    ``HPU_OPS_TPC=0`` in the environment, and, with one warning, when ``LIB`` does not exist."""
    global _INSTALLED, _PATH_SET, _LATE_WARNED, _MISSING_WARNED
    if os.environ.get(OPT_OUT) == "0":
        _INSTALLED = False
        return False
    if not _lib_exists(LIB):
        _INSTALLED = False
        if not _MISSING_WARNED:
            _MISSING_WARNED = True
            warnings.warn(MISSING_WARNING.format(path=LIB), stacklevel=2)
        return False
    if _initialized():
        # enable() again after its undo: the path set by an earlier successful call is still the one Habana read
        if _PATH_SET and str(LIB) in (os.environ.get("GC_KERNEL_PATH") or "").split(":"):
            _INSTALLED = True
            return True
        _INSTALLED = False
        if not _LATE_WARNED:
            _LATE_WARNED = True
            warnings.warn(LATE_WARNING, stacklevel=2)
        return False
    entries = [e for e in (os.environ.get("GC_KERNEL_PATH") or DEFAULT_KERNELS).split(":") if e]
    if str(LIB) not in entries:
        os.environ["GC_KERNEL_PATH"] = ":".join([str(LIB), *entries])
    _INSTALLED = _PATH_SET = True
    return True


@functools.cache
def _lib_exists(path):
    return path.exists()


def available():
    """True when ``install_kernel_path`` succeeded in this process and the library is there (the file check is cached
    per path)."""
    return _INSTALLED and _lib_exists(LIB)


def _device_name():
    """The name of the HPU device (cached once read): ``habana_frameworks.torch.hpu.get_device_name()``, else
    ``torch.hpu.get_device_name()``; "" if neither answers."""
    global _DEVICE
    if _DEVICE is None:
        for module in (_habana_hpu(), getattr(torch, "hpu", None)):
            try:
                _DEVICE = str(module.get_device_name())
                break
            except Exception:  # noqa: BLE001  (no such module or name: try the next, else decline)
                continue
        else:
            return ""
    return _DEVICE


def _gaudi2():
    """The kernels are compiled for Gaudi2 only: on another generation they would fail or run wrong."""
    return "GAUDI2" in _device_name().upper()


def ready():
    """``available()``, a Gaudi2 device and the C++ extension built. The build happens on the first call; if it
    fails, one warning, then False for the rest of the process (the ops run through hpu-ops' own CPU route, with no
    retry mid-forward)."""
    return available() and _gaudi2() and _built()


@functools.cache
def _built():
    global _INSTALLED
    try:
        _load()
    except Exception as error:  # noqa: BLE001  (a missing toolchain or header: decline, do not crash the model)
        _INSTALLED = False
        warnings.warn(BUILD_WARNING.format(error=str(error)[:300]), stacklevel=2)
        return False
    return True


def lazy_mode():
    """Whether the bridge runs in lazy mode: Habana's own ``is_lazy()`` once ``habana_frameworks.torch`` is imported,
    else ``PT_HPU_LAZY_MODE == "1"``. Bridge 1.21.4 is eager when the variable is unset (measured in the pod)."""
    import sys

    if "habana_frameworks.torch" in sys.modules:
        try:
            from habana_frameworks.torch.utils.internal import is_lazy
        except ImportError:
            pass
        else:
            return bool(is_lazy())
    return os.environ.get("PT_HPU_LAZY_MODE", "0") == "1"


# The build of the C++ ops (``_load``) waits at most this many seconds (environment variable, default 600) for another
# process's build; past it ``ready()`` warns once and is False. torch's own lock (``FileBaton``, a ``lock`` file in the
# build directory) is a file that a killed builder leaves behind, and every later process then waited for it forever
# (code review #4: a job preempted during its first linalg op hung the next one at its first linalg op).
BUILD_TIMEOUT = "HPU_OPS_BUILD_TIMEOUT"
# hpu-ops' own build lock: an ``flock`` (released by the kernel when its holder dies, never stale) on this file in the
# build directory, which also records the holder (host and pid).
BUILD_LOCK = "hpu_ops.flock"
# A torch ``lock`` file found while holding ``BUILD_LOCK`` is removed when the builder recorded there ran on this host
# and is gone, or when it is older than this (s): a process without ``BUILD_LOCK`` (an older hpu-ops) may hold it.
STALE_LOCK_S = 120.0
STALE_WARNING = "hpu_ops: removed the stale build lock {path} ({age:.0f} s old, {why})."


def _alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # another user's process
        return True
    return True


@contextlib.contextmanager
def _exclusive(path, deadline):
    """An exclusive ``flock`` on ``path`` (created if missing), waited for until ``deadline`` (``time.monotonic``),
    else ``TimeoutError``. Yields the previous holder's record ``(host, pid)`` (None if unreadable) and writes this
    process's."""
    import fcntl
    import socket

    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() > deadline:
                    raise TimeoutError(f"another process holds {path}") from None
                time.sleep(0.2)
        try:
            host, pid = os.pread(fd, 256, 0).decode().split()
            previous = (host, int(pid))
        except (ValueError, UnicodeDecodeError):
            previous = None
        record = f"{socket.gethostname()} {os.getpid()}\n".encode()
        os.ftruncate(fd, 0)
        os.pwrite(fd, record, 0)
        yield previous
    finally:
        os.close(fd)  # releases the lock


def _clear_baton(path, deadline, previous):
    """Wait for torch's ``FileBaton`` file ``path`` to go, holding ``BUILD_LOCK``. Every hpu-ops builder holds that
    lock while it builds, so the file is left by a dead builder or held by a process without the lock (an older
    hpu-ops). It is removed, with one warning, when ``previous`` (the last holder of ``BUILD_LOCK``) ran on this host
    and is gone, or when it is older than ``STALE_LOCK_S``; past ``deadline``, ``TimeoutError``."""
    import socket

    dead = previous is not None and previous[0] == socket.gethostname() and not _alive(previous[1])
    while True:
        try:
            age = time.time() - os.stat(path).st_mtime
        except FileNotFoundError:
            return
        if dead or age > STALE_LOCK_S:
            try:
                os.remove(path)
            except FileNotFoundError:
                return
            why = f"its builder, pid {previous[1]}, is gone" if dead else "no builder holds hpu-ops' lock"
            warnings.warn(STALE_WARNING.format(path=path, age=age, why=why), stacklevel=3)
            return
        if time.monotonic() > deadline:
            raise TimeoutError(f"{path} is held by another process ({age:.0f} s)")
        time.sleep(0.2)


@functools.cache
def _load():
    """Compile and load the C++ ops: the legacy API of lazy mode when ``lazy_mode()``, the eager and compile API
    otherwise. The build waits at most ``HPU_OPS_BUILD_TIMEOUT`` seconds for another builder (``TimeoutError``,
    which ``ready()`` turns into one warning and False) and clears a stale torch lock (``_clear_baton``)."""
    from habana_frameworks.torch.utils.lib_utils import get_include_dir, get_lib_dir
    from torch.utils.cpp_extension import _get_build_directory, load_inline

    lazy = lazy_mode()
    name = "linalg_tpc_plugin_lazy" if lazy else "linalg_tpc_plugin"
    build_dir = _get_build_directory(name, verbose=False)
    deadline = time.monotonic() + float(os.environ.get(BUILD_TIMEOUT) or 600)
    with _exclusive(os.path.join(build_dir, BUILD_LOCK), deadline) as previous:
        _clear_baton(os.path.join(build_dir, "lock"), deadline, previous)
        load_inline(name=name, cpp_sources=[CPP_LAZY if lazy else CPP], verbose=False, is_python_module=False,
                    build_directory=build_dir, extra_include_paths=[get_include_dir(), "/usr/include/habanalabs"],
                    extra_ldflags=[f"-L{get_lib_dir()}",
                                   "-lhabana_pytorch_plugin" if lazy else "-lhabana_pytorch2_plugin"])


def _op():
    """``torch.ops.linalg_tpc.gj_solve``, built on first use."""
    _load()
    return torch.ops.linalg_tpc.gj_solve


def _inv_op():
    """``torch.ops.linalg_tpc.gj_inv``, built on first use."""
    _load()
    return torch.ops.linalg_tpc.gj_inv


def _large_op():
    """``torch.ops.linalg_tpc.gj_large``, built on first use."""
    _load()
    return torch.ops.linalg_tpc.gj_large


def _svd_op():
    """``torch.ops.linalg_tpc.jacobi_svd``, built on first use."""
    _load()
    return torch.ops.linalg_tpc.jacobi_svd


def _svd_full_op():
    """``torch.ops.linalg_tpc.jacobi_svd_full``, built on first use."""
    _load()
    return torch.ops.linalg_tpc.jacobi_svd_full


def _svdvals_op():
    """``torch.ops.linalg_tpc.jacobi_svdvals``, built on first use."""
    _load()
    return torch.ops.linalg_tpc.jacobi_svdvals


def _hh_op():
    """``torch.ops.linalg_tpc.householder_product``, built on first use."""
    _load()
    return torch.ops.linalg_tpc.householder_product


def _expm_op():
    """``torch.ops.linalg_tpc.matrix_exp``, built on first use."""
    _load()
    return torch.ops.linalg_tpc.matrix_exp


def _expm_frechet_op():
    """``torch.ops.linalg_tpc.matrix_exp_frechet``, built on first use."""
    _load()
    return torch.ops.linalg_tpc.matrix_exp_frechet


def _lu_op():
    """``torch.ops.linalg_tpc.lu_factor``, built on first use."""
    _load()
    return torch.ops.linalg_tpc.lu_factor


def _lu_perm_op():
    """``torch.ops.linalg_tpc.lu_perm``, built on first use."""
    _load()
    return torch.ops.linalg_tpc.lu_perm


def _lu_solve_op():
    """``torch.ops.linalg_tpc.lu_solve``, built on first use."""
    _load()
    return torch.ops.linalg_tpc.lu_solve


def _eigh_op():
    """``torch.ops.linalg_tpc.jacobi_eigh``, built on first use."""
    _load()
    return torch.ops.linalg_tpc.jacobi_eigh


def _big_op(refine=False):
    """``torch.ops.linalg_tpc.jacobi_big`` (``_refine``: the short pass), built on first use."""
    _load()
    return torch.ops.linalg_tpc.jacobi_big_refine if refine else torch.ops.linalg_tpc.jacobi_big


def _eigh_refine_op():
    """``torch.ops.linalg_tpc.jacobi_eigh_refine``, built on first use."""
    _load()
    return torch.ops.linalg_tpc.jacobi_eigh_refine


def _tri_op(upper, unitriangular):
    """``torch.ops.linalg_tpc.tri_solve_*``, built on first use."""
    _load()
    name = "tri_solve" + ("_upper" if upper else "_lower") + ("_unit" if unitriangular else "")
    return getattr(torch.ops.linalg_tpc, name)


def _lazy():
    """Lazy mode (read from the bridge, see ``lazy_mode``), and not tracing under ``torch.compile``."""
    return lazy_mode() and not torch.compiler.is_compiling()


# Measured on Gaudi2, bridge 1.21.4, lazy mode: a TPC custom op whose inputs come from the pending lazy graph can fail
# with "ValidateSyncInputTensors tensor_data is empty" (nondeterministic). A mark_step BEFORE the op prevents it (2 of
# 2 gate runs race-free); a mark_step only AFTER the op does not, and packing the outputs into one tensor does not
# either. The cost is one graph boundary per call; eager mode needs none.
def _flush():
    """``mark_step`` in lazy mode; nothing in eager mode, while tracing under ``torch.compile``, or without Habana."""
    if not _lazy():
        return
    try:
        import habana_frameworks.torch.core as htcore
    except ImportError:
        return
    htcore.mark_step()


# Measured on Gaudi2, bridge 1.21.4, lazy mode (repro/lazy_custom_op_output_cpu.py): the output of a custom op read
# with ``.cpu()`` and no ATen op after it comes back wrong (relative error about 1); one ATen op after it gives the
# right values. So in lazy mode each output is cloned: a copy, no extra graph boundary.
def _call(op, *args):
    """``op(*args)`` after a lazy graph boundary (``mark_step`` before the op only); in lazy mode the outputs are
    copies, so an ATen node always follows the custom op."""
    _flush()
    out = op(*args)
    if not _lazy():
        return out
    return tuple(t.clone() for t in out) if isinstance(out, tuple) else out.clone()


def _fresh(t):
    """``t``, or a copy of it when it may alias other storage: a view (``_base``), an offset into its storage, or a
    storage larger than ``t`` (inference-mode views have ``_base`` None), so the kernel never reads a view of a graph
    input, which breaks lazy mode."""
    if t._base is not None or t.storage_offset() != 0:
        return t.clone()
    try:
        whole = t.untyped_storage().nbytes() == t.numel() * t.element_size()
    except (RuntimeError, NotImplementedError):  # a tensor without accessible storage: copy, to be safe
        whole = False
    return t if whole else t.clone()


def _flat(x, batch, *shape):
    return _fresh(x.expand(*batch, *shape).reshape(-1, *shape).to(torch.float32).contiguous())


def _unflatten(x, batch, *tail):
    if not batch:
        return x.squeeze(0)
    if len(batch) == 1:
        return x
    return x.reshape(*batch, *tail)


def _chunked(A, B):
    """The kernel on ``A[b, n, n]`` and ``B[b, n, m]`` for any ``m``: columns of ``B`` in contiguous chunks of at most
    ``MAXN``, ``X`` concatenated, ``D`` and ``S`` from the first chunk (they depend on ``A`` alone)."""
    op = _op()
    if B.shape[-1] <= MAXN:
        return _call(op, A, B)
    parts = [_call(op, A, _fresh(chunk.contiguous())) for chunk in B.split(MAXN, -1)]
    return torch.cat([p[0] for p in parts], -1), parts[0][1], parts[0][2]


def _tri_chunked(op, A, B):
    if B.shape[-1] <= MAXN:
        return _call(op, A, B)
    return torch.cat([_call(op, A, _fresh(chunk.contiguous())) for chunk in B.split(MAXN, -1)], -1)


def _check(A, n, *others, limit=MAXN):
    if A.dim() < 2 or (limit is not None and n > limit) or A.shape[-2] != n:
        raise ValueError(f"hpu_ops.tpc takes square A with n <= {limit}, got {tuple(A.shape)}")
    if any(t.dtype != torch.float32 for t in (A, *others)):
        # the kernels compute in float32: a float64 input would be downcast silently
        raise ValueError(f"hpu_ops.tpc takes float32 tensors, got {[str(t.dtype) for t in (A, *others)]}")


def large_pays(n, batch):
    """Whether the routing sends ``batch`` matrices of order ``n > MAXN`` to the in-place kernels: every size and
    batch, eager and lazy, but not while tracing under ``torch.compile`` (main's route there; the compiled path was not
    measured). Tracing (make_fx, AOT autograd, where is_compiling() is False) is declined by the caller, which sees the
    tensors. Past ``LARGE_MAXN``, and for small batches below it (``_square_pays``), the blocked route runs."""
    return not torch.compiler.is_compiling()


# The blocked route (n > LARGE_MAXN, or a batch too small to fill the card's 24 TPCs with whole matrices): block
# Gauss-Jordan over panels of about PANEL columns, each panel's pivots chosen by partial pivoting over the whole
# remaining height (``gj_tall``, the in-place kernel on a tall panel: the same pivot rows as unblocked GEPP), the
# trailing updates two matmuls per panel. Measured on Gaudi2 (bridge 1.21.4, batch 1, eager): the square kernel takes
# 2.3 / 9.1 / 24.9 ms at n = 220 / 384 / 512 (one matrix per TPC, n^2 ceil(n / 64) vector-row steps); a panel of width
# c over r rows costs r c ceil(c / 64), so the panels' sum is about n^2 PANEL / 128.
PANEL = 128
TALL_MAXM = 2048  # rows of a gj_tall panel (VR = 32 in gj_large_f32_gaudi2.c); taller panels take a tournament
TPCS = 24  # Gaudi2: one matrix per TPC
# What one panel of the blocked route costs on top of its kernel steps (its launches or graph boundary, the reorder,
# gathers and matmuls), in the kernel's vector-row steps (about 13 ns each): fitted to the same measurements, n = 512
# batch 1 inverse forward and backward, square 28.2 ms against blocked 9.1 (lazy) and 29.5 against 27.4 (eager).
PANEL_OVERHEAD = {"lazy": 90000, "eager": 400000}


def _panel_widths(n):
    """Near-equal panel widths summing to ``n``, each of at most about ``PANEL`` and more than ``MAXN`` columns (the
    kernel's lower limit)."""
    p = max(1, min(-(-n // PANEL), n // (MAXN + 1)))
    q, r = divmod(n, p)
    return [q + 1] * r + [q] * (p - r)


def _square_pays(n, batch):
    """Whether one square in-place kernel call beats the blocked route for ``batch`` matrices of order ``n`` (<=
    ``LARGE_MAXN``), by the cost model above (kernel steps per wave of ``TPCS`` matrices, plus each panel's overhead)."""
    waves = -(-batch // TPCS)
    square = waves * n * n * (-(-n // 64))
    overhead = PANEL_OVERHEAD["lazy" if _lazy() else "eager"]
    widths, rows, blocked = _panel_widths(n), n, 0
    for c in widths:
        blocked += waves * rows * c * (-(-c // 64)) + overhead
        rows -= c
    return square <= blocked


def _parity(P):
    """``(-1)^inversions`` of the permutations ``P[b, n]`` (float, exact integers): the sign of the row order."""
    n = P.shape[-1]
    later = torch.ones(n, n, dtype=torch.bool, device=P.device).triu(1)
    inversions = ((P[..., :, None] > P[..., None, :]) & later).sum((-2, -1))
    return (1 - 2 * (inversions % 2)).to(P.dtype)


def _rows(X, index):
    """``X[..., index, :]`` for an index ``[b, n]`` (int64, or float holding exact integers): a gather of float data."""
    return X.gather(-2, index.long()[..., None].expand(*index.shape, X.shape[-1]))


def _arange(n, like, batch):
    return torch.arange(n, dtype=torch.float32, device=like.device).expand(batch, n)


def _panel(W):
    """Partial pivoting over the whole height of the panel ``W[b, r, c]`` (MAXN < c <= LARGE_MAXN, c <= r):
    ``(rows, X, D, info, key)`` with ``rows[b, c]`` the pivot row of each step (float, exact integers), ``X[b, c, c] =
    W[rows]^-1``, ``D[b, c]`` the pivots (``det W[rows] = prod D``), ``info[b]`` 1 + the first zero pivot (else 0) and
    ``key[b, r]`` whose ascending order puts the pivot rows first, in order, then the others in theirs, and ``R[b, r,
    c]``, the kernel's in-place panel (``R[rows] = X``, the other rows ``-W[o] X``, computed by the elimination: the
    multipliers of the Schur update, more accurate than the matmul ``W[o] @ X``), None past ``TALL_MAXM``. One
    ``gj_tall`` launch up to ``TALL_MAXM`` rows; past it a tournament (CALU): each chunk of rows nominates its own
    pivot rows, and the panel of the nominees picks the final ones."""
    b, r, c = W.shape
    if r > TALL_MAXM:
        assert 2 * c <= TALL_MAXM, "a tournament round must halve the candidates"
        k = -(-r // TALL_MAXM)
        q, extra = divmod(r, k)
        starts = [i * q + min(i, extra) for i in range(k + 1)]
        nominees = torch.cat([_panel(_fresh(W[:, a:e].contiguous()))[0] + a for a, e in zip(starts, starts[1:])], -1)
        rows_c, X, D, info, _, _ = _panel(_rows(W, nominees))
        rows = nominees.gather(-1, rows_c.long())
        key = (_arange(r, W, b) + c).scatter(-1, rows.long(), _arange(c, W, b))
        return rows, X, D, info, key, None
    R, P, Q, D, S = _call(_large_op(), W)
    R = R[..., :c]
    return P, _rows(R, P), D, S[..., 1], Q, R


def _cat(parts, dim):
    """``torch.cat`` of the non-empty ``parts``: on Gaudi2 (bridge 1.21.4, eager) a cat along the rows with an empty
    slice of a gather's output among its inputs failed its launch (synStatus 30, "Sections' validation failure";
    ``tests/probe_blocked_ops.py``)."""
    kept = [p for p in parts if p.shape[dim] > 0] or parts[:1]
    return kept[0] if len(kept) == 1 else torch.cat(kept, dim)


def _blocked(A, inverse=True, sign=True):
    """Block Gauss-Jordan with partial pivoting over the whole height for ``A[b, n, n]`` (float32, contiguous):
    ``(X, D, S)`` with ``X = A^-1`` (None unless ``inverse``), ``D`` the pivots and ``S[:, 0]`` the sign of the row
    order (0 unless ``sign``), ``S[:, 1]`` the info. Panel by panel: the panel's pivot rows (``_panel``) are moved
    before the rows not yet taken, the pivot block inverted by the kernel, the other rows updated by one matmul (with
    ``inverse``, the rows already taken too: the in-place inverse of ``A[perm]``, whose columns are then put back in
    ``A``'s row order; without it only the trailing Schur complement, LU-style, for det and slogdet)."""
    b, n, _ = A.shape
    perm = _arange(n, A, b)
    M = A  # with ``inverse`` the whole matrix, else the trailing block of the rows and columns not yet taken
    Ds, s = [], 0
    for c in _panel_widths(n):
        e = s + c
        W = M[:, s:, s:e] if inverse else M[..., :c]
        rows, X, D, _, key, R = _panel(_fresh(W.contiguous()))
        order = key.argsort(-1)
        perm = _cat([perm[:, :s], perm[:, s:].gather(-1, order)], -1)
        Ds.append(D)
        # K: the rows not taken yet, -W[o] X, from the kernel's elimination (the matmul past TALL_MAXM)
        if inverse:
            tail = _rows(M[:, s:], order)
            Mp = tail[:, :c]
            K = _rows(R, order)[:, c:] if R is not None else -tail[:, c:, s:e] @ X
            if s:  # the rows already taken: -M[taken, panel] X by a matmul (they were not the kernel's candidates)
                K = _cat([-M[:, :s, s:e] @ X, K], 1)
            rest = _cat([M[:, :s], tail[:, c:]], 1)
            Mp0 = _cat([Mp[..., :s], torch.zeros_like(X), Mp[..., e:]], -1)
            XM = X @ Mp0
            G = _cat([XM[..., :s], X, XM[..., e:]], -1)
            rest = _cat([rest[..., :s], K, rest[..., e:]], -1) + K @ Mp0
            M = _cat([rest[:, :s], G, rest[:, s:]], 1)
        elif e < n:
            T = _rows(M, order)
            K = _rows(R, order)[:, c:] if R is not None else -T[:, c:, :c] @ X
            M = T[:, c:, c:] + K @ T[:, :c, c:]
        s = e
    D = _cat(Ds, -1)
    # info: 1 + the first zero pivot (the kernel's own info, per panel, is the same test on its D)
    zero = torch.where(D == 0, _arange(n, A, b) + 1, float(n + 1)).amin(-1)
    info = torch.where(zero > n, 0.0, zero)
    S = torch.stack([_parity(perm) if sign else torch.zeros_like(info), info], -1)
    if not inverse:
        return None, D, S
    # M is the inverse of A[perm]: A^-1 = M with its columns put back in A's row order
    return M.gather(-1, perm.argsort(-1)[:, None, :].expand(b, n, n)), D, S


def _square(A, sign=True):
    """The square in-place kernel's ``(R, P, Q, D, S)`` for ``A[b, n, n]`` (MAXN < n <= LARGE_MAXN), ``S[:, 0]`` the
    sign of the pivot order (0 when not ``sign``: solve and inv do not read it, and its [b, n, n] comparison costs
    launches). The kernel makes R and D NaN for a matrix with a NaN (LAPACK's results are NaN there)."""
    R, P, Q, D, S = _call(_large_op(), A)
    if sign:
        S = torch.stack([_parity(P), S[..., 1]], -1)
    return R, P, Q, D, S


# det and slogdet past this n take the blocked route whatever the cost model says: its Schur updates are matmuls,
# where the square kernel accumulates n rank-1 updates per entry; measured on Gaudi2 (bridge 1.21.4, linalg_on_hpu's
# near-identity matrices, batch 4, n = 384): logabsdet 3.4e-6 square, 1.4e-6 blocked, CPU float32 1.9e-7
DET_BLOCKED_N = 256


def _square_pays_for(A, det=False):
    n = A.shape[-1]
    if det and n > DET_BLOCKED_N:
        return False
    return n <= LARGE_MAXN and _square_pays(n, A.shape[0])


def large_factors(A):
    """``(D, S)`` of the in-place kernels for ``A[..., n, n]`` (n > ``MAXN``), for ``det`` and ``slogdet``: the pivots
    and (sign, info), without forming the inverse. Run on ``A^T`` (see ``_large``)."""
    n = A.shape[-1]
    _check(A, n, limit=None)
    batch = A.shape[:-2]
    At = _flat(A, batch, n, n).mT.contiguous()
    if _square_pays_for(At, det=True):
        _, _, _, D, S = _square(At)
    else:
        _, D, S = _blocked(At, inverse=False)
    return _unflatten(D, batch, n), _unflatten(S, batch, 2)


def _large(A, B=None, sign=True, det=False):
    """``(X, D, S)`` of the in-place Gauss-Jordan kernels for ``A[b, n, n]`` (n > ``MAXN``): ``X = A^-1`` (``B``
    None) or ``A^-1 B``, ``S[:, 0]`` the sign of the pivot order. Run on ``A^T``, as torch: its CPU solve, inv and det
    factor the row-major A as LAPACK's column-major A^T, so the pivot order is LAPACK's on the matrix torch hands it.
    On ``A`` the kernel took the other order, and partial pivoting's 2^(n-1) growth on Wilkinson's matrix (``W = I -
    tril(1, -1)``, last column 1; condition 58 at n = 130) overflowed where torch's factors stay bounded (Codex review
    #1); torch overflows on ``W^T``, and so do we. One square kernel call (``A^-T = R[P, :][:, Q]``, so ``A^-1 =
    R^T[Q, :][:, P]`` and ``A^-1 B = (R^T @ B[Q])[Q]``) when it pays, else the blocked route (``_blocked``)."""
    n = A.shape[-1]
    At = A.mT.contiguous()
    if not _square_pays_for(At, det):
        Xt, D, S = _blocked(At, sign=sign)
        # one Newton step X += X (I - A X): the block updates multiply by each pivot block's explicit inverse, which
        # left the inverse up to 15x CPU float32's error (replica, n = 600); after it, 0.2-0.5x. A non-finite
        # correction (a singular block: zero rows) is dropped.
        C = Xt @ (torch.eye(n, dtype=At.dtype, device=At.device) - At @ Xt)
        Xt = Xt + torch.where(torch.isfinite(C), C, 0.0)
        return (Xt.mT if B is None else Xt.mT @ B), D, S
    R, P, Q, D, S = _square(At, sign)
    Rt = R[..., :n].mT
    # the kernel divides each row by its pivot itself (a torch division by a pivot of 1e38 zeroed the row in lazy mode:
    # the card's division multiplies by a reciprocal, subnormal hence flushed, and a 2^-64 prescale was folded away)
    if B is None:
        # no Newton step: on the kernel's float32 replica it gained little at condition <= 1e4 and lost past 1e6
        # (n = 220, condition 1e7: 6.8e-2 against 4.5e-2 without, LAPACK 4.7e-2)
        return _rows(Rt, Q).gather(-1, P.long()[..., None, :].expand(*P.shape[:-1], n, n)), D, S
    return _rows(Rt @ _rows(B, Q), Q), D, S


def _small_inv(A):
    """``(A^-1, D, S)`` for ``A[b, n, n]`` (n <= ``MAXN``) from the identity kernel run on ``A^T``, as the large path
    and torch's CPU kernels do (LAPACK factors the row-major A as its column-major A^T): on ``A`` itself the pivot
    order differs, and Wilkinson-type matrices (``I - tril(1, -1)``, last column 1; condition 11-29) lost every digit
    to partial pivoting's 2^(n-1) growth, duplicate rows cancelled to det exactly 0 with a zero gradient (torch: the
    adjugate), ``[[2^100, 2^100], [2^-100, 2^-100]]`` reported det 1 and info 0, and ``[[0, 1], [nan, 1]]`` info 1
    (code review of main, finding 3). ``D`` and ``S`` are those of ``A^T``'s elimination; ``det A = det A^T``."""
    X, D, S = _call(_inv_op(), A.mT.contiguous())
    return X.mT, D, S


def _inv_solve(A, B, Ainv):
    """``A^-1 B`` from the inverse, with one residual step ``X += A^-1 (B - A X)`` (a non-finite correction, from Inf
    in A, is dropped): on the in-place kernel's float32 replica the forward error then matches LAPACK's from condition
    1 to 1e7."""
    X = Ainv @ B
    C = Ainv @ (B - A @ X)
    return X + torch.where(torch.isfinite(C), C, 0.0)


def gj_solve(A, B):
    """``(X, D, S)`` for ``A[..., n, n]`` and ``B[..., n, m]`` (float32) from the TPC kernels: ``X = A^-1 B`` (the
    inverse of ``A^T``'s elimination, ``_small_inv`` / ``_large``, times ``B``, one residual step for n <= ``MAXN``),
    ``D`` the pivots, ``S[..., 0]`` the sign of the permutation and ``S[..., 1]`` the info (1 + the first zero pivot
    in the kernel's elimination order, else 0); shaped ``[..., n, m]``, ``[..., n]``, ``[..., 2]``."""
    n, m = A.shape[-1], B.shape[-1]
    _check(A, n, B, limit=None)
    if B.dim() < 2 or B.shape[-2] != n:
        raise ValueError(f"hpu_ops.tpc.gj_solve takes B with n rows, got {tuple(A.shape)} and {tuple(B.shape)}")
    batch = torch.broadcast_shapes(A.shape[:-2], B.shape[:-2])
    A, B = _flat(A, batch, n, n), _flat(B, batch, n, m)
    if n <= MAXN:
        Ainv, D, S = _small_inv(A)
        X = _inv_solve(A, B, Ainv)
    else:
        X, D, S = _large(A, B)
    return _unflatten(X, batch, n, m), _unflatten(D, batch, n), _unflatten(S, batch, 2)


def gj_inv(A, sign=True, det=False):
    """``(A^-1, D, S)`` for ``A[..., n, n]``: the kernel on ``A^T`` with the identity as right-hand side, which the
    kernel builds itself (``_small_inv``; past ``MAXN`` the in-place kernels, ``_large``). Past ``MAXN``,
    ``sign=False`` leaves ``S[..., 0]`` 0 (not computed)."""
    n = A.shape[-1]
    _check(A, n, limit=None)
    batch = A.shape[:-2]
    A = _flat(A, batch, n, n)
    X, D, S = _small_inv(A) if n <= MAXN else _large(A, sign=sign, det=det)
    return _unflatten(X, batch, n, n), _unflatten(D, batch, n), _unflatten(S, batch, 2)


def _tri_blocked(op, A, B, upper):
    """Blocked substitution for ``A[b, n, n]`` (n > ``MAXN``), ``B[b, n, m]``: diagonal blocks of at most ``MAXN``
    through the substitution kernel ``op``, each block's right-hand side first updated by one matmul with the blocks
    already solved (forward for lower, backward for upper). The other triangle is never read: the off-diagonal blocks
    used lie in ``A``'s own triangle."""
    n = A.shape[-1]
    p = -(-n // MAXN)
    q, r = divmod(n, p)
    bounds = [0]
    for i in range(p):
        bounds.append(bounds[-1] + q + (i < r))
    X = [None] * p
    for i in (range(p - 1, -1, -1) if upper else range(p)):
        a, e = bounds[i], bounds[i + 1]
        rhs = B[:, a:e]
        done = range(i + 1, p) if upper else range(i)
        if done:
            lo, hi = (e, n) if upper else (0, a)
            rhs = rhs - A[:, a:e, lo:hi] @ torch.cat([X[j] for j in done], 1)
        X[i] = _tri_chunked(op, _fresh(A[:, a:e, a:e].contiguous()), _fresh(rhs.contiguous()))
    return torch.cat(X, 1)


def tri_solve(A, B, upper=False, left=True, unitriangular=False):
    """Solve ``A X = B`` (``X A = B`` if not ``left``) for triangular ``A[..., n, n]``: n <= ``MAXN`` one substitution
    kernel call per chunk of 64 columns of ``B``, past it the blocked substitution (``_tri_blocked``)."""
    n = A.shape[-1]
    _check(A, n, B, limit=None)
    if B.dim() < 2 or B.shape[-2 if left else -1] != n:
        raise ValueError(f"hpu_ops.tpc.tri_solve takes compatible B for left={left}, "
                         f"got {tuple(A.shape)} and {tuple(B.shape)}")
    A_eff = A if left else A.mT
    B_eff = B if left else B.mT
    upper_eff = upper if left else not upper
    batch = torch.broadcast_shapes(A.shape[:-2], B.shape[:-2])
    m = B_eff.shape[-1]
    solve = _tri_chunked if n <= MAXN else functools.partial(_tri_blocked, upper=upper_eff)
    X = solve(_tri_op(upper_eff, unitriangular), _flat(A_eff, batch, n, n), _flat(B_eff, batch, n, m))
    X = _unflatten(X, batch, n, m)
    return X if left else X.mT


def tri_solve_reference(A, B, upper=False, left=True, unitriangular=False):
    """``tri_solve`` in plain torch, the contract of the kernel (torch's own ``linalg.solve_triangular`` is what
    the CPU tests override). Opposite triangle (and diagonal when ``unitriangular``) is never read. Detached: no
    autograd through the kernel. For the CPU tests."""
    if not left:
        return tri_solve_reference(A.mT, B.mT, upper=not upper, left=True, unitriangular=unitriangular).mT
    batch = torch.broadcast_shapes(A.shape[:-2], B.shape[:-2])
    A = A.detach().expand(*batch, *A.shape[-2:])
    B = B.detach().expand(*batch, *B.shape[-2:])
    n = A.shape[-1]
    rows = [None] * n
    for i in (range(n - 1, -1, -1) if upper else range(n)):
        rhs = B[..., i, :]
        for j in (range(n - 1, i, -1) if upper else range(i)):  # the kernel's (and LAPACK's) order
            rhs = rhs - A[..., i, j:j + 1] * rows[j]
        if not unitriangular:
            rhs = rhs / A[..., i, i:i + 1]
        rows[i] = rhs
    return torch.stack(rows, dim=-2)


def _sort(x, descending=False):
    """``(values, indices)`` of ``x`` sorted along the last dimension, NaN kept as NaN with an index in range: Gaudi's
    sort returns NaN as +inf with out-of-range indices (measured, ``hpu_ops.extrema``), so the eigenvalues and singular
    values of a matrix with a NaN came out +-inf, the vectors gathered with bad indices (code review #13). The order
    key is ``x`` with NaN as +inf (``nan_to_num``, one op; NaN and +inf then tie, where torch puts NaN last); Gaudi
    sorts a key without NaN right at these sizes (n <= 32; its non-stable sort went wrong past 1000 elements with
    infs, and its 1-D stable sort runs on the CPU, so the sort is the plain one). The values are gathered from ``x``
    (float data, an int64 index): two ops more than the sort alone."""
    key = torch.nan_to_num(x, nan=float("inf"), posinf=float("inf"), neginf=float("-inf"))
    order = internal_sort(key, -1, descending)[1]  # Gaudi's own sort: the key has no NaN
    return x.gather(-1, order), order


def eigh(A, upper):
    """``(w, V)`` of the symmetric matrix held in one triangle of ``A[..., n, n]`` (float32, n <= ``EIGH_MAXN``) from
    the TPC kernel: ``w`` ascending, the columns of ``V`` the eigenvectors. The kernel returns ``w`` unsorted and the
    eigenvectors as rows; the sort and the gather of the float data run on the card."""
    n = A.shape[-1]
    if n > EIGH_MAXN or A.shape[-2] != n:
        raise ValueError(f"hpu_ops.tpc.eigh takes square A with n <= {EIGH_MAXN}, got {tuple(A.shape)}")
    batch = A.shape[:-2]
    if n == 1:  # trivial, and the kernel's graph does not compile at n = 1 in lazy mode (synStatus 26)
        A = A.to(torch.float32)
        return A.reshape(*batch, 1).clone(), torch.ones_like(A)
    # a fresh tensor, never a view of the caller's input: views break lazy mode
    S = torch.triu(A) + torch.triu(A, 1).mT if upper else torch.tril(A) + torch.tril(A, -1).mT
    S_flat = S.reshape(-1, n, n).to(torch.float32).contiguous()
    _, Vt1 = _call(_eigh_op(), S_flat)
    W, Vt = _call(_eigh_refine_op(), S_flat, Vt1)
    w, order = _sort(W)
    V = Vt.gather(-2, order[..., :, None].expand_as(Vt)).mT
    return w.reshape(*batch, n), V.reshape(*batch, n, n)


BIG_EIGH_MAXN = 128  # jacobi_big: 32 < n <= BIG_EIGH_MAXN (VA = 1, 2); past it the block Jacobi of hpu_ops.jacobi
BIG_SHIFT = 1.001  # S = A / ||A||_inf + BIG_SHIFT I: positive definite, eigenvalues in [0.001, 2.001]


def _big_pass(X, sigma, refine, shift=None):
    """One kernel pass (``jacobi_big``, 8 sweeps; ``refine``: 3) on ``X[b, n, n]`` symmetric, shifted positive
    definite by its row-sum bound ``sigma[b]``: the rows of the eigenvector estimate ``[b, n, n]``."""
    b, n, _ = X.shape
    w = 64 * (-(-n // 64))
    eye = torch.eye(n, dtype=X.dtype, device=X.device)
    S = X / sigma[:, None, None] + (BIG_SHIFT if shift is None else shift[:, None, None]) * eye
    U = _call(_big_op(refine), _fresh(torch.nn.functional.pad(S, (0, w - n)).contiguous()))[..., :n]
    # the rows of S V are lambda_j v_j at convergence (S positive definite): normalized, the eigenvectors
    return U * torch.rsqrt((U * U).sum(-1, keepdim=True).clamp_min(1e-36))


def eigh_big(A):
    """``(w, V)`` of the symmetric ``A[..., n, n]`` (both triangles read; float32, 32 < n <= ``BIG_EIGH_MAXN``): ``w``
    ascending, the columns of ``V`` the eigenvectors. The one-sided Jacobi kernel ``jacobi_big`` (rows in global
    memory, one matrix per TPC) on ``A`` shifted positive definite (``BIG_SHIFT``), then a refinement pass on
    ``V A V^T`` (nearly diagonal: 3 sweeps), two Newton-Schulz steps on the composed rows and Rayleigh quotients with
    ``A`` itself. Measured on Gaudi2 (bridge 1.21.4, batch 32): EEGSPDNet's eigh at n = 110 and 55 took 360 and 121
    ms through the block Jacobi of 32 x 32 launches (``hpu_ops.jacobi``), 27 and 13 on hpu-ops' CPU route."""
    batch, n = A.shape[:-2], A.shape[-1]
    if A.dim() < 2 or A.shape[-2] != n or not EIGH_MAXN < n <= BIG_EIGH_MAXN:
        raise ValueError(f"hpu_ops.tpc.eigh_big takes square A with {EIGH_MAXN} < n <= {BIG_EIGH_MAXN}, got {tuple(A.shape)}")
    X = A.reshape(-1, n, n).to(torch.float32)
    sigma = X.abs().sum(-1).amax(-1)
    sigma = torch.where(sigma > 0, sigma, 1.0)
    Xs = X / sigma[:, None, None]
    ones = torch.ones_like(sigma)
    V = _big_pass(Xs, ones, False)  # rows
    T = V @ Xs @ V.mT
    T = (T + T.mT) / 2
    # The refinement on T = V Xs V^T, nearly diagonal: unshifted where Gershgorin proves T positive definite (one-sided
    # Jacobi on a definite matrix keeps its small eigenvalues' relative accuracy; the shift of 1.001 compressed them:
    # 1e-6 came back as 1.41e-6 at n = 127, code review of #51), else shifted as the first pass
    off = T.abs().sum(-1) - torch.diagonal(T, dim1=-2, dim2=-1).abs()
    definite = (torch.diagonal(T, dim1=-2, dim2=-1) - off > 1e-12).all(-1)
    V = _big_pass(T, ones, True, torch.where(definite, 0.0, BIG_SHIFT)) @ V
    eye = torch.eye(n, dtype=X.dtype, device=X.device)
    for _ in range(2):
        V = (1.5 * eye - 0.5 * (V @ V.mT)) @ V
    w = ((V @ Xs) * V).sum(-1) * sigma[:, None]
    w, order = w.sort(-1)
    V = V.gather(-2, order[..., :, None].expand_as(V)).mT
    return w.reshape(*batch, n), V.reshape(*batch, n, n)


def jacobi_big_reference(S0, sweeps=8):
    """``jacobi_big`` in float64 numpy, the kernel's contract (round-robin one-sided Jacobi on the rows of ``S0``; a
    pair orthogonal to 1e-7 left alone), for the CPU tests."""
    import numpy as np

    U = S0.detach().cpu().double().numpy().copy()
    V = U.copy()
    n = U.shape[-2]
    M = n if n % 2 else n - 1
    for b in range(U.shape[0]):
        u, v = U[b], V[b]
        for _ in range(sweeps):
            for r in range(M):
                pairs = [(r, M)] if M < n else []
                for k in range(1, (M + 1) // 2):
                    x, y = (r + k) % M, (r - k) % M
                    pairs.append((min(x, y), max(x, y)))
                for p, q in pairs:
                    a, bb, g = u[p] @ u[p], u[q] @ u[q], u[p] @ u[q]
                    if g * g <= 1e-14 * a * bb:
                        continue
                    zeta = (bb - a) / (2 * g)
                    t = np.sign(zeta) / (abs(zeta) + np.sqrt(1 + zeta * zeta)) if zeta != 0 else 1.0
                    c = 1 / np.sqrt(1 + t * t)
                    s = c * t
                    u[p], u[q] = c * u[p] - s * u[q], s * u[p] + c * u[q]
                    v[p], v[q] = c * v[p] - s * v[q], s * v[p] + c * v[q]
    return torch.from_numpy(U).to(S0.dtype)


def eigh_reference(A, upper):
    """``eigh`` through numpy's LAPACK, the contract of the kernel path (torch's own ``linalg.eigh`` is what the CPU
    tests override). Detached: no autograd through the kernel. For the CPU tests."""
    import numpy as np  # only here: ``import hpu_ops`` does not need numpy

    w, V = np.linalg.eigh(A.detach().cpu().numpy(), UPLO="U" if upper else "L")
    return torch.from_numpy(w).to(A.device), torch.from_numpy(V).to(A.device)


def gj_solve_reference(A, B):
    """``gj_solve`` in plain torch, the contract of the kernel: ``lu`` and ``lu_solve`` of ``hpu_ops.linalg``, ``D``
    the diagonal of U (a zero pivot leaves later pivots finite; the kernel's row of X for it is zero, this
    reference's is not finite), ``S[..., 0]`` the parity of the pivots and ``S[..., 1]`` the info. For the CPU tests."""
    from .. import linalg

    LU, pivots = linalg.lu(A)
    X = linalg.lu_solve(LU, pivots, B)
    D = torch.diagonal(LU, dim1=-2, dim2=-1)
    S = torch.stack([linalg._parity(pivots).to(A.dtype), linalg.lu_info_of_diagonal(D).to(A.dtype)], -1)
    return X, D, S


def _float32(*tensors):
    """The kernels compute in float32 on real data: a complex tensor would be cast with its imaginary part dropped
    (Codex review, minor: ``matrix_exp(tensor([[1 + 1j]]))`` gave 2.71828), so a direct call with one raises. Real
    floating dtypes are cast as before: on the card a float64 tensor holds float32 values (hpu-ops#16), and the
    routes pass float32 only."""
    if any(t.is_complex() or not t.is_floating_point() for t in tensors):
        raise ValueError(f"hpu_ops.tpc takes real floating (float32) tensors, got {[str(t.dtype) for t in tensors]}")


def _check_small(A, n):
    if A.dim() < 2 or n > SMALL_MAXN or A.shape[-2] != n:
        raise ValueError(f"hpu_ops.tpc takes square A with n <= {SMALL_MAXN}, got {tuple(A.shape)}")


def _check_svd(A):
    if A.dim() < 2 or A.shape[-2] > SMALL_MAXN or A.shape[-1] > SMALL_MAXN:
        raise ValueError(f"hpu_ops.tpc takes 2D+ A with m, n <= {SMALL_MAXN}, got {tuple(A.shape)}")


def jacobi_svd(A, full_matrices=False, sort=True, flush=True):
    """SVD ``(U, S, Vh)`` for ``A[..., m, n]`` (float32, m, n <= ``SMALL_MAXN``) from the one-sided Jacobi kernel."""
    _check_svd(A)
    _float32(A)
    batch, m, n = A.shape[:-2], A.shape[-2], A.shape[-1]
    k = min(m, n)
    full = full_matrices and m != n
    if A.numel() == 0:
        return (A.new_empty(*batch, m, m if full else k, dtype=torch.float32),
                A.new_empty(*batch, k, dtype=torch.float32),
                A.new_empty(*batch, n if full else k, n, dtype=torch.float32))
    swap = m < n or (m == n and A.is_contiguous())
    big = max(m, n)
    A_t = _fresh((A if swap else A.mT).reshape(-1, k, big).to(torch.float32).contiguous())
    op = _svd_full_op() if full else _svd_op()
    Ut, S, Vh = _call(op, A_t) if flush else op(A_t)
    if swap:
        U, Vh = Vh.mT.reshape(*batch, m, k), Ut.reshape(*batch, n if full else k, n)
    else:
        U, Vh = Ut.mT.reshape(*batch, m, m if full else k), Vh.reshape(*batch, k, n)
    S = S.reshape(*batch, k)
    if not sort:
        return U, S, Vh
    S, order = _sort(S, descending=True)
    if full:
        full_order = torch.cat([order, torch.arange(k, big, device=order.device).expand(*batch, big - k)], -1)
        u_ord, v_ord = (order, full_order) if swap else (full_order, order)
    else:
        u_ord = v_ord = order
    return U.gather(-1, u_ord[..., None, :].expand_as(U)), S, Vh.gather(-2, v_ord[..., :, None].expand_as(Vh))


def jacobi_svdvals(A, sort=True, flush=True):
    """Singular values ``S`` for ``A[..., m, n]`` (float32, m, n <= ``SMALL_MAXN``) from ``-DNO_UV``."""
    _check_svd(A)
    _float32(A)
    batch, m, n = A.shape[:-2], A.shape[-2], A.shape[-1]
    k = min(m, n)
    if A.numel() == 0:
        return A.new_empty(*batch, k, dtype=torch.float32)
    no_t = m < n or (m == n and A.is_contiguous())
    A_t = _fresh((A if no_t else A.mT).reshape(-1, k, max(m, n)).to(torch.float32).contiguous())
    S = (_call(_svdvals_op(), A_t) if flush else _svdvals_op()(A_t)).reshape(*batch, k)
    return _sort(S, descending=True)[0] if sort else S


def _ftz32(t):
    """``t`` with float32 values below the smallest normal flushed to 0, as the card's arithmetic does (float64
    unchanged)."""
    if t.dtype != torch.float32:
        return t
    return torch.where(t.abs() < torch.finfo(torch.float32).tiny, 0.0, t)


def svd_reference(A, compute_uv=True, full_matrices=False):
    """``jacobi_svd`` / ``jacobi_svdvals`` in plain torch, matching the kernel contract: the rows of A orthogonalized
    when m < n or A is square and contiguous (else the rows of A^T), 1/max|a| scaling (not of a zero matrix), the
    kernel's round-robin rotations and skip rules on alpha, beta, gamma scaled by 1 / (alpha + beta) (gamma = 0,
    gamma^2 <= tol alpha beta, r^2 < 1e-34; exactly 45 degrees for equal norms), row norms as max|x| ||x / max|x|||,
    MGS completion of the zero and near-zero rows (below 1e-5 of the largest), optional full
    orthogonal extension, descending ``S``. For the CPU tests (``tests/kernel_replica.py`` replays the float32
    arithmetic)."""
    _check_svd(A)
    batch, m, n = A.shape[:-2], A.shape[-2], A.shape[-1]
    k = min(m, n)
    full = full_matrices and m != n
    if A.numel() == 0:
        S = A.new_empty(*batch, k)
        if not compute_uv:
            return A.new_empty(0), S, A.new_empty(0)
        return A.new_empty(*batch, m, m if full else k), S, A.new_empty(*batch, n if full else k, n)

    swap = m < n or (m == n and A.is_contiguous())
    big = max(m, n)
    W = (A if swap else A.mT).reshape(-1, k, big).clone()
    b = W.shape[0]
    sigma = W.abs().amax((-2, -1), keepdim=True)
    safe_sigma = torch.where(sigma == 0, 1.0, sigma)
    W = W / safe_sigma
    V = torch.eye(k, dtype=A.dtype, device=A.device).expand(b, k, k).clone() if compute_uv else None

    def row_norms(X):  # the kernel's ROW_NORM: max|x| * ||x / max|x||| (no squares of tiny rows), and x / max|x|
        mx = X.abs().amax(-1, keepdim=True)
        cw = X / torch.where(mx == 0, 1.0, mx)
        return mx.squeeze(-1), cw, cw.norm(dim=-1) * mx.squeeze(-1)

    if k > 1:
        k_even = (k + 1) & ~1
        rr_m = k_even - 1
        rounds = []
        for r in range(rr_m):
            ps, qs = [], []
            for j in range(k & 1, k_even >> 1):
                if j == 0:
                    p, q = r, rr_m
                else:
                    p, q = (r + j) % rr_m, (r - j) % rr_m
                    if p > q:
                        p, q = q, p
                ps.append(p)
                qs.append(q)
            rounds.append((torch.tensor(ps, device=A.device), torch.tensor(qs, device=A.device)))
        sweeps = 6 if k <= 4 else 10 if k <= 16 else 12
        tol_scale = 1e-30 if A.dtype == torch.float64 else 1e-14
        for _ in range(sweeps):
            for ps, qs in rounds:
                wp, wq = W[:, ps], W[:, qs]
                # the card flushes each product below float32's smallest normal (FTZ): so does the reference
                alpha = _ftz32(wp * wp).sum(-1)
                beta = _ftz32(wq * wq).sum(-1)
                gamma = _ftz32(wp * wq).sum(-1)
                # the kernel's JACOBI_CS: alpha, beta, gamma scaled by 1 / (alpha + beta), skip rules relative
                ab = alpha + beta
                scale = 1.0 / torch.where(ab == 0, 1.0, ab)
                a_, b_, g_ = alpha * scale, beta * scale, gamma * scale
                d = b_ - a_
                g2 = 2.0 * g_
                r = torch.hypot(d, g2)
                skip = (gamma == 0) | (r * r < 1e-34) | (g_ * g_ <= tol_scale * a_ * b_)
                sr = torch.where(skip, 1.0, r)
                c2 = 0.5 * (1.0 + d.abs() / sr)
                c = torch.where(skip, 1.0, c2.sqrt())
                s = torch.where(skip, 0.0, torch.where(d < 0, -1.0, 1.0) * 0.5 * g2 / (sr * torch.where(skip, 1.0, c)))
                half = math.sqrt(0.5)  # equal norms: exactly 45 degrees, so identical rows cancel exactly
                c = torch.where((d == 0) & ~skip, half, c)
                s = torch.where((d == 0) & ~skip, torch.where(g2 < 0, -half, half), s)
                c_, s_ = c.unsqueeze(-1), s.unsqueeze(-1)
                W[:, ps], W[:, qs] = c_ * wp - s_ * wq, s_ * wp + c_ * wq
                if compute_uv:
                    vp, vq = V[:, ps], V[:, qs]
                    V[:, ps], V[:, qs] = c_ * vp - s_ * vq, s_ * vp + c_ * vq

    mx, cw, norm = row_norms(W)
    zero = mx == 0
    norm = torch.where(zero, 0.0, norm)
    S = norm * safe_sigma.squeeze(-1)
    if not compute_uv:
        return A.new_empty(0), S.reshape(*batch, k).sort(-1, descending=True).values, A.new_empty(0)

    U_t = torch.where(zero.unsqueeze(-1), 0.0, cw / torch.where(zero, 1.0, cw.norm(dim=-1)).unsqueeze(-1))
    bad_mask = norm < torch.clamp(1e-5 * norm.amax(-1, keepdim=True), min=1e-37)  # zero and near-zero rows
    if bad_mask.any():
        for bi in range(b):
            if not bad_mask[bi].any():
                continue
            done = [j for j in range(k) if not bad_mask[bi, j]]
            # the near-zero rows (own directions) before the zero rows, which would take them (the kernel: diag_p)
            for j in sorted((j for j in range(k) if bad_mask[bi, j]), key=lambda j: bool(norm[bi, j] == 0)):
                accepted = False
                if norm[bi, j] > 0:
                    cand = U_t[bi, j].clone()
                    for _ in range(2):
                        for d_idx in done:
                            cand = cand - (cand @ U_t[bi, d_idx]) * U_t[bi, d_idx]
                    cn = cand.norm()
                    if cn > 0.5:
                        U_t[bi, j] = cand / cn
                        done.append(j)
                        accepted = True
                if not accepted:
                    for e in range(big):
                        cand = torch.zeros(big, dtype=A.dtype, device=A.device)
                        cand[e] = 1.0
                        for _ in range(2):
                            for d_idx in done:
                                cand = cand - (cand @ U_t[bi, d_idx]) * U_t[bi, d_idx]
                        cn = cand.norm()
                        if cn > 1e-2:
                            U_t[bi, j] = cand / cn
                            done.append(j)
                            break

    if full:
        U_full = torch.empty(b, big, big, dtype=A.dtype, device=A.device)
        U_full[:, :k] = U_t
        for bi in range(b):
            diag_p = 1.0 - (U_full[bi, :k] * U_full[bi, :k]).sum(0)
            for j in range(k, big):
                e = int(diag_p.argmax())
                cand = torch.zeros(big, dtype=A.dtype, device=A.device)
                cand[e] = 1.0
                for _ in range(2):
                    for prev in range(j):
                        uk = U_full[bi, prev]
                        cand = cand - (cand @ uk) * uk
                cand = cand / cand.norm().clamp(min=1e-36)
                U_full[bi, j] = cand
                diag_p = diag_p - cand * cand
        U_t = U_full

    if swap:
        U, Vh = V.mT.reshape(*batch, m, k), U_t.reshape(*batch, n if full else k, n)
    else:
        U, Vh = U_t.mT.reshape(*batch, m, m if full else k), V.reshape(*batch, k, n)
    S, order = S.reshape(*batch, k).sort(-1, descending=True)
    if full:
        full_order = torch.cat([order, torch.arange(k, big, device=order.device).expand(*batch, big - k)], -1)
        u_ord, v_ord = (order, full_order) if swap else (full_order, order)
    else:
        u_ord = v_ord = order
    return U.gather(-1, u_ord[..., None, :].expand_as(U)), S, Vh.gather(-2, v_ord[..., :, None].expand_as(Vh))


def householder_product(A, tau, flush=True):
    """Householder product ``Q`` for ``A[..., m, n]`` and ``tau[..., k]`` (float32, m <= ``MAXHH``, m >= n >= k)."""
    if A.dim() < 2:
        raise ValueError(f"hpu_ops.tpc takes 2D+ A, got {tuple(A.shape)}")
    _float32(A, tau)
    batch, m, n = A.shape[:-2], A.shape[-2], A.shape[-1]
    k = tau.shape[-1]
    if m > MAXHH or m < n or k > n or tau.dim() != A.dim() - 1 or tau.shape[:-1] != batch:
        raise ValueError(
            f"hpu_ops.tpc takes m >= n >= k with m <= {MAXHH}, got {tuple(A.shape)} and {tuple(tau.shape)}")
    if A.numel() == 0 or k == 0:
        return torch.eye(m, n, dtype=torch.float32, device=A.device).expand(*batch, m, n).clone()
    A_t = _fresh(A.mT.reshape(-1, n, m).to(torch.float32).contiguous())
    tau_f = _fresh(tau.reshape(-1, k).to(torch.float32).contiguous())
    op = _hh_op()
    Q_t = _call(op, A_t, tau_f) if flush else op(A_t, tau_f)
    return Q_t.mT.reshape(*batch, m, n)


def householder_product_reference(A, tau):
    """``householder_product`` in plain torch, matching the TPC kernel contract. For the CPU tests."""
    batch, m, n = A.shape[:-2], A.shape[-2], A.shape[-1]
    k = tau.shape[-1]
    if m > MAXHH or m < n or k > n or tau.dim() != A.dim() - 1 or tau.shape[:-1] != batch:
        raise ValueError(
            f"hpu_ops.tpc takes m >= n >= k with m <= {MAXHH}, got {tuple(A.shape)} and {tuple(tau.shape)}")
    if A.numel() == 0 or k == 0:
        return torch.eye(m, n, dtype=A.dtype, device=A.device).expand(*batch, m, n).clone()
    A_f = A.reshape(-1, m, n)
    tau_f = tau.reshape(-1, k)
    b = A_f.shape[0]
    Q = torch.eye(m, n, dtype=A.dtype, device=A.device).expand(b, m, n).clone()
    rows = torch.arange(m, device=A.device)
    for i in range(k - 1, -1, -1):
        v = torch.where(rows > i, A_f[..., :, i], torch.where(rows == i, 1.0, 0.0))
        t_v = tau_f[..., i:i + 1] * v
        dots = v.unsqueeze(-2) @ Q[..., :, i + 1:]
        Q[..., :, i + 1:] = Q[..., :, i + 1:] - t_v.unsqueeze(-1) * dots
        Q[..., :, i] = (rows == i).to(A.dtype) - t_v
    return Q.reshape(*batch, m, n)


def matrix_exp(A, flush=True):
    """Matrix exponential ``exp(A)`` for ``A[..., n, n]`` (float32, n <= ``SMALL_MAXN``) from the Gaudi2 TPC kernel."""
    n = A.shape[-1]
    _check_small(A, n)
    _float32(A)
    if A.numel() == 0:
        return A.new_empty(A.shape, dtype=torch.float32)
    batch = A.shape[:-2]
    A_f = _fresh(A.reshape(-1, n, n).to(torch.float32).contiguous())
    op = _expm_op()
    Y = _call(op, A_f) if flush else op(A_f)
    return Y.reshape(*batch, n, n)


def matrix_exp_frechet(A, G, flush=True):
    """Adjoint Fréchet derivative ``L_exp(A^T, G)`` for ``A, G[..., n, n]`` (float32, n <= ``SMALL_MAXN``) on Gaudi2 TPC."""
    n = A.shape[-1]
    _check_small(A, n)
    _float32(A, G)
    batch = torch.broadcast_shapes(A.shape[:-2], G.shape[:-2])
    if A.numel() == 0:
        return A.new_empty(*batch, n, n, dtype=torch.float32)
    flat = lambda x: _fresh(x.expand(*batch, n, n).reshape(-1, n, n).to(torch.float32).contiguous())  # noqa: E731
    op = _expm_frechet_op()
    Y = _call(op, flat(A), flat(G)) if flush else op(flat(A), flat(G))
    return Y.reshape(*batch, n, n)


def matrix_exp_reference(A):
    """``matrix_exp`` in plain torch with the TPC kernel's contract: each matrix scaled by 2^-s, s its own count of
    halvings of the 1-norm until <= theta_13 (at most 126, 0 if not finite: ``linalg._expm_scaling``), Padé-13,
    then s squarings. Detached. For the CPU tests."""
    from .. import linalg

    n = A.shape[-1]
    flat = A.detach().reshape(-1, n, n)
    out = [linalg._matrix_exp_composite(a, False) for a in flat]
    return (torch.stack(out) if out else flat.clone()).reshape(A.shape)


def matrix_exp_frechet_reference(A, G):
    """``matrix_exp_frechet`` in plain torch with the TPC kernel's contract (per-matrix scaling of A^T, as
    ``matrix_exp_reference``). Detached. For the CPU tests."""
    from .. import linalg

    n = A.shape[-1]
    batch = torch.broadcast_shapes(A.shape[:-2], G.shape[:-2])
    fa = A.detach().expand(*batch, n, n).reshape(-1, n, n)
    fg = G.detach().expand(*batch, n, n).reshape(-1, n, n)
    out = [linalg._matrix_exp_frechet_torch(a, g) for a, g in zip(fa, fg)]
    return (torch.stack(out) if out else fa.clone()).reshape(*batch, n, n)


def lu_factor(A, pivots=True):
    """``(LU, pivots, perm, info)`` for square ``A[..., n, n]`` (float32, n <= ``SMALL_MAXN``) from the TPC LU kernel:
    ``LU`` packed as ``torch.linalg.lu_factor`` returns it, ``pivots`` LAPACK's (int32, 1-based), ``perm`` the row
    permutation (float32, exact below 2**24: ``LU``'s row i is ``A``'s row ``perm[i]``) and ``info`` LAPACK's
    (int32). The kernel writes pivots, perm and info as float32; the int casts run on the card. With
    ``pivots=False`` only ``(LU, perm)``: ``linalg.lu`` needs no pivots or info."""
    n = A.shape[-1]
    _check_small(A, n)
    _float32(A)
    batch = A.shape[:-2]
    LU, PV = _call(_lu_op(), _fresh(A.reshape(-1, n, n).to(torch.float32).contiguous()))
    if not pivots:
        return LU.reshape(*batch, n, n), PV[:, 1].reshape(*batch, n)
    pivots = PV[:, 0].to(torch.int32).reshape(*batch, n)
    perm = PV[:, 1].reshape(*batch, n)
    info = PV[:, 2, 0].to(torch.int32).reshape(batch)
    return LU.reshape(*batch, n, n), pivots, perm, info


def lu_factor_reference(A):
    """``lu_factor`` in plain torch, the contract of the kernel: ``hpu_ops.linalg.lu`` (partial pivoting, first
    largest pivot, as LAPACK). Detached. For the CPU tests."""
    from .. import linalg

    LU, pivots = linalg.lu(A.detach())
    return LU, pivots, linalg.pivots_to_perm(pivots).to(A.dtype), linalg.lu_info(LU)


def lu_perm(pivots):
    """The row permutation (float32, exact below 2**24; ``LU``'s row i is ``A``'s row ``perm[i]``) of LAPACK's
    1-based ``pivots[..., n]`` (n <= ``SMALL_MAXN``) from the TPC kernel: the swaps replayed in order, one launch."""
    n = pivots.shape[-1]
    if n > SMALL_MAXN:
        raise ValueError(f"hpu_ops.tpc.lu_perm takes n <= {SMALL_MAXN} pivots, got {tuple(pivots.shape)}")
    batch = pivots.shape[:-1]
    perm = _call(_lu_perm_op(), _fresh(pivots.reshape(-1, n).to(torch.float32).contiguous()))
    return perm.reshape(*batch, n)


def lu_solve(LU, pivots, B):
    """``A^-1 B`` from LAPACK's factors ``LU[..., n, n]`` and 1-based ``pivots[..., n]`` (float32, n <= ``SMALL_MAXN``)
    and ``B[..., n, m]`` (any m, in chunks of ``SMALL_MAXN`` columns), batches broadcast, by the TPC getrs kernel: the row
    swaps, then forward and back substitution. One launch per chunk."""
    n, m = LU.shape[-1], B.shape[-1]
    _check_small(LU, n)
    _float32(LU, B)
    if B.shape[-2] != n:
        raise ValueError(f"hpu_ops.tpc.lu_solve takes B with n rows, got {tuple(LU.shape)} and {tuple(B.shape)}")
    batch = torch.broadcast_shapes(LU.shape[:-2], pivots.shape[:-1], B.shape[:-2])
    flat = lambda x, *shape: _fresh(x.expand(*batch, *shape).reshape(-1, *shape).to(torch.float32).contiguous())  # noqa: E731
    lu_f, piv_f = flat(LU, n, n), flat(pivots, n)
    B_f = flat(B, n, m)
    op = _lu_solve_op()
    if m <= SMALL_MAXN:
        X = _call(op, lu_f, piv_f, B_f)
    else:
        X = torch.cat([_call(op, lu_f, piv_f, _fresh(chunk.contiguous())) for chunk in B_f.split(SMALL_MAXN, -1)], -1)
    return X.reshape(*batch, n, m)


# complex64 kernels (for hpu-complex): ``hpu_ops.tpc.complex_linalg``; imported last, it reads this module's helpers
from . import complex_linalg  # noqa: E402
