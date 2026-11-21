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
- ``torch.ops.linalg_tpc.gj_large(A[b, n, n]) -> (R, P, Q, D, S)`` for 64 < n <= ``LARGE_MAXN`` (512): the in-place
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

Habana reads ``GC_KERNEL_PATH`` when the device initialises, so ``install_kernel_path`` must run before the first HPU
op of the process; ``hpu_ops.enable()`` calls it. ``HPU_OPS_TPC=0`` in the environment keeps the kernels off.
"""
import contextlib
import functools
import os
import pathlib
import time
import warnings

import torch

MAXN = 64  # MAXN of gj_solve and tri_solve (main)
SMALL_MAXN = 32  # MAXN of jacobi_svd, matrix_exp, lu_factor, lu_perm and lu_solve
MAXHH = 64
LARGE_MAXN = 512  # NMAX of gj_large_f32_gaudi2.c: gj_solve and gj_inv take 64 < n <= LARGE_MAXN through it
# The in-place kernel runs one matrix per TPC (24 on Gaudi2), so a small batch leaves the card idle: measured on
# Gaudi2 (bridge 1.21.4, eager, mean of synced calls), batch 1 n = 384 and batch 1 to 4 n = 512 were slower than
# Habana's CPU fallback, batch 1 n <= 300 and batch 32 n <= 256 not (see ``large_pays``).
LARGE_ANY_BATCH_N = 256  # up to here the kernel takes any batch
LARGE_MIN_BATCH = 8  # past it, at least this many matrices
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
  auto batch = a.size(0), n = a.size(1);
  return {habana::PartialOutputMetaData{a.scalar_type(), {batch, n, 64 * ((n + 63) / 64)}},
          habana::PartialOutputMetaData{a.scalar_type(), {batch, n}},
          habana::PartialOutputMetaData{a.scalar_type(), {batch, n}},
          habana::PartialOutputMetaData{a.scalar_type(), {batch, n}},
          habana::PartialOutputMetaData{a.scalar_type(), {batch, 2}}};
}
static bool registered = [] {
  habana::custom_op::registerUserCustomOp("linalg_tpc::gj_large", "gj_large_f32_gaudi2", large_meta, nullptr);
  habana::custom_op::registerUserCustomOp("linalg_tpc::jacobi_eigh", "jacobi_eigh_f32_gaudi2", eigh_meta, nullptr);
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
TORCH_LIBRARY(linalg_tpc, m) {
  m.def("gj_large(Tensor a) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");
  m.def("gj_solve(Tensor a, Tensor b) -> (Tensor, Tensor, Tensor)");
  m.def("gj_inv(Tensor a) -> (Tensor, Tensor, Tensor)");
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
}
TORCH_LIBRARY_IMPL(linalg_tpc, HPU, m) {
  m.impl("gj_large", gj_large);
  m.impl("gj_solve", gj_solve);
  m.impl("gj_inv", gj_inv);
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
}
TORCH_LIBRARY_IMPL(linalg_tpc, Meta, m) {
  m.impl("gj_large", [](at::Tensor a) {
    auto bt = a.size(0), n = a.size(1);
    auto v = a.new_empty({bt, n});
    return std::make_tuple(a.new_empty({bt, n, 64 * ((n + 63) / 64)}), v, a.new_empty({bt, n}), a.new_empty({bt, n}),
                           a.new_empty({bt, 2}));
  });
  m.impl("gj_solve", [](at::Tensor a, at::Tensor b) {
    return std::make_tuple(b.new_empty(b.sizes()), b.new_empty({a.size(0), a.size(1)}), b.new_empty({a.size(0), 2}));
  });
  m.impl("gj_inv", [](at::Tensor a) {
    return std::make_tuple(a.new_empty(a.sizes()), a.new_empty({a.size(0), a.size(1)}), a.new_empty({a.size(0), 2}));
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
  auto n = a.size(1);
  return std::vector<int64_t>{a.size(0), n, 64 * ((n + 63) / 64)};
};
static bool registered_large = [] {
  std::vector<habana::custom_op::InputDesc> in{{habana::custom_op::input_type::TENSOR, 0}};
  std::vector<habana::custom_op::OutputDesc> out{{0, c10::ScalarType::Float, rshape},
                                                 {1, c10::ScalarType::Float, dshape},
                                                 {2, c10::ScalarType::Float, dshape},
                                                 {3, c10::ScalarType::Float, dshape},
                                                 {4, c10::ScalarType::Float, sshape}};
  REGISTER_CUSTOM_OP_ATTRIBUTES("linalg_tpc::gj_large", "gj_large_f32_gaudi2", in, out, nullptr);
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
TORCH_LIBRARY(linalg_tpc, m) {
  m.def("gj_large(Tensor a) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");
  m.def("gj_solve(Tensor a, Tensor b) -> (Tensor, Tensor, Tensor)");
  m.def("gj_inv(Tensor a) -> (Tensor, Tensor, Tensor)");
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
}
TORCH_LIBRARY_IMPL(linalg_tpc, HPU, m) {
  m.impl("gj_large", gj_large);
  m.impl("gj_solve", gj_solve);
  m.impl("gj_inv", gj_inv);
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
    if A.dim() < 2 or n > limit or A.shape[-2] != n:
        raise ValueError(f"hpu_ops.tpc takes square A with n <= {limit}, got {tuple(A.shape)}")
    if any(t.dtype != torch.float32 for t in (A, *others)):
        # the kernels compute in float32: a float64 input would be downcast silently
        raise ValueError(f"hpu_ops.tpc takes float32 tensors, got {[str(t.dtype) for t in (A, *others)]}")


def large_pays(n, batch):
    """Whether the routing sends ``batch`` matrices of order ``n`` (``MAXN`` < n <= ``LARGE_MAXN``) to the in-place
    kernel: eager mode, not tracing under ``torch.compile``, and n <= ``LARGE_ANY_BATCH_N`` or at least
    ``LARGE_MIN_BATCH`` matrices. In lazy mode (and compiled) the call keeps main's route (Habana's fallback):
    measured on Gaudi2, bridge 1.21.4, EEGSPDNet's lazy step was 73.2 against 63.0 ms median with the kernel (its graph
    boundary, gathers and checks), and the compiled path was not measured faster."""
    # lazy_mode(), the one reading of the mode every route uses (the variable unset is eager on bridge 1.21.4,
    # measured in the pod; reading the variable here with another default declined default eager calls). Tracing
    # (make_fx, AOT autograd, where is_compiling() is False) is declined by the caller, which sees the tensors.
    eager = not lazy_mode() and not torch.compiler.is_compiling()
    return eager and n <= LARGE_MAXN and (n <= LARGE_ANY_BATCH_N or batch >= LARGE_MIN_BATCH)


def _parity(P):
    """``(-1)^inversions`` of the permutations ``P[b, n]`` (float, exact integers): the sign of the row order."""
    n = P.shape[-1]
    later = torch.ones(n, n, dtype=torch.bool, device=P.device).triu(1)
    inversions = ((P[..., :, None] > P[..., None, :]) & later).sum((-2, -1))
    return (1 - 2 * (inversions % 2)).to(P.dtype)


def _rows(X, index):
    """``X[..., index, :]`` for a float index ``[b, n]`` (exact integers): a gather of float data."""
    return X.gather(-2, index.long()[..., None].expand(*index.shape, X.shape[-1]))


def _large_raw(A, sign=True):
    """The kernel's ``(R, P, Q, D, S)`` for ``A[b, n, n]`` run on ``A^T`` (``R[P, :][:, Q] = A^-T``), ``S[:, 0]`` set
    to the sign of the pivot order (0 when not ``sign``: solve and inv do not read it, and its [b, n, n] comparison
    costs eager launches) and a matrix with a NaN given NaN pivots (LAPACK's results are NaN there; the kernel's need
    not be). ``A^T``, as torch: its CPU solve, inv and det factor the row-major A as LAPACK's column-major A^T, so
    the pivot order is LAPACK's on the matrix torch hands it. On ``A`` the kernel took the other order, and partial
    pivoting's 2^(n-1) growth on Wilkinson's matrix (``W = I - tril(1, -1)``, last column 1; condition 58 at n =
    130) overflowed where torch's factors stay bounded (Codex review #1); torch overflows on ``W^T``, and so do we."""
    R, P, Q, D, S = _call(_large_op(), A.mT.contiguous())
    nan = A.isnan().any(-1).any(-1)
    D = D.masked_fill(nan[..., None], float("nan"))
    if sign:
        S = torch.stack([_parity(P), S[..., 1]], -1)
    return R, P, Q, D, S, nan


def large_factors(A):
    """``(D, S)`` of the in-place kernel for ``A[..., n, n]`` (64 < n <= ``LARGE_MAXN``), for ``det`` and
    ``slogdet``: the pivots and (sign, info), without forming the inverse."""
    n = A.shape[-1]
    _check(A, n, limit=LARGE_MAXN)
    batch = A.shape[:-2]
    _, _, _, D, S, _ = _large_raw(_flat(A, batch, n, n))
    return _unflatten(D, batch, n), _unflatten(S, batch, 2)


def _large(A, B=None, sign=True):
    """``(X, D, S)`` of the in-place Gauss-Jordan kernel for ``A[b, n, n]`` (64 < n <= ``LARGE_MAXN``): ``X =
    A^-1`` (``B`` None) or ``A^-1 B``, ``S[:, 0]`` the sign of the pivot order. The kernel runs on ``A^T`` (see
    ``_large_raw``) and divides each row by its pivot at the end: ``A^-T = R[P, :][:, Q]``, so ``A^-1 = R^T[Q, :][:,
    P]`` and ``A^-1 B = (R^T @ B[Q])[Q]`` (with ``Q = P^-1``); a zero pivot gives a zero column (info is set)."""
    n = A.shape[-1]
    R, P, Q, D, S, nan = _large_raw(A, sign)
    Rt = R[..., :n].mT
    # the kernel divides each row by its pivot itself (a torch division by a pivot of 1e38 zeroed the row in lazy mode:
    # the card's division multiplies by a reciprocal, subnormal hence flushed, and a 2^-64 prescale was folded away)
    Y = _rows(Rt, Q) if B is None else _rows(Rt @ _rows(B, Q), Q)
    if B is None:
        # no Newton step: on the kernel's float32 replica it gained little at condition <= 1e4 and lost past 1e6
        # (n = 220, condition 1e7: 6.8e-2 against 4.5e-2 without, LAPACK 4.7e-2)
        X = Y.gather(-1, P.long()[..., None, :].expand(*P.shape[:-1], n, n))
    else:
        X = Y
    X = X.masked_fill(nan[..., None, None], float("nan"))
    return X, D, S


def gj_solve(A, B):
    """``(X, D, S)`` for ``A[..., n, n]`` and ``B[..., n, m]`` (float32, n <= ``MAXN``) from the TPC kernel: ``X = A^-1
    B``, ``D`` the pivots, ``S[..., 0]`` the sign of the permutation and ``S[..., 1]`` the info (1 + the first
    zero pivot in the kernel's elimination order, else 0; it can differ from LAPACK's);
    shaped ``[..., n, m]``, ``[..., n]``, ``[..., 2]``."""
    n, m = A.shape[-1], B.shape[-1]
    _check(A, n, B, limit=LARGE_MAXN)
    if B.dim() < 2 or B.shape[-2] != n:
        raise ValueError(f"hpu_ops.tpc.gj_solve takes B with n rows, got {tuple(A.shape)} and {tuple(B.shape)}")
    batch = torch.broadcast_shapes(A.shape[:-2], B.shape[:-2])
    solve = _chunked if n <= MAXN else _large
    X, D, S = solve(_flat(A, batch, n, n), _flat(B, batch, n, m))
    return _unflatten(X, batch, n, m), _unflatten(D, batch, n), _unflatten(S, batch, 2)


def gj_inv(A, sign=True):
    """``(A^-1, D, S)`` for ``A[..., n, n]``: ``gj_solve`` against the identity, which the kernel builds itself.
    Past ``MAXN``, ``sign=False`` leaves ``S[..., 0]`` 0 (not computed)."""
    n = A.shape[-1]
    _check(A, n, limit=LARGE_MAXN)
    batch = A.shape[:-2]
    A = _flat(A, batch, n, n)
    X, D, S = _call(_inv_op(), A) if n <= MAXN else _large(A, sign=sign)
    return _unflatten(X, batch, n, n), _unflatten(D, batch, n), _unflatten(S, batch, 2)


def tri_solve(A, B, upper=False, left=True, unitriangular=False):
    """Solve ``A X = B`` (``X A = B`` if not ``left``) for triangular ``A[..., n, n]`` (n <= ``MAXN``)."""
    n = A.shape[-1]
    _check(A, n, B)
    if B.dim() < 2 or B.shape[-2 if left else -1] != n:
        raise ValueError(f"hpu_ops.tpc.tri_solve takes compatible B for left={left}, "
                         f"got {tuple(A.shape)} and {tuple(B.shape)}")
    A_eff = A if left else A.mT
    B_eff = B if left else B.mT
    upper_eff = upper if left else not upper
    batch = torch.broadcast_shapes(A.shape[:-2], B.shape[:-2])
    m = B_eff.shape[-1]
    X = _tri_chunked(_tri_op(upper_eff, unitriangular), _flat(A_eff, batch, n, n), _flat(B_eff, batch, n, m))
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
    order = key.sort(dim=-1, descending=descending)[1]
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


def svd_reference(A, compute_uv=True, full_matrices=False):
    """``jacobi_svd`` / ``jacobi_svdvals`` in plain torch, matching the kernel contract: the rows of A orthogonalized
    when m < n or A is square and contiguous (else the rows of A^T), 1/max|a| scaling (not of a zero matrix), the
    kernel's round-robin rotations and skip rules (gamma = 0, gamma^2 <= tol alpha beta, r^2 < 1e-34), singular
    values of squared row norms below 1e-36 set to 0, MGS completion for zero/degenerate rows, optional full
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
                alpha = (wp * wp).sum(-1)
                beta = (wq * wq).sum(-1)
                gamma = (wp * wq).sum(-1)
                d = beta - alpha
                g2 = 2.0 * gamma
                r = torch.hypot(d, g2)
                skip = (gamma == 0) | (r * r < 1e-34) | (gamma * gamma <= tol_scale * alpha * beta)
                sr = torch.where(skip, 1.0, r)
                c2 = 0.5 * (1.0 + d.abs() / sr)
                c = torch.where(skip, 1.0, c2.sqrt())
                s = torch.where(skip, 0.0, torch.where(d < 0, -1.0, 1.0) * 0.5 * g2 / (sr * torch.where(skip, 1.0, c)))
                c_, s_ = c.unsqueeze(-1), s.unsqueeze(-1)
                W[:, ps], W[:, qs] = c_ * wp - s_ * wq, s_ * wp + c_ * wq
                if compute_uv:
                    vp, vq = V[:, ps], V[:, qs]
                    V[:, ps], V[:, qs] = c_ * vp - s_ * vq, s_ * vp + c_ * vq

    sq = (W * W).sum(-1)
    norm = sq.sqrt()
    S = torch.where(sq < 1e-36, 0.0, norm * safe_sigma.squeeze(-1))
    if not compute_uv:
        return A.new_empty(0), S.reshape(*batch, k).sort(-1, descending=True).values, A.new_empty(0)

    thresh = torch.clamp(1e-10 * sq.amax(-1, keepdim=True), min=1e-36)
    U_t = torch.where(sq.unsqueeze(-1) < 1e-36, 0.0, W / torch.where(norm.unsqueeze(-1) > 0, norm.unsqueeze(-1), 1.0))
    bad_mask = sq < thresh
    if bad_mask.any():
        for bi in range(b):
            if not bad_mask[bi].any():
                continue
            done = [j for j in range(k) if not bad_mask[bi, j]]
            for j in range(k):
                if not bad_mask[bi, j]:
                    continue
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
