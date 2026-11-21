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
  ``MAXN`` to it: ``inv`` is two gathers of ``R``, ``solve`` the matmul ``(R @ B[P])[P]``, the sign of ``det`` the
  parity of ``P``.
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
import functools
import os
import pathlib
import warnings

import torch

MAXN = 64
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
    global _INSTALLED, _LATE_WARNED, _MISSING_WARNED
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
        _INSTALLED = False
        if not _LATE_WARNED:
            _LATE_WARNED = True
            warnings.warn(LATE_WARNING, stacklevel=2)
        return False
    entries = [e for e in (os.environ.get("GC_KERNEL_PATH") or DEFAULT_KERNELS).split(":") if e]
    if str(LIB) not in entries:
        os.environ["GC_KERNEL_PATH"] = ":".join([str(LIB), *entries])
    _INSTALLED = True
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


@functools.cache
def _load():
    """Compile and load the C++ ops: the eager and compile API when ``PT_HPU_LAZY_MODE`` is "0", the legacy API of
    lazy mode otherwise."""
    from habana_frameworks.torch.utils.lib_utils import get_include_dir, get_lib_dir
    from torch.utils.cpp_extension import load_inline

    lazy = os.environ.get("PT_HPU_LAZY_MODE", "1") != "0"
    load_inline(name="linalg_tpc_plugin_lazy" if lazy else "linalg_tpc_plugin", cpp_sources=[CPP_LAZY if lazy else CPP],
                verbose=False, is_python_module=False,
                extra_include_paths=[get_include_dir(), "/usr/include/habanalabs"],
                extra_ldflags=[f"-L{get_lib_dir()}", "-lhabana_pytorch_plugin" if lazy else "-lhabana_pytorch2_plugin"])


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
    """Lazy mode, and not tracing under ``torch.compile``."""
    return os.environ.get("PT_HPU_LAZY_MODE", "1") != "0" and not torch.compiler.is_compiling()


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
    kernel: n <= ``LARGE_ANY_BATCH_N``, or at least ``LARGE_MIN_BATCH`` matrices."""
    return n <= LARGE_MAXN and (n <= LARGE_ANY_BATCH_N or batch >= LARGE_MIN_BATCH)


def _parity(P):
    """``(-1)^inversions`` of the permutations ``P[b, n]`` (float, exact integers): the sign of the row order."""
    n = P.shape[-1]
    later = torch.ones(n, n, dtype=torch.bool, device=P.device).triu(1)
    inversions = ((P[..., :, None] > P[..., None, :]) & later).sum((-2, -1))
    return (1 - 2 * (inversions % 2)).to(P.dtype)


def _rows(X, index):
    """``X[..., index, :]`` for a float index ``[b, n]`` (exact integers): a gather of float data."""
    return X.gather(-2, index.long()[..., None].expand(*index.shape, X.shape[-1]))


def _large(A, B=None):
    """``(X, D, S)`` of the in-place Gauss-Jordan kernel for ``A[b, n, n]`` (64 < n <= ``LARGE_MAXN``): ``X =
    A^-1`` (``B`` None; refined by one Newton step) or ``A^-1 B``, ``S[:, 0]`` the sign of the pivot order. The kernel leaves ``R`` with ``A^-1 =
    R[P, :][:, Q]``, so ``A^-1 B = (R @ B[Q^-1 order])[P]``: with ``Q = P^-1``, ``R[:, Q] @ B = R @ B[P]``."""
    n = A.shape[-1]
    R, P, Q, D, S = _call(_large_op(), A)
    R = R[..., :n]
    if B is None:
        X = _rows(R, P).gather(-1, Q.long()[..., None, :].expand(*Q.shape[:-1], n, n))
        # One Newton step on the MME, X + X (I - A X): Gauss-Jordan's inverse is about 2x LAPACK's error (the kernel
        # equals the same algorithm on the CPU); measured on Gaudi2 at n = 220, 1.7e-6 to 8.7e-7 (CPU float32 7.0e-7)
        # (a non-finite correction, from an Inf in A, is dropped: the kernel's result keeps LAPACK's pattern there)
        C = X @ (torch.eye(n, dtype=X.dtype, device=X.device) - A @ X)
        X = X + torch.where(torch.isfinite(C), C, 0.0)
    else:
        X = _rows(R @ _rows(B, P), P)
    # A NaN in A: LAPACK's results are NaN; the kernel's are not always (measured on Gaudi2: a NaN at (0, 0) of a
    # 70 x 70 matrix left R finite), so a matrix with a NaN gets NaN outputs here
    nan = A.isnan().any(-1).any(-1)
    X = X.masked_fill(nan[..., None, None], float("nan"))
    D = D.masked_fill(nan[..., None], float("nan"))
    return X, D, torch.stack([_parity(P), S[..., 1]], -1)


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


def gj_inv(A):
    """``(A^-1, D, S)`` for ``A[..., n, n]``: ``gj_solve`` against the identity, which the kernel builds itself."""
    n = A.shape[-1]
    _check(A, n, limit=LARGE_MAXN)
    batch = A.shape[:-2]
    A = _flat(A, batch, n, n)
    X, D, S = _call(_inv_op(), A) if n <= MAXN else _large(A)
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
    w, order = W.sort(-1)
    V = Vt.gather(-2, order[..., :, None].expand_as(Vt)).mT
    return w.reshape(*batch, n), V.reshape(*batch, n, n)


def eigh_reference(A, upper):
    """``eigh`` through numpy's LAPACK, the contract of the kernel path (torch's own ``linalg.eigh`` is what the CPU
    tests override). Detached: no autograd through the kernel. For the CPU tests."""
    import numpy as np  # only here: ``import hpu_ops`` does not need numpy

    w, V = np.linalg.eigh(A.detach().cpu().numpy(), UPLO="U" if upper else "L")
    return torch.from_numpy(w).to(A.device), torch.from_numpy(V).to(A.device)


def gj_large_reference(A):
    """``torch.ops.linalg_tpc.gj_large`` in plain torch, the contract of the in-place kernel, for ``A[b, n, n]``:
    ``(R, P, Q, D, S)`` with ``R`` padded to whole 64-lane vectors and ``A^-1 = R[P, :][:, Q]``. Implicit partial
    pivoting (the first unused row of largest |column k|), a zero pivot zeroes its row. For the CPU tests."""
    A = A.detach()
    b, n = A.shape[0], A.shape[-1]
    rows = torch.arange(b)
    R = A.clone()
    used = torch.zeros(b, n, dtype=torch.bool)
    P, D, info = A.new_zeros(b, n), A.new_zeros(b, n), A.new_zeros(b)
    for k in range(n):
        p = R[:, :, k].abs().masked_fill(used, -1.0).argmax(-1)
        pivot = R[rows, p, k]
        P[:, k], D[:, k] = p.to(A.dtype), pivot
        info = torch.where((pivot == 0) & (info == 0), float(k + 1), info)
        used[rows, p] = True
        prow = R[rows, p].clone()
        prow[:, k] = 1.0
        prow = prow * torch.where(pivot == 0, 0.0, 1.0 / torch.where(pivot == 0, 1.0, pivot))[:, None]
        f = R[:, :, k].clone()
        R[:, :, k] = 0.0
        R = R - f[:, :, None] * prow[:, None, :]
        R[rows, p] = prow
    Q = P.argsort(-1).to(A.dtype)
    R = torch.nn.functional.pad(R, (0, 64 * ((n + 63) // 64) - n))
    return R, P, Q, D, torch.stack([torch.zeros_like(info), info], -1)


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
