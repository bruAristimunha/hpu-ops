"""The Gaudi2 TPC kernels behind ``linalg.solve``, ``inv``, ``det``, ``slogdet``, ``solve_triangular`` and
``cholesky_solve``:
- ``torch.ops.linalg_tpc.gj_solve(A[b, n, n], B[b, n, m]) -> (X[b, n, m], D[b, n], S[b, 2])`` and
  ``gj_inv(A) -> (X, D, S)`` (B = identity, built in the kernel) for float32, n <= 64 (B columns chunked by 64). One
  matrix per index-space member, rows in local memory, partial pivoting. ``D`` holds the pivots (U's diagonal),
  ``S[:, 0]`` the sign (-1)^swaps of the row permutation, ``S[:, 1]`` the info: the index follows the kernel's own
  elimination order, can differ from LAPACK's, and is nonzero exactly when a pivot is exactly zero.
- ``torch.ops.linalg_tpc.tri_solve_{lower,upper}[_unit](A[b, n, n], B[b, n, m]) -> X[b, n, m]``: dedicated
  non-pivoting forward/back substitution for float32, n <= 64 (B columns chunked by 64; ``left=False`` via ``.mT`` and
  flipped ``upper``); opposite triangle (and diagonal for ``_unit``) is never read.

``jacobi_eigh(A[b, n, n]) -> (W[b, n], Vt[b, n, n])`` is a one-sided Jacobi eigensolver of the same shape (float32,
n <= ``EIGH_MAXN``): ``W`` unsorted, the rows of ``Vt`` the eigenvectors; ``eigh`` sorts them. Eager, batch 32: n = 22
0.30 ms against 3.7 ms for the CPU fallback, n = 32 0.54 ms against 5.0 ms, batch 1024 3.1 ms against 52 ms; accuracy
within 3.2x of CPU float32.

The torch-op composites of ``hpu_ops.linalg`` are correct but 3-25x slower than Habana's CPU fallback on Gaudi2 (one
launch per op). The kernel, eager, batch 32: n = 22 solve 0.20 ms against 4.7 ms for the fallback, n = 32 0.28 ms,
batch 1024 1.9 ms against 20 ms; the pivots equal LAPACK's U diagonal to 6e-6, the determinant to 9e-6.

Habana reads ``GC_KERNEL_PATH`` when the first graph compiles, so ``install_kernel_path`` must run before the first HPU
op of the process; ``hpu_ops.enable()`` calls it.
"""
import functools
import os
import pathlib
import warnings

import numpy as np
import torch

MAXN = 64
EIGH_MAXN = 32  # MAXN of jacobi_eigh_f32_gaudi2.c
LIB = pathlib.Path(__file__).with_name("liblinalg_tpc.so")
DEFAULT_KERNELS = "/usr/lib/habanalabs/libtpc_kernels.so"
LATE_WARNING = ("hpu_ops: the HPU was used before hpu_ops.enable(); TPC linalg kernels are off for this process "
                "(call enable() before the first HPU op)")
BUILD_WARNING = ("hpu_ops: building the TPC linalg extension failed ({error}); solve, inv, det, slogdet and the "
                 "triangular solves keep Habana's CPU fallback.")
_INSTALLED = False
_LATE_WARNED = False

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
static bool registered = [] {
  habana::custom_op::registerUserCustomOp("linalg_tpc::jacobi_eigh", "jacobi_eigh_f32_gaudi2", eigh_meta, nullptr);
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
TORCH_LIBRARY(linalg_tpc, m) {
  m.def("gj_solve(Tensor a, Tensor b) -> (Tensor, Tensor, Tensor)");
  m.def("gj_inv(Tensor a) -> (Tensor, Tensor, Tensor)");
  m.def("jacobi_eigh(Tensor a) -> (Tensor, Tensor)");
  m.def("tri_solve_lower(Tensor a, Tensor b) -> Tensor");
  m.def("tri_solve_upper(Tensor a, Tensor b) -> Tensor");
  m.def("tri_solve_lower_unit(Tensor a, Tensor b) -> Tensor");
  m.def("tri_solve_upper_unit(Tensor a, Tensor b) -> Tensor");
}
TORCH_LIBRARY_IMPL(linalg_tpc, HPU, m) {
  m.impl("gj_solve", gj_solve);
  m.impl("gj_inv", gj_inv);
  m.impl("jacobi_eigh", jacobi_eigh);
  m.impl("tri_solve_lower", tri_solve_lower);
  m.impl("tri_solve_upper", tri_solve_upper);
  m.impl("tri_solve_lower_unit", tri_solve_lower_unit);
  m.impl("tri_solve_upper_unit", tri_solve_upper_unit);
}
TORCH_LIBRARY_IMPL(linalg_tpc, Meta, m) {
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
TORCH_LIBRARY(linalg_tpc, m) {
  m.def("gj_solve(Tensor a, Tensor b) -> (Tensor, Tensor, Tensor)");
  m.def("gj_inv(Tensor a) -> (Tensor, Tensor, Tensor)");
  m.def("jacobi_eigh(Tensor a) -> (Tensor, Tensor)");
  m.def("tri_solve_lower(Tensor a, Tensor b) -> Tensor");
  m.def("tri_solve_upper(Tensor a, Tensor b) -> Tensor");
  m.def("tri_solve_lower_unit(Tensor a, Tensor b) -> Tensor");
  m.def("tri_solve_upper_unit(Tensor a, Tensor b) -> Tensor");
}
TORCH_LIBRARY_IMPL(linalg_tpc, HPU, m) {
  m.impl("gj_solve", gj_solve);
  m.impl("gj_inv", gj_inv);
  m.impl("jacobi_eigh", jacobi_eigh);
  m.impl("tri_solve_lower", tri_solve_lower);
  m.impl("tri_solve_upper", tri_solve_upper);
  m.impl("tri_solve_lower_unit", tri_solve_lower_unit);
  m.impl("tri_solve_upper_unit", tri_solve_upper_unit);
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
    when the first graph compiles: set later, the kernel's graph fails to compile."""
    global _INSTALLED, _LATE_WARNED
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


def available():
    """True when ``install_kernel_path`` succeeded in this process and the library is there."""
    return _INSTALLED and LIB.exists()


def ready():
    """``available()`` and the C++ extension built. The build happens on the first call; if it fails, one warning,
    then False for the rest of the process (the ops keep Habana's CPU fallback, with no retry mid-forward)."""
    return available() and _built()


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


def _eigh_op():
    """``torch.ops.linalg_tpc.jacobi_eigh``, built on first use."""
    _load()
    return torch.ops.linalg_tpc.jacobi_eigh


def _tri_op(upper, unitriangular):
    """``torch.ops.linalg_tpc.tri_solve_*``, built on first use."""
    _load()
    name = "tri_solve" + ("_upper" if upper else "_lower") + ("_unit" if unitriangular else "")
    return getattr(torch.ops.linalg_tpc, name)


# Lazy mode on bridge 1.21.4 can drop the unused outputs of a multi-output custom op (the callers use some of X, D and
# S), and the next graph then fails at mark_step with "ValidateSyncInputTensors tensor_data is empty". A mark_step
# before and after each call keeps the op alone in its graph. The cost is two graph boundaries per call; eager mode
# needs none.
def _lazy():
    return os.environ.get("PT_HPU_LAZY_MODE", "1") != "0" and not torch.compiler.is_compiling()


def _flush():
    """``mark_step`` in lazy mode; nothing in eager mode, while tracing under ``torch.compile``, or without Habana."""
    if not _lazy():
        return
    try:
        import habana_frameworks.torch.core as htcore
    except ImportError:
        return
    htcore.mark_step()


def _call(op, *args):
    """``op(*args)`` as a graph of its own in lazy mode, wrapped in an ATen node so ``.cpu()`` materializes it."""
    _flush()
    out = op(*args)
    _flush()
    if _lazy():
        if isinstance(out, tuple):
            return tuple(t.clone() if isinstance(t, torch.Tensor) else t for t in out)
        return out.clone() if isinstance(out, torch.Tensor) else out
    return out


def _fresh(t):
    """Detaches view storage so even size-1 ``.mT`` / slice views never alias graph inputs in lazy mode."""
    return t.clone() if t._base is not None else t


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


def _check(A, n):
    if A.dim() < 2 or n > MAXN or A.shape[-2] != n:
        raise ValueError(f"hpu_ops.tpc takes square A with n <= {MAXN}, got {tuple(A.shape)}")


def gj_solve(A, B):
    """``(X, D, S)`` for ``A[..., n, n]`` and ``B[..., n, m]`` (float32, n <= ``MAXN``) from the TPC kernel: ``X = A^-1
    B``, ``D`` the pivots, ``S[..., 0]`` the sign of the permutation and ``S[..., 1]`` the info (1 + the first
    zero pivot in the kernel's elimination order, else 0; it can differ from LAPACK's);
    shaped ``[..., n, m]``, ``[..., n]``, ``[..., 2]``."""
    n, m = A.shape[-1], B.shape[-1]
    _check(A, n)
    if B.dim() < 2 or B.shape[-2] != n:
        raise ValueError(f"hpu_ops.tpc.gj_solve takes B with n rows, got {tuple(A.shape)} and {tuple(B.shape)}")
    batch = torch.broadcast_shapes(A.shape[:-2], B.shape[:-2])
    X, D, S = _chunked(_flat(A, batch, n, n), _flat(B, batch, n, m))
    return _unflatten(X, batch, n, m), _unflatten(D, batch, n), _unflatten(S, batch, 2)


def gj_inv(A):
    """``(A^-1, D, S)`` for ``A[..., n, n]``: ``gj_solve`` against the identity, which the kernel builds itself."""
    n = A.shape[-1]
    _check(A, n)
    batch = A.shape[:-2]
    X, D, S = _call(_inv_op(), _flat(A, batch, n, n))
    return _unflatten(X, batch, n, n), _unflatten(D, batch, n), _unflatten(S, batch, 2)


def tri_solve(A, B, upper=False, left=True, unitriangular=False):
    """Solve ``A X = B`` (``X A = B`` if not ``left``) for triangular ``A[..., n, n]`` (n <= ``MAXN``)."""
    n = A.shape[-1]
    _check(A, n)
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
        for j in (range(i + 1, n) if upper else range(i)):
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
    # a fresh tensor, never a view of the caller's input: views break lazy mode
    S = torch.triu(A) + torch.triu(A, 1).mT if upper else torch.tril(A) + torch.tril(A, -1).mT
    W, Vt = _call(_eigh_op(), S.reshape(-1, n, n).to(torch.float32).contiguous())
    w, order = W.sort(-1)
    V = Vt.gather(-2, order[..., :, None].expand_as(Vt)).mT
    return w.reshape(*batch, n), V.reshape(*batch, n, n)


def eigh_reference(A, upper):
    """``eigh`` through numpy's LAPACK, the contract of the kernel path (torch's own ``linalg.eigh`` is what the CPU
    tests override). Detached: no autograd through the kernel. For the CPU tests."""
    w, V = np.linalg.eigh(A.detach().cpu().numpy(), UPLO="U" if upper else "L")
    return torch.from_numpy(w).to(A.device), torch.from_numpy(V).to(A.device)


def gj_solve_reference(A, B):
    """``gj_solve`` in plain torch, the contract of the kernel: ``lu`` and ``lu_solve`` of ``hpu_ops.linalg``, ``D``
    the diagonal of U (a zero pivot leaves later pivots finite, as in the kernel), ``S[..., 0]`` the parity of the
    pivots and ``S[..., 1]`` the info. For the CPU tests."""
    from .. import linalg

    LU, pivots = linalg.lu(A)
    X = linalg.lu_solve(LU, pivots, B)
    D = torch.diagonal(LU, dim1=-2, dim2=-1)
    S = torch.stack([linalg._parity(pivots).to(A.dtype), linalg.lu_info_of_diagonal(D).to(A.dtype)], -1)
    return X, D, S
