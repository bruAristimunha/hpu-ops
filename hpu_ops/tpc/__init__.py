"""The Gaudi2 TPC Gauss-Jordan kernel behind ``linalg.solve``, ``inv``, ``det``, ``slogdet``, ``solve_triangular`` and
``cholesky_solve``: ``torch.ops.linalg_tpc.gj_solve(A[b, n, n], B[b, n, m]) -> (X[b, n, m], D[b, n], S[b, 1])`` for
float32, n and m <= 32. One matrix per index-space member, rows in local memory, partial pivoting. ``D`` holds the
pivots (U's diagonal), ``S`` the sign (-1)^swaps of the row permutation.

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

import torch

MAXN = 32
LIB = pathlib.Path(__file__).with_name("liblinalg_tpc.so")
DEFAULT_KERNELS = "/usr/lib/habanalabs/libtpc_kernels.so"
LATE_WARNING = ("hpu_ops: the HPU was used before hpu_ops.enable(); TPC linalg kernels are off for this process "
                "(call enable() before the first HPU op)")
_INSTALLED = False
_LATE_WARNED = False

# The registration strings are those measured on Gaudi2. CPP: the eager and compile API (bridge 2), CPP_LAZY: the
# legacy API of lazy mode.
CPP = r"""
#include <torch/extension.h>
#include "hpu_custom_op_pt2.h"

static habana::PartialOutputMetaDataVector solve_meta(const at::Stack& inputs) {
  auto a = inputs[0].toTensor(), b = inputs[1].toTensor();
  auto batch = a.size(0), n = a.size(1);
  return {habana::PartialOutputMetaData{b.scalar_type(), b.sizes().vec()},
          habana::PartialOutputMetaData{b.scalar_type(), {batch, n}},
          habana::PartialOutputMetaData{b.scalar_type(), {batch, 1}}};
}
static bool registered = [] {
  habana::custom_op::registerUserCustomOp("linalg_tpc::gj_solve", "gj_solve_f32_gaudi2", solve_meta, nullptr);
  return true;
}();

std::tuple<at::Tensor, at::Tensor, at::Tensor> gj_solve(at::Tensor a, at::Tensor b) {
  std::vector<c10::IValue> inputs{a, b};
  auto desc = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor("linalg_tpc::gj_solve");
  auto out = desc.execute(inputs);
  return {out[0], out[1], out[2]};
}
TORCH_LIBRARY(linalg_tpc, m) { m.def("gj_solve(Tensor a, Tensor b) -> (Tensor, Tensor, Tensor)"); }
TORCH_LIBRARY_IMPL(linalg_tpc, HPU, m) { m.impl("gj_solve", gj_solve); }
TORCH_LIBRARY_IMPL(linalg_tpc, Meta, m) { m.impl("gj_solve", [](at::Tensor a, at::Tensor b) {
  return std::make_tuple(b.new_empty(b.sizes()), b.new_empty({a.size(0), a.size(1)}), b.new_empty({a.size(0), 1})); }); }
"""
CPP_LAZY = r"""
#include <torch/extension.h>
#include "hpu_custom_op.h"

static bool registered = [] {
  std::vector<habana::custom_op::InputDesc> in{{habana::custom_op::input_type::TENSOR, 0},
                                               {habana::custom_op::input_type::TENSOR, 1}};
  auto shape = [](const at::Stack& s) { return s[1].toTensor().sizes().vec(); };
  auto dshape = [](const at::Stack& s) { auto a = s[0].toTensor(); return std::vector<int64_t>{a.size(0), a.size(1)}; };
  auto sshape = [](const at::Stack& s) { return std::vector<int64_t>{s[0].toTensor().size(0), 1}; };
  std::vector<habana::custom_op::OutputDesc> out{{0, c10::ScalarType::Float, shape}, {1, c10::ScalarType::Float, dshape},
                                                 {2, c10::ScalarType::Float, sshape}};
  REGISTER_CUSTOM_OP_ATTRIBUTES("linalg_tpc::gj_solve", "gj_solve_f32_gaudi2", in, out, nullptr);
  return true;
}();

std::tuple<at::Tensor, at::Tensor, at::Tensor> gj_solve(at::Tensor a, at::Tensor b) {
  auto op = habana::custom_op::HabanaCustomOpDescriptor::getCustomOpDescriptor("linalg_tpc::gj_solve");
  std::vector<c10::IValue> inputs{a, b};
  auto out = op.execute(inputs);
  return {out[0], out[1], out[2]};
}
TORCH_LIBRARY(linalg_tpc, m) { m.def("gj_solve(Tensor a, Tensor b) -> (Tensor, Tensor, Tensor)"); }
TORCH_LIBRARY_IMPL(linalg_tpc, HPU, m) { m.impl("gj_solve", gj_solve); }
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
    """Put ``LIB`` first on ``GC_KERNEL_PATH`` (default ``DEFAULT_KERNELS`` when unset, never duplicated) if the HPU is
    not initialized yet and return True; otherwise warn once and return False, as Habana reads the path when the first
    graph compiles: set later, the kernel's graph fails to compile."""
    global _INSTALLED, _LATE_WARNED
    if _initialized():
        _INSTALLED = False
        if not _LATE_WARNED:
            _LATE_WARNED = True
            warnings.warn(LATE_WARNING, stacklevel=2)
        return False
    entries = [e for e in os.environ.get("GC_KERNEL_PATH", DEFAULT_KERNELS).split(":") if e]
    if str(LIB) not in entries:
        os.environ["GC_KERNEL_PATH"] = ":".join([str(LIB), *entries])
    _INSTALLED = True
    return True


def available():
    """True when ``install_kernel_path`` succeeded in this process."""
    return _INSTALLED


@functools.cache
def _op():
    """``torch.ops.linalg_tpc.gj_solve``, compiled and loaded on the first call: the eager and compile API when
    ``PT_HPU_LAZY_MODE`` is "0", the legacy API otherwise."""
    from habana_frameworks.torch.utils.lib_utils import get_include_dir, get_lib_dir
    from torch.utils.cpp_extension import load_inline

    lazy = os.environ.get("PT_HPU_LAZY_MODE", "1") != "0"
    load_inline(name="linalg_tpc_plugin_lazy" if lazy else "linalg_tpc_plugin", cpp_sources=[CPP_LAZY if lazy else CPP],
                verbose=False, is_python_module=False, extra_include_paths=[get_include_dir(), "/usr/include/habanalabs"],
                extra_ldflags=[f"-L{get_lib_dir()}", "-lhabana_pytorch_plugin" if lazy else "-lhabana_pytorch2_plugin"])
    return torch.ops.linalg_tpc.gj_solve


def _chunked(A, B):
    """The kernel on ``A[b, n, n]`` and ``B[b, n, m]`` for any ``m``: columns of ``B`` in chunks of at most ``MAXN``,
    ``X`` concatenated, ``D`` and ``S`` from the first chunk (they depend on ``A`` alone)."""
    op = _op()
    parts = [op(A, chunk) for chunk in B.split(MAXN, -1)]
    X = parts[0][0] if len(parts) == 1 else torch.cat([p[0] for p in parts], -1)
    return X, parts[0][1], parts[0][2]


def gj_solve(A, B):
    """``(X, D, S)`` for ``A[..., n, n]`` and ``B[..., n, m]`` (float32, n <= ``MAXN``) from the TPC kernel: ``X = A^-1
    B``, ``D`` the pivots, ``S`` the sign of the permutation, shaped ``[..., n, m]``, ``[..., n]``, ``[..., 1]``."""
    n, m = A.shape[-1], B.shape[-1]
    if n > MAXN or A.shape[-2] != n or B.shape[-2] != n:
        raise ValueError(f"hpu_ops.tpc.gj_solve takes square A with n <= {MAXN} and B with n rows, got {tuple(A.shape)}"
                         f" and {tuple(B.shape)}")
    batch = torch.broadcast_shapes(A.shape[:-2], B.shape[:-2])
    flat = lambda x, *shape: x.expand(*batch, *shape).reshape(-1, *shape).to(torch.float32).contiguous()  # noqa: E731
    X, D, S = _chunked(flat(A, n, n), flat(B, n, m))
    return X.reshape(*batch, n, m), D.reshape(*batch, n), S.reshape(*batch, 1)


def gj_solve_reference(A, B):
    """``gj_solve`` in plain torch, the contract of the kernel: ``lu`` and ``lu_solve`` of ``hpu_ops.linalg``, ``D``
    the diagonal of U, ``S`` the parity of the pivots. For the CPU tests."""
    from .. import linalg

    LU, pivots = linalg.lu(A)
    X = linalg.lu_solve(LU, pivots, B)
    D = torch.diagonal(LU, dim1=-2, dim2=-1)
    return X, D, linalg._parity(pivots).to(A.dtype)[..., None]
