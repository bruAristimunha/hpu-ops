"""CPU tests for the linalg/TPC findings of the max-effort code review of main (code-review-hpu-ops.txt): 1 (autocast),
4 (build lock), 8 (matrix_exp float32 accuracy), 10 (outputs that are inputs or views), 11 (the kernel's ASO barrier),
13 (NaN through the eigh/SVD sort) and the cut items (traced det of a singular matrix on LAPACK, compiled gels on a
rank-deficient input, make_fx of lstsq, build.sh's copy over a loaded library). Each uses the review's own input; every
test failed on main (bda890b) and passes with the fix. Kernel changes are checked on the kernels' replicas
(tests/kernel_replica.py) and on the card (repro/probe_code_review_linalg.py)."""
import contextlib
import os
import pathlib
import re
import subprocess
import sys
import textwrap

import numpy as np
import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

import hpu_ops
import kernel_replica as kr
from test_cpu import compiled, cpu_as_hpu
from hpu_ops import linalg, tpc

ROOT = pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def eager_tpc(monkeypatch):
    monkeypatch.setenv("PT_HPU_LAZY_MODE", "0")


def rel(x, ref):
    x, ref = torch.as_tensor(x).double(), torch.as_tensor(ref).double()
    return ((x - ref).abs().max() / ref.abs().max()).item()


# ---- 1: the Functions' inner matmuls ran under the caller's autocast


@contextlib.contextmanager
def hpu_autocast_table(*ops):
    """Habana's AutocastHPU falls through for every linalg op (no linalg op is in its lists or in hpu-ops' CUDA files);
    the CPU has AutocastCPU fp32 kernels for a few (linalg_solve, linalg_lstsq), replaced here by fallthroughs."""
    lib = torch.library.Library("aten", "IMPL")
    for op in ops:
        lib.impl(op, torch.library.fallthrough_kernel, "AutocastCPU")
    try:
        yield
    finally:
        lib._destroy()


def tri_like_hpu(A, B, upper, left, unitriangular):
    tpc._check(A, A.shape[-1], B)  # tpc.tri_solve's first line on the card: float32 operands only
    return tpc.tri_solve_reference(A, B, upper, left, unitriangular)


def getrs_like_hpu(LU, pivots, B):
    tpc._check_small(LU, LU.shape[-1])
    tpc._float32(LU, B)
    batch = torch.broadcast_shapes(LU.shape[:-2], B.shape[:-2])
    n = LU.shape[-1]
    with torch.autocast("cpu", enabled=False):  # a TPC custom op: float32 in, float32 out
        return linalg.lu_solve(LU.expand(*batch, n, n), pivots.expand(*batch, n), B.expand(*batch, *B.shape[-2:]))


@pytest.fixture
def hpu_like_kernels(monkeypatch):
    monkeypatch.setattr(linalg, "_TRI", tri_like_hpu)
    monkeypatch.setattr(linalg, "_LU_SOLVE", getrs_like_hpu)


@pytest.mark.parametrize("left", [True, False])
@pytest.mark.parametrize("adjoint", [False, True])
def test_lu_solve_under_autocast_is_float32(hpu_like_kernels, left, adjoint):
    """Review: lu_solve(left=False, adjoint=True) raised 'hpu_ops.tpc takes float32 tensors' (B @ P in bf16);
    adjoint=True and left=False returned bfloat16. torch returns float32 for all four."""
    torch.manual_seed(0)
    X = torch.randn(3, 6, 6)
    LU, piv = torch.linalg.lu_factor(X @ X.mT + 6 * torch.eye(6))
    B = torch.randn(3, 6, 6, requires_grad=True)
    ref = torch.linalg.lu_solve(LU.double(), piv, B.detach().double(), left=left, adjoint=adjoint)
    with cpu_as_hpu(hpu_ops._install) as calls, torch.autocast("cpu", dtype=torch.bfloat16):
        out = torch.linalg.lu_solve(LU, piv, B, left=left, adjoint=adjoint)
        out.sum().backward()
    assert calls["lu_solve"] >= 1
    assert out.dtype == torch.float32 and B.grad.dtype == torch.float32
    assert rel(out.detach(), ref) < 1e-5


def test_large_solve_under_autocast_is_float32(monkeypatch):
    """Review: linalg.solve(A[8, 96, 96], B) on the large-kernel route returned bfloat16, rel err 2.8e-3 (torch float32
    1.9e-7): STaRNet's n = 96 Cayley solve. The kernel is its float32 replica, as in test_large_cpu."""
    monkeypatch.setattr(tpc, "_large_op", lambda: kr.gj_large_op())
    monkeypatch.setattr(linalg, "_LARGE_ON_CPU", True)
    torch.manual_seed(0)
    n = 96
    A = torch.randn(8, n, n) + n * torch.eye(n)
    B = torch.randn(8, n, 4)
    ref = torch.linalg.solve(A.double(), B.double())
    with hpu_autocast_table("linalg_solve"), cpu_as_hpu(hpu_ops._install) as calls:
        with torch.autocast("cpu", dtype=torch.bfloat16):
            X = torch.linalg.solve(A, B)
            Xe = torch.linalg.solve_ex(A, B)[0]
    assert calls["solve"] == 2
    assert X.dtype == Xe.dtype == torch.float32
    assert rel(X, ref) < 1e-6 and rel(Xe, ref) < 1e-6


def test_lstsq_under_autocast_is_float32():
    """Review: the lstsq solution was bfloat16 under autocast (HPU: linalg_lstsq falls through AutocastHPU)."""
    torch.manual_seed(0)
    A, B = torch.randn(4, 6, 5, requires_grad=True), torch.randn(4, 6, 2)
    with hpu_autocast_table("linalg_lstsq"), cpu_as_hpu(hpu_ops._install) as calls:
        with torch.autocast("cpu", dtype=torch.bfloat16):
            X = torch.linalg.lstsq(A, B).solution
            X.sum().backward()
    assert calls["lstsq"] == 1
    assert X.dtype == A.grad.dtype == torch.float32


# ---- 4: a killed build left torch's lock and the next process hung in ready()


CHILD = textwrap.dedent("""
    import sys, time, types, warnings
    for name in ("habana_frameworks", "habana_frameworks.torch", "habana_frameworks.torch.utils",
                 "habana_frameworks.torch.utils.lib_utils", "habana_frameworks.torch.utils.internal"):
        sys.modules[name] = types.ModuleType(name)
    sys.modules["habana_frameworks.torch.utils.lib_utils"].get_include_dir = lambda: "/nonexistent/include"
    sys.modules["habana_frameworks.torch.utils.lib_utils"].get_lib_dir = lambda: "/nonexistent/lib"
    sys.modules["habana_frameworks.torch.utils.internal"].is_lazy = lambda: False
    import torch.utils.cpp_extension as ext
    def load_inline(name, build_directory, **_):  # torch's FileBaton, then the build fails (no Habana headers)
        baton = ext.FileBaton(build_directory + "/lock")
        if baton.try_acquire():
            baton.release()
            raise RuntimeError("no Habana headers here")
        baton.wait()
        raise RuntimeError("waited for another build")
    ext.load_inline = load_inline
    from hpu_ops import tpc
    tpc._INSTALLED = True
    tpc._device_name = lambda: "GAUDI2"
    t0 = time.time()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        r = tpc.ready()
    print("READY", r, round(time.time() - t0, 1), [str(x.message)[:60] for x in w])
""")


def run_child(tmp_path, timeout=60, **env):
    full = {**os.environ, "TORCH_EXTENSIONS_DIR": str(tmp_path), "PYTHONPATH": str(ROOT), **env}
    return subprocess.run([sys.executable, "-c", CHILD], env=full, capture_output=True, text=True, timeout=timeout)


def leave_lock(tmp_path, age=0.0, record=None):
    build = tmp_path / "linalg_tpc_plugin"
    build.mkdir(parents=True, exist_ok=True)
    lock = build / "lock"
    lock.write_text("")
    if age:
        t = lock.stat().st_mtime - age
        os.utime(lock, (t, t))
    if record is not None:
        (build / tpc.BUILD_LOCK).write_text(record)
    return lock


def dead_pid():
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    return p.pid


def test_stale_lock_of_a_dead_builder_is_cleared(tmp_path):
    """Review: a job killed while holding torch's lock (about 2.5-4 s of every first ready()) left the lock file, and
    every later process blocked forever in ready(). Now the builder's record (this host, a dead pid) clears it at
    once, and ready() takes the 'one warning, then False' path."""
    import socket

    lock = leave_lock(tmp_path, record=f"{socket.gethostname()} {dead_pid()}\n")
    out = run_child(tmp_path).stdout
    assert "READY False" in out and "removed the stale build lock" in out and "building the TPC" in out
    assert not lock.exists()


def test_old_lock_without_a_record_is_cleared(tmp_path, monkeypatch):
    """A lock without hpu-ops' record (an older hpu-ops) is removed once older than STALE_LOCK_S."""
    lock = leave_lock(tmp_path, age=tpc.STALE_LOCK_S + 5)
    out = run_child(tmp_path).stdout
    assert "READY False" in out and "removed the stale build lock" in out
    assert not lock.exists()


def test_live_build_times_out_with_one_warning(tmp_path):
    """Another process holding hpu-ops' build lock: the wait ends at HPU_OPS_BUILD_TIMEOUT with one warning and
    ready() False (the docstring's path, unreachable on main)."""
    import fcntl

    build = tmp_path / "linalg_tpc_plugin"
    build.mkdir(parents=True)
    with open(build / tpc.BUILD_LOCK, "w") as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        out = run_child(tmp_path, HPU_OPS_BUILD_TIMEOUT="1").stdout
    assert "READY False" in out and out.count("building the TPC") == 1


def test_young_foreign_lock_is_waited_for_then_times_out(tmp_path):
    """A fresh torch lock without a record (an older hpu-ops building now) is not removed: waited for, then the
    timeout."""
    lock = leave_lock(tmp_path)
    out = run_child(tmp_path, HPU_OPS_BUILD_TIMEOUT="1").stdout
    assert "READY False" in out and "removed" not in out
    assert lock.exists()


# ---- 8: Padé-13 at Higham's double-precision theta lost 66-75x in float32


def test_replica_matrix_exp_dominant_positive_eigenvalue():
    """Review: diag(5, .5, -1) rel err 5.4e-6 against torch float32 8.2e-8; diag(80, .5, -1) 8.6e-5 against 1.1e-6
    (the kernel's replica). Bar: max(1e-6, 10x torch float32)."""
    for x, legacy_err in ((5.0, 5.4e-6), (80.0, 8.6e-5)):
        A = np.diag([x, 0.5, -1.0]).astype(np.float32)
        truth = torch.linalg.matrix_exp(torch.tensor(A, dtype=torch.float64))
        bar = max(1e-6, 10 * rel(torch.linalg.matrix_exp(torch.tensor(A)), truth))
        assert rel(kr.matrix_exp(A, theta=kr.THETA_LEGACY), truth) == pytest.approx(legacy_err, rel=0.1)  # main's
        assert rel(kr.matrix_exp(A), truth) <= bar


@pytest.mark.parametrize("x", [5.0, 20.0, 80.0])
def test_matrix_exp_gradient_dominant_positive_eigenvalue(x):
    """Review: the gradient of exp(A).sum() at diag(5, .5, -1) was 1.1e-5 against 1.2e-7 (20-170x). Through the
    Function with the kernels' contracts (Padé-13 composite and Fréchet), which share the kernel's THETA."""
    A = torch.diag(torch.tensor([x, 0.5, -1.0]))
    Ad = A.double().requires_grad_()
    torch.linalg.matrix_exp(Ad).sum().backward()
    At = A.clone().requires_grad_()
    torch.linalg.matrix_exp(At).sum().backward()
    Ah = A.clone().requires_grad_()
    with cpu_as_hpu(linalg.install_matrix_exp) as calls:
        torch.linalg.matrix_exp(Ah).sum().backward()
    assert calls["matrix_exp"] == 1
    assert rel(Ah.grad, Ad.grad) <= max(1e-6, 10 * rel(At.grad, Ad.grad))


def test_matrix_exp_theta_is_one_constant():
    """The kernel's THETA, the Python contract's and the replica's are the same number (the routes agree)."""
    source = (ROOT / "hpu_ops/tpc/matrix_exp_f32_gaudi2.c").read_text()
    theta = float(re.search(r"#define THETA ([0-9.]+)f", source).group(1))
    assert theta == linalg._PADE13_THETA == float(kr.THETA_13) == 2.0


# ---- 10: Functions returned their inputs or views


def test_lstsq_solution_can_be_scaled_in_place():
    """Review: X = lstsq(A, B).solution; X.mul_(0.5) raised 'Output 0 of _LstsqBackward is a view'."""
    torch.manual_seed(0)
    A = torch.randn(6, 4, dtype=torch.float64, requires_grad=True)
    B = torch.randn(6, 2, dtype=torch.float64)
    with cpu_as_hpu(linalg.install_lstsq) as calls:
        X = torch.linalg.lstsq(A, B).solution
        X.mul_(0.5)
        X.sum().backward()
    assert calls["lstsq"] == 1 and torch.isfinite(A.grad).all()


def test_lstsq_singular_values_can_be_clamped_in_place():
    """Review: r.singular_values.clamp_(min=1e-3) then r.solution.sum().backward() raised 'modified by an inplace
    operation' (S was saved and returned)."""
    torch.manual_seed(0)
    A = torch.randn(6, 4, dtype=torch.float64, requires_grad=True)
    B = torch.randn(6, 2, dtype=torch.float64)
    Ar = A.detach().clone().requires_grad_()
    torch.linalg.lstsq(Ar, B, driver="gelsd").solution.sum().backward()
    with cpu_as_hpu(linalg.install_lstsq) as calls:
        r = torch.linalg.lstsq(A, B, driver="gelsd")
        r.singular_values.clamp_(min=1e-3)
        r.solution.sum().backward()
    assert calls["lstsq"] == 1
    torch.testing.assert_close(A.grad, Ar.grad)
    assert not r.rank.requires_grad and not r.singular_values.requires_grad


def test_matrix_exp_result_can_be_edited_in_place(monkeypatch):
    """Review: Y = matrix_exp(A) (the kernel's output reshaped: a view); Y.diagonal().add_(1) and Y[..., :3, 3] = t
    raised. The kernel's contract (its output layout) behind the hook."""
    monkeypatch.setattr(linalg, "_MATRIX_EXP", lambda A: tpc.matrix_exp_reference(A).reshape(-1, 4, 4).reshape(A.shape))
    torch.manual_seed(0)
    A = (0.3 * torch.randn(2, 4, 4)).requires_grad_()
    with cpu_as_hpu(linalg.install_matrix_exp) as calls:
        Y = torch.linalg.matrix_exp(A)
        Y.diagonal(dim1=-2, dim2=-1).add_(1.0)
        Y[..., :3, 3] = 0.0
        Y.sum().backward()
    assert calls["matrix_exp"] == 1 and torch.isfinite(A.grad).all()


def test_lu_factor_and_inverse_results_can_be_edited_in_place(monkeypatch):
    """Review: lu_factor's LU (tpc.lu_factor's reshape) has the same issue; so had a 2-D inverse (the kernel's batch
    squeezed) and a right solve (a transpose)."""
    monkeypatch.setattr(linalg, "_LU", lambda A: tuple(t.reshape(t.shape) for t in tpc.lu_factor_reference(A)))
    monkeypatch.setattr(linalg, "_GJ", lambda A, B: tuple(t.squeeze(0).unsqueeze(0) for t in
                                                         tpc.gj_solve_reference(A.detach(), B.detach())))
    torch.manual_seed(0)
    A = (torch.randn(5, 5) + 5 * torch.eye(5)).requires_grad_()
    with cpu_as_hpu(hpu_ops._install) as calls:
        LU, piv = torch.linalg.lu_factor(A)
        LU.mul_(2.0)
        X = torch.linalg.inv(A)
        X.add_(1.0)  # as torch: allowed; a backward through X then raises (torch saves the inverse, so do we)
        Y = torch.linalg.solve(A, torch.randn(3, 5), left=False)
        Y.mul_(2.0)  # as torch: allowed (its backward then raises in torch too: the solution is saved)
    assert calls["lu_factor"] == 1 and calls["inv"] == 1 and calls["solve"] == 1
    assert not piv.requires_grad and LU.requires_grad and X.requires_grad


# ---- 11: matrix_exp's s read back by a scalar load with no ASO after the vector store


def test_matrix_exp_kernel_orders_the_scalar_read_after_the_store():
    """Review: v_f32_st_tnsr_partial of s (L320) then s_f32_ld_g (L31) with no aso: Habana's coherency table lists
    st_tnsr -> scalar ld_g as not kept coherent by hardware. The card check is repro/probe_code_review_linalg.py
    (aso: [[0, 1e6], [0, 0]] x 64 after other calls); here, the source: an aso(SW_VPU) before the load."""
    source = (ROOT / "hpu_ops/tpc/matrix_exp_f32_gaudi2.c").read_text()
    body = source[source.index("static int squaring_trips"):source.index("__local__ float64 a_mat")]
    assert body.index("aso(SW_VPU);") < body.index("s_f32_ld_g")
    assert source.index("v_f32_st_tnsr_partial(c0, Y_out, s_vec") < source.index("squaring_trips(Y_out, bt)")


# ---- 13: Gaudi's sort returns NaN as +inf with out-of-range indices


@contextlib.contextmanager
def gaudi_sort():
    """torch.Tensor.sort as Gaudi's (measured, hpu_ops.extrema): NaN sorted as +inf, its value +inf and its index out
    of range. Stable sorts without NaN are right."""
    real = torch.Tensor.sort

    def sort(self, dim=-1, descending=False, stable=False):
        nan = torch.isnan(self)
        values, index = real(torch.where(nan, float("inf"), self), dim=dim, descending=descending, stable=True)
        moved = nan.gather(dim, index)
        return torch.return_types.sort((values, torch.where(moved, self.shape[dim] + 7, index)))

    torch.Tensor.sort = sort
    try:
        yield
    finally:
        torch.Tensor.sort = real


def nan_symmetric():
    A = torch.randn(4, 4)
    A = A + A.T
    A[1, 2] = A[2, 1] = float("nan")
    return A


def test_tpc_eigh_keeps_nan_and_gathers_in_range(monkeypatch):
    """Review: a symmetric 4x4 with A[1,2] = A[2,1] = nan gave eigvalsh [inf]*4, the vectors gathered with bad
    indices. The kernel's outputs stand in (NaN eigenvalues, as from the Jacobi rotations on NaN)."""
    A = nan_symmetric()
    W = torch.tensor([[2.0, float("nan"), -1.0, float("nan")]])
    Vt = torch.eye(4).unsqueeze(0)
    monkeypatch.setattr(tpc, "_eigh_op", lambda: None)
    monkeypatch.setattr(tpc, "_eigh_refine_op", lambda: None)
    monkeypatch.setattr(tpc, "_call", lambda op, *args: (W, Vt))
    with gaudi_sort():
        w, V = tpc.eigh(A, upper=False)
    assert w[:2].tolist() == [-1.0, 2.0] and w[2:].isnan().all()
    assert torch.equal(V[:, :2], torch.eye(4)[:, [2, 0]])


def test_tpc_svd_keeps_nan_values(monkeypatch):
    """The same sort in jacobi_svd and jacobi_svdvals (descending): NaN singular values stay NaN, never +-inf."""
    S = torch.tensor([[1.0, float("nan"), 3.0, 2.0]])
    monkeypatch.setattr(tpc, "_svdvals_op", lambda: None)
    monkeypatch.setattr(tpc, "_svd_op", lambda: None)
    U = torch.eye(4).unsqueeze(0)
    with gaudi_sort():
        monkeypatch.setattr(tpc, "_call", lambda op, *args: S)
        sv = tpc.jacobi_svdvals(nan_symmetric())
        monkeypatch.setattr(tpc, "_call", lambda op, *args: (U, S, U))
        _, S2, _ = tpc.jacobi_svd(nan_symmetric())
    for s in (sv, S2):
        assert not s.isinf().any() and s.isnan().sum() == 1


# ---- cut items


def replica_gj(A, B):
    batch = torch.broadcast_shapes(A.shape[:-2], B.shape[:-2])
    n, m = A.shape[-1], B.shape[-1]
    a = A.detach().expand(*batch, n, n).reshape(-1, n, n)
    b = B.detach().expand(*batch, n, m).reshape(-1, n, m)
    outs = [kr.gj_solve(x.numpy(), y.numpy()) for x, y in zip(a, b)]
    X = torch.from_numpy(np.stack([o[0] for o in outs])).to(A.dtype).reshape(*batch, n, m)
    D = torch.from_numpy(np.stack([o[1] for o in outs])).to(A.dtype).reshape(*batch, n)
    S = torch.from_numpy(np.stack([np.array(o[2], np.float32) for o in outs])).to(A.dtype).reshape(*batch, 2)
    return X, D, S


def test_traced_det_of_singular_matrix_on_lapack_is_finite(monkeypatch):
    """Review (linalg.py:941): compiled det of diag(1, 0, 2) declined with grad (float64 on hpu, or the kernel off),
    its backward's inverse on LAPACK: NaN gradient (0 * inf); torch gives 0, as does the kernel route."""
    monkeypatch.setattr(linalg, "_takes_tensors", lambda device_type, name, tensors: False)
    A = torch.diag(torch.tensor([1.0, 0.0, 2.0], dtype=torch.float64)).requires_grad_()
    with cpu_as_hpu(linalg.install_inv_det):
        compiled(lambda a: torch.linalg.det(a).sum(), [], fullgraph=True)(A).backward()
    Ar = A.detach().clone().requires_grad_()
    torch.linalg.det(Ar).backward()
    torch.testing.assert_close(A.grad, Ar.grad)


def test_traced_det_kernel_route_unchanged(monkeypatch):
    """The kernel route of the same compiled det (symbolic batch, the replica at run time) still gives torch's."""
    monkeypatch.setattr(linalg, "_GJ", replica_gj)
    A2 = torch.stack([torch.diag(torch.tensor([1.0, 0.0, 2.0])), 2 * torch.eye(3)]).requires_grad_()
    with cpu_as_hpu(linalg.install_inv_det):
        compiled(lambda a: torch.linalg.det(a).sum(), [], fullgraph=True, dynamic=True)(A2).backward()
    Ar = A2.detach().clone().requires_grad_()
    torch.linalg.det(Ar).sum().backward()
    torch.testing.assert_close(A2.grad, Ar.grad)


def traceable_svd(A, compute_uv):
    """The SVD hook as the tests' (svd_reference on real tensors, torch's traceable SVD under fake tensors and
    make_fx, as the TPC op is traceable on HPU)."""
    proxy = torch._C._get_dispatch_mode(torch._C._TorchDispatchModeKey.PROXY) is not None
    if not linalg._is_fake(A) and not proxy:
        return tpc.svd_reference(A, compute_uv=compute_uv)
    if not compute_uv:
        return A.new_empty(0), torch.linalg.svdvals(A), A.new_empty(0)
    return torch.linalg.svd(A, full_matrices=False)


def test_compiled_gels_rank_deficient_raises(monkeypatch):
    """Review (linalg.py:1312): compiled lstsq(driver='gels') of a rank-deficient A (a zero column) returned
    [nan, nan, nan, nan, -inf, inf]; torch raises LinAlgError, compiled or not (error code 3, rank 2 + 1)."""
    monkeypatch.setattr(linalg, "_SVD", traceable_svd)
    torch.manual_seed(0)
    A = torch.zeros(5, 3)
    A[:, :2] = torch.randn(5, 2)
    B = torch.randn(5, 2)
    f = lambda a, b: torch.linalg.lstsq(a, b, driver="gels").solution  # noqa: E731
    with pytest.raises(torch.linalg.LinAlgError, match=r"full rank \(error code: 3\)"):
        f(A, B)
    with cpu_as_hpu(linalg.install_lstsq) as calls:
        with pytest.raises(torch.linalg.LinAlgError, match=r"full rank \(error code: 3\)"):
            compiled(f, [])(A, B)
        assert calls["lstsq"] >= 1
        full = torch.randn(5, 3)
        torch.testing.assert_close(compiled(f, [])(full, B), torch.linalg.lstsq(full, B, driver="gels").solution)


@pytest.mark.parametrize("driver", [None, "gelsy", "gelsd", "gelss", "gels"])
def test_make_fx_of_lstsq(monkeypatch, driver):
    """Review (linalg.py:1299): make_fx (real tensors, a proxy mode) of lstsq raised 'trying to get value out of a
    tracing tensor' for every driver: the rank check read a proxy. Now traced calls are the tracers' case."""
    monkeypatch.setattr(linalg, "_SVD", traceable_svd)
    torch.manual_seed(0)
    A, B = torch.randn(5, 3), torch.randn(5, 2)
    g = lambda a, b: torch.linalg.lstsq(a, b, driver=driver).solution  # noqa: E731
    with cpu_as_hpu(linalg.install_lstsq):
        out = make_fx(g)(A, B)(A, B)
    torch.testing.assert_close(out, g(A, B), atol=1e-5, rtol=1e-5)


def test_build_script_renames_the_new_library():
    """Review (build.sh:20): cp over the loaded .so crashed running jobs (SIGBUS/SIGSEGV). The new library is copied to
    a temporary name in the same directory and renamed over the old one."""
    script = (ROOT / "hpu_ops/tpc/build.sh").read_text()
    commands = [line for line in script.splitlines() if not line.lstrip().startswith("#")]
    assert not any(re.match(r'\s*cp .*"\$here/liblinalg_tpc\.so"\s*$', line) for line in commands)
    assert any(re.match(r'\s*mv -f "\$here/\.liblinalg_tpc\.so\.new\.\$\$" "\$here/liblinalg_tpc\.so"', line)
               for line in commands)


# ---- adversarial pass on the fixes


def test_sort_nan_inf_ties_and_shapes():
    """``tpc._sort``: NaN sorted as +inf (NaN and +inf tie: torch puts NaN last; eigenvalues and singular values of
    finite input are finite), indices in range, 1-D and batched. (The CPU's sort keeps ties in order here.)"""
    x = torch.tensor([1.0, float("nan"), -float("inf"), 1.0, float("inf"), float("nan")])
    with gaudi_sort():
        v, i = tpc._sort(x)
        vd, idd = tpc._sort(x, descending=True)
        vb, ib = tpc._sort(x.expand(3, 2, 6))
    assert i.tolist() == [2, 0, 3, 1, 4, 5] and torch.equal(v.nan_to_num(7.0), x[i].nan_to_num(7.0))
    assert idd.tolist() == [1, 4, 5, 0, 3, 2] and torch.equal(vd.nan_to_num(7.0), x[idd].nan_to_num(7.0))
    assert ((ib >= 0) & (ib < 6)).all() and torch.equal(ib[1, 1], i) and vb.shape == (3, 2, 6)
    allnan = torch.full((2, 3), float("nan"))
    with gaudi_sort():
        v, i = tpc._sort(allnan)
    assert v.isnan().all() and i.tolist() == [[0, 1, 2], [0, 1, 2]]


def test_autocast_inference_mode_and_compiled(hpu_like_kernels):
    """The backend kernels (inference mode) and a traced graph (aot) under autocast: float32 too."""
    torch.manual_seed(0)
    X = torch.randn(2, 5, 5)
    LU, piv = torch.linalg.lu_factor(X @ X.mT + 5 * torch.eye(5))
    B = torch.randn(2, 5, 3)
    with cpu_as_hpu(hpu_ops._install) as calls, torch.autocast("cpu", dtype=torch.bfloat16):
        with torch.inference_mode():
            a = torch.linalg.lu_solve(LU, piv, B, adjoint=True)
        b = compiled(lambda l, p, x: torch.linalg.lu_solve(l, p, x, adjoint=True), [])(LU, piv, B)
    assert calls["lu_solve"] >= 2 and a.dtype == b.dtype == torch.float32
    ref = torch.linalg.lu_solve(LU, piv, B, adjoint=True)
    torch.testing.assert_close(a, ref)
    torch.testing.assert_close(b, ref)


def test_owned_outputs_odd_shapes():
    """In-place edits of results at n = 1, a vector right-hand side, a broadcast batch; non-differentiable outputs
    stay so; double backward through a right solve (a transposed kernel output) still matches torch."""
    torch.manual_seed(0)
    A = torch.randn(3, 1, 1, dtype=torch.float64, requires_grad=True)
    with cpu_as_hpu(linalg.install_matrix_exp):
        Y = torch.linalg.matrix_exp(A)
        Y.add_(1.0)
        Y.sum().backward()
    torch.testing.assert_close(A.grad, A.detach().exp())
    A = torch.randn(6, 4, dtype=torch.float64, requires_grad=True)
    b = torch.randn(6, dtype=torch.float64)
    with cpu_as_hpu(linalg.install_lstsq) as calls:
        r = torch.linalg.lstsq(A, b)
        r.solution.mul_(2.0)
        r.solution.sum().backward()
    assert calls["lstsq"] == 1 and not r.rank.requires_grad
    M = (torch.randn(2, 1, 4, 4, dtype=torch.float64) + 4 * torch.eye(4)).requires_grad_()
    Bm = torch.randn(3, 5, 4, dtype=torch.float64, requires_grad=True)
    with cpu_as_hpu(linalg.install_solve):
        torch.autograd.gradgradcheck(lambda a, x: torch.linalg.solve(a, x, left=False), (M, Bm))
        X, info = torch.linalg.solve_ex(M, Bm, left=False)
        LU, pivots = torch.linalg.lu_factor(M)
    assert X.requires_grad and not info.requires_grad and not pivots.requires_grad


# ---- team Z's rsqrt bias, in the real one-sided Jacobi SVD kernel


def low_estimate(x):
    """The hardware rsqrt estimate 1 ulp low (as the card's sometimes is)."""
    return np.nextafter(kr.hw_rsqrt(x), np.float32(0)).astype(np.float32)


def test_svd_rsqrt_newton_is_unbiased():
    """The plain Newton step r (1.5 - x r^2 / 2) keeps a 1-ulp-low estimate at 1 (0.99999994); the residual form gives
    1. Over the kernel's replica, the low estimate with the plain step drifts the accumulated rotations' orthogonality
    with n (1.0e-5 at n = 32; Gaudi2 measured 6.1e-6 before, 3.0e-6 after, torch 1.3e-6); with the residual step it
    no longer depends on the estimate's bias."""
    import functools

    assert kr._rsqrt_newton(np.float32(1), legacy=True, estimate=low_estimate)[()] == np.float32(0.99999994)
    assert kr._rsqrt_newton(np.float32(1), estimate=low_estimate)[()] == 1
    torch.manual_seed(5)
    A = torch.randn(32, 32).numpy()
    real = kr._rsqrt_newton

    def u_ortho(**kw):
        kr._rsqrt_newton = functools.partial(real, **kw)
        try:
            U = np.asarray(kr.jacobi_svd(A)[0], np.float64)
        finally:
            kr._rsqrt_newton = real
        return np.abs(U.T @ U - np.eye(32)).max()

    torch_u = torch.linalg.svd(torch.tensor(A))[0].double()
    bar = (torch_u.mT @ torch_u - torch.eye(32, dtype=torch.float64)).abs().max().item()
    fixed_low, fixed_exact, legacy_low = u_ortho(estimate=low_estimate), u_ortho(), u_ortho(legacy=True,
                                                                                         estimate=low_estimate)
    assert legacy_low > 10 * bar and legacy_low > 4 * fixed_low  # the drift, and the fix of it
    assert fixed_low < 2 * fixed_exact and fixed_exact < 5 * bar  # no longer depends on the estimate's bias
