"""Adversarial round 2 (team D): cases that break what the branch ships, reproduced on the CPU stand-in for HPU and
measured on Gaudi2 (tests/probe_adversarial_d.py, runs/adv_r2*_*.txt). Each test compares
with torch's own result for the same call without the rewrite. They failed on 10bed70."""
import pytest
import torch
from test_cpu import cpu_as_hpu
from torch.autograd import forward_ad

from hpu_ops import linalg, tpc


@pytest.fixture
def svd_hook(monkeypatch):
    monkeypatch.setattr(linalg, "_SVD", lambda A, uv: tpc.svd_reference(A.detach(), compute_uv=uv))


def _A():
    return torch.randn(3, 5, 4, dtype=torch.float64, generator=torch.Generator().manual_seed(0))


def _T():
    return torch.randn(3, 5, 4, dtype=torch.float64, generator=torch.Generator().manual_seed(1))


SVD_FNS = {
    "svdvals": torch.linalg.svdvals,
    "matrix_norm_2": lambda a: torch.linalg.matrix_norm(a, 2),
    "matrix_norm_nuc": lambda a: torch.linalg.matrix_norm(a, "nuc"),
    "svd_S": lambda a: torch.linalg.svd(a, full_matrices=False)[1],
    "pinv": torch.linalg.pinv,
}
INSTALL = {"pinv": linalg.install_pinv}


@pytest.mark.parametrize("name", list(SVD_FNS))
def test_forward_ad_matches_torch(name, svd_hook):
    """torch.autograd.forward_ad: the tangent is dropped (None) for svdvals / matrix_norm; svd and pinv raise."""
    f, A, T = SVD_FNS[name], _A(), _T()

    def run():
        with forward_ad.dual_level():
            return forward_ad.unpack_dual(f(forward_ad.make_dual(A, T))).tangent

    want = run()
    with cpu_as_hpu(INSTALL.get(name, linalg.install_svd)):
        got = run()
    assert got is not None
    torch.testing.assert_close(got, want)


@pytest.mark.parametrize("driver", ["gelsd", "gelss"])
def test_compiled_lstsq_rank_deficient_rank_and_residuals(driver, svd_hook):
    """Under torch.compile the rank check is skipped (fake tensors): a rank-2 7x4 input gets rank 4 and residuals that
    torch leaves empty when rank < n. Eager declines the same call."""
    g = torch.Generator().manual_seed(5)
    A = torch.randn(7, 2, generator=g, dtype=torch.float64) @ torch.randn(2, 4, generator=g, dtype=torch.float64)
    B = torch.randn(7, 3, generator=g, dtype=torch.float64)
    want = torch.linalg.lstsq(A, B, driver=driver)
    with cpu_as_hpu(linalg.install_lstsq) as calls:
        torch._dynamo.reset()
        got = torch.compile(lambda a, b: torch.linalg.lstsq(a, b, driver=driver), backend="aot_eager")(A, B)
    assert got.rank.tolist() == want.rank.tolist()
    assert got.residuals.shape == want.residuals.shape


def test_compiled_lstsq_gels_still_taken(svd_hook):
    """gels reports no rank and assumes full rank, so it stays on the rewrite under torch.compile."""
    g = torch.Generator().manual_seed(6)
    A = torch.randn(7, 4, generator=g, dtype=torch.float64)
    B = torch.randn(7, 3, generator=g, dtype=torch.float64)
    want = torch.linalg.lstsq(A, B, driver="gels")
    with cpu_as_hpu(linalg.install_lstsq) as calls:
        torch._dynamo.reset()
        got = torch.compile(lambda a, b: torch.linalg.lstsq(a, b, driver="gels"), backend="aot_eager")(A, B)
        assert calls["lstsq"] >= 1
    torch.testing.assert_close(got.solution, want.solution)
    torch.testing.assert_close(got.residuals, want.residuals)


def test_lu_installed_in_default_mode_of_bridge_1_21(monkeypatch):
    """Bridge 1.21.4 runs eager when PT_HPU_LAZY_MODE is unset (measured in the pod: is_lazy() False), but
    ``_lu_wanted`` (and ``tpc._load`` / ``tpc._flush``) read an unset variable as lazy, so the eager-only LU ops are
    not installed in a default process."""
    monkeypatch.delenv("PT_HPU_LAZY_MODE", raising=False)
    monkeypatch.delenv(linalg.LU_SLOW, raising=False)
    assert linalg._lu_wanted("hpu", "lu_factor")


@pytest.fixture
def hh_and_svd_hooks(monkeypatch):
    monkeypatch.setattr(linalg, "_HOUSEHOLDER", tpc.householder_product_reference)
    monkeypatch.setattr(linalg, "_GJ", tpc.gj_solve_reference)
    monkeypatch.setattr(linalg, "_SVD", lambda A, uv: tpc.svd_reference(A.detach(), compute_uv=uv))


@pytest.mark.parametrize("name", ["householder_product", "lstsq"])
def test_forward_ad_householder_and_lstsq_match_torch(name, hh_and_svd_hooks):
    """Their ``takes`` do not read ``_transformed()`` either: householder_product raises (no jvp in the Function),
    lstsq without requires_grad returns the solution as a plain tensor, so the tangent is dropped (None)."""
    g = torch.Generator().manual_seed(7)
    A = torch.randn(3, 6, 4, dtype=torch.float64, generator=g)
    T = torch.randn(3, 6, 4, dtype=torch.float64, generator=g)
    if name == "householder_product":
        A = A.tril(-1)
        other = torch.rand(3, 4, dtype=torch.float64, generator=g)
        f, install = (lambda a: torch.linalg.householder_product(a, other)), linalg.install_householder_product
    else:
        other = torch.randn(3, 6, 2, dtype=torch.float64, generator=g)
        f, install = (lambda a: torch.linalg.lstsq(a, other).solution), linalg.install_lstsq

    def run():
        with forward_ad.dual_level():
            return forward_ad.unpack_dual(f(forward_ad.make_dual(A, T))).tangent

    want = run()
    with cpu_as_hpu(install):
        got = run()
    assert got is not None
    torch.testing.assert_close(got, want)


def test_install_kernel_path_again_on_an_initialized_device_keeps_the_kernels(monkeypatch):
    """``enable()`` after its undo calls ``install_kernel_path`` again; the device is initialized by then, so it sets
    ``_INSTALLED = False`` although the path from the first call is still in effect: every TPC op is declined for the
    rest of the process (Gaudi2: svdvals and solve uncounted after undo + enable, runs/adv_r2b_*.txt)."""
    from hpu_ops import tpc as tpc_mod

    monkeypatch.delenv(tpc_mod.OPT_OUT, raising=False)
    monkeypatch.setattr(tpc_mod, "_lib_exists", lambda path: True)
    monkeypatch.setattr(tpc_mod, "_INSTALLED", False)
    monkeypatch.setattr(tpc_mod, "_LATE_WARNED", True)
    monkeypatch.setenv("GC_KERNEL_PATH", "")
    state = {"init": False}
    monkeypatch.setattr(tpc_mod, "_initialized", lambda: state["init"])
    assert tpc_mod.install_kernel_path()
    assert tpc_mod.available()
    state["init"] = True  # first HPU op ran with the kernel path set
    tpc_mod.install_kernel_path()  # enable() after undo
    assert tpc_mod.available()


def test_pinv_of_non_finite_input_is_non_finite_not_an_error(svd_hook):
    """Documented difference (README, SVD row): torch's pinv of an all-inf matrix raises LinAlgError (the SVD does not
    converge); the rewrite returns non-finite values, as a check would cost a host sync per call."""
    A = torch.full((2, 3, 3), float("inf"), dtype=torch.float64)
    with pytest.raises(torch.linalg.LinAlgError):
        torch.linalg.pinv(A)
    with cpu_as_hpu(linalg.install_pinv):
        got = torch.linalg.pinv(A)
    assert not torch.isfinite(got).all()


def test_tpc_load_picks_the_eager_api_when_lazy_mode_is_unset(monkeypatch):
    """Bridge 1.21.4 is eager when PT_HPU_LAZY_MODE is unset (pod: is_lazy() False), but ``tpc._load`` reads unset as
    lazy and builds the legacy lazy-API plugin. On Gaudi2 that leaves ``tpc.ready()`` False: no TPC kernel runs in a
    default process (svd, pinv, matrix_exp, householder decline to hpu-ops' CPU route; runs/adv_r2b_eager2_main_unset.txt)."""
    import sys
    import types

    import torch.utils.cpp_extension

    from hpu_ops import tpc as tpc_mod

    names = []
    lib_utils = types.ModuleType("habana_frameworks.torch.utils.lib_utils")
    lib_utils.get_include_dir = lambda: "/inc"
    lib_utils.get_lib_dir = lambda: "/lib"
    for mod in ("habana_frameworks", "habana_frameworks.torch", "habana_frameworks.torch.utils"):
        monkeypatch.setitem(sys.modules, mod, sys.modules.get(mod) or types.ModuleType(mod))
    monkeypatch.setitem(sys.modules, "habana_frameworks.torch.utils.lib_utils", lib_utils)
    monkeypatch.setattr(torch.utils.cpp_extension, "load_inline", lambda name, **kw: names.append(name))
    monkeypatch.delenv("PT_HPU_LAZY_MODE", raising=False)
    tpc_mod._load.__wrapped__()
    assert names == ["linalg_tpc_plugin"]
