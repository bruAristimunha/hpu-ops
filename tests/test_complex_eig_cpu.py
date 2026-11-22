"""CPU tests of the general complex64 ``eig`` / ``eigvals`` MPS-recipe replica (``kernel_replica.ceig_kernel``)
and ``hpu_ops.tpc.complex_linalg.eig`` / ``eigvals`` API contracts.

No scipy dependency: uses a pure-NumPy O(n^3) Hungarian (Jonker-Volgenant) minimum-cost bipartite matcher for
eigenvalue matching against ``complex128`` reference eigenvalues.
"""
import inspect
import itertools
import math

import kernel_replica as kr
import numpy as np
import pytest
import torch

from hpu_ops.tpc import complex_linalg as cl

C64, C128 = torch.complex64, torch.complex128
EPS32 = float(2.0 ** -23)


# ---- Pure-NumPy Hungarian assignment & error helpers ---------------------------------------------------------------


def linear_sum_assignment_np(cost):
    """Exact O(n^3) Hungarian minimum-sum assignment for an ``(n, n)`` cost matrix (no scipy)."""
    cost = np.asarray(cost, dtype=np.float64)
    n, m = cost.shape
    assert n == m
    if n == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    u = np.zeros(n + 1, dtype=np.float64)
    v = np.zeros(n + 1, dtype=np.float64)
    p = np.zeros(n + 1, dtype=np.int64)
    way = np.zeros(n + 1, dtype=np.int64)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(n + 1, np.inf, dtype=np.float64)
        used = np.zeros(n + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = np.inf
            j1 = 0
            for j in range(1, n + 1):
                if not used[j]:
                    cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    col_ind = np.empty(n, dtype=np.int64)
    for j in range(1, n + 1):
        col_ind[p[j] - 1] = j - 1
    return np.arange(n, dtype=np.int64), col_ind


def eig_match_err(w_cand, w_ref):
    """Hungarian-matched max eigenvalue distance relative to ``max(1e-30, max|w_ref|)``."""
    w1 = np.asarray(w_cand, dtype=np.complex128).ravel()
    w2 = np.asarray(w_ref, dtype=np.complex128).ravel()
    if w1.size == 0:
        return 0.0
    cost = np.abs(w1[:, None] - w2[None, :])
    r, c = linear_sum_assignment_np(cost)
    max_diff = float(cost[r, c].max())
    denom = float(np.abs(w2).max())
    if denom == 0.0:
        return max_diff
    return max_diff / max(1e-30, denom)


def eig_residual(A, w, V):
    """Frobenius residual ``||A V - V diag(w)||_F / ||A||_F`` (or unnormalized if A == 0)."""
    A128 = np.asarray(A, dtype=np.complex128)
    w128 = np.asarray(w, dtype=np.complex128)
    V128 = np.asarray(V, dtype=np.complex128)
    res = float(np.linalg.norm(A128 @ V128 - V128 * w128[None, :], ord="fro"))
    nrm = float(np.linalg.norm(A128, ord="fro"))
    return res if nrm == 0.0 else res / nrm


def planar_to_complex(W_planar, V_planar=None):
    w = (W_planar[0] + 1j * W_planar[1]).astype(np.complex64)
    if V_planar is None:
        return w, None
    n = W_planar.shape[-1]
    V = (V_planar[:n] + 1j * V_planar[n:]).astype(np.complex64)
    return w, V


def complex_to_planar(A_c64):
    A_c64 = np.asarray(A_c64, dtype=np.complex64)
    return np.vstack([A_c64.real.astype(np.float32), A_c64.imag.astype(np.float32)])


def crandn(*shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.complex(torch.randn(*shape, generator=g), torch.randn(*shape, generator=g)).to(C64)


def pairs(Z):
    return torch.view_as_real(Z.to(C64)).contiguous()


def cplx(P):
    return torch.view_as_complex(P.double().contiguous())


def _ceig_needs_info():
    if not hasattr(cl, "_ceig_op"):
        return False
    src = inspect.getsource(cl.eig) if hasattr(cl, "eig") else ""
    return "info" in src or ", _, " in src or ", _ =" in src and src.count(",") >= 2


def _ceigvals_needs_info():
    if not hasattr(cl, "_ceigvals_op"):
        return False
    src = inspect.getsource(cl.eigvals) if hasattr(cl, "eigvals") else ""
    return ", _" in src or "info" in src


@pytest.fixture
def cl_eig_replica(monkeypatch):
    if not (hasattr(cl, "_ceig_op") and hasattr(cl, "_ceigvals_op") and hasattr(cl, "eig") and hasattr(cl, "eigvals")):
        pytest.skip("hpu_ops.tpc.complex_linalg.eig/eigvals not yet implemented")
    monkeypatch.setenv("PT_HPU_LAZY_MODE", "0")
    monkeypatch.setattr(cl, "_ceig_op", lambda: kr.ceig_op(with_info=_ceig_needs_info()))
    monkeypatch.setattr(cl, "_ceigvals_op", lambda: kr.ceigvals_op(with_info=_ceigvals_needs_info()))
    if hasattr(cl, "_ceig_big_op"):
        monkeypatch.setattr(cl, "_ceig_big_op", lambda: kr.ceig_big_op(with_info=_ceig_needs_info()))
    if hasattr(cl, "_ceigvals_big_op"):
        monkeypatch.setattr(cl, "_ceigvals_big_op", lambda: kr.ceigvals_big_op(with_info=_ceig_needs_info()))


# ---- Test cases ----------------------------------------------------------------------------------------------------


def _jordan_sim(m, lam, seed):
    rng = np.random.default_rng(seed)
    J = np.diag([lam] * m).astype(np.complex128)
    for i in range(m - 1):
        J[i, i + 1] = 1.0
    Q, _ = np.linalg.qr(rng.normal(size=(m, m)) + 1j * rng.normal(size=(m, m)))
    return (Q @ J @ Q.conj().T).astype(np.complex64)


def _embed_jordan4(n, lam, seed):
    rng = np.random.default_rng(seed)
    J = np.diag(np.concatenate([np.full(4, lam, dtype=np.complex128), np.arange(5, n + 1, dtype=np.complex128)]))
    for i in range(3):
        J[i, i + 1] = 1.0
    Q, _ = np.linalg.qr(rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)))
    return (Q @ J @ Q.conj().T).astype(np.complex64)


def _illcond_nonnormal(n, cond=1e5, seed=0):
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)))
    T = np.diag(np.geomspace(1.0, 1.0 / cond, n).astype(np.complex128))
    iu = np.triu_indices(n, 1)
    T[iu] = 0.05 * (rng.normal(size=len(iu[0])) + 1j * rng.normal(size=len(iu[0])))
    return (Q @ T @ Q.conj().T).astype(np.complex64)


def _adversarial_cases():
    rng = np.random.default_rng(123)
    cases = []

    # 1. Zero & identity across sizes (n <= 32 and 32 < n <= 64)
    for n in (1, 4, 16, 32, 33, 48, 64):
        cases.append((f"zero_{n}", np.zeros((n, n), dtype=np.complex64), None))
        cases.append((f"eye_{n}", np.eye(n, dtype=np.complex64), None))

    # 1b. Subnormal matrix at n=64 (FTZ contract: flushes to zero -> W=0, V=I)
    cases.append(("subnormal_64", (rng.normal(size=(64, 64)) * 1e-40).astype(np.complex64), None))

    # 1c. Real float32 non-Hermitian (R, 0) for 32 < n <= 64
    for n in (33, 48, 64):
        R_real = rng.normal(size=(n, n)).astype(np.float32)
        cases.append((f"rand_real_{n}", R_real.astype(np.complex64), None))

    # 2. Diagonal with zero & complex entries
    cases.append(("diag_8", np.diag(np.array([1, -2, 3j, -4j, 1 + 2j, -1e-3, 1e3, 0], dtype=np.complex64)), None))

    # 3. Upper & lower triangular
    U = np.triu(rng.normal(size=(12, 12)) + 1j * rng.normal(size=(12, 12))).astype(np.complex64)
    cases.append(("upper_tri_12", U, None))
    cases.append(("lower_tri_12", U.T.copy(), None))

    # 4. Cyclic shift C_8 (8th roots of unity) & random companion (n=16)
    C8 = np.zeros((8, 8), dtype=np.complex64)
    C8[1:, :-1] = np.eye(7)
    C8[0, -1] = 1.0
    cases.append(("cyclic_8", C8, None))

    C16 = np.zeros((16, 16), dtype=np.complex64)
    C16[1:, :-1] = np.eye(15)
    C16[:, -1] = -(rng.normal(size=16) + 1j * rng.normal(size=16)) * 0.2
    cases.append(("companion_16", C16, None))

    # 5. Defective Jordan blocks J_2, J_3, J_4 (similarity-transformed)
    for m, lam, sd in ((2, 1.5 - 0.5j, 201), (3, -1.0 + 2.0j, 202), (4, 2.0 + 1.0j, 203)):
        cases.append((f"jordan_{m}_sim", _jordan_sim(m, lam, sd), m))

    # 6. Near-defective 2x2
    cases.append(("near_def_2", np.array([[1.0, 1.0], [1e-6, 1.0 + 1e-6]], dtype=np.complex64), 2))

    # 7. Repeated eigenvalues (diagonal & similarity-transformed)
    rep_eigs = np.array([0.0, 0.0, 0.0, 1.0, 1.0 + 1e-6, 2.0, -3j, 5.0 + 2j], dtype=np.complex64)
    cases.append(("repeated_diag_8", np.diag(rep_eigs), None))
    Q8, _ = np.linalg.qr(rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8)))
    cases.append(("repeated_sim_8", (Q8 @ np.diag(rep_eigs.astype(np.complex128)) @ Q8.conj().T).astype(np.complex64), None))

    # 8. Real rotation / conjugate-pair blocks
    R = np.zeros((6, 6), dtype=np.complex64)
    for k, th in enumerate([0.3, 1.1, 2.5]):
        R[2 * k:2 * k + 2, 2 * k:2 * k + 2] = [[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]]
    cases.append(("real_rot_6", R, None))

    # 9. Badly scaled (2^[-30..30])
    D_bad = np.diag(2.0 ** np.linspace(-30, 30, 8))
    M_rand = rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8))
    cases.append(("bad_scale_8", (D_bad @ M_rand @ np.linalg.inv(D_bad)).astype(np.complex64), None))

    # 10. Extreme scales (1e-30, 1e30, 3e37, and e == 254 boundary 2.5e38)
    base6 = (rng.normal(size=(6, 6)) + 1j * rng.normal(size=(6, 6))).astype(np.complex64)
    for sc_name, sc in (("tiny_1e-30", 1e-30), ("huge_1e30", 1e30), ("huge_3e37", 3e37)):
        cases.append((sc_name, (base6 * np.float32(sc)).astype(np.complex64), None))
    cases.append(("scale_eye_2p5e38", np.eye(4, dtype=np.complex64) * np.float32(2.5e38), None))
    Q6, _ = np.linalg.qr(rng.normal(size=(6, 6)) + 1j * rng.normal(size=(6, 6)))
    cases.append(("huge_2p5e38", (Q6 * 2.5e38).astype(np.complex64), None))

    # 11. Forsythe perturbed nilpotent Jordan blocks (triggers clartg |f| <= 2^-64 |g| underflow)
    for fn in (16, 24, 32):
        Fo = np.diag(np.ones(fn - 1, dtype=np.complex64), 1)
        Fo[fn - 1, 0] = np.float32(0.1)
        cases.append((f"forsythe_n{fn}_eps0.1", Fo, None))

    # 12. Graded Frank matrix (n=16 within residual tolerance)
    F16 = np.zeros((16, 16), dtype=np.complex64)
    for i in range(16):
        for j in range(i - 1 if i > 0 else 0, 16):
            F16[i, j] = float(16 - max(i, j))
    cases.append(("frank_n16", F16, None))

    # 13. 4x4 upper Hessenberg matrices exercising clartg f2 in [2^-126, 2^-125) (rsqrt_fine e == 1 FTZ boundary)
    hessenberg_e1_f = (
        ("clartg_f_1p1e-19", 1.1e-19),
        ("clartg_f_1p2e-19", 1.2e-19),
        ("clartg_f_1p3e-19", 1.3e-19),
        ("clartg_f_1p4e-19", 1.4e-19),
        ("clartg_f_1p5e-19", 1.5e-19),
        ("clartg_f_1p2e-19_cpx", 1.2e-19 + 0.3e-19j),
    )
    for cname, fv in hessenberg_e1_f:
        H = np.array(
            [[fv, 0.2, 0.3, 0.4], [1.0, 0.5, 0.1, 0.2], [0.0, 0.5, 0.7, 0.0], [0.0, 0.0, 0.5, 0.0]],
            dtype=np.complex64,
        )
        cases.append((cname, H, None))
    H_sym_g = np.array(
        [[1.0, 0.2, 0.3, 0.4], [1.2e-19, 0.5, 0.1, 0.2], [0.0, 0.5, 0.7, 0.0], [0.0, 0.0, 0.5, 0.0]],
        dtype=np.complex64,
    )
    cases.append(("clartg_g_1p2e-19", H_sym_g, None))

    # 14. 32 < n <= 64 adversarial classes (ceig_big)
    U64 = np.triu(rng.normal(size=(64, 64)) + 1j * rng.normal(size=(64, 64))).astype(np.complex64)
    cases.append(("upper_tri_64", U64, None))
    cases.append(("lower_tri_64", U64.T.copy(), None))

    C64 = np.zeros((64, 64), dtype=np.complex64)
    C64[1:, :-1] = np.eye(63)
    C64[0, -1] = 1.0
    cases.append(("cyclic_64", C64, None))

    for n in (33, 48, 64):
        rep_n = np.concatenate([
            np.array([0.0, 0.0, 0.0, 1.0, 1.0 + 1e-6, -3j, 5.0 + 2j], dtype=np.complex64),
            np.arange(10, 10 + n - 7, dtype=np.complex64),
        ])
        Qn, _ = np.linalg.qr(rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)))
        cases.append((f"repeated_sim_{n}", (Qn @ np.diag(rep_n.astype(np.complex128)) @ Qn.conj().T).astype(np.complex64), None))
        cases.append((f"jordan_4_sim_{n}", _embed_jordan4(n, 2.0 + 1.0j, 210 + n), 4))
        D_bad_n = np.diag(2.0 ** np.linspace(-30, 30, n))
        Mn = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
        cases.append((f"bad_scale_{n}", (D_bad_n @ Mn @ np.linalg.inv(D_bad_n)).astype(np.complex64), None))

    cases.append(("illcond_1e5_33", _illcond_nonnormal(33, 1e5, seed=310), None))
    cases.append(("illcond_1e5_64", _illcond_nonnormal(64, 1e5, seed=311), None))
    Q_n64, _ = np.linalg.qr(rng.normal(size=(64, 64)) + 1j * rng.normal(size=(64, 64)))
    D_n64 = np.diag(np.geomspace(1.0, 1e-5, 64).astype(np.complex128))
    cases.append(("illcond_norm_1e5_64", (Q_n64 @ D_n64 @ Q_n64.conj().T).astype(np.complex64), None))

    for fn in (48, 64):
        Fo = np.diag(np.ones(fn - 1, dtype=np.complex64), 1)
        Fo[fn - 1, 0] = np.float32(0.1)
        cases.append((f"forsythe_n{fn}_eps0.1", Fo, None))

    base64 = (rng.normal(size=(64, 64)) + 1j * rng.normal(size=(64, 64))).astype(np.complex64)
    cases.append(("tiny_1e-30_64", (base64 * np.float32(1e-30)).astype(np.complex64), None))
    cases.append(("huge_1e30_64", (base64 * np.float32(1e30)).astype(np.complex64), None))
    Q64, _ = np.linalg.qr(rng.normal(size=(64, 64)) + 1j * rng.normal(size=(64, 64)))
    cases.append(("huge_2p5e38_64", (Q64 * 2.5e38).astype(np.complex64), None))

    return cases


# ---- Unit tests ----------------------------------------------------------------------------------------------------


def test_hungarian_matcher_exact_against_bruteforce():
    rng = np.random.default_rng(77)
    for n in range(1, 7):
        cost = rng.uniform(0.0, 10.0, size=(n, n))
        r, c = linear_sum_assignment_np(cost)
        best = min(sum(cost[i, perm[i]] for i in range(n)) for perm in itertools.permutations(range(n)))
        assert math.isclose(float(cost[r, c].sum()), best, rel_tol=0.0, abs_tol=1e-12)


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 8, 16, 22, 32])
def test_replica_random_sizes(n):
    A = crandn(3, n, n, seed=100 + n).numpy()
    for b in range(A.shape[0]):
        a = A[b]
        Wp, Vp, info = kr.ceig_kernel(complex_to_planar(a), compute_vectors=True)
        Wvp, Vvp, infov = kr.ceig_kernel(complex_to_planar(a), compute_vectors=False)
        assert float(info[0]) == 0.0 and float(infov[0]) == 0.0
        assert Vvp is None
        assert np.array_equal(Wp, Wvp), "eigvals must be bit-identical to eig"

        w, V = planar_to_complex(Wp, Vp)
        res = eig_residual(a, w, V)
        assert res < 50 * n * EPS32, f"n={n} b={b} residual {res}"

        v_norms = np.linalg.norm(V, axis=0)
        assert np.max(np.abs(v_norms - 1.0)) < 1e-5, f"n={n} b={b} norms {v_norms}"

        w128 = np.linalg.eigvals(a.astype(np.complex128))
        w32 = np.linalg.eigvals(a)
        ew = eig_match_err(w, w128)
        ew32 = eig_match_err(w32, w128)
        assert ew <= max(1e-5, 20.0 * ew32), f"n={n} b={b} ew={ew} cpu32={ew32}"


@pytest.mark.parametrize("n", [33, 48, 64])
def test_replica_big_random_sizes(n):
    A = crandn(2, n, n, seed=100 + n).numpy()
    for b in range(A.shape[0]):
        a = A[b]
        Wp, Vtp, info = kr.ceig_big_kernel(complex_to_planar(a), compute_vectors=True)
        Wvp, Vvp, infov = kr.ceig_big_kernel(complex_to_planar(a), compute_vectors=False)
        assert float(info[0]) == 0.0 and float(infov[0]) == 0.0
        assert Vvp is None
        assert np.array_equal(Wp, Wvp), "eigvals_big must be bit-identical to eig_big"

        Vp = Vtp.reshape(2, n, n).transpose(0, 2, 1).reshape(2 * n, n)
        w, V = planar_to_complex(Wp, Vp)
        res = eig_residual(a, w, V)
        assert res < 50 * n * EPS32, f"n={n} b={b} residual {res}"

        v_norms = np.linalg.norm(V, axis=0)
        assert np.max(np.abs(v_norms - 1.0)) < 1e-5, f"n={n} b={b} norms {v_norms}"

        w128 = np.linalg.eigvals(a.astype(np.complex128))
        w32 = np.linalg.eigvals(a)
        ew = eig_match_err(w, w128)
        ew32 = eig_match_err(w32, w128)
        assert ew <= max(1e-5, 20.0 * ew32), f"n={n} b={b} ew={ew} cpu32={ew32}"


@pytest.mark.parametrize("name,A,jordan_m", _adversarial_cases())
def test_replica_adversarial_classes(name, A, jordan_m):
    n = A.shape[-1]
    if n > 32:
        Wp, Vtp, info = kr.ceig_big_kernel(complex_to_planar(A), compute_vectors=True)
        Wvp, Vvp, infov = kr.ceig_big_kernel(complex_to_planar(A), compute_vectors=False)
        Vp = Vtp.reshape(2, n, n).transpose(0, 2, 1).reshape(2 * n, n)
    else:
        Wp, Vp, info = kr.ceig_kernel(complex_to_planar(A), compute_vectors=True)
        Wvp, Vvp, infov = kr.ceig_kernel(complex_to_planar(A), compute_vectors=False)
    assert float(info[0]) == 0.0 and float(infov[0]) == 0.0
    assert Vvp is None
    assert np.array_equal(Wp, Wvp), f"{name}: eigvals parity mismatch"

    w, V = planar_to_complex(Wp, Vp)
    if name.startswith("subnormal_"):
        assert np.all(w == 0)
        assert np.array_equal(V, np.eye(n, dtype=np.complex64))
        return

    res = eig_residual(A, w, V)
    assert res < 50 * n * EPS32, f"{name}: residual {res}"

    v_norms = np.linalg.norm(V, axis=0)
    assert np.max(np.abs(v_norms - 1.0)) < 1e-5, f"{name}: v_norms {v_norms}"

    w128 = np.linalg.eigvals(A.astype(np.complex128))
    w32 = np.linalg.eigvals(A)
    ew = eig_match_err(w, w128)
    ew32 = eig_match_err(w32, w128)
    if name.startswith("zero_"):
        assert np.all(w == 0)
        assert np.array_equal(V, np.eye(n, dtype=np.complex64))
    elif jordan_m is not None:
        tol = 2.0 * (EPS32 ** (1.0 / jordan_m))
        assert ew <= tol, f"{name}: jordan ew={ew} > {tol}"
    elif name.startswith("lower_tri"):
        # Without cgebal permutation (JOB='P'), lower triangular matrices go through Hessenberg+QR
        # (cpu32 gets 0.0 via permutation). At n=12 error is O(100*n*eps32); at n=64 cond(V)~1e17
        # so 1*eps32 perturbation already shifts eigenvalues by ~0.43 (backward residual stays < 50*n*eps32).
        tol = 1.0 if n > 32 else max(1e-5, 100.0 * n * EPS32, 20.0 * ew32)
        assert ew <= tol, f"{name}: ew={ew} cpu32={ew32}"
    elif name.startswith("frank"):
        # Graded Frank matrices have ill-conditioned small eigenvalues (cond ~ 10^5 at n=16); residual is O(n*eps32).
        assert ew <= 0.02, f"{name}: ew={ew} cpu32={ew32}"
    elif name.startswith("illcond_1e5_"):
        # Non-normal ill-conditioned matrices (cond(V) ~ 1e8) shift eigenvalues by ~0.01-0.05 under 1*eps32.
        assert ew <= 0.1, f"{name}: ew={ew} cpu32={ew32}"
    else:
        assert ew <= max(1e-5, 20.0 * ew32), f"{name}: ew={ew} cpu32={ew32}"


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_replica_nonfinite_poisoning_per_batch(bad):
    A = crandn(3, 6, 6, seed=300).numpy()
    A[0, 2, 4] = complex(bad, 0.0)
    A[2, 1, 3] = complex(0.0, bad)
    planar_batch = torch.from_numpy(np.stack([complex_to_planar(x) for x in A]))
    W, V = kr.ceig_op()(planar_batch)
    Wv = kr.ceigvals_op()(planar_batch)
    assert torch.equal(W.nan_to_num(), Wv.nan_to_num())
    assert torch.isnan(W[0]).all() and torch.isnan(V[0]).all() and torch.isnan(Wv[0]).all()
    assert torch.isnan(W[2]).all() and torch.isnan(V[2]).all() and torch.isnan(Wv[2]).all()
    assert torch.isfinite(W[1]).all() and torch.isfinite(V[1]).all() and torch.isfinite(Wv[1]).all()
    w1, V1 = planar_to_complex(W[1].numpy(), V[1].numpy())
    assert eig_residual(A[1], w1, V1) < 50 * 6 * EPS32


def test_replica_ceig_op_interface_and_batch_dims():
    A = crandn(5, 7, 7, seed=301).numpy()
    planar_batch = torch.from_numpy(np.stack([complex_to_planar(x) for x in A]))
    W, V = kr.ceig_op()(planar_batch)
    Wv = kr.ceigvals_op()(planar_batch)
    assert W.shape == (5, 2, 7) and V.shape == (5, 14, 7) and Wv.shape == (5, 2, 7)
    assert torch.equal(W, Wv)
    W3, V3, info3 = kr.ceig_op(with_info=True)(planar_batch)
    Wv2, info2 = kr.ceigvals_op(with_info=True)(planar_batch)
    assert torch.equal(W3, W) and torch.equal(V3, V) and torch.equal(Wv2, Wv)
    assert info3.shape == (5, 1) and info2.shape == (5, 1) and torch.all(info3 == 0) and torch.all(info2 == 0)


def test_replica_ceig_big_op_interface_and_batch_dims():
    A = crandn(3, 48, 48, seed=302).numpy()
    planar_batch = torch.from_numpy(np.stack([complex_to_planar(x) for x in A]))
    W, V_t = kr.ceig_big_op()(planar_batch)
    Wv = kr.ceigvals_big_op()(planar_batch)
    assert W.shape == (3, 2, 48) and V_t.shape == (3, 96, 48) and Wv.shape == (3, 2, 48)
    assert torch.equal(W, Wv)
    W3, V3, info3 = kr.ceig_big_op(with_info=True)(planar_batch)
    Wv2, info2 = kr.ceigvals_big_op(with_info=True)(planar_batch)
    assert torch.equal(W3, W) and torch.equal(V3, V_t) and torch.equal(Wv2, Wv)
    assert info3.shape == (3, 1) and info2.shape == (3, 1) and torch.all(info3 == 0) and torch.all(info2 == 0)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_replica_big_nonfinite_poisoning_per_batch(bad):
    A = crandn(3, 48, 48, seed=303).numpy()
    A[0, 10, 35] = complex(bad, 0.0)
    A[2, 33, 12] = complex(0.0, bad)
    planar_batch = torch.from_numpy(np.stack([complex_to_planar(x) for x in A]))
    W, V_t = kr.ceig_big_op()(planar_batch)
    Wv = kr.ceigvals_big_op()(planar_batch)
    assert torch.equal(W.nan_to_num(), Wv.nan_to_num())
    assert torch.isnan(W[0]).all() and torch.isnan(V_t[0]).all() and torch.isnan(Wv[0]).all()
    assert torch.isnan(W[2]).all() and torch.isnan(V_t[2]).all() and torch.isnan(Wv[2]).all()
    assert torch.isfinite(W[1]).all() and torch.isfinite(V_t[1]).all() and torch.isfinite(Wv[1]).all()
    V1_p = V_t[1].numpy().reshape(2, 48, 48).transpose(0, 2, 1).reshape(96, 48)
    w1, V1 = planar_to_complex(W[1].numpy(), V1_p)
    assert eig_residual(A[1], w1, V1) < 50 * 48 * EPS32


# ---- hpu_ops.tpc.complex_linalg API tests (run automatically once cl.eig / cl.eigvals exist) ----------------------


def test_cl_eig_constants_preserved(cl_eig_replica):
    assert cl.CEIG_MAXN == 64
    assert cl.COMPLEX_MAXN == 32
    with pytest.raises(ValueError):
        cl.inv(torch.zeros(1, 33, 33, 2))
    with pytest.raises(ValueError):
        cl.eigh(torch.zeros(1, 33, 33, 2))


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 8, 16, 22, 32, 33, 48, 64])
def test_cl_eig_sizes_and_layouts(cl_eig_replica, n):
    bs = 1 if n > 32 else 3
    A = crandn(bs, n, n, seed=400 + n)
    W_p, V_p = cl.eig(pairs(A), layout="pairs")
    assert W_p.shape == (bs, n, 2) and V_p.shape == (bs, n, n, 2)
    Wv_p = cl.eigvals(pairs(A), layout="pairs")
    assert Wv_p.shape == (bs, n, 2)
    assert torch.equal(W_p, Wv_p)

    # Interleaved layout
    Ai = pairs(A).reshape(bs, n, 2 * n)
    W_i, V_i = cl.eig(Ai, layout="interleaved")
    Wv_i = cl.eigvals(Ai, layout="interleaved")
    assert W_i.shape == (bs, 2 * n) and V_i.shape == (bs, n, 2 * n) and Wv_i.shape == (bs, 2 * n)
    assert torch.equal(W_i, W_p.reshape(bs, 2 * n))
    assert torch.equal(V_i, V_p.reshape(bs, n, 2 * n))
    assert torch.equal(Wv_i, Wv_p.reshape(bs, 2 * n))

    # complex64 tensor input directly
    W_c, V_c = cl.eig(A)
    assert torch.equal(W_c, W_p) and torch.equal(V_c, V_p)
    assert torch.equal(cl.eigvals(A), Wv_p)

    # Accuracy & unit norms
    w_c = cplx(W_p).numpy()
    V_c_np = cplx(V_p).numpy()
    for b in range(bs):
        a_np = A[b].numpy()
        assert eig_residual(a_np, w_c[b], V_c_np[b]) < 50 * n * EPS32
        assert np.max(np.abs(np.linalg.norm(V_c_np[b], axis=0) - 1.0)) < 1e-5
        w128 = np.linalg.eigvals(a_np.astype(np.complex128))
        w32 = np.linalg.eigvals(a_np)
        assert eig_match_err(w_c[b], w128) <= max(1e-5, 20.0 * eig_match_err(w32, w128))


def test_cl_eig_unbatched_multidim_empty_and_views(cl_eig_replica):
    # Unbatched (n, n)
    A1 = crandn(6, 6, seed=500)
    w_u, V_u = cl.eig(pairs(A1))
    assert w_u.shape == (6, 2) and V_u.shape == (6, 6, 2)
    assert torch.equal(cl.eigvals(pairs(A1)), w_u)
    assert eig_residual(A1.numpy(), cplx(w_u).numpy(), cplx(V_u).numpy()) < 50 * 6 * EPS32

    # Multi-dim batch (2, 3, 6, 6)
    Am = crandn(2, 3, 6, 6, seed=501)
    Wm, Vm = cl.eig(pairs(Am))
    assert Wm.shape == (2, 3, 6, 2) and Vm.shape == (2, 3, 6, 6, 2)
    assert torch.equal(cl.eigvals(pairs(Am)), Wm)

    # Empty batch (0, 6, 6)
    We, Ve = cl.eig(pairs(Am)[0, :0])
    assert We.shape == (0, 6, 2) and Ve.shape == (0, 6, 6, 2)
    assert cl.eigvals(pairs(Am)[0, :0]).shape == (0, 6, 2)
    We_i, Ve_i = cl.eig(pairs(Am)[0, :0].reshape(0, 6, 12), layout="interleaved")
    assert We_i.shape == (0, 12) and Ve_i.shape == (0, 6, 12)

    # Non-contiguous / strided / .mT / .conj() views
    A4 = crandn(4, 6, 6, seed=502)
    P4 = torch.view_as_real(A4)
    W_s, V_s = cl.eig(P4[::2])
    W_s_ref, V_s_ref = cl.eig(P4[::2].contiguous())
    assert torch.equal(W_s, W_s_ref) and torch.equal(V_s, V_s_ref)

    W_t, V_t = cl.eig(A4.mT)
    W_t_ref, V_t_ref = cl.eig(A4.mT.contiguous())
    assert torch.equal(W_t, W_t_ref) and torch.equal(V_t, V_t_ref)

    W_c, V_c = cl.eig(A4.conj())
    W_c_ref, V_c_ref = cl.eig(A4.conj().contiguous())
    assert torch.equal(W_c, W_c_ref) and torch.equal(V_c, V_c_ref)


def test_replica_clartg_underflow_and_pow2_e254():
    cs, snr, sni, rr, ri = kr._clartg(np.float32(1e-20), np.float32(0.0), np.float32(1.0), np.float32(0.0))
    assert float(cs) == 0.0
    assert math.isclose(float(snr) ** 2 + float(sni) ** 2, 1.0, rel_tol=1e-6)
    assert math.isclose(float(rr), 1.0, rel_tol=1e-6) and float(ri) == 0.0

    for val in (1.71e38, 2.5e38, float(np.finfo(np.float32).max)):
        m = np.float32(val)
        sc = kr._ceig_pow2_scale(m)
        unsc = kr._ceig_pow2_unscale(m)
        assert float(sc) > 0.0 and math.isfinite(float(sc))
        assert float(sc * unsc) == 1.0
        assert 2.0 <= float(m * sc) < 4.0


def test_replica_rsqrt_fine_all_exponents_and_clartg_e1():
    for exp in range(1, 255):
        for mant in (0, 0x200000, 0x7FFFFF):
            x = kr._f32((exp << 23) | mant)
            got = float(kr._ceig_rsqrt_fine(x))
            exact = 1.0 / math.sqrt(float(x))
            assert math.isfinite(got) and abs(got - exact) / exact < 1e-6, (exp, float(x), got, exact)

    for fv in (1.1e-19, 1.2e-19, 1.3e-19, 1.4e-19, 1.5e-19, 1.2e-19 + 0.3e-19j):
        cs, snr, sni, rr, ri = kr._clartg(np.float32(fv.real), np.float32(fv.imag), np.float32(1.0), np.float32(0.0))
        assert 0.0 <= float(cs) <= 1.0 + 1e-6
        assert math.isclose(float(cs) ** 2 + float(snr) ** 2 + float(sni) ** 2, 1.0, rel_tol=1e-6)
        assert math.isclose(math.hypot(float(rr), float(ri)), 1.0, rel_tol=1e-6)

        cs_g, snr_g, sni_g, rr_g, ri_g = kr._clartg(
            np.float32(1.0), np.float32(0.0), np.float32(fv.real), np.float32(fv.imag)
        )
        assert 0.0 <= float(cs_g) <= 1.0 + 1e-6
        assert math.isclose(float(cs_g) ** 2 + float(snr_g) ** 2 + float(sni_g) ** 2, 1.0, rel_tol=1e-6)
        assert math.isclose(math.hypot(float(rr_g), float(ri_g)), 1.0, rel_tol=1e-6)


def test_cl_eig_invalid_inputs_raise(cl_eig_replica):
    with pytest.raises(ValueError):
        cl.eig(torch.zeros(4, dtype=torch.complex64))
    with pytest.raises(ValueError):
        cl.eigvals(torch.zeros(4, dtype=torch.complex64))
    with pytest.raises(ValueError):
        cl.eig(torch.zeros((), dtype=torch.complex64))
    with pytest.raises(ValueError):
        cl.eig(torch.zeros(4, 2))
    with pytest.raises(ValueError):
        cl.eig(torch.zeros(2, 0, 0, 2))
    with pytest.raises(ValueError):
        cl.eigvals(torch.zeros(2, 0, 0, 2))
    with pytest.raises(ValueError):
        cl.eig(torch.zeros(1, 65, 65, 2))
    with pytest.raises(ValueError):
        cl.eigvals(torch.zeros(1, 65, 65, 2))
    with pytest.raises(ValueError):
        cl.eig(torch.zeros(1, 65, 65, dtype=torch.complex64))
    with pytest.raises(ValueError):
        cl.eigvals(torch.zeros(1, 65, 65, dtype=torch.complex64))
    with pytest.raises(ValueError):
        cl.eig(torch.zeros(2, 3, 3, 2, dtype=torch.float64))
    with pytest.raises(ValueError):
        cl.eig(torch.zeros(2, 3, 3, dtype=torch.complex128))
    with pytest.raises(ValueError):
        cl.eigvals(torch.zeros(2, 3, 3, dtype=torch.complex128))
    with pytest.raises(ValueError):
        cl.eig(torch.zeros(2, 3, 4, 2))
    with pytest.raises(ValueError):
        cl.eig(torch.zeros(2, 3, 5), layout="interleaved")


@pytest.mark.parametrize("bad", [math.nan, math.inf])
def test_cl_eig_nonfinite(cl_eig_replica, bad):
    A = crandn(2, 5, 5, seed=503)
    A[0, 2, 3] = complex(bad, 0.0)
    W, V = cl.eig(pairs(A))
    Wv = cl.eigvals(pairs(A))
    assert torch.isnan(W[0]).all() and torch.isnan(V[0]).all() and torch.isnan(Wv[0]).all()
    assert torch.isfinite(W[1]).all() and torch.isfinite(V[1]).all() and torch.isfinite(Wv[1]).all()


@pytest.mark.parametrize("n", [33, 48, 64])
def test_cl_eig_real_float32_pairs(cl_eig_replica, n):
    g = torch.Generator().manual_seed(600 + n)
    R = torch.randn(2, n, n, generator=g, dtype=torch.float32)
    P_real = torch.stack([R, torch.zeros_like(R)], dim=-1)
    W_p, V_p = cl.eig(P_real)
    Wv_p = cl.eigvals(P_real)
    assert torch.equal(W_p, Wv_p)
    w_c = cplx(W_p).numpy()
    V_c = cplx(V_p).numpy()
    for b in range(2):
        a_np = R[b].numpy().astype(np.complex64)
        assert eig_residual(a_np, w_c[b], V_c[b]) < 50 * n * EPS32
        assert np.max(np.abs(np.linalg.norm(V_c[b], axis=0) - 1.0)) < 1e-5
        w128 = np.linalg.eigvals(R[b].double().numpy())
        w32 = np.linalg.eigvals(R[b].numpy())
        assert eig_match_err(w_c[b], w128) <= max(1e-5, 20.0 * eig_match_err(w32, w128))


def test_cl_eig_big_unbatched_multidim_empty_views_and_nonfinite(cl_eig_replica):
    A1 = crandn(33, 33, seed=700)
    w_u, V_u = cl.eig(pairs(A1))
    assert w_u.shape == (33, 2) and V_u.shape == (33, 33, 2)
    assert torch.equal(cl.eigvals(pairs(A1)), w_u)
    assert eig_residual(A1.numpy(), cplx(w_u).numpy(), cplx(V_u).numpy()) < 50 * 33 * EPS32

    Am = crandn(2, 1, 33, 33, seed=701)
    Wm, Vm = cl.eig(pairs(Am))
    assert Wm.shape == (2, 1, 33, 2) and Vm.shape == (2, 1, 33, 33, 2)
    assert torch.equal(cl.eigvals(pairs(Am)), Wm)

    We, Ve = cl.eig(pairs(Am)[0, :0])
    assert We.shape == (0, 33, 2) and Ve.shape == (0, 33, 33, 2)
    assert cl.eigvals(pairs(Am)[0, :0]).shape == (0, 33, 2)

    A4 = crandn(4, 33, 33, seed=702)
    P4 = torch.view_as_real(A4)
    W_s, V_s = cl.eig(P4[::2])
    W_s_ref, V_s_ref = cl.eig(P4[::2].contiguous())
    assert torch.equal(W_s, W_s_ref) and torch.equal(V_s, V_s_ref)

    W_t, V_t = cl.eig(A4[:2].mT)
    W_t_ref, V_t_ref = cl.eig(A4[:2].mT.contiguous())
    assert torch.equal(W_t, W_t_ref) and torch.equal(V_t, V_t_ref)

    A3 = A4[:3].clone()
    A3[0, 20, 10] = complex(math.nan, 0.0)
    Wn, Vn = cl.eig(pairs(A3))
    Wvn = cl.eigvals(pairs(A3))
    assert torch.isnan(Wn[0]).all() and torch.isnan(Vn[0]).all() and torch.isnan(Wvn[0]).all()
    assert torch.isfinite(Wn[1:]).all() and torch.isfinite(Vn[1:]).all() and torch.isfinite(Wvn[1:]).all()
