"""complex64 linear algebra from the Gaudi2 TPC kernels, for hpu-complex.

A complex matrix is a float32 tensor of (re, im) pairs: ``(..., n, n, 2)`` (``torch.view_as_real`` of a complex64
tensor, ``layout="pairs"``, the default) or the same memory as ``(..., n, 2n)`` (``layout="interleaved"``: row i is
re0, im0, re1, im1, ...). A complex64 tensor is also taken (viewed as pairs). Results come back as float32 pairs in
the input's layout; nothing runs on the CPU.

- ``cgj_solve`` / ``cgj_inv`` (``cgj_solve_f32_gaudi2.c``): Gauss-Jordan with partial pivoting by
  cabs1 = |re| + |im| (LAPACK icamax, as MPS's complex LU), n <= ``COMPLEX_MAXN``. One launch, plus one transpose
  in (pairs to planes) and one out.
- ``eigh`` / ``eigvalsh`` (``jacobi_ceigh_f32_gaudi2.c``, two launches): complex one-sided Jacobi on the shifted
  Hermitian matrix, then the adaptive-shift refinement pass (the real ``jacobi_eigh``'s structure; the rotation makes
  the rows' inner product real with a phase, as XLA's eigh_expander and MPS's Eigh kernel), n <= ``COMPLEX_MAXN``.
- ``svd`` / ``svdvals`` (``jacobi_csvd_f32_gaudi2.c``, one launch): complex one-sided Jacobi SVD, thin factors,
  m, n <= ``COMPLEX_MAXN``.
- ``solve``, ``inv``, ``slogdet``, ``det`` on top of the Gauss-Jordan kernels. The phase of the determinant is computed in the kernel as
  (-1)^swaps * prod(u_kk / |u_kk|) (torch's ``lu_det_P`` times ``sgn(diag U)``, JAX's ``_slogdet_lu``), each factor
  normalized, the product renormalized once; logabsdet = sum(log |u_kk|) (one log and one sum on the card), det =
  phase * prod |u_kk|. The rsqrt is unbiased and |u_kk| gets one residual step, so exact inputs (the identity, i times
  a permutation, small integers) give an exact sign, logabsdet and det.

Semantics: torch's on the finite range. A singular matrix (an exactly zero pivot) gives sign 0 and logabsdet -inf,
a zero row of X for that pivot and ``info`` = 1 + its index; any NaN or infinity in A (or B) gives NaN everywhere
(torch gives NaN or inf there). Subnormal inputs count as zero (the card flushes them).
"""
import torch

from . import _call, _fresh, lazy_mode  # noqa: F401  (lazy_mode: re-exported for the callers' checks)
from . import _load
from ..extrema import internal_sort

COMPLEX_MAXN = 32  # MAXN of the complex kernels: a row of each plane is one 64-lane vector, 4 planes in 32 KB


def _cgj_op():
    """``torch.ops.linalg_tpc.cgj_solve``, built on first use."""
    _load()
    return torch.ops.linalg_tpc.cgj_solve


def _ceigh_op():
    """``torch.ops.linalg_tpc.jacobi_ceigh`` (pass 1), built on first use."""
    _load()
    return torch.ops.linalg_tpc.jacobi_ceigh


def _ceigh_refine_op():
    """``torch.ops.linalg_tpc.jacobi_ceigh_refine`` (pass 2), built on first use."""
    _load()
    return torch.ops.linalg_tpc.jacobi_ceigh_refine


def _csvd_op():
    """``torch.ops.linalg_tpc.jacobi_csvd``, built on first use."""
    _load()
    return torch.ops.linalg_tpc.jacobi_csvd


def _csvdvals_op():
    """``torch.ops.linalg_tpc.jacobi_csvdvals``, built on first use."""
    _load()
    return torch.ops.linalg_tpc.jacobi_csvdvals


def _cgj_inv_op():
    """``torch.ops.linalg_tpc.cgj_inv``, built on first use."""
    _load()
    return torch.ops.linalg_tpc.cgj_inv


def _ceig_op():
    """``torch.ops.linalg_tpc.ceig``, built on first use."""
    _load()
    return torch.ops.linalg_tpc.ceig


def _ceigvals_op():
    """``torch.ops.linalg_tpc.ceigvals``, built on first use."""
    _load()
    return torch.ops.linalg_tpc.ceigvals


def _cqr_op():
    """``torch.ops.linalg_tpc.cqr``, built on first use."""
    _load()
    return torch.ops.linalg_tpc.cqr


def _cqr_complete_op():
    """``torch.ops.linalg_tpc.cqr_complete``, built on first use."""
    _load()
    return torch.ops.linalg_tpc.cqr_complete


def _cqr_r_op():
    """``torch.ops.linalg_tpc.cqr_r``, built on first use."""
    _load()
    return torch.ops.linalg_tpc.cqr_r


# ---- layouts -------------------------------------------------------------------------------------------------------


def as_pairs(A, layout="pairs"):
    """``A`` as a float32 ``(..., r, c, 2)`` tensor: complex input viewed as pairs; ``"interleaved"`` ``(..., r, 2c)``
    viewed as pairs (a copy only when the last dim is strided)."""
    if A.is_complex():
        if A.dtype != torch.complex64:
            raise ValueError(f"hpu_ops.tpc.complex takes complex64, got {A.dtype}")
        return torch.view_as_real(A.resolve_conj().resolve_neg())
    if A.dtype != torch.float32:
        raise ValueError(f"hpu_ops.tpc.complex takes float32 (re, im) pairs or complex64, got {A.dtype}")
    if layout == "pairs":
        if A.dim() < 3 or A.shape[-1] != 2:
            raise ValueError(f"layout='pairs' takes (..., r, c, 2), got {tuple(A.shape)}")
        return A
    if layout == "interleaved":
        if A.dim() < 2 or A.shape[-1] % 2:
            raise ValueError(f"layout='interleaved' takes (..., r, 2c), got {tuple(A.shape)}")
        return A.reshape(*A.shape[:-1], A.shape[-1] // 2, 2)
    raise ValueError(f"layout is 'pairs' or 'interleaved', got {layout!r}")


def from_pairs(P, layout="pairs"):
    """``(..., r, c, 2)`` pairs back in ``layout``."""
    return P if layout == "pairs" else P.reshape(*P.shape[:-2], P.shape[-2] * 2)


def _planar(P, batch):
    """Pairs ``(*batch, r, c, 2)`` -> the kernels' planar ``(b, 2r, c)``: rows 0..r-1 the real parts, r..2r-1 the
    imaginary parts (one transpose on the card)."""
    r, c = P.shape[-3], P.shape[-2]
    P = P.expand(*batch, r, c, 2).reshape(-1, r, c, 2)
    return _fresh(P.permute(0, 3, 1, 2).reshape(-1, 2 * r, c).contiguous())


def _unplanar(X, batch, r, c):
    """Planar ``(b, 2r, c)`` -> pairs ``(*batch, r, c, 2)`` (one transpose)."""
    return X.reshape(-1, 2, r, c).permute(0, 2, 3, 1).contiguous().reshape(*batch, r, c, 2)


def _planar_t(P, batch):
    """Pairs ``(*batch, r, c, 2)`` -> column-contiguous planar ``(b, 2c, r)`` (TPC ``[r, 2c, b]``):
    planes 0..c-1 the real columns, c..2c-1 the imaginary columns."""
    r, c = P.shape[-3], P.shape[-2]
    P = P.expand(*batch, r, c, 2).reshape(-1, r, c, 2)
    return _fresh(P.permute(0, 3, 2, 1).reshape(-1, 2 * c, r).contiguous())


def _unplanar_t(X, batch, r, c):
    """Column-contiguous planar ``(b, 2c, r)`` -> pairs ``(*batch, r, c, 2)``."""
    return X.reshape(-1, 2, c, r).permute(0, 3, 2, 1).contiguous().reshape(*batch, r, c, 2)


def _empty(batch):
    return any(b == 0 for b in batch)


def _square(P, limit=COMPLEX_MAXN):
    if P.dim() < 3:
        raise ValueError(f"hpu_ops.tpc.complex takes (..., n, n) matrices, got pairs {tuple(P.shape)}")
    n = P.shape[-2]
    if P.shape[-3] != n or n > limit or n < 1:
        raise ValueError(f"hpu_ops.tpc.complex takes square A with 1 <= n <= {limit}, got pairs {tuple(P.shape)}")
    return n


# ---- Gauss-Jordan ----------------------------------------------------------------------------------------------------


def cgj_solve(A, B, layout="pairs"):
    """``(X, D, S)`` for complex ``A (..., n, n)`` and ``B (..., n, m)`` (n <= ``COMPLEX_MAXN``, any m, broadcast
    batch): ``X = A^-1 B`` in ``layout``; ``D (..., 3, n)`` the pivots' re, im and modulus (U's diagonal of
    P A = L U); ``S (..., 4)`` = (phase re, phase im, info, (-1)^swaps), info = 1 + the first zero pivot (0 if none).
    B's columns go through the kernel in chunks of ``COMPLEX_MAXN``; D and S describe A and come from the first
    chunk. A non-finite entry anywhere in A or B makes that batch member's X, D and S all NaN, whatever chunk it is
    in (each launch poisons only its own chunk, so the flag is OR-ed across the chunks here)."""
    PA, PB = as_pairs(A, layout), as_pairs(B, layout)
    n = _square(PA)
    if PB.shape[-3] != n:
        raise ValueError(f"hpu_ops.tpc.cgj_solve takes B with n rows, got pairs {tuple(PA.shape)} and {tuple(PB.shape)}")
    m = PB.shape[-2]
    batch = torch.broadcast_shapes(PA.shape[:-3], PB.shape[:-3])
    if _empty(batch) or m == 0:  # no launch for an empty batch; D and S still describe A when only m is 0
        X = PB.new_zeros(*batch, n, m, 2)
        if _empty(batch):
            return from_pairs(X, layout), PA.new_zeros(*batch, 3, n), PA.new_zeros(*batch, 4)
        _, D, S = cgj_inv(PA.expand(*batch, n, n, 2))
        return from_pairs(X, layout), D, S
    a = _planar(PA, batch)
    op = _cgj_op()
    parts = [_call(op, a, _planar(chunk, batch)) for chunk in PB.split(COMPLEX_MAXN, -2)]
    X = parts[0][0] if len(parts) == 1 else torch.cat([p[0].reshape(-1, 2, n, p[0].shape[-1]) for p in parts],
                                                       -1).reshape(-1, 2 * n, m)
    D, S = parts[0][1], parts[0][2]
    if len(parts) > 1:  # a NaN/inf in a later chunk of B: poison X, D and S of that member, as one launch would
        bad = torch.stack([p[2] for p in parts]).isnan().any(0).any(-1)
        nan = X.new_full((), float("nan"))
        X = torch.where(bad[:, None, None], nan, X)
        D = torch.where(bad[:, None, None], nan, D)
        S = torch.where(bad[:, None], nan, S)
    return from_pairs(_unplanar(X, batch, n, m), layout), D.reshape(*batch, 3, n), S.reshape(*batch, 4)


def cgj_inv(A, layout="pairs"):
    """``(A^-1, D, S)`` for complex ``A (..., n, n)``: ``cgj_solve`` against the identity, built in the kernel."""
    PA = as_pairs(A, layout)
    n = _square(PA)
    batch = PA.shape[:-3]
    if _empty(batch):
        return from_pairs(PA.new_zeros(PA.shape), layout), PA.new_zeros(*batch, 3, n), PA.new_zeros(*batch, 4)
    X, D, S = _call(_cgj_inv_op(), _planar(PA, batch))
    return from_pairs(_unplanar(X, batch, n, n), layout), D.reshape(*batch, 3, n), S.reshape(*batch, 4)


def solve(A, B, layout="pairs"):
    """``torch.linalg.solve`` for complex A (n <= ``COMPLEX_MAXN``) and B ``(..., n, m)``, as pairs."""
    return cgj_solve(A, B, layout)[0]


def inv(A, layout="pairs"):
    """``torch.linalg.inv`` for complex A (n <= ``COMPLEX_MAXN``), as pairs."""
    return cgj_inv(A, layout)[0]


def slogdet(A, layout="pairs"):
    """``(sign, logabsdet)`` of ``torch.linalg.slogdet`` for complex A: ``sign (..., 2)`` the unit phase as a
    (re, im) pair (0 for a singular A), ``logabsdet (...)`` float32 (-inf for a singular A)."""
    PA = as_pairs(A, layout)
    _square(PA)
    batch = PA.shape[:-3]
    if _empty(batch):
        return PA.new_zeros(*batch, 2), PA.new_zeros(batch)
    _, D, S = _call(_cgj_inv_op(), _planar(PA, batch))  # X (A^-1) is not read: no transpose back
    return S[:, :2].reshape(*batch, 2), D[:, 2, :].log().sum(-1).reshape(batch)


def det(A, layout="pairs"):
    """``torch.linalg.det`` for complex A as a ``(..., 2)`` pair: phase * prod |u_kk| (as torch's product of U's
    diagonal: exact for exact pivots, where exp(sum log |u_kk|) is not; it over/underflows where torch's does)."""
    PA = as_pairs(A, layout)
    _square(PA)
    batch = PA.shape[:-3]
    if _empty(batch):
        return PA.new_zeros(*batch, 2)
    _, D, S = _call(_cgj_inv_op(), _planar(PA, batch))
    return (S[:, :2] * D[:, 2, :].prod(-1)[:, None]).reshape(*batch, 2)


def cgj_reference(A, B=None, layout="pairs"):
    """The kernel's contract in complex128 torch (for the CPU tests): ``(X, D, S)`` with the pivots of
    ``torch.linalg.lu_factor`` (LAPACK's pivot rule is |re| + |im| as well) and the sign of the permutation."""
    PA = as_pairs(A, layout).double()
    Z = torch.view_as_complex(PA.contiguous())
    n = Z.shape[-1]
    LU, piv = torch.linalg.lu_factor(Z)
    d = LU.diagonal(dim1=-2, dim2=-1)
    swaps = (piv != torch.arange(1, n + 1, dtype=piv.dtype)).sum(-1)
    perm_sign = (1 - 2 * (swaps % 2)).double()
    phase = torch.sgn(d).prod(-1) * perm_sign
    X = torch.linalg.solve(Z, torch.eye(n, dtype=Z.dtype) if B is None else
                           torch.view_as_complex(as_pairs(B, layout).double().contiguous()))
    D = torch.stack([d.real, d.imag, d.abs()], -2)
    S = torch.stack([phase.real, phase.imag, torch.zeros_like(perm_sign), perm_sign], -1)
    return torch.view_as_real(X), D, S


# ---- Jacobi: eigh, svd ---------------------------------------------------------------------------------------------


def _conj_t_planar(X, r, c):
    """Planar ``(b, 2r, c)`` of M -> planar ``(b, 2c, r)`` of M^H (a transpose and a negation of the imaginary plane)."""
    X = X.reshape(-1, 2, r, c).transpose(-1, -2)
    return torch.cat([X[:, :1], -X[:, 1:]], 1).reshape(-1, 2 * c, r).contiguous()


def _eigh_raw(PA, upper):
    """``(W [b, n] unsorted, rows [b, 2, n, n])`` of the two kernel passes; rows[:, :, j] = x_j^H, planar."""
    n = _square(PA)
    a = _planar(PA, PA.shape[:-3])
    if upper:  # the lower triangle of A^H is the conjugate of A's upper triangle
        a = _fresh(_conj_t_planar(a, n, n))
    V1, H = _call(_ceigh_op(), a)
    W, V = _call(_ceigh_refine_op(), H, V1)
    return W, V.reshape(-1, 2, n, n)


def _upper(UPLO):
    """``True`` for ``UPLO`` = "U"/"u", ``False`` for "L"/"l"; anything else raises as ``torch.linalg.eigh`` does."""
    if not isinstance(UPLO, str) or UPLO.upper() not in ("L", "U"):
        raise RuntimeError(f"Expected UPLO argument to be 'L' or 'U', but got {UPLO}")
    return UPLO.upper() == "U"


def eigh(A, UPLO="L", layout="pairs"):
    """``torch.linalg.eigh`` for a complex Hermitian A (n <= ``COMPLEX_MAXN``) held in the ``UPLO`` triangle (the
    imaginary parts of the diagonal are ignored, as LAPACK's heevd): ``(w (...,  n) ascending float32, V)`` with V's
    columns the eigenvectors as pairs (``(..., n, n, 2)`` or interleaved); each column's phase is arbitrary, as in
    torch. Two kernel launches, a sort and a gather on the card."""
    upper = _upper(UPLO)
    PA = as_pairs(A, layout)
    n = _square(PA)
    batch = PA.shape[:-3]
    if _empty(batch):
        return PA.new_zeros(*batch, n), from_pairs(PA.new_zeros(PA.shape), layout)
    if n == 1:  # trivial (and kernels with no rotation at all are not worth a launch)
        w = PA[..., 0, 0, 0].reshape(*batch, 1).clone()
        V = torch.stack([torch.ones_like(w), torch.zeros_like(w)], -1)[..., None, :]
        return w, from_pairs(V, layout)
    W, R = _eigh_raw(PA, upper)
    w, order = internal_sort(W, -1)
    R = R.gather(2, order[:, None, :, None].expand_as(R))  # rows j sorted
    V = torch.stack([R[:, 0].mT, -R[:, 1].mT], -1)  # V[i, j] = conj(x_j^H[i])
    return w.reshape(*batch, n), from_pairs(V.reshape(*batch, n, n, 2), layout)


def eigvalsh(A, UPLO="L", layout="pairs"):
    """``torch.linalg.eigvalsh`` for a complex Hermitian A: the eigenvalues of ``eigh``, ascending."""
    upper = _upper(UPLO)
    PA = as_pairs(A, layout)
    n = _square(PA)
    if _empty(PA.shape[:-3]):
        return PA.new_zeros(*PA.shape[:-3], n)
    if n == 1:
        return PA[..., 0, 0, 0].reshape(*PA.shape[:-3], 1).clone()
    W, _ = _eigh_raw(PA, upper)
    return internal_sort(W, -1)[0].reshape(*PA.shape[:-3], n)


def _svd_input(A, layout):
    P = as_pairs(A, layout)
    m, n = P.shape[-3], P.shape[-2]
    if max(m, n) > COMPLEX_MAXN or min(m, n) < 1:
        raise ValueError(f"hpu_ops.tpc.complex_linalg.svd takes 1 <= m, n <= {COMPLEX_MAXN}, got pairs {tuple(P.shape)}")
    a = _planar(P, P.shape[:-3])
    wide = m <= n  # orthogonalize the rows of A (k = m), else those of A^H (k = n)
    return P, m, n, wide, (a if wide else _fresh(_conj_t_planar(a, m, n)))


def svd(A, full_matrices=False, layout="pairs"):
    """``torch.linalg.svd`` for a complex A (m, n <= ``COMPLEX_MAXN``): ``(U, S, Vh)`` with U ``(..., m, k)`` and Vh
    ``(..., k, n)`` as pairs (k = min(m, n)), S descending float32. ``full_matrices=True`` is taken for square A only
    (it is the thin SVD there); for non-square A it raises NotImplementedError (no null-space completion of the
    longer side yet). One kernel launch, a sort and gathers on the card."""
    P, m, n, wide, a = _svd_input(A, layout)
    if full_matrices and m != n:
        raise NotImplementedError("hpu_ops.tpc.complex_linalg.svd: full_matrices=True for non-square A")
    batch, k = P.shape[:-3], min(m, n)
    big = max(m, n)
    if _empty(batch):
        return (from_pairs(P.new_zeros(*batch, m, k, 2), layout), P.new_zeros(*batch, k),
                from_pairs(P.new_zeros(*batch, k, n, 2), layout))
    Wn, S, J = _call(_csvd_op(), a)
    Wn = Wn.reshape(-1, 2, k, big)
    J = J.reshape(-1, 2, k, k)
    S, order = internal_sort(S, -1, descending=True)
    Wn = Wn.gather(2, order[:, None, :, None].expand_as(Wn))
    J = J.gather(2, order[:, None, :, None].expand_as(J))
    # A_t = J^H S Wn: wide (A_t = A): U = J^H, Vh = Wn; tall (A_t = A^H): U = Wn^H, Vh = J
    X, Y = (J, Wn) if wide else (Wn, J)
    U = torch.stack([X[:, 0].mT, -X[:, 1].mT], -1)
    Vh = torch.stack([Y[:, 0], Y[:, 1]], -1)
    return (from_pairs(U.reshape(*batch, m, k, 2), layout), S.reshape(*batch, k),
            from_pairs(Vh.reshape(*batch, k, n, 2), layout))


def svdvals(A, layout="pairs"):
    """``torch.linalg.svdvals`` for a complex A (m, n <= ``COMPLEX_MAXN``): descending float32, from the kernel without
    U and V (no polish sweeps there)."""
    P, m, n, _, a = _svd_input(A, layout)
    if _empty(P.shape[:-3]):
        return P.new_zeros(*P.shape[:-3], min(m, n))
    S = _call(_csvdvals_op(), a)
    return internal_sort(S, -1, descending=True)[0].reshape(*P.shape[:-3], min(m, n))


# ---- general eig, eigvals (MPS recipe) -----------------------------------------------------------------------------


def eig(A, layout="pairs"):
    """``torch.linalg.eig`` for a general complex64 matrix A (``1 <= n <= COMPLEX_MAXN``): returns
    ``(eigenvalues, eigenvectors)`` as float32 pairs in ``layout`` (``(..., n, 2)`` / ``(..., 2n)`` and
    ``(..., n, n, 2)`` / ``(..., n, 2n)``). Column ``k`` of ``eigenvectors`` has unit 2-norm and satisfies
    ``A V = V diag(W)``. Non-finite inputs produce ``NaN`` outputs."""
    PA = as_pairs(A, layout)
    n = _square(PA)
    batch = PA.shape[:-3]
    if _empty(batch):
        return from_pairs(PA.new_zeros(*batch, n, 2), layout), from_pairs(PA.new_zeros(PA.shape), layout)
    if n == 1:
        w = PA[..., 0, 0, :].reshape(*batch, 1, 2).clone()
        bad = ~torch.isfinite(w).all(-1, keepdim=True)
        w = torch.where(bad, float("nan"), w)
        v = torch.stack([torch.ones_like(w[..., :1]), torch.zeros_like(w[..., :1])], -1)
        v = torch.where(bad[..., None], float("nan"), v)
        return from_pairs(w, layout), from_pairs(v, layout)
    W_raw, V_raw, _ = _call(_ceig_op(), _planar(PA, batch))
    W = W_raw.transpose(-1, -2).contiguous().reshape(*batch, n, 2)
    V = _unplanar(V_raw, batch, n, n)
    return from_pairs(W, layout), from_pairs(V, layout)


def eigvals(A, layout="pairs"):
    """``torch.linalg.eigvals`` for a general complex64 matrix A (``1 <= n <= COMPLEX_MAXN``): returns
    ``eigenvalues`` as float32 pairs in ``layout`` (``(..., n, 2)`` or ``(..., 2n)``), without accumulating Q or V."""
    PA = as_pairs(A, layout)
    n = _square(PA)
    batch = PA.shape[:-3]
    if _empty(batch):
        return from_pairs(PA.new_zeros(*batch, n, 2), layout)
    if n == 1:
        w = PA[..., 0, 0, :].reshape(*batch, 1, 2).clone()
        bad = ~torch.isfinite(w).all(-1, keepdim=True)
        w = torch.where(bad, float("nan"), w)
        return from_pairs(w, layout)
    W_raw, _ = _call(_ceigvals_op(), _planar(PA, batch))
    W = W_raw.transpose(-1, -2).contiguous().reshape(*batch, n, 2)
    return from_pairs(W, layout)


# ---- Householder QR ------------------------------------------------------------------------------------------------


def qr(A, mode="reduced", layout="pairs"):
    """``torch.linalg.qr`` for complex64 ``A (..., m, n)`` with ``1 <= n <= m <= COMPLEX_MAXN``:
    returns ``(Q, R)`` as float32 pairs in ``layout``. ``mode`` is ``"reduced"``, ``"complete"``, or ``"r"``
    (for ``"r"``, ``Q`` is an empty 1D float32 tensor of shape ``(0,)``)."""
    if not isinstance(mode, str) or mode not in ("reduced", "complete", "r"):
        raise RuntimeError(
            f"qr received unrecognized mode {mode!r} but expected one of 'reduced' (default), 'r', or 'complete'"
        )
    PA = as_pairs(A, layout)
    if PA.dim() < 3:
        raise ValueError(f"hpu_ops.tpc.complex_linalg.qr takes (..., m, n) matrices, got pairs {tuple(PA.shape)}")
    m, n = PA.shape[-3], PA.shape[-2]
    if n < 1 or m > COMPLEX_MAXN or m < n:
        raise ValueError(
            f"hpu_ops.tpc.complex_linalg.qr takes 1 <= n <= m <= {COMPLEX_MAXN}, got pairs {tuple(PA.shape)}"
        )
    batch = PA.shape[:-3]
    if _empty(batch):
        if mode == "r":
            return PA.new_zeros(0), from_pairs(PA.new_zeros(*batch, n, n, 2), layout)
        k_q = m if mode == "complete" else n
        return (
            from_pairs(PA.new_zeros(*batch, m, k_q, 2), layout),
            from_pairs(PA.new_zeros(*batch, k_q, n, 2), layout),
        )
    a_t = _planar_t(PA, batch)
    if mode == "r":
        R_t = _call(_cqr_r_op(), a_t)
        return PA.new_zeros(0), from_pairs(_unplanar_t(R_t, batch, n, n), layout)
    op = _cqr_complete_op() if mode == "complete" else _cqr_op()
    Q_t, R_t = _call(op, a_t)
    k_q = m if mode == "complete" else n
    return (
        from_pairs(_unplanar_t(Q_t, batch, m, k_q), layout),
        from_pairs(_unplanar_t(R_t, batch, k_q, n), layout),
    )
