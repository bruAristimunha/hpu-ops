"""complex64 linear algebra from the Gaudi2 TPC kernels, for hpu-complex.

A complex matrix is a float32 tensor of (re, im) pairs: ``(..., n, n, 2)`` (``torch.view_as_real`` of a complex64
tensor, ``layout="pairs"``, the default) or the same memory as ``(..., n, 2n)`` (``layout="interleaved"``: row i is
re0, im0, re1, im1, ...). A complex64 tensor is also taken (viewed as pairs). Results come back as float32 pairs in
the input's layout; nothing runs on the CPU.

- ``cgj_solve`` / ``cgj_inv`` (``cgj_solve_f32_gaudi2.c``): Gauss-Jordan with partial pivoting by
  cabs1 = |re| + |im| (LAPACK icamax, as MPS's complex LU), n <= ``COMPLEX_MAXN``. One launch, plus one transpose
  in (pairs to planes) and one out.
- ``solve``, ``inv``, ``slogdet``, ``det`` on top of them. The phase of the determinant is computed in the kernel as
  (-1)^swaps * prod(u_kk / |u_kk|) (torch's ``lu_det_P`` times ``sgn(diag U)``, JAX's ``_slogdet_lu``), each factor
  normalized, the product renormalized once; logabsdet = sum(log |u_kk|) (one log and one sum on the card).

Semantics: torch's on the finite range. A singular matrix (an exactly zero pivot) gives sign 0 and logabsdet -inf,
a zero row of X for that pivot and ``info`` = 1 + its index; any NaN or infinity in A (or B) gives NaN everywhere
(torch gives NaN or inf there). Subnormal inputs count as zero (the card flushes them).
"""
import torch

from . import _call, _fresh, lazy_mode  # noqa: F401  (lazy_mode: re-exported for the callers' checks)
from . import _load

COMPLEX_MAXN = 32  # MAXN of the complex kernels: a row of each plane is one 64-lane vector, 4 planes in 32 KB


def _cgj_op():
    """``torch.ops.linalg_tpc.cgj_solve``, built on first use."""
    _load()
    return torch.ops.linalg_tpc.cgj_solve


def _cgj_inv_op():
    """``torch.ops.linalg_tpc.cgj_inv``, built on first use."""
    _load()
    return torch.ops.linalg_tpc.cgj_inv


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


def _square(P, limit=COMPLEX_MAXN):
    n = P.shape[-2]
    if P.shape[-3] != n or n > limit or n < 1:
        raise ValueError(f"hpu_ops.tpc.complex takes square A with 1 <= n <= {limit}, got pairs {tuple(P.shape)}")
    return n


# ---- Gauss-Jordan ----------------------------------------------------------------------------------------------------


def cgj_solve(A, B, layout="pairs"):
    """``(X, D, S)`` for complex ``A (..., n, n)`` and ``B (..., n, m)`` (n <= ``COMPLEX_MAXN``, any m, broadcast
    batch): ``X = A^-1 B`` in ``layout``; ``D (..., 3, n)`` the pivots' re, im and modulus (U's diagonal of
    P A = L U); ``S (..., 4)`` = (phase re, phase im, info, (-1)^swaps), info = 1 + the first zero pivot (0 if none).
    B's columns go through the kernel in chunks of ``COMPLEX_MAXN`` (D and S from the first)."""
    PA, PB = as_pairs(A, layout), as_pairs(B, layout)
    n = _square(PA)
    if PB.shape[-3] != n:
        raise ValueError(f"hpu_ops.tpc.cgj_solve takes B with n rows, got pairs {tuple(PA.shape)} and {tuple(PB.shape)}")
    m = PB.shape[-2]
    batch = torch.broadcast_shapes(PA.shape[:-3], PB.shape[:-3])
    a = _planar(PA, batch)
    op = _cgj_op()
    parts = [_call(op, a, _planar(chunk, batch)) for chunk in PB.split(COMPLEX_MAXN, -2)]
    X = parts[0][0] if len(parts) == 1 else torch.cat([p[0].reshape(-1, 2, n, p[0].shape[-1]) for p in parts],
                                                       -1).reshape(-1, 2 * n, m)
    D, S = parts[0][1], parts[0][2]
    return from_pairs(_unplanar(X, batch, n, m), layout), D.reshape(*batch, 3, n), S.reshape(*batch, 4)


def cgj_inv(A, layout="pairs"):
    """``(A^-1, D, S)`` for complex ``A (..., n, n)``: ``cgj_solve`` against the identity, built in the kernel."""
    PA = as_pairs(A, layout)
    n = _square(PA)
    batch = PA.shape[:-3]
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
    _, D, S = _call(_cgj_inv_op(), _planar(PA, batch))  # X (A^-1) is not read: no transpose back
    return S[:, :2].reshape(*batch, 2), D[:, 2, :].log().sum(-1).reshape(batch)


def det(A, layout="pairs"):
    """``torch.linalg.det`` for complex A as a ``(..., 2)`` pair: phase * exp(logabsdet)."""
    sign, logabsdet = slogdet(A, layout)
    return sign * logabsdet.exp()[..., None]


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
