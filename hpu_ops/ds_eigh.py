"""Double-single (DS) eigenvalue refinement for float64 ``eigh`` (#64), opt-in.

Gaudi2 has no float64 arithmetic (a float64 tensor there holds float32 values), so a
float64 eigh today routes to the CPU. The Jacobi/TPC forward in float32 leaves eigenvalue
errors around 1e-6..1e-5 on ill-conditioned spectra; one Rayleigh-quotient refinement in
double-single arithmetic (each value a pair of float32s, ~48-bit mantissa) is quadratically
accurate in the eigenvector error and brings the eigenvalues to the stored matrix's own
float64 accuracy (~1e-12 measured), on card. Eigenvectors are the float32 ones (their
storage is float32 anyway). Opt-in: ``HPU_OPS_EIGH_DS=1``.
"""
import os

import torch


def two_sum(a, b):
    """Knuth's error-free transformation: (s, e) with s + e == a + b exactly (float32)."""
    s = a + b
    bb = s - a
    e = (a - (s - bb)) + (b - bb)
    return s, e


def ds_add(ah, al, bh, bl):
    s, e = two_sum(ah, bh)
    e = e + (al + bl)
    hi = s + e
    lo = e - (hi - s)
    return hi, lo


def ds_mul(ah, al, bh, bl):
    """Dekker product (Veltkamp split; float32 has a 24-bit mantissa, split at 2**12 + 1)."""
    p = ah * bh
    ca, cb = 8193.0 * ah, 8193.0 * bh
    ahh = ca - (ca - ah)
    ahl = ah - ahh
    bhh = cb - (cb - bh)
    bhl = bh - bhh
    e = ((ahh * bhh - p) + ahh * bhl + ahl * bhh) + ahl * bhl
    e = e + (ah * bl + al * bh)
    hi = p + e
    lo = e - (hi - p)
    return hi, lo


def ds_matmul(ah, al, bh, bl):
    """DS matmul with compensated accumulation over the contraction axis: each product
    is a Dekker pair and the running sum a DS accumulator, so the result carries ~48-bit
    accuracy (a plain fp32 matmul's accumulation error, ~1e-7, would swamp it)."""
    k = ah.shape[-1]
    hi = (ah @ bh) * 0  # output shape, zeroed
    lo = torch.zeros_like(hi)
    for i in range(k):
        ph, pl = ds_mul(ah[..., :, i : i + 1], al[..., :, i : i + 1],
                        bh[..., i : i + 1, :], bl[..., i : i + 1, :])
        hi, lo = ds_add(hi, lo, ph, pl)
    return hi, lo


def ds_split(x):
    """Split a float64 tensor into its float32 (hi, lo) pair."""
    hi = x.float()
    lo = (x.double() - hi.double()).float()
    return hi, lo


def ds_eigh_refine(A, w, V):
    """Refined eigenvalues of symmetric ``A`` (any dtype/device) from float32 eigenpairs
    ``(w, V)``: Rayleigh quotients ``v'Av`` in double-single arithmetic. Returns float64
    eigenvalues on ``A``'s device; ``V`` is returned unchanged."""
    if V.shape[-1] == 0 or A.shape[-1] == 0:
        return w.double()
    if A.dtype == torch.float64 and A.device.type != "hpu":
        ah, al = ds_split(A.double())
    else:
        # on Gaudi a float64 tensor already holds float32 values: the lo part is exactly 0
        ah, al = A.float(), torch.zeros_like(A.float())
    vh = V.float()
    zero = torch.zeros_like(vh)
    avh, avl = ds_matmul(ah, al, vh, zero)
    qh, ql = ds_matmul(vh.mT, zero, avh, avl)  # Rayleigh quotients on the diagonal
    lam_hi = torch.diagonal(qh, dim1=-2, dim2=-1)
    lam_lo = torch.diagonal(ql, dim1=-2, dim2=-1)
    return lam_hi.double() + lam_lo.double()


def ds_div(ah, al, bh, bl):
    """DS division: fp32 quotient plus one residual correction."""
    q1 = ah / bh
    # residual a - q1 * b in DS
    ph, pl = ds_mul(q1, torch.zeros_like(q1), bh, bl)
    rh, rl = ds_add(ah, al, -ph, -pl)
    q2 = rh / bh
    hi = q1 + q2
    lo = q2 - (hi - q1)
    return hi, lo


def ds_sqrt(ah, al):
    """DS square root: fp32 root plus two Newton corrections in DS."""
    y = torch.sqrt(ah)
    for _ in range(2):
        qh, ql = ds_div(ah, al, y, torch.zeros_like(y))
        hi, lo = ds_add(y, torch.zeros_like(y), qh, ql)
        y = 0.5 * hi + 0.5 * lo
    return y, torch.zeros_like(y)


def ds_refine_vectors(A, w, V):
    """One inverse-iteration refinement per eigenvector, in DS: solve the shifted system
    in float32, correct by the DS residual, renormalize. Returns (refined w, refined V)
    with w recomputed as Rayleigh quotients of the refined vectors."""
    if A.dtype == torch.float64 and A.device.type != "hpu":
        ah, al = ds_split(A.double())
    else:
        ah, al = A.float(), torch.zeros_like(A.float())
    vh = V.float()
    zero = torch.zeros_like(vh)
    w = w.float()
    n = A.shape[-1]
    eye = torch.eye(n, dtype=torch.float32, device=A.device)
    cols = []
    for i in range(n):
        lam = w[..., i]
        shifted = ah - lam[..., None, None] * eye
        y = torch.linalg.solve(shifted, vh[..., :, i : i + 1])
        # DS residual r = v - (A - lam I) y, then one correction solve
        mh, ml = ah - lam[..., None, None] * eye, al
        ryh, ryl = ds_matmul(mh, ml, y, torch.zeros_like(y))
        rh = vh[..., :, i : i + 1] - ryh
        rl = -ryl
        dy = torch.linalg.solve(shifted, rh + rl)
        y = y + dy
        # normalize in DS
        nh, nl = ds_matmul(y.mT, torch.zeros_like(y.mT), y, torch.zeros_like(y))
        inv = 1.0 / ds_sqrt(nh, nl)[0]
        cols.append(y * inv)
    V2 = torch.cat(cols, dim=-1)
    # Rayleigh quotients of the refined vectors (sign-fixed by alignment with V)
    sign = torch.where((V2 * vh).sum(-2) < 0, -1.0, 1.0)
    V2 = V2 * sign
    avh, avl = ds_matmul(ah, al, V2, zero)
    qh, ql = ds_matmul(V2.mT, zero, avh, avl)
    lam_hi = torch.diagonal(qh, dim1=-2, dim2=-1)
    lam_lo = torch.diagonal(ql, dim1=-2, dim2=-1)
    return lam_hi.double() + lam_lo.double(), V2


def ds_enabled():
    return os.environ.get("HPU_OPS_EIGH_DS") == "1"
