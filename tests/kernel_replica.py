"""Float32-exact Python replicas of the arithmetic of the TPC kernels ``gj_solve_f32_gaudi2.c`` and
``tri_solve_f32_gaudi2.c``, one matrix at a time, for the CPU tests.

The CPU tests otherwise stand torch's own solvers in for the kernels (``linalg._GJ`` / ``linalg._TRI`` hooks), which
hides any bug of the kernel's own arithmetic (operation order, reciprocal, scaling). These replicas follow the C code
operation by operation in float32: every product, sum and FMA rounds to float32 (an FMA is computed exactly in
float64 and rounded once to float32; the product of two float32 is exact in float64, so only the final sum can round
twice, which does not happen on the inputs tested), results and inputs below 2^-126 are flushed to zero as the card
does (FTZ/DAZ), and the reciprocal is the kernels' bit-level Newton ``lut_free_recip``.

``legacy=True`` replays the arithmetic of the kernels before the Codex review (Gauss-Jordan normalizing the pivot row
by 1/pivot; substitution summing the products before one subtraction), so a test can show that it reproduces the
reported failures and that the current arithmetic does not."""
import numpy as np

F32 = np.float32
TINY = 2.0 ** -126


def _ftz(x):
    x = np.asarray(x, dtype=F32)
    return np.where(np.abs(x) < TINY, F32(0) * np.sign(x), x).astype(F32)


def mul(a, b):
    return _ftz(np.asarray(_ftz(a), F32) * np.asarray(_ftz(b), F32))


def sub(a, b):
    return _ftz(np.asarray(_ftz(a), F32) - np.asarray(_ftz(b), F32))


def mac(a, b, acc):
    """``v_f32_mac_b(a, b, acc)`` = acc + a * b with one rounding."""
    a, b, acc = (np.asarray(_ftz(v), np.float64) for v in (a, b, acc))
    with np.errstate(over="ignore", invalid="ignore"):
        return _ftz((acc + a * b).astype(F32))


def _u32(x):
    return np.asarray(x, F32).view(np.uint32).astype(np.int64)


def _f32(bits):
    return np.asarray(np.asarray(bits, np.int64) & 0xFFFFFFFF, np.uint32).view(F32)


def recip(x, zero_to_inf):
    """``lut_free_recip`` of the kernels, elementwise: ``zero_to_inf`` False is Gauss-Jordan's (0 -> +0), True the
    substitution's (+-0 -> +-inf)."""
    x = np.atleast_1d(np.asarray(x, F32))
    bits = _u32(x)
    sign = bits & 0x80000000
    abits = bits & 0x7FFFFFFF
    sub_bits = _u32(sub(_f32(abits | 0x3F800000), F32(1)))
    nbits = np.where(abits < 0x00800000, sub_bits, abits)
    e_adj = np.where(abits < 0x00800000, 126, 0)
    mant = nbits & 0x007FFFFF
    m = _f32(mant | 0x3F000000)
    y = mac(m, F32(2.58586), F32(-5.81818))
    y = mac(m, y, F32(4.24242))
    nm = sub(F32(0), m)
    for _ in range(3):
        err = mac(nm, y, F32(1))
        y = mac(y, err, y)
    yb = _u32(y)
    ey = yb >> 23
    ymant = yb & 0x007FFFFF
    e_in = nbits >> 23
    e_biased = (ey + 382 + e_adj - e_in) & 0xFFFFFFFF
    res_norm = sign | ((((e_biased - 256) & 0xFFFFFFFF) << 23) & 0xFFFFFFFF) | ymant
    shift = np.where(e_biased < 257, 257 - e_biased, 0)
    res_sub = sign | ((0x00800000 | ymant) >> np.minimum(shift, 63))
    res = np.where(e_biased < 257, res_sub, res_norm)
    res = np.where(e_biased >= 511, sign | 0x7F800000, res)
    res = np.where(abits == 0x7F800000, sign, res)
    res = np.where(abits == 0, (sign | 0x7F800000) if zero_to_inf else 0, res)
    res = np.where(abits > 0x7F800000, bits, res)
    return _ftz(_f32(res))


def _scale(v):
    """2^-64 where |v| >= 2^64 (``0x5F800000``), else 1."""
    return np.where((_u32(v) & 0x7FFFFFFF) >= 0x5F800000, F32(2.0 ** -64), F32(1)).astype(F32)


def gj_solve(A, B=None, legacy=False, legacy_tiny=False):
    """``(X, D, S)`` of ``gj_solve_f32_gaudi2.c`` (``gj_inv`` with ``B=None``) for one matrix ``A[n, n]`` and
    ``B[n, m]`` (numpy or torch, any float dtype; converted to float32). ``legacy_tiny=True``: the LU-style kernel
    before multipliers below 2^-126 were carried scaled (commit 090da6b)."""
    a = _ftz(np.array(A, dtype=F32))
    n = a.shape[0]
    b = np.eye(n, dtype=F32) if B is None else _ftz(np.array(B, dtype=F32))
    dv = np.zeros(n, F32)
    sv, info = F32(1), 0
    for k in range(n):
        col = a[:, k].copy()
        cand = np.abs(col[k:])
        top = max(F32(0), np.max(cand))
        hits = np.nonzero(cand == top)[0]
        p = k + int(hits[0]) if hits.size else k
        if p != k:
            a[[k, p]], b[[k, p]] = a[[p, k]], b[[p, k]]
            col[[k, p]] = col[[p, k]]
            sv = -sv
        pk = a[k, k]
        dv[k] = pk
        if pk == 0 and info == 0:
            info = k + 1
        s = _scale(pk)
        if legacy:
            inv = recip(mul(pk, s), zero_to_inf=False)[0]
            a[k] = mul(mul(a[k], s), inv)
            b[k] = mul(mul(b[k], s), inv)
            for i in range(n):
                if i != k:
                    nf = sub(F32(0), a[i, k])
                    a[i] = mac(a[k], nf, a[i])
                    b[i] = mac(b[k], nf, b[i])
            continue
        pks = mul(pk, s)
        ninv = sub(F32(0), recip(pks, zero_to_inf=False)[0])
        cs = mul(col, s)
        q = mul(cs, ninv)
        r = mac(q, pks, cs)
        q = q if ninv == 0 else mac(r, ninv, q)
        if legacy_tiny:
            qs, aps, bps = np.zeros_like(q), np.zeros_like(a[k]), np.zeros_like(b[k])
        else:  # multipliers below 2^-126: q * 2^64 against the pivot row * 2^-64
            with np.errstate(invalid="ignore", over="ignore"):
                qs = mul(mul(col, F32(2.0 ** 64) if s == 1 else F32(1)), ninv)
            qs = np.where(q == 0, qs, F32(0)).astype(F32)
            with np.errstate(invalid="ignore"):
                aps = np.where(np.abs(a[k]) < np.inf, mul(a[k], F32(2.0 ** -64)), F32(0)).astype(F32)
                bps = np.where(np.abs(b[k]) < np.inf, mul(b[k], F32(2.0 ** -64)), F32(0)).astype(F32)
        for i in range(n):
            if i != k:
                a[i] = mac(aps, qs[i], mac(a[k], q[i], a[i]))
                b[i] = mac(bps, qs[i], mac(b[k], q[i], b[i]))
    if not legacy:
        ds = _scale(dv)
        dinv = recip(mul(dv, ds), zero_to_inf=False)
        b = mul(mul(b, ds[:, None]), dinv[:, None])
    return b, dv, np.array([sv, info], F32)


def tri_solve(A, B, upper=False, unitriangular=False, legacy=False):
    """``X`` of ``tri_solve_{lower,upper}[_unit]_f32_gaudi2.c`` for one matrix ``A[n, n]`` and ``B[n, m]``."""
    a = _ftz(np.array(A, dtype=F32))
    b = _ftz(np.array(B, dtype=F32))
    n = a.shape[0]
    x = np.zeros_like(b)
    for i in (range(n - 1, -1, -1) if upper else range(n)):
        if legacy:
            total = np.zeros(b.shape[1], F32)
            for k in (range(i + 1, n) if upper else range(i)):
                total = mac(a[i, k], x[k], total)
            acc = sub(b[i], total)
        else:
            acc = b[i].copy()
            for k in (range(n - 1, i, -1) if upper else range(i)):
                acc = mac(sub(F32(0), a[i, k]), x[k], acc)
        if not unitriangular:
            d = a[i, i]
            s = _scale(d)
            acc = mul(mul(acc, s), recip(mul(d, s), zero_to_inf=True)[0])
        x[i] = acc
    return x


def gj_large(A, legacy=False, legacy_growth=False):
    """``(R, P, Q, D, S)`` of ``gj_large_f32_gaudi2.c`` (the in-place Gauss-Jordan inverse, 64 < n <= 512) for one
    matrix ``A[n, n]``, row by row in the kernel's order (each step's pass over the rows; the next pivot row is the
    first unused row of largest |new column k + 1|, a NaN candidate counting as 0). ``legacy=True`` replays the kernel
    before the review: the pivot row normalized by 1/pivot before the elimination (tiny quotients flushed) and NaN
    candidates never taken. ``legacy_growth=True``: the kernel before the stored pivot row was normalized by a power of
    two (Codex review #17: the quotients of used rows overflowed)."""
    a = np.array(A, dtype=F32)  # tensor loads are not flushed; every op is
    n = a.shape[0]
    w = 64 * ((n + 63) // 64)
    R = np.zeros((n, w), F32)
    R[:, :n] = a
    used = np.zeros(n, bool)
    P, Q, D = np.zeros(n, F32), np.zeros(n, F32), np.zeros(n, F32)
    drv = np.zeros(n, F32)  # row i's pivot, times the 2^-e that normalized the row as stored
    info = F32(0)
    lanes = np.arange(w)

    def pick(cand):
        best, idx = F32(-1), 0
        for i in range(n):
            c = cand[i] if legacy or cand[i] == cand[i] else F32(0)
            if c > best:
                best, idx = c, i
        return idx

    p = pick(np.abs(R[:, 0]))
    nw = R[p].copy()
    for k in range(n):
        more = k + 1 < n
        pk = nw[k]
        D[k], P[k] = pk, p
        # the stored pivot row normalized by 2^-e (pivot = 2^e m, m in [1, 2)): 2^-126 for e = 127; 1 for a zero,
        # subnormal or non-finite pivot (Codex review #17)
        pe = (int(_u32(pk)) >> 23) & 0xFF
        ns = F32(1) if legacy_growth or pe in (0, 255) else _f32((max(254 - pe, 1)) << 23)
        drv[p] = mul(pk, ns)
        if pk == 0 and info == 0:
            info = F32(k + 1)
        s = _scale(pk)
        used[p] = True
        Q[p] = k
        col = R[:, k].copy()
        y = R.copy()
        y[:, k] = 0
        if legacy:
            inv = mul(recip(mul(pk, s), zero_to_inf=False)[0], s)
            pw = mul(np.where(lanes == k, F32(1), nw), inv)
            nf = sub(F32(0), col)
            y = mac(pw[None, :], nf[:, None], y)
        else:
            pw = np.where(lanes == k, F32(1), nw).astype(F32)
            with np.errstate(invalid="ignore", over="ignore"):
                pws = np.where(np.abs(pw) < np.inf, mul(pw, F32(2.0 ** -64)), F32(0)).astype(F32)
            pks = mul(pk, s)
            ninv = sub(F32(0), recip(pks, zero_to_inf=False)[0])
            cs = mul(col, s)
            q = mul(cs, ninv)
            r = mac(q, pks, cs)
            q = q if ninv == 0 else mac(r, ninv, q)
            with np.errstate(invalid="ignore", over="ignore"):
                qs = mul(mul(col, F32(2.0 ** 64) if s == 1 else F32(1)), ninv)
            qs = np.where(q == 0, qs, F32(0)).astype(F32)
            y = mac(pws[None, :], qs[:, None], mac(pw[None, :], q[:, None], y))
        y[p] = pw if legacy or legacy_growth else mul(pw, ns)
        R = y.astype(F32)
        if more:
            cand = np.where(used, F32(-1), np.abs(R[:, k + 1]))
            p = pick(cand)
            nw = R[p].copy()
    if not legacy:  # each row divided by its pivot (scaled with the row): |pivot| >= 2^64 scaled by 2^-64 first
        dr = drv
        ds = _scale(dr)
        R = mul(mul(R, ds[:, None]), recip(mul(dr, ds), zero_to_inf=False)[:, None])
    return R, P, Q, D, np.array([0, info], F32)


def gj_large_op(legacy=False, legacy_growth=False):
    """A stand-in for ``torch.ops.linalg_tpc.gj_large``: the replica per batch element, torch tensors out."""
    import torch

    def op(A):
        outs = [gj_large(a.detach().cpu().numpy(), legacy, legacy_growth) for a in A]
        return tuple(torch.from_numpy(np.stack([o[j] for o in outs])) for j in range(5))

    return op
# ---- team D kernels: matrix_exp, lu_factor / lu_solve, jacobi_svd -------------------------------------------------
# The same conventions as above (float32 rounding at every op, FTZ/DAZ, an FMA rounded once). Two hardware functions
# are modelled, not replayed bit for bit: ``v_reciprocal_f32`` as the correctly rounded float32 reciprocal and
# ``v_rsqrt_f32`` as the correctly rounded float32 1/sqrt (the SVD kernel follows it with one Newton step). Lane
# reductions follow the kernels' butterflies (``reduce_*_half32``: partners at lane distance 16, 8, 4, 2, 1).

THETA_13 = F32(5.371920351148152)
EXPM_S_MAX = 126  # S_MAX of matrix_exp_f32_gaudi2.c
EXPM_S_LEGACY = 16  # its cap before review round 3


def add(a, b):
    return _ftz(np.asarray(_ftz(a), F32) + np.asarray(_ftz(b), F32))


def hw_recip(x):
    """``v_reciprocal_f32``, modelled as the correctly rounded float32 reciprocal (1/0 = inf), then FTZ."""
    x = _ftz(x)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        return _ftz(np.divide(F32(1), x, dtype=F32))


def hw_rsqrt(x):
    """``v_rsqrt_f32``, modelled as the correctly rounded float32 1/sqrt."""
    x = np.asarray(_ftz(x), np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return _ftz((1.0 / np.sqrt(x)).astype(F32))


def matmul(P, Q):
    """``matmul_nn``: ``R[i] = sum_k P[i, k] Q[k]``, one FMA per k in order, from 0."""
    R = np.zeros((P.shape[0], Q.shape[1]), F32)
    for k in range(P.shape[1]):
        R = mac(P[:, k:k + 1], Q[k:k + 1, :], R)
    return R


def rcp_newton(x, legacy=False):
    """``rcp_newton`` of matrix_exp_f32_gaudi2.c: a bit-level guess, four Newton steps and one in FMA error form (not
    with ``legacy``, the kernel before Codex review #8); 0 -> 0."""
    x = np.atleast_1d(_ftz(np.asarray(x, F32)))
    ax = np.abs(x)
    y = _f32((0x7EF311C3 - _u32(ax)) & 0xFFFFFFFF)
    for _ in range(4):
        y = mul(y, sub(F32(2), mul(ax, y)))
    if not legacy:  # one step in FMA error form (Codex review #8)
        y = mac(y, mac(sub(F32(0), ax), y, F32(1)), y)
    y = np.where(x < 0, sub(F32(0), y), y)
    return np.where(x == 0, F32(0), y).astype(F32)


def expm_scaling(A, legacy=False, transpose=False):
    """The kernel's scaling count of one matrix: halvings of ||A||_1 (column sums in row order; of A^T with
    ``transpose``) until <= theta_13, at most ``EXPM_S_MAX`` (``EXPM_S_LEGACY`` with ``legacy``); 0 if not finite."""
    a = _ftz(np.array(A, dtype=F32))
    a = a.T if transpose else a
    col = np.zeros(a.shape[1], F32)
    for i in range(a.shape[0]):
        col = add(col, np.abs(a[i]))
    norm = F32(np.max(col))
    if not np.isfinite(norm):  # the kernel's bit test (Codex review #10: the literal 3.4e38 was below FLT_MAX)
        return 0
    s = 0
    while norm > THETA_13 and s < (EXPM_S_LEGACY if legacy else EXPM_S_MAX):
        norm = mul(norm, F32(0.5))
        s += 1
    return s


def _gj_expm(M, B, legacy_rcp=False):
    """``gj_solve_inplace`` of matrix_exp_f32_gaudi2.c: first largest pivot, ``rcp_newton``, pivot row normalized."""
    M, B = M.copy(), B.copy()
    n = M.shape[0]
    for k in range(n):
        cand = np.abs(M[k:, k])
        p = k + int(np.flatnonzero(cand == cand.max())[0]) if np.isfinite(cand).all() else k
        M[[k, p]], B[[k, p]] = M[[p, k]], B[[p, k]]
        inv = rcp_newton(M[k, k], legacy_rcp)[0]
        M[k], B[k] = mul(M[k], inv), mul(B[k], inv)
        for i in range(n):
            if i != k:
                nf = sub(F32(0), M[i, k])
                M[i], B[i] = mac(M[k], nf, M[i]), mac(B[k], nf, B[i])
    return B


def matrix_exp(A, legacy=False, legacy_rcp=False):
    """``exp(A)`` of matrix_exp_f32_gaudi2.c for one matrix ``A[n, n]``: scaling by 2^-s, Padé-13 in the kernel's
    order, the Gauss-Jordan solve, s squarings. ``legacy``: the scaling capped at 16, as before review round 3;
    ``legacy_rcp``: the reciprocal before Codex review #8 (four plain Newton steps)."""
    a = _ftz(np.array(A, dtype=F32))
    n = a.shape[0]
    s = expm_scaling(a, legacy)
    a = mul(a, F32(2.0 ** -s))
    c = [F32(v) for v in (1.0, 0.5, 0.12, 11.0 / 600.0, 11.0 / 5520.0, 3.0 / 18400.0, 1.0 / 96600.0, 1.0 / 1932000.0,
                          1.0 / 48944000.0, 1.0 / 1585785600.0, 1.0 / 67395888000.0, 1.0 / 3953892096000.0,
                          1.0 / 355850288640000.0, 1.0 / 64764752532480000.0)]
    eye = np.eye(n, dtype=F32)
    x2 = matmul(a, a)
    x4 = matmul(x2, x2)
    x6 = matmul(x2, x4)
    u = mac(c[13], x6, mac(c[11], x4, mul(c[9], x2)))
    v = add(matmul(x6, u), mac(c[7], x6, mac(c[5], x4, mac(c[3], x2, mul(eye, c[1])))))
    u = matmul(a, v)
    v = mac(c[12], x6, mac(c[10], x4, mul(c[8], x2)))
    vi = add(matmul(x6, v), mac(c[6], x6, mac(c[4], x4, mac(c[2], x2, eye))))
    with np.errstate(over="ignore", invalid="ignore"):
        r = _gj_expm(sub(vi, u), add(vi, u), legacy_rcp)
        for _ in range(s):
            r = matmul(r, r)
    return r


def _first_max(cand):
    hits = np.flatnonzero(cand == cand.max())
    return int(hits[0]) if hits.size else 0


def _last_max(cand):
    hits = np.flatnonzero(cand == cand.max())
    return int(hits[-1]) if hits.size else 0


def lu_factor(A, argmax="first", legacy_recip=False, legacy_tiny=False):
    """``(LU, pivots, perm, info)`` of lu_factor_f32_gaudi2.c for one matrix ``A[n, n]``: pivots 1-based, perm the
    row permutation. ``argmax="last"`` breaks pivot ties by the last row (a tie order other than LAPACK's);
    ``legacy_recip``: 1/pivot without the 2^-64 scaling, flushed to zero past |pivot| = 2^126 (before round 3);
    ``legacy_tiny``: a multiplier below 2^-126 flushed with its product (before Codex review #5)."""
    a = _ftz(np.array(A, dtype=F32))
    n = a.shape[0]
    piv, perm, info = np.zeros(n, np.int32), np.arange(n), 0
    pick = _first_max if argmax == "first" else _last_max
    for k in range(n):
        p = k + pick(np.abs(a[k:, k]))
        piv[k] = p + 1
        perm[[k, p]] = perm[[p, k]]
        a[[k, p]] = a[[p, k]]
        pk = a[k, k]
        if pk == 0 and info == 0:
            info = k + 1
        sc = F32(1) if (legacy_recip or abs(pk) < F32(2.0 ** 64)) else F32(2.0 ** -64)
        inv = F32(0) if pk == 0 else hw_recip(mul(pk, sc))
        up = F32(2.0 ** 64) if sc == 1 else F32(1)
        with np.errstate(invalid="ignore", over="ignore"):
            aps = np.where(np.abs(a[k]) < np.inf, mul(a[k], F32(2.0 ** -64)), F32(0)).astype(F32)
        for i in range(k + 1, n):
            l_ik = mul(mul(a[i, k], sc), inv)
            with np.errstate(invalid="ignore", over="ignore"):
                ls = F32(0) if (legacy_tiny or legacy_recip or l_ik != 0) else mul(mul(a[i, k], up), inv)
            upd = mac(aps, sub(F32(0), ls), mac(a[k], sub(F32(0), l_ik), a[i]))
            upd[:k + 1] = a[i, :k + 1]
            upd[k] = l_ik
            a[i] = upd
    return a, piv, perm, info


def lu_solve(LU, pivots, B, legacy_recip=False):
    """``X`` of lu_solve_f32_gaudi2.c for one matrix: LAPACK's row swaps on ``B``, then L, then U substitution, each
    product subtracted in turn; the diagonal's reciprocal scaled by 2^-64 past 2^64 (not with ``legacy_recip``)."""
    lu = _ftz(np.array(LU, dtype=F32))
    b = _ftz(np.array(B, dtype=F32))
    n = lu.shape[0]
    for k in range(n):
        p = int(pivots[k]) - 1
        b[[k, p]] = b[[p, k]]
    for i in range(1, n):
        acc = b[i].copy()
        for j in range(i):
            acc = mac(b[j], sub(F32(0), lu[i, j]), acc)
        b[i] = acc
    for i in range(n - 1, -1, -1):
        acc = b[i].copy()
        for j in range(i + 1, n):
            acc = mac(b[j], sub(F32(0), lu[i, j]), acc)
        d = lu[i, i]
        sc = F32(1) if (legacy_recip or abs(d) < F32(2.0 ** 64)) else F32(2.0 ** -64)
        b[i] = mul(mul(acc, sc), hw_recip(mul(d, sc)))
    return b


def _red_add(v):
    """``reduce_add_half32`` of a 32-lane vector (the butterfly; every lane ends with the same sum)."""
    v = np.asarray(v, F32).copy()
    idx = np.arange(32)
    for sh in (16, 8, 4, 2, 1):
        v = add(v, v[idx ^ sh])
    return v[0]


def _rsqrt_newton(x):
    r = hw_rsqrt(x)
    return mul(r, mac(mul(F32(-0.5), x), mul(r, r), F32(1.5)))


def _rr_pairs(r, n):
    """The kernel's round-robin pairs of round ``r`` for ``n`` columns (``RR_PAIR``)."""
    n_even = (n + 1) & ~1
    rr_m, out = n_even - 1, []
    for k in range(n & 1, n_even >> 1):
        if k == 0:
            out.append((r, rr_m))
        else:
            a_, b_ = (r + k) % rr_m, (r - k) % rr_m
            out.append((min(a_, b_), max(a_, b_)))
    return out


def jacobi_svd_kernel(A_t, compute_uv=True, full_u=False, legacy_tiny=False):
    """``(Ut, S, Vt)`` of jacobi_svd_f32_gaudi2.c for one ``A_t[k, big]`` (the rows it orthogonalizes, k <= big <=
    32): ``S`` unsorted; ``Ut`` [k or big, big] and ``Vt`` [k, k] (None without ``compute_uv``). ``legacy_tiny``: a
    matrix with max|a| < 1e-36 not scaled (before review round 3), so its squares underflow and S is 0."""
    w0 = _ftz(np.array(A_t, dtype=F32))
    n, m = w0.shape
    w = np.zeros((n, 32), F32)
    w[:, :m] = w0
    v = np.eye(n, 32, dtype=F32)
    sigma = F32(np.max(np.abs(w))) if w.size else F32(0)
    pre = F32(1e-4) if F32(1e35) < sigma else F32(1)
    safe_sigma = F32(1) if (sigma < F32(1e-36) if legacy_tiny else sigma == 0) else sigma
    adj = mul(safe_sigma, pre)
    inv = hw_recip(adj)
    inv = mul(inv, sub(F32(2), mul(adj, inv)))
    w = mul(mul(w, pre), inv)
    sweeps = 6 if n <= 4 else 10 if n <= 16 else 12
    for _ in range(sweeps):
        for r in range(((n + 1) & ~1) - 1):
            for p, q in _rr_pairs(r, n):
                up, uq = w[p].copy(), w[q].copy()
                alpha, beta, gamma = _red_add(mul(up, up)), _red_add(mul(uq, uq)), _red_add(mul(up, uq))
                d, g2 = sub(beta, alpha), add(gamma, gamma)
                r2 = mac(g2, g2, mul(d, d))
                inv_r = _rsqrt_newton(F32(1) if r2 < F32(1e-34) else r2)
                c2 = mul(F32(0.5), mac(np.abs(d), inv_r, F32(1)))
                inv_c = _rsqrt_newton(c2)
                c = mul(c2, inv_c)
                s = mul(mul(F32(0.5), g2), mul(inv_r, inv_c))
                s = sub(F32(0), s) if d < 0 else s
                if not mul(F32(1e-14), mul(alpha, beta)) < mul(gamma, gamma) or r2 < F32(1e-34) or gamma == 0:
                    c, s = F32(1), F32(0)
                w[p], w[q] = mac(uq, sub(F32(0), s), mul(up, c)), mac(up, s, mul(uq, c))
                if compute_uv:
                    vp, vq = v[p].copy(), v[q].copy()
                    v[p], v[q] = mac(vq, sub(F32(0), s), mul(vp, c)), mac(vp, s, mul(vq, c))
    S = np.zeros(n, F32)
    sq_all = np.zeros(n, F32)
    for j in range(n):
        sq = _red_add(mul(w[j], w[j]))
        sq_all[j] = sq
        tiny = sq < F32(1e-36)
        rn = _rsqrt_newton(F32(1) if tiny else sq)
        S[j] = F32(0) if tiny else mul(mul(sq, rn), safe_sigma)
        if compute_uv:
            w[j] = mul(w[j], F32(0) if tiny else rn)
    if not compute_uv:
        return None, S, None
    # MGS completion of the columns below the threshold, in the kernel's order (done flags updated as it goes)
    thresh = max(F32(1e-36), mul(F32(1e-10), np.max(sq_all)))
    done = (~(sq_all < thresh)).astype(F32)
    diag = np.where(np.arange(32) < m, F32(1), F32(-1)).astype(F32)
    for k in range(n):
        uk = mul(w[k], done[k])
        diag = sub(diag, mul(uk, uk))
    for j in range(n):
        dj = done[j]
        cand1, cand2 = w[j].copy(), np.zeros(32, F32)
        cand2[_first_max(diag)] = 1
        for _ in range(2):
            for k in range(n):
                uk = mul(w[k], done[k])
                d1, d2 = _red_add(mul(cand1, uk)), _red_add(mul(cand2, uk))
                cand1, cand2 = sub(cand1, mul(d1, uk)), sub(cand2, mul(d2, uk))
        cn1, cn2 = _red_add(mul(cand1, cand1)), _red_add(mul(cand2, cand2))
        cand, cn = (cand1, cn1) if F32(0.25) < cn1 else (cand2, cn2)
        cand = mul(cand, _rsqrt_newton(F32(1) if cn < F32(1e-36) else cn))
        if dj == 0:
            w[j] = cand
            diag = sub(diag, mul(cand, cand))
        done[j] = 1
    Ut = w[:, :m].copy()
    if full_u:
        rows = [np.where(np.arange(32) < m, w[j], F32(0)).astype(F32) for j in range(n)]
        for j in range(n, m):
            cand = np.zeros(32, F32)
            cand[_first_max(diag)] = 1
            for _ in range(2):
                for uk in rows:
                    cand = sub(cand, mul(_red_add(mul(cand, uk)), uk))
            cn = _red_add(mul(cand, cand))
            cand = mul(cand, _rsqrt_newton(F32(1) if cn < F32(1e-36) else cn))
            cand[m:] = 0
            rows.append(cand)
            diag = sub(diag, mul(cand, cand))
        Ut = np.stack(rows)[:, :m]
    return Ut, S, v[:, :n].copy()


def jacobi_svd(A, compute_uv=True, full_matrices=False, contiguous=True, legacy_tiny=False):
    """``torch.linalg.svd`` of one ``A[m, n]`` as ``hpu_ops.tpc.jacobi_svd`` computes it: the rows of A are
    orthogonalized when m < n, or when A is square and contiguous (``contiguous``), else the rows of A^T; then the
    kernel, then the singular values sorted descending. Returns ``(U, S, Vh)`` (U, Vh None without ``compute_uv``)."""
    a = np.array(A, dtype=F32)
    m, n = a.shape
    k, big = min(m, n), max(m, n)
    full = full_matrices and m != n
    swap = m < n or (m == n and contiguous)
    Ut, S, Vt = jacobi_svd_kernel(a if swap else a.T, compute_uv, full, legacy_tiny)
    order = np.argsort(-S, kind="stable")
    if not compute_uv:
        return None, S[order], None
    if swap:
        U, Vh = Vt.T, Ut
    else:
        U, Vh = Ut.T, Vt
    full_order = np.concatenate([order, np.arange(k, big)]) if full else order
    u_ord, v_ord = ((order, full_order) if swap else (full_order, order)) if full else (order, order)
    return U[:, u_ord], S[order], Vh[v_ord, :]


# ---- householder_product_f32_gaudi2.c ------------------------------------------------------------------------------


def _reduce_add_64(x):
    """``reduce_add_64``: butterfly over the 64 lanes (dual groups 16, 32, then lane distance 8, 4, 2, 1)."""
    lanes = np.arange(64)
    for d in (16, 32, 8, 4, 2, 1):
        x = add(x, x[lanes ^ d])
    return x


def householder_product(A, tau, legacy=False):
    """``Q`` of householder_product_f32_gaudi2.c for one ``A[m, n]`` (m <= 64) and ``tau[k]``: the reflectors applied
    from the last, each column a dot product (``_reduce_add_64``) and one FMA. ``legacy``: before Codex review #16 (a
    zero tau multiplied its reflector's dot product, NaN where that overflowed)."""
    a = np.array(A, dtype=F32)
    t = np.array(tau, dtype=F32)
    m, n = a.shape
    k = t.shape[0]
    lanes = np.arange(64)
    q = [np.where(lanes < m, np.pad(a[:, j], (0, 64 - m)), F32(0)).astype(F32) for j in range(k)]
    q += [np.where(lanes == j, F32(1), F32(0)).astype(F32) for j in range(k, n)]
    for i in range(k - 1, -1, -1):
        tau_i = t[i]
        vi = np.where(lanes < i, F32(0), q[i])
        vi = np.where(lanes == i, F32(1), vi).astype(F32)
        with np.errstate(invalid="ignore", over="ignore"):
            ntv = mul(sub(F32(0), tau_i), vi)
            for j in range(i + 1, n):
                d = _reduce_add_64(mul(vi, q[j]))
                upd = mac(ntv, d, q[j])
                q[j] = upd if legacy or tau_i != 0 else q[j]
            e = np.where(lanes == i, F32(1), F32(0)).astype(F32)
            q[i] = add(e, ntv) if legacy or tau_i != 0 else e
    return np.stack([x[:m] for x in q], 1)
