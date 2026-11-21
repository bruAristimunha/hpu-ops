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


# ---- complex64 kernels (planar re/im rows): cgj_solve / cgj_inv ----------------------------------------------------
# LUT-free throughout (lut_free_recip, rsqrt_fine, power-of-two scaling), so these replays are
# meant to be bit-identical to the card.


def _nonfinite(*xs):
    return any(bool(np.any((_u32(np.asarray(x, F32)) & 0x7F800000) == 0x7F800000)) for x in xs)


def _pow2_scale(m):
    """``pow2_scale``: 2^(127 - e) for the biased exponent e of m (1 for m = 0)."""
    e = (_u32(m) >> 23) & 0xFF
    return np.where(e == 0, F32(1), _f32(((254 - e) & 0xFFFFFFFF) << 23)).astype(F32)


def _pow2_unscale(m):
    e = _u32(m) & 0x7F800000
    return np.where(e == 0, F32(1), _f32(e)).astype(F32)


def _big_scale(m):
    return np.where((_u32(m) & 0x7FFFFFFF) >= 0x5F800000, F32(2.0 ** -64), F32(1)).astype(F32)


def rsqrt_fine(x):
    """``rsqrt_fine`` of the Jacobi kernels: 0 -> 1, the guess, two Newton steps, then one in residual form
    y += y/2 (1 - (x y) y) with the product in an FMA, so the result is unbiased (rsqrt(1) = 1; the plain three-step
    form returned 0.99999994 there, which shrank every rotation by an ulp)."""
    x = np.where(np.asarray(x, F32) == 0, F32(1), np.asarray(x, F32)).astype(F32)
    y = _f32((0x5F375A86 - (_u32(x) >> 1)) & 0xFFFFFFFF)
    hx = mul(x, F32(-0.5))
    for _ in range(2):
        y = mul(y, mac(hx, mul(y, y), F32(1.5)))
    e = mac(sub(F32(0), mul(x, y)), y, F32(1))
    return mac(mul(y, F32(0.5)), e, y)


def _cneg_recip_parts(pr, pi, pm):
    """The kernel's -1/p, with the 2^-64 prescale: (scale, psr, psi, ninvr, ninvi)."""
    scale = _big_scale(pm)
    psr, psi = mul(pr, scale), mul(pi, scale)
    t = _pow2_scale(mul(pm, scale))
    tr, ti = mul(psr, t), mul(psi, t)
    rt = mul(recip(mac(ti, ti, mul(tr, tr)), zero_to_inf=False), t)
    return scale, psr, psi, mul(sub(F32(0), tr), rt), mul(ti, rt)


def cgj_solve(A, B=None):
    with np.errstate(all="ignore"):
        return _cgj_solve(A, B)


def _cgj_solve(A, B=None):
    """``(X, D, S)`` of cgj_solve_f32_gaudi2.c (cgj_inv with ``B=None``) for one planar ``A[2n, n]`` (rows 0..n-1
    real parts, n..2n-1 imaginary parts) and ``B[2n, m]``; ``X[2n, m]``, ``D[3, n]`` (pivot re, im, abs), ``S[4]``
    (phase re, im, info, (-1)^swaps)."""
    A = np.array(A, dtype=F32)
    n = A.shape[1]
    ar, ai = _ftz(A[:n].copy()), _ftz(A[n:].copy())
    if B is None:
        br, bi = np.eye(n, dtype=F32), np.zeros((n, n), F32)
    else:
        B = np.array(B, dtype=F32)
        br, bi = _ftz(B[:n].copy()), _ftz(B[n:].copy())
    m = br.shape[1]
    if _nonfinite(A) or (B is not None and _nonfinite(B)):
        return np.full((2 * n, m), np.nan, F32), np.full((3, n), np.nan, F32), np.full(4, np.nan, F32)
    dvr, dvi = np.zeros(n, F32), np.zeros(n, F32)
    sv, info = F32(1), F32(0)
    for k in range(n):
        colr, coli = ar[:, k].copy(), ai[:, k].copy()
        cand = add(np.abs(colr), np.abs(coli))
        cand[:k] = -1
        top = max(F32(0), np.max(cand))
        hits = np.nonzero(cand == top)[0]
        p = int(hits[0]) if hits.size else k
        if p != k:
            for x in (ar, ai, br, bi):
                x[[k, p]] = x[[p, k]]
            colr[[k, p]], coli[[k, p]] = colr[[p, k]], coli[[p, k]]
            sv = -sv
        pr, pi = ar[k, k], ai[k, k]
        dvr[k], dvi[k] = pr, pi
        pm = max(abs(pr), abs(pi))
        if pm == 0 and info == 0:
            info = F32(k + 1)
        scale, psr, psi, ninvr, ninvi = _cneg_recip_parts(pr, pi, pm)
        csr, csi = mul(colr, scale), mul(coli, scale)
        qr = mac(sub(F32(0), csi), ninvi, mul(csr, ninvr))
        qi = mac(csi, ninvr, mul(csr, ninvi))
        rr = mac(sub(F32(0), qi), psi, mac(qr, psr, csr))
        ri = mac(qi, psr, mac(qr, psi, csi))
        qr = mac(sub(F32(0), ri), ninvi, mac(rr, ninvr, qr))
        qi = mac(ri, ninvr, mac(rr, ninvi, qi))
        nqi = sub(F32(0), qi)
        apr, api, bpr, bpi = ar[k].copy(), ai[k].copy(), br[k].copy(), bi[k].copy()
        for i in range(n):
            if i != k:
                ar[i] = mac(api, nqi[i], mac(apr, qr[i], ar[i]))
                ai[i] = mac(apr, qi[i], mac(api, qr[i], ai[i]))
                br[i] = mac(bpi, nqi[i], mac(bpr, qr[i], br[i]))
                bi[i] = mac(bpr, qi[i], mac(bpi, qr[i], bi[i]))
    dm = np.maximum(np.abs(dvr), np.abs(dvi)).astype(F32)
    dscale = _big_scale(dm)
    dms = mul(dm, dscale)
    t, inv_t = _pow2_scale(dms), _pow2_unscale(dms)
    tr, ti = mul(mul(dvr, dscale), t), mul(mul(dvi, dscale), t)
    den = mac(ti, ti, mul(tr, tr))
    rt = mul(recip(den, zero_to_inf=False), t)
    invr, invi = mul(tr, rt), mul(sub(F32(0), ti), rt)
    rs = rsqrt_fine(den)
    # |t| = den * rsqrt(den), then one residual step sq += rs/2 ((a - sq)(a + sq) + b^2), a = max(|tr|, |ti|),
    # b = min: the residual is taken against the components, not the rounded den, so |t| = |tr| exactly when ti = 0
    ta, tb = np.abs(tr).astype(F32), np.abs(ti).astype(F32)
    ta, tb = np.maximum(ta, tb).astype(F32), np.minimum(ta, tb).astype(F32)
    sq = mul(den, rs)
    sq = mac(mul(rs, F32(0.5)), mac(tb, tb, mul(sub(ta, sq), add(ta, sq))), sq)
    absd = mul(mul(sq, inv_t), np.where(dscale == 1, F32(1), F32(2.0 ** 64)).astype(F32))
    # u = t / |t|: q = t rs, then one division residual step q += rs (t - q |t|) (u = 1 exactly for a positive pivot)
    ur, ui = mul(tr, rs), mul(ti, rs)
    ur = mac(mac(sub(F32(0), ur), sq, tr), rs, ur)
    ui = mac(mac(sub(F32(0), ui), sq, ti), rs, ui)
    X = np.zeros((2 * n, m), F32)
    for i in range(n):
        bsr, bsi = mul(br[i], dscale[i]), mul(bi[i], dscale[i])
        X[i] = mac(sub(F32(0), bsi), invi[i], mul(bsr, invr[i]))
        X[n + i] = mac(bsi, invr[i], mul(bsr, invi[i]))
    phr, phi = F32(sv), F32(0)
    for k in range(n):
        phr, phi = mac(sub(F32(0), phi), ui[k], mul(phr, ur[k])), mac(phi, ur[k], mul(phr, ui[k]))
    pn = rsqrt_fine(mac(phi, phi, mul(phr, phr)))
    phr, phi = mul(phr, pn), mul(phi, pn)
    return X, np.stack([dvr, dvi, absd]), np.array([phr, phi, info, sv], F32)


def cgj_op(identity=False):
    """A stand-in for ``torch.ops.linalg_tpc.cgj_solve`` (``cgj_inv`` with ``identity``): the replica per batch
    element on the planar tensors, torch tensors out."""
    import torch

    def op(A, B=None):
        outs = [cgj_solve(a.detach().cpu().numpy(), None if identity else b.detach().cpu().numpy())
                for a, b in zip(A, A if identity else B)]
        return tuple(torch.from_numpy(np.stack([o[j] for o in outs])) for j in range(3))

    return (lambda A: op(A)) if identity else op


# ---- complex64 Jacobi: jacobi_ceigh (two passes) and jacobi_csvd ---------------------------------------------------
# Planar rows, two matrices per vector (lanes 0-31 and 32-63) on the card; one matrix at a time here (the halves never
# mix: every reduction is the 32-lane butterfly ``_red_add``). LUT-free: rsqrt_fine / lut_free_recip.


def _rr_rounds(n):
    """Round-robin rounds of the kernels (``RR_PAIR``): disjoint pairs per round, so their order within a round does
    not change the result."""
    return [_rr_pairs(r, n) for r in range(((n + 1) & ~1) - 1)]


def _crot_coeffs(ur, ui, p, q, skip_tol=None):
    """The complex one-sided Jacobi rotation of rows p, q: the real rotation of (p, e q) with e = <p, q> / |<p, q>|,
    so p' = c p - s e q and q' = s p + c e q are orthogonal. Returns the coefficients ``_capply`` uses."""
    upr, upi, uqr, uqi = ur[p], ui[p], ur[q], ui[q]
    alpha = _red_add(mac(upi, upi, mul(upr, upr)))
    beta = _red_add(mac(uqi, uqi, mul(uqr, uqr)))
    gr = _red_add(mac(upi, uqi, mul(upr, uqr)))
    gi = _red_add(mac(sub(F32(0), upr), uqi, mul(upi, uqr)))
    # |g| and e = g / |g| from g scaled by a power of two into [1, 2): squaring the raw parts flushed a tiny one
    # (|e| came out 2.6 on a rank-deficient matrix); g = 0 gives e = 0 (and the rotation is skipped)
    gm = max(np.abs(gr), np.abs(gi))
    t = _pow2_scale(gm)
    grt, git = mul(gr, t), mul(gi, t)
    n2 = mac(git, git, mul(grt, grt))
    rn = rsqrt_fine(n2)
    ag = mul(mul(n2, rn), _pow2_unscale(gm))
    er, ei = mul(grt, rn), mul(git, rn)
    g2s = mul(ag, ag)
    d = sub(beta, alpha)
    g2 = add(ag, ag)
    r2 = mac(g2, g2, mul(d, d))
    inv_r = rsqrt_fine(r2)
    c2 = mul(F32(0.5), mac(np.abs(d), inv_r, F32(1)))
    inv_c = rsqrt_fine(c2)
    c = mul(c2, inv_c)
    s = mul(ag, mul(inv_r, inv_c))
    s = sub(F32(0), s) if d < 0 else s
    skip = g2s == 0 or r2 == 0
    if skip_tol is not None:  # the SVD's rule: |g|^2 <= tol alpha beta, or r2 < 1e-34
        skip = skip or not mul(F32(skip_tol), mul(alpha, beta)) < g2s or r2 < F32(1e-34)
    if skip:
        c, s, er, ei = F32(1), F32(0), F32(1), F32(0)
    cer, cei, ser, sei = mul(c, er), mul(c, ei), mul(s, er), mul(s, ei)
    return c, s, cer, cei, sei, sub(F32(0), ser), sub(F32(0), sei), sub(F32(0), cei)


def _capply(ur, ui, p, q, co):
    c, s, cer, cei, sei, nser, nsei, ncei = co
    upr, upi, uqr, uqi = ur[p].copy(), ui[p].copy(), ur[q].copy(), ui[q].copy()
    ur[p] = mac(uqi, sei, mac(uqr, nser, mul(upr, c)))
    ui[p] = mac(uqr, nsei, mac(uqi, nser, mul(upi, c)))
    ur[q] = mac(uqi, ncei, mac(uqr, cer, mul(upr, s)))
    ui[q] = mac(uqr, cei, mac(uqi, cer, mul(upi, s)))


def _csweeps(ur, ui, n, sweeps):
    for _ in range(sweeps):
        for pairs_ in _rr_rounds(n):
            for p, q in pairs_:
                _capply(ur, ui, p, q, _crot_coeffs(ur, ui, p, q))


def _cnormalize(ur, ui, n):
    for j in range(n):
        rn = rsqrt_fine(_red_add(mac(ui[j], ui[j], mul(ur[j], ur[j]))))
        ur[j], ui[j] = mul(ur[j], rn), mul(ui[j], rn)


def _prescale(max_abs):
    """The real eigh kernel's prescale / inv_prescale from max|a|."""
    pre = F32(2.0 ** -20) if max_abs > F32(1e33) else F32(1)
    inv = F32(2.0 ** 20) if max_abs > F32(1e33) else F32(1)
    if max_abs < F32(1e-20):
        pre, inv = (pre, inv) if max_abs == 0 else (F32(2.0 ** 40), F32(2.0 ** -40))
    return pre, inv


def _safe_inv_sigma(sigma, hi):
    safe = F32(1) if (sigma == 0 or sigma > F32(hi) or sigma < F32(1e-37)) else sigma
    return recip(safe, zero_to_inf=False)[0]


def _pad32(x):
    out = np.zeros((x.shape[0], 32), F32)
    out[:, :x.shape[1]] = x
    return out


def ceigh_pass1(A):
    """``(V1, H)`` of jacobi_ceigh_f32_gaudi2.c for one planar ``A[2n, n]``: H the Hermitian matrix of A's lower
    triangle (the imaginary parts of the diagonal dropped), V1 the rows u_j = x_j^H (planar ``[2n, n]``) of the
    eigenvectors of S = H/sigma + 1.001 I after one-sided Jacobi, normalized. Non-finite input: NaN."""
    with np.errstate(all="ignore"):
        A = np.array(A, dtype=F32)
        n = A.shape[1]
        ar, ai = _pad32(_ftz(A[:n])), _pad32(_ftz(A[n:]))
        hr, hi = np.zeros_like(ar), np.zeros_like(ai)
        for i in range(n):
            for j in range(n):
                if j < i:
                    hr[i, j], hi[i, j] = ar[i, j], ai[i, j]
                elif j > i:
                    hr[i, j], hi[i, j] = ar[j, i], sub(F32(0), ai[j, i])
                else:
                    hr[i, j] = ar[i, i]
        H = np.concatenate([hr[:, :n], hi[:, :n]])
        if _nonfinite(H):
            return np.full((2 * n, n), np.nan, F32), np.full((2 * n, n), np.nan, F32)
        max_abs = max(F32(np.max(np.abs(hr))), F32(np.max(np.abs(hi))))
        pre, _ = _prescale(max_abs)
        ur, ui = mul(hr, pre), mul(hi, pre)
        sigma = F32(0)
        for i in range(n):
            sigma = max(sigma, _red_add(add(np.abs(ur[i]), np.abs(ui[i]))))
        inv_sigma = _safe_inv_sigma(sigma, 1e35)
        ur, ui = mul(ur, inv_sigma), mul(ui, inv_sigma)
        for i in range(n):
            ur[i, i] = add(ur[i, i], F32(1.001))
        sweeps = 4 if n <= 4 else 8 if n <= 16 else 9 if n <= 22 else 10
        _csweeps(ur, ui, n, sweeps)
        _cnormalize(ur, ui, n)
        return np.concatenate([ur[:n, :n], ui[:n, :n]]), H


def _crow_combo(wr, wi, Mr, Mi, n):
    """sum_i w[i] M_i for the complex row w and planar rows M (the kernels' order: re*Mr, -im*Mi, re*Mi, im*Mr)."""
    yr, yi = np.zeros(32, F32), np.zeros(32, F32)
    for i in range(n):
        br, bi = wr[i], wi[i]
        nbi = sub(F32(0), bi)
        yr = mac(Mi[i], nbi, mac(Mr[i], br, yr))
        yi = mac(Mr[i], bi, mac(Mi[i], br, yi))
    return yr, yi


def ceigh_pass2(H, V1):
    """``(W, V)`` of jacobi_ceigh_refine_f32_gaudi2.c for one planar Hermitian ``H[2n, n]`` and the pass-1 rows
    ``V1[2n, n]``: T = V1 (H * prescale) V1^H, adaptive shift max(0, 0.01 - g_min / sigma_T), 2/4 sweeps, rows
    composed with V1 and renormalized; W the Rayleigh quotients u_j H u_j^H (unsorted)."""
    with np.errstate(all="ignore"):
        H, V1 = np.array(H, dtype=F32), np.array(V1, dtype=F32)
        n = H.shape[1]
        if _nonfinite(H) or _nonfinite(V1):
            return np.full(n, np.nan, F32), np.full((2 * n, n), np.nan, F32)
        hr0, hi0 = _pad32(H[:n]), _pad32(H[n:])
        vr, vi = _pad32(V1[:n]), _pad32(V1[n:])
        max_abs = max(F32(np.max(np.abs(hr0))), F32(np.max(np.abs(hi0))))
        pre, inv_pre = _prescale(max_abs)
        hr, hi = mul(hr0, pre), mul(hi0, pre)
        ur, ui = np.zeros((n, 32), F32), np.zeros((n, 32), F32)
        for j in range(n):
            yr, yi = _crow_combo(vr[j], vi[j], hr, hi, n)
            for l in range(n):
                ur[j, l] = _red_add(mac(yi, vi[l], mul(yr, vr[l])))
                ui[j, l] = _red_add(mac(sub(F32(0), yr), vi[l], mul(yi, vr[l])))
        sigma, g_min = F32(0), F32(0)
        for i in range(n):
            row_sum = _red_add(add(np.abs(ur[i]), np.abs(ui[i])))
            sigma = max(sigma, row_sum)
            dg = ur[i, i]
            g = sub(add(dg, np.abs(dg)), row_sum)
            g_min = g if i == 0 else min(g_min, g)
        inv_sigma = _safe_inv_sigma(sigma, 1e36)
        shift = max(F32(0), sub(F32(0.01), mul(g_min, inv_sigma)))
        ur, ui = mul(ur, inv_sigma), mul(ui, inv_sigma)
        for i in range(n):
            ur[i, i] = add(ur[i, i], shift)
        _csweeps(ur, ui, n, 2 if n <= 4 else 4)
        _cnormalize(ur, ui, n)
        fr, fi = np.zeros((n, 32), F32), np.zeros((n, 32), F32)
        for j in range(n):
            fr[j], fi[j] = _crow_combo(ur[j], ui[j], vr, vi, n)
        _cnormalize(fr, fi, n)
        W = np.zeros(n, F32)
        for j in range(n):
            yr, yi = _crow_combo(fr[j], fi[j], hr, hi, n)
            W[j] = mul(_red_add(mac(yi, fi[j], mul(yr, fr[j]))), inv_pre)
        return W, np.concatenate([fr[:, :n], fi[:, :n]])


def ceigh(A):
    """Both passes on one planar ``A[2n, n]``: ``(W unsorted, V planar rows)``."""
    V1, H = ceigh_pass1(A)
    return ceigh_pass2(H, V1)


def _cdot(ar_, ai_, br_, bi_):
    """<a, b> = sum a conj(b): (re, im) by the 32-lane butterfly."""
    return (_red_add(mac(ai_, bi_, mul(ar_, br_))), _red_add(mac(sub(F32(0), ar_), bi_, mul(ai_, br_))))


def _caxpy_neg(xr, xi, dr, di, ur_, ui_):
    """x - d u for the complex scalar d."""
    ndr, ndi = sub(F32(0), dr), sub(F32(0), di)
    return mac(ui_, di, mac(ur_, ndr, xr)), mac(ur_, ndi, mac(ui_, ndr, xi))


def csvd_kernel(A_t, compute_uv=True, polish=True):
    """``(Wn, S, J)`` of jacobi_csvd_f32_gaudi2.c (jacobi_csvdvals with ``compute_uv=False``: S alone) for one planar
    ``A_t[2k, big]`` (k <= big <= 32): one-sided Jacobi on the k complex rows, A_t = J^H diag(S) Wn with Wn's rows
    orthonormal (rows of tiny norm completed by complex MGS) and J [2k, k] unitary rows. S unsorted."""
    with np.errstate(all="ignore"):
        A_t = np.array(A_t, dtype=F32)
        k, big = A_t.shape[0] // 2, A_t.shape[1]
        if _nonfinite(A_t):
            nan = np.full
            return (nan((2 * k, big), np.nan, F32) if compute_uv else None, nan(k, np.nan, F32),
                    nan((2 * k, k), np.nan, F32) if compute_uv else None)
        wr, wi = _pad32(_ftz(A_t[:k])), _pad32(_ftz(A_t[k:]))
        vr, vi = np.eye(k, 32, dtype=F32), np.zeros((k, 32), F32)
        sigma = max(F32(np.max(np.abs(wr))), F32(np.max(np.abs(wi))))
        pre = F32(1e-4) if F32(1e35) < sigma else F32(1)
        safe_sigma = F32(1) if sigma == 0 else sigma
        inv = recip(mul(safe_sigma, pre), zero_to_inf=False)[0]
        wr, wi = mul(mul(wr, pre), inv), mul(mul(wi, pre), inv)
        ar_, ai_ = wr.copy(), wi.copy()  # the scaled A_t, for W = J A_t after J is reorthonormalized
        sweeps = 6 if k <= 4 else 10 if k <= 16 else 12

        def run(count):
            for _ in range(count):
                for pairs_ in _rr_rounds(k):
                    for p, q in pairs_:
                        co = _crot_coeffs(wr, wi, p, q, skip_tol=1e-14)
                        _capply(wr, wi, p, q, co)
                        if compute_uv:
                            _capply(vr, vi, p, q, co)

        run(sweeps)
        if compute_uv and polish:
            # the accumulated rotations drift from unitary by ~sqrt(#rotations) ulp (5e-6 at k = 16, as in the real
            # kernel): two MGS passes make J unitary to ~1 ulp, W = J A_t, and two polish sweeps of near-identity
            # rotations (c rounds to 1) re-orthogonalize W's rows without moving J off the unitary group
            for _ in range(2):
                for j in range(k):
                    for kk in range(j):
                        dr, di = _cdot(vr[j], vi[j], vr[kk], vi[kk])
                        vr[j], vi[j] = _caxpy_neg(vr[j], vi[j], dr, di, vr[kk], vi[kk])
                    rn = rsqrt_fine(_red_add(mac(vi[j], vi[j], mul(vr[j], vr[j]))))
                    vr[j], vi[j] = mul(vr[j], rn), mul(vi[j], rn)
            for j in range(k):
                wr[j], wi[j] = _crow_combo(vr[j], vi[j], ar_, ai_, k)
            run(2)
        S, sq_all = np.zeros(k, F32), np.zeros(k, F32)
        for j in range(k):
            sq = _red_add(mac(wi[j], wi[j], mul(wr[j], wr[j])))
            sq_all[j] = sq
            tiny = sq < F32(1e-36)
            rn = rsqrt_fine(F32(1) if tiny else sq)
            S[j] = F32(0) if tiny else mul(mul(sq, rn), safe_sigma)
            if compute_uv:
                f = F32(0) if tiny else rn
                wr[j], wi[j] = mul(wr[j], f), mul(wi[j], f)
        if not compute_uv:
            return None, S, None
        thresh = max(F32(1e-36), mul(F32(1e-10), np.max(sq_all)))
        done = (~(sq_all < thresh)).astype(F32)
        diag = np.where(np.arange(32) < big, F32(1), F32(-1)).astype(F32)
        for j in range(k):
            ukr, uki = mul(wr[j], done[j]), mul(wi[j], done[j])
            diag = sub(diag, mac(uki, uki, mul(ukr, ukr)))
        for j in range(k):
            c1r, c1i = wr[j].copy(), wi[j].copy()
            c2r, c2i = np.zeros(32, F32), np.zeros(32, F32)
            c2r[_first_max(diag)] = 1
            for _ in range(2):
                for kk in range(k):
                    ukr, uki = mul(wr[kk], done[kk]), mul(wi[kk], done[kk])
                    d1r, d1i = _cdot(c1r, c1i, ukr, uki)
                    d2r, d2i = _cdot(c2r, c2i, ukr, uki)
                    c1r, c1i = _caxpy_neg(c1r, c1i, d1r, d1i, ukr, uki)
                    c2r, c2i = _caxpy_neg(c2r, c2i, d2r, d2i, ukr, uki)
            cn1 = _red_add(mac(c1i, c1i, mul(c1r, c1r)))
            cn2 = _red_add(mac(c2i, c2i, mul(c2r, c2r)))
            cr, ci, cn = (c1r, c1i, cn1) if F32(0.25) < cn1 else (c2r, c2i, cn2)
            rn = rsqrt_fine(F32(1) if cn < F32(1e-36) else cn)
            cr, ci = mul(cr, rn), mul(ci, rn)
            if done[j] == 0:
                wr[j], wi[j] = cr, ci
                diag = sub(diag, mac(ci, ci, mul(cr, cr)))
            done[j] = 1
        return (np.concatenate([wr[:, :big], wi[:, :big]]), S, np.concatenate([vr[:, :k], vi[:, :k]]))


def _torch_op(fn, n_out):
    import torch

    def op(*tensors):
        outs = [fn(*[t.detach().cpu().numpy() for t in ts]) for ts in zip(*tensors)]
        res = tuple(torch.from_numpy(np.stack([o[j] for o in outs])) for j in range(n_out))
        return res if n_out > 1 else res[0]

    return op


def ceigh_op():
    """Stand-in for ``torch.ops.linalg_tpc.jacobi_ceigh``: planar A [b, 2n, n] -> (V1, H)."""
    return _torch_op(ceigh_pass1, 2)


def ceigh_refine_op():
    """Stand-in for ``torch.ops.linalg_tpc.jacobi_ceigh_refine``: (H, V1) -> (W, V)."""
    return _torch_op(ceigh_pass2, 2)


def csvd_op():
    """Stand-in for ``torch.ops.linalg_tpc.jacobi_csvd``: planar A_t [b, 2k, big] -> (Wn, S, J)."""
    return _torch_op(lambda a: csvd_kernel(a), 3)


def csvdvals_op():
    """Stand-in for ``torch.ops.linalg_tpc.jacobi_csvdvals``: S alone."""
    return _torch_op(lambda a: (csvd_kernel(a, compute_uv=False)[1],), 1)
