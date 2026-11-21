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


def gj_large(A, legacy=False):
    """``(R, P, Q, D, S)`` of ``gj_large_f32_gaudi2.c`` (the in-place Gauss-Jordan inverse, 64 < n <= 512) for one
    matrix ``A[n, n]``, row by row in the kernel's order (each step's pass over the rows; the next pivot row is the
    first unused row of largest |new column k + 1|, a NaN candidate counting as 0). ``legacy=True`` replays the kernel
    before the review: the pivot row normalized by 1/pivot before the elimination (tiny quotients flushed) and NaN
    candidates never taken."""
    a = np.array(A, dtype=F32)  # tensor loads are not flushed; every op is
    n = a.shape[0]
    w = 64 * ((n + 63) // 64)
    R = np.zeros((n, w), F32)
    R[:, :n] = a
    used = np.zeros(n, bool)
    P, Q, D = np.zeros(n, F32), np.zeros(n, F32), np.zeros(n, F32)
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
        y[p] = pw
        R = y.astype(F32)
        if more:
            cand = np.where(used, F32(-1), np.abs(R[:, k + 1]))
            p = pick(cand)
            nw = R[p].copy()
    if not legacy:  # each row divided by its pivot: |pivot| >= 2^64 scaled by 2^-64 first
        dr = np.zeros(n, F32)
        dr[P.astype(int)] = D
        ds = _scale(dr)
        R = mul(mul(R, ds[:, None]), recip(mul(dr, ds), zero_to_inf=False)[:, None])
    return R, P, Q, D, np.array([0, info], F32)


def gj_large_op(legacy=False):
    """A stand-in for ``torch.ops.linalg_tpc.gj_large``: the replica per batch element, torch tensors out."""
    import torch

    def op(A):
        outs = [gj_large(a.detach().cpu().numpy(), legacy) for a in A]
        return tuple(torch.from_numpy(np.stack([o[j] for o in outs])) for j in range(5))

    return op
