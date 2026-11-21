/* Batched in-place inverse of A on Gaudi2 TPC for 64 < n <= 512: Gauss-Jordan with partial pivoting, one matrix
 * per index-space member, built once per VA = ceil(n / 64) in 2..8. The rows do not fit the 80 KB of local memory:
 * the first RC rows stay there, the others are read and written in global memory (the output R) on every step.
 *
 * Tensors (TPC geometry, dim0 innermost; torch [batch, rows, cols] row-major):
 *   A [n, m, batch]   f32, m = n rows (square) or, built with -DTALL, n <= m <= 2048 rows (a tall panel)
 *   R [w, m, batch]   f32, w = 64 * ceil(n / 64): the in-place result, A^-1 = R[P, :][:, Q] (lanes >= n are 0);
 *                     each row taken as a pivot is divided by its pivot at the end, the caller does the gathers.
 *                     Tall: R[P, :] = A[P, :]^-1 and the rows never taken hold -A[o, :] A[P, :]^-1
 *   P [n, batch]      f32: p_k, the row taken as the k-th pivot (exact integers)
 *   Q [m, batch]      f32: Q[p_k] = k (the inverse permutation when square), n + i for a row i never taken
 *   D [n, batch]      f32: the k-th pivot (U's diagonal of P A = L U)
 *   S [2, batch]      f32: lane 0 unused (0; the caller takes the sign from P), lane 1 = info (1 + first zero pivot)
 * A NaN anywhere in A makes every entry of R and D NaN and the info 0 (LAPACK's results are NaN there and its NaN
 * pivots are not zero; the elimination's need not be), so the caller needs no check of its own.
 * Pivoting is implicit (no row moves): step k takes the unused row p of largest |column k| (the first among ties; a
 * NaN counts as 0, so the order stays a permutation), stores 1 at (p, k) and leaves the rest of row p unnormalized;
 * every other row i gets q_i = -R[i][k] / pivot at (i, k) and q_i times the pivot row added elsewhere (LU-style, as
 * gj_solve: nothing is scaled by 1 / pivot, small quotients are carried scaled by 2^64, huge pivots scaled by 2^-64).
 * Row p is then stored normalized by the power of two 2^-e of its pivot (exact), so the quotients of used rows stay
 * bounded by their entries over the new pivot (Codex review #17).
 * So column k of the storage ends up holding column p_k of the transform T with T A = diag(D) in rows p_k, and dividing row p_k by D[k] gives T A = (rows p_k = e_k). The pivot
 * row of step k + 1 is found and copied to local memory during the pass of step k, so each step is one
 * read-modify-write pass over the rows. A zero pivot makes its step a no-op (info set). One ASO per pass: a vector
 * load after a vector store to the same address is not coherent without it. LUT-free (lane broadcast, Newton
 * reciprocal), so local memory stays in its 80 KB mode.
 */
#ifndef VA
#define VA 4  // 64-lane vectors per row, ceil(n / 64): one build per VA in 2..8, so every loop over a row unrolls
#endif
#ifdef TALL
#define VR 32  // tall panels (-DTALL): up to 64 * VR = 2048 rows, the per-row state in VR vectors
#else
#define VR VA  // square: one row per column
#endif
#if VA <= 6
#define NR 2  // rows per group: two independent chains for the scheduler (VA 7, 8: one, as two spill registers)
#else
#define NR 1
#endif
#if defined(TALL)
#define CACHE (280 - 5 * VA - 2 * VR)  // the per-row state of 2048 rows takes 64 vectors of the cache's room
#elif VA < 8
#define CACHE 232  // vectors of local memory for the first rows (58 KB; VA = 8 spills past it)
#else
#define CACHE 216  // VA = 8: 27 rows (54 KB); with the normalized pivot row 224 and 232 spilled past 80 KB (measured)
#endif

__local__ float64 cache[CACHE];  // row i < RC, vector v at cache[i * VA + v]
__local__ float64 nwl[VA];       // the next pivot row, spilled once per step for its lanes k, k + 1
__local__ float64 pwl[VA];
__local__ float64 pwsl[VA];
__local__ float64 dvv[VA];
__local__ float64 pvv[VA];
__local__ float64 qvv[VR];  // lane i % 64 of qvv[i / 64]: the step k that took row i as its pivot, n + i while unused
__local__ float64 drv[VR];  // lane i % 64 of drv[i / 64]: the pivot of the step that took row i, times the 2^-e
                            // that normalized the row as stored (1 while unused: a row never taken is not divided)

static inline float64 lane_bcast(float64 v, int k)
{
    return as_float64(v_broadcast_element_i32(as_int64(v), k));
}

static inline float64 lut_free_recip(float64 x)
{
    uint64 bits = as_uint64(x);
    uint64 sign = v_u32_and_b(bits, 0x80000000u);
    uint64 abits = v_u32_and_b(bits, 0x7FFFFFFFu);

    // FTZ-proof subnormal normalization: (abits | 1.0f) - 1.0f == |x| * 2^126
    uint64 sub_bits = as_uint64(v_f32_sub_b(as_float64(v_u32_or_b(abits, 0x3F800000u)), 1.0f));
    uint64 nbits = v_u32_sel_less_u32_b(abits, 0x00800000u, sub_bits, abits);
    uint64 e_adj = v_u32_sel_less_u32_b(abits, 0x00800000u, 126u, 0u);

    uint64 mant = v_u32_and_b(nbits, 0x007FFFFFu);
    float64 m = as_float64(v_u32_or_b(mant, 0x3F000000u));  // [0.5, 1.0)

    // Cephes quadratic guess + 3 FMA error-form NR steps (<= 1 ULP everywhere)
    float64 y = v_f32_mac_b(m, 2.58586f, -5.81818f);
    y = v_f32_mac_b(m, y, 4.24242f);
    float64 nm = v_f32_sub_b(0.0f, m);
    float64 err = v_f32_mac_b(nm, y, 1.0f);
    y = v_f32_mac_b(y, err, y);
    err = v_f32_mac_b(nm, y, 1.0f);
    y = v_f32_mac_b(y, err, y);
    err = v_f32_mac_b(nm, y, 1.0f);
    y = v_f32_mac_b(y, err, y);

    uint64 yb = as_uint64(y);
    uint64 ey = v_u32_shr_b(yb, 23);
    uint64 ymant = v_u32_and_b(yb, 0x007FFFFFu);
    uint64 e_in = v_u32_shr_b(nbits, 23);

    // Biased exponent with +256 offset so arithmetic stays unsigned: [255, 532] for finite inputs
    uint64 e_biased = (ey + 382u + e_adj) - e_in;

    uint64 res_norm = v_u32_or_b(sign, v_u32_or_b(v_u32_shl_b(e_biased - 256u, 23), ymant));
    uint64 shift = v_u32_sel_less_u32_b(e_biased, 257u, 257u - e_biased, 0u);
    uint64 res_sub = v_u32_or_b(sign, v_u32_shr_b(v_u32_or_b(0x00800000u, ymant), as_int64(shift)));
    uint64 res = v_u32_sel_less_u32_b(e_biased, 257u, res_sub, res_norm);

    res = v_u32_sel_geq_u32_b(e_biased, 511u, v_u32_or_b(sign, 0x7F800000u), res);  // overflow -> +/-Inf
    res = v_u32_sel_eq_u32_b(abits, 0x7F800000u, sign, res);                         // +/-Inf -> +/-0.0f
    res = v_u32_sel_eq_u32_b(abits, 0u, 0u, res);                                    // +/-0.0f -> +0.0f
    res = v_u32_sel_grt_u32_b(abits, 0x7F800000u, bits, res);                        // NaN -> NaN
    return as_float64(res);
}

// q = -col / pivot rounded (Markstein: one FMA residual correction)
static inline float64 quotient(float64 col, float64 scale, float64 pks, float64 ninv)
{
    float64 cs = v_f32_mul_b(col, scale);
    float64 t = v_f32_mul_b(cs, ninv);
    float64 r = v_f32_mac_b(t, pks, cs);
    // no correction when 1/pivot is 0 (an infinite or zero pivot): 0 * inf in r would make q NaN
    return v_f32_sel_eq_f32_b(ninv, 0.0f, t, v_f32_mac_b(r, ninv, t));
}

// qs = q * 2^64 where q flushed to zero, else 0 (used against the pivot row * 2^-64)
static inline float64 small_quotient(float64 col, float64 q, float64 up, float64 ninv)
{
    return v_f32_sel_eq_f32_b(q, 0.0f, v_f32_mul_b(v_f32_mul_b(col, up), ninv), 0.0f);
}

void main(tensor A, tensor R, tensor P, tensor Q, tensor D, tensor S)
{
    const int n = get_dim_size(A, 0);                 // columns, one elimination step each
    const int m = get_dim_size(A, 1);                 // rows: n, or more for a tall panel
    const int RC = CACHE / VA < m ? CACHE / VA : m;  // rows kept in local memory
    const int last = n - 64 * (VA - 1);              // lanes in the last vector
    const int vm = (m + 63) / 64;                    // vectors of per-row state
    const int lastm = m - 64 * (vm - 1);
    const uint64 lane = read_lane_id_4b_b();
    const float64 lanef = v_convert_i32_to_f32_b(as_int64(lane), 0);
    const int5 off = get_index_space_offset();
    const int5 size = get_index_space_size();

    for (int bt = off[0]; bt < off[0] + size[0]; bt++)
    {
        float64 pw[VA], nw[VA];
#pragma unroll
        for (int v = 0; v < VA; v++)
        {
            dvv[v] = 0.0f;
            pvv[v] = 0.0f;
            nw[v] = 0.0f;
        }
        for (int v = 0; v < vm; v++)
        {
            qvv[v] = v_f32_add_b(lanef, (float)(64 * v + n));  // n + i: row i not taken yet
            drv[v] = 1.0f;
        }
        float64 infov = 0.0f;
        float64 nanv = 0.0f;  // lane-wise: 1 where a NaN was seen

        // pass 0: A into the cache and R (lanes >= n zeroed), find the pivot row of column 0
        float64 best = -1.0f;
        float64 bidx = 0.0f;
        for (int i = 0; i < m; i++)
        {
            int5 c = {0, i, bt, 0, 0};
            float64 x[VA];
#pragma unroll
            for (int v = 0; v < VA; v++)
            {
                c[0] = 64 * v;
                x[v] = v_f32_ld_tnsr_b(c, A);
            }
            x[VA - 1] = v_f32_sel_geq_u32_b(lane, (unsigned)last, 0.0f, x[VA - 1]);
#pragma unroll
            for (int v = 0; v < VA; v++)
            {
                nanv = v_f32_sel_eq_f32_b(x[v], x[v], nanv, 1.0f);
            }
            float64 cand = v_f32_abs_b(lane_bcast(x[0], 0));
            cand = v_f32_sel_eq_f32_b(cand, cand, cand, 0.0f);  // NaN counts as 0: the order stays a permutation
            const float64 prev = best;
            bidx = v_f32_sel_grt_f32_b(cand, prev, (float)i, bidx);
            best = v_f32_sel_grt_f32_b(cand, prev, cand, prev);
#pragma unroll
            for (int v = 0; v < VA; v++)
            {
                nw[v] = v_f32_sel_grt_f32_b(cand, prev, x[v], nw[v]);
            }
            if (i < RC)
            {
#pragma unroll
                for (int v = 0; v < VA; v++)
                {
                    cache[i * VA + v] = x[v];
                }
            }
            else
            {
#pragma unroll
                for (int v = 0; v < VA; v++)
                {
                    c[0] = 64 * v;
                    v_f32_st_tnsr(c, R, x[v]);
                }
            }
        }
        aso(SW_VPU);

        for (int k = 0; k < n; k++)
        {
            const int vk = k / 64, lk = k % 64;
            const int kn = k + 1, lkn = kn % 64;
            const bool more = kn < n;
            const int vkn = more ? kn / 64 : vk;
            const float64 p = bidx;  // pivot row index, in every lane
#pragma unroll
            for (int v = 0; v < VA; v++)
            {
                nwl[v] = nw[v];
            }
            const float64 pk = lane_bcast(nwl[vk], lk);
            dvv[vk] = v_f32_sel_eq_u32_b(lane, (unsigned)lk, pk, dvv[vk]);
            pvv[vk] = v_f32_sel_eq_u32_b(lane, (unsigned)lk, p, pvv[vk]);
            infov = v_f32_sel_eq_f32_b(pk, 0.0f, v_f32_sel_eq_f32_b(infov, 0.0f, (float)(k + 1), infov), infov);
            // LU-style, as gj_solve: the pivot row is never scaled by 1/pivot (entries / pivot below 2^-126 would
            // flush); row i gets q_i = -a_ik / pivot (one FMA correction) times the pivot row, the stored pivot row has
            // 1 at column k, and the caller divides row p_k by the k-th pivot at the end. |pivot| >= 2^64: column and
            // pivot scaled by 2^-64 (exact) so recip is never subnormal. |q| < 2^-126 flushes, so those rows also
            // take qs = q * 2^64 against the pivot row * 2^-64. A zero pivot gives q = qs = 0: a no-op step.
            const float64 scale = v_f32_sel_geq_u32_b(v_u32_and_b(as_uint64(pk), 0x7FFFFFFFu), 0x5F800000u,
                                                      0x1p-64f, 1.0f);
            const float64 pks = v_f32_mul_b(pk, scale);
            const float64 ninv = v_f32_sub_b(0.0f, lut_free_recip(pks));
            const float64 up = v_f32_sel_eq_f32_b(scale, 1.0f, 0x1p64f, 1.0f);
            // The stored pivot row is normalized by ns = 2^-e (pivot = 2^e m, m in [1, 2); exact): a used row's
            // quotient is then bounded by its entries over the new pivot, not by them times its own pivot, which
            // overflowed for [[2^70, 2^70], [0, 2^-70]] (2^140; Codex review #17). The elimination still uses the
            // unnormalized row (its entries below 2^-126 x pivot would flush), and drv holds pivot x ns, so the
            // division at the end gives the same values. e = 127 takes 2^-126 (2^-127 is subnormal); a zero,
            // subnormal or non-finite pivot is not scaled.
            const uint64 pe = v_u32_shr_b(v_u32_and_b(as_uint64(pk), 0x7F800000u), 23);
            uint64 nsb = v_u32_shl_b(254u - pe, 23);
            nsb = v_u32_sel_geq_u32_b(pe, 254u, 0x00800000u, nsb);
            nsb = v_u32_sel_eq_u32_b(pe, 255u, 0x3F800000u, nsb);
            nsb = v_u32_sel_eq_u32_b(pe, 0u, 0x3F800000u, nsb);
            const float64 ns = as_float64(nsb);
            unsigned zl[VA];  // lane of vector v zeroed in every non-pivot row: lk in vector vk, none elsewhere
            float64 pws[VA];
            const float64 pkns = v_f32_mul_b(pk, ns);
            for (int v = 0; v < vm; v++)
            {
                float64 rowid = v_f32_add_b(lanef, (float)(64 * v));
                qvv[v] = v_f32_sel_eq_f32_b(rowid, p, (float)k, qvv[v]);
                drv[v] = v_f32_sel_eq_f32_b(rowid, p, pkns, drv[v]);
            }
#pragma unroll
            for (int v = 0; v < VA; v++)
            {
                zl[v] = v == vk ? (unsigned)lk : 64u;
                pw[v] = v_f32_sel_eq_u32_b(lane, zl[v], 1.0f, nw[v]);
                pws[v] = v_f32_sel_less_u32_b(v_u32_and_b(as_uint64(pw[v]), 0x7FFFFFFFu), 0x7F800000u,
                                              v_f32_mul_b(pw[v], 0x1p-64f), 0.0f);
                pwl[v] = pw[v];
                pwsl[v] = pws[v];
            }
            const float64 pkn = lane_bcast(pwl[vkn], lkn);  // the pivot row at column k + 1
            const float64 pksn = lane_bcast(pwsl[vkn], lkn);
#pragma unroll
            for (int v = 0; v < VA; v++)
            {
                pwl[v] = v_f32_mul_b(pw[v], ns);  // the pivot row as stored: normalized (read once per row below)
            }
            best = -1.0f;
            bidx = 0.0f;

            // Rows NR at a time; with an odd count the last pair is (i, i): the same row twice gives the same values,
            // stored twice, and never beats itself as a candidate.
            // rows in local memory
            for (int i = 0; i < RC; i += NR)
            {
                const int ii[2] = {i, i + 1 < RC ? i + 1 : i};  // ii[1] unused when NR = 1
                float64 x[NR][VA], cand[NR], q[NR], qs[NR];
#pragma unroll
                for (int r = 0; r < NR; r++)
                {
                    const int base = ii[r] * VA;
#pragma unroll
                    for (int v = 0; v < VA; v++)
                    {
                        x[r][v] = cache[base + v];
                    }
                    const float64 col = lane_bcast(cache[base + vk], lk);
                    q[r] = quotient(col, scale, pks, ninv);
                    qs[r] = small_quotient(col, q[r], up, ninv);
                    // candidate for step k + 1: the new value at column k + 1 of an unused row (same fused ops as below)
                    float64 cd = v_f32_abs_b(v_f32_mac_b(pksn, qs[r], v_f32_mac_b(pkn, q[r], lane_bcast(cache[base + vkn], lkn))));
                    cd = v_f32_sel_eq_f32_b(cd, cd, cd, 0.0f);  // NaN counts as 0
                    cd = v_f32_sel_less_f32_b(lane_bcast(qvv[ii[r] / 64], ii[r] % 64), (float)n, -1.0f, cd);  // taken
                    cand[r] = more ? cd : -1.0f;
                }
#pragma unroll
                for (int r = 0; r < NR; r++)
                {
                    const int base = ii[r] * VA;
                    const float fi = (float)ii[r];
                    const float64 prev = best;
                    bidx = v_f32_sel_grt_f32_b(cand[r], prev, fi, bidx);
                    best = v_f32_sel_grt_f32_b(cand[r], prev, cand[r], prev);
#pragma unroll
                    for (int v = 0; v < VA; v++)
                    {
                        float64 y = v_f32_sel_eq_u32_b(lane, zl[v], 0.0f, x[r][v]);
                        y = v_f32_mac_b(pws[v], qs[r], v_f32_mac_b(pw[v], q[r], y));
                        y = v_f32_sel_eq_f32_b(p, fi, pwl[v], y);
                        cache[base + v] = y;
                        nw[v] = v_f32_sel_grt_f32_b(cand[r], prev, y, nw[v]);
                    }
                }
            }

            // rows in global memory: all vectors of a pair loaded before any is stored
            for (int i = RC; i < m; i += NR)
            {
                const int ii[2] = {i, i + 1 < m ? i + 1 : i};  // ii[1] unused when NR = 1
                float64 x[NR][VA], cand[NR], q[NR], qs[NR];
#pragma unroll
                for (int r = 0; r < NR; r++)
                {
                    int5 c = {0, ii[r], bt, 0, 0};
#pragma unroll
                    for (int v = 0; v < VA; v++)
                    {
                        c[0] = 64 * v;
                        x[r][v] = v_f32_ld_tnsr_b(c, R);
                    }
                    c[0] = 64 * vk;
                    float64 xk = v_f32_ld_tnsr_b(c, R);
                    c[0] = 64 * vkn;
                    float64 xkn = v_f32_ld_tnsr_b(c, R);
                    const float64 col = lane_bcast(xk, lk);
                    q[r] = quotient(col, scale, pks, ninv);
                    qs[r] = small_quotient(col, q[r], up, ninv);
                    float64 cd = v_f32_abs_b(v_f32_mac_b(pksn, qs[r], v_f32_mac_b(pkn, q[r], lane_bcast(xkn, lkn))));
                    cd = v_f32_sel_eq_f32_b(cd, cd, cd, 0.0f);  // NaN counts as 0
                    cd = v_f32_sel_less_f32_b(lane_bcast(qvv[ii[r] / 64], ii[r] % 64), (float)n, -1.0f, cd);  // taken
                    cand[r] = more ? cd : -1.0f;
                }
#pragma unroll
                for (int r = 0; r < NR; r++)
                {
                    int5 c = {0, ii[r], bt, 0, 0};
                    const float fi = (float)ii[r];
                    const float64 prev = best;
                    bidx = v_f32_sel_grt_f32_b(cand[r], prev, fi, bidx);
                    best = v_f32_sel_grt_f32_b(cand[r], prev, cand[r], prev);
#pragma unroll
                    for (int v = 0; v < VA; v++)
                    {
                        float64 y = v_f32_sel_eq_u32_b(lane, zl[v], 0.0f, x[r][v]);
                        y = v_f32_mac_b(pws[v], qs[r], v_f32_mac_b(pw[v], q[r], y));
                        y = v_f32_sel_eq_f32_b(p, fi, pwl[v], y);
                        c[0] = 64 * v;
                        v_f32_st_tnsr(c, R, y);
                        nw[v] = v_f32_sel_grt_f32_b(cand[r], prev, y, nw[v]);
                    }
                }
            }
            aso(SW_VPU);
        }

        // Each row divided by its pivot (as gj_solve's B / diag(D)): |pivot| >= 2^64 scaled by 2^-64 first, so its
        // reciprocal is never subnormal; a zero pivot gives a zero row, a row never taken (tall) is kept. Cached rows
        // from local memory, the others re-read (after the last step's ASO) and written back.
        // a NaN in any lane of any row: every output of this matrix NaN
        float64 anynan = 0.0f;
        for (int l = 0; l < 64; l++)
        {
            anynan = v_f32_sel_eq_f32_b(lane_bcast(nanv, l), 1.0f, 1.0f, anynan);
        }
        const float64 qnan = as_float64(v_u32_or_b(v_u32_and_b(as_uint64(anynan), 0u), 0x7FC00000u));
        for (int i = 0; i < m; i++)
        {
            const float64 dr = lane_bcast(drv[i / 64], i % 64);
            const float64 ds = v_f32_sel_geq_u32_b(v_u32_and_b(as_uint64(dr), 0x7FFFFFFFu), 0x5F800000u, 0x1p-64f, 1.0f);
            const float64 dinv = lut_free_recip(v_f32_mul_b(dr, ds));
#pragma unroll
            for (int v = 0; v < VA; v++)
            {
                int5 c = {64 * v, i, bt, 0, 0};
                float64 y = i < RC ? cache[i * VA + v] : v_f32_ld_tnsr_b(c, R);
                y = v_f32_mul_b(v_f32_mul_b(y, ds), dinv);
                v_f32_st_tnsr(c, R, v_f32_sel_eq_f32_b(anynan, 1.0f, qnan, y));
            }
        }
#pragma unroll
        for (int v = 0; v < VA; v++)
        {
            int5 cd = {64 * v, bt, 0, 0, 0};
            const float64 dv = v_f32_sel_eq_f32_b(anynan, 1.0f, qnan, dvv[v]);
            if (v == VA - 1 && last < 64)
            {
                v_f32_st_tnsr_partial(cd, D, dv, (char)(last - 1), 0);
                v_f32_st_tnsr_partial(cd, P, pvv[v], (char)(last - 1), 0);
            }
            else
            {
                v_f32_st_tnsr(cd, D, dv);
                v_f32_st_tnsr(cd, P, pvv[v]);
            }
        }
        for (int v = 0; v < vm; v++)
        {
            int5 cq = {64 * v, bt, 0, 0, 0};
            if (v == vm - 1 && lastm < 64)
            {
                v_f32_st_tnsr_partial(cq, Q, qvv[v], (char)(lastm - 1), 0);
            }
            else
            {
                v_f32_st_tnsr(cq, Q, qvv[v]);
            }
        }
        int5 cs = {0, bt, 0, 0, 0};
        infov = v_f32_sel_eq_f32_b(anynan, 1.0f, 0.0f, infov);  // a NaN: info 0, as LAPACK's (NaN pivots are not 0)
        v_f32_st_tnsr_partial(cs, S, v_f32_sel_eq_u32_b(lane, 1u, infov, 0.0f), 1, 0);
    }
}
