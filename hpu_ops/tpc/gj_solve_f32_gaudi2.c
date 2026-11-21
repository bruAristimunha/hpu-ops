/* Batched solve A X = B on Gaudi2 TPC: Gauss-Jordan with partial pivoting, one matrix per index-space member.
 *
 * Tensors (TPC geometry, dim0 innermost; torch [batch, rows, cols] row-major):
 *   A [n, n, batch]  f32     B [m, n, batch]  f32     X [m, n, batch]  f32
 *   D [n, batch]  f32: the pivot at each step (U's diagonal of P A = L U)
 *   S [2, batch]  f32: lane 0 = (-1)^swaps, lane 1 = info (1 + first zero pivot, kernel's order)
 * D and S give det = S0 * prod(D) and slogdet. Elimination is LU-style (row -= (a_ik / pivot) * pivot row, the
 * pivot row unnormalized; X = B / diag at the end): equal rows cancel exactly, so a rank-deficient matrix gets an exactly
 * zero pivot when its rows are exact duplicates (or a power-of-two multiple), where LAPACK's rounded multipliers can
 * leave a residue and report info 0; other rank-deficient matrices (e.g. integer ones) are not detected, as in torch.
 * B is never scaled by 1/pivot before the cancellation (no intermediate overflow). A multiplier below 2^-126 is
 * carried as q * 2^64 against the pivot row * 2^-64, so it is not flushed when its product is normal. A zero pivot
 * leaves the elimination a no-op and its row of X zero, so later pivots stay finite.
 * With -DIDENTITY_B the kernel takes A alone and solves against the identity (inv): m = n, no B tensor.
 * Row i of A and of B is one 64-lane vector in local memory (lane j = column j), so n, m <= MAXN.
 * All operations (lane broadcast, argmax, reciprocal) are LUT-free so VLM stays in 80 KB mode (a[64]+b[64] = 32 KB).
 */
#define MAXN 64

__local__ float64 a[MAXN];
__local__ float64 b[MAXN];

static inline float64 lane_bcast(float64 v, int k)
{
    return as_float64(v_broadcast_element_i32(as_int64(v), k));
}

static inline float64 reduce_max_64(float64 p)
{
    p = v_f32_max_b(p, v_f32_mov_dual_group_all_b(p, 0xFFFFFFFF, 1, 0, 3, 2, SW_MDG_ALL, p));
    p = v_f32_max_b(p, v_f32_mov_dual_group_all_b(p, 0xFFFFFFFF, 2, 3, 0, 1, SW_MDG_ALL, p));
    p = v_f32_max_b(p, as_float64(v_element_shift_xor_i32(as_int64(p), 8)));
    p = v_f32_max_b(p, as_float64(v_element_shift_xor_i32(as_int64(p), 4)));
    p = v_f32_max_b(p, as_float64(v_element_shift_xor_i32(as_int64(p), 2)));
    p = v_f32_max_b(p, as_float64(v_element_shift_xor_i32(as_int64(p), 1)));
    return p;
}

static inline uint64 reduce_min_u32_64(uint64 idx)
{
    idx = v_u32_min_b(idx, as_uint64(v_f32_mov_dual_group_all_b(
        as_float64(idx), 0xFFFFFFFF, 1, 0, 3, 2, SW_MDG_ALL, as_float64(idx))));
    idx = v_u32_min_b(idx, as_uint64(v_f32_mov_dual_group_all_b(
        as_float64(idx), 0xFFFFFFFF, 2, 3, 0, 1, SW_MDG_ALL, as_float64(idx))));
    idx = v_u32_min_b(idx, as_uint64(v_element_shift_xor_i32(as_int64(idx), 8)));
    idx = v_u32_min_b(idx, as_uint64(v_element_shift_xor_i32(as_int64(idx), 4)));
    idx = v_u32_min_b(idx, as_uint64(v_element_shift_xor_i32(as_int64(idx), 2)));
    idx = v_u32_min_b(idx, as_uint64(v_element_shift_xor_i32(as_int64(idx), 1)));
    return idx;
}

static inline uint64 reduce_argmax_64(float64 cand, uint64 lane, int n, int k)
{
    float64 max_val = v_f32_max_b(reduce_max_64(cand), 0.0f);
    uint64 idx = v_u32_sel_eq_f32_b(cand, max_val, lane, 0xFFFFFFFFu);
    uint64 p = reduce_min_u32_64(idx);
    return v_u32_sel_geq_u32_b(p, (unsigned)n, (unsigned)k, p);
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

#ifdef IDENTITY_B
void main(tensor A, tensor X, tensor D, tensor S)
#else
void main(tensor A, tensor B, tensor X, tensor D, tensor S)
#endif
{
    const int n = get_dim_size(A, 0);
#ifdef IDENTITY_B
    const int m = n;
#else
    const int m = get_dim_size(B, 0);
#endif
    const uint64 lane = read_lane_id_4b_b();
    const int5 off = get_index_space_offset();
    const int5 size = get_index_space_size();

    for (int bt = off[0]; bt < off[0] + size[0]; bt++)
    {
        float64 nanv = 0.0f;  // lane-wise: 1 where a NaN of A was seen
        for (int i = 0; i < n; i++)
        {
            int5 c = {0, i, bt, 0, 0};
            a[i] = v_f32_sel_geq_u32_b(lane, (unsigned)n, 0.0f, v_f32_ld_tnsr_b(c, A));
            nanv = v_f32_sel_eq_f32_b(a[i], a[i], nanv, 1.0f);
#ifdef IDENTITY_B
            b[i] = v_f32_sel_eq_u32_b(lane, (unsigned)i, 1.0f, 0.0f);
#else
            b[i] = v_f32_sel_geq_u32_b(lane, (unsigned)m, 0.0f, v_f32_ld_tnsr_b(c, B));
#endif
        }
        float64 dv = 0.0f;
        float64 dsv = 0.0f;  // lane k: the k-th pivot times the 2^-e that normalized its row as stored
        float64 sv = 1.0f;
        float64 infov = 0.0f;
        for (int k = 0; k < n; k++)
        {
            // column k as a vector over rows: lane i = A[i][k], every row (the multipliers need the rows above k too)
            // four independent select chains (rows i mod 4), so the n dependent selects do not serialize; lanes are
            // disjoint, so the sum of the four is exact
            float64 c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;
            int i4 = 0;
            for (; i4 + 3 < n; i4 += 4)
            {
                c0 = v_f32_sel_eq_u32_b(lane, (unsigned)i4, lane_bcast(a[i4], k), c0);
                c1 = v_f32_sel_eq_u32_b(lane, (unsigned)(i4 + 1), lane_bcast(a[i4 + 1], k), c1);
                c2 = v_f32_sel_eq_u32_b(lane, (unsigned)(i4 + 2), lane_bcast(a[i4 + 2], k), c2);
                c3 = v_f32_sel_eq_u32_b(lane, (unsigned)(i4 + 3), lane_bcast(a[i4 + 3], k), c3);
            }
            for (; i4 < n; i4++)
            {
                c0 = v_f32_sel_eq_u32_b(lane, (unsigned)i4, lane_bcast(a[i4], k), c0);
            }
            float64 col = v_f32_add_b(v_f32_add_b(c0, c1), v_f32_add_b(c2, c3));
            float64 cand = v_f32_abs_b(col);
            cand = v_f32_sel_less_u32_b(lane, (unsigned)k, -1.0f, cand);
            cand = v_f32_sel_geq_u32_b(lane, (unsigned)n, -1.0f, cand);
            uint64 p = reduce_argmax_64(cand, lane, n, k);  // pivot row, in every lane

            // swap rows k and p with selects (no data-dependent branch)
            float64 ak = a[k], bk = b[k], ap = ak, bp = bk;
            for (int i = k + 1; i < n; i++)
            {
                ap = v_f32_sel_eq_u32_b(p, (unsigned)i, a[i], ap);
                bp = v_f32_sel_eq_u32_b(p, (unsigned)i, b[i], bp);
            }
            for (int i = k + 1; i < n; i++)
            {
                a[i] = v_f32_sel_eq_u32_b(p, (unsigned)i, ak, a[i]);
                b[i] = v_f32_sel_eq_u32_b(p, (unsigned)i, bk, b[i]);
            }
            a[k] = ap;
            b[k] = bp;
            float64 pk = lane_bcast(ap, k);
            // the column after the swap: lane p gets the old A[k][k], lane k the pivot
            col = v_f32_sel_eq_u32_b(lane, p, lane_bcast(col, k), col);
            col = v_f32_sel_eq_u32_b(lane, (unsigned)k, pk, col);
            dv = v_f32_sel_eq_u32_b(lane, (unsigned)k, pk, dv);
            sv = v_f32_sel_eq_u32_b(p, (unsigned)k, sv, v_f32_sub_b(0.0f, sv));
            infov = v_f32_sel_eq_f32_b(pk, 0.0f, v_f32_sel_eq_f32_b(infov, 0.0f, (float)(k + 1), infov), infov);
            // LU-style elimination: row i -= (A[i][k] / pivot) * pivot row, the pivot row left unnormalized, so equal
            // rows cancel exactly and B is never scaled by 1/pivot before the cancellation. The quotients of all rows
            // at once, one vector: q = -col / pivot, then one FMA residual correction (Markstein), so q is the
            // rounded quotient (-1 exactly for an equal row). |pivot| >= 2^64: col and pivot scaled by 2^-64 (exact)
            // so recip(pivot) is never subnormal, which the card flushes to zero. A zero pivot gives recip 0, so
            // q = 0 and the elimination is a no-op, as in LAPACK.
            float64 scale = v_f32_sel_geq_u32_b(v_u32_and_b(as_uint64(pk), 0x7FFFFFFFu), 0x5F800000u,
                                                0x1p-64f, 1.0f);
            float64 pks = v_f32_mul_b(pk, scale);
            float64 ninv = v_f32_sub_b(0.0f, lut_free_recip(pks));
            float64 cs = v_f32_mul_b(col, scale);
            float64 q = v_f32_mul_b(cs, ninv);
            float64 r = v_f32_mac_b(q, pks, cs);  // cs + q * pivot, exact in one FMA
            // no correction when 1/pivot is 0 (an infinite or zero pivot): 0 * inf in r would make every q NaN
            q = v_f32_sel_eq_f32_b(ninv, 0.0f, q, v_f32_mac_b(r, ninv, q));
            // |a_ik / pivot| < 2^-126 flushes q to zero although q * (pivot row) can be normal: those rows also take
            // qs = q * 2^64 (from col * 2^64, |col| < 2^-126 |pivot| <= 4 so it does not overflow) against the pivot
            // row * 2^-64; any other row gets qs = 0. Entries of the scaled row that flush, or are not finite, are 0:
            // their true products are subnormal (|q| < 2^-126, |entry| < 2^-62), or q * inf is already NaN in the
            // first FMA, and a 0 * inf there would turn the normal rows' inf into NaN.
            float64 qs = v_f32_mul_b(v_f32_mul_b(col, v_f32_sel_eq_f32_b(scale, 1.0f, 0x1p64f, 1.0f)), ninv);
            qs = v_f32_sel_eq_f32_b(q, 0.0f, qs, 0.0f);
            float64 aps = v_f32_sel_less_u32_b(v_u32_and_b(as_uint64(ap), 0x7FFFFFFFu), 0x7F800000u,
                                               v_f32_mul_b(ap, 0x1p-64f), 0.0f);
            float64 bps = v_f32_sel_less_u32_b(v_u32_and_b(as_uint64(bp), 0x7FFFFFFFu), 0x7F800000u,
                                               v_f32_mul_b(bp, 0x1p-64f), 0.0f);
            for (int i = 0; i < n; i++)
            {
                if (i == k)
                {
                    continue;
                }
                float64 l = lane_bcast(q, i);
                float64 ls = lane_bcast(qs, i);
                a[i] = v_f32_mac_b(aps, ls, v_f32_mac_b(ap, l, a[i]));
                b[i] = v_f32_mac_b(bps, ls, v_f32_mac_b(bp, l, b[i]));
            }
            // A pivot row of |pivot| >= 2 is stored scaled down by ns = 2^-e (pivot = 2^e m, m in [1, 2); exact but
            // for entries below 2^-126 x pivot, which flush): a used row's later quotient is then bounded by its
            // entries over the new pivot, not by them times its own pivot, which overflowed on A^T of [[2^64, 2^-64],
            // [2^64, 2^-63]] (the used row's quotient 2^128, NaN; as Codex review #17 in gj_large). Never scaled up:
            // B's row times 2^-e > 1 could overflow (Codex #3's 2^127 right-hand side). e = 127 takes 2^-126 (2^-127
            // is subnormal); a non-finite pivot is not scaled. The final division takes pivot x ns.
            const uint64 pe = v_u32_shr_b(v_u32_and_b(as_uint64(pk), 0x7F800000u), 23);
            uint64 nsb = v_u32_shl_b(254u - pe, 23);
            nsb = v_u32_sel_geq_u32_b(pe, 254u, 0x00800000u, nsb);
            nsb = v_u32_sel_eq_u32_b(pe, 255u, 0x3F800000u, nsb);
            nsb = v_u32_sel_less_u32_b(pe, 128u, 0x3F800000u, nsb);  // |pivot| < 2 (and zero, subnormal): not scaled
            const float64 ns = as_float64(nsb);
            a[k] = v_f32_mul_b(ap, ns);
            b[k] = v_f32_mul_b(bp, ns);
            dsv = v_f32_sel_eq_u32_b(lane, (unsigned)k, v_f32_mul_b(pk, ns), dsv);
        }
        // A is now diagonal (the pivots dv, up to the unread residues left of each diagonal): X row i = B row i / d_i,
        // with the same 2^-64 scaling of huge pivots; a zero pivot gives a zero row
        float64 dscale = v_f32_sel_geq_u32_b(v_u32_and_b(as_uint64(dsv), 0x7FFFFFFFu), 0x5F800000u, 0x1p-64f, 1.0f);
        float64 dinv = lut_free_recip(v_f32_mul_b(dsv, dscale));
        // a NaN in A: X and D all NaN and info 0, as LAPACK's (its NaN pivots are not zero); the elimination's need
        // not be (a NaN candidate never wins the pivot search, so a column of zeros could report a zero pivot)
        float64 anynan = 0.0f;
        for (int l = 0; l < 64; l++)
        {
            anynan = v_f32_sel_eq_f32_b(lane_bcast(nanv, l), 1.0f, 1.0f, anynan);
        }
        const float64 qnan = as_float64(v_u32_or_b(v_u32_and_b(as_uint64(anynan), 0u), 0x7FC00000u));
        for (int i = 0; i < n; i++)
        {
            b[i] = v_f32_mul_b(v_f32_mul_b(b[i], lane_bcast(dscale, i)), lane_bcast(dinv, i));
            b[i] = v_f32_sel_eq_f32_b(anynan, 1.0f, qnan, b[i]);
        }
        dv = v_f32_sel_eq_f32_b(anynan, 1.0f, qnan, dv);
        infov = v_f32_sel_eq_f32_b(anynan, 1.0f, 0.0f, infov);
        for (int i = 0; i < n; i++)
        {
            int5 c = {0, i, bt, 0, 0};
            if (m < 64)
            {
                v_f32_st_tnsr_partial(c, X, b[i], (char)(m - 1), 0);
            }
            else
            {
                v_f32_st_tnsr(c, X, b[i]);
            }
        }
        int5 cd = {0, bt, 0, 0, 0};
        if (n < 64)
        {
            v_f32_st_tnsr_partial(cd, D, dv, (char)(n - 1), 0);
        }
        else
        {
            v_f32_st_tnsr(cd, D, dv);
        }
        v_f32_st_tnsr_partial(cd, S, v_f32_sel_eq_u32_b(lane, 1u, infov, sv), 1, 0);
    }
}
