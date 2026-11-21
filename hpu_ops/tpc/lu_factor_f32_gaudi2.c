/* Batched LU with partial pivoting on Gaudi2 TPC (LAPACK getrf's factors), one matrix per index-space member.
 *
 * Tensors (TPC geometry, dim0 innermost; torch [batch, rows, cols] row-major):
 *   A  [n, n, batch]  f32
 *   LU [n, n, batch]  f32: L (unit diagonal, not stored) below the diagonal and U on and above it, P A = L U
 *   PV [n, 3, batch]  f32: row 0 = LAPACK's 1-based pivots (row k was swapped with row pivots[k] - 1 at step k),
 *                          row 1 = the row permutation perm (LU's row i is A's row perm[i]),
 *                          row 2 lane 0 = info (1 + the first zero pivot, LAPACK's rule, else 0)
 * Row i of A is one 64-lane vector in local memory (lane j = column j), so n <= MAXN. Whole rows are swapped, as
 * LAPACK's laswp does, so the multipliers of earlier steps follow their row. A zero pivot (its whole column below
 * is zero, as the pivot is the largest) gives zero multipliers: the step is a no-op, as in LAPACK.
 * Lane k of a row is broadcast to all lanes by a masked reduce; no lane is ever moved to a scalar. The pivot is the
 * first largest |a_ik| (LAPACK's rule, also with ties); l_ik = (a_ik s) (1 / (pivot s)), s = 2^-64 for |pivot| >= 2^64.
 *
 * With -DPERM_FROM_PIVOTS the kernel takes PIVF [n, batch] f32 (LAPACK's 1-based pivots as floats) and writes
 * PERM [n, batch] f32, the row permutation (row 1 of PV above): the swaps replayed in order. lu_solve and
 * lu_unpack use it, so a permutation costs one launch instead of n gathers.
 */
#define MAXN 32
#define TWO_64 1.8446744073709552e19f    // 2^64: pivots at or above it are scaled by 2^-64 before the reciprocal,
#define TWO_M64 5.421010862427522e-20f   // so 1/pivot (below 2^-126 past 2^126) is never flushed to zero

__local__ float64 a[MAXN];

#ifndef PERM_FROM_PIVOTS
// Max / min over lanes 0..31 (and, separately, over lanes 32..63), in every lane: n <= 32, so the lower half holds
// every row. The same reductions as matrix_exp_f32_gaudi2.c.
static inline float64 reduce_max_half32(float64 p)
{
    p = v_f32_max_b(p, v_f32_mov_dual_group_all_b(p, 0xFFFFFFFF, 1, 0, 3, 2, SW_MDG_ALL, p));
    p = v_f32_max_b(p, as_float64(v_element_shift_xor_i32(as_int64(p), 8)));
    p = v_f32_max_b(p, as_float64(v_element_shift_xor_i32(as_int64(p), 4)));
    p = v_f32_max_b(p, as_float64(v_element_shift_xor_i32(as_int64(p), 2)));
    p = v_f32_max_b(p, as_float64(v_element_shift_xor_i32(as_int64(p), 1)));
    return p;
}

static inline float64 reduce_min_half32(float64 p)
{
    p = v_f32_min_b(p, v_f32_mov_dual_group_all_b(p, 0xFFFFFFFF, 1, 0, 3, 2, SW_MDG_ALL, p));
    p = v_f32_min_b(p, as_float64(v_element_shift_xor_i32(as_int64(p), 8)));
    p = v_f32_min_b(p, as_float64(v_element_shift_xor_i32(as_int64(p), 4)));
    p = v_f32_min_b(p, as_float64(v_element_shift_xor_i32(as_int64(p), 2)));
    p = v_f32_min_b(p, as_float64(v_element_shift_xor_i32(as_int64(p), 1)));
    return p;
}
#endif

float64 lane_bcast(float64 v, uint64 lane, int k)
{
    return v_f32_reduce_add(v_f32_sel_eq_u32_b(lane, (unsigned)k, v, 0.0f));
}

float64 lane_bcast_v(float64 v, uint64 lane, uint64 k)
{
    return v_f32_reduce_add(v_f32_sel_eq_u32_b(lane, k, v, 0.0f));
}

#ifdef PERM_FROM_PIVOTS
void main(tensor PIVF, tensor PERM)
{
    const int n = get_dim_size(PIVF, 0);
    const uint64 lane = read_lane_id_4b_b();
    const int5 off = get_index_space_offset();
    const int5 size = get_index_space_size();

    float64 iota = 0.0f;
    for (int i = 0; i < n; i++)
    {
        iota = v_f32_sel_eq_u32_b(lane, (unsigned)i, (float)i, iota);
    }
    for (int bt = off[0]; bt < off[0] + size[0]; bt++)
    {
        int5 c = {0, bt, 0, 0, 0};
        float64 pv = v_f32_ld_tnsr_b(c, PIVF);
        float64 perm = iota;
        for (int k = 0; k < n; k++)
        {
            float64 pf = v_f32_sub_b(lane_bcast(pv, lane, k), 1.0f);  // 0-based pivot row, in every lane
            float64 perm_k = lane_bcast(perm, lane, k);
            float64 perm_p = v_f32_reduce_add(v_f32_sel_eq_f32_b(iota, pf, perm, 0.0f));
            perm = v_f32_sel_eq_f32_b(iota, pf, perm_k, perm);
            perm = v_f32_sel_eq_u32_b(lane, (unsigned)k, perm_p, perm);
        }
        v_f32_st_tnsr(c, PERM, perm);
    }
}
#else
void main(tensor A, tensor LU, tensor PV)
{
    const int n = get_dim_size(A, 0);
    const uint64 lane = read_lane_id_4b_b();
    const int5 off = get_index_space_offset();
    const int5 size = get_index_space_size();

    float64 iota = 0.0f;  // lane i = i, as a float
    for (int i = 0; i < n; i++)
    {
        iota = v_f32_sel_eq_u32_b(lane, (unsigned)i, (float)i, iota);
    }
    const float64 iota_all = v_convert_u32_to_f32_b(lane);  // lane i = i, every lane

    for (int bt = off[0]; bt < off[0] + size[0]; bt++)
    {
        for (int i = 0; i < n; i++)
        {
            int5 c = {0, i, bt, 0, 0};
            a[i] = v_f32_ld_tnsr_b(c, A);
        }
        float64 piv = 0.0f;
        float64 perm = iota;
        float64 infov = 0.0f;
        for (int k = 0; k < n; k++)
        {
            // column k as a vector over rows: lane i = A[i][k]
            float64 col = 0.0f;
            for (int i = k; i < n; i++)
            {
                col = v_f32_sel_eq_u32_b(lane, (unsigned)i, lane_bcast(a[i], lane, k), col);
            }
            float64 cand = v_f32_abs_b(col);
            cand = v_f32_sel_less_u32_b(lane, (unsigned)k, -1.0f, cand);
            cand = v_f32_sel_geq_u32_b(lane, (unsigned)n, -1.0f, cand);
            // pivot row, in every lane: the FIRST row of largest |a_ik| (LAPACK's idamax; the hardware argmax breaks
            // ties in another order): the smallest lane whose candidate equals the max
            float64 mx = reduce_max_half32(cand);
            float64 lane_f = v_f32_sel_eq_f32_b(cand, mx, iota_all, 100.0f);
            uint64 p = v_convert_f32_to_u32_b(reduce_min_half32(lane_f));
            p = v_u32_sel_less_u32_b(p, (unsigned)k, (unsigned)k, p);
            p = v_u32_sel_geq_u32_b(p, (unsigned)n, (unsigned)k, p);

            // pivots[k] = p + 1; perm: swap lanes k and p
            float64 pf = lane_bcast_v(iota, lane, p);
            piv = v_f32_sel_eq_u32_b(lane, (unsigned)k, v_f32_add_b(pf, 1.0f), piv);
            float64 perm_k = lane_bcast(perm, lane, k);
            float64 perm_p = lane_bcast_v(perm, lane, p);
            perm = v_f32_sel_eq_u32_b(lane, p, perm_k, perm);
            perm = v_f32_sel_eq_u32_b(lane, (unsigned)k, perm_p, perm);

            // swap whole rows k and p with selects (no data-dependent branch)
            float64 ak = a[k], ap = ak;
            for (int i = k + 1; i < n; i++)
            {
                ap = v_f32_sel_eq_u32_b(p, (unsigned)i, a[i], ap);
            }
            for (int i = k + 1; i < n; i++)
            {
                a[i] = v_f32_sel_eq_u32_b(p, (unsigned)i, ak, a[i]);
            }
            a[k] = ap;
            float64 pk = lane_bcast(ap, lane, k);
            infov = v_f32_sel_eq_f32_b(pk, 0.0f, v_f32_sel_eq_f32_b(infov, 0.0f, (float)(k + 1), infov), infov);
            float64 sc = v_f32_sel_less_f32_b(v_f32_abs_b(pk), TWO_64, 1.0f, TWO_M64);
            float64 inv = v_f32_sel_eq_f32_b(pk, 0.0f, 0.0f, v_reciprocal_f32(v_f32_mul_b(pk, sc)));
            // a multiplier below 2^-126 flushes (L keeps 0, as the card holds no subnormal), but its product with the
            // pivot row can be normal: those rows also take ls = l * 2^64 against the pivot row * 2^-64, so equal
            // rows still cancel (Codex review #5: [[2^100, 2^100], [2^-30, 2^-30]] kept U[1, 1] = 2^-30, info 0)
            float64 up = v_f32_sel_eq_f32_b(sc, 1.0f, TWO_64, 1.0f);
            float64 aps = v_f32_sel_less_u32_b(v_u32_and_b(as_uint64(ap), 0x7FFFFFFFu), 0x7F800000u,
                                               v_f32_mul_b(ap, TWO_M64), 0.0f);
            for (int i = k + 1; i < n; i++)
            {
                float64 aik = lane_bcast(a[i], lane, k);
                float64 l = v_f32_mul_b(v_f32_mul_b(aik, sc), inv);
                float64 ls = v_f32_sel_eq_f32_b(l, 0.0f, v_f32_mul_b(v_f32_mul_b(aik, up), inv), 0.0f);
                float64 upd = v_f32_mac_b(aps, v_f32_sub_b(0.0f, ls), v_f32_mac_b(ap, v_f32_sub_b(0.0f, l), a[i]));
                float64 row = v_f32_sel_less_u32_b(lane, (unsigned)(k + 1), a[i], upd);  // lanes <= k keep L
                a[i] = v_f32_sel_eq_u32_b(lane, (unsigned)k, l, row);
            }
        }
        for (int i = 0; i < n; i++)
        {
            int5 c = {0, i, bt, 0, 0};
            v_f32_st_tnsr(c, LU, a[i]);
        }
        int5 c0 = {0, 0, bt, 0, 0};
        int5 c1 = {0, 1, bt, 0, 0};
        int5 c2 = {0, 2, bt, 0, 0};
        v_f32_st_tnsr(c0, PV, piv);
        v_f32_st_tnsr(c1, PV, perm);
        v_f32_st_tnsr(c2, PV, infov);
    }
}
#endif
