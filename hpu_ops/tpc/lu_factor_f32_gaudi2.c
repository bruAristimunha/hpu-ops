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
 * Lane k of a row is broadcast to all lanes by a masked reduce; no lane is ever moved to a scalar.
 *
 * With -DPERM_FROM_PIVOTS the kernel takes PIVF [n, batch] f32 (LAPACK's 1-based pivots as floats) and writes
 * PERM [n, batch] f32, the row permutation (row 1 of PV above): the swaps replayed in order. lu_solve and
 * lu_unpack use it, so a permutation costs one launch instead of n gathers.
 */
#define MAXN 32

__local__ float64 a[MAXN];

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
            uint64 p = v_f32_reduce_argmax(cand).v1;  // pivot row, in every lane

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
            float64 inv = v_f32_sel_eq_f32_b(pk, 0.0f, 0.0f, v_reciprocal_f32(pk));
            for (int i = k + 1; i < n; i++)
            {
                float64 l = v_f32_mul_b(lane_bcast(a[i], lane, k), inv);
                float64 upd = v_f32_mac_b(ap, v_f32_sub_b(0.0f, l), a[i]);
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
