/* Batched A X = B from LAPACK's LU factors on Gaudi2 TPC (getrs, no transpose), one matrix per index-space member.
 *
 * Tensors (TPC geometry, dim0 innermost; torch [batch, rows, cols] row-major):
 *   LU   [n, n, batch]  f32: packed factors, P A = L U (L unit lower, not stored diagonal)
 *   PIVF [n, batch]     f32: LAPACK's 1-based pivots as floats
 *   B    [m, n, batch]  f32     X [m, n, batch]  f32
 * The row swaps are replayed on B in order, then forward substitution with L and back substitution with U, as LAPACK
 * does: no inverse is formed. A zero on U's diagonal gives inf/NaN, as LAPACK's trsm. Row i of LU and of B is one
 * 64-lane vector in local memory, so n, m <= MAXN. Lane k of a row is broadcast by a masked reduce.
 */
#define MAXN 32
#define TWO_64 1.8446744073709552e19f   // 2^64: a diagonal at or above it is scaled by 2^-64 before the reciprocal
#define TWO_M64 5.421010862427522e-20f

__local__ float64 lu[MAXN];
__local__ float64 b[MAXN];

float64 lane_bcast(float64 v, uint64 lane, int k)
{
    return v_f32_reduce_add(v_f32_sel_eq_u32_b(lane, (unsigned)k, v, 0.0f));
}

void main(tensor LU, tensor PIVF, tensor B, tensor X)
{
    const int n = get_dim_size(LU, 0);
    const uint64 lane = read_lane_id_4b_b();
    const int5 off = get_index_space_offset();
    const int5 size = get_index_space_size();

    for (int bt = off[0]; bt < off[0] + size[0]; bt++)
    {
        for (int i = 0; i < n; i++)
        {
            int5 c = {0, i, bt, 0, 0};
            lu[i] = v_f32_ld_tnsr_b(c, LU);
            b[i] = v_f32_ld_tnsr_b(c, B);
        }
        int5 cp = {0, bt, 0, 0, 0};
        float64 pv = v_f32_ld_tnsr_b(cp, PIVF);

        // P^T B: swap rows k and pivots[k] - 1, in order
        for (int k = 0; k < n; k++)
        {
            float64 pf = v_f32_sub_b(lane_bcast(pv, lane, k), 1.0f);  // 0-based pivot row, in every lane
            float64 bk = b[k], bp = bk;
            for (int i = k + 1; i < n; i++)
            {
                bp = v_f32_sel_eq_f32_b(pf, (float)i, b[i], bp);
            }
            for (int i = k + 1; i < n; i++)
            {
                b[i] = v_f32_sel_eq_f32_b(pf, (float)i, bk, b[i]);
            }
            b[k] = bp;
        }
        // L Y = P^T B
        for (int i = 1; i < n; i++)
        {
            float64 acc = b[i];
            for (int j = 0; j < i; j++)
            {
                acc = v_f32_mac_b(b[j], v_f32_sub_b(0.0f, lane_bcast(lu[i], lane, j)), acc);
            }
            b[i] = acc;
        }
        // U X = Y
        for (int i = n - 1; i >= 0; i--)
        {
            float64 acc = b[i];
            for (int j = i + 1; j < n; j++)
            {
                acc = v_f32_mac_b(b[j], v_f32_sub_b(0.0f, lane_bcast(lu[i], lane, j)), acc);
            }
            float64 d = lane_bcast(lu[i], lane, i);
            float64 sc = v_f32_sel_less_f32_b(v_f32_abs_b(d), TWO_64, 1.0f, TWO_M64);  // 1/d never flushed to zero
            b[i] = v_f32_mul_b(v_f32_mul_b(acc, sc), v_reciprocal_f32(v_f32_mul_b(d, sc)));
        }
        for (int i = 0; i < n; i++)
        {
            int5 c = {0, i, bt, 0, 0};
            v_f32_st_tnsr(c, X, b[i]);
        }
    }
}
