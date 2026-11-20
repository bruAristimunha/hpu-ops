/* Batched solve A X = B on Gaudi2 TPC: Gauss-Jordan with partial pivoting, one matrix per index-space member.
 *
 * Tensors (TPC geometry, dim0 innermost; torch [batch, rows, cols] row-major):
 *   A [n, n, batch]  f32     B [m, n, batch]  f32     P [m + 2, n, batch]  f32: the ONE output, row i =
 *   lanes 0..m-1 X[i], lane m = D[i] (the pivot at step i, U's diagonal of P A = L U), lane m + 1 = (-1)^swaps in
 *   row 0, info (1 + first zero pivot, kernel's order) in row 1, 0 in the other rows.
 * D and S give det = S0 * prod(D) and slogdet. A zero pivot zeroes the pivot row, so later pivots stay finite.
 * With -DIDENTITY_B the kernel takes A alone and solves against the identity (inv): m = n, no B tensor.
 * Row i of A and of B is one 64-lane vector in local memory (lane j = column j), so n, m <= MAXN.
 * Lane k of a row is broadcast to all lanes by a masked reduce; no lane is ever moved to a scalar.
 * ponytail: masked-reduce broadcasts (~2n^2 per matrix); the shuffle broadcast is cheaper once its map is pinned.
 */
#define MAXN 32

__local__ float64 a[MAXN];
__local__ float64 b[MAXN];

float64 lane_bcast(float64 v, uint64 lane, int k)
{
    return v_f32_reduce_add(v_f32_sel_eq_u32_b(lane, (unsigned)k, v, 0.0f));
}

#ifdef IDENTITY_B
void main(tensor A, tensor P)
#else
void main(tensor A, tensor B, tensor P)
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
        for (int i = 0; i < n; i++)
        {
            int5 c = {0, i, bt, 0, 0};
            a[i] = v_f32_ld_tnsr_b(c, A);
#ifdef IDENTITY_B
            b[i] = v_f32_sel_eq_u32_b(lane, (unsigned)i, 1.0f, 0.0f);
#else
            b[i] = v_f32_ld_tnsr_b(c, B);
#endif
        }
        float64 dv = 0.0f;
        float64 sv = 1.0f;
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
            float64 pk = lane_bcast(ap, lane, k);
            dv = v_f32_sel_eq_u32_b(lane, (unsigned)k, pk, dv);
            sv = v_f32_sel_eq_u32_b(p, (unsigned)k, sv, v_f32_sub_b(0.0f, sv));
            infov = v_f32_sel_eq_f32_b(pk, 0.0f, v_f32_sel_eq_f32_b(infov, 0.0f, (float)(k + 1), infov), infov);
            // a zero pivot gives a zero pivot row: its elimination is a no-op, as in LAPACK
            float64 inv = v_f32_sel_eq_f32_b(pk, 0.0f, 0.0f, v_reciprocal_f32(pk));
            ak = v_f32_mul_b(ap, inv);
            bk = v_f32_mul_b(bp, inv);
            a[k] = ak;
            b[k] = bk;
            for (int i = 0; i < n; i++)
            {
                if (i == k)
                {
                    continue;
                }
                float64 nf = v_f32_sub_b(0.0f, lane_bcast(a[i], lane, k));
                a[i] = v_f32_mac_b(ak, nf, a[i]);
                b[i] = v_f32_mac_b(bk, nf, b[i]);
            }
        }
        for (int i = 0; i < n; i++)
        {
            int5 c = {0, i, bt, 0, 0};
            float64 row = v_f32_sel_eq_u32_b(lane, (unsigned)m, lane_bcast(dv, lane, i), b[i]);
            float64 extra = (i == 0) ? sv : ((i == 1) ? infov : 0.0f);
            row = v_f32_sel_eq_u32_b(lane, (unsigned)(m + 1), extra, row);
            v_f32_st_tnsr(c, P, row);
        }
    }
}
