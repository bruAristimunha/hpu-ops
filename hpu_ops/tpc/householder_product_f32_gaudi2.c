/* Batched Householder product Q = (H_0 ... H_{k-1})[:, :n] on Gaudi2 TPC for m >= n >= k >= 1, m <= 64.
 *
 * Tensors (TPC geometry, dim0 innermost; torch [batch, rows, cols] row-major):
 *   A_t [m, n, batch]  f32: A^T so column j of A sits in contiguous TPC row j
 *   tau [k, batch]     f32: Householder scalar multipliers
 *   Q_t [m, n, batch]  f32: Q^T (row j = column j of Q)
 *
 * LUT-free (no reduce_*, reciprocal, rsqrt, sqrt) -> 80 KB VLM; q[64] uses 16 KB.
 */
#define MAXM 64

__local__ float64 q[MAXM];

static inline float64 reduce_add_64(float64 p)
{
    p = v_f32_add_b(p, v_f32_mov_dual_group_all_b(p, 0xFFFFFFFF, 1, 0, 3, 2, SW_MDG_ALL, p));
    p = v_f32_add_b(p, v_f32_mov_dual_group_all_b(p, 0xFFFFFFFF, 2, 3, 0, 1, SW_MDG_ALL, p));
    p = v_f32_add_b(p, as_float64(v_element_shift_xor_i32(as_int64(p), 8)));
    p = v_f32_add_b(p, as_float64(v_element_shift_xor_i32(as_int64(p), 4)));
    p = v_f32_add_b(p, as_float64(v_element_shift_xor_i32(as_int64(p), 2)));
    p = v_f32_add_b(p, as_float64(v_element_shift_xor_i32(as_int64(p), 1)));
    return p;
}

void main(tensor A_t, tensor tau, tensor Q_t)
{
    const int m = get_dim_size(A_t, 0);
    const int n = get_dim_size(A_t, 1);
    const int k = get_dim_size(tau, 0);
    const uint64 lane = read_lane_id_4b_b();
    const int5 off = get_index_space_offset();
    const int5 size = get_index_space_size();

    for (int bt = off[0]; bt < off[0] + size[0]; bt++)
    {
        for (int j = 0; j < k; j++)
        {
            int5 c = {0, j, bt, 0, 0};
            q[j] = v_f32_sel_geq_u32_b(lane, (unsigned)m, 0.0f, v_f32_ld_tnsr_b(c, A_t));
        }
        for (int j = k; j < n; j++)
        {
            q[j] = v_f32_sel_eq_u32_b(lane, (unsigned)j, 1.0f, 0.0f);
        }
        int5 ct = {0, bt, 0, 0, 0};
        float64 tau_vec = v_f32_sel_geq_u32_b(lane, (unsigned)k, 0.0f, v_f32_ld_tnsr_b(ct, tau));

        for (int i = k - 1; i >= 0; i--)
        {
            float64 tau_i = reduce_add_64(v_f32_sel_eq_u32_b(lane, (unsigned)i, tau_vec, 0.0f));
            float64 vi = v_f32_sel_less_u32_b(lane, (unsigned)i, 0.0f, q[i]);
            vi = v_f32_sel_eq_u32_b(lane, (unsigned)i, 1.0f, vi);
            float64 neg_tau_vi = v_f32_mul_b(v_f32_sub_b(0.0f, tau_i), vi);

            int j = i + 1;
            for (; j + 3 < n; j += 4)
            {
                float64 q0 = q[j], q1 = q[j + 1], q2 = q[j + 2], q3 = q[j + 3];
                float64 d0 = reduce_add_64(v_f32_mul_b(vi, q0));
                float64 d1 = reduce_add_64(v_f32_mul_b(vi, q1));
                float64 d2 = reduce_add_64(v_f32_mul_b(vi, q2));
                float64 d3 = reduce_add_64(v_f32_mul_b(vi, q3));
                q[j]     = v_f32_mac_b(neg_tau_vi, d0, q0);
                q[j + 1] = v_f32_mac_b(neg_tau_vi, d1, q1);
                q[j + 2] = v_f32_mac_b(neg_tau_vi, d2, q2);
                q[j + 3] = v_f32_mac_b(neg_tau_vi, d3, q3);
            }
            for (; j < n; j++)
            {
                float64 qj = q[j];
                float64 d = reduce_add_64(v_f32_mul_b(vi, qj));
                q[j] = v_f32_mac_b(neg_tau_vi, d, qj);
            }
            q[i] = v_f32_add_b(v_f32_sel_eq_u32_b(lane, (unsigned)i, 1.0f, 0.0f), neg_tau_vi);
        }

        for (int j = 0; j < n; j++)
        {
            int5 c = {0, j, bt, 0, 0};
            if (m == 64)
            {
                v_f32_st_tnsr(c, Q_t, q[j]);
            }
            else
            {
                v_f32_st_tnsr_partial(c, Q_t, q[j], m - 1, 0);
            }
        }
    }
}
