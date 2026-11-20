/* Batched symmetric eigendecomposition on Gaudi2 TPC: one-sided (Hestenes) Jacobi, one matrix per index-space member.
 *
 * Tensors (TPC geometry, dim0 innermost; torch [batch, rows, cols] row-major):
 *   A [n, n, batch]  f32 (symmetric; the lower triangle is mirrored by the caller)
 *   W [n, batch]     f32: eigenvalues, unsorted          V [n, n, batch]  f32: row j = eigenvector of W[j]
 * A is normalized by 1/sigma (sigma = Gershgorin bound) and shifted to S = A/sigma + 1.001 I so S is strictly
 * positive definite with O(1) entries. For n <= 32, lanes 32..63 are unused, so two disjoint round-robin pairs
 * are packed into lanes 0..31 and 32..63 with a LUT-free 32-lane half-vector reduction (reduce_add_half32),
 * and two packed units (K=4 pairs) are interleaved per loop step. Rayleigh quotient W[j] = v_j^T A v_j is
 * tiled in 8-column VRF blocks using v_broadcast_element_i32 (O(n) global loads and O(n) reduce_add calls).
 */
#define MAXN 32  // u[32] = 8 KB of 16 KB LUT-mode VLM, leaving 8 KB for register spills

__local__ float64 u[MAXN];

static inline float64 reduce_add_half32(float64 p)
{
    p = v_f32_add_b(p, v_f32_mov_dual_group_all_b(p, 0xFFFFFFFF, 1, 0, 3, 2, SW_MDG_ALL, p));
    p = v_f32_add_b(p, as_float64(v_element_shift_xor_i32(as_int64(p), 8)));
    p = v_f32_add_b(p, as_float64(v_element_shift_xor_i32(as_int64(p), 4)));
    p = v_f32_add_b(p, as_float64(v_element_shift_xor_i32(as_int64(p), 2)));
    p = v_f32_add_b(p, as_float64(v_element_shift_xor_i32(as_int64(p), 1)));
    return p;
}

static inline float64 pack_halves(float64 lo, float64 hi)
{
    float64 p = v_f32_mov_dual_group_b(hi, 0xFFFFFFFF, 0, 2, MkWr(1, 1), lo);
    return v_f32_mov_dual_group_b(hi, 0xFFFFFFFF, 1, 3, MkWr(1, 1), p);
}

static inline float64 unpack_hi_half(float64 p)
{
    float64 z = 0.0f;
    float64 out = v_f32_mov_dual_group_b(p, 0xFFFFFFFF, 2, 0, MkWr(1, 1), z);
    return v_f32_mov_dual_group_b(p, 0xFFFFFFFF, 3, 1, MkWr(1, 1), out);
}

#define RR_PAIR(r, k, m, p, q)          \
    do                                  \
    {                                   \
        if ((k) == 0)                   \
        {                               \
            (p) = (r);                  \
            (q) = (m);                  \
        }                               \
        else                            \
        {                               \
            int a_ = (r) + (k);         \
            if (a_ >= (m)) a_ -= (m);   \
            int b_ = (r) - (k);         \
            if (b_ < 0) b_ += (m);      \
            (p) = (a_ < b_) ? a_ : b_;  \
            (q) = (a_ < b_) ? b_ : a_;  \
        }                               \
    } while (0)

void main(tensor A, tensor W, tensor V)
{
    const int n = get_dim_size(A, 0);
    const uint64 lane = read_lane_id_4b_b();
    const int5 off = get_index_space_offset();
    const int5 size = get_index_space_size();
    const int n_even = (n + 1) & ~1;
    const int m = n_even - 1;
    const int half_n = n_even >> 1;
    const int k_start = n & 1;  // when n is odd, k==0 has q == m == n (padding) and is skipped
    const int sweeps = (n <= 4) ? 4 : (n <= 16) ? 6 : (n <= 22) ? 7 : 8;

    for (int bt = off[0]; bt < off[0] + size[0]; bt++)
    {
        float64 sigma = 0.0f;
        for (int i = 0; i < n; i++)
        {
            int5 c = {0, i, bt, 0, 0};
            float64 row = v_f32_ld_tnsr_b(c, A);
            row = v_f32_sel_geq_u32_b(lane, (unsigned)n, 0.0f, row);  // lanes past n are padding
            u[i] = row;
            sigma = v_f32_max_b(sigma, v_f32_reduce_add(v_f32_abs_b(row)));
        }
        // Normalize A by 1/sigma (safe when sigma == 0) and shift diagonal by 1.001f
        float64 safe_sigma = v_f32_sel_eq_f32_b(sigma, 0.0f, 1.0f, sigma);
        float64 inv_sigma = v_reciprocal_f32(safe_sigma);
        inv_sigma = v_f32_mul_b(inv_sigma, v_f32_sub_b(2.0f, v_f32_mul_b(safe_sigma, inv_sigma)));
        const float shift_norm = 1.001f;
        for (int i = 0; i < n; i++)
        {
            float64 row = v_f32_mul_b(u[i], inv_sigma);
            u[i] = v_f32_sel_eq_u32_b(lane, (unsigned)i, v_f32_add_b(row, shift_norm), row);
        }

        for (int sweep = 0; sweep < sweeps; sweep++)
        {
            for (int r = 0; r < m; r++)
            {
                int k = k_start;
                for (; k + 3 < half_n; k += 4)
                {
                    int p0, q0, p1, q1, p2, q2, p3, q3;
                    RR_PAIR(r, k, m, p0, q0);
                    RR_PAIR(r, k + 1, m, p1, q1);
                    RR_PAIR(r, k + 2, m, p2, q2);
                    RR_PAIR(r, k + 3, m, p3, q3);

                    float64 up_a = pack_halves(u[p0], u[p1]);
                    float64 uq_a = pack_halves(u[q0], u[q1]);
                    float64 up_b = pack_halves(u[p2], u[p3]);
                    float64 uq_b = pack_halves(u[q2], u[q3]);

                    float64 alpha_a = reduce_add_half32(v_f32_mul_b(up_a, up_a));
                    float64 beta_a = reduce_add_half32(v_f32_mul_b(uq_a, uq_a));
                    float64 gamma_a = reduce_add_half32(v_f32_mul_b(up_a, uq_a));
                    float64 alpha_b = reduce_add_half32(v_f32_mul_b(up_b, up_b));
                    float64 beta_b = reduce_add_half32(v_f32_mul_b(uq_b, uq_b));
                    float64 gamma_b = reduce_add_half32(v_f32_mul_b(up_b, uq_b));

                    float64 d_a = v_f32_sub_b(beta_a, alpha_a);
                    float64 d_b = v_f32_sub_b(beta_b, alpha_b);
                    float64 g2_a = v_f32_add_b(gamma_a, gamma_a);
                    float64 g2_b = v_f32_add_b(gamma_b, gamma_b);
                    float64 r2_a = v_f32_mac_b(g2_a, g2_a, v_f32_mul_b(d_a, d_a));
                    float64 r2_b = v_f32_mac_b(g2_b, g2_b, v_f32_mul_b(d_b, d_b));
                    float64 sr2_a = v_f32_sel_eq_f32_b(r2_a, 0.0f, 1.0f, r2_a);
                    float64 sr2_b = v_f32_sel_eq_f32_b(r2_b, 0.0f, 1.0f, r2_b);

                    float64 inv_r_a = v_rsqrt_f32(sr2_a);
                    float64 inv_r_b = v_rsqrt_f32(sr2_b);

                    float64 c2_a = v_f32_mul_b(0.5f, v_f32_mac_b(v_f32_abs_b(d_a), inv_r_a, 1.0f));
                    float64 c2_b = v_f32_mul_b(0.5f, v_f32_mac_b(v_f32_abs_b(d_b), inv_r_b, 1.0f));

                    float64 inv_c_a = v_rsqrt_f32(c2_a);
                    float64 inv_c_b = v_rsqrt_f32(c2_b);

                    float64 c_a = v_f32_mul_b(c2_a, inv_c_a);
                    float64 c_b = v_f32_mul_b(c2_b, inv_c_b);
                    float64 s_a = v_f32_mul_b(v_f32_mul_b(0.5f, g2_a), v_f32_mul_b(inv_r_a, inv_c_a));
                    float64 s_b = v_f32_mul_b(v_f32_mul_b(0.5f, g2_b), v_f32_mul_b(inv_r_b, inv_c_b));
                    s_a = v_f32_sel_less_f32_b(d_a, 0.0f, v_f32_sub_b(0.0f, s_a), s_a);
                    s_b = v_f32_sel_less_f32_b(d_b, 0.0f, v_f32_sub_b(0.0f, s_b), s_b);

                    c_a = v_f32_sel_eq_f32_b(gamma_a, 0.0f, 1.0f, c_a);
                    s_a = v_f32_sel_eq_f32_b(gamma_a, 0.0f, 0.0f, s_a);
                    c_a = v_f32_sel_eq_f32_b(r2_a, 0.0f, 1.0f, c_a);
                    s_a = v_f32_sel_eq_f32_b(r2_a, 0.0f, 0.0f, s_a);

                    c_b = v_f32_sel_eq_f32_b(gamma_b, 0.0f, 1.0f, c_b);
                    s_b = v_f32_sel_eq_f32_b(gamma_b, 0.0f, 0.0f, s_b);
                    c_b = v_f32_sel_eq_f32_b(r2_b, 0.0f, 1.0f, c_b);
                    s_b = v_f32_sel_eq_f32_b(r2_b, 0.0f, 0.0f, s_b);

                    float64 ns_a = v_f32_sub_b(0.0f, s_a);
                    float64 ns_b = v_f32_sub_b(0.0f, s_b);
                    float64 np_a = v_f32_mac_b(uq_a, ns_a, v_f32_mul_b(up_a, c_a));
                    float64 nq_a = v_f32_mac_b(up_a, s_a, v_f32_mul_b(uq_a, c_a));
                    float64 np_b = v_f32_mac_b(uq_b, ns_b, v_f32_mul_b(up_b, c_b));
                    float64 nq_b = v_f32_mac_b(up_b, s_b, v_f32_mul_b(uq_b, c_b));

                    u[p0] = v_f32_sel_geq_u32_b(lane, 32u, 0.0f, np_a);
                    u[q0] = v_f32_sel_geq_u32_b(lane, 32u, 0.0f, nq_a);
                    u[p1] = unpack_hi_half(np_a);
                    u[q1] = unpack_hi_half(nq_a);
                    u[p2] = v_f32_sel_geq_u32_b(lane, 32u, 0.0f, np_b);
                    u[q2] = v_f32_sel_geq_u32_b(lane, 32u, 0.0f, nq_b);
                    u[p3] = unpack_hi_half(np_b);
                    u[q3] = unpack_hi_half(nq_b);
                }
                if (k + 1 < half_n)
                {
                    int p0, q0, p1, q1;
                    RR_PAIR(r, k, m, p0, q0);
                    RR_PAIR(r, k + 1, m, p1, q1);
                    k += 2;

                    float64 up_a = pack_halves(u[p0], u[p1]);
                    float64 uq_a = pack_halves(u[q0], u[q1]);

                    float64 alpha_a = reduce_add_half32(v_f32_mul_b(up_a, up_a));
                    float64 beta_a = reduce_add_half32(v_f32_mul_b(uq_a, uq_a));
                    float64 gamma_a = reduce_add_half32(v_f32_mul_b(up_a, uq_a));

                    float64 d_a = v_f32_sub_b(beta_a, alpha_a);
                    float64 g2_a = v_f32_add_b(gamma_a, gamma_a);
                    float64 r2_a = v_f32_mac_b(g2_a, g2_a, v_f32_mul_b(d_a, d_a));
                    float64 sr2_a = v_f32_sel_eq_f32_b(r2_a, 0.0f, 1.0f, r2_a);
                    float64 inv_r_a = v_rsqrt_f32(sr2_a);
                    float64 c2_a = v_f32_mul_b(0.5f, v_f32_mac_b(v_f32_abs_b(d_a), inv_r_a, 1.0f));
                    float64 inv_c_a = v_rsqrt_f32(c2_a);
                    float64 c_a = v_f32_mul_b(c2_a, inv_c_a);
                    float64 s_a = v_f32_mul_b(v_f32_mul_b(0.5f, g2_a), v_f32_mul_b(inv_r_a, inv_c_a));
                    s_a = v_f32_sel_less_f32_b(d_a, 0.0f, v_f32_sub_b(0.0f, s_a), s_a);

                    c_a = v_f32_sel_eq_f32_b(gamma_a, 0.0f, 1.0f, c_a);
                    s_a = v_f32_sel_eq_f32_b(gamma_a, 0.0f, 0.0f, s_a);
                    c_a = v_f32_sel_eq_f32_b(r2_a, 0.0f, 1.0f, c_a);
                    s_a = v_f32_sel_eq_f32_b(r2_a, 0.0f, 0.0f, s_a);

                    float64 ns_a = v_f32_sub_b(0.0f, s_a);
                    float64 np_a = v_f32_mac_b(uq_a, ns_a, v_f32_mul_b(up_a, c_a));
                    float64 nq_a = v_f32_mac_b(up_a, s_a, v_f32_mul_b(uq_a, c_a));

                    u[p0] = v_f32_sel_geq_u32_b(lane, 32u, 0.0f, np_a);
                    u[q0] = v_f32_sel_geq_u32_b(lane, 32u, 0.0f, nq_a);
                    u[p1] = unpack_hi_half(np_a);
                    u[q1] = unpack_hi_half(nq_a);
                }
                if (k < half_n)
                {
                    int p, q;
                    RR_PAIR(r, k, m, p, q);
                    float64 up = u[p], uq = u[q];
                    float64 alpha = v_f32_reduce_add(v_f32_mul_b(up, up));
                    float64 beta = v_f32_reduce_add(v_f32_mul_b(uq, uq));
                    float64 gamma = v_f32_reduce_add(v_f32_mul_b(up, uq));

                    float64 d = v_f32_sub_b(beta, alpha);
                    float64 g2 = v_f32_add_b(gamma, gamma);
                    float64 r2 = v_f32_mac_b(g2, g2, v_f32_mul_b(d, d));
                    float64 safe_r2 = v_f32_sel_eq_f32_b(r2, 0.0f, 1.0f, r2);
                    float64 inv_r = v_rsqrt_f32(safe_r2);
                    float64 c2 = v_f32_mul_b(0.5f, v_f32_mac_b(v_f32_abs_b(d), inv_r, 1.0f));
                    float64 inv_c = v_rsqrt_f32(c2);
                    float64 c = v_f32_mul_b(c2, inv_c);
                    float64 s = v_f32_mul_b(v_f32_mul_b(0.5f, g2), v_f32_mul_b(inv_r, inv_c));
                    s = v_f32_sel_less_f32_b(d, 0.0f, v_f32_sub_b(0.0f, s), s);

                    c = v_f32_sel_eq_f32_b(gamma, 0.0f, 1.0f, c);
                    s = v_f32_sel_eq_f32_b(gamma, 0.0f, 0.0f, s);
                    c = v_f32_sel_eq_f32_b(r2, 0.0f, 1.0f, c);
                    s = v_f32_sel_eq_f32_b(r2, 0.0f, 0.0f, s);

                    float64 ns = v_f32_sub_b(0.0f, s);
                    u[p] = v_f32_mac_b(uq, ns, v_f32_mul_b(up, c));
                    u[q] = v_f32_mac_b(up, s, v_f32_mul_b(uq, c));
                }
            }
        }

        int j = 0;
        for (; j + 1 < n; j += 2)
        {
            float64 up = pack_halves(u[j], u[j + 1]);
            float64 sq = reduce_add_half32(v_f32_mul_b(up, up));
            float64 rn = v_rsqrt_f32(sq);
            // 1 Newton-Raphson step on rn = 1/sqrt(sq)
            rn = v_f32_mul_b(rn, v_f32_mac_b(v_f32_mul_b(-0.5f, sq), v_f32_mul_b(rn, rn), 1.5f));
            float64 vp = v_f32_mul_b(up, rn);
            float64 vj0 = v_f32_sel_geq_u32_b(lane, 32u, 0.0f, vp);
            float64 vj1 = unpack_hi_half(vp);
            u[j] = vj0;
            u[j + 1] = vj1;
            int5 c0 = {0, j, bt, 0, 0};
            int5 c1 = {0, j + 1, bt, 0, 0};
            v_f32_st_tnsr_partial(c0, V, vj0, n - 1, 0);
            v_f32_st_tnsr_partial(c1, V, vj1, n - 1, 0);
        }
        if (j < n)
        {
            float64 sq = v_f32_reduce_add(v_f32_mul_b(u[j], u[j]));
            float64 rn = v_rsqrt_f32(sq);
            rn = v_f32_mul_b(rn, v_f32_mac_b(v_f32_mul_b(-0.5f, sq), v_f32_mul_b(rn, rn), 1.5f));
            float64 vj = v_f32_mul_b(u[j], rn);
            u[j] = vj;
            int5 c = {0, j, bt, 0, 0};
            v_f32_st_tnsr_partial(c, V, vj, n - 1, 0);
        }
        // Rayleigh quotient W[j] = v_j^T A v_j in 8-column VRF blocks using v_broadcast_element_i32
        float64 w = 0.0f;
        for (int j0 = 0; j0 < n; j0 += 8)
        {
            const int c0 = j0;
            const int c1 = (j0 + 1 < n) ? j0 + 1 : 0;
            const int c2 = (j0 + 2 < n) ? j0 + 2 : 0;
            const int c3 = (j0 + 3 < n) ? j0 + 3 : 0;
            const int c4 = (j0 + 4 < n) ? j0 + 4 : 0;
            const int c5 = (j0 + 5 < n) ? j0 + 5 : 0;
            const int c6 = (j0 + 6 < n) ? j0 + 6 : 0;
            const int c7 = (j0 + 7 < n) ? j0 + 7 : 0;
            float64 v0 = u[c0], v1 = u[c1], v2 = u[c2], v3 = u[c3];
            float64 v4 = u[c4], v5 = u[c5], v6 = u[c6], v7 = u[c7];
            float64 av0 = 0.0f, av1 = 0.0f, av2 = 0.0f, av3 = 0.0f;
            float64 av4 = 0.0f, av5 = 0.0f, av6 = 0.0f, av7 = 0.0f;
            for (int i = 0; i < n; i++)
            {
                int5 c = {0, i, bt, 0, 0};
                float64 row = v_f32_ld_tnsr_b(c, A);
                row = v_f32_sel_geq_u32_b(lane, (unsigned)n, 0.0f, row);
                av0 = v_f32_mac_b(row, as_float64(v_broadcast_element_i32(as_int64(v0), i)), av0);
                av1 = v_f32_mac_b(row, as_float64(v_broadcast_element_i32(as_int64(v1), i)), av1);
                av2 = v_f32_mac_b(row, as_float64(v_broadcast_element_i32(as_int64(v2), i)), av2);
                av3 = v_f32_mac_b(row, as_float64(v_broadcast_element_i32(as_int64(v3), i)), av3);
                av4 = v_f32_mac_b(row, as_float64(v_broadcast_element_i32(as_int64(v4), i)), av4);
                av5 = v_f32_mac_b(row, as_float64(v_broadcast_element_i32(as_int64(v5), i)), av5);
                av6 = v_f32_mac_b(row, as_float64(v_broadcast_element_i32(as_int64(v6), i)), av6);
                av7 = v_f32_mac_b(row, as_float64(v_broadcast_element_i32(as_int64(v7), i)), av7);
            }
            float64 w0 = v_f32_reduce_add(v_f32_mul_b(v0, av0));
            float64 w1 = v_f32_reduce_add(v_f32_mul_b(v1, av1));
            float64 w2 = v_f32_reduce_add(v_f32_mul_b(v2, av2));
            float64 w3 = v_f32_reduce_add(v_f32_mul_b(v3, av3));
            float64 w4 = v_f32_reduce_add(v_f32_mul_b(v4, av4));
            float64 w5 = v_f32_reduce_add(v_f32_mul_b(v5, av5));
            float64 w6 = v_f32_reduce_add(v_f32_mul_b(v6, av6));
            float64 w7 = v_f32_reduce_add(v_f32_mul_b(v7, av7));
            w = v_f32_sel_eq_u32_b(lane, (unsigned)(j0 + 0), w0, w);
            if (j0 + 1 < n) w = v_f32_sel_eq_u32_b(lane, (unsigned)(j0 + 1), w1, w);
            if (j0 + 2 < n) w = v_f32_sel_eq_u32_b(lane, (unsigned)(j0 + 2), w2, w);
            if (j0 + 3 < n) w = v_f32_sel_eq_u32_b(lane, (unsigned)(j0 + 3), w3, w);
            if (j0 + 4 < n) w = v_f32_sel_eq_u32_b(lane, (unsigned)(j0 + 4), w4, w);
            if (j0 + 5 < n) w = v_f32_sel_eq_u32_b(lane, (unsigned)(j0 + 5), w5, w);
            if (j0 + 6 < n) w = v_f32_sel_eq_u32_b(lane, (unsigned)(j0 + 6), w6, w);
            if (j0 + 7 < n) w = v_f32_sel_eq_u32_b(lane, (unsigned)(j0 + 7), w7, w);
        }
        int5 cw = {0, bt, 0, 0, 0};
        v_f32_st_tnsr_partial(cw, W, w, n - 1, 0);
    }
}
