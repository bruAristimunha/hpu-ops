/* Batched Householder QR on Gaudi2 TPC, LUT-free (80 KB VLM), m <= 256, n <= 64.
 *
 * Modes:
 *   - Default (VA in {1..4}): returns (Q_t [m, k, batch], R_t [k, n, batch]), k = min(m, n).
 *   - -DR_ONLY (VA in {1..4}): returns R_t [k, n, batch] only.
 *   - -DY_TAU (VA in {1..4}): returns packed (YR_t [m, n, batch], tau [k, batch]) where upper triangle of YR
 *     is R and strict lower triangle holds Householder vectors v_i (implicit 1 on diagonal).
 *
 * Tensors (TPC geometry, dim0 innermost; torch [batch, rows, cols] row-major):
 *   A_t  [m, n, batch] f32: A^T so column j of A sits in contiguous TPC row j
 *   Q_t  [m, k, batch] f32: Q^T (row j = column j of Q)
 *   R_t  [k, n, batch] f32: R^T (row j = column j of R, upper triangular)
 *   YR_t [m, n, batch] f32: packed Householder factors & R (transposed)
 *   tau  [k, batch]    f32: Householder scalar multipliers
 *
 * Column-max prescaling (xs = x / mx, zero_col guard -> tau = 0) and per-column power-of-2 scaling during
 * trailing updates keep 1e-37..1e37 and zero/rank-deficient columns free of overflow and FTZ underflow.
 * LUT-free arithmetic preserves 80 KB VLM: q[64 * 4] = 256 vectors = 64 KB at VA = 4.
 */
#ifndef VA
#define VA 4
#endif
#define MAXN 64

__local__ float64 q[MAXN * VA];

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

static inline float64 lut_free_recip(float64 x)
{
    uint64 bits = as_uint64(x);
    uint64 sign = v_u32_and_b(bits, 0x80000000u);
    uint64 abits = v_u32_and_b(bits, 0x7FFFFFFFu);

    uint64 sub_bits = as_uint64(v_f32_sub_b(as_float64(v_u32_or_b(abits, 0x3F800000u)), 1.0f));
    uint64 nbits = v_u32_sel_less_u32_b(abits, 0x00800000u, sub_bits, abits);
    uint64 e_adj = v_u32_sel_less_u32_b(abits, 0x00800000u, 126u, 0u);

    uint64 mant = v_u32_and_b(nbits, 0x007FFFFFu);
    float64 m = as_float64(v_u32_or_b(mant, 0x3F000000u));

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
    uint64 e_biased = (ey + 382u + e_adj) - e_in;

    uint64 res_norm = v_u32_or_b(sign, v_u32_or_b(v_u32_shl_b(e_biased - 256u, 23), ymant));
    uint64 shift = v_u32_sel_less_u32_b(e_biased, 257u, 257u - e_biased, 0u);
    uint64 res_sub = v_u32_or_b(sign, v_u32_shr_b(v_u32_or_b(0x00800000u, ymant), as_int64(shift)));
    uint64 res = v_u32_sel_less_u32_b(e_biased, 257u, res_sub, res_norm);

    res = v_u32_sel_geq_u32_b(e_biased, 511u, v_u32_or_b(sign, 0x7F800000u), res);
    res = v_u32_sel_eq_u32_b(abits, 0x7F800000u, sign, res);
    res = v_u32_sel_eq_u32_b(abits, 0u, 0u, res);
    res = v_u32_sel_grt_u32_b(abits, 0x7F800000u, bits, res);
    return as_float64(res);
}

static inline float64 rsqrt_fine(float64 x)
{
    x = v_f32_sel_eq_f32_b(x, 0.0f, 1.0f, x);
    float64 y = as_float64(0x5F375A86u - v_u32_shr_b(as_uint64(x), 1));
    float64 hx = v_f32_mul_b(x, -0.5f);
    y = v_f32_mul_b(y, v_f32_mac_b(hx, v_f32_mul_b(y, y), 1.5f));
    y = v_f32_mul_b(y, v_f32_mac_b(hx, v_f32_mul_b(y, y), 1.5f));
    float64 e = v_f32_mac_b(v_f32_sub_b(0.0f, v_f32_mul_b(x, y)), y, 1.0f);
    return v_f32_mac_b(v_f32_mul_b(y, 0.5f), e, y);
}

#ifdef R_ONLY
void main(tensor A_t, tensor R_t)
#elif defined(Y_TAU)
void main(tensor A_t, tensor YR_t, tensor tau_out)
#else
void main(tensor A_t, tensor Q_t, tensor R_t)
#endif
{
    const int m = get_dim_size(A_t, 0);
    const int n = get_dim_size(A_t, 1);
    const int k = (m < n) ? m : n;
    const uint64 lane = read_lane_id_4b_b();
    const int5 off = get_index_space_offset();
    const int5 size = get_index_space_size();

    for (int bt = off[0]; bt < off[0] + size[0]; bt++)
    {
        for (int j = 0; j < n; j++)
        {
#pragma unroll
            for (int iv = 0; iv < VA; iv++)
            {
                float64 val = 0.0f;
                if (64 * iv < m)
                {
                    int5 c = {64 * iv, j, bt, 0, 0};
                    val = v_f32_sel_geq_u32_b(lane + (unsigned)(64 * iv), (unsigned)m, 0.0f, v_f32_ld_tnsr_b(c, A_t));
                }
                q[j * VA + iv] = val;
            }
        }

#ifndef R_ONLY
        float64 tau_vec = 0.0f;
#endif

        for (int i = 0; i < k; i++)
        {
            float64 x[VA];
            float64 mx_loc = 0.0f;
#pragma unroll
            for (int iv = 0; iv < VA; iv++)
            {
                uint64 gid = lane + (unsigned)(64 * iv);
                float64 qi = q[i * VA + iv];
                x[iv] = v_f32_sel_less_u32_b(gid, (unsigned)i, 0.0f, v_f32_sel_geq_u32_b(gid, (unsigned)m, 0.0f, qi));
                mx_loc = v_f32_max_b(mx_loc, v_f32_abs_b(x[iv]));
            }
            float64 mx = reduce_max_64(mx_loc);
            float64 zero_col = v_f32_sel_eq_f32_b(mx, 0.0f, 1.0f, 0.0f);
            float64 smx = v_f32_sel_eq_f32_b(zero_col, 1.0f, 1.0f, mx);
            float64 inv_mx = lut_free_recip(smx);

            float64 xs[VA];
            float64 sq_rest_loc = 0.0f;
#pragma unroll
            for (int iv = 0; iv < VA; iv++)
            {
                uint64 gid = lane + (unsigned)(64 * iv);
                xs[iv] = v_f32_mul_b(x[iv], inv_mx);
                float64 tail = v_f32_sel_grt_u32_b(gid, (unsigned)i, xs[iv], 0.0f);
                sq_rest_loc = v_f32_mac_b(tail, tail, sq_rest_loc);
            }
            float64 sigma_rest = reduce_add_64(sq_rest_loc);

            int vi_idx = i >> 6;
            float64 xs_vi = xs[0];
            float64 orig_vi = q[i * VA + 0];
#pragma unroll
            for (int iv = 1; iv < VA; iv++)
            {
                if (iv == vi_idx)
                {
                    xs_vi = xs[iv];
                    orig_vi = q[i * VA + iv];
                }
            }
            float64 xs0 = reduce_add_64(v_f32_sel_eq_u32_b(lane, (unsigned)(i & 63), xs_vi, 0.0f));
            float64 orig_ii = reduce_add_64(v_f32_sel_eq_u32_b(lane, (unsigned)(i & 63), orig_vi, 0.0f));

            float64 sq = v_f32_mac_b(xs0, xs0, sigma_rest);
            float64 safe_sq = v_f32_sel_eq_f32_b(sq, 0.0f, 1.0f, sq);
            float64 inv_nrm = rsqrt_fine(safe_sq);
            float64 nrm_xs = v_f32_mul_b(sq, inv_nrm);
            float64 beta_xs = v_f32_sel_less_f32_b(xs0, 0.0f, nrm_xs, v_f32_sub_b(0.0f, nrm_xs));
            float64 v0 = v_f32_sub_b(xs0, beta_xs);
            float64 safe_v0 = v_f32_sel_eq_f32_b(v0, 0.0f, 1.0f, v0);
            float64 inv_v0 = lut_free_recip(safe_v0);

            float64 tau_i = v_f32_sel_eq_f32_b(sigma_rest, 0.0f, 0.0f, v_f32_mul_b(v_f32_abs_b(v0), inv_nrm));
            tau_i = v_f32_sel_eq_f32_b(zero_col, 1.0f, 0.0f, tau_i);
            float64 r_ii = v_f32_sel_eq_f32_b(sigma_rest, 0.0f, orig_ii, v_f32_mul_b(beta_xs, smx));

            float64 v_vec[VA];
#pragma unroll
            for (int iv = 0; iv < VA; iv++)
            {
                uint64 gid = lane + (unsigned)(64 * iv);
                float64 vv = v_f32_mul_b(xs[iv], inv_v0);
                vv = v_f32_sel_less_u32_b(gid, (unsigned)i, 0.0f, vv);
                vv = v_f32_sel_eq_u32_b(gid, (unsigned)i, 1.0f, vv);
                vv = v_f32_sel_eq_f32_b(tau_i, 0.0f, 0.0f, vv);
                v_vec[iv] = vv;

                float64 qi = q[i * VA + iv];
#ifdef R_ONLY
                float64 new_qi = v_f32_sel_grt_u32_b(gid, (unsigned)i, 0.0f, qi);
                new_qi = v_f32_sel_eq_u32_b(gid, (unsigned)i, r_ii, new_qi);
#else
                float64 new_qi = v_f32_sel_grt_u32_b(gid, (unsigned)i, vv, qi);
                new_qi = v_f32_sel_eq_u32_b(gid, (unsigned)i, r_ii, new_qi);
#endif
                q[i * VA + iv] = new_qi;
            }
#ifndef R_ONLY
            tau_vec = v_f32_sel_eq_u32_b(lane, (unsigned)i, tau_i, tau_vec);
#endif

            for (int j = i + 1; j < n; j++)
            {
                float64 mx_j_loc = 0.0f;
#pragma unroll
                for (int iv = 0; iv < VA; iv++)
                {
                    uint64 gid = lane + (unsigned)(64 * iv);
                    float64 qj_act = v_f32_sel_less_u32_b(gid, (unsigned)i, 0.0f, q[j * VA + iv]);
                    mx_j_loc = v_f32_max_b(mx_j_loc, v_f32_abs_b(qj_act));
                }
                float64 mx_j = reduce_max_64(mx_j_loc);
                float64 is_huge = v_f32_sel_less_f32_b(1.0e30f, mx_j, 1.0f, 0.0f);
                float64 is_tiny = v_f32_sel_eq_f32_b(mx_j, 0.0f, 0.0f, v_f32_sel_less_f32_b(mx_j, 1.0e-25f, 1.0f, 0.0f));
                float64 sc = v_f32_sel_eq_f32_b(is_huge, 1.0f, 0x1p-40f,
                                                v_f32_sel_eq_f32_b(is_tiny, 1.0f, 0x1p+40f, 1.0f));
                float64 sc_inv = v_f32_sel_eq_f32_b(is_huge, 1.0f, 0x1p+40f,
                                                    v_f32_sel_eq_f32_b(is_tiny, 1.0f, 0x1p-40f, 1.0f));

                float64 dot_loc = 0.0f;
#pragma unroll
                for (int iv = 0; iv < VA; iv++)
                {
                    uint64 gid = lane + (unsigned)(64 * iv);
                    float64 qj_act = v_f32_sel_less_u32_b(gid, (unsigned)i, 0.0f, q[j * VA + iv]);
                    float64 qj_sc = v_f32_mul_b(qj_act, sc);
                    dot_loc = v_f32_mac_b(v_vec[iv], qj_sc, dot_loc);
                }
                float64 dot = reduce_add_64(dot_loc);
                float64 neg_tau_dot = v_f32_mul_b(v_f32_sub_b(0.0f, tau_i), dot);

#pragma unroll
                for (int iv = 0; iv < VA; iv++)
                {
                    uint64 gid = lane + (unsigned)(64 * iv);
                    float64 qjv = q[j * VA + iv];
                    float64 qj_act = v_f32_sel_less_u32_b(gid, (unsigned)i, 0.0f, qjv);
                    float64 qj_sc = v_f32_mul_b(qj_act, sc);
                    float64 qj_upd = v_f32_mul_b(v_f32_mac_b(neg_tau_dot, v_vec[iv], qj_sc), sc_inv);
                    float64 qj_fin = v_f32_sel_eq_f32_b(tau_i, 0.0f, qjv, qj_upd);
                    q[j * VA + iv] = v_f32_sel_less_u32_b(gid, (unsigned)i, qjv, qj_fin);
                }
            }
        }

#ifdef Y_TAU
        for (int j = 0; j < n; j++)
        {
#pragma unroll
            for (int iv = 0; iv < VA; iv++)
            {
                if (64 * iv < m)
                {
                    int5 cy = {64 * iv, j, bt, 0, 0};
                    int rem = m - 64 * iv;
                    if (rem >= 64)
                    {
                        v_f32_st_tnsr(cy, YR_t, q[j * VA + iv]);
                    }
                    else
                    {
                        v_f32_st_tnsr_partial(cy, YR_t, q[j * VA + iv], (char)(rem - 1), 0);
                    }
                }
            }
        }
        int5 ct = {0, bt, 0, 0, 0};
        if (k == 64)
        {
            v_f32_st_tnsr(ct, tau_out, tau_vec);
        }
        else
        {
            v_f32_st_tnsr_partial(ct, tau_out, tau_vec, (char)(k - 1), 0);
        }
#else
        for (int j = 0; j < n; j++)
        {
#pragma unroll
            for (int iv = 0; iv < VA; iv++)
            {
                if (64 * iv < k)
                {
                    uint64 gid = lane + (unsigned)(64 * iv);
                    float64 rj = v_f32_sel_grt_u32_b(gid, (unsigned)j, 0.0f, q[j * VA + iv]);
                    int5 cr = {64 * iv, j, bt, 0, 0};
                    int rem = k - 64 * iv;
                    if (rem >= 64)
                    {
                        v_f32_st_tnsr(cr, R_t, rj);
                    }
                    else
                    {
                        v_f32_st_tnsr_partial(cr, R_t, rj, (char)(rem - 1), 0);
                    }
                }
            }
        }
#ifndef R_ONLY
        // Form thin Q_t [m, k, batch] in-place via backward Householder accumulation
        for (int i = k - 1; i >= 0; i--)
        {
            float64 tau_i = reduce_add_64(v_f32_sel_eq_u32_b(lane, (unsigned)i, tau_vec, 0.0f));
            float64 vi[VA], neg_tau_vi[VA];
#pragma unroll
            for (int iv = 0; iv < VA; iv++)
            {
                uint64 gid = lane + (unsigned)(64 * iv);
                float64 qi = q[i * VA + iv];
                float64 vv = v_f32_sel_less_u32_b(gid, (unsigned)i, 0.0f, qi);
                vv = v_f32_sel_eq_u32_b(gid, (unsigned)i, 1.0f, vv);
                vv = v_f32_sel_geq_u32_b(gid, (unsigned)m, 0.0f, vv);
                vi[iv] = vv;
                neg_tau_vi[iv] = v_f32_mul_b(v_f32_sub_b(0.0f, tau_i), vv);
            }

            for (int j = i + 1; j < k; j++)
            {
                float64 d_loc = 0.0f;
#pragma unroll
                for (int iv = 0; iv < VA; iv++)
                {
                    d_loc = v_f32_mac_b(vi[iv], q[j * VA + iv], d_loc);
                }
                float64 d = reduce_add_64(d_loc);
#pragma unroll
                for (int iv = 0; iv < VA; iv++)
                {
                    float64 qjv = q[j * VA + iv];
                    q[j * VA + iv] = v_f32_sel_eq_f32_b(tau_i, 0.0f, qjv, v_f32_mac_b(neg_tau_vi[iv], d, qjv));
                }
            }

#pragma unroll
            for (int iv = 0; iv < VA; iv++)
            {
                uint64 gid = lane + (unsigned)(64 * iv);
                float64 ei = v_f32_sel_eq_u32_b(gid, (unsigned)i, 1.0f, 0.0f);
                q[i * VA + iv] = v_f32_sel_eq_f32_b(tau_i, 0.0f, ei, v_f32_add_b(ei, neg_tau_vi[iv]));
            }
        }

        for (int j = 0; j < k; j++)
        {
#pragma unroll
            for (int iv = 0; iv < VA; iv++)
            {
                if (64 * iv < m)
                {
                    int5 cq = {64 * iv, j, bt, 0, 0};
                    int rem = m - 64 * iv;
                    if (rem >= 64)
                    {
                        v_f32_st_tnsr(cq, Q_t, q[j * VA + iv]);
                    }
                    else
                    {
                        v_f32_st_tnsr_partial(cq, Q_t, q[j * VA + iv], (char)(rem - 1), 0);
                    }
                }
            }
        }
#endif
#endif
    }
}
