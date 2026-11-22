/* Batched complex64 Householder QR on Gaudi2 TPC (100% LUT-free, 16 KB VLM), 1 <= n <= m <= 32.
 *
 * Modes:
 *   - Default:     returns (Q_t [m, 2n, batch], R_t [n, 2n, batch])  (reduced)
 *   - -DFULL_Q:    returns (Q_t [m, 2m, batch], R_t [m, 2n, batch])  (complete, rows n..m-1 of R zeroed)
 *   - -DR_ONLY:    returns R_t [n, 2n, batch] only
 *
 * Transposed planar layout (TPC geometry, dim0 innermost; torch [batch, 2*cols, rows] row-major):
 *   A_t [m, 2n, batch]   f32: planes 0..n-1 = Re(col j), n..2n-1 = Im(col j)
 *   Q_t [m, 2k_q, batch] f32: planes 0..k_q-1 = Re(col j of Q), k_q..2k_q-1 = Im(col j of Q)
 *   R_t [k_r, 2n, batch] f32: planes 0..n-1 = Re(col j of R), n..2n-1 = Im(col j of R)
 *
 * Uses exact LAPACK clarfg sign convention (real diagonal of R; tau = (0, 0) when tail is zero and head is real),
 * power-of-2 column prescaling (pow2_scale / pow2_unscale) across [2^-126, FLT_MAX], and non-finite per-batch
 * poisoning (any NaN/Inf in active lanes 0..m-1 produces NaN everywhere for that batch item).
 * VLM usage: qr[32] + qi[32] = 64 * 256 B = 16 KB (80 KB VLM mode).
 */
#define MAXN 32

__local__ float64 qr[MAXN];
__local__ float64 qi[MAXN];

static inline float64 lane_bcast(float64 v, int k)
{
    return as_float64(v_broadcast_element_i32(as_int64(v), k));
}

static inline float64 reduce_add_half32(float64 p)
{
    p = v_f32_add_b(p, v_f32_mov_dual_group_all_b(p, 0xFFFFFFFF, 1, 0, 3, 2, SW_MDG_ALL, p));
    p = v_f32_add_b(p, as_float64(v_element_shift_xor_i32(as_int64(p), 8)));
    p = v_f32_add_b(p, as_float64(v_element_shift_xor_i32(as_int64(p), 4)));
    p = v_f32_add_b(p, as_float64(v_element_shift_xor_i32(as_int64(p), 2)));
    p = v_f32_add_b(p, as_float64(v_element_shift_xor_i32(as_int64(p), 1)));
    return p;
}

static inline float64 reduce_max_half32(float64 p)
{
    p = v_f32_max_b(p, v_f32_mov_dual_group_all_b(p, 0xFFFFFFFF, 1, 0, 3, 2, SW_MDG_ALL, p));
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
    float64 sc = v_f32_sel_less_f32_b(x, 0x1p-125f, 4.0f, 1.0f);
    float64 out_m = v_f32_sel_less_f32_b(x, 0x1p-125f, 2.0f, 1.0f);
    sc = v_f32_sel_geq_f32_b(x, 0x1p+120f, 0x1p-8f, sc);
    out_m = v_f32_sel_geq_f32_b(x, 0x1p+120f, 0x1p-4f, out_m);
    x = v_f32_mul_b(x, sc);
    float64 y = as_float64(0x5F375A86u - v_u32_shr_b(as_uint64(x), 1));
    float64 hx = v_f32_mul_b(x, -0.5f);
    y = v_f32_mul_b(y, v_f32_mac_b(hx, v_f32_mul_b(y, y), 1.5f));
    y = v_f32_mul_b(y, v_f32_mac_b(hx, v_f32_mul_b(y, y), 1.5f));
    float64 e = v_f32_mac_b(v_f32_sub_b(0.0f, v_f32_mul_b(x, y)), y, 1.0f);
    return v_f32_mul_b(v_f32_mac_b(v_f32_mul_b(y, 0.5f), e, y), out_m);
}

static inline float64 pow2_scale(float64 m)
{
    uint64 e = v_u32_shr_b(v_u32_and_b(as_uint64(m), 0x7F800000u), 23);
    uint64 e_eff = v_u32_min_b(e, 253u);
    return as_float64(v_u32_sel_eq_u32_b(e, 0u, 0x3F800000u, v_u32_shl_b(254u - e_eff, 23)));
}

static inline float64 pow2_unscale(float64 m)
{
    uint64 e = v_u32_shr_b(v_u32_and_b(as_uint64(m), 0x7F800000u), 23);
    uint64 e_eff = v_u32_min_b(e, 253u);
    return as_float64(v_u32_sel_eq_u32_b(e, 0u, 0x3F800000u, v_u32_shl_b(e_eff, 23)));
}

static inline float64 nonfinite(float64 x, float64 bad)
{
    return v_f32_sel_eq_u32_b(v_u32_and_b(as_uint64(x), 0x7F800000u), 0x7F800000u, 1.0f, bad);
}

#ifdef R_ONLY
void main(tensor A_t, tensor R_t)
#else
void main(tensor A_t, tensor Q_t, tensor R_t)
#endif
{
    const int m = get_dim_size(A_t, 0);
    const int n = get_dim_size(A_t, 1) >> 1;
#ifdef FULL_Q
    const int k_r = m;
    const int k_q = m;
#else
    const int k_r = n;
#ifndef R_ONLY
    const int k_q = n;
#endif
#endif
    const uint64 lane = read_lane_id_4b_b();
    const int5 off = get_index_space_offset();
    const int5 size = get_index_space_size();
    const uint64 nan_bits = 0x7FC00000u;
    const float64 nanv = as_float64(nan_bits);

    for (int bt = off[0]; bt < off[0] + size[0]; bt++)
    {
        float64 bad = 0.0f;
        for (int j = 0; j < n; j++)
        {
            int5 cr = {0, j, bt, 0, 0};
            int5 ci = {0, n + j, bt, 0, 0};
            float64 xr = v_f32_sel_geq_u32_b(lane, (unsigned)m, 0.0f, v_f32_ld_tnsr_b(cr, A_t));
            float64 xi = v_f32_sel_geq_u32_b(lane, (unsigned)m, 0.0f, v_f32_ld_tnsr_b(ci, A_t));
            bad = nonfinite(xi, nonfinite(xr, bad));
            qr[j] = v_f32_sel_eq_u32_b(v_u32_and_b(as_uint64(xr), 0x7F800000u), 0u, 0.0f, xr);
            qi[j] = v_f32_sel_eq_u32_b(v_u32_and_b(as_uint64(xi), 0x7F800000u), 0u, 0.0f, xi);
        }
        bad = lane_bcast(reduce_max_half32(bad), 0);
        for (int j = 0; j < n; j++)
        {
            qr[j] = v_f32_sel_grt_f32_b(bad, 0.0f, 0.0f, qr[j]);
            qi[j] = v_f32_sel_grt_f32_b(bad, 0.0f, 0.0f, qi[j]);
        }

#ifndef R_ONLY
        float64 tau_r_vec = 0.0f;
        float64 tau_i_vec = 0.0f;
#endif

        // Forward complex Householder reduction (LAPACK clarfg convention)
        for (int i = 0; i < n; i++)
        {
            float64 qir = qr[i];
            float64 qii = qi[i];
            float64 xr = v_f32_sel_less_u32_b(lane, (unsigned)i, 0.0f, v_f32_sel_geq_u32_b(lane, (unsigned)m, 0.0f, qir));
            float64 xi = v_f32_sel_less_u32_b(lane, (unsigned)i, 0.0f, v_f32_sel_geq_u32_b(lane, (unsigned)m, 0.0f, qii));
            float64 mx = reduce_max_half32(v_f32_max_b(v_f32_abs_b(xr), v_f32_abs_b(xi)));
            float64 sc = pow2_scale(mx);
            float64 unsc = pow2_unscale(mx);
            float64 xsr = v_f32_mul_b(xr, sc);
            float64 xsi = v_f32_mul_b(xi, sc);

            float64 tail_r = v_f32_sel_grt_u32_b(lane, (unsigned)i, xsr, 0.0f);
            float64 tail_i = v_f32_sel_grt_u32_b(lane, (unsigned)i, xsi, 0.0f);
            float64 sigma_rest = reduce_add_half32(v_f32_mac_b(tail_i, tail_i, v_f32_mul_b(tail_r, tail_r)));

            float64 xsr0 = lane_bcast(xsr, i);
            float64 xsi0 = lane_bcast(xsi, i);
            float64 orig_r = lane_bcast(qir, i);

            float64 no_reflect = v_f32_sel_eq_f32_b(
                mx, 0.0f, 1.0f,
                v_f32_sel_eq_f32_b(sigma_rest, 0.0f, v_f32_sel_eq_f32_b(xsi0, 0.0f, 1.0f, 0.0f), 0.0f));

            float64 sq = v_f32_mac_b(xsi0, xsi0, v_f32_mac_b(xsr0, xsr0, sigma_rest));
            float64 safe_sq = v_f32_sel_eq_f32_b(no_reflect, 1.0f, 1.0f, sq);
            float64 inv_nrm = rsqrt_fine(safe_sq);
            float64 nrm_xs = v_f32_mul_b(safe_sq, inv_nrm);
            float64 beta = v_f32_sel_less_f32_b(xsr0, 0.0f, nrm_xs, v_f32_sub_b(0.0f, nrm_xs));
            float64 safe_beta = v_f32_sel_eq_f32_b(no_reflect, 1.0f, 1.0f, beta);
            float64 inv_beta = lut_free_recip(safe_beta);

            float64 tau_r_i = v_f32_sel_eq_f32_b(no_reflect, 1.0f, 0.0f, v_f32_sub_b(1.0f, v_f32_mul_b(xsr0, inv_beta)));
            float64 tau_i_i = v_f32_sel_eq_f32_b(no_reflect, 1.0f, 0.0f, v_f32_sub_b(0.0f, v_f32_mul_b(xsi0, inv_beta)));
            float64 r_ii_r = v_f32_sel_eq_f32_b(no_reflect, 1.0f, orig_r, v_f32_mul_b(beta, unsc));

            float64 u0r = v_f32_sub_b(xsr0, beta);
            float64 u0i = xsi0;
            float64 u0sq = v_f32_mac_b(u0i, u0i, v_f32_mul_b(u0r, u0r));
            float64 safe_u0sq = v_f32_sel_eq_f32_b(no_reflect, 1.0f, 1.0f, u0sq);
            float64 inv_u0sq = lut_free_recip(safe_u0sq);
            float64 inv_u0_r = v_f32_mul_b(u0r, inv_u0sq);
            float64 inv_u0_i = v_f32_mul_b(v_f32_sub_b(0.0f, u0i), inv_u0sq);

            float64 vr = v_f32_mac_b(v_f32_sub_b(0.0f, xsi), inv_u0_i, v_f32_mul_b(xsr, inv_u0_r));
            float64 vi = v_f32_mac_b(xsi, inv_u0_r, v_f32_mul_b(xsr, inv_u0_i));
            vr = v_f32_sel_less_u32_b(lane, (unsigned)i, 0.0f, vr);
            vi = v_f32_sel_less_u32_b(lane, (unsigned)i, 0.0f, vi);
            vr = v_f32_sel_eq_u32_b(lane, (unsigned)i, 1.0f, vr);
            vi = v_f32_sel_eq_u32_b(lane, (unsigned)i, 0.0f, vi);
            vr = v_f32_sel_eq_f32_b(no_reflect, 1.0f, 0.0f, vr);
            vi = v_f32_sel_eq_f32_b(no_reflect, 1.0f, 0.0f, vi);

#ifdef R_ONLY
            qr[i] = v_f32_sel_eq_u32_b(lane, (unsigned)i, r_ii_r, qir);
            qi[i] = v_f32_sel_eq_u32_b(lane, (unsigned)i, 0.0f, qii);
#else
            float64 new_qir = v_f32_sel_grt_u32_b(lane, (unsigned)i, vr, qir);
            float64 new_qii = v_f32_sel_grt_u32_b(lane, (unsigned)i, vi, qii);
            qr[i] = v_f32_sel_eq_u32_b(lane, (unsigned)i, r_ii_r, new_qir);
            qi[i] = v_f32_sel_eq_u32_b(lane, (unsigned)i, 0.0f, new_qii);
            tau_r_vec = v_f32_sel_eq_u32_b(lane, (unsigned)i, tau_r_i, tau_r_vec);
            tau_i_vec = v_f32_sel_eq_u32_b(lane, (unsigned)i, tau_i_i, tau_i_vec);
#endif

            // Apply H_i^H = I - conj(tau_i) v_i v_i^H to trailing columns j > i
            for (int j = i + 1; j < n; j++)
            {
                float64 qjr = qr[j];
                float64 qji = qi[j];
                float64 act_r = v_f32_sel_less_u32_b(lane, (unsigned)i, 0.0f, v_f32_sel_geq_u32_b(lane, (unsigned)m, 0.0f, qjr));
                float64 act_i = v_f32_sel_less_u32_b(lane, (unsigned)i, 0.0f, v_f32_sel_geq_u32_b(lane, (unsigned)m, 0.0f, qji));
                float64 mx_j = reduce_max_half32(v_f32_max_b(v_f32_abs_b(act_r), v_f32_abs_b(act_i)));
                float64 sc_j = pow2_scale(mx_j);
                float64 unsc_j = pow2_unscale(mx_j);
                float64 qjr_sc = v_f32_mul_b(act_r, sc_j);
                float64 qji_sc = v_f32_mul_b(act_i, sc_j);

                float64 dr = reduce_add_half32(v_f32_mac_b(vi, qji_sc, v_f32_mul_b(vr, qjr_sc)));
                float64 di = reduce_add_half32(v_f32_mac_b(v_f32_sub_b(0.0f, vi), qjr_sc, v_f32_mul_b(vr, qji_sc)));

                float64 cr_c = v_f32_mac_b(tau_i_i, di, v_f32_mul_b(tau_r_i, dr));
                float64 ci_c = v_f32_mac_b(v_f32_sub_b(0.0f, tau_i_i), dr, v_f32_mul_b(tau_r_i, di));

                float64 upd_r = v_f32_mul_b(
                    v_f32_mac_b(ci_c, vi, v_f32_mac_b(v_f32_sub_b(0.0f, cr_c), vr, qjr_sc)), unsc_j);
                float64 upd_i = v_f32_mul_b(
                    v_f32_mac_b(v_f32_sub_b(0.0f, ci_c), vr, v_f32_mac_b(v_f32_sub_b(0.0f, cr_c), vi, qji_sc)),
                    unsc_j);

                float64 keep = v_f32_sel_less_u32_b(lane, (unsigned)i, 1.0f,
                                                    v_f32_sel_geq_u32_b(lane, (unsigned)m, 1.0f, no_reflect));
                qr[j] = v_f32_sel_eq_f32_b(keep, 1.0f, qjr, upd_r);
                qi[j] = v_f32_sel_eq_f32_b(keep, 1.0f, qji, upd_i);
            }
        }

        // Store R_t [k_r, 2n, batch] before overwriting qr/qi with Q
        for (int j = 0; j < n; j++)
        {
            float64 rr = v_f32_sel_grt_u32_b(lane, (unsigned)j, 0.0f, qr[j]);
            float64 ri = v_f32_sel_grt_u32_b(lane, (unsigned)j, 0.0f, qi[j]);
            rr = v_f32_sel_grt_f32_b(bad, 0.0f, nanv, rr);
            ri = v_f32_sel_grt_f32_b(bad, 0.0f, nanv, ri);
            int5 cr = {0, j, bt, 0, 0};
            int5 ci = {0, n + j, bt, 0, 0};
            v_f32_st_tnsr_partial(cr, R_t, rr, (char)(k_r - 1), 0);
            v_f32_st_tnsr_partial(ci, R_t, ri, (char)(k_r - 1), 0);
        }

#ifndef R_ONLY
#ifdef FULL_Q
        for (int j = n; j < m; j++)
        {
            qr[j] = v_f32_sel_eq_u32_b(lane, (unsigned)j, 1.0f, 0.0f);
            qi[j] = 0.0f;
        }
#endif
        // Backward Q accumulation: Q[:, :k_q] = H_0 H_1 ... H_{n-1} I[:, :k_q]
        for (int i = n - 1; i >= 0; i--)
        {
            float64 tau_r_i = lane_bcast(tau_r_vec, i);
            float64 tau_i_i = lane_bcast(tau_i_vec, i);
            float64 zero_tau = v_f32_sel_eq_f32_b(tau_r_i, 0.0f, 1.0f, 0.0f);

            float64 vr = v_f32_sel_less_u32_b(lane, (unsigned)i, 0.0f,
                                              v_f32_sel_geq_u32_b(lane, (unsigned)m, 0.0f, qr[i]));
            float64 vi = v_f32_sel_less_u32_b(lane, (unsigned)i, 0.0f,
                                              v_f32_sel_geq_u32_b(lane, (unsigned)m, 0.0f, qi[i]));
            vr = v_f32_sel_eq_u32_b(lane, (unsigned)i, 1.0f, vr);
            vi = v_f32_sel_eq_u32_b(lane, (unsigned)i, 0.0f, vi);

            float64 ntr = v_f32_sub_b(0.0f, tau_r_i);
            float64 nti = v_f32_sub_b(0.0f, tau_i_i);
            float64 ntv_r = v_f32_mac_b(v_f32_sub_b(0.0f, nti), vi, v_f32_mul_b(ntr, vr));
            float64 ntv_i = v_f32_mac_b(nti, vr, v_f32_mul_b(ntr, vi));

            for (int j = i + 1; j < k_q; j++)
            {
                float64 qjr = qr[j];
                float64 qji = qi[j];
                float64 dr = reduce_add_half32(v_f32_mac_b(vi, qji, v_f32_mul_b(vr, qjr)));
                float64 di = reduce_add_half32(v_f32_mac_b(v_f32_sub_b(0.0f, vi), qjr, v_f32_mul_b(vr, qji)));
                float64 upd_r = v_f32_mac_b(v_f32_sub_b(0.0f, ntv_i), di, v_f32_mac_b(ntv_r, dr, qjr));
                float64 upd_i = v_f32_mac_b(ntv_i, dr, v_f32_mac_b(ntv_r, di, qji));
                qr[j] = v_f32_sel_eq_f32_b(zero_tau, 1.0f, qjr, upd_r);
                qi[j] = v_f32_sel_eq_f32_b(zero_tau, 1.0f, qji, upd_i);
            }

            float64 eir = v_f32_sel_eq_u32_b(lane, (unsigned)i, 1.0f, 0.0f);
            qr[i] = v_f32_sel_eq_f32_b(zero_tau, 1.0f, eir, v_f32_add_b(eir, ntv_r));
            qi[i] = v_f32_sel_eq_f32_b(zero_tau, 1.0f, 0.0f, ntv_i);
        }

        for (int j = 0; j < k_q; j++)
        {
            float64 qjr = v_f32_sel_grt_f32_b(bad, 0.0f, nanv, qr[j]);
            float64 qji = v_f32_sel_grt_f32_b(bad, 0.0f, nanv, qi[j]);
            int5 cqr = {0, j, bt, 0, 0};
            int5 cqi = {0, k_q + j, bt, 0, 0};
            v_f32_st_tnsr_partial(cqr, Q_t, qjr, (char)(m - 1), 0);
            v_f32_st_tnsr_partial(cqi, Q_t, qji, (char)(m - 1), 0);
        }
#endif
    }
}
