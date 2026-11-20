/* Batched symmetric eigendecomposition on Gaudi2 TPC: one-sided (Hestenes) Jacobi, 2-matrix SIMD half-vector packing.
 *
 * Tensors (TPC geometry, dim0 innermost; torch [batch, rows, cols] row-major):
 *   A [n, n, batch]  f32 (symmetric; the lower triangle is mirrored by the caller)
 *   W [n, batch]     f32: eigenvalues, unsorted          V [n, n, batch]  f32: row j = eigenvector of W[j]
 *
 * Pass 1 (jacobi_eigh_f32_gaudi2): prescales A by max_abs, normalizes by tight prescaled Gershgorin sigma_p,
 * shifts S = A/sigma_p + 1.001 I so S is strictly SPD with O(1) entries, and runs 4/8/9/10 sweeps of one-sided
 * Jacobi with K=3 interleaved round-robin pairs (+ K=2, K=1 tails) in 8 KB VLM (u[MAXN]).
 *
 * Pass 2 (-DREFINE_PASS, jacobi_eigh_refine_f32_gaudi2): takes A and Pass-1 eigenvector rows V1 as read-only inputs,
 * forms T = V1^T (A * prescale) V1 in u[0..n-1], computes T's Gershgorin lower bound g_min / sigma_T, applies an
 * adaptive shift max(0, 0.01 - g_norm), runs 2/4 refinement sweeps to get U2, composes
 * V = V1 @ U2, renormalizes, and computes Rayleigh quotient W[j] = v_j^T (A * prescale) v_j * inv_prescale.
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

static inline float64 reduce_max_half32(float64 p)
{
    p = v_f32_max_b(p, v_f32_mov_dual_group_all_b(p, 0xFFFFFFFF, 1, 0, 3, 2, SW_MDG_ALL, p));
    p = v_f32_max_b(p, as_float64(v_element_shift_xor_i32(as_int64(p), 8)));
    p = v_f32_max_b(p, as_float64(v_element_shift_xor_i32(as_int64(p), 4)));
    p = v_f32_max_b(p, as_float64(v_element_shift_xor_i32(as_int64(p), 2)));
    p = v_f32_max_b(p, as_float64(v_element_shift_xor_i32(as_int64(p), 1)));
    return p;
}

static inline float64 inv_sqrt_unit(float64 c2)
{
    // Degree-4 relative-error minimax polynomial on [0.5, 1.0] + 1 NR step (8.2e-9 theoretical, <= 1.52e-7 / <= 1.3 ULP in IEEE float32)
    float64 p = 1.1498469f;
    p = v_f32_mac_b(c2, p, -4.40869479f);
    p = v_f32_mac_b(c2, p, 6.8050256f);
    p = v_f32_mac_b(c2, p, -5.47635557f);
    p = v_f32_mac_b(c2, p, 2.9302518f);
    float64 half_c2 = v_f32_mul_b(-0.5f, c2);
    float64 p2 = v_f32_mul_b(p, p);
    float64 nr = v_f32_mac_b(half_c2, p2, 1.5f);
    return v_f32_mul_b(p, nr);
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

static inline void run_jacobi_sweeps(int m, int half_n, int k_start, int sweeps)
{
    for (int sweep = 0; sweep < sweeps; sweep++)
    {
        for (int r = 0; r < m; r++)
        {
            int k = k_start;
            for (; k + 2 < half_n; k += 3)
            {
                int p0, q0, p1, q1, p2, q2;
                RR_PAIR(r, k, m, p0, q0);
                RR_PAIR(r, k + 1, m, p1, q1);
                RR_PAIR(r, k + 2, m, p2, q2);

                float64 up0 = u[p0], uq0 = u[q0];
                float64 up1 = u[p1], uq1 = u[q1];
                float64 up2 = u[p2], uq2 = u[q2];

                float64 alpha0 = reduce_add_half32(v_f32_mul_b(up0, up0));
                float64 beta0 = reduce_add_half32(v_f32_mul_b(uq0, uq0));
                float64 gamma0 = reduce_add_half32(v_f32_mul_b(up0, uq0));
                float64 alpha1 = reduce_add_half32(v_f32_mul_b(up1, up1));
                float64 beta1 = reduce_add_half32(v_f32_mul_b(uq1, uq1));
                float64 gamma1 = reduce_add_half32(v_f32_mul_b(up1, uq1));
                float64 alpha2 = reduce_add_half32(v_f32_mul_b(up2, up2));
                float64 beta2 = reduce_add_half32(v_f32_mul_b(uq2, uq2));
                float64 gamma2 = reduce_add_half32(v_f32_mul_b(up2, uq2));

                float64 d0 = v_f32_sub_b(beta0, alpha0);
                float64 d1 = v_f32_sub_b(beta1, alpha1);
                float64 d2 = v_f32_sub_b(beta2, alpha2);
                float64 g20 = v_f32_add_b(gamma0, gamma0);
                float64 g21 = v_f32_add_b(gamma1, gamma1);
                float64 g22 = v_f32_add_b(gamma2, gamma2);
                float64 r20 = v_f32_mac_b(g20, g20, v_f32_mul_b(d0, d0));
                float64 r21 = v_f32_mac_b(g21, g21, v_f32_mul_b(d1, d1));
                float64 r22 = v_f32_mac_b(g22, g22, v_f32_mul_b(d2, d2));
                float64 sr20 = v_f32_sel_eq_f32_b(r20, 0.0f, 1.0f, r20);
                float64 sr21 = v_f32_sel_eq_f32_b(r21, 0.0f, 1.0f, r21);
                float64 sr22 = v_f32_sel_eq_f32_b(r22, 0.0f, 1.0f, r22);

                float64 inv_r0 = v_rsqrt_f32(sr20);
                float64 inv_r1 = v_rsqrt_f32(sr21);
                float64 inv_r2 = v_rsqrt_f32(sr22);

                float64 c20 = v_f32_mul_b(0.5f, v_f32_mac_b(v_f32_abs_b(d0), inv_r0, 1.0f));
                float64 c21 = v_f32_mul_b(0.5f, v_f32_mac_b(v_f32_abs_b(d1), inv_r1, 1.0f));
                float64 c22 = v_f32_mul_b(0.5f, v_f32_mac_b(v_f32_abs_b(d2), inv_r2, 1.0f));

                float64 inv_c0 = inv_sqrt_unit(c20);
                float64 inv_c1 = inv_sqrt_unit(c21);
                float64 inv_c2 = inv_sqrt_unit(c22);

                float64 c0 = v_f32_mul_b(c20, inv_c0);
                float64 c1 = v_f32_mul_b(c21, inv_c1);
                float64 c2 = v_f32_mul_b(c22, inv_c2);
                float64 s0 = v_f32_mul_b(gamma0, v_f32_mul_b(inv_r0, inv_c0));
                float64 s1 = v_f32_mul_b(gamma1, v_f32_mul_b(inv_r1, inv_c1));
                float64 s2 = v_f32_mul_b(gamma2, v_f32_mul_b(inv_r2, inv_c2));
                s0 = v_f32_sel_less_f32_b(d0, 0.0f, v_f32_sub_b(0.0f, s0), s0);
                s1 = v_f32_sel_less_f32_b(d1, 0.0f, v_f32_sub_b(0.0f, s1), s1);
                s2 = v_f32_sel_less_f32_b(d2, 0.0f, v_f32_sub_b(0.0f, s2), s2);

                c0 = v_f32_sel_eq_f32_b(gamma0, 0.0f, 1.0f, c0);
                s0 = v_f32_sel_eq_f32_b(gamma0, 0.0f, 0.0f, s0);
                c0 = v_f32_sel_eq_f32_b(r20, 0.0f, 1.0f, c0);
                s0 = v_f32_sel_eq_f32_b(r20, 0.0f, 0.0f, s0);

                c1 = v_f32_sel_eq_f32_b(gamma1, 0.0f, 1.0f, c1);
                s1 = v_f32_sel_eq_f32_b(gamma1, 0.0f, 0.0f, s1);
                c1 = v_f32_sel_eq_f32_b(r21, 0.0f, 1.0f, c1);
                s1 = v_f32_sel_eq_f32_b(r21, 0.0f, 0.0f, s1);

                c2 = v_f32_sel_eq_f32_b(gamma2, 0.0f, 1.0f, c2);
                s2 = v_f32_sel_eq_f32_b(gamma2, 0.0f, 0.0f, s2);
                c2 = v_f32_sel_eq_f32_b(r22, 0.0f, 1.0f, c2);
                s2 = v_f32_sel_eq_f32_b(r22, 0.0f, 0.0f, s2);

                float64 ns0 = v_f32_sub_b(0.0f, s0);
                float64 ns1 = v_f32_sub_b(0.0f, s1);
                float64 ns2 = v_f32_sub_b(0.0f, s2);
                u[p0] = v_f32_mac_b(uq0, ns0, v_f32_mul_b(up0, c0));
                u[q0] = v_f32_mac_b(up0, s0, v_f32_mul_b(uq0, c0));
                u[p1] = v_f32_mac_b(uq1, ns1, v_f32_mul_b(up1, c1));
                u[q1] = v_f32_mac_b(up1, s1, v_f32_mul_b(uq1, c1));
                u[p2] = v_f32_mac_b(uq2, ns2, v_f32_mul_b(up2, c2));
                u[q2] = v_f32_mac_b(up2, s2, v_f32_mul_b(uq2, c2));
            }
            if (k + 1 < half_n)
            {
                int p0, q0, p1, q1;
                RR_PAIR(r, k, m, p0, q0);
                RR_PAIR(r, k + 1, m, p1, q1);
                k += 2;

                float64 up0 = u[p0], uq0 = u[q0];
                float64 up1 = u[p1], uq1 = u[q1];

                float64 alpha0 = reduce_add_half32(v_f32_mul_b(up0, up0));
                float64 beta0 = reduce_add_half32(v_f32_mul_b(uq0, uq0));
                float64 gamma0 = reduce_add_half32(v_f32_mul_b(up0, uq0));
                float64 alpha1 = reduce_add_half32(v_f32_mul_b(up1, up1));
                float64 beta1 = reduce_add_half32(v_f32_mul_b(uq1, uq1));
                float64 gamma1 = reduce_add_half32(v_f32_mul_b(up1, uq1));

                float64 d0 = v_f32_sub_b(beta0, alpha0);
                float64 d1 = v_f32_sub_b(beta1, alpha1);
                float64 g20 = v_f32_add_b(gamma0, gamma0);
                float64 g21 = v_f32_add_b(gamma1, gamma1);
                float64 r20 = v_f32_mac_b(g20, g20, v_f32_mul_b(d0, d0));
                float64 r21 = v_f32_mac_b(g21, g21, v_f32_mul_b(d1, d1));
                float64 sr20 = v_f32_sel_eq_f32_b(r20, 0.0f, 1.0f, r20);
                float64 sr21 = v_f32_sel_eq_f32_b(r21, 0.0f, 1.0f, r21);

                float64 inv_r0 = v_rsqrt_f32(sr20);
                float64 inv_r1 = v_rsqrt_f32(sr21);

                float64 c20 = v_f32_mul_b(0.5f, v_f32_mac_b(v_f32_abs_b(d0), inv_r0, 1.0f));
                float64 c21 = v_f32_mul_b(0.5f, v_f32_mac_b(v_f32_abs_b(d1), inv_r1, 1.0f));

                float64 inv_c0 = inv_sqrt_unit(c20);
                float64 inv_c1 = inv_sqrt_unit(c21);

                float64 c0 = v_f32_mul_b(c20, inv_c0);
                float64 c1 = v_f32_mul_b(c21, inv_c1);
                float64 s0 = v_f32_mul_b(gamma0, v_f32_mul_b(inv_r0, inv_c0));
                float64 s1 = v_f32_mul_b(gamma1, v_f32_mul_b(inv_r1, inv_c1));
                s0 = v_f32_sel_less_f32_b(d0, 0.0f, v_f32_sub_b(0.0f, s0), s0);
                s1 = v_f32_sel_less_f32_b(d1, 0.0f, v_f32_sub_b(0.0f, s1), s1);

                c0 = v_f32_sel_eq_f32_b(gamma0, 0.0f, 1.0f, c0);
                s0 = v_f32_sel_eq_f32_b(gamma0, 0.0f, 0.0f, s0);
                c0 = v_f32_sel_eq_f32_b(r20, 0.0f, 1.0f, c0);
                s0 = v_f32_sel_eq_f32_b(r20, 0.0f, 0.0f, s0);

                c1 = v_f32_sel_eq_f32_b(gamma1, 0.0f, 1.0f, c1);
                s1 = v_f32_sel_eq_f32_b(gamma1, 0.0f, 0.0f, s1);
                c1 = v_f32_sel_eq_f32_b(r21, 0.0f, 1.0f, c1);
                s1 = v_f32_sel_eq_f32_b(r21, 0.0f, 0.0f, s1);

                float64 ns0 = v_f32_sub_b(0.0f, s0);
                float64 ns1 = v_f32_sub_b(0.0f, s1);
                u[p0] = v_f32_mac_b(uq0, ns0, v_f32_mul_b(up0, c0));
                u[q0] = v_f32_mac_b(up0, s0, v_f32_mul_b(uq0, c0));
                u[p1] = v_f32_mac_b(uq1, ns1, v_f32_mul_b(up1, c1));
                u[q1] = v_f32_mac_b(up1, s1, v_f32_mul_b(uq1, c1));
            }
            if (k < half_n)
            {
                int p, q;
                RR_PAIR(r, k, m, p, q);
                float64 up = u[p], uq = u[q];
                float64 alpha = reduce_add_half32(v_f32_mul_b(up, up));
                float64 beta = reduce_add_half32(v_f32_mul_b(uq, uq));
                float64 gamma = reduce_add_half32(v_f32_mul_b(up, uq));

                float64 d = v_f32_sub_b(beta, alpha);
                float64 g2 = v_f32_add_b(gamma, gamma);
                float64 r2 = v_f32_mac_b(g2, g2, v_f32_mul_b(d, d));
                float64 sr2 = v_f32_sel_eq_f32_b(r2, 0.0f, 1.0f, r2);
                float64 inv_r = v_rsqrt_f32(sr2);
                float64 c2 = v_f32_mul_b(0.5f, v_f32_mac_b(v_f32_abs_b(d), inv_r, 1.0f));
                float64 inv_c = inv_sqrt_unit(c2);
                float64 c = v_f32_mul_b(c2, inv_c);
                float64 s = v_f32_mul_b(gamma, v_f32_mul_b(inv_r, inv_c));
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
}

static inline void normalize_u_rows(int n)
{
    int j = 0;
    for (; j + 1 < n; j += 2)
    {
        float64 u0 = u[j], u1 = u[j + 1];
        float64 sq0 = reduce_add_half32(v_f32_mul_b(u0, u0));
        float64 sq1 = reduce_add_half32(v_f32_mul_b(u1, u1));
        float64 rn0 = v_rsqrt_f32(sq0);
        float64 rn1 = v_rsqrt_f32(sq1);
        rn0 = v_f32_mul_b(rn0, v_f32_mac_b(v_f32_mul_b(-0.5f, sq0), v_f32_mul_b(rn0, rn0), 1.5f));
        rn1 = v_f32_mul_b(rn1, v_f32_mac_b(v_f32_mul_b(-0.5f, sq1), v_f32_mul_b(rn1, rn1), 1.5f));
        u[j] = v_f32_mul_b(u0, rn0);
        u[j + 1] = v_f32_mul_b(u1, rn1);
    }
    if (j < n)
    {
        float64 u0 = u[j];
        float64 sq0 = reduce_add_half32(v_f32_mul_b(u0, u0));
        float64 rn0 = v_rsqrt_f32(sq0);
        rn0 = v_f32_mul_b(rn0, v_f32_mac_b(v_f32_mul_b(-0.5f, sq0), v_f32_mul_b(rn0, rn0), 1.5f));
        u[j] = v_f32_mul_b(u0, rn0);
    }
}

#ifdef REFINE_PASS
void main(tensor A, tensor V1, tensor W, tensor V)
#else
void main(tensor A, tensor W, tensor V)
#endif
{
    const int n = get_dim_size(A, 0);
    const uint64 lane = read_lane_id_4b_b();
    const uint64 lane_mod32 = lane & 31u;
    const int5 off = get_index_space_offset();
    const int5 size = get_index_space_size();
    const int n_even = (n + 1) & ~1;
    const int m = n_even - 1;
    const int half_n = n_even >> 1;
    const int k_start = n & 1;  // when n is odd, k==0 has q == m == n (padding) and is skipped
#ifdef REFINE_PASS
    const int sweeps = (n <= 4) ? 2 : 4;
#else
    const int sweeps = (n <= 4) ? 4 : (n <= 16) ? 8 : (n <= 22) ? 9 : 10;
    const float shift_norm = 1.001f;
#endif

    // Unified 2-matrix SIMD half-vector packed path: bt in lanes 0..31, bt+1 (if has2) in lanes 32..63
    for (int bt = off[0]; bt < off[0] + size[0]; bt += 2)
    {
        const int has2 = (bt + 1 < off[0] + size[0]);
        float64 max_abs = 0.0f;
        for (int i = 0; i < n; i++)
        {
            int5 c0 = {0, i, bt, 0, 0};
            float64 r0 = v_f32_sel_geq_u32_b(lane, (unsigned)n, 0.0f, v_f32_ld_tnsr_b(c0, A));
            float64 r1 = 0.0f;
            if (has2)
            {
                int5 c1 = {0, i, bt + 1, 0, 0};
                r1 = v_f32_sel_geq_u32_b(lane, (unsigned)n, 0.0f, v_f32_ld_tnsr_b(c1, A));
            }
            float64 packed = pack_halves(r0, r1);
#ifndef REFINE_PASS
            u[i] = packed;
#endif
            max_abs = v_f32_max_b(max_abs, v_f32_abs_b(packed));
        }
        max_abs = reduce_max_half32(max_abs);
        // Per-half branchless overflow/subnormal prescaling from max_abs
        float64 prescale = v_f32_sel_grt_f32_b(max_abs, 1e33f, 0x1p-20f, 1.0f);
        float64 inv_prescale = v_f32_sel_grt_f32_b(max_abs, 1e33f, 0x1p+20f, 1.0f);
        float64 sub_prescale = v_f32_sel_eq_f32_b(max_abs, 0.0f, prescale, 0x1p+40f);
        float64 sub_inv_prescale = v_f32_sel_eq_f32_b(max_abs, 0.0f, inv_prescale, 0x1p-40f);
        prescale = v_f32_sel_less_f32_b(max_abs, 1e-20f, sub_prescale, prescale);
        inv_prescale = v_f32_sel_less_f32_b(max_abs, 1e-20f, sub_inv_prescale, inv_prescale);

#ifdef REFINE_PASS
        // Form T = V1^T (A * prescale) V1 directly into u[0..n-1] in 8-column VRF blocks
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
            float64 v0, v1, v2, v3, v4, v5, v6, v7;
            {
                int5 p0a = {0, c0, bt, 0, 0}, p1a = {0, c1, bt, 0, 0}, p2a = {0, c2, bt, 0, 0}, p3a = {0, c3, bt, 0, 0};
                int5 p4a = {0, c4, bt, 0, 0}, p5a = {0, c5, bt, 0, 0}, p6a = {0, c6, bt, 0, 0}, p7a = {0, c7, bt, 0, 0};
                float64 l0 = v_f32_sel_geq_u32_b(lane, (unsigned)n, 0.0f, v_f32_ld_tnsr_b(p0a, V1));
                float64 l1 = v_f32_sel_geq_u32_b(lane, (unsigned)n, 0.0f, v_f32_ld_tnsr_b(p1a, V1));
                float64 l2 = v_f32_sel_geq_u32_b(lane, (unsigned)n, 0.0f, v_f32_ld_tnsr_b(p2a, V1));
                float64 l3 = v_f32_sel_geq_u32_b(lane, (unsigned)n, 0.0f, v_f32_ld_tnsr_b(p3a, V1));
                float64 l4 = v_f32_sel_geq_u32_b(lane, (unsigned)n, 0.0f, v_f32_ld_tnsr_b(p4a, V1));
                float64 l5 = v_f32_sel_geq_u32_b(lane, (unsigned)n, 0.0f, v_f32_ld_tnsr_b(p5a, V1));
                float64 l6 = v_f32_sel_geq_u32_b(lane, (unsigned)n, 0.0f, v_f32_ld_tnsr_b(p6a, V1));
                float64 l7 = v_f32_sel_geq_u32_b(lane, (unsigned)n, 0.0f, v_f32_ld_tnsr_b(p7a, V1));
                float64 h0 = 0.0f, h1 = 0.0f, h2 = 0.0f, h3 = 0.0f, h4 = 0.0f, h5 = 0.0f, h6 = 0.0f, h7 = 0.0f;
                if (has2)
                {
                    int5 p0b = {0, c0, bt + 1, 0, 0}, p1b = {0, c1, bt + 1, 0, 0}, p2b = {0, c2, bt + 1, 0, 0}, p3b = {0, c3, bt + 1, 0, 0};
                    int5 p4b = {0, c4, bt + 1, 0, 0}, p5b = {0, c5, bt + 1, 0, 0}, p6b = {0, c6, bt + 1, 0, 0}, p7b = {0, c7, bt + 1, 0, 0};
                    h0 = v_f32_sel_geq_u32_b(lane, (unsigned)n, 0.0f, v_f32_ld_tnsr_b(p0b, V1));
                    h1 = v_f32_sel_geq_u32_b(lane, (unsigned)n, 0.0f, v_f32_ld_tnsr_b(p1b, V1));
                    h2 = v_f32_sel_geq_u32_b(lane, (unsigned)n, 0.0f, v_f32_ld_tnsr_b(p2b, V1));
                    h3 = v_f32_sel_geq_u32_b(lane, (unsigned)n, 0.0f, v_f32_ld_tnsr_b(p3b, V1));
                    h4 = v_f32_sel_geq_u32_b(lane, (unsigned)n, 0.0f, v_f32_ld_tnsr_b(p4b, V1));
                    h5 = v_f32_sel_geq_u32_b(lane, (unsigned)n, 0.0f, v_f32_ld_tnsr_b(p5b, V1));
                    h6 = v_f32_sel_geq_u32_b(lane, (unsigned)n, 0.0f, v_f32_ld_tnsr_b(p6b, V1));
                    h7 = v_f32_sel_geq_u32_b(lane, (unsigned)n, 0.0f, v_f32_ld_tnsr_b(p7b, V1));
                }
                v0 = pack_halves(l0, h0); v1 = pack_halves(l1, h1);
                v2 = pack_halves(l2, h2); v3 = pack_halves(l3, h3);
                v4 = pack_halves(l4, h4); v5 = pack_halves(l5, h5);
                v6 = pack_halves(l6, h6); v7 = pack_halves(l7, h7);
            }
            float64 av0 = 0.0f, av1 = 0.0f, av2 = 0.0f, av3 = 0.0f;
            float64 av4 = 0.0f, av5 = 0.0f, av6 = 0.0f, av7 = 0.0f;
            for (int i = 0; i < n; i++)
            {
                int5 ca = {0, i, bt, 0, 0};
                float64 r0 = v_f32_sel_geq_u32_b(lane, (unsigned)n, 0.0f, v_f32_ld_tnsr_b(ca, A));
                float64 r1 = 0.0f;
                if (has2)
                {
                    int5 cb = {0, i, bt + 1, 0, 0};
                    r1 = v_f32_sel_geq_u32_b(lane, (unsigned)n, 0.0f, v_f32_ld_tnsr_b(cb, A));
                }
                float64 arow = v_f32_mul_b(pack_halves(r0, r1), prescale);
                float64 b0 = v_f32_sel_less_u32_b(lane, 32u,
                                                  as_float64(v_broadcast_element_i32(as_int64(v0), i)),
                                                  as_float64(v_broadcast_element_i32(as_int64(v0), i + 32)));
                float64 b1 = v_f32_sel_less_u32_b(lane, 32u,
                                                  as_float64(v_broadcast_element_i32(as_int64(v1), i)),
                                                  as_float64(v_broadcast_element_i32(as_int64(v1), i + 32)));
                float64 b2 = v_f32_sel_less_u32_b(lane, 32u,
                                                  as_float64(v_broadcast_element_i32(as_int64(v2), i)),
                                                  as_float64(v_broadcast_element_i32(as_int64(v2), i + 32)));
                float64 b3 = v_f32_sel_less_u32_b(lane, 32u,
                                                  as_float64(v_broadcast_element_i32(as_int64(v3), i)),
                                                  as_float64(v_broadcast_element_i32(as_int64(v3), i + 32)));
                float64 b4 = v_f32_sel_less_u32_b(lane, 32u,
                                                  as_float64(v_broadcast_element_i32(as_int64(v4), i)),
                                                  as_float64(v_broadcast_element_i32(as_int64(v4), i + 32)));
                float64 b5 = v_f32_sel_less_u32_b(lane, 32u,
                                                  as_float64(v_broadcast_element_i32(as_int64(v5), i)),
                                                  as_float64(v_broadcast_element_i32(as_int64(v5), i + 32)));
                float64 b6 = v_f32_sel_less_u32_b(lane, 32u,
                                                  as_float64(v_broadcast_element_i32(as_int64(v6), i)),
                                                  as_float64(v_broadcast_element_i32(as_int64(v6), i + 32)));
                float64 b7 = v_f32_sel_less_u32_b(lane, 32u,
                                                  as_float64(v_broadcast_element_i32(as_int64(v7), i)),
                                                  as_float64(v_broadcast_element_i32(as_int64(v7), i + 32)));
                av0 = v_f32_mac_b(arow, b0, av0);
                av1 = v_f32_mac_b(arow, b1, av1);
                av2 = v_f32_mac_b(arow, b2, av2);
                av3 = v_f32_mac_b(arow, b3, av3);
                av4 = v_f32_mac_b(arow, b4, av4);
                av5 = v_f32_mac_b(arow, b5, av5);
                av6 = v_f32_mac_b(arow, b6, av6);
                av7 = v_f32_mac_b(arow, b7, av7);
            }
            float64 t0 = 0.0f, t1 = 0.0f, t2 = 0.0f, t3 = 0.0f;
            float64 t4 = 0.0f, t5 = 0.0f, t6 = 0.0f, t7 = 0.0f;
            for (int k = 0; k < n; k++)
            {
                int5 cka = {0, k, bt, 0, 0};
                float64 r0 = v_f32_sel_geq_u32_b(lane, (unsigned)n, 0.0f, v_f32_ld_tnsr_b(cka, V1));
                float64 r1 = 0.0f;
                if (has2)
                {
                    int5 ckb = {0, k, bt + 1, 0, 0};
                    r1 = v_f32_sel_geq_u32_b(lane, (unsigned)n, 0.0f, v_f32_ld_tnsr_b(ckb, V1));
                }
                float64 vk = pack_halves(r0, r1);
                t0 = v_f32_sel_eq_u32_b(lane_mod32, (unsigned)k, reduce_add_half32(v_f32_mul_b(vk, av0)), t0);
                t1 = v_f32_sel_eq_u32_b(lane_mod32, (unsigned)k, reduce_add_half32(v_f32_mul_b(vk, av1)), t1);
                t2 = v_f32_sel_eq_u32_b(lane_mod32, (unsigned)k, reduce_add_half32(v_f32_mul_b(vk, av2)), t2);
                t3 = v_f32_sel_eq_u32_b(lane_mod32, (unsigned)k, reduce_add_half32(v_f32_mul_b(vk, av3)), t3);
                t4 = v_f32_sel_eq_u32_b(lane_mod32, (unsigned)k, reduce_add_half32(v_f32_mul_b(vk, av4)), t4);
                t5 = v_f32_sel_eq_u32_b(lane_mod32, (unsigned)k, reduce_add_half32(v_f32_mul_b(vk, av5)), t5);
                t6 = v_f32_sel_eq_u32_b(lane_mod32, (unsigned)k, reduce_add_half32(v_f32_mul_b(vk, av6)), t6);
                t7 = v_f32_sel_eq_u32_b(lane_mod32, (unsigned)k, reduce_add_half32(v_f32_mul_b(vk, av7)), t7);
            }
            u[c0] = t0;
            if (j0 + 1 < n) u[c1] = t1;
            if (j0 + 2 < n) u[c2] = t2;
            if (j0 + 3 < n) u[c3] = t3;
            if (j0 + 4 < n) u[c4] = t4;
            if (j0 + 5 < n) u[c5] = t5;
            if (j0 + 6 < n) u[c6] = t6;
            if (j0 + 7 < n) u[c7] = t7;
        }

        // Compute tight Gershgorin bound sigma_T and lower bound g_min of T, normalize, and apply adaptive shift
        float64 sigma_p = 0.0f;
        float64 g_min = 0.0f;
        for (int i = 0; i < n; i++)
        {
            float64 row = u[i];
            float64 row_sum = reduce_add_half32(v_f32_abs_b(row));
            sigma_p = v_f32_max_b(sigma_p, row_sum);
            float64 diag_i = v_f32_sel_less_u32_b(lane, 32u,
                                                  as_float64(v_broadcast_element_i32(as_int64(row), i)),
                                                  as_float64(v_broadcast_element_i32(as_int64(row), i + 32)));
            float64 g_i = v_f32_sub_b(v_f32_add_b(diag_i, v_f32_abs_b(diag_i)), row_sum);
            g_min = (i == 0) ? g_i : v_f32_min_b(g_min, g_i);
        }
        float64 safe_sigma = v_f32_sel_eq_f32_b(sigma_p, 0.0f, 1.0f, sigma_p);
        safe_sigma = v_f32_sel_grt_f32_b(safe_sigma, 1e36f, 1.0f, safe_sigma);
        safe_sigma = v_f32_sel_less_f32_b(safe_sigma, 1e-37f, 1.0f, safe_sigma);
        float64 inv_sigma = v_reciprocal_f32(safe_sigma);
        inv_sigma = v_f32_mul_b(inv_sigma, v_f32_sub_b(2.0f, v_f32_mul_b(safe_sigma, inv_sigma)));
        float64 g_norm = v_f32_mul_b(g_min, inv_sigma);
        float64 shift_norm = v_f32_max_b(0.0f, v_f32_sub_b(0.01f, g_norm));
        for (int i = 0; i < n; i++)
        {
            float64 row = v_f32_mul_b(u[i], inv_sigma);
            u[i] = v_f32_sel_eq_u32_b(lane_mod32, (unsigned)i, v_f32_add_b(row, shift_norm), row);
        }

        run_jacobi_sweeps(m, half_n, k_start, sweeps);
        normalize_u_rows(n);

        // Compose V_final = V1 @ U2 into u[0..n-1] in 8-column VRF blocks (row j = sum_i U2[j,i] * V1[i])
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
            float64 uj0 = u[c0], uj1 = u[c1], uj2 = u[c2], uj3 = u[c3];
            float64 uj4 = u[c4], uj5 = u[c5], uj6 = u[c6], uj7 = u[c7];
            float64 vf0 = 0.0f, vf1 = 0.0f, vf2 = 0.0f, vf3 = 0.0f;
            float64 vf4 = 0.0f, vf5 = 0.0f, vf6 = 0.0f, vf7 = 0.0f;
            for (int i = 0; i < n; i++)
            {
                int5 ca = {0, i, bt, 0, 0};
                float64 r0 = v_f32_sel_geq_u32_b(lane, (unsigned)n, 0.0f, v_f32_ld_tnsr_b(ca, V1));
                float64 r1 = 0.0f;
                if (has2)
                {
                    int5 cb = {0, i, bt + 1, 0, 0};
                    r1 = v_f32_sel_geq_u32_b(lane, (unsigned)n, 0.0f, v_f32_ld_tnsr_b(cb, V1));
                }
                float64 v1row = pack_halves(r0, r1);
                float64 b0 = v_f32_sel_less_u32_b(lane, 32u,
                                                  as_float64(v_broadcast_element_i32(as_int64(uj0), i)),
                                                  as_float64(v_broadcast_element_i32(as_int64(uj0), i + 32)));
                float64 b1 = v_f32_sel_less_u32_b(lane, 32u,
                                                  as_float64(v_broadcast_element_i32(as_int64(uj1), i)),
                                                  as_float64(v_broadcast_element_i32(as_int64(uj1), i + 32)));
                float64 b2 = v_f32_sel_less_u32_b(lane, 32u,
                                                  as_float64(v_broadcast_element_i32(as_int64(uj2), i)),
                                                  as_float64(v_broadcast_element_i32(as_int64(uj2), i + 32)));
                float64 b3 = v_f32_sel_less_u32_b(lane, 32u,
                                                  as_float64(v_broadcast_element_i32(as_int64(uj3), i)),
                                                  as_float64(v_broadcast_element_i32(as_int64(uj3), i + 32)));
                float64 b4 = v_f32_sel_less_u32_b(lane, 32u,
                                                  as_float64(v_broadcast_element_i32(as_int64(uj4), i)),
                                                  as_float64(v_broadcast_element_i32(as_int64(uj4), i + 32)));
                float64 b5 = v_f32_sel_less_u32_b(lane, 32u,
                                                  as_float64(v_broadcast_element_i32(as_int64(uj5), i)),
                                                  as_float64(v_broadcast_element_i32(as_int64(uj5), i + 32)));
                float64 b6 = v_f32_sel_less_u32_b(lane, 32u,
                                                  as_float64(v_broadcast_element_i32(as_int64(uj6), i)),
                                                  as_float64(v_broadcast_element_i32(as_int64(uj6), i + 32)));
                float64 b7 = v_f32_sel_less_u32_b(lane, 32u,
                                                  as_float64(v_broadcast_element_i32(as_int64(uj7), i)),
                                                  as_float64(v_broadcast_element_i32(as_int64(uj7), i + 32)));
                vf0 = v_f32_mac_b(v1row, b0, vf0);
                vf1 = v_f32_mac_b(v1row, b1, vf1);
                vf2 = v_f32_mac_b(v1row, b2, vf2);
                vf3 = v_f32_mac_b(v1row, b3, vf3);
                vf4 = v_f32_mac_b(v1row, b4, vf4);
                vf5 = v_f32_mac_b(v1row, b5, vf5);
                vf6 = v_f32_mac_b(v1row, b6, vf6);
                vf7 = v_f32_mac_b(v1row, b7, vf7);
            }
            u[c0] = vf0;
            if (j0 + 1 < n) u[c1] = vf1;
            if (j0 + 2 < n) u[c2] = vf2;
            if (j0 + 3 < n) u[c3] = vf3;
            if (j0 + 4 < n) u[c4] = vf4;
            if (j0 + 5 < n) u[c5] = vf5;
            if (j0 + 6 < n) u[c6] = vf6;
            if (j0 + 7 < n) u[c7] = vf7;
        }
        normalize_u_rows(n);
#else
        // Tight prescaled Gershgorin bound sigma_p
        float64 sigma_p = 0.0f;
        int ip = 0;
        for (; ip + 1 < n; ip += 2)
        {
            float64 r0 = v_f32_mul_b(u[ip], prescale);
            float64 r1 = v_f32_mul_b(u[ip + 1], prescale);
            u[ip] = r0;
            u[ip + 1] = r1;
            float64 s0 = reduce_add_half32(v_f32_abs_b(r0));
            float64 s1 = reduce_add_half32(v_f32_abs_b(r1));
            sigma_p = v_f32_max_b(sigma_p, v_f32_max_b(s0, s1));
        }
        if (ip < n)
        {
            float64 r0 = v_f32_mul_b(u[ip], prescale);
            u[ip] = r0;
            sigma_p = v_f32_max_b(sigma_p, reduce_add_half32(v_f32_abs_b(r0)));
        }

        float64 safe_sigma = v_f32_sel_eq_f32_b(sigma_p, 0.0f, 1.0f, sigma_p);
        safe_sigma = v_f32_sel_grt_f32_b(safe_sigma, 1e35f, 1.0f, safe_sigma);
        safe_sigma = v_f32_sel_less_f32_b(safe_sigma, 1e-37f, 1.0f, safe_sigma);
        float64 inv_sigma = v_reciprocal_f32(safe_sigma);
        inv_sigma = v_f32_mul_b(inv_sigma, v_f32_sub_b(2.0f, v_f32_mul_b(safe_sigma, inv_sigma)));
        for (int i = 0; i < n; i++)
        {
            float64 row = v_f32_mul_b(u[i], inv_sigma);
            u[i] = v_f32_sel_eq_u32_b(lane_mod32, (unsigned)i, v_f32_add_b(row, shift_norm), row);
        }

        run_jacobi_sweeps(m, half_n, k_start, sweeps);
        normalize_u_rows(n);
#endif

        // Store V rows from u[0..n-1]
        for (int j = 0; j < n; j++)
        {
            float64 vj = u[j];
            int5 ca = {0, j, bt, 0, 0};
            v_f32_st_tnsr_partial(ca, V, vj, n - 1, 0);
            if (has2)
            {
                int5 cb = {0, j, bt + 1, 0, 0};
                v_f32_st_tnsr_partial(cb, V, unpack_hi_half(vj), n - 1, 0);
            }
        }

        // Paired Rayleigh quotient W[j] = v_j^T (A * prescale) v_j * inv_prescale in 8-column VRF blocks
        float64 w_packed = 0.0f;
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
                int5 ca = {0, i, bt, 0, 0};
                float64 r0 = v_f32_sel_geq_u32_b(lane, (unsigned)n, 0.0f, v_f32_ld_tnsr_b(ca, A));
                float64 r1 = 0.0f;
                if (has2)
                {
                    int5 cb = {0, i, bt + 1, 0, 0};
                    r1 = v_f32_sel_geq_u32_b(lane, (unsigned)n, 0.0f, v_f32_ld_tnsr_b(cb, A));
                }
                float64 arow = v_f32_mul_b(pack_halves(r0, r1), prescale);
                float64 b0 = v_f32_sel_less_u32_b(lane, 32u,
                                                  as_float64(v_broadcast_element_i32(as_int64(v0), i)),
                                                  as_float64(v_broadcast_element_i32(as_int64(v0), i + 32)));
                float64 b1 = v_f32_sel_less_u32_b(lane, 32u,
                                                  as_float64(v_broadcast_element_i32(as_int64(v1), i)),
                                                  as_float64(v_broadcast_element_i32(as_int64(v1), i + 32)));
                float64 b2 = v_f32_sel_less_u32_b(lane, 32u,
                                                  as_float64(v_broadcast_element_i32(as_int64(v2), i)),
                                                  as_float64(v_broadcast_element_i32(as_int64(v2), i + 32)));
                float64 b3 = v_f32_sel_less_u32_b(lane, 32u,
                                                  as_float64(v_broadcast_element_i32(as_int64(v3), i)),
                                                  as_float64(v_broadcast_element_i32(as_int64(v3), i + 32)));
                float64 b4 = v_f32_sel_less_u32_b(lane, 32u,
                                                  as_float64(v_broadcast_element_i32(as_int64(v4), i)),
                                                  as_float64(v_broadcast_element_i32(as_int64(v4), i + 32)));
                float64 b5 = v_f32_sel_less_u32_b(lane, 32u,
                                                  as_float64(v_broadcast_element_i32(as_int64(v5), i)),
                                                  as_float64(v_broadcast_element_i32(as_int64(v5), i + 32)));
                float64 b6 = v_f32_sel_less_u32_b(lane, 32u,
                                                  as_float64(v_broadcast_element_i32(as_int64(v6), i)),
                                                  as_float64(v_broadcast_element_i32(as_int64(v6), i + 32)));
                float64 b7 = v_f32_sel_less_u32_b(lane, 32u,
                                                  as_float64(v_broadcast_element_i32(as_int64(v7), i)),
                                                  as_float64(v_broadcast_element_i32(as_int64(v7), i + 32)));
                av0 = v_f32_mac_b(arow, b0, av0);
                av1 = v_f32_mac_b(arow, b1, av1);
                av2 = v_f32_mac_b(arow, b2, av2);
                av3 = v_f32_mac_b(arow, b3, av3);
                av4 = v_f32_mac_b(arow, b4, av4);
                av5 = v_f32_mac_b(arow, b5, av5);
                av6 = v_f32_mac_b(arow, b6, av6);
                av7 = v_f32_mac_b(arow, b7, av7);
            }
            float64 w0 = reduce_add_half32(v_f32_mul_b(v0, av0));
            float64 w1 = reduce_add_half32(v_f32_mul_b(v1, av1));
            float64 w2 = reduce_add_half32(v_f32_mul_b(v2, av2));
            float64 w3 = reduce_add_half32(v_f32_mul_b(v3, av3));
            float64 w4 = reduce_add_half32(v_f32_mul_b(v4, av4));
            float64 w5 = reduce_add_half32(v_f32_mul_b(v5, av5));
            float64 w6 = reduce_add_half32(v_f32_mul_b(v6, av6));
            float64 w7 = reduce_add_half32(v_f32_mul_b(v7, av7));
            w_packed = v_f32_sel_eq_u32_b(lane_mod32, (unsigned)(j0 + 0), w0, w_packed);
            if (j0 + 1 < n) w_packed = v_f32_sel_eq_u32_b(lane_mod32, (unsigned)(j0 + 1), w1, w_packed);
            if (j0 + 2 < n) w_packed = v_f32_sel_eq_u32_b(lane_mod32, (unsigned)(j0 + 2), w2, w_packed);
            if (j0 + 3 < n) w_packed = v_f32_sel_eq_u32_b(lane_mod32, (unsigned)(j0 + 3), w3, w_packed);
            if (j0 + 4 < n) w_packed = v_f32_sel_eq_u32_b(lane_mod32, (unsigned)(j0 + 4), w4, w_packed);
            if (j0 + 5 < n) w_packed = v_f32_sel_eq_u32_b(lane_mod32, (unsigned)(j0 + 5), w5, w_packed);
            if (j0 + 6 < n) w_packed = v_f32_sel_eq_u32_b(lane_mod32, (unsigned) (j0 + 6), w6, w_packed);
            if (j0 + 7 < n) w_packed = v_f32_sel_eq_u32_b(lane_mod32, (unsigned)(j0 + 7), w7, w_packed);
        }
        w_packed = v_f32_mul_b(w_packed, inv_prescale);
        int5 cwa = {0, bt, 0, 0, 0};
        v_f32_st_tnsr_partial(cwa, W, w_packed, n - 1, 0);
        if (has2)
        {
            int5 cwb = {0, bt + 1, 0, 0, 0};
            v_f32_st_tnsr_partial(cwb, W, unpack_hi_half(w_packed), n - 1, 0);
        }
    }
}
