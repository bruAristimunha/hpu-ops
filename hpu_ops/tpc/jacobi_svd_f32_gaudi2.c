/* Batched thin one-sided (Hestenes) Jacobi SVD on Gaudi2 TPC for m >= n, m,n <= 32.
 *
 * Tensors (TPC geometry, dim0 innermost; torch [batch, rows, cols] row-major):
 *   A_t [m, n, batch]  f32: A^T so column j of A sits in contiguous TPC row j
 *   Ut  [m, n, batch]  f32: U^T (row j = left singular vector u_j; [m, m, batch] under -DFULL_U; omitted under -DNO_UV)
 *   S   [n, batch]     f32: singular values >= 0, unsorted
 *   Vt  [n, n, batch]  f32: V^T = Vh (row j = right singular vector v_j; omitted under -DNO_UV)
 *
 * Full kernel: 1 matrix per iteration; uv[j] packs w_j in lanes 0..31 and v_j in lanes 32..63 (8 KB VLM).
 * -DFULL_U: same 8 KB VLM; after Vt is stored, extends Ut[0..n-1] to Ut[0..m-1] via 2-pass MGS in uv[0..m-1].
 * -DNO_UV: 2 matrices per iteration (bt in lanes 0..31, bt+1 in lanes 32..63); w[j] in 8 KB VLM.
 */
#define MAXN 32

#ifdef NO_UV
__local__ float64 w[MAXN];
#define ARR w
#else
__local__ float64 uv[MAXN];
#define ARR uv
#endif

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

// 1/sqrt(x): the hardware estimate, then one Newton step in residual form, e = 1 - (x r) r with the product rounded
// once (an FMA), r + (r / 2) e. The plain step r (1.5 - x r^2 / 2) keeps an estimate 1 ulp low near 1 (its fixed
// point), which shrank every rotation by an ulp: the accumulated rotations' orthogonality drifted with n, 6.1e-6 at
// n = 32 on Gaudi2 against torch's 1.3e-6 (found by team Z on its complex kernels).
static inline float64 rsqrt_newton(float64 x)
{
    float64 r = v_rsqrt_f32(x);
    float64 e = v_f32_mac_b(v_f32_sub_b(0.0f, v_f32_mul_b(x, r)), r, 1.0f);
    return v_f32_mac_b(v_f32_mul_b(r, 0.5f), e, r);
}

#define RR_PAIR(r, k, rr_m, p, q)          \
    do                                     \
    {                                      \
        if ((k) == 0)                      \
        {                                  \
            (p) = (r);                     \
            (q) = (rr_m);                  \
        }                                  \
        else                               \
        {                                  \
            int a_ = (r) + (k);            \
            if (a_ >= (rr_m)) a_ -= (rr_m);\
            int b_ = (r) - (k);            \
            if (b_ < 0) b_ += (rr_m);      \
            (p) = (a_ < b_) ? a_ : b_;     \
            (q) = (a_ < b_) ? b_ : a_;     \
        }                                  \
    } while (0)

#ifdef NO_UV
#define BROADCAST_CS(c, s) ((void)0)
#else
#define BROADCAST_CS(c, s)                                                     \
    do                                                                         \
    {                                                                          \
        (c) = v_f32_mov_dual_group_all_b((c), 0xFFFFFFFF, 0, 1, 0, 1, SW_MDG_ALL, (c)); \
        (s) = v_f32_mov_dual_group_all_b((s), 0xFFFFFFFF, 0, 1, 0, 1, SW_MDG_ALL, (s)); \
    } while (0)
#endif

#ifdef NO_UV
void main(tensor A_t, tensor S)
#else
void main(tensor A_t, tensor Ut, tensor S, tensor Vt)
#endif
{
    const int m = get_dim_size(A_t, 0);
    const int n = get_dim_size(A_t, 1);
    const uint64 lane = read_lane_id_4b_b();
    const uint64 lane_mod32 = lane & 31u;
    const int5 off = get_index_space_offset();
    const int5 size = get_index_space_size();
    const int n_even = (n + 1) & ~1;
    const int rr_m = n_even - 1;
    const int half_n = n_even >> 1;
    const int k_start = n & 1;
    const int sweeps = (n <= 4) ? 6 : (n <= 16) ? 10 : 12;

#ifdef NO_UV
    for (int bt = off[0]; bt < off[0] + size[0]; bt += 2)
    {
        const int has2 = (bt + 1 < off[0] + size[0]);
        float64 sigma = 0.0f;
        for (int j = 0; j < n; j++)
        {
            int5 c0 = {0, j, bt, 0, 0};
            float64 r0 = v_f32_sel_geq_u32_b(lane, (unsigned)m, 0.0f, v_f32_ld_tnsr_b(c0, A_t));
            float64 r1 = 0.0f;
            if (has2)
            {
                int5 c1 = {0, j, bt + 1, 0, 0};
                r1 = v_f32_sel_geq_u32_b(lane, (unsigned)m, 0.0f, v_f32_ld_tnsr_b(c1, A_t));
            }
            float64 packed = pack_halves(r0, r1);
            w[j] = packed;
            sigma = v_f32_max_b(sigma, reduce_max_half32(v_f32_abs_b(packed)));
        }
#else
    for (int bt = off[0]; bt < off[0] + size[0]; bt += 1)
    {
        float64 sigma = 0.0f;
        for (int j = 0; j < n; j++)
        {
            int5 c0 = {0, j, bt, 0, 0};
            float64 wj = v_f32_sel_geq_u32_b(lane, (unsigned)m, 0.0f, v_f32_ld_tnsr_b(c0, A_t));
            float64 vj = v_f32_sel_eq_u32_b(lane, (unsigned)j, 1.0f, 0.0f);
            uv[j] = pack_halves(wj, vj);
            sigma = v_f32_max_b(sigma, reduce_max_half32(v_f32_abs_b(wj)));
        }
#endif
        float64 pre_scale = v_f32_sel_less_f32_b(1.0e35f, sigma, 1.0e-4f, 1.0f);
        // every matrix is scaled by 1/max|a| (also a tiny one: 1/sigma is normal for any normal sigma), except a zero one
        float64 safe_sigma = v_f32_sel_eq_f32_b(sigma, 0.0f, 1.0f, sigma);
        float64 adj_sigma = v_f32_mul_b(safe_sigma, pre_scale);
        float64 inv_sigma = v_reciprocal_f32(adj_sigma);
        inv_sigma = v_f32_mul_b(inv_sigma, v_f32_sub_b(2.0f, v_f32_mul_b(adj_sigma, inv_sigma)));
#ifndef NO_UV
        pre_scale = v_f32_sel_geq_u32_b(lane, 32u, 1.0f, pre_scale);
        inv_sigma = v_f32_sel_geq_u32_b(lane, 32u, 1.0f, inv_sigma);
#endif
        for (int j = 0; j < n; j++)
        {
            ARR[j] = v_f32_mul_b(v_f32_mul_b(ARR[j], pre_scale), inv_sigma);
        }

        for (int sweep = 0; sweep < sweeps; sweep++)
        {
            for (int r = 0; r < rr_m; r++)
            {
                int k = k_start;
                for (; k + 2 < half_n; k += 3)
                {
                    int p0, q0, p1, q1, p2, q2;
                    RR_PAIR(r, k, rr_m, p0, q0);
                    RR_PAIR(r, k + 1, rr_m, p1, q1);
                    RR_PAIR(r, k + 2, rr_m, p2, q2);

                    float64 up0 = ARR[p0], uq0 = ARR[q0];
                    float64 up1 = ARR[p1], uq1 = ARR[q1];
                    float64 up2 = ARR[p2], uq2 = ARR[q2];

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
                    float64 sr20 = v_f32_sel_less_f32_b(r20, 1.0e-34f, 1.0f, r20);
                    float64 sr21 = v_f32_sel_less_f32_b(r21, 1.0e-34f, 1.0f, r21);
                    float64 sr22 = v_f32_sel_less_f32_b(r22, 1.0e-34f, 1.0f, r22);

                    float64 inv_r0 = rsqrt_newton(sr20);
                    float64 inv_r1 = rsqrt_newton(sr21);
                    float64 inv_r2 = rsqrt_newton(sr22);

                    float64 c20 = v_f32_mul_b(0.5f, v_f32_mac_b(v_f32_abs_b(d0), inv_r0, 1.0f));
                    float64 c21 = v_f32_mul_b(0.5f, v_f32_mac_b(v_f32_abs_b(d1), inv_r1, 1.0f));
                    float64 c22 = v_f32_mul_b(0.5f, v_f32_mac_b(v_f32_abs_b(d2), inv_r2, 1.0f));

                    float64 inv_c0 = rsqrt_newton(c20);
                    float64 inv_c1 = rsqrt_newton(c21);
                    float64 inv_c2 = rsqrt_newton(c22);

                    float64 c0 = v_f32_mul_b(c20, inv_c0);
                    float64 c1 = v_f32_mul_b(c21, inv_c1);
                    float64 c2 = v_f32_mul_b(c22, inv_c2);
                    float64 s0 = v_f32_mul_b(v_f32_mul_b(0.5f, g20), v_f32_mul_b(inv_r0, inv_c0));
                    float64 s1 = v_f32_mul_b(v_f32_mul_b(0.5f, g21), v_f32_mul_b(inv_r1, inv_c1));
                    float64 s2 = v_f32_mul_b(v_f32_mul_b(0.5f, g22), v_f32_mul_b(inv_r2, inv_c2));
                    s0 = v_f32_sel_less_f32_b(d0, 0.0f, v_f32_sub_b(0.0f, s0), s0);
                    s1 = v_f32_sel_less_f32_b(d1, 0.0f, v_f32_sub_b(0.0f, s1), s1);
                    s2 = v_f32_sel_less_f32_b(d2, 0.0f, v_f32_sub_b(0.0f, s2), s2);

                    float64 gsq0 = v_f32_mul_b(gamma0, gamma0);
                    float64 tol0 = v_f32_mul_b(1.0e-14f, v_f32_mul_b(alpha0, beta0));
                    c0 = v_f32_sel_less_f32_b(tol0, gsq0, c0, 1.0f);
                    s0 = v_f32_sel_less_f32_b(tol0, gsq0, s0, 0.0f);
                    c0 = v_f32_sel_less_f32_b(r20, 1.0e-34f, 1.0f, c0);
                    s0 = v_f32_sel_less_f32_b(r20, 1.0e-34f, 0.0f, s0);
                    c0 = v_f32_sel_eq_f32_b(gamma0, 0.0f, 1.0f, c0);
                    s0 = v_f32_sel_eq_f32_b(gamma0, 0.0f, 0.0f, s0);

                    float64 gsq1 = v_f32_mul_b(gamma1, gamma1);
                    float64 tol1 = v_f32_mul_b(1.0e-14f, v_f32_mul_b(alpha1, beta1));
                    c1 = v_f32_sel_less_f32_b(tol1, gsq1, c1, 1.0f);
                    s1 = v_f32_sel_less_f32_b(tol1, gsq1, s1, 0.0f);
                    c1 = v_f32_sel_less_f32_b(r21, 1.0e-34f, 1.0f, c1);
                    s1 = v_f32_sel_less_f32_b(r21, 1.0e-34f, 0.0f, s1);
                    c1 = v_f32_sel_eq_f32_b(gamma1, 0.0f, 1.0f, c1);
                    s1 = v_f32_sel_eq_f32_b(gamma1, 0.0f, 0.0f, s1);

                    float64 gsq2 = v_f32_mul_b(gamma2, gamma2);
                    float64 tol2 = v_f32_mul_b(1.0e-14f, v_f32_mul_b(alpha2, beta2));
                    c2 = v_f32_sel_less_f32_b(tol2, gsq2, c2, 1.0f);
                    s2 = v_f32_sel_less_f32_b(tol2, gsq2, s2, 0.0f);
                    c2 = v_f32_sel_less_f32_b(r22, 1.0e-34f, 1.0f, c2);
                    s2 = v_f32_sel_less_f32_b(r22, 1.0e-34f, 0.0f, s2);
                    c2 = v_f32_sel_eq_f32_b(gamma2, 0.0f, 1.0f, c2);
                    s2 = v_f32_sel_eq_f32_b(gamma2, 0.0f, 0.0f, s2);

                    BROADCAST_CS(c0, s0);
                    BROADCAST_CS(c1, s1);
                    BROADCAST_CS(c2, s2);

                    float64 ns0 = v_f32_sub_b(0.0f, s0);
                    float64 ns1 = v_f32_sub_b(0.0f, s1);
                    float64 ns2 = v_f32_sub_b(0.0f, s2);
                    ARR[p0] = v_f32_mac_b(uq0, ns0, v_f32_mul_b(up0, c0));
                    ARR[q0] = v_f32_mac_b(up0, s0, v_f32_mul_b(uq0, c0));
                    ARR[p1] = v_f32_mac_b(uq1, ns1, v_f32_mul_b(up1, c1));
                    ARR[q1] = v_f32_mac_b(up1, s1, v_f32_mul_b(uq1, c1));
                    ARR[p2] = v_f32_mac_b(uq2, ns2, v_f32_mul_b(up2, c2));
                    ARR[q2] = v_f32_mac_b(up2, s2, v_f32_mul_b(uq2, c2));
                }
                if (k + 1 < half_n)
                {
                    int p0, q0, p1, q1;
                    RR_PAIR(r, k, rr_m, p0, q0);
                    RR_PAIR(r, k + 1, rr_m, p1, q1);
                    k += 2;

                    float64 up0 = ARR[p0], uq0 = ARR[q0];
                    float64 up1 = ARR[p1], uq1 = ARR[q1];

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
                    float64 sr20 = v_f32_sel_less_f32_b(r20, 1.0e-34f, 1.0f, r20);
                    float64 sr21 = v_f32_sel_less_f32_b(r21, 1.0e-34f, 1.0f, r21);

                    float64 inv_r0 = rsqrt_newton(sr20);
                    float64 inv_r1 = rsqrt_newton(sr21);

                    float64 c20 = v_f32_mul_b(0.5f, v_f32_mac_b(v_f32_abs_b(d0), inv_r0, 1.0f));
                    float64 c21 = v_f32_mul_b(0.5f, v_f32_mac_b(v_f32_abs_b(d1), inv_r1, 1.0f));

                    float64 inv_c0 = rsqrt_newton(c20);
                    float64 inv_c1 = rsqrt_newton(c21);

                    float64 c0 = v_f32_mul_b(c20, inv_c0);
                    float64 c1 = v_f32_mul_b(c21, inv_c1);
                    float64 s0 = v_f32_mul_b(v_f32_mul_b(0.5f, g20), v_f32_mul_b(inv_r0, inv_c0));
                    float64 s1 = v_f32_mul_b(v_f32_mul_b(0.5f, g21), v_f32_mul_b(inv_r1, inv_c1));
                    s0 = v_f32_sel_less_f32_b(d0, 0.0f, v_f32_sub_b(0.0f, s0), s0);
                    s1 = v_f32_sel_less_f32_b(d1, 0.0f, v_f32_sub_b(0.0f, s1), s1);

                    float64 gsq0 = v_f32_mul_b(gamma0, gamma0);
                    float64 tol0 = v_f32_mul_b(1.0e-14f, v_f32_mul_b(alpha0, beta0));
                    c0 = v_f32_sel_less_f32_b(tol0, gsq0, c0, 1.0f);
                    s0 = v_f32_sel_less_f32_b(tol0, gsq0, s0, 0.0f);
                    c0 = v_f32_sel_less_f32_b(r20, 1.0e-34f, 1.0f, c0);
                    s0 = v_f32_sel_less_f32_b(r20, 1.0e-34f, 0.0f, s0);
                    c0 = v_f32_sel_eq_f32_b(gamma0, 0.0f, 1.0f, c0);
                    s0 = v_f32_sel_eq_f32_b(gamma0, 0.0f, 0.0f, s0);

                    float64 gsq1 = v_f32_mul_b(gamma1, gamma1);
                    float64 tol1 = v_f32_mul_b(1.0e-14f, v_f32_mul_b(alpha1, beta1));
                    c1 = v_f32_sel_less_f32_b(tol1, gsq1, c1, 1.0f);
                    s1 = v_f32_sel_less_f32_b(tol1, gsq1, s1, 0.0f);
                    c1 = v_f32_sel_less_f32_b(r21, 1.0e-34f, 1.0f, c1);
                    s1 = v_f32_sel_less_f32_b(r21, 1.0e-34f, 0.0f, s1);
                    c1 = v_f32_sel_eq_f32_b(gamma1, 0.0f, 1.0f, c1);
                    s1 = v_f32_sel_eq_f32_b(gamma1, 0.0f, 0.0f, s1);

                    BROADCAST_CS(c0, s0);
                    BROADCAST_CS(c1, s1);

                    float64 ns0 = v_f32_sub_b(0.0f, s0);
                    float64 ns1 = v_f32_sub_b(0.0f, s1);
                    ARR[p0] = v_f32_mac_b(uq0, ns0, v_f32_mul_b(up0, c0));
                    ARR[q0] = v_f32_mac_b(up0, s0, v_f32_mul_b(uq0, c0));
                    ARR[p1] = v_f32_mac_b(uq1, ns1, v_f32_mul_b(up1, c1));
                    ARR[q1] = v_f32_mac_b(up1, s1, v_f32_mul_b(uq1, c1));
                }
                if (k < half_n)
                {
                    int p, q;
                    RR_PAIR(r, k, rr_m, p, q);
                    float64 up = ARR[p], uq = ARR[q];
                    float64 alpha = reduce_add_half32(v_f32_mul_b(up, up));
                    float64 beta = reduce_add_half32(v_f32_mul_b(uq, uq));
                    float64 gamma = reduce_add_half32(v_f32_mul_b(up, uq));

                    float64 d = v_f32_sub_b(beta, alpha);
                    float64 g2 = v_f32_add_b(gamma, gamma);
                    float64 r2 = v_f32_mac_b(g2, g2, v_f32_mul_b(d, d));
                    float64 sr2 = v_f32_sel_less_f32_b(r2, 1.0e-34f, 1.0f, r2);
                    float64 inv_r = rsqrt_newton(sr2);
                    float64 c2 = v_f32_mul_b(0.5f, v_f32_mac_b(v_f32_abs_b(d), inv_r, 1.0f));
                    float64 inv_c = rsqrt_newton(c2);
                    float64 c = v_f32_mul_b(c2, inv_c);
                    float64 s = v_f32_mul_b(v_f32_mul_b(0.5f, g2), v_f32_mul_b(inv_r, inv_c));
                    s = v_f32_sel_less_f32_b(d, 0.0f, v_f32_sub_b(0.0f, s), s);

                    float64 gsq = v_f32_mul_b(gamma, gamma);
                    float64 tol = v_f32_mul_b(1.0e-14f, v_f32_mul_b(alpha, beta));
                    c = v_f32_sel_less_f32_b(tol, gsq, c, 1.0f);
                    s = v_f32_sel_less_f32_b(tol, gsq, s, 0.0f);
                    c = v_f32_sel_less_f32_b(r2, 1.0e-34f, 1.0f, c);
                    s = v_f32_sel_less_f32_b(r2, 1.0e-34f, 0.0f, s);
                    c = v_f32_sel_eq_f32_b(gamma, 0.0f, 1.0f, c);
                    s = v_f32_sel_eq_f32_b(gamma, 0.0f, 0.0f, s);

                    BROADCAST_CS(c, s);

                    float64 ns = v_f32_sub_b(0.0f, s);
                    ARR[p] = v_f32_mac_b(uq, ns, v_f32_mul_b(up, c));
                    ARR[q] = v_f32_mac_b(up, s, v_f32_mul_b(uq, c));
                }
            }
        }

        // Compute singular values and (in full mode) normalize U columns + MGS completion
        float64 s_vec = 0.0f;
#ifdef NO_UV
        for (int j = 0; j < n; j++)
        {
            float64 sq = reduce_add_half32(v_f32_mul_b(w[j], w[j]));
            float64 safe_sq = v_f32_sel_less_f32_b(sq, 1.0e-36f, 1.0f, sq);
            float64 rn = rsqrt_newton(safe_sq);
            float64 sj = v_f32_sel_less_f32_b(sq, 1.0e-36f, 0.0f, v_f32_mul_b(v_f32_mul_b(sq, rn), safe_sigma));
            s_vec = v_f32_sel_eq_u32_b(lane_mod32, (unsigned)j, sj, s_vec);
        }
        int5 cs0 = {0, bt, 0, 0, 0};
        v_f32_st_tnsr_partial(cs0, S, s_vec, n - 1, 0);
        if (has2)
        {
            int5 cs1 = {0, bt + 1, 0, 0, 0};
            v_f32_st_tnsr_partial(cs1, S, unpack_hi_half(s_vec), n - 1, 0);
        }
#else
        float64 sq_vec = 0.0f;
        float64 max_sq = 0.0f;
        for (int j = 0; j < n; j++)
        {
            float64 cur = uv[j];
            float64 sq = reduce_add_half32(v_f32_mul_b(cur, cur));
            sq_vec = v_f32_sel_eq_u32_b(lane_mod32, (unsigned)j, sq, sq_vec);
            max_sq = v_f32_max_b(max_sq, sq);

            float64 safe_sq = v_f32_sel_less_f32_b(sq, 1.0e-36f, 1.0f, sq);
            float64 rn = rsqrt_newton(safe_sq);
            float64 sj = v_f32_sel_less_f32_b(sq, 1.0e-36f, 0.0f, v_f32_mul_b(v_f32_mul_b(sq, rn), safe_sigma));
            s_vec = v_f32_sel_eq_u32_b(lane_mod32, (unsigned)j, sj, s_vec);

            float64 scale_w = v_f32_sel_less_f32_b(sq, 1.0e-36f, 0.0f, rn);
            float64 scale_uv = v_f32_sel_geq_u32_b(lane, 32u, 1.0f, scale_w);
            uv[j] = v_f32_mul_b(cur, scale_uv);
        }

        // Vectorized 2-pass MGS completion for any zero/near-zero columns (predicated by done_vec, no scalar sync)
        float64 thresh = v_f32_max_b(1.0e-36f, v_f32_mul_b(1.0e-10f, max_sq));
        float64 done_vec = v_f32_sel_less_f32_b(sq_vec, thresh, 0.0f, 1.0f);
        done_vec = v_f32_sel_geq_u32_b(lane, (unsigned)n, 0.0f, done_vec);
        float64 diag_p = v_f32_sel_less_u32_b(lane, (unsigned)m, 1.0f, -1.0f);
        for (int k = 0; k < n; k++)
        {
            float64 dk = reduce_add_half32(v_f32_sel_eq_u32_b(lane_mod32, (unsigned)k, done_vec, 0.0f));
            float64 uk = v_f32_sel_geq_u32_b(lane, (unsigned)m, 0.0f, v_f32_mul_b(uv[k], dk));
            diag_p = v_f32_sub_b(diag_p, v_f32_mul_b(uk, uk));
        }

        for (int j = 0; j < n; j++)
        {
            float64 dj = reduce_add_half32(v_f32_sel_eq_u32_b(lane_mod32, (unsigned)j, done_vec, 0.0f));
            float64 cand1 = v_f32_sel_geq_u32_b(lane, (unsigned)m, 0.0f, uv[j]);
            uint64 p_idx = v_f32_reduce_argmax(diag_p).v1;
            float64 cand2 = v_f32_sel_eq_u32_b(lane, p_idx, 1.0f, 0.0f);

            for (int pass = 0; pass < 2; pass++)
            {
                for (int k = 0; k < n; k++)
                {
                    float64 dk = reduce_add_half32(v_f32_sel_eq_u32_b(lane_mod32, (unsigned)k, done_vec, 0.0f));
                    float64 uk = v_f32_sel_geq_u32_b(lane, (unsigned)m, 0.0f, v_f32_mul_b(uv[k], dk));
                    float64 dot1 = reduce_add_half32(v_f32_mul_b(cand1, uk));
                    float64 dot2 = reduce_add_half32(v_f32_mul_b(cand2, uk));
                    cand1 = v_f32_sub_b(cand1, v_f32_mul_b(dot1, uk));
                    cand2 = v_f32_sub_b(cand2, v_f32_mul_b(dot2, uk));
                }
            }
            float64 cn1 = reduce_add_half32(v_f32_mul_b(cand1, cand1));
            float64 cn2 = reduce_add_half32(v_f32_mul_b(cand2, cand2));
            float64 cand = v_f32_sel_less_f32_b(0.25f, cn1, cand1, cand2);
            float64 cn_sel = v_f32_sel_less_f32_b(0.25f, cn1, cn1, cn2);
            float64 safe_cn = v_f32_sel_less_f32_b(cn_sel, 1.0e-36f, 1.0f, cn_sel);
            cand = v_f32_mul_b(cand, rsqrt_newton(safe_cn));

            float64 cur_uv = uv[j];
            float64 new_w = v_f32_sel_eq_f32_b(dj, 0.0f, cand, cur_uv);
            uv[j] = v_f32_sel_geq_u32_b(lane, 32u, cur_uv, new_w);

            float64 new_uk = v_f32_sel_eq_f32_b(dj, 0.0f, cand, 0.0f);
            diag_p = v_f32_sub_b(diag_p, v_f32_mul_b(new_uk, new_uk));
            done_vec = v_f32_sel_eq_u32_b(lane_mod32, (unsigned)j, 1.0f, done_vec);
        }

        int5 cs = {0, bt, 0, 0, 0};
        v_f32_st_tnsr_partial(cs, S, s_vec, n - 1, 0);
        for (int j = 0; j < n; j++)
        {
            int5 cu = {0, j, bt, 0, 0};
            float64 cur = uv[j];
            v_f32_st_tnsr_partial(cu, Ut, cur, m - 1, 0);
            v_f32_st_tnsr_partial(cu, Vt, unpack_hi_half(cur), n - 1, 0);
#ifdef FULL_U
            // Clear v_j (lanes 32..63) and any >= m garbage before extending Ut to m rows in uv[0..m-1]
            uv[j] = v_f32_sel_geq_u32_b(lane, (unsigned)m, 0.0f, cur);
#endif
        }
#ifdef FULL_U
        for (int j = n; j < m; j++)
        {
            uint64 p_idx = v_f32_reduce_argmax(diag_p).v1;
            float64 cand = v_f32_sel_eq_u32_b(lane, p_idx, 1.0f, 0.0f);
            for (int pass = 0; pass < 2; pass++)
            {
                for (int k = 0; k < j; k++)
                {
                    float64 uk = uv[k];
                    float64 dot = reduce_add_half32(v_f32_mul_b(cand, uk));
                    cand = v_f32_sub_b(cand, v_f32_mul_b(dot, uk));
                }
            }
            float64 cn = reduce_add_half32(v_f32_mul_b(cand, cand));
            float64 safe_cn = v_f32_sel_less_f32_b(cn, 1.0e-36f, 1.0f, cn);
            cand = v_f32_mul_b(cand, rsqrt_newton(safe_cn));
            cand = v_f32_sel_geq_u32_b(lane, (unsigned)m, 0.0f, cand);
            uv[j] = cand;
            diag_p = v_f32_sub_b(diag_p, v_f32_mul_b(cand, cand));
            int5 cu = {0, j, bt, 0, 0};
            v_f32_st_tnsr_partial(cu, Ut, cand, m - 1, 0);
        }
#endif
#endif
    }
}
