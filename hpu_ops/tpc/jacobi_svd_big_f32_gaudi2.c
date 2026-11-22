/* Batched thin one-sided (Hestenes) Jacobi SVD on Gaudi2 TPC, LUT-free (80 KB VLM).
 *
 * Modes:
 *   - Default (MAXN = 64, VA in {1, 2}): m <= 128, n <= 64, m >= n -> Ut [m, n, batch], S [n, batch], Vt [n, n, batch].
 *   - -DNO_UV (MAXN = 128, VA in {1, 2}): m <= 128, n <= 128, m >= n -> S [n, batch].
 *
 * Tensors (TPC geometry, dim0 innermost; torch [batch, rows, cols] row-major):
 *   A_t [m, n, batch]  f32: A^T so column j of A sits in contiguous TPC row j (w_j)
 *   Ut  [m, n, batch]  f32: U^T (row j = left singular vector u_j; omitted under -DNO_UV)
 *   S   [n, batch]     f32: singular values >= 0, unsorted
 *   Vt  [n, n, batch]  f32: V^T = Vh (row j = right singular vector v_j; omitted under -DNO_UV)
 *
 * LUT-free arithmetic (no v_rsqrt_f32, v_reciprocal_f32, v_sqrt_f32, v_div_f32, v_f32_reduce_*) preserves 80 KB VLM:
 *   - Default VA = 2: w[128] + v[64] = 192 vectors = 48 KB
 *   - NO_UV   VA = 2: w[256]         = 256 vectors = 64 KB
 */
#ifndef VA
#define VA 2
#endif

#ifdef NO_UV
#define MAXN 128
__local__ float64 w[MAXN * VA];
#else
#define MAXN 64
__local__ float64 w[MAXN * VA];
__local__ float64 v[MAXN];
#endif

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

#ifndef NO_UV
static inline float64 reduce_min_64(float64 p)
{
    p = v_f32_min_b(p, v_f32_mov_dual_group_all_b(p, 0xFFFFFFFF, 1, 0, 3, 2, SW_MDG_ALL, p));
    p = v_f32_min_b(p, v_f32_mov_dual_group_all_b(p, 0xFFFFFFFF, 2, 3, 0, 1, SW_MDG_ALL, p));
    p = v_f32_min_b(p, as_float64(v_element_shift_xor_i32(as_int64(p), 8)));
    p = v_f32_min_b(p, as_float64(v_element_shift_xor_i32(as_int64(p), 4)));
    p = v_f32_min_b(p, as_float64(v_element_shift_xor_i32(as_int64(p), 2)));
    p = v_f32_min_b(p, as_float64(v_element_shift_xor_i32(as_int64(p), 1)));
    return p;
}
#endif

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

// LUT-free 1/sqrt(x), x >= 0 normal (0 -> 1): bit guess + 2 Newton steps + 1 residual-form Newton step (unbiased)
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

static inline void rotate_pair(int p, int q)
{
    float64 up[VA], uq[VA];
    float64 a = 0.0f, b = 0.0f, g = 0.0f;
#pragma unroll
    for (int iv = 0; iv < VA; iv++)
    {
        up[iv] = w[p * VA + iv];
        uq[iv] = w[q * VA + iv];
        a = v_f32_mac_b(up[iv], up[iv], a);
        b = v_f32_mac_b(uq[iv], uq[iv], b);
        g = v_f32_mac_b(up[iv], uq[iv], g);
    }
    float64 alpha = reduce_add_64(a);
    float64 beta = reduce_add_64(b);
    float64 gamma = reduce_add_64(g);

    float64 ab = v_f32_add_b(alpha, beta);
    float64 k = rsqrt_fine(v_f32_sel_eq_f32_b(ab, 0.0f, 1.0f, ab));
    k = v_f32_mul_b(k, k);
    float64 a_ = v_f32_mul_b(alpha, k);
    float64 b_ = v_f32_mul_b(beta, k);
    float64 g_ = v_f32_mul_b(gamma, k);
    float64 d = v_f32_sub_b(b_, a_);
    float64 g2 = v_f32_add_b(g_, g_);
    float64 r2 = v_f32_mac_b(g2, g2, v_f32_mul_b(d, d));
    float64 inv_r = rsqrt_fine(v_f32_sel_less_f32_b(r2, 1.0e-34f, 1.0f, r2));
    float64 c2 = v_f32_mul_b(0.5f, v_f32_mac_b(v_f32_abs_b(d), inv_r, 1.0f));
    float64 inv_c = rsqrt_fine(c2);
    float64 c = v_f32_mul_b(c2, inv_c);
    float64 s = v_f32_mul_b(v_f32_mul_b(0.5f, g2), v_f32_mul_b(inv_r, inv_c));
    s = v_f32_sel_less_f32_b(d, 0.0f, v_f32_sub_b(0.0f, s), s);
    // Equal norms: exact 45 degrees so identical rows cancel to exact 0
    c = v_f32_sel_eq_f32_b(d, 0.0f, 0.70710677f, c);
    s = v_f32_sel_eq_f32_b(d, 0.0f, v_f32_sel_less_f32_b(g2, 0.0f, -0.70710677f, 0.70710677f), s);

    float64 gsq = v_f32_mul_b(g_, g_);
    float64 tol = v_f32_mul_b(1.0e-14f, v_f32_mul_b(a_, b_));
    c = v_f32_sel_less_f32_b(tol, gsq, c, 1.0f);
    s = v_f32_sel_less_f32_b(tol, gsq, s, 0.0f);
    c = v_f32_sel_less_f32_b(r2, 1.0e-34f, 1.0f, c);
    s = v_f32_sel_less_f32_b(r2, 1.0e-34f, 0.0f, s);
    c = v_f32_sel_eq_f32_b(gamma, 0.0f, 1.0f, c);
    s = v_f32_sel_eq_f32_b(gamma, 0.0f, 0.0f, s);

#pragma unroll
    for (int iv = 0; iv < VA; iv++)
    {
        w[p * VA + iv] = v_f32_sub_b(v_f32_mul_b(up[iv], c), v_f32_mul_b(uq[iv], s));
        w[q * VA + iv] = v_f32_mac_b(up[iv], s, v_f32_mul_b(uq[iv], c));
    }
#ifndef NO_UV
    float64 vp = v[p], vq = v[q];
    v[p] = v_f32_sub_b(v_f32_mul_b(vp, c), v_f32_mul_b(vq, s));
    v[q] = v_f32_mac_b(vp, s, v_f32_mul_b(vq, c));
#endif
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
void main(tensor A_t, tensor S)
#else
void main(tensor A_t, tensor Ut, tensor S, tensor Vt)
#endif
{
    const int m = get_dim_size(A_t, 0);
    const int n = get_dim_size(A_t, 1);
    const uint64 lane = read_lane_id_4b_b();
#ifndef NO_UV
    const float64 lanef = v_convert_i32_to_f32_b(as_int64(lane), 0);
#endif
    const int5 off = get_index_space_offset();
    const int5 size = get_index_space_size();
    const int n_even = (n + 1) & ~1;
    const int rr_m = n_even - 1;
    const int half_n = n_even >> 1;
    const int k_start = n & 1;
    const int sweeps = (n <= 4) ? 6 : (n <= 16) ? 10 : (n <= 32) ? 12 : (n <= 64) ? 18 : 20;

    for (int bt = off[0]; bt < off[0] + size[0]; bt++)
    {
        float64 sigma_loc = 0.0f;
        for (int j = 0; j < n; j++)
        {
#pragma unroll
            for (int iv = 0; iv < VA; iv++)
            {
                float64 wj = 0.0f;
                if (64 * iv < m)
                {
                    int5 c = {64 * iv, j, bt, 0, 0};
                    wj = v_f32_sel_geq_u32_b(lane + (unsigned)(64 * iv), (unsigned)m, 0.0f, v_f32_ld_tnsr_b(c, A_t));
                }
                w[j * VA + iv] = wj;
                sigma_loc = v_f32_max_b(sigma_loc, v_f32_abs_b(wj));
            }
#ifndef NO_UV
            v[j] = v_f32_sel_eq_u32_b(lane, (unsigned)j, 1.0f, 0.0f);
#endif
        }
        float64 sigma = reduce_max_64(sigma_loc);
        float64 pre_scale = v_f32_sel_less_f32_b(1.0e35f, sigma, 1.0e-4f, 1.0f);
        float64 safe_sigma = v_f32_sel_eq_f32_b(sigma, 0.0f, 1.0f, sigma);
        float64 inv_sigma = lut_free_recip(v_f32_mul_b(safe_sigma, pre_scale));
        for (int j = 0; j < n; j++)
        {
#pragma unroll
            for (int iv = 0; iv < VA; iv++)
            {
                w[j * VA + iv] = v_f32_mul_b(v_f32_mul_b(w[j * VA + iv], pre_scale), inv_sigma);
            }
        }

        for (int sweep = 0; sweep < sweeps; sweep++)
        {
            for (int r = 0; r < rr_m; r++)
            {
                for (int k = k_start; k < half_n; k++)
                {
                    int p, q;
                    RR_PAIR(r, k, rr_m, p, q);
                    rotate_pair(p, q);
                }
            }
        }

#ifdef NO_UV
        float64 s_vec0 = 0.0f;
        float64 s_vec1 = 0.0f;
        for (int j = 0; j < n; j++)
        {
            float64 mx_loc = 0.0f;
#pragma unroll
            for (int iv = 0; iv < VA; iv++)
            {
                mx_loc = v_f32_max_b(mx_loc, v_f32_abs_b(w[j * VA + iv]));
            }
            float64 mx = reduce_max_64(mx_loc);
            float64 smx = v_f32_sel_eq_f32_b(mx, 0.0f, 1.0f, mx);
            float64 rmx = lut_free_recip(smx);
            float64 sq_loc = 0.0f;
#pragma unroll
            for (int iv = 0; iv < VA; iv++)
            {
                float64 cw = v_f32_mul_b(w[j * VA + iv], rmx);
                sq_loc = v_f32_mac_b(cw, cw, sq_loc);
            }
            float64 sq = reduce_add_64(sq_loc);
            float64 rn = rsqrt_fine(v_f32_sel_less_f32_b(sq, 1.0e-36f, 1.0f, sq));
            float64 nrm = v_f32_mul_b(v_f32_mul_b(sq, rn), smx);
            float64 sj = v_f32_sel_eq_f32_b(mx, 0.0f, 0.0f, v_f32_mul_b(nrm, safe_sigma));
            if (j < 64)
            {
                s_vec0 = v_f32_sel_eq_u32_b(lane, (unsigned)j, sj, s_vec0);
            }
            else
            {
                s_vec1 = v_f32_sel_eq_u32_b(lane, (unsigned)(j - 64), sj, s_vec1);
            }
        }
        int5 cs0 = {0, bt, 0, 0, 0};
        if (n >= 64)
        {
            v_f32_st_tnsr(cs0, S, s_vec0);
            if (n > 64)
            {
                int5 cs1 = {64, bt, 0, 0, 0};
                if (n == 128)
                {
                    v_f32_st_tnsr(cs1, S, s_vec1);
                }
                else
                {
                    v_f32_st_tnsr_partial(cs1, S, s_vec1, (char)(n - 65), 0);
                }
            }
        }
        else
        {
            v_f32_st_tnsr_partial(cs0, S, s_vec0, (char)(n - 1), 0);
        }
#else
        float64 s_vec = 0.0f;
        float64 nrm_vec = 0.0f;
        float64 max_nrm = 0.0f;
        for (int j = 0; j < n; j++)
        {
            float64 mx_loc = 0.0f;
#pragma unroll
            for (int iv = 0; iv < VA; iv++)
            {
                mx_loc = v_f32_max_b(mx_loc, v_f32_abs_b(w[j * VA + iv]));
            }
            float64 mx = reduce_max_64(mx_loc);
            float64 smx = v_f32_sel_eq_f32_b(mx, 0.0f, 1.0f, mx);
            float64 rmx = lut_free_recip(smx);
            float64 cw[VA];
            float64 sq_loc = 0.0f;
#pragma unroll
            for (int iv = 0; iv < VA; iv++)
            {
                cw[iv] = v_f32_mul_b(w[j * VA + iv], rmx);
                sq_loc = v_f32_mac_b(cw[iv], cw[iv], sq_loc);
            }
            float64 sq = reduce_add_64(sq_loc);
            float64 rn = rsqrt_fine(v_f32_sel_less_f32_b(sq, 1.0e-36f, 1.0f, sq));
            float64 nrm = v_f32_mul_b(v_f32_mul_b(sq, rn), smx);
            float64 zero = v_f32_sel_eq_f32_b(mx, 0.0f, 1.0f, 0.0f);
            nrm = v_f32_sel_eq_f32_b(zero, 1.0f, 0.0f, nrm);
            nrm_vec = v_f32_sel_eq_u32_b(lane, (unsigned)j, nrm, nrm_vec);
            max_nrm = v_f32_max_b(max_nrm, nrm);
            s_vec = v_f32_sel_eq_u32_b(lane, (unsigned)j, v_f32_mul_b(nrm, safe_sigma), s_vec);
            float64 scale_u = v_f32_sel_eq_f32_b(zero, 1.0f, 0.0f, rn);
#pragma unroll
            for (int iv = 0; iv < VA; iv++)
            {
                w[j * VA + iv] = v_f32_mul_b(cw[iv], scale_u);
            }
        }

        // Vectorized 2-pass MGS completion for zero/near-zero rows (LUT-free argmax)
        float64 thresh = v_f32_max_b(1.0e-37f, v_f32_mul_b(1.0e-5f, max_nrm));
        float64 done_vec = v_f32_sel_less_f32_b(nrm_vec, thresh, 0.0f, 1.0f);
        done_vec = v_f32_sel_geq_u32_b(lane, (unsigned)n, 0.0f, done_vec);
        float64 diag_p[VA];
#pragma unroll
        for (int iv = 0; iv < VA; iv++)
        {
            diag_p[iv] = v_f32_sel_less_u32_b(lane + (unsigned)(64 * iv), (unsigned)m, 1.0f, -1.0f);
        }
        for (int k = 0; k < n; k++)
        {
            float64 dk = reduce_add_64(v_f32_sel_eq_u32_b(lane, (unsigned)k, done_vec, 0.0f));
#pragma unroll
            for (int iv = 0; iv < VA; iv++)
            {
                float64 uk = v_f32_sel_geq_u32_b(lane + (unsigned)(64 * iv), (unsigned)m, 0.0f,
                                                 v_f32_mul_b(w[k * VA + iv], dk));
                diag_p[iv] = v_f32_sub_b(diag_p[iv], v_f32_mul_b(uk, uk));
            }
        }

        for (int phase = 0; phase < 2; phase++)
        {
            for (int j = 0; j < n; j++)
            {
                float64 dj = reduce_add_64(v_f32_sel_eq_u32_b(lane, (unsigned)j, done_vec, 0.0f));
                float64 nj = reduce_add_64(v_f32_sel_eq_u32_b(lane, (unsigned)j, nrm_vec, 0.0f));
                float64 act = v_f32_sel_eq_f32_b(dj, 0.0f, 1.0f, 0.0f);
                act = phase == 0 ? v_f32_sel_eq_f32_b(nj, 0.0f, 0.0f, act) : v_f32_sel_eq_f32_b(nj, 0.0f, act, 0.0f);

                float64 dmax = reduce_max_64(diag_p[0]);
#pragma unroll
                for (int iv = 1; iv < VA; iv++)
                {
                    dmax = v_f32_max_b(dmax, reduce_max_64(diag_p[iv]));
                }
                float64 best_id = 1024.0f;
#pragma unroll
                for (int iv = 0; iv < VA; iv++)
                {
                    float64 gid = v_f32_add_b(lanef, (float)(64 * iv));
                    float64 id_v = v_f32_sel_eq_f32_b(diag_p[iv], dmax, gid, 1024.0f);
                    best_id = v_f32_min_b(best_id, reduce_min_64(id_v));
                }

                float64 cand1[VA], cand2[VA];
#pragma unroll
                for (int iv = 0; iv < VA; iv++)
                {
                    float64 gid = v_f32_add_b(lanef, (float)(64 * iv));
                    cand1[iv] = v_f32_sel_geq_u32_b(lane + (unsigned)(64 * iv), (unsigned)m, 0.0f, w[j * VA + iv]);
                    cand2[iv] = v_f32_sel_eq_f32_b(gid, best_id, 1.0f, 0.0f);
                }

                for (int pass = 0; pass < 2; pass++)
                {
                    for (int k = 0; k < n; k++)
                    {
                        float64 dk = reduce_add_64(v_f32_sel_eq_u32_b(lane, (unsigned)k, done_vec, 0.0f));
                        float64 uk[VA];
                        float64 d1_loc = 0.0f, d2_loc = 0.0f;
#pragma unroll
                        for (int iv = 0; iv < VA; iv++)
                        {
                            uk[iv] = v_f32_sel_geq_u32_b(lane + (unsigned)(64 * iv), (unsigned)m, 0.0f,
                                                         v_f32_mul_b(w[k * VA + iv], dk));
                            d1_loc = v_f32_mac_b(cand1[iv], uk[iv], d1_loc);
                            d2_loc = v_f32_mac_b(cand2[iv], uk[iv], d2_loc);
                        }
                        float64 dot1 = reduce_add_64(d1_loc);
                        float64 dot2 = reduce_add_64(d2_loc);
#pragma unroll
                        for (int iv = 0; iv < VA; iv++)
                        {
                            cand1[iv] = v_f32_sub_b(cand1[iv], v_f32_mul_b(dot1, uk[iv]));
                            cand2[iv] = v_f32_sub_b(cand2[iv], v_f32_mul_b(dot2, uk[iv]));
                        }
                    }
                }

                float64 cn1_loc = 0.0f, cn2_loc = 0.0f;
#pragma unroll
                for (int iv = 0; iv < VA; iv++)
                {
                    cn1_loc = v_f32_mac_b(cand1[iv], cand1[iv], cn1_loc);
                    cn2_loc = v_f32_mac_b(cand2[iv], cand2[iv], cn2_loc);
                }
                float64 cn1 = reduce_add_64(cn1_loc);
                float64 cn2 = reduce_add_64(cn2_loc);
                float64 cn_sel = v_f32_sel_less_f32_b(0.25f, cn1, cn1, cn2);
                float64 safe_cn = v_f32_sel_less_f32_b(cn_sel, 1.0e-36f, 1.0f, cn_sel);
                float64 rcn = rsqrt_fine(safe_cn);

#pragma unroll
                for (int iv = 0; iv < VA; iv++)
                {
                    float64 cand = v_f32_sel_less_f32_b(0.25f, cn1, cand1[iv], cand2[iv]);
                    cand = v_f32_mul_b(cand, rcn);
                    w[j * VA + iv] = v_f32_sel_eq_f32_b(act, 1.0f, cand, w[j * VA + iv]);
                    float64 new_uk = v_f32_sel_eq_f32_b(act, 1.0f, cand, 0.0f);
                    diag_p[iv] = v_f32_sub_b(diag_p[iv], v_f32_mul_b(new_uk, new_uk));
                }
                done_vec = v_f32_sel_eq_u32_b(lane, (unsigned)j, v_f32_sel_eq_f32_b(act, 1.0f, 1.0f, done_vec),
                                              done_vec);
            }
        }

        int5 cs = {0, bt, 0, 0, 0};
        if (n == 64)
        {
            v_f32_st_tnsr(cs, S, s_vec);
        }
        else
        {
            v_f32_st_tnsr_partial(cs, S, s_vec, (char)(n - 1), 0);
        }
        for (int j = 0; j < n; j++)
        {
#pragma unroll
            for (int iv = 0; iv < VA; iv++)
            {
                if (64 * iv < m)
                {
                    int5 cu = {64 * iv, j, bt, 0, 0};
                    int rem = m - 64 * iv;
                    if (rem >= 64)
                    {
                        v_f32_st_tnsr(cu, Ut, w[j * VA + iv]);
                    }
                    else
                    {
                        v_f32_st_tnsr_partial(cu, Ut, w[j * VA + iv], (char)(rem - 1), 0);
                    }
                }
            }
            int5 cv = {0, j, bt, 0, 0};
            if (n == 64)
            {
                v_f32_st_tnsr(cv, Vt, v[j]);
            }
            else
            {
                v_f32_st_tnsr_partial(cv, Vt, v[j], (char)(n - 1), 0);
            }
        }
#endif
    }
}
