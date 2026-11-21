/* Shared by the complex Jacobi kernels (jacobi_ceigh_f32_gaudi2.c, jacobi_csvd_f32_gaudi2.c): half-vector packing
 * (two matrices per vector), LUT-free reciprocal and rsqrt, and the complex one-sided rotation crot of rows p, q of
 * the __local__ planes ur / ui (and, with APPLY_V, the same rotation of vr / vi), which the including file declares.
 * SVD_TOL adds the SVD's skip rule. tests/kernel_replica.py (_crot_coeffs, _capply) replays it.
 */
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

static inline float64 pack_halves(float64 lo, float64 hi_)
{
    float64 p = v_f32_mov_dual_group_b(hi_, 0xFFFFFFFF, 0, 2, MkWr(1, 1), lo);
    return v_f32_mov_dual_group_b(hi_, 0xFFFFFFFF, 1, 3, MkWr(1, 1), p);
}

static inline float64 unpack_hi_half(float64 p)
{
    float64 z = 0.0f;
    float64 out = v_f32_mov_dual_group_b(p, 0xFFFFFFFF, 2, 0, MkWr(1, 1), z);
    return v_f32_mov_dual_group_b(p, 0xFFFFFFFF, 3, 1, MkWr(1, 1), out);
}

// element i of each half (lane i for lanes 0-31, lane 32 + i for lanes 32-63)
static inline float64 bcast_half(float64 v, int i, uint64 lane)
{
    return v_f32_sel_less_u32_b(lane, 32u, as_float64(v_broadcast_element_i32(as_int64(v), i)),
                                as_float64(v_broadcast_element_i32(as_int64(v), i + 32)));
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

// 1/sqrt(x), x >= 0 normal (0 -> 1): the bit guess, two Newton steps, one in residual form (unbiased: rsqrt(1) = 1)
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

static inline float64 pow2_scale(float64 m)
{
    uint64 e = v_u32_shr_b(v_u32_and_b(as_uint64(m), 0x7F800000u), 23);
    return as_float64(v_u32_sel_eq_u32_b(e, 0u, 0x3F800000u, v_u32_shl_b(254u - e, 23)));
}

static inline float64 pow2_unscale(float64 m)
{
    uint64 e = v_u32_and_b(as_uint64(m), 0x7F800000u);
    return as_float64(v_u32_sel_eq_u32_b(e, 0u, 0x3F800000u, e));
}

static inline float64 nonfinite(float64 x, float64 bad)
{
    return v_f32_sel_eq_u32_b(v_u32_and_b(as_uint64(x), 0x7F800000u), 0x7F800000u, 1.0f, bad);
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

// one complex one-sided rotation of rows p, q of (ur, ui): p' = c p - s e q, q' = s p + c e q
#ifdef APPLY_V
static inline void crot(int p, int q, int apply_v)
#else
static inline void crot(int p, int q)
#endif
{
    float64 upr = ur[p], upi = ui[p], uqr = ur[q], uqi = ui[q];
    float64 alpha = reduce_add_half32(v_f32_mac_b(upi, upi, v_f32_mul_b(upr, upr)));
    float64 beta = reduce_add_half32(v_f32_mac_b(uqi, uqi, v_f32_mul_b(uqr, uqr)));
    float64 gr = reduce_add_half32(v_f32_mac_b(upi, uqi, v_f32_mul_b(upr, uqr)));
    float64 gi = reduce_add_half32(v_f32_mac_b(v_f32_sub_b(0.0f, upr), uqi, v_f32_mul_b(upi, uqr)));
    // |g| and e = g / |g| from g scaled into [1, 2) by a power of two (a tiny part would flush when squared)
    float64 gm = v_f32_max_b(v_f32_abs_b(gr), v_f32_abs_b(gi));
    float64 t = pow2_scale(gm);
    float64 grt = v_f32_mul_b(gr, t);
    float64 git = v_f32_mul_b(gi, t);
    float64 n2 = v_f32_mac_b(git, git, v_f32_mul_b(grt, grt));
    float64 rn = rsqrt_fine(n2);
    float64 ag = v_f32_mul_b(v_f32_mul_b(n2, rn), pow2_unscale(gm));
    float64 er = v_f32_mul_b(grt, rn);
    float64 ei = v_f32_mul_b(git, rn);
    float64 g2s = v_f32_mul_b(ag, ag);
    float64 d = v_f32_sub_b(beta, alpha);
    float64 g2 = v_f32_add_b(ag, ag);
    float64 r2 = v_f32_mac_b(g2, g2, v_f32_mul_b(d, d));
    float64 inv_r = rsqrt_fine(r2);
    float64 c2 = v_f32_mul_b(0.5f, v_f32_mac_b(v_f32_abs_b(d), inv_r, 1.0f));
    float64 inv_c = rsqrt_fine(c2);
    float64 c = v_f32_mul_b(c2, inv_c);
    float64 s = v_f32_mul_b(ag, v_f32_mul_b(inv_r, inv_c));
    s = v_f32_sel_less_f32_b(d, 0.0f, v_f32_sub_b(0.0f, s), s);
#ifdef SVD_TOL
    // the SVD's rule (jacobi_svd_f32_gaudi2.c): skip unless 1e-14 alpha beta < |g|^2, and when r2 < 1e-34
    float64 keep = v_f32_sel_less_f32_b(v_f32_mul_b(1e-14f, v_f32_mul_b(alpha, beta)), g2s, 1.0f, 0.0f);
    keep = v_f32_sel_less_f32_b(r2, 1e-34f, 0.0f, keep);
#else
    float64 keep = 1.0f;
#endif
    keep = v_f32_sel_eq_f32_b(g2s, 0.0f, 0.0f, keep);
    keep = v_f32_sel_eq_f32_b(r2, 0.0f, 0.0f, keep);
    c = v_f32_sel_eq_f32_b(keep, 0.0f, 1.0f, c);
    s = v_f32_sel_eq_f32_b(keep, 0.0f, 0.0f, s);
    er = v_f32_sel_eq_f32_b(keep, 0.0f, 1.0f, er);
    ei = v_f32_sel_eq_f32_b(keep, 0.0f, 0.0f, ei);
    float64 cer = v_f32_mul_b(c, er), cei = v_f32_mul_b(c, ei);
    float64 ser = v_f32_mul_b(s, er), sei = v_f32_mul_b(s, ei);
    float64 nser = v_f32_sub_b(0.0f, ser), nsei = v_f32_sub_b(0.0f, sei), ncei = v_f32_sub_b(0.0f, cei);
    ur[p] = v_f32_mac_b(uqi, sei, v_f32_mac_b(uqr, nser, v_f32_mul_b(upr, c)));
    ui[p] = v_f32_mac_b(uqr, nsei, v_f32_mac_b(uqi, nser, v_f32_mul_b(upi, c)));
    ur[q] = v_f32_mac_b(uqi, ncei, v_f32_mac_b(uqr, cer, v_f32_mul_b(upr, s)));
    ui[q] = v_f32_mac_b(uqr, cei, v_f32_mac_b(uqi, cer, v_f32_mul_b(upi, s)));
#ifdef APPLY_V
    if (apply_v)
    {
        float64 vpr = vr[p], vpi = vi[p], vqr = vr[q], vqi = vi[q];
        vr[p] = v_f32_mac_b(vqi, sei, v_f32_mac_b(vqr, nser, v_f32_mul_b(vpr, c)));
        vi[p] = v_f32_mac_b(vqr, nsei, v_f32_mac_b(vqi, nser, v_f32_mul_b(vpi, c)));
        vr[q] = v_f32_mac_b(vqi, ncei, v_f32_mac_b(vqr, cer, v_f32_mul_b(vpr, s)));
        vi[q] = v_f32_mac_b(vqr, cei, v_f32_mac_b(vqi, cer, v_f32_mul_b(vpi, s)));
    }
#endif
}

#ifdef APPLY_V
static inline void run_sweeps(int m, int half_n, int k_start, int sweeps, int apply_v)
#else
static inline void run_sweeps(int m, int half_n, int k_start, int sweeps)
#endif
{
    for (int sweep = 0; sweep < sweeps; sweep++)
    {
        for (int r = 0; r < m; r++)
        {
            for (int k = k_start; k < half_n; k++)
            {
                int p, q;
                RR_PAIR(r, k, m, p, q);
#ifdef APPLY_V
                crot(p, q, apply_v);
#else
                crot(p, q);
#endif
            }
        }
    }
}

static inline void normalize_rows(int n)
{
    for (int j = 0; j < n; j++)
    {
        float64 xr = ur[j], xi = ui[j];
        float64 rn = rsqrt_fine(reduce_add_half32(v_f32_mac_b(xi, xi, v_f32_mul_b(xr, xr))));
        ur[j] = v_f32_mul_b(xr, rn);
        ui[j] = v_f32_mul_b(xi, rn);
    }
}

static inline float64 load_packed(tensor T, int row, int bt, int has2, int n, uint64 lane)
{
    int5 c0 = {0, row, bt, 0, 0};
    float64 r0 = v_f32_sel_geq_u32_b(lane, (unsigned)n, 0.0f, v_f32_ld_tnsr_b(c0, T));
    float64 r1 = 0.0f;
    if (has2)
    {
        int5 c1 = {0, row, bt + 1, 0, 0};
        r1 = v_f32_sel_geq_u32_b(lane, (unsigned)n, 0.0f, v_f32_ld_tnsr_b(c1, T));
    }
    return pack_halves(r0, r1);
}

static inline void store_packed(tensor T, int row, int bt, int has2, int n, float64 v)
{
    int5 c0 = {0, row, bt, 0, 0};
    v_f32_st_tnsr_partial(c0, T, v, n - 1, 0);
    if (has2)
    {
        int5 c1 = {0, row, bt + 1, 0, 0};
        v_f32_st_tnsr_partial(c1, T, unpack_hi_half(v), n - 1, 0);
    }
}

// the real eigh kernel's prescale (inv = 0) or inv_prescale (inv = 1) from max|a| (per half)
static inline float64 prescale_of(float64 max_abs, int inv)
{
    float64 pre = v_f32_sel_grt_f32_b(max_abs, 1e33f, inv ? 0x1p+20f : 0x1p-20f, 1.0f);
    float64 sub_pre = v_f32_sel_eq_f32_b(max_abs, 0.0f, pre, inv ? 0x1p-40f : 0x1p+40f);
    return v_f32_sel_less_f32_b(max_abs, 1e-20f, sub_pre, pre);
}

static inline float64 safe_inv(float64 sigma, float hi_)
{
    float64 safe = v_f32_sel_eq_f32_b(sigma, 0.0f, 1.0f, sigma);
    safe = v_f32_sel_grt_f32_b(safe, hi_, 1.0f, safe);
    safe = v_f32_sel_less_f32_b(safe, 1e-37f, 1.0f, safe);
    return lut_free_recip(safe);
}

