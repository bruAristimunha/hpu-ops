/* Batched complex64 general eigendecomposition on Gaudi2 TPC (MPS recipe):
 *   1. Entry non-finite check (NaN/Inf -> NaN everywhere), zero fast-path (W=0, V=I), and exact power-of-2 prescaling
 *      to [1, 2) (eigenvalues unscaled at exit).
 *   2. Parlett-Reinsch radix-2 diagonal balancing (up to 16 passes, branchless binary-power scaling).
 *   3. Complex Householder Hessenberg reduction (H row-major in hr/hi; Q column-major in qr/qi).
 *   4. Explicit complex single-shift QR (Wilkinson shift with LUT-free c_sqrt, exceptional shift every 10 stalled
 *      iterations, Ahues-Kressner/LAPACK deflation, complex Givens clartg, 30 * max(10, n) cap).
 *   5. Stage 4 (eig only): overflow-guarded triangular back-substitution on T, v_k = Q[:, 0:k+1] @ y_k,
 *      inverse balance d_scale, unit 2-norm normalization, transpose column-major V to row-major V_out.
 *
 * Tensors (TPC geometry, dim0 innermost; torch [batch, rows, cols] row-major), planar (rows 0..n-1 re, n..2n-1 im):
 *   ceig_f32_gaudi2:     A [n, 2n, batch] -> W [n, 2, batch], V [n, 2n, batch], S [2, batch] (info, iters)
 *   ceigvals_f32_gaudi2: A [n, 2n, batch] -> W [n, 2, batch], S [2, batch] (-DNO_VECTORS)
 *
 * 100% LUT-free (no v_reciprocal_f32, v_rsqrt_f32, v_sqrt_f32, v_div_f32, v_f32_reduce_*, or float `/`),
 * keeping VLM in 80 KB mode: 6 * 32 * 256 B = 48 KB for ceig; 2 * 32 * 256 B = 16 KB for ceigvals.
 */
#define MAXN 32

__local__ float64 hr[MAXN];
__local__ float64 hi[MAXN];
#ifndef NO_VECTORS
__local__ float64 qr[MAXN];
__local__ float64 qi[MAXN];
__local__ float64 vr[MAXN];
__local__ float64 vi[MAXN];
#endif

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

static inline float64 c_abs1(float64 re, float64 im)
{
    return v_f32_add_b(v_f32_abs_b(re), v_f32_abs_b(im));
}

typedef struct { float64 r; float64 i; } cvec_t;
typedef struct { float64 cs; float64 snr; float64 sni; float64 rr; float64 ri; } givens_t;

static inline cvec_t c_div(float64 ar, float64 ai, float64 br, float64 bi)
{
    float64 bm = v_f32_max_b(v_f32_abs_b(br), v_f32_abs_b(bi));
    float64 am = v_f32_max_b(v_f32_abs_b(ar), v_f32_abs_b(ai));
    float64 tb = pow2_scale(bm);
    float64 btr = v_f32_mul_b(br, tb);
    float64 bti = v_f32_mul_b(bi, tb);
    float64 den = v_f32_mac_b(bti, bti, v_f32_mul_b(btr, btr));
    float64 inv_den = lut_free_recip(den);
    float64 ta = pow2_scale(am);
    float64 atr = v_f32_mul_b(ar, ta);
    float64 ati = v_f32_mul_b(ai, ta);
    float64 nr = v_f32_mac_b(ati, bti, v_f32_mul_b(atr, btr));
    float64 ni = v_f32_mac_b(v_f32_sub_b(0.0f, atr), bti, v_f32_mul_b(ati, btr));
    float64 qr = v_f32_mul_b(nr, inv_den);
    float64 qi = v_f32_mul_b(ni, inv_den);

    int64 ea = v_i32_min_b(253, v_i32_and_b(v_i32_shr_b(as_int64(am), 23), 0xFF));
    int64 eb = v_i32_min_b(253, v_i32_and_b(v_i32_shr_b(as_int64(bm), 23), 0xFF));
    int64 ediff = v_i32_sub_b(ea, eb);
    int64 exp_biased = v_i32_add_b(127, ediff);
    int64 e1 = v_i32_shr_b(ediff, 1);
    int64 e2 = v_i32_sub_b(ediff, e1);
    int64 b1 = v_i32_min_b(254, v_i32_max_b(1, v_i32_add_b(127, e1)));
    int64 b2 = v_i32_min_b(254, v_i32_max_b(1, v_i32_add_b(127, e2)));
    float64 s1 = as_float64(v_u32_shl_b(as_uint64(b1), 23));
    float64 s2 = as_float64(v_u32_shl_b(as_uint64(b2), 23));
    int64 exp_clamped = v_i32_min_b(254, v_i32_max_b(1, exp_biased));
    float64 scale_q = as_float64(v_u32_shl_b(as_uint64(exp_clamped), 23));
    float64 q_norm_r = v_f32_mul_b(qr, scale_q);
    float64 q_norm_i = v_f32_mul_b(qi, scale_q);
    float64 q_hi_r = v_f32_mul_b(v_f32_mul_b(qr, s1), s2);
    float64 q_hi_i = v_f32_mul_b(v_f32_mul_b(qi, s1), s2);
    float64 zr = v_f32_sel_geq_i32_b(exp_biased, 255, q_hi_r, q_norm_r);
    float64 zi = v_f32_sel_geq_i32_b(exp_biased, 255, q_hi_i, q_norm_i);
    zr = v_f32_sel_leq_i32_b(exp_biased, 0, 0.0f, zr);
    zi = v_f32_sel_leq_i32_b(exp_biased, 0, 0.0f, zi);
    float64 zero_ab = v_f32_sel_eq_f32_b(am, 0.0f, 1.0f, v_f32_sel_eq_f32_b(bm, 0.0f, 1.0f, 0.0f));
    cvec_t out;
    out.r = v_f32_sel_eq_f32_b(zero_ab, 1.0f, 0.0f, zr);
    out.i = v_f32_sel_eq_f32_b(zero_ab, 1.0f, 0.0f, zi);
    return out;
}

static inline cvec_t c_sqrt(float64 zr, float64 zi)
{
    float64 m = v_f32_max_b(v_f32_abs_b(zr), v_f32_abs_b(zi));
    uint64 e = v_u32_shr_b(v_u32_and_b(as_uint64(m), 0x7F800000u), 23);
    uint64 e_safe = v_u32_sel_eq_u32_b(e, 0u, 127u, e);
    uint64 e_even = e_safe - ((e_safe & 1u) ^ 1u);
    float64 sc = as_float64(v_u32_shl_b(254u - e_even, 23));
    float64 un_sqrt = as_float64(v_u32_shl_b((e_even + 127u) >> 1, 23));
    float64 u = v_f32_mul_b(zr, sc);
    float64 v = v_f32_mul_b(zi, sc);
    float64 n2 = v_f32_mac_b(v, v, v_f32_mul_b(u, u));
    float64 r = v_f32_mul_b(n2, rsqrt_fine(n2));
    float64 abs_u = v_f32_abs_b(u);
    float64 s2 = v_f32_mul_b(v_f32_add_b(r, abs_u), 0.5f);
    float64 s = v_f32_mul_b(s2, rsqrt_fine(s2));
    float64 inv_2s = lut_free_recip(v_f32_mul_b(2.0f, s));
    float64 q = v_f32_mul_b(v, inv_2s);
    float64 s_signed = v_f32_sel_less_f32_b(v, 0.0f, v_f32_sub_b(0.0f, s), s);
    float64 q_signed = v_f32_sel_less_f32_b(v, 0.0f, v_f32_sub_b(0.0f, q), q);
    float64 wr = v_f32_sel_geq_f32_b(u, 0.0f, s, q_signed);
    float64 wi = v_f32_sel_geq_f32_b(u, 0.0f, q, s_signed);
    cvec_t out;
    out.r = v_f32_sel_eq_f32_b(m, 0.0f, 0.0f, v_f32_mul_b(wr, un_sqrt));
    out.i = v_f32_sel_eq_f32_b(m, 0.0f, 0.0f, v_f32_mul_b(wi, un_sqrt));
    return out;
}

static inline givens_t clartg(float64 fr, float64 fi, float64 gr, float64 gi)
{
    float64 fm = v_f32_max_b(v_f32_abs_b(fr), v_f32_abs_b(fi));
    float64 gm = v_f32_max_b(v_f32_abs_b(gr), v_f32_abs_b(gi));
    float64 mx = v_f32_max_b(fm, gm);
    float64 t = pow2_scale(mx);
    float64 un_t = pow2_unscale(mx);
    float64 fsr = v_f32_mul_b(fr, t), fsi = v_f32_mul_b(fi, t);
    float64 gsr = v_f32_mul_b(gr, t), gsi = v_f32_mul_b(gi, t);
    float64 f2 = v_f32_mac_b(fsi, fsi, v_f32_mul_b(fsr, fsr));
    float64 g2 = v_f32_mac_b(gsi, gsi, v_f32_mul_b(gsr, gsr));
    float64 fg2 = v_f32_add_b(f2, g2);
    float64 inv_fg = rsqrt_fine(fg2);
    float64 inv_f = v_f32_sel_grt_f32_b(f2, 0.0f, rsqrt_fine(f2), 0.0f);
    float64 f_abs_s = v_f32_mul_b(f2, inv_f);
    float64 cs = v_f32_mul_b(f_abs_s, inv_fg);
    float64 pfr = v_f32_mul_b(fsr, inv_f), pfi = v_f32_mul_b(fsi, inv_f);
    float64 snr = v_f32_mac_b(pfi, gsi, v_f32_mul_b(pfr, gsr));
    float64 sni = v_f32_mac_b(v_f32_sub_b(0.0f, pfr), gsi, v_f32_mul_b(pfi, gsr));
    snr = v_f32_mul_b(snr, inv_fg);
    sni = v_f32_mul_b(sni, inv_fg);
    float64 r_abs = v_f32_mul_b(v_f32_mul_b(fg2, inv_fg), un_t);
    float64 rr = v_f32_mul_b(pfr, r_abs);
    float64 ri = v_f32_mul_b(pfi, r_abs);

    float64 g_abs = v_f32_mul_b(v_f32_mul_b(g2, inv_fg), un_t);
    float64 f0_snr = v_f32_mul_b(gsr, inv_fg);
    float64 f0_sni = v_f32_mul_b(v_f32_sub_b(0.0f, gsi), inv_fg);
    cs = v_f32_sel_eq_f32_b(f2, 0.0f, 0.0f, cs);
    snr = v_f32_sel_eq_f32_b(f2, 0.0f, f0_snr, snr);
    sni = v_f32_sel_eq_f32_b(f2, 0.0f, f0_sni, sni);
    rr = v_f32_sel_eq_f32_b(f2, 0.0f, g_abs, rr);
    ri = v_f32_sel_eq_f32_b(f2, 0.0f, 0.0f, ri);

    givens_t out;
    out.cs = v_f32_sel_eq_f32_b(g2, 0.0f, 1.0f, cs);
    out.snr = v_f32_sel_eq_f32_b(g2, 0.0f, 0.0f, snr);
    out.sni = v_f32_sel_eq_f32_b(g2, 0.0f, 0.0f, sni);
    out.rr = v_f32_sel_eq_f32_b(g2, 0.0f, fr, rr);
    out.ri = v_f32_sel_eq_f32_b(g2, 0.0f, fi, ri);
    return out;
}

// Zero scalar reads (read_vpu_scalar removed): Stage 3 uses vector-predicated QR.

#ifdef NO_VECTORS
void main(tensor A, tensor W, tensor S)
#else
void main(tensor A, tensor W, tensor V, tensor S)
#endif
{
    const int n = get_dim_size(A, 0);
    const uint64 lane = read_lane_id_4b_b();
    const int5 off = get_index_space_offset();
    const int5 size = get_index_space_size();
    const uint64 nan_bits = 0x7FC00000u;
    const float64 nanv = as_float64(nan_bits);
    const float eps32 = 0x1p-23f;
    const float smlnum = 0x1p-103f;  // 2^-126 / 2^-23
    const float sml_base = (float)n * smlnum;
    const float sml = (sml_base > smlnum) ? sml_base : smlnum;

    for (int bt = off[0]; bt < off[0] + size[0]; bt++)
    {
        float64 bad = 0.0f;
        float64 max_abs = 0.0f;
        for (int i = 0; i < n; i++)
        {
            int5 cr = {0, i, bt, 0, 0};
            int5 ci = {0, n + i, bt, 0, 0};
            float64 xr = v_f32_sel_geq_u32_b(lane, (unsigned)n, 0.0f, v_f32_ld_tnsr_b(cr, A));
            float64 xi = v_f32_sel_geq_u32_b(lane, (unsigned)n, 0.0f, v_f32_ld_tnsr_b(ci, A));
            hr[i] = xr;
            hi[i] = xi;
            bad = nonfinite(xi, nonfinite(xr, bad));
            max_abs = v_f32_max_b(max_abs, v_f32_max_b(v_f32_abs_b(xr), v_f32_abs_b(xi)));
        }
        bad = reduce_max_half32(bad);
        max_abs = reduce_max_half32(max_abs);
        float64 skip_calc = v_f32_sel_grt_f32_b(bad, 0.0f, 1.0f, v_f32_sel_eq_f32_b(max_abs, 0.0f, 1.0f, 0.0f));

        // Exact power-of-2 prescaling to [1, 2) (zero out hr/hi if bad or zero so Stage 1-4 stay finite & fast)
        float64 sc_entry = v_f32_sel_eq_f32_b(skip_calc, 1.0f, 1.0f, pow2_scale(max_abs));
        float64 unsc_entry = v_f32_sel_eq_f32_b(skip_calc, 1.0f, 1.0f, pow2_unscale(max_abs));
        for (int i = 0; i < n; i++)
        {
            float64 zr = v_f32_sel_eq_f32_b(skip_calc, 1.0f, 0.0f, hr[i]);
            float64 zi = v_f32_sel_eq_f32_b(skip_calc, 1.0f, 0.0f, hi[i]);
            hr[i] = v_f32_mul_b(zr, sc_entry);
            hi[i] = v_f32_mul_b(zi, sc_entry);
        }

        // Stage 1: Parlett-Reinsch radix-2 diagonal balancing (16 sweeps, branchless binary-power steps)
#ifndef NO_VECTORS
        float64 d_scale = 1.0f;
#endif
        const float sfmin2 = 0x1p-101f;
        const float sfmax2 = 0x1p+101f;
        const float p2s[8] = {0x1p+64f, 0x1p+32f, 0x1p+16f, 0x1p+8f, 0x1p+4f, 0x1p+2f, 0x1p+1f, 0x1p+1f};
        const float ip2s[8] = {0x1p-64f, 0x1p-32f, 0x1p-16f, 0x1p-8f, 0x1p-4f, 0x1p-2f, 0x1p-1f, 0x1p-1f};
        for (int pass = 0; pass < 16; pass++)
        {
            for (int i = 0; i < n; i++)
            {
                float64 row_a = c_abs1(hr[i], hi[i]);
                row_a = v_f32_sel_eq_u32_b(lane, (unsigned)i, 0.0f, row_a);
                float64 r = reduce_add_half32(row_a);
                float64 ra = reduce_max_half32(row_a);

                float64 col_a = 0.0f;
                for (int j = 0; j < n; j++)
                {
                    if (j != i)
                    {
                        float64 crj = lane_bcast(hr[j], i);
                        float64 cij = lane_bcast(hi[j], i);
                        col_a = v_f32_sel_eq_u32_b(lane, (unsigned)j, c_abs1(crj, cij), col_a);
                    }
                }
                float64 c = reduce_add_half32(col_a);
                float64 ca = reduce_max_half32(col_a);

                float64 nz = v_f32_sel_grt_f32_b(c, 0.0f, v_f32_sel_grt_f32_b(r, 0.0f, 1.0f, 0.0f), 0.0f);
                float64 s0 = v_f32_add_b(c, r);
                float64 f = 1.0f;
                float64 inv_f = 1.0f;
                float64 g = v_f32_mul_b(r, 0.5f);
                for (int st = 0; st < 8; st++)
                {
                    float p = p2s[st];
                    float ip = ip2s[st];
                    // Test whether scaling by (p, ip) keeps c < g (for st < 7 we test post-step c*p < g*ip; for st == 7 we test pre-step c < g)
                    float64 c_test = (st < 7) ? v_f32_mul_b(c, p) : c;
                    float64 g_test = (st < 7) ? v_f32_mul_b(g, ip) : g;
                    float64 f_n = v_f32_mul_b(f, p);
                    float64 inv_n = v_f32_mul_b(inv_f, ip);
                    float64 c_n = v_f32_mul_b(c, p);
                    float64 ca_n = v_f32_mul_b(ca, p);
                    float64 r_n = v_f32_mul_b(r, ip);
                    float64 g_n = v_f32_mul_b(g, ip);
                    float64 ra_n = v_f32_mul_b(ra, ip);
                    float64 mx_up = v_f32_max_b(f, v_f32_max_b(c, ca));
                    float64 mn_up = v_f32_min_b(r, v_f32_min_b(g, ra));
                    float64 ok_up = v_f32_sel_less_f32_b(c_test, g_test, 1.0f, 0.0f);
                    ok_up = v_f32_sel_less_f32_b(mx_up, sfmax2, ok_up, 0.0f);
                    ok_up = v_f32_sel_grt_f32_b(mn_up, sfmin2, ok_up, 0.0f);
                    if (st < 7)
                    {
                        ok_up = v_f32_sel_less_f32_b(v_f32_max_b(f_n, v_f32_max_b(c_n, ca_n)), sfmax2, ok_up, 0.0f);
                        ok_up = v_f32_sel_grt_f32_b(v_f32_min_b(r_n, v_f32_min_b(g_n, ra_n)), sfmin2, ok_up, 0.0f);
                    }
                    f = v_f32_sel_eq_f32_b(ok_up, 1.0f, f_n, f);
                    inv_f = v_f32_sel_eq_f32_b(ok_up, 1.0f, inv_n, inv_f);
                    c = v_f32_sel_eq_f32_b(ok_up, 1.0f, c_n, c);
                    ca = v_f32_sel_eq_f32_b(ok_up, 1.0f, ca_n, ca);
                    r = v_f32_sel_eq_f32_b(ok_up, 1.0f, r_n, r);
                    g = v_f32_sel_eq_f32_b(ok_up, 1.0f, g_n, g);
                    ra = v_f32_sel_eq_f32_b(ok_up, 1.0f, ra_n, ra);
                }
                g = v_f32_mul_b(c, 0.5f);
                for (int st = 0; st < 8; st++)
                {
                    float p = p2s[st];
                    float ip = ip2s[st];
                    float64 g_test = (st < 7) ? v_f32_mul_b(g, ip) : g;
                    float64 r_test = (st < 7) ? v_f32_mul_b(r, p) : r;
                    float64 f_n = v_f32_mul_b(f, ip);
                    float64 inv_n = v_f32_mul_b(inv_f, p);
                    float64 r_n = v_f32_mul_b(r, p);
                    float64 ra_n = v_f32_mul_b(ra, p);
                    float64 c_n = v_f32_mul_b(c, ip);
                    float64 g_n = v_f32_mul_b(g, ip);
                    float64 ca_n = v_f32_mul_b(ca, ip);
                    float64 mx_dn = v_f32_max_b(f, v_f32_max_b(r, ra));
                    float64 mn_dn = v_f32_min_b(inv_f, v_f32_min_b(c, v_f32_min_b(g, ca)));
                    float64 ok_dn = v_f32_sel_geq_f32_b(g_test, r_test, 1.0f, 0.0f);
                    ok_dn = v_f32_sel_less_f32_b(mx_dn, sfmax2, ok_dn, 0.0f);
                    ok_dn = v_f32_sel_grt_f32_b(mn_dn, sfmin2, ok_dn, 0.0f);
                    if (st < 7)
                    {
                        ok_dn = v_f32_sel_less_f32_b(v_f32_max_b(f_n, v_f32_max_b(r_n, ra_n)), sfmax2, ok_dn, 0.0f);
                        ok_dn = v_f32_sel_grt_f32_b(v_f32_min_b(inv_n, v_f32_min_b(c_n, v_f32_min_b(g_n, ca_n))), sfmin2, ok_dn, 0.0f);
                    }
                    f = v_f32_sel_eq_f32_b(ok_dn, 1.0f, f_n, f);
                    inv_f = v_f32_sel_eq_f32_b(ok_dn, 1.0f, inv_n, inv_f);
                    r = v_f32_sel_eq_f32_b(ok_dn, 1.0f, r_n, r);
                    ra = v_f32_sel_eq_f32_b(ok_dn, 1.0f, ra_n, ra);
                    c = v_f32_sel_eq_f32_b(ok_dn, 1.0f, c_n, c);
                    g = v_f32_sel_eq_f32_b(ok_dn, 1.0f, g_n, g);
                    ca = v_f32_sel_eq_f32_b(ok_dn, 1.0f, ca_n, ca);
                }

                float64 accept = v_f32_sel_less_f32_b(v_f32_add_b(c, r), v_f32_mul_b(0.95f, s0), nz, 0.0f);
                float64 f_eff = v_f32_sel_eq_f32_b(accept, 1.0f, f, 1.0f);
                float64 inv_eff = v_f32_sel_eq_f32_b(accept, 1.0f, inv_f, 1.0f);
#ifndef NO_VECTORS
                d_scale = v_f32_sel_eq_u32_b(lane, (unsigned)i, v_f32_mul_b(d_scale, f_eff), d_scale);
#endif
                hr[i] = v_f32_mul_b(hr[i], inv_eff);
                hi[i] = v_f32_mul_b(hi[i], inv_eff);
                for (int j = 0; j < n; j++)
                {
                    hr[j] = v_f32_sel_eq_u32_b(lane, (unsigned)i, v_f32_mul_b(hr[j], f_eff), hr[j]);
                    hi[j] = v_f32_sel_eq_u32_b(lane, (unsigned)i, v_f32_mul_b(hi[j], f_eff), hi[j]);
                }
            }
        }

        // Stage 2: Complex Householder Hessenberg reduction (Q stored column-major in qr/qi)
#ifndef NO_VECTORS
        for (int j = 0; j < n; j++)
        {
            qr[j] = v_f32_sel_eq_u32_b(lane, (unsigned)j, 1.0f, 0.0f);
            qi[j] = 0.0f;
        }
#endif
        for (int k = 0; k < n - 2; k++)
        {
            float64 xcol_r = 0.0f, xcol_i = 0.0f, tail_nz = 0.0f;
            for (int i = k + 1; i < n; i++)
            {
                float64 er = lane_bcast(hr[i], k);
                float64 ei = lane_bcast(hi[i], k);
                xcol_r = v_f32_sel_eq_u32_b(lane, (unsigned)i, er, xcol_r);
                xcol_i = v_f32_sel_eq_u32_b(lane, (unsigned)i, ei, xcol_i);
                if (i >= k + 2)
                {
                    tail_nz = v_f32_max_b(tail_nz, v_f32_max_b(v_f32_abs_b(er), v_f32_abs_b(ei)));
                }
            }
            float64 mx = reduce_max_half32(v_f32_max_b(v_f32_abs_b(xcol_r), v_f32_abs_b(xcol_i)));
            float64 t = pow2_scale(mx);
            float64 un_t = pow2_unscale(mx);
            float64 sxr = v_f32_mul_b(xcol_r, t);
            float64 sxi = v_f32_mul_b(xcol_i, t);
            float64 sq = reduce_add_half32(v_f32_mac_b(sxi, sxi, v_f32_mul_b(sxr, sxr)));
            float64 xnorm = v_f32_mul_b(v_f32_mul_b(sq, rsqrt_fine(sq)), un_t);
            float64 alr = lane_bcast(hr[k + 1], k);
            float64 ali = lane_bcast(hi[k + 1], k);
            float64 beta = v_f32_sel_geq_f32_b(alr, 0.0f, v_f32_sub_b(0.0f, xnorm), xnorm);
            float64 keep = v_f32_sel_grt_f32_b(tail_nz, 0.0f, v_f32_sel_grt_f32_b(xnorm, 0.0f, 1.0f, 0.0f), 0.0f);
            float64 inv_beta = lut_free_recip(v_f32_sel_eq_f32_b(keep, 1.0f, beta, 1.0f));
            float64 taur = v_f32_sel_eq_f32_b(keep, 1.0f, v_f32_sub_b(1.0f, v_f32_mul_b(alr, inv_beta)), 0.0f);
            float64 taui = v_f32_sel_eq_f32_b(keep, 1.0f, v_f32_sub_b(0.0f, v_f32_mul_b(ali, inv_beta)), 0.0f);

            cvec_t vq = c_div(xcol_r, xcol_i, v_f32_sub_b(alr, beta), ali);
            float64 vr_v = vq.r, vi_v = vq.i;
            vr_v = v_f32_sel_leq_u32_b(lane, (unsigned)k, 0.0f, vr_v);
            vi_v = v_f32_sel_leq_u32_b(lane, (unsigned)(k + 1), 0.0f, vi_v);
            vr_v = v_f32_sel_eq_u32_b(lane, (unsigned)(k + 1), 1.0f, vr_v);
            vr_v = v_f32_sel_geq_u32_b(lane, (unsigned)n, 0.0f, vr_v);
            vi_v = v_f32_sel_geq_u32_b(lane, (unsigned)n, 0.0f, vi_v);
            vr_v = v_f32_sel_eq_f32_b(keep, 1.0f, vr_v, 0.0f);
            vi_v = v_f32_sel_eq_f32_b(keep, 1.0f, vi_v, 0.0f);

            // Left multiply H[k+1:n, k+1:n] <- (I - conj(tau) v v^H) H[k+1:n, k+1:n]
            float64 drr = 0.0f, dri = 0.0f;
            for (int i = k + 1; i < n; i++)
            {
                float64 vri = lane_bcast(vr_v, i);
                float64 cvi = v_f32_sub_b(0.0f, lane_bcast(vi_v, i));
                drr = v_f32_add_b(drr, v_f32_mac_b(v_f32_sub_b(0.0f, cvi), hi[i], v_f32_mul_b(vri, hr[i])));
                dri = v_f32_add_b(dri, v_f32_mac_b(cvi, hr[i], v_f32_mul_b(vri, hi[i])));
            }
            drr = v_f32_sel_leq_u32_b(lane, (unsigned)k, 0.0f, drr);
            dri = v_f32_sel_leq_u32_b(lane, (unsigned)k, 0.0f, dri);
            float64 ctaur = taur, ctaui = v_f32_sub_b(0.0f, taui);
            for (int i = k + 1; i < n; i++)
            {
                float64 vri = lane_bcast(vr_v, i);
                float64 vii = lane_bcast(vi_v, i);
                float64 mr = v_f32_mac_b(v_f32_sub_b(0.0f, ctaui), vii, v_f32_mul_b(ctaur, vri));
                float64 mi = v_f32_mac_b(ctaui, vri, v_f32_mul_b(ctaur, vii));
                float64 nmr = v_f32_sub_b(0.0f, mr);
                hr[i] = v_f32_mac_b(mi, dri, v_f32_mac_b(nmr, drr, hr[i]));
                hi[i] = v_f32_mac_b(v_f32_sub_b(0.0f, mi), drr, v_f32_mac_b(nmr, dri, hi[i]));
            }
            hr[k + 1] = v_f32_sel_eq_u32_b(lane, (unsigned)k, v_f32_sel_eq_f32_b(keep, 1.0f, beta, alr), hr[k + 1]);
            hi[k + 1] = v_f32_sel_eq_u32_b(lane, (unsigned)k, v_f32_sel_eq_f32_b(keep, 1.0f, 0.0f, ali), hi[k + 1]);
            for (int i = k + 2; i < n; i++)
            {
                hr[i] = v_f32_sel_eq_u32_b(lane, (unsigned)k, v_f32_sel_eq_f32_b(keep, 1.0f, 0.0f, hr[i]), hr[i]);
                hi[i] = v_f32_sel_eq_u32_b(lane, (unsigned)k, v_f32_sel_eq_f32_b(keep, 1.0f, 0.0f, hi[i]), hi[i]);
            }

            // Right multiply H[0:n, k+1:n] <- H[0:n, k+1:n] (I - tau v v^H)
            for (int i = 0; i < n; i++)
            {
                float64 sr = reduce_add_half32(v_f32_mac_b(v_f32_sub_b(0.0f, hi[i]), vi_v, v_f32_mul_b(hr[i], vr_v)));
                float64 si = reduce_add_half32(v_f32_mac_b(hr[i], vi_v, v_f32_mul_b(hi[i], vr_v)));
                float64 tsr = v_f32_mac_b(v_f32_sub_b(0.0f, taui), si, v_f32_mul_b(taur, sr));
                float64 tsi = v_f32_mac_b(taui, sr, v_f32_mul_b(taur, si));
                float64 ntsr = v_f32_sub_b(0.0f, tsr), ntsi = v_f32_sub_b(0.0f, tsi);
                hr[i] = v_f32_mac_b(ntsi, vi_v, v_f32_mac_b(ntsr, vr_v, hr[i]));
                hi[i] = v_f32_mac_b(tsr, vi_v, v_f32_mac_b(ntsi, vr_v, hi[i]));
            }

#ifndef NO_VECTORS
            // Accumulate Q[:, k+1:n] <- Q[:, k+1:n] (I - tau v v^H) (column-major)
            float64 sr_q = 0.0f, si_q = 0.0f;
            for (int j = k + 1; j < n; j++)
            {
                float64 vjr = lane_bcast(vr_v, j);
                float64 vji = lane_bcast(vi_v, j);
                sr_q = v_f32_mac_b(v_f32_sub_b(0.0f, vji), qi[j], v_f32_mac_b(vjr, qr[j], sr_q));
                si_q = v_f32_mac_b(vji, qr[j], v_f32_mac_b(vjr, qi[j], si_q));
            }
            float64 tsr_q = v_f32_mac_b(v_f32_sub_b(0.0f, taui), si_q, v_f32_mul_b(taur, sr_q));
            float64 tsi_q = v_f32_mac_b(taui, sr_q, v_f32_mul_b(taur, si_q));
            float64 ntsr_q = v_f32_sub_b(0.0f, tsr_q), ntsi_q = v_f32_sub_b(0.0f, tsi_q);
            for (int j = k + 1; j < n; j++)
            {
                float64 cvjr = lane_bcast(vr_v, j);
                float64 cvji = v_f32_sub_b(0.0f, lane_bcast(vi_v, j));
                qr[j] = v_f32_mac_b(tsi_q, cvji, v_f32_mac_b(ntsr_q, cvjr, qr[j]));
                qi[j] = v_f32_mac_b(ntsi_q, cvjr, v_f32_mac_b(ntsr_q, cvji, qi[j]));
            }
#endif
        }

        // Stage 3: Explicit complex single-shift QR iteration (scalar-read-free vector-predicated loop)
        float64 lane_f = 0.0f;
        for (int i = 0; i < MAXN; i++)
        {
            lane_f = v_f32_sel_eq_u32_b(lane, (unsigned)i, (float)i, lane_f);
        }
        const int max_itn = (n > 1) ? (30 * ((n > 10) ? n : 10)) : 0;
        float64 itn_used = 0.0f;
        float64 exc_cnt = 0.0f;
        float64 m_vec = (float)(n - 1);
        float64 prev_m = m_vec;

        for (int step = 0; step < max_itn; step++)
        {
            // Deflation scan: zero deflated subdiagonals, find highest non-deflated m_vec and highest deflated l_vec < m_vec
            float64 new_m = 0.0f;
            float64 defl_mask = 0.0f;
            for (int i = 1; i < n; i++)
            {
                float64 d0 = c_abs1(lane_bcast(hr[i - 1], i - 1), lane_bcast(hi[i - 1], i - 1));
                float64 d1 = c_abs1(lane_bcast(hr[i], i), lane_bcast(hi[i], i));
                float64 s_diag = v_f32_add_b(d0, d1);
                float64 tol = v_f32_max_b(sml, v_f32_mul_b(eps32, s_diag));
                float64 sub_d = c_abs1(lane_bcast(hr[i], i - 1), lane_bcast(hi[i], i - 1));
                float64 in_m = v_f32_sel_leq_f32_b((float)i, m_vec, 1.0f, 0.0f);
                float64 def_in = v_f32_sel_leq_f32_b(sub_d, tol, in_m, 0.0f);
                hr[i] = v_f32_sel_eq_u32_b(lane, (unsigned)(i - 1),
                                           v_f32_sel_eq_f32_b(def_in, 1.0f, 0.0f, hr[i]), hr[i]);
                hi[i] = v_f32_sel_eq_u32_b(lane, (unsigned)(i - 1),
                                           v_f32_sel_eq_f32_b(def_in, 1.0f, 0.0f, hi[i]), hi[i]);
                float64 keep_m = v_f32_sel_eq_f32_b(def_in, 0.0f, in_m, 0.0f);
                new_m = v_f32_sel_eq_f32_b(keep_m, 1.0f, (float)i, new_m);
                defl_mask = v_f32_sel_eq_u32_b(lane, (unsigned)i, def_in, defl_mask);
            }
            m_vec = new_m;
            float64 active = v_f32_sel_geq_f32_b(m_vec, 1.0f, 1.0f, 0.0f);
            float64 m_changed = v_f32_sel_eq_f32_b(m_vec, prev_m, 0.0f, 1.0f);
            prev_m = m_vec;
            exc_cnt = v_f32_sel_eq_f32_b(m_changed, 1.0f, 0.0f, exc_cnt);
            exc_cnt = v_f32_sel_eq_f32_b(active, 1.0f, v_f32_add_b(exc_cnt, 1.0f), 0.0f);
            itn_used = v_f32_sel_eq_f32_b(active, 1.0f, v_f32_add_b(itn_used, 1.0f), itn_used);

            float64 l_vec = 0.0f;
            for (int i = 1; i < n; i++)
            {
                float64 di = lane_bcast(defl_mask, i);
                float64 lt_m = v_f32_sel_less_f32_b((float)i, m_vec, 1.0f, 0.0f);
                l_vec = v_f32_sel_eq_f32_b(v_f32_mul_b(di, lt_m), 1.0f, (float)i, l_vec);
            }

            // Extract rows m_vec and m_vec - 1 for shift calculation
            float64 m_m1 = v_f32_sub_b(m_vec, 1.0f);
            float64 m_m2 = v_f32_sub_b(m_vec, 2.0f);
            float64 row_mr = 0.0f, row_mi = 0.0f, row_m1r = 0.0f, row_m1i = 0.0f;
            for (int i = 0; i < n; i++)
            {
                float64 is_m = v_f32_sel_eq_f32_b(m_vec, (float)i, 1.0f, 0.0f);
                float64 is_m1 = v_f32_sel_eq_f32_b(m_m1, (float)i, 1.0f, 0.0f);
                row_mr = v_f32_sel_eq_f32_b(is_m, 1.0f, hr[i], row_mr);
                row_mi = v_f32_sel_eq_f32_b(is_m, 1.0f, hi[i], row_mi);
                row_m1r = v_f32_sel_eq_f32_b(is_m1, 1.0f, hr[i], row_m1r);
                row_m1i = v_f32_sel_eq_f32_b(is_m1, 1.0f, hi[i], row_m1i);
            }
            float64 mask_cm = v_f32_sel_eq_f32_b(lane_f, m_vec, 1.0f, 0.0f);
            float64 mask_cm1 = v_f32_sel_eq_f32_b(lane_f, m_m1, 1.0f, 0.0f);
            float64 mask_cm2 = v_f32_sel_eq_f32_b(lane_f, m_m2, 1.0f, 0.0f);
            float64 tr_ = reduce_add_half32(v_f32_sel_eq_f32_b(mask_cm, 1.0f, row_mr, 0.0f));
            float64 ti_ = reduce_add_half32(v_f32_sel_eq_f32_b(mask_cm, 1.0f, row_mi, 0.0f));
            float64 hm1m1_r = reduce_add_half32(v_f32_sel_eq_f32_b(mask_cm1, 1.0f, row_m1r, 0.0f));
            float64 hm1m1_i = reduce_add_half32(v_f32_sel_eq_f32_b(mask_cm1, 1.0f, row_m1i, 0.0f));
            float64 br_ = reduce_add_half32(v_f32_sel_eq_f32_b(mask_cm, 1.0f, row_m1r, 0.0f));
            float64 bi_ = reduce_add_half32(v_f32_sel_eq_f32_b(mask_cm, 1.0f, row_m1i, 0.0f));
            float64 cr_ = reduce_add_half32(v_f32_sel_eq_f32_b(mask_cm1, 1.0f, row_mr, 0.0f));
            float64 ci_ = reduce_add_half32(v_f32_sel_eq_f32_b(mask_cm1, 1.0f, row_mi, 0.0f));
            float64 hm1m2_r = reduce_add_half32(v_f32_sel_eq_f32_b(mask_cm2, 1.0f, row_m1r, 0.0f));
            float64 hm1m2_i = reduce_add_half32(v_f32_sel_eq_f32_b(mask_cm2, 1.0f, row_m1i, 0.0f));

            // Exceptional shift (every 10 stalled steps)
            float64 is_exc = v_f32_sel_eq_f32_b(exc_cnt, 10.0f, 1.0f, 0.0f);
            exc_cnt = v_f32_sel_eq_f32_b(is_exc, 1.0f, 0.0f, exc_cnt);
            float64 s_exc = c_abs1(cr_, ci_);
            float64 has_m2 = v_f32_sel_grt_f32_b(m_m1, l_vec, 1.0f, 0.0f);
            s_exc = v_f32_sel_eq_f32_b(has_m2, 1.0f, v_f32_add_b(s_exc, c_abs1(hm1m2_r, hm1m2_i)), s_exc);
            float64 sigr_exc = v_f32_mac_b(0.75f, s_exc, tr_);
            float64 sigi_exc = v_f32_mac_b(-0.4375f, s_exc, ti_);

            // Wilkinson shift
            float64 ur_ = v_f32_sub_b(hm1m1_r, tr_);
            float64 ui_ = v_f32_sub_b(hm1m1_i, ti_);
            float64 sm = v_f32_max_b(v_f32_max_b(v_f32_abs_b(ur_), v_f32_abs_b(ui_)),
                                     v_f32_max_b(v_f32_max_b(v_f32_abs_b(br_), v_f32_abs_b(bi_)),
                                                 v_f32_max_b(v_f32_abs_b(cr_), v_f32_abs_b(ci_))));
            float64 sc = pow2_scale(sm);
            float64 unsc = pow2_unscale(sm);
            float64 xr = v_f32_mul_b(v_f32_mul_b(ur_, sc), 0.5f);
            float64 xi = v_f32_mul_b(v_f32_mul_b(ui_, sc), 0.5f);
            float64 bsr = v_f32_mul_b(br_, sc), bsi = v_f32_mul_b(bi_, sc);
            float64 csr = v_f32_mul_b(cr_, sc), csi = v_f32_mul_b(ci_, sc);
            float64 vr_ = v_f32_mac_b(v_f32_sub_b(0.0f, bsi), csi, v_f32_mul_b(bsr, csr));
            float64 vi_ = v_f32_mac_b(bsi, csr, v_f32_mul_b(bsr, csi));
            float64 x2r = v_f32_mac_b(v_f32_sub_b(0.0f, xi), xi, v_f32_mul_b(xr, xr));
            float64 x2i = v_f32_mac_b(xi, xr, v_f32_mul_b(xr, xi));
            cvec_t sqv = c_sqrt(v_f32_add_b(x2r, vr_), v_f32_add_b(x2i, vi_));
            float64 sr = sqv.r, si = sqv.i;
            float64 dot_xs = v_f32_mac_b(xi, si, v_f32_mul_b(xr, sr));
            sr = v_f32_sel_less_f32_b(dot_xs, 0.0f, v_f32_sub_b(0.0f, sr), sr);
            si = v_f32_sel_less_f32_b(dot_xs, 0.0f, v_f32_sub_b(0.0f, si), si);
            float64 denr = v_f32_add_b(xr, sr), deni = v_f32_add_b(xi, si);
            cvec_t dv = c_div(vr_, vi_, denr, deni);
            float64 dr = dv.r, di = dv.i;
            float64 den_zero = v_f32_sel_eq_f32_b(denr, 0.0f, v_f32_sel_eq_f32_b(deni, 0.0f, 1.0f, 0.0f), 0.0f);
            float64 skip_s = v_f32_sel_eq_f32_b(sm, 0.0f, 1.0f, den_zero);
            float64 sigr_w = v_f32_sel_eq_f32_b(skip_s, 1.0f, tr_, v_f32_sub_b(tr_, v_f32_mul_b(dr, unsc)));
            float64 sigi_w = v_f32_sel_eq_f32_b(skip_s, 1.0f, ti_, v_f32_sub_b(ti_, v_f32_mul_b(di, unsc)));
            float64 sigr = v_f32_sel_eq_f32_b(is_exc, 1.0f, sigr_exc, sigr_w);
            float64 sigi = v_f32_sel_eq_f32_b(is_exc, 1.0f, sigi_exc, sigi_w);

            // Single-shift bulge chase k = 0 .. n - 2 (predicated by l_vec <= k < m_vec)
            for (int k = 0; k < n - 1; k++)
            {
                float64 k_ge_l = v_f32_sel_geq_f32_b((float)k, l_vec, 1.0f, 0.0f);
                float64 k_lt_m = v_f32_sel_less_f32_b((float)k, m_vec, 1.0f, 0.0f);
                float64 k_act = v_f32_mul_b(active, v_f32_mul_b(k_ge_l, k_lt_m));
                float64 k_eq_l = v_f32_sel_eq_f32_b((float)k, l_vec, 1.0f, 0.0f);

                float64 fr = v_f32_sub_b(lane_bcast(hr[k], k), sigr);
                float64 fi = v_f32_sub_b(lane_bcast(hi[k], k), sigi);
                float64 gr = lane_bcast(hr[k + 1], k);
                float64 gi = lane_bcast(hi[k + 1], k);
                if (k > 0)
                {
                    float64 fr_n = lane_bcast(hr[k], k - 1);
                    float64 fi_n = lane_bcast(hi[k], k - 1);
                    float64 gr_n = lane_bcast(hr[k + 1], k - 1);
                    float64 gi_n = lane_bcast(hi[k + 1], k - 1);
                    fr = v_f32_sel_eq_f32_b(k_eq_l, 1.0f, fr, fr_n);
                    fi = v_f32_sel_eq_f32_b(k_eq_l, 1.0f, fi, fi_n);
                    gr = v_f32_sel_eq_f32_b(k_eq_l, 1.0f, gr, gr_n);
                    gi = v_f32_sel_eq_f32_b(k_eq_l, 1.0f, gi, gi_n);
                }
                givens_t gv = clartg(fr, fi, gr, gi);
                float64 cs = gv.cs, snr = gv.snr, sni = gv.sni, rr = gv.rr, ri = gv.ri;
                if (k > 0)
                {
                    float64 k_gt_l = v_f32_sel_eq_f32_b(k_eq_l, 0.0f, k_act, 0.0f);
                    hr[k] = v_f32_sel_eq_u32_b(lane, (unsigned)(k - 1),
                                               v_f32_sel_eq_f32_b(k_gt_l, 1.0f, rr, hr[k]), hr[k]);
                    hi[k] = v_f32_sel_eq_u32_b(lane, (unsigned)(k - 1),
                                               v_f32_sel_eq_f32_b(k_gt_l, 1.0f, ri, hi[k]), hi[k]);
                    hr[k + 1] = v_f32_sel_eq_u32_b(lane, (unsigned)(k - 1),
                                                   v_f32_sel_eq_f32_b(k_gt_l, 1.0f, 0.0f, hr[k + 1]), hr[k + 1]);
                    hi[k + 1] = v_f32_sel_eq_u32_b(lane, (unsigned)(k - 1),
                                                   v_f32_sel_eq_f32_b(k_gt_l, 1.0f, 0.0f, hi[k + 1]), hi[k + 1]);
                }

                // Left update: rows k, k+1, cols k..c_end-1
                float64 r0r = hr[k], r0i = hi[k], r1r = hr[k + 1], r1i = hi[k + 1];
                float64 nr0r = v_f32_mac_b(v_f32_sub_b(0.0f, sni), r1i, v_f32_mac_b(snr, r1r, v_f32_mul_b(cs, r0r)));
                float64 nr0i = v_f32_mac_b(sni, r1r, v_f32_mac_b(snr, r1i, v_f32_mul_b(cs, r0i)));
                float64 nsnr = v_f32_sub_b(0.0f, snr), nsni = sni;
                float64 nr1r = v_f32_mac_b(v_f32_sub_b(0.0f, nsni), r0i, v_f32_mac_b(nsnr, r0r, v_f32_mul_b(cs, r1r)));
                float64 nr1i = v_f32_mac_b(nsni, r0r, v_f32_mac_b(nsnr, r0i, v_f32_mul_b(cs, r1i)));
#ifdef NO_VECTORS
                float64 in_cols = v_f32_sel_leq_f32_b(lane_f, m_vec, 1.0f, 0.0f);
#else
                float64 in_cols = v_f32_sel_less_u32_b(lane, (unsigned)n, 1.0f, 0.0f);
#endif
                float64 mask_l = v_f32_sel_geq_u32_b(lane, (unsigned)k, v_f32_mul_b(k_act, in_cols), 0.0f);
                hr[k] = v_f32_sel_eq_f32_b(mask_l, 1.0f, nr0r, r0r);
                hi[k] = v_f32_sel_eq_f32_b(mask_l, 1.0f, nr0i, r0i);
                hr[k + 1] = v_f32_sel_eq_f32_b(mask_l, 1.0f, nr1r, r1r);
                hi[k + 1] = v_f32_sel_eq_f32_b(mask_l, 1.0f, nr1i, r1i);

                // Right update: cols k, k+1, rows r_start..min(k+2, m)
                const int r_end = (k + 3 < n) ? (k + 3) : n;
                float64 csnr = snr, csni = v_f32_sub_b(0.0f, sni);
                float64 nsr = v_f32_sub_b(0.0f, snr), nsi = v_f32_sub_b(0.0f, sni);
                for (int i = 0; i < r_end; i++)
                {
#ifdef NO_VECTORS
                    float64 r_ok = v_f32_sel_geq_f32_b((float)i, l_vec,
                                                       v_f32_sel_leq_f32_b((float)i, m_vec, k_act, 0.0f), 0.0f);
#else
                    float64 r_ok = v_f32_sel_leq_f32_b((float)i, m_vec, k_act, 0.0f);
#endif
                    float64 ckr = lane_bcast(hr[i], k), cki = lane_bcast(hi[i], k);
                    float64 ck1r = lane_bcast(hr[i], k + 1), ck1i = lane_bcast(hi[i], k + 1);
                    float64 nckr = v_f32_mac_b(v_f32_sub_b(0.0f, csni), ck1i, v_f32_mac_b(csnr, ck1r, v_f32_mul_b(cs, ckr)));
                    float64 ncki = v_f32_mac_b(csni, ck1r, v_f32_mac_b(csnr, ck1i, v_f32_mul_b(cs, cki)));
                    float64 nck1r = v_f32_mac_b(v_f32_sub_b(0.0f, nsi), cki, v_f32_mac_b(nsr, ckr, v_f32_mul_b(cs, ck1r)));
                    float64 nck1i = v_f32_mac_b(nsi, ckr, v_f32_mac_b(nsr, cki, v_f32_mul_b(cs, ck1i)));
                    float64 h_upd_r = v_f32_sel_eq_u32_b(lane, (unsigned)k, nckr, v_f32_sel_eq_u32_b(lane, (unsigned)(k + 1), nck1r, hr[i]));
                    float64 h_upd_i = v_f32_sel_eq_u32_b(lane, (unsigned)k, ncki, v_f32_sel_eq_u32_b(lane, (unsigned)(k + 1), nck1i, hi[i]));
                    hr[i] = v_f32_sel_eq_f32_b(r_ok, 1.0f, h_upd_r, hr[i]);
                    hi[i] = v_f32_sel_eq_f32_b(r_ok, 1.0f, h_upd_i, hi[i]);
                }

#ifndef NO_VECTORS
                float64 q0r = qr[k], q0i = qi[k], q1r = qr[k + 1], q1i = qi[k + 1];
                float64 nq0r = v_f32_mac_b(v_f32_sub_b(0.0f, csni), q1i, v_f32_mac_b(csnr, q1r, v_f32_mul_b(cs, q0r)));
                float64 nq0i = v_f32_mac_b(csni, q1r, v_f32_mac_b(csnr, q1i, v_f32_mul_b(cs, q0i)));
                float64 nq1r = v_f32_mac_b(v_f32_sub_b(0.0f, nsi), q0i, v_f32_mac_b(nsr, q0r, v_f32_mul_b(cs, q1r)));
                float64 nq1i = v_f32_mac_b(nsi, q0r, v_f32_mac_b(nsr, q0i, v_f32_mul_b(cs, q1i)));
                qr[k] = v_f32_sel_eq_f32_b(k_act, 1.0f, nq0r, q0r);
                qi[k] = v_f32_sel_eq_f32_b(k_act, 1.0f, nq0i, q0i);
                qr[k + 1] = v_f32_sel_eq_f32_b(k_act, 1.0f, nq1r, q1r);
                qi[k + 1] = v_f32_sel_eq_f32_b(k_act, 1.0f, nq1i, q1i);
#endif
            }
        }

        // Final deflation check for info
        float64 final_m = 0.0f;
        for (int i = 1; i < n; i++)
        {
            float64 d0 = c_abs1(lane_bcast(hr[i - 1], i - 1), lane_bcast(hi[i - 1], i - 1));
            float64 d1 = c_abs1(lane_bcast(hr[i], i), lane_bcast(hi[i], i));
            float64 s_diag = v_f32_add_b(d0, d1);
            float64 tol = v_f32_max_b(sml, v_f32_mul_b(eps32, s_diag));
            float64 sub_d = c_abs1(lane_bcast(hr[i], i - 1), lane_bcast(hi[i], i - 1));
            float64 in_m = v_f32_sel_leq_f32_b((float)i, m_vec, 1.0f, 0.0f);
            float64 def_in = v_f32_sel_leq_f32_b(sub_d, tol, in_m, 0.0f);
            hr[i] = v_f32_sel_eq_u32_b(lane, (unsigned)(i - 1),
                                       v_f32_sel_eq_f32_b(def_in, 1.0f, 0.0f, hr[i]), hr[i]);
            hi[i] = v_f32_sel_eq_u32_b(lane, (unsigned)(i - 1),
                                       v_f32_sel_eq_f32_b(def_in, 1.0f, 0.0f, hi[i]), hi[i]);
            float64 keep_m = v_f32_sel_eq_f32_b(def_in, 0.0f, in_m, 0.0f);
            final_m = v_f32_sel_eq_f32_b(keep_m, 1.0f, (float)i, final_m);
        }
        float64 info_vec = v_f32_sel_geq_f32_b(final_m, 1.0f, v_f32_add_b(final_m, 1.0f), 0.0f);

        // Write eigenvalues W [n, 2, batch] = diag(H) * unsc_entry
        float64 wr_out = 0.0f, wi_out = 0.0f;
        for (int i = 0; i < n; i++)
        {
            wr_out = v_f32_sel_eq_u32_b(lane, (unsigned)i, lane_bcast(hr[i], i), wr_out);
            wi_out = v_f32_sel_eq_u32_b(lane, (unsigned)i, lane_bcast(hi[i], i), wi_out);
        }
        wr_out = v_f32_mul_b(wr_out, unsc_entry);
        wi_out = v_f32_mul_b(wi_out, unsc_entry);
        wr_out = v_f32_sel_eq_f32_b(max_abs, 0.0f, 0.0f, wr_out);
        wi_out = v_f32_sel_eq_f32_b(max_abs, 0.0f, 0.0f, wi_out);
        wr_out = v_f32_sel_grt_f32_b(bad, 0.0f, nanv, wr_out);
        wi_out = v_f32_sel_grt_f32_b(bad, 0.0f, nanv, wi_out);
        int5 cw0 = {0, 0, bt, 0, 0};
        int5 cw1 = {0, 1, bt, 0, 0};
        v_f32_st_tnsr_partial(cw0, W, wr_out, (char)(n - 1), 0);
        v_f32_st_tnsr_partial(cw1, W, wi_out, (char)(n - 1), 0);

        float64 sv = v_f32_sel_eq_u32_b(lane, 0u, info_vec, itn_used);
        sv = v_f32_sel_eq_f32_b(max_abs, 0.0f, 0.0f, sv);
        sv = v_f32_sel_grt_f32_b(bad, 0.0f, nanv, sv);
        int5 cs = {0, bt, 0, 0, 0};
        v_f32_st_tnsr_partial(cs, S, sv, 1, 0);

#ifndef NO_VECTORS
        // Stage 4: Back-substitution on upper triangular T, Q @ y, inverse balance, unit 2-norm
        float64 norm_T = 0.0f;
        for (int i = 0; i < n; i++)
        {
            float64 row_m = v_f32_sel_geq_u32_b(lane, (unsigned)i, c_abs1(hr[i], hi[i]), 0.0f);
            row_m = v_f32_sel_geq_u32_b(lane, (unsigned)n, 0.0f, row_m);
            norm_T = v_f32_max_b(norm_T, row_m);
        }
        norm_T = reduce_max_half32(norm_T);

        for (int k = 0; k < n; k++)
        {
            float64 lam_r = lane_bcast(hr[k], k);
            float64 lam_i = lane_bcast(hi[k], k);
            float64 smin = v_f32_max_b(sml, v_f32_mul_b(eps32, v_f32_max_b(c_abs1(lam_r, lam_i), norm_T)));
            float64 yr = v_f32_sel_eq_u32_b(lane, (unsigned)k, 1.0f, 0.0f);
            float64 yi = 0.0f;
            for (int i = k - 1; i >= 0; i--)
            {
                float64 mask_t = v_f32_sel_grt_u32_b(lane, (unsigned)i,
                                                     v_f32_sel_leq_u32_b(lane, (unsigned)k, 1.0f, 0.0f), 0.0f);
                float64 tr_i = v_f32_sel_eq_f32_b(mask_t, 1.0f, hr[i], 0.0f);
                float64 ti_i = v_f32_sel_eq_f32_b(mask_t, 1.0f, hi[i], 0.0f);
                float64 sr = reduce_add_half32(v_f32_mac_b(v_f32_sub_b(0.0f, ti_i), yi, v_f32_mul_b(tr_i, yr)));
                float64 si = reduce_add_half32(v_f32_mac_b(ti_i, yr, v_f32_mul_b(tr_i, yi)));
                float64 rhsr = v_f32_sub_b(0.0f, sr), rhsi = v_f32_sub_b(0.0f, si);
                float64 dr = v_f32_sub_b(lane_bcast(hr[i], i), lam_r);
                float64 di = v_f32_sub_b(lane_bcast(hi[i], i), lam_i);
                float64 d_small = v_f32_sel_less_f32_b(c_abs1(dr, di), smin, 1.0f, 0.0f);
                dr = v_f32_sel_eq_f32_b(d_small, 1.0f, smin, dr);
                di = v_f32_sel_eq_f32_b(d_small, 1.0f, 0.0f, di);
                float64 rm = v_f32_max_b(v_f32_abs_b(rhsr), v_f32_abs_b(rhsi));
                float64 dm = v_f32_max_b(v_f32_abs_b(dr), v_f32_abs_b(di));
                float64 need_rs1 = v_f32_sel_grt_f32_b(rm, v_f32_mul_b(0x1p+28f, dm), 1.0f, 0.0f);
                float64 sc1 = v_f32_sel_eq_f32_b(need_rs1, 1.0f, pow2_scale(rm), 1.0f);
                yr = v_f32_mul_b(yr, sc1);
                yi = v_f32_mul_b(yi, sc1);
                rhsr = v_f32_mul_b(rhsr, sc1);
                rhsi = v_f32_mul_b(rhsi, sc1);
                cvec_t qy = c_div(rhsr, rhsi, dr, di);
                yr = v_f32_sel_eq_u32_b(lane, (unsigned)i, qy.r, yr);
                yi = v_f32_sel_eq_u32_b(lane, (unsigned)i, qy.i, yi);
                float64 ym = reduce_max_half32(v_f32_max_b(v_f32_abs_b(yr), v_f32_abs_b(yi)));
                float64 need_rs2 = v_f32_sel_grt_f32_b(ym, 0x1p+28f, 1.0f, 0.0f);
                float64 sc2 = v_f32_sel_eq_f32_b(need_rs2, 1.0f, pow2_scale(ym), 1.0f);
                yr = v_f32_mul_b(yr, sc2);
                yi = v_f32_mul_b(yi, sc2);
            }
            float64 ym_end = reduce_max_half32(v_f32_max_b(v_f32_abs_b(yr), v_f32_abs_b(yi)));
            float64 sc_y = v_f32_sel_grt_f32_b(ym_end, 0.0f, pow2_scale(ym_end), 1.0f);
            yr = v_f32_mul_b(yr, sc_y);
            yi = v_f32_mul_b(yi, sc_y);

            // v_k = Q[:, 0:k+1] @ y[0:k+1]
            float64 vkr = 0.0f, vki = 0.0f;
            for (int j = 0; j <= k; j++)
            {
                float64 yjr = lane_bcast(yr, j);
                float64 yji = lane_bcast(yi, j);
                vkr = v_f32_mac_b(v_f32_sub_b(0.0f, yji), qi[j], v_f32_mac_b(yjr, qr[j], vkr));
                vki = v_f32_mac_b(yji, qr[j], v_f32_mac_b(yjr, qi[j], vki));
            }
            vkr = v_f32_mul_b(vkr, d_scale);
            vki = v_f32_mul_b(vki, d_scale);
            float64 vm = reduce_max_half32(v_f32_max_b(v_f32_abs_b(vkr), v_f32_abs_b(vki)));
            float64 sc_v = v_f32_sel_grt_f32_b(vm, 0.0f, pow2_scale(vm), 1.0f);
            float64 vsr = v_f32_mul_b(vkr, sc_v);
            float64 vsi = v_f32_mul_b(vki, sc_v);
            float64 vsq = reduce_add_half32(v_f32_mac_b(vsi, vsi, v_f32_mul_b(vsr, vsr)));
            float64 rn = v_f32_sel_grt_f32_b(vm, 0.0f, rsqrt_fine(vsq), 0.0f);
            vr[k] = v_f32_mul_b(vsr, rn);
            vi[k] = v_f32_mul_b(vsi, rn);
        }

        // Transpose column-major vr/vi to row-major V [n, 2n, batch]
        for (int i = 0; i < n; i++)
        {
            float64 r_out = 0.0f, i_out = 0.0f;
            for (int k = 0; k < n; k++)
            {
                r_out = v_f32_sel_eq_u32_b(lane, (unsigned)k, lane_bcast(vr[k], i), r_out);
                i_out = v_f32_sel_eq_u32_b(lane, (unsigned)k, lane_bcast(vi[k], i), i_out);
            }
            float64 er_id = v_f32_sel_eq_u32_b(lane, (unsigned)i, 1.0f, 0.0f);
            r_out = v_f32_sel_eq_f32_b(max_abs, 0.0f, er_id, r_out);
            i_out = v_f32_sel_eq_f32_b(max_abs, 0.0f, 0.0f, i_out);
            r_out = v_f32_sel_grt_f32_b(bad, 0.0f, nanv, r_out);
            i_out = v_f32_sel_grt_f32_b(bad, 0.0f, nanv, i_out);
            int5 cvr = {0, i, bt, 0, 0};
            int5 cvi = {0, n + i, bt, 0, 0};
            v_f32_st_tnsr_partial(cvr, V, r_out, (char)(n - 1), 0);
            v_f32_st_tnsr_partial(cvi, V, i_out, (char)(n - 1), 0);
        }
#endif
    }
}
