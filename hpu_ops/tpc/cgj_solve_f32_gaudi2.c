/* Batched complex64 solve A X = B on Gaudi2 TPC: Gauss-Jordan with partial pivoting by cabs1 = |re| + |im| (LAPACK
 * icamax, as MPS's complex LU), one matrix per index-space member, complex numbers as float32 (re, im) planes.
 *
 * Tensors (TPC geometry, dim0 innermost; torch [batch, rows, cols] row-major), planar: rows 0..n-1 hold the real
 * parts, rows n..2n-1 the imaginary parts (torch: view_as_real(A).movedim(-1, -3).reshape(batch, 2n, n)):
 *   A [n, 2n, batch]  f32     B [m, 2n, batch]  f32     X [m, 2n, batch]  f32
 *   D [n, 3, batch]   f32: row 0/1 = re/im of the pivots (U's diagonal of P A = L U), row 2 = |pivot|
 *   S [4, batch]      f32: lane 0/1 = re/im of the phase of det A = (-1)^swaps prod(u_kk / |u_kk|) (0 when a pivot
 *                          is zero), lane 2 = info (1 + first zero pivot), lane 3 = (-1)^swaps
 * slogdet = (S0 + i S1, sum log D2); det = phase * prod |u_kk|. Any non-finite entry of A or B makes X, D and S NaN.
 * Elimination is LU-style (row -= (a_ik / pivot) * pivot row, the pivot row unnormalized; X = B / diag at the end),
 * the complex quotient -a_ik / pivot from a power-of-two scaled reciprocal of the pivot (|re|, |im| of the pivot scaled
 * into [1, 2) before squaring, so |pivot|^2 neither overflows nor underflows) with one residual correction; pivots
 * >= 2^64 are scaled by 2^-64 with their column first, as in the real kernel. A zero pivot leaves its step a no-op and
 * its row of X zero. With -DIDENTITY_B the kernel takes A alone and solves against the identity (inv): m = n.
 * Row i of each plane is one 64-lane vector in local memory (lane j = column j), so n, m <= MAXN; 4 planes x 32 rows =
 * 32 KB. LUT-free (no special function), so local memory stays in 80 KB mode and the arithmetic is replayable bit for
 * bit (tests/kernel_replica.py cgj_solve).
 */
#define MAXN 32

__local__ float64 ar[MAXN];
__local__ float64 ai[MAXN];
__local__ float64 br[MAXN];
__local__ float64 bi[MAXN];

static inline float64 lane_bcast(float64 v, int k)
{
    return as_float64(v_broadcast_element_i32(as_int64(v), k));
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

static inline uint64 reduce_min_u32_64(uint64 idx)
{
    idx = v_u32_min_b(idx, as_uint64(v_f32_mov_dual_group_all_b(
        as_float64(idx), 0xFFFFFFFF, 1, 0, 3, 2, SW_MDG_ALL, as_float64(idx))));
    idx = v_u32_min_b(idx, as_uint64(v_f32_mov_dual_group_all_b(
        as_float64(idx), 0xFFFFFFFF, 2, 3, 0, 1, SW_MDG_ALL, as_float64(idx))));
    idx = v_u32_min_b(idx, as_uint64(v_element_shift_xor_i32(as_int64(idx), 8)));
    idx = v_u32_min_b(idx, as_uint64(v_element_shift_xor_i32(as_int64(idx), 4)));
    idx = v_u32_min_b(idx, as_uint64(v_element_shift_xor_i32(as_int64(idx), 2)));
    idx = v_u32_min_b(idx, as_uint64(v_element_shift_xor_i32(as_int64(idx), 1)));
    return idx;
}

static inline uint64 reduce_argmax_64(float64 cand, uint64 lane, int n, int k)
{
    float64 max_val = v_f32_max_b(reduce_max_64(cand), 0.0f);
    uint64 idx = v_u32_sel_eq_f32_b(cand, max_val, lane, 0xFFFFFFFFu);
    uint64 p = reduce_min_u32_64(idx);
    return v_u32_sel_geq_u32_b(p, (unsigned)n, (unsigned)k, p);
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

// 1/sqrt(x) for x >= 0 in the normal range (here |z|^2 of a number scaled into [1, 2), so x in [1, 8), or 0): the
// bit-level guess and three Newton steps; x = 0 gives 1 (so x * result = 0; the guess for 0 would overflow).
static inline float64 lut_free_rsqrt(float64 x)
{
    x = v_f32_sel_eq_f32_b(x, 0.0f, 1.0f, x);
    float64 y = as_float64(0x5F375A86u - v_u32_shr_b(as_uint64(x), 1));
    float64 hx = v_f32_mul_b(x, -0.5f);
    y = v_f32_mul_b(y, v_f32_mac_b(hx, v_f32_mul_b(y, y), 1.5f));
    y = v_f32_mul_b(y, v_f32_mac_b(hx, v_f32_mul_b(y, y), 1.5f));
    y = v_f32_mul_b(y, v_f32_mac_b(hx, v_f32_mul_b(y, y), 1.5f));
    return y;
}

// The power of two t = 2^(127 - e) for the biased exponent e of m >= 0 (m < 2^64 here): m * t in [1, 2); t = 1 for
// m = 0. pow2_unscale is its inverse 2^(e - 127) (1 for m = 0).
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

// 2^-64 where max(|re|, |im|) >= 2^64, else 1
static inline float64 big_scale(float64 m)
{
    return v_f32_sel_geq_u32_b(v_u32_and_b(as_uint64(m), 0x7FFFFFFFu), 0x5F800000u, 0x1p-64f, 1.0f);
}

#ifdef IDENTITY_B
void main(tensor A, tensor X, tensor D, tensor S)
#else
void main(tensor A, tensor B, tensor X, tensor D, tensor S)
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
    const uint64 nan_bits = 0x7FC00000u;
    const float64 nanv = as_float64(nan_bits);

    for (int bt = off[0]; bt < off[0] + size[0]; bt++)
    {
        float64 bad = 0.0f;  // 1 in some lane when A or B has a non-finite entry
        for (int i = 0; i < n; i++)
        {
            int5 cr = {0, i, bt, 0, 0};
            int5 ci = {0, n + i, bt, 0, 0};
            float64 xr = v_f32_sel_geq_u32_b(lane, (unsigned)n, 0.0f, v_f32_ld_tnsr_b(cr, A));
            float64 xi = v_f32_sel_geq_u32_b(lane, (unsigned)n, 0.0f, v_f32_ld_tnsr_b(ci, A));
            ar[i] = xr;
            ai[i] = xi;
            bad = v_f32_sel_eq_u32_b(v_u32_and_b(as_uint64(xr), 0x7F800000u), 0x7F800000u, 1.0f, bad);
            bad = v_f32_sel_eq_u32_b(v_u32_and_b(as_uint64(xi), 0x7F800000u), 0x7F800000u, 1.0f, bad);
#ifdef IDENTITY_B
            br[i] = v_f32_sel_eq_u32_b(lane, (unsigned)i, 1.0f, 0.0f);
            bi[i] = 0.0f;
#else
            float64 yr = v_f32_sel_geq_u32_b(lane, (unsigned)m, 0.0f, v_f32_ld_tnsr_b(cr, B));
            float64 yi = v_f32_sel_geq_u32_b(lane, (unsigned)m, 0.0f, v_f32_ld_tnsr_b(ci, B));
            br[i] = yr;
            bi[i] = yi;
            bad = v_f32_sel_eq_u32_b(v_u32_and_b(as_uint64(yr), 0x7F800000u), 0x7F800000u, 1.0f, bad);
            bad = v_f32_sel_eq_u32_b(v_u32_and_b(as_uint64(yi), 0x7F800000u), 0x7F800000u, 1.0f, bad);
#endif
        }
        bad = reduce_max_64(bad);
        float64 dvr = 0.0f, dvi = 0.0f;
        float64 sv = 1.0f;
        float64 infov = 0.0f;
        for (int k = 0; k < n; k++)
        {
            // column k over rows (lane i = A[i][k]), two select chains per plane; lanes disjoint, so the sums are exact
            float64 r0 = 0.0f, r1 = 0.0f, i0 = 0.0f, i1 = 0.0f;
            int i2 = 0;
            for (; i2 + 1 < n; i2 += 2)
            {
                r0 = v_f32_sel_eq_u32_b(lane, (unsigned)i2, lane_bcast(ar[i2], k), r0);
                i0 = v_f32_sel_eq_u32_b(lane, (unsigned)i2, lane_bcast(ai[i2], k), i0);
                r1 = v_f32_sel_eq_u32_b(lane, (unsigned)(i2 + 1), lane_bcast(ar[i2 + 1], k), r1);
                i1 = v_f32_sel_eq_u32_b(lane, (unsigned)(i2 + 1), lane_bcast(ai[i2 + 1], k), i1);
            }
            if (i2 < n)
            {
                r0 = v_f32_sel_eq_u32_b(lane, (unsigned)i2, lane_bcast(ar[i2], k), r0);
                i0 = v_f32_sel_eq_u32_b(lane, (unsigned)i2, lane_bcast(ai[i2], k), i0);
            }
            float64 colr = v_f32_add_b(r0, r1);
            float64 coli = v_f32_add_b(i0, i1);
            float64 cand = v_f32_add_b(v_f32_abs_b(colr), v_f32_abs_b(coli));
            cand = v_f32_sel_less_u32_b(lane, (unsigned)k, -1.0f, cand);
            cand = v_f32_sel_geq_u32_b(lane, (unsigned)n, -1.0f, cand);
            uint64 p = reduce_argmax_64(cand, lane, n, k);

            // swap rows k and p with selects (no data-dependent branch)
            float64 akr = ar[k], aki = ai[k], bkr = br[k], bki = bi[k];
            float64 apr = akr, api = aki, bpr = bkr, bpi = bki;
            for (int i = k + 1; i < n; i++)
            {
                apr = v_f32_sel_eq_u32_b(p, (unsigned)i, ar[i], apr);
                api = v_f32_sel_eq_u32_b(p, (unsigned)i, ai[i], api);
                bpr = v_f32_sel_eq_u32_b(p, (unsigned)i, br[i], bpr);
                bpi = v_f32_sel_eq_u32_b(p, (unsigned)i, bi[i], bpi);
            }
            for (int i = k + 1; i < n; i++)
            {
                ar[i] = v_f32_sel_eq_u32_b(p, (unsigned)i, akr, ar[i]);
                ai[i] = v_f32_sel_eq_u32_b(p, (unsigned)i, aki, ai[i]);
                br[i] = v_f32_sel_eq_u32_b(p, (unsigned)i, bkr, br[i]);
                bi[i] = v_f32_sel_eq_u32_b(p, (unsigned)i, bki, bi[i]);
            }
            ar[k] = apr;
            ai[k] = api;
            br[k] = bpr;
            bi[k] = bpi;
            float64 pr = lane_bcast(apr, k);
            float64 pi = lane_bcast(api, k);
            colr = v_f32_sel_eq_u32_b(lane, p, lane_bcast(colr, k), colr);
            coli = v_f32_sel_eq_u32_b(lane, p, lane_bcast(coli, k), coli);
            colr = v_f32_sel_eq_u32_b(lane, (unsigned)k, pr, colr);
            coli = v_f32_sel_eq_u32_b(lane, (unsigned)k, pi, coli);
            dvr = v_f32_sel_eq_u32_b(lane, (unsigned)k, pr, dvr);
            dvi = v_f32_sel_eq_u32_b(lane, (unsigned)k, pi, dvi);
            sv = v_f32_sel_eq_u32_b(p, (unsigned)k, sv, v_f32_sub_b(0.0f, sv));
            float64 pm = v_f32_max_b(v_f32_abs_b(pr), v_f32_abs_b(pi));
            infov = v_f32_sel_eq_f32_b(pm, 0.0f, v_f32_sel_eq_f32_b(infov, 0.0f, (float)(k + 1), infov), infov);

            // -1/pivot: pivot (and its column) times 2^-64 when huge, then conj(p t) / |p t|^2 * t with p t in [1, 2)
            float64 scale = big_scale(pm);
            float64 psr = v_f32_mul_b(pr, scale);
            float64 psi = v_f32_mul_b(pi, scale);
            float64 t = pow2_scale(v_f32_mul_b(pm, scale));
            float64 tr = v_f32_mul_b(psr, t);
            float64 ti = v_f32_mul_b(psi, t);
            float64 rt = v_f32_mul_b(lut_free_recip(v_f32_mac_b(ti, ti, v_f32_mul_b(tr, tr))), t);
            float64 ninvr = v_f32_mul_b(v_f32_sub_b(0.0f, tr), rt);
            float64 ninvi = v_f32_mul_b(ti, rt);
            // q = -col / pivot for every row at once, then one residual correction q += (cs + q * ps) * (-1 / ps)
            float64 csr = v_f32_mul_b(colr, scale);
            float64 csi = v_f32_mul_b(coli, scale);
            float64 qr = v_f32_mac_b(v_f32_sub_b(0.0f, csi), ninvi, v_f32_mul_b(csr, ninvr));
            float64 qi = v_f32_mac_b(csi, ninvr, v_f32_mul_b(csr, ninvi));
            float64 rr = v_f32_mac_b(v_f32_sub_b(0.0f, qi), psi, v_f32_mac_b(qr, psr, csr));
            float64 ri = v_f32_mac_b(qi, psr, v_f32_mac_b(qr, psi, csi));
            qr = v_f32_mac_b(v_f32_sub_b(0.0f, ri), ninvi, v_f32_mac_b(rr, ninvr, qr));
            qi = v_f32_mac_b(ri, ninvr, v_f32_mac_b(rr, ninvi, qi));
            float64 nqi = v_f32_sub_b(0.0f, qi);
            for (int i = 0; i < n; i++)
            {
                if (i == k)
                {
                    continue;
                }
                float64 lr = lane_bcast(qr, i);
                float64 li = lane_bcast(qi, i);
                float64 nli = lane_bcast(nqi, i);
                ar[i] = v_f32_mac_b(api, nli, v_f32_mac_b(apr, lr, ar[i]));
                ai[i] = v_f32_mac_b(apr, li, v_f32_mac_b(api, lr, ai[i]));
                br[i] = v_f32_mac_b(bpi, nli, v_f32_mac_b(bpr, lr, br[i]));
                bi[i] = v_f32_mac_b(bpr, li, v_f32_mac_b(bpi, lr, bi[i]));
            }
        }
        // X row i = B row i / d_i (the same scaled reciprocal, lane i = pivot i); |d_i| and the unit phase d_i / |d_i|
        float64 dm = v_f32_max_b(v_f32_abs_b(dvr), v_f32_abs_b(dvi));
        float64 dscale = big_scale(dm);
        float64 dms = v_f32_mul_b(dm, dscale);
        float64 t = pow2_scale(dms);
        float64 inv_t = pow2_unscale(dms);
        float64 tr = v_f32_mul_b(v_f32_mul_b(dvr, dscale), t);
        float64 ti = v_f32_mul_b(v_f32_mul_b(dvi, dscale), t);
        float64 den = v_f32_mac_b(ti, ti, v_f32_mul_b(tr, tr));
        float64 rt = v_f32_mul_b(lut_free_recip(den), t);
        float64 invr = v_f32_mul_b(tr, rt);
        float64 invi = v_f32_mul_b(v_f32_sub_b(0.0f, ti), rt);
        float64 rs = lut_free_rsqrt(den);
        float64 absd = v_f32_mul_b(v_f32_mul_b(v_f32_mul_b(den, rs), inv_t),
                                   v_f32_sel_eq_f32_b(dscale, 1.0f, 1.0f, 0x1p64f));
        float64 ur = v_f32_mul_b(tr, rs);
        float64 ui = v_f32_mul_b(ti, rs);
        for (int i = 0; i < n; i++)
        {
            float64 s_i = lane_bcast(dscale, i);
            float64 ir = lane_bcast(invr, i);
            float64 ii = lane_bcast(invi, i);
            float64 bsr = v_f32_mul_b(br[i], s_i);
            float64 bsi = v_f32_mul_b(bi[i], s_i);
            float64 xr = v_f32_mac_b(v_f32_sub_b(0.0f, bsi), ii, v_f32_mul_b(bsr, ir));
            float64 xi = v_f32_mac_b(bsi, ir, v_f32_mul_b(bsr, ii));
            xr = v_f32_sel_grt_f32_b(bad, 0.0f, nanv, xr);
            xi = v_f32_sel_grt_f32_b(bad, 0.0f, nanv, xi);
            int5 cr = {0, i, bt, 0, 0};
            int5 ci = {0, n + i, bt, 0, 0};
            v_f32_st_tnsr_partial(cr, X, xr, (char)(m - 1), 0);
            v_f32_st_tnsr_partial(ci, X, xi, (char)(m - 1), 0);
        }
        // phase = (-1)^swaps prod_k u_k, renormalized once (0 when a pivot is zero)
        float64 phr = sv, phi = 0.0f;
        for (int k = 0; k < n; k++)
        {
            float64 uk_r = lane_bcast(ur, k);
            float64 uk_i = lane_bcast(ui, k);
            float64 nr = v_f32_mac_b(v_f32_sub_b(0.0f, phi), uk_i, v_f32_mul_b(phr, uk_r));
            float64 ni = v_f32_mac_b(phi, uk_r, v_f32_mul_b(phr, uk_i));
            phr = nr;
            phi = ni;
        }
        float64 pn = lut_free_rsqrt(v_f32_mac_b(phi, phi, v_f32_mul_b(phr, phr)));
        phr = v_f32_mul_b(phr, pn);
        phi = v_f32_mul_b(phi, pn);

        dvr = v_f32_sel_grt_f32_b(bad, 0.0f, nanv, dvr);
        dvi = v_f32_sel_grt_f32_b(bad, 0.0f, nanv, dvi);
        absd = v_f32_sel_grt_f32_b(bad, 0.0f, nanv, absd);
        int5 d0 = {0, 0, bt, 0, 0};
        int5 d1 = {0, 1, bt, 0, 0};
        int5 d2 = {0, 2, bt, 0, 0};
        v_f32_st_tnsr_partial(d0, D, dvr, (char)(n - 1), 0);
        v_f32_st_tnsr_partial(d1, D, dvi, (char)(n - 1), 0);
        v_f32_st_tnsr_partial(d2, D, absd, (char)(n - 1), 0);
        float64 sw = v_f32_sel_eq_u32_b(lane, 0u, phr, v_f32_sel_eq_u32_b(lane, 1u, phi,
                                        v_f32_sel_eq_u32_b(lane, 2u, infov, sv)));
        sw = v_f32_sel_grt_f32_b(bad, 0.0f, nanv, sw);
        int5 cs = {0, bt, 0, 0, 0};
        v_f32_st_tnsr_partial(cs, S, sw, 3, 0);
    }
}
