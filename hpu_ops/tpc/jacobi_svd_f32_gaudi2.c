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

static inline float64 rsqrt_newton(float64 x)
{
    float64 r = v_rsqrt_f32(x);
    return v_f32_mul_b(r, v_f32_mac_b(v_f32_mul_b(-0.5f, x), v_f32_mul_b(r, r), 1.5f));
}

// The rotation (c, s) of the pair with squared norms alpha, beta and dot product gamma, computed on alpha, beta,
// gamma scaled by ~1 / (alpha + beta): c and s depend on their ratios only, and r^2 = d^2 + (2 gamma)^2 and
// alpha * beta, products of squares, underflowed for rows of norm ~1e-10 (Codex review #6: two identical rows
// [0, 1e-10, 0] never rotated, rank 3 for 2). The skip rules (gamma = 0, gamma^2 <= 1e-14 alpha beta, r^2 < 1e-34)
// read the scaled values, so they are relative to the pair. The update of row p is c up - s uq as two rounded
// products (no FMA), so that with c = s two identical rows give exactly 0.
#define JACOBI_CS(alpha, beta, gamma, c, s)                                                        \
    do                                                                                             \
    {                                                                                              \
        float64 ab_ = v_f32_add_b((alpha), (beta));                                                \
        float64 k_ = rsqrt_newton(v_f32_sel_eq_f32_b(ab_, 0.0f, 1.0f, ab_));                       \
        k_ = v_f32_mul_b(k_, k_);                                                                  \
        float64 a_ = v_f32_mul_b((alpha), k_);                                                     \
        float64 b_ = v_f32_mul_b((beta), k_);                                                      \
        float64 g_ = v_f32_mul_b((gamma), k_);                                                     \
        float64 d_ = v_f32_sub_b(b_, a_);                                                          \
        float64 g2_ = v_f32_add_b(g_, g_);                                                         \
        float64 r2_ = v_f32_mac_b(g2_, g2_, v_f32_mul_b(d_, d_));                                  \
        float64 inv_r_ = rsqrt_newton(v_f32_sel_less_f32_b(r2_, 1.0e-34f, 1.0f, r2_));             \
        float64 c2_ = v_f32_mul_b(0.5f, v_f32_mac_b(v_f32_abs_b(d_), inv_r_, 1.0f));               \
        float64 inv_c_ = rsqrt_newton(c2_);                                                        \
        (c) = v_f32_mul_b(c2_, inv_c_);                                                            \
        (s) = v_f32_mul_b(v_f32_mul_b(0.5f, g2_), v_f32_mul_b(inv_r_, inv_c_));                    \
        (s) = v_f32_sel_less_f32_b(d_, 0.0f, v_f32_sub_b(0.0f, (s)), (s));                         \
        /* equal norms: exactly 45 degrees, c = |s| (two identical rows then cancel to an exact zero row) */      \
        (c) = v_f32_sel_eq_f32_b(d_, 0.0f, 0.70710677f, (c));                                      \
        (s) = v_f32_sel_eq_f32_b(d_, 0.0f, v_f32_sel_less_f32_b(g2_, 0.0f, -0.70710677f, 0.70710677f), (s)); \
        float64 gsq_ = v_f32_mul_b(g_, g_);                                                        \
        float64 tol_ = v_f32_mul_b(1.0e-14f, v_f32_mul_b(a_, b_));                                 \
        (c) = v_f32_sel_less_f32_b(tol_, gsq_, (c), 1.0f);                                         \
        (s) = v_f32_sel_less_f32_b(tol_, gsq_, (s), 0.0f);                                         \
        (c) = v_f32_sel_less_f32_b(r2_, 1.0e-34f, 1.0f, (c));                                      \
        (s) = v_f32_sel_less_f32_b(r2_, 1.0e-34f, 0.0f, (s));                                      \
        (c) = v_f32_sel_eq_f32_b((gamma), 0.0f, 1.0f, (c));                                        \
        (s) = v_f32_sel_eq_f32_b((gamma), 0.0f, 0.0f, (s));                                        \
    } while (0)

// The norm of row x (per 32-lane half) without squaring it into underflow: max|x| * ||x / max|x|||, the row
// scaled by 1 / max|x| (cw) and 1 / ||cw|| (rn): cw * rn is the unit row (Codex review #6). nrm is 0 for a zero row.
#define ROW_NORM(x, mx, cw, nrm, rn)                                                               \
    do                                                                                             \
    {                                                                                              \
        (mx) = reduce_max_half32(v_f32_abs_b(x));                                                  \
        float64 smx_ = v_f32_sel_eq_f32_b((mx), 0.0f, 1.0f, (mx));                                 \
        float64 rmx_ = v_reciprocal_f32(smx_);                                                     \
        rmx_ = v_f32_mul_b(rmx_, v_f32_sub_b(2.0f, v_f32_mul_b(smx_, rmx_)));                      \
        (cw) = v_f32_mul_b((x), rmx_);                                                             \
        float64 sq_ = reduce_add_half32(v_f32_mul_b((cw), (cw)));                                  \
        (rn) = rsqrt_newton(v_f32_sel_less_f32_b(sq_, 1.0e-36f, 1.0f, sq_));                       \
        (nrm) = v_f32_mul_b(v_f32_mul_b(sq_, (rn)), smx_);                                         \
    } while (0)

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

                    float64 c0, s0, c1, s1, c2, s2;
                    JACOBI_CS(alpha0, beta0, gamma0, c0, s0);
                    JACOBI_CS(alpha1, beta1, gamma1, c1, s1);
                    JACOBI_CS(alpha2, beta2, gamma2, c2, s2);

                    BROADCAST_CS(c0, s0);
                    BROADCAST_CS(c1, s1);
                    BROADCAST_CS(c2, s2);
                    ARR[p0] = v_f32_sub_b(v_f32_mul_b(up0, c0), v_f32_mul_b(uq0, s0));
                    ARR[q0] = v_f32_mac_b(up0, s0, v_f32_mul_b(uq0, c0));
                    ARR[p1] = v_f32_sub_b(v_f32_mul_b(up1, c1), v_f32_mul_b(uq1, s1));
                    ARR[q1] = v_f32_mac_b(up1, s1, v_f32_mul_b(uq1, c1));
                    ARR[p2] = v_f32_sub_b(v_f32_mul_b(up2, c2), v_f32_mul_b(uq2, s2));
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

                    float64 c0, s0, c1, s1;
                    JACOBI_CS(alpha0, beta0, gamma0, c0, s0);
                    JACOBI_CS(alpha1, beta1, gamma1, c1, s1);

                    BROADCAST_CS(c0, s0);
                    BROADCAST_CS(c1, s1);
                    ARR[p0] = v_f32_sub_b(v_f32_mul_b(up0, c0), v_f32_mul_b(uq0, s0));
                    ARR[q0] = v_f32_mac_b(up0, s0, v_f32_mul_b(uq0, c0));
                    ARR[p1] = v_f32_sub_b(v_f32_mul_b(up1, c1), v_f32_mul_b(uq1, s1));
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

                    float64 c, s;
                    JACOBI_CS(alpha, beta, gamma, c, s);

                    BROADCAST_CS(c, s);
                    ARR[p] = v_f32_sub_b(v_f32_mul_b(up, c), v_f32_mul_b(uq, s));
                    ARR[q] = v_f32_mac_b(up, s, v_f32_mul_b(uq, c));
                }
            }
        }

        // Compute singular values and (in full mode) normalize U columns + MGS completion
        float64 s_vec = 0.0f;
#ifdef NO_UV
        for (int j = 0; j < n; j++)
        {
            // the norm of the row scaled by 1 / max|row| (Codex review #6: the squared norm of a row of 1e-20
            // underflowed, svdvals(diag(1e20, 1)) = [1e20, 0]); 0 for a zero row
            float64 mx, cw, nrm, rn;
            ROW_NORM(w[j], mx, cw, nrm, rn);
            float64 sj = v_f32_sel_eq_f32_b(mx, 0.0f, 0.0f, v_f32_mul_b(nrm, safe_sigma));
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
        float64 nrm_vec = 0.0f;
        float64 max_nrm = 0.0f;
        for (int j = 0; j < n; j++)
        {
            // the norm of w_j scaled by 1 / max|w_j| (Codex review #6: the squared norm of a row of 1e-20
            // underflowed, svdvals(diag(1e20, 1)) = [1e20, 0]); 0 for a zero row.
            // Lanes >= 32 (v_j) keep their values; their norm is unused.
            float64 cur = uv[j];
            float64 mx, cw, nrm, rn;
            ROW_NORM(cur, mx, cw, nrm, rn);
            float64 zero = v_f32_sel_eq_f32_b(mx, 0.0f, 1.0f, 0.0f);
            nrm = v_f32_sel_eq_f32_b(zero, 1.0f, 0.0f, nrm);
            nrm_vec = v_f32_sel_eq_u32_b(lane_mod32, (unsigned)j, nrm, nrm_vec);
            max_nrm = v_f32_max_b(max_nrm, nrm);

            s_vec = v_f32_sel_eq_u32_b(lane_mod32, (unsigned)j, v_f32_mul_b(nrm, safe_sigma), s_vec);

            // u_j = cw * rn, the unit row (0 for a zeroed row, then completed below); v_j kept
            float64 u = v_f32_mul_b(cw, v_f32_sel_eq_f32_b(zero, 1.0f, 0.0f, rn));
            uv[j] = v_f32_sel_geq_u32_b(lane, 32u, cur, u);
        }

        // Vectorized 2-pass MGS completion for zero/near-zero rows, norms below 1e-5 of the largest (squared: 1e-10,
        // as before), predicated by done_vec, no scalar sync: each takes its own direction orthogonalized against the
        // rows done so far when that survives, else the basis vector of most remaining room (diag_p, the done rows'
        // room, as before). In two phases: the nonzero near-zero rows first, then the zero rows, so a zero row cannot
        // take a small row's direction (Codex review #6's [[1,0,0],[0,1e-10,0],[0,1e-10,0]] after its rotation), and a
        // row that duplicates an earlier one is re-orthogonalized against it (counting both in diag_p up front left no
        // room: [[1,0,0],[0,s,s],[0,s,s]], s <= 1e-19, gave a zero row of Vh).
        float64 thresh = v_f32_max_b(1.0e-37f, v_f32_mul_b(1.0e-5f, max_nrm));
        float64 done_vec = v_f32_sel_less_f32_b(nrm_vec, thresh, 0.0f, 1.0f);
        done_vec = v_f32_sel_geq_u32_b(lane, (unsigned)n, 0.0f, done_vec);
        float64 diag_p = v_f32_sel_less_u32_b(lane, (unsigned)m, 1.0f, -1.0f);
        for (int k = 0; k < n; k++)
        {
            float64 dk = reduce_add_half32(v_f32_sel_eq_u32_b(lane_mod32, (unsigned)k, done_vec, 0.0f));
            float64 uk = v_f32_sel_geq_u32_b(lane, (unsigned)m, 0.0f, v_f32_mul_b(uv[k], dk));
            diag_p = v_f32_sub_b(diag_p, v_f32_mul_b(uk, uk));
        }

        for (int phase = 0; phase < 2; phase++)
        {
        for (int j = 0; j < n; j++)
        {
            float64 dj = reduce_add_half32(v_f32_sel_eq_u32_b(lane_mod32, (unsigned)j, done_vec, 0.0f));
            float64 nj = reduce_add_half32(v_f32_sel_eq_u32_b(lane_mod32, (unsigned)j, nrm_vec, 0.0f));
            // active: not done, and nonzero in phase 0, zero in phase 1 (else 0, which leaves the row and diag_p)
            float64 act = v_f32_sel_eq_f32_b(dj, 0.0f, 1.0f, 0.0f);
            act = phase == 0 ? v_f32_sel_eq_f32_b(nj, 0.0f, 0.0f, act) : v_f32_sel_eq_f32_b(nj, 0.0f, act, 0.0f);
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
            float64 new_w = v_f32_sel_eq_f32_b(act, 1.0f, cand, cur_uv);
            uv[j] = v_f32_sel_geq_u32_b(lane, 32u, cur_uv, new_w);

            float64 new_uk = v_f32_sel_eq_f32_b(act, 1.0f, cand, 0.0f);
            diag_p = v_f32_sub_b(diag_p, v_f32_mul_b(new_uk, new_uk));
            done_vec = v_f32_sel_eq_u32_b(lane_mod32, (unsigned)j, v_f32_sel_eq_f32_b(act, 1.0f, 1.0f, done_vec),
                                          done_vec);
        }
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
