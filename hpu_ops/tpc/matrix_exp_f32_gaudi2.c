/* Batched matrix exponential exp(A) and Frechet derivative L_exp(A^T, G) on Gaudi2 TPC for n <= 32.
 *
 * Uses Higham Padé-13 scaling-and-squaring with normalized coefficients c_k = b_k / b_0 <= 1.0, the scaled
 * 1-norm at most THETA = 2 (not Higham's double-precision theta_13 = 5.37), and Gauss-Jordan with partial pivoting.
 * At 5.37 the denominator q = V - U cancels for a dominant positive eigenvalue (V / (V - U) = 75 at x = 5), and
 * float32 lost 66-75x against torch's float32 (diag(5, .5, -1): 5.4e-6 against 8.2e-8; gradients 20-170x, code
 * review #8). At 2 the cancellation is about 3x; the truncation error of Padé-13 stays far below float32's unit
 * roundoff, and the cost is up to 2 more squarings (none for ||A||_1 <= 2).
 *
 * Tensors (TPC geometry, dim0 innermost; torch [batch, n, n] row-major):
 *   A [n, n, batch]  f32
 *   G [n, n, batch]  f32 (under -DFRECHET only)
 *   Y [n, n, batch]  f32 (exp(A), or grad_A = L_exp(A^T, G) under -DFRECHET)
 *
 * 100% LUT-free (no v_f32_reduce_*, v_reciprocal_f32, sqrt, rsqrt) -> 80 KB VLM.
 * Forward uses 6 x 8 KB = 48 KB; -DFRECHET uses 8 x 8 KB = 64 KB.
 *
 * Scaling: s = the number of halvings of ||A||_1 (of A^T under -DFRECHET) until it is <= THETA, at most S_MAX
 * (0 when the norm is not finite; the result is then non-finite, from the inputs). S_MAX = 126, as 2^-126 is the
 * smallest normal float: up to ||A||_1 = THETA * 2^126 = 1.7e38 the scaled norm is <= THETA, past it (to FLT_MAX)
 * <= 4. s lives in a vector (all lanes equal);
 * the squarings are masked by it. S_FIXED masked squarings always run (cheap; ||A||_1 <= 1.3e5 needs no more). For
 * s > S_FIXED the scalar loop needs s as a scalar: it is stored to Y (overwritten at the end) and read back by one
 * scalar load after the Pade solve, behind an ASO barrier: Habana's TPC coherency table does not keep a vector
 * st_tnsr and a later scalar ld_g of the same address coherent in hardware (code review #11). A read that is not a number in (S_FIXED, S_MAX] is ignored, so the kernel
 * can never run fewer than S_FIXED squarings, nor more than S_MAX.
 */
#define MAXN 32
#define THETA 2.0f
#define S_MAX 126
#define S_FIXED 16

// The squaring count for the scalar loop: s read back from Y (see above), else S_FIXED. The ASO orders the read
// after the vector store of s (st_tnsr -> scalar ld_g is not coherent in hardware), as gj_large_f32_gaudi2.c does.
static int squaring_trips(tensor Y_out, int bt)
{
    int5 c0 = {0, 0, bt, 0, 0};
    aso(SW_VPU);
    float sv = s_f32_ld_g((__global__ float*)gen_addr(c0, Y_out));
    return (sv > (float)S_FIXED && sv <= (float)S_MAX) ? (int)sv : S_FIXED;
}

__local__ float64 a_mat[MAXN];
__local__ float64 u_mat[MAXN];
__local__ float64 v_mat[MAXN];
__local__ float64 x2_mat[MAXN];

#ifdef FRECHET
__local__ float64 e_mat[MAXN];
__local__ float64 m2_mat[MAXN];
__local__ float64 lu_mat[MAXN];
__local__ float64 lv_mat[MAXN];
#else
__local__ float64 x4_mat[MAXN];
__local__ float64 x6_mat[MAXN];
#endif

static inline float64 reduce_max_half32(float64 p)
{
    p = v_f32_max_b(p, v_f32_mov_dual_group_all_b(p, 0xFFFFFFFF, 1, 0, 3, 2, SW_MDG_ALL, p));
    p = v_f32_max_b(p, as_float64(v_element_shift_xor_i32(as_int64(p), 8)));
    p = v_f32_max_b(p, as_float64(v_element_shift_xor_i32(as_int64(p), 4)));
    p = v_f32_max_b(p, as_float64(v_element_shift_xor_i32(as_int64(p), 2)));
    p = v_f32_max_b(p, as_float64(v_element_shift_xor_i32(as_int64(p), 1)));
    return p;
}

static inline float64 reduce_min_half32(float64 p)
{
    p = v_f32_min_b(p, v_f32_mov_dual_group_all_b(p, 0xFFFFFFFF, 1, 0, 3, 2, SW_MDG_ALL, p));
    p = v_f32_min_b(p, as_float64(v_element_shift_xor_i32(as_int64(p), 8)));
    p = v_f32_min_b(p, as_float64(v_element_shift_xor_i32(as_int64(p), 4)));
    p = v_f32_min_b(p, as_float64(v_element_shift_xor_i32(as_int64(p), 2)));
    p = v_f32_min_b(p, as_float64(v_element_shift_xor_i32(as_int64(p), 1)));
    return p;
}

static inline float64 lane_bcast32(float64 v, uint64 lane, int k)
{
    (void)lane;
    return as_float64(v_broadcast_element_i32(as_int64(v), k));
}

static inline float64 rcp_newton(float64 x)
{
    float64 ax = v_f32_abs_b(x);
    int64 ix = as_int64(ax);
    int64 magic = 0x7EF311C3;
    float64 y = as_float64(v_i32_sub_b(magic, ix));
    y = v_f32_mul_b(y, v_f32_sub_b(2.0f, v_f32_mul_b(ax, y)));
    y = v_f32_mul_b(y, v_f32_sub_b(2.0f, v_f32_mul_b(ax, y)));
    y = v_f32_mul_b(y, v_f32_sub_b(2.0f, v_f32_mul_b(ax, y)));
    y = v_f32_mul_b(y, v_f32_sub_b(2.0f, v_f32_mul_b(ax, y)));
    // one step in FMA error form: e = 1 - ax y exactly rounded once, y += y e. The plain steps leave 1 ulp
    // (rcp(1) = 0.99999994), which s squarings amplify: exp(diag(-1e8, 0))[1, 1] was 0.1353 (Codex review #8)
    y = v_f32_mac_b(y, v_f32_mac_b(v_f32_sub_b(0.0f, ax), y, 1.0f), y);
    y = v_f32_sel_less_f32_b(x, 0.0f, v_f32_sub_b(0.0f, y), y);
    return v_f32_sel_eq_f32_b(x, 0.0f, 0.0f, y);
}

static void matmul_nn(int n, uint64 lane, __local__ float64* P, __local__ float64* Q, __local__ float64* R)
{
    (void)lane;
    int i = 0;
    for (; i + 3 < n; i += 4)
    {
        float64 p0 = P[i], p1 = P[i + 1], p2 = P[i + 2], p3 = P[i + 3];
        float64 a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        for (int k = 0; k < n; k++)
        {
            float64 qk = Q[k];
            a0 = v_f32_mac_b(as_float64(v_broadcast_element_i32(as_int64(p0), k)), qk, a0);
            a1 = v_f32_mac_b(as_float64(v_broadcast_element_i32(as_int64(p1), k)), qk, a1);
            a2 = v_f32_mac_b(as_float64(v_broadcast_element_i32(as_int64(p2), k)), qk, a2);
            a3 = v_f32_mac_b(as_float64(v_broadcast_element_i32(as_int64(p3), k)), qk, a3);
        }
        R[i] = a0;
        R[i + 1] = a1;
        R[i + 2] = a2;
        R[i + 3] = a3;
    }
    for (; i < n; i++)
    {
        float64 pi = P[i];
        float64 acc = 0.0f;
        for (int k = 0; k < n; k++)
        {
            acc = v_f32_mac_b(as_float64(v_broadcast_element_i32(as_int64(pi), k)), Q[k], acc);
        }
        R[i] = acc;
    }
}

#ifdef FRECHET
static void matmul_add_nn(int n, uint64 lane, __local__ float64* P1, __local__ float64* Q1, __local__ float64* P2,
                          __local__ float64* Q2, __local__ float64* R)
{
    (void)lane;
    int i = 0;
    for (; i + 3 < n; i += 4)
    {
        float64 p10 = P1[i], p11 = P1[i + 1], p12 = P1[i + 2], p13 = P1[i + 3];
        float64 p20 = P2[i], p21 = P2[i + 1], p22 = P2[i + 2], p23 = P2[i + 3];
        float64 a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        for (int k = 0; k < n; k++)
        {
            float64 q1k = Q1[k];
            float64 q2k = Q2[k];
            a0 = v_f32_mac_b(as_float64(v_broadcast_element_i32(as_int64(p10), k)), q1k, a0);
            a1 = v_f32_mac_b(as_float64(v_broadcast_element_i32(as_int64(p11), k)), q1k, a1);
            a2 = v_f32_mac_b(as_float64(v_broadcast_element_i32(as_int64(p12), k)), q1k, a2);
            a3 = v_f32_mac_b(as_float64(v_broadcast_element_i32(as_int64(p13), k)), q1k, a3);
            a0 = v_f32_mac_b(as_float64(v_broadcast_element_i32(as_int64(p20), k)), q2k, a0);
            a1 = v_f32_mac_b(as_float64(v_broadcast_element_i32(as_int64(p21), k)), q2k, a1);
            a2 = v_f32_mac_b(as_float64(v_broadcast_element_i32(as_int64(p22), k)), q2k, a2);
            a3 = v_f32_mac_b(as_float64(v_broadcast_element_i32(as_int64(p23), k)), q2k, a3);
        }
        R[i] = a0;
        R[i + 1] = a1;
        R[i + 2] = a2;
        R[i + 3] = a3;
    }
    for (; i < n; i++)
    {
        float64 p1 = P1[i];
        float64 p2 = P2[i];
        float64 acc = 0.0f;
        for (int k = 0; k < n; k++)
        {
            acc = v_f32_mac_b(as_float64(v_broadcast_element_i32(as_int64(p1), k)), Q1[k], acc);
            acc = v_f32_mac_b(as_float64(v_broadcast_element_i32(as_int64(p2), k)), Q2[k], acc);
        }
        R[i] = acc;
    }
}
#endif

static void gj_solve_inplace(int n, uint64 lane, __local__ float64* M, __local__ float64* B)
{
    for (int k = 0; k < n; k++)
    {
        float64 col = 0.0f;
        for (int i = k; i < n; i++)
        {
            col = v_f32_sel_eq_u32_b(lane, (unsigned)i, lane_bcast32(M[i], lane, k), col);
        }
        float64 cand = v_f32_abs_b(col);
        cand = v_f32_sel_less_u32_b(lane, (unsigned)k, -1.0f, cand);
        cand = v_f32_sel_geq_u32_b(lane, (unsigned)n, -1.0f, cand);
        float64 mx = reduce_max_half32(cand);
        float64 lane_f = v_f32_sel_eq_f32_b(cand, mx, v_convert_u32_to_f32_b(lane), 100.0f);
        uint64 p = v_convert_f32_to_u32_b(reduce_min_half32(lane_f));
        p = v_u32_sel_less_u32_b(p, (unsigned)k, (unsigned)k, p);
        p = v_u32_sel_geq_u32_b(p, (unsigned)n, (unsigned)k, p);

        float64 mk = M[k], bk = B[k], mp = mk, bp = bk;
        for (int i = k + 1; i < n; i++)
        {
            mp = v_f32_sel_eq_u32_b(p, (unsigned)i, M[i], mp);
            bp = v_f32_sel_eq_u32_b(p, (unsigned)i, B[i], bp);
        }
        for (int i = k + 1; i < n; i++)
        {
            M[i] = v_f32_sel_eq_u32_b(p, (unsigned)i, mk, M[i]);
            B[i] = v_f32_sel_eq_u32_b(p, (unsigned)i, bk, B[i]);
        }
        float64 pk = lane_bcast32(mp, lane, k);
        float64 inv = rcp_newton(pk);
        mk = v_f32_mul_b(mp, inv);
        bk = v_f32_mul_b(bp, inv);
        M[k] = mk;
        B[k] = bk;
        for (int i = 0; i < n; i++)
        {
            if (i == k)
            {
                continue;
            }
            float64 nf = v_f32_sub_b(0.0f, lane_bcast32(M[i], lane, k));
            M[i] = v_f32_mac_b(mk, nf, M[i]);
            B[i] = v_f32_mac_b(bk, nf, B[i]);
        }
    }
}

#ifdef FRECHET
static void horner_step(int n, uint64 lane, float c, __local__ float64* P, __local__ float64* LP,
                        __local__ float64* SCR_P, __local__ float64* SCR_LP)
{
    matmul_nn(n, lane, x2_mat, P, SCR_P);
    matmul_add_nn(n, lane, x2_mat, LP, m2_mat, P, SCR_LP);
    for (int i = 0; i < n; i++)
    {
        P[i] = v_f32_add_b(SCR_P[i], v_f32_sel_eq_u32_b(lane, (unsigned)i, c, 0.0f));
        LP[i] = SCR_LP[i];
    }
}

static void horner_frechet(int n, uint64 lane, __local__ float64* P, __local__ float64* LP, __local__ float64* SCR_P,
                           __local__ float64* SCR_LP, float c0, float c1, float c2, float c3, float c4, float c5,
                           float c6)
{
    for (int i = 0; i < n; i++)
    {
        P[i] = v_f32_sel_eq_u32_b(lane, (unsigned)i, c0, 0.0f);
        LP[i] = 0.0f;
    }
    horner_step(n, lane, c1, P, LP, SCR_P, SCR_LP);
    horner_step(n, lane, c2, P, LP, SCR_P, SCR_LP);
    horner_step(n, lane, c3, P, LP, SCR_P, SCR_LP);
    horner_step(n, lane, c4, P, LP, SCR_P, SCR_LP);
    horner_step(n, lane, c5, P, LP, SCR_P, SCR_LP);
    horner_step(n, lane, c6, P, LP, SCR_P, SCR_LP);
}
#endif

#ifdef FRECHET
void main(tensor A_in, tensor G_in, tensor Y_out)
#else
void main(tensor A_in, tensor Y_out)
#endif
{
    const int n = get_dim_size(A_in, 0);
    const uint64 lane = read_lane_id_4b_b();
    const int5 off = get_index_space_offset();
    const int5 size = get_index_space_size();

    const float c1 = 0.5f;
    const float c2 = 0.12f;
    const float c3 = 11.0f / 600.0f;
    const float c4 = 11.0f / 5520.0f;
    const float c5 = 3.0f / 18400.0f;
    const float c6 = 1.0f / 96600.0f;
    const float c7 = 1.0f / 1932000.0f;
    const float c8 = 1.0f / 48944000.0f;
    const float c9 = 1.0f / 1585785600.0f;
    const float c10 = 1.0f / 67395888000.0f;
    const float c11 = 1.0f / 3953892096000.0f;
    const float c12 = 1.0f / 355850288640000.0f;
    const float c13 = 1.0f / 64764752532480000.0f;

    for (int bt = off[0]; bt < off[0] + size[0]; bt++)
    {
        float64 col_sum = 0.0f;
        for (int i = 0; i < n; i++)
        {
            int5 c = {0, i, bt, 0, 0};
            float64 row = v_f32_sel_geq_u32_b(lane, (unsigned)n, 0.0f, v_f32_ld_tnsr_b(c, A_in));
#ifdef FRECHET
            x2_mat[i] = row;
            e_mat[i] = v_f32_sel_geq_u32_b(lane, (unsigned)n, 0.0f, v_f32_ld_tnsr_b(c, G_in));
#else
            a_mat[i] = row;
            col_sum = v_f32_add_b(col_sum, v_f32_abs_b(row));
#endif
        }

#ifdef FRECHET
        // Transpose A_in into a_mat so a_mat holds A^T, and accumulate ||A^T||_1.
        for (int i = 0; i < n; i++)
        {
            float64 row = 0.0f;
            for (int j = 0; j < n; j++)
            {
                row = v_f32_sel_eq_u32_b(lane, (unsigned)j, lane_bcast32(x2_mat[j], lane, i), row);
            }
            a_mat[i] = row;
            col_sum = v_f32_add_b(col_sum, v_f32_abs_b(row));
        }
#endif

        float64 norm1 = reduce_max_half32(col_sum);
        // finite: below the Inf bit pattern (the literal 3.4e38f is under FLT_MAX: a finite norm in between was
        // taken for Inf, 0 halvings, NaN; Codex review #10)
        float64 finite_ok = v_f32_sel_less_u32_b(v_u32_and_b(as_uint64(norm1), 0x7FFFFFFFu), 0x7F800000u, 1.0f, 0.0f);
        float64 cur_norm = v_f32_mul_b(norm1, finite_ok);
        float64 scale = 1.0f;
        float64 s_vec = 0.0f;
        for (int k = 0; k < S_MAX; k++)
        {
            float64 need = v_f32_sel_grt_f32_b(cur_norm, THETA, 1.0f, 0.0f);
            scale = v_f32_sel_eq_f32_b(need, 1.0f, v_f32_mul_b(scale, 0.5f), scale);
            cur_norm = v_f32_sel_eq_f32_b(need, 1.0f, v_f32_mul_b(cur_norm, 0.5f), cur_norm);
            s_vec = v_f32_sel_eq_f32_b(need, 1.0f, v_f32_add_b(s_vec, 1.0f), s_vec);
        }
        {
            int5 c0 = {0, 0, bt, 0, 0};
            v_f32_st_tnsr_partial(c0, Y_out, s_vec, 0, 0);  // s, read back by squaring_trips
        }
        for (int i = 0; i < n; i++)
        {
            a_mat[i] = v_f32_mul_b(a_mat[i], scale);
#ifdef FRECHET
            e_mat[i] = v_f32_mul_b(e_mat[i], scale);
#endif
        }

#ifdef FRECHET
        matmul_nn(n, lane, a_mat, a_mat, x2_mat);
        matmul_add_nn(n, lane, a_mat, e_mat, e_mat, a_mat, m2_mat);

        horner_frechet(n, lane, u_mat, lu_mat, v_mat, lv_mat, c13, c11, c9, c7, c5, c3, c1);
        // u_mat = A * W, lu_mat = E * W + A * LW (using v_mat, lv_mat as scratch)
        matmul_nn(n, lane, a_mat, u_mat, v_mat);
        matmul_add_nn(n, lane, e_mat, u_mat, a_mat, lu_mat, lv_mat);
        for (int i = 0; i < n; i++)
        {
            u_mat[i] = v_mat[i];
            lu_mat[i] = lv_mat[i];
        }
        // a_mat and e_mat are now free as scratch for Z, LZ
        horner_frechet(n, lane, v_mat, lv_mat, a_mat, e_mat, c12, c10, c8, c6, c4, c2, 1.0f);

        // D = V - U -> a_mat and x2_mat; N = V + U -> v_mat; L_D = LV - LU -> u_mat; L_N = LV + LU -> lv_mat
        for (int i = 0; i < n; i++)
        {
            float64 vi = v_mat[i], ui = u_mat[i];
            float64 lvi = lv_mat[i], lui = lu_mat[i];
            float64 di = v_f32_sub_b(vi, ui);
            a_mat[i] = di;
            x2_mat[i] = di;
            v_mat[i] = v_f32_add_b(vi, ui);
            u_mat[i] = v_f32_sub_b(lvi, lui);
            lv_mat[i] = v_f32_add_b(lvi, lui);
        }
        gj_solve_inplace(n, lane, a_mat, v_mat);  // v_mat = R
        // lv_mat = L_N - L_D * R (using a_mat as scratch)
        matmul_nn(n, lane, u_mat, v_mat, a_mat);
        for (int i = 0; i < n; i++)
        {
            lv_mat[i] = v_f32_sub_b(lv_mat[i], a_mat[i]);
        }
        gj_solve_inplace(n, lane, x2_mat, lv_mat);  // lv_mat = L_0

        const int trips = squaring_trips(Y_out, bt);
        for (int step = 0; step < trips; step++)
        {
            float64 active = v_f32_sel_grt_f32_b(s_vec, (float)step, 1.0f, 0.0f);
            matmul_add_nn(n, lane, v_mat, lv_mat, lv_mat, v_mat, a_mat);
            for (int i = 0; i < n; i++)
            {
                lv_mat[i] = v_f32_sel_eq_f32_b(active, 1.0f, a_mat[i], lv_mat[i]);
            }
            if (step + 1 < trips)
            {
                matmul_nn(n, lane, v_mat, v_mat, a_mat);
                for (int i = 0; i < n; i++)
                {
                    v_mat[i] = v_f32_sel_eq_f32_b(active, 1.0f, a_mat[i], v_mat[i]);
                }
            }
        }
        for (int i = 0; i < n; i++)
        {
            int5 c = {0, i, bt, 0, 0};
            v_f32_st_tnsr_partial(c, Y_out, lv_mat[i], n - 1, 0);
        }
#else
        matmul_nn(n, lane, a_mat, a_mat, x2_mat);
        matmul_nn(n, lane, x2_mat, x2_mat, x4_mat);
        matmul_nn(n, lane, x2_mat, x4_mat, x6_mat);

        for (int i = 0; i < n; i++)
        {
            float64 x2 = x2_mat[i], x4 = x4_mat[i], x6 = x6_mat[i];
            u_mat[i] = v_f32_mac_b(c13, x6, v_f32_mac_b(c11, x4, v_f32_mul_b(c9, x2)));
        }
        matmul_nn(n, lane, x6_mat, u_mat, v_mat);
        for (int i = 0; i < n; i++)
        {
            float64 x2 = x2_mat[i], x4 = x4_mat[i], x6 = x6_mat[i];
            float64 w2 = v_f32_mac_b(c7, x6, v_f32_mac_b(c5, x4, v_f32_mac_b(c3, x2, v_f32_sel_eq_u32_b(lane, (unsigned)i, c1, 0.0f))));
            v_mat[i] = v_f32_add_b(v_mat[i], w2);
        }
        matmul_nn(n, lane, a_mat, v_mat, u_mat);
        for (int i = 0; i < n; i++)
        {
            float64 x2 = x2_mat[i], x4 = x4_mat[i], x6 = x6_mat[i];
            v_mat[i] = v_f32_mac_b(c12, x6, v_f32_mac_b(c10, x4, v_f32_mul_b(c8, x2)));
        }
        matmul_nn(n, lane, x6_mat, v_mat, a_mat);
        for (int i = 0; i < n; i++)
        {
            float64 x2 = x2_mat[i], x4 = x4_mat[i], x6 = x6_mat[i];
            float64 z2 = v_f32_mac_b(c6, x6, v_f32_mac_b(c4, x4, v_f32_mac_b(c2, x2, v_f32_sel_eq_u32_b(lane, (unsigned)i, 1.0f, 0.0f))));
            float64 vi = v_f32_add_b(a_mat[i], z2);
            float64 ui = u_mat[i];
            a_mat[i] = v_f32_sub_b(vi, ui);
            v_mat[i] = v_f32_add_b(vi, ui);
        }
        gj_solve_inplace(n, lane, a_mat, v_mat);
        const int trips = squaring_trips(Y_out, bt);
        for (int step = 0; step < trips; step++)
        {
            float64 active = v_f32_sel_grt_f32_b(s_vec, (float)step, 1.0f, 0.0f);
            matmul_nn(n, lane, v_mat, v_mat, a_mat);
            for (int i = 0; i < n; i++)
            {
                v_mat[i] = v_f32_sel_eq_f32_b(active, 1.0f, a_mat[i], v_mat[i]);
            }
        }
        for (int i = 0; i < n; i++)
        {
            int5 c = {0, i, bt, 0, 0};
            v_f32_st_tnsr_partial(c, Y_out, v_mat[i], n - 1, 0);
        }
#endif
    }
}
