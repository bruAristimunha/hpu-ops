/* One-sided (Hestenes) Jacobi for 32 < n <= 128 on Gaudi2 TPC, one matrix per index-space member, the rows in local
 * memory (LUT-free: a bit-level reciprocal square root, so local memory keeps its 80 KB). Built once per VA =
 * ceil(n / 64) in 1..2 and per SWEEPS (-DSWEEPS: the full pass, and the short refinement pass).
 *
 * Tensors (TPC geometry, dim0 innermost; torch [batch, rows, cols] row-major), w = 64 * VA:
 *   S0 [w, n, batch]  f32: S symmetric positive definite (lanes >= n zero)
 *   U  [w, n, batch]  f32: the rows of S V for the rotation V the sweeps built: at convergence they are mutually
 *                     orthogonal, row j = lambda_j v_j with v_j an eigenvector of S, so the caller normalizes them
 * Each sweep visits every pair (p, q) once, round-robin (the circle method over an odd count M: n - 1 rotating
 * indices and n - 1 fixed when n is even; n, with a bye, when n is odd). The rotation is the Jacobi kernel's
 * (jacobi_eigh_f32_gaudi2.c): zeroes u_p . u_q; a pair already orthogonal to 1e-7 is left alone. A first version
 * kept the rows (and V) in global memory: a row stored and read back in the next round came back stale (zeros) even
 * with aso(SW_INC | SW_VPU) and aso(SW_DEC | SW_VPU) between the rounds.
 */
#ifndef VA
#define VA 2
#endif
#ifndef SWEEPS
#define SWEEPS 8
#endif
#define MAXR 128  // rows: u[MAXR * VA] = 64 KB of the 80 KB local memory (LUT-free, so the 80 KB mode)

__local__ float64 u[MAXR * VA];  // row j, vector v at u[j * VA + v]

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

// LUT-free 1/sqrt(x) for x > 0 normal: the bit-level guess and three Newton steps (float32 accuracy)
static inline float64 rsqrt_lut_free(float64 x)
{
    uint64 i = as_uint64(x);
    uint64 g = v_u32_sub_b(0x5f3759dfu, v_u32_shr_b(i, 1));
    float64 y = as_float64(g);
    float64 hx = v_f32_mul_b(-0.5f, x);
#pragma unroll
    for (int it = 0; it < 3; it++)
    {
        y = v_f32_mul_b(y, v_f32_mac_b(hx, v_f32_mul_b(y, y), 1.5f));
    }
    return y;
}

static inline float64 inv_sqrt_unit(float64 c2)
{
    // as jacobi_eigh_f32_gaudi2.c: minimax polynomial on [0.5, 1.0] and one Newton step
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

static inline void rotate(int p, int q)
{
    float64 up[VA], uq[VA];
    float64 a = 0.0f, b = 0.0f, g = 0.0f;
#pragma unroll
    for (int v = 0; v < VA; v++)
    {
        up[v] = u[p * VA + v];
        uq[v] = u[q * VA + v];
        a = v_f32_mac_b(up[v], up[v], a);
        b = v_f32_mac_b(uq[v], uq[v], b);
        g = v_f32_mac_b(up[v], uq[v], g);
    }
    float64 alpha = reduce_add_64(a);
    float64 beta = reduce_add_64(b);
    float64 gamma = reduce_add_64(g);

    float64 d = v_f32_sub_b(beta, alpha);
    float64 g2 = v_f32_add_b(gamma, gamma);
    float64 r2 = v_f32_mac_b(g2, g2, v_f32_mul_b(d, d));
    float64 sr2 = v_f32_sel_eq_f32_b(r2, 0.0f, 1.0f, r2);
    float64 inv_r = rsqrt_lut_free(sr2);
    float64 c2 = v_f32_mul_b(0.5f, v_f32_mac_b(v_f32_abs_b(d), inv_r, 1.0f));
    float64 inv_c = inv_sqrt_unit(c2);
    float64 c = v_f32_mul_b(c2, inv_c);
    float64 s = v_f32_mul_b(gamma, v_f32_mul_b(inv_r, inv_c));
    s = v_f32_sel_less_f32_b(d, 0.0f, v_f32_sub_b(0.0f, s), s);
    // converged (|gamma| <= 1e-7 sqrt(alpha beta)), or nothing to rotate: the identity
    float64 conv = v_f32_sub_b(v_f32_mul_b(1e-14f, v_f32_mul_b(alpha, beta)), v_f32_mul_b(gamma, gamma));
    c = v_f32_sel_geq_f32_b(conv, 0.0f, 1.0f, c);
    s = v_f32_sel_geq_f32_b(conv, 0.0f, 0.0f, s);
    c = v_f32_sel_eq_f32_b(r2, 0.0f, 1.0f, c);
    s = v_f32_sel_eq_f32_b(r2, 0.0f, 0.0f, s);
    float64 ns = v_f32_sub_b(0.0f, s);
#pragma unroll
    for (int v = 0; v < VA; v++)
    {
        u[p * VA + v] = v_f32_mac_b(uq[v], ns, v_f32_mul_b(up[v], c));
        u[q * VA + v] = v_f32_mac_b(up[v], s, v_f32_mul_b(uq[v], c));
    }
}

void main(tensor S0, tensor U)
{
    const int n = get_dim_size(S0, 1);
    const int5 off = get_index_space_offset();
    const int5 size = get_index_space_size();
    const int M = (n % 2 == 1) ? n : n - 1;  // odd: M rounds of (M + 1) / 2 pairs
    const int npairs = (M + 1) / 2;          // ("half" is a TPC type name)

    for (int bt = off[0]; bt < off[0] + size[0]; bt++)
    {
        for (int i = 0; i < n; i++)
        {
#pragma unroll
            for (int v = 0; v < VA; v++)
            {
                int5 c = {64 * v, i, bt, 0, 0};
                u[i * VA + v] = v_f32_ld_tnsr_b(c, S0);
            }
        }
        for (int sweep = 0; sweep < SWEEPS; sweep++)
        {
            for (int r = 0; r < M; r++)
            {
                if (M < n)  // n even: the fixed index n - 1 meets r
                {
                    rotate(r, M);
                }
                for (int k = 1; k < npairs; k++)
                {
                    int x = r + k;
                    if (x >= M) x -= M;
                    int y = r - k;
                    if (y < 0) y += M;
                    rotate(x < y ? x : y, x < y ? y : x);
                }
            }
        }
        for (int i = 0; i < n; i++)
        {
#pragma unroll
            for (int v = 0; v < VA; v++)
            {
                int5 c = {64 * v, i, bt, 0, 0};
                v_f32_st_tnsr(c, U, u[i * VA + v]);
            }
        }
    }
}
