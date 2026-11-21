/* Batched complex64 Hermitian eigendecomposition on Gaudi2 TPC: complex one-sided (Hestenes) Jacobi, the structure of
 * jacobi_eigh_f32_gaudi2.c (shift to positive definite, round-robin sweeps, adaptive-shift refinement pass), with the
 * complex rotation of XLA's eigh_expander / MPS's Eigh kernel: rows p, q are rotated as (p, e q) with
 * e = <p, q> / |<p, q>|, which makes their inner product real, then the real rotation.
 *
 * Tensors (TPC geometry, dim0 innermost; torch [batch, rows, cols] row-major), planar: rows 0..n-1 the real parts,
 * rows n..2n-1 the imaginary parts. Two matrices per vector (lanes 0-31 and 32-63), as the real kernel.
 * Pass 1 (jacobi_ceigh_f32_gaudi2):  A [n, 2n, batch] -> V [n, 2n, batch], H [n, 2n, batch]
 *   H = the Hermitian matrix of A's lower triangle (the imaginary parts of the diagonal dropped, as LAPACK's heevd),
 *   V = rows u_j = x_j^H of S = H / sigma + 1.001 I (sigma: cabs1 Gershgorin bound) after 4/8/9/10 sweeps, normalized.
 * Pass 2 (-DREFINE_PASS, jacobi_ceigh_refine_f32_gaudi2): H, V1 -> W [n, batch], V [n, 2n, batch]
 *   T = V1 (H * prescale) V1^H, adaptive shift max(0, 0.01 - g_min / sigma_T), 2/4 sweeps, rows composed with V1,
 *   renormalized; W[j] = u_j H u_j^H (the Rayleigh quotient), unsorted. The eigenvector of W[j] is conj(row j of V).
 * Any non-finite entry of the used triangle makes the outputs NaN. LUT-free (rsqrt_fine, lut_free_recip), so local
 * memory is in 80 KB mode (6 x 8 KB used) and the arithmetic is replayable bit for bit
 * (tests/kernel_replica.py ceigh_pass1 / ceigh_pass2).
 */
#define MAXN 32

__local__ float64 ur[MAXN];
__local__ float64 ui[MAXN];
#ifdef REFINE_PASS
__local__ float64 hr[MAXN];
__local__ float64 hi[MAXN];
__local__ float64 v1r[MAXN];
__local__ float64 v1i[MAXN];
#endif

#include "jacobi_cjacobi_common.h"

#ifdef REFINE_PASS
// sum_i w[i] M_i (complex row w, planar rows M): the replica's _crow_combo order
#define CROW_COMBO(wr_, wi_, Mr, Mi, n, lane, yr, yi)                                   \
    do                                                                                  \
    {                                                                                   \
        yr = 0.0f;                                                                      \
        yi = 0.0f;                                                                      \
        for (int i_ = 0; i_ < (n); i_++)                                                \
        {                                                                               \
            float64 br_ = bcast_half((wr_), i_, (lane));                                \
            float64 bi_ = bcast_half((wi_), i_, (lane));                                \
            float64 nbi_ = v_f32_sub_b(0.0f, bi_);                                      \
            yr = v_f32_mac_b(Mi[i_], nbi_, v_f32_mac_b(Mr[i_], br_, yr));               \
            yi = v_f32_mac_b(Mr[i_], bi_, v_f32_mac_b(Mi[i_], br_, yi));                \
        }                                                                               \
    } while (0)
#endif

#ifdef REFINE_PASS
void main(tensor H, tensor V1, tensor W, tensor V)
#else
void main(tensor A, tensor V, tensor H)
#endif
{
#ifdef REFINE_PASS
    const int n = get_dim_size(H, 0);
#else
    const int n = get_dim_size(A, 0);
#endif
    const uint64 lane = read_lane_id_4b_b();
    const uint64 lane_mod32 = lane & 31u;
    const int5 off = get_index_space_offset();
    const int5 size = get_index_space_size();
    const int n_even = (n + 1) & ~1;
    const int m = n_even - 1;
    const int half_n = n_even >> 1;
    const int k_start = n & 1;
    const uint64 nan_bits = 0x7FC00000u;
    const float64 nanv = as_float64(nan_bits);

    for (int bt = off[0]; bt < off[0] + size[0]; bt += 2)
    {
        const int has2 = (bt + 1 < off[0] + size[0]);
        float64 bad = 0.0f;
        float64 max_abs = 0.0f;
#ifndef REFINE_PASS
        for (int i = 0; i < n; i++)
        {
            ur[i] = load_packed(A, i, bt, has2, n, lane);
            ui[i] = load_packed(A, n + i, bt, has2, n, lane);
        }
        // the Hermitian matrix of the lower triangle: lane j > i of row i = conj(row j, lane i); Im diag = 0
        for (int i = 0; i < n; i++)
        {
            float64 xr = ur[i], xi = ui[i];
            for (int j = i + 1; j < n; j++)
            {
                xr = v_f32_sel_eq_u32_b(lane_mod32, (unsigned)j, bcast_half(ur[j], i, lane), xr);
                xi = v_f32_sel_eq_u32_b(lane_mod32, (unsigned)j, v_f32_sub_b(0.0f, bcast_half(ui[j], i, lane)), xi);
            }
            xi = v_f32_sel_eq_u32_b(lane_mod32, (unsigned)i, 0.0f, xi);
            ur[i] = xr;
            ui[i] = xi;
            bad = nonfinite(xi, nonfinite(xr, bad));
            max_abs = v_f32_max_b(max_abs, v_f32_max_b(v_f32_abs_b(xr), v_f32_abs_b(xi)));
        }
        bad = reduce_max_half32(bad);
        for (int i = 0; i < n; i++)
        {
            store_packed(H, i, bt, has2, n, v_f32_sel_grt_f32_b(bad, 0.0f, nanv, ur[i]));
            store_packed(H, n + i, bt, has2, n, v_f32_sel_grt_f32_b(bad, 0.0f, nanv, ui[i]));
        }
        max_abs = reduce_max_half32(max_abs);
        float64 pre = prescale_of(max_abs, 0);
        float64 sigma = 0.0f;
        for (int i = 0; i < n; i++)
        {
            float64 xr = v_f32_mul_b(ur[i], pre), xi = v_f32_mul_b(ui[i], pre);
            ur[i] = xr;
            ui[i] = xi;
            sigma = v_f32_max_b(sigma, reduce_add_half32(v_f32_add_b(v_f32_abs_b(xr), v_f32_abs_b(xi))));
        }
        float64 inv_sigma = safe_inv(sigma, 1e35f);
        for (int i = 0; i < n; i++)
        {
            float64 xr = v_f32_mul_b(ur[i], inv_sigma);
            ur[i] = v_f32_sel_eq_u32_b(lane_mod32, (unsigned)i, v_f32_add_b(xr, 1.001f), xr);
            ui[i] = v_f32_mul_b(ui[i], inv_sigma);
        }
        const int sweeps = (n <= 4) ? 4 : (n <= 16) ? 8 : (n <= 22) ? 9 : 10;
        run_sweeps(m, half_n, k_start, sweeps);
        normalize_rows(n);
        for (int j = 0; j < n; j++)
        {
            store_packed(V, j, bt, has2, n, v_f32_sel_grt_f32_b(bad, 0.0f, nanv, ur[j]));
            store_packed(V, n + j, bt, has2, n, v_f32_sel_grt_f32_b(bad, 0.0f, nanv, ui[j]));
        }
#else
        for (int i = 0; i < n; i++)
        {
            float64 xr = load_packed(H, i, bt, has2, n, lane), xi = load_packed(H, n + i, bt, has2, n, lane);
            float64 yr = load_packed(V1, i, bt, has2, n, lane), yi = load_packed(V1, n + i, bt, has2, n, lane);
            hr[i] = xr;
            hi[i] = xi;
            v1r[i] = yr;
            v1i[i] = yi;
            bad = nonfinite(yi, nonfinite(yr, nonfinite(xi, nonfinite(xr, bad))));
            max_abs = v_f32_max_b(max_abs, v_f32_max_b(v_f32_abs_b(xr), v_f32_abs_b(xi)));
        }
        bad = reduce_max_half32(bad);
        max_abs = reduce_max_half32(max_abs);
        float64 pre = prescale_of(max_abs, 0);
        float64 inv_pre = prescale_of(max_abs, 1);
        for (int i = 0; i < n; i++)
        {
            hr[i] = v_f32_mul_b(hr[i], pre);
            hi[i] = v_f32_mul_b(hi[i], pre);
        }
        // T = V1 H V1^H: row j = (sum_i v_j[i] H_i) V1^H
        for (int j = 0; j < n; j++)
        {
            float64 yr, yi;
            CROW_COMBO(v1r[j], v1i[j], hr, hi, n, lane, yr, yi);
            float64 tr = 0.0f, ti = 0.0f;
            for (int l = 0; l < n; l++)
            {
                float64 xr = v1r[l], xi = v1i[l];
                tr = v_f32_sel_eq_u32_b(lane_mod32, (unsigned)l,
                                        reduce_add_half32(v_f32_mac_b(yi, xi, v_f32_mul_b(yr, xr))), tr);
                ti = v_f32_sel_eq_u32_b(lane_mod32, (unsigned)l,
                                        reduce_add_half32(v_f32_mac_b(v_f32_sub_b(0.0f, yr), xi, v_f32_mul_b(yi, xr))),
                                        ti);
            }
            ur[j] = tr;
            ui[j] = ti;
        }
        float64 sigma = 0.0f, g_min = 0.0f;
        for (int i = 0; i < n; i++)
        {
            float64 row_sum = reduce_add_half32(v_f32_add_b(v_f32_abs_b(ur[i]), v_f32_abs_b(ui[i])));
            sigma = v_f32_max_b(sigma, row_sum);
            float64 dg = bcast_half(ur[i], i, lane);
            float64 g = v_f32_sub_b(v_f32_add_b(dg, v_f32_abs_b(dg)), row_sum);
            g_min = (i == 0) ? g : v_f32_min_b(g_min, g);
        }
        float64 inv_sigma = safe_inv(sigma, 1e36f);
        float64 shift = v_f32_max_b(0.0f, v_f32_sub_b(0.01f, v_f32_mul_b(g_min, inv_sigma)));
        for (int i = 0; i < n; i++)
        {
            float64 xr = v_f32_mul_b(ur[i], inv_sigma);
            ur[i] = v_f32_sel_eq_u32_b(lane_mod32, (unsigned)i, v_f32_add_b(xr, shift), xr);
            ui[i] = v_f32_mul_b(ui[i], inv_sigma);
        }
        run_sweeps(m, half_n, k_start, (n <= 4) ? 2 : 4);
        normalize_rows(n);
        for (int j = 0; j < n; j++)  // row j = u_j V1 (only u_j is read, so in place)
        {
            float64 fr, fi;
            CROW_COMBO(ur[j], ui[j], v1r, v1i, n, lane, fr, fi);
            ur[j] = fr;
            ui[j] = fi;
        }
        normalize_rows(n);
        float64 w = 0.0f;
        for (int j = 0; j < n; j++)
        {
            float64 fr = ur[j], fi = ui[j];
            float64 yr, yi;
            CROW_COMBO(fr, fi, hr, hi, n, lane, yr, yi);
            float64 wj = v_f32_mul_b(reduce_add_half32(v_f32_mac_b(yi, fi, v_f32_mul_b(yr, fr))), inv_pre);
            w = v_f32_sel_eq_u32_b(lane_mod32, (unsigned)j, wj, w);
            store_packed(V, j, bt, has2, n, v_f32_sel_grt_f32_b(bad, 0.0f, nanv, fr));
            store_packed(V, n + j, bt, has2, n, v_f32_sel_grt_f32_b(bad, 0.0f, nanv, fi));
        }
        w = v_f32_sel_grt_f32_b(bad, 0.0f, nanv, w);
        int5 cw0 = {0, bt, 0, 0, 0};
        v_f32_st_tnsr_partial(cw0, W, w, n - 1, 0);
        if (has2)
        {
            int5 cw1 = {0, bt + 1, 0, 0, 0};
            v_f32_st_tnsr_partial(cw1, W, unpack_hi_half(w), n - 1, 0);
        }
#endif
    }
}
