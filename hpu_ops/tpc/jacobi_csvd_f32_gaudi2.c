/* Batched complex64 thin SVD on Gaudi2 TPC: one-sided (Hestenes) complex Jacobi on the k rows of A_t (k <= big <= 32),
 * the structure of jacobi_svd_f32_gaudi2.c (1/max|a| scaling, 6/10/12 round-robin sweeps, the skip rule, tiny rows
 * completed by Gram-Schmidt) with the complex rotation of jacobi_cjacobi_common.h.
 *
 * Tensors (TPC geometry, dim0 innermost), planar (rows 0..k-1 real parts, k..2k-1 imaginary parts), two matrices per
 * vector (lanes 0-31, 32-63):
 *   A  [big, 2k, batch] -> Wn [big, 2k, batch], S [k, batch], J [k, 2k, batch]   (jacobi_csvd_f32_gaudi2)
 *   A  [big, 2k, batch] -> S [k, batch]                                         (-DNO_UV: jacobi_csvdvals_f32_gaudi2)
 * A_t = J^H diag(S) Wn: Wn's rows orthonormal (rows of norm^2 below max(1e-36, 1e-10 max) replaced by complex MGS
 * completions), J the accumulated rotations, unitary, S unsorted. After the sweeps J is made unitary again by two MGS
 * passes, W = J A_t recomputed, and two polish sweeps run (their rotations are near the identity): the accumulated
 * rotations otherwise drift from unitary by sqrt(#rotations) ulp. Non-finite input: NaN outputs. LUT-free; local
 * memory 6 x 8 KB of the 80 KB mode; tests/kernel_replica.py csvd_kernel replays it bit for bit.
 */
#define MAXN 32
#define SVD_TOL
#ifndef NO_UV
#define APPLY_V
#endif

__local__ float64 ur[MAXN];  // W (the rows being orthogonalized)
__local__ float64 ui[MAXN];
#ifndef NO_UV
__local__ float64 vr[MAXN];  // J
__local__ float64 vi[MAXN];
__local__ float64 ar[MAXN];  // the scaled A_t
__local__ float64 ai[MAXN];
#endif

#include "jacobi_cjacobi_common.h"

#ifndef NO_UV
static inline uint64 reduce_min_u32_half32(uint64 idx)
{
    idx = v_u32_min_b(idx, as_uint64(v_f32_mov_dual_group_all_b(
        as_float64(idx), 0xFFFFFFFF, 1, 0, 3, 2, SW_MDG_ALL, as_float64(idx))));
    idx = v_u32_min_b(idx, as_uint64(v_element_shift_xor_i32(as_int64(idx), 8)));
    idx = v_u32_min_b(idx, as_uint64(v_element_shift_xor_i32(as_int64(idx), 4)));
    idx = v_u32_min_b(idx, as_uint64(v_element_shift_xor_i32(as_int64(idx), 2)));
    idx = v_u32_min_b(idx, as_uint64(v_element_shift_xor_i32(as_int64(idx), 1)));
    return idx;
}
#endif

#ifdef NO_UV
void main(tensor A, tensor S)
#else
void main(tensor A, tensor Wn, tensor S, tensor J)
#endif
{
    const int big = get_dim_size(A, 0);
    const int k = get_dim_size(A, 1) / 2;
    const uint64 lane = read_lane_id_4b_b();
    const uint64 lane_mod32 = lane & 31u;
    const int5 off = get_index_space_offset();
    const int5 size = get_index_space_size();
    const int n_even = (k + 1) & ~1;
    const int m = n_even - 1;
    const int half_n = n_even >> 1;
    const int k_start = k & 1;
    const uint64 nan_bits = 0x7FC00000u;
    const float64 nanv = as_float64(nan_bits);
    const int sweeps = (k <= 4) ? 6 : (k <= 16) ? 10 : 12;

    for (int bt = off[0]; bt < off[0] + size[0]; bt += 2)
    {
        const int has2 = (bt + 1 < off[0] + size[0]);
        float64 bad = 0.0f, sigma = 0.0f;
        for (int i = 0; i < k; i++)
        {
            float64 xr = load_packed(A, i, bt, has2, big, lane), xi = load_packed(A, k + i, bt, has2, big, lane);
            ur[i] = xr;
            ui[i] = xi;
            bad = nonfinite(xi, nonfinite(xr, bad));
            sigma = v_f32_max_b(sigma, v_f32_max_b(v_f32_abs_b(xr), v_f32_abs_b(xi)));
        }
        bad = reduce_max_half32(bad);
        sigma = reduce_max_half32(sigma);
        float64 pre = v_f32_sel_less_f32_b(1e35f, sigma, 1e-4f, 1.0f);
        float64 safe_sigma = v_f32_sel_eq_f32_b(sigma, 0.0f, 1.0f, sigma);
        float64 inv = lut_free_recip(v_f32_mul_b(safe_sigma, pre));
        for (int i = 0; i < k; i++)
        {
            float64 xr = v_f32_mul_b(v_f32_mul_b(ur[i], pre), inv);
            float64 xi = v_f32_mul_b(v_f32_mul_b(ui[i], pre), inv);
            ur[i] = xr;
            ui[i] = xi;
#ifndef NO_UV
            ar[i] = xr;
            ai[i] = xi;
            vr[i] = v_f32_sel_eq_u32_b(lane_mod32, (unsigned)i, 1.0f, 0.0f);
            vi[i] = 0.0f;
#endif
        }
#ifdef NO_UV
        run_sweeps(m, half_n, k_start, sweeps);
#else
        run_sweeps(m, half_n, k_start, sweeps, 1);
        // J back to unitary (two MGS passes), W = J A_t, two polish sweeps
        for (int pass = 0; pass < 2; pass++)
        {
            for (int j = 0; j < k; j++)
            {
                float64 xr = vr[j], xi = vi[j];
                for (int kk = 0; kk < j; kk++)
                {
                    float64 br = vr[kk], bi = vi[kk];
                    float64 dr = reduce_add_half32(v_f32_mac_b(xi, bi, v_f32_mul_b(xr, br)));
                    float64 di = reduce_add_half32(v_f32_mac_b(v_f32_sub_b(0.0f, xr), bi, v_f32_mul_b(xi, br)));
                    float64 ndr = v_f32_sub_b(0.0f, dr), ndi = v_f32_sub_b(0.0f, di);
                    float64 nr = v_f32_mac_b(bi, di, v_f32_mac_b(br, ndr, xr));
                    xi = v_f32_mac_b(br, ndi, v_f32_mac_b(bi, ndr, xi));
                    xr = nr;
                }
                float64 rn = rsqrt_fine(reduce_add_half32(v_f32_mac_b(xi, xi, v_f32_mul_b(xr, xr))));
                vr[j] = v_f32_mul_b(xr, rn);
                vi[j] = v_f32_mul_b(xi, rn);
            }
        }
        for (int j = 0; j < k; j++)
        {
            float64 yr = 0.0f, yi = 0.0f;
            float64 wjr = vr[j], wji = vi[j];
            for (int i = 0; i < k; i++)
            {
                float64 br = bcast_half(wjr, i, lane);
                float64 bi = bcast_half(wji, i, lane);
                float64 nbi = v_f32_sub_b(0.0f, bi);
                yr = v_f32_mac_b(ai[i], nbi, v_f32_mac_b(ar[i], br, yr));
                yi = v_f32_mac_b(ar[i], bi, v_f32_mac_b(ai[i], br, yi));
            }
            ur[j] = yr;
            ui[j] = yi;
        }
        run_sweeps(m, half_n, k_start, 2, 1);
#endif
        // singular values; rows normalized (a row with norm^2 < 1e-36 gets S = 0 and is zeroed)
        float64 sv = 0.0f, sq_all = 0.0f;
        for (int j = 0; j < k; j++)
        {
            float64 xr = ur[j], xi = ui[j];
            float64 sq = reduce_add_half32(v_f32_mac_b(xi, xi, v_f32_mul_b(xr, xr)));
            sq_all = v_f32_sel_eq_u32_b(lane_mod32, (unsigned)j, sq, sq_all);
            float64 rn = rsqrt_fine(v_f32_sel_less_f32_b(sq, 1e-36f, 1.0f, sq));
            float64 sj = v_f32_sel_less_f32_b(sq, 1e-36f, 0.0f, v_f32_mul_b(v_f32_mul_b(sq, rn), safe_sigma));
            sv = v_f32_sel_eq_u32_b(lane_mod32, (unsigned)j, sj, sv);
#ifndef NO_UV
            float64 f = v_f32_sel_less_f32_b(sq, 1e-36f, 0.0f, rn);
            ur[j] = v_f32_mul_b(xr, f);
            ui[j] = v_f32_mul_b(xi, f);
#endif
        }
#ifndef NO_UV
        // complete the rows below the threshold by MGS, in order (done flags updated as it goes)
        float64 thresh = v_f32_max_b(1e-36f, v_f32_mul_b(1e-10f, reduce_max_half32(sq_all)));
        float64 done = v_f32_sel_less_f32_b(sq_all, thresh, 0.0f, 1.0f);
        float64 diag = v_f32_sel_less_u32_b(lane_mod32, (unsigned)big, 1.0f, -1.0f);
        for (int j = 0; j < k; j++)
        {
            float64 dj = bcast_half(done, j, lane);
            float64 ukr = v_f32_mul_b(ur[j], dj), uki = v_f32_mul_b(ui[j], dj);
            diag = v_f32_sub_b(diag, v_f32_mac_b(uki, uki, v_f32_mul_b(ukr, ukr)));
        }
        for (int j = 0; j < k; j++)
        {
            float64 c1r = ur[j], c1i = ui[j];
            uint64 idx = v_u32_sel_eq_f32_b(diag, reduce_max_half32(diag), lane_mod32, 0xFFFFFFFFu);
            idx = reduce_min_u32_half32(idx);
            float64 c2r = v_f32_sel_eq_u32_b(lane_mod32, idx, 1.0f, 0.0f), c2i = 0.0f;
            for (int pass = 0; pass < 2; pass++)
            {
                for (int kk = 0; kk < k; kk++)
                {
                    float64 dk = bcast_half(done, kk, lane);
                    float64 br = v_f32_mul_b(ur[kk], dk), bi = v_f32_mul_b(ui[kk], dk);
                    float64 d1r = reduce_add_half32(v_f32_mac_b(c1i, bi, v_f32_mul_b(c1r, br)));
                    float64 d1i = reduce_add_half32(v_f32_mac_b(v_f32_sub_b(0.0f, c1r), bi, v_f32_mul_b(c1i, br)));
                    float64 d2r = reduce_add_half32(v_f32_mac_b(c2i, bi, v_f32_mul_b(c2r, br)));
                    float64 d2i = reduce_add_half32(v_f32_mac_b(v_f32_sub_b(0.0f, c2r), bi, v_f32_mul_b(c2i, br)));
                    float64 n1r = v_f32_sub_b(0.0f, d1r), n1i = v_f32_sub_b(0.0f, d1i);
                    float64 n2r = v_f32_sub_b(0.0f, d2r), n2i = v_f32_sub_b(0.0f, d2i);
                    float64 t1 = v_f32_mac_b(bi, d1i, v_f32_mac_b(br, n1r, c1r));
                    c1i = v_f32_mac_b(br, n1i, v_f32_mac_b(bi, n1r, c1i));
                    c1r = t1;
                    float64 t2 = v_f32_mac_b(bi, d2i, v_f32_mac_b(br, n2r, c2r));
                    c2i = v_f32_mac_b(br, n2i, v_f32_mac_b(bi, n2r, c2i));
                    c2r = t2;
                }
            }
            float64 cn1 = reduce_add_half32(v_f32_mac_b(c1i, c1i, v_f32_mul_b(c1r, c1r)));
            float64 cn2 = reduce_add_half32(v_f32_mac_b(c2i, c2i, v_f32_mul_b(c2r, c2r)));
            float64 cr = v_f32_sel_less_f32_b(0.25f, cn1, c1r, c2r);
            float64 ci = v_f32_sel_less_f32_b(0.25f, cn1, c1i, c2i);
            float64 cn = v_f32_sel_less_f32_b(0.25f, cn1, cn1, cn2);
            float64 rn = rsqrt_fine(v_f32_sel_less_f32_b(cn, 1e-36f, 1.0f, cn));
            cr = v_f32_mul_b(cr, rn);
            ci = v_f32_mul_b(ci, rn);
            float64 dj = bcast_half(done, j, lane);
            ur[j] = v_f32_sel_eq_f32_b(dj, 0.0f, cr, ur[j]);
            ui[j] = v_f32_sel_eq_f32_b(dj, 0.0f, ci, ui[j]);
            diag = v_f32_sel_eq_f32_b(dj, 0.0f, v_f32_sub_b(diag, v_f32_mac_b(ci, ci, v_f32_mul_b(cr, cr))), diag);
            done = v_f32_sel_eq_u32_b(lane_mod32, (unsigned)j, 1.0f, done);
        }
        for (int j = 0; j < k; j++)
        {
            store_packed(Wn, j, bt, has2, big, v_f32_sel_grt_f32_b(bad, 0.0f, nanv, ur[j]));
            store_packed(Wn, k + j, bt, has2, big, v_f32_sel_grt_f32_b(bad, 0.0f, nanv, ui[j]));
            store_packed(J, j, bt, has2, k, v_f32_sel_grt_f32_b(bad, 0.0f, nanv, vr[j]));
            store_packed(J, k + j, bt, has2, k, v_f32_sel_grt_f32_b(bad, 0.0f, nanv, vi[j]));
        }
#endif
        sv = v_f32_sel_grt_f32_b(bad, 0.0f, nanv, sv);
        int5 cs0 = {0, bt, 0, 0, 0};
        v_f32_st_tnsr_partial(cs0, S, sv, k - 1, 0);
        if (has2)
        {
            int5 cs1 = {0, bt + 1, 0, 0, 0};
            v_f32_st_tnsr_partial(cs1, S, unpack_hi_half(sv), k - 1, 0);
        }
    }
}
