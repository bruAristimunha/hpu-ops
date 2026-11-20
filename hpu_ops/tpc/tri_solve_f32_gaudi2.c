/* Batched triangular solve A X = B on Gaudi2 TPC: forward/back substitution without pivoting,
 * one matrix per index-space member.
 *
 * Tensors (TPC geometry, dim0 innermost; torch [batch, rows, cols] row-major):
 *   A [n, n, batch]  f32     B [m, n, batch]  f32     X [m, n, batch]  f32
 * Build variants via -DUPPER (back substitution, else forward) and -DUNIT_DIAG (assume diag = 1, skip recip).
 * Row i of X is one 64-lane vector in local memory (lane j = column j), so n, m <= MAXN.
 * Opposite triangle (and diagonal under -DUNIT_DIAG) is never read; all ops are LUT-free (80 KB VLM, x[64] = 16 KB).
 */
#define MAXN 64

__local__ float64 x[MAXN];

static inline float64 lane_bcast(float64 v, int k)
{
    return as_float64(v_broadcast_element_i32(as_int64(v), k));
}

#ifndef UNIT_DIAG
static inline float64 lut_free_recip(float64 v)
{
    uint64 bits = as_uint64(v);
    uint64 sign = v_u32_and_b(bits, 0x80000000u);
    uint64 abits = v_u32_and_b(bits, 0x7FFFFFFFu);

    // FTZ-proof subnormal normalization: (abits | 1.0f) - 1.0f == |v| * 2^126
    uint64 sub_bits = as_uint64(v_f32_sub_b(as_float64(v_u32_or_b(abits, 0x3F800000u)), 1.0f));
    uint64 nbits = v_u32_sel_less_u32_b(abits, 0x00800000u, sub_bits, abits);
    uint64 e_adj = v_u32_sel_less_u32_b(abits, 0x00800000u, 126u, 0u);

    uint64 mant = v_u32_and_b(nbits, 0x007FFFFFu);
    float64 m = as_float64(v_u32_or_b(mant, 0x3F000000u));  // [0.5, 1.0)

    // Cephes quadratic guess + 3 FMA error-form NR steps (<= 1 ULP everywhere)
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

    // Biased exponent with +256 offset so arithmetic stays unsigned: [255, 532] for finite inputs
    uint64 e_biased = (ey + 382u + e_adj) - e_in;

    uint64 res_norm = v_u32_or_b(sign, v_u32_or_b(v_u32_shl_b(e_biased - 256u, 23), ymant));
    uint64 shift = v_u32_sel_less_u32_b(e_biased, 257u, 257u - e_biased, 0u);
    uint64 res_sub = v_u32_or_b(sign, v_u32_shr_b(v_u32_or_b(0x00800000u, ymant), as_int64(shift)));
    uint64 res = v_u32_sel_less_u32_b(e_biased, 257u, res_sub, res_norm);

    res = v_u32_sel_geq_u32_b(e_biased, 511u, v_u32_or_b(sign, 0x7F800000u), res);  // overflow -> +/-Inf
    res = v_u32_sel_eq_u32_b(abits, 0x7F800000u, sign, res);                         // +/-Inf -> +/-0.0f
    res = v_u32_sel_eq_u32_b(abits, 0u, v_u32_or_b(sign, 0x7F800000u), res);         // +/-0.0f -> +/-Inf
    res = v_u32_sel_grt_u32_b(abits, 0x7F800000u, bits, res);                        // NaN -> NaN
    return as_float64(res);
}
#endif

void main(tensor A, tensor B, tensor X)
{
    const int n = get_dim_size(A, 0);
    const int m = get_dim_size(B, 0);
    const uint64 lane = read_lane_id_4b_b();
    const int5 off = get_index_space_offset();
    const int5 size = get_index_space_size();

    for (int bt = off[0]; bt < off[0] + size[0]; bt++)
    {
#ifdef UPPER
        for (int i = n - 1; i >= 0; i--)
#else
        for (int i = 0; i < n; i++)
#endif
        {
            int5 c = {0, i, bt, 0, 0};
            float64 row_a = v_f32_sel_geq_u32_b(lane, (unsigned)n, 0.0f, v_f32_ld_tnsr_b(c, A));
            float64 row_b = v_f32_sel_geq_u32_b(lane, (unsigned)m, 0.0f, v_f32_ld_tnsr_b(c, B));
            float64 sum = 0.0f;
#ifdef UPPER
            for (int k = i + 1; k < n; k++)
#else
            for (int k = 0; k < i; k++)
#endif
            {
                float64 aik = lane_bcast(row_a, k);
                sum = v_f32_mac_b(aik, x[k], sum);
            }
            float64 acc = v_f32_sub_b(row_b, sum);
#ifndef UNIT_DIAG
            float64 diag = lane_bcast(row_a, i);
            float64 inv = lut_free_recip(diag);
            acc = v_f32_mul_b(acc, inv);
#endif
            x[i] = acc;
            if (m < 64)
            {
                v_f32_st_tnsr_partial(c, X, acc, (char)(m - 1), 0);
            }
            else
            {
                v_f32_st_tnsr(c, X, acc);
            }
        }
    }
}
