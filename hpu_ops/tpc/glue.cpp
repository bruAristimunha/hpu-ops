// Glue for the batched linalg TPC kernel (Gaudi2). One index-space member per matrix (batch on dim 0).
#include <cstring>

#include "gc_interface.h"
#include "tpc_kernel_lib_interface.h"

extern "C" {
extern const char _binary___gj_solve_f32_gaudi2_o_start[];
extern const char _binary___gj_solve_f32_gaudi2_o_end[];
extern const char _binary___gj_inv_f32_gaudi2_o_start[];
extern const char _binary___gj_inv_f32_gaudi2_o_end[];
extern const char _binary___jacobi_eigh_f32_gaudi2_o_start[];
extern const char _binary___jacobi_eigh_f32_gaudi2_o_end[];
extern const char _binary___tri_solve_lower_f32_gaudi2_o_start[];
extern const char _binary___tri_solve_lower_f32_gaudi2_o_end[];
extern const char _binary___tri_solve_upper_f32_gaudi2_o_start[];
extern const char _binary___tri_solve_upper_f32_gaudi2_o_end[];
extern const char _binary___tri_solve_lower_unit_f32_gaudi2_o_start[];
extern const char _binary___tri_solve_lower_unit_f32_gaudi2_o_end[];
extern const char _binary___tri_solve_upper_unit_f32_gaudi2_o_start[];
extern const char _binary___tri_solve_upper_unit_f32_gaudi2_o_end[];
extern const char _binary___jacobi_eigh_refine_f32_gaudi2_o_start[];
extern const char _binary___jacobi_eigh_refine_f32_gaudi2_o_end[];
extern const char _binary___gj_large2_f32_gaudi2_o_start[];
extern const char _binary___gj_large2_f32_gaudi2_o_end[];
extern const char _binary___gj_large3_f32_gaudi2_o_start[];
extern const char _binary___gj_large3_f32_gaudi2_o_end[];
extern const char _binary___gj_large4_f32_gaudi2_o_start[];
extern const char _binary___gj_large4_f32_gaudi2_o_end[];
extern const char _binary___gj_large5_f32_gaudi2_o_start[];
extern const char _binary___gj_large5_f32_gaudi2_o_end[];
extern const char _binary___gj_large6_f32_gaudi2_o_start[];
extern const char _binary___gj_large6_f32_gaudi2_o_end[];
extern const char _binary___gj_large7_f32_gaudi2_o_start[];
extern const char _binary___gj_large7_f32_gaudi2_o_end[];
extern const char _binary___gj_large8_f32_gaudi2_o_start[];
extern const char _binary___gj_large8_f32_gaudi2_o_end[];
extern const char _binary___gj_tall2_f32_gaudi2_o_start[];
extern const char _binary___gj_tall2_f32_gaudi2_o_end[];
extern const char _binary___gj_tall3_f32_gaudi2_o_start[];
extern const char _binary___gj_tall3_f32_gaudi2_o_end[];
extern const char _binary___gj_tall4_f32_gaudi2_o_start[];
extern const char _binary___gj_tall4_f32_gaudi2_o_end[];
extern const char _binary___gj_tall5_f32_gaudi2_o_start[];
extern const char _binary___gj_tall5_f32_gaudi2_o_end[];
extern const char _binary___gj_tall6_f32_gaudi2_o_start[];
extern const char _binary___gj_tall6_f32_gaudi2_o_end[];
extern const char _binary___gj_tall7_f32_gaudi2_o_start[];
extern const char _binary___gj_tall7_f32_gaudi2_o_end[];
extern const char _binary___gj_tall8_f32_gaudi2_o_start[];
extern const char _binary___gj_tall8_f32_gaudi2_o_end[];
extern const char _binary___jacobi_big1_f32_gaudi2_o_start[];
extern const char _binary___jacobi_big1_f32_gaudi2_o_end[];
extern const char _binary___jacobi_big1r_f32_gaudi2_o_start[];
extern const char _binary___jacobi_big1r_f32_gaudi2_o_end[];
extern const char _binary___jacobi_big2_f32_gaudi2_o_start[];
extern const char _binary___jacobi_big2_f32_gaudi2_o_end[];
extern const char _binary___jacobi_big2r_f32_gaudi2_o_start[];
extern const char _binary___jacobi_big2r_f32_gaudi2_o_end[];
extern const char _binary___jacobi_svd_f32_gaudi2_o_start[];
extern const char _binary___jacobi_svd_f32_gaudi2_o_end[];
extern const char _binary___jacobi_svd_full_f32_gaudi2_o_start[];
extern const char _binary___jacobi_svd_full_f32_gaudi2_o_end[];
extern const char _binary___jacobi_svdvals_f32_gaudi2_o_start[];
extern const char _binary___jacobi_svdvals_f32_gaudi2_o_end[];
extern const char _binary___householder_product_f32_gaudi2_o_start[];
extern const char _binary___householder_product_f32_gaudi2_o_end[];
extern const char _binary___matrix_exp_f32_gaudi2_o_start[];
extern const char _binary___matrix_exp_f32_gaudi2_o_end[];
extern const char _binary___matrix_exp_frechet_f32_gaudi2_o_start[];
extern const char _binary___matrix_exp_frechet_f32_gaudi2_o_end[];
extern const char _binary___lu_factor_f32_gaudi2_o_start[];
extern const char _binary___lu_factor_f32_gaudi2_o_end[];
extern const char _binary___lu_perm_f32_gaudi2_o_start[];
extern const char _binary___lu_perm_f32_gaudi2_o_end[];
extern const char _binary___lu_solve_f32_gaudi2_o_start[];
extern const char _binary___lu_solve_f32_gaudi2_o_end[];
extern const char _binary___cgj_solve_f32_gaudi2_o_start[];
extern const char _binary___cgj_solve_f32_gaudi2_o_end[];
extern const char _binary___cgj_inv_f32_gaudi2_o_start[];
extern const char _binary___cgj_inv_f32_gaudi2_o_end[];
extern const char _binary___jacobi_ceigh_f32_gaudi2_o_start[];
extern const char _binary___jacobi_ceigh_f32_gaudi2_o_end[];
extern const char _binary___jacobi_ceigh_refine_f32_gaudi2_o_start[];
extern const char _binary___jacobi_ceigh_refine_f32_gaudi2_o_end[];
extern const char _binary___jacobi_csvd_f32_gaudi2_o_start[];
extern const char _binary___jacobi_csvd_f32_gaudi2_o_end[];
extern const char _binary___jacobi_csvdvals_f32_gaudi2_o_start[];
extern const char _binary___jacobi_csvdvals_f32_gaudi2_o_end[];
extern const char _binary___jacobi_svd_big1_f32_gaudi2_o_start[];
extern const char _binary___jacobi_svd_big1_f32_gaudi2_o_end[];
extern const char _binary___jacobi_svd_big2_f32_gaudi2_o_start[];
extern const char _binary___jacobi_svd_big2_f32_gaudi2_o_end[];
extern const char _binary___jacobi_svdvals_big1_f32_gaudi2_o_start[];
extern const char _binary___jacobi_svdvals_big1_f32_gaudi2_o_end[];
extern const char _binary___jacobi_svdvals_big2_f32_gaudi2_o_start[];
extern const char _binary___jacobi_svdvals_big2_f32_gaudi2_o_end[];
extern const char _binary___hh_qr1_f32_gaudi2_o_start[];
extern const char _binary___hh_qr1_f32_gaudi2_o_end[];
extern const char _binary___hh_qr2_f32_gaudi2_o_start[];
extern const char _binary___hh_qr2_f32_gaudi2_o_end[];
extern const char _binary___hh_qr3_f32_gaudi2_o_start[];
extern const char _binary___hh_qr3_f32_gaudi2_o_end[];
extern const char _binary___hh_qr4_f32_gaudi2_o_start[];
extern const char _binary___hh_qr4_f32_gaudi2_o_end[];
extern const char _binary___hh_qr_r1_f32_gaudi2_o_start[];
extern const char _binary___hh_qr_r1_f32_gaudi2_o_end[];
extern const char _binary___hh_qr_r2_f32_gaudi2_o_start[];
extern const char _binary___hh_qr_r2_f32_gaudi2_o_end[];
extern const char _binary___hh_qr_r3_f32_gaudi2_o_start[];
extern const char _binary___hh_qr_r3_f32_gaudi2_o_end[];
extern const char _binary___hh_qr_r4_f32_gaudi2_o_start[];
extern const char _binary___hh_qr_r4_f32_gaudi2_o_end[];
extern const char _binary___hh_qr_ytau1_f32_gaudi2_o_start[];
extern const char _binary___hh_qr_ytau1_f32_gaudi2_o_end[];
extern const char _binary___hh_qr_ytau2_f32_gaudi2_o_start[];
extern const char _binary___hh_qr_ytau2_f32_gaudi2_o_end[];
extern const char _binary___hh_qr_ytau3_f32_gaudi2_o_start[];
extern const char _binary___hh_qr_ytau3_f32_gaudi2_o_end[];
extern const char _binary___hh_qr_ytau4_f32_gaudi2_o_start[];
extern const char _binary___hh_qr_ytau4_f32_gaudi2_o_end[];
extern const char _binary___ceig_f32_gaudi2_o_start[];
extern const char _binary___ceig_f32_gaudi2_o_end[];
extern const char _binary___ceigvals_f32_gaudi2_o_start[];
extern const char _binary___ceigvals_f32_gaudi2_o_end[];
}

namespace {

const char* kSolve = "gj_solve_f32_gaudi2";
const char* kInv = "gj_inv_f32_gaudi2";  // the same source with -DIDENTITY_B: A alone, B is the identity
const char* kEigh = "jacobi_eigh_f32_gaudi2";
const char* kEighRefine = "jacobi_eigh_refine_f32_gaudi2";
const char* kBig = "jacobi_big_f32_gaudi2";
const char* kBigRefine = "jacobi_big_refine_f32_gaudi2";
constexpr uint64_t kMaxBigN = 128;  // rows of jacobi_big (VA = 1, 2)
const char* kSvd = "jacobi_svd_f32_gaudi2";
const char* kLarge = "gj_large_f32_gaudi2";
const char* kSvdFull = "jacobi_svd_full_f32_gaudi2";
const char* kSvdvals = "jacobi_svdvals_f32_gaudi2";
const char* kHouseholder = "householder_product_f32_gaudi2";
const char* kExpm = "matrix_exp_f32_gaudi2";
const char* kExpmFrechet = "matrix_exp_frechet_f32_gaudi2";
const char* kLu = "lu_factor_f32_gaudi2";
const char* kLuPerm = "lu_perm_f32_gaudi2";  // the same source with -DPERM_FROM_PIVOTS
const char* kLuSolve = "lu_solve_f32_gaudi2";
const char* kTriLower = "tri_solve_lower_f32_gaudi2";
const char* kTriUpper = "tri_solve_upper_f32_gaudi2";
const char* kTriLowerUnit = "tri_solve_lower_unit_f32_gaudi2";
const char* kTriUpperUnit = "tri_solve_upper_unit_f32_gaudi2";
const char* kCSolve = "cgj_solve_f32_gaudi2";  // complex64 Gauss-Jordan, planar re/im rows
const char* kCInv = "cgj_inv_f32_gaudi2";      // the same source with -DIDENTITY_B
const char* kCEigh = "jacobi_ceigh_f32_gaudi2";
const char* kCEighRefine = "jacobi_ceigh_refine_f32_gaudi2";  // -DREFINE_PASS
const char* kCSvd = "jacobi_csvd_f32_gaudi2";
const char* kCSvdvals = "jacobi_csvdvals_f32_gaudi2";  // -DNO_UV
const char* kCEig = "ceig_f32_gaudi2";
const char* kCEigvals = "ceigvals_f32_gaudi2";  // -DNO_VECTORS
constexpr uint64_t kMaxComplexN = 32;  // MAXN of the complex kernels (n complex columns, 2n planar rows)
constexpr uint64_t kMaxN = 64;      // MAXN of gj_solve and tri_solve
constexpr uint64_t kMaxSmallN = 32; // MAXN of jacobi_svd, matrix_exp, lu_factor and lu_solve
constexpr uint64_t kMaxHH = 64;     // MAXHH of householder_product
constexpr uint64_t kMaxEighN = 32;  // MAXN of jacobi_eigh
constexpr uint64_t kMaxLargeN = 512;  // NMAX of gj_large
constexpr uint64_t kMaxTallM = 2048;  // rows of a tall gj_large panel (-DTALL: VR = 32)
const char* kSvdBig = "jacobi_svd_big_f32_gaudi2";
const char* kSvdvalsBig = "jacobi_svdvals_big_f32_gaudi2";
const char* kHhQr = "hh_qr_f32_gaudi2";
const char* kHhQrR = "hh_qr_r_f32_gaudi2";
const char* kHhQrYtau = "hh_qr_ytau_f32_gaudi2";
constexpr uint64_t kMaxBigSvdN = 64;
constexpr uint64_t kMaxBigSvdM = 128;
constexpr uint64_t kMaxBigSvdvalsN = 128;
constexpr uint64_t kMaxHhQrN = 64;
constexpr uint64_t kMaxHhQrM = 256;

void base_pattern(tpc_lib_api::TensorAccessPattern& p)
{
    for (int d = 0; d < 5; d++)
    {
        p.mapping[d].indexSpaceDim = 0;
        p.mapping[d].a = 0;
        p.mapping[d].start_b = 0;
        p.mapping[d].end_b = 0;
    }
}

void full_dim(tpc_lib_api::TensorAccessPattern& p, int d, uint64_t size)
{
    p.mapping[d].indexSpaceDim = 0;
    p.mapping[d].allRequired = true;
    p.mapping[d].a = 0;
    p.mapping[d].start_b = 0;
    p.mapping[d].end_b = size - 1;
}

void batch_dim(tpc_lib_api::TensorAccessPattern& p, int d)
{
    p.mapping[d].indexSpaceDim = 0;
    p.mapping[d].allRequired = true;
    p.mapping[d].a = 1;
    p.mapping[d].start_b = 0;
    p.mapping[d].end_b = 0;
}

// Every tensor of these kernels is [cols, rows, batch] (the batch flattened by the Python side): a 4-D or 5-D tensor
// would have its extra dimensions silently ignored by the kernel's coordinates, so it is refused. (Fewer dims are
// kept: a graph may drop a trailing size-1 batch.)
bool rank3(const tpc_lib_api::HabanaKernelParams* in, unsigned inputs)
{
    for (unsigned i = 0; i < inputs; i++)
    {
        if (in->inputTensors[i].geometry.dims > 3)
        {
            return false;
        }
    }
    return true;
}

tpc_lib_api::GlueCodeReturn copy_elf(tpc_lib_api::HabanaKernelInstantiation* out, const char* start, const char* end)
{
    const unsigned isa = static_cast<unsigned>(end - start);
    if (out->kernel.elfSize < isa)
    {
        out->kernel.elfSize = isa;
        return tpc_lib_api::GLUE_INSUFFICIENT_ELF_BUFFER;
    }
    out->kernel.elfSize = isa;
    memcpy(out->kernel.kernelElf, start, isa);
    return tpc_lib_api::GLUE_SUCCESS;
}

tpc_lib_api::GlueCodeReturn solve_defs(tpc_lib_api::HabanaKernelParams* in, tpc_lib_api::HabanaKernelInstantiation* out,
                                       bool identity)
{
    const unsigned inputs = identity ? 1 : 2;
    if (in->inputTensorNr != inputs)
    {
        in->inputTensorNr = inputs;
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_COUNT;
    }
    if (in->outputTensorNr != 3)
    {
        in->outputTensorNr = 3;
        return tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    }
    for (unsigned i = 0; i < inputs; i++)
    {
        if (in->inputTensors[i].geometry.dataType != tpc_lib_api::DATA_F32)
        {
            return tpc_lib_api::GLUE_INCOMPATIBLE_DATA_TYPE;
        }
    }
    const uint64_t n = in->inputTensors[0].geometry.maxSizes[0];
    const uint64_t m = identity ? n : in->inputTensors[1].geometry.maxSizes[0];
    const uint64_t batch = in->inputTensors[0].geometry.maxSizes[2];
    if (!rank3(in, inputs) || n > kMaxN || m > kMaxN ||
        in->inputTensors[0].geometry.maxSizes[1] != n ||
        (!identity && (in->inputTensors[1].geometry.maxSizes[1] != n ||
                       in->inputTensors[1].geometry.maxSizes[2] != batch)))
    {
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;
    }
    out->indexSpaceRank = 1;
    out->indexSpaceGeometry[0] = batch;
    out->preferredSplitDim = 1;  // (preferredSplitDim - 1) = index-space dim 0

    auto& pa = out->inputTensorAccessPattern[0];
    base_pattern(pa);
    full_dim(pa, 0, n);
    full_dim(pa, 1, n);
    batch_dim(pa, 2);
    if (!identity)
    {
        auto& pb = out->inputTensorAccessPattern[1];
        base_pattern(pb);
        full_dim(pb, 0, m);
        full_dim(pb, 1, n);
        batch_dim(pb, 2);
    }
    auto& px = out->outputTensorAccessPattern[0];
    base_pattern(px);
    full_dim(px, 0, m);
    full_dim(px, 1, n);
    batch_dim(px, 2);
    auto& pd = out->outputTensorAccessPattern[1];
    base_pattern(pd);
    full_dim(pd, 0, n);
    batch_dim(pd, 1);
    auto& ps = out->outputTensorAccessPattern[2];
    base_pattern(ps);
    full_dim(ps, 0, 2);
    batch_dim(ps, 1);

    return identity ? copy_elf(out, _binary___gj_inv_f32_gaudi2_o_start, _binary___gj_inv_f32_gaudi2_o_end)
                    : copy_elf(out, _binary___gj_solve_f32_gaudi2_o_start, _binary___gj_solve_f32_gaudi2_o_end);
}

// Complex Gauss-Jordan: A [n, 2n, batch] (rows 0..n-1 the real parts, n..2n-1 the imaginary parts), B and X
// [m, 2n, batch], D [n, 3, batch] (pivot re, im, abs), S [4, batch] (phase re, im, info, sign of the permutation).
tpc_lib_api::GlueCodeReturn csolve_defs(tpc_lib_api::HabanaKernelParams* in,
                                        tpc_lib_api::HabanaKernelInstantiation* out, bool identity)
{
    const unsigned inputs = identity ? 1 : 2;
    if (in->inputTensorNr != inputs)
    {
        in->inputTensorNr = inputs;
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_COUNT;
    }
    if (in->outputTensorNr != 3)
    {
        in->outputTensorNr = 3;
        return tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    }
    for (unsigned i = 0; i < inputs; i++)
    {
        if (in->inputTensors[i].geometry.dataType != tpc_lib_api::DATA_F32)
        {
            return tpc_lib_api::GLUE_INCOMPATIBLE_DATA_TYPE;
        }
    }
    const uint64_t n = in->inputTensors[0].geometry.maxSizes[0];
    const uint64_t m = identity ? n : in->inputTensors[1].geometry.maxSizes[0];
    const uint64_t batch = in->inputTensors[0].geometry.maxSizes[2];
    if (!rank3(in, inputs) || n > kMaxComplexN || m > kMaxComplexN ||
        in->inputTensors[0].geometry.maxSizes[1] != 2 * n ||
        (!identity && (in->inputTensors[1].geometry.maxSizes[1] != 2 * n ||
                       in->inputTensors[1].geometry.maxSizes[2] != batch)))
    {
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;
    }
    out->indexSpaceRank = 1;
    out->indexSpaceGeometry[0] = batch;
    out->preferredSplitDim = 1;

    auto& pa = out->inputTensorAccessPattern[0];
    base_pattern(pa);
    full_dim(pa, 0, n);
    full_dim(pa, 1, 2 * n);
    batch_dim(pa, 2);
    if (!identity)
    {
        auto& pb = out->inputTensorAccessPattern[1];
        base_pattern(pb);
        full_dim(pb, 0, m);
        full_dim(pb, 1, 2 * n);
        batch_dim(pb, 2);
    }
    auto& px = out->outputTensorAccessPattern[0];
    base_pattern(px);
    full_dim(px, 0, m);
    full_dim(px, 1, 2 * n);
    batch_dim(px, 2);
    auto& pd = out->outputTensorAccessPattern[1];
    base_pattern(pd);
    full_dim(pd, 0, n);
    full_dim(pd, 1, 3);
    batch_dim(pd, 2);
    auto& ps = out->outputTensorAccessPattern[2];
    base_pattern(ps);
    full_dim(ps, 0, 4);
    batch_dim(ps, 1);

    return identity ? copy_elf(out, _binary___cgj_inv_f32_gaudi2_o_start, _binary___cgj_inv_f32_gaudi2_o_end)
                    : copy_elf(out, _binary___cgj_solve_f32_gaudi2_o_start, _binary___cgj_solve_f32_gaudi2_o_end);
}

void full3(tpc_lib_api::TensorAccessPattern& p, uint64_t d0, uint64_t d1)
{
    base_pattern(p);
    full_dim(p, 0, d0);
    full_dim(p, 1, d1);
    batch_dim(p, 2);
}

void full2(tpc_lib_api::TensorAccessPattern& p, uint64_t d0)
{
    base_pattern(p);
    full_dim(p, 0, d0);
    batch_dim(p, 1);
}

tpc_lib_api::GlueCodeReturn check_io(tpc_lib_api::HabanaKernelParams* in, unsigned inputs, unsigned outputs)
{
    if (in->inputTensorNr != inputs)
    {
        in->inputTensorNr = inputs;
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_COUNT;
    }
    if (in->outputTensorNr != outputs)
    {
        in->outputTensorNr = outputs;
        return tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    }
    for (unsigned i = 0; i < inputs; i++)
    {
        if (in->inputTensors[i].geometry.dataType != tpc_lib_api::DATA_F32)
        {
            return tpc_lib_api::GLUE_INCOMPATIBLE_DATA_TYPE;
        }
    }
    return rank3(in, inputs) ? tpc_lib_api::GLUE_SUCCESS : tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;
}

// Complex Hermitian Jacobi. Pass 1: A [n, 2n, batch] -> V, H [n, 2n, batch]; refine: H, V1 [n, 2n, batch] -> W [n,
// batch], V [n, 2n, batch]. Planar rows (re then im).
tpc_lib_api::GlueCodeReturn ceigh_defs(tpc_lib_api::HabanaKernelParams* in, tpc_lib_api::HabanaKernelInstantiation* out,
                                       bool refine)
{
    const unsigned inputs = refine ? 2 : 1;
    auto status = check_io(in, inputs, 2);
    if (status != tpc_lib_api::GLUE_SUCCESS)
    {
        return status;
    }
    const uint64_t n = in->inputTensors[0].geometry.maxSizes[0];
    const uint64_t batch = in->inputTensors[0].geometry.maxSizes[2];
    if (n > kMaxComplexN || in->inputTensors[0].geometry.maxSizes[1] != 2 * n ||
        (refine && (in->inputTensors[1].geometry.maxSizes[0] != n || in->inputTensors[1].geometry.maxSizes[1] != 2 * n ||
                    in->inputTensors[1].geometry.maxSizes[2] != batch)))
    {
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;
    }
    out->indexSpaceRank = 1;
    out->indexSpaceGeometry[0] = batch;
    out->preferredSplitDim = 1;
    for (unsigned i = 0; i < inputs; i++)
    {
        full3(out->inputTensorAccessPattern[i], n, 2 * n);
    }
    if (refine)
    {
        full2(out->outputTensorAccessPattern[0], n);
    }
    else
    {
        full3(out->outputTensorAccessPattern[0], n, 2 * n);
    }
    full3(out->outputTensorAccessPattern[1], n, 2 * n);
    return refine ? copy_elf(out, _binary___jacobi_ceigh_refine_f32_gaudi2_o_start,
                             _binary___jacobi_ceigh_refine_f32_gaudi2_o_end)
                  : copy_elf(out, _binary___jacobi_ceigh_f32_gaudi2_o_start, _binary___jacobi_ceigh_f32_gaudi2_o_end);
}

// Complex one-sided Jacobi SVD: A_t [big, 2k, batch] (k <= big <= 32) -> Wn [big, 2k, batch], S [k, batch], J [k, 2k,
// batch]; svdvals: S alone.
tpc_lib_api::GlueCodeReturn csvd_defs(tpc_lib_api::HabanaKernelParams* in, tpc_lib_api::HabanaKernelInstantiation* out,
                                      bool vals)
{
    auto status = check_io(in, 1, vals ? 1 : 3);
    if (status != tpc_lib_api::GLUE_SUCCESS)
    {
        return status;
    }
    const uint64_t big = in->inputTensors[0].geometry.maxSizes[0];
    const uint64_t rows = in->inputTensors[0].geometry.maxSizes[1];
    const uint64_t k = rows / 2;
    const uint64_t batch = in->inputTensors[0].geometry.maxSizes[2];
    if (big > kMaxComplexN || rows % 2 || k > big || k == 0)
    {
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;
    }
    out->indexSpaceRank = 1;
    out->indexSpaceGeometry[0] = batch;
    out->preferredSplitDim = 1;
    full3(out->inputTensorAccessPattern[0], big, 2 * k);
    if (vals)
    {
        full2(out->outputTensorAccessPattern[0], k);
        return copy_elf(out, _binary___jacobi_csvdvals_f32_gaudi2_o_start, _binary___jacobi_csvdvals_f32_gaudi2_o_end);
    }
    full3(out->outputTensorAccessPattern[0], big, 2 * k);
    full2(out->outputTensorAccessPattern[1], k);
    full3(out->outputTensorAccessPattern[2], k, 2 * k);
    return copy_elf(out, _binary___jacobi_csvd_f32_gaudi2_o_start, _binary___jacobi_csvd_f32_gaudi2_o_end);
}

// General complex64 eig / eigvals (MPS recipe): A [n, 2n, batch] -> W [n, 2, batch], V [n, 2n, batch], S [2, batch]
// (or W, S only under -DNO_VECTORS).
tpc_lib_api::GlueCodeReturn ceig_defs(tpc_lib_api::HabanaKernelParams* in, tpc_lib_api::HabanaKernelInstantiation* out,
                                      bool vals)
{
    auto status = check_io(in, 1, vals ? 2 : 3);
    if (status != tpc_lib_api::GLUE_SUCCESS)
    {
        return status;
    }
    const uint64_t n = in->inputTensors[0].geometry.maxSizes[0];
    const uint64_t batch = in->inputTensors[0].geometry.maxSizes[2];
    if (n < 1 || n > kMaxComplexN || in->inputTensors[0].geometry.maxSizes[1] != 2 * n)
    {
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;
    }
    out->indexSpaceRank = 1;
    out->indexSpaceGeometry[0] = batch;
    out->preferredSplitDim = 1;
    full3(out->inputTensorAccessPattern[0], n, 2 * n);
    if (vals)
    {
        full3(out->outputTensorAccessPattern[0], n, 2);
        full2(out->outputTensorAccessPattern[1], 2);
        return copy_elf(out, _binary___ceigvals_f32_gaudi2_o_start, _binary___ceigvals_f32_gaudi2_o_end);
    }
    full3(out->outputTensorAccessPattern[0], n, 2);
    full3(out->outputTensorAccessPattern[1], n, 2 * n);
    full2(out->outputTensorAccessPattern[2], 2);
    return copy_elf(out, _binary___ceig_f32_gaudi2_o_start, _binary___ceig_f32_gaudi2_o_end);
}

tpc_lib_api::GlueCodeReturn eigh_defs(tpc_lib_api::HabanaKernelParams* in, tpc_lib_api::HabanaKernelInstantiation* out,
                                       bool refine)
{
    const unsigned inputs = refine ? 2 : 1;
    if (in->inputTensorNr != inputs)
    {
        in->inputTensorNr = inputs;
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_COUNT;
    }
    if (in->outputTensorNr != 2)
    {
        in->outputTensorNr = 2;
        return tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    }
    for (unsigned i = 0; i < inputs; i++)
    {
        if (in->inputTensors[i].geometry.dataType != tpc_lib_api::DATA_F32)
        {
            return tpc_lib_api::GLUE_INCOMPATIBLE_DATA_TYPE;
        }
    }
    const uint64_t n = in->inputTensors[0].geometry.maxSizes[0];
    const uint64_t batch = in->inputTensors[0].geometry.maxSizes[2];
    if (n > kMaxEighN || in->inputTensors[0].geometry.maxSizes[1] != n ||
        (refine && (in->inputTensors[1].geometry.maxSizes[0] != n || in->inputTensors[1].geometry.maxSizes[1] != n ||
                    in->inputTensors[1].geometry.maxSizes[2] != batch)))
    {
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;
    }
    out->indexSpaceRank = 1;
    out->indexSpaceGeometry[0] = batch;
    out->preferredSplitDim = 1;
    auto& pa = out->inputTensorAccessPattern[0];
    base_pattern(pa);
    full_dim(pa, 0, n);
    full_dim(pa, 1, n);
    batch_dim(pa, 2);
    if (refine)
    {
        auto& pv1 = out->inputTensorAccessPattern[1];
        base_pattern(pv1);
        full_dim(pv1, 0, n);
        full_dim(pv1, 1, n);
        batch_dim(pv1, 2);
    }
    auto& pw = out->outputTensorAccessPattern[0];
    base_pattern(pw);
    full_dim(pw, 0, n);
    batch_dim(pw, 1);
    auto& pv = out->outputTensorAccessPattern[1];
    base_pattern(pv);
    full_dim(pv, 0, n);
    full_dim(pv, 1, n);
    batch_dim(pv, 2);
    return refine ? copy_elf(out, _binary___jacobi_eigh_refine_f32_gaudi2_o_start,
                             _binary___jacobi_eigh_refine_f32_gaudi2_o_end)
                  : copy_elf(out, _binary___jacobi_eigh_f32_gaudi2_o_start, _binary___jacobi_eigh_f32_gaudi2_o_end);
}

// jacobi_big: input S0 (w, n, batch), w = 64 * ceil(n / 64), 32 < n <= 128; output U of the same shape. One
// index-space member per matrix; one ELF per VA and pass (8 sweeps, the refinement's 3).
tpc_lib_api::GlueCodeReturn big_defs(tpc_lib_api::HabanaKernelParams* in, tpc_lib_api::HabanaKernelInstantiation* out,
                                     bool refine)
{
    if (in->inputTensorNr != 1)
    {
        in->inputTensorNr = 1;
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_COUNT;
    }
    if (in->outputTensorNr != 1)
    {
        in->outputTensorNr = 1;
        return tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    }
    if (in->inputTensors[0].geometry.dataType != tpc_lib_api::DATA_F32)
    {
        return tpc_lib_api::GLUE_INCOMPATIBLE_DATA_TYPE;
    }
    const uint64_t w = in->inputTensors[0].geometry.maxSizes[0];
    const uint64_t n = in->inputTensors[0].geometry.maxSizes[1];
    const uint64_t batch = in->inputTensors[0].geometry.maxSizes[2];
    if (n <= kMaxEighN || n > kMaxBigN || w != 64 * ((n + 63) / 64))
    {
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;
    }
    out->indexSpaceRank = 1;
    out->indexSpaceGeometry[0] = batch;
    out->preferredSplitDim = 1;
    auto& pi = out->inputTensorAccessPattern[0];
    base_pattern(pi);
    full_dim(pi, 0, w);
    full_dim(pi, 1, n);
    batch_dim(pi, 2);
    auto& po = out->outputTensorAccessPattern[0];
    base_pattern(po);
    full_dim(po, 0, w);
    full_dim(po, 1, n);
    batch_dim(po, 2);
    if (w == 64)
    {
        return refine ? copy_elf(out, _binary___jacobi_big1r_f32_gaudi2_o_start, _binary___jacobi_big1r_f32_gaudi2_o_end)
                      : copy_elf(out, _binary___jacobi_big1_f32_gaudi2_o_start, _binary___jacobi_big1_f32_gaudi2_o_end);
    }
    return refine ? copy_elf(out, _binary___jacobi_big2r_f32_gaudi2_o_start, _binary___jacobi_big2r_f32_gaudi2_o_end)
                  : copy_elf(out, _binary___jacobi_big2_f32_gaudi2_o_start, _binary___jacobi_big2_f32_gaudi2_o_end);
}

tpc_lib_api::GlueCodeReturn tri_solve_defs(tpc_lib_api::HabanaKernelParams* in,
                                           tpc_lib_api::HabanaKernelInstantiation* out,
                                           bool upper, bool unit)
{
    if (in->inputTensorNr != 2)
    {
        in->inputTensorNr = 2;
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_COUNT;
    }
    if (in->outputTensorNr != 1)
    {
        in->outputTensorNr = 1;
        return tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    }
    for (unsigned i = 0; i < 2; i++)
    {
        if (in->inputTensors[i].geometry.dataType != tpc_lib_api::DATA_F32)
        {
            return tpc_lib_api::GLUE_INCOMPATIBLE_DATA_TYPE;
        }
    }
    const uint64_t n = in->inputTensors[0].geometry.maxSizes[0];
    const uint64_t m = in->inputTensors[1].geometry.maxSizes[0];
    const uint64_t batch = in->inputTensors[0].geometry.maxSizes[2];
    if (!rank3(in, 2) || n > kMaxN || m > kMaxN ||
        in->inputTensors[0].geometry.maxSizes[1] != n ||
        in->inputTensors[1].geometry.maxSizes[1] != n ||
        in->inputTensors[1].geometry.maxSizes[2] != batch)
    {
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;
    }
    out->indexSpaceRank = 1;
    out->indexSpaceGeometry[0] = batch;
    out->preferredSplitDim = 1;

    auto& pa = out->inputTensorAccessPattern[0];
    base_pattern(pa);
    full_dim(pa, 0, n);
    full_dim(pa, 1, n);
    batch_dim(pa, 2);

    auto& pb = out->inputTensorAccessPattern[1];
    base_pattern(pb);
    full_dim(pb, 0, m);
    full_dim(pb, 1, n);
    batch_dim(pb, 2);

    auto& px = out->outputTensorAccessPattern[0];
    base_pattern(px);
    full_dim(px, 0, m);
    full_dim(px, 1, n);
    batch_dim(px, 2);

    if (upper)
    {
        return unit ? copy_elf(out, _binary___tri_solve_upper_unit_f32_gaudi2_o_start,
                               _binary___tri_solve_upper_unit_f32_gaudi2_o_end)
                    : copy_elf(out, _binary___tri_solve_upper_f32_gaudi2_o_start,
                               _binary___tri_solve_upper_f32_gaudi2_o_end);
    }
    return unit ? copy_elf(out, _binary___tri_solve_lower_unit_f32_gaudi2_o_start,
                           _binary___tri_solve_lower_unit_f32_gaudi2_o_end)
                : copy_elf(out, _binary___tri_solve_lower_f32_gaudi2_o_start,
                           _binary___tri_solve_lower_f32_gaudi2_o_end);
}

tpc_lib_api::GlueCodeReturn svd_defs(tpc_lib_api::HabanaKernelParams* in, tpc_lib_api::HabanaKernelInstantiation* out,
                                      bool no_uv, bool full_u)
{
    if (in->inputTensorNr != 1)
    {
        in->inputTensorNr = 1;
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_COUNT;
    }
    const unsigned outputs = no_uv ? 1 : 3;
    if (in->outputTensorNr != outputs)
    {
        in->outputTensorNr = outputs;
        return tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    }
    if (in->inputTensors[0].geometry.dataType != tpc_lib_api::DATA_F32)
    {
        return tpc_lib_api::GLUE_INCOMPATIBLE_DATA_TYPE;
    }
    const uint64_t m = in->inputTensors[0].geometry.maxSizes[0];
    const uint64_t n = in->inputTensors[0].geometry.maxSizes[1];
    const uint64_t batch = in->inputTensors[0].geometry.maxSizes[2];
    if (!rank3(in, 1) || m < 1 || n < 1 || m > kMaxSmallN || n > kMaxSmallN || m < n || batch < 1)
    {
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;
    }
    out->indexSpaceRank = 1;
    out->indexSpaceGeometry[0] = batch;
    out->preferredSplitDim = 1;

    auto& pa = out->inputTensorAccessPattern[0];
    base_pattern(pa);
    full_dim(pa, 0, m);
    full_dim(pa, 1, n);
    batch_dim(pa, 2);

    if (no_uv)
    {
        auto& ps = out->outputTensorAccessPattern[0];
        base_pattern(ps);
        full_dim(ps, 0, n);
        batch_dim(ps, 1);
        return copy_elf(out, _binary___jacobi_svdvals_f32_gaudi2_o_start, _binary___jacobi_svdvals_f32_gaudi2_o_end);
    }

    auto& pu = out->outputTensorAccessPattern[0];
    base_pattern(pu);
    full_dim(pu, 0, m);
    full_dim(pu, 1, full_u ? m : n);
    batch_dim(pu, 2);
    auto& ps = out->outputTensorAccessPattern[1];
    base_pattern(ps);
    full_dim(ps, 0, n);
    batch_dim(ps, 1);
    auto& pv = out->outputTensorAccessPattern[2];
    base_pattern(pv);
    full_dim(pv, 0, n);
    full_dim(pv, 1, n);
    batch_dim(pv, 2);
    return full_u
               ? copy_elf(out, _binary___jacobi_svd_full_f32_gaudi2_o_start, _binary___jacobi_svd_full_f32_gaudi2_o_end)
               : copy_elf(out, _binary___jacobi_svd_f32_gaudi2_o_start, _binary___jacobi_svd_f32_gaudi2_o_end);
}

tpc_lib_api::GlueCodeReturn householder_defs(tpc_lib_api::HabanaKernelParams* in,
                                             tpc_lib_api::HabanaKernelInstantiation* out)
{
    if (in->inputTensorNr != 2)
    {
        in->inputTensorNr = 2;
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_COUNT;
    }
    if (in->outputTensorNr != 1)
    {
        in->outputTensorNr = 1;
        return tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    }
    for (unsigned i = 0; i < 2; i++)
    {
        if (in->inputTensors[i].geometry.dataType != tpc_lib_api::DATA_F32)
        {
            return tpc_lib_api::GLUE_INCOMPATIBLE_DATA_TYPE;
        }
    }
    const uint64_t m = in->inputTensors[0].geometry.maxSizes[0];
    const uint64_t n = in->inputTensors[0].geometry.maxSizes[1];
    const uint64_t batch = in->inputTensors[0].geometry.maxSizes[2];
    const uint64_t k = in->inputTensors[1].geometry.maxSizes[0];
    if (!rank3(in, 2) || m < 1 || n < 1 || m > kMaxHH || n > m || k < 1 || k > n || batch < 1 ||
        in->inputTensors[1].geometry.maxSizes[1] != batch)
    {
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;
    }
    out->indexSpaceRank = 1;
    out->indexSpaceGeometry[0] = batch;
    out->preferredSplitDim = 1;

    auto& pa = out->inputTensorAccessPattern[0];
    base_pattern(pa);
    full_dim(pa, 0, m);
    full_dim(pa, 1, n);
    batch_dim(pa, 2);

    auto& pt = out->inputTensorAccessPattern[1];
    base_pattern(pt);
    full_dim(pt, 0, k);
    batch_dim(pt, 1);

    auto& pq = out->outputTensorAccessPattern[0];
    base_pattern(pq);
    full_dim(pq, 0, m);
    full_dim(pq, 1, n);
    batch_dim(pq, 2);

    return copy_elf(out, _binary___householder_product_f32_gaudi2_o_start,
                    _binary___householder_product_f32_gaudi2_o_end);
}

tpc_lib_api::GlueCodeReturn expm_defs(tpc_lib_api::HabanaKernelParams* in,
                                      tpc_lib_api::HabanaKernelInstantiation* out, bool frechet)
{
    const unsigned inputs = frechet ? 2 : 1;
    if (in->inputTensorNr != inputs)
    {
        in->inputTensorNr = inputs;
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_COUNT;
    }
    if (in->outputTensorNr != 1)
    {
        in->outputTensorNr = 1;
        return tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    }
    for (unsigned i = 0; i < inputs; i++)
    {
        if (in->inputTensors[i].geometry.dataType != tpc_lib_api::DATA_F32)
        {
            return tpc_lib_api::GLUE_INCOMPATIBLE_DATA_TYPE;
        }
    }
    const uint64_t n = in->inputTensors[0].geometry.maxSizes[0];
    const uint64_t batch = in->inputTensors[0].geometry.maxSizes[2];
    if (!rank3(in, inputs) || n < 1 || n > kMaxSmallN || in->inputTensors[0].geometry.maxSizes[1] != n || batch < 1 ||
        (frechet && (in->inputTensors[1].geometry.maxSizes[0] != n ||
                     in->inputTensors[1].geometry.maxSizes[1] != n ||
                     in->inputTensors[1].geometry.maxSizes[2] != batch)))
    {
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;
    }
    out->indexSpaceRank = 1;
    out->indexSpaceGeometry[0] = batch;
    out->preferredSplitDim = 1;

    auto& pa = out->inputTensorAccessPattern[0];
    base_pattern(pa);
    full_dim(pa, 0, n);
    full_dim(pa, 1, n);
    batch_dim(pa, 2);
    if (frechet)
    {
        auto& pg = out->inputTensorAccessPattern[1];
        base_pattern(pg);
        full_dim(pg, 0, n);
        full_dim(pg, 1, n);
        batch_dim(pg, 2);
    }
    auto& py = out->outputTensorAccessPattern[0];
    base_pattern(py);
    full_dim(py, 0, n);
    full_dim(py, 1, n);
    batch_dim(py, 2);

    return frechet ? copy_elf(out, _binary___matrix_exp_frechet_f32_gaudi2_o_start,
                              _binary___matrix_exp_frechet_f32_gaudi2_o_end)
                   : copy_elf(out, _binary___matrix_exp_f32_gaudi2_o_start, _binary___matrix_exp_f32_gaudi2_o_end);
}

tpc_lib_api::GlueCodeReturn lu_defs(tpc_lib_api::HabanaKernelParams* in, tpc_lib_api::HabanaKernelInstantiation* out)
{
    if (in->inputTensorNr != 1)
    {
        in->inputTensorNr = 1;
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_COUNT;
    }
    if (in->outputTensorNr != 2)
    {
        in->outputTensorNr = 2;
        return tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    }
    if (in->inputTensors[0].geometry.dataType != tpc_lib_api::DATA_F32)
    {
        return tpc_lib_api::GLUE_INCOMPATIBLE_DATA_TYPE;
    }
    const uint64_t n = in->inputTensors[0].geometry.maxSizes[0];
    const uint64_t batch = in->inputTensors[0].geometry.maxSizes[2];
    if (!rank3(in, 1) || n < 1 || n > kMaxSmallN || in->inputTensors[0].geometry.maxSizes[1] != n || batch < 1)
    {
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;
    }
    out->indexSpaceRank = 1;
    out->indexSpaceGeometry[0] = batch;
    out->preferredSplitDim = 1;

    auto& pa = out->inputTensorAccessPattern[0];
    base_pattern(pa);
    full_dim(pa, 0, n);
    full_dim(pa, 1, n);
    batch_dim(pa, 2);
    auto& plu = out->outputTensorAccessPattern[0];
    base_pattern(plu);
    full_dim(plu, 0, n);
    full_dim(plu, 1, n);
    batch_dim(plu, 2);
    auto& ppv = out->outputTensorAccessPattern[1];
    base_pattern(ppv);
    full_dim(ppv, 0, n);
    full_dim(ppv, 1, 3);
    batch_dim(ppv, 2);
    return copy_elf(out, _binary___lu_factor_f32_gaudi2_o_start, _binary___lu_factor_f32_gaudi2_o_end);
}

tpc_lib_api::GlueCodeReturn lu_perm_defs(tpc_lib_api::HabanaKernelParams* in,
                                         tpc_lib_api::HabanaKernelInstantiation* out)
{
    if (in->inputTensorNr != 1)
    {
        in->inputTensorNr = 1;
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_COUNT;
    }
    if (in->outputTensorNr != 1)
    {
        in->outputTensorNr = 1;
        return tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    }
    if (in->inputTensors[0].geometry.dataType != tpc_lib_api::DATA_F32)
    {
        return tpc_lib_api::GLUE_INCOMPATIBLE_DATA_TYPE;
    }
    const uint64_t n = in->inputTensors[0].geometry.maxSizes[0];
    const uint64_t batch = in->inputTensors[0].geometry.maxSizes[1];
    if (!rank3(in, 1) || n < 1 || n > kMaxSmallN || batch < 1)
    {
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;
    }
    out->indexSpaceRank = 1;
    out->indexSpaceGeometry[0] = batch;
    out->preferredSplitDim = 1;
    auto& pp = out->inputTensorAccessPattern[0];
    base_pattern(pp);
    full_dim(pp, 0, n);
    batch_dim(pp, 1);
    auto& po = out->outputTensorAccessPattern[0];
    base_pattern(po);
    full_dim(po, 0, n);
    batch_dim(po, 1);
    return copy_elf(out, _binary___lu_perm_f32_gaudi2_o_start, _binary___lu_perm_f32_gaudi2_o_end);
}

tpc_lib_api::GlueCodeReturn lu_solve_defs(tpc_lib_api::HabanaKernelParams* in,
                                          tpc_lib_api::HabanaKernelInstantiation* out)
{
    if (in->inputTensorNr != 3)
    {
        in->inputTensorNr = 3;
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_COUNT;
    }
    if (in->outputTensorNr != 1)
    {
        in->outputTensorNr = 1;
        return tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    }
    for (unsigned i = 0; i < 3; i++)
    {
        if (in->inputTensors[i].geometry.dataType != tpc_lib_api::DATA_F32)
        {
            return tpc_lib_api::GLUE_INCOMPATIBLE_DATA_TYPE;
        }
    }
    const uint64_t n = in->inputTensors[0].geometry.maxSizes[0];
    const uint64_t batch = in->inputTensors[0].geometry.maxSizes[2];
    const uint64_t m = in->inputTensors[2].geometry.maxSizes[0];
    if (!rank3(in, 3) || n < 1 || n > kMaxSmallN || m < 1 || m > kMaxSmallN || batch < 1 || in->inputTensors[0].geometry.maxSizes[1] != n ||
        in->inputTensors[1].geometry.maxSizes[0] != n || in->inputTensors[1].geometry.maxSizes[1] != batch ||
        in->inputTensors[2].geometry.maxSizes[1] != n || in->inputTensors[2].geometry.maxSizes[2] != batch)
    {
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;
    }
    out->indexSpaceRank = 1;
    out->indexSpaceGeometry[0] = batch;
    out->preferredSplitDim = 1;
    auto& pl = out->inputTensorAccessPattern[0];
    base_pattern(pl);
    full_dim(pl, 0, n);
    full_dim(pl, 1, n);
    batch_dim(pl, 2);
    auto& pp = out->inputTensorAccessPattern[1];
    base_pattern(pp);
    full_dim(pp, 0, n);
    batch_dim(pp, 1);
    auto& pb = out->inputTensorAccessPattern[2];
    base_pattern(pb);
    full_dim(pb, 0, m);
    full_dim(pb, 1, n);
    batch_dim(pb, 2);
    auto& px = out->outputTensorAccessPattern[0];
    base_pattern(px);
    full_dim(px, 0, m);
    full_dim(px, 1, n);
    batch_dim(px, 2);
    return copy_elf(out, _binary___lu_solve_f32_gaudi2_o_start, _binary___lu_solve_f32_gaudi2_o_end);
}

// gj_large: input A (n, m, batch), m = n (square) or n < m <= 2048 (a tall panel: the -DTALL build); outputs R
// (64 * ceil(n / 64), m, batch), the in-place inverse, P and D (n, batch), Q (m, batch), S (2, batch). One
// index-space member per matrix.
tpc_lib_api::GlueCodeReturn large_defs(tpc_lib_api::HabanaKernelParams* in, tpc_lib_api::HabanaKernelInstantiation* out)
{
    if (in->inputTensorNr != 1)
    {
        in->inputTensorNr = 1;
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_COUNT;
    }
    if (in->outputTensorNr != 5)
    {
        in->outputTensorNr = 5;
        return tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    }
    if (in->inputTensors[0].geometry.dataType != tpc_lib_api::DATA_F32)
    {
        return tpc_lib_api::GLUE_INCOMPATIBLE_DATA_TYPE;
    }
    const uint64_t n = in->inputTensors[0].geometry.maxSizes[0];
    const uint64_t m = in->inputTensors[0].geometry.maxSizes[1];
    const uint64_t batch = in->inputTensors[0].geometry.maxSizes[2];
    const uint64_t w = 64 * ((n + 63) / 64);
    const bool tall = m != n;
    if (!rank3(in, 1) || n <= 64 || n > kMaxLargeN || m < n || m > kMaxTallM)
    {
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;
    }
    if (in->outputTensors[0].geometry.maxSizes[0] != w || in->outputTensors[0].geometry.maxSizes[1] != m ||
        in->outputTensors[2].geometry.maxSizes[0] != m)
    {
        return tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    }
    out->indexSpaceRank = 1;
    out->indexSpaceGeometry[0] = batch;
    out->preferredSplitDim = 1;

    auto& pa = out->inputTensorAccessPattern[0];
    base_pattern(pa);
    full_dim(pa, 0, n);
    full_dim(pa, 1, m);
    batch_dim(pa, 2);
    auto& pr = out->outputTensorAccessPattern[0];
    base_pattern(pr);
    full_dim(pr, 0, w);
    full_dim(pr, 1, m);
    batch_dim(pr, 2);
    for (int t = 1; t < 4; t++)
    {
        auto& pv = out->outputTensorAccessPattern[t];
        base_pattern(pv);
        full_dim(pv, 0, t == 2 ? m : n);
        batch_dim(pv, 1);
    }
    auto& ps = out->outputTensorAccessPattern[4];
    base_pattern(ps);
    full_dim(ps, 0, 2);
    batch_dim(ps, 1);
    // one ELF per row width in vectors (-DVA), so the kernel's loops over a row unroll; tall panels their own
    if (tall)
    {
        switch ((n + 63) / 64)
        {
            case 2: return copy_elf(out, _binary___gj_tall2_f32_gaudi2_o_start, _binary___gj_tall2_f32_gaudi2_o_end);
            case 3: return copy_elf(out, _binary___gj_tall3_f32_gaudi2_o_start, _binary___gj_tall3_f32_gaudi2_o_end);
            case 4: return copy_elf(out, _binary___gj_tall4_f32_gaudi2_o_start, _binary___gj_tall4_f32_gaudi2_o_end);
            case 5: return copy_elf(out, _binary___gj_tall5_f32_gaudi2_o_start, _binary___gj_tall5_f32_gaudi2_o_end);
            case 6: return copy_elf(out, _binary___gj_tall6_f32_gaudi2_o_start, _binary___gj_tall6_f32_gaudi2_o_end);
            case 7: return copy_elf(out, _binary___gj_tall7_f32_gaudi2_o_start, _binary___gj_tall7_f32_gaudi2_o_end);
            default: return copy_elf(out, _binary___gj_tall8_f32_gaudi2_o_start, _binary___gj_tall8_f32_gaudi2_o_end);
        }
    }
    switch ((n + 63) / 64)
    {
        case 2: return copy_elf(out, _binary___gj_large2_f32_gaudi2_o_start, _binary___gj_large2_f32_gaudi2_o_end);
        case 3: return copy_elf(out, _binary___gj_large3_f32_gaudi2_o_start, _binary___gj_large3_f32_gaudi2_o_end);
        case 4: return copy_elf(out, _binary___gj_large4_f32_gaudi2_o_start, _binary___gj_large4_f32_gaudi2_o_end);
        case 5: return copy_elf(out, _binary___gj_large5_f32_gaudi2_o_start, _binary___gj_large5_f32_gaudi2_o_end);
        case 6: return copy_elf(out, _binary___gj_large6_f32_gaudi2_o_start, _binary___gj_large6_f32_gaudi2_o_end);
        case 7: return copy_elf(out, _binary___gj_large7_f32_gaudi2_o_start, _binary___gj_large7_f32_gaudi2_o_end);
        default: return copy_elf(out, _binary___gj_large8_f32_gaudi2_o_start, _binary___gj_large8_f32_gaudi2_o_end);
    }
}

tpc_lib_api::GlueCodeReturn svd_big_defs(tpc_lib_api::HabanaKernelParams* in,
                                          tpc_lib_api::HabanaKernelInstantiation* out, bool no_uv)
{
    auto status = check_io(in, 1, no_uv ? 1 : 3);
    if (status != tpc_lib_api::GLUE_SUCCESS)
    {
        return status;
    }
    const uint64_t m = in->inputTensors[0].geometry.maxSizes[0];
    const uint64_t n = in->inputTensors[0].geometry.maxSizes[1];
    const uint64_t batch = in->inputTensors[0].geometry.maxSizes[2];
    const uint64_t max_n = no_uv ? kMaxBigSvdvalsN : kMaxBigSvdN;
    if (m < 1 || n < 1 || m > kMaxBigSvdM || n > max_n || m < n || batch < 1)
    {
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;
    }
    out->indexSpaceRank = 1;
    out->indexSpaceGeometry[0] = batch;
    out->preferredSplitDim = 1;
    full3(out->inputTensorAccessPattern[0], m, n);
    if (no_uv)
    {
        full2(out->outputTensorAccessPattern[0], n);
        return (m <= 64) ? copy_elf(out, _binary___jacobi_svdvals_big1_f32_gaudi2_o_start,
                                    _binary___jacobi_svdvals_big1_f32_gaudi2_o_end)
                         : copy_elf(out, _binary___jacobi_svdvals_big2_f32_gaudi2_o_start,
                                    _binary___jacobi_svdvals_big2_f32_gaudi2_o_end);
    }
    full3(out->outputTensorAccessPattern[0], m, n);
    full2(out->outputTensorAccessPattern[1], n);
    full3(out->outputTensorAccessPattern[2], n, n);
    return (m <= 64) ? copy_elf(out, _binary___jacobi_svd_big1_f32_gaudi2_o_start,
                                _binary___jacobi_svd_big1_f32_gaudi2_o_end)
                     : copy_elf(out, _binary___jacobi_svd_big2_f32_gaudi2_o_start,
                                _binary___jacobi_svd_big2_f32_gaudi2_o_end);
}

// mode: 0 = (Q_t, R_t), 1 = R_t only, 2 = (YR_t, tau)
tpc_lib_api::GlueCodeReturn hh_qr_defs(tpc_lib_api::HabanaKernelParams* in,
                                        tpc_lib_api::HabanaKernelInstantiation* out, int mode)
{
    auto status = check_io(in, 1, mode == 1 ? 1 : 2);
    if (status != tpc_lib_api::GLUE_SUCCESS)
    {
        return status;
    }
    const uint64_t m = in->inputTensors[0].geometry.maxSizes[0];
    const uint64_t n = in->inputTensors[0].geometry.maxSizes[1];
    const uint64_t batch = in->inputTensors[0].geometry.maxSizes[2];
    if (m < 1 || n < 1 || m > kMaxHhQrM || n > kMaxHhQrN || batch < 1)
    {
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;
    }
    const uint64_t k = m < n ? m : n;
    out->indexSpaceRank = 1;
    out->indexSpaceGeometry[0] = batch;
    out->preferredSplitDim = 1;
    full3(out->inputTensorAccessPattern[0], m, n);
    const uint64_t va = (m + 63) / 64;
    if (mode == 1)
    {
        full3(out->outputTensorAccessPattern[0], k, n);
        switch (va)
        {
            case 1: return copy_elf(out, _binary___hh_qr_r1_f32_gaudi2_o_start, _binary___hh_qr_r1_f32_gaudi2_o_end);
            case 2: return copy_elf(out, _binary___hh_qr_r2_f32_gaudi2_o_start, _binary___hh_qr_r2_f32_gaudi2_o_end);
            case 3: return copy_elf(out, _binary___hh_qr_r3_f32_gaudi2_o_start, _binary___hh_qr_r3_f32_gaudi2_o_end);
            default: return copy_elf(out, _binary___hh_qr_r4_f32_gaudi2_o_start, _binary___hh_qr_r4_f32_gaudi2_o_end);
        }
    }
    if (mode == 2)
    {
        full3(out->outputTensorAccessPattern[0], m, n);
        full2(out->outputTensorAccessPattern[1], k);
        switch (va)
        {
            case 1: return copy_elf(out, _binary___hh_qr_ytau1_f32_gaudi2_o_start, _binary___hh_qr_ytau1_f32_gaudi2_o_end);
            case 2: return copy_elf(out, _binary___hh_qr_ytau2_f32_gaudi2_o_start, _binary___hh_qr_ytau2_f32_gaudi2_o_end);
            case 3: return copy_elf(out, _binary___hh_qr_ytau3_f32_gaudi2_o_start, _binary___hh_qr_ytau3_f32_gaudi2_o_end);
            default: return copy_elf(out, _binary___hh_qr_ytau4_f32_gaudi2_o_start, _binary___hh_qr_ytau4_f32_gaudi2_o_end);
        }
    }
    full3(out->outputTensorAccessPattern[0], m, k);
    full3(out->outputTensorAccessPattern[1], k, n);
    switch (va)
    {
        case 1: return copy_elf(out, _binary___hh_qr1_f32_gaudi2_o_start, _binary___hh_qr1_f32_gaudi2_o_end);
        case 2: return copy_elf(out, _binary___hh_qr2_f32_gaudi2_o_start, _binary___hh_qr2_f32_gaudi2_o_end);
        case 3: return copy_elf(out, _binary___hh_qr3_f32_gaudi2_o_start, _binary___hh_qr3_f32_gaudi2_o_end);
        default: return copy_elf(out, _binary___hh_qr4_f32_gaudi2_o_start, _binary___hh_qr4_f32_gaudi2_o_end);
    }
}

}  // namespace

extern "C" {

tpc_lib_api::GlueCodeReturn GetKernelGuids(tpc_lib_api::DeviceId deviceId, uint32_t* kernelCount,
                                           tpc_lib_api::GuidInfo* guids)
{
    const uint32_t count = deviceId == tpc_lib_api::DEVICE_ID_GAUDI2 ? 33 : 0;
    if (guids != nullptr && count)
    {
        strcpy(guids[0].name, kSolve);
        strcpy(guids[1].name, kInv);
        strcpy(guids[2].name, kEigh);
        strcpy(guids[3].name, kEighRefine);
        strcpy(guids[4].name, kSvd);
        strcpy(guids[5].name, kSvdvals);
        strcpy(guids[6].name, kSvdFull);
        strcpy(guids[7].name, kHouseholder);
        strcpy(guids[8].name, kExpm);
        strcpy(guids[9].name, kExpmFrechet);
        strcpy(guids[10].name, kLu);
        strcpy(guids[11].name, kLuPerm);
        strcpy(guids[12].name, kLuSolve);
        strcpy(guids[13].name, kTriLower);
        strcpy(guids[14].name, kTriUpper);
        strcpy(guids[15].name, kTriLowerUnit);
        strcpy(guids[16].name, kTriUpperUnit);
        strcpy(guids[17].name, kLarge);
        strcpy(guids[18].name, kBig);
        strcpy(guids[19].name, kBigRefine);
        strcpy(guids[20].name, kCSolve);
        strcpy(guids[21].name, kCInv);
        strcpy(guids[22].name, kCEigh);
        strcpy(guids[23].name, kCEighRefine);
        strcpy(guids[24].name, kCSvd);
        strcpy(guids[25].name, kCSvdvals);
        strcpy(guids[26].name, kSvdBig);
        strcpy(guids[27].name, kSvdvalsBig);
        strcpy(guids[28].name, kHhQr);
        strcpy(guids[29].name, kHhQrR);
        strcpy(guids[30].name, kHhQrYtau);
        strcpy(guids[31].name, kCEig);
        strcpy(guids[32].name, kCEigvals);
    }
    if (kernelCount != nullptr)
    {
        *kernelCount = count;
    }
    return tpc_lib_api::GLUE_SUCCESS;
}

tpc_lib_api::GlueCodeReturn InstantiateTpcKernel(tpc_lib_api::HabanaKernelParams* params,
                                                 tpc_lib_api::HabanaKernelInstantiation* instance)
{
    if (strcmp(params->guid.name, kSolve) == 0)
    {
        return solve_defs(params, instance, false);
    }
    if (strcmp(params->guid.name, kInv) == 0)
    {
        return solve_defs(params, instance, true);
    }
    if (strcmp(params->guid.name, kEigh) == 0)
    {
        return eigh_defs(params, instance, false);
    }
    if (strcmp(params->guid.name, kEighRefine) == 0)
    {
        return eigh_defs(params, instance, true);
    }
    if (strcmp(params->guid.name, kBig) == 0)
    {
        return big_defs(params, instance, false);
    }
    if (strcmp(params->guid.name, kBigRefine) == 0)
    {
        return big_defs(params, instance, true);
    }
    if (strcmp(params->guid.name, kSvd) == 0)
    {
        return svd_defs(params, instance, false, false);
    }
    if (strcmp(params->guid.name, kSvdFull) == 0)
    {
        return svd_defs(params, instance, false, true);
    }
    if (strcmp(params->guid.name, kSvdvals) == 0)
    {
        return svd_defs(params, instance, true, false);
    }
    if (strcmp(params->guid.name, kHouseholder) == 0)
    {
        return householder_defs(params, instance);
    }
    if (strcmp(params->guid.name, kExpm) == 0)
    {
        return expm_defs(params, instance, false);
    }
    if (strcmp(params->guid.name, kExpmFrechet) == 0)
    {
        return expm_defs(params, instance, true);
    }
    if (strcmp(params->guid.name, kLu) == 0)
    {
        return lu_defs(params, instance);
    }
    if (strcmp(params->guid.name, kLuPerm) == 0)
    {
        return lu_perm_defs(params, instance);
    }
    if (strcmp(params->guid.name, kLuSolve) == 0)
    {
        return lu_solve_defs(params, instance);
    }
    if (strcmp(params->guid.name, kTriLower) == 0)
    {
        return tri_solve_defs(params, instance, false, false);
    }
    if (strcmp(params->guid.name, kTriUpper) == 0)
    {
        return tri_solve_defs(params, instance, true, false);
    }
    if (strcmp(params->guid.name, kTriLowerUnit) == 0)
    {
        return tri_solve_defs(params, instance, false, true);
    }
    if (strcmp(params->guid.name, kTriUpperUnit) == 0)
    {
        return tri_solve_defs(params, instance, true, true);
    }
    if (strcmp(params->guid.name, kLarge) == 0)
    {
        return large_defs(params, instance);
    }
    if (strcmp(params->guid.name, kCSolve) == 0)
    {
        return csolve_defs(params, instance, false);
    }
    if (strcmp(params->guid.name, kCInv) == 0)
    {
        return csolve_defs(params, instance, true);
    }
    if (strcmp(params->guid.name, kCEigh) == 0)
    {
        return ceigh_defs(params, instance, false);
    }
    if (strcmp(params->guid.name, kCEighRefine) == 0)
    {
        return ceigh_defs(params, instance, true);
    }
    if (strcmp(params->guid.name, kCSvd) == 0)
    {
        return csvd_defs(params, instance, false);
    }
    if (strcmp(params->guid.name, kCSvdvals) == 0)
    {
        return csvd_defs(params, instance, true);
    }
    if (strcmp(params->guid.name, kSvdBig) == 0)
    {
        return svd_big_defs(params, instance, false);
    }
    if (strcmp(params->guid.name, kSvdvalsBig) == 0)
    {
        return svd_big_defs(params, instance, true);
    }
    if (strcmp(params->guid.name, kHhQr) == 0)
    {
        return hh_qr_defs(params, instance, 0);
    }
    if (strcmp(params->guid.name, kHhQrR) == 0)
    {
        return hh_qr_defs(params, instance, 1);
    }
    if (strcmp(params->guid.name, kHhQrYtau) == 0)
    {
        return hh_qr_defs(params, instance, 2);
    }
    if (strcmp(params->guid.name, kCEig) == 0)
    {
        return ceig_defs(params, instance, false);
    }
    if (strcmp(params->guid.name, kCEigvals) == 0)
    {
        return ceig_defs(params, instance, true);
    }
    return tpc_lib_api::GLUE_NODE_NOT_FOUND;
}

tpc_lib_api::GlueCodeReturn GetShapeInference(tpc_lib_api::DeviceId, const tpc_lib_api::ShapeInferenceParams*,
                                              tpc_lib_api::ShapeInferenceOutput*)
{
    return tpc_lib_api::GLUE_SUCCESS;
}

}  // extern "C"
