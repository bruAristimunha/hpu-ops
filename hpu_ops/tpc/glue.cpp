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
}

namespace {

const char* kSolve = "gj_solve_f32_gaudi2";
const char* kInv = "gj_inv_f32_gaudi2";  // the same source with -DIDENTITY_B: A alone, B is the identity
const char* kEigh = "jacobi_eigh_f32_gaudi2";
const char* kEighRefine = "jacobi_eigh_refine_f32_gaudi2";
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
constexpr uint64_t kMaxN = 64;      // MAXN of gj_solve and tri_solve
constexpr uint64_t kMaxSmallN = 32; // MAXN of jacobi_svd, matrix_exp, lu_factor and lu_solve
constexpr uint64_t kMaxHH = 64;     // MAXHH of householder_product
constexpr uint64_t kMaxEighN = 32;  // MAXN of jacobi_eigh
constexpr uint64_t kMaxLargeN = 512;  // NMAX of gj_large

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

// gj_large: input A (n, n, batch); outputs R (64 * ceil(n / 64), n, batch), the in-place inverse, and P, Q, D
// (n, batch), S (2, batch). One index-space member per matrix.
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
    const uint64_t batch = in->inputTensors[0].geometry.maxSizes[2];
    const uint64_t w = 64 * ((n + 63) / 64);
    if (!rank3(in, 1) || n <= 64 || n > kMaxLargeN || in->inputTensors[0].geometry.maxSizes[1] != n)
    {
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;
    }
    if (in->outputTensors[0].geometry.maxSizes[0] != w || in->outputTensors[0].geometry.maxSizes[1] != n)
    {
        return tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    }
    out->indexSpaceRank = 1;
    out->indexSpaceGeometry[0] = batch;
    out->preferredSplitDim = 1;

    auto& pa = out->inputTensorAccessPattern[0];
    base_pattern(pa);
    full_dim(pa, 0, n);
    full_dim(pa, 1, n);
    batch_dim(pa, 2);
    auto& pr = out->outputTensorAccessPattern[0];
    base_pattern(pr);
    full_dim(pr, 0, w);
    full_dim(pr, 1, n);
    batch_dim(pr, 2);
    for (int t = 1; t < 4; t++)
    {
        auto& pv = out->outputTensorAccessPattern[t];
        base_pattern(pv);
        full_dim(pv, 0, n);
        batch_dim(pv, 1);
    }
    auto& ps = out->outputTensorAccessPattern[4];
    base_pattern(ps);
    full_dim(ps, 0, 2);
    batch_dim(ps, 1);
    // one ELF per row width in vectors (-DVA), so the kernel's loops over a row unroll
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

}  // namespace

extern "C" {

tpc_lib_api::GlueCodeReturn GetKernelGuids(tpc_lib_api::DeviceId deviceId, uint32_t* kernelCount,
                                           tpc_lib_api::GuidInfo* guids)
{
    const uint32_t count = deviceId == tpc_lib_api::DEVICE_ID_GAUDI2 ? 18 : 0;
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
    return tpc_lib_api::GLUE_NODE_NOT_FOUND;
}

tpc_lib_api::GlueCodeReturn GetShapeInference(tpc_lib_api::DeviceId, const tpc_lib_api::ShapeInferenceParams*,
                                              tpc_lib_api::ShapeInferenceOutput*)
{
    return tpc_lib_api::GLUE_SUCCESS;
}

}  // extern "C"
