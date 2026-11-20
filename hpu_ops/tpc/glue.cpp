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
}

namespace {

const char* kSolve = "gj_solve_f32_gaudi2";
const char* kInv = "gj_inv_f32_gaudi2";  // the same source with -DIDENTITY_B: A alone, B is the identity
const char* kEigh = "jacobi_eigh_f32_gaudi2";
const char* kEighRefine = "jacobi_eigh_refine_f32_gaudi2";
const char* kTriLower = "tri_solve_lower_f32_gaudi2";
const char* kTriUpper = "tri_solve_upper_f32_gaudi2";
const char* kTriLowerUnit = "tri_solve_lower_unit_f32_gaudi2";
const char* kTriUpperUnit = "tri_solve_upper_unit_f32_gaudi2";
constexpr uint64_t kMaxN = 64;      // MAXN of gj_solve and tri_solve
constexpr uint64_t kMaxEighN = 32;  // MAXN of jacobi_eigh

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
    if (n > kMaxN || m > kMaxN ||
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
    if (n > kMaxN || m > kMaxN ||
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

}  // namespace

extern "C" {

tpc_lib_api::GlueCodeReturn GetKernelGuids(tpc_lib_api::DeviceId deviceId, uint32_t* kernelCount,
                                           tpc_lib_api::GuidInfo* guids)
{
    const uint32_t count = deviceId == tpc_lib_api::DEVICE_ID_GAUDI2 ? 8 : 0;
    if (guids != nullptr && count)
    {
        strcpy(guids[0].name, kSolve);
        strcpy(guids[1].name, kInv);
        strcpy(guids[2].name, kEigh);
        strcpy(guids[3].name, kEighRefine);
        strcpy(guids[4].name, kTriLower);
        strcpy(guids[5].name, kTriUpper);
        strcpy(guids[6].name, kTriLowerUnit);
        strcpy(guids[7].name, kTriUpperUnit);
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
    return tpc_lib_api::GLUE_NODE_NOT_FOUND;
}

tpc_lib_api::GlueCodeReturn GetShapeInference(tpc_lib_api::DeviceId, const tpc_lib_api::ShapeInferenceParams*,
                                              tpc_lib_api::ShapeInferenceOutput*)
{
    return tpc_lib_api::GLUE_SUCCESS;
}

}  // extern "C"
