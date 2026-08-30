#pragma once

#include "DebugStateRaw.h"

#include <cstddef>
#include <cstdint>

namespace HpShark {

static constexpr auto MaxReferenceBlocks = 256;
static constexpr auto AdditionalGlobalSyncSpace = 128 * (MaxReferenceBlocks + 1);
static constexpr uint32_t CohortBarrierCount = 2u;
static constexpr uint32_t CohortBarrierStrideUInt32 = 32u;
static constexpr uint32_t CohortBarrierArrivalOffsetUInt32 = 0u;
static constexpr uint32_t CohortBarrierGenerationOffsetUInt32 = 2u;
static_assert(CohortBarrierCount * CohortBarrierStrideUInt32 * sizeof(uint32_t) <=
              AdditionalGlobalSyncSpace * sizeof(uint64_t));
static constexpr auto AdditionalGlobalDebugCountSpace =
    DebugGlobalState ? 1024u * MaxReferenceBlocks * sizeof(DebugGlobalCountRaw) : 0u;
static constexpr auto AdditionalGlobalChecksumSpace = DebugChecksums ? 1024 * 1024 : 0;

static constexpr auto AdditionalGlobalSyncSpaceOffset = 0;
static constexpr auto AdditionalDebugCountsOffset =
    AdditionalGlobalSyncSpaceOffset + AdditionalGlobalSyncSpace;
static constexpr auto AdditionalChecksumsOffset =
    AdditionalDebugCountsOffset + AdditionalGlobalDebugCountSpace;

static constexpr auto AdditionalUInt64Global =
    AdditionalGlobalSyncSpace + AdditionalGlobalDebugCountSpace + AdditionalGlobalChecksumSpace;

// Release reference iterations use one 4096-word plane for non-NR or two 2048-word planes for NR
// (32 KiB total), plus one 128-word twiddle cache reused phase-wise by forward/pointwise/inverse.
// One 4096-word plane or two 2048-word planes, plus the first eleven stages of twiddles.
static constexpr size_t ReferenceReleaseSharedMemory = 48u * 1024u;
static_assert(ReferenceReleaseSharedMemory == 49152u);

static constexpr size_t ReferenceDefaultSharedMemory = ReferenceReleaseSharedMemory;

template <class SharkFloatParams>
constexpr int32_t
CalculateReferenceSharedMemorySize()
{
    return static_cast<int32_t>(ReferenceDefaultSharedMemory);
}

} // namespace HpShark
