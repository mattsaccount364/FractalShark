#pragma once

#include <cstdint>

namespace HpShark {

static constexpr auto MaxReferenceBlocks = 256;
static constexpr auto AdditionalGlobalSyncSpace = 128 * (MaxReferenceBlocks + 1);
static constexpr auto AdditionalGlobalDebugCountSpace = DebugGlobalState ? 1024 * 1024 : 0;
static constexpr auto AdditionalGlobalChecksumSpace = DebugChecksums ? 1024 * 1024 : 0;

static constexpr auto AdditionalGlobalSyncSpaceOffset = 0;
static constexpr auto AdditionalDebugCountsOffset =
    AdditionalGlobalSyncSpaceOffset + AdditionalGlobalSyncSpace;
static constexpr auto AdditionalChecksumsOffset =
    AdditionalDebugCountsOffset + AdditionalGlobalDebugCountSpace;

static constexpr auto AdditionalUInt64Global =
    AdditionalGlobalSyncSpace + AdditionalGlobalDebugCountSpace + AdditionalGlobalChecksumSpace;

template <class SharkFloatParams>
constexpr int32_t
CalculateReferenceSharedMemorySize()
{
    return 3 * 2048 * sizeof(uint64_t);
}

} // namespace HpShark
