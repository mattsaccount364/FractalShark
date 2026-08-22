#pragma once

#include "DebugStateRaw.h"

#include <cstddef>
#include <cstdint>

namespace HpShark {

static constexpr auto MaxReferenceBlocks = 256;
static constexpr auto AdditionalGlobalSyncSpace = 128 * (MaxReferenceBlocks + 1);
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

static constexpr size_t ReferenceSharedMemoryLimit = 96u * 1024u;
static constexpr size_t ReferenceCarryPrefixMaxWarps = 32u;
static constexpr size_t ReferenceCarryPrefixSharedBytes =
    2u * ReferenceCarryPrefixMaxWarps * sizeof(uint64_t);
static constexpr size_t ReferenceSharedMemoryDynamicLimit =
    ReferenceSharedMemoryLimit - ReferenceCarryPrefixSharedBytes;
static constexpr size_t ReferenceDefaultSharedMemory = 48u * 1024u;
static constexpr size_t ReferenceSharedOnlyTileSize = 1u << 11u;
static constexpr size_t ReferenceSharedOnlyCachedTwiddleCount = 1u << 7u;
static constexpr size_t ReferenceSharedOnlyDynamicBytes =
    (4u * ReferenceSharedOnlyTileSize + 2u * ReferenceSharedOnlyCachedTwiddleCount) * sizeof(uint64_t);
static_assert(ReferenceSharedOnlyDynamicBytes <= ReferenceSharedMemoryDynamicLimit);

template <class SharkFloatParams>
constexpr int32_t
CalculateReferenceSharedMemorySize()
{
    if constexpr (SharkFloatParams::SharedOnly) {
        return static_cast<int32_t>(ReferenceSharedOnlyDynamicBytes);
    } else {
        return static_cast<int32_t>(ReferenceDefaultSharedMemory);
    }
}

} // namespace HpShark
