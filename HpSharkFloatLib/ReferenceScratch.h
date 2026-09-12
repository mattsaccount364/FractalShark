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

// The 48 KiB footprint is the minimum layout guaranteed by the supported devices.  The host
// launch path opts into 96 KiB automatically when the active device can accommodate it.
static constexpr size_t ReferenceMinimumSharedMemory = 48u * 1024u;

template <class SharkFloatParams>
constexpr int32_t
CalculateReferenceSharedMemorySize()
{
    return static_cast<int32_t>(ReferenceMinimumSharedMemory);
}

} // namespace HpShark
