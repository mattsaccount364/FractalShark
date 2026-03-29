#pragma once

// GPU scratch memory layout constants and frame size calculators.
// Standalone header — no dependency on HpSharkFloat class or type aliases.
// Template functions access SharkFloatParams members via template parameter only.

namespace HpShark {

// This one should account for maximum call index, e.g. if we generate 500 calls
// recursively then we need this to be at 500.
static constexpr auto ScratchMemoryCopies = 256llu;

// Number of arrays of digits on each frame (non-NR baseline)
static constexpr auto ScratchMemoryArraysForMultiply = 64;
static constexpr auto ScratchMemoryArraysForMultiplyNR = 90;
static constexpr auto ScratchMemoryArraysForAdd = 40;
static constexpr auto ScratchMemoryArraysForAddNR = 76;

// Additional space per frame:
static constexpr auto AdditionalUInt64PerFrame = 256;

// Additional space up front, globally-shared:
// Units are uint64_t
static constexpr auto MaxBlocks = 256;

static constexpr auto AdditionalGlobalSyncSpace = 128 * (MaxBlocks + 1);
static constexpr auto AdditionalGlobalDebugPerThread = DebugGlobalState ? 1024 * 1024 : 0;
static constexpr auto AdditionalGlobalChecksumSpace = DebugChecksums ? 1024 * 1024 : 0;

static constexpr auto AdditionalGlobalSyncSpaceOffset = 0;
static constexpr auto AdditionalMultipliesOffset =
    AdditionalGlobalSyncSpaceOffset + AdditionalGlobalSyncSpace;
static constexpr auto AdditionalChecksumsOffset =
    AdditionalMultipliesOffset + AdditionalGlobalDebugPerThread;

// Use the order of these three variables being added as the
// definition of how they are laid out in memory.
static constexpr auto AdditionalUInt64Global =
    AdditionalGlobalSyncSpace + AdditionalGlobalDebugPerThread + AdditionalGlobalChecksumSpace;

template <class SharkFloatParams>
static constexpr auto
CalculateKaratsubaFrameSize()
{
    constexpr auto arrays = SharkFloatParams::EnableNewtonRaphson
                                ? ScratchMemoryArraysForMultiplyNR
                                : ScratchMemoryArraysForMultiply;
    constexpr auto retval = arrays * SharkFloatParams::GlobalNumUint32 + AdditionalUInt64PerFrame;
    constexpr auto alignAt16BytesConstant = (retval % 16 == 0) ? 0 : (16 - retval % 16);
    return retval + alignAt16BytesConstant;
}

template <class SharkFloatParams>
static constexpr auto
CalculateNTTFrameSize()
{
    constexpr auto arrays = SharkFloatParams::EnableNewtonRaphson
                                ? ScratchMemoryArraysForMultiplyNR
                                : ScratchMemoryArraysForMultiply;
    constexpr auto retval = arrays * SharkFloatParams::GlobalNumUint32 + AdditionalUInt64PerFrame;
    constexpr auto alignAt16BytesConstant = (retval % 16 == 0) ? 0 : (16 - retval % 16);
    return retval + alignAt16BytesConstant;
}

template <class SharkFloatParams>
constexpr int32_t
CalculateNTTSharedMemorySize()
{
    // HpShark::ConstantSharedRequiredBytes
    constexpr auto sharedAmountBytes = 3 * 2048 * sizeof(uint64_t);
    return sharedAmountBytes;
}

template <class SharkFloatParams>
static constexpr auto
CalculateAddFrameSize()
{
    constexpr auto arrays = SharkFloatParams::EnableNewtonRaphson
                                ? ScratchMemoryArraysForAddNR
                                : ScratchMemoryArraysForAdd;
    return arrays * SharkFloatParams::GlobalNumUint32 + AdditionalUInt64PerFrame;
}

// Returns the maximum of NTT and Add frame sizes (both share the same scratch allocation).
template <class SharkFloatParams>
static constexpr auto
CalculateMaxFrameSize()
{
    constexpr auto ntt = CalculateNTTFrameSize<SharkFloatParams>();
    constexpr auto add = CalculateAddFrameSize<SharkFloatParams>();
    return (ntt > add) ? ntt : add;
}

static constexpr auto LowPrec = 32;

} // namespace HpShark
