#include "DebugChecksum.h"
#include "Exceptions.h"
#include "KernelHpSharkReferenceOrbit.h"
#include "LaunchParamsCalculator.h"
#include "ReferenceNTT.h"

#include <algorithm>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cstddef>
#include <cstdint>
#include <cuda/atomic>
#include <cuda/barrier>
#include <cuda_runtime.h>
#include <sstream>

namespace cg = cooperative_groups;

namespace ReferenceDetail {

// Runtime-purpose debug-state storage used by Reference checkpoints.
template <class SharkFloatParams, typename ArrayType>
static __device__ SharkForceInlineReleaseOnly void
StoreCurrentDebugState(DebugState<SharkFloatParams> *SharkRestrict debugStates,
                       cooperative_groups::grid_group &grid,
                       cooperative_groups::thread_block &block,
                       DebugStatePurpose purpose,
                       const ArrayType *arrayToChecksum,
                       size_t arraySize)
{
    const auto curPurpose = static_cast<int32_t>(purpose);
    constexpr auto recursionDepth = 0;
    constexpr auto useConvolution = UseConvolution::No;
    constexpr auto callIndex = 0;

    debugStates[curPurpose].Reset(
        useConvolution, grid, block, arrayToChecksum, arraySize, purpose, recursionDepth, callIndex);
}

template <class SharkFloatParams>
static __device__ SharkForceInlineReleaseOnly void
StoreCurrentDebugValue(DebugState<SharkFloatParams> *SharkRestrict debugStates,
                       cooperative_groups::grid_group &grid,
                       cooperative_groups::thread_block &block,
                       DebugStatePurpose purpose,
                       const HpSharkFloat<SharkFloatParams> &value)
{
    const auto curPurpose = static_cast<int32_t>(purpose);
    constexpr auto recursionDepth = 0;
    constexpr auto useConvolution = UseConvolution::No;
    constexpr auto callIndex = 0;

    debugStates[curPurpose].Reset(
        useConvolution, grid, block, value, purpose, recursionDepth, callIndex);
}

namespace NTT {

static __device__ SharkForceInlineReleaseOnly uint64_t
AddP(uint64_t a, uint64_t b)
{
    uint64_t s = a + b;
    if (s < a || s >= SharkNTT::MagicPrime)
        s -= SharkNTT::MagicPrime;
    return s;
}

static __device__ SharkForceInlineReleaseOnly uint64_t
SubP(uint64_t a, uint64_t b)
{
    return (a >= b) ? (a - b) : (a + SharkNTT::MagicPrime - b);
}

template <class SharkFloatParams>
static __device__ SharkForceInlineReleaseOnly uint64_t
MontgomeryMul(DebugGlobalCount<SharkFloatParams> *debugCombo, uint64_t a, uint64_t b)
{
    // Debug instrumentation (optionally compiled out via if constexpr).
    // Count as 7 "64-bit mul-equivalents": 3 for a*b, 3 for m*p, 1 for the add path.
    if constexpr (HpShark::DebugGlobalState) {
        cooperative_groups::grid_group grid = cooperative_groups::this_grid();
        cooperative_groups::thread_block block = cooperative_groups::this_thread_block();
        DebugMultiplyIncrement<SharkFloatParams>(debugCombo, grid, block, 7);
    }

    // ---------------------------------------------------------------------
    // The modulus for this Montgomery domain is SharkNTT::MagicPrime.
    // Montgomery reduction computes:
    //
    //     r = (a * b * R^{-1}) mod p     (where R = 2^64)
    //
    // Given:
    //   NINV = -p^{-1} mod 2^64
    //
    // We compute:
    //   t      = a * b           (128-bit)
    //   m      = (tLo * NINV)   (mod R)
    //   mp     = m * p           (128-bit)
    //   u      = t + mp          (128-bit)
    //   r      = uHi (upper 64 bits)
    //
    // And finally, ensure r < p by subtracting p if needed.
    //
    // PTX is used to explicitly control 128-bit math via mul.lo/mul.hi and add.cc/addc.
    // ---------------------------------------------------------------------

    uint64_t tLo, tHi;   // 128-bit product of a*b
    uint64_t m;          // m = tLo * MagicPrimeInv (mod 2^64)
    uint64_t mpLo, mpHi; // 128-bit product m * MagicPrime

    // ---------------------------------------------------------------------
    // Compute:
    //   tLo  = (a * b) low  64 bits
    //   tHi  = (a * b) high 64 bits
    //   m     = (tLo * MagicPrimeInv) mod 2^64   (Montgomery trick)
    //   mpLo = (m * MagicPrime) low  64 bits
    //   mpHi = (m * MagicPrime) high 64 bits
    //
    // All in a single asm block so the compiler can't interleave or reorder them.
    // Using "=&l" marks outputs early-clobber, ensuring no operand overlap.
    // ---------------------------------------------------------------------
    asm("{\n\t"
        "  mul.lo.u64 %0, %5, %6;   // tLo = a * b (low 64 bits)\n\t"
        "  mul.hi.u64 %1, %5, %6;   // tHi = a * b (high 64 bits)\n\t"
        "  mul.lo.u64 %2, %0, %7;   // m    = tLo * MagicPrimeInv (mod 2^64)\n\t"
        "  mul.lo.u64 %3, %2, %8;   // mpLo = m * MagicPrime (low 64 bits)\n\t"
        "  mul.hi.u64 %4, %2, %8;   // mpHi = m * MagicPrime (high 64 bits)\n\t"
        "}\n\t"
        : "=&l"(tLo), "=&l"(tHi), "=&l"(m), "=&l"(mpLo), "=&l"(mpHi)
        : "l"(a),
          "l"(b),
          "l"(SharkNTT::MagicPrimeInv), // constant folded into immediate or const space
          "l"(SharkNTT::MagicPrime));   // same

    uint64_t uHi, carry1;

    // ---------------------------------------------------------------------
    // Now compute 128-bit addition:
    //     u = t + mp
    //
    // We only need uHi (upper 64 bits) for Montgomery reduction; the low limb is discarded.
    // Reuse mpLo as the low-sum scratch to reduce register pressure.
    //
    // add.cc        sets the carry flag (CC) from the low-limb addition.
    // addc.cc       adds the high limbs *plus the carry*, again updating CC.
    // addc          writes out the final carry (0 or 1) to carry1.
    // ---------------------------------------------------------------------
    asm("add.cc.u64  %0, %3, %4;\n\t" // mpLo = tLo + mpLo   (sets carry0)
        "addc.cc.u64 %1, %5, %6;\n\t" // uHi = tHi + mpHi + carry0   (sets carry1)
        "addc.u64    %2, 0, 0;\n\t"   // carry1 = final carry out
        : "+l"(mpLo), "=&l"(uHi), "=&l"(carry1)
        : "l"(tLo), "l"(mpLo), "l"(tHi), "l"(mpHi));

    // Candidate Montgomery result before final correction
    uint64_t r = uHi;

    // ---------------------------------------------------------------------
    // Final conditional subtraction:
    //
    //   if (carry1 || r >= p)
    //       r -= p;
    //
    // Implemented branchlessly using PTX predicates:
    //
    //   p_carry = (carry1 != 0)
    //   p_ge    = (r >= p)
    //   p_do    = p_carry || p_ge
    //
    // We perform:
    //   r = r - p
    //   if (!p_do) r = r + p   // undo subtraction when not needed
    //
    // This avoids warp divergence and yields a constant-latency path.
    // ---------------------------------------------------------------------
    {
        uint64_t p = SharkNTT::MagicPrime;

        asm volatile("{\n\t"
                     "  .reg .pred p_carry, p_ge, p_do;\n\t"
                     "  setp.ne.u64 p_carry, %1, 0;     // p_carry = (carry1 != 0)\n\t"
                     "  setp.ge.u64 p_ge, %0, %2;       // p_ge = (r >= p)\n\t"
                     "  or.pred p_do, p_carry, p_ge;    // p_do = p_carry || p_ge\n\t"
                     "  sub.u64 %0, %0, %2;             // r = r - p   (tentative)\n\t"
                     "  @!p_do add.u64 %0, %0, %2;      // if not doing reduction, restore r += p\n\t"
                     "}\n\t"
                     : "+l"(r)
                     : "l"(carry1), "l"(p));
    }

    return r; // Fully normalized Montgomery product mod p
}

template <class SharkFloatParams>
static __device__ SharkForceInlineReleaseOnly uint64_t
ToMontgomery(DebugGlobalCount<SharkFloatParams> *debugCombo, uint64_t x)
{
    return MontgomeryMul(debugCombo, x, SharkNTT::R2);
}

template <class SharkFloatParams>
static __device__ SharkForceInlineReleaseOnly uint64_t
FromMontgomery(DebugGlobalCount<SharkFloatParams> *debugCombo, uint64_t x)
{
    return MontgomeryMul(debugCombo, x, 1);
}

template <class SharkFloatParams>
static __device__ SharkForceInlineReleaseOnly void
StoreRadix2DIFButterfly(DebugGlobalCount<SharkFloatParams> *debugCombo,
                        uint64_t *SharkRestrict data,
                        uint32_t upperIndex,
                        uint32_t lowerIndex,
                        uint64_t twiddle)
{
    const uint64_t upper = data[upperIndex];
    const uint64_t lower = data[lowerIndex];
    data[upperIndex] = AddP(upper, lower);
    data[lowerIndex] = MontgomeryMul(debugCombo, SubP(upper, lower), twiddle);
}

template <class SharkFloatParams>
static __device__ SharkForceInlineReleaseOnly void
StoreRadix2DITButterfly(DebugGlobalCount<SharkFloatParams> *debugCombo,
                        uint64_t *SharkRestrict data,
                        uint32_t upperIndex,
                        uint32_t lowerIndex,
                        uint64_t twiddle)
{
    const uint64_t upper = data[upperIndex];
    const uint64_t lower = data[lowerIndex];
    const uint64_t product = MontgomeryMul(debugCombo, lower, twiddle);
    data[upperIndex] = AddP(upper, product);
    data[lowerIndex] = SubP(upper, product);
}

template <class SharkFloatParams>
static __device__ SharkForceInlineReleaseOnly void
StoreRadix4DIFStagePair(DebugGlobalCount<SharkFloatParams> *debugCombo,
                        uint64_t *SharkRestrict data,
                        uint32_t index0,
                        uint32_t firstHalfSpan,
                        uint32_t j,
                        const uint64_t *SharkRestrict firstTwiddles,
                        const uint64_t *SharkRestrict secondTwiddles)
{
    const uint32_t secondHalfSpan = firstHalfSpan >> 1u;
    const uint32_t index1 = index0 + secondHalfSpan;
    const uint32_t index2 = index0 + firstHalfSpan;
    const uint32_t index3 = index2 + secondHalfSpan;
    const uint64_t value0 = data[index0];
    const uint64_t value1 = data[index1];
    const uint64_t value2 = data[index2];
    const uint64_t value3 = data[index3];
    const uint64_t firstStageTwiddle0 = firstTwiddles[j];
    const uint64_t firstStageTwiddle1 = firstTwiddles[j + secondHalfSpan];
    const uint64_t secondStageTwiddle = secondTwiddles[j];
    const uint64_t firstValue0 = AddP(value0, value2);
    const uint64_t firstValue1 = AddP(value1, value3);
    const uint64_t firstProduct0 = MontgomeryMul(debugCombo, SubP(value0, value2), firstStageTwiddle0);
    const uint64_t firstProduct1 = MontgomeryMul(debugCombo, SubP(value1, value3), firstStageTwiddle1);
    const uint64_t secondValue0 = AddP(firstValue0, firstValue1);
    const uint64_t secondValue1 =
        MontgomeryMul(debugCombo, SubP(firstValue0, firstValue1), secondStageTwiddle);
    const uint64_t secondValue2 = AddP(firstProduct0, firstProduct1);
    const uint64_t secondValue3 =
        MontgomeryMul(debugCombo, SubP(firstProduct0, firstProduct1), secondStageTwiddle);
    data[index0] = secondValue0;
    data[index1] = secondValue1;
    data[index2] = secondValue2;
    data[index3] = secondValue3;
}

template <class SharkFloatParams>
static __device__ SharkForceInlineReleaseOnly void
StoreRadix4DITStagePair(DebugGlobalCount<SharkFloatParams> *debugCombo,
                        uint64_t *SharkRestrict data,
                        uint32_t index0,
                        uint32_t firstHalfSpan,
                        uint32_t j,
                        const uint64_t *SharkRestrict lowerStageTwiddles,
                        const uint64_t *SharkRestrict upperStageTwiddles)
{
    const uint32_t secondHalfSpan = firstHalfSpan >> 1u;
    const uint32_t index1 = index0 + secondHalfSpan;
    const uint32_t index2 = index0 + firstHalfSpan;
    const uint32_t index3 = index2 + secondHalfSpan;
    const uint64_t value0 = data[index0];
    const uint64_t value1 = data[index1];
    const uint64_t value2 = data[index2];
    const uint64_t value3 = data[index3];
    const uint64_t lowerStageTwiddle = lowerStageTwiddles[j];
    const uint64_t upperStageTwiddle0 = upperStageTwiddles[j];
    const uint64_t upperStageTwiddle1 = upperStageTwiddles[j + secondHalfSpan];
    const uint64_t firstProduct0 = MontgomeryMul(debugCombo, value1, lowerStageTwiddle);
    const uint64_t firstProduct1 = MontgomeryMul(debugCombo, value3, lowerStageTwiddle);
    const uint64_t firstValue0 = AddP(value0, firstProduct0);
    const uint64_t firstValue1 = SubP(value0, firstProduct0);
    const uint64_t firstValue2 = AddP(value2, firstProduct1);
    const uint64_t firstValue3 = SubP(value2, firstProduct1);
    const uint64_t secondProduct0 = MontgomeryMul(debugCombo, firstValue2, upperStageTwiddle0);
    const uint64_t secondProduct1 = MontgomeryMul(debugCombo, firstValue3, upperStageTwiddle1);
    data[index0] = AddP(firstValue0, secondProduct0);
    data[index2] = SubP(firstValue0, secondProduct0);
    data[index1] = AddP(firstValue1, secondProduct1);
    data[index3] = SubP(firstValue1, secondProduct1);
}

static __device__ SharkForceInlineReleaseOnly size_t
CohortThreadIndex()
{
    const uint32_t cohortBlockIndex = gridDim.x >= 2u ? blockIdx.x >> 1u : 0u;
    return static_cast<size_t>(cohortBlockIndex) * blockDim.x + threadIdx.x;
}

static __device__ SharkForceInlineReleaseOnly size_t
CohortGridSize()
{
    const uint32_t cohort = gridDim.x >= 2u ? blockIdx.x & 1u : 0u;
    const uint32_t cohortBlockCount = gridDim.x >= 2u ? (gridDim.x + 1u - cohort) >> 1u : 1u;
    return static_cast<size_t>(cohortBlockCount) * blockDim.x;
}

template <class SharkFloatParams>
static __device__ __noinline__ void
ForwardRadix2One(DebugGlobalCount<SharkFloatParams> *debugCombo,
                 uint64_t *SharkRestrict A,
                 const uint64_t *SharkRestrict stageTwiddles,
                 uint32_t transformSize,
                 uint32_t stageIndex)
{
    const uint32_t butterflySpan = 1u << stageIndex;
    const uint32_t halfSpan = butterflySpan >> 1u;
    const uint32_t twiddleOffset = halfSpan - 1u;
    const size_t threadIndex = CohortThreadIndex();
    const size_t gridSize = CohortGridSize();
    const size_t butterflyCount = static_cast<size_t>(transformSize) >> 1u;
    for (size_t task = threadIndex; task < butterflyCount; task += gridSize) {
        const uint32_t blockIndex = static_cast<uint32_t>(task / halfSpan);
        const uint32_t j = static_cast<uint32_t>(task - static_cast<size_t>(blockIndex) * halfSpan);
        const uint32_t upperIndex = blockIndex * butterflySpan + j;
        const uint32_t lowerIndex = upperIndex + halfSpan;
        StoreRadix2DIFButterfly(debugCombo, A, upperIndex, lowerIndex, stageTwiddles[twiddleOffset + j]);
    }
}

template <class SharkFloatParams>
static __device__ __noinline__ void
ForwardRadix2Two(DebugGlobalCount<SharkFloatParams> *debugCombo,
                 uint64_t *SharkRestrict A,
                 uint64_t *SharkRestrict B,
                 const uint64_t *SharkRestrict stageTwiddles,
                 uint32_t transformSize,
                 uint32_t stageIndex)
{
    ForwardRadix2One(debugCombo, A, stageTwiddles, transformSize, stageIndex);
    ForwardRadix2One(debugCombo, B, stageTwiddles, transformSize, stageIndex);
}

template <class SharkFloatParams>
static __device__ __noinline__ void
ForwardRadix4One(DebugGlobalCount<SharkFloatParams> *debugCombo,
                 uint64_t *SharkRestrict A,
                 const uint64_t *SharkRestrict stageTwiddles,
                 uint32_t transformSize,
                 uint32_t firstStageIndex)
{
    const uint32_t firstHalfSpan = 1u << (firstStageIndex - 1u);
    const uint32_t secondHalfSpan = firstHalfSpan >> 1u;
    const uint32_t combinedSpan = firstHalfSpan << 1u;
    const uint64_t *firstTwiddles = stageTwiddles + firstHalfSpan - 1u;
    const uint64_t *secondTwiddles = stageTwiddles + secondHalfSpan - 1u;
    const uint32_t numBlocks = transformSize / combinedSpan;
    const size_t threadIndex = CohortThreadIndex();
    const size_t gridSize = CohortGridSize();
    for (size_t task = threadIndex; task < static_cast<size_t>(numBlocks) * secondHalfSpan;
         task += gridSize) {
        const uint32_t blockIndex = static_cast<uint32_t>(task / secondHalfSpan);
        const uint32_t j =
            static_cast<uint32_t>(task - static_cast<size_t>(blockIndex) * secondHalfSpan);
        const uint32_t index0 = blockIndex * combinedSpan + j;
        StoreRadix4DIFStagePair(debugCombo, A, index0, firstHalfSpan, j, firstTwiddles, secondTwiddles);
    }
}

template <class SharkFloatParams>
static __device__ __noinline__ void
ForwardRadix4Two(DebugGlobalCount<SharkFloatParams> *debugCombo,
                 uint64_t *SharkRestrict A,
                 uint64_t *SharkRestrict B,
                 const uint64_t *SharkRestrict stageTwiddles,
                 uint32_t transformSize,
                 uint32_t firstStageIndex)
{
    ForwardRadix4One(debugCombo, A, stageTwiddles, transformSize, firstStageIndex);
    ForwardRadix4One(debugCombo, B, stageTwiddles, transformSize, firstStageIndex);
}

template <class SharkFloatParams>
static __device__ __noinline__ void
InverseRadix2One(DebugGlobalCount<SharkFloatParams> *debugCombo,
                 uint64_t *SharkRestrict A,
                 const uint64_t *SharkRestrict stageTwiddles,
                 uint32_t transformSize,
                 uint32_t stageIndex)
{
    const uint32_t butterflySpan = 1u << stageIndex;
    const uint32_t halfSpan = butterflySpan >> 1u;
    const uint32_t twiddleOffset = halfSpan - 1u;
    const size_t threadIndex = CohortThreadIndex();
    const size_t gridSize = CohortGridSize();
    const size_t butterflyCount = static_cast<size_t>(transformSize) >> 1u;
    for (size_t task = threadIndex; task < butterflyCount; task += gridSize) {
        const uint32_t blockIndex = static_cast<uint32_t>(task / halfSpan);
        const uint32_t j = static_cast<uint32_t>(task - static_cast<size_t>(blockIndex) * halfSpan);
        const uint32_t upperIndex = blockIndex * butterflySpan + j;
        const uint32_t lowerIndex = upperIndex + halfSpan;
        StoreRadix2DITButterfly(debugCombo, A, upperIndex, lowerIndex, stageTwiddles[twiddleOffset + j]);
    }
}

// The fused DIT operation consumes upperStageIndex - 1 followed by upperStageIndex.
template <class SharkFloatParams>
static __device__ __noinline__ void
InverseRadix4One(DebugGlobalCount<SharkFloatParams> *debugCombo,
                 uint64_t *SharkRestrict A,
                 const uint64_t *SharkRestrict stageTwiddles,
                 uint32_t transformSize,
                 uint32_t upperStageIndex)
{
    const uint32_t firstHalfSpan = 1u << (upperStageIndex - 1u);
    const uint32_t secondHalfSpan = firstHalfSpan >> 1u;
    const uint32_t combinedSpan = firstHalfSpan << 1u;
    const uint64_t *upperStageTwiddles = stageTwiddles + firstHalfSpan - 1u;
    const uint64_t *lowerStageTwiddles = stageTwiddles + secondHalfSpan - 1u;
    const uint32_t numBlocks = transformSize / combinedSpan;
    const size_t threadIndex = CohortThreadIndex();
    const size_t gridSize = CohortGridSize();
    for (size_t task = threadIndex; task < static_cast<size_t>(numBlocks) * secondHalfSpan;
         task += gridSize) {
        const uint32_t blockIndex = static_cast<uint32_t>(task / secondHalfSpan);
        const uint32_t j =
            static_cast<uint32_t>(task - static_cast<size_t>(blockIndex) * secondHalfSpan);
        const uint32_t index0 = blockIndex * combinedSpan + j;
        StoreRadix4DITStagePair(
            debugCombo, A, index0, firstHalfSpan, j, lowerStageTwiddles, upperStageTwiddles);
    }
}

// The fused DIT operation consumes upperStageIndex - 1 followed by upperStageIndex.
template <class SharkFloatParams>
static __device__ __noinline__ void
InverseRadix4Two(DebugGlobalCount<SharkFloatParams> *debugCombo,
                 uint64_t *SharkRestrict A,
                 uint64_t *SharkRestrict B,
                 const uint64_t *SharkRestrict stageTwiddles,
                 uint32_t transformSize,
                 uint32_t upperStageIndex)
{
    InverseRadix4One(debugCombo, A, stageTwiddles, transformSize, upperStageIndex);
    InverseRadix4One(debugCombo, B, stageTwiddles, transformSize, upperStageIndex);
}

template <class SharkFloatParams>
static __device__ __noinline__ void
InverseRadix2Two(DebugGlobalCount<SharkFloatParams> *debugCombo,
                 uint64_t *SharkRestrict A,
                 uint64_t *SharkRestrict B,
                 const uint64_t *SharkRestrict stageTwiddles,
                 uint32_t transformSize,
                 uint32_t stageIndex)
{
    InverseRadix2One(debugCombo, A, stageTwiddles, transformSize, stageIndex);
    InverseRadix2One(debugCombo, B, stageTwiddles, transformSize, stageIndex);
}

constexpr uint32_t TileSixWarpStageCount = 6u;
constexpr uint32_t TileSevenWarpStageCount = 7u;
constexpr uint32_t TileSevenStageMinimumLength = 1u << 10u;
constexpr uint32_t TileTwiddleCacheWords = 1u << 11u;
constexpr uint32_t TileSharedWords = 48u * 1024u / sizeof(uint64_t);

static __device__ SharkForceInlineReleaseOnly uint32_t
CohortBlockIndex()
{
    return gridDim.x >= 2u ? blockIdx.x >> 1u : 0u;
}

static __device__ SharkForceInlineReleaseOnly uint32_t
CohortBlockCount()
{
    return gridDim.x >= 2u ? (gridDim.x + 1u - (blockIdx.x & 1u)) >> 1u : 1u;
}

static __device__ SharkForceInlineReleaseOnly uint32_t
LargestCohortBlockCount()
{
    return gridDim.x >= 2u ? (gridDim.x + 1u) >> 1u : 1u;
}

static __device__ SharkForceInlineReleaseOnly uint32_t
SelectTileSizeLog2(uint32_t transformSize, uint32_t maximumTileSizeLog2)
{
    uint32_t candidate = maximumTileSizeLog2;
    while (candidate > 9u) {
        const uint32_t tileSize = 1u << candidate;
        const uint32_t tileCount = (transformSize + tileSize - 1u) / tileSize;
        if (tileCount >= LargestCohortBlockCount())
            return candidate;
        --candidate;
    }
    return 9u;
}

static __device__ SharkForceInlineReleaseOnly uint32_t
SelectTileWarpStageCount(uint32_t length)
{
    return length >= TileSevenStageMinimumLength ? TileSevenWarpStageCount : TileSixWarpStageCount;
}

template <int Width>
static __device__ SharkForceInlineReleaseOnly uint64_t
ShuffleXorUint64Width(uint64_t value, uint32_t laneMask)
{
    const uint32_t low = __shfl_xor_sync(0xFFFF'FFFFu, static_cast<uint32_t>(value), laneMask, Width);
    const uint32_t high =
        __shfl_xor_sync(0xFFFF'FFFFu, static_cast<uint32_t>(value >> 32u), laneMask, Width);
    return static_cast<uint64_t>(low) | (static_cast<uint64_t>(high) << 32u);
}

template <int Width>
static __device__ SharkForceInlineReleaseOnly uint64_t
ShuffleUint64Width(uint64_t value, uint32_t sourceLane)
{
    const uint32_t low = __shfl_sync(0xFFFF'FFFFu, static_cast<uint32_t>(value), sourceLane, Width);
    const uint32_t high =
        __shfl_sync(0xFFFF'FFFFu, static_cast<uint32_t>(value >> 32u), sourceLane, Width);
    return static_cast<uint64_t>(low) | (static_cast<uint64_t>(high) << 32u);
}

template <int Width>
static __device__ SharkForceInlineReleaseOnly void
RegroupWarpButterfly(uint64_t &upper, uint64_t &lower, uint32_t shuffleDistance, bool ownsLowerInput)
{
    const uint64_t previousUpper = upper;
    const uint64_t previousLower = lower;
    const uint64_t ownValue = ownsLowerInput ? previousLower : previousUpper;
    const uint64_t partnerCandidate = ownsLowerInput ? previousUpper : previousLower;
    const uint64_t partner = ShuffleXorUint64Width<Width>(partnerCandidate, shuffleDistance);
    upper = ownsLowerInput ? partner : ownValue;
    lower = ownsLowerInput ? ownValue : partner;
}

static __device__ SharkForceInlineReleaseOnly uint64_t
LoadWarpSubgroupTwiddle(const uint64_t *twiddleCache, uint32_t stage, uint32_t subgroupLane)
{
    const uint32_t halfSpan = 1u << (stage - 1u);
    const uint32_t j = subgroupLane & (halfSpan - 1u);
    uint64_t twiddle = 0ull;
    if (subgroupLane < halfSpan)
        twiddle = twiddleCache[halfSpan - 1u + subgroupLane];
    return ShuffleUint64Width<16>(twiddle, j);
}

template <class SharkFloatParams>
static __device__ SharkForceInlineReleaseOnly void
ApplyWarpDIFButterfly(DebugGlobalCount<SharkFloatParams> *debugCombo,
                      uint64_t twiddle,
                      uint64_t &upper,
                      uint64_t &lower)
{
    const uint64_t originalUpper = upper;
    const uint64_t originalLower = lower;
    upper = AddP(originalUpper, originalLower);
    lower = MontgomeryMul(debugCombo, SubP(originalUpper, originalLower), twiddle);
}

template <class SharkFloatParams>
static __device__ SharkForceInlineReleaseOnly void
ApplyWarpDITButterfly(DebugGlobalCount<SharkFloatParams> *debugCombo,
                      uint64_t twiddle,
                      uint64_t &upper,
                      uint64_t &lower)
{
    const uint64_t originalUpper = upper;
    const uint64_t product = MontgomeryMul(debugCombo, lower, twiddle);
    upper = AddP(originalUpper, product);
    lower = SubP(originalUpper, product);
}

template <class SharkFloatParams>
static __device__ SharkForceInlineReleaseOnly void
ForwardWarpSubgroupStagesOne(DebugGlobalCount<SharkFloatParams> *debugCombo,
                             const uint64_t *twiddleCache,
                             uint32_t subgroupLane,
                             uint64_t &upper,
                             uint64_t &lower)
{
    uint64_t twiddle = LoadWarpSubgroupTwiddle(twiddleCache, 5u, subgroupLane);
    ApplyWarpDIFButterfly(debugCombo, twiddle, upper, lower);
#pragma unroll
    for (uint32_t stage = 4u; stage > 0u; --stage) {
        const uint32_t shuffleDistance = 1u << (stage - 1u);
        const bool ownsLowerInput = (subgroupLane & shuffleDistance) != 0u;
        RegroupWarpButterfly<16>(upper, lower, shuffleDistance, ownsLowerInput);
        twiddle = LoadWarpSubgroupTwiddle(twiddleCache, stage, subgroupLane);
        ApplyWarpDIFButterfly(debugCombo, twiddle, upper, lower);
    }
}

template <class SharkFloatParams>
static __device__ SharkForceInlineReleaseOnly void
ForwardWarpSubgroupStagesTwo(DebugGlobalCount<SharkFloatParams> *debugCombo,
                             const uint64_t *twiddleCache,
                             uint32_t subgroupLane,
                             uint64_t &firstUpper,
                             uint64_t &firstLower,
                             uint64_t &secondUpper,
                             uint64_t &secondLower)
{
    uint64_t twiddle = LoadWarpSubgroupTwiddle(twiddleCache, 5u, subgroupLane);
    ApplyWarpDIFButterfly(debugCombo, twiddle, firstUpper, firstLower);
    ApplyWarpDIFButterfly(debugCombo, twiddle, secondUpper, secondLower);
#pragma unroll
    for (uint32_t stage = 4u; stage > 0u; --stage) {
        const uint32_t shuffleDistance = 1u << (stage - 1u);
        const bool ownsLowerInput = (subgroupLane & shuffleDistance) != 0u;
        RegroupWarpButterfly<16>(firstUpper, firstLower, shuffleDistance, ownsLowerInput);
        RegroupWarpButterfly<16>(secondUpper, secondLower, shuffleDistance, ownsLowerInput);
        twiddle = LoadWarpSubgroupTwiddle(twiddleCache, stage, subgroupLane);
        ApplyWarpDIFButterfly(debugCombo, twiddle, firstUpper, firstLower);
        ApplyWarpDIFButterfly(debugCombo, twiddle, secondUpper, secondLower);
    }
}

template <class SharkFloatParams>
static __device__ SharkForceInlineReleaseOnly void
InverseWarpSubgroupStagesOne(DebugGlobalCount<SharkFloatParams> *debugCombo,
                             const uint64_t *twiddleCache,
                             uint32_t subgroupLane,
                             uint64_t &upper,
                             uint64_t &lower)
{
    uint64_t twiddle = LoadWarpSubgroupTwiddle(twiddleCache, 1u, subgroupLane);
    ApplyWarpDITButterfly(debugCombo, twiddle, upper, lower);
#pragma unroll
    for (uint32_t stage = 2u; stage <= 5u; ++stage) {
        const uint32_t shuffleDistance = 1u << (stage - 2u);
        const bool ownsLowerInput = (subgroupLane & shuffleDistance) != 0u;
        RegroupWarpButterfly<16>(upper, lower, shuffleDistance, ownsLowerInput);
        twiddle = LoadWarpSubgroupTwiddle(twiddleCache, stage, subgroupLane);
        ApplyWarpDITButterfly(debugCombo, twiddle, upper, lower);
    }
}

template <class SharkFloatParams>
static __device__ SharkForceInlineReleaseOnly void
InverseWarpSubgroupStagesTwo(DebugGlobalCount<SharkFloatParams> *debugCombo,
                             const uint64_t *twiddleCache,
                             uint32_t subgroupLane,
                             uint64_t &firstUpper,
                             uint64_t &firstLower,
                             uint64_t &secondUpper,
                             uint64_t &secondLower)
{
    uint64_t twiddle = LoadWarpSubgroupTwiddle(twiddleCache, 1u, subgroupLane);
    ApplyWarpDITButterfly(debugCombo, twiddle, firstUpper, firstLower);
    ApplyWarpDITButterfly(debugCombo, twiddle, secondUpper, secondLower);
#pragma unroll
    for (uint32_t stage = 2u; stage <= 5u; ++stage) {
        const uint32_t shuffleDistance = 1u << (stage - 2u);
        const bool ownsLowerInput = (subgroupLane & shuffleDistance) != 0u;
        RegroupWarpButterfly<16>(firstUpper, firstLower, shuffleDistance, ownsLowerInput);
        RegroupWarpButterfly<16>(secondUpper, secondLower, shuffleDistance, ownsLowerInput);
        twiddle = LoadWarpSubgroupTwiddle(twiddleCache, stage, subgroupLane);
        ApplyWarpDITButterfly(debugCombo, twiddle, firstUpper, firstLower);
        ApplyWarpDITButterfly(debugCombo, twiddle, secondUpper, secondLower);
    }
}

template <class SharkFloatParams>
static __device__ SharkForceInlineReleaseOnly void
ForwardSharedStage(DebugGlobalCount<SharkFloatParams> *debugCombo,
                   uint64_t *tileData,
                   const uint64_t *twiddleCache,
                   const uint64_t *stageTwiddles,
                   uint32_t cachedWords,
                   uint32_t length,
                   uint32_t stage)
{
    const uint32_t butterflySpan = 1u << stage;
    const uint32_t halfSpan = butterflySpan >> 1u;
    const uint32_t twiddleOffset = halfSpan - 1u;
    for (uint32_t pair = threadIdx.x; pair < length / 2u; pair += blockDim.x) {
        const uint32_t group = pair / halfSpan;
        const uint32_t j = pair - group * halfSpan;
        const uint32_t upperIndex = group * butterflySpan + j;
        const uint32_t lowerIndex = upperIndex + halfSpan;
        const uint64_t upper = tileData[upperIndex];
        const uint64_t lower = tileData[lowerIndex];
        const uint32_t twiddleIndex = twiddleOffset + j;
        const uint64_t twiddle =
            twiddleIndex < cachedWords ? twiddleCache[twiddleIndex] : stageTwiddles[twiddleIndex];
        tileData[upperIndex] = AddP(upper, lower);
        tileData[lowerIndex] = MontgomeryMul(debugCombo, SubP(upper, lower), twiddle);
    }
}

template <class SharkFloatParams>
static __device__ SharkForceInlineReleaseOnly void
InverseSharedStage(DebugGlobalCount<SharkFloatParams> *debugCombo,
                   uint64_t *tileData,
                   const uint64_t *twiddleCache,
                   const uint64_t *stageTwiddles,
                   uint32_t cachedWords,
                   uint32_t length,
                   uint32_t stage)
{
    const uint32_t butterflySpan = 1u << stage;
    const uint32_t halfSpan = butterflySpan >> 1u;
    const uint32_t twiddleOffset = halfSpan - 1u;
    for (uint32_t pair = threadIdx.x; pair < length / 2u; pair += blockDim.x) {
        const uint32_t group = pair / halfSpan;
        const uint32_t j = pair - group * halfSpan;
        const uint32_t upperIndex = group * butterflySpan + j;
        const uint32_t lowerIndex = upperIndex + halfSpan;
        const uint64_t upper = tileData[upperIndex];
        const uint64_t lower = tileData[lowerIndex];
        const uint32_t twiddleIndex = twiddleOffset + j;
        const uint64_t twiddle =
            twiddleIndex < cachedWords ? twiddleCache[twiddleIndex] : stageTwiddles[twiddleIndex];
        const uint64_t product = MontgomeryMul(debugCombo, lower, twiddle);
        tileData[upperIndex] = AddP(upper, product);
        tileData[lowerIndex] = SubP(upper, product);
    }
}

template <class SharkFloatParams>
static __device__ SharkForceInlineReleaseOnly void
ForwardWarpStagesSix(DebugGlobalCount<SharkFloatParams> *debugCombo,
                     const uint64_t *tileData,
                     uint64_t *output,
                     const uint64_t *twiddleCache,
                     uint32_t tileOffset,
                     uint32_t length)
{
    constexpr uint32_t CoefficientsPerWarp = 1u << TileSixWarpStageCount;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t subgroupLane = lane & 15u;
    const uint32_t subgroupIndex = lane >> 4u;
    const uint32_t warpIndex = threadIdx.x >> 5u;
    const uint32_t warpCount = blockDim.x >> 5u;
    const uint32_t groupCount = length / CoefficientsPerWarp;
    for (uint32_t group = warpIndex; group < groupCount; group += warpCount) {
        const uint32_t groupBase = group * CoefficientsPerWarp;
        uint64_t upper = tileData[groupBase + lane];
        uint64_t lower = tileData[groupBase + lane + 32u];
        const uint64_t stageSixTwiddle = twiddleCache[31u + lane];
        ApplyWarpDIFButterfly(debugCombo, stageSixTwiddle, upper, lower);

        const bool ownsLowerInput = (lane & 16u) != 0u;
        RegroupWarpButterfly<32>(upper, lower, 16u, ownsLowerInput);
        ForwardWarpSubgroupStagesOne(debugCombo, twiddleCache, subgroupLane, upper, lower);

        const uint32_t outputBase = groupBase + subgroupIndex * 32u + subgroupLane * 2u;
        output[tileOffset + outputBase] = upper;
        output[tileOffset + outputBase + 1u] = lower;
    }
}

template <class SharkFloatParams>
static __device__ SharkForceInlineReleaseOnly void
ForwardWarpStagesSeven(DebugGlobalCount<SharkFloatParams> *debugCombo,
                       const uint64_t *tileData,
                       uint64_t *output,
                       const uint64_t *twiddleCache,
                       uint32_t tileOffset,
                       uint32_t length)
{
    constexpr uint32_t CoefficientsPerWarp = 1u << TileSevenWarpStageCount;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t subgroupLane = lane & 15u;
    const uint32_t subgroupIndex = lane >> 4u;
    const uint32_t warpIndex = threadIdx.x >> 5u;
    const uint32_t warpCount = blockDim.x >> 5u;
    const uint32_t groupCount = length / CoefficientsPerWarp;
    for (uint32_t group = warpIndex; group < groupCount; group += warpCount) {
        const uint32_t groupBase = group * CoefficientsPerWarp;
        uint64_t firstUpper = tileData[groupBase + lane];
        uint64_t firstLower = tileData[groupBase + lane + 32u];
        uint64_t secondUpper = tileData[groupBase + lane + 64u];
        uint64_t secondLower = tileData[groupBase + lane + 96u];

        uint64_t twiddle = twiddleCache[63u + lane];
        ApplyWarpDIFButterfly(debugCombo, twiddle, firstUpper, secondUpper);
        twiddle = twiddleCache[95u + lane];
        ApplyWarpDIFButterfly(debugCombo, twiddle, firstLower, secondLower);

        twiddle = twiddleCache[31u + lane];
        ApplyWarpDIFButterfly(debugCombo, twiddle, firstUpper, firstLower);
        ApplyWarpDIFButterfly(debugCombo, twiddle, secondUpper, secondLower);

        const bool ownsLowerInput = (lane & 16u) != 0u;
        RegroupWarpButterfly<32>(firstUpper, firstLower, 16u, ownsLowerInput);
        RegroupWarpButterfly<32>(secondUpper, secondLower, 16u, ownsLowerInput);
        ForwardWarpSubgroupStagesTwo(
            debugCombo, twiddleCache, subgroupLane, firstUpper, firstLower, secondUpper, secondLower);

        const uint32_t firstOutputBase = groupBase + subgroupIndex * 32u + subgroupLane * 2u;
        const uint32_t secondOutputBase = firstOutputBase + 64u;
        output[tileOffset + firstOutputBase] = firstUpper;
        output[tileOffset + firstOutputBase + 1u] = firstLower;
        output[tileOffset + secondOutputBase] = secondUpper;
        output[tileOffset + secondOutputBase + 1u] = secondLower;
    }
}

template <class SharkFloatParams>
static __device__ SharkForceInlineReleaseOnly void
ForwardWarpStages(DebugGlobalCount<SharkFloatParams> *debugCombo,
                  const uint64_t *tileData,
                  uint64_t *output,
                  const uint64_t *twiddleCache,
                  uint32_t tileOffset,
                  uint32_t length,
                  uint32_t warpStageCount)
{
    if (warpStageCount == TileSevenWarpStageCount) {
        ForwardWarpStagesSeven(debugCombo, tileData, output, twiddleCache, tileOffset, length);
    } else {
        ForwardWarpStagesSix(debugCombo, tileData, output, twiddleCache, tileOffset, length);
    }
}

template <class SharkFloatParams>
static __device__ SharkForceInlineReleaseOnly void
InverseWarpStagesSix(DebugGlobalCount<SharkFloatParams> *debugCombo,
                     uint64_t *tileData,
                     const uint64_t *twiddleCache,
                     uint32_t length)
{
    constexpr uint32_t CoefficientsPerWarp = 1u << TileSixWarpStageCount;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t subgroupLane = lane & 15u;
    const uint32_t subgroupIndex = lane >> 4u;
    const uint32_t warpIndex = threadIdx.x >> 5u;
    const uint32_t warpCount = blockDim.x >> 5u;
    const uint32_t groupCount = length / CoefficientsPerWarp;
    for (uint32_t group = warpIndex; group < groupCount; group += warpCount) {
        const uint32_t groupBase = group * CoefficientsPerWarp;
        const uint32_t inputBase = groupBase + subgroupIndex * 32u + subgroupLane * 2u;
        uint64_t upper = tileData[inputBase];
        uint64_t lower = tileData[inputBase + 1u];
        InverseWarpSubgroupStagesOne(debugCombo, twiddleCache, subgroupLane, upper, lower);

        const bool ownsLowerInput = (lane & 16u) != 0u;
        RegroupWarpButterfly<32>(upper, lower, 16u, ownsLowerInput);
        const uint64_t stageSixTwiddle = twiddleCache[31u + lane];
        ApplyWarpDITButterfly(debugCombo, stageSixTwiddle, upper, lower);
        tileData[groupBase + lane] = upper;
        tileData[groupBase + lane + 32u] = lower;
    }
}

template <class SharkFloatParams>
static __device__ SharkForceInlineReleaseOnly void
InverseWarpStagesSeven(DebugGlobalCount<SharkFloatParams> *debugCombo,
                       uint64_t *tileData,
                       const uint64_t *twiddleCache,
                       uint32_t length)
{
    constexpr uint32_t CoefficientsPerWarp = 1u << TileSevenWarpStageCount;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t subgroupLane = lane & 15u;
    const uint32_t subgroupIndex = lane >> 4u;
    const uint32_t warpIndex = threadIdx.x >> 5u;
    const uint32_t warpCount = blockDim.x >> 5u;
    const uint32_t groupCount = length / CoefficientsPerWarp;
    for (uint32_t group = warpIndex; group < groupCount; group += warpCount) {
        const uint32_t groupBase = group * CoefficientsPerWarp;
        const uint32_t firstInputBase = groupBase + subgroupIndex * 32u + subgroupLane * 2u;
        const uint32_t secondInputBase = firstInputBase + 64u;
        uint64_t firstUpper = tileData[firstInputBase];
        uint64_t firstLower = tileData[firstInputBase + 1u];
        uint64_t secondUpper = tileData[secondInputBase];
        uint64_t secondLower = tileData[secondInputBase + 1u];
        InverseWarpSubgroupStagesTwo(
            debugCombo, twiddleCache, subgroupLane, firstUpper, firstLower, secondUpper, secondLower);

        const bool ownsLowerInput = (lane & 16u) != 0u;
        RegroupWarpButterfly<32>(firstUpper, firstLower, 16u, ownsLowerInput);
        RegroupWarpButterfly<32>(secondUpper, secondLower, 16u, ownsLowerInput);

        uint64_t twiddle = twiddleCache[31u + lane];
        ApplyWarpDITButterfly(debugCombo, twiddle, firstUpper, firstLower);
        ApplyWarpDITButterfly(debugCombo, twiddle, secondUpper, secondLower);

        twiddle = twiddleCache[63u + lane];
        ApplyWarpDITButterfly(debugCombo, twiddle, firstUpper, secondUpper);
        twiddle = twiddleCache[95u + lane];
        ApplyWarpDITButterfly(debugCombo, twiddle, firstLower, secondLower);

        tileData[groupBase + lane] = firstUpper;
        tileData[groupBase + lane + 32u] = firstLower;
        tileData[groupBase + lane + 64u] = secondUpper;
        tileData[groupBase + lane + 96u] = secondLower;
    }
}

template <class SharkFloatParams>
static __device__ SharkForceInlineReleaseOnly void
InverseWarpStages(DebugGlobalCount<SharkFloatParams> *debugCombo,
                  uint64_t *tileData,
                  const uint64_t *twiddleCache,
                  uint32_t length,
                  uint32_t warpStageCount)
{
    if (warpStageCount == TileSevenWarpStageCount) {
        InverseWarpStagesSeven(debugCombo, tileData, twiddleCache, length);
    } else {
        InverseWarpStagesSix(debugCombo, tileData, twiddleCache, length);
    }
}

template <class SharkFloatParams, uint32_t TileSizeLog2>
static __device__ __noinline__ void
ForwardTileOne(DebugGlobalCount<SharkFloatParams> *debugCombo,
               uint64_t *sharedData,
               uint64_t *input,
               const uint64_t *stageTwiddles,
               uint32_t transformSize,
               uint32_t stageCount)
{
    cooperative_groups::thread_block block = cooperative_groups::this_thread_block();
    constexpr uint32_t TileSize = 1u << TileSizeLog2;
    static_assert(TileSizeLog2 >= 9u && TileSizeLog2 <= 12u);
    static_assert(TileSize + TileTwiddleCacheWords <= TileSharedWords);
    uint64_t *const tileData = sharedData;
    uint64_t *const twiddleCache = tileData + TileSize;
    const uint32_t smallStages = stageCount < TileSizeLog2 ? stageCount : TileSizeLog2;
    const uint32_t cachedStages = smallStages < 11u ? smallStages : 11u;
    const uint32_t cachedWords = cachedStages == 0u ? 0u : (1u << cachedStages) - 1u;
    const uint32_t cachedCopyWords = cachedStages == 0u ? 0u : 1u << cachedStages;
    if (cachedCopyWords != 0u) {
        cooperative_groups::memcpy_async(block,
                                         twiddleCache,
                                         stageTwiddles,
                                         cuda::aligned_size_t<16>(cachedCopyWords * sizeof(uint64_t)));
        cooperative_groups::wait(block);
    }
    const uint32_t tileCount = (transformSize + TileSize - 1u) / TileSize;
    const uint32_t cohortBlockIndex = CohortBlockIndex();
    const uint32_t cohortBlockCount = CohortBlockCount();
    for (uint32_t tile = cohortBlockIndex; tile < tileCount; tile += cohortBlockCount) {
        const uint32_t remainder = transformSize & (TileSize - 1u);
        const uint32_t length = tile + 1u == tileCount && remainder != 0u ? remainder : TileSize;
        const uint32_t tileOffset = tile * TileSize;
        const uint32_t warpStageCount = SelectTileWarpStageCount(length);
        cooperative_groups::memcpy_async(
            block, tileData, input + tileOffset, cuda::aligned_size_t<16>(length * sizeof(uint64_t)));
        cooperative_groups::wait(block);
        for (uint32_t stage = smallStages; stage > warpStageCount; --stage) {
            ForwardSharedStage(
                debugCombo, tileData, twiddleCache, stageTwiddles, cachedWords, length, stage);
            block.sync();
        }
        ForwardWarpStages(debugCombo, tileData, input, twiddleCache, tileOffset, length, warpStageCount);
        if (tile + cohortBlockCount < tileCount)
            block.sync();
    }
}

template <class SharkFloatParams, uint32_t TileSizeLog2>
static __device__ __noinline__ void
ForwardTileTwo(DebugGlobalCount<SharkFloatParams> *debugCombo,
               uint64_t *sharedData,
               uint64_t *inputA,
               uint64_t *inputB,
               const uint64_t *stageTwiddles,
               uint32_t transformSize,
               uint32_t stageCount)
{
    cooperative_groups::thread_block block = cooperative_groups::this_thread_block();
    constexpr uint32_t TileSize = 1u << TileSizeLog2;
    static_assert(TileSizeLog2 >= 9u && TileSizeLog2 <= 11u);
    static_assert(2u * TileSize + TileTwiddleCacheWords <= TileSharedWords);
    uint64_t *const tileDataA = sharedData;
    uint64_t *const tileDataB = tileDataA + TileSize;
    uint64_t *const twiddleCache = tileDataB + TileSize;
    const uint32_t smallStages = stageCount < TileSizeLog2 ? stageCount : TileSizeLog2;
    const uint32_t cachedStages = smallStages < 11u ? smallStages : 11u;
    const uint32_t cachedWords = cachedStages == 0u ? 0u : (1u << cachedStages) - 1u;
    const uint32_t cachedCopyWords = cachedStages == 0u ? 0u : 1u << cachedStages;
    if (cachedCopyWords != 0u) {
        cooperative_groups::memcpy_async(block,
                                         twiddleCache,
                                         stageTwiddles,
                                         cuda::aligned_size_t<16>(cachedCopyWords * sizeof(uint64_t)));
        cooperative_groups::wait(block);
    }
    const uint32_t tileCount = (transformSize + TileSize - 1u) / TileSize;
    const uint32_t cohortBlockIndex = CohortBlockIndex();
    const uint32_t cohortBlockCount = CohortBlockCount();
    for (uint32_t tile = cohortBlockIndex; tile < tileCount; tile += cohortBlockCount) {
        const uint32_t remainder = transformSize & (TileSize - 1u);
        const uint32_t length = tile + 1u == tileCount && remainder != 0u ? remainder : TileSize;
        const uint32_t tileOffset = tile * TileSize;
        const uint32_t warpStageCount = SelectTileWarpStageCount(length);
        cooperative_groups::memcpy_async(
            block, tileDataA, inputA + tileOffset, cuda::aligned_size_t<16>(length * sizeof(uint64_t)));
        cooperative_groups::memcpy_async(
            block, tileDataB, inputB + tileOffset, cuda::aligned_size_t<16>(length * sizeof(uint64_t)));
        cooperative_groups::wait(block);
        for (uint32_t stage = smallStages; stage > warpStageCount; --stage) {
            ForwardSharedStage(
                debugCombo, tileDataA, twiddleCache, stageTwiddles, cachedWords, length, stage);
            ForwardSharedStage(
                debugCombo, tileDataB, twiddleCache, stageTwiddles, cachedWords, length, stage);
            block.sync();
        }
        ForwardWarpStages(
            debugCombo, tileDataA, inputA, twiddleCache, tileOffset, length, warpStageCount);
        ForwardWarpStages(
            debugCombo, tileDataB, inputB, twiddleCache, tileOffset, length, warpStageCount);
        if (tile + cohortBlockCount < tileCount)
            block.sync();
    }
}

enum class InverseTileOneLoad : uint8_t {
    Copy,
    OrbitReal,
    OrbitImag,
};

enum class InverseTileTwoLoad : uint8_t {
    Copy,
    OrbitPair,
    DerivativePair,
};

template <class SharkFloatParams, uint32_t TileSizeLog2, InverseTileOneLoad Load>
static __device__ __noinline__ void
InverseTileOneCore(DebugGlobalCount<SharkFloatParams> *debugCombo,
                   uint64_t *sharedData,
                   const uint64_t *sourceA,
                   const uint64_t *sourceB,
                   uint64_t *output,
                   bool productEnabled,
                   const uint64_t *stageTwiddles,
                   uint32_t transformSize,
                   uint32_t stageCount)
{
    if constexpr (Load == InverseTileOneLoad::Copy) {
        (void)sourceB;
        (void)productEnabled;
    }

    cooperative_groups::thread_block block = cooperative_groups::this_thread_block();
    constexpr uint32_t TileSize = 1u << TileSizeLog2;
    static_assert(TileSizeLog2 >= 9u && TileSizeLog2 <= 12u);
    static_assert(TileSize + TileTwiddleCacheWords <= TileSharedWords);
    uint64_t *const tileData = sharedData;
    uint64_t *const twiddleCache = tileData + TileSize;
    const uint32_t smallStages = stageCount < TileSizeLog2 ? stageCount : TileSizeLog2;
    const uint32_t cachedStages = smallStages < 11u ? smallStages : 11u;
    const uint32_t cachedWords = cachedStages == 0u ? 0u : (1u << cachedStages) - 1u;
    const uint32_t cachedCopyWords = cachedStages == 0u ? 0u : 1u << cachedStages;
    if (cachedCopyWords != 0u) {
        cooperative_groups::memcpy_async(block,
                                         twiddleCache,
                                         stageTwiddles,
                                         cuda::aligned_size_t<16>(cachedCopyWords * sizeof(uint64_t)));
        cooperative_groups::wait(block);
    }
    const uint32_t tileCount = (transformSize + TileSize - 1u) / TileSize;
    const uint32_t cohortBlockIndex = CohortBlockIndex();
    const uint32_t cohortBlockCount = CohortBlockCount();
    for (uint32_t tile = cohortBlockIndex; tile < tileCount; tile += cohortBlockCount) {
        const uint32_t remainder = transformSize & (TileSize - 1u);
        const uint32_t length = tile + 1u == tileCount && remainder != 0u ? remainder : TileSize;
        const uint32_t tileOffset = tile * TileSize;
        const uint32_t warpStageCount = SelectTileWarpStageCount(length);
        if constexpr (Load == InverseTileOneLoad::Copy) {
            cooperative_groups::memcpy_async(block,
                                             tileData,
                                             sourceA + tileOffset,
                                             cuda::aligned_size_t<16>(length * sizeof(uint64_t)));
            cooperative_groups::wait(block);
        } else {
            for (uint32_t i = block.thread_index().x; i < length; i += block.size()) {
                const uint32_t index = tileOffset + i;
                if (!productEnabled) {
                    tileData[i] = 0ull;
                } else {
                    const uint64_t real = sourceA[index];
                    const uint64_t imag = sourceB[index];
                    if constexpr (Load == InverseTileOneLoad::OrbitReal) {
                        tileData[i] = MontgomeryMul(debugCombo, AddP(real, imag), SubP(real, imag));
                    } else {
                        const uint64_t product = MontgomeryMul(debugCombo, real, imag);
                        tileData[i] = AddP(product, product);
                    }
                }
            }
            block.sync();
        }

        InverseWarpStages(debugCombo, tileData, twiddleCache, length, warpStageCount);
        block.sync();
        for (uint32_t stage = warpStageCount + 1u; stage < smallStages; ++stage) {
            InverseSharedStage(
                debugCombo, tileData, twiddleCache, stageTwiddles, cachedWords, length, stage);
            block.sync();
        }
        const uint32_t butterflySpan = 1u << smallStages;
        const uint32_t halfSpan = butterflySpan >> 1u;
        const uint32_t twiddleOffset = halfSpan - 1u;
        for (uint32_t pair = block.thread_index().x; pair < length / 2u; pair += block.size()) {
            const uint32_t group = pair / halfSpan;
            const uint32_t j = pair - group * halfSpan;
            const uint32_t upperIndex = group * butterflySpan + j;
            const uint32_t lowerIndex = upperIndex + halfSpan;
            const uint64_t upper = tileData[upperIndex];
            const uint64_t lower = tileData[lowerIndex];
            const uint32_t twiddleIndex = twiddleOffset + j;
            const uint64_t twiddle =
                twiddleIndex < cachedWords ? twiddleCache[twiddleIndex] : stageTwiddles[twiddleIndex];
            const uint64_t product = MontgomeryMul(debugCombo, lower, twiddle);
            output[tileOffset + upperIndex] = AddP(upper, product);
            output[tileOffset + lowerIndex] = SubP(upper, product);
        }
        if (tile + cohortBlockCount < tileCount)
            block.sync();
    }
}

template <class SharkFloatParams, uint32_t TileSizeLog2, InverseTileTwoLoad Load>
static __device__ __noinline__ void
InverseTileTwoCore(DebugGlobalCount<SharkFloatParams> *debugCombo,
                   uint64_t *sharedData,
                   const uint64_t *sourceA,
                   const uint64_t *sourceB,
                   const uint64_t *sourceC,
                   const uint64_t *sourceD,
                   uint64_t *outputA,
                   uint64_t *outputB,
                   bool firstProductEnabled,
                   bool secondProductEnabled,
                   const uint64_t *stageTwiddles,
                   uint32_t transformSize,
                   uint32_t stageCount)
{
    if constexpr (Load == InverseTileTwoLoad::Copy) {
        (void)sourceC;
        (void)sourceD;
        (void)firstProductEnabled;
        (void)secondProductEnabled;
    } else if constexpr (Load == InverseTileTwoLoad::OrbitPair) {
        (void)sourceC;
        (void)sourceD;
    } else {
        (void)secondProductEnabled;
    }

    cooperative_groups::thread_block block = cooperative_groups::this_thread_block();
    constexpr uint32_t TileSize = 1u << TileSizeLog2;
    static_assert(TileSizeLog2 >= 9u && TileSizeLog2 <= 11u);
    static_assert(2u * TileSize + TileTwiddleCacheWords <= TileSharedWords);
    uint64_t *const tileDataA = sharedData;
    uint64_t *const tileDataB = tileDataA + TileSize;
    uint64_t *const twiddleCache = tileDataB + TileSize;
    const uint32_t smallStages = stageCount < TileSizeLog2 ? stageCount : TileSizeLog2;
    const uint32_t cachedStages = smallStages < 11u ? smallStages : 11u;
    const uint32_t cachedWords = cachedStages == 0u ? 0u : (1u << cachedStages) - 1u;
    const uint32_t cachedCopyWords = cachedStages == 0u ? 0u : 1u << cachedStages;
    if (cachedCopyWords != 0u) {
        cooperative_groups::memcpy_async(block,
                                         twiddleCache,
                                         stageTwiddles,
                                         cuda::aligned_size_t<16>(cachedCopyWords * sizeof(uint64_t)));
        cooperative_groups::wait(block);
    }
    const uint32_t tileCount = (transformSize + TileSize - 1u) / TileSize;
    const uint32_t cohortBlockIndex = CohortBlockIndex();
    const uint32_t cohortBlockCount = CohortBlockCount();
    for (uint32_t tile = cohortBlockIndex; tile < tileCount; tile += cohortBlockCount) {
        const uint32_t remainder = transformSize & (TileSize - 1u);
        const uint32_t length = tile + 1u == tileCount && remainder != 0u ? remainder : TileSize;
        const uint32_t tileOffset = tile * TileSize;
        const uint32_t warpStageCount = SelectTileWarpStageCount(length);
        if constexpr (Load == InverseTileTwoLoad::Copy) {
            cooperative_groups::memcpy_async(block,
                                             tileDataA,
                                             sourceA + tileOffset,
                                             cuda::aligned_size_t<16>(length * sizeof(uint64_t)));
            cooperative_groups::memcpy_async(block,
                                             tileDataB,
                                             sourceB + tileOffset,
                                             cuda::aligned_size_t<16>(length * sizeof(uint64_t)));
            cooperative_groups::wait(block);
        } else {
            for (uint32_t i = block.thread_index().x; i < length; i += block.size()) {
                const uint32_t index = tileOffset + i;
                if constexpr (Load == InverseTileTwoLoad::OrbitPair) {
                    if (firstProductEnabled || secondProductEnabled) {
                        const uint64_t real = sourceA[index];
                        const uint64_t imag = sourceB[index];
                        if (firstProductEnabled) {
                            tileDataA[i] = MontgomeryMul(debugCombo, AddP(real, imag), SubP(real, imag));
                        } else {
                            tileDataA[i] = 0ull;
                        }
                        if (secondProductEnabled) {
                            const uint64_t product = MontgomeryMul(debugCombo, real, imag);
                            tileDataB[i] = AddP(product, product);
                        } else {
                            tileDataB[i] = 0ull;
                        }
                    } else {
                        tileDataA[i] = 0ull;
                        tileDataB[i] = 0ull;
                    }
                } else if (firstProductEnabled) {
                    const uint64_t real = sourceA[index];
                    const uint64_t imag = sourceB[index];
                    const uint64_t derivativeReal = sourceC[index];
                    const uint64_t derivativeImag = sourceD[index];
                    const uint64_t p1 = MontgomeryMul(debugCombo, real, derivativeReal);
                    const uint64_t p2 = MontgomeryMul(debugCombo, imag, derivativeImag);
                    const uint64_t p3 = MontgomeryMul(
                        debugCombo, AddP(real, imag), AddP(derivativeReal, derivativeImag));
                    const uint64_t realDifference = SubP(p1, p2);
                    const uint64_t imagDifference = SubP(SubP(p3, p1), p2);
                    tileDataA[i] = AddP(realDifference, realDifference);
                    tileDataB[i] = AddP(imagDifference, imagDifference);
                } else {
                    tileDataA[i] = 0ull;
                    tileDataB[i] = 0ull;
                }
            }
            block.sync();
        }

        InverseWarpStages(debugCombo, tileDataA, twiddleCache, length, warpStageCount);
        InverseWarpStages(debugCombo, tileDataB, twiddleCache, length, warpStageCount);
        block.sync();
        for (uint32_t stage = warpStageCount + 1u; stage < smallStages; ++stage) {
            InverseSharedStage(
                debugCombo, tileDataA, twiddleCache, stageTwiddles, cachedWords, length, stage);
            InverseSharedStage(
                debugCombo, tileDataB, twiddleCache, stageTwiddles, cachedWords, length, stage);
            block.sync();
        }
        const uint32_t butterflySpan = 1u << smallStages;
        const uint32_t halfSpan = butterflySpan >> 1u;
        const uint32_t twiddleOffset = halfSpan - 1u;
        for (uint32_t pair = block.thread_index().x; pair < length / 2u; pair += block.size()) {
            const uint32_t group = pair / halfSpan;
            const uint32_t j = pair - group * halfSpan;
            const uint32_t upperIndex = group * butterflySpan + j;
            const uint32_t lowerIndex = upperIndex + halfSpan;
            const uint32_t twiddleIndex = twiddleOffset + j;
            const uint64_t twiddle =
                twiddleIndex < cachedWords ? twiddleCache[twiddleIndex] : stageTwiddles[twiddleIndex];
            const uint64_t upper = tileDataA[upperIndex];
            const uint64_t product = MontgomeryMul(debugCombo, tileDataA[lowerIndex], twiddle);
            outputA[tileOffset + upperIndex] = AddP(upper, product);
            outputA[tileOffset + lowerIndex] = SubP(upper, product);
        }
        for (uint32_t pair = block.thread_index().x; pair < length / 2u; pair += block.size()) {
            const uint32_t group = pair / halfSpan;
            const uint32_t j = pair - group * halfSpan;
            const uint32_t upperIndex = group * butterflySpan + j;
            const uint32_t lowerIndex = upperIndex + halfSpan;
            const uint32_t twiddleIndex = twiddleOffset + j;
            const uint64_t twiddle =
                twiddleIndex < cachedWords ? twiddleCache[twiddleIndex] : stageTwiddles[twiddleIndex];
            const uint64_t upper = tileDataB[upperIndex];
            const uint64_t product = MontgomeryMul(debugCombo, tileDataB[lowerIndex], twiddle);
            outputB[tileOffset + upperIndex] = AddP(upper, product);
            outputB[tileOffset + lowerIndex] = SubP(upper, product);
        }
        if (tile + cohortBlockCount < tileCount)
            block.sync();
    }
}

template <class SharkFloatParams>
static __device__ __noinline__ void
ForwardTileOneSelected(DebugGlobalCount<SharkFloatParams> *debugCombo,
                       uint64_t *sharedData,
                       uint64_t *input,
                       const uint64_t *stageTwiddles,
                       uint32_t transformSize,
                       uint32_t stageCount,
                       uint32_t tileSizeLog2)
{
    if (tileSizeLog2 == 12u) {
        ForwardTileOne<SharkFloatParams, 12u>(
            debugCombo, sharedData, input, stageTwiddles, transformSize, stageCount);
    } else if (tileSizeLog2 == 11u) {
        ForwardTileOne<SharkFloatParams, 11u>(
            debugCombo, sharedData, input, stageTwiddles, transformSize, stageCount);
    } else if (tileSizeLog2 == 10u) {
        ForwardTileOne<SharkFloatParams, 10u>(
            debugCombo, sharedData, input, stageTwiddles, transformSize, stageCount);
    } else {
        ForwardTileOne<SharkFloatParams, 9u>(
            debugCombo, sharedData, input, stageTwiddles, transformSize, stageCount);
    }
}

template <class SharkFloatParams>
static __device__ __noinline__ void
ForwardTileTwoSelected(DebugGlobalCount<SharkFloatParams> *debugCombo,
                       uint64_t *sharedData,
                       uint64_t *inputA,
                       uint64_t *inputB,
                       const uint64_t *stageTwiddles,
                       uint32_t transformSize,
                       uint32_t stageCount,
                       uint32_t tileSizeLog2)
{
    if (tileSizeLog2 == 11u) {
        ForwardTileTwo<SharkFloatParams, 11u>(
            debugCombo, sharedData, inputA, inputB, stageTwiddles, transformSize, stageCount);
    } else if (tileSizeLog2 == 10u) {
        ForwardTileTwo<SharkFloatParams, 10u>(
            debugCombo, sharedData, inputA, inputB, stageTwiddles, transformSize, stageCount);
    } else {
        ForwardTileTwo<SharkFloatParams, 9u>(
            debugCombo, sharedData, inputA, inputB, stageTwiddles, transformSize, stageCount);
    }
}

template <class SharkFloatParams, InverseTileOneLoad Load>
static __device__ __noinline__ void
InverseTileOneSelected(DebugGlobalCount<SharkFloatParams> *debugCombo,
                       uint64_t *sharedData,
                       const uint64_t *sourceA,
                       const uint64_t *sourceB,
                       uint64_t *output,
                       bool productEnabled,
                       const uint64_t *stageTwiddles,
                       uint32_t transformSize,
                       uint32_t stageCount,
                       uint32_t tileSizeLog2)
{
    if (tileSizeLog2 == 12u) {
        InverseTileOneCore<SharkFloatParams, 12u, Load>(debugCombo,
                                                        sharedData,
                                                        sourceA,
                                                        sourceB,
                                                        output,
                                                        productEnabled,
                                                        stageTwiddles,
                                                        transformSize,
                                                        stageCount);
    } else if (tileSizeLog2 == 11u) {
        InverseTileOneCore<SharkFloatParams, 11u, Load>(debugCombo,
                                                        sharedData,
                                                        sourceA,
                                                        sourceB,
                                                        output,
                                                        productEnabled,
                                                        stageTwiddles,
                                                        transformSize,
                                                        stageCount);
    } else if (tileSizeLog2 == 10u) {
        InverseTileOneCore<SharkFloatParams, 10u, Load>(debugCombo,
                                                        sharedData,
                                                        sourceA,
                                                        sourceB,
                                                        output,
                                                        productEnabled,
                                                        stageTwiddles,
                                                        transformSize,
                                                        stageCount);
    } else {
        InverseTileOneCore<SharkFloatParams, 9u, Load>(debugCombo,
                                                       sharedData,
                                                       sourceA,
                                                       sourceB,
                                                       output,
                                                       productEnabled,
                                                       stageTwiddles,
                                                       transformSize,
                                                       stageCount);
    }
}

template <class SharkFloatParams, InverseTileTwoLoad Load>
static __device__ __noinline__ void
InverseTileTwoSelected(DebugGlobalCount<SharkFloatParams> *debugCombo,
                       uint64_t *sharedData,
                       const uint64_t *sourceA,
                       const uint64_t *sourceB,
                       const uint64_t *sourceC,
                       const uint64_t *sourceD,
                       uint64_t *outputA,
                       uint64_t *outputB,
                       bool firstProductEnabled,
                       bool secondProductEnabled,
                       const uint64_t *stageTwiddles,
                       uint32_t transformSize,
                       uint32_t stageCount,
                       uint32_t tileSizeLog2)
{
    if (tileSizeLog2 == 11u) {
        InverseTileTwoCore<SharkFloatParams, 11u, Load>(debugCombo,
                                                        sharedData,
                                                        sourceA,
                                                        sourceB,
                                                        sourceC,
                                                        sourceD,
                                                        outputA,
                                                        outputB,
                                                        firstProductEnabled,
                                                        secondProductEnabled,
                                                        stageTwiddles,
                                                        transformSize,
                                                        stageCount);
    } else if (tileSizeLog2 == 10u) {
        InverseTileTwoCore<SharkFloatParams, 10u, Load>(debugCombo,
                                                        sharedData,
                                                        sourceA,
                                                        sourceB,
                                                        sourceC,
                                                        sourceD,
                                                        outputA,
                                                        outputB,
                                                        firstProductEnabled,
                                                        secondProductEnabled,
                                                        stageTwiddles,
                                                        transformSize,
                                                        stageCount);
    } else {
        InverseTileTwoCore<SharkFloatParams, 9u, Load>(debugCombo,
                                                       sharedData,
                                                       sourceA,
                                                       sourceB,
                                                       sourceC,
                                                       sourceD,
                                                       outputA,
                                                       outputB,
                                                       firstProductEnabled,
                                                       secondProductEnabled,
                                                       stageTwiddles,
                                                       transformSize,
                                                       stageCount);
    }
}

} // namespace NTT

#ifdef _DEBUG
static __device__ SharkForceInlineReleaseOnly void
MattsCudaAssert(bool cond)
{
    if (!cond)
        asm volatile("trap;");
}
#else
static __device__ SharkForceInlineReleaseOnly void
MattsCudaAssert(bool)
{
    // no-op in release builds
}
#endif

struct FusedTerm {
    bool IsZero;
    int32_t Exponent;
};

static __device__ bool
IsLeader(const cooperative_groups::thread_block &block)
{
    return block.group_index().x == 0 && block.thread_index().x == 0;
}

static __device__ uint32_t
GridThreadRank(const cooperative_groups::thread_block &block)
{
    return block.thread_index().x + block.group_index().x * blockDim.x;
}

static __device__ uint32_t
GridThreadRank()
{
    return threadIdx.x + blockIdx.x * blockDim.x;
}

// The producer phase must publish all input arrays with grid.sync() before this checkpoint.
template <class SharkFloatParams, class ArrayType>
__device__ void
StoreReferenceDebugStateBatchAfterSync(DebugState<SharkFloatParams> *debugStates,
                                       cooperative_groups::grid_group &grid,
                                       cooperative_groups::thread_block &block,
                                       DebugStatePurpose purpose0,
                                       const ArrayType *array0,
                                       DebugStatePurpose purpose1,
                                       const ArrayType *array1,
                                       size_t arraySize)
{
    if constexpr (HpShark::DebugChecksums) {
        StoreCurrentDebugState<SharkFloatParams, ArrayType>(
            debugStates, grid, block, purpose0, array0, arraySize);
        StoreCurrentDebugState<SharkFloatParams, ArrayType>(
            debugStates, grid, block, purpose1, array1, arraySize);
        grid.sync();
    }
}

template <class SharkFloatParams, class ArrayType>
__device__ void
StoreReferenceDebugStateBatchAfterSync(DebugState<SharkFloatParams> *debugStates,
                                       cooperative_groups::grid_group &grid,
                                       cooperative_groups::thread_block &block,
                                       DebugStatePurpose purpose0,
                                       const ArrayType *array0,
                                       DebugStatePurpose purpose1,
                                       const ArrayType *array1,
                                       DebugStatePurpose purpose2,
                                       const ArrayType *array2,
                                       DebugStatePurpose purpose3,
                                       const ArrayType *array3,
                                       size_t arraySize)
{
    if constexpr (HpShark::DebugChecksums) {
        StoreCurrentDebugState<SharkFloatParams, ArrayType>(
            debugStates, grid, block, purpose0, array0, arraySize);
        StoreCurrentDebugState<SharkFloatParams, ArrayType>(
            debugStates, grid, block, purpose1, array1, arraySize);
        StoreCurrentDebugState<SharkFloatParams, ArrayType>(
            debugStates, grid, block, purpose2, array2, arraySize);
        StoreCurrentDebugState<SharkFloatParams, ArrayType>(
            debugStates, grid, block, purpose3, array3, arraySize);
        grid.sync();
    }
}

template <class SharkFloatParams, class ArrayType>
__device__ void
StoreReferenceDebugStateBatchAfterSync(DebugState<SharkFloatParams> *debugStates,
                                       cooperative_groups::grid_group &grid,
                                       cooperative_groups::thread_block &block,
                                       DebugStatePurpose purpose0,
                                       const ArrayType *array0,
                                       size_t arraySize0,
                                       DebugStatePurpose purpose1,
                                       const ArrayType *array1,
                                       size_t arraySize1)
{
    if constexpr (HpShark::DebugChecksums) {
        StoreCurrentDebugState<SharkFloatParams, ArrayType>(
            debugStates, grid, block, purpose0, array0, arraySize0);
        StoreCurrentDebugState<SharkFloatParams, ArrayType>(
            debugStates, grid, block, purpose1, array1, arraySize1);
        grid.sync();
    }
}

template <class SharkFloatParams, class ArrayType>
__device__ void
StoreReferenceDebugStateBatchAfterSync(DebugState<SharkFloatParams> *debugStates,
                                       cooperative_groups::grid_group &grid,
                                       cooperative_groups::thread_block &block,
                                       DebugStatePurpose purpose0,
                                       const ArrayType *array0,
                                       size_t arraySize0,
                                       DebugStatePurpose purpose1,
                                       const ArrayType *array1,
                                       size_t arraySize1,
                                       DebugStatePurpose purpose2,
                                       const ArrayType *array2,
                                       size_t arraySize2,
                                       DebugStatePurpose purpose3,
                                       const ArrayType *array3,
                                       size_t arraySize3)
{
    if constexpr (HpShark::DebugChecksums) {
        StoreCurrentDebugState<SharkFloatParams, ArrayType>(
            debugStates, grid, block, purpose0, array0, arraySize0);
        StoreCurrentDebugState<SharkFloatParams, ArrayType>(
            debugStates, grid, block, purpose1, array1, arraySize1);
        StoreCurrentDebugState<SharkFloatParams, ArrayType>(
            debugStates, grid, block, purpose2, array2, arraySize2);
        StoreCurrentDebugState<SharkFloatParams, ArrayType>(
            debugStates, grid, block, purpose3, array3, arraySize3);
        grid.sync();
    }
}

template <class SharkFloatParams>
__device__ void
StoreReferenceDebugValueBatch(DebugState<SharkFloatParams> *debugStates,
                              cooperative_groups::grid_group &grid,
                              cooperative_groups::thread_block &block,
                              DebugStatePurpose purpose0,
                              const HpSharkFloat<SharkFloatParams> &value0,
                              DebugStatePurpose purpose1,
                              const HpSharkFloat<SharkFloatParams> &value1)
{
    if constexpr (HpShark::DebugChecksums) {
        grid.sync();
        StoreCurrentDebugValue<SharkFloatParams>(debugStates, grid, block, purpose0, value0);
        StoreCurrentDebugValue<SharkFloatParams>(debugStates, grid, block, purpose1, value1);
        grid.sync();
    }
}

template <class SharkFloatParams>
__device__ void
StoreReferenceDebugValueBatch(DebugState<SharkFloatParams> *debugStates,
                              cooperative_groups::grid_group &grid,
                              cooperative_groups::thread_block &block,
                              DebugStatePurpose purpose0,
                              const HpSharkFloat<SharkFloatParams> &value0,
                              DebugStatePurpose purpose1,
                              const HpSharkFloat<SharkFloatParams> &value1,
                              DebugStatePurpose purpose2,
                              const HpSharkFloat<SharkFloatParams> &value2,
                              DebugStatePurpose purpose3,
                              const HpSharkFloat<SharkFloatParams> &value3)
{
    if constexpr (HpShark::DebugChecksums) {
        grid.sync();
        StoreCurrentDebugValue<SharkFloatParams>(debugStates, grid, block, purpose0, value0);
        StoreCurrentDebugValue<SharkFloatParams>(debugStates, grid, block, purpose1, value1);
        StoreCurrentDebugValue<SharkFloatParams>(debugStates, grid, block, purpose2, value2);
        StoreCurrentDebugValue<SharkFloatParams>(debugStates, grid, block, purpose3, value3);
        grid.sync();
    }
}

static __device__ uint64_t
CeilPowerOfTwo(uint64_t value)
{
    if (value <= 1)
        return 1;
    --value;
    value |= value >> 1;
    value |= value >> 2;
    value |= value >> 4;
    value |= value >> 8;
    value |= value >> 16;
    value |= value >> 32;
    return value + 1;
}

static __device__ uint32_t
CountTrailingZeros(uint32_t value)
{
    uint32_t count = 0;
    while ((value & 1u) == 0u) {
        value >>= 1;
        ++count;
    }
    return count;
}

static __device__ uint64_t
AddPSerial(uint64_t a, uint64_t b)
{
    const uint64_t sum = a + b;
    return (sum < a || sum >= SharkNTT::MagicPrime) ? sum - SharkNTT::MagicPrime : sum;
}

static __device__ uint64_t
SubPSerial(uint64_t a, uint64_t b)
{
    return (a >= b) ? a - b : a + SharkNTT::MagicPrime - b;
}

template <class SharkFloatParams>
__device__ bool
IsZero(const HpSharkFloat<SharkFloatParams> &value)
{
    // reference kernel finalization keeps every nonzero value normalized to the high bit of the top
    // limb.
    const uint32_t top = value.Digits[SharkFloatParams::GlobalNumUint32 - 1];
    MattsCudaAssert(top == 0u || (top & 0x8000'0000u) != 0u);
    return top == 0u;
}

template <class SharkFloatParams>
__device__ void
SetZeroDigits(cooperative_groups::grid_group &grid,
              cooperative_groups::thread_block &block,
              HpSharkFloat<SharkFloatParams> *output)
{
    constexpr uint32_t DigitCount = SharkFloatParams::GlobalNumUint32;
    const uint32_t gridSize = static_cast<uint32_t>(grid.size());
    for (uint32_t digitIndex = GridThreadRank(block); digitIndex < DigitCount; digitIndex += gridSize) {
        output->Digits[digitIndex] = 0u;
    }
}

template <class SharkFloatParams>
__device__ void
SetZeroMetadata(cooperative_groups::thread_block &block, HpSharkFloat<SharkFloatParams> *output)
{
    if (IsLeader(block)) {
        output->Exponent = -100'000'000;
        output->SetNegative(false);
    }
}

template <class SharkFloatParams>
__device__ typename SharkFloatParams::Float
ToNormalizedHDRFloat(const HpSharkFloat<SharkFloatParams> &value)
{
    using SubType = typename SharkFloatParams::SubType;
    using Hdr = typename SharkFloatParams::Float;
    constexpr int TopIndex = SharkFloatParams::GlobalNumUint32 - 1;
    constexpr int MsbInWindow = 63;
    constexpr int32_t MantissaExponent = TopIndex * 32 + 31;

    const uint32_t high = value.Digits[TopIndex];
    if (high == 0u)
        return Hdr{};

    MattsCudaAssert((high & 0x8000'0000u) != 0u);
    const uint32_t low = TopIndex > 0 ? value.Digits[TopIndex - 1] : 0u;
    const uint64_t window = (static_cast<uint64_t>(high) << 32u) | low;
    const int32_t finalExponent = MantissaExponent + value.Exponent;

    if constexpr (std::is_same_v<SubType, CudaDblflt<dblflt>>) {
        double mantissa = static_cast<double>(window) / static_cast<double>(1ull << MsbInWindow);
        if (value.GetNegative())
            mantissa = -mantissa;
        HDRFloat<double> temporary(finalExponent, mantissa);
        HdrReduce(temporary);
        return Hdr{temporary};
    } else {
        SubType mantissa = SubType(window) / std::ldexp(SubType(1), MsbInWindow);
        if (value.GetNegative())
            mantissa = -mantissa;
        Hdr result(finalExponent, mantissa);
        HdrReduce(result);
        return result;
    }
}

template <class SharkFloatParams>
__device__ SharkForceInlineReleaseOnly FusedTerm
MakeLinearTerm(const HpSharkFloat<SharkFloatParams> &a, uint32_t ignoredPrecisionBits)
{
    if (IsZero(a))
        return {true, 0};
    const int64_t exponent =
        static_cast<int64_t>(a.Exponent) + static_cast<int64_t>(ignoredPrecisionBits);
    MattsCudaAssert(exponent >= INT32_MIN && exponent <= INT32_MAX);
    return {false, static_cast<int32_t>(exponent)};
}

template <class SharkFloatParams>
__device__ SharkForceInlineReleaseOnly bool
ResolveAlignedValueExponent(int32_t *commonExponent,
                            const HpSharkFloat<SharkFloatParams> &value0,
                            const HpSharkFloat<SharkFloatParams> &value1)
{
    const bool value0Zero = IsZero(value0);
    const bool value1Zero = IsZero(value1);
    if (value0Zero && value1Zero) {
        *commonExponent = 0;
        return true;
    }
    if (value0Zero) {
        *commonExponent = value1.Exponent;
        return false;
    }
    if (value1Zero) {
        *commonExponent = value0.Exponent;
        return false;
    }
    *commonExponent = value0.Exponent < value1.Exponent ? value0.Exponent : value1.Exponent;
    return false;
}

static __device__ SharkForceInlineReleaseOnly FusedTerm
MakeAlignedProductTerm(bool isZero, int32_t exponent)
{
    return {isZero, isZero ? 0 : exponent};
}

static __device__ SharkForceInlineReleaseOnly void
IncludeTermInCommonExponent(FusedTerm term, bool &any, int32_t &common)
{
    if (term.IsZero)
        return;
    common = any && common < term.Exponent ? common : term.Exponent;
    any = true;
}

template <class... RemainingTerms>
__device__ SharkForceInlineReleaseOnly bool
ResolveCommonExponent(int32_t *commonExponent, FusedTerm firstTerm, RemainingTerms... remainingTerms)
{
    bool any = false;
    int32_t common = 0;
    IncludeTermInCommonExponent(firstTerm, any, common);
    (IncludeTermInCommonExponent(remainingTerms, any, common), ...);
    *commonExponent = any ? common : 0;
    return !any;
}

template <class SharkFloatParams>
__device__ uint64_t
ReadBitsSimple(const HpSharkFloat<SharkFloatParams> &value, int64_t bitIndex, int bitCount)
{
    constexpr int TotalBits = SharkFloatParams::GlobalNumUint32 * 32;
    if (bitIndex < 0 || bitIndex >= TotalBits)
        return 0;

    uint64_t result = 0;
    int needed = bitCount;
    int outputBit = 0;
    while (needed > 0 && bitIndex < TotalBits) {
        const int64_t word = bitIndex / 32;
        const int offset = static_cast<int>(bitIndex % 32);
        const uint32_t limb = value.Digits[static_cast<int>(word)];
        const uint32_t chunk = offset == 0 ? limb : limb >> offset;
        const int take = (32 - offset) < needed ? 32 - offset : needed;
        const uint32_t mask = take == 32 ? 0xffffffffu : (1u << take) - 1u;
        result |= static_cast<uint64_t>(chunk & mask) << outputBit;
        outputBit += take;
        needed -= take;
        bitIndex += take;
    }
    return bitCount == 64 ? result : result & ((1ull << bitCount) - 1ull);
}

template <class SharkFloatParams>
__device__ uint64_t
ReadAlignedBits(const HpSharkFloat<SharkFloatParams> &value,
                uint32_t inputBitOffset,
                int64_t sourceBit,
                int bitCount)
{
    constexpr int TotalBits = SharkFloatParams::GlobalNumUint32 * 32;
    const int64_t lowerBit = static_cast<int64_t>(inputBitOffset);
    const int64_t upperBit = static_cast<int64_t>(TotalBits);
    const int64_t sourceEnd = sourceBit + static_cast<int64_t>(bitCount);
    if (sourceEnd <= lowerBit || sourceBit >= upperBit)
        return 0;

    const int64_t readStart = sourceBit < lowerBit ? lowerBit : sourceBit;
    const int64_t readEnd = sourceEnd > upperBit ? upperBit : sourceEnd;
    if (readStart >= readEnd)
        return 0;

    const int leadingZeroBits = static_cast<int>(readStart - sourceBit);
    const int availableBits = static_cast<int>(readEnd - readStart);
    const uint64_t valueBits = ReadBitsSimple(value, readStart, availableBits);
    return valueBits << leadingZeroBits;
}

template <class SharkFloatParams>
__device__ uint64_t
MontgomeryPowSerial(DebugGlobalCount<SharkFloatParams> *debugCombo, uint64_t value, uint64_t exponent)
{
    uint64_t result = NTT::ToMontgomery<SharkFloatParams>(debugCombo, 1);
    while (exponent != 0) {
        if ((exponent & 1ull) != 0)
            result = NTT::MontgomeryMul<SharkFloatParams>(debugCombo, result, value);
        value = NTT::MontgomeryMul<SharkFloatParams>(debugCombo, value, value);
        exponent >>= 1;
    }
    return result;
}

template <class SharkFloatParams>
__device__ void
GenerateCachedPlan(cooperative_groups::grid_group &grid,
                   cooperative_groups::thread_block &block,
                   DebugGlobalCount<SharkFloatParams> *debugCombo,
                   uint32_t activeN,
                   HpSharkReferenceWorkspace<SharkFloatParams> &workspace)
{
    using Workspace = HpSharkReferenceWorkspace<SharkFloatParams>;
    MattsCudaAssert(activeN >= workspace.ActiveMinFusedN && activeN <= workspace.ActiveMaxFusedN);
    MattsCudaAssert((activeN & (activeN - 1u)) == 0u);
    const uint32_t stages = CountTrailingZeros(activeN);
    MattsCudaAssert(stages >= workspace.ActiveMinFusedStages &&
                    stages <= workspace.ActiveMaxFusedStages);
    const uint32_t slot = stages - Workspace::MinFusedStages;
    MattsCudaAssert(slot < Workspace::PlanCacheEntryCount);
    const uint32_t planBit = 1u << slot;
    if ((workspace.ValidPlanMask & planBit) != 0u)
        return;

    const SharkNTT::Plan &plan = workspace.Plans[slot];
    SharkNTT::RootTables &roots = workspace.PlanRoots[slot];
    MattsCudaAssert(static_cast<uint32_t>(plan.N) == activeN);
    MattsCudaAssert(static_cast<uint32_t>(roots.N) == activeN);
    const uint32_t rank = GridThreadRank(block);
    const uint32_t gridSize = static_cast<uint32_t>(grid.size());

    constexpr uint64_t Generator = SharkNTT::FindGeneratorConstexpr();
    const uint64_t generatorMont = NTT::ToMontgomery<SharkFloatParams>(debugCombo, Generator);
    const uint64_t omega =
        MontgomeryPowSerial<SharkFloatParams>(debugCombo, generatorMont, SharkNTT::PHI / activeN);
    const uint64_t omegaInverse =
        MontgomeryPowSerial<SharkFloatParams>(debugCombo, omega, SharkNTT::PHI - 1ull);

    const uint32_t firstMissingStage = workspace.GeneratedStages + 1u;
    for (uint32_t stage = firstMissingStage; stage <= stages; ++stage) {
        const uint32_t width = 1u << stage;
        const uint32_t half = width >> 1u;
        const uint32_t offset = half - 1u;
        if (IsLeader(block)) {
            roots.stage_omegas[stage - 1u] =
                MontgomeryPowSerial<SharkFloatParams>(debugCombo, omega, activeN / width);
            roots.stage_omegas_inv[stage - 1u] =
                MontgomeryPowSerial<SharkFloatParams>(debugCombo, omegaInverse, activeN / width);
        }
        grid.sync();
        if (rank < half) {
            uint64_t forwardTwiddle =
                MontgomeryPowSerial<SharkFloatParams>(debugCombo, roots.stage_omegas[stage - 1u], rank);
            uint64_t inverseTwiddle = MontgomeryPowSerial<SharkFloatParams>(
                debugCombo, roots.stage_omegas_inv[stage - 1u], rank);
            const uint64_t forwardStride = MontgomeryPowSerial<SharkFloatParams>(
                debugCombo, roots.stage_omegas[stage - 1u], gridSize);
            const uint64_t inverseStride = MontgomeryPowSerial<SharkFloatParams>(
                debugCombo, roots.stage_omegas_inv[stage - 1u], gridSize);
            for (uint32_t index = rank; index < half; index += gridSize) {
                roots.stage_twiddles_fwd[offset + index] = forwardTwiddle;
                roots.stage_twiddles_inv[offset + index] = inverseTwiddle;
                if (index + gridSize < half) {
                    forwardTwiddle =
                        NTT::MontgomeryMul<SharkFloatParams>(debugCombo, forwardTwiddle, forwardStride);
                    inverseTwiddle =
                        NTT::MontgomeryMul<SharkFloatParams>(debugCombo, inverseTwiddle, inverseStride);
                }
            }
        }
    }

    if (IsLeader(block)) {
        if (workspace.GeneratedStages < stages)
            workspace.GeneratedStages = stages;
        const uint64_t inverseTwo =
            NTT::ToMontgomery<SharkFloatParams>(debugCombo, (SharkNTT::MagicPrime + 1ull) >> 1u);
        roots.Ninvm_mont = MontgomeryPowSerial<SharkFloatParams>(debugCombo, inverseTwo, stages);
        roots.Ninv = NTT::FromMontgomery<SharkFloatParams>(debugCombo, roots.Ninvm_mont);
        workspace.ValidPlanMask |= planBit;
    }
    grid.sync();
}

static __device__ SharkForceInlineReleaseOnly uint64_t
MultiplyB16ByMontgomeryConstant(uint64_t coefficient, uint64_t scaleR)
{
    MattsCudaAssert(coefficient <= 0xffffull);

    const uint64_t coefficient32 = static_cast<uint32_t>(coefficient);
    const uint64_t lowProduct = coefficient32 * static_cast<uint32_t>(scaleR);
    const uint64_t highProduct = coefficient32 * static_cast<uint32_t>(scaleR >> 32u);
    const uint64_t low = lowProduct + (highProduct << 32u);
    const uint64_t high = (highProduct >> 32u) + (low < lowProduct ? 1ull : 0ull);

    // The input is only 16 bits, so the product is at most 80 bits. Fold the upper
    // word with 2^64 == 2^32 - 1 (mod p), including a possible carry from the fold.
    const uint64_t folded = (high << 32u) - high;
    uint64_t result = low + folded;
    if (result < low) {
        const uint64_t beforeCarryFold = result;
        result += SharkNTT::MontgomeryR;
        if (result < beforeCarryFold)
            result += SharkNTT::MontgomeryR;
    }
    if (result >= SharkNTT::MagicPrime)
        result -= SharkNTT::MagicPrime;
    return result;
}

template <class SharkFloatParams>
static __device__ SharkForceInlineReleaseOnly uint64_t
PackAlignedForwardCoefficient(DebugGlobalCount<SharkFloatParams> *debugCombo,
                              const HpSharkFloat<SharkFloatParams> *value,
                              const SharkNTT::Plan &plan,
                              uint32_t outputIndex,
                              uint32_t inputBitOffset,
                              uint32_t coefficientShift,
                              uint32_t residualBitShift)
{
    const bool hasCoefficient = outputIndex >= coefficientShift &&
                                outputIndex - coefficientShift <
                                    static_cast<uint32_t>(plan.L) + (residualBitShift != 0u ? 1u : 0u);
    uint64_t packed = 0ull;
    if (hasCoefficient) {
        const uint32_t inputIndex = outputIndex - coefficientShift;
        const int64_t sourceBit = static_cast<int64_t>(inputBitOffset) +
                                  static_cast<int64_t>(inputIndex) * static_cast<int64_t>(plan.b) -
                                  static_cast<int64_t>(residualBitShift);
        const uint64_t coefficient =
            ReadAlignedBits(*value, inputBitOffset, sourceBit, static_cast<int>(plan.b));
        packed = NTT::ToMontgomery<SharkFloatParams>(debugCombo, coefficient % SharkNTT::MagicPrime);
        if (value->GetNegative() && coefficient != 0u)
            packed = SubPSerial(0ull, packed);
    }
    return packed;
}

template <class SharkFloatParams>
__device__ uint32_t
ReadAlignedB16Half(const HpSharkFloat<SharkFloatParams> &value,
                   uint32_t inputBitOffset,
                   int64_t halfIndex,
                   uint32_t coefficientCount)
{
    if (halfIndex < 0 || halfIndex >= static_cast<int64_t>(coefficientCount))
        return 0u;

    constexpr uint64_t TotalBits = static_cast<uint64_t>(SharkFloatParams::GlobalNumUint32) * 32ull;
    const uint64_t sourceBit =
        static_cast<uint64_t>(inputBitOffset) + static_cast<uint64_t>(halfIndex) * 16ull;
    if ((inputBitOffset & 31u) != 0u || sourceBit + 16ull > TotalBits)
        return static_cast<uint32_t>(ReadAlignedBits(value, inputBitOffset, sourceBit, 16));

    const uint32_t wordIndex = (inputBitOffset >> 5u) + static_cast<uint32_t>(halfIndex >> 1u);
    const uint32_t word = value.Digits[wordIndex];
    return (halfIndex & 1ll) == 0ll ? word & 0xffffu : word >> 16u;
}

template <class SharkFloatParams>
__device__ uint64_t
ReadAlignedB16Coefficient(const HpSharkFloat<SharkFloatParams> &value,
                          uint32_t inputBitOffset,
                          uint32_t inputIndex,
                          uint32_t coefficientCount,
                          uint32_t residualBitShift)
{
    const uint32_t current =
        ReadAlignedB16Half(value, inputBitOffset, static_cast<int64_t>(inputIndex), coefficientCount);
    if (residualBitShift == 0u)
        return current;

    const uint32_t previous = ReadAlignedB16Half(
        value, inputBitOffset, static_cast<int64_t>(inputIndex) - 1ll, coefficientCount);
    return (static_cast<uint64_t>(previous >> (16u - residualBitShift)) |
            (static_cast<uint64_t>(current) << residualBitShift)) &
           0xffffull;
}

template <class SharkFloatParams>
__device__ uint64_t
PackAlignedForwardCoefficientScaled(const HpSharkFloat<SharkFloatParams> *value,
                                    const SharkNTT::Plan &plan,
                                    uint64_t inputScaleR,
                                    uint32_t outputIndex,
                                    uint32_t inputBitOffset,
                                    uint32_t coefficientShift,
                                    uint32_t residualBitShift)
{
    MattsCudaAssert(plan.b == 16);
    const bool hasCoefficient = outputIndex >= coefficientShift &&
                                outputIndex - coefficientShift <
                                    static_cast<uint32_t>(plan.L) + (residualBitShift != 0u ? 1u : 0u);
    if (!hasCoefficient)
        return 0ull;

    const uint32_t inputIndex = outputIndex - coefficientShift;
    const uint32_t sourceCoefficientCount =
        static_cast<uint32_t>(plan.L) + (residualBitShift != 0u ? 1u : 0u);
    const uint64_t coefficient = ReadAlignedB16Coefficient(
        *value, inputBitOffset, inputIndex, sourceCoefficientCount, residualBitShift);
    uint64_t packed;
    const uint32_t stageCount = static_cast<uint32_t>(plan.stages);
    if ((stageCount & 1u) == 0u) {
        // For even stages the scale is exactly a power of two in the standard domain.
        // The compile-time table validation in NTTConstexprGenerator.h covers every
        // supported even stage, so this remains valid for non-View5 transform lengths.
        const uint32_t shift = 32u - stageCount / 2u;
        packed = coefficient << shift;
    } else {
        packed = MultiplyB16ByMontgomeryConstant(coefficient, inputScaleR);
    }
    if (value->GetNegative())
        packed = SubPSerial(0ull, packed);
    return packed;
}

template <class SharkFloatParams>
static __device__ __noinline__ void
PackForwardOne(const SharkNTT::Plan &plan,
               DebugGlobalCount<SharkFloatParams> *debugCombo,
               uint64_t inputScaleR,
               const HpSharkFloat<SharkFloatParams> *value,
               uint64_t *output,
               uint32_t inputBitOffset,
               uint32_t coefficientShift,
               uint32_t residualBitShift)
{
    const uint32_t cohortBlockIndex = gridDim.x >= 2u ? blockIdx.x >> 1u : 0u;
    const size_t threadIndex = static_cast<size_t>(cohortBlockIndex) * blockDim.x + threadIdx.x;
    const size_t gridSize = NTT::CohortGridSize();
    const uint32_t transformSize = static_cast<uint32_t>(plan.N);
    for (size_t index = threadIndex; index < transformSize; index += gridSize) {
        if (plan.b == 16u) {
            output[index] = PackAlignedForwardCoefficientScaled(value,
                                                                plan,
                                                                inputScaleR,
                                                                static_cast<uint32_t>(index),
                                                                inputBitOffset,
                                                                coefficientShift,
                                                                residualBitShift);
        } else {
            output[index] = PackAlignedForwardCoefficient<SharkFloatParams>(debugCombo,
                                                                            value,
                                                                            plan,
                                                                            static_cast<uint32_t>(index),
                                                                            inputBitOffset,
                                                                            coefficientShift,
                                                                            residualBitShift);
        }
    }
}

static __device__ __noinline__ void
PointwiseZeroOne(uint64_t *output, uint32_t transformSize)
{
    const size_t threadIndex = NTT::CohortThreadIndex();
    const size_t gridSize = NTT::CohortGridSize();
    for (size_t index = threadIndex; index < transformSize; index += gridSize)
        output[index] = 0ull;
}

static __device__ __noinline__ void
PointwiseZeroTwo(uint64_t *outputA, uint64_t *outputB, uint32_t transformSize)
{
    const size_t threadIndex = NTT::CohortThreadIndex();
    const size_t gridSize = NTT::CohortGridSize();
    for (size_t index = threadIndex; index < transformSize; index += gridSize) {
        outputA[index] = 0ull;
        outputB[index] = 0ull;
    }
}

template <class SharkFloatParams>
static __device__ __noinline__ void
PointwiseOrbitRealOne(DebugGlobalCount<SharkFloatParams> *debugCombo,
                      const uint64_t *zReal,
                      const uint64_t *zImag,
                      uint64_t *output,
                      uint32_t transformSize)
{
    const size_t threadIndex = NTT::CohortThreadIndex();
    const size_t gridSize = NTT::CohortGridSize();
    for (size_t index = threadIndex; index < transformSize; index += gridSize) {
        const uint64_t real = zReal[index];
        const uint64_t imag = zImag[index];
        const uint64_t sum = NTT::AddP(real, imag);
        const uint64_t difference = NTT::SubP(real, imag);
        output[index] = NTT::MontgomeryMul(debugCombo, sum, difference);
    }
}

template <class SharkFloatParams>
static __device__ __noinline__ void
PointwiseOrbitImagOne(DebugGlobalCount<SharkFloatParams> *debugCombo,
                      const uint64_t *zReal,
                      const uint64_t *zImag,
                      uint64_t *output,
                      uint32_t transformSize)
{
    const size_t threadIndex = NTT::CohortThreadIndex();
    const size_t gridSize = NTT::CohortGridSize();
    for (size_t index = threadIndex; index < transformSize; index += gridSize) {
        const uint64_t product = NTT::MontgomeryMul(debugCombo, zReal[index], zImag[index]);
        output[index] = NTT::AddP(product, product);
    }
}

template <class SharkFloatParams>
static __device__ SharkForceInlineReleaseOnly void
PointwiseOrbitPairTwo(DebugGlobalCount<SharkFloatParams> *debugCombo,
                      const uint64_t *zReal,
                      const uint64_t *zImag,
                      uint64_t *realOutput,
                      uint64_t *imagOutput,
                      uint32_t transformSize)
{
    const size_t threadIndex = NTT::CohortThreadIndex();
    const size_t gridSize = NTT::CohortGridSize();
    for (size_t index = threadIndex; index < transformSize; index += gridSize) {
        const uint64_t real = zReal[index];
        const uint64_t imag = zImag[index];
        const uint64_t sum = NTT::AddP(real, imag);
        const uint64_t difference = NTT::SubP(real, imag);
        realOutput[index] = NTT::MontgomeryMul(debugCombo, sum, difference);
        const uint64_t product = NTT::MontgomeryMul(debugCombo, real, imag);
        imagOutput[index] = NTT::AddP(product, product);
    }
}

template <class SharkFloatParams>
static __device__ SharkForceInlineReleaseOnly void
PointwiseDerivativePairTwo(DebugGlobalCount<SharkFloatParams> *debugCombo,
                           const uint64_t *zReal,
                           const uint64_t *zImag,
                           const uint64_t *dzdcReal,
                           const uint64_t *dzdcImag,
                           uint64_t *realOutput,
                           uint64_t *imagOutput,
                           uint32_t transformSize)
{
    const size_t threadIndex = NTT::CohortThreadIndex();
    const size_t gridSize = NTT::CohortGridSize();
    for (size_t index = threadIndex; index < transformSize; index += gridSize) {
        const uint64_t real = zReal[index];
        const uint64_t imag = zImag[index];
        const uint64_t derivativeReal = dzdcReal[index];
        const uint64_t derivativeImag = dzdcImag[index];
        const uint64_t p1 = NTT::MontgomeryMul(debugCombo, real, derivativeReal);
        const uint64_t p2 = NTT::MontgomeryMul(debugCombo, imag, derivativeImag);
        const uint64_t stateSum = NTT::AddP(real, imag);
        const uint64_t derivativeSum = NTT::AddP(derivativeReal, derivativeImag);
        const uint64_t p3 = NTT::MontgomeryMul(debugCombo, stateSum, derivativeSum);
        const uint64_t realDifference = NTT::SubP(p1, p2);
        const uint64_t imagDifference = NTT::SubP(NTT::SubP(p3, p1), p2);
        realOutput[index] = NTT::AddP(realDifference, realDifference);
        imagOutput[index] = NTT::AddP(imagDifference, imagDifference);
    }
}
template <class IntT>
__device__ uint32_t
FunnelShiftRight(const IntT *data, int index, int count, int bitOffset)
{
    const int wordOffset = bitOffset / 32;
    const int bit = bitOffset % 32;
    const uint32_t low =
        (index + wordOffset >= count) ? 0u : static_cast<uint32_t>(data[index + wordOffset]);
    if (bit == 0)
        return low;
    const uint32_t high =
        (index + wordOffset + 1 >= count) ? 0u : static_cast<uint32_t>(data[index + wordOffset + 1]);
    return (low >> bit) | (high << (32 - bit));
}

template <class IntT>
__device__ uint32_t
FunnelShiftLeft(const IntT *data, int index, int count, int bitOffset)
{
    const int wordOffset = bitOffset / 32;
    const int bit = bitOffset % 32;
    const uint32_t low = (index - wordOffset < 0) ? 0u : static_cast<uint32_t>(data[index - wordOffset]);
    if (bit == 0)
        return low;
    const uint32_t high =
        (index - wordOffset - 1 < 0) ? 0u : static_cast<uint32_t>(data[index - wordOffset - 1]);
    return (low << bit) | (high >> (32 - bit));
}

static __device__ SharkForceInlineReleaseOnly int64_t
SignedResidueContribution(uint64_t residue,
                          uint64_t coefficientIndex,
                          uint32_t limbIndex,
                          uint32_t bitsPerCoefficient,
                          uint64_t halfPrime,
                          uint64_t outputBitOffset)
{
    if (residue == 0)
        return 0;

    const bool negative = residue > halfPrime;
    const uint64_t magnitude = negative ? SharkNTT::MagicPrime - residue : residue;
    const uint64_t shiftedBits =
        outputBitOffset + coefficientIndex * static_cast<uint64_t>(bitsPerCoefficient);
    const uint32_t q = static_cast<uint32_t>(shiftedBits >> 5);
    if (q > limbIndex || limbIndex - q > 3)
        return 0;

    const int r = static_cast<int>(shiftedBits & 31);
    const uint64_t lo = r == 0 ? magnitude : magnitude << r;
    const uint64_t hi = r == 0 ? 0ull : magnitude >> (64 - r);
    uint32_t contribution = 0;
    switch (limbIndex - q) {
        case 0:
            contribution = static_cast<uint32_t>(lo);
            break;
        case 1:
            contribution = static_cast<uint32_t>(lo >> 32);
            break;
        case 2:
            contribution = static_cast<uint32_t>(hi);
            break;
        case 3:
            contribution = static_cast<uint32_t>(hi >> 32);
            break;
    }
    return negative ? -static_cast<int64_t>(contribution) : static_cast<int64_t>(contribution);
}

template <class SharkFloatParams>
static __device__ SharkForceInlineReleaseOnly int64_t
SignedLinearLimbContributionWithLinear(const HpSharkFloat<SharkFloatParams> *value,
                                       uint32_t inputBitOffset,
                                       uint64_t outputBitOffset,
                                       uint32_t limbIndex)
{
    const uint64_t limbBit = static_cast<uint64_t>(limbIndex) * 32ull;
    if (limbBit + 32ull <= outputBitOffset)
        return 0;

    uint32_t contribution = 0u;
    if (limbBit < outputBitOffset) {
        const uint32_t gap = static_cast<uint32_t>(outputBitOffset - limbBit);
        if (gap < 32u) {
            const uint32_t bitCount = 32u - gap;
            const uint64_t source =
                ReadBitsSimple(*value, static_cast<int64_t>(inputBitOffset), static_cast<int>(bitCount));
            contribution = static_cast<uint32_t>(source << gap);
        }
    } else {
        const uint64_t sourceBit = limbBit - outputBitOffset;
        const uint64_t source =
            ReadBitsSimple(*value, static_cast<int64_t>(inputBitOffset + sourceBit), 32);
        contribution = static_cast<uint32_t>(source);
    }

    const int64_t signedContribution = static_cast<int64_t>(contribution);
    return value->GetNegative() ? -signedContribution : signedContribution;
}

static __device__ SharkForceInlineReleaseOnly int64_t
UnpackAlignedResidueLimbContributionNoLinear(const uint64_t *spectrum,
                                             const SharkNTT::Plan &plan,
                                             uint32_t coefficientCount,
                                             uint64_t productBitOffset,
                                             uint32_t limbIndex)
{
    const uint64_t halfPrime = (SharkNTT::MagicPrime - 1ull) >> 1;
    const uint64_t firstBit = limbIndex >= 3u ? static_cast<uint64_t>(limbIndex - 3u) * 32ull : 0ull;
    const uint64_t lastBit = (static_cast<uint64_t>(limbIndex) + 1ull) * 32ull - 1ull;
    const uint64_t firstCoefficient = firstBit > productBitOffset
                                          ? (firstBit - productBitOffset) / static_cast<uint64_t>(plan.b)
                                          : 0ull;
    const uint64_t lastCoefficient = lastBit >= productBitOffset
                                         ? (lastBit - productBitOffset) / static_cast<uint64_t>(plan.b)
                                         : 0ull;
    int64_t total = 0;
    if (firstBit >= productBitOffset || productBitOffset <= lastBit) {
        for (uint64_t i = firstCoefficient; i <= lastCoefficient && i < coefficientCount; ++i) {
            uint64_t residue = spectrum[i];
            total += SignedResidueContribution(
                residue, i, limbIndex, static_cast<uint32_t>(plan.b), halfPrime, productBitOffset);
        }
    }
    return total;
}

template <class SharkFloatParams>
static __device__ SharkForceInlineReleaseOnly int64_t
UnpackAlignedResidueLimbContributionWithLinear(const uint64_t *spectrum,
                                               const SharkNTT::Plan &plan,
                                               uint32_t coefficientCount,
                                               uint64_t productBitOffset,
                                               const HpSharkFloat<SharkFloatParams> *linearValue,
                                               uint32_t linearInputBitOffset,
                                               uint64_t linearBitOffset,
                                               uint32_t limbIndex)
{
    return UnpackAlignedResidueLimbContributionNoLinear(
               spectrum, plan, coefficientCount, productBitOffset, limbIndex) +
           SignedLinearLimbContributionWithLinear(
               linearValue, linearInputBitOffset, linearBitOffset, limbIndex);
}

template <class SharkFloatParams>
static __device__ SharkForceInlineReleaseOnly void
GatherLinearToSignedLimbsOne(const HpSharkFloat<SharkFloatParams> &linearValue,
                             uint32_t linearInputBitOffset,
                             int64_t *limbs,
                             uint32_t limbCount)
{
    const uint32_t rank = GridThreadRank();
    const uint32_t gridSize = gridDim.x * blockDim.x;
    for (uint32_t j = rank; j < limbCount; j += gridSize)
        limbs[j] = SignedLinearLimbContributionWithLinear(&linearValue, linearInputBitOffset, 0u, j);
}

static __device__ SharkForceInlineReleaseOnly void
ZeroSignedLimbsOne(int64_t *limbs, uint32_t limbCount)
{
    const uint32_t rank = GridThreadRank();
    const uint32_t gridSize = gridDim.x * blockDim.x;
    for (uint32_t j = rank; j < limbCount; j += gridSize)
        limbs[j] = 0;
}

template <class SharkFloatParams>
static __device__ SharkForceInlineReleaseOnly void
UnpackAlignedResiduesToSignedLimbsOneNoLinear(const uint64_t *spectrum,
                                              const SharkNTT::Plan &plan,
                                              uint32_t coefficientCount,
                                              uint64_t productBitOffset,
                                              int64_t *limbs,
                                              uint32_t limbCount)
{
    MattsCudaAssert(plan.b != 16u);
    for (uint32_t limbIndex = GridThreadRank(); limbIndex < limbCount;
         limbIndex += gridDim.x * blockDim.x) {
        limbs[limbIndex] = UnpackAlignedResidueLimbContributionNoLinear(
            spectrum, plan, coefficientCount, productBitOffset, limbIndex);
    }
}

template <class SharkFloatParams>
static __device__ SharkForceInlineReleaseOnly void
UnpackAlignedResiduesToSignedLimbsOneWithLinear(const uint64_t *spectrum,
                                                const SharkNTT::Plan &plan,
                                                uint32_t coefficientCount,
                                                uint64_t productBitOffset,
                                                const HpSharkFloat<SharkFloatParams> *linearValue,
                                                uint32_t linearInputBitOffset,
                                                uint64_t linearBitOffset,
                                                int64_t *limbs,
                                                uint32_t limbCount)
{
    MattsCudaAssert(plan.b != 16u);
    for (uint32_t limbIndex = GridThreadRank(); limbIndex < limbCount;
         limbIndex += gridDim.x * blockDim.x) {
        limbs[limbIndex] =
            UnpackAlignedResidueLimbContributionWithLinear<SharkFloatParams>(spectrum,
                                                                             plan,
                                                                             coefficientCount,
                                                                             productBitOffset,
                                                                             linearValue,
                                                                             linearInputBitOffset,
                                                                             linearBitOffset,
                                                                             limbIndex);
    }
}
static __device__ SharkForceInlineReleaseOnly uint64_t
ShuffleUint64(unsigned mask, uint64_t value, int sourceLane)
{
    const uint32_t low = __shfl_sync(mask, static_cast<uint32_t>(value), sourceLane);
    const uint32_t high = __shfl_sync(mask, static_cast<uint32_t>(value >> 32), sourceLane);
    return (static_cast<uint64_t>(high) << 32) | low;
}

static __device__ SharkForceInlineReleaseOnly uint64_t
ShuffleUpUint64(unsigned mask, uint64_t value, unsigned delta)
{
    const uint32_t low = __shfl_up_sync(mask, static_cast<uint32_t>(value), delta);
    const uint32_t high = __shfl_up_sync(mask, static_cast<uint32_t>(value >> 32), delta);
    return (static_cast<uint64_t>(high) << 32) | low;
}

static __device__ SharkForceInlineReleaseOnly void
SignedB16ShiftPieces(uint64_t residue,
                     uint32_t shift,
                     uint64_t halfPrime,
                     int64_t &piece0,
                     int64_t &piece1,
                     int64_t &piece2)
{
    if (residue == 0ull) {
        piece0 = 0;
        piece1 = 0;
        piece2 = 0;
        return;
    }

    const bool negative = residue > halfPrime;
    const uint64_t magnitude = negative ? SharkNTT::MagicPrime - residue : residue;
    const uint32_t magnitudeHigh = static_cast<uint32_t>(magnitude >> 32u);
    const uint32_t value0 = static_cast<uint32_t>(magnitude << shift);
    const uint32_t value1 =
        shift == 0u ? magnitudeHigh : static_cast<uint32_t>(magnitude >> (32u - shift));
    const uint32_t value2 = shift == 0u ? 0u : static_cast<uint32_t>(magnitude >> (64u - shift));
    piece0 = negative ? -static_cast<int64_t>(value0) : static_cast<int64_t>(value0);
    piece1 = negative ? -static_cast<int64_t>(value1) : static_cast<int64_t>(value1);
    piece2 = negative ? -static_cast<int64_t>(value2) : static_cast<int64_t>(value2);
}

static __device__ SharkForceInlineReleaseOnly uint64_t
NormalizeB16ResidueForTileStandard(const uint64_t *spectrum,
                                   uint32_t coefficientCount,
                                   uint64_t firstCoefficient,
                                   uint64_t lastCoefficient,
                                   int64_t coefficientIndex)
{
    if (coefficientIndex < 0)
        return 0;
    const uint64_t index = static_cast<uint64_t>(coefficientIndex);
    if (index < firstCoefficient || index > lastCoefficient || index >= coefficientCount)
        return 0;
    return spectrum[index];
}

template <class SharkFloatParams>
static __device__ SharkForceInlineReleaseOnly int64_t
ComputeAlignedB16SignedLimbCore(const uint64_t *spectrum,
                                uint32_t coefficientCount,
                                uint64_t productBitOffset,
                                uint32_t limbBegin,
                                uint32_t tileLimbCount,
                                uint32_t laneIndex,
                                int64_t linearTotal)
{
    constexpr unsigned FullWarpMask = 0xFFFF'FFFFu;
    const uint32_t limbIndex = limbBegin + laneIndex;
    const bool limbIsValid = laneIndex < tileLimbCount;
    const uint32_t limbEnd = limbBegin + tileLimbCount - 1u;
    const int64_t productLimbOffset = static_cast<int64_t>(productBitOffset >> 5u);
    const uint32_t productResidualBitOffset = static_cast<uint32_t>(productBitOffset & 31ull);
    const uint64_t halfPrime = (SharkNTT::MagicPrime - 1ull) >> 1;

    const uint64_t firstBit = limbBegin >= 3u ? static_cast<uint64_t>(limbBegin - 3u) * 32ull : 0ull;
    const uint64_t lastBit = (static_cast<uint64_t>(limbEnd) + 1ull) * 32ull - 1ull;
    const bool productOverlapsTile = productBitOffset <= lastBit;
    uint64_t firstCoefficient = 0;
    uint64_t lastCoefficient = 0;
    if (productOverlapsTile) {
        firstCoefficient =
            firstBit > productBitOffset ? (firstBit - productBitOffset + 15ull) / 16ull : 0ull;
        lastCoefficient = (lastBit - productBitOffset) / 16ull;
        if (coefficientCount != 0u && lastCoefficient >= coefficientCount)
            lastCoefficient = static_cast<uint64_t>(coefficientCount - 1u);
    }
    const bool hasProduct = productOverlapsTile && coefficientCount != 0u &&
                            firstCoefficient <= lastCoefficient && firstCoefficient < coefficientCount;

    int64_t total = linearTotal;
    if (!hasProduct)
        return total;

    const int64_t pairIndex = static_cast<int64_t>(limbIndex) - productLimbOffset;
    const int64_t evenCoefficientIndex = pairIndex * 2ll;
    const int64_t oddCoefficientIndex = evenCoefficientIndex + 1ll;
    const uint64_t evenResidue = NormalizeB16ResidueForTileStandard(
        spectrum, coefficientCount, firstCoefficient, lastCoefficient, evenCoefficientIndex);
    const uint64_t oddResidue = NormalizeB16ResidueForTileStandard(
        spectrum, coefficientCount, firstCoefficient, lastCoefficient, oddCoefficientIndex);

    uint64_t haloEven1Residue = 0;
    uint64_t haloEven2Residue = 0;
    uint64_t haloOdd1Residue = 0;
    uint64_t haloOdd2Residue = 0;
    uint64_t haloOdd3Residue = 0;
    if (laneIndex == 0u) {
        const int64_t tilePairIndex = static_cast<int64_t>(limbBegin) - productLimbOffset;
        haloEven1Residue = NormalizeB16ResidueForTileStandard(
            spectrum, coefficientCount, firstCoefficient, lastCoefficient, (tilePairIndex - 1ll) * 2ll);
        haloEven2Residue = NormalizeB16ResidueForTileStandard(
            spectrum, coefficientCount, firstCoefficient, lastCoefficient, (tilePairIndex - 2ll) * 2ll);
        haloOdd1Residue = NormalizeB16ResidueForTileStandard(spectrum,
                                                             coefficientCount,
                                                             firstCoefficient,
                                                             lastCoefficient,
                                                             (tilePairIndex - 1ll) * 2ll + 1ll);
        haloOdd2Residue = NormalizeB16ResidueForTileStandard(spectrum,
                                                             coefficientCount,
                                                             firstCoefficient,
                                                             lastCoefficient,
                                                             (tilePairIndex - 2ll) * 2ll + 1ll);
        if (productResidualBitOffset >= 16u) {
            haloOdd3Residue = NormalizeB16ResidueForTileStandard(spectrum,
                                                                 coefficientCount,
                                                                 firstCoefficient,
                                                                 lastCoefficient,
                                                                 (tilePairIndex - 3ll) * 2ll + 1ll);
        }
    }

    const uint32_t oddShift =
        productResidualBitOffset < 16u ? productResidualBitOffset + 16u : productResidualBitOffset - 16u;
    int64_t evenPiece0 = 0;
    int64_t evenPiece1 = 0;
    int64_t evenPiece2 = 0;
    int64_t oddPiece0 = 0;
    int64_t oddPiece1 = 0;
    int64_t oddPiece2 = 0;
    SignedB16ShiftPieces(
        evenResidue, productResidualBitOffset, halfPrime, evenPiece0, evenPiece1, evenPiece2);
    SignedB16ShiftPieces(oddResidue, oddShift, halfPrime, oddPiece0, oddPiece1, oddPiece2);

    int64_t haloEven1Piece0 = 0;
    int64_t haloEven1Piece1 = 0;
    int64_t haloEven1Piece2 = 0;
    int64_t haloEven2Piece0 = 0;
    int64_t haloEven2Piece1 = 0;
    int64_t haloEven2Piece2 = 0;
    int64_t haloOdd1Piece0 = 0;
    int64_t haloOdd1Piece1 = 0;
    int64_t haloOdd1Piece2 = 0;
    int64_t haloOdd2Piece0 = 0;
    int64_t haloOdd2Piece1 = 0;
    int64_t haloOdd2Piece2 = 0;
    int64_t haloOdd3Piece0 = 0;
    int64_t haloOdd3Piece1 = 0;
    int64_t haloOdd3Piece2 = 0;
    if (laneIndex == 0u) {
        SignedB16ShiftPieces(haloEven1Residue,
                             productResidualBitOffset,
                             halfPrime,
                             haloEven1Piece0,
                             haloEven1Piece1,
                             haloEven1Piece2);
        SignedB16ShiftPieces(haloEven2Residue,
                             productResidualBitOffset,
                             halfPrime,
                             haloEven2Piece0,
                             haloEven2Piece1,
                             haloEven2Piece2);
        SignedB16ShiftPieces(
            haloOdd1Residue, oddShift, halfPrime, haloOdd1Piece0, haloOdd1Piece1, haloOdd1Piece2);
        SignedB16ShiftPieces(
            haloOdd2Residue, oddShift, halfPrime, haloOdd2Piece0, haloOdd2Piece1, haloOdd2Piece2);
        if (productResidualBitOffset >= 16u) {
            SignedB16ShiftPieces(
                haloOdd3Residue, oddShift, halfPrime, haloOdd3Piece0, haloOdd3Piece1, haloOdd3Piece2);
        }
    }

    haloEven1Piece1 =
        static_cast<int64_t>(ShuffleUint64(FullWarpMask, static_cast<uint64_t>(haloEven1Piece1), 0));
    haloEven2Piece2 =
        static_cast<int64_t>(ShuffleUint64(FullWarpMask, static_cast<uint64_t>(haloEven2Piece2), 0));
    haloEven1Piece2 =
        static_cast<int64_t>(ShuffleUint64(FullWarpMask, static_cast<uint64_t>(haloEven1Piece2), 0));
    if (productResidualBitOffset < 16u) {
        haloOdd1Piece1 =
            static_cast<int64_t>(ShuffleUint64(FullWarpMask, static_cast<uint64_t>(haloOdd1Piece1), 0));
        haloOdd2Piece2 =
            static_cast<int64_t>(ShuffleUint64(FullWarpMask, static_cast<uint64_t>(haloOdd2Piece2), 0));
        haloOdd1Piece2 =
            static_cast<int64_t>(ShuffleUint64(FullWarpMask, static_cast<uint64_t>(haloOdd1Piece2), 0));
    } else {
        haloOdd1Piece0 =
            static_cast<int64_t>(ShuffleUint64(FullWarpMask, static_cast<uint64_t>(haloOdd1Piece0), 0));
        haloOdd2Piece1 =
            static_cast<int64_t>(ShuffleUint64(FullWarpMask, static_cast<uint64_t>(haloOdd2Piece1), 0));
        haloOdd3Piece2 =
            static_cast<int64_t>(ShuffleUint64(FullWarpMask, static_cast<uint64_t>(haloOdd3Piece2), 0));
        haloOdd1Piece1 =
            static_cast<int64_t>(ShuffleUint64(FullWarpMask, static_cast<uint64_t>(haloOdd1Piece1), 0));
        haloOdd2Piece2 =
            static_cast<int64_t>(ShuffleUint64(FullWarpMask, static_cast<uint64_t>(haloOdd2Piece2), 0));
        haloOdd1Piece2 =
            static_cast<int64_t>(ShuffleUint64(FullWarpMask, static_cast<uint64_t>(haloOdd1Piece2), 0));
    }

    int64_t evenPrevious =
        static_cast<int64_t>(ShuffleUpUint64(FullWarpMask, static_cast<uint64_t>(evenPiece1), 1u));
    int64_t evenTwoBack =
        static_cast<int64_t>(ShuffleUpUint64(FullWarpMask, static_cast<uint64_t>(evenPiece2), 2u));
    if (laneIndex == 0u) {
        evenPrevious = haloEven1Piece1;
        evenTwoBack = haloEven2Piece2;
    } else if (laneIndex == 1u) {
        evenTwoBack = haloEven1Piece2;
    }

    total += evenPiece0 + evenPrevious + evenTwoBack;

    if (productResidualBitOffset < 16u) {
        int64_t oddPrevious =
            static_cast<int64_t>(ShuffleUpUint64(FullWarpMask, static_cast<uint64_t>(oddPiece1), 1u));
        int64_t oddTwoBack =
            static_cast<int64_t>(ShuffleUpUint64(FullWarpMask, static_cast<uint64_t>(oddPiece2), 2u));
        if (laneIndex == 0u) {
            oddPrevious = haloOdd1Piece1;
            oddTwoBack = haloOdd2Piece2;
        } else if (laneIndex == 1u) {
            oddTwoBack = haloOdd1Piece2;
        }
        total += oddPiece0 + oddPrevious + oddTwoBack;
    } else {
        int64_t oddPrevious =
            static_cast<int64_t>(ShuffleUpUint64(FullWarpMask, static_cast<uint64_t>(oddPiece0), 1u));
        int64_t oddTwoBack =
            static_cast<int64_t>(ShuffleUpUint64(FullWarpMask, static_cast<uint64_t>(oddPiece1), 2u));
        int64_t oddThreeBack =
            static_cast<int64_t>(ShuffleUpUint64(FullWarpMask, static_cast<uint64_t>(oddPiece2), 3u));
        if (laneIndex == 0u) {
            oddPrevious = haloOdd1Piece0;
            oddTwoBack = haloOdd2Piece1;
            oddThreeBack = haloOdd3Piece2;
        } else if (laneIndex == 1u) {
            oddTwoBack = haloOdd1Piece1;
            oddThreeBack = haloOdd2Piece2;
        } else if (laneIndex == 2u) {
            oddThreeBack = haloOdd1Piece2;
        }
        total += oddPrevious + oddTwoBack + oddThreeBack;
    }

    return limbIsValid ? total : 0;
}

template <class SharkFloatParams>
static __device__ SharkForceInlineReleaseOnly int64_t
PrepareAlignedB16SignedLimbNoLinear(const uint64_t *spectrum,
                                    uint32_t coefficientCount,
                                    uint64_t productBitOffset,
                                    uint32_t limbBegin,
                                    uint32_t tileLimbCount)
{
    const uint32_t laneIndex = threadIdx.x & 31u;
    return ComputeAlignedB16SignedLimbCore<SharkFloatParams>(
        spectrum, coefficientCount, productBitOffset, limbBegin, tileLimbCount, laneIndex, 0);
}

template <class SharkFloatParams>
static __device__ SharkForceInlineReleaseOnly int64_t
PrepareAlignedB16SignedLimbWithLinear(const uint64_t *spectrum,
                                      uint32_t coefficientCount,
                                      uint64_t productBitOffset,
                                      const HpSharkFloat<SharkFloatParams> *linearValue,
                                      uint32_t linearInputBitOffset,
                                      uint64_t linearBitOffset,
                                      uint32_t limbBegin,
                                      uint32_t tileLimbCount)
{
    const uint32_t laneIndex = threadIdx.x & 31u;
    const uint32_t limbIndex = limbBegin + laneIndex;
    const int64_t linearTotal = laneIndex < tileLimbCount
                                    ? SignedLinearLimbContributionWithLinear(
                                          linearValue, linearInputBitOffset, linearBitOffset, limbIndex)
                                    : 0;
    return ComputeAlignedB16SignedLimbCore<SharkFloatParams>(
        spectrum, coefficientCount, productBitOffset, limbBegin, tileLimbCount, laneIndex, linearTotal);
}

constexpr uint32_t FinalizationDigitLengthControl = 0;
constexpr uint32_t FinalizationNegativeControl = 1;
constexpr uint32_t FinalizationLowestNonZeroControl = 2;
constexpr uint32_t FinalizationHighestNonZeroControl = 3;
// Shared alias for the general non-B16 finalization helpers below.
constexpr uint32_t FinalizationNonZeroReductionControl = FinalizationLowestNonZeroControl;

constexpr int32_t CarryPrefixMin = -8;
constexpr int32_t CarryPrefixMax = 7;
constexpr uint32_t CarryPrefixStateCount = CarryPrefixMax - CarryPrefixMin + 1;
constexpr uint32_t CarryPrefixMaxWarps = 32;
constexpr uint32_t CarryPrefixWarpAggregatesOffset = 0u;
constexpr uint32_t CarryPrefixWarpPrefixesOffset = CarryPrefixWarpAggregatesOffset + CarryPrefixMaxWarps;
constexpr uint32_t CarryPrefixLookbackTransformsOffset =
    CarryPrefixWarpPrefixesOffset + CarryPrefixMaxWarps;
constexpr uint32_t CarryPrefixLookbackStatesOffset =
    CarryPrefixLookbackTransformsOffset + CarryPrefixMaxWarps;
constexpr uint32_t CarryPrefixControlSlot = 0u;

static __device__ void
InitializeLowestNonZeroStream(cooperative_groups::thread_block &block,
                              uint32_t *control,
                              uint32_t *blockMinimum,
                              uint32_t minimumSlot)
{
    const uint32_t count = control[FinalizationDigitLengthControl];
    const bool enabled = control[FinalizationNegativeControl] != 0u;
    if (IsLeader(block))
        control[FinalizationNonZeroReductionControl] = enabled ? count : 0u;
    if (block.thread_index().x == 0u)
        blockMinimum[minimumSlot] = enabled ? count : 0u;
}

static __device__ void
AccumulateLowestNonZeroStream(cooperative_groups::thread_block &block,
                              uint32_t gridSize,
                              uint32_t maximumCount,
                              uint32_t *digits,
                              uint32_t *control,
                              uint32_t *blockMinimum,
                              uint32_t minimumSlot)
{
    const uint32_t count = control[FinalizationDigitLengthControl];
    const bool enabled = control[FinalizationNegativeControl] != 0u;
    uint32_t localMinimum = enabled ? count : 0u;
    for (uint32_t index = GridThreadRank(block); index < maximumCount; index += gridSize) {
        if (enabled && index < count && digits[index] != 0u)
            localMinimum = localMinimum < index ? localMinimum : index;
    }
    if (enabled && localMinimum != count)
        atomicMin(&blockMinimum[minimumSlot], localMinimum);
}

static __device__ void
PublishLowestNonZeroStream(cooperative_groups::thread_block &block,
                           uint32_t *control,
                           uint32_t *blockMinimum,
                           uint32_t minimumSlot)
{
    if (block.thread_index().x == 0u) {
        const uint32_t count = control[FinalizationDigitLengthControl];
        const bool enabled = control[FinalizationNegativeControl] != 0u;
        if (enabled && blockMinimum[minimumSlot] != count)
            atomicMin(&control[FinalizationNonZeroReductionControl], blockMinimum[minimumSlot]);
    }
}

static __device__ void
VerifyLowestNonZeroStream(cooperative_groups::thread_block &block,
                          uint32_t gridSize,
                          uint32_t maximumCount,
                          uint32_t *digits,
                          uint32_t *control)
{
    const uint32_t count = control[FinalizationDigitLengthControl];
    const bool enabled = control[FinalizationNegativeControl] != 0u;
    const uint32_t lowest = control[FinalizationNonZeroReductionControl];
    for (uint32_t index = GridThreadRank(block); index < maximumCount; index += gridSize) {
        if (enabled && index < count) {
            if (index < lowest)
                MattsCudaAssert(digits[index] == 0u);
            if (index == lowest)
                MattsCudaAssert(digits[index] != 0u);
        }
    }
}

template <class SharkFloatParams>
__device__ void
FindLowestNonNr(uint32_t *realDigits,
                uint32_t *realControl,
                uint32_t *imagDigits,
                uint32_t *imagControl,
                uint64_t *sharedStorage)
{
    cooperative_groups::grid_group grid = cooperative_groups::this_grid();
    cooperative_groups::thread_block block = cooperative_groups::this_thread_block();
    const uint32_t gridSize = static_cast<uint32_t>(grid.size());
    uint32_t *blockMinimum = reinterpret_cast<uint32_t *>(sharedStorage);
    const uint32_t realCount = realControl[FinalizationDigitLengthControl];
    const uint32_t imagCount = imagControl[FinalizationDigitLengthControl];
    const bool realEnabled = realControl[FinalizationNegativeControl] != 0u;
    const bool imagEnabled = imagControl[FinalizationNegativeControl] != 0u;
    InitializeLowestNonZeroStream(block, realControl, blockMinimum, 0u);
    InitializeLowestNonZeroStream(block, imagControl, blockMinimum, 1u);
    grid.sync();

    uint32_t maximumCount = realEnabled ? realCount : 0u;
    maximumCount = imagEnabled && imagCount > maximumCount ? imagCount : maximumCount;
    AccumulateLowestNonZeroStream(
        block, gridSize, maximumCount, realDigits, realControl, blockMinimum, 0u);
    AccumulateLowestNonZeroStream(
        block, gridSize, maximumCount, imagDigits, imagControl, blockMinimum, 1u);
    __syncthreads();

    PublishLowestNonZeroStream(block, realControl, blockMinimum, 0u);
    PublishLowestNonZeroStream(block, imagControl, blockMinimum, 1u);
    grid.sync();

    if constexpr (HpShark::Debug) {
        VerifyLowestNonZeroStream(block, gridSize, maximumCount, realDigits, realControl);
        VerifyLowestNonZeroStream(block, gridSize, maximumCount, imagDigits, imagControl);
        grid.sync();
    }
}

template <class SharkFloatParams>
__device__ void
FindLowestNr(uint32_t *realDigits,
             uint32_t *realControl,
             uint32_t *imagDigits,
             uint32_t *imagControl,
             uint32_t *dzdcRealDigits,
             uint32_t *dzdcRealControl,
             uint32_t *dzdcImagDigits,
             uint32_t *dzdcImagControl,
             uint64_t *sharedStorage)
{
    cooperative_groups::grid_group grid = cooperative_groups::this_grid();
    cooperative_groups::thread_block block = cooperative_groups::this_thread_block();
    const uint32_t gridSize = static_cast<uint32_t>(grid.size());
    uint32_t *blockMinimum = reinterpret_cast<uint32_t *>(sharedStorage);
    const uint32_t realCount = realControl[FinalizationDigitLengthControl];
    const uint32_t imagCount = imagControl[FinalizationDigitLengthControl];
    const uint32_t dzdcRealCount = dzdcRealControl[FinalizationDigitLengthControl];
    const uint32_t dzdcImagCount = dzdcImagControl[FinalizationDigitLengthControl];
    const bool realEnabled = realControl[FinalizationNegativeControl] != 0u;
    const bool imagEnabled = imagControl[FinalizationNegativeControl] != 0u;
    const bool dzdcRealEnabled = dzdcRealControl[FinalizationNegativeControl] != 0u;
    const bool dzdcImagEnabled = dzdcImagControl[FinalizationNegativeControl] != 0u;

    InitializeLowestNonZeroStream(block, realControl, blockMinimum, 0u);
    InitializeLowestNonZeroStream(block, imagControl, blockMinimum, 1u);
    InitializeLowestNonZeroStream(block, dzdcRealControl, blockMinimum, 2u);
    InitializeLowestNonZeroStream(block, dzdcImagControl, blockMinimum, 3u);
    grid.sync();

    uint32_t maximumCount = realEnabled ? realCount : 0u;
    maximumCount = imagEnabled && imagCount > maximumCount ? imagCount : maximumCount;
    maximumCount = dzdcRealEnabled && dzdcRealCount > maximumCount ? dzdcRealCount : maximumCount;
    maximumCount = dzdcImagEnabled && dzdcImagCount > maximumCount ? dzdcImagCount : maximumCount;
    AccumulateLowestNonZeroStream(
        block, gridSize, maximumCount, realDigits, realControl, blockMinimum, 0u);
    AccumulateLowestNonZeroStream(
        block, gridSize, maximumCount, imagDigits, imagControl, blockMinimum, 1u);
    AccumulateLowestNonZeroStream(
        block, gridSize, maximumCount, dzdcRealDigits, dzdcRealControl, blockMinimum, 2u);
    AccumulateLowestNonZeroStream(
        block, gridSize, maximumCount, dzdcImagDigits, dzdcImagControl, blockMinimum, 3u);
    __syncthreads();

    PublishLowestNonZeroStream(block, realControl, blockMinimum, 0u);
    PublishLowestNonZeroStream(block, imagControl, blockMinimum, 1u);
    PublishLowestNonZeroStream(block, dzdcRealControl, blockMinimum, 2u);
    PublishLowestNonZeroStream(block, dzdcImagControl, blockMinimum, 3u);
    grid.sync();

    if constexpr (HpShark::Debug) {
        VerifyLowestNonZeroStream(block, gridSize, maximumCount, realDigits, realControl);
        VerifyLowestNonZeroStream(block, gridSize, maximumCount, imagDigits, imagControl);
        VerifyLowestNonZeroStream(block, gridSize, maximumCount, dzdcRealDigits, dzdcRealControl);
        VerifyLowestNonZeroStream(block, gridSize, maximumCount, dzdcImagDigits, dzdcImagControl);
        grid.sync();
    }
}

enum class CarryPrefixDescriptorState : uint32_t {
    Empty = 0,
    Aggregate = 1,
    Prefix = 2,
};

enum class CarryPrefixLookbackStatus : uint32_t {
    Pending = 0,
    Ready = 1,
    Prefix = 2,
    End = 3,
};

constexpr uint32_t CarryPrefixLookbackStatusMask = 3u;

static __device__ uint32_t
MakeCarryPrefixLookbackToken(uint32_t part, uint32_t batch, uint32_t batchCount)
{
    const uint64_t token = static_cast<uint64_t>(part) * batchCount + batch;
    MattsCudaAssert(token < (1ull << 30u));
    return static_cast<uint32_t>(token);
}

static __device__ uint32_t
PackCarryPrefixLookbackStatus(uint32_t token, CarryPrefixLookbackStatus status)
{
    MattsCudaAssert(token < (1u << 30u));
    return (token << 2u) | static_cast<uint32_t>(status);
}

static __device__ int32_t
CarryOutForSignedLimb(int64_t limb, int32_t carryIn)
{
    constexpr int64_t Base = 1ll << 32;
    const int64_t sum = limb + carryIn;
    const uint32_t digit = static_cast<uint32_t>(static_cast<uint64_t>(sum));
    return static_cast<int32_t>((sum - static_cast<int64_t>(digit)) / Base);
}

static __device__ uint32_t
MakeSignedCarryPrefixByte(int64_t limb)
{
    constexpr int64_t Base = 1ll << 32;
    const int32_t carryAtMin = CarryOutForSignedLimb(limb, CarryPrefixMin);
    const int32_t carryAtMax = CarryOutForSignedLimb(limb, CarryPrefixMax);
    MattsCudaAssert(carryAtMin >= CarryPrefixMin && carryAtMin <= CarryPrefixMax);
    MattsCudaAssert(carryAtMax >= CarryPrefixMin && carryAtMax <= CarryPrefixMax);

    const uint32_t output = static_cast<uint32_t>(carryAtMin - CarryPrefixMin);
    if (carryAtMin == carryAtMax)
        return output;

    MattsCudaAssert(carryAtMax == carryAtMin + 1);
    const int64_t transitionCarry = (static_cast<int64_t>(carryAtMin) + 1) * Base - limb;
    const uint32_t threshold = static_cast<uint32_t>(transitionCarry - CarryPrefixMin);
    MattsCudaAssert(threshold >= 1u && threshold < CarryPrefixStateCount);
    return output | (threshold << 4u);
}

static __device__ uint32_t
ApplyCarryPrefixByte(uint32_t transform, int32_t carry)
{
    MattsCudaAssert(carry >= CarryPrefixMin && carry <= CarryPrefixMax);
    if (transform == 0xFFu)
        return static_cast<uint32_t>(carry - CarryPrefixMin);

    const uint32_t input = static_cast<uint32_t>(carry - CarryPrefixMin);
    const uint32_t base = transform & 0xFu;
    const uint32_t threshold = transform >> 4u;
    return base + (threshold != 0u && input >= threshold ? 1u : 0u);
}

static __device__ uint32_t
ComposeCarryPrefixBytes(uint32_t earlier, uint32_t later)
{
    if (earlier == 0xFFu)
        return later;
    if (later == 0xFFu)
        return earlier;

    const uint32_t earlierBase = earlier & 0xFu;
    const uint32_t earlierThreshold = earlier >> 4u;
    const uint32_t laterBase = later & 0xFu;
    const uint32_t laterThreshold = later >> 4u;
    if (laterThreshold == 0u)
        return laterBase;
    if (earlierThreshold == 0u)
        return ApplyCarryPrefixByte(later, static_cast<int32_t>(earlierBase) + CarryPrefixMin);

    const uint32_t outputAtBase =
        ApplyCarryPrefixByte(later, static_cast<int32_t>(earlierBase) + CarryPrefixMin);
    const uint32_t outputAfterStep =
        ApplyCarryPrefixByte(later, static_cast<int32_t>(earlierBase + 1u) + CarryPrefixMin);
    return outputAtBase == outputAfterStep ? outputAtBase : outputAtBase | (earlierThreshold << 4u);
}

static __device__ SharkForceInlineReleaseOnly uint32_t
ComposePackedCarryPrefixes(uint32_t earlier, uint32_t later)
{
    uint32_t combined = ComposeCarryPrefixBytes(earlier & 0xFFu, later & 0xFFu);
    combined |= ComposeCarryPrefixBytes((earlier >> 8u) & 0xFFu, (later >> 8u) & 0xFFu) << 8u;
    combined |= ComposeCarryPrefixBytes((earlier >> 16u) & 0xFFu, (later >> 16u) & 0xFFu) << 16u;
    combined |= ComposeCarryPrefixBytes(earlier >> 24u, later >> 24u) << 24u;
    return combined;
}

static __device__ uint32_t
ApplyPackedCarryPrefix(uint32_t transform, int32_t carry)
{
    uint32_t packedStates = ApplyCarryPrefixByte(transform & 0xFFu, carry);
    packedStates |= ApplyCarryPrefixByte((transform >> 8u) & 0xFFu, carry) << 8u;
    packedStates |= ApplyCarryPrefixByte((transform >> 16u) & 0xFFu, carry) << 16u;
    packedStates |= ApplyCarryPrefixByte(transform >> 24u, carry) << 24u;
    return packedStates;
}

static __device__ void
StoreSignedCarryDigit(int64_t signedLimb,
                      int32_t carryIn,
                      uint32_t index,
                      uint32_t limbCount,
                      uint32_t capacity,
                      uint32_t *digits,
                      uint32_t *control)
{
    digits[index] = static_cast<uint32_t>(static_cast<uint64_t>(signedLimb + carryIn));
    if (index + 1u != limbCount)
        return;

    int32_t finalCarry = CarryOutForSignedLimb(signedLimb, carryIn);
    uint32_t digitLength = limbCount;
    while (finalCarry != 0 && finalCarry != -1 && digitLength < capacity) {
        digits[digitLength++] = static_cast<uint32_t>(static_cast<uint64_t>(finalCarry));
        finalCarry = CarryOutForSignedLimb(finalCarry, 0);
    }
    control[FinalizationDigitLengthControl] = digitLength;
    control[FinalizationNegativeControl] = finalCarry < 0 ? 1u : 0u;
}

static __device__ void
PublishCarryPrefixState(uint32_t *state, CarryPrefixDescriptorState value)
{
    cuda::atomic_ref<uint32_t, cuda::thread_scope_device> atomicState(*state);
    atomicState.store(static_cast<uint32_t>(value), cuda::memory_order_release);
}

static __device__ uint32_t
LoadCarryPrefixState(uint32_t *state)
{
    cuda::atomic_ref<uint32_t, cuda::thread_scope_device> atomicState(*state);
    return atomicState.load(cuda::memory_order_acquire);
}

static __device__ uint32_t
LoadCarryPrefixTransform(uint32_t *transform)
{
    cuda::atomic_ref<uint32_t, cuda::thread_scope_device> atomicTransform(*transform);
    return atomicTransform.load(cuda::memory_order_relaxed);
}

static __device__ uint32_t
LoadCarryPrefixLookbackStatus(uint32_t *status)
{
    cuda::atomic_ref<uint32_t, cuda::thread_scope_block> atomicStatus(*status);
    return atomicStatus.load(cuda::memory_order_acquire);
}

static __device__ void
StoreCarryPrefixLookbackStatus(uint32_t *status, uint32_t value)
{
    cuda::atomic_ref<uint32_t, cuda::thread_scope_block> atomicStatus(*status);
    atomicStatus.store(value, cuda::memory_order_release);
}

static __device__ bool
IsCarryPrefixLookbackComplete(uint32_t *control, uint32_t token, uint32_t lane)
{
    if (control == nullptr)
        return false;

    const uint32_t loadedControl = lane == 0u ? LoadCarryPrefixLookbackStatus(control) : 0u;
    const uint32_t controlStatus = __shfl_sync(0xFFFF'FFFFu, loadedControl, 0);
    return controlStatus == PackCarryPrefixLookbackStatus(token, CarryPrefixLookbackStatus::Ready);
}

static __device__ void
StoreCarryPrefixTransform(uint32_t *transform, uint32_t value)
{
    cuda::atomic_ref<uint32_t, cuda::thread_scope_device> atomicTransform(*transform);
    atomicTransform.store(value, cuda::memory_order_relaxed);
}

static __device__ void
PublishCarryPrefixDescriptorAggregate(HpSharkReferencePackedCarryPrefixDescriptor &descriptor,
                                      uint32_t aggregate)
{
    StoreCarryPrefixTransform(&descriptor.AggregateTransform, aggregate);
    PublishCarryPrefixState(&descriptor.State, CarryPrefixDescriptorState::Aggregate);
}

static __device__ void
PublishCarryPrefixDescriptorPrefix(HpSharkReferencePackedCarryPrefixDescriptor &descriptor,
                                   uint32_t prefix)
{
    StoreCarryPrefixTransform(&descriptor.PrefixTransform, prefix);
    PublishCarryPrefixState(&descriptor.State, CarryPrefixDescriptorState::Prefix);
}

static __device__ uint32_t
ResolveCarryPrefixHistory(HpSharkReferencePackedCarryPrefixDescriptor *descriptors,
                          uint32_t part,
                          uint32_t lane)
{
    constexpr uint32_t Identity = 0xFFFF'FFFFu;
    uint32_t exclusive = Identity;
    int32_t previousPart = static_cast<int32_t>(part) - 1;
    int spin = 0;

    while (previousPart >= 0) {
        const int32_t descriptorIndex = previousPart - static_cast<int32_t>(lane);
        const bool validDescriptor = descriptorIndex >= 0;
        CarryPrefixDescriptorState state = CarryPrefixDescriptorState::Empty;
        uint32_t descriptorCount = 0;
        uint32_t validDescriptorCount = 0;
        bool foundPrefix = false;

        do {
            if (validDescriptor && state == CarryPrefixDescriptorState::Empty) {
                state = static_cast<CarryPrefixDescriptorState>(
                    LoadCarryPrefixState(&descriptors[descriptorIndex].State));
            }

            const unsigned validMask = __ballot_sync(0xFFFF'FFFFu, validDescriptor);
            const unsigned readyMask = __ballot_sync(
                0xFFFF'FFFFu, !validDescriptor || state != CarryPrefixDescriptorState::Empty);
            const unsigned unresolvedMask = validMask & ~readyMask;
            validDescriptorCount = static_cast<uint32_t>(__popc(validMask));
            const uint32_t contiguousReadyCount = unresolvedMask == 0u
                                                      ? validDescriptorCount
                                                      : static_cast<uint32_t>(__ffs(unresolvedMask) - 1);
            const unsigned contiguousReadyMask =
                contiguousReadyCount == 32u ? 0xFFFF'FFFFu : ((1u << contiguousReadyCount) - 1u);
            const unsigned prefixMask =
                __ballot_sync(0xFFFF'FFFFu,
                              validDescriptor && state == CarryPrefixDescriptorState::Prefix) &
                contiguousReadyMask;

            if (prefixMask != 0u) {
                descriptorCount = static_cast<uint32_t>(__ffs(prefixMask));
                foundPrefix = true;
            } else if (contiguousReadyCount == validDescriptorCount) {
                descriptorCount = validDescriptorCount;
            }

            if (descriptorCount == 0u) {
                if (++spin == 64) {
                    __nanosleep(64);
                    spin = 0;
                }
            }
        } while (descriptorCount == 0u);

        MattsCudaAssert(descriptorCount <= validDescriptorCount);
        MattsCudaAssert(lane >= descriptorCount || state == CarryPrefixDescriptorState::Aggregate ||
                        state == CarryPrefixDescriptorState::Prefix);
        uint32_t transform = Identity;
        if (lane < descriptorCount) {
            transform = state == CarryPrefixDescriptorState::Prefix
                            ? LoadCarryPrefixTransform(&descriptors[descriptorIndex].PrefixTransform)
                            : LoadCarryPrefixTransform(&descriptors[descriptorIndex].AggregateTransform);
        }

        uint32_t windowTransform = transform;
#pragma unroll
        for (uint32_t offset = 1u; offset < 32u; offset <<= 1u) {
            const uint32_t older = __shfl_down_sync(0xFFFF'FFFFu, windowTransform, offset);
            if (lane + offset < descriptorCount)
                windowTransform = ComposePackedCarryPrefixes(older, windowTransform);
        }

        if (lane == 0u)
            exclusive = ComposePackedCarryPrefixes(windowTransform, exclusive);
        if (foundPrefix)
            break;
        const int32_t nextPreviousPart = previousPart - static_cast<int32_t>(descriptorCount);
        MattsCudaAssert(nextPreviousPart < previousPart);
        previousPart = nextPreviousPart;
    }

    return __shfl_sync(0xFFFF'FFFFu, exclusive, 0);
}

static __device__ uint32_t
ResolveCarryPrefixWindow(HpSharkReferencePackedCarryPrefixDescriptor *descriptors,
                         uint32_t part,
                         uint32_t window,
                         uint32_t lane,
                         uint32_t controlToken,
                         uint32_t *lookbackControl,
                         uint32_t *windowStatus,
                         bool *cancelled)
{
    constexpr uint32_t Identity = 0xFFFF'FFFFu;
    const int32_t windowStart = static_cast<int32_t>(part) - 1 - static_cast<int32_t>(window * 32u);
    *windowStatus = static_cast<uint32_t>(CarryPrefixLookbackStatus::Pending);
    *cancelled = false;

    if (windowStart < 0) {
        if (IsCarryPrefixLookbackComplete(lookbackControl, controlToken, lane)) {
            *cancelled = true;
            return Identity;
        }
        *windowStatus = static_cast<uint32_t>(CarryPrefixLookbackStatus::End);
        return Identity;
    }

    const int32_t descriptorIndex = windowStart - static_cast<int32_t>(lane);
    const bool validDescriptor = descriptorIndex >= 0;
    CarryPrefixDescriptorState state = CarryPrefixDescriptorState::Empty;
    uint32_t descriptorCount = 0u;
    uint32_t validDescriptorCount = 0u;
    bool foundPrefix = false;
    int spin = 0;

    do {
        if (IsCarryPrefixLookbackComplete(lookbackControl, controlToken, lane)) {
            *cancelled = true;
            return Identity;
        }

        if (validDescriptor && state == CarryPrefixDescriptorState::Empty) {
            state = static_cast<CarryPrefixDescriptorState>(
                LoadCarryPrefixState(&descriptors[descriptorIndex].State));
        }

        const unsigned validMask = __ballot_sync(0xFFFF'FFFFu, validDescriptor);
        const unsigned readyMask =
            __ballot_sync(0xFFFF'FFFFu, !validDescriptor || state != CarryPrefixDescriptorState::Empty);
        const unsigned unresolvedMask = validMask & ~readyMask;
        validDescriptorCount = static_cast<uint32_t>(__popc(validMask));
        const uint32_t contiguousReadyCount = unresolvedMask == 0u
                                                  ? validDescriptorCount
                                                  : static_cast<uint32_t>(__ffs(unresolvedMask) - 1);
        const unsigned contiguousReadyMask =
            contiguousReadyCount == 32u ? 0xFFFF'FFFFu : ((1u << contiguousReadyCount) - 1u);
        const unsigned prefixMask =
            __ballot_sync(0xFFFF'FFFFu, validDescriptor && state == CarryPrefixDescriptorState::Prefix) &
            contiguousReadyMask;

        if (prefixMask != 0u) {
            descriptorCount = static_cast<uint32_t>(__ffs(prefixMask));
            foundPrefix = true;
        } else if (contiguousReadyCount == validDescriptorCount) {
            descriptorCount = validDescriptorCount;
        }

        if (descriptorCount == 0u) {
            if (++spin == 64) {
                __nanosleep(64);
                spin = 0;
            }
        }
    } while (descriptorCount == 0u);

    MattsCudaAssert(descriptorCount <= validDescriptorCount);
    MattsCudaAssert(lane >= descriptorCount || state == CarryPrefixDescriptorState::Aggregate ||
                    state == CarryPrefixDescriptorState::Prefix);
    uint32_t transform = Identity;
    if (lane < descriptorCount) {
        transform = state == CarryPrefixDescriptorState::Prefix
                        ? LoadCarryPrefixTransform(&descriptors[descriptorIndex].PrefixTransform)
                        : LoadCarryPrefixTransform(&descriptors[descriptorIndex].AggregateTransform);
    }

    uint32_t windowTransform = transform;
#pragma unroll
    for (uint32_t offset = 1u; offset < 32u; offset <<= 1u) {
        const uint32_t older = __shfl_down_sync(0xFFFF'FFFFu, windowTransform, offset);
        if (lane + offset < descriptorCount)
            windowTransform = ComposePackedCarryPrefixes(older, windowTransform);
    }

    if (foundPrefix)
        *windowStatus = static_cast<uint32_t>(CarryPrefixLookbackStatus::Prefix);
    else if (windowStart < 32)
        *windowStatus = static_cast<uint32_t>(CarryPrefixLookbackStatus::End);
    else
        *windowStatus = static_cast<uint32_t>(CarryPrefixLookbackStatus::Ready);

    return __shfl_sync(0xFFFF'FFFFu, windowTransform, 0);
}

static __device__ uint32_t
ResolveCarryPrefixBlockExclusive(HpSharkReferencePackedCarryPrefixDescriptor *descriptors,
                                 uint32_t part,
                                 uint32_t lane,
                                 uint32_t warp,
                                 uint32_t numWarps,
                                 uint32_t lookbackBatchCount,
                                 uint32_t *packedLookbackTransforms,
                                 uint32_t *packedLookbackStates)
{
    constexpr uint32_t Identity = 0xFFFF'FFFFu;
    const uint32_t initialToken = MakeCarryPrefixLookbackToken(part, 0u, lookbackBatchCount);

    if (warp == 0u && lane == 0u)
        StoreCarryPrefixLookbackStatus(
            &packedLookbackStates[CarryPrefixControlSlot],
            PackCarryPrefixLookbackStatus(initialToken, CarryPrefixLookbackStatus::Pending));

    if (numWarps == 1u) {
        const uint32_t packedBlockExclusive = ResolveCarryPrefixHistory(descriptors, part, lane);
        if (warp == 0u && lane == 0u) {
            packedLookbackTransforms[CarryPrefixControlSlot] = packedBlockExclusive;
            StoreCarryPrefixLookbackStatus(
                &packedLookbackStates[CarryPrefixControlSlot],
                PackCarryPrefixLookbackStatus(initialToken, CarryPrefixLookbackStatus::Ready));
        }
        return packedBlockExclusive;
    }

    if (warp == 0u) {
        // Warp zero consumes the per-warp windows in order and owns the control slot.
        uint32_t batch = 0u;
        uint32_t accumulated = Identity;
        bool done = false;
        while (!done) {
            const uint32_t token = MakeCarryPrefixLookbackToken(part, batch, lookbackBatchCount);
            uint32_t windowStatus = static_cast<uint32_t>(CarryPrefixLookbackStatus::Pending);
            bool cancelled = false;
            const uint32_t windowTransform = ResolveCarryPrefixWindow(
                descriptors, part, batch * numWarps, lane, token, nullptr, &windowStatus, &cancelled);
            MattsCudaAssert(!cancelled);

            uint32_t lane0Done = 0u;
            uint32_t nextBatch = batch;
            if (lane == 0u) {
                uint32_t batchTransform = windowTransform;
                bool batchDone =
                    windowStatus == static_cast<uint32_t>(CarryPrefixLookbackStatus::Prefix) ||
                    windowStatus == static_cast<uint32_t>(CarryPrefixLookbackStatus::End);

                if (!batchDone) {
                    for (uint32_t windowWarp = 1u; windowWarp < numWarps; ++windowWarp) {
                        uint32_t slotStatus = 0u;
                        int spin = 0;
                        do {
                            slotStatus =
                                LoadCarryPrefixLookbackStatus(&packedLookbackStates[windowWarp]);
                            if (slotStatus == PackCarryPrefixLookbackStatus(
                                                  token, CarryPrefixLookbackStatus::Ready) ||
                                slotStatus == PackCarryPrefixLookbackStatus(
                                                  token, CarryPrefixLookbackStatus::Prefix) ||
                                slotStatus ==
                                    PackCarryPrefixLookbackStatus(token, CarryPrefixLookbackStatus::End))
                                break;
                            if (++spin == 64) {
                                __nanosleep(64);
                                spin = 0;
                            }
                        } while (true);

                        const CarryPrefixLookbackStatus status = static_cast<CarryPrefixLookbackStatus>(
                            slotStatus & CarryPrefixLookbackStatusMask);
                        batchTransform = ComposePackedCarryPrefixes(packedLookbackTransforms[windowWarp],
                                                                    batchTransform);
                        if (status == CarryPrefixLookbackStatus::Prefix ||
                            status == CarryPrefixLookbackStatus::End) {
                            batchDone = true;
                            break;
                        }
                    }
                }

                accumulated = ComposePackedCarryPrefixes(batchTransform, accumulated);
                if (batchDone) {
                    packedLookbackTransforms[CarryPrefixControlSlot] = accumulated;
                    StoreCarryPrefixLookbackStatus(
                        &packedLookbackStates[CarryPrefixControlSlot],
                        PackCarryPrefixLookbackStatus(token, CarryPrefixLookbackStatus::Ready));
                    lane0Done = 1u;
                } else {
                    nextBatch = batch + 1u;
                    const uint32_t nextToken =
                        MakeCarryPrefixLookbackToken(part, nextBatch, lookbackBatchCount);
                    StoreCarryPrefixLookbackStatus(
                        &packedLookbackStates[CarryPrefixControlSlot],
                        PackCarryPrefixLookbackStatus(nextToken, CarryPrefixLookbackStatus::Pending));
                }
            }

            done = __shfl_sync(0xFFFF'FFFFu, lane0Done, 0) != 0u;
            if (!done)
                batch = __shfl_sync(0xFFFF'FFFFu, nextBatch, 0);
        }
    } else {
        // Other warps resolve one window, then follow the coordinator command.
        for (uint32_t batch = 0u; batch < lookbackBatchCount; ++batch) {
            const uint32_t token = MakeCarryPrefixLookbackToken(part, batch, lookbackBatchCount);
            uint32_t windowStatus = static_cast<uint32_t>(CarryPrefixLookbackStatus::Pending);
            bool cancelled = false;
            const uint32_t windowTransform = ResolveCarryPrefixWindow(descriptors,
                                                                      part,
                                                                      batch * numWarps + warp,
                                                                      lane,
                                                                      token,
                                                                      packedLookbackStates,
                                                                      &windowStatus,
                                                                      &cancelled);
            if (cancelled)
                break;

            if (lane == 0u) {
                packedLookbackTransforms[warp] = windowTransform;
                StoreCarryPrefixLookbackStatus(
                    &packedLookbackStates[warp],
                    PackCarryPrefixLookbackStatus(token,
                                                  static_cast<CarryPrefixLookbackStatus>(windowStatus)));
            }

            bool complete = false;
            bool advance = false;
            do {
                const uint32_t controlStatus =
                    lane == 0u
                        ? LoadCarryPrefixLookbackStatus(&packedLookbackStates[CarryPrefixControlSlot])
                        : 0u;
                const uint32_t command = __shfl_sync(0xFFFF'FFFFu, controlStatus, 0);
                const uint32_t nextToken =
                    batch + 1u < lookbackBatchCount
                        ? MakeCarryPrefixLookbackToken(part, batch + 1u, lookbackBatchCount)
                        : 0u;
                if (command == PackCarryPrefixLookbackStatus(token, CarryPrefixLookbackStatus::Ready)) {
                    complete = true;
                } else if (batch + 1u < lookbackBatchCount &&
                           command == PackCarryPrefixLookbackStatus(nextToken,
                                                                    CarryPrefixLookbackStatus::Ready)) {
                    complete = true;
                } else if (batch + 1u < lookbackBatchCount &&
                           command == PackCarryPrefixLookbackStatus(
                                          nextToken, CarryPrefixLookbackStatus::Pending)) {
                    advance = true;
                } else if (lane == 0u) {
                    __nanosleep(64);
                }
            } while (!complete && !advance);

            if (complete)
                break;
        }
    }

    return packedLookbackTransforms[CarryPrefixControlSlot];
}

template <class SharkFloatParams>
__device__ void
InitializeCarryPrefixTransformsDLB(cooperative_groups::grid_group &grid,
                                   cooperative_groups::thread_block &block,
                                   uint32_t count,
                                   uint32_t capacity,
                                   HpSharkReferencePackedCarryPrefixDescriptor *descriptors,
                                   uint64_t *sharedStorage)
{
    if (count == 0u)
        return;

    const uint32_t blockSize = block.dim_threads().x;
    const uint32_t numParts = (count + blockSize - 1u) / blockSize;
    const uint32_t threadIndex = block.thread_index().x;
    const uint32_t lane = threadIndex & 31u;
    const uint32_t warp = threadIndex >> 5u;
    const uint32_t numWarps = (blockSize + 31u) >> 5u;
    uint32_t *packedCarryPrefixShared = reinterpret_cast<uint32_t *>(sharedStorage);
    uint32_t *packedLookbackStates = packedCarryPrefixShared + CarryPrefixLookbackStatesOffset;

    MattsCudaAssert(blockSize >= 32u && (blockSize & 31u) == 0u);
    MattsCudaAssert(numWarps <= CarryPrefixMaxWarps);
    MattsCudaAssert(capacity >= count);

    const uint32_t lookbackWindowsPerBatch = numWarps * 32u;
    const uint32_t lookbackBatchCount =
        numWarps == 1u ? 1u : (numParts + lookbackWindowsPerBatch - 1u) / lookbackWindowsPerBatch;
    MattsCudaAssert(lookbackBatchCount != 0u);
    MattsCudaAssert(numParts <= (1u << 30u) / lookbackBatchCount);

    const uint32_t gridSize = static_cast<uint32_t>(grid.size());
    const uint32_t processorId = block.group_index().x;
    const uint32_t firstPart = processorId < numParts ? processorId : 0u;
    const uint32_t firstToken = MakeCarryPrefixLookbackToken(firstPart, 0u, lookbackBatchCount);
    for (uint32_t part = GridThreadRank(block); part < numParts; part += gridSize)
        PublishCarryPrefixState(&descriptors[part].State, CarryPrefixDescriptorState::Empty);
    if (lane == 0u)
        StoreCarryPrefixLookbackStatus(
            &packedLookbackStates[warp],
            PackCarryPrefixLookbackStatus(firstToken, CarryPrefixLookbackStatus::Pending));
    // The caller must grid-sync before the scalar DLB runners consume these states.
}

static __device__ SharkForceInlineReleaseOnly uint32_t MakePackedCarryPrefixTwo(int64_t realLimb,
                                                                                int64_t imagLimb,
                                                                                bool hasValue);

static __device__ SharkForceInlineReleaseOnly uint32_t MakePackedCarryPrefixFour(
    int64_t realLimb, int64_t imagLimb, int64_t dzdcRealLimb, int64_t dzdcImagLimb, bool hasValue);

static __device__ SharkForceInlineReleaseOnly void EmitPackedCarryPrefixLane(uint32_t packedCarries,
                                                                             uint32_t streamShift,
                                                                             int64_t signedLimb,
                                                                             uint32_t index,
                                                                             uint32_t count,
                                                                             uint32_t capacity,
                                                                             uint32_t *digits,
                                                                             uint32_t *control);

static __device__ SharkForceInlineReleaseOnly uint32_t
ScanPackedCarryPrefixPart(uint32_t part,
                          uint32_t count,
                          uint32_t &packedInclusive,
                          HpSharkReferencePackedCarryPrefixDescriptor *descriptors,
                          uint64_t *sharedStorage,
                          uint32_t &packedWarpPrefix,
                          uint32_t &packedLocalExclusive);

template <class SharkFloatParams>
static __device__ SharkForceInlineReleaseOnly void
RunCarryPrefixNonNr(HpSharkReferenceWorkspace<SharkFloatParams> &workspace, uint64_t *sharedStorage)
{
    cooperative_groups::grid_group grid = cooperative_groups::this_grid();
    cooperative_groups::thread_block block = cooperative_groups::this_thread_block();
    auto *descriptors = reinterpret_cast<HpSharkReferencePackedCarryPrefixDescriptor *>(workspace.ZReal);
    using Workspace = HpSharkReferenceWorkspace<SharkFloatParams>;
    using Descriptor = HpSharkReferencePackedCarryPrefixDescriptor;
    const auto &iterationPlan = workspace.IterationPlan;
    const uint32_t count = iterationPlan.LimbCount;
    const uint32_t capacity = workspace.ActiveMaxFusedLimbs;
    if (count == 0u)
        return;

    const uint32_t descriptorWords =
        (workspace.ActiveMaxCarryPrefixParts * sizeof(Descriptor) + sizeof(uint64_t) - 1u) /
        sizeof(uint64_t);
    uint32_t *realDigits = reinterpret_cast<uint32_t *>(workspace.RealOutput);
    uint32_t *realControl =
        reinterpret_cast<uint32_t *>(workspace.RealOutput + capacity + descriptorWords);
    uint32_t *imagDigits = reinterpret_cast<uint32_t *>(workspace.ImagOutput);
    uint32_t *imagControl =
        reinterpret_cast<uint32_t *>(workspace.ImagOutput + capacity + descriptorWords);
    const uint32_t blockSize = block.dim_threads().x;
    const uint32_t threadIndex = block.thread_index().x;
    const uint32_t numParts = (count + blockSize - 1u) / blockSize;
    const uint32_t processorId = block.group_index().x;
    const uint32_t activeProcessors = gridDim.x;

    if (IsLeader(block)) {
        realControl[FinalizationHighestNonZeroControl] = 0u;
        imagControl[FinalizationHighestNonZeroControl] = 0u;
    }

    for (uint32_t part = processorId; part < numParts; part += activeProcessors) {
        const uint32_t base = part * blockSize;
        const uint32_t index = base + threadIndex;
        const bool hasValue = index < count;
        int64_t realLimb = 0;
        int64_t imagLimb = 0;
        if (hasValue) {
            realLimb = workspace.RealLimbs[index];
            imagLimb = workspace.ImagLimbs[index];
        }

        uint32_t packedInclusive = MakePackedCarryPrefixTwo(realLimb, imagLimb, hasValue);
        uint32_t packedWarpPrefix = 0u;
        uint32_t packedLocalExclusive = 0u;
        const uint32_t packedBlockExclusive = ScanPackedCarryPrefixPart(part,
                                                                        count,
                                                                        packedInclusive,
                                                                        descriptors,
                                                                        sharedStorage,
                                                                        packedWarpPrefix,
                                                                        packedLocalExclusive);
        const uint32_t packedExclusive = ComposePackedCarryPrefixes(
            packedBlockExclusive, ComposePackedCarryPrefixes(packedWarpPrefix, packedLocalExclusive));
        const uint32_t packedCarries = ApplyPackedCarryPrefix(packedExclusive, 0);
        EmitPackedCarryPrefixLane(
            packedCarries, 0u, realLimb, index, count, capacity, realDigits, realControl);
        EmitPackedCarryPrefixLane(
            packedCarries, 8u, imagLimb, index, count, capacity, imagDigits, imagControl);
    }
    grid.sync();
}

template <class SharkFloatParams>
static __device__ SharkForceInlineReleaseOnly void
RunCarryPrefixNr(HpSharkReferenceWorkspace<SharkFloatParams> &workspace, uint64_t *sharedStorage)
{
    cooperative_groups::grid_group grid = cooperative_groups::this_grid();
    cooperative_groups::thread_block block = cooperative_groups::this_thread_block();
    auto *descriptors = reinterpret_cast<HpSharkReferencePackedCarryPrefixDescriptor *>(workspace.ZReal);
    using Workspace = HpSharkReferenceWorkspace<SharkFloatParams>;
    using Descriptor = HpSharkReferencePackedCarryPrefixDescriptor;
    const auto &iterationPlan = workspace.IterationPlan;
    const uint32_t count = iterationPlan.LimbCount;
    const uint32_t capacity = workspace.ActiveMaxFusedLimbs;
    if (count == 0u)
        return;

    const uint32_t descriptorWords =
        (workspace.ActiveMaxCarryPrefixParts * sizeof(Descriptor) + sizeof(uint64_t) - 1u) /
        sizeof(uint64_t);
    uint32_t *realDigits = reinterpret_cast<uint32_t *>(workspace.RealOutput);
    uint32_t *realControl =
        reinterpret_cast<uint32_t *>(workspace.RealOutput + capacity + descriptorWords);
    uint32_t *imagDigits = reinterpret_cast<uint32_t *>(workspace.ImagOutput);
    uint32_t *imagControl =
        reinterpret_cast<uint32_t *>(workspace.ImagOutput + capacity + descriptorWords);
    uint32_t *dzdcRealDigits = reinterpret_cast<uint32_t *>(workspace.DzdcRealOutput);
    uint32_t *dzdcRealControl =
        reinterpret_cast<uint32_t *>(workspace.DzdcRealOutput + capacity + descriptorWords);
    uint32_t *dzdcImagDigits = reinterpret_cast<uint32_t *>(workspace.DzdcImagOutput);
    uint32_t *dzdcImagControl =
        reinterpret_cast<uint32_t *>(workspace.DzdcImagOutput + capacity + descriptorWords);
    const uint32_t blockSize = block.dim_threads().x;
    const uint32_t threadIndex = block.thread_index().x;
    const uint32_t numParts = (count + blockSize - 1u) / blockSize;
    const uint32_t processorId = block.group_index().x;
    const uint32_t activeProcessors = gridDim.x;

    if (IsLeader(block)) {
        realControl[FinalizationHighestNonZeroControl] = 0u;
        imagControl[FinalizationHighestNonZeroControl] = 0u;
        dzdcRealControl[FinalizationHighestNonZeroControl] = 0u;
        dzdcImagControl[FinalizationHighestNonZeroControl] = 0u;
    }

    for (uint32_t part = processorId; part < numParts; part += activeProcessors) {
        const uint32_t base = part * blockSize;
        const uint32_t index = base + threadIndex;
        const bool hasValue = index < count;
        int64_t realLimb = 0;
        int64_t imagLimb = 0;
        int64_t dzdcRealLimb = 0;
        int64_t dzdcImagLimb = 0;
        if (hasValue) {
            realLimb = workspace.RealLimbs[index];
            imagLimb = workspace.ImagLimbs[index];
            dzdcRealLimb = workspace.DzdcRealLimbs[index];
            dzdcImagLimb = workspace.DzdcImagLimbs[index];
        }

        uint32_t packedInclusive =
            MakePackedCarryPrefixFour(realLimb, imagLimb, dzdcRealLimb, dzdcImagLimb, hasValue);
        uint32_t packedWarpPrefix = 0u;
        uint32_t packedLocalExclusive = 0u;
        const uint32_t packedBlockExclusive = ScanPackedCarryPrefixPart(part,
                                                                        count,
                                                                        packedInclusive,
                                                                        descriptors,
                                                                        sharedStorage,
                                                                        packedWarpPrefix,
                                                                        packedLocalExclusive);
        const uint32_t packedExclusive = ComposePackedCarryPrefixes(
            packedBlockExclusive, ComposePackedCarryPrefixes(packedWarpPrefix, packedLocalExclusive));
        const uint32_t packedCarries = ApplyPackedCarryPrefix(packedExclusive, 0);
        EmitPackedCarryPrefixLane(
            packedCarries, 0u, realLimb, index, count, capacity, realDigits, realControl);
        EmitPackedCarryPrefixLane(
            packedCarries, 8u, imagLimb, index, count, capacity, imagDigits, imagControl);
        EmitPackedCarryPrefixLane(
            packedCarries, 16u, dzdcRealLimb, index, count, capacity, dzdcRealDigits, dzdcRealControl);
        EmitPackedCarryPrefixLane(
            packedCarries, 24u, dzdcImagLimb, index, count, capacity, dzdcImagDigits, dzdcImagControl);
    }
    grid.sync();
}

// B16 finalization keeps the packed four-lane DLB scan, but stream preparation and
// emission are scalar.  The caller selects the enabled linear contribution explicitly so no
// optional stream pointer reaches the scan.
static __device__ SharkForceInlineReleaseOnly uint32_t
MakePackedCarryPrefixTwo(int64_t realLimb, int64_t imagLimb, bool hasValue)
{
    constexpr uint32_t Identity = 0xFFFF'FFFFu;
    if (!hasValue)
        return Identity;
    return MakeSignedCarryPrefixByte(realLimb) | (MakeSignedCarryPrefixByte(imagLimb) << 8u);
}

static __device__ SharkForceInlineReleaseOnly uint32_t
MakePackedCarryPrefixFour(
    int64_t realLimb, int64_t imagLimb, int64_t dzdcRealLimb, int64_t dzdcImagLimb, bool hasValue)
{
    constexpr uint32_t Identity = 0xFFFF'FFFFu;
    if (!hasValue)
        return Identity;
    return MakeSignedCarryPrefixByte(realLimb) | (MakeSignedCarryPrefixByte(imagLimb) << 8u) |
           (MakeSignedCarryPrefixByte(dzdcRealLimb) << 16u) |
           (MakeSignedCarryPrefixByte(dzdcImagLimb) << 24u);
}

static __device__ SharkForceInlineReleaseOnly void
EmitPackedCarryPrefixLane(uint32_t packedCarries,
                          uint32_t streamShift,
                          int64_t signedLimb,
                          uint32_t index,
                          uint32_t count,
                          uint32_t capacity,
                          uint32_t *digits,
                          uint32_t *control)
{
    if (index >= count)
        return;
    const int32_t carryIn =
        static_cast<int32_t>((packedCarries >> streamShift) & 0xFFu) + CarryPrefixMin;
    StoreSignedCarryDigit(signedLimb, carryIn, index, count, capacity, digits, control);
}

static __device__ SharkForceInlineReleaseOnly uint32_t
ScanPackedCarryPrefixPart(uint32_t part,
                          uint32_t count,
                          uint32_t &packedInclusive,
                          HpSharkReferencePackedCarryPrefixDescriptor *descriptors,
                          uint64_t *sharedStorage,
                          uint32_t &packedWarpPrefix,
                          uint32_t &packedLocalExclusive)
{
    constexpr uint32_t Identity = 0xFFFF'FFFFu;
    constexpr uint32_t WarpSize = 32u;
    cooperative_groups::thread_block block = cooperative_groups::this_thread_block();
    const uint32_t blockSize = block.dim_threads().x;
    const uint32_t numParts = (count + blockSize - 1u) / blockSize;
    const uint32_t threadIndex = block.thread_index().x;
    const uint32_t lane = threadIndex & (WarpSize - 1u);
    const uint32_t warp = threadIndex >> 5u;
    const uint32_t numWarps = (blockSize + WarpSize - 1u) / WarpSize;
    uint32_t *packedCarryPrefixShared = reinterpret_cast<uint32_t *>(sharedStorage);
    uint32_t *packedWarpAggregates = packedCarryPrefixShared + CarryPrefixWarpAggregatesOffset;
    uint32_t *packedWarpPrefixes = packedCarryPrefixShared + CarryPrefixWarpPrefixesOffset;
    uint32_t *packedLookbackTransforms = packedCarryPrefixShared + CarryPrefixLookbackTransformsOffset;
    uint32_t *packedLookbackStates = packedCarryPrefixShared + CarryPrefixLookbackStatesOffset;

    MattsCudaAssert(blockSize >= WarpSize && (blockSize & (WarpSize - 1u)) == 0u);
    MattsCudaAssert(numWarps <= CarryPrefixMaxWarps);
    const uint32_t lookbackWindowsPerBatch = numWarps * WarpSize;
    const uint32_t lookbackBatchCount =
        numWarps == 1u ? 1u : (numParts + lookbackWindowsPerBatch - 1u) / lookbackWindowsPerBatch;
    MattsCudaAssert(lookbackBatchCount != 0u);
    MattsCudaAssert(numParts <= (1u << 30u) / lookbackBatchCount);

#pragma unroll
    for (uint32_t offset = 1u; offset < WarpSize; offset <<= 1u) {
        const uint32_t previous = __shfl_up_sync(0xFFFF'FFFFu, packedInclusive, offset);
        if (lane >= offset)
            packedInclusive = ComposePackedCarryPrefixes(previous, packedInclusive);
    }
    const uint32_t packedPrevious = __shfl_up_sync(0xFFFF'FFFFu, packedInclusive, 1);
    packedLocalExclusive = lane == 0u ? Identity : packedPrevious;

    const uint32_t warpEnd = (warp + 1u) * WarpSize;
    const uint32_t warpLastThread = (warpEnd < blockSize ? warpEnd : blockSize) - 1u;
    if (threadIndex == warpLastThread)
        packedWarpAggregates[warp] = packedInclusive;
    __syncthreads();

    uint32_t packedAggregate = Identity;
    if (threadIndex < WarpSize) {
        uint32_t packedWarpInclusive = lane < numWarps ? packedWarpAggregates[lane] : Identity;
#pragma unroll
        for (uint32_t offset = 1u; offset < WarpSize; offset <<= 1u) {
            const uint32_t previous = __shfl_up_sync(0xFFFF'FFFFu, packedWarpInclusive, offset);
            if (lane >= offset && lane < numWarps)
                packedWarpInclusive = ComposePackedCarryPrefixes(previous, packedWarpInclusive);
        }

        const uint32_t previous = __shfl_up_sync(0xFFFF'FFFFu, packedWarpInclusive, 1);
        if (lane < numWarps)
            packedWarpPrefixes[lane] = lane == 0u ? Identity : previous;
        packedAggregate =
            __shfl_sync(0xFFFF'FFFFu, packedWarpInclusive, static_cast<int>(numWarps - 1u));
    }

    if (threadIndex == 0u)
        PublishCarryPrefixDescriptorAggregate(descriptors[part], packedAggregate);

    const uint32_t resolvedBlockExclusive = ResolveCarryPrefixBlockExclusive(descriptors,
                                                                             part,
                                                                             lane,
                                                                             warp,
                                                                             numWarps,
                                                                             lookbackBatchCount,
                                                                             packedLookbackTransforms,
                                                                             packedLookbackStates);

    if (threadIndex == 0u) {
        PublishCarryPrefixDescriptorPrefix(
            descriptors[part], ComposePackedCarryPrefixes(resolvedBlockExclusive, packedAggregate));
    }
    __syncthreads();

    packedWarpPrefix = packedWarpPrefixes[warp];
    return packedLookbackTransforms[CarryPrefixControlSlot];
}

template <class SharkFloatParams>
static __device__ SharkForceInlineReleaseOnly void
RunAlignedB16CarryPrefix(HpSharkReferenceWorkspace<SharkFloatParams> &workspace,
                         HpSharkReferenceResults<SharkFloatParams> *combo,
                         uint64_t *carryPrefixShared)
{
    cooperative_groups::grid_group grid = cooperative_groups::this_grid();
    cooperative_groups::thread_block block = cooperative_groups::this_thread_block();
    const auto &iterationPlan = workspace.IterationPlan;
    const uint32_t count = iterationPlan.LimbCount;
    const uint32_t capacity = workspace.ActiveMaxFusedLimbs;
    if (count == 0u)
        return;

    MattsCudaAssert(iterationPlan.PlanSlot <
                    HpSharkReferenceWorkspace<SharkFloatParams>::PlanCacheEntryCount);
    const SharkNTT::Plan &plan = workspace.Plans[iterationPlan.PlanSlot];
    MattsCudaAssert(plan.b == 16);

    auto *descriptors = reinterpret_cast<HpSharkReferencePackedCarryPrefixDescriptor *>(workspace.ZReal);
    const uint32_t descriptorWords =
        (workspace.ActiveMaxCarryPrefixParts * sizeof(HpSharkReferencePackedCarryPrefixDescriptor) +
         sizeof(uint64_t) - 1u) /
        sizeof(uint64_t);
    uint32_t *realDigits = reinterpret_cast<uint32_t *>(workspace.ZReal + descriptorWords);
    uint32_t *realControl = reinterpret_cast<uint32_t *>(workspace.ZReal + descriptorWords + capacity);
    uint32_t *imagDigits = reinterpret_cast<uint32_t *>(workspace.ZImag + descriptorWords);
    uint32_t *imagControl = reinterpret_cast<uint32_t *>(workspace.ZImag + descriptorWords + capacity);

    const uint32_t flags = iterationPlan.Flags;
    const bool realLinearEnabled = (flags & HpSharkReferencePlanRealLinear) != 0u;
    const bool imagLinearEnabled = (flags & HpSharkReferencePlanImagLinear) != 0u;
    const uint32_t blockSize = block.dim_threads().x;
    const uint32_t threadIndex = block.thread_index().x;
    const uint32_t numParts = (count + blockSize - 1u) / blockSize;
    const uint32_t processorId = block.group_index().x;
    const uint32_t activeProcessors = gridDim.x;

    if (IsLeader(block)) {
        realControl[FinalizationHighestNonZeroControl] = 0u;
        imagControl[FinalizationHighestNonZeroControl] = 0u;
    }

    for (uint32_t part = processorId; part < numParts; part += activeProcessors) {
        const uint32_t base = part * blockSize;
        const uint32_t index = base + threadIndex;
        const bool hasValue = index < count;
        const uint32_t limbBegin = base + (threadIndex & ~31u);
        const uint32_t remainingLimbs = limbBegin < count ? count - limbBegin : 0u;
        const uint32_t tileLimbCount = remainingLimbs < 32u ? remainingLimbs : 32u;

        int64_t realLimb = 0;
        int64_t imagLimb = 0;
        if (tileLimbCount != 0u) {
            if (realLinearEnabled) {
                realLimb = PrepareAlignedB16SignedLimbWithLinear<SharkFloatParams>(
                    workspace.RealOutput,
                    iterationPlan.RealCoefficientCount,
                    iterationPlan.RealProductBitOffset,
                    &combo->CReal,
                    workspace.IgnoredPrecisionBits,
                    iterationPlan.RealLinearBitOffset,
                    limbBegin,
                    tileLimbCount);
            } else {
                realLimb = PrepareAlignedB16SignedLimbNoLinear<SharkFloatParams>(
                    workspace.RealOutput,
                    iterationPlan.RealCoefficientCount,
                    iterationPlan.RealProductBitOffset,
                    limbBegin,
                    tileLimbCount);
            }
            if (imagLinearEnabled) {
                imagLimb = PrepareAlignedB16SignedLimbWithLinear<SharkFloatParams>(
                    workspace.ImagOutput,
                    iterationPlan.ImagCoefficientCount,
                    iterationPlan.ImagProductBitOffset,
                    &combo->CImag,
                    workspace.IgnoredPrecisionBits,
                    iterationPlan.ImagLinearBitOffset,
                    limbBegin,
                    tileLimbCount);
            } else {
                imagLimb = PrepareAlignedB16SignedLimbNoLinear<SharkFloatParams>(
                    workspace.ImagOutput,
                    iterationPlan.ImagCoefficientCount,
                    iterationPlan.ImagProductBitOffset,
                    limbBegin,
                    tileLimbCount);
            }
        }

        if constexpr (HpShark::DebugChecksums) {
            if (hasValue) {
                workspace.RealLimbs[index] = realLimb;
                workspace.ImagLimbs[index] = imagLimb;
            }
        }

        uint32_t packedInclusive = MakePackedCarryPrefixTwo(realLimb, imagLimb, hasValue);
        uint32_t packedWarpPrefix = 0u;
        uint32_t packedLocalExclusive = 0u;
        const uint32_t packedBlockExclusive = ScanPackedCarryPrefixPart(part,
                                                                        count,
                                                                        packedInclusive,
                                                                        descriptors,
                                                                        carryPrefixShared,
                                                                        packedWarpPrefix,
                                                                        packedLocalExclusive);
        const uint32_t packedExclusive = ComposePackedCarryPrefixes(
            packedBlockExclusive, ComposePackedCarryPrefixes(packedWarpPrefix, packedLocalExclusive));
        const uint32_t packedCarries = ApplyPackedCarryPrefix(packedExclusive, 0);
        EmitPackedCarryPrefixLane(
            packedCarries, 0u, realLimb, index, count, capacity, realDigits, realControl);
        EmitPackedCarryPrefixLane(
            packedCarries, 8u, imagLimb, index, count, capacity, imagDigits, imagControl);
    }
    grid.sync();
}

template <class SharkFloatParams>
static __device__ SharkForceInlineReleaseOnly void
RunAlignedB16CarryPrefixNr(HpSharkReferenceWorkspace<SharkFloatParams> &workspace,
                           HpSharkReferenceResults<SharkFloatParams> *combo,
                           uint64_t *carryPrefixShared)
{
    cooperative_groups::grid_group grid = cooperative_groups::this_grid();
    cooperative_groups::thread_block block = cooperative_groups::this_thread_block();
    const auto &iterationPlan = workspace.IterationPlan;
    const uint32_t count = iterationPlan.LimbCount;
    const uint32_t capacity = workspace.ActiveMaxFusedLimbs;
    if (count == 0u)
        return;

    MattsCudaAssert(iterationPlan.PlanSlot <
                    HpSharkReferenceWorkspace<SharkFloatParams>::PlanCacheEntryCount);
    const SharkNTT::Plan &plan = workspace.Plans[iterationPlan.PlanSlot];
    MattsCudaAssert(plan.b == 16);

    auto *descriptors = reinterpret_cast<HpSharkReferencePackedCarryPrefixDescriptor *>(workspace.ZReal);
    const uint32_t descriptorWords =
        (workspace.ActiveMaxCarryPrefixParts * sizeof(HpSharkReferencePackedCarryPrefixDescriptor) +
         sizeof(uint64_t) - 1u) /
        sizeof(uint64_t);
    uint32_t *realDigits = reinterpret_cast<uint32_t *>(workspace.ZReal + descriptorWords);
    uint32_t *realControl = reinterpret_cast<uint32_t *>(workspace.ZReal + descriptorWords + capacity);
    uint32_t *imagDigits = reinterpret_cast<uint32_t *>(workspace.ZImag + descriptorWords);
    uint32_t *imagControl = reinterpret_cast<uint32_t *>(workspace.ZImag + descriptorWords + capacity);
    uint32_t *dzdcRealDigits = reinterpret_cast<uint32_t *>(workspace.DzdcReal + descriptorWords);
    uint32_t *dzdcRealControl =
        reinterpret_cast<uint32_t *>(workspace.DzdcReal + descriptorWords + capacity);
    uint32_t *dzdcImagDigits = reinterpret_cast<uint32_t *>(workspace.DzdcImag + descriptorWords);
    uint32_t *dzdcImagControl =
        reinterpret_cast<uint32_t *>(workspace.DzdcImag + descriptorWords + capacity);

    const uint32_t flags = iterationPlan.Flags;
    const bool realLinearEnabled = (flags & HpSharkReferencePlanRealLinear) != 0u;
    const bool imagLinearEnabled = (flags & HpSharkReferencePlanImagLinear) != 0u;
    const bool dzdcOneEnabled = (flags & HpSharkReferencePlanDzdcOne) != 0u;
    const uint32_t blockSize = block.dim_threads().x;
    const uint32_t threadIndex = block.thread_index().x;
    const uint32_t numParts = (count + blockSize - 1u) / blockSize;
    const uint32_t processorId = block.group_index().x;
    const uint32_t activeProcessors = gridDim.x;

    if (IsLeader(block)) {
        realControl[FinalizationHighestNonZeroControl] = 0u;
        imagControl[FinalizationHighestNonZeroControl] = 0u;
        dzdcRealControl[FinalizationHighestNonZeroControl] = 0u;
        dzdcImagControl[FinalizationHighestNonZeroControl] = 0u;
    }

    for (uint32_t part = processorId; part < numParts; part += activeProcessors) {
        const uint32_t base = part * blockSize;
        const uint32_t index = base + threadIndex;
        const bool hasValue = index < count;
        const uint32_t limbBegin = base + (threadIndex & ~31u);
        const uint32_t remainingLimbs = limbBegin < count ? count - limbBegin : 0u;
        const uint32_t tileLimbCount = remainingLimbs < 32u ? remainingLimbs : 32u;

        int64_t realLimb = 0;
        int64_t imagLimb = 0;
        int64_t dzdcRealLimb = 0;
        int64_t dzdcImagLimb = 0;
        if (tileLimbCount != 0u) {
            if (realLinearEnabled) {
                realLimb = PrepareAlignedB16SignedLimbWithLinear<SharkFloatParams>(
                    workspace.RealOutput,
                    iterationPlan.RealCoefficientCount,
                    iterationPlan.RealProductBitOffset,
                    &combo->CReal,
                    workspace.IgnoredPrecisionBits,
                    iterationPlan.RealLinearBitOffset,
                    limbBegin,
                    tileLimbCount);
            } else {
                realLimb = PrepareAlignedB16SignedLimbNoLinear<SharkFloatParams>(
                    workspace.RealOutput,
                    iterationPlan.RealCoefficientCount,
                    iterationPlan.RealProductBitOffset,
                    limbBegin,
                    tileLimbCount);
            }
            if (imagLinearEnabled) {
                imagLimb = PrepareAlignedB16SignedLimbWithLinear<SharkFloatParams>(
                    workspace.ImagOutput,
                    iterationPlan.ImagCoefficientCount,
                    iterationPlan.ImagProductBitOffset,
                    &combo->CImag,
                    workspace.IgnoredPrecisionBits,
                    iterationPlan.ImagLinearBitOffset,
                    limbBegin,
                    tileLimbCount);
            } else {
                imagLimb = PrepareAlignedB16SignedLimbNoLinear<SharkFloatParams>(
                    workspace.ImagOutput,
                    iterationPlan.ImagCoefficientCount,
                    iterationPlan.ImagProductBitOffset,
                    limbBegin,
                    tileLimbCount);
            }
            if (dzdcOneEnabled) {
                dzdcRealLimb = PrepareAlignedB16SignedLimbWithLinear<SharkFloatParams>(
                    workspace.DzdcRealOutput,
                    iterationPlan.DzdcRealCoefficientCount,
                    iterationPlan.DzdcRealProductBitOffset,
                    &combo->One,
                    workspace.IgnoredPrecisionBits,
                    iterationPlan.DzdcRealLinearBitOffset,
                    limbBegin,
                    tileLimbCount);
            } else {
                dzdcRealLimb = PrepareAlignedB16SignedLimbNoLinear<SharkFloatParams>(
                    workspace.DzdcRealOutput,
                    iterationPlan.DzdcRealCoefficientCount,
                    iterationPlan.DzdcRealProductBitOffset,
                    limbBegin,
                    tileLimbCount);
            }
            dzdcImagLimb = PrepareAlignedB16SignedLimbNoLinear<SharkFloatParams>(
                workspace.DzdcImagOutput,
                iterationPlan.DzdcImagCoefficientCount,
                iterationPlan.DzdcImagProductBitOffset,
                limbBegin,
                tileLimbCount);
        }

        if constexpr (HpShark::DebugChecksums) {
            if (hasValue) {
                workspace.RealLimbs[index] = realLimb;
                workspace.ImagLimbs[index] = imagLimb;
                workspace.DzdcRealLimbs[index] = dzdcRealLimb;
                workspace.DzdcImagLimbs[index] = dzdcImagLimb;
            }
        }

        uint32_t packedInclusive =
            MakePackedCarryPrefixFour(realLimb, imagLimb, dzdcRealLimb, dzdcImagLimb, hasValue);
        uint32_t packedWarpPrefix = 0u;
        uint32_t packedLocalExclusive = 0u;
        const uint32_t packedBlockExclusive = ScanPackedCarryPrefixPart(part,
                                                                        count,
                                                                        packedInclusive,
                                                                        descriptors,
                                                                        carryPrefixShared,
                                                                        packedWarpPrefix,
                                                                        packedLocalExclusive);
        const uint32_t packedExclusive = ComposePackedCarryPrefixes(
            packedBlockExclusive, ComposePackedCarryPrefixes(packedWarpPrefix, packedLocalExclusive));
        const uint32_t packedCarries = ApplyPackedCarryPrefix(packedExclusive, 0);
        EmitPackedCarryPrefixLane(
            packedCarries, 0u, realLimb, index, count, capacity, realDigits, realControl);
        EmitPackedCarryPrefixLane(
            packedCarries, 8u, imagLimb, index, count, capacity, imagDigits, imagControl);
        EmitPackedCarryPrefixLane(
            packedCarries, 16u, dzdcRealLimb, index, count, capacity, dzdcRealDigits, dzdcRealControl);
        EmitPackedCarryPrefixLane(
            packedCarries, 24u, dzdcImagLimb, index, count, capacity, dzdcImagDigits, dzdcImagControl);
    }
    grid.sync();
}

template <class SharkFloatParams>
__device__ SharkForceInlineReleaseOnly void
FinalizeSignedStream(DebugState<SharkFloatParams> *debugStates,
                     uint64_t *carryPrefixShared,
                     HpSharkReferenceWorkspace<SharkFloatParams> &workspace,
                     HpSharkReferenceResults<SharkFloatParams> *combo)
{
    cooperative_groups::grid_group grid = cooperative_groups::this_grid();
    cooperative_groups::thread_block block = cooperative_groups::this_thread_block();
    using Workspace = HpSharkReferenceWorkspace<SharkFloatParams>;
    using Descriptor = HpSharkReferencePackedCarryPrefixDescriptor;
    constexpr uint32_t MaxCapacity = Workspace::MaxFusedLimbs;
    constexpr uint32_t MaxDescriptorWords =
        (Workspace::MaxCarryPrefixParts * sizeof(Descriptor) + sizeof(uint64_t) - 1u) / sizeof(uint64_t);
    constexpr uint32_t MaxControlWords =
        (Workspace::CarryPrefixControlCount * sizeof(uint32_t) + sizeof(uint64_t) - 1u) /
        sizeof(uint64_t);
    static_assert((MaxCapacity * sizeof(uint64_t)) % alignof(Descriptor) == 0u);
    static_assert(MaxCapacity + MaxDescriptorWords + MaxControlWords <= Workspace::MaxFusedN);
    const uint32_t capacity = workspace.ActiveMaxFusedLimbs;
    const auto &iterationPlan = workspace.IterationPlan;
    const uint32_t limbCount = iterationPlan.LimbCount;
    const int32_t realExponent = iterationPlan.RealExponent;
    const int32_t imagExponent = iterationPlan.ImagExponent;
    const int32_t dzdcRealExponent = iterationPlan.DzdcRealExponent;
    const int32_t dzdcImagExponent = iterationPlan.DzdcImagExponent;
    const bool carryPrefixReady = workspace.Plans[iterationPlan.PlanSlot].b == 16u;
    const uint32_t descriptorWords =
        (workspace.ActiveMaxCarryPrefixParts * sizeof(Descriptor) + sizeof(uint64_t) - 1u) /
        sizeof(uint64_t);
    const uint32_t controlWords =
        (Workspace::CarryPrefixControlCount * sizeof(uint32_t) + sizeof(uint64_t) - 1u) /
        sizeof(uint64_t);
    MattsCudaAssert(capacity <= MaxCapacity);
    MattsCudaAssert(capacity + descriptorWords + controlWords <= workspace.ActiveMaxFusedN);

    // Fused B16 carries remain in the dead Z arenas; avoid copying the full workspace capacity.
    uint64_t *realOutputArena =
        carryPrefixReady ? workspace.ZReal + descriptorWords : workspace.RealOutput;
    int64_t *realLimbs = workspace.RealLimbs;
    uint32_t *realDigits = reinterpret_cast<uint32_t *>(realOutputArena);
    uint32_t *realControl = reinterpret_cast<uint32_t *>(realOutputArena + capacity +
                                                         (carryPrefixReady ? 0u : descriptorWords));
    HpSharkFloat<SharkFloatParams> *realOutput = &combo->ZReal;

    uint64_t *imagOutputArena =
        carryPrefixReady ? workspace.ZImag + descriptorWords : workspace.ImagOutput;
    int64_t *imagLimbs = workspace.ImagLimbs;
    uint32_t *imagDigits = reinterpret_cast<uint32_t *>(imagOutputArena);
    uint32_t *imagControl = reinterpret_cast<uint32_t *>(imagOutputArena + capacity +
                                                         (carryPrefixReady ? 0u : descriptorWords));
    HpSharkFloat<SharkFloatParams> *imagOutput = &combo->ZImag;

    int64_t *dzdcRealLimbs = workspace.DzdcRealLimbs;
    uint64_t *dzdcRealOutputArena =
        carryPrefixReady ? workspace.DzdcReal + descriptorWords : workspace.DzdcRealOutput;
    uint32_t *dzdcRealDigits = reinterpret_cast<uint32_t *>(dzdcRealOutputArena);
    uint32_t *dzdcRealControl = reinterpret_cast<uint32_t *>(dzdcRealOutputArena + capacity +
                                                             (carryPrefixReady ? 0u : descriptorWords));
    HpSharkFloat<SharkFloatParams> *dzdcRealOutput = &combo->DzdcReal;
    int64_t *dzdcImagLimbs = workspace.DzdcImagLimbs;
    uint64_t *dzdcImagOutputArena =
        carryPrefixReady ? workspace.DzdcImag + descriptorWords : workspace.DzdcImagOutput;
    uint32_t *dzdcImagDigits = reinterpret_cast<uint32_t *>(dzdcImagOutputArena);
    uint32_t *dzdcImagControl = reinterpret_cast<uint32_t *>(dzdcImagOutputArena + capacity +
                                                             (carryPrefixReady ? 0u : descriptorWords));
    HpSharkFloat<SharkFloatParams> *dzdcImagOutput = &combo->DzdcImag;

    MattsCudaAssert(limbCount > 0u && limbCount <= capacity);
    const uint32_t gridSize = static_cast<uint32_t>(grid.size());

    if (!carryPrefixReady) {
        if (IsLeader(block)) {
            realControl[FinalizationHighestNonZeroControl] = 0u;
            imagControl[FinalizationHighestNonZeroControl] = 0u;
            if constexpr (SharkFloatParams::EnableNewtonRaphson) {
                dzdcRealControl[FinalizationHighestNonZeroControl] = 0u;
                dzdcImagControl[FinalizationHighestNonZeroControl] = 0u;
            }
        }
        if constexpr (SharkFloatParams::EnableNewtonRaphson) {
            RunCarryPrefixNr<SharkFloatParams>(workspace, carryPrefixShared);
        } else {
            RunCarryPrefixNonNr<SharkFloatParams>(workspace, carryPrefixShared);
        }
    }

    const uint32_t realDigitLength = realControl[FinalizationDigitLengthControl];
    const uint32_t imagDigitLength = imagControl[FinalizationDigitLengthControl];
    uint32_t maximumDigitLength = realDigitLength > imagDigitLength ? realDigitLength : imagDigitLength;
    if constexpr (SharkFloatParams::EnableNewtonRaphson) {
        const uint32_t dzdcRealDigitLength = dzdcRealControl[FinalizationDigitLengthControl];
        const uint32_t dzdcImagDigitLength = dzdcImagControl[FinalizationDigitLengthControl];
        maximumDigitLength =
            maximumDigitLength > dzdcRealDigitLength ? maximumDigitLength : dzdcRealDigitLength;
        maximumDigitLength =
            maximumDigitLength > dzdcImagDigitLength ? maximumDigitLength : dzdcImagDigitLength;
    }

    if constexpr (HpShark::DebugChecksums) {
        if constexpr (SharkFloatParams::EnableNewtonRaphson) {
            StoreReferenceDebugStateBatchAfterSync<SharkFloatParams>(
                debugStates,
                grid,
                block,
                DebugStatePurpose::SignedCarry1,
                realDigits,
                realControl[FinalizationDigitLengthControl],
                DebugStatePurpose::SignedCarry2,
                imagDigits,
                imagControl[FinalizationDigitLengthControl],
                DebugStatePurpose::SignedCarryDzdc1,
                dzdcRealDigits,
                dzdcRealControl[FinalizationDigitLengthControl],
                DebugStatePurpose::SignedCarryDzdc2,
                dzdcImagDigits,
                dzdcImagControl[FinalizationDigitLengthControl]);
        } else {
            StoreReferenceDebugStateBatchAfterSync<SharkFloatParams>(
                debugStates,
                grid,
                block,
                DebugStatePurpose::SignedCarry1,
                realDigits,
                realControl[FinalizationDigitLengthControl],
                DebugStatePurpose::SignedCarry2,
                imagDigits,
                imagControl[FinalizationDigitLengthControl]);
        }
    }

    // In (~digits) + 1, the carry reaches the lowest nonzero digit and stops there.
    // Locating that digit avoids a second cross-block carry-prefix scan.
    if constexpr (SharkFloatParams::EnableNewtonRaphson) {
        FindLowestNr<SharkFloatParams>(realDigits,
                                       realControl,
                                       imagDigits,
                                       imagControl,
                                       dzdcRealDigits,
                                       dzdcRealControl,
                                       dzdcImagDigits,
                                       dzdcImagControl,
                                       carryPrefixShared);
    } else {
        FindLowestNonNr<SharkFloatParams>(
            realDigits, realControl, imagDigits, imagControl, carryPrefixShared);
    }

    uint32_t realLocalHighest = 0u;
    uint32_t imagLocalHighest = 0u;
    uint32_t dzdcRealLocalHighest = 0u;
    uint32_t dzdcImagLocalHighest = 0u;
    for (uint32_t index = GridThreadRank(block); index < maximumDigitLength; index += gridSize) {
        if (realControl[FinalizationNegativeControl] != 0u && index < realDigitLength) {
            const uint32_t lowestNonZero = realControl[FinalizationLowestNonZeroControl];
            if (index < lowestNonZero)
                realDigits[index] = 0u;
            else if (index == lowestNonZero)
                realDigits[index] = 0u - realDigits[index];
            else
                realDigits[index] = ~realDigits[index];
        }
        if (imagControl[FinalizationNegativeControl] != 0u && index < imagDigitLength) {
            const uint32_t lowestNonZero = imagControl[FinalizationLowestNonZeroControl];
            if (index < lowestNonZero)
                imagDigits[index] = 0u;
            else if (index == lowestNonZero)
                imagDigits[index] = 0u - imagDigits[index];
            else
                imagDigits[index] = ~imagDigits[index];
        }
        if (index < realDigitLength && realDigits[index] != 0u)
            realLocalHighest = index + 1u;
        if (index < imagDigitLength && imagDigits[index] != 0u)
            imagLocalHighest = index + 1u;
        if constexpr (SharkFloatParams::EnableNewtonRaphson) {
            const uint32_t dzdcRealDigitLength = dzdcRealControl[FinalizationDigitLengthControl];
            if (dzdcRealControl[FinalizationNegativeControl] != 0u && index < dzdcRealDigitLength) {
                const uint32_t lowestNonZero = dzdcRealControl[FinalizationLowestNonZeroControl];
                if (index < lowestNonZero)
                    dzdcRealDigits[index] = 0u;
                else if (index == lowestNonZero)
                    dzdcRealDigits[index] = 0u - dzdcRealDigits[index];
                else
                    dzdcRealDigits[index] = ~dzdcRealDigits[index];
            }

            const uint32_t dzdcImagDigitLength = dzdcImagControl[FinalizationDigitLengthControl];
            if (dzdcImagControl[FinalizationNegativeControl] != 0u && index < dzdcImagDigitLength) {
                const uint32_t lowestNonZero = dzdcImagControl[FinalizationLowestNonZeroControl];
                if (index < lowestNonZero)
                    dzdcImagDigits[index] = 0u;
                else if (index == lowestNonZero)
                    dzdcImagDigits[index] = 0u - dzdcImagDigits[index];
                else
                    dzdcImagDigits[index] = ~dzdcImagDigits[index];
            }
            if (index < dzdcRealDigitLength && dzdcRealDigits[index] != 0u)
                dzdcRealLocalHighest = index + 1u;
            if (index < dzdcImagDigitLength && dzdcImagDigits[index] != 0u)
                dzdcImagLocalHighest = index + 1u;
        }
    }

    const uint32_t threadIndex = block.thread_index().x;
    uint32_t *blockMaximum = reinterpret_cast<uint32_t *>(carryPrefixShared);
    if (threadIndex == 0u) {
        blockMaximum[0] = 0u;
        blockMaximum[1] = 0u;
        if constexpr (SharkFloatParams::EnableNewtonRaphson) {
            blockMaximum[2] = 0u;
            blockMaximum[3] = 0u;
        }
    }
    __syncthreads();
    if (realLocalHighest != 0u)
        atomicMax(&blockMaximum[0], realLocalHighest);
    if (imagLocalHighest != 0u)
        atomicMax(&blockMaximum[1], imagLocalHighest);
    if constexpr (SharkFloatParams::EnableNewtonRaphson) {
        if (dzdcRealLocalHighest != 0u)
            atomicMax(&blockMaximum[2], dzdcRealLocalHighest);
        if (dzdcImagLocalHighest != 0u)
            atomicMax(&blockMaximum[3], dzdcImagLocalHighest);
    }
    __syncthreads();
    if (threadIndex == 0u) {
        atomicMax(&realControl[FinalizationHighestNonZeroControl], blockMaximum[0]);
        atomicMax(&imagControl[FinalizationHighestNonZeroControl], blockMaximum[1]);
        if constexpr (SharkFloatParams::EnableNewtonRaphson) {
            atomicMax(&dzdcRealControl[FinalizationHighestNonZeroControl], blockMaximum[2]);
            atomicMax(&dzdcImagControl[FinalizationHighestNonZeroControl], blockMaximum[3]);
        }
    }
    grid.sync();

    if (IsLeader(block)) {
        uint32_t currentRealDigitLength = realControl[FinalizationDigitLengthControl];
        if (realControl[FinalizationNegativeControl] != 0u &&
            realControl[FinalizationLowestNonZeroControl] == currentRealDigitLength) {
            MattsCudaAssert(currentRealDigitLength < capacity);
            if (currentRealDigitLength < capacity)
                realDigits[currentRealDigitLength++] = 1u;
        }
        realControl[FinalizationDigitLengthControl] = currentRealDigitLength;
        if (realControl[FinalizationHighestNonZeroControl] < currentRealDigitLength &&
            currentRealDigitLength != 0u && realDigits[currentRealDigitLength - 1u] != 0u)
            realControl[FinalizationHighestNonZeroControl] = currentRealDigitLength;

        uint32_t currentImagDigitLength = imagControl[FinalizationDigitLengthControl];
        if (imagControl[FinalizationNegativeControl] != 0u &&
            imagControl[FinalizationLowestNonZeroControl] == currentImagDigitLength) {
            MattsCudaAssert(currentImagDigitLength < capacity);
            if (currentImagDigitLength < capacity)
                imagDigits[currentImagDigitLength++] = 1u;
        }
        imagControl[FinalizationDigitLengthControl] = currentImagDigitLength;
        if (imagControl[FinalizationHighestNonZeroControl] < currentImagDigitLength &&
            currentImagDigitLength != 0u && imagDigits[currentImagDigitLength - 1u] != 0u)
            imagControl[FinalizationHighestNonZeroControl] = currentImagDigitLength;

        if constexpr (SharkFloatParams::EnableNewtonRaphson) {
            uint32_t currentDzdcRealDigitLength = dzdcRealControl[FinalizationDigitLengthControl];
            if (dzdcRealControl[FinalizationNegativeControl] != 0u &&
                dzdcRealControl[FinalizationLowestNonZeroControl] == currentDzdcRealDigitLength) {
                MattsCudaAssert(currentDzdcRealDigitLength < capacity);
                if (currentDzdcRealDigitLength < capacity)
                    dzdcRealDigits[currentDzdcRealDigitLength++] = 1u;
            }
            dzdcRealControl[FinalizationDigitLengthControl] = currentDzdcRealDigitLength;
            if (dzdcRealControl[FinalizationHighestNonZeroControl] < currentDzdcRealDigitLength &&
                currentDzdcRealDigitLength != 0u &&
                dzdcRealDigits[currentDzdcRealDigitLength - 1u] != 0u)
                dzdcRealControl[FinalizationHighestNonZeroControl] = currentDzdcRealDigitLength;

            uint32_t currentDzdcImagDigitLength = dzdcImagControl[FinalizationDigitLengthControl];
            if (dzdcImagControl[FinalizationNegativeControl] != 0u &&
                dzdcImagControl[FinalizationLowestNonZeroControl] == currentDzdcImagDigitLength) {
                MattsCudaAssert(currentDzdcImagDigitLength < capacity);
                if (currentDzdcImagDigitLength < capacity)
                    dzdcImagDigits[currentDzdcImagDigitLength++] = 1u;
            }
            dzdcImagControl[FinalizationDigitLengthControl] = currentDzdcImagDigitLength;
            if (dzdcImagControl[FinalizationHighestNonZeroControl] < currentDzdcImagDigitLength &&
                currentDzdcImagDigitLength != 0u &&
                dzdcImagDigits[currentDzdcImagDigitLength - 1u] != 0u)
                dzdcImagControl[FinalizationHighestNonZeroControl] = currentDzdcImagDigitLength;
        }
    }
    grid.sync();

    if constexpr (HpShark::DebugChecksums) {
        if constexpr (SharkFloatParams::EnableNewtonRaphson) {
            StoreReferenceDebugStateBatchAfterSync<SharkFloatParams>(
                debugStates,
                grid,
                block,
                DebugStatePurpose::FinalAdd1,
                realDigits,
                realControl[FinalizationHighestNonZeroControl],
                DebugStatePurpose::FinalAdd2,
                imagDigits,
                imagControl[FinalizationHighestNonZeroControl],
                DebugStatePurpose::FinalAddDzdc1,
                dzdcRealDigits,
                dzdcRealControl[FinalizationHighestNonZeroControl],
                DebugStatePurpose::FinalAddDzdc2,
                dzdcImagDigits,
                dzdcImagControl[FinalizationHighestNonZeroControl]);
        } else {
            StoreReferenceDebugStateBatchAfterSync<SharkFloatParams>(
                debugStates,
                grid,
                block,
                DebugStatePurpose::FinalAdd1,
                realDigits,
                realControl[FinalizationHighestNonZeroControl],
                DebugStatePurpose::FinalAdd2,
                imagDigits,
                imagControl[FinalizationHighestNonZeroControl]);
        }
    }

    constexpr uint32_t ActualDigits = SharkFloatParams::GlobalNumUint32;
    constexpr int DesiredBit = (static_cast<int>(ActualDigits) - 1) * 32 + 31;
    if (IsLeader(block)) {
        const uint32_t realHighestNonZeroPlusOne = realControl[FinalizationHighestNonZeroControl];
        if (realHighestNonZeroPlusOne == 0u) {
            realOutput->Exponent = -100'000'000;
            realOutput->SetNegative(false);
        } else {
            const uint32_t highestNonZero = realHighestNonZeroPlusOne - 1u;
            const int currentBit =
                static_cast<int>(highestNonZero) * 32 + 31 - __clz(realDigits[highestNonZero]);
            realOutput->Exponent = realExponent + currentBit - DesiredBit;
            realOutput->SetNegative(realControl[FinalizationNegativeControl] != 0u);
        }

        const uint32_t imagHighestNonZeroPlusOne = imagControl[FinalizationHighestNonZeroControl];
        if (imagHighestNonZeroPlusOne == 0u) {
            imagOutput->Exponent = -100'000'000;
            imagOutput->SetNegative(false);
        } else {
            const uint32_t highestNonZero = imagHighestNonZeroPlusOne - 1u;
            const int currentBit =
                static_cast<int>(highestNonZero) * 32 + 31 - __clz(imagDigits[highestNonZero]);
            imagOutput->Exponent = imagExponent + currentBit - DesiredBit;
            imagOutput->SetNegative(imagControl[FinalizationNegativeControl] != 0u);
        }

        if constexpr (SharkFloatParams::EnableNewtonRaphson) {
            const uint32_t dzdcRealHighestNonZeroPlusOne =
                dzdcRealControl[FinalizationHighestNonZeroControl];
            if (dzdcRealHighestNonZeroPlusOne == 0u) {
                dzdcRealOutput->Exponent = -100'000'000;
                dzdcRealOutput->SetNegative(false);
            } else {
                const uint32_t highestNonZero = dzdcRealHighestNonZeroPlusOne - 1u;
                const int currentBit =
                    static_cast<int>(highestNonZero) * 32 + 31 - __clz(dzdcRealDigits[highestNonZero]);
                dzdcRealOutput->Exponent = dzdcRealExponent + currentBit - DesiredBit;
                dzdcRealOutput->SetNegative(dzdcRealControl[FinalizationNegativeControl] != 0u);
            }

            const uint32_t dzdcImagHighestNonZeroPlusOne =
                dzdcImagControl[FinalizationHighestNonZeroControl];
            if (dzdcImagHighestNonZeroPlusOne == 0u) {
                dzdcImagOutput->Exponent = -100'000'000;
                dzdcImagOutput->SetNegative(false);
            } else {
                const uint32_t highestNonZero = dzdcImagHighestNonZeroPlusOne - 1u;
                const int currentBit =
                    static_cast<int>(highestNonZero) * 32 + 31 - __clz(dzdcImagDigits[highestNonZero]);
                dzdcImagOutput->Exponent = dzdcImagExponent + currentBit - DesiredBit;
                dzdcImagOutput->SetNegative(dzdcImagControl[FinalizationNegativeControl] != 0u);
            }
        }
    }

    for (uint32_t digit = GridThreadRank(block); digit < ActualDigits; digit += gridSize) {
        const int digitIndex = static_cast<int>(digit);
        {
            const uint32_t highestNonZeroPlusOne = realControl[FinalizationHighestNonZeroControl];
            if (highestNonZeroPlusOne == 0u) {
                realOutput->Digits[digitIndex] = 0u;
            } else {
                const uint32_t highestNonZero = highestNonZeroPlusOne - 1u;
                const int magnitudeLength = static_cast<int>(highestNonZeroPlusOne);
                const int currentBit =
                    static_cast<int>(highestNonZero) * 32 + 31 - __clz(realDigits[highestNonZero]);
                const int shift = currentBit - DesiredBit;
                realOutput->Digits[digitIndex] =
                    shift > 0 ? FunnelShiftRight(realDigits, digitIndex, magnitudeLength, shift)
                              : FunnelShiftLeft(realDigits, digitIndex, magnitudeLength, -shift);
            }
        }

        {
            const uint32_t highestNonZeroPlusOne = imagControl[FinalizationHighestNonZeroControl];
            if (highestNonZeroPlusOne == 0u) {
                imagOutput->Digits[digitIndex] = 0u;
            } else {
                const uint32_t highestNonZero = highestNonZeroPlusOne - 1u;
                const int magnitudeLength = static_cast<int>(highestNonZeroPlusOne);
                const int currentBit =
                    static_cast<int>(highestNonZero) * 32 + 31 - __clz(imagDigits[highestNonZero]);
                const int shift = currentBit - DesiredBit;
                imagOutput->Digits[digitIndex] =
                    shift > 0 ? FunnelShiftRight(imagDigits, digitIndex, magnitudeLength, shift)
                              : FunnelShiftLeft(imagDigits, digitIndex, magnitudeLength, -shift);
            }
        }

        if constexpr (SharkFloatParams::EnableNewtonRaphson) {
            {
                const uint32_t highestNonZeroPlusOne =
                    dzdcRealControl[FinalizationHighestNonZeroControl];
                if (highestNonZeroPlusOne == 0u) {
                    dzdcRealOutput->Digits[digitIndex] = 0u;
                } else {
                    const uint32_t highestNonZero = highestNonZeroPlusOne - 1u;
                    const int magnitudeLength = static_cast<int>(highestNonZeroPlusOne);
                    const int currentBit = static_cast<int>(highestNonZero) * 32 + 31 -
                                           __clz(dzdcRealDigits[highestNonZero]);
                    const int shift = currentBit - DesiredBit;
                    dzdcRealOutput->Digits[digitIndex] =
                        shift > 0 ? FunnelShiftRight(dzdcRealDigits, digitIndex, magnitudeLength, shift)
                                  : FunnelShiftLeft(dzdcRealDigits, digitIndex, magnitudeLength, -shift);
                }
            }

            {
                const uint32_t highestNonZeroPlusOne =
                    dzdcImagControl[FinalizationHighestNonZeroControl];
                if (highestNonZeroPlusOne == 0u) {
                    dzdcImagOutput->Digits[digitIndex] = 0u;
                } else {
                    const uint32_t highestNonZero = highestNonZeroPlusOne - 1u;
                    const int magnitudeLength = static_cast<int>(highestNonZeroPlusOne);
                    const int currentBit = static_cast<int>(highestNonZero) * 32 + 31 -
                                           __clz(dzdcImagDigits[highestNonZero]);
                    const int shift = currentBit - DesiredBit;
                    dzdcImagOutput->Digits[digitIndex] =
                        shift > 0 ? FunnelShiftRight(dzdcImagDigits, digitIndex, magnitudeLength, shift)
                                  : FunnelShiftLeft(dzdcImagDigits, digitIndex, magnitudeLength, -shift);
                }
            }
        }
    }

    if constexpr (HpShark::Debug) {
        grid.sync();
        if (IsLeader(block)) {
            if (realControl[FinalizationHighestNonZeroControl] != 0u) {
                MattsCudaAssert((realOutput->Digits[ActualDigits - 1u] & 0x8000'0000u) != 0u);
            }
            if (imagControl[FinalizationHighestNonZeroControl] != 0u) {
                MattsCudaAssert((imagOutput->Digits[ActualDigits - 1u] & 0x8000'0000u) != 0u);
            }
            if constexpr (SharkFloatParams::EnableNewtonRaphson) {
                if (dzdcRealControl[FinalizationHighestNonZeroControl] != 0u) {
                    MattsCudaAssert((dzdcRealOutput->Digits[ActualDigits - 1u] & 0x8000'0000u) != 0u);
                }
                if (dzdcImagControl[FinalizationHighestNonZeroControl] != 0u) {
                    MattsCudaAssert((dzdcImagOutput->Digits[ActualDigits - 1u] & 0x8000'0000u) != 0u);
                }
            }
        }
    }
}

template <class SharkFloatParams>
__device__ void
BuildReferenceIterationPlan(HpSharkReferenceResults<SharkFloatParams> *combo)
{
    auto &workspace = *combo->Workspace;
    auto &iterationPlan = workspace.IterationPlan;
    const auto &zReal = combo->ZReal;
    const auto &zImag = combo->ZImag;
    const auto &cReal = combo->CReal;
    const auto &cImag = combo->CImag;
    const SharkNTT::Plan basePlan = workspace.Plans[0];
    const uint32_t ignoredPrecisionBits = workspace.IgnoredPrecisionBits;
    const uint32_t bitsPerCoefficient = static_cast<uint32_t>(basePlan.b);

    iterationPlan = {};

    int32_t stateCommonExponent = 0;
    const bool stateBothZero = ResolveAlignedValueExponent(&stateCommonExponent, zReal, zImag);
    const bool stateRealZero = IsZero(zReal);
    const bool stateImagZero = IsZero(zImag);
    const int64_t stateProductExponent64 = static_cast<int64_t>(stateCommonExponent) * 2ll +
                                           2ll * static_cast<int64_t>(ignoredPrecisionBits);
    MattsCudaAssert(stateProductExponent64 >= INT32_MIN && stateProductExponent64 <= INT32_MAX);
    const int32_t stateProductExponent = static_cast<int32_t>(stateProductExponent64);
    const FusedTerm realProductTerm = MakeAlignedProductTerm(stateBothZero, stateProductExponent);
    const FusedTerm imagProductTerm =
        MakeAlignedProductTerm(stateRealZero || stateImagZero, stateProductExponent);
    const FusedTerm realConstantTerm = MakeLinearTerm(cReal, ignoredPrecisionBits);
    const FusedTerm imagConstantTerm = MakeLinearTerm(cImag, ignoredPrecisionBits);
    int32_t realExponent = 0;
    int32_t imagExponent = 0;
    ResolveCommonExponent(&realExponent, realProductTerm, realConstantTerm);
    ResolveCommonExponent(&imagExponent, imagProductTerm, imagConstantTerm);

    int32_t derivativeCommonExponent = 0;
    bool derivativeBothZero = true;
    bool derivativeRealZero = true;
    bool derivativeImagZero = true;
    int32_t derivativeProductExponent = 0;
    FusedTerm dzdcP1Term = MakeAlignedProductTerm(true, 0);
    FusedTerm dzdcP2Term = MakeAlignedProductTerm(true, 0);
    FusedTerm dzdcP3Term = MakeAlignedProductTerm(true, 0);
    FusedTerm dzdcOneTerm = MakeLinearTerm(combo->One, ignoredPrecisionBits);
    int32_t dzdcRealExponent = 0;
    int32_t dzdcImagExponent = 0;
    if constexpr (SharkFloatParams::EnableNewtonRaphson) {
        derivativeBothZero =
            ResolveAlignedValueExponent(&derivativeCommonExponent, combo->DzdcReal, combo->DzdcImag);
        derivativeRealZero = IsZero(combo->DzdcReal);
        derivativeImagZero = IsZero(combo->DzdcImag);
        const int64_t derivativeProductExponent64 = static_cast<int64_t>(stateCommonExponent) +
                                                    static_cast<int64_t>(derivativeCommonExponent) +
                                                    2ll * static_cast<int64_t>(ignoredPrecisionBits);
        MattsCudaAssert(derivativeProductExponent64 >= INT32_MIN &&
                        derivativeProductExponent64 <= INT32_MAX);
        derivativeProductExponent = static_cast<int32_t>(derivativeProductExponent64);
        dzdcP1Term =
            MakeAlignedProductTerm(stateRealZero || derivativeRealZero, derivativeProductExponent);
        dzdcP2Term =
            MakeAlignedProductTerm(stateImagZero || derivativeImagZero, derivativeProductExponent);
        dzdcP3Term =
            MakeAlignedProductTerm(stateBothZero || derivativeBothZero, derivativeProductExponent);
        ResolveCommonExponent(&dzdcRealExponent, dzdcP1Term, dzdcP2Term, dzdcP3Term, dzdcOneTerm);
        ResolveCommonExponent(&dzdcImagExponent, dzdcP1Term, dzdcP2Term, dzdcP3Term);
    }

    const uint64_t stateRealShiftBits =
        stateRealZero ? 0ull : static_cast<uint64_t>(zReal.Exponent - stateCommonExponent);
    const uint64_t stateImagShiftBits =
        stateImagZero ? 0ull : static_cast<uint64_t>(zImag.Exponent - stateCommonExponent);
    const uint64_t stateRealCoefficientShift = stateRealShiftBits / bitsPerCoefficient;
    const uint64_t stateImagCoefficientShift = stateImagShiftBits / bitsPerCoefficient;
    const uint64_t stateRealResidualBitShift = stateRealShiftBits % bitsPerCoefficient;
    const uint64_t stateImagResidualBitShift = stateImagShiftBits % bitsPerCoefficient;
    const uint64_t stateRealInputCoefficients =
        static_cast<uint64_t>(basePlan.L) + (stateRealResidualBitShift != 0ull ? 1ull : 0ull);
    const uint64_t stateImagInputCoefficients =
        static_cast<uint64_t>(basePlan.L) + (stateImagResidualBitShift != 0ull ? 1ull : 0ull);
    const uint64_t stateRealLastCoefficient =
        stateRealCoefficientShift + stateRealInputCoefficients - 1ull;
    const uint64_t stateImagLastCoefficient =
        stateImagCoefficientShift + stateImagInputCoefficients - 1ull;
    const uint64_t stateMaxLastCoefficient = stateRealLastCoefficient > stateImagLastCoefficient
                                                 ? stateRealLastCoefficient
                                                 : stateImagLastCoefficient;
    const uint64_t realRequiredCoefficients =
        stateBothZero ? 0ull : 2ull * stateMaxLastCoefficient + 1ull;
    const uint64_t imagRequiredCoefficients =
        (stateRealZero || stateImagZero) ? 0ull
                                         : stateRealLastCoefficient + stateImagLastCoefficient + 1ull;
    uint64_t requiredCoefficients = realRequiredCoefficients > imagRequiredCoefficients
                                        ? realRequiredCoefficients
                                        : imagRequiredCoefficients;

    uint64_t derivativeRealShiftBits = 0;
    uint64_t derivativeImagShiftBits = 0;
    uint64_t derivativeRealCoefficientShift = 0;
    uint64_t derivativeImagCoefficientShift = 0;
    uint64_t derivativeRealResidualBitShift = 0;
    uint64_t derivativeImagResidualBitShift = 0;
    uint64_t derivativeRealInputCoefficients = 0;
    uint64_t derivativeImagInputCoefficients = 0;
    uint64_t derivativeRealLastCoefficient = 0;
    uint64_t derivativeImagLastCoefficient = 0;
    uint64_t derivativeMaxLastCoefficient = 0;
    uint64_t derivativeP1RequiredCoefficients = 0;
    uint64_t derivativeP2RequiredCoefficients = 0;
    uint64_t derivativeP3RequiredCoefficients = 0;
    if constexpr (SharkFloatParams::EnableNewtonRaphson) {
        derivativeRealShiftBits =
            derivativeRealZero
                ? 0ull
                : static_cast<uint64_t>(combo->DzdcReal.Exponent - derivativeCommonExponent);
        derivativeImagShiftBits =
            derivativeImagZero
                ? 0ull
                : static_cast<uint64_t>(combo->DzdcImag.Exponent - derivativeCommonExponent);
        derivativeRealCoefficientShift = derivativeRealShiftBits / bitsPerCoefficient;
        derivativeImagCoefficientShift = derivativeImagShiftBits / bitsPerCoefficient;
        derivativeRealResidualBitShift = derivativeRealShiftBits % bitsPerCoefficient;
        derivativeImagResidualBitShift = derivativeImagShiftBits % bitsPerCoefficient;
        derivativeRealInputCoefficients =
            static_cast<uint64_t>(basePlan.L) + (derivativeRealResidualBitShift != 0ull ? 1ull : 0ull);
        derivativeImagInputCoefficients =
            static_cast<uint64_t>(basePlan.L) + (derivativeImagResidualBitShift != 0ull ? 1ull : 0ull);
        derivativeRealLastCoefficient =
            derivativeRealCoefficientShift + derivativeRealInputCoefficients - 1ull;
        derivativeImagLastCoefficient =
            derivativeImagCoefficientShift + derivativeImagInputCoefficients - 1ull;
        derivativeMaxLastCoefficient = derivativeRealLastCoefficient > derivativeImagLastCoefficient
                                           ? derivativeRealLastCoefficient
                                           : derivativeImagLastCoefficient;
        derivativeP1RequiredCoefficients =
            dzdcP1Term.IsZero ? 0ull : stateRealLastCoefficient + derivativeRealLastCoefficient + 1ull;
        derivativeP2RequiredCoefficients =
            dzdcP2Term.IsZero ? 0ull : stateImagLastCoefficient + derivativeImagLastCoefficient + 1ull;
        derivativeP3RequiredCoefficients =
            dzdcP3Term.IsZero ? 0ull : stateMaxLastCoefficient + derivativeMaxLastCoefficient + 1ull;
        requiredCoefficients = requiredCoefficients > derivativeP1RequiredCoefficients
                                   ? requiredCoefficients
                                   : derivativeP1RequiredCoefficients;
        requiredCoefficients = requiredCoefficients > derivativeP2RequiredCoefficients
                                   ? requiredCoefficients
                                   : derivativeP2RequiredCoefficients;
        requiredCoefficients = requiredCoefficients > derivativeP3RequiredCoefficients
                                   ? requiredCoefficients
                                   : derivativeP3RequiredCoefficients;
    }

    uint32_t flags = 0u;
    if (!realProductTerm.IsZero)
        flags |= HpSharkReferencePlanRealProduct;
    if (!imagProductTerm.IsZero)
        flags |= HpSharkReferencePlanImagProduct;
    if constexpr (SharkFloatParams::EnableNewtonRaphson) {
        if (!dzdcP1Term.IsZero)
            flags |= HpSharkReferencePlanDzdcP1;
        if (!dzdcP2Term.IsZero)
            flags |= HpSharkReferencePlanDzdcP2;
        if (!dzdcP3Term.IsZero)
            flags |= HpSharkReferencePlanDzdcP3;
        if (!dzdcOneTerm.IsZero)
            flags |= HpSharkReferencePlanDzdcOne;
    }
    if (!realConstantTerm.IsZero)
        flags |= HpSharkReferencePlanRealLinear;
    if (!imagConstantTerm.IsZero)
        flags |= HpSharkReferencePlanImagLinear;
    iterationPlan.Flags = flags;
    iterationPlan.RealExponent = realExponent;
    iterationPlan.ImagExponent = imagExponent;
    iterationPlan.DzdcRealExponent = dzdcRealExponent;
    iterationPlan.DzdcImagExponent = dzdcImagExponent;
    iterationPlan.RealProductBitOffset =
        realProductTerm.IsZero ? 0ull : static_cast<uint64_t>(realProductTerm.Exponent - realExponent);
    iterationPlan.ImagProductBitOffset =
        imagProductTerm.IsZero ? 0ull : static_cast<uint64_t>(imagProductTerm.Exponent - imagExponent);
    iterationPlan.DzdcRealProductBitOffset =
        dzdcP1Term.IsZero && dzdcP2Term.IsZero && dzdcP3Term.IsZero
            ? 0ull
            : static_cast<uint64_t>(derivativeProductExponent - dzdcRealExponent);
    iterationPlan.DzdcImagProductBitOffset =
        dzdcP1Term.IsZero && dzdcP2Term.IsZero && dzdcP3Term.IsZero
            ? 0ull
            : static_cast<uint64_t>(derivativeProductExponent - dzdcImagExponent);
    iterationPlan.RealLinearBitOffset =
        realConstantTerm.IsZero ? 0ull : static_cast<uint64_t>(realConstantTerm.Exponent - realExponent);
    iterationPlan.ImagLinearBitOffset =
        imagConstantTerm.IsZero ? 0ull : static_cast<uint64_t>(imagConstantTerm.Exponent - imagExponent);
    iterationPlan.DzdcRealLinearBitOffset =
        dzdcOneTerm.IsZero ? 0ull : static_cast<uint64_t>(dzdcOneTerm.Exponent - dzdcRealExponent);

    const bool hasLinearTerm = !realConstantTerm.IsZero || !imagConstantTerm.IsZero ||
                               (SharkFloatParams::EnableNewtonRaphson && !dzdcOneTerm.IsZero);
    if (requiredCoefficients == 0ull) {
        iterationPlan.Kind =
            static_cast<uint32_t>(hasLinearTerm ? HpSharkReferenceIterationKind::LinearOnly
                                                : HpSharkReferenceIterationKind::Zero);
        iterationPlan.LimbCount = hasLinearTerm ? SharkFloatParams::GlobalNumUint32 + 2u : 0u;
        return;
    }

    const uint64_t requiredN = CeilPowerOfTwo(requiredCoefficients);
    if (requiredN > HpSharkReferenceWorkspace<SharkFloatParams>::MaxFusedN) {
        combo->PeriodicityStatus = PeriodicityResult::Unknown;
        return;
    }

    using Workspace = HpSharkReferenceWorkspace<SharkFloatParams>;
    const uint32_t activeN = requiredN < workspace.ActiveMinFusedN ? workspace.ActiveMinFusedN
                                                                   : static_cast<uint32_t>(requiredN);
    MattsCudaAssert(activeN >= workspace.ActiveMinFusedN);
    MattsCudaAssert(requiredCoefficients <= activeN);
    const uint32_t planSlot = CountTrailingZeros(activeN) - Workspace::MinFusedStages;
    MattsCudaAssert(planSlot < Workspace::PlanCacheEntryCount);
    MattsCudaAssert((workspace.ValidPlanMask & (1u << planSlot)) != 0u);
    const SharkNTT::Plan &plan = workspace.Plans[planSlot];
    MattsCudaAssert(static_cast<uint32_t>(plan.N) == activeN);

    const uint64_t linearBits =
        static_cast<uint64_t>(SharkFloatParams::GlobalNumUint32) * 32ull - ignoredPrecisionBits;
    uint64_t outputBits =
        iterationPlan.RealProductBitOffset + realRequiredCoefficients * bitsPerCoefficient;
    const uint64_t realLinearBits = iterationPlan.RealLinearBitOffset +
                                    ((flags & HpSharkReferencePlanRealLinear) != 0u ? linearBits : 0ull);
    outputBits = outputBits > realLinearBits ? outputBits : realLinearBits;
    const uint64_t imagProductBits =
        iterationPlan.ImagProductBitOffset + imagRequiredCoefficients * bitsPerCoefficient;
    const uint64_t imagLinearBits = iterationPlan.ImagLinearBitOffset +
                                    ((flags & HpSharkReferencePlanImagLinear) != 0u ? linearBits : 0ull);
    outputBits = outputBits > imagProductBits ? outputBits : imagProductBits;
    outputBits = outputBits > imagLinearBits ? outputBits : imagLinearBits;

    uint64_t derivativeRequiredCoefficients = 0ull;
    if constexpr (SharkFloatParams::EnableNewtonRaphson) {
        derivativeRequiredCoefficients =
            derivativeP1RequiredCoefficients > derivativeP2RequiredCoefficients
                ? (derivativeP1RequiredCoefficients > derivativeP3RequiredCoefficients
                       ? derivativeP1RequiredCoefficients
                       : derivativeP3RequiredCoefficients)
                : (derivativeP2RequiredCoefficients > derivativeP3RequiredCoefficients
                       ? derivativeP2RequiredCoefficients
                       : derivativeP3RequiredCoefficients);
        const uint64_t dzdcRealProductBits =
            iterationPlan.DzdcRealProductBitOffset + derivativeRequiredCoefficients * bitsPerCoefficient;
        const uint64_t dzdcRealLinearBits =
            iterationPlan.DzdcRealLinearBitOffset +
            ((flags & HpSharkReferencePlanDzdcOne) != 0u ? linearBits : 0ull);
        outputBits = outputBits > dzdcRealProductBits ? outputBits : dzdcRealProductBits;
        outputBits = outputBits > dzdcRealLinearBits ? outputBits : dzdcRealLinearBits;
        const uint64_t dzdcImagProductBits =
            iterationPlan.DzdcImagProductBitOffset + derivativeRequiredCoefficients * bitsPerCoefficient;
        outputBits = outputBits > dzdcImagProductBits ? outputBits : dzdcImagProductBits;
    }

    const uint64_t limbCount64 = (outputBits + 31ull) / 32ull + 2ull;
    MattsCudaAssert(limbCount64 <= workspace.ActiveMaxFusedLimbs);
    if (limbCount64 > workspace.ActiveMaxFusedLimbs) {
        combo->PeriodicityStatus = PeriodicityResult::Unknown;
        return;
    }

    iterationPlan.Kind = static_cast<uint32_t>(HpSharkReferenceIterationKind::Ntt);
    iterationPlan.PlanSlot = planSlot;
    iterationPlan.ActiveN = activeN;
    iterationPlan.LimbCount = static_cast<uint32_t>(limbCount64);
    iterationPlan.ZRealCoefficientShift = static_cast<uint32_t>(stateRealCoefficientShift);
    iterationPlan.ZImagCoefficientShift = static_cast<uint32_t>(stateImagCoefficientShift);
    iterationPlan.DzdcRealCoefficientShift = static_cast<uint32_t>(derivativeRealCoefficientShift);
    iterationPlan.DzdcImagCoefficientShift = static_cast<uint32_t>(derivativeImagCoefficientShift);
    iterationPlan.ZRealResidualBitShift = static_cast<uint32_t>(stateRealResidualBitShift);
    iterationPlan.ZImagResidualBitShift = static_cast<uint32_t>(stateImagResidualBitShift);
    iterationPlan.DzdcRealResidualBitShift = static_cast<uint32_t>(derivativeRealResidualBitShift);
    iterationPlan.DzdcImagResidualBitShift = static_cast<uint32_t>(derivativeImagResidualBitShift);
    MattsCudaAssert(realRequiredCoefficients <= activeN);
    MattsCudaAssert(imagRequiredCoefficients <= activeN);
    iterationPlan.RealCoefficientCount =
        realProductTerm.IsZero ? 0u : static_cast<uint32_t>(realRequiredCoefficients);
    iterationPlan.ImagCoefficientCount =
        imagProductTerm.IsZero ? 0u : static_cast<uint32_t>(imagRequiredCoefficients);
    if constexpr (SharkFloatParams::EnableNewtonRaphson) {
        MattsCudaAssert(derivativeRequiredCoefficients <= activeN);
        const uint32_t derivativeCoefficientCount =
            dzdcP1Term.IsZero && dzdcP2Term.IsZero && dzdcP3Term.IsZero
                ? 0u
                : static_cast<uint32_t>(derivativeRequiredCoefficients);
        iterationPlan.DzdcRealCoefficientCount = derivativeCoefficientCount;
        iterationPlan.DzdcImagCoefficientCount = derivativeCoefficientCount;
    }
}

template <class SharkFloatParams>
__device__ SharkForceInlineReleaseOnly void
ExecuteReferenceIterationNonNrNtt(cooperative_groups::grid_group &grid,
                                  cooperative_groups::thread_block &block,
                                  uint64_t *sharedData,
                                  DebugGlobalCount<SharkFloatParams> *debugCombo,
                                  DebugState<SharkFloatParams> *debugStates,
                                  uint64_t *carryPrefixShared,
                                  HpSharkReferenceResults<SharkFloatParams> *combo)
{
    auto &workspace = *combo->Workspace;
    const auto &iterationPlan = workspace.IterationPlan;
    const auto &zReal = combo->ZReal;
    const auto &zImag = combo->ZImag;
    const auto &cReal = combo->CReal;
    const auto &cImag = combo->CImag;
    const uint32_t flags = iterationPlan.Flags;
    const bool realProductEnabled = (flags & HpSharkReferencePlanRealProduct) != 0u;
    const bool imagProductEnabled = (flags & HpSharkReferencePlanImagProduct) != 0u;
    const bool realLinearEnabled = (flags & HpSharkReferencePlanRealLinear) != 0u;
    const bool imagLinearEnabled = (flags & HpSharkReferencePlanImagLinear) != 0u;
    const uint32_t limbCount = iterationPlan.LimbCount;
    const uint32_t carryPrefixCapacity = workspace.ActiveMaxFusedLimbs;
    auto *carryPrefixDescriptors =
        reinterpret_cast<HpSharkReferencePackedCarryPrefixDescriptor *>(workspace.ZReal);
    const uint32_t carryPrefixDescriptorWords =
        (workspace.ActiveMaxCarryPrefixParts * sizeof(HpSharkReferencePackedCarryPrefixDescriptor) +
         sizeof(uint64_t) - 1u) /
        sizeof(uint64_t);
    uint32_t *tempRealDigits =
        reinterpret_cast<uint32_t *>(workspace.ZReal + carryPrefixDescriptorWords);
    uint32_t *tempRealControl =
        reinterpret_cast<uint32_t *>(workspace.ZReal + carryPrefixDescriptorWords + carryPrefixCapacity);
    uint32_t *tempImagDigits =
        reinterpret_cast<uint32_t *>(workspace.ZImag + carryPrefixDescriptorWords);
    uint32_t *tempImagControl =
        reinterpret_cast<uint32_t *>(workspace.ZImag + carryPrefixDescriptorWords + carryPrefixCapacity);

    const uint32_t activeN = iterationPlan.ActiveN;
    MattsCudaAssert(iterationPlan.PlanSlot <
                    HpSharkReferenceWorkspace<SharkFloatParams>::PlanCacheEntryCount);
    const SharkNTT::Plan &plan = workspace.Plans[iterationPlan.PlanSlot];
    SharkNTT::RootTables &roots = workspace.PlanRoots[iterationPlan.PlanSlot];
    MattsCudaAssert(static_cast<uint32_t>(plan.N) == activeN);
    MattsCudaAssert(static_cast<uint32_t>(roots.N) == activeN);
    const uint32_t stageCount = static_cast<uint32_t>(plan.stages);
    MattsCudaAssert(static_cast<uint32_t>(roots.stages) == stageCount);
    const uint32_t transformSize = static_cast<uint32_t>(plan.N);
    const uint32_t tileSizeLog2 = NTT::SelectTileSizeLog2(transformSize, 12u);
    const bool useFusedInversePointwise = !HpShark::DebugChecksums;

    if (gridDim.x == 1u) {
        PackForwardOne(plan,
                       debugCombo,
                       roots.InputScaleR,
                       &zReal,
                       workspace.ZReal,
                       workspace.IgnoredPrecisionBits,
                       iterationPlan.ZRealCoefficientShift,
                       iterationPlan.ZRealResidualBitShift);
        PackForwardOne(plan,
                       debugCombo,
                       roots.InputScaleR,
                       &zImag,
                       workspace.ZImag,
                       workspace.IgnoredPrecisionBits,
                       iterationPlan.ZImagCoefficientShift,
                       iterationPlan.ZImagResidualBitShift);
    } else if ((blockIdx.x & 1u) == 0u) {
        PackForwardOne(plan,
                       debugCombo,
                       roots.InputScaleR,
                       &zReal,
                       workspace.ZReal,
                       workspace.IgnoredPrecisionBits,
                       iterationPlan.ZRealCoefficientShift,
                       iterationPlan.ZRealResidualBitShift);
    } else {
        PackForwardOne(plan,
                       debugCombo,
                       roots.InputScaleR,
                       &zImag,
                       workspace.ZImag,
                       workspace.IgnoredPrecisionBits,
                       iterationPlan.ZImagCoefficientShift,
                       iterationPlan.ZImagResidualBitShift);
    }
    grid.sync();

    if constexpr (HpShark::DebugChecksums) {
        StoreReferenceDebugStateBatchAfterSync<SharkFloatParams>(debugStates,
                                                                 grid,
                                                                 block,
                                                                 DebugStatePurpose::Z0XX,
                                                                 workspace.ZReal,
                                                                 DebugStatePurpose::Z0YY,
                                                                 workspace.ZImag,
                                                                 transformSize);
    }

    uint32_t forwardStage = stageCount;
    while (forwardStage > tileSizeLog2 + 1u) {
        if (gridDim.x == 1u) {
            NTT::ForwardRadix4One(
                debugCombo, workspace.ZReal, roots.stage_twiddles_fwd, transformSize, forwardStage);
            NTT::ForwardRadix4One(
                debugCombo, workspace.ZImag, roots.stage_twiddles_fwd, transformSize, forwardStage);
        } else if ((blockIdx.x & 1u) == 0u) {
            NTT::ForwardRadix4One(
                debugCombo, workspace.ZReal, roots.stage_twiddles_fwd, transformSize, forwardStage);
        } else {
            NTT::ForwardRadix4One(
                debugCombo, workspace.ZImag, roots.stage_twiddles_fwd, transformSize, forwardStage);
        }
        grid.sync();
        forwardStage -= 2u;
    }
    if (forwardStage > tileSizeLog2) {
        if (gridDim.x == 1u) {
            NTT::ForwardRadix2One(
                debugCombo, workspace.ZReal, roots.stage_twiddles_fwd, transformSize, forwardStage);
            NTT::ForwardRadix2One(
                debugCombo, workspace.ZImag, roots.stage_twiddles_fwd, transformSize, forwardStage);
        } else if ((blockIdx.x & 1u) == 0u) {
            NTT::ForwardRadix2One(
                debugCombo, workspace.ZReal, roots.stage_twiddles_fwd, transformSize, forwardStage);
        } else {
            NTT::ForwardRadix2One(
                debugCombo, workspace.ZImag, roots.stage_twiddles_fwd, transformSize, forwardStage);
        }
        grid.sync();
    }
    if (gridDim.x == 1u) {
        NTT::ForwardTileOneSelected(debugCombo,
                                    sharedData,
                                    workspace.ZReal,
                                    roots.stage_twiddles_fwd,
                                    transformSize,
                                    stageCount,
                                    tileSizeLog2);
        block.sync();
        NTT::ForwardTileOneSelected(debugCombo,
                                    sharedData,
                                    workspace.ZImag,
                                    roots.stage_twiddles_fwd,
                                    transformSize,
                                    stageCount,
                                    tileSizeLog2);
    } else if ((blockIdx.x & 1u) == 0u) {
        NTT::ForwardTileOneSelected(debugCombo,
                                    sharedData,
                                    workspace.ZReal,
                                    roots.stage_twiddles_fwd,
                                    transformSize,
                                    stageCount,
                                    tileSizeLog2);
    } else {
        NTT::ForwardTileOneSelected(debugCombo,
                                    sharedData,
                                    workspace.ZImag,
                                    roots.stage_twiddles_fwd,
                                    transformSize,
                                    stageCount,
                                    tileSizeLog2);
    }
    grid.sync();

    if constexpr (HpShark::DebugChecksums) {
        StoreReferenceDebugStateBatchAfterSync<SharkFloatParams>(debugStates,
                                                                 grid,
                                                                 block,
                                                                 DebugStatePurpose::Z2XX,
                                                                 workspace.ZReal,
                                                                 DebugStatePurpose::Z2YY,
                                                                 workspace.ZImag,
                                                                 transformSize);
    }

    // DIT inverse stages run from low to high; the adaptive tile owns the local stages.
    if (!useFusedInversePointwise) {
        if (gridDim.x == 1u) {
            if (realProductEnabled) {
                PointwiseOrbitRealOne(
                    debugCombo, workspace.ZReal, workspace.ZImag, workspace.RealOutput, transformSize);
            } else {
                PointwiseZeroOne(workspace.RealOutput, transformSize);
            }
            if (imagProductEnabled) {
                PointwiseOrbitImagOne(
                    debugCombo, workspace.ZReal, workspace.ZImag, workspace.ImagOutput, transformSize);
            } else {
                PointwiseZeroOne(workspace.ImagOutput, transformSize);
            }
        } else if ((blockIdx.x & 1u) == 0u) {
            if (realProductEnabled) {
                PointwiseOrbitRealOne(
                    debugCombo, workspace.ZReal, workspace.ZImag, workspace.RealOutput, transformSize);
            } else {
                PointwiseZeroOne(workspace.RealOutput, transformSize);
            }
        } else {
            if (imagProductEnabled) {
                PointwiseOrbitImagOne(
                    debugCombo, workspace.ZReal, workspace.ZImag, workspace.ImagOutput, transformSize);
            } else {
                PointwiseZeroOne(workspace.ImagOutput, transformSize);
            }
        }
        grid.sync();

        if constexpr (HpShark::DebugChecksums) {
            StoreReferenceDebugStateBatchAfterSync<SharkFloatParams>(debugStates,
                                                                     grid,
                                                                     block,
                                                                     DebugStatePurpose::Z2_Perm1,
                                                                     workspace.RealOutput,
                                                                     DebugStatePurpose::Z2_Perm2,
                                                                     workspace.ImagOutput,
                                                                     transformSize);
        }
        InitializeCarryPrefixTransformsDLB<SharkFloatParams>(
            grid, block, limbCount, carryPrefixCapacity, carryPrefixDescriptors, carryPrefixShared);

        if (gridDim.x == 1u) {
            NTT::InverseTileOneSelected<SharkFloatParams, NTT::InverseTileOneLoad::Copy>(
                debugCombo,
                sharedData,
                workspace.RealOutput,
                nullptr,
                workspace.RealOutput,
                false,
                roots.stage_twiddles_inv,
                transformSize,
                stageCount,
                tileSizeLog2);
            block.sync();
            NTT::InverseTileOneSelected<SharkFloatParams, NTT::InverseTileOneLoad::Copy>(
                debugCombo,
                sharedData,
                workspace.ImagOutput,
                nullptr,
                workspace.ImagOutput,
                false,
                roots.stage_twiddles_inv,
                transformSize,
                stageCount,
                tileSizeLog2);
        } else if ((blockIdx.x & 1u) == 0u) {
            NTT::InverseTileOneSelected<SharkFloatParams, NTT::InverseTileOneLoad::Copy>(
                debugCombo,
                sharedData,
                workspace.RealOutput,
                nullptr,
                workspace.RealOutput,
                false,
                roots.stage_twiddles_inv,
                transformSize,
                stageCount,
                tileSizeLog2);
        } else {
            NTT::InverseTileOneSelected<SharkFloatParams, NTT::InverseTileOneLoad::Copy>(
                debugCombo,
                sharedData,
                workspace.ImagOutput,
                nullptr,
                workspace.ImagOutput,
                false,
                roots.stage_twiddles_inv,
                transformSize,
                stageCount,
                tileSizeLog2);
        }
        grid.sync();
    } else {
        if (gridDim.x == 1u) {
            NTT::InverseTileOneSelected<SharkFloatParams, NTT::InverseTileOneLoad::OrbitReal>(
                debugCombo,
                sharedData,
                workspace.ZReal,
                workspace.ZImag,
                workspace.RealOutput,
                realProductEnabled,
                roots.stage_twiddles_inv,
                transformSize,
                stageCount,
                tileSizeLog2);
            block.sync();
            NTT::InverseTileOneSelected<SharkFloatParams, NTT::InverseTileOneLoad::OrbitImag>(
                debugCombo,
                sharedData,
                workspace.ZReal,
                workspace.ZImag,
                workspace.ImagOutput,
                imagProductEnabled,
                roots.stage_twiddles_inv,
                transformSize,
                stageCount,
                tileSizeLog2);
        } else if ((blockIdx.x & 1u) == 0u) {
            NTT::InverseTileOneSelected<SharkFloatParams, NTT::InverseTileOneLoad::OrbitReal>(
                debugCombo,
                sharedData,
                workspace.ZReal,
                workspace.ZImag,
                workspace.RealOutput,
                realProductEnabled,
                roots.stage_twiddles_inv,
                transformSize,
                stageCount,
                tileSizeLog2);
        } else {
            NTT::InverseTileOneSelected<SharkFloatParams, NTT::InverseTileOneLoad::OrbitImag>(
                debugCombo,
                sharedData,
                workspace.ZReal,
                workspace.ZImag,
                workspace.ImagOutput,
                imagProductEnabled,
                roots.stage_twiddles_inv,
                transformSize,
                stageCount,
                tileSizeLog2);
        }
        grid.sync();
        InitializeCarryPrefixTransformsDLB<SharkFloatParams>(
            grid, block, limbCount, carryPrefixCapacity, carryPrefixDescriptors, carryPrefixShared);
        if (stageCount < tileSizeLog2 + 1u)
            grid.sync();
    }

    uint32_t inverseStage = tileSizeLog2 + 1u;
    while (inverseStage + 1u <= stageCount) {
        if (gridDim.x == 1u) {
            NTT::InverseRadix4One(debugCombo,
                                  workspace.RealOutput,
                                  roots.stage_twiddles_inv,
                                  transformSize,
                                  inverseStage + 1u);
            NTT::InverseRadix4One(debugCombo,
                                  workspace.ImagOutput,
                                  roots.stage_twiddles_inv,
                                  transformSize,
                                  inverseStage + 1u);
        } else if ((blockIdx.x & 1u) == 0u) {
            NTT::InverseRadix4One(debugCombo,
                                  workspace.RealOutput,
                                  roots.stage_twiddles_inv,
                                  transformSize,
                                  inverseStage + 1u);
        } else {
            NTT::InverseRadix4One(debugCombo,
                                  workspace.ImagOutput,
                                  roots.stage_twiddles_inv,
                                  transformSize,
                                  inverseStage + 1u);
        }
        grid.sync();
        inverseStage += 2u;
    }
    if (inverseStage <= stageCount) {
        if (gridDim.x == 1u) {
            NTT::InverseRadix2One(
                debugCombo, workspace.RealOutput, roots.stage_twiddles_inv, transformSize, inverseStage);
            NTT::InverseRadix2One(
                debugCombo, workspace.ImagOutput, roots.stage_twiddles_inv, transformSize, inverseStage);
        } else if ((blockIdx.x & 1u) == 0u) {
            NTT::InverseRadix2One(
                debugCombo, workspace.RealOutput, roots.stage_twiddles_inv, transformSize, inverseStage);
        } else {
            NTT::InverseRadix2One(
                debugCombo, workspace.ImagOutput, roots.stage_twiddles_inv, transformSize, inverseStage);
        }
        grid.sync();
    }

    if constexpr (HpShark::DebugChecksums) {
        StoreReferenceDebugStateBatchAfterSync<SharkFloatParams>(debugStates,
                                                                 grid,
                                                                 block,
                                                                 DebugStatePurpose::Z2_Perm4,
                                                                 workspace.RealOutput,
                                                                 DebugStatePurpose::Z2_Perm5,
                                                                 workspace.ImagOutput,
                                                                 transformSize);
    }
    if (plan.b == 16) {
        RunAlignedB16CarryPrefix<SharkFloatParams>(workspace, combo, carryPrefixShared);
    } else {
        if (realLinearEnabled) {
            UnpackAlignedResiduesToSignedLimbsOneWithLinear<SharkFloatParams>(
                workspace.RealOutput,
                plan,
                iterationPlan.RealCoefficientCount,
                iterationPlan.RealProductBitOffset,
                &cReal,
                workspace.IgnoredPrecisionBits,
                iterationPlan.RealLinearBitOffset,
                workspace.RealLimbs,
                limbCount);
        } else {
            UnpackAlignedResiduesToSignedLimbsOneNoLinear<SharkFloatParams>(
                workspace.RealOutput,
                plan,
                iterationPlan.RealCoefficientCount,
                iterationPlan.RealProductBitOffset,
                workspace.RealLimbs,
                limbCount);
        }
        if (imagLinearEnabled) {
            UnpackAlignedResiduesToSignedLimbsOneWithLinear<SharkFloatParams>(
                workspace.ImagOutput,
                plan,
                iterationPlan.ImagCoefficientCount,
                iterationPlan.ImagProductBitOffset,
                &cImag,
                workspace.IgnoredPrecisionBits,
                iterationPlan.ImagLinearBitOffset,
                workspace.ImagLimbs,
                limbCount);
        } else {
            UnpackAlignedResiduesToSignedLimbsOneNoLinear<SharkFloatParams>(
                workspace.ImagOutput,
                plan,
                iterationPlan.ImagCoefficientCount,
                iterationPlan.ImagProductBitOffset,
                workspace.ImagLimbs,
                limbCount);
        }
        grid.sync();
    }
    if constexpr (HpShark::DebugChecksums) {
        StoreReferenceDebugStateBatchAfterSync<SharkFloatParams>(
            debugStates,
            grid,
            block,
            DebugStatePurpose::UnpackXX,
            reinterpret_cast<const uint64_t *>(workspace.RealLimbs),
            DebugStatePurpose::UnpackYY,
            reinterpret_cast<const uint64_t *>(workspace.ImagLimbs),
            limbCount);
    }
    FinalizeSignedStream<SharkFloatParams>(debugStates, carryPrefixShared, workspace, combo);
    StoreReferenceDebugValueBatch<SharkFloatParams>(debugStates,
                                                    grid,
                                                    block,
                                                    DebugStatePurpose::Result_Add1,
                                                    combo->ZReal,
                                                    DebugStatePurpose::Result_Add2,
                                                    combo->ZImag);
}

template <class SharkFloatParams>
__device__ SharkForceInlineReleaseOnly void
ExecuteReferenceIterationNrNtt(cooperative_groups::grid_group &grid,
                               cooperative_groups::thread_block &block,
                               uint64_t *sharedData,
                               DebugGlobalCount<SharkFloatParams> *debugCombo,
                               DebugState<SharkFloatParams> *debugStates,
                               uint64_t *carryPrefixShared,
                               HpSharkReferenceResults<SharkFloatParams> *combo)
{
    auto &workspace = *combo->Workspace;
    const auto &iterationPlan = workspace.IterationPlan;
    const auto &zReal = combo->ZReal;
    const auto &zImag = combo->ZImag;
    const auto &cReal = combo->CReal;
    const auto &cImag = combo->CImag;
    const uint32_t flags = iterationPlan.Flags;
    const bool realProductEnabled = (flags & HpSharkReferencePlanRealProduct) != 0u;
    const bool imagProductEnabled = (flags & HpSharkReferencePlanImagProduct) != 0u;
    const bool dzdcP1Enabled = (flags & HpSharkReferencePlanDzdcP1) != 0u;
    const bool dzdcP2Enabled = (flags & HpSharkReferencePlanDzdcP2) != 0u;
    const bool dzdcP3Enabled = (flags & HpSharkReferencePlanDzdcP3) != 0u;
    const bool dzdcOneEnabled = (flags & HpSharkReferencePlanDzdcOne) != 0u;
    const bool realLinearEnabled = (flags & HpSharkReferencePlanRealLinear) != 0u;
    const bool imagLinearEnabled = (flags & HpSharkReferencePlanImagLinear) != 0u;
    const bool derivativeEnabled = dzdcP1Enabled || dzdcP2Enabled || dzdcP3Enabled;
    const uint32_t limbCount = iterationPlan.LimbCount;
    const uint32_t carryPrefixCapacity = workspace.ActiveMaxFusedLimbs;
    auto *carryPrefixDescriptors =
        reinterpret_cast<HpSharkReferencePackedCarryPrefixDescriptor *>(workspace.ZReal);
    const uint32_t carryPrefixDescriptorWords =
        (workspace.ActiveMaxCarryPrefixParts * sizeof(HpSharkReferencePackedCarryPrefixDescriptor) +
         sizeof(uint64_t) - 1u) /
        sizeof(uint64_t);
    uint32_t *tempRealDigits =
        reinterpret_cast<uint32_t *>(workspace.ZReal + carryPrefixDescriptorWords);
    uint32_t *tempRealControl =
        reinterpret_cast<uint32_t *>(workspace.ZReal + carryPrefixDescriptorWords + carryPrefixCapacity);
    uint32_t *tempImagDigits =
        reinterpret_cast<uint32_t *>(workspace.ZImag + carryPrefixDescriptorWords);
    uint32_t *tempImagControl =
        reinterpret_cast<uint32_t *>(workspace.ZImag + carryPrefixDescriptorWords + carryPrefixCapacity);
    uint32_t *tempDzdcRealDigits =
        reinterpret_cast<uint32_t *>(workspace.DzdcReal + carryPrefixDescriptorWords);
    uint32_t *tempDzdcRealControl = reinterpret_cast<uint32_t *>(
        workspace.DzdcReal + carryPrefixDescriptorWords + carryPrefixCapacity);
    uint32_t *tempDzdcImagDigits =
        reinterpret_cast<uint32_t *>(workspace.DzdcImag + carryPrefixDescriptorWords);
    uint32_t *tempDzdcImagControl = reinterpret_cast<uint32_t *>(
        workspace.DzdcImag + carryPrefixDescriptorWords + carryPrefixCapacity);

    const uint32_t activeN = iterationPlan.ActiveN;
    MattsCudaAssert(iterationPlan.PlanSlot <
                    HpSharkReferenceWorkspace<SharkFloatParams>::PlanCacheEntryCount);
    const SharkNTT::Plan &plan = workspace.Plans[iterationPlan.PlanSlot];
    SharkNTT::RootTables &roots = workspace.PlanRoots[iterationPlan.PlanSlot];
    MattsCudaAssert(static_cast<uint32_t>(plan.N) == activeN);
    MattsCudaAssert(static_cast<uint32_t>(roots.N) == activeN);
    const uint32_t stageCount = static_cast<uint32_t>(plan.stages);
    MattsCudaAssert(static_cast<uint32_t>(roots.stages) == stageCount);
    const uint32_t transformSize = static_cast<uint32_t>(plan.N);
    const uint32_t tileSizeLog2 = NTT::SelectTileSizeLog2(transformSize, 11u);
    const bool useFusedInversePointwise = !HpShark::DebugChecksums;

    if (gridDim.x == 1u) {
        PackForwardOne(plan,
                       debugCombo,
                       roots.InputScaleR,
                       &zReal,
                       workspace.ZReal,
                       workspace.IgnoredPrecisionBits,
                       iterationPlan.ZRealCoefficientShift,
                       iterationPlan.ZRealResidualBitShift);
        PackForwardOne(plan,
                       debugCombo,
                       roots.InputScaleR,
                       &zImag,
                       workspace.ZImag,
                       workspace.IgnoredPrecisionBits,
                       iterationPlan.ZImagCoefficientShift,
                       iterationPlan.ZImagResidualBitShift);
        PackForwardOne(plan,
                       debugCombo,
                       roots.InputScaleR,
                       &combo->DzdcReal,
                       workspace.DzdcReal,
                       workspace.IgnoredPrecisionBits,
                       iterationPlan.DzdcRealCoefficientShift,
                       iterationPlan.DzdcRealResidualBitShift);
        PackForwardOne(plan,
                       debugCombo,
                       roots.InputScaleR,
                       &combo->DzdcImag,
                       workspace.DzdcImag,
                       workspace.IgnoredPrecisionBits,
                       iterationPlan.DzdcImagCoefficientShift,
                       iterationPlan.DzdcImagResidualBitShift);
    } else if ((blockIdx.x & 1u) == 0u) {
        PackForwardOne(plan,
                       debugCombo,
                       roots.InputScaleR,
                       &zReal,
                       workspace.ZReal,
                       workspace.IgnoredPrecisionBits,
                       iterationPlan.ZRealCoefficientShift,
                       iterationPlan.ZRealResidualBitShift);
        PackForwardOne(plan,
                       debugCombo,
                       roots.InputScaleR,
                       &zImag,
                       workspace.ZImag,
                       workspace.IgnoredPrecisionBits,
                       iterationPlan.ZImagCoefficientShift,
                       iterationPlan.ZImagResidualBitShift);
    } else {
        PackForwardOne(plan,
                       debugCombo,
                       roots.InputScaleR,
                       &combo->DzdcReal,
                       workspace.DzdcReal,
                       workspace.IgnoredPrecisionBits,
                       iterationPlan.DzdcRealCoefficientShift,
                       iterationPlan.DzdcRealResidualBitShift);
        PackForwardOne(plan,
                       debugCombo,
                       roots.InputScaleR,
                       &combo->DzdcImag,
                       workspace.DzdcImag,
                       workspace.IgnoredPrecisionBits,
                       iterationPlan.DzdcImagCoefficientShift,
                       iterationPlan.DzdcImagResidualBitShift);
    }
    grid.sync();

    if constexpr (HpShark::DebugChecksums) {
        StoreReferenceDebugStateBatchAfterSync<SharkFloatParams>(debugStates,
                                                                 grid,
                                                                 block,
                                                                 DebugStatePurpose::Z0XX,
                                                                 workspace.ZReal,
                                                                 DebugStatePurpose::Z0YY,
                                                                 workspace.ZImag,
                                                                 DebugStatePurpose::Z0W1,
                                                                 workspace.DzdcReal,
                                                                 DebugStatePurpose::Z0W2,
                                                                 workspace.DzdcImag,
                                                                 transformSize);
    }

    uint32_t forwardStage = stageCount;
    while (forwardStage > tileSizeLog2 + 1u) {
        if (gridDim.x == 1u) {
            NTT::ForwardRadix4Two(debugCombo,
                                  workspace.ZReal,
                                  workspace.ZImag,
                                  roots.stage_twiddles_fwd,
                                  transformSize,
                                  forwardStage);
            NTT::ForwardRadix4Two(debugCombo,
                                  workspace.DzdcReal,
                                  workspace.DzdcImag,
                                  roots.stage_twiddles_fwd,
                                  transformSize,
                                  forwardStage);
        } else if ((blockIdx.x & 1u) == 0u) {
            NTT::ForwardRadix4Two(debugCombo,
                                  workspace.ZReal,
                                  workspace.ZImag,
                                  roots.stage_twiddles_fwd,
                                  transformSize,
                                  forwardStage);
        } else {
            NTT::ForwardRadix4Two(debugCombo,
                                  workspace.DzdcReal,
                                  workspace.DzdcImag,
                                  roots.stage_twiddles_fwd,
                                  transformSize,
                                  forwardStage);
        }
        grid.sync();
        forwardStage -= 2u;
    }
    if (forwardStage > tileSizeLog2) {
        if (gridDim.x == 1u) {
            NTT::ForwardRadix2Two(debugCombo,
                                  workspace.ZReal,
                                  workspace.ZImag,
                                  roots.stage_twiddles_fwd,
                                  transformSize,
                                  forwardStage);
            NTT::ForwardRadix2Two(debugCombo,
                                  workspace.DzdcReal,
                                  workspace.DzdcImag,
                                  roots.stage_twiddles_fwd,
                                  transformSize,
                                  forwardStage);
        } else if ((blockIdx.x & 1u) == 0u) {
            NTT::ForwardRadix2Two(debugCombo,
                                  workspace.ZReal,
                                  workspace.ZImag,
                                  roots.stage_twiddles_fwd,
                                  transformSize,
                                  forwardStage);
        } else {
            NTT::ForwardRadix2Two(debugCombo,
                                  workspace.DzdcReal,
                                  workspace.DzdcImag,
                                  roots.stage_twiddles_fwd,
                                  transformSize,
                                  forwardStage);
        }
        grid.sync();
    }
    if (gridDim.x == 1u) {
        NTT::ForwardTileTwoSelected(debugCombo,
                                    sharedData,
                                    workspace.ZReal,
                                    workspace.ZImag,
                                    roots.stage_twiddles_fwd,
                                    transformSize,
                                    stageCount,
                                    tileSizeLog2);
        block.sync();
        NTT::ForwardTileTwoSelected(debugCombo,
                                    sharedData,
                                    workspace.DzdcReal,
                                    workspace.DzdcImag,
                                    roots.stage_twiddles_fwd,
                                    transformSize,
                                    stageCount,
                                    tileSizeLog2);
    } else if ((blockIdx.x & 1u) == 0u) {
        NTT::ForwardTileTwoSelected(debugCombo,
                                    sharedData,
                                    workspace.ZReal,
                                    workspace.ZImag,
                                    roots.stage_twiddles_fwd,
                                    transformSize,
                                    stageCount,
                                    tileSizeLog2);
    } else {
        NTT::ForwardTileTwoSelected(debugCombo,
                                    sharedData,
                                    workspace.DzdcReal,
                                    workspace.DzdcImag,
                                    roots.stage_twiddles_fwd,
                                    transformSize,
                                    stageCount,
                                    tileSizeLog2);
    }
    grid.sync();

    if constexpr (HpShark::DebugChecksums) {
        StoreReferenceDebugStateBatchAfterSync<SharkFloatParams>(debugStates,
                                                                 grid,
                                                                 block,
                                                                 DebugStatePurpose::Z2XX,
                                                                 workspace.ZReal,
                                                                 DebugStatePurpose::Z2YY,
                                                                 workspace.ZImag,
                                                                 DebugStatePurpose::Z2W1,
                                                                 workspace.DzdcReal,
                                                                 DebugStatePurpose::Z2W2,
                                                                 workspace.DzdcImag,
                                                                 transformSize);
    }

    // DIT inverse stages run from low to high; the adaptive tile owns the local stages.
    if (!useFusedInversePointwise) {
        if (gridDim.x == 1u || (blockIdx.x & 1u) == 0u) {
            if (realProductEnabled && imagProductEnabled) {
                PointwiseOrbitPairTwo(debugCombo,
                                      workspace.ZReal,
                                      workspace.ZImag,
                                      workspace.RealOutput,
                                      workspace.ImagOutput,
                                      transformSize);
            } else {
                if (realProductEnabled) {
                    PointwiseOrbitRealOne(debugCombo,
                                          workspace.ZReal,
                                          workspace.ZImag,
                                          workspace.RealOutput,
                                          transformSize);
                } else {
                    PointwiseZeroOne(workspace.RealOutput, transformSize);
                }
                if (imagProductEnabled) {
                    PointwiseOrbitImagOne(debugCombo,
                                          workspace.ZReal,
                                          workspace.ZImag,
                                          workspace.ImagOutput,
                                          transformSize);
                } else {
                    PointwiseZeroOne(workspace.ImagOutput, transformSize);
                }
            }
        }
        if (gridDim.x == 1u || (blockIdx.x & 1u) != 0u) {
            if (derivativeEnabled) {
                PointwiseDerivativePairTwo(debugCombo,
                                           workspace.ZReal,
                                           workspace.ZImag,
                                           workspace.DzdcReal,
                                           workspace.DzdcImag,
                                           workspace.DzdcRealOutput,
                                           workspace.DzdcImagOutput,
                                           transformSize);
            } else {
                PointwiseZeroTwo(workspace.DzdcRealOutput, workspace.DzdcImagOutput, transformSize);
            }
        }
        grid.sync();

        if constexpr (HpShark::DebugChecksums) {
            StoreReferenceDebugStateBatchAfterSync<SharkFloatParams>(debugStates,
                                                                     grid,
                                                                     block,
                                                                     DebugStatePurpose::Z2_Perm1,
                                                                     workspace.RealOutput,
                                                                     DebugStatePurpose::Z2_Perm2,
                                                                     workspace.ImagOutput,
                                                                     DebugStatePurpose::Z2_PermW0,
                                                                     workspace.DzdcRealOutput,
                                                                     DebugStatePurpose::Z2_PermW1,
                                                                     workspace.DzdcImagOutput,
                                                                     transformSize);
        }
        InitializeCarryPrefixTransformsDLB<SharkFloatParams>(
            grid, block, limbCount, carryPrefixCapacity, carryPrefixDescriptors, carryPrefixShared);

        if (gridDim.x == 1u) {
            NTT::InverseTileTwoSelected<SharkFloatParams, NTT::InverseTileTwoLoad::Copy>(
                debugCombo,
                sharedData,
                workspace.RealOutput,
                workspace.ImagOutput,
                nullptr,
                nullptr,
                workspace.RealOutput,
                workspace.ImagOutput,
                false,
                false,
                roots.stage_twiddles_inv,
                transformSize,
                stageCount,
                tileSizeLog2);
            block.sync();
            NTT::InverseTileTwoSelected<SharkFloatParams, NTT::InverseTileTwoLoad::Copy>(
                debugCombo,
                sharedData,
                workspace.DzdcRealOutput,
                workspace.DzdcImagOutput,
                nullptr,
                nullptr,
                workspace.DzdcRealOutput,
                workspace.DzdcImagOutput,
                false,
                false,
                roots.stage_twiddles_inv,
                transformSize,
                stageCount,
                tileSizeLog2);
        } else if ((blockIdx.x & 1u) == 0u) {
            NTT::InverseTileTwoSelected<SharkFloatParams, NTT::InverseTileTwoLoad::Copy>(
                debugCombo,
                sharedData,
                workspace.RealOutput,
                workspace.ImagOutput,
                nullptr,
                nullptr,
                workspace.RealOutput,
                workspace.ImagOutput,
                false,
                false,
                roots.stage_twiddles_inv,
                transformSize,
                stageCount,
                tileSizeLog2);
        } else {
            NTT::InverseTileTwoSelected<SharkFloatParams, NTT::InverseTileTwoLoad::Copy>(
                debugCombo,
                sharedData,
                workspace.DzdcRealOutput,
                workspace.DzdcImagOutput,
                nullptr,
                nullptr,
                workspace.DzdcRealOutput,
                workspace.DzdcImagOutput,
                false,
                false,
                roots.stage_twiddles_inv,
                transformSize,
                stageCount,
                tileSizeLog2);
        }
        grid.sync();
    } else {
        if (gridDim.x == 1u) {
            NTT::InverseTileTwoSelected<SharkFloatParams, NTT::InverseTileTwoLoad::OrbitPair>(
                debugCombo,
                sharedData,
                workspace.ZReal,
                workspace.ZImag,
                nullptr,
                nullptr,
                workspace.RealOutput,
                workspace.ImagOutput,
                realProductEnabled,
                imagProductEnabled,
                roots.stage_twiddles_inv,
                transformSize,
                stageCount,
                tileSizeLog2);
            block.sync();
            NTT::InverseTileTwoSelected<SharkFloatParams, NTT::InverseTileTwoLoad::DerivativePair>(
                debugCombo,
                sharedData,
                workspace.ZReal,
                workspace.ZImag,
                workspace.DzdcReal,
                workspace.DzdcImag,
                workspace.DzdcRealOutput,
                workspace.DzdcImagOutput,
                derivativeEnabled,
                false,
                roots.stage_twiddles_inv,
                transformSize,
                stageCount,
                tileSizeLog2);
        } else if ((blockIdx.x & 1u) == 0u) {
            NTT::InverseTileTwoSelected<SharkFloatParams, NTT::InverseTileTwoLoad::OrbitPair>(
                debugCombo,
                sharedData,
                workspace.ZReal,
                workspace.ZImag,
                nullptr,
                nullptr,
                workspace.RealOutput,
                workspace.ImagOutput,
                realProductEnabled,
                imagProductEnabled,
                roots.stage_twiddles_inv,
                transformSize,
                stageCount,
                tileSizeLog2);
        } else {
            NTT::InverseTileTwoSelected<SharkFloatParams, NTT::InverseTileTwoLoad::DerivativePair>(
                debugCombo,
                sharedData,
                workspace.ZReal,
                workspace.ZImag,
                workspace.DzdcReal,
                workspace.DzdcImag,
                workspace.DzdcRealOutput,
                workspace.DzdcImagOutput,
                derivativeEnabled,
                false,
                roots.stage_twiddles_inv,
                transformSize,
                stageCount,
                tileSizeLog2);
        }
        grid.sync();
        InitializeCarryPrefixTransformsDLB<SharkFloatParams>(
            grid, block, limbCount, carryPrefixCapacity, carryPrefixDescriptors, carryPrefixShared);
        if (stageCount < tileSizeLog2 + 1u)
            grid.sync();
    }

    uint32_t inverseStage = tileSizeLog2 + 1u;
    while (inverseStage + 1u <= stageCount) {
        if (gridDim.x == 1u) {
            NTT::InverseRadix4Two(debugCombo,
                                  workspace.RealOutput,
                                  workspace.ImagOutput,
                                  roots.stage_twiddles_inv,
                                  transformSize,
                                  inverseStage + 1u);
            NTT::InverseRadix4Two(debugCombo,
                                  workspace.DzdcRealOutput,
                                  workspace.DzdcImagOutput,
                                  roots.stage_twiddles_inv,
                                  transformSize,
                                  inverseStage + 1u);
        } else if ((blockIdx.x & 1u) == 0u) {
            NTT::InverseRadix4Two(debugCombo,
                                  workspace.RealOutput,
                                  workspace.ImagOutput,
                                  roots.stage_twiddles_inv,
                                  transformSize,
                                  inverseStage + 1u);
        } else {
            NTT::InverseRadix4Two(debugCombo,
                                  workspace.DzdcRealOutput,
                                  workspace.DzdcImagOutput,
                                  roots.stage_twiddles_inv,
                                  transformSize,
                                  inverseStage + 1u);
        }
        grid.sync();
        inverseStage += 2u;
    }
    if (inverseStage <= stageCount) {
        if (gridDim.x == 1u) {
            NTT::InverseRadix2Two(debugCombo,
                                  workspace.RealOutput,
                                  workspace.ImagOutput,
                                  roots.stage_twiddles_inv,
                                  transformSize,
                                  inverseStage);
            NTT::InverseRadix2Two(debugCombo,
                                  workspace.DzdcRealOutput,
                                  workspace.DzdcImagOutput,
                                  roots.stage_twiddles_inv,
                                  transformSize,
                                  inverseStage);
        } else if ((blockIdx.x & 1u) == 0u) {
            NTT::InverseRadix2Two(debugCombo,
                                  workspace.RealOutput,
                                  workspace.ImagOutput,
                                  roots.stage_twiddles_inv,
                                  transformSize,
                                  inverseStage);
        } else {
            NTT::InverseRadix2Two(debugCombo,
                                  workspace.DzdcRealOutput,
                                  workspace.DzdcImagOutput,
                                  roots.stage_twiddles_inv,
                                  transformSize,
                                  inverseStage);
        }
        grid.sync();
    }

    if constexpr (HpShark::DebugChecksums) {
        StoreReferenceDebugStateBatchAfterSync<SharkFloatParams>(debugStates,
                                                                 grid,
                                                                 block,
                                                                 DebugStatePurpose::Z2_Perm4,
                                                                 workspace.RealOutput,
                                                                 DebugStatePurpose::Z2_Perm5,
                                                                 workspace.ImagOutput,
                                                                 DebugStatePurpose::Z2_PermW0b,
                                                                 workspace.DzdcRealOutput,
                                                                 DebugStatePurpose::Z2_PermW1b,
                                                                 workspace.DzdcImagOutput,
                                                                 transformSize);
    }
    if (plan.b == 16) {
        RunAlignedB16CarryPrefixNr<SharkFloatParams>(workspace, combo, carryPrefixShared);
    } else {
        if (realLinearEnabled) {
            UnpackAlignedResiduesToSignedLimbsOneWithLinear<SharkFloatParams>(
                workspace.RealOutput,
                plan,
                iterationPlan.RealCoefficientCount,
                iterationPlan.RealProductBitOffset,
                &cReal,
                workspace.IgnoredPrecisionBits,
                iterationPlan.RealLinearBitOffset,
                workspace.RealLimbs,
                limbCount);
        } else {
            UnpackAlignedResiduesToSignedLimbsOneNoLinear<SharkFloatParams>(
                workspace.RealOutput,
                plan,
                iterationPlan.RealCoefficientCount,
                iterationPlan.RealProductBitOffset,
                workspace.RealLimbs,
                limbCount);
        }
        if (imagLinearEnabled) {
            UnpackAlignedResiduesToSignedLimbsOneWithLinear<SharkFloatParams>(
                workspace.ImagOutput,
                plan,
                iterationPlan.ImagCoefficientCount,
                iterationPlan.ImagProductBitOffset,
                &cImag,
                workspace.IgnoredPrecisionBits,
                iterationPlan.ImagLinearBitOffset,
                workspace.ImagLimbs,
                limbCount);
        } else {
            UnpackAlignedResiduesToSignedLimbsOneNoLinear<SharkFloatParams>(
                workspace.ImagOutput,
                plan,
                iterationPlan.ImagCoefficientCount,
                iterationPlan.ImagProductBitOffset,
                workspace.ImagLimbs,
                limbCount);
        }
        if (dzdcOneEnabled) {
            UnpackAlignedResiduesToSignedLimbsOneWithLinear<SharkFloatParams>(
                workspace.DzdcRealOutput,
                plan,
                iterationPlan.DzdcRealCoefficientCount,
                iterationPlan.DzdcRealProductBitOffset,
                &combo->One,
                workspace.IgnoredPrecisionBits,
                iterationPlan.DzdcRealLinearBitOffset,
                workspace.DzdcRealLimbs,
                limbCount);
        } else {
            UnpackAlignedResiduesToSignedLimbsOneNoLinear<SharkFloatParams>(
                workspace.DzdcRealOutput,
                plan,
                iterationPlan.DzdcRealCoefficientCount,
                iterationPlan.DzdcRealProductBitOffset,
                workspace.DzdcRealLimbs,
                limbCount);
        }
        UnpackAlignedResiduesToSignedLimbsOneNoLinear<SharkFloatParams>(
            workspace.DzdcImagOutput,
            plan,
            iterationPlan.DzdcImagCoefficientCount,
            iterationPlan.DzdcImagProductBitOffset,
            workspace.DzdcImagLimbs,
            limbCount);
        grid.sync();
    }
    if constexpr (HpShark::DebugChecksums) {
        StoreReferenceDebugStateBatchAfterSync<SharkFloatParams>(
            debugStates,
            grid,
            block,
            DebugStatePurpose::UnpackXX,
            reinterpret_cast<const uint64_t *>(workspace.RealLimbs),
            DebugStatePurpose::UnpackYY,
            reinterpret_cast<const uint64_t *>(workspace.ImagLimbs),
            DebugStatePurpose::UnpackW0,
            reinterpret_cast<const uint64_t *>(workspace.DzdcRealLimbs),
            DebugStatePurpose::UnpackW1,
            reinterpret_cast<const uint64_t *>(workspace.DzdcImagLimbs),
            limbCount);
    }
    FinalizeSignedStream<SharkFloatParams>(debugStates, carryPrefixShared, workspace, combo);
    StoreReferenceDebugValueBatch<SharkFloatParams>(debugStates,
                                                    grid,
                                                    block,
                                                    DebugStatePurpose::Result_AddDzdc1,
                                                    combo->DzdcReal,
                                                    DebugStatePurpose::Result_AddDzdc2,
                                                    combo->DzdcImag,
                                                    DebugStatePurpose::Result_Add1,
                                                    combo->ZReal,
                                                    DebugStatePurpose::Result_Add2,
                                                    combo->ZImag);
}

template <class SharkFloatParams>
__device__ SharkForceInlineReleaseOnly void
ExecuteReferenceIteration(cooperative_groups::grid_group &grid,
                          cooperative_groups::thread_block &block,
                          uint64_t *sharedData,
                          DebugGlobalCount<SharkFloatParams> *debugCombo,
                          DebugState<SharkFloatParams> *debugStates,
                          uint64_t *carryPrefixShared,
                          HpSharkReferenceResults<SharkFloatParams> *combo)
{
    auto &workspace = *combo->Workspace;
    const auto &iterationPlan = workspace.IterationPlan;
    const uint32_t flags = iterationPlan.Flags;
    const bool dzdcOneEnabled = (flags & HpSharkReferencePlanDzdcOne) != 0u;
    const bool realLinearEnabled = (flags & HpSharkReferencePlanRealLinear) != 0u;
    const bool imagLinearEnabled = (flags & HpSharkReferencePlanImagLinear) != 0u;
    const auto kind = static_cast<HpSharkReferenceIterationKind>(iterationPlan.Kind);

    if (kind == HpSharkReferenceIterationKind::Zero) {
        SetZeroDigits(grid, block, &combo->ZReal);
        SetZeroDigits(grid, block, &combo->ZImag);
        SetZeroMetadata(block, &combo->ZReal);
        SetZeroMetadata(block, &combo->ZImag);
        if constexpr (SharkFloatParams::EnableNewtonRaphson) {
            SetZeroDigits(grid, block, &combo->DzdcReal);
            SetZeroDigits(grid, block, &combo->DzdcImag);
            SetZeroMetadata(block, &combo->DzdcReal);
            SetZeroMetadata(block, &combo->DzdcImag);
        }
        if constexpr (SharkFloatParams::EnableNewtonRaphson) {
            StoreReferenceDebugValueBatch<SharkFloatParams>(debugStates,
                                                            grid,
                                                            block,
                                                            DebugStatePurpose::Result_Add1,
                                                            combo->ZReal,
                                                            DebugStatePurpose::Result_Add2,
                                                            combo->ZImag,
                                                            DebugStatePurpose::Result_AddDzdc1,
                                                            combo->DzdcReal,
                                                            DebugStatePurpose::Result_AddDzdc2,
                                                            combo->DzdcImag);
        } else {
            StoreReferenceDebugValueBatch<SharkFloatParams>(debugStates,
                                                            grid,
                                                            block,
                                                            DebugStatePurpose::Result_Add1,
                                                            combo->ZReal,
                                                            DebugStatePurpose::Result_Add2,
                                                            combo->ZImag);
        }
        return;
    }

    if (kind == HpSharkReferenceIterationKind::LinearOnly) {
        const uint32_t limbCount = iterationPlan.LimbCount;
        const uint32_t carryPrefixCapacity = workspace.ActiveMaxFusedLimbs;
        auto *carryPrefixDescriptors =
            reinterpret_cast<HpSharkReferencePackedCarryPrefixDescriptor *>(workspace.ZReal);
        const auto &cReal = combo->CReal;
        const auto &cImag = combo->CImag;
        InitializeCarryPrefixTransformsDLB<SharkFloatParams>(
            grid, block, limbCount, carryPrefixCapacity, carryPrefixDescriptors, carryPrefixShared);
        if constexpr (SharkFloatParams::EnableNewtonRaphson) {
            if (realLinearEnabled) {
                GatherLinearToSignedLimbsOne(
                    cReal, workspace.IgnoredPrecisionBits, workspace.RealLimbs, limbCount);
            } else {
                ZeroSignedLimbsOne(workspace.RealLimbs, limbCount);
            }
            if (imagLinearEnabled) {
                GatherLinearToSignedLimbsOne(
                    cImag, workspace.IgnoredPrecisionBits, workspace.ImagLimbs, limbCount);
            } else {
                ZeroSignedLimbsOne(workspace.ImagLimbs, limbCount);
            }
            if (dzdcOneEnabled) {
                GatherLinearToSignedLimbsOne(
                    combo->One, workspace.IgnoredPrecisionBits, workspace.DzdcRealLimbs, limbCount);
            } else {
                ZeroSignedLimbsOne(workspace.DzdcRealLimbs, limbCount);
            }
            ZeroSignedLimbsOne(workspace.DzdcImagLimbs, limbCount);
        } else {
            if (realLinearEnabled) {
                GatherLinearToSignedLimbsOne(
                    cReal, workspace.IgnoredPrecisionBits, workspace.RealLimbs, limbCount);
            } else {
                ZeroSignedLimbsOne(workspace.RealLimbs, limbCount);
            }
            if (imagLinearEnabled) {
                GatherLinearToSignedLimbsOne(
                    cImag, workspace.IgnoredPrecisionBits, workspace.ImagLimbs, limbCount);
            } else {
                ZeroSignedLimbsOne(workspace.ImagLimbs, limbCount);
            }
        }
        grid.sync();
        FinalizeSignedStream<SharkFloatParams>(debugStates, carryPrefixShared, workspace, combo);
        if constexpr (SharkFloatParams::EnableNewtonRaphson) {
            StoreReferenceDebugValueBatch<SharkFloatParams>(debugStates,
                                                            grid,
                                                            block,
                                                            DebugStatePurpose::Result_Add1,
                                                            combo->ZReal,
                                                            DebugStatePurpose::Result_Add2,
                                                            combo->ZImag,
                                                            DebugStatePurpose::Result_AddDzdc1,
                                                            combo->DzdcReal,
                                                            DebugStatePurpose::Result_AddDzdc2,
                                                            combo->DzdcImag);
        } else {
            StoreReferenceDebugValueBatch<SharkFloatParams>(debugStates,
                                                            grid,
                                                            block,
                                                            DebugStatePurpose::Result_Add1,
                                                            combo->ZReal,
                                                            DebugStatePurpose::Result_Add2,
                                                            combo->ZImag);
        }
        return;
    }

    MattsCudaAssert(kind == HpSharkReferenceIterationKind::Ntt);
    if constexpr (SharkFloatParams::EnableNewtonRaphson) {
        ExecuteReferenceIterationNrNtt(
            grid, block, sharedData, debugCombo, debugStates, carryPrefixShared, combo);
    } else {
        ExecuteReferenceIterationNonNrNtt(
            grid, block, sharedData, debugCombo, debugStates, carryPrefixShared, combo);
    }
    return;
}

template <class SharkFloatParams>
__device__ void
UpdateD2(HpSharkReferenceResults<SharkFloatParams> *combo)
{
    if constexpr (SharkFloatParams::EnableNewtonRaphson) {
        using Hdr = typename SharkFloatParams::Float;
        const Hdr zr = ToNormalizedHDRFloat(combo->ZReal);
        const Hdr zi = ToNormalizedHDRFloat(combo->ZImag);
        const Hdr dzr = ToNormalizedHDRFloat(combo->DzdcReal);
        const Hdr dzi = ToNormalizedHDRFloat(combo->DzdcImag);
        Hdr dz2r = dzr * dzr - dzi * dzi;
        HdrReduce(dz2r);
        Hdr dz2i = Hdr{2.0f} * (dzr * dzi);
        HdrReduce(dz2i);
        Hdr zd2r = zr * combo->D2Real - zi * combo->D2Imag;
        HdrReduce(zd2r);
        Hdr zd2i = zr * combo->D2Imag + zi * combo->D2Real;
        HdrReduce(zd2i);
        Hdr sumr = dz2r + zd2r;
        HdrReduce(sumr);
        Hdr sumi = dz2i + zd2i;
        HdrReduce(sumi);
        combo->D2Real = Hdr{2.0f} * sumr;
        combo->D2Imag = Hdr{2.0f} * sumi;
    }
}

template <class SharkFloatParams>
__device__ bool
CheckPeriodicity(HpSharkReferenceResults<SharkFloatParams> *combo, uint64_t iteration)
{
    if constexpr (!SharkFloatParams::EnablePeriodicity) {
        (void)combo;
        (void)iteration;
        return false;
    } else {
        using Hdr = typename SharkFloatParams::Float;
        Hdr zx = ToNormalizedHDRFloat(combo->ZReal);
        Hdr zy = ToNormalizedHDRFloat(combo->ZImag);
        if (iteration < HpSharkReferenceResults<SharkFloatParams>::MaxOutputIters) {
            combo->OutputIters[iteration].x = zx;
            combo->OutputIters[iteration].y = zy;
        }
        HdrReduce(combo->DzdcX);
        const Hdr dxAbs = HdrAbs(combo->DzdcX);
        HdrReduce(combo->DzdcY);
        const Hdr dyAbs = HdrAbs(combo->DzdcY);
        HdrReduce(zx);
        const Hdr zxAbs = HdrAbs(zx);
        HdrReduce(zy);
        const Hdr zyAbs = HdrAbs(zy);
        const Hdr n2 = HdrMaxPositiveReduced(zxAbs, zyAbs);
        const Hdr n3 = combo->RadiusY * HdrMaxPositiveReduced(dxAbs, dyAbs) * Hdr{2.0f};
        if (HdrCompareToBothPositiveReducedLT(n2, n3)) {
            combo->PeriodicityStatus = PeriodicityResult::PeriodFound;
            ++combo->OutputIterCount;
            return true;
        }
        const Hdr dx = combo->DzdcX;
        combo->DzdcX = Hdr{2.0f} * (zx * combo->DzdcX - zy * combo->DzdcY) + Hdr{1.0f};
        combo->DzdcY = Hdr{2.0f} * (zx * combo->DzdcY + zy * dx);
        const Hdr cx = ToNormalizedHDRFloat(combo->CReal);
        const Hdr cy = ToNormalizedHDRFloat(combo->CImag);
        const Hdr tx = zx + cx;
        const Hdr ty = zy + cy;
        const Hdr size = tx * tx + ty * ty;
        if (HdrCompareToBothPositiveReducedGT(size, Hdr{256.0f})) {
            combo->PeriodicityStatus = PeriodicityResult::Escaped;
            ++combo->OutputIterCount;
            return true;
        }
        (void)iteration;
        return false;
    }
}

} // namespace ReferenceDetail

template <class SharkFloatParams>
__global__ void
__maxnreg__(HpShark::RegisterLimit)
    HpSharkReferenceSetupKernel(HpSharkReferenceWorkspace<SharkFloatParams> *workspace,
                                uint64_t *tempData)
{
    namespace cg = cooperative_groups;
    cg::grid_group grid = cg::this_grid();
    cg::thread_block block = cg::this_thread_block();
    DebugGlobalCount<SharkFloatParams> *debugCombo = nullptr;
    DebugState<SharkFloatParams> *debugStates = nullptr;
    if constexpr (HpShark::DebugGlobalState) {
        debugCombo = reinterpret_cast<DebugGlobalCount<SharkFloatParams> *>(
            &tempData[HpShark::AdditionalGlobalSyncSpace]);
        if (ReferenceDetail::IsLeader(block))
            debugCombo->DebugMultiplyErase();
    }
    if constexpr (HpShark::DebugChecksums) {
        debugStates = reinterpret_cast<DebugState<SharkFloatParams> *>(
            &tempData[HpShark::AdditionalChecksumsOffset]);
        EraseAllDebugStates(debugStates, grid, block);
    }

    using Workspace = HpSharkReferenceWorkspace<SharkFloatParams>;
    for (uint32_t stage = workspace->ActiveMinFusedStages; stage <= workspace->ActiveMaxFusedStages;
         ++stage) {
        const uint32_t activeN = 1u << stage;
        ReferenceDetail::GenerateCachedPlan<SharkFloatParams>(
            grid, block, debugCombo, activeN, *workspace);
    }

    const uint32_t firstSlot = workspace->ActiveMinFusedStages - Workspace::MinFusedStages;
    const uint32_t activePlanMask = workspace->ActivePlanCacheEntryCount == 32u
                                        ? ~0u
                                        : (1u << workspace->ActivePlanCacheEntryCount) - 1u;
    const uint32_t fullPlanMask = activePlanMask << firstSlot;
    ReferenceDetail::MattsCudaAssert(workspace->ValidPlanMask == fullPlanMask);
}

template <class SharkFloatParams>
__global__ void
__maxnreg__(HpShark::RegisterLimit)
    HpSharkReferenceGpuLoop(HpSharkReferenceResults<SharkFloatParams> *combo, uint64_t *tempData)
{
    namespace cg = cooperative_groups;
    cg::grid_group grid = cg::this_grid();
    cg::thread_block block = cg::this_thread_block();
    extern __shared__ __align__(16) uint64_t sharedData[];
    __shared__ uint64_t carryPrefixShared[2u * ReferenceDetail::CarryPrefixMaxWarps];
    const bool leader = ReferenceDetail::IsLeader(block);
    DebugGlobalCount<SharkFloatParams> *debugCombo = nullptr;
    DebugState<SharkFloatParams> *debugStates = nullptr;
    if constexpr (HpShark::DebugGlobalState) {
        const auto offset = HpShark::AdditionalGlobalSyncSpace;
        debugCombo = reinterpret_cast<DebugGlobalCount<SharkFloatParams> *>(&tempData[offset]);
        if (leader)
            debugCombo->DebugMultiplyErase();
    }
    if constexpr (HpShark::DebugChecksums) {
        debugStates = reinterpret_cast<DebugState<SharkFloatParams> *>(
            &tempData[HpShark::AdditionalChecksumsOffset]);
        EraseAllDebugStates(debugStates, grid, block);
    }

    if (leader) {
        combo->OutputIterCount = 0;
        combo->PeriodicityStatus = PeriodicityResult::Continue;
    }

    ReferenceDetail::StoreReferenceDebugValueBatch<SharkFloatParams>(
        debugStates,
        grid,
        block,
        DebugStatePurpose::ReferenceEntryZReal,
        combo->ZReal,
        DebugStatePurpose::ReferenceEntryZImag,
        combo->ZImag,
        DebugStatePurpose::ReferenceEntryCReal,
        combo->CReal,
        DebugStatePurpose::ReferenceEntryCImag,
        combo->CImag);

    for (uint64_t iteration = 0; iteration < combo->MaxRuntimeIters; ++iteration) {
        if (leader)
            ReferenceDetail::CheckPeriodicity<SharkFloatParams>(combo, iteration);
        if (leader && combo->PeriodicityStatus == PeriodicityResult::Continue)
            ReferenceDetail::BuildReferenceIterationPlan<SharkFloatParams>(combo);
        grid.sync();
        if (combo->PeriodicityStatus != PeriodicityResult::Continue)
            break;

        if (leader)
            ReferenceDetail::UpdateD2<SharkFloatParams>(combo);

        ReferenceDetail::ExecuteReferenceIteration<SharkFloatParams>(
            grid, block, sharedData, debugCombo, debugStates, carryPrefixShared, combo);
        // Publish every output and PeriodicityStatus before any thread consumes them.
        grid.sync();
        if (combo->PeriodicityStatus == PeriodicityResult::Unknown)
            break;

        if (leader)
            ++combo->OutputIterCount;
    }

    ReferenceDetail::StoreReferenceDebugValueBatch<SharkFloatParams>(
        debugStates,
        grid,
        block,
        DebugStatePurpose::ReferenceExitZReal,
        combo->ZReal,
        DebugStatePurpose::ReferenceExitZImag,
        combo->ZImag);
}

namespace ReferenceLaunchDetail {

inline void
CheckCuda(cudaError_t result, const char *operation)
{
    if (result == cudaSuccess)
        return;
    std::ostringstream message;
    message << operation << " failed: " << cudaGetErrorString(result);
    throw FractalSharkSeriousException(message.str());
}

inline void
CheckLaunch(cudaError_t result, const char *kernelName, const HpShark::LaunchParams &resolved)
{
    if (result == cudaSuccess)
        return;
    std::ostringstream message;
    message << "cudaLaunchCooperativeKernel(" << kernelName << ") failed: " << cudaGetErrorString(result)
            << " | blocks=" << resolved.NumBlocks << " threads=" << resolved.ThreadsPerBlock;
    throw FractalSharkSeriousException(message.str());
}

} // namespace ReferenceLaunchDetail

template <class SharkFloatParams>
void
ComputeHpSharkReferenceSetup(const HpShark::LaunchParams &launchParams,
                             cudaStream_t &stream,
                             void *kernelArgs[])
{
    constexpr auto SharedMemSize = 0u;
    const cudaError_t attribute = cudaFuncSetAttribute(HpSharkReferenceSetupKernel<SharkFloatParams>,
                                                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                       SharedMemSize);
    ReferenceLaunchDetail::CheckCuda(attribute, "cudaFuncSetAttribute(HpSharkReferenceSetupKernel)");

    HpShark::LaunchParams resolved{launchParams};
    if (resolved.NumBlocks == 0) {
        HpShark::CudaLaunchConfig config;
        const cudaError_t result =
            config.compute(reinterpret_cast<const void *>(HpSharkReferenceSetupKernel<SharkFloatParams>),
                           SharedMemSize,
                           resolved);
        ReferenceLaunchDetail::CheckCuda(result, "LaunchConfig.compute(HpSharkReferenceSetupKernel)");
    }

    const cudaError_t launch = cudaLaunchCooperativeKernel(
        reinterpret_cast<void *>(HpSharkReferenceSetupKernel<SharkFloatParams>),
        dim3(resolved.NumBlocks),
        dim3(resolved.ThreadsPerBlock),
        kernelArgs,
        SharedMemSize,
        stream);
    ReferenceLaunchDetail::CheckLaunch(launch, "HpSharkReferenceSetupKernel", resolved);
    const cudaError_t immediate = cudaGetLastError();
    ReferenceLaunchDetail::CheckCuda(immediate,
                                     "cudaGetLastError() after HpSharkReferenceSetupKernel launch");
    const cudaError_t synchronized = cudaDeviceSynchronize();
    ReferenceLaunchDetail::CheckCuda(synchronized,
                                     "cudaDeviceSynchronize() after HpSharkReferenceSetupKernel");
}

template <class SharkFloatParams>
void
ComputeHpSharkReferenceGpuLoop(const HpShark::LaunchParams &launchParams,
                               cudaStream_t &stream,
                               void *kernelArgs[])
{
    constexpr auto SharedMemSize = HpShark::CalculateReferenceSharedMemorySize<SharkFloatParams>();
    const cudaError_t attribute = cudaFuncSetAttribute(HpSharkReferenceGpuLoop<SharkFloatParams>,
                                                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                       SharedMemSize);
    ReferenceLaunchDetail::CheckCuda(attribute, "cudaFuncSetAttribute(HpSharkReferenceGpuLoop)");

    HpShark::LaunchParams resolved{launchParams};
    if (resolved.NumBlocks == 0) {
        HpShark::CudaLaunchConfig config;
        const cudaError_t result =
            config.compute(reinterpret_cast<const void *>(HpSharkReferenceGpuLoop<SharkFloatParams>),
                           SharedMemSize,
                           resolved);
        ReferenceLaunchDetail::CheckCuda(result, "LaunchConfig.compute(HpSharkReferenceGpuLoop)");
    } else {
        int device = 0;
        const cudaError_t getDevice = cudaGetDevice(&device);
        ReferenceLaunchDetail::CheckCuda(getDevice, "cudaGetDevice for HpSharkReferenceGpuLoop");

        int maxThreadsPerBlock = 0;
        const cudaError_t getLimit =
            cudaDeviceGetAttribute(&maxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, device);
        ReferenceLaunchDetail::CheckCuda(getLimit,
                                         "cudaDeviceGetAttribute(cudaDevAttrMaxThreadsPerBlock)");

        if (resolved.NumBlocks < 1 || resolved.ThreadsPerBlock < 32 ||
            (resolved.ThreadsPerBlock & 31) != 0 || resolved.ThreadsPerBlock > maxThreadsPerBlock) {
            std::ostringstream message;
            message << "Invalid explicit HpSharkReferenceGpuLoop launch shape: blocks="
                    << resolved.NumBlocks << " threads=" << resolved.ThreadsPerBlock
                    << " (threads must be a warp multiple from 32 through the device maximum of "
                    << maxThreadsPerBlock << ")";
            throw FractalSharkSeriousException(message.str());
        }
    }

    const cudaError_t launch =
        cudaLaunchCooperativeKernel(reinterpret_cast<void *>(HpSharkReferenceGpuLoop<SharkFloatParams>),
                                    dim3(resolved.NumBlocks),
                                    dim3(resolved.ThreadsPerBlock),
                                    kernelArgs,
                                    SharedMemSize,
                                    stream);
    ReferenceLaunchDetail::CheckLaunch(launch, "HpSharkReferenceGpuLoop", resolved);
    const cudaError_t immediate = cudaGetLastError();
    ReferenceLaunchDetail::CheckCuda(immediate,
                                     "cudaGetLastError() after HpSharkReferenceGpuLoop launch");
    const cudaError_t synchronized = cudaDeviceSynchronize();
    ReferenceLaunchDetail::CheckCuda(synchronized,
                                     "cudaDeviceSynchronize() after HpSharkReferenceGpuLoop");
}
