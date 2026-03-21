#include "HpSharkFloat.h"
#include "DebugChecksum.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdint>
#include <cassert>
#include <cstring>

#include <iostream>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

//
// Helper functions to perform bit shifts on a fixed-width digit array.
// They mirror the CUDA device functions but work sequentially on the full array.
//

// #define MattsCudaAssert assert

#ifdef _DEBUG
static __device__ SharkForceInlineReleaseOnly void
MattsCudaAssert (bool cond) {
    if (!cond) {
        //assert(cond);
        // asm("brkpt;");
        for (;;);
    }
}
#else
static __device__ SharkForceInlineReleaseOnly void
MattsCudaAssert (bool) {
    // no-op in release builds
}
#endif


// Direction tag for our funnel-shift helper
enum class Dir { Left, Right };

/// Funnels two 32-bit words from 'data' around a bit-offset boundary.
/// - For Dir::Right, this emulates a right shift across word boundaries.
/// - For Dir::Left,  this emulates a left  shift across word boundaries.
/// 'N' is the number of valid words in 'data'; out-of-range indices yield 0.
/// Casts values in input array to 32-bits even if it's 64-bit
template<Dir D>
static __device__ SharkForceInlineReleaseOnly uint32_t
FunnelShift32 (
    const auto *SharkRestrict data,
    int              idx,
    int              N,
    int              bitOffset) {
    int wordOff = bitOffset / 32;
    int b = bitOffset % 32;

    auto pick = [&](int i) -> uint32_t {
        return (i < 0 || i >= N) ? 0u : static_cast<uint32_t>(data[i]);
        };

    uint32_t low, high;
    if constexpr (D == Dir::Right) {
        low = pick(idx + wordOff);
        high = pick(idx + wordOff + 1);
    } else {
        low = pick(idx - wordOff);
        high = pick(idx - wordOff - 1);
    }

    if (b == 0) return low;
    if constexpr (D == Dir::Right)
        return (low >> b) | (high << (32 - b));
    else
        return (low << b) | (high >> (32 - b));
}

/// Retrieves the digit at 'idx' after a left shift by 'shiftBits',
/// treating words beyond 'actualDigits' as zero (within an 'numActualDigitsPlusGuard' buffer).
static __device__ SharkForceInlineReleaseOnly uint32_t
GetNormalizedDigit(
    const uint32_t *SharkRestrict digits,
    int32_t         actualDigits,
    int32_t         numActualDigitsPlusGuard,
    int32_t         shiftBits,
    int32_t         idx) {
    // ensure idx is within the extended buffer
    MattsCudaAssert(idx >= 0 && idx < numActualDigitsPlusGuard);

    // funnel-shift left within the 'actualDigits' region
    return FunnelShift32<Dir::Left>(
        digits,
        idx,
        actualDigits,
        shiftBits
    );
}

// Counts the number of leading zero bits in a 32-bit unsigned integer.
// This is a portable implementation of the count leading zeros operation.
static __device__ SharkForceInlineReleaseOnly int32_t
CountLeadingZeros (
    const uint32_t x) {
#if defined(__CUDA_ARCH__)
    // __clz returns 0–32 inclusive, even for x==0
    return __clz(static_cast<int>(x));
#else
    // std::countl_zero is constexpr in C++20 and returns 32 for x==0
    return static_cast<int>(std::countl_zero(x));
#endif
}

//
// Multi-word shift routines for little-endian arrays
//

// Generic multi-word shift using the requested parameter names
template<Dir D>
static __device__ SharkForceInlineReleaseOnly void
MultiWordShift (
    const int gridSize,
    const int32_t idx,
    const auto *SharkRestrict in,
    const int32_t  numActualDigitsPlusGuard,
    const int32_t  shiftNeeded,
    auto *SharkRestrict out,
    const int32_t  outSz
)
{
    MattsCudaAssert(numActualDigitsPlusGuard >= outSz);

    for (int32_t i = idx; i < outSz; i += gridSize) {
        out[i] = FunnelShift32<D>(
            in,
            i,
            numActualDigitsPlusGuard,
            shiftNeeded
        );
    }
}

// Retrieves a limb from an extended array, returning zero for indices beyond the actual digit count.
// This handles the boundary between actual digits and guard digits in extended precision arithmetic.
static __device__ SharkForceInlineReleaseOnly uint32_t
GetExtLimb (
    const uint32_t *SharkRestrict ext,
    const int32_t actualDigits,
    const int32_t numActualDigitsPlusGuard,
    const int32_t idx) {

    if (idx < actualDigits) {
        return ext[idx];
    } else {
        MattsCudaAssert(idx < numActualDigitsPlusGuard);
        return 0;
    }
}

//
// New ExtendedNormalize routine
//
// Instead of copying (shifting) the entire array, this routine computes
// a shift offset (L) such that if you were to left-shift the original array by L bits,
// its most-significant set bit would be in the highest bit position of the extended field.
// It then adjusts the stored exponent accordingly and returns the shift offset.
//

/// Compute the left-shift needed to bring each MSB to the top.
/// For NR types, processes all 10 arrays (5 orbit + 5 NR) in one scan.
/// Uses a two-tier approach:
///   Tier 1: Quick top-limb check — if top limb is non-zero, MSD found in O(1).
///   Tier 2: Warp-parallel scan via __ballot_sync for remaining arrays (32 limbs/tile).
template<class SharkFloatParams>
static __device__ SharkForceInlineReleaseOnly void
ExtendedNormalizeShiftIndexAll(
    const uint32_t *SharkRestrict extA,
    const uint32_t *SharkRestrict extB,
    const uint32_t *SharkRestrict extC,
    const uint32_t *SharkRestrict extD,
    const uint32_t *SharkRestrict extE,
    int32_t actualDigits,
    int32_t numActualDigitsPlusGuard,
    int32_t &expA, int32_t &expB, int32_t &expC, int32_t &expD, int32_t &expE,
    bool &isZeroA, bool &isZeroB, bool &isZeroC, bool &isZeroD, bool &isZeroE,
    int32_t &shiftA, int32_t &shiftB, int32_t &shiftC, int32_t &shiftD, int32_t &shiftE,
    // NR inputs (only accessed when EnableNewtonRaphson; pass nullptr otherwise)
    const uint32_t *SharkRestrict extNR0 = nullptr,
    const uint32_t *SharkRestrict extNR1 = nullptr,
    const uint32_t *SharkRestrict extNR2 = nullptr,
    const uint32_t *SharkRestrict extNR3 = nullptr,
    const uint32_t *SharkRestrict extNR4 = nullptr,
    int32_t *expNR0 = nullptr, int32_t *expNR1 = nullptr,
    int32_t *expNR2 = nullptr, int32_t *expNR3 = nullptr, int32_t *expNR4 = nullptr,
    bool *isZeroNR0 = nullptr, bool *isZeroNR1 = nullptr,
    bool *isZeroNR2 = nullptr, bool *isZeroNR3 = nullptr, bool *isZeroNR4 = nullptr,
    int32_t *shiftNR0 = nullptr, int32_t *shiftNR1 = nullptr,
    int32_t *shiftNR2 = nullptr, int32_t *shiftNR3 = nullptr, int32_t *shiftNR4 = nullptr
) {
    int32_t msdA = -1, msdB = -1, msdC = -1, msdD = -1, msdE = -1;
    int32_t msdNR0 = -1, msdNR1 = -1, msdNR2 = -1, msdNR3 = -1, msdNR4 = -1;

    const int32_t topIdx = actualDigits - 1;

    // ---- Tier 1: Quick top-limb check ----
    // NTT multiply normalizes MSB near the top, so Digits[N-1] is almost always non-zero.
    if (extA[topIdx] != 0) msdA = topIdx;
    if (extB[topIdx] != 0) msdB = topIdx;
    if (extC[topIdx] != 0) msdC = topIdx;
    if (extD[topIdx] != 0) msdD = topIdx;
    if (extE[topIdx] != 0) msdE = topIdx;

    if constexpr (SharkFloatParams::EnableNewtonRaphson) {
        if (extNR0[topIdx] != 0) msdNR0 = topIdx;
        if (extNR1[topIdx] != 0) msdNR1 = topIdx;
        if (extNR2[topIdx] != 0) msdNR2 = topIdx;
        if (extNR3[topIdx] != 0) msdNR3 = topIdx;
        if (extNR4[topIdx] != 0) msdNR4 = topIdx;
    }

    bool allFound = (msdA >= 0 && msdB >= 0 && msdC >= 0 && msdD >= 0 && msdE >= 0);
    if constexpr (SharkFloatParams::EnableNewtonRaphson) {
        allFound = allFound &&
            (msdNR0 >= 0 && msdNR1 >= 0 && msdNR2 >= 0 && msdNR3 >= 0 && msdNR4 >= 0);
    }

    // ---- Tier 2: Warp-parallel scan for any remaining unfound MSDs ----
    if (!allFound) {
        const int lane = threadIdx.x & 31;
        const unsigned fullMask = 0xffff'ffff;
        const int32_t numTiles = (actualDigits + 31) / 32;

        for (int32_t tile = 0; tile < numTiles; ++tile) {
            const int32_t idxFromTop = tile * 32 + lane;
            const int32_t i = topIdx - idxFromTop;
            const bool inBounds = (i >= 0);

            // Load one limb per lane; zero if MSD already found or out of bounds
            const uint32_t lA = (inBounds && msdA < 0) ? extA[i] : 0;
            const uint32_t lB = (inBounds && msdB < 0) ? extB[i] : 0;
            const uint32_t lC = (inBounds && msdC < 0) ? extC[i] : 0;
            const uint32_t lD = (inBounds && msdD < 0) ? extD[i] : 0;
            const uint32_t lE = (inBounds && msdE < 0) ? extE[i] : 0;

            // Always call __ballot_sync (all lanes must participate).
            // Only update MSD from the result if not yet found.
            const unsigned maskA = __ballot_sync(fullMask, lA != 0);
            const unsigned maskB = __ballot_sync(fullMask, lB != 0);
            const unsigned maskC = __ballot_sync(fullMask, lC != 0);
            const unsigned maskD = __ballot_sync(fullMask, lD != 0);
            const unsigned maskE = __ballot_sync(fullMask, lE != 0);

            if (msdA < 0 && maskA != 0) msdA = topIdx - (tile * 32 + (__ffs(maskA) - 1));
            if (msdB < 0 && maskB != 0) msdB = topIdx - (tile * 32 + (__ffs(maskB) - 1));
            if (msdC < 0 && maskC != 0) msdC = topIdx - (tile * 32 + (__ffs(maskC) - 1));
            if (msdD < 0 && maskD != 0) msdD = topIdx - (tile * 32 + (__ffs(maskD) - 1));
            if (msdE < 0 && maskE != 0) msdE = topIdx - (tile * 32 + (__ffs(maskE) - 1));

            allFound = (msdA >= 0 && msdB >= 0 && msdC >= 0 && msdD >= 0 && msdE >= 0);

            if constexpr (SharkFloatParams::EnableNewtonRaphson) {
                const uint32_t lNR0 = (inBounds && msdNR0 < 0) ? extNR0[i] : 0;
                const uint32_t lNR1 = (inBounds && msdNR1 < 0) ? extNR1[i] : 0;
                const uint32_t lNR2 = (inBounds && msdNR2 < 0) ? extNR2[i] : 0;
                const uint32_t lNR3 = (inBounds && msdNR3 < 0) ? extNR3[i] : 0;
                const uint32_t lNR4 = (inBounds && msdNR4 < 0) ? extNR4[i] : 0;

                const unsigned mNR0 = __ballot_sync(fullMask, lNR0 != 0);
                const unsigned mNR1 = __ballot_sync(fullMask, lNR1 != 0);
                const unsigned mNR2 = __ballot_sync(fullMask, lNR2 != 0);
                const unsigned mNR3 = __ballot_sync(fullMask, lNR3 != 0);
                const unsigned mNR4 = __ballot_sync(fullMask, lNR4 != 0);

                if (msdNR0 < 0 && mNR0 != 0) msdNR0 = topIdx - (tile * 32 + (__ffs(mNR0) - 1));
                if (msdNR1 < 0 && mNR1 != 0) msdNR1 = topIdx - (tile * 32 + (__ffs(mNR1) - 1));
                if (msdNR2 < 0 && mNR2 != 0) msdNR2 = topIdx - (tile * 32 + (__ffs(mNR2) - 1));
                if (msdNR3 < 0 && mNR3 != 0) msdNR3 = topIdx - (tile * 32 + (__ffs(mNR3) - 1));
                if (msdNR4 < 0 && mNR4 != 0) msdNR4 = topIdx - (tile * 32 + (__ffs(mNR4) - 1));

                allFound = allFound &&
                    (msdNR0 >= 0 && msdNR1 >= 0 && msdNR2 >= 0 && msdNR3 >= 0 && msdNR4 >= 0);
            }

            if (allFound) break;
        }
    }

    isZeroA = (msdA < 0);
    isZeroB = (msdB < 0);
    isZeroC = (msdC < 0);
    isZeroD = (msdD < 0);
    isZeroE = (msdE < 0);

    const int32_t totalExtBits = numActualDigitsPlusGuard * 32;
    auto computeShift = [&](const uint32_t *ext, int32_t msd, bool isz, int32_t &exp, int32_t &outShift) {
        if (isz) {
            outShift = 0;
        } else {
            int32_t limb = GetExtLimb(ext, actualDigits, numActualDigitsPlusGuard, msd);
            int32_t clz = CountLeadingZeros(static_cast<uint32_t>(limb));
            int32_t current_msb = msd * 32 + (31 - clz);
            int32_t L = (totalExtBits - 1) - current_msb;
            exp -= L;
            outShift = L;
        }
    };

    computeShift(extA, msdA, isZeroA, expA, shiftA);
    computeShift(extB, msdB, isZeroB, expB, shiftB);
    computeShift(extC, msdC, isZeroC, expC, shiftC);
    computeShift(extD, msdD, isZeroD, expD, shiftD);
    computeShift(extE, msdE, isZeroE, expE, shiftE);

    if constexpr (SharkFloatParams::EnableNewtonRaphson) {
        *isZeroNR0 = (msdNR0 < 0);
        *isZeroNR1 = (msdNR1 < 0);
        *isZeroNR2 = (msdNR2 < 0);
        *isZeroNR3 = (msdNR3 < 0);
        *isZeroNR4 = (msdNR4 < 0);

        computeShift(extNR0, msdNR0, *isZeroNR0, *expNR0, *shiftNR0);
        computeShift(extNR1, msdNR1, *isZeroNR1, *expNR1, *shiftNR1);
        computeShift(extNR2, msdNR2, *isZeroNR2, *expNR2, *shiftNR2);
        computeShift(extNR3, msdNR3, *isZeroNR3, *expNR3, *shiftNR3);
        computeShift(extNR4, msdNR4, *isZeroNR4, *expNR4, *shiftNR4);
    }
}

// New helper: Computes the aligned digit for the normalized value on the fly.
// 'diffDE' is the additional right shift required for alignment.
template <class SharkFloatParams>
static __device__ SharkForceInlineReleaseOnly uint32_t
GetShiftedNormalizedDigit (
    const uint32_t *SharkRestrict ext,
    const int32_t actualDigits,
    const int32_t numActualDigitsPlusGuard,
    const int32_t shiftOffset,
    const int32_t diff,
    const int32_t idx) {
    // const int32_t n = SharkFloatParams::GlobalNumUint32; // normalized length
    const int32_t wordShift = diff / 32;
    const int32_t bitShift = diff % 32;
    const bool inBounds = (idx + wordShift < numActualDigitsPlusGuard && idx + wordShift >= 0);
    const uint32_t lower = inBounds ?
        GetNormalizedDigit(ext, actualDigits, numActualDigitsPlusGuard, shiftOffset, idx + wordShift) : 0;
    const bool inBoundsPlus1 = (idx + wordShift + 1 < numActualDigitsPlusGuard && idx + wordShift + 1 >= 0);
    const uint32_t upper = inBoundsPlus1 ?
        GetNormalizedDigit(ext, actualDigits, numActualDigitsPlusGuard, shiftOffset, idx + wordShift + 1) : 0;
    if (bitShift == 0)
        return lower;
    else
        return (lower >> bitShift) | (upper << (32 - bitShift));
}


static __device__ SharkForceInlineReleaseOnly bool
CompareMagnitudes2Way (
    const int32_t effExpA,
    const int32_t effExpB,
    const int32_t actualDigits,
    const int32_t numActualDigitsPlusGuard,
    const int32_t shiftA,
    const int32_t shiftB,
    const uint32_t *SharkRestrict extA,
    const uint32_t *SharkRestrict extB) {
    bool AIsBiggerMagnitude;

    if (effExpA > effExpB) {
        AIsBiggerMagnitude = true;
    } else if (effExpA < effExpB) {
        AIsBiggerMagnitude = false;
    } else {
        AIsBiggerMagnitude = false; // default if equal
        for (int32_t i = numActualDigitsPlusGuard - 1; i >= 0; i--) {
            uint32_t digitA = GetNormalizedDigit(extA, actualDigits, numActualDigitsPlusGuard, shiftA, i);
            uint32_t digitB = GetNormalizedDigit(extB, actualDigits, numActualDigitsPlusGuard, shiftB, i);
            if (digitA > digitB) {
                AIsBiggerMagnitude = true;
                break;
            } else if (digitA < digitB) {
                AIsBiggerMagnitude = false;
                break;
            }
        }
    }

    return AIsBiggerMagnitude;
}

template<class SharkFloatParams>
static __device__ SharkForceInlineReleaseOnly void
GetCorrespondingLimbs (
    const uint32_t *SharkRestrict ext_A_X2,
    const int32_t actualASize,
    const int32_t extASize,
    const uint32_t *SharkRestrict ext_B_Y2,
    const int32_t actualBSize,
    const int32_t extBSize,
    const int32_t shiftALeftToGetMsb,
    const int32_t shiftBLeftToGetMsb,
    const bool AIsBiggerMagnitude,
    const int32_t diff,
    const int32_t index,
    uint64_t &alignedA,
    uint64_t &alignedB) {
    if (AIsBiggerMagnitude) {
        // A is larger: normalized A is used as is.
        // For B, we normalize and then shift right by 'diff'.
        alignedA = GetNormalizedDigit(ext_A_X2, actualASize, extASize, shiftALeftToGetMsb, index);
        alignedB = GetShiftedNormalizedDigit<SharkFloatParams>(
            ext_B_Y2,
            actualBSize,
            extBSize,
            shiftBLeftToGetMsb,
            diff,
            index);
    } else {
        // B is larger: normalized B is used as is.
        // For A, we normalize and shift right by 'diff'.
        alignedB = GetNormalizedDigit(ext_B_Y2, actualBSize, extBSize, shiftBLeftToGetMsb, index);
        alignedA = GetShiftedNormalizedDigit<SharkFloatParams>(
            ext_A_X2,
            actualASize,
            extASize,
            shiftALeftToGetMsb,
            diff,
            index);
    }
}

template<
    class SharkFloatParams,
    DebugStatePurpose Purpose>
static __device__ SharkForceInlineReleaseOnly void
EraseCurrentDebugStateAdd (
    DebugState<SharkFloatParams> *SharkRestrict debugStates,
    cooperative_groups::grid_group &grid,
    cooperative_groups::thread_block &block) {

    constexpr auto RecursionDepth = 0;
    constexpr auto CallIndex = 0;
    constexpr auto curPurpose = static_cast<int32_t>(Purpose);
    debugStates[curPurpose].Erase(
        grid, block, Purpose, RecursionDepth, CallIndex);
}

template<
    class SharkFloatParams,
    DebugStatePurpose Purpose,
    typename ArrayType>
static __device__ SharkForceInlineReleaseOnly void
StoreCurrentDebugStateAdd (
    DebugState<SharkFloatParams> *SharkRestrict debugStates,
    cooperative_groups::grid_group &grid,
    cooperative_groups::thread_block &block,
    const ArrayType *arrayToChecksum,
    size_t arraySize) {

    constexpr auto CurPurpose = static_cast<int32_t>(Purpose);
    constexpr auto RecursionDepth = 0;
    constexpr auto UseConvolutionHere = UseConvolution::No;
    constexpr auto CallIndex = 0;

    debugStates[CurPurpose].Reset(
        UseConvolutionHere, grid, block, arrayToChecksum, arraySize, Purpose, RecursionDepth, CallIndex);
}

#include "Add_ABC.h"
#include "Add_DE.h"

static __device__ inline void
find_msd_warp_ABC_DE(const uint64_t *extABC,
                     const uint64_t *extDE,
                     int32_t N,     // highest valid digit index (0..N)
                     int32_t &msdA, // out
                     int32_t &msdD) // out
{
    const int lane = threadIdx.x & 31;
    const unsigned fullMask = 0xffff'ffff;

    msdA = 0;
    msdD = 0;

    const int32_t totalDigits = N + 1; // indices 0..N inclusive
    const int32_t numTiles = (totalDigits + 31) / 32;

    // Scan from top: tile 0 is highest 32 digits, tile 1 next 32, etc.
    for (int32_t tile = 0; tile < numTiles && (msdA == 0 || msdD == 0); ++tile) {
        const int32_t idxFromTop = tile * 32 + lane;
        const int32_t i = N - idxFromTop;

        uint32_t a = 0u, d = 0u;
        if (i >= 0) {
            a = static_cast<uint32_t>(extABC[i]);
            d = static_cast<uint32_t>(extDE[i]);
        }

        const unsigned maskA = __ballot_sync(fullMask, a != 0u);
        const unsigned maskD = __ballot_sync(fullMask, d != 0u);

        // Use LSB (smallest lane index) – matches N..0 descent.
        if (msdA == 0 && maskA != 0u) {
            const int lsbLaneA = __ffs(maskA) - 1; // 0..31
            msdA = N - (tile * 32 + lsbLaneA);
        }

        if (msdD == 0 && maskD != 0u) {
            const int lsbLaneD = __ffs(maskD) - 1;
            msdD = N - (tile * 32 + lsbLaneD);
        }
    }
}


template<class SharkFloatParams>
static __device__ SharkForceInlineReleaseOnly void
NormalizeAndCopyResult(
    const int stride,
    const int32_t                      idx,
    const int32_t                      actualDigits,    // = SharkFloatParams::GlobalNumUint32
    const int32_t                      numActualDigitsPlusGuard, // = SharkFloatParams::GlobalNumUint32 + SharkFloatParams::Guard

    //  6) A−B+C exponent
    int32_t &outExponentABC,
    //  7) D+E exponent
    int32_t &outExponentDE,

    //  8) A−B+C carry
    int32_t &carryABC,
    //  9) D+E carry
    int32_t &carryDE,

    // 10) raw 64‐bit limbs from Phase1_ABC (true or false branch)
    auto *SharkRestrict extABC,
    // 11) raw 64‐bit limbs from Phase1_DE
    auto *SharkRestrict extDE,

    // 12) output HpSharkFloat for A−B+C
    HpSharkFloat<SharkFloatParams> *OutABC,
    // 13) output HpSharkFloat for D+E
    HpSharkFloat<SharkFloatParams> *OutDE,

    // 14) sign for A−B+C
    const bool                         outSignABC,
    // 15) sign for D+E
    const bool                         outSignDE
) noexcept {
    // compile‐time sizes
    constexpr int N = SharkFloatParams::GlobalNumUint32 + SharkFloatParams::Guard;
    constexpr int Np1 = N + 1;

    // 1) inject carries into the guard slot at index N
    extABC[N] = static_cast<uint32_t>(carryABC);
    extDE[N] = static_cast<uint32_t>(carryDE);

#ifdef TEST_SMALL_NORMALIZE_WARP
    static constexpr bool OriginalImpl = true;
#else
    static constexpr bool OriginalImpl = false;
#endif

    int32_t msdA = 0, msdD = 0;
    if constexpr (OriginalImpl) {
        // 2) find MSB index in each (all threads may do it, but it's cheap)
        for (int32_t i = N; i >= 0; --i) {
            // cast down to 32 bits — that’s exactly what FunnelShift32 will see
            uint32_t a = static_cast<uint32_t>(extABC[i]);
            uint32_t d = static_cast<uint32_t>(extDE[i]);

            if (msdA == 0 && a != 0u)
                msdA = i;
            if (msdD == 0 && d != 0u)
                msdD = i;

            // once *both* are non-zero, we’re done
            if (msdA != 0 && msdD != 0)
                break;
        }
    } else {
        // New parallel MSB finder
        find_msd_warp_ABC_DE(
            extABC,
            extDE,
            N,
            msdA,
            msdD);
    }

    // 3) compute total shiftNeeded for each and update exponents
    auto calcShift = [&](auto *w, int32_t msd, int32_t &exp) {
        uint32_t v = w[msd];
        int      c = CountLeadingZeros(v);
        int      curr = msd * 32 + (31 - c);
        int      des = (SharkFloatParams::GlobalNumUint32 - 1) * 32 + 31;
        int      s = curr - des;
        exp += s;
        return s;
    };

    int shiftA = calcShift(extABC, msdA, outExponentABC);
    int shiftD = calcShift(extDE, msdD, outExponentDE);

    // 4) funnel‐shift each extended array into its 32‐bit Digits
    if (shiftA > 0) {
        MultiWordShift<Dir::Right>(stride, idx,
            extABC, Np1, shiftA,
            OutABC->Digits, actualDigits
        );
    } else if (shiftA < 0) {
        MultiWordShift<Dir::Left>(stride, idx,
            extABC, Np1, -shiftA,
            OutABC->Digits, actualDigits
        );
    } else {
        MultiWordShift<Dir::Left>(stride, idx,
            extABC, Np1, 0,
            OutABC->Digits, actualDigits
        );
    }

    if (shiftD > 0) {
        MultiWordShift<Dir::Right>(stride, idx,
            extDE, Np1, shiftD,
            OutDE->Digits, actualDigits
        );
    } else if (shiftD < 0) {
        MultiWordShift<Dir::Left>(stride, idx,
            extDE, Np1, -shiftD,
            OutDE->Digits, actualDigits
        );
    } else {
        MultiWordShift<Dir::Left>(stride, idx,
            extDE, Np1, 0,
            OutDE->Digits, actualDigits
        );
    }

    // 5) write back exponent & sign on thread 0
    if (idx == 0) {
        OutABC->Exponent = outExponentABC;
        OutABC->SetNegative(outSignABC);
        OutDE->Exponent = outExponentDE;
        OutDE->SetNegative(outSignDE);
    }
}


template <class SharkFloatParams>
static __device__ void AddHelperSeparates(
    cg::grid_group &grid,
    cg::thread_block &block,
    const HpSharkFloat<SharkFloatParams> *A_X2,
    const HpSharkFloat<SharkFloatParams> *B_Y2,
    const HpSharkFloat<SharkFloatParams> *C_A,
    const HpSharkFloat<SharkFloatParams> *D_2X,
    const HpSharkFloat<SharkFloatParams> *E_B,
    HpSharkFloat<SharkFloatParams> *Out_A_B_C,
    HpSharkFloat<SharkFloatParams> *Out_D_E,
    uint64_t *tempData,
    // NR params (only used when EnableNewtonRaphson)
    const HpSharkFloat<SharkFloatParams> *W0 = nullptr,
    const HpSharkFloat<SharkFloatParams> *W1 = nullptr,
    const HpSharkFloat<SharkFloatParams> *W2 = nullptr,
    const HpSharkFloat<SharkFloatParams> *W3 = nullptr,
    const HpSharkFloat<SharkFloatParams> *One = nullptr,
    HpSharkFloat<SharkFloatParams> *Out_DzdcReal = nullptr,
    HpSharkFloat<SharkFloatParams> *Out_DzdcImag = nullptr)
{
    extern __shared__ uint64_t shared_data[];

    // --- Constants and Parameters ---
    constexpr int32_t guard = SharkFloatParams::Guard;
    constexpr int32_t numActualDigits = SharkFloatParams::GlobalNumUint32;
    constexpr int32_t numActualDigitsPlusGuard = SharkFloatParams::GlobalNumUint32 + guard;
    const int32_t idx = block.thread_index().x + block.group_index().x * block.dim_threads().x;
    constexpr int32_t NewN = SharkFloatParams::GlobalNumUint32;

    const auto *ext_A_X2 = A_X2->Digits;
    const auto *ext_B_Y2 = B_Y2->Digits;
    const auto *ext_C_A = C_A->Digits;
    const auto *ext_D_2X = D_2X->Digits;
    const auto *ext_E_B = E_B->Digits;

    const bool IsNegativeA = A_X2->GetNegative();
    const bool IsNegativeB = !B_Y2->GetNegative(); // A - B + C
    const bool IsNegativeC = C_A->GetNegative();
    const bool IsNegativeD = D_2X->GetNegative();
    const bool IsNegativeE = E_B->GetNegative();

    constexpr auto GlobalSync1_offset = 0;
    constexpr auto GlobalSync2_offset = 128 / sizeof(uint64_t);
    //constexpr auto GlobalSync3_offset = GlobalSync2_offset + 128 / sizeof(uint64_t);
    constexpr auto DebugGlobals_offset = HpShark::AdditionalGlobalSyncSpace;
    constexpr auto DebugChecksum_offset = DebugGlobals_offset + HpShark::AdditionalGlobalDebugPerThread;
    constexpr auto Final128Offset_ABC_True = HpShark::AdditionalUInt64Global;
    constexpr auto Final128Offset_ABC_False = Final128Offset_ABC_True + 2 * SharkFloatParams::GlobalNumUint32;
    constexpr auto Final128Offset_DE = Final128Offset_ABC_False + 2 * SharkFloatParams::GlobalNumUint32;
    constexpr auto PropagatedFinal128Offset_ABC_True = Final128Offset_DE + 2 * SharkFloatParams::GlobalNumUint32;
    constexpr auto PropagatedFinal128Offset_ABC_False = PropagatedFinal128Offset_ABC_True + 2 * SharkFloatParams::GlobalNumUint32;
    constexpr auto PropagatedFinal128Offset_DE = PropagatedFinal128Offset_ABC_False + 2 * SharkFloatParams::GlobalNumUint32;
    constexpr auto Carry1_offset = PropagatedFinal128Offset_DE + 2 * SharkFloatParams::GlobalNumUint32;
    constexpr auto Carry2_offset = Carry1_offset + 4 * SharkFloatParams::GlobalNumUint32;
    constexpr auto Carry3_offset = Carry2_offset + 4 * SharkFloatParams::GlobalNumUint32;
    constexpr auto Carry4_offset = Carry3_offset + 4 * SharkFloatParams::GlobalNumUint32;
    constexpr auto Carry5_offset = Carry4_offset + 4 * SharkFloatParams::GlobalNumUint32;
    constexpr auto Carry6_offset = Carry5_offset + 4 * SharkFloatParams::GlobalNumUint32;
    constexpr auto TotalGlobalNumUint32Multiple = 2 + 2 + 2 + 2 + 2 + 2 + 6 * 4;
    static_assert(TotalGlobalNumUint32Multiple == 36);

    // NR scratch offsets — separate buffers for NR, following orbit buffers
    constexpr auto NR_Final128_ABC_True = Carry6_offset + 4 * NewN;
    constexpr auto NR_Final128_ABC_False = NR_Final128_ABC_True + 2 * NewN;
    constexpr auto NR_Final128_DE = NR_Final128_ABC_False + 2 * NewN;
    constexpr auto NR_Prop_ABC_True = NR_Final128_DE + 2 * NewN;
    constexpr auto NR_Prop_ABC_False = NR_Prop_ABC_True + 2 * NewN;
    constexpr auto NR_Prop_DE = NR_Prop_ABC_False + 2 * NewN;
    constexpr auto NR_Carry1 = NR_Prop_DE + 2 * NewN;
    constexpr auto NR_Carry2 = NR_Carry1 + 4 * NewN;
    constexpr auto NR_Carry3 = NR_Carry2 + 4 * NewN;
    constexpr auto NR_Carry4 = NR_Carry3 + 4 * NewN;
    constexpr auto NR_Carry5 = NR_Carry4 + 4 * NewN;
    constexpr auto NR_Carry6 = NR_Carry5 + 4 * NewN;

    auto *SharkRestrict globalSync1 =
        reinterpret_cast<uint32_t *>(&tempData[GlobalSync1_offset]);
    auto *SharkRestrict globalSync2 =
        reinterpret_cast<uint32_t *>(&tempData[GlobalSync2_offset]);
    //auto *SharkRestrict globalSync3 =
    //    reinterpret_cast<uint32_t *>(&tempData[GlobalSync3_offset]);
    auto *SharkRestrict debugStates =
        reinterpret_cast<DebugState<SharkFloatParams>*>(&tempData[DebugChecksum_offset]);
    auto *SharkRestrict debugGlobalState =
        reinterpret_cast<DebugGlobalCount<SharkFloatParams>*>(&tempData[DebugGlobals_offset]);
    auto *SharkRestrict extResultTrue =
        reinterpret_cast<uint64_t *>(&tempData[Final128Offset_ABC_True]);
    auto *SharkRestrict extResultFalse =
        reinterpret_cast<uint64_t *>(&tempData[Final128Offset_ABC_False]);
    auto *SharkRestrict final128_DE =
        reinterpret_cast<uint64_t *>(&tempData[Final128Offset_DE]);
    auto *SharkRestrict extResultTrue32 =
        reinterpret_cast<uint64_t *>(&tempData[PropagatedFinal128Offset_ABC_True]);
    auto *SharkRestrict extResultFalse32 =
        reinterpret_cast<uint64_t *>(&tempData[PropagatedFinal128Offset_ABC_False]);
    auto *SharkRestrict final128_DE32 =
        reinterpret_cast<uint64_t *>(&tempData[PropagatedFinal128Offset_DE]);
    auto *SharkRestrict carry1 =
        reinterpret_cast<uint32_t *>(&tempData[Carry1_offset]);
    auto *SharkRestrict carry2 =
        reinterpret_cast<uint32_t *>(&tempData[Carry2_offset]);
    auto *SharkRestrict carry3 =
        reinterpret_cast<uint32_t *>(&tempData[Carry3_offset]);
    auto *SharkRestrict carry4 =
        reinterpret_cast<uint32_t *>(&tempData[Carry4_offset]);
    auto *SharkRestrict carry5 =
        reinterpret_cast<uint32_t *>(&tempData[Carry5_offset]);
    auto *SharkRestrict carry6 =
        reinterpret_cast<uint32_t *>(&tempData[Carry6_offset]);

    // NR scratch pointers — only assigned when NR is enabled
    uint64_t *nrExtResultTrue = nullptr;
    uint64_t *nrExtResultFalse = nullptr;
    uint64_t *nrFinal128_DE = nullptr;
    uint64_t *nrExtResultTrue32 = nullptr;
    uint64_t *nrExtResultFalse32 = nullptr;
    uint64_t *nrFinal128_DE32 = nullptr;
    uint32_t *nrCarry1 = nullptr, *nrCarry2 = nullptr, *nrCarry3 = nullptr;
    uint32_t *nrCarry4 = nullptr, *nrCarry5 = nullptr, *nrCarry6 = nullptr;

    if constexpr (SharkFloatParams::EnableNewtonRaphson) {
        nrExtResultTrue = reinterpret_cast<uint64_t *>(&tempData[NR_Final128_ABC_True]);
        nrExtResultFalse = reinterpret_cast<uint64_t *>(&tempData[NR_Final128_ABC_False]);
        nrFinal128_DE = reinterpret_cast<uint64_t *>(&tempData[NR_Final128_DE]);
        nrExtResultTrue32 = reinterpret_cast<uint64_t *>(&tempData[NR_Prop_ABC_True]);
        nrExtResultFalse32 = reinterpret_cast<uint64_t *>(&tempData[NR_Prop_ABC_False]);
        nrFinal128_DE32 = reinterpret_cast<uint64_t *>(&tempData[NR_Prop_DE]);
        nrCarry1 = reinterpret_cast<uint32_t *>(&tempData[NR_Carry1]);
        nrCarry2 = reinterpret_cast<uint32_t *>(&tempData[NR_Carry2]);
        nrCarry3 = reinterpret_cast<uint32_t *>(&tempData[NR_Carry3]);
        nrCarry4 = reinterpret_cast<uint32_t *>(&tempData[NR_Carry4]);
        nrCarry5 = reinterpret_cast<uint32_t *>(&tempData[NR_Carry5]);
        nrCarry6 = reinterpret_cast<uint32_t *>(&tempData[NR_Carry6]);
    }

    static constexpr auto CallIndex = 0;

    if constexpr (HpShark::DebugGlobalState) {
        const auto CurBlock = block.group_index().x;
        const auto CurThread = block.thread_index().x;
        const auto ThreadsPerBlock = block.dim_threads().x;
        debugGlobalState[CurBlock * ThreadsPerBlock + CurThread]
            .DebugMultiplyErase();
    }

    if constexpr (HpShark::DebugChecksums) {
        EraseCurrentDebugStateAdd<SharkFloatParams, DebugStatePurpose::Invalid>(debugStates, grid, block);
        EraseCurrentDebugStateAdd<SharkFloatParams, DebugStatePurpose::ADigits>(debugStates, grid, block);
        EraseCurrentDebugStateAdd<SharkFloatParams, DebugStatePurpose::BDigits>(debugStates, grid, block);
        EraseCurrentDebugStateAdd<SharkFloatParams, DebugStatePurpose::CDigits>(debugStates, grid, block);
        EraseCurrentDebugStateAdd<SharkFloatParams, DebugStatePurpose::DDigits>(debugStates, grid, block);
        EraseCurrentDebugStateAdd<SharkFloatParams, DebugStatePurpose::EDigits>(debugStates, grid, block);
        EraseCurrentDebugStateAdd<SharkFloatParams, DebugStatePurpose::AHalfHigh>(debugStates, grid, block);
        EraseCurrentDebugStateAdd<SharkFloatParams, DebugStatePurpose::AHalfLow>(debugStates, grid, block);
        EraseCurrentDebugStateAdd<SharkFloatParams, DebugStatePurpose::BHalfHigh>(debugStates, grid, block);
        EraseCurrentDebugStateAdd<SharkFloatParams, DebugStatePurpose::BHalfLow>(debugStates, grid, block);
        EraseCurrentDebugStateAdd<SharkFloatParams, DebugStatePurpose::XDiff>(debugStates, grid, block);
        EraseCurrentDebugStateAdd<SharkFloatParams, DebugStatePurpose::YDiff>(debugStates, grid, block);
        EraseCurrentDebugStateAdd<SharkFloatParams, DebugStatePurpose::Z0XX>(debugStates, grid, block);
        EraseCurrentDebugStateAdd<SharkFloatParams, DebugStatePurpose::Z0XY>(debugStates, grid, block);
        EraseCurrentDebugStateAdd<SharkFloatParams, DebugStatePurpose::Z0YY>(debugStates, grid, block);
        EraseCurrentDebugStateAdd<SharkFloatParams, DebugStatePurpose::Z1XX>(debugStates, grid, block);
        EraseCurrentDebugStateAdd<SharkFloatParams, DebugStatePurpose::Z1XY>(debugStates, grid, block);
        EraseCurrentDebugStateAdd<SharkFloatParams, DebugStatePurpose::Z1YY>(debugStates, grid, block);
        EraseCurrentDebugStateAdd<SharkFloatParams, DebugStatePurpose::Z2XX>(debugStates, grid, block);
        EraseCurrentDebugStateAdd<SharkFloatParams, DebugStatePurpose::Z2XY>(debugStates, grid, block);
        EraseCurrentDebugStateAdd<SharkFloatParams, DebugStatePurpose::Z2YY>(debugStates, grid, block);
        EraseCurrentDebugStateAdd<SharkFloatParams, DebugStatePurpose::Z3XX>(debugStates, grid, block);
        EraseCurrentDebugStateAdd<SharkFloatParams, DebugStatePurpose::Z3XY>(debugStates, grid, block);
        EraseCurrentDebugStateAdd<SharkFloatParams, DebugStatePurpose::Z3YY>(debugStates, grid, block);
        EraseCurrentDebugStateAdd<SharkFloatParams, DebugStatePurpose::Z4XX>(debugStates, grid, block);
        EraseCurrentDebugStateAdd<SharkFloatParams, DebugStatePurpose::Z4XY>(debugStates, grid, block);
        EraseCurrentDebugStateAdd<SharkFloatParams, DebugStatePurpose::Z4YY>(debugStates, grid, block);
        EraseCurrentDebugStateAdd<SharkFloatParams, DebugStatePurpose::Z2_Perm1>(debugStates, grid, block);
        EraseCurrentDebugStateAdd<SharkFloatParams, DebugStatePurpose::Z2_Perm2>(debugStates, grid, block);
        EraseCurrentDebugStateAdd<SharkFloatParams, DebugStatePurpose::Z2_Perm3>(debugStates, grid, block);
        EraseCurrentDebugStateAdd<SharkFloatParams, DebugStatePurpose::Z2_Perm4>(debugStates, grid, block);
        EraseCurrentDebugStateAdd<SharkFloatParams, DebugStatePurpose::Z2_Perm5>(debugStates, grid, block);
        EraseCurrentDebugStateAdd<SharkFloatParams, DebugStatePurpose::Z2_Perm6>(debugStates, grid, block);
        EraseCurrentDebugStateAdd<SharkFloatParams, DebugStatePurpose::Z1_offsetXX>(debugStates, grid, block);
        EraseCurrentDebugStateAdd<SharkFloatParams, DebugStatePurpose::Z1_offsetXY>(debugStates, grid, block);
        EraseCurrentDebugStateAdd<SharkFloatParams, DebugStatePurpose::Z1_offsetYY>(debugStates, grid, block);
        EraseCurrentDebugStateAdd<SharkFloatParams, DebugStatePurpose::Final128XX>(debugStates, grid, block);
        EraseCurrentDebugStateAdd<SharkFloatParams, DebugStatePurpose::Final128XY>(debugStates, grid, block);
        EraseCurrentDebugStateAdd<SharkFloatParams, DebugStatePurpose::Final128YY>(debugStates, grid, block);
        EraseCurrentDebugStateAdd<SharkFloatParams, DebugStatePurpose::FinalAdd1>(debugStates, grid, block);
        EraseCurrentDebugStateAdd<SharkFloatParams, DebugStatePurpose::FinalAdd2>(debugStates, grid, block);
        EraseCurrentDebugStateAdd<SharkFloatParams, DebugStatePurpose::FinalAdd3>(debugStates, grid, block);
        EraseCurrentDebugStateAdd<SharkFloatParams, DebugStatePurpose::Result_offsetXX>(debugStates, grid, block);
        EraseCurrentDebugStateAdd<SharkFloatParams, DebugStatePurpose::Result_offsetXY>(debugStates, grid, block);
        EraseCurrentDebugStateAdd<SharkFloatParams, DebugStatePurpose::Result_offsetYY>(debugStates, grid, block);
        EraseCurrentDebugStateAdd<SharkFloatParams, DebugStatePurpose::Result_Add1>(debugStates, grid, block);
        EraseCurrentDebugStateAdd<SharkFloatParams, DebugStatePurpose::Result_Add2>(debugStates, grid, block);
        static_assert(static_cast<int32_t>(DebugStatePurpose::NumPurposes) == 87, "Unexpected number of purposes");

        StoreCurrentDebugStateAdd<SharkFloatParams, DebugStatePurpose::ADigits, uint32_t>(
            debugStates, grid, block, A_X2->Digits, NewN);
        StoreCurrentDebugStateAdd<SharkFloatParams, DebugStatePurpose::BDigits, uint32_t>(
            debugStates, grid, block, B_Y2->Digits, NewN);
        StoreCurrentDebugStateAdd<SharkFloatParams, DebugStatePurpose::CDigits, uint32_t>(
            debugStates, grid, block, C_A->Digits, NewN);
        StoreCurrentDebugStateAdd<SharkFloatParams, DebugStatePurpose::DDigits, uint32_t>(
            debugStates, grid, block, D_2X->Digits, NewN);
        StoreCurrentDebugStateAdd<SharkFloatParams, DebugStatePurpose::EDigits, uint32_t>(
            debugStates, grid, block, E_B->Digits, NewN);

        grid.sync();
    }

    // --- Extended Normalization using shift indices ---
    const bool sameSign = (D_2X->GetNegative() == E_B->GetNegative());

    bool normA_isZero = false;
    bool normB_isZero = false;
    bool normC_isZero = false;
    bool normD_isZero = false;
    bool normE_isZero = false;

    int32_t newAExponent = A_X2->Exponent;
    int32_t newBExponent = B_Y2->Exponent;
    int32_t newCExponent = C_A->Exponent;
    int32_t newDExponent = D_2X->Exponent;
    int32_t newEExponent = E_B->Exponent;

    int32_t shiftALeftToGetMsb;
    int32_t shiftBLeftToGetMsb;
    int32_t shiftCLeftToGetMsb;
    int32_t shiftDLeftToGetMsb;
    int32_t shiftELeftToGetMsb;

    // NR variable declarations (always present, populated only for NR types)
    const uint32_t *ext_W0 = nullptr;
    const uint32_t *ext_W1 = nullptr;
    const uint32_t *ext_W2 = nullptr;
    const uint32_t *ext_W3 = nullptr;
    const uint32_t *ext_One = nullptr;
    bool nrNormW0Zero = false, nrNormW1Zero = false, nrNormOneZero = false;
    bool nrNormW2Zero = false, nrNormW3Zero = false;
    int32_t nrExpW0 = 0, nrExpW1 = 0, nrExpOne = 0, nrExpW2 = 0, nrExpW3 = 0;
    int32_t nrShiftW0 = 0, nrShiftW1 = 0, nrShiftOne = 0, nrShiftW2 = 0, nrShiftW3 = 0;

    if constexpr (SharkFloatParams::EnableNewtonRaphson) {
        ext_W0 = W0->Digits;
        ext_W1 = W1->Digits;
        ext_W2 = W2->Digits;
        ext_W3 = W3->Digits;
        ext_One = One->Digits;
        nrExpW0 = W0->Exponent;
        nrExpW1 = W1->Exponent;
        nrExpOne = One->Exponent;
        nrExpW2 = W2->Exponent;
        nrExpW3 = W3->Exponent;
    }

    // Single call: orbit + NR (NR arrays are nullptr/defaults for non-NR types,
    // and if constexpr inside the function skips NR processing)
    ExtendedNormalizeShiftIndexAll<SharkFloatParams>(
        ext_A_X2, ext_B_Y2, ext_C_A, ext_D_2X, ext_E_B,
        numActualDigits, numActualDigitsPlusGuard,
        newAExponent, newBExponent, newCExponent, newDExponent, newEExponent,
        normA_isZero, normB_isZero, normC_isZero, normD_isZero, normE_isZero,
        shiftALeftToGetMsb, shiftBLeftToGetMsb, shiftCLeftToGetMsb, shiftDLeftToGetMsb, shiftELeftToGetMsb,
        ext_One, ext_W1, ext_W0, ext_W2, ext_W3,
        &nrExpOne, &nrExpW1, &nrExpW0, &nrExpW2, &nrExpW3,
        &nrNormOneZero, &nrNormW1Zero, &nrNormW0Zero, &nrNormW2Zero, &nrNormW3Zero,
        &nrShiftOne, &nrShiftW1, &nrShiftW0, &nrShiftW2, &nrShiftW3);


    // --- Compute Effective Exponents ---
    const auto bias = (SharkFloatParams::GlobalNumUint32 * 32 - 32);
    const int32_t effExpA = normA_isZero ? -100'000'000 : newAExponent + bias;
    const int32_t effExpB = normB_isZero ? -100'000'000 : newBExponent + bias;
    const int32_t effExpC = normC_isZero ? -100'000'000 : newCExponent + bias;
    const int32_t effExpD = normD_isZero ? -100'000'000 : newDExponent + bias;
    const int32_t effExpE = normE_isZero ? -100'000'000 : newEExponent + bias;


    // --- Determine which operand has larger magnitude ---

    // Do a 3-way comparison of the other three operands.
    // We need to compare A_X2, B_Y2, and C_A.
    // The result is a 3-way ordering of the three operands.

    // A, B, C:
    const auto threeWayMagnitude = CompareMagnitudes3Way(
        effExpA,
        effExpB,
        effExpC,
        numActualDigits,
        numActualDigitsPlusGuard,
        shiftALeftToGetMsb,
        shiftBLeftToGetMsb,
        shiftCLeftToGetMsb,
        ext_A_X2,
        ext_B_Y2,
        ext_C_A);

    const bool DIsBiggerMagnitude = CompareMagnitudes2Way(
        effExpD,
        effExpE,
        numActualDigits,
        numActualDigitsPlusGuard,
        shiftDLeftToGetMsb,
        shiftELeftToGetMsb,
        ext_D_2X,
        ext_E_B);

    // --- NR normalization, effective exponents, and magnitude comparison ---
    // --- NR signs, effective exponents, and magnitude comparison ---
    bool nrIsNegW0 = false, nrIsNegW1 = false, nrIsNegOne = false;
    bool nrIsNegW2 = false, nrIsNegW3 = false;
    int32_t nrEffW0 = 0, nrEffW1 = 0, nrEffOne = 0, nrEffW2 = 0, nrEffW3 = 0;
    ThreeWayLargestOrdering nrThreeWay{};
    bool nrW2Bigger = false;

    if constexpr (SharkFloatParams::EnableNewtonRaphson) {
        // Signs: One - W1 + W0 = W0 - W1 + One (reordered so fallthrough
        // to C in CompareMagnitudes3Way selects W0, not tiny One)
        nrIsNegW0 = W0->GetNegative();         // C position (W0, as-is)
        nrIsNegW1 = !W1->GetNegative();        // B position (W1, subtracted)
        nrIsNegOne = One->GetNegative();        // A position (One, as-is)
        // Signs: W2 + W3 (DE pattern, unchanged)
        nrIsNegW2 = W2->GetNegative();
        nrIsNegW3 = W3->GetNegative();

        // NR effective exponents
        nrEffW0 = nrNormW0Zero ? -100'000'000 : nrExpW0 + bias;
        nrEffW1 = nrNormW1Zero ? -100'000'000 : nrExpW1 + bias;
        nrEffOne = nrNormOneZero ? -100'000'000 : nrExpOne + bias;
        nrEffW2 = nrNormW2Zero ? -100'000'000 : nrExpW2 + bias;
        nrEffW3 = nrNormW3Zero ? -100'000'000 : nrExpW3 + bias;

        // NR 3-way magnitude comparison: A=One, B=W1, C=W0
        nrThreeWay = CompareMagnitudes3Way(
            nrEffOne, nrEffW1, nrEffW0,
            numActualDigits, numActualDigitsPlusGuard,
            nrShiftOne, nrShiftW1, nrShiftW0,
            ext_One, ext_W1, ext_W0);

        // NR 2-way magnitude comparison for W2, W3
        nrW2Bigger = CompareMagnitudes2Way(
            nrEffW2, nrEffW3,
            numActualDigits, numActualDigitsPlusGuard,
            nrShiftW2, nrShiftW3,
            ext_W2, ext_W3);
    }

    // --- Phase 1: A - B + C ---
    int32_t outExponentTrue = 0;
    int32_t outExponentFalse = 0;

    bool outSignTrue = false;
    bool outSignFalse = false;

    bool nrSignTrue = false, nrSignFalse = false;
    int32_t nrOutExpTrue = 0, nrOutExpFalse = 0;

    Phase1_ABC<SharkFloatParams, CallIndex>(
        block,
        grid,
        idx,
        threeWayMagnitude,
        IsNegativeA,
        IsNegativeB,
        IsNegativeC,
        numActualDigitsPlusGuard,
        numActualDigits,
        ext_A_X2,
        ext_B_Y2,
        ext_C_A,
        shiftALeftToGetMsb,
        shiftBLeftToGetMsb,
        shiftCLeftToGetMsb,
        effExpA,
        effExpB,
        effExpC,
        bias,

        outSignTrue,
        outSignFalse,

        outExponentTrue,
        outExponentFalse,

        extResultTrue,
        extResultFalse,

        // NR params (A=One, B=W1, C=W0)
        nrThreeWay,
        nrIsNegOne,
        nrIsNegW1,
        nrIsNegW0,
        ext_One,
        ext_W1,
        ext_W0,
        nrShiftOne,
        nrShiftW1,
        nrShiftW0,
        nrEffOne,
        nrEffW1,
        nrEffW0,
        nrSignTrue,
        nrSignFalse,
        nrOutExpTrue,
        nrOutExpFalse,
        nrExtResultTrue,
        nrExtResultFalse,

        debugStates
    );

    int32_t outExponent_DE = 0;
    int32_t nrOutExpDE = 0;

    Phase1_DE<SharkFloatParams, CallIndex>(
        block,
        grid,
        idx,
        DIsBiggerMagnitude,
        IsNegativeD,
        IsNegativeE,
        numActualDigitsPlusGuard,
        numActualDigits,
        ext_D_2X,
        ext_E_B,
        shiftDLeftToGetMsb,
        shiftELeftToGetMsb,
        effExpD,
        effExpE,
        newDExponent,
        newEExponent,
        outExponent_DE,
        final128_DE,  // the extended result digits

        // NR params
        nrW2Bigger,
        nrIsNegW2,
        nrIsNegW3,
        ext_W2,
        ext_W3,
        nrShiftW2,
        nrShiftW3,
        nrEffW2,
        nrEffW3,
        nrExpW2,
        nrExpW3,
        nrOutExpDE,
        nrFinal128_DE,

        debugStates);

    int32_t carryAcc_ABC_True = 0;
    int32_t carryAcc_ABC_False = 0;
    int32_t carryAcc_DE = 0;

    int32_t nrCarryTrue = 0, nrCarryFalse = 0, nrCarryDE = 0;

    if constexpr (!SharkFloatParams::DisableCarryPropagation) {
        CarryPropagation_ABC_PPv5<SharkFloatParams>(
            globalSync1,
            globalSync2,
            shared_data,
            idx,
            numActualDigitsPlusGuard,
            extResultTrue,
            extResultFalse,
            final128_DE,
            carry1,
            carry2,
            carry3,
            carry4,
            carry5,
            carry6,
            carryAcc_ABC_True,
            carryAcc_ABC_False,
            carryAcc_DE,
            // NR carry params
            nrExtResultTrue,
            nrExtResultFalse,
            nrFinal128_DE,
            nrCarry1,
            nrCarry2,
            nrCarry3,
            nrCarry4,
            nrCarry5,
            nrCarry6,
            nrCarryTrue,
            nrCarryFalse,
            nrCarryDE,
            block,
            grid,
            debugGlobalState);
    }

    if constexpr (HpShark::DebugChecksums) {
        grid.sync();
        StoreCurrentDebugStateAdd<SharkFloatParams, DebugStatePurpose::FinalAdd1, uint64_t>(
            debugStates, grid, block, extResultTrue, numActualDigitsPlusGuard);
        StoreCurrentDebugStateAdd<SharkFloatParams, DebugStatePurpose::FinalAdd2, uint64_t>(
            debugStates, grid, block, extResultFalse, numActualDigitsPlusGuard);
        StoreCurrentDebugStateAdd<SharkFloatParams, DebugStatePurpose::FinalAdd3, uint64_t>(
            debugStates, grid, block, final128_DE, numActualDigitsPlusGuard);

        if constexpr (SharkFloatParams::EnableNewtonRaphson) {
            StoreCurrentDebugStateAdd<SharkFloatParams, DebugStatePurpose::FinalAddDzdc1, uint64_t>(
                debugStates, grid, block, nrExtResultTrue, numActualDigitsPlusGuard);
            StoreCurrentDebugStateAdd<SharkFloatParams, DebugStatePurpose::FinalAddDzdc2, uint64_t>(
                debugStates, grid, block, nrExtResultFalse, numActualDigitsPlusGuard);
            StoreCurrentDebugStateAdd<SharkFloatParams, DebugStatePurpose::FinalAddDzdc3, uint64_t>(
                debugStates, grid, block, nrFinal128_DE, numActualDigitsPlusGuard);
        }

        grid.sync();
    } else {
        grid.sync();
    }

    // 1) Decide which A−B+C branch to use
    const bool sameSignDE = (D_2X->GetNegative() == E_B->GetNegative());
    bool useTrueBranch = (carryAcc_ABC_True >= carryAcc_ABC_False);

    // 3) And handle D+E exactly once
    bool deSign = sameSignDE
        ? D_2X->GetNegative()
        : (DIsBiggerMagnitude ? D_2X->GetNegative() : E_B->GetNegative());

    // 2) Normalize & write *only* that branch
    if (useTrueBranch) {
        NormalizeAndCopyResult<SharkFloatParams>(grid.size(),
            idx,
            numActualDigits,
            numActualDigitsPlusGuard,
            outExponentTrue,
            outExponent_DE,
            carryAcc_ABC_True,
            carryAcc_DE,
            extResultTrue,
            final128_DE,
            Out_A_B_C,
            Out_D_E,
            outSignTrue,
            deSign
        );
    } else {
        NormalizeAndCopyResult<SharkFloatParams>(grid.size(),
            idx,
            numActualDigits,
            numActualDigitsPlusGuard,
            outExponentFalse,
            outExponent_DE,
            carryAcc_ABC_False,
            carryAcc_DE,
            extResultFalse,
            final128_DE,
            Out_A_B_C,
            Out_D_E,
            outSignFalse,
            deSign
        );
    }

    // NR: NormalizeAndCopyResult for NR outputs (no sync needed — writes to different buffers)
    if constexpr (SharkFloatParams::EnableNewtonRaphson) {
        const bool nrSameDE = (W2->GetNegative() == W3->GetNegative());
        const bool nrUseTrueBranch = (nrCarryTrue >= nrCarryFalse);
        const bool nrDeSign = nrSameDE
            ? W2->GetNegative()
            : (nrW2Bigger ? W2->GetNegative() : W3->GetNegative());

        if (nrUseTrueBranch) {
            NormalizeAndCopyResult<SharkFloatParams>(grid.size(),
                idx, numActualDigits, numActualDigitsPlusGuard,
                nrOutExpTrue, nrOutExpDE,
                nrCarryTrue, nrCarryDE,
                nrExtResultTrue, nrFinal128_DE,
                Out_DzdcReal, Out_DzdcImag,
                nrSignTrue, nrDeSign);
        } else {
            NormalizeAndCopyResult<SharkFloatParams>(grid.size(),
                idx, numActualDigits, numActualDigitsPlusGuard,
                nrOutExpFalse, nrOutExpDE,
                nrCarryFalse, nrCarryDE,
                nrExtResultFalse, nrFinal128_DE,
                Out_DzdcReal, Out_DzdcImag,
                nrSignFalse, nrDeSign);
        }
    }

    //const int32_t stride = blockDim.x * gridDim.x;
    //FinalResolutionDE(
    //    idx,
    //    stride,
    //    carryAcc_DE,
    //    numActualDigitsPlusGuard,
    //    numActualDigits,
    //    final128_DE,
    //    Out_D_E,
    //    outExponent_DE
    //     );

    if constexpr (HpShark::DebugChecksums) {
        grid.sync();
        StoreCurrentDebugStateAdd<SharkFloatParams, DebugStatePurpose::Result_Add1, uint32_t>(
            debugStates, grid, block, Out_A_B_C->Digits, numActualDigits);
        StoreCurrentDebugStateAdd<SharkFloatParams, DebugStatePurpose::Result_Add2, uint32_t>(
            debugStates, grid, block, Out_D_E->Digits, numActualDigits);

        if constexpr (SharkFloatParams::EnableNewtonRaphson) {
            StoreCurrentDebugStateAdd<SharkFloatParams, DebugStatePurpose::Result_AddDzdc1, uint32_t>(
                debugStates, grid, block, Out_DzdcReal->Digits, numActualDigits);
            StoreCurrentDebugStateAdd<SharkFloatParams, DebugStatePurpose::Result_AddDzdc2, uint32_t>(
                debugStates, grid, block, Out_DzdcImag->Digits, numActualDigits);
        }

        grid.sync();
    } else {
        grid.sync();
    }
}

template <class SharkFloatParams>
static __device__ void AddHelper (
    cg::grid_group &grid,
    cg::thread_block &block,
    HpSharkAddComboResults<SharkFloatParams> *SharkRestrict combo,
    uint64_t *tempData) {

    AddHelperSeparates<SharkFloatParams>(
        grid,
        block,
        &combo->A_X2,
        &combo->B_Y2,
        &combo->C_A,
        &combo->D_2X,
        &combo->E_B,
        &combo->Result1_A_B_C,
        &combo->Result2_D_E,
        tempData,
        SharkFloatParams::EnableNewtonRaphson ? &combo->W0 : nullptr,
        SharkFloatParams::EnableNewtonRaphson ? &combo->W1 : nullptr,
        SharkFloatParams::EnableNewtonRaphson ? &combo->W2 : nullptr,
        SharkFloatParams::EnableNewtonRaphson ? &combo->W3 : nullptr,
        SharkFloatParams::EnableNewtonRaphson ? &combo->One : nullptr,
        SharkFloatParams::EnableNewtonRaphson ? &combo->ResultDzdcReal : nullptr,
        SharkFloatParams::EnableNewtonRaphson ? &combo->ResultDzdcImag : nullptr);
}

