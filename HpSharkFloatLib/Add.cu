#include "HpSharkFloat.cuh"
#include "DebugChecksum.cuh"

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
    const cooperative_groups::grid_group &grid,
    const cooperative_groups::thread_block &block,
    const int32_t idx,
    const auto *SharkRestrict in,
    const int32_t  numActualDigitsPlusGuard,
    const int32_t  shiftNeeded,
    auto *SharkRestrict out,
    const int32_t  outSz
)
{
    MattsCudaAssert(numActualDigitsPlusGuard >= outSz);
    const auto stride = grid.size();

    for (int32_t i = idx; i < outSz; i += stride) {
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

/// Compute the left‐shift needed to bring each MSB to the top, for five operands.
template<class SharkFloatParams>
static __device__ SharkForceInlineReleaseOnly void
ExtendedNormalizeShiftIndexAll(
    // inputs: five little‐endian extended arrays (length numActualDigitsPlusGuard)
    const uint32_t *SharkRestrict extA,
    const uint32_t *SharkRestrict extB,
    const uint32_t *SharkRestrict extC,
    const uint32_t *SharkRestrict extD,
    const uint32_t *SharkRestrict extE,
    int32_t actualDigits,
    int32_t numActualDigitsPlusGuard,

    // in/out: stored exponents
    int32_t &expA,
    int32_t &expB,
    int32_t &expC,
    int32_t &expD,
    int32_t &expE,

    // out: “is zero?” flags
    bool &isZeroA,
    bool &isZeroB,
    bool &isZeroC,
    bool &isZeroD,
    bool &isZeroE,

    // out: left‐shift amounts
    int32_t &shiftA,
    int32_t &shiftB,
    int32_t &shiftC,
    int32_t &shiftD,
    int32_t &shiftE
) {
    // initialize msd indices to “not found”
    int32_t msdA = -1, msdB = -1, msdC = -1, msdD = -1, msdE = -1;

    // scan downward once
    for (int32_t i = numActualDigitsPlusGuard - 1; i >= 0; --i) {
        // 1) grab all five limbs up front (no data‐dep between them)
        const uint32_t limbA = GetExtLimb(extA, actualDigits, numActualDigitsPlusGuard, i);
        const uint32_t limbB = GetExtLimb(extB, actualDigits, numActualDigitsPlusGuard, i);
        const uint32_t limbC = GetExtLimb(extC, actualDigits, numActualDigitsPlusGuard, i);
        const uint32_t limbD = GetExtLimb(extD, actualDigits, numActualDigitsPlusGuard, i);
        const uint32_t limbE = GetExtLimb(extE, actualDigits, numActualDigitsPlusGuard, i);

        // 2) now update each msd only if we haven’t found it yet
        if (msdA < 0 && limbA != 0) msdA = i;
        if (msdB < 0 && limbB != 0) msdB = i;
        if (msdC < 0 && limbC != 0) msdC = i;
        if (msdD < 0 && limbD != 0) msdD = i;
        if (msdE < 0 && limbE != 0) msdE = i;

        // 3) exit early once all five are found
        if (msdA >= 0 && msdB >= 0 && msdC >= 0 && msdD >= 0 && msdE >= 0)
            break;
    }

    // handle zero‐cases
    isZeroA = (msdA < 0);
    isZeroB = (msdB < 0);
    isZeroC = (msdC < 0);
    isZeroD = (msdD < 0);
    isZeroE = (msdE < 0);

    // common bias: highest bit index in full ext array
    const int32_t totalExtBits = numActualDigitsPlusGuard * 32;
    auto computeShift = [&](const uint32_t *ext, int32_t &msd, bool &isz, int32_t &exp, int32_t &outShift) {
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
    RecordIt record,
    DebugState<SharkFloatParams> *SharkRestrict debugStates,
    cooperative_groups::grid_group &grid,
    cooperative_groups::thread_block &block) {

    constexpr auto RecursionDepth = 0;
    constexpr auto CallIndex = 0;
    constexpr auto curPurpose = static_cast<int32_t>(Purpose);
    debugStates[curPurpose].Erase(
        record, grid, block, Purpose, RecursionDepth, CallIndex);
}

template<
    class SharkFloatParams,
    DebugStatePurpose Purpose,
    typename ArrayType>
static __device__ SharkForceInlineReleaseOnly void
StoreCurrentDebugStateAdd (
    RecordIt record,
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
        record, UseConvolutionHere, grid, block, arrayToChecksum, arraySize, Purpose, RecursionDepth, CallIndex);
}

#include "Add_ABC.cuh"
#include "Add_DE.cuh"


template<class SharkFloatParams>
static __device__ SharkForceInlineReleaseOnly void
NormalizeAndCopyResult(
    cooperative_groups::grid_group &grid,
    cooperative_groups::thread_block &block,
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
    const int stride = int(grid.size());

    // 1) inject carries into the guard slot at index N
    extABC[N] = static_cast<uint32_t>(carryABC);
    extDE[N] = static_cast<uint32_t>(carryDE);

    // 2) find MSB index in each (all threads may do it, but it's cheap)
    int32_t msdA = 0, msdD = 0;
    for (int32_t i = N; i >= 0; --i) {
        // cast down to 32 bits — that’s exactly what FunnelShift32 will see
        uint32_t a = static_cast<uint32_t>(extABC[i]);
        uint32_t d = static_cast<uint32_t>(extDE[i]);

        if (msdA == 0 && a != 0u) msdA = i;
        if (msdD == 0 && d != 0u) msdD = i;

        // once *both* are non-zero, we’re done
        if (msdA != 0 && msdD != 0)
            break;
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
        MultiWordShift<Dir::Right>(
            grid, block, idx,
            extABC, Np1, shiftA,
            OutABC->Digits, actualDigits
        );
    } else if (shiftA < 0) {
        MultiWordShift<Dir::Left>(
            grid, block, idx,
            extABC, Np1, -shiftA,
            OutABC->Digits, actualDigits
        );
    } else {
        MultiWordShift<Dir::Left>(
            grid, block, idx,
            extABC, Np1, 0,
            OutABC->Digits, actualDigits
        );
    }

    if (shiftD > 0) {
        MultiWordShift<Dir::Right>(
            grid, block, idx,
            extDE, Np1, shiftD,
            OutDE->Digits, actualDigits
        );
    } else if (shiftD < 0) {
        MultiWordShift<Dir::Left>(
            grid, block, idx,
            extDE, Np1, -shiftD,
            OutDE->Digits, actualDigits
        );
    } else {
        MultiWordShift<Dir::Left>(
            grid, block, idx,
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
    uint64_t *tempData)
{
    extern __shared__ uint32_t sharedData[];

    // --- Constants and Parameters ---
    constexpr int32_t guard = SharkFloatParams::Guard;
    constexpr int32_t numActualDigits = SharkFloatParams::GlobalNumUint32;
    constexpr int32_t numActualDigitsPlusGuard = SharkFloatParams::GlobalNumUint32 + guard;
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
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

    constexpr auto GlobalSync_offset = 0;
    constexpr auto Checksum_offset = AdditionalGlobalSyncSpace;
    constexpr auto Final128Offset_ABC_True = AdditionalUInt64Global;
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

    auto *SharkRestrict globalSync =
        reinterpret_cast<uint32_t *>(&tempData[GlobalSync_offset]);
    auto *SharkRestrict debugStates =
        reinterpret_cast<DebugState<SharkFloatParams>*>(&tempData[Checksum_offset]);
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

    const RecordIt record =
        (block.thread_index().x == 0 && block.group_index().x == 0) ?
        RecordIt::Yes :
        RecordIt::No;

    static constexpr auto CallIndex = 0;

    if constexpr (SharkDebugChecksums) {
        EraseCurrentDebugStateAdd<SharkFloatParams, DebugStatePurpose::Invalid>(record, debugStates, grid, block);
        EraseCurrentDebugStateAdd<SharkFloatParams, DebugStatePurpose::ADigits>(record, debugStates, grid, block);
        EraseCurrentDebugStateAdd<SharkFloatParams, DebugStatePurpose::BDigits>(record, debugStates, grid, block);
        EraseCurrentDebugStateAdd<SharkFloatParams, DebugStatePurpose::CDigits>(record, debugStates, grid, block);
        EraseCurrentDebugStateAdd<SharkFloatParams, DebugStatePurpose::DDigits>(record, debugStates, grid, block);
        EraseCurrentDebugStateAdd<SharkFloatParams, DebugStatePurpose::EDigits>(record, debugStates, grid, block);
        EraseCurrentDebugStateAdd<SharkFloatParams, DebugStatePurpose::AHalfHigh>(record, debugStates, grid, block);
        EraseCurrentDebugStateAdd<SharkFloatParams, DebugStatePurpose::AHalfLow>(record, debugStates, grid, block);
        EraseCurrentDebugStateAdd<SharkFloatParams, DebugStatePurpose::BHalfHigh>(record, debugStates, grid, block);
        EraseCurrentDebugStateAdd<SharkFloatParams, DebugStatePurpose::BHalfLow>(record, debugStates, grid, block);
        EraseCurrentDebugStateAdd<SharkFloatParams, DebugStatePurpose::XDiff>(record, debugStates, grid, block);
        EraseCurrentDebugStateAdd<SharkFloatParams, DebugStatePurpose::YDiff>(record, debugStates, grid, block);
        EraseCurrentDebugStateAdd<SharkFloatParams, DebugStatePurpose::Z0XX>(record, debugStates, grid, block);
        EraseCurrentDebugStateAdd<SharkFloatParams, DebugStatePurpose::Z0XY>(record, debugStates, grid, block);
        EraseCurrentDebugStateAdd<SharkFloatParams, DebugStatePurpose::Z0YY>(record, debugStates, grid, block);
        EraseCurrentDebugStateAdd<SharkFloatParams, DebugStatePurpose::Z1XX>(record, debugStates, grid, block);
        EraseCurrentDebugStateAdd<SharkFloatParams, DebugStatePurpose::Z1XY>(record, debugStates, grid, block);
        EraseCurrentDebugStateAdd<SharkFloatParams, DebugStatePurpose::Z1YY>(record, debugStates, grid, block);
        EraseCurrentDebugStateAdd<SharkFloatParams, DebugStatePurpose::Z2XX>(record, debugStates, grid, block);
        EraseCurrentDebugStateAdd<SharkFloatParams, DebugStatePurpose::Z2XY>(record, debugStates, grid, block);
        EraseCurrentDebugStateAdd<SharkFloatParams, DebugStatePurpose::Z2YY>(record, debugStates, grid, block);
        EraseCurrentDebugStateAdd<SharkFloatParams, DebugStatePurpose::Z3XX>(record, debugStates, grid, block);
        EraseCurrentDebugStateAdd<SharkFloatParams, DebugStatePurpose::Z3XY>(record, debugStates, grid, block);
        EraseCurrentDebugStateAdd<SharkFloatParams, DebugStatePurpose::Z3YY>(record, debugStates, grid, block);
        EraseCurrentDebugStateAdd<SharkFloatParams, DebugStatePurpose::Z4XX>(record, debugStates, grid, block);
        EraseCurrentDebugStateAdd<SharkFloatParams, DebugStatePurpose::Z4XY>(record, debugStates, grid, block);
        EraseCurrentDebugStateAdd<SharkFloatParams, DebugStatePurpose::Z4YY>(record, debugStates, grid, block);
        EraseCurrentDebugStateAdd<SharkFloatParams, DebugStatePurpose::Z2_Perm1>(record, debugStates, grid, block);
        EraseCurrentDebugStateAdd<SharkFloatParams, DebugStatePurpose::Z2_Perm2>(record, debugStates, grid, block);
        EraseCurrentDebugStateAdd<SharkFloatParams, DebugStatePurpose::Z2_Perm3>(record, debugStates, grid, block);
        EraseCurrentDebugStateAdd<SharkFloatParams, DebugStatePurpose::Z2_Perm4>(record, debugStates, grid, block);
        EraseCurrentDebugStateAdd<SharkFloatParams, DebugStatePurpose::Z2_Perm5>(record, debugStates, grid, block);
        EraseCurrentDebugStateAdd<SharkFloatParams, DebugStatePurpose::Z2_Perm6>(record, debugStates, grid, block);
        EraseCurrentDebugStateAdd<SharkFloatParams, DebugStatePurpose::Z1_offsetXX>(record, debugStates, grid, block);
        EraseCurrentDebugStateAdd<SharkFloatParams, DebugStatePurpose::Z1_offsetXY>(record, debugStates, grid, block);
        EraseCurrentDebugStateAdd<SharkFloatParams, DebugStatePurpose::Z1_offsetYY>(record, debugStates, grid, block);
        EraseCurrentDebugStateAdd<SharkFloatParams, DebugStatePurpose::Final128XX>(record, debugStates, grid, block);
        EraseCurrentDebugStateAdd<SharkFloatParams, DebugStatePurpose::Final128XY>(record, debugStates, grid, block);
        EraseCurrentDebugStateAdd<SharkFloatParams, DebugStatePurpose::Final128YY>(record, debugStates, grid, block);
        EraseCurrentDebugStateAdd<SharkFloatParams, DebugStatePurpose::FinalAdd1>(record, debugStates, grid, block);
        EraseCurrentDebugStateAdd<SharkFloatParams, DebugStatePurpose::FinalAdd2>(record, debugStates, grid, block);
        EraseCurrentDebugStateAdd<SharkFloatParams, DebugStatePurpose::FinalAdd3>(record, debugStates, grid, block);
        EraseCurrentDebugStateAdd<SharkFloatParams, DebugStatePurpose::Result_offsetXX>(record, debugStates, grid, block);
        EraseCurrentDebugStateAdd<SharkFloatParams, DebugStatePurpose::Result_offsetXY>(record, debugStates, grid, block);
        EraseCurrentDebugStateAdd<SharkFloatParams, DebugStatePurpose::Result_offsetYY>(record, debugStates, grid, block);
        EraseCurrentDebugStateAdd<SharkFloatParams, DebugStatePurpose::Result_Add1>(record, debugStates, grid, block);
        EraseCurrentDebugStateAdd<SharkFloatParams, DebugStatePurpose::Result_Add2>(record, debugStates, grid, block);
        static_assert(static_cast<int32_t>(DebugStatePurpose::NumPurposes) == 47, "Unexpected number of purposes");

        StoreCurrentDebugStateAdd<SharkFloatParams, DebugStatePurpose::ADigits, uint32_t>(
            record, debugStates, grid, block, A_X2->Digits, NewN);
        StoreCurrentDebugStateAdd<SharkFloatParams, DebugStatePurpose::BDigits, uint32_t>(
            record, debugStates, grid, block, B_Y2->Digits, NewN);
        StoreCurrentDebugStateAdd<SharkFloatParams, DebugStatePurpose::CDigits, uint32_t>(
            record, debugStates, grid, block, C_A->Digits, NewN);
        StoreCurrentDebugStateAdd<SharkFloatParams, DebugStatePurpose::DDigits, uint32_t>(
            record, debugStates, grid, block, D_2X->Digits, NewN);
        StoreCurrentDebugStateAdd<SharkFloatParams, DebugStatePurpose::EDigits, uint32_t>(
            record, debugStates, grid, block, E_B->Digits, NewN);

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

    ExtendedNormalizeShiftIndexAll<SharkFloatParams>(
        ext_A_X2,
        ext_B_Y2,
        ext_C_A,
        ext_D_2X,
        ext_E_B,
        numActualDigits,
        numActualDigitsPlusGuard,

        newAExponent,
        newBExponent,
        newCExponent,
        newDExponent,
        newEExponent,

        normA_isZero,
        normB_isZero,
        normC_isZero,
        normD_isZero,
        normE_isZero,

        shiftALeftToGetMsb,
        shiftBLeftToGetMsb,
        shiftCLeftToGetMsb,
        shiftDLeftToGetMsb,
        shiftELeftToGetMsb
    );

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
    int32_t biasedExpABC = 0;
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
        ext_C_A,
        biasedExpABC);

    const bool DIsBiggerMagnitude = CompareMagnitudes2Way(
        effExpD,
        effExpE,
        numActualDigits,
        numActualDigitsPlusGuard,
        shiftDLeftToGetMsb,
        shiftELeftToGetMsb,
        ext_D_2X,
        ext_E_B);

    // --- Phase 1: A - B + C ---
    int32_t outExponentTrue = 0;
    int32_t outExponentFalse = 0;

    bool outSignTrue = false;
    bool outSignFalse = false;

    Phase1_ABC<SharkFloatParams, CallIndex>(
        block,
        grid,
        record,
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

        debugStates
    );

    int32_t outExponent_DE = 0;
    Phase1_DE<SharkFloatParams, CallIndex>(
        block,
        grid,
        record,
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
        debugStates);

    int32_t carryAcc_ABC_True = 0;
    int32_t carryAcc_ABC_False = 0;
    int32_t carryAcc_DE = 0;

    if constexpr (!SharkFloatParams::DisableCarryPropagation) {
        CarryPropagation_ABC<SharkFloatParams>(
            globalSync,
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
            block,
            grid);
    }

    if constexpr (SharkDebugChecksums) {
        grid.sync();
        StoreCurrentDebugStateAdd<SharkFloatParams, DebugStatePurpose::FinalAdd1, uint64_t>(
            record, debugStates, grid, block, extResultTrue, numActualDigitsPlusGuard);
        StoreCurrentDebugStateAdd<SharkFloatParams, DebugStatePurpose::FinalAdd2, uint64_t>(
            record, debugStates, grid, block, extResultFalse, numActualDigitsPlusGuard);
        StoreCurrentDebugStateAdd<SharkFloatParams, DebugStatePurpose::FinalAdd3, uint64_t>(
            record, debugStates, grid, block, final128_DE, numActualDigitsPlusGuard);
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
        NormalizeAndCopyResult<SharkFloatParams>(
            grid,
            block,
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
        NormalizeAndCopyResult<SharkFloatParams>(
            grid,
            block,
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

    if constexpr (SharkDebugChecksums) {
        grid.sync();
        StoreCurrentDebugStateAdd<SharkFloatParams, DebugStatePurpose::Result_Add1, uint32_t>(
            record, debugStates, grid, block, Out_A_B_C->Digits, numActualDigits);
        StoreCurrentDebugStateAdd<SharkFloatParams, DebugStatePurpose::Result_Add2, uint32_t>(
            record, debugStates, grid, block, Out_D_E->Digits, numActualDigits);
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
        tempData);
}

