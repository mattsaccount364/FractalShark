#include "HpSharkFloat.cuh"
#include "BenchmarkTimer.h"
#include "TestTracker.h"
#include "KernelInvoke.cuh"
#include "Tests.h"
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
    const auto *data,
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
/// treating words beyond 'actualDigits' as zero (within an 'extDigits' buffer).
static __device__ SharkForceInlineReleaseOnly uint32_t
GetNormalizedDigit(
    const uint32_t *digits,
    int32_t         actualDigits,
    int32_t         extDigits,
    int32_t         shiftBits,
    int32_t         idx) {
    // ensure idx is within the extended buffer
    assert(idx >= 0 && idx < extDigits);

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
    const auto *in,
    const int32_t  extDigits,
    const int32_t  shiftNeeded,
    uint32_t *out,
    const int32_t  outSz
)
{
    assert(extDigits >= outSz);
    const auto stride = grid.size();

    for (int32_t i = idx; i < outSz; i += stride) {
        out[i] = FunnelShift32<D>(
            in,
            i,
            extDigits,
            shiftNeeded
        );
    }
}

// Retrieves a limb from an extended array, returning zero for indices beyond the actual digit count.
// This handles the boundary between actual digits and guard digits in extended precision arithmetic.
static __device__ SharkForceInlineReleaseOnly uint32_t
GetExtLimb (
    const uint32_t *ext,
    const int32_t actualDigits,
    const int32_t extDigits,
    const int32_t idx) {

    if (idx < actualDigits) {
        return ext[idx];
    } else {
        assert(idx < extDigits);
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

// Computes the bit shift offset needed to normalize an extended precision number.
// Returns the shift amount and updates the exponent accordingly, without actually performing the shift.
static __device__ SharkForceInlineReleaseOnly int32_t
ExtendedNormalizeShiftIndex (
    const uint32_t *ext,
    const int32_t actualDigits,
    const int32_t extDigits,
    int32_t &storedExp,
    bool &isZero) {
    int32_t msd = extDigits - 1;
    while (msd >= 0 && GetExtLimb(ext, actualDigits, extDigits, msd) == 0)
        msd--;
    if (msd < 0) {
        isZero = true;
        return 0;  // For zero, the shift offset is irrelevant.
    }
    isZero = false;
    const int32_t clz = CountLeadingZeros(GetExtLimb(ext, actualDigits, extDigits, msd));
    // In little-endian, the overall bit index of the MSB is:
    //    current_msb = msd * 32 + (31 - clz)
    const int32_t current_msb = msd * 32 + (31 - clz);
    const int32_t totalExtBits = extDigits * 32;
    // Compute the left-shift needed so that the MSB moves to bit (totalExtBits - 1).
    const int32_t L = (totalExtBits - 1) - current_msb;
    // Adjust the exponent as if we had shifted the number left by L bits.
    storedExp -= L;
    return L;
}

// New helper: Computes the aligned digit for the normalized value on the fly.
// 'diffDE' is the additional right shift required for alignment.
template <class SharkFloatParams>
static __device__ SharkForceInlineReleaseOnly uint32_t
GetShiftedNormalizedDigit (
    const uint32_t *ext,
    const int32_t actualDigits,
    const int32_t extDigits,
    const int32_t shiftOffset,
    const int32_t diff,
    const int32_t idx) {
    // const int32_t n = SharkFloatParams::GlobalNumUint32; // normalized length
    const int32_t wordShift = diff / 32;
    const int32_t bitShift = diff % 32;
    const uint32_t lower = (idx + wordShift < extDigits) ?
        GetNormalizedDigit(ext, actualDigits, extDigits, shiftOffset, idx + wordShift) : 0;
    const uint32_t upper = (idx + wordShift + 1 < extDigits) ?
        GetNormalizedDigit(ext, actualDigits, extDigits, shiftOffset, idx + wordShift + 1) : 0;
    if (bitShift == 0)
        return lower;
    else
        return (lower >> bitShift) | (upper << (32 - bitShift));
}


__device__ SharkForceInlineReleaseOnly bool
CompareMagnitudes2Way (
    const int32_t effExpA,
    const int32_t effExpB,
    const int32_t actualDigits,
    const int32_t extDigits,
    const int32_t shiftA,
    const int32_t shiftB,
    const uint32_t *extA,
    const uint32_t *extB) {
    bool AIsBiggerMagnitude;

    if (effExpA > effExpB) {
        AIsBiggerMagnitude = true;
    } else if (effExpA < effExpB) {
        AIsBiggerMagnitude = false;
    } else {
        AIsBiggerMagnitude = false; // default if equal
        for (int32_t i = extDigits - 1; i >= 0; i--) {
            uint32_t digitA = GetNormalizedDigit(extA, actualDigits, extDigits, shiftA, i);
            uint32_t digitB = GetNormalizedDigit(extB, actualDigits, extDigits, shiftB, i);
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
__device__ SharkForceInlineReleaseOnly void
GetCorrespondingLimbs (
    const uint32_t *ext_A_X2,
    const int32_t actualASize,
    const int32_t extASize,
    const uint32_t *ext_B_Y2,
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
__device__ SharkForceInlineReleaseOnly void
EraseCurrentDebugState (
    RecordIt record,
    DebugState<SharkFloatParams> *debugStates,
    cooperative_groups::grid_group &grid,
    cooperative_groups::thread_block &block) {

    constexpr auto RecursionDepth = 0;
    constexpr auto CallIndex = 0;
    constexpr auto maxPurposes = static_cast<int32_t>(DebugStatePurpose::NumPurposes);
    constexpr auto curPurpose = static_cast<int32_t>(Purpose);
    debugStates[CallIndex * maxPurposes + curPurpose].Erase(
        record, grid, block, Purpose, RecursionDepth, CallIndex);
}

template<
    class SharkFloatParams,
    int32_t CallIndex,
    DebugStatePurpose Purpose,
    typename ArrayType>
__device__ SharkForceInlineReleaseOnly void
StoreCurrentDebugState (
    RecordIt record,
    DebugState<SharkFloatParams> *debugStates,
    cooperative_groups::grid_group &grid,
    cooperative_groups::thread_block &block,
    const ArrayType *arrayToChecksum,
    size_t arraySize) {

    constexpr auto CurPurpose = static_cast<int32_t>(Purpose);
    constexpr auto RecursionDepth = 0;
    constexpr auto UseConvolutionHere = UseConvolution::No;

    debugStates[CurPurpose].Reset(
        record, UseConvolutionHere, grid, block, arrayToChecksum, arraySize, Purpose, RecursionDepth, CallIndex);
}

#include "Add_ABC.cuh"
#include "Add_DE.cuh"


template<class SharkFloatParams>
static __device__ SharkForceInlineReleaseOnly void
NormalizeAndCopyResult (
    cooperative_groups::grid_group &grid,
    cooperative_groups::thread_block &block,
    const int32_t idx,
    const int32_t actualDigits,
    const int32_t extDigits,
    int32_t exponent,
    int32_t &carry,
    auto *scratch,
    HpSharkFloat<SharkFloatParams> *ResultOut,
    const bool outSign
) noexcept {
    // --- 1) Inject any carry/borrow back into the digit stream ---
    if (idx == 0) {
        if (carry != 0) {
            if (carry < 0) {
                return;
            }
            int shift = (carry == 2 ? 2 : 1);
            exponent += shift;

            uint32_t highBits = static_cast<uint32_t>(carry);
            for (int32_t i = extDigits - 1; i >= 0; --i) {
                uint32_t w = scratch[i];
                uint32_t lowMask = (1u << shift) - 1;
                uint32_t nextHB = w & lowMask;
                scratch[i] = (w >> shift) | (highBits << (32 - shift));
                highBits = nextHB;
            }
        }
    }

    grid.sync();

    // --- 2) Locate most‐significant non‐zero word ---
    int32_t msdResult = 0;
    for (int32_t i = extDigits - 1; i >= 0; --i) {
        if (scratch[i] != 0) {
            msdResult = i;
            break;
        }
    }

    // --- 3) Compute current vs desired bit positions ---
    int32_t clzResult = CountLeadingZeros(scratch[msdResult]);
    int32_t currentBit = msdResult * 32 + (31 - clzResult);
    int32_t desiredBit = (SharkFloatParams::GlobalNumUint32 - 1) * 32 + 31;

    // --- 4) Normalize by shifting left or right ---
    int32_t shiftNeeded = currentBit - desiredBit;
    if (shiftNeeded > 0) {
        MultiWordShift<Dir::Right>(
            grid,
            block,
            idx,
            scratch,
            extDigits,
            shiftNeeded,
            ResultOut->Digits,
            actualDigits
        );
        exponent += shiftNeeded;
    } else if (shiftNeeded < 0) {
        int32_t L = -shiftNeeded;
        MultiWordShift<Dir::Left>(
            grid,
            block,
            idx,
            scratch,
            extDigits,
            L,
            ResultOut->Digits,
            actualDigits
        );
        exponent -= L;

   } else {
        int32_t L = 0;
        MultiWordShift<Dir::Left>(
            grid,
            block,
            idx,
            scratch,
            extDigits,
            L,
            ResultOut->Digits,
            actualDigits
        );
    }

    // --- 5) Set final exponent and sign ---
    if (idx == 0) {
        ResultOut->Exponent = exponent;
        ResultOut->SetNegative(outSign);
    }

    grid.sync();
}


template <class SharkFloatParams>
__device__ void AddHelper (
    cg::grid_group &grid,
    cg::thread_block &block,
    HpSharkAddComboResults<SharkFloatParams> *SharkRestrict combo,
    uint64_t *tempData) {

    extern __shared__ uint32_t sharedData[];

    // --- Constants and Parameters ---
    constexpr int32_t guard = SharkFloatParams::Guard;
    constexpr int32_t numActualDigits = SharkFloatParams::GlobalNumUint32;
    constexpr int32_t numActualDigitsPlusGuard = SharkFloatParams::GlobalNumUint32 + guard;
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    constexpr int32_t NewN = SharkFloatParams::GlobalNumUint32;

    const auto *A_X2 = &combo->A_X2;
    const auto *B_Y2 = &combo->B_Y2;
    const auto *C_A = &combo->C_A;
    const auto *D_2X = &combo->D_2X;
    const auto *E_B = &combo->E_B;

    const auto *ext_A_X2 = combo->A_X2.Digits;
    const auto *ext_B_Y2 = combo->B_Y2.Digits;
    const auto *ext_C_A = combo->C_A.Digits;
    const auto *ext_D_2X = combo->D_2X.Digits;
    const auto *ext_E_B = combo->E_B.Digits;

    const bool IsNegativeA = A_X2->GetNegative();
    const bool IsNegativeB = !B_Y2->GetNegative(); // A - B + C
    const bool IsNegativeC = C_A->GetNegative();
    const bool IsNegativeD = D_2X->GetNegative();
    const bool IsNegativeE = E_B->GetNegative();

    auto *Out_A_B_C = &combo->Result1_A_B_C;
    auto *Out_D_E = &combo->Result2_D_E;

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

    const RecordIt record =
        (block.thread_index().x == 0 && block.group_index().x == 0) ?
        RecordIt::Yes :
        RecordIt::No;

    static constexpr auto CallIndex = 0;

    if constexpr (SharkDebugChecksums) {
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Invalid>(record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::ADigits>(record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::BDigits>(record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::CDigits>(record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::DDigits>(record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::EDigits>(record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::AHalfHigh>(record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::AHalfLow>(record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::BHalfHigh>(record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::BHalfLow>(record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::XDiff>(record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::YDiff>(record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z0XX>(record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z0XY>(record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z0YY>(record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z1XX>(record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z1XY>(record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z1YY>(record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z2XX>(record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z2XY>(record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z2YY>(record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z2_Perm1>(record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z2_Perm2>(record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z2_Perm3>(record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z2_Perm4>(record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z2_Perm5>(record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z2_Perm6>(record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z1_offsetXX>(record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z1_offsetXY>(record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z1_offsetYY>(record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Final128XX>(record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Final128XY>(record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Final128YY>(record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::FinalAdd1>(record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::FinalAdd2>(record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::FinalAdd3>(record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Result_offsetXX>(record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Result_offsetXY>(record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Result_offsetYY>(record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Result_Add1>(record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Result_Add2>(record, debugStates, grid, block);
        static_assert(static_cast<int32_t>(DebugStatePurpose::NumPurposes) == 41, "Unexpected number of purposes");

        StoreCurrentDebugState<SharkFloatParams, CallIndex, DebugStatePurpose::ADigits, uint32_t>(
            record, debugStates, grid, block, A_X2->Digits, NewN);
        StoreCurrentDebugState<SharkFloatParams, CallIndex, DebugStatePurpose::BDigits, uint32_t>(
            record, debugStates, grid, block, B_Y2->Digits, NewN);
        StoreCurrentDebugState<SharkFloatParams, CallIndex, DebugStatePurpose::CDigits, uint32_t>(
            record, debugStates, grid, block, C_A->Digits, NewN);
        StoreCurrentDebugState<SharkFloatParams, CallIndex, DebugStatePurpose::DDigits, uint32_t>(
            record, debugStates, grid, block, D_2X->Digits, NewN);
        StoreCurrentDebugState<SharkFloatParams, CallIndex, DebugStatePurpose::EDigits, uint32_t>(
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

    const int32_t shiftALeftToGetMsb = ExtendedNormalizeShiftIndex(
        ext_A_X2,
        numActualDigits,
        numActualDigitsPlusGuard,
        newAExponent,
        normA_isZero);

    const int32_t shiftBLeftToGetMsb = ExtendedNormalizeShiftIndex(
        ext_B_Y2,
        numActualDigits,
        numActualDigitsPlusGuard,
        newBExponent,
        normB_isZero);

    const int32_t shiftCLeftToGetMsb = ExtendedNormalizeShiftIndex(
        ext_C_A,
        numActualDigits,
        numActualDigitsPlusGuard,
        newCExponent,
        normC_isZero);

    const int32_t shiftDLeftToGetMsb = ExtendedNormalizeShiftIndex(
        ext_D_2X,
        numActualDigits,
        numActualDigitsPlusGuard,
        newDExponent,
        normD_isZero);

    const int32_t shiftELeftToGetMsb = ExtendedNormalizeShiftIndex(
        ext_E_B,
        numActualDigits,
        numActualDigitsPlusGuard,
        newEExponent,
        normE_isZero);

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

    int32_t carryTrue = 0;
    int32_t carryFalse = 0;

    int32_t carry_DE = 0;

    if constexpr (!SharkFloatParams::DisableCarryPropagation) {
        CarryPropagation_ABC<SharkFloatParams>(
            sharedData,
            globalSync,
            idx,
            numActualDigitsPlusGuard,
            final128_DE,
            carry1,
            carry2,
            carry_DE,
            block,
            grid);

        CarryPropagation_ABC<SharkFloatParams>(
            sharedData,
            globalSync,
            idx,
            numActualDigitsPlusGuard,
            extResultTrue,
            carry1,
            carry2,
            carryTrue,
            block,
            grid);

        CarryPropagation_ABC<SharkFloatParams>(
            sharedData,
            globalSync,
            idx,
            numActualDigitsPlusGuard,
            extResultFalse,
            carry1,
            carry2,
            carryFalse,
            block,
            grid);
    }

    if constexpr (SharkDebugChecksums) {
        grid.sync();
        StoreCurrentDebugState<SharkFloatParams, CallIndex, DebugStatePurpose::FinalAdd1, uint64_t>(
            record, debugStates, grid, block, extResultTrue, numActualDigitsPlusGuard);
        StoreCurrentDebugState<SharkFloatParams, CallIndex, DebugStatePurpose::FinalAdd2, uint64_t>(
            record, debugStates, grid, block, extResultFalse, numActualDigitsPlusGuard);
        StoreCurrentDebugState<SharkFloatParams, CallIndex, DebugStatePurpose::FinalAdd3, uint64_t>(
            record, debugStates, grid, block, final128_DE, numActualDigitsPlusGuard);
        grid.sync();
    } else {
        grid.sync();
    }

    // 1) Decide which A−B+C branch to use
    const bool sameSignDE = (D_2X->GetNegative() == E_B->GetNegative());
    bool useTrueBranch = (carryTrue >= carryFalse);

    // 2) Normalize & write *only* that branch
    if (useTrueBranch) {
        NormalizeAndCopyResult<SharkFloatParams>(
            grid,
            block,
            idx,
            numActualDigits,
            numActualDigitsPlusGuard,
            outExponentTrue,
            carryTrue,
            extResultTrue,
            Out_A_B_C,
            outSignTrue
        );
    } else {
        NormalizeAndCopyResult<SharkFloatParams>(
            grid,
            block,
            idx,
            numActualDigits,
            numActualDigitsPlusGuard,
            outExponentFalse,
            carryFalse,
            extResultFalse,
            Out_A_B_C,
            outSignFalse
        );
    }

    // 3) And handle D+E exactly once
    bool deSign = sameSignDE
        ? D_2X->GetNegative()
        : (DIsBiggerMagnitude ? D_2X->GetNegative() : E_B->GetNegative());

    NormalizeAndCopyResult<SharkFloatParams>(
        grid,
        block,
        idx,
        numActualDigits,
        numActualDigitsPlusGuard,
        outExponent_DE,
        carry_DE,
        final128_DE,
        Out_D_E,
        deSign
    );

    //const int32_t stride = blockDim.x * gridDim.x;
    //FinalResolutionDE(
    //    idx,
    //    stride,
    //    carry_DE,
    //    numActualDigitsPlusGuard,
    //    numActualDigits,
    //    final128_DE,
    //    Out_D_E,
    //    outExponent_DE
    //     );

    if constexpr (SharkDebugChecksums) {
        grid.sync();
        StoreCurrentDebugState<SharkFloatParams, CallIndex, DebugStatePurpose::Result_Add1, uint32_t>(
            record, debugStates, grid, block, Out_A_B_C->Digits, numActualDigits);
        StoreCurrentDebugState<SharkFloatParams, CallIndex, DebugStatePurpose::Result_Add2, uint32_t>(
            record, debugStates, grid, block, Out_D_E->Digits, numActualDigits);
        grid.sync();
    } else {
        grid.sync();
    }
}


template<class SharkFloatParams>
__global__ void AddKernel (
    HpSharkAddComboResults<SharkFloatParams> *SharkRestrict combo,
    uint64_t *tempData) {

    // Initialize cooperative grid group
    cg::grid_group grid = cg::this_grid();
    cg::thread_block block = cg::this_thread_block();

    // Call the AddHelper function
    AddHelper(grid, block, combo, tempData);
}

template<class SharkFloatParams>
__global__ void AddKernelTestLoop (
    HpSharkAddComboResults<SharkFloatParams> *SharkRestrict combo,
    uint64_t numIters,
    uint64_t *tempData) {

    // Initialize cooperative grid group
    cg::grid_group grid = cg::this_grid();
    cg::thread_block block = cg::this_thread_block();

    for (int32_t i = 0; i < numIters; ++i) {
        AddHelper(grid, block, combo, tempData);
    }
}

template<class SharkFloatParams>
void ComputeAddGpu (void *kernelArgs[]) {

    constexpr auto ExpandedNumDigits = SharkFloatParams::GlobalNumUint32;
    constexpr size_t SharedMemSize = sizeof(uint32_t) * ExpandedNumDigits; // Adjust as necessary
    cudaError_t err = cudaLaunchCooperativeKernel(
        (void *)AddKernel<SharkFloatParams>,
        dim3(SharkFloatParams::GlobalNumBlocks),
        dim3(SharkFloatParams::GlobalThreadsPerBlock),
        kernelArgs,
        SharedMemSize, // Shared memory size
        0 // Stream
    );

    cudaDeviceSynchronize();

    if (err != cudaSuccess) {
        std::cerr << "CUDA error in ComputeAddGpu: " << cudaGetErrorString(err) << std::endl;
        DebugBreak();
    }
}

template<class SharkFloatParams>
void ComputeAddGpuTestLoop (void *kernelArgs[]) {

    constexpr auto ExpandedNumDigits = SharkFloatParams::GlobalNumUint32;
    constexpr size_t SharedMemSize = sizeof(uint32_t) * ExpandedNumDigits; // Adjust as necessary

    cudaError_t err = cudaLaunchCooperativeKernel(
        (void *)AddKernelTestLoop<SharkFloatParams>,
        dim3(SharkFloatParams::GlobalNumBlocks),
        dim3(SharkFloatParams::GlobalThreadsPerBlock),
        kernelArgs,
        SharedMemSize, // Shared memory size
        0 // Stream
    );

    cudaDeviceSynchronize();

    if (err != cudaSuccess) {
        std::cerr << "CUDA error in ComputeAddGpuTestLoop: " << cudaGetErrorString(err) << std::endl;
        DebugBreak();
    }
}

#define ExplicitlyInstantiate(SharkFloatParams) \
    template void ComputeAddGpu<SharkFloatParams>(void *kernelArgs[]); \
    template void ComputeAddGpuTestLoop<SharkFloatParams>(void *kernelArgs[]);

#ifdef SHARK_INCLUDE_KERNELS
ExplicitInstantiateAll();
#endif
