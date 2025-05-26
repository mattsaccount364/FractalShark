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

// ShiftRight: Shifts the number (given by its digit array) right by shiftBits.
// idx is the index of the digit to compute. The parameter numDigits prevents out-of-bounds access.
__device__ SharkForceInlineReleaseOnly uint32_t
ShiftRight (
    const uint32_t *digits,
    const int32_t shiftBits,
    const int32_t idx,
    const int32_t numDigits) {
    const int32_t shiftWords = shiftBits / 32;
    const int32_t shiftBitsMod = shiftBits % 32;
    const uint32_t lower = (idx + shiftWords < numDigits) ? digits[idx + shiftWords] : 0;
    const uint32_t upper = (idx + shiftWords + 1 < numDigits) ? digits[idx + shiftWords + 1] : 0;
    if (shiftBitsMod == 0) {
        return lower;
    } else {
        return (lower >> shiftBitsMod) | (upper << (32 - shiftBitsMod));
    }
}

// ShiftLeft: Shifts the number (given by its digit array) left by shiftBits.
// idx is the index of the digit to compute.
__device__ SharkForceInlineReleaseOnly uint32_t
ShiftLeft (
    const uint32_t *digits,
    const int32_t numActualDigits,
    const int32_t numActualDigitsPlusGuard,
    const int32_t shiftBits,
    const int32_t idx) {
    const int32_t shiftWords = shiftBits / 32;
    const int32_t shiftBitsMod = shiftBits % 32;
    const int32_t srcIdx = idx - shiftWords;

    int32_t srcDigitLower = (srcIdx >= 0 && srcIdx < numActualDigits) ? digits[srcIdx] : 0;
    int32_t srcDigitUpper = (srcIdx - 1 >= 0 && srcIdx - 1 < numActualDigits) ? digits[srcIdx - 1] : 0;

    const uint32_t lower = (srcIdx >= 0) ? srcDigitLower : 0;
    const uint32_t upper = (srcIdx - 1 >= 0) ? srcDigitUpper : 0;
    if (shiftBitsMod == 0) {
        return lower;
    } else {
        return (lower << shiftBitsMod) | (upper >> (32 - shiftBitsMod));
    }
}

// Portable helper: CountLeadingZeros for a 32-bit integer.
// On CUDA consider:
// __device__ int32_t CountLeadingZerosCUDA(uint32_t x) {
// return __clz(x);
// }
__device__ SharkForceInlineReleaseOnly int32_t
CountLeadingZeros (
    const uint32_t x) {
    int32_t count = 0;
    for (int32_t bit = 31; bit >= 0; --bit) {
        if (x & (1u << bit))
            break;
        ++count;
    }
    return count;
}

//
// Multi-word shift routines for little-endian arrays
//

// MultiWordRightShift_LittleEndian: shift an array 'in' (of length n) right by L bits,
// storing the result in 'out'. (out and in may be distinct.)
__device__ SharkForceInlineReleaseOnly void
MultiWordRightShift_LittleEndian (
    const uint32_t *in,
    const int32_t inN,
    const int32_t L,
    uint32_t *out,
    const int32_t outSz) {
    assert(inN >= outSz);

    for (int32_t i = 0; i < outSz; i++) {
        out[i] = ShiftRight(in, L, i, inN);
    }
}

// MultiWordLeftShift_LittleEndian: shift an array 'in' (of length n) left by L bits,
// storing the result in 'out'.
__device__ SharkForceInlineReleaseOnly void
MultiWordLeftShift_LittleEndian (
    const uint32_t *in,
    const int32_t numActualDigitsPlusGuard,
    const int32_t numActualDigits,
    const int32_t L,
    uint32_t *out,
    const int32_t outSz) {
    assert(numActualDigitsPlusGuard >= outSz);

    for (int32_t i = 0; i < outSz; i++) {
        out[i] = ShiftLeft(in, numActualDigits, numActualDigitsPlusGuard, L, i);
    }
}

__device__ SharkForceInlineReleaseOnly uint32_t
GetExtLimb (
    const uint32_t *ext,
    const int32_t numActualDigits,
    const int32_t numActualDigitsPlusGuard,
    const int32_t idx) {

    if (idx < numActualDigits) {
        return ext[idx];
    } else {
        assert(idx < numActualDigitsPlusGuard);
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
__device__ SharkForceInlineReleaseOnly int32_t
ExtendedNormalizeShiftIndex (
    const uint32_t *ext,
    const int32_t numActualDigits,
    const int32_t numActualDigitsPlusGuard,
    int32_t &storedExp,
    bool &isZero) {
    int32_t msd = numActualDigitsPlusGuard - 1;
    while (msd >= 0 && GetExtLimb(ext, numActualDigits, numActualDigitsPlusGuard, msd) == 0)
        msd--;
    if (msd < 0) {
        isZero = true;
        return 0;  // For zero, the shift offset is irrelevant.
    }
    isZero = false;
    const int32_t clz = CountLeadingZeros(GetExtLimb(ext, numActualDigits, numActualDigitsPlusGuard, msd));
    // In little-endian, the overall bit index of the MSB is:
    //    current_msb = msd * 32 + (31 - clz)
    const int32_t current_msb = msd * 32 + (31 - clz);
    const int32_t totalExtBits = numActualDigitsPlusGuard * 32;
    // Compute the left-shift needed so that the MSB moves to bit (totalExtBits - 1).
    const int32_t L = (totalExtBits - 1) - current_msb;
    // Adjust the exponent as if we had shifted the number left by L bits.
    storedExp -= L;
    return L;
}

//
// Helper to retrieve a normalized digit on the fly.
// Given the original extended array and a shift offset (obtained from ExtendedNormalizeShiftIndex),
// this returns the digit at index 'idx' as if the array had been left-shifted by shiftOffset bits.
//
__device__ SharkForceInlineReleaseOnly uint32_t
GetNormalizedDigit (
    const uint32_t *ext,
    const int32_t numActualDigits,
    const int32_t numActualDigitsPlusGuard,
    const int32_t shiftOffset,
    const int32_t idx) {
    return ShiftLeft(ext, numActualDigits, numActualDigitsPlusGuard, shiftOffset, idx);
}

// New helper: Computes the aligned digit for the normalized value on the fly.
// 'diff' is the additional right shift required for alignment.
template <class SharkFloatParams>
__device__ SharkForceInlineReleaseOnly uint32_t
GetShiftedNormalizedDigit (
    const uint32_t *ext,
    const int32_t numActualDigits,
    const int32_t numActualDigitsPlusGuard,
    const int32_t shiftOffset,
    const int32_t diff,
    const int32_t idx) {
    // const int32_t n = SharkFloatParams::GlobalNumUint32; // normalized length
    const int32_t wordShift = diff / 32;
    const int32_t bitShift = diff % 32;
    const uint32_t lower = (idx + wordShift < numActualDigitsPlusGuard) ?
        GetNormalizedDigit(ext, numActualDigits, numActualDigitsPlusGuard, shiftOffset, idx + wordShift) : 0;
    const uint32_t upper = (idx + wordShift + 1 < numActualDigitsPlusGuard) ?
        GetNormalizedDigit(ext, numActualDigits, numActualDigitsPlusGuard, shiftOffset, idx + wordShift + 1) : 0;
    if (bitShift == 0)
        return lower;
    else
        return (lower >> bitShift) | (upper << (32 - bitShift));
}

__device__ SharkForceInlineReleaseOnly bool
CompareMagnitudes2Way(
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
GetCorrespondingLimbs(
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

template <class SharkFloatParams>
__device__ void AddHelper (
    cg::grid_group &grid,
    cg::thread_block &block,
    HpSharkAddComboResults<SharkFloatParams> *SharkRestrict combo,
    uint64_t *tempData) {

    extern __shared__ uint32_t sharedData[];

    // --- Constants and Parameters ---
    constexpr int32_t guard = 4;
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

    const bool IsNegativeA = A_X2->IsNegative;
    const bool IsNegativeB = !B_Y2->IsNegative; // A - B + C
    const bool IsNegativeC = C_A->IsNegative;
    const bool IsNegativeD = D_2X->IsNegative;
    const bool IsNegativeE = E_B->IsNegative;

    auto *Out_A_B_C = &combo->Result1_A_B_C;
    auto *Out_D_E = &combo->Result2_D_E;

    constexpr auto GlobalSync_offset = 0;
    constexpr auto Checksum_offset = AdditionalGlobalSyncSpace;
    constexpr auto Final128Offset_ABC = AdditionalUInt64Global;
    constexpr auto Final128Offset_DE = Final128Offset_ABC + 2 * SharkFloatParams::GlobalNumUint32;
    constexpr auto Carry1_offset = Final128Offset_DE + 8 * SharkFloatParams::GlobalNumUint32;
    constexpr auto Carry2_offset = Carry1_offset + 4 * SharkFloatParams::GlobalNumUint32;

    auto *SharkRestrict globalSync =
        reinterpret_cast<uint32_t *>(&tempData[GlobalSync_offset]);
    auto *SharkRestrict debugStates =
        reinterpret_cast<DebugState<SharkFloatParams>*>(&tempData[Checksum_offset]);
    auto *SharkRestrict final128_ABC =
        reinterpret_cast<uint64_t *>(&tempData[Final128Offset_ABC]);
    auto *SharkRestrict final128_DE =
        reinterpret_cast<uint64_t *>(&tempData[Final128Offset_DE]);
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
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z1_offsetXX>(record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z1_offsetXY>(record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z1_offsetYY>(record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Final128XX>(record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Final128XY>(record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Final128YY>(record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::FinalAdd1>(record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::FinalAdd2>(record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Result_offsetXX>(record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Result_offsetXY>(record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Result_offsetYY>(record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Result_Add1>(record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Result_Add2>(record, debugStates, grid, block);
        static_assert(static_cast<int32_t>(DebugStatePurpose::NumPurposes) == 34, "Unexpected number of purposes");

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
    const bool sameSign = (D_2X->IsNegative == E_B->IsNegative);

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
    int32_t outExponent_ABC = 0;
    bool isNegative_ABC = false;
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
        biasedExpABC,
        bias,
        isNegative_ABC,
        outExponent_ABC,
        final128_ABC,
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

    int32_t carryAcc_DE = 0;
    int32_t carryAcc_ABC = 0;
    if constexpr (!SharkFloatParams::DisableCarryPropagation) {
        // --- Carry/Borrow Propagation ---
        CarryPropagationDE<SharkFloatParams>(
            sharedData,
            globalSync,
            final128_DE,
            numActualDigitsPlusGuard,
            carry1,
            carry2,
            carryAcc_DE,
            sameSign,
            block,
            grid);

        CarryPropagation_ABC<SharkFloatParams>(
            sharedData,
            globalSync,
            idx,
            numActualDigitsPlusGuard,
            final128_ABC,
            carry1,
            carry2,
            carryAcc_ABC,
            block,
            grid);
    }

    if constexpr (SharkDebugChecksums) {
        grid.sync();
        StoreCurrentDebugState<SharkFloatParams, CallIndex, DebugStatePurpose::FinalAdd1, uint64_t>(
            record, debugStates, grid, block, final128_ABC, numActualDigitsPlusGuard);
        StoreCurrentDebugState<SharkFloatParams, CallIndex, DebugStatePurpose::FinalAdd2, uint64_t>(
            record, debugStates, grid, block, final128_DE, numActualDigitsPlusGuard);
        grid.sync();
    } else {
        grid.sync();
    }

    // --- Final Normalization ---
    // Thread 0 finds the most-significant digit (msd) and computes the shift needed.
    auto handleFinalCarry = [](
        int32_t &carryAcc,
        const int32_t numActualDigitsPlusGuard,
        const int32_t numActualDigits,
        uint64_t *final128
        ) {
            int32_t msdResult;

            if (carryAcc) {
                int32_t msdResult = numActualDigitsPlusGuard - 1;
                int32_t clzResult = 0;
                int32_t currentOverall = msdResult * 32 + (31 - clzResult);
                int32_t desiredOverall = (numActualDigits - 1) * 32 + 31;
                carryAcc += currentOverall - desiredOverall;
            } else {
                msdResult = 0;
                for (int32_t i = numActualDigitsPlusGuard - 1; i >= 0; i--) {
                    if (final128[i] != 0) {
                        msdResult = i;
                        break;
                    }
                }

                int32_t clzResult = __clz(final128[msdResult]);
                int32_t currentOverall = msdResult * 32 + (31 - clzResult);
                int32_t desiredOverall = (numActualDigits - 1) * 32 + 31;
                carryAcc += currentOverall - desiredOverall;
            }
        };

    handleFinalCarry(
        carryAcc_ABC,
        numActualDigitsPlusGuard,
        numActualDigits,
        final128_ABC
    );

    handleFinalCarry(
        carryAcc_DE,
        numActualDigitsPlusGuard,
        numActualDigits,
        final128_DE
    );

    const int32_t stride = blockDim.x * gridDim.x;

    auto finalResolution = [](
        const int32_t idx,
        const int32_t stride,
        const int32_t carryAcc_DE,
        const int32_t numActualDigitsPlusGuard,
        const int32_t numActualDigits,
        const uint64_t *final128_DE,
        HpSharkFloat<SharkFloatParams> *OutSharkFloat,
        int32_t &outExponent_DE
        ) {
            if (carryAcc_DE > 0) {
                // Make sure carryAcc_DE, numActualDigitsPlusGuard, numActualDigits, and final128_DE are computed and 
                // available to all threads
                int32_t wordShift = carryAcc_DE / 32;
                int32_t bitShift = carryAcc_DE % 32;

                // Each thread handles a subset of indices.
                for (int32_t i = idx; i < numActualDigits; i += stride) {
                    uint32_t lower = (i + wordShift < numActualDigitsPlusGuard) ? final128_DE[i + wordShift] : 0;
                    uint32_t upper = (i + wordShift + 1 < numActualDigitsPlusGuard) ? final128_DE[i + wordShift + 1] : 0;
                    OutSharkFloat->Digits[i] = (bitShift == 0) ? lower : (lower >> bitShift) | (upper << (32 - bitShift));

                    if (i == numActualDigits - 1) {
                        // Set the high-order bit of the last digit.
                        OutSharkFloat->Digits[numActualDigits - 1] |= (1u << 31);
                    }
                }

                outExponent_DE += carryAcc_DE;
            } else if (carryAcc_DE < 0) {
                int32_t wordShift = (-carryAcc_DE) / 32;
                int32_t bitShift = (-carryAcc_DE) % 32;

                for (int32_t i = idx; i < numActualDigits; i += stride) {
                    int32_t srcIdx = i - wordShift;
                    uint32_t lower = (srcIdx >= 0 && srcIdx < numActualDigitsPlusGuard) ? final128_DE[srcIdx] : 0;
                    uint32_t upper = (srcIdx - 1 >= 0 && srcIdx - 1 < numActualDigitsPlusGuard) ? final128_DE[srcIdx - 1] : 0;
                    OutSharkFloat->Digits[i] = (bitShift == 0) ? lower : (lower << bitShift) | (upper >> (32 - bitShift));
                }

                if (idx == 0) {
                    outExponent_DE -= (-carryAcc_DE);
                }
            } else {
                // No shifting needed; simply copy.  Convert to uint32_t along the way

                for (int32_t i = idx; i < numActualDigits; i += stride) {
                    OutSharkFloat->Digits[i] = final128_DE[i];
                }
            }
        };

    finalResolution(
        idx,
        stride,
        carryAcc_ABC,
        numActualDigitsPlusGuard,
        numActualDigits,
        final128_ABC,
        Out_A_B_C,
        outExponent_ABC
    );

    finalResolution(
        idx,
        stride,
        carryAcc_DE,
        numActualDigitsPlusGuard,
        numActualDigits,
        final128_DE,
        Out_D_E,
        outExponent_DE
    );

    if (idx == 0) {
        Out_D_E->Exponent = outExponent_DE;
        Out_D_E->IsNegative = sameSign ? A_X2->IsNegative : (DIsBiggerMagnitude ? A_X2->IsNegative : B_Y2->IsNegative);

        Out_A_B_C->Exponent = outExponent_ABC;
        Out_A_B_C->IsNegative = isNegative_ABC;
    }

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
__global__ void AddKernel(
    HpSharkAddComboResults<SharkFloatParams> *SharkRestrict combo,
    uint64_t *tempData) {

    // Initialize cooperative grid group
    cg::grid_group grid = cg::this_grid();
    cg::thread_block block = cg::this_thread_block();

    // Call the AddHelper function
    AddHelper(grid, block, combo, tempData);
}

template<class SharkFloatParams>
__global__ void AddKernelTestLoop(
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
void ComputeAddGpu(void *kernelArgs[]) {

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
void ComputeAddGpuTestLoop(void *kernelArgs[]) {

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
