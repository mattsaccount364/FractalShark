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
    const int32_t actualDigits,
    const int32_t extDigits,
    const int32_t shiftBits,
    const int32_t idx) {
    const int32_t shiftWords = shiftBits / 32;
    const int32_t shiftBitsMod = shiftBits % 32;
    const int32_t srcIdx = idx - shiftWords;

    int32_t srcDigitLower = (srcIdx >= 0 && srcIdx < actualDigits) ? digits[srcIdx] : 0;
    int32_t srcDigitUpper = (srcIdx - 1 >= 0 && srcIdx - 1 < actualDigits) ? digits[srcIdx - 1] : 0;

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
    const int32_t extDigits,
    const int32_t actualDigits,
    const int32_t L,
    uint32_t *out,
    const int32_t outSz) {
    assert(extDigits >= outSz);

    for (int32_t i = 0; i < outSz; i++) {
        out[i] = ShiftLeft(in, actualDigits, extDigits, L, i);
    }
}

__device__ SharkForceInlineReleaseOnly uint32_t
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
__device__ SharkForceInlineReleaseOnly int32_t
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

//
// Helper to retrieve a normalized digit on the fly.
// Given the original extended array and a shift offset (obtained from ExtendedNormalizeShiftIndex),
// this returns the digit at index 'idx' as if the array had been left-shifted by shiftOffset bits.
//
__device__ SharkForceInlineReleaseOnly uint32_t
GetNormalizedDigit (
    const uint32_t *ext,
    const int32_t actualDigits,
    const int32_t extDigits,
    const int32_t shiftOffset,
    const int32_t idx) {
    return ShiftLeft(ext, actualDigits, extDigits, shiftOffset, idx);
}

// New helper: Computes the aligned digit for the normalized value on the fly.
// 'diff' is the additional right shift required for alignment.
template <class SharkFloatParams>
__device__ SharkForceInlineReleaseOnly uint32_t
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

template<class SharkFloatParams>
__device__ SharkForceInlineReleaseOnly void
GetCorrespondingLimbs (
    const uint32_t *extA,
    const int32_t actualASize,
    const int32_t extASize,
    const uint32_t *extB,
    const int32_t actualBSize,
    const int32_t extBSize,
    const int32_t shiftA,
    const int32_t shiftB,
    const bool AIsBiggerMagnitude,
    const int32_t diff,
    const int32_t index,
    uint64_t &alignedA,
    uint64_t &alignedB) {
    if (AIsBiggerMagnitude) {
        // A is larger: normalized A is used as is.
        // For B, we normalize and then shift right by 'diff'.
        alignedA = GetNormalizedDigit(extA, actualASize, extASize, shiftA, index);
        alignedB = GetShiftedNormalizedDigit<SharkFloatParams>(
            extB,
            actualBSize,
            extBSize,
            shiftB,
            diff,
            index);
    } else {
        // B is larger: normalized B is used as is.
        // For A, we normalize and shift right by 'diff'.
        alignedB = GetNormalizedDigit(extB, actualBSize, extBSize, shiftB, index);
        alignedA = GetShiftedNormalizedDigit<SharkFloatParams>(
            extA,
            actualASize,
            extASize,
            shiftA,
            diff,
            index);
    }
}

__device__ SharkForceInlineReleaseOnly bool
CompareMagnitudes (
    int32_t effExpA,
    int32_t effExpB,
    const int32_t actualDigits,
    const int32_t extDigits,
    const uint32_t *extA,
    const int32_t shiftA,
    const uint32_t *extB,
    const int32_t shiftB) {
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

template<
    class SharkFloatParams,
    DebugStatePurpose Purpose>
__device__ SharkForceInlineReleaseOnly void
EraseCurrentDebugState (
    RecordIt record,
    DebugState<SharkFloatParams> *debugChecksumArray,
    cooperative_groups::grid_group &grid,
    cooperative_groups::thread_block &block) {

    constexpr auto RecursionDepth = 0;
    constexpr auto CallIndex = 0;
    constexpr auto maxPurposes = static_cast<int>(DebugStatePurpose::NumPurposes);
    constexpr auto curPurpose = static_cast<int>(Purpose);
    debugChecksumArray[CallIndex * maxPurposes + curPurpose].Erase(
        record, grid, block, Purpose, RecursionDepth, CallIndex);
}

template<
    class SharkFloatParams,
    int CallIndex,
    DebugStatePurpose Purpose,
    typename ArrayType>
__device__ SharkForceInlineReleaseOnly void
StoreCurrentDebugState (
    RecordIt record,
    DebugState<SharkFloatParams> *debugChecksumArray,
    cooperative_groups::grid_group &grid,
    cooperative_groups::thread_block &block,
    const ArrayType *arrayToChecksum,
    size_t arraySize) {

    constexpr auto CurPurpose = static_cast<int>(Purpose);
    constexpr auto RecursionDepth = 0;
    constexpr auto UseConvolutionHere = UseConvolution::No;

    debugChecksumArray[CurPurpose].Reset(
        record, UseConvolutionHere, grid, block, arrayToChecksum, arraySize, Purpose, RecursionDepth, CallIndex);
}

template <class SharkFloatParams>
__device__ SharkForceInlineReleaseOnly void
SerialCarryPropagation (
    uint32_t *sharedData,
    uint32_t *globalSync,
    uint64_t *final128,
    const int32_t extDigits,
    uint32_t *carry,      // global memory array for intermediate carries (length extDigits+1)
    int32_t &outExponent,  // note: if you need the updated exponent outside, you might pass this by reference
    bool sameSign,
    cg::thread_block &block,
    cg::grid_group &grid)
{
    // (For brevity, we perform a sequential prefix scan in thread 0.
    // In a production code you would replace this with a parallel prefix-sum algorithm.)
    if (block.thread_index().x == 0 && block.group_index().x == 0) {
        if (sameSign) {
            // Addition: propagate carry.
            uint64_t carry = 0;
            for (int i = 0; i < extDigits; i++) {
                uint64_t sum = (uint64_t)final128[i] + carry;
                final128[i] = (uint32_t)(sum & 0xFFFFFFFF);
                carry = sum >> 32;
            }
            uint32_t finalCarry = (uint32_t)carry;
            if (finalCarry > 0) {
                outExponent += 1;
                // Right-shift the extended result by one bit.
                uint32_t nextBit = finalCarry & 1;
                for (int i = extDigits - 1; i >= 0; i--) {
                    uint32_t current = final128[i];
                    final128[i] = (current >> 1) | (nextBit << 31);
                    nextBit = current & 1;
                }
            }
        } else {
            // Subtraction: propagate borrow.
            uint32_t borrow = 0;
            for (int i = 0; i < extDigits; i++) {
                int64_t diffVal = (int64_t)final128[i] - borrow;
                if (diffVal < 0) {
                    diffVal += ((uint64_t)1 << 32);
                    borrow = 1;
                } else {
                    borrow = 0;
                }
                final128[i] = (uint32_t)(diffVal & 0xFFFFFFFF);
            }
            // Optionally, check that the final borrow is zero.
            assert(borrow == 0);
        }
    }
}

template <class SharkFloatParams>
__device__ SharkForceInlineReleaseOnly void
CarryPropagation (
    uint32_t *sharedData,
    uint32_t *globalSync,   // global sync array; element 0 is used for borrow/carry count
    uint64_t *final128,
    const int32_t extDigits,
    uint32_t *carry1,        // global memory array for intermediate carries/borrows (length extDigits+1)
    uint32_t *carry2,        // global memory array for intermediate carries/borrows (length extDigits+1)
    int32_t &outExponent,    // note: if you need the updated exponent outside, you might pass this by reference
    int32_t &finalShiftRight,
    bool sameSign,
    cg::thread_block &block,
    cg::grid_group &grid) {

    // Compute a grid-global thread id and stride.
    const int globalIdx = block.thread_index().x + block.group_index().x * blockDim.x;
    const int stride = grid.size();

    // We'll use a fixed number of iterations.
    constexpr int maxIter = 1000;

    // Pointer to a single global counter used for convergence.
    auto *carries_remaining_global = &globalSync[0];
    auto *curCarry = carry1;
    auto *nextCarry = carry2;

    // Set an initial nonzero carry flag.
    if (block.group_index().x == 0 && block.thread_index().x == 0) {
        *carries_remaining_global = 1;
    }

    auto existing_carries_remaining = 0;

    // Zero out the carry array.
    for (int i = globalIdx; i < extDigits; i += stride) {
        carry1[i] = 0;
        carry2[i] = 0;
    }
    grid.sync();

    if (sameSign) {
        // --- Addition: Propagate carry ---
        // Loop a fixed number of iterations. For random numbers the carry
        // propagation will usually "settle" in only a few iterations.
        for (int iter = 0; iter < maxIter; iter++) {

            // Convergence check and reset the counter.
            // If no carries were produced in the previous iteration, we can exit.
            if (*carries_remaining_global == existing_carries_remaining) {
                // Break out of the loop.
                // (This branch is taken if no thread produced any new carry.)
                // Note: The counter was set during the previous iteration.
                // Reset is done before starting the next pass.
                break;
            }

            // Reset the counter for the next iteration.
            existing_carries_remaining = *carries_remaining_global;

            grid.sync();

            uint32_t totalNewCarry = 0;
            // Each thread updates its assigned digits in a grid-stride loop.
            for (int i = globalIdx; i < extDigits; i += stride) {
                uint32_t incoming = (i == 0) ? 0u : curCarry[i];
                uint64_t sum = final128[i] + incoming;
                uint32_t newDigit = (uint32_t)(sum & 0xFFFFFFFFu);
                uint32_t newCarry = (uint32_t)(sum >> 32);
                final128[i] = newDigit;
                if (i < extDigits - 1) {
                    nextCarry[i + 1] = newCarry;
                } else {
                    // Always merge the new carry with the previous final carry.
                    nextCarry[i + 1] = curCarry[i + 1] | newCarry;
                }

                totalNewCarry += newCarry;
            }

            grid.sync();

            if (totalNewCarry > 0) {
                // Atomically accumulate the new carry count.
                atomicAdd(carries_remaining_global, 1);
            }

            // Swap the carry arrays for the next iteration.
            auto *temp = curCarry;
            curCarry = nextCarry;
            nextCarry = temp;

            grid.sync();
        }

        // After the loop, thread 0 checks if a final carry remains.
        uint32_t finalCarry = curCarry[extDigits];
        if (finalCarry > 0u) {
            finalShiftRight = 1;
        }

    } else {
        // --- Subtraction: Propagate borrow ---
        for (int iter = 0; iter < maxIter; iter++) {
            // Convergence check and reset the counter.
            // If no carries were produced in the previous iteration, we can exit.
            if (*carries_remaining_global == existing_carries_remaining) {
                // Break out of the loop.
                // (This branch is taken if no thread produced any new carry.)
                // Note: The counter was set during the previous iteration.
                // Reset is done before starting the next pass.
                break;
            }

            // Reset the counter for the next iteration.
            existing_carries_remaining = *carries_remaining_global;

            // Reset the counter for the next iteration.
            grid.sync();

            uint32_t totalNewBorrow = 0;
            // Each thread processes its grid-stride subset.
            for (int i = globalIdx; i < extDigits; i += stride) {
                // For digit 0, no incoming borrow.
                uint32_t incoming = (i == 0) ? 0u : curCarry[i];
                int64_t diffVal = (int64_t)final128[i] - incoming;
                uint32_t newBorrow = 0;

                if (diffVal < 0) {
                    diffVal += (1ULL << 32);
                    newBorrow = 1;
                }

                final128[i] = (uint32_t)(diffVal & 0xFFFFFFFFu);
                nextCarry[i + 1] = newBorrow;
                totalNewBorrow += newBorrow;
            }

            grid.sync();

            if (totalNewBorrow > 0) {
                // Accumulate new borrows into the global counter.
                atomicAdd(carries_remaining_global, 1);
            }

            // Swap the carry arrays for the next iteration.
            auto *temp = curCarry;
            curCarry = nextCarry;
            nextCarry = temp;

            grid.sync();
        }

        // Finally, thread 0 checks that no borrow remains.
        if (globalIdx == 0) {
            assert(curCarry[extDigits] == 0);
        }
    }

    grid.sync();
}

template <class SharkFloatParams>
__device__ void AddHelper(
    cg::grid_group &grid,
    cg::thread_block &block,
    HpSharkAddComboResults<SharkFloatParams> *SharkRestrict combo,
    uint32_t *tempData) {

    extern __shared__ uint32_t sharedData[];

    // --- Constants and Parameters ---
    constexpr int32_t guard = 2;
    constexpr int32_t actualDigits = SharkFloatParams::GlobalNumUint32;
    constexpr int32_t extDigits = SharkFloatParams::GlobalNumUint32 + guard;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    constexpr int NewN = SharkFloatParams::GlobalNumUint32;

    const auto *A = &combo->A;
    const auto *B = &combo->B;
    auto *OutXY = &combo->ResultX2;
    const auto *extA = A->Digits;
    const auto *extB = B->Digits;

    constexpr auto GlobalSync_offset = 0;
    auto *SharkRestrict globalSync =
        reinterpret_cast<uint32_t *>(&tempData[GlobalSync_offset]);
    
    constexpr auto Checksum_offset = AdditionalGlobalSyncSpace;
    auto *SharkRestrict debugChecksumArray =
        reinterpret_cast<DebugState<SharkFloatParams>*>(&tempData[Checksum_offset]);

    constexpr auto Final128Offset = AdditionalUInt64Global;
    auto *SharkRestrict final128 =
        reinterpret_cast<uint64_t *>(&tempData[Final128Offset]);
    
    constexpr auto Carry1_offset = Final128Offset + 4 * SharkFloatParams::GlobalNumUint32;
    auto *SharkRestrict carry1 =
        reinterpret_cast<uint32_t *>(&tempData[Carry1_offset]);

    constexpr auto Carry2_offset = Carry1_offset + 4 * SharkFloatParams::GlobalNumUint32;
    auto *SharkRestrict carry2 =
        reinterpret_cast<uint32_t *>(&tempData[Carry2_offset]);

    const RecordIt record =
        (block.thread_index().x == 0 && block.group_index().x == 0) ?
        RecordIt::Yes :
        RecordIt::No;

    static constexpr auto CallIndex = 0;

    if constexpr (SharkDebugChecksums) {
        const RecordIt record =
            (block.thread_index().x == 0 && block.group_index().x == 0) ?
            RecordIt::Yes :
            RecordIt::No;
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Invalid>(record, debugChecksumArray, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::ADigits>(record, debugChecksumArray, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::BDigits>(record, debugChecksumArray, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::AHalfHigh>(record, debugChecksumArray, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::AHalfLow>(record, debugChecksumArray, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::BHalfHigh>(record, debugChecksumArray, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::BHalfLow>(record, debugChecksumArray, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::XDiff>(record, debugChecksumArray, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::YDiff>(record, debugChecksumArray, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z0XX>(record, debugChecksumArray, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z0XY>(record, debugChecksumArray, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z0YY>(record, debugChecksumArray, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z1XX>(record, debugChecksumArray, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z1XY>(record, debugChecksumArray, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z1YY>(record, debugChecksumArray, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z2XX>(record, debugChecksumArray, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z2XY>(record, debugChecksumArray, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z2YY>(record, debugChecksumArray, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z1_offsetXX>(record, debugChecksumArray, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z1_offsetXY>(record, debugChecksumArray, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z1_offsetYY>(record, debugChecksumArray, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Final128XX>(record, debugChecksumArray, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Final128XY>(record, debugChecksumArray, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Final128YY>(record, debugChecksumArray, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Result_offsetXX>(record, debugChecksumArray, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Result_offsetXY>(record, debugChecksumArray, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Result_offsetYY>(record, debugChecksumArray, grid, block);
        static_assert(static_cast<int>(DebugStatePurpose::NumPurposes) == 27, "Unexpected number of purposes");

        StoreCurrentDebugState<SharkFloatParams, CallIndex, DebugStatePurpose::ADigits, uint32_t>(
            record, debugChecksumArray, grid, block, A->Digits, NewN);
        StoreCurrentDebugState<SharkFloatParams, CallIndex, DebugStatePurpose::BDigits, uint32_t>(
            record, debugChecksumArray, grid, block, B->Digits, NewN);

        grid.sync();
    }

    // --- Extended Normalization using shift indices ---
    const bool sameSign = (A->IsNegative == B->IsNegative);
    bool normA_isZero = false, normB_isZero = false;
    int32_t newAExponent = A->Exponent;
    int32_t newBExponent = B->Exponent;
    const int32_t shiftA = ExtendedNormalizeShiftIndex(
        extA,
        actualDigits,
        extDigits,
        newAExponent,
        normA_isZero);

    const int32_t shiftB = ExtendedNormalizeShiftIndex(
        extB,
        actualDigits,
        extDigits,
        newBExponent,
        normB_isZero);

    // --- Compute Effective Exponents ---
    const int32_t effExpA = normA_isZero ? -100'000'000 : newAExponent + (SharkFloatParams::GlobalNumUint32 * 32 - 32);
    const int32_t effExpB = normB_isZero ? -100'000'000 : newBExponent + (SharkFloatParams::GlobalNumUint32 * 32 - 32);


    // --- Determine which operand has larger magnitude ---
    // If effective exponents differ, use them. If equal, compare normalized digits on the fly.
    const bool AIsBiggerMagnitude = CompareMagnitudes(
        effExpA,
        effExpB,
        actualDigits,
        extDigits,
        extA,
        shiftA,
        extB,
        shiftB);

    const int32_t diff = AIsBiggerMagnitude ? (effExpA - effExpB) : (effExpB - effExpA);
    int32_t outExponent = AIsBiggerMagnitude ? newAExponent : newBExponent;

    // --- Each thread computes its aligned limb.
    for (int i = idx; i < extDigits; i += blockDim.x * gridDim.x) {
        uint64_t alignedA = 0, alignedB = 0;

        uint64_t prelim = 0;
        if (sameSign) {
            GetCorrespondingLimbs<SharkFloatParams>(
                extA,
                actualDigits,
                extDigits,
                extB,
                actualDigits,
                extDigits,
                shiftA,
                shiftB,
                AIsBiggerMagnitude,
                diff,
                i,
                alignedA,
                alignedB);
            prelim = alignedA + alignedB;
        } else {
            // ---- Subtraction Branch ----
            if (AIsBiggerMagnitude) {
                uint64_t alignedA = 0, alignedB = 0;
                GetCorrespondingLimbs<SharkFloatParams>(
                    extA,
                    actualDigits,
                    extDigits,
                    extB,
                    actualDigits,
                    extDigits,
                    shiftA,
                    shiftB,
                    AIsBiggerMagnitude,
                    diff,
                    i,
                    alignedA,
                    alignedB);
                int64_t diffVal = (int64_t)alignedA - (int64_t)alignedB;
                prelim = diffVal;
            } else {
                uint64_t alignedA = 0, alignedB = 0;
                GetCorrespondingLimbs<SharkFloatParams>(
                    extA,
                    actualDigits,
                    extDigits,
                    extB,
                    actualDigits,
                    extDigits,
                    shiftA,
                    shiftB,
                    AIsBiggerMagnitude,
                    diff,
                    i,
                    alignedA,
                    alignedB);
                int64_t diffVal = (int64_t)alignedB - (int64_t)alignedA;
                prelim = diffVal;
            }
        }

        // Write preliminary result (without carry/borrow propagation) to global temporary.
        final128[i] = prelim;
    }

    if constexpr (SharkDebugChecksums) {
        grid.sync();
        StoreCurrentDebugState<SharkFloatParams, CallIndex, DebugStatePurpose::Z2XY, uint64_t>(
            record, debugChecksumArray, grid, block, final128, extDigits);
        grid.sync();
    } else {
        grid.sync();
    }

    // --- Carry/Borrow Propagation ---
    int shiftNeeded = 0;
    CarryPropagation<SharkFloatParams>(
        sharedData,
        globalSync,
        final128,
        extDigits,
        carry1,
        carry2,
        outExponent,
        shiftNeeded,
        sameSign,
        block,
        grid);

    if constexpr (SharkDebugChecksums) {
        grid.sync();
        StoreCurrentDebugState<SharkFloatParams, CallIndex, DebugStatePurpose::Final128XY, uint64_t>(
            record, debugChecksumArray, grid, block, final128, extDigits);
        grid.sync();
    } else {
        grid.sync();
    }

    // --- Final Normalization ---
    // Thread 0 finds the most-significant digit (msd) and computes the shift needed.
    int msdResult;

    bool injectHighOrderBit = (shiftNeeded > 0);

    if (injectHighOrderBit) {
        shiftNeeded = 1;

        int msdResult = extDigits - 1;
        int clzResult = 0;
        int currentOverall = msdResult * 32 + (31 - clzResult);
        int desiredOverall = (actualDigits - 1) * 32 + 31;
        shiftNeeded += currentOverall - desiredOverall;

    } else {
        msdResult = 0;
        for (int i = extDigits - 1; i >= 0; i--) {
            if (final128[i] != 0) {
                msdResult = i;
                break;
            }
        }

        int clzResult = __clz(final128[msdResult]);
        int currentOverall = msdResult * 32 + (31 - clzResult);
        int desiredOverall = (actualDigits - 1) * 32 + 31;
        shiftNeeded += currentOverall - desiredOverall;
    }

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    if (shiftNeeded > 0) {
        // Make sure shiftNeeded, extDigits, actualDigits, and final128 are computed and 
        // available to all threads (e.g. computed by thread 0 and then synchronized).
        int wordShift = shiftNeeded / 32;
        int bitShift = shiftNeeded % 32;

        // Each thread handles a subset of indices.
        for (int i = tid; i < actualDigits; i += stride) {
            uint32_t lower = (i + wordShift < extDigits) ? final128[i + wordShift] : 0;
            uint32_t upper = (i + wordShift + 1 < extDigits) ? final128[i + wordShift + 1] : 0;
            OutXY->Digits[i] = (bitShift == 0) ? lower : (lower >> bitShift) | (upper << (32 - bitShift));

            if (i == actualDigits - 1) {
                if (injectHighOrderBit) {
                    // Set the high-order bit of the last digit.
                    OutXY->Digits[actualDigits - 1] |= (1u << 31);
                }
            }
        }

        outExponent += shiftNeeded;
    } else if (shiftNeeded < 0) {
        int wordShift = (-shiftNeeded) / 32;
        int bitShift = (-shiftNeeded) % 32;

        for (int i = tid; i < actualDigits; i += stride) {
            int srcIdx = i - wordShift;
            uint32_t lower = (srcIdx >= 0 && srcIdx < extDigits) ? final128[srcIdx] : 0;
            uint32_t upper = (srcIdx - 1 >= 0 && srcIdx - 1 < extDigits) ? final128[srcIdx - 1] : 0;
            OutXY->Digits[i] = (bitShift == 0) ? lower : (lower << bitShift) | (upper >> (32 - bitShift));
        }

        if (tid == 0) {
            outExponent -= (-shiftNeeded);
        }
    } else {
        // No shifting needed; simply copy.  Convert to uint32_t along the way

        for (int i = tid; i < actualDigits; i += stride) {
            OutXY->Digits[i] = final128[i];
        }
    }

    if (idx == 0) {
        OutXY->Exponent = outExponent;
        // Set result sign.
        OutXY->IsNegative = sameSign ? A->IsNegative : (AIsBiggerMagnitude ? A->IsNegative : B->IsNegative);
    }

    if constexpr (SharkDebugChecksums) {
        grid.sync();
        StoreCurrentDebugState<SharkFloatParams, CallIndex, DebugStatePurpose::Result_offsetXY, uint32_t>(
            record, debugChecksumArray, grid, block, OutXY->Digits, actualDigits);
        grid.sync();
    } else {
        grid.sync();
    }
}


template<class SharkFloatParams>
__global__ void AddKernel(
    HpSharkAddComboResults<SharkFloatParams> *SharkRestrict combo,
    uint32_t *tempData) {

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
    uint32_t *tempData) {

    // Initialize cooperative grid group
    cg::grid_group grid = cg::this_grid();
    cg::thread_block block = cg::this_thread_block();

    for (int i = 0; i < numIters; ++i) {
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
