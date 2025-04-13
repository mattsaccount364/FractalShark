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

// Define our custom combine operator as a device lambda.
__device__ uint32_t
CombineBorrow(
    uint32_t x,
    uint32_t y
) {
    // x and y are encoded as: (sat << 1) | b.
    uint32_t sat_x = (x >> 1) & 1;
    uint32_t b_x = x & 1;
    uint32_t sat_y = (y >> 1) & 1;
    uint32_t b_y = y & 1;
    // The combined borrow propagates if:
    //   new_b = b_y OR (b_x AND sat_x)
    uint32_t new_b = b_y | (b_x & sat_x);
    // The saturation value is simply taken from the right element.
    return (sat_y << 1) | new_b;
}

// A small structure to hold the generate/propagate pair for a digit.
struct GenProp {
    uint32_t g; // generate: indicates that this digit produces a carry regardless of incoming carry.
    uint32_t p; // propagate: indicates that if an incoming carry exists, it will be passed along.
};

// The combine operator for two GenProp pairs.
// If you have a block with operator f(x) = g OR (p AND x),
// then the combination for two adjacent blocks is given by:
__device__ inline GenProp Combine (
    const GenProp &left,
    const GenProp &right) {
    GenProp out;
    out.g = right.g | (right.p & left.g);
    out.p = right.p & left.p;
    return out;
}

template <class SharkFloatParams>
__device__ inline void CarryPropagationPP2 (
    uint32_t *sharedData,   // allocated with at least 2*n*sizeof(uint32_t); unused here.
    uint32_t *globalSync,   // unused (provided for interface compatibility)
    uint64_t *final128,     // the extended result digits
    const int32_t extDigits,  // number of digits in final128 to process
    uint32_t *carry1,         // will be reinterpreted as an array of GenProp (working vector)
    uint32_t *carry2,         // will be reinterpreted as an array of GenProp (scratch buffer)
    int32_t &outExponent,     // (unchanged; provided for interface compatibility)
    int32_t &finalShiftRight, // will be set to 1 if there is an overall carry; 0 otherwise
    bool sameSign,            // true for addition; false for subtraction (borrow propagation)
    cg::thread_block &block,
    cg::grid_group &grid) {
    // Determine grid-stride parameters.
    const int totalThreads = grid.size();
    const int tid = block.thread_index().x + block.group_index().x * blockDim.x;

    // Reinterpret carry1 and carry2 as arrays of GenProp.
    GenProp *working = reinterpret_cast<GenProp *>(carry1);  // working array
    GenProp *scratch = reinterpret_cast<GenProp *>(carry2);    // scratch array

    //--------------------------------------------------------------------------
    // Phase 1: Initialize the per-digit signals.
    //
    // For addition (sameSign):
    //   working[i].g = (high 32 bits of final128[i] != 0) ? 1 : 0;
    //   working[i].p = (low 32 bits of final128[i] == 0xFFFFFFFF) ? 1 : 0;
    //
    // For subtraction:
    //   working[i].g = (int64_t(final128[i]) < 0) ? 1 : 0;
    //   working[i].p = ((low 32 bits == 0) && (high 32 bits == 0)) ? 1 : 0;
    //--------------------------------------------------------------------------
    for (int i = tid; i < extDigits; i += totalThreads) {
        uint64_t digit = final128[i];
        uint32_t lo = static_cast<uint32_t>(digit);
        uint32_t hi = static_cast<uint32_t>(digit >> 32);
        if (sameSign) {
            working[i].g = (hi != 0) ? 1 : 0;
            working[i].p = (lo == 0xFFFFFFFFu) ? 1 : 0;
        } else {
            int64_t raw = static_cast<int64_t>(digit);
            working[i].g = (raw < 0) ? 1 : 0;
            working[i].p = ((lo == 0) && (hi == 0)) ? 1 : 0;
        }
    }
    grid.sync();

    //--------------------------------------------------------------------------
    // Phase 2: Perform an inclusive scan on the working vector.
    //
    // We perform log2(extDigits) passes. In each pass, each thread processes 
    // one or more elements via a grid-stride loop. For each element i >= offset,
    // the new signal is computed as:
    //    scratch[i] = Combine(working[i-offset], working[i])
    // Otherwise, we simply copy working[i] to scratch[i].
    // Then we swap working and scratch.
    //--------------------------------------------------------------------------
    if (extDigits > 1) {
        for (int offset = 1; offset < extDigits; offset *= 2) {
            for (int i = tid; i < extDigits; i += totalThreads) {
                if (i >= offset) {
                    scratch[i] = Combine(working[i - offset], working[i]);
                } else {
                    scratch[i] = working[i];
                }
            }
            grid.sync();

            // Swap pointers so that 'working' always points to the most up-to-date array.
            GenProp *temp = working;
            working = scratch;
            scratch = temp;
            grid.sync();
        }
    }
    // At this point, working[0..extDigits-1] holds the inclusive scan results.

    //--------------------------------------------------------------------------
    // Phase 3: Convert to an exclusive scan and update the digits.
    //
    // The exclusive scan is defined by taking an identity for index 0 and
    // for i >= 1 using working[i-1]. (The identity is equivalent to a GenProp 
    // of {0, 1}, but since the overall initial carry is 0, we simply use 0.)
    //
    // For addition: update each digit with final128[i] = (final128[i] + incomingCarry) & 0xFFFFFFFF.
    // For subtraction: subtract the incoming borrow.
    //--------------------------------------------------------------------------
    for (int i = tid; i < extDigits; i += totalThreads) {
        uint32_t incoming = (i == 0) ? 0 : working[i - 1].g;
        if (sameSign) {
            uint64_t sum = final128[i] + incoming;
            final128[i] = sum & 0xFFFFFFFFULL;
        } else {
            int64_t diff = static_cast<int64_t>(final128[i]) - incoming;
            final128[i] = static_cast<uint32_t>(diff & 0xFFFFFFFFULL);
        }
    }
    grid.sync();

    //--------------------------------------------------------------------------
    // Phase 4: Final carry/borrrow update.
    //
    // For addition: the overall carry is given by working[extDigits - 1].g.
    // If it is nonzero, then we set finalShiftRight = 1.
    // For subtraction, we assume the final borrow is zero.
    //--------------------------------------------------------------------------
    if (sameSign) {
        uint32_t overallCarry = working[extDigits - 1].g;
        finalShiftRight = (overallCarry > 0) ? 1 : 0;
    }
    grid.sync();
}

// Older implementation buggy with add samesign == true.
template <class SharkFloatParams>
__device__ SharkForceInlineReleaseOnly void
CarryPropagationPP (
    uint32_t *sharedData,  // must be allocated with at least 2*n*sizeof(uint32_t)
    uint32_t *globalSync,  // unused in this version (still provided for interface compatibility)
    uint64_t *final128,
    const int32_t extDigits,
    uint32_t *carry1,
    uint32_t *carry2,
    int32_t &outExponent,
    int32_t &finalShiftRight,
    bool sameSign,
    cg::thread_block &block,
    cg::grid_group &grid) {
    // We assume that extDigits is small. Let n be the next power of two >= extDigits.
    const auto n = extDigits;

    // We use sharedData to hold two arrays (each of length n):
    // s_g[0..n-1] will hold the "generate" flag (0 or 1) for each digit,
    // s_p[0..n-1] will hold the "propagate" flag (0 or 1).
    // (For the exclusive scan we use the Blelloch algorithm.)
    uint32_t *s_g = carry1;       // first n elements
    uint32_t *s_p = carry2;       // next n elements

    static constexpr auto SequentialBits = true;

    const int totalThreads = grid.size();
    int tid = block.thread_index().x + block.group_index().x * blockDim.x;

    if (sameSign) {
        // --- Initialization ---
        // Only the first extDigits threads load a digit; for i >= extDigits we initialize to identity: (0,1).
        for (int i = tid; i < n; i += totalThreads) {
            // i < extDigits always here
            uint64_t x = final128[i];
            uint32_t low = (uint32_t)x;
            uint32_t hi = (uint32_t)(x >> 32);
            s_g[i] = (hi == 1) ? 1 : 0;
            s_p[i] = (low == 0xFFFFFFFF) ? 1 : 0;
        }
        grid.sync();

        // --- Upsweep phase (reduce) ---
        // For d = 1,2,4,..., n/2, each thread whose index fits combines a pair of nodes.
        if constexpr (!SequentialBits) {
            for (int d = 1; d < n; d *= 2) {
                int index = (tid + 1) * d * 2 - 1;  // each thread works on one index
                if (index < n) {
                    uint32_t g1 = s_g[index - d];
                    uint32_t p1 = s_p[index - d];
                    uint32_t g2 = s_g[index];
                    uint32_t p2 = s_p[index];
                    // Combine according to our operator:
                    // (g, p) = (g2 OR (p2 AND g1), p2 AND p1)
                    s_g[index] = g2 | (p2 & g1);
                    s_p[index] = p2 & p1;
                }
                grid.sync();
            }

            // --- Set the last element to the identity (for exclusive scan) ---
            if (tid == 0) {
                s_g[n - 1] = 0;
                s_p[n - 1] = 1;
            }
        } else {
            // Upsweep phase (reduce) � corrected operator (note the order of operands)
            if (block.thread_index().x == 0 && block.group_index().x == 0) {
                for (int d = 1; d < n; d *= 2) {
                    // Process every segment of 2*d elements.
                    for (int index = 2 * d - 1; index < n; index += 2 * d) {
                        // left child is at index - d, right child is at index.
                        uint32_t left_g = s_g[index - d];
                        uint32_t left_p = s_p[index - d];
                        uint32_t right_g = s_g[index];
                        uint32_t right_p = s_p[index];
                        // Combine using the left-to-right operator:
                        // (g, p) = (left_g OR (left_p & right_g), left_p AND right_p)
                        s_g[index] = left_g | (left_p & right_g);
                        s_p[index] = left_p & right_p;
                    }
                }
            }
            grid.sync();

            // --- Set the last element to the identity (for exclusive scan) ---
            if (tid == 0) {
                s_g[n - 1] = 0;
                s_p[n - 1] = 1;
            }
        }
        grid.sync();

        // --- Downsweep phase (exclusive scan for addition) ---
        // Now perform the downsweep: update only indices k satisfying ((k+1) mod (2*d)) == 0.
        if constexpr (!SequentialBits) {
            for (int d = n / 2; d >= 1; d /= 2) {
                // Use grid-stride loops to cover all indices.
                for (int k = block.thread_index().x + block.group_index().x * blockDim.x;
                    k < n;
                    k += grid.size()) {
                    // Check if k is the last index of its block of 2*d elements.
                    if (((k + 1) % (2 * d)) == 0) {
                        uint32_t temp = s_g[k - d];
                        s_g[k - d] = s_g[k];
                        if (d == 1) {
                            // On the final downsweep pass, simply assign the left value.
                            s_g[k] = temp;
                        } else {
                            s_g[k] = s_g[k] | (s_p[k] & temp); // standard combine for d > 1
                        }
                    }
                }

                grid.sync();
            }
        } else {
            if (block.thread_index().x == 0 && block.group_index().x == 0) {
                for (int d = n / 2; d >= 1; d /= 2) {
                    for (int base = 0; base < n; base += 2 * d) {
                        int k = base + 2 * d - 1; // rightmost node of the segment
                        uint32_t temp = s_g[k - d]; // save left child's g
                        s_g[k - d] = s_g[k];        // set left value to what the right child held
                        if (d == 1)
                            s_g[k] = temp;
                        else
                            s_g[k] = temp | (s_p[k - d] & s_g[k]);
                    }
                }
            }

            grid.sync();
        }

        // Now s_g[0..extDigits-1] contains the exclusive scan results.
        // In particular, for digit i the carry into that digit is given by s_g[i].
        grid.sync();

        // --- Update digits using the computed carries ---
        auto original_last_digit = final128[extDigits - 1]; // save the original last digit
        grid.sync();
        for (int i = tid; i < extDigits; i += totalThreads) {
            uint32_t carryIn = s_g[i]; // carry into digit i
            uint64_t sum = final128[i] + carryIn;
            // Write the 32-bit result back.
            final128[i] = sum & 0xFFFFFFFF;
        }
        grid.sync();

        // --- Determine overall final carry ---
        uint32_t carryIn = s_g[extDigits - 1];
        uint64_t lastVal = original_last_digit; // saved before updating final128[extDigits-1]
        uint64_t S = lastVal + carryIn;
        uint32_t finalCarry = S >> 32;
        if (finalCarry > 0u) {
            finalShiftRight = 1;
        }
    } else {
        // --- Subtraction branch (parallel custom scan for borrow propagation) ---
        // For subtraction we assume the final result is nonnegative so no borrow should remain.
        // The sequential logic is:
        //
        //   borrow[0] = 0,
        //   for i >= 1:
        //     borrow[i] = ( (final128[i-1] - incoming) < 0 )
        //               and then, if final128[i-1] (as 32-bit) equals 0xFFFFFFFF, the borrow propagates.
        //
        // We encode for each digit i (for i>=1) a 2-bit value:
        //   s_g[i] = (sat << 1) | b,
        // where b = 1 if final128[i-1] (the 64-bit preliminary difference) is negative,
        // and sat = 1 if the lower 32 bits of final128[i-1] equal 0xFFFFFFFF.
        //
        // For i == 0, no incoming borrow: we use 0.
        // For indices i >= extDigits (padding) we use the identity element for our operator,
        // which we choose as (sat=1, b=0) --> value 2.
        for (int i = tid; i < n; i += totalThreads) {
            if (i == 0) {
                // No incoming borrow.
                s_g[0] = 0;
            } else if (i < extDigits) {
                uint32_t b = (((int64_t)final128[i - 1] < 0) ? 1 : 0);
                uint32_t sat = (((uint32_t)final128[i - 1] == 0xFFFFFFFFu) ? 1 : 0);
                s_g[i] = (sat << 1) | b;
            } else {
                // For padded indices use the identity element (sat=1, b=0) so that it does not cancel a borrow.
                s_g[i] = 2;
            }
            // s_p is not used in this branch.
        }
        grid.sync();

        // --- Upsweep phase using custom operator ---
        for (int d = 1; d < n; d *= 2) {
            // Each thread processes one or more indices via a grid-stride loop.
            for (int index = (tid + 1) * d * 2 - 1; index < n; index += grid.size() * d * 2) {
                s_g[index] = CombineBorrow(s_g[index - d], s_g[index]);
            }
            grid.sync();
        }

        // --- Set the last element to the identity (0) for the downsweep ---
        if (tid == 0) {
            s_g[n - 1] = 2;
        }
        grid.sync();

        // --- Downsweep phase using custom operator ---
        // Parallel Hillis�Steele inclusive scan using CombineBorrow.
        // Note: 'I' is the identity for our CombineBorrow operator; for our encoding, we take I = 2.
        // (Assumes extDigits is <= the number of elements in s_g.)
        for (int d = 1; d < extDigits; d *= 2) {
            // Each thread processes indices in grid stride.
            for (int i = block.thread_index().x + block.group_index().x * blockDim.x;
                i < extDigits;
                i += grid.size()) {
                // For indices with i >= d, combine the element from i-d with s_g[i].
                uint32_t prev = (i >= d) ? s_g[i - d] : 2;  // Use identity for out-of-bound indexes.
                uint32_t cur = s_g[i];
                // Compute the inclusive scan value at index i.
                uint32_t combined = (i >= d) ? CombineBorrow(prev, cur) : cur;
                // Write this back to s_g[i]. (It is ok if the update is not strictly barrier-synchronized
                // between iterations, as long as we put an overall __syncthreads() after each step.)
                s_g[i] = combined;
            }
            grid.sync();
        }

        // Now s_g[0..extDigits-1] holds, in its lower bit, the incoming borrow for each digit.
        // Decode the borrow flag (the lower bit) and update final128 accordingly.
        for (int i = tid; i < extDigits; i += totalThreads) {
            uint32_t borrow = s_g[i] & 1;
            uint64_t diffVal = (uint64_t)final128[i] - borrow;
            final128[i] = (uint32_t)(diffVal & 0xFFFFFFFFu);
        }
    }
    grid.sync();
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
__device__ void AddHelper (
    cg::grid_group &grid,
    cg::thread_block &block,
    HpSharkAddComboResults<SharkFloatParams> *SharkRestrict combo,
    uint64_t *tempData) {

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
    
    constexpr auto Carry1_offset = Final128Offset + 8 * SharkFloatParams::GlobalNumUint32;
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
    CarryPropagationPP2<SharkFloatParams>(
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
