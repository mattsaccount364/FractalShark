#include "ReferenceKaratsuba.h"
#include "HpSharkFloat.cuh"
#include "DebugChecksumHost.h"
#include "DebugChecksum.cuh"

#include <cstdint>
#include <algorithm>
#include <cstring> // for memset
#include <vector>
#include <iostream>
#include <assert.h>
#include "ReferenceAdd.h"

static constexpr auto UseBellochPropagation = true;

//
// Helper functions to perform bit shifts on a fixed-width digit array.
// They mirror the CUDA device functions but work sequentially on the full array.
//

// ShiftRight: Shifts the number (given by its digit array) right by shiftBits.
// idx is the index of the digit to compute. The parameter numDigits prevents out-of-bounds access.
static uint32_t
ShiftRight (
    const uint32_t *digits,
    const int32_t shiftBits,
    const int32_t idx,
    const int32_t numDigits)
{
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
static uint32_t
ShiftLeft (
    const uint32_t *digits,
    const int32_t actualDigits,
    const int32_t extDigits,
    const int32_t shiftBits,
    const int32_t idx)
{
    const int32_t shiftWords = shiftBits / 32;
    const int32_t shiftBitsMod = shiftBits % 32;
    const int32_t srcIdx = idx - shiftWords;

    int32_t srcDigitLower;
    if (srcIdx < actualDigits) {
        srcDigitLower = digits[srcIdx];
    } else {
        assert(srcIdx < extDigits);
        srcDigitLower = 0;
    }

    int32_t srcDigitUpper;
    if (srcIdx - 1 >= 0) {
        srcDigitUpper = digits[srcIdx - 1];
    } else {
        assert(srcIdx - 1 < extDigits);
        srcDigitUpper = 0;
    }

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
static int32_t
CountLeadingZeros (
    const uint32_t x)
{
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
static void
MultiWordRightShift_LittleEndian (
    const uint32_t *in,
    const int32_t inN,
    const int32_t L,
    uint32_t *out,
    const int32_t outSz)
{
    assert(inN >= outSz);

    for (int32_t i = 0; i < outSz; i++) {
        out[i] = ShiftRight(in, L, i, inN);
    }
}

// MultiWordLeftShift_LittleEndian: shift an array 'in' (of length n) left by L bits,
// storing the result in 'out'.
static void
MultiWordLeftShift_LittleEndian(
    const uint32_t *in,
    const int32_t extDigits,
    const int32_t actualDigits,
    const int32_t L,
    uint32_t *out,
    const int32_t outSz)
{
    assert(extDigits >= outSz);

    for (int32_t i = 0; i < outSz; i++) {
        out[i] = ShiftLeft(in, actualDigits, extDigits, L, i);
    }
}

static uint32_t
GetExtLimb(
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
static int32_t
ExtendedNormalizeShiftIndex (
    const uint32_t *ext,
    const int32_t actualDigits,
    const int32_t extDigits,
    int32_t &storedExp,
    bool &isZero)
{
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
static uint32_t
GetNormalizedDigit (
    const uint32_t *ext,
    const int32_t actualDigits,
    const int32_t extDigits,
    const int32_t shiftOffset,
    const int32_t idx)
{
    return ShiftLeft(ext, actualDigits, extDigits, shiftOffset, idx);
}

// New helper: Computes the aligned digit for the normalized value on the fly.
// 'diff' is the additional right shift required for alignment.
template <class SharkFloatParams>
static uint32_t
GetShiftedNormalizedDigit (
    const uint32_t *ext,
    const int32_t actualDigits,
    const int32_t extDigits,
    const int32_t shiftOffset,
    const int32_t diff,
    const int32_t idx)
{
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
static void
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
    uint64_t &alignedB)
{
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

template<
    class SharkFloatParams,
    DebugStatePurpose Purpose,
    typename ArrayType>
static const DebugStateHost<SharkFloatParams> &
GetCurrentDebugState(
    std::vector<DebugStateHost<SharkFloatParams>> &debugChecksumArray,
    const ArrayType *arrayToChecksum,
    size_t arraySize) {

    constexpr auto curPurpose = static_cast<int>(Purpose);
    constexpr auto CallIndex = 0;
    constexpr auto UseConvolution = UseConvolution::No;
    constexpr auto RecursionDepth = 0;

    auto &retval = debugChecksumArray[curPurpose];
    retval.Reset(
        arrayToChecksum, arraySize, Purpose, RecursionDepth, CallIndex, UseConvolution);
    return retval;
}

static bool
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


// A small structure to hold the generate/propagate pair for a digit.
struct GenProp {
    uint32_t g; // generate: indicates that this digit produces a carry regardless of incoming carry.
    uint32_t p; // propagate: indicates that if an incoming carry exists, it will be passed along.
};

// The combine operator for two GenProp pairs.
// If you have a block with operator f(x) = g OR (p AND x),
// then the combination for two adjacent blocks is given by:
inline GenProp Combine (
    const GenProp &left,
    const GenProp &right) {
    GenProp out;
    out.g = right.g | (right.p & left.g);
    out.p = right.p & left.p;
    return out;
}

// Unified CarryPropagationPP.
// If sameSign is true, this is the addition branch;
// if false, it is the subtraction branch (we guarantee that the final result is positive).
template<class SharkFloatParams>
void CarryPropagationPP (
    const bool sameSign,
    const int32_t extDigits,
    const std::vector<uint64_t> extResultVector,
    uint32_t &carry,
    std::vector<uint32_t> &propagatedResultVector)
{
    // Check that the sizes are as expected.
    assert(extResultVector.size() == extDigits);
    const auto *extResult = extResultVector.data();
    assert(propagatedResultVector.size() == extDigits);
    uint32_t *propagatedResult = propagatedResultVector.data();

    // Step 1. Build the sigma vector (per-digit signals).
    std::vector<GenProp> working(extDigits);
    for (int i = 0; i < extDigits; i++) {
        if (sameSign) {
            // Addition case.
            uint32_t lo = static_cast<uint32_t>(extResult[i] & 0xFFFFFFFFULL);
            uint32_t hi = static_cast<uint32_t>(extResult[i] >> 32);
            working[i].g = hi;  // generates a carry if the high half is nonzero.
            working[i].p = (lo == 0xFFFFFFFFUL) ? 1 : 0;  // propagates incoming carry if low half is full.
        } else {
            // Subtraction case.
            int64_t raw = static_cast<int64_t>(extResult[i]);
            working[i].g = (raw < 0) ? 1 : 0;  // generate a borrow if the raw difference is negative.
            // For borrow propagation, a digit will propagate a borrow if it's exactly 0
            // (if we subtract 1 from 0, we need to borrow)
            uint32_t lo = static_cast<uint32_t>(extResult[i] & 0xFFFFFFFFULL);
            uint32_t hi = static_cast<uint32_t>(extResult[i] >> 32);
            working[i].p = (lo == 0 && hi == 0) ? 1 : 0;  // propagate if the entire digit is 0
        }
    }

    if constexpr (SharkFloatParams::HostVerbose) {
        std::cout << "CarryPropagationPP Working array:" << std::endl;
        for (int i = 0; i < extDigits; i++) {
            std::cout << "  " << i << ": g = " << working[i].g << ", p = " << working[i].p << std::endl;
        }
    }

    // Step 2. Perform an inclusive (upsweep) scan on the per-digit signals.
    // The inclusive array at index i contains the combined operator for sigma[0..i].
    assert(working.size() == extDigits);
    std::vector<GenProp> scratch(extDigits); // one scratch array of size extDigits

    // Use raw pointers that point to the current input and output buffers.
    GenProp *in = working.data();
    GenProp *out = scratch.data();

    // Perform the upsweep (inclusive scan) in log2(extDigits) passes.
    for (int offset = 1; offset < extDigits; offset *= 2) {
        for (int i = 0; i < extDigits; i++) {
            if (i >= offset)
                out[i] = Combine(in[i - offset], in[i]);
            else
                out[i] = in[i];
        }
        // Swap the roles for the next pass.
        std::swap(in, out);
    }

    // Now "in" points to the final inclusive scan result for indices 0 .. extDigits-1.
    const GenProp *inclusive = in;

    if constexpr (SharkFloatParams::HostVerbose) {
        std::cout << "CarryPropagationPP Inclusive array:" << std::endl;
        for (int i = 0; i < extDigits; i++) {
            std::cout << "  " << i << ": g = " << inclusive[i].g << ", p = " << inclusive[i].p << std::endl;
        }
    }

    // Step 3. Compute the carries (or borrows) via an exclusive scan.
    // The exclusive operator for digit i is taken to be:
    //   - For digit 0: use the identity operator {0, 1}.
    //   - For digit i (i>=1): use inclusive[i-1].
    // Then the carry/borrow applied to the digit is: op.g OR (op.p AND initialCarry).
    // We assume an initial carry (or borrow) of 0.
    GenProp identity = { 0, 1 };
    constexpr auto initialValue = 0;
    std::vector<uint32_t> carries(extDigits + 1, 0);
    carries[0] = initialValue;
    for (int i = 1; i <= extDigits; i++) {
        carries[i] = inclusive[i - 1].g | (inclusive[i - 1].p & initialValue);
    }

    if constexpr (SharkFloatParams::HostVerbose) {
        std::cout << "CarryPropagationPP Carries array:" << std::endl;
        for (int i = 0; i <= extDigits; i++) {
            std::cout << "  " << i << ": " << carries[i] << std::endl;
        }
    }

    // Step 4. Apply the computed carry/borrow to get the final 32-bit result.
    if (sameSign) {
        // Addition: add the carry.
        for (int i = 0; i < extDigits; i++) {
            uint64_t sum = extResult[i] + carries[i];
            propagatedResult[i] = static_cast<uint32_t>(sum & 0xFFFFFFFFULL);
        }
    } else {
        // Subtraction: subtract the borrow.
        // (By construction, the final overall borrow is guaranteed to be zero.)
        for (int i = 0; i < extDigits; i++) {
            int64_t diff = static_cast<int64_t>(extResult[i]) - carries[i];
            propagatedResult[i] = static_cast<uint32_t>(diff & 0xFFFFFFFFULL);
        }
    }

    if constexpr (SharkFloatParams::HostVerbose) {
        std::cout << "CarryPropagationPP Propagated result:" << std::endl;
        for (int i = 0; i < extDigits; i++) {
            std::cout << "  " << i << ": " << propagatedResult[i] << std::endl;
        }
    }

    // The final element (at position extDigits in the carries array)
    // is the overall carry (or borrow). For subtraction we expect this to be 0.
    carry = carries[extDigits];

    if constexpr (SharkFloatParams::HostVerbose) {
        std::cout << "CarryPropagationPP Final carry: " << carry << std::endl;
    }
}

template<class SharkFloatParams>
void CarryPropagation (
    const bool sameSign,
    const int32_t extDigits,
    std::vector<uint64_t> &extResult,
    uint32_t &carry,
    std::vector<uint32_t> &propagatedResult)
{
    if (sameSign) {
        // Propagate carry for addition.
        for (int32_t i = 0; i < extDigits; i++) {
            int64_t sum = (int64_t)extResult[i] + carry;
            propagatedResult[i] = (uint32_t)(sum & 0xFFFFFFFFULL);
            carry = sum >> 32;
        }

        // Note we'll handle final carry later.
    } else {
        // Propagate borrow for subtraction.
        int64_t borrow = 0;
        for (int32_t i = 0; i < extDigits; i++) {
            int64_t diffVal = (int64_t)extResult[i] - borrow;
            if (diffVal < 0) {
                diffVal += (1LL << 32);
                borrow = 1;
            } else {
                borrow = 0;
            }
            propagatedResult[i] = (uint32_t)(diffVal & 0xFFFFFFFFULL);
        }
        assert(borrow == 0 && "Final borrow in subtraction should be zero");
    }
}

//
// Extended arithmetic using little-endian representation.
// This version uses the new normalization approach, where the extended operands
// are not copied; instead, a shift index is returned and used later to compute
// normalized digits on the fly.
//
template<class SharkFloatParams>
void
AddHelper (
    const HpSharkFloat<SharkFloatParams> *A,
    const HpSharkFloat<SharkFloatParams> *B,
    HpSharkFloat<SharkFloatParams> *OutXY,
    std::vector<DebugStateHost<SharkFloatParams>> &debugStates
) {
    if constexpr (SharkDebugChecksums) {
        constexpr auto NewDebugStateSize = static_cast<int>(DebugStatePurpose::NumPurposes);
        debugStates.resize(NewDebugStateSize);
    }

    // Make local copies.
    const auto *extA = A->Digits;
    const auto *extB = B->Digits;

    // --- Set up extended working precision ---
    constexpr int32_t guard = 2;
    constexpr int32_t actualDigits = SharkFloatParams::GlobalNumUint32;
    constexpr int32_t extDigits = SharkFloatParams::GlobalNumUint32 + guard;
    // Create extended arrays (little-endian, index 0 is LSB).
    std::vector<uint64_t> extResult(extDigits, 0);

    // The guard words (indices GlobalNumUint32 to extDigits-1) are left as zero.

    if constexpr (SharkFloatParams::HostVerbose) {
        std::cout << "extA: " << VectorUintToHexString(extA, actualDigits) << std::endl;
        std::cout << "extA exponent: " << A->Exponent << std::endl;
        std::cout << "extB: " << VectorUintToHexString(extB, actualDigits) << std::endl;
        std::cout << "extB exponent: " << B->Exponent << std::endl;
    }

    if constexpr (SharkDebugChecksums) {
        const auto &debugAState = GetCurrentDebugState<SharkFloatParams, DebugStatePurpose::ADigits>(
            debugStates, extA, actualDigits);
        const auto &debugBState = GetCurrentDebugState<SharkFloatParams, DebugStatePurpose::BDigits>(
            debugStates, extB, actualDigits);

        if constexpr (SharkFloatParams::HostVerbose) {
            std::cout << "A->Digits checksum: " << debugAState.GetStr() << std::endl;
            std::cout << "B->Digits checksum: " << debugBState.GetStr() << std::endl;
        }
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

    if constexpr (SharkFloatParams::HostVerbose) {
        std::cout << "extA after normalization: " << VectorUintToHexString(extA, actualDigits) << std::endl;
        std::cout << "extB after normalization: " << VectorUintToHexString(extB, actualDigits) << std::endl;
        std::cout << "shiftA: " << shiftA << std::endl;
        std::cout << "shiftB: " << shiftB << std::endl;
    }

    // --- Compute Effective Exponents ---
    const int32_t effExpA = normA_isZero ? -100'000'000 : newAExponent + (SharkFloatParams::GlobalNumUint32 * 32 - 32);
    const int32_t effExpB = normB_isZero ? -100'000'000 : newBExponent + (SharkFloatParams::GlobalNumUint32 * 32 - 32);

    if constexpr (SharkFloatParams::HostVerbose) {
        std::cout << "effExpA: " << effExpA << std::endl;
        std::cout << "effExpB: " << effExpB << std::endl;
    }

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

    if constexpr (SharkFloatParams::HostVerbose) {
        std::cout << "AIsBiggerMagnitude: " << AIsBiggerMagnitude << std::endl;
        std::cout << "diff: " << diff << std::endl;
        std::cout << "outExponent: " << outExponent << std::endl;
    }

    // --- Phase 1: Raw Extended Arithmetic ---
    // Compute the raw limb-wise result without propagation.
    if (sameSign) {
        // Addition branch.
        for (int32_t i = 0; i < extDigits; i++) {
            uint64_t alignedA = 0, alignedB = 0;
            GetCorrespondingLimbs<SharkFloatParams>(
                extA, actualDigits, extDigits,
                extB, actualDigits, extDigits,
                shiftA, shiftB,
                AIsBiggerMagnitude, diff, i,
                alignedA, alignedB);
            extResult[i] = alignedA + alignedB;
        }
    } else {
        // Subtraction branch.
        std::vector<uint64_t> alignedADebug;
        std::vector<uint64_t> alignedBDebug;

        if (AIsBiggerMagnitude) {
            for (int32_t i = 0; i < extDigits; i++) {
                uint64_t alignedA = 0, alignedB = 0;
                GetCorrespondingLimbs<SharkFloatParams>(
                    extA, actualDigits, extDigits,
                    extB, actualDigits, extDigits,
                    shiftA, shiftB,
                    AIsBiggerMagnitude, diff, i,
                    alignedA, alignedB);
                // Compute raw difference (which may be negative).
                int64_t rawDiff = (int64_t)alignedA - (int64_t)alignedB;
                extResult[i] = (uint64_t)rawDiff;

                alignedADebug.push_back(alignedA);
                alignedBDebug.push_back(alignedB);
            }
        } else {
            for (int32_t i = 0; i < extDigits; i++) {
                uint64_t alignedA = 0, alignedB = 0;
                GetCorrespondingLimbs<SharkFloatParams>(
                    extA, actualDigits, extDigits,
                    extB, actualDigits, extDigits,
                    shiftA, shiftB,
                    AIsBiggerMagnitude, diff, i,
                    alignedA, alignedB);
                int64_t rawDiff = (int64_t)alignedB - (int64_t)alignedA;
                extResult[i] = (uint64_t)rawDiff;

                alignedADebug.push_back(alignedA);
                alignedBDebug.push_back(alignedB);
            }
        }

        if constexpr (SharkFloatParams::HostVerbose) {
            std::cout << "These are effectively the arrays we're adding and subtracting:" << std::endl;
            std::cout << "alignedADebug: " << VectorUintToHexString(alignedADebug) << std::endl;
            std::cout << "alignedBDebug: " << VectorUintToHexString(alignedBDebug) << std::endl;
        }
    }

    if constexpr (SharkDebugChecksums) {
        const auto &debugResultState = GetCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z2XY>(
            debugStates, extResult.data(), extDigits);

        if constexpr (SharkFloatParams::HostVerbose) {
            std::cout << "extResult checksum: " << debugResultState.GetStr() << std::endl;
            std::cout << "extResult after arithmetic: " << VectorUintToHexString(extResult) << std::endl;
        }
    }

    // --- Phase 2: Propagation ---
    // Propagate carries (if addition) or borrows (if subtraction)
    // and store the corrected 32-bit digit into propagatedResult.

    uint32_t carry = 0;
    std::vector<uint32_t> propagatedResult(extDigits, 0); // Result after propagation
    if constexpr (UseBellochPropagation) {
        CarryPropagationPP<SharkFloatParams>(
            sameSign,
            extDigits,
            extResult,
            carry,
            propagatedResult);
    } else {
        CarryPropagation<SharkFloatParams>(
            sameSign,
            extDigits,
            extResult,
            carry,
            propagatedResult);
    }

    // At this point, the propagatedResult array holds the result of the borrow/carry propagation.
    // A subsequent normalization step would adjust these digits (and the exponent) so that the most-significant
    // bit is in the desired position. This normalization step is omitted here.

    if constexpr (SharkFloatParams::HostVerbose) {
        std::cout << "propagatedResult after arithmetic: " << VectorUintToHexString(propagatedResult) << std::endl;
        std::cout << "outExponent after arithmetic, before renormalization: " << outExponent << std::endl;
        std::cout << "propagatedResult: " << VectorUintToHexString(propagatedResult) << std::endl;
        std::cout << "carry out: 0x" << std::hex << carry << std::endl;
    }

    if constexpr (SharkDebugChecksums) {
        const auto &debugResultState = GetCurrentDebugState<SharkFloatParams, DebugStatePurpose::Final128XY>(
            debugStates, propagatedResult.data(), extDigits);

        if constexpr (SharkFloatParams::HostVerbose) {
            std::cout << "propagatedResult checksum: " << debugResultState.GetStr() << std::endl;
        }
    }

    if (carry > 0) {
        outExponent += 1;
        // (Optionally, handle a final carry by shifting right by 1 bit.)
        // For example:
        uint32_t nextBit = (uint32_t)(carry & 1U);
        for (int32_t i = extDigits - 1; i >= 0; i--) {
            uint32_t current = propagatedResult[i];
            propagatedResult[i] = (current >> 1) | (nextBit << 31);
            nextBit = current & 1;
        }
    }

    // --- Final Normalization ---
    int32_t msdResult = 0;
    for (int32_t i = extDigits - 1; i >= 0; i--) {
        if (propagatedResult[i] != 0) {
            msdResult = i;
            break;
        }
    }

    const int32_t clzResult = CountLeadingZeros(propagatedResult[msdResult]);
    const int32_t currentOverall = msdResult * 32 + (31 - clzResult);
    const int32_t desiredOverall = (SharkFloatParams::GlobalNumUint32 - 1) * 32 + 31;

    if constexpr (SharkFloatParams::HostVerbose) {
        std::cout << "Count leading zeros: " << clzResult << std::endl;
        std::cout << "Current MSB index: " << msdResult << std::endl;
        std::cout << "Current overall bit position: " << currentOverall << std::endl;
        std::cout << "Desired overall bit position: " << desiredOverall << std::endl;
    }

    const int32_t shiftNeeded = currentOverall - desiredOverall;
    if (shiftNeeded > 0) {
        if constexpr (SharkFloatParams::HostVerbose) {
            std::cout << "Shift needed branch A: " << shiftNeeded << std::endl;
        }

        const auto shiftedSz = SharkFloatParams::GlobalNumUint32;
        MultiWordRightShift_LittleEndian(propagatedResult.data(), extDigits, shiftNeeded, OutXY->Digits, shiftedSz);
        outExponent += shiftNeeded;

        if constexpr (SharkFloatParams::HostVerbose) {
            std::cout << "Final propagatedResult after right shift: " <<
                VectorUintToHexString(OutXY->Digits, shiftedSz) <<
                std::endl;
            std::cout << "ShiftNeeded after right shift: " << shiftNeeded << std::endl;
            std::cout << "Final outExponent after right shift: " << outExponent << std::endl;
        }
    } else if (shiftNeeded < 0) {
        if constexpr (SharkFloatParams::HostVerbose) {
            std::cout << "Shift needed branch B: " << shiftNeeded << std::endl;
        }

        const int32_t L = -shiftNeeded;
        const auto shiftedSz = static_cast<int32_t>(SharkFloatParams::GlobalNumUint32);
        MultiWordLeftShift_LittleEndian(
            propagatedResult.data(),
            actualDigits,
            extDigits,
            L,
            OutXY->Digits,
            shiftedSz);
        outExponent -= L;

        if constexpr (SharkFloatParams::HostVerbose) {
            std::cout << "Final propagatedResult after left shift: " <<
                VectorUintToHexString(OutXY->Digits, shiftedSz) <<
                std::endl;
            std::cout << "L after left shift: " << L << std::endl;
            std::cout << "Final outExponent after left shift: " << outExponent << std::endl;
        }
    } else {
        if constexpr (SharkFloatParams::HostVerbose) {
            std::cout << "No shift needed: " << shiftNeeded << std::endl;
        }
        // No shift needed, just copy the result.
        memcpy(OutXY->Digits, propagatedResult.data(), SharkFloatParams::GlobalNumUint32 * sizeof(uint32_t));
    }

    OutXY->Exponent = outExponent;
    // Set the result sign.
    if (sameSign)
        OutXY->IsNegative = A->IsNegative;
    else
        OutXY->IsNegative = AIsBiggerMagnitude ? A->IsNegative : B->IsNegative;

    if constexpr (SharkDebugChecksums) {
        const auto &debugResultState = GetCurrentDebugState<SharkFloatParams, DebugStatePurpose::Result_offsetXY>(
            debugStates, OutXY->Digits, SharkFloatParams::GlobalNumUint32);

        if constexpr (SharkFloatParams::HostVerbose) {
            std::cout << "OutXY->Digits checksum: " << debugResultState.GetStr() << std::endl;
        }
    }
}

//
// Explicit instantiation macro (assumes ExplicitInstantiateAll is defined elsewhere)
//
#define ExplicitlyInstantiate(SharkFloatParams) \
    template void AddHelper<SharkFloatParams>( \
        const HpSharkFloat<SharkFloatParams> *, \
        const HpSharkFloat<SharkFloatParams> *, \
        HpSharkFloat<SharkFloatParams> *, \
        std::vector<DebugStateHost<SharkFloatParams>> &debugStates);

ExplicitInstantiateAll();
