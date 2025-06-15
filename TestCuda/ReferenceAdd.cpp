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

static constexpr auto UseBellochPropagation = false;

// A - B + C
// D + E

//
// Helper functions to perform bit shifts on a fixed-width digit array.
// They mirror the CUDA device functions but work sequentially on the full array.
//

// ShiftRight: Shifts the number (given by its digit array) right by shiftBits.
// idx is the index of the digit to compute. The parameter numDigits prevents out-of-bounds access.
static uint32_t
ShiftRight(
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
static uint32_t
ShiftLeft(
    const uint32_t *digits,
    const int32_t actualDigits,
    const int32_t extDigits,
    const int32_t shiftBits,
    const int32_t idx) {
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
CountLeadingZeros(
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
static void
MultiWordRightShift_LittleEndian(
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
static void
MultiWordLeftShift_LittleEndian(
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
ExtendedNormalizeShiftIndex(
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
static uint32_t
GetNormalizedDigit(
    const uint32_t *ext,
    const int32_t actualDigits,
    const int32_t extDigits,
    const int32_t shiftOffset,
    const int32_t idx) {
    return ShiftLeft(ext, actualDigits, extDigits, shiftOffset, idx);
}

// New helper: Computes the aligned digit for the normalized value on the fly.
// 'diffDE' is the additional right shift required for alignment.
template <class SharkFloatParams>
static uint32_t
GetShiftedNormalizedDigit(
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
GetCorrespondingLimbs(
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
        // For B, we normalize and then shift right by 'diffDE'.
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
        // For A, we normalize and shift right by 'diffDE'.
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
    std::vector<DebugStateHost<SharkFloatParams>> &debugStates,
    const ArrayType *arrayToChecksum,
    size_t arraySize) {

    constexpr auto curPurpose = static_cast<int>(Purpose);
    constexpr auto CallIndex = 0;
    constexpr auto UseConvolution = UseConvolution::No;
    constexpr auto RecursionDepth = 0;

    auto &retval = debugStates[curPurpose];
    retval.Reset(
        arrayToChecksum, arraySize, Purpose, RecursionDepth, CallIndex, UseConvolution);
    return retval;
}

static bool
CompareMagnitudes2Way (
    const int32_t effExpA,
    const int32_t effExpB,
    const int32_t actualDigits,
    const int32_t extDigits,
    const int32_t shiftA,
    const int32_t shiftB,
    const uint32_t *extA,
    const uint32_t *extB)
{
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

// "Strict" ordering of three magnitudes (ignores exact ties - see note below)
enum class ThreeWayMagnitude {
    A_GT_B_GT_C,  // A > B > C
    A_GT_C_GT_B,  // A > C > B
    B_GT_A_GT_C,  // B > A > C
    B_GT_C_GT_A,  // B > C > A
    C_GT_A_GT_B,  // C > A > B
    C_GT_B_GT_A   // C > B > A
};

// Convert ThreeWayMagnitude to string
static std::string
ThreeWayMagnitudeToString (ThreeWayMagnitude cmp) {
    switch (cmp) {
    case ThreeWayMagnitude::A_GT_B_GT_C: return "A > B > C";
    case ThreeWayMagnitude::A_GT_C_GT_B: return "A > C > B";
    case ThreeWayMagnitude::B_GT_A_GT_C: return "B > A > C";
    case ThreeWayMagnitude::B_GT_C_GT_A: return "B > C > A";
    case ThreeWayMagnitude::C_GT_A_GT_B: return "C > A > B";
    case ThreeWayMagnitude::C_GT_B_GT_A: return "C > B > A";
    }
    return "Unknown";
}

static ThreeWayMagnitude
CompareMagnitudes3Way (
    const int32_t effExpA,
    const int32_t effExpB,
    const int32_t effExpC,
    const int32_t actualDigits,
    const int32_t extDigits,
    const int32_t shiftA,
    const int32_t shiftB,
    const int32_t shiftC,
    const uint32_t *extA,
    const uint32_t *extB,
    const uint32_t *extC,
    int32_t &outExp
    )
{
    // Helper: returns true if "first" is strictly bigger than "second"
    auto cmp = [&](const uint32_t *e1, int32_t s1, int32_t exp1,
        const uint32_t *e2, int32_t s2, int32_t exp2) {
            if (exp1 != exp2)
                return exp1 > exp2;
            // exponents equal -> compare normalized digits high->low
            for (int32_t i = extDigits - 1; i >= 0; --i) {
                uint32_t d1 = GetNormalizedDigit(e1, actualDigits, extDigits, s1, i);
                uint32_t d2 = GetNormalizedDigit(e2, actualDigits, extDigits, s2, i);
                if (d1 != d2)
                    return d1 > d2;
            }
            return false;  // treat exact equality as "not greater"
        };

    // 1) Is A the strict max?
    if (cmp(extA, shiftA, effExpA, extB, shiftB, effExpB) &&
        cmp(extA, shiftA, effExpA, extC, shiftC, effExpC)) {
        // now order B vs C
        outExp = effExpA;
        if (cmp(extB, shiftB, effExpB, extC, shiftC, effExpC))
            return ThreeWayMagnitude::A_GT_B_GT_C;
        else
            return ThreeWayMagnitude::A_GT_C_GT_B;
    }
    // 2) Is B the strict max?
    else if (cmp(extB, shiftB, effExpB, extA, shiftA, effExpA) &&
        cmp(extB, shiftB, effExpB, extC, shiftC, effExpC)) {
        // now order A vs C
        outExp = effExpB;
        if (cmp(extA, shiftA, effExpA, extC, shiftC, effExpC))
            return ThreeWayMagnitude::B_GT_A_GT_C;
        else
            return ThreeWayMagnitude::B_GT_C_GT_A;
    }
    // 3) Otherwise C is the (strict) max
    else {
        // order A vs B
        outExp = effExpC;
        if (cmp(extA, shiftA, effExpA, extB, shiftB, effExpB))
            return ThreeWayMagnitude::C_GT_A_GT_B;
        else
            return ThreeWayMagnitude::C_GT_B_GT_A;
    }
}

template<class SharkFloatParams>
static ThreeWayMagnitude
CompareMagnitudes3WayRelativeToBase(
    // the “base” exponent (after normalization + bias)
    const int32_t effExpBase,

    // raw shift to bring each into their MSB position
    const int32_t shiftA,
    const int32_t shiftB,
    const int32_t shiftC,

    // extra right‐shifts to align to effExpBase
    const int32_t diffA,
    const int32_t diffB,
    const int32_t diffC,

    // mantissa arrays (little‐endian, length = extDigits)
    const uint32_t *extA,
    const uint32_t *extB,
    const uint32_t *extC,

    // digit counts
    const int32_t actualDigits,
    const int32_t extDigits,

    // out: the chosen exponent for the result
    int32_t &outExp
) {
    // We'll always report back the base exponent.
    outExp = effExpBase;

    // Helper: lex compare two aligned mantissas (no exponent check)
    auto cmpAligned = [&](
        const uint32_t *e1, int32_t s1, int32_t d1,
        const uint32_t *e2, int32_t s2, int32_t d2
        ) {
            for (int32_t i = extDigits - 1; i >= 0; --i) {
                uint32_t m1 = GetShiftedNormalizedDigit<SharkFloatParams>(
                    e1, actualDigits, extDigits, s1, d1, i);
                uint32_t m2 = GetShiftedNormalizedDigit<SharkFloatParams>(
                    e2, actualDigits, extDigits, s2, d2, i);
                if (m1 != m2)
                    return (m1 > m2);
            }
            return false;  // tie → not greater
        };

    // 1) Is A the strict max?
    if (cmpAligned(extA, shiftA, diffA, extB, shiftB, diffB) &&
        cmpAligned(extA, shiftA, diffA, extC, shiftC, diffC)) {
        // now order B vs C
        if (cmpAligned(extB, shiftB, diffB, extC, shiftC, diffC))
            return ThreeWayMagnitude::A_GT_B_GT_C;
        else
            return ThreeWayMagnitude::A_GT_C_GT_B;
    }
    // 2) Is B the strict max?
    else if (cmpAligned(extB, shiftB, diffB, extA, shiftA, diffA) &&
        cmpAligned(extB, shiftB, diffB, extC, shiftC, diffC)) {
        // now order A vs C
        if (cmpAligned(extA, shiftA, diffA, extC, shiftC, diffC))
            return ThreeWayMagnitude::B_GT_A_GT_C;
        else
            return ThreeWayMagnitude::B_GT_C_GT_A;
    }
    // 3) Otherwise C is the strict max
    else {
        // order A vs B
        if (cmpAligned(extA, shiftA, diffA, extB, shiftB, diffB))
            return ThreeWayMagnitude::C_GT_A_GT_B;
        else
            return ThreeWayMagnitude::C_GT_B_GT_A;
    }
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

// Unified CarryPropagationPP_DE.
// If sameSign is true, this is the addition branch;
// if false, it is the subtraction branch (we guarantee that the final result is positive).
template<class SharkFloatParams>
void CarryPropagationPP_DE (
    const bool sameSign,
    const int32_t extDigits,
    const std::vector<uint64_t> extResultVector,
    uint32_t &carry,
    std::vector<uint32_t> &propagatedResultVector) {
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
        std::cout << "CarryPropagationPP_DE Working array:" << std::endl;
        for (int i = 0; i < extDigits; i++) {
            std::cout << "  " << i << ": g = " << working[i].g << ", p = " << working[i].p << std::endl;
        }
    }

    // Step 2. Perform an inclusive (upsweep) scan on the per-digit signals.
    // The inclusive array at index i contains the combined operator for sigma[0..i].
    assert(working.size() == extDigits);
    std::vector<GenProp> scratch(extDigits); // one scratch array of size numActualDigitsPlusGuard

    // Use raw pointers that point to the current input and output buffers.
    GenProp *in = working.data();
    GenProp *out = scratch.data();

    // Perform the upsweep (inclusive scan) in log2(numActualDigitsPlusGuard) passes.
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

    // Now "in" points to the final inclusive scan result for indices 0 .. numActualDigitsPlusGuard-1.
    const GenProp *inclusive = in;

    if constexpr (SharkFloatParams::HostVerbose) {
        std::cout << "CarryPropagationPP_DE Inclusive array:" << std::endl;
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
        std::cout << "CarryPropagationPP_DE Carries array:" << std::endl;
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
        std::cout << "CarryPropagationPP_DE Propagated result:" << std::endl;
        for (int i = 0; i < extDigits; i++) {
            std::cout << "  " << i << ": " << propagatedResult[i] << std::endl;
        }
    }

    // The final element (at position numActualDigitsPlusGuard in the carries array)
    // is the overall carry (or borrow). For subtraction we expect this to be 0.
    carry = carries[extDigits];

    if constexpr (SharkFloatParams::HostVerbose) {
        std::cout << "CarryPropagationPP_DE Final carry: " << carry << std::endl;
    }
}

template<class SharkFloatParams>
void CarryPropagation_DE (
    const bool sameSign,
    const int32_t extDigits,
    std::vector<uint64_t> &extResult,
    uint32_t &carry,
    std::vector<uint32_t> &propagatedResult) {
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



template<class SharkFloatParams>
void Phase1_DE (
    const bool DIsBiggerMagnitude,
    const bool IsNegativeD,
    const bool IsNegativeE,
    const int32_t extDigits,
    const int32_t actualDigits,
    const auto *ext_D_2X,
    const auto *ext_E_B,
    const int32_t shiftD,
    const int32_t shiftE,
    const int32_t effExpD,
    const int32_t effExpE,
    const int32_t newDExponent,
    const int32_t newEExponent,
    int32_t &outExponent_DE,
    std::vector<uint64_t> &extResult_D_E,
    std::vector<DebugStateHost<SharkFloatParams>> &debugStates)
{
    const bool sameSignDE = (IsNegativeD == IsNegativeE);
    const int32_t diffDE = DIsBiggerMagnitude ? (effExpD - effExpE) : (effExpE - effExpD);
    outExponent_DE = DIsBiggerMagnitude ? newDExponent : newEExponent;

    if constexpr (SharkFloatParams::HostVerbose) {
        std::cout << "diffDE: " << diffDE << std::endl;
        std::cout << "outExponent_DE: " << outExponent_DE << std::endl;
    }

    // --- Phase 1: Raw Extended Arithmetic ---
    // Compute the raw limb-wise result without propagation.
    if (sameSignDE) {
        // Addition branch.
        for (int32_t i = 0; i < extDigits; i++) {
            uint64_t alignedA = 0, alignedB = 0;
            GetCorrespondingLimbs<SharkFloatParams>(
                ext_D_2X, actualDigits, extDigits,
                ext_E_B, actualDigits, extDigits,
                shiftD, shiftE,
                DIsBiggerMagnitude, diffDE, i,
                alignedA, alignedB);
            extResult_D_E[i] = alignedA + alignedB;
        }
    } else {
        // Subtraction branch.
        std::vector<uint64_t> alignedDDebug;
        std::vector<uint64_t> alignedEDebug;

        if (DIsBiggerMagnitude) {
            for (int32_t i = 0; i < extDigits; i++) {
                uint64_t alignedA = 0, alignedB = 0;
                GetCorrespondingLimbs<SharkFloatParams>(
                    ext_D_2X, actualDigits, extDigits,
                    ext_E_B, actualDigits, extDigits,
                    shiftD, shiftE,
                    DIsBiggerMagnitude, diffDE, i,
                    alignedA, alignedB);
                // Compute raw difference (which may be negative).
                const int64_t rawDiff = (int64_t)alignedA - (int64_t)alignedB;
                extResult_D_E[i] = (uint64_t)rawDiff;

                alignedDDebug.push_back(alignedA);
                alignedEDebug.push_back(alignedB);
            }
        } else {
            for (int32_t i = 0; i < extDigits; i++) {
                uint64_t alignedA = 0, alignedB = 0;
                GetCorrespondingLimbs<SharkFloatParams>(
                    ext_D_2X, actualDigits, extDigits,
                    ext_E_B, actualDigits, extDigits,
                    shiftD, shiftE,
                    DIsBiggerMagnitude, diffDE, i,
                    alignedA, alignedB);
                const int64_t rawDiff = (int64_t)alignedB - (int64_t)alignedA;
                extResult_D_E[i] = (uint64_t)rawDiff;

                alignedDDebug.push_back(alignedA);
                alignedEDebug.push_back(alignedB);
            }
        }

        if constexpr (SharkFloatParams::HostVerbose) {
            std::cout << "Phase1_DE - These are effectively the arrays we're adding and subtracting:" << std::endl;
            std::cout << "alignedDDebug: " << VectorUintToHexString(alignedDDebug) << std::endl;
            std::cout << "alignedEDebug: " << VectorUintToHexString(alignedEDebug) << std::endl;
        }
    }

    if constexpr (SharkDebugChecksums) {
        const auto &debugResultState = GetCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z2XY>(
            debugStates, extResult_D_E.data(), extDigits);

        if constexpr (SharkFloatParams::HostVerbose) {
            std::cout << "extResult_D_E checksum: " << debugResultState.GetStr() << std::endl;
            std::cout << "extResult_D_E after arithmetic: " << VectorUintToHexString(extResult_D_E) << std::endl;
        }
    }
}


template<class SharkFloatParams>
void CarryPropagation_ABC (
    const int32_t            extDigits,
    std::vector<uint64_t> &extResult,         // raw signed limbs from Phase1_ABC
    int32_t &carryAcc,         // signed carry-in/out (init to 0)
    std::vector<uint32_t> &propagatedResult   // size numActualDigitsPlusGuard
) {
    // Start with zero carry/borrow
    carryAcc = 0;

    for (int32_t i = 0; i < extDigits; ++i) {
        // reinterpret the 64-bit limb as signed
        int64_t limb = static_cast<int64_t>(extResult[i]);

        // add in the previous carry (or borrow, if negative)
        int64_t sum = limb + carryAcc;

        // low 32 bits become the output digit
        uint32_t low32 = static_cast<uint32_t>(sum & 0xFFFFFFFFULL);
        propagatedResult[i] = low32;

        // compute next carryAcc = floor(sum/2^32)
        // (sum - low32) is a multiple of 2^32, so this division is exact
        carryAcc = (sum - static_cast<int64_t>(low32)) >> 32;
        // -or equivalently-
        // carryAcc = (sum - static_cast<int64_t>(low32)) / (1LL << 32);
    }

    // On exit, carryAcc may be positive (overflow) or negative (net borrow).
    // You can inspect it to adjust exponent / final sign:
    if constexpr (SharkFloatParams::HostVerbose) {
        assert(carryAcc >= 0);
        std::cout << "CarryPropagation3 final carryAcc = " << carryAcc << std::endl;
    }
}

template<class SharkFloatParams>
static void
ComputeABCComparison (
    // normalized, extended digit arrays (little‐endian; index 0 = LSB)
    //   extA, extB, extC each have length = extDigits (actualDigits + guardWords)
    const uint32_t *extA,
    const uint32_t *extB,
    const uint32_t *extC,

    // sizes
    const int32_t actualDigits,  // number of real digits in A, B, C
    const int32_t extDigits,     // = actualDigits + #guardWords

    // normalization shifts (how many bits left each was shifted to bring MSB to top)
    const int32_t shiftA,
    const int32_t shiftB,
    const int32_t shiftC,

    // effective exponents of A, B, C after normalization + bias
    const int32_t effExpA,
    const int32_t effExpB,
    const int32_t effExpC,

    // three‐way ordering (precomputed by CompareMagnitudes3Way in Phase1_ABC)
    const ThreeWayMagnitude ordering,
    const int32_t              biasedExpABC,

    // input signs (for A–B, B–C, etc.); in Phase1_ABC the caller already flipped
    // signB if you are doing A–B+C, but here we assume signA, signB, signC are
    // exactly “true if negative” for each operand in the three‐way.
    const bool signA,
    const bool signB,
    const bool signC,

    // outputs (same as before):
    bool &ABIsBiggerThanC,  // “is |(±A) – (±B)| > |C| ?”
    bool &ACIsBiggerThanB,  // “is |(±A) – (±C)| > |B| ?”
    bool &BCIsBiggerThanA   // “is |(±B) – (±C)| > |A| ?”
)
{
    // 1) Compute how far each must be right‐shifted to line up with biasedExpABC:
    int32_t diffA = biasedExpABC - effExpA;
    int32_t diffB = biasedExpABC - effExpB;
    int32_t diffC = biasedExpABC - effExpC;

    // just after you compute `ordering`:
    // compare extBase vs extOther after both have been normalized & shifted to the same exponent frame
    auto CompareMagnitudes2WayRelativeToBase = [&](
        const uint32_t *extBase, int32_t shiftBase, int32_t diffBase,
        const uint32_t *extOther, int32_t shiftOther, int32_t diffOther
        ) {
            // lex‐compare high→low
            for (int32_t i = extDigits - 1; i >= 0; --i) {
                uint32_t mB = GetShiftedNormalizedDigit<SharkFloatParams>(
                    extBase, actualDigits, extDigits, shiftBase, diffBase, i);
                uint32_t mO = GetShiftedNormalizedDigit<SharkFloatParams>(
                    extOther, actualDigits, extDigits, shiftOther, diffOther, i);
                if (mB > mO) return true;
                if (mB < mO) return false;
            }
            // treat exact equality as “greater or equal”
            return true;
        };

    // now compute each X≥Y flag with one call apiece:
    const bool AB_XgeY = CompareMagnitudes2WayRelativeToBase(
        extA, shiftA, diffA,
        extB, shiftB, diffB
    );
    const bool AC_XgeY = CompareMagnitudes2WayRelativeToBase(
        extA, shiftA, diffA,
        extC, shiftC, diffC
    );
    const bool BC_XgeY = CompareMagnitudes2WayRelativeToBase(
        extB, shiftB, diffB,
        extC, shiftC, diffC
    );

    auto CmpAlignedPairVsThird = [&](
        const uint32_t *extX,
        const uint32_t *extY,
        const uint32_t *extZ,
        int32_t shiftX,
        int32_t diffX,
        int32_t shiftY,
        int32_t diffY,
        int32_t shiftZ,
        int32_t diffZ,
        bool    sX,
        bool    sY,
        bool    XgeY,
        const int32_t actualDigits,
        const int32_t extDigits,
        bool &outXYgtZ
        )
    {
        // --- Phase A: single‐limb early‐exit test at i = extDigits−1 ---
        {
            int32_t i = extDigits - 1;

            // (a) Fetch top 64-bit limb of X and Y
            uint64_t a = GetShiftedNormalizedDigit<SharkFloatParams>(
                extX, actualDigits, extDigits, shiftX, diffX, i);
            uint64_t b = GetShiftedNormalizedDigit<SharkFloatParams>(
                extY, actualDigits, extDigits, shiftY, diffY, i);

            // (b) Compute signed raw and absolute value
            int64_t raw64 = (sX == sY
                ? int64_t(a) + int64_t(b)
                : (a >= b ? int64_t(a) - int64_t(b)
                    : int64_t(b) - int64_t(a)));
            uint64_t mag64 = raw64 < 0 ? uint64_t(-raw64) : uint64_t(raw64);

            // (c) Split into low-word and carry-out
            uint32_t carry = uint32_t(mag64 >> 32);               // overflow bit
            uint32_t D_low = uint32_t(mag64 & 0xFFFFFFFFULL);     // low 32 bits

            // (d) Fetch top aligned Z word
            uint32_t dZ = uint32_t(
                GetShiftedNormalizedDigit<SharkFloatParams>(
                    extZ, actualDigits, extDigits, shiftZ, diffZ, i));

            // (e) Early exits by simple inequalities:
            //     - any carry → |X±Y| has a higher bit
            //     - D_low > dZ+1 → even a borrow of 1 can’t drop it below Z
            if (carry != 0U || D_low > dZ + 1U) {
                outXYgtZ = true;
                return;
            }
            if (D_low < dZ) {
                outXYgtZ = false;
                return;
            }

            // else we’re in the narrow window (D_low == dZ or dZ+1):
            // fall through into the borrow‐aware Phase B loop below
        }


        //
        // --- Phase B: exponents tied → lexicographic compare of 32-bit words ---
        //
        auto computeBorrowIn = [&](int32_t i) -> uint32_t {
            // scan all lower limbs j = i–1 … 0
            for (int32_t j = i - 1; j >= 0; --j) {
                uint64_t a_j = GetShiftedNormalizedDigit<SharkFloatParams>(
                    extX, actualDigits, extDigits, shiftX, diffX, j);
                uint64_t b_j = GetShiftedNormalizedDigit<SharkFloatParams>(
                    extY, actualDigits, extDigits, shiftY, diffY, j);

                // exactly the same signed raw you did in Phase B:
                int64_t raw_j = (sX != sY)
                    ? (XgeY ? int64_t(a_j) - int64_t(b_j)
                        : int64_t(b_j) - int64_t(a_j))
                    : int64_t(a_j) + int64_t(b_j);

                if (raw_j < 0) {
                    // borrow was generated at j
                    return 1U;
                }
                if (raw_j > 0) {
                    // no borrow could pass upward
                    return 0U;
                }
                // raw_j == 0 → keep scanning (propagate)
            }
            // if we get here, everything below was zero → no borrow
            return 0U;
            };


        // assume before this loop you computed:
        bool doSubtract = (sX != sY);

        // word-by-word compare
        for (int32_t i = extDigits - 1; i >= 0; --i) {
            uint64_t a = GetShiftedNormalizedDigit<SharkFloatParams>(
                extX, actualDigits, extDigits, shiftX, diffX, i);
            uint64_t b = GetShiftedNormalizedDigit<SharkFloatParams>(
                extY, actualDigits, extDigits, shiftY, diffY, i);

            uint32_t D_low;
            uint32_t carry_or_borrow = 0;

            if (!doSubtract) {
                // addition branch
                uint64_t sum = a + b;
                D_low = uint32_t(sum);
                carry_or_borrow = uint32_t(sum >> 32);  // any overflow → carry
            } else {
                // subtraction branch: always X - Y when XgeY, or Y - X otherwise
                if (XgeY) {
                    uint64_t diff = a - b;
                    D_low = uint32_t(diff);
                    // no immediate borrow here (we'll detect cross-digit borrows in computeBorrowIn)
                } else {
                    uint64_t diff = b - a;
                    D_low = uint32_t(diff);
                    // likewise no per-digit borrow
                }
                // we deliberately leave carry_or_borrow == 0,
                // because any actual borrow will be found by computeBorrowIn
            }

            uint32_t dZ = uint32_t(
                GetShiftedNormalizedDigit<SharkFloatParams>(
                    extZ, actualDigits, extDigits, shiftZ, diffZ, i));

            // fast-exit on addition‐overflow or clear non‐borrow
            if (!doSubtract) {
                if (carry_or_borrow != 0U) {
                    outXYgtZ = true;
                    return;
                }
            }

            // fast-exit on magnitude compare without borrow
            if (D_low < dZ) {
                outXYgtZ = false;
                return;
            }
            if (D_low > dZ + 1U) {
                outXYgtZ = true;
                return;
            }

            // slow‐path: inject borrow from lower limbs
            uint32_t borrow = computeBorrowIn(i);
            uint32_t D_prop = D_low - borrow;

            if (D_prop < dZ) {
                outXYgtZ = false;
                return;
            }
            if (D_prop > dZ) {
                outXYgtZ = true;
                return;
            }
            // else tie → continue
        }

        // exact tie
        outXYgtZ = false;
    };

    // Now call that helper three times, each time aligning all three mantissas to biasedExpABC:
    //  i)  “Is |(±A) – (±B)| > |C| ?”
    CmpAlignedPairVsThird(
        /* extX      */ extA, 
        /* extY      */ extB,
        /* extZ      */ extC,
        /* shiftX    */ shiftA,
        /* diffX     */ diffA,
        /* shiftY    */ shiftB,
        /* diffY     */ diffB,
        /* shiftZ    */ shiftC,
        /* diffZ     */ diffC,
        /* sX        */ signA,
        /* sY        */ signB,
        AB_XgeY,
        /* actualDig */ actualDigits,
        /* extDig    */ extDigits,
        /* out       */ ABIsBiggerThanC
    );

    //  ii) “Is |(±A) – (±C)| > |B| ?”
    CmpAlignedPairVsThird(
        /* extX      */ extA,
        /* extY      */ extC,
        /* extZ      */ extB,
        /* shiftX    */ shiftA,
        /* diffX     */ diffA,
        /* shiftY    */ shiftC,
        /* diffY     */ diffC,
        /* shiftZ    */ shiftB,
        /* diffZ     */ diffB,
        /* sX        */ signA,
        /* sY        */ signC,
        AC_XgeY,
        /* actualDig */ actualDigits,
        /* extDig    */ extDigits,
        /* out       */ ACIsBiggerThanB
    );

    // iii) “Is |(±B) – (±C)| > |A| ?”
    CmpAlignedPairVsThird(
        /* extX      */ extB,
        /* extY      */ extC,
        /* extZ      */ extA,
        /* shiftX    */ shiftB,
        /* diffX     */ diffB,
        /* shiftY    */ shiftC,
        /* diffY     */ diffC,
        /* shiftZ    */ shiftA,
        /* diffZ     */ diffA,
        /* sX        */ signB,
        /* sY        */ signC,
        BC_XgeY,
        /* actualDig */ actualDigits,
        /* extDig    */ extDigits,
        /* out       */ BCIsBiggerThanA
    );
}



template<class SharkFloatParams>
void Phase1_ABC (
    const ThreeWayMagnitude ordering,
    const bool IsNegativeA,
    const bool IsNegativeB,
    const bool IsNegativeC,
    const int32_t  extDigits,
    const int32_t  actualDigits,
    const uint32_t *extA,
    const uint32_t *extB,
    const uint32_t *extC,
    const int32_t  shiftA,
    const int32_t  shiftB,
    const int32_t  shiftC,
    const int32_t  effExpA,
    const int32_t  effExpB,
    const int32_t  effExpC,
    const int32_t  biasedExpABC_local,
    const int32_t  bias,
    bool &IsNegativeABC,
    int32_t &outExponent_ABC,
    std::vector<uint64_t> &extResult_ABC,
    std::vector<DebugStateHost<SharkFloatParams>> &debugStates
) {
    // Final exponent before bias correction
    outExponent_ABC = biasedExpABC_local - bias;
    extResult_ABC.assign(extDigits, 0ull);

    bool ABIsBiggerThanC, ACIsBiggerThanB, BCIsBiggerThanA;
    ComputeABCComparison<SharkFloatParams>(
        extA, extB, extC,
        actualDigits, extDigits,
        shiftA, shiftB, shiftC,
        effExpA, effExpB, effExpC,
        ordering, biasedExpABC_local,
        IsNegativeA, IsNegativeB, IsNegativeC,
        ABIsBiggerThanC,
        ACIsBiggerThanB,
        BCIsBiggerThanA);

    // How far each input must be shifted right to align at biasedExpABC_local
    int32_t diffA = biasedExpABC_local - effExpA;
    int32_t diffB = biasedExpABC_local - effExpB;
    int32_t diffC = biasedExpABC_local - effExpC;

    // Debug buffers
    std::vector<uint64_t> alignedXDebug(extDigits), alignedYDebug(extDigits), alignedZDebug(extDigits);

    // --- Fused loop: subtract middle from largest, then add smallest ---
    uint64_t X = 0;
    uint64_t Y = 0;
    uint64_t Z = 0;

    bool signX = false;
    bool signY = false;
    bool signZ = false;

    switch (ordering) {
    case ThreeWayMagnitude::A_GT_B_GT_C:
        signX = IsNegativeA;
        signY = IsNegativeB;
        signZ = IsNegativeC;
        break;
    case ThreeWayMagnitude::A_GT_C_GT_B:
        signX = IsNegativeA;
        signY = IsNegativeC;
        signZ = IsNegativeB;
        break;
    case ThreeWayMagnitude::B_GT_A_GT_C:
        signX = IsNegativeB;
        signY = IsNegativeA;
        signZ = IsNegativeC;
        break;
    case ThreeWayMagnitude::B_GT_C_GT_A:
        signX = IsNegativeB;
        signY = IsNegativeC;
        signZ = IsNegativeA;
        break;
    case ThreeWayMagnitude::C_GT_A_GT_B:
        signX = IsNegativeC;
        signY = IsNegativeA;
        signZ = IsNegativeB;
        break;
    case ThreeWayMagnitude::C_GT_B_GT_A:
        signX = IsNegativeC;
        signY = IsNegativeB;
        signZ = IsNegativeA;
        break;
    default:
        assert(false);
    }

    bool XYgtZ = false;
    switch (ordering) {
    case ThreeWayMagnitude::A_GT_B_GT_C:
    case ThreeWayMagnitude::B_GT_A_GT_C:
        XYgtZ = ABIsBiggerThanC;    break;
    case ThreeWayMagnitude::A_GT_C_GT_B:
    case ThreeWayMagnitude::C_GT_A_GT_B:
        XYgtZ = ACIsBiggerThanB;    break;
    case ThreeWayMagnitude::B_GT_C_GT_A:
    case ThreeWayMagnitude::C_GT_B_GT_A:
        XYgtZ = BCIsBiggerThanA;    break;
    }

    std::string arrayXStr, arrayYStr, arrayZStr;
    switch (ordering) {
    case ThreeWayMagnitude::A_GT_B_GT_C:
        arrayXStr = "A"; arrayYStr = "B"; arrayZStr = "C"; break;
    case ThreeWayMagnitude::A_GT_C_GT_B:
        arrayXStr = "A"; arrayYStr = "C"; arrayZStr = "B"; break;
    case ThreeWayMagnitude::B_GT_A_GT_C:
        arrayXStr = "B"; arrayYStr = "A"; arrayZStr = "C"; break;
    case ThreeWayMagnitude::B_GT_C_GT_A:
        arrayXStr = "B"; arrayYStr = "C"; arrayZStr = "A"; break;
    case ThreeWayMagnitude::C_GT_A_GT_B:
        arrayXStr = "C"; arrayYStr = "A"; arrayZStr = "B"; break;
    case ThreeWayMagnitude::C_GT_B_GT_A:
        arrayXStr = "C"; arrayYStr = "B"; arrayZStr = "A"; break;
    default:
        assert(false);
    }

    // before the loop: we'll stash the final sign here
    IsNegativeABC = false;
    for (int32_t i = 0; i < extDigits; ++i) {
        // Pick (X, Y, Z) = (largest, middle, smallest) per 'ordering'
        switch (ordering) {
        case ThreeWayMagnitude::A_GT_B_GT_C:
            X = GetNormalizedDigit(extA, actualDigits, extDigits, shiftA, i);
            Y = GetShiftedNormalizedDigit<SharkFloatParams>(extB, actualDigits, extDigits, shiftB, diffB, i);
            Z = GetShiftedNormalizedDigit<SharkFloatParams>(extC, actualDigits, extDigits, shiftC, diffC, i);
            break;
        case ThreeWayMagnitude::A_GT_C_GT_B:
            X = GetNormalizedDigit(extA, actualDigits, extDigits, shiftA, i);
            Y = GetShiftedNormalizedDigit<SharkFloatParams>(extC, actualDigits, extDigits, shiftC, diffC, i);
            Z = GetShiftedNormalizedDigit<SharkFloatParams>(extB, actualDigits, extDigits, shiftB, diffB, i);
            break;
        case ThreeWayMagnitude::B_GT_A_GT_C:
            X = GetNormalizedDigit(extB, actualDigits, extDigits, shiftB, i);
            Y = GetShiftedNormalizedDigit<SharkFloatParams>(extA, actualDigits, extDigits, shiftA, diffA, i);
            Z = GetShiftedNormalizedDigit<SharkFloatParams>(extC, actualDigits, extDigits, shiftC, diffC, i);
            break;
        case ThreeWayMagnitude::B_GT_C_GT_A:
            X = GetNormalizedDigit(extB, actualDigits, extDigits, shiftB, i);
            Y = GetShiftedNormalizedDigit<SharkFloatParams>(extC, actualDigits, extDigits, shiftC, diffC, i);
            Z = GetShiftedNormalizedDigit<SharkFloatParams>(extA, actualDigits, extDigits, shiftA, diffA, i);
            break;
        case ThreeWayMagnitude::C_GT_A_GT_B:
            X = GetNormalizedDigit(extC, actualDigits, extDigits, shiftC, i);
            Y = GetShiftedNormalizedDigit<SharkFloatParams>(extA, actualDigits, extDigits, shiftA, diffA, i);
            Z = GetShiftedNormalizedDigit<SharkFloatParams>(extB, actualDigits, extDigits, shiftB, diffB, i);
            break;
        case ThreeWayMagnitude::C_GT_B_GT_A:
            X = GetNormalizedDigit(extC, actualDigits, extDigits, shiftC, i);
            Y = GetShiftedNormalizedDigit<SharkFloatParams>(extB, actualDigits, extDigits, shiftB, diffB, i);
            Z = GetShiftedNormalizedDigit<SharkFloatParams>(extA, actualDigits, extDigits, shiftA, diffA, i);
            break;
        default:
            assert(false);
        }

        // 2) always “larger - smaller” when signs differ, otherwise add
        uint64_t magXY = (signX == signY) ? (X + Y)
            : (X - Y);
        bool   signXY = signX;  // if we subtracted, X was the larger so X’s sign wins

        uint64_t magABC;
        if (signXY == signZ) {
            // same sign → addition
            magABC = magXY + Z;
            IsNegativeABC = signXY;
        } else if (XYgtZ) {
            // |X±Y| ≥ |Z| → subtraction in that order
            magABC = magXY - Z;
            IsNegativeABC = signXY;
        } else {
            // |Z| > |X±Y| → subtraction the other way
            magABC = Z - magXY;
            IsNegativeABC = signZ;
        }

        // 3) store the always-nonnegative magnitude
        extResult_ABC[i] = magABC;

        // (optional) debug:
        alignedXDebug[i] = X;
        alignedYDebug[i] = Y;
        alignedZDebug[i] = Z;
    }

    // — now that extResult_ABC is fully populated, find its MS‐non‐zero limb —
    int32_t msd = -1;
    for (int32_t i = extDigits - 1; i >= 0; --i) {
        if (extResult_ABC[i] != 0) {
            msd = i;
            break;
        }
    }

    // Debug printing
    if constexpr (SharkFloatParams::HostVerbose) {
        std::cout << "Phase1_ABC - These are effectively the arrays we're adding and subtracting:\n";
        std::cout << "alignedXDebug (" << arrayXStr << "): " << (signX ? "-" : "+") << VectorUintToHexString(alignedXDebug) << "\n";
        std::cout << "alignedYDebug (" << arrayYStr << "): " << (signY ? "-" : "+") << VectorUintToHexString(alignedYDebug) << "\n";
        std::cout << "alignedZDebug (" << arrayZStr << "): " << (signZ ? "-" : "+") << VectorUintToHexString(alignedZDebug) << "\n";
        std::cout << "extResult_ABC: " << VectorUintToHexString(extResult_ABC) << "\n";
        std::cout << "Phase1_ABC - Final sign: " << (IsNegativeABC ? "-" : "+") << "\n";
        std::cout << "Phase1_ABC msd: " << msd << "\n";
    }

    if constexpr (SharkDebugChecksums) {
        const auto &debugResultState = GetCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z2XX>(
            debugStates, extResult_ABC.data(), extDigits);
        if constexpr (SharkFloatParams::HostVerbose) {
            std::cout << "Phase1_ABC checksum: " << debugResultState.GetStr() << "\n";
        }
    }

    // if it wasn’t all zero, we may have a carry out but no borrow.
    //if (msd >= 0) {
    //    uint64_t top = extResult_ABC[msd];
    //    assert((top & 0xFFFF'FFF0'0000'0000ULL) == 0ULL);
    //}
}

//
// Ternary operator that calculates:
// OutXY1 = A_X2 - B_Y2 + C_A
// ResultOut = D_2X + E_B
//
// Extended arithmetic using little-endian representation.
// This version uses the new normalization approach, where the extended operands
// are not copied; instead, a shift index is returned and used later to compute
// normalized digits on the fly.
//
template<class SharkFloatParams>
void
AddHelper (
    const HpSharkFloat<SharkFloatParams> *A_X2,
    const HpSharkFloat<SharkFloatParams> *B_Y2,
    const HpSharkFloat<SharkFloatParams> *C_A,
    const HpSharkFloat<SharkFloatParams> *D_2X,
    const HpSharkFloat<SharkFloatParams> *E_B,
    HpSharkFloat<SharkFloatParams> *OutXY1,
    HpSharkFloat<SharkFloatParams> *OutXY2,
    std::vector<DebugStateHost<SharkFloatParams>> &debugStates
) {
    if constexpr (SharkDebugChecksums) {
        constexpr auto NewDebugStateSize = static_cast<int>(DebugStatePurpose::NumPurposes);
        debugStates.resize(NewDebugStateSize);
    }

    // Make local copies.
    const auto *ext_A_X2 = A_X2->Digits;
    const auto *ext_B_Y2 = B_Y2->Digits;
    const auto *ext_C_A = C_A->Digits;
    const auto *ext_D_2X = D_2X->Digits;
    const auto *ext_E_B = E_B->Digits;

    const bool IsNegativeA = A_X2->IsNegative;
    const bool IsNegativeB = !B_Y2->IsNegative; // A - B + C
    const bool IsNegativeC = C_A->IsNegative;
    const bool IsNegativeD = D_2X->IsNegative;
    const bool IsNegativeE = E_B->IsNegative;

    // --- Set up extended working precision ---
    constexpr int32_t guard = 4;
    constexpr int32_t numActualDigits = SharkFloatParams::GlobalNumUint32;
    constexpr int32_t numActualDigitsPlusGuard = SharkFloatParams::GlobalNumUint32 + guard;
    // Create extended arrays (little-endian, index 0 is LSB).
    std::vector<uint64_t> extResult_ABC(numActualDigitsPlusGuard, 0);
    std::vector<uint64_t> extResult_D_E(numActualDigitsPlusGuard, 0);

    // The guard words (indices GlobalNumUint32 to numActualDigitsPlusGuard-1) are left as zero.

    if constexpr (SharkFloatParams::HostVerbose) {
        std::cout << "ext_A_X2: " << VectorUintToHexString(ext_A_X2, numActualDigits) << std::endl;
        std::cout << "ext_A_X2 exponent: " << A_X2->Exponent << std::endl;
        std::cout << "ext_A_X2 sign: " << (IsNegativeA ? "-" : "+") << std::endl;

        std::cout << "ext_B_Y2: " << VectorUintToHexString(ext_B_Y2, numActualDigits) << std::endl;
        std::cout << "ext_B_Y2 exponent: " << B_Y2->Exponent << std::endl;
        std::cout << "ext_B_Y2 sign: " << (IsNegativeB ? "-" : "+") << std::endl;

        std::cout << "ext_C_A: " << VectorUintToHexString(ext_C_A, numActualDigits) << std::endl;
        std::cout << "ext_C_A exponent: " << C_A->Exponent << std::endl;
        std::cout << "ext_C_A sign: " << (IsNegativeC ? "-" : "+") << std::endl;

        std::cout << "ext_D_2X: " << VectorUintToHexString(ext_D_2X, numActualDigits) << std::endl;
        std::cout << "ext_D_2X exponent: " << D_2X->Exponent << std::endl;
        std::cout << "ext_D_2X sign: " << (IsNegativeD ? "-" : "+") << std::endl;

        std::cout << "ext_E_B: " << VectorUintToHexString(ext_E_B, numActualDigits) << std::endl;
        std::cout << "ext_E_B exponent: " << E_B->Exponent << std::endl;
        std::cout << "ext_E_B sign: " << (IsNegativeE ? "-" : "+") << std::endl;
    }

    if constexpr (SharkDebugChecksums) {
        // Compute checksums for the extended arrays.
        // Note: we use the actual digits (not the extended size) for the checksum.

        const auto &debugState = GetCurrentDebugState<SharkFloatParams, DebugStatePurpose::ADigits>(
            debugStates, ext_A_X2, numActualDigits);

        const auto &debugBState = GetCurrentDebugState<SharkFloatParams, DebugStatePurpose::BDigits>(
            debugStates, ext_B_Y2, numActualDigits);

        const auto &debugCState = GetCurrentDebugState<SharkFloatParams, DebugStatePurpose::CDigits>(
            debugStates, ext_C_A, numActualDigits);

        const auto &debugDState = GetCurrentDebugState<SharkFloatParams, DebugStatePurpose::DDigits>(
            debugStates, ext_D_2X, numActualDigits);

        const auto &debugEState = GetCurrentDebugState<SharkFloatParams, DebugStatePurpose::EDigits>(
            debugStates, ext_E_B, numActualDigits);

        if constexpr (SharkFloatParams::HostVerbose) {
            std::cout << "A_X2->Digits checksum: " << debugState.GetStr() << std::endl;
            std::cout << "B_Y2->Digits checksum: " << debugBState.GetStr() << std::endl;
            std::cout << "C_A->Digits checksum: " << debugCState.GetStr() << std::endl;
            std::cout << "D_2X->Digits checksum: " << debugDState.GetStr() << std::endl;
            std::cout << "E_B->Digits checksum: " << debugEState.GetStr() << std::endl;
        }
    }

    // --- Extended Normalization using shift indices ---
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

    // Normalize the extended operands.
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

    if constexpr (SharkFloatParams::HostVerbose) {
        std::cout << "shiftALeftToGetMsb: " << shiftALeftToGetMsb << std::endl;
        std::cout << "shiftBLeftToGetMsb: " << shiftBLeftToGetMsb << std::endl;
        std::cout << "shiftCLeftToGetMsb: " << shiftCLeftToGetMsb << std::endl;
        std::cout << "shiftDLeftToGetMsb: " << shiftDLeftToGetMsb << std::endl;
        std::cout << "shiftELeftToGetMsb: " << shiftELeftToGetMsb << std::endl;
    }

    // --- Compute Effective Exponents ---
    const auto bias = (SharkFloatParams::GlobalNumUint32 * 32 - 32);
    const int32_t effExpA = normA_isZero ? -100'000'000 : newAExponent + bias;
    const int32_t effExpB = normB_isZero ? -100'000'000 : newBExponent + bias;
    const int32_t effExpC = normC_isZero ? -100'000'000 : newCExponent + bias;
    const int32_t effExpD = normD_isZero ? -100'000'000 : newDExponent + bias;
    const int32_t effExpE = normE_isZero ? -100'000'000 : newEExponent + bias;

    if constexpr (SharkFloatParams::HostVerbose) {
        std::cout << "effExpA: " << effExpA << std::endl;
        std::cout << "effExpB: " << effExpB << std::endl;
        std::cout << "effExpC: " << effExpC << std::endl;
        std::cout << "effExpD: " << effExpD << std::endl;
        std::cout << "effExpE: " << effExpE << std::endl;

        // Print the each array normalized according to
        // their resepective effective exponents

        auto PrintOneNormalizedArray = [&](const uint32_t *ext, int32_t shift, const std::string &name) {
            std::cout << name << " normalized: ";
            std::vector<uint32_t> normalizedDigits(numActualDigitsPlusGuard, 0);
            for (int32_t i = 0; i < numActualDigitsPlusGuard; ++i) {
                uint32_t digit = GetNormalizedDigit(ext, numActualDigits, numActualDigitsPlusGuard, shift, i);
                normalizedDigits[i] = digit;
            }

            std::cout << VectorUintToHexString(normalizedDigits.data(), numActualDigitsPlusGuard) << std::endl;
            };

        PrintOneNormalizedArray(ext_A_X2, shiftALeftToGetMsb, "ext_A_X2");
        PrintOneNormalizedArray(ext_B_Y2, shiftBLeftToGetMsb, "ext_B_Y2");
        PrintOneNormalizedArray(ext_C_A, shiftCLeftToGetMsb, "ext_C_A");
        PrintOneNormalizedArray(ext_D_2X, shiftDLeftToGetMsb, "ext_D_2X");
        PrintOneNormalizedArray(ext_E_B, shiftELeftToGetMsb, "ext_E_B");
    }

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

    {
        auto PrintOneShiftedNormalizedArray = [&](const uint32_t *ext, int32_t shift, int32_t diff, const std::string &name) {
            std::cout << name << " shifted normalized: ";
            std::vector<uint32_t> shiftedNormalizedDigits(numActualDigitsPlusGuard, 0);
            for (int32_t i = 0; i < numActualDigitsPlusGuard; ++i) {
                uint32_t digit = GetShiftedNormalizedDigit<SharkFloatParams>(ext, numActualDigits, numActualDigitsPlusGuard, shift, diff, i);
                shiftedNormalizedDigits[i] = digit;
            }
            std::cout << VectorUintToHexString(shiftedNormalizedDigits.data(), numActualDigitsPlusGuard) << std::endl;
            };

        PrintOneShiftedNormalizedArray(ext_A_X2, shiftALeftToGetMsb, biasedExpABC - effExpA, "ext_A_X2");
        PrintOneShiftedNormalizedArray(ext_B_Y2, shiftBLeftToGetMsb, biasedExpABC - effExpB, "ext_B_Y2");
        PrintOneShiftedNormalizedArray(ext_C_A, shiftCLeftToGetMsb, biasedExpABC - effExpC, "ext_C_A");
        PrintOneShiftedNormalizedArray(ext_D_2X, shiftDLeftToGetMsb, biasedExpABC - effExpD, "ext_D_2X");
        PrintOneShiftedNormalizedArray(ext_E_B, shiftELeftToGetMsb, biasedExpABC - effExpE, "ext_E_B");
    }

    int32_t outExponent_ABC = 0;

    if constexpr (SharkFloatParams::HostVerbose) {
        std::cout << "threeWayMagnitude: " << ThreeWayMagnitudeToString(threeWayMagnitude) << std::endl;
        std::cout << "outExponent_ABC: " << outExponent_ABC << std::endl;
    }

    // --- Phase 1: A - B + C ---
    bool isNegative_ABC = false;
    Phase1_ABC<SharkFloatParams>(
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
        extResult_ABC,
        debugStates
    );

    // D + E
    // If effective exponents differ, use them. If equal, compare normalized digits on the fly.
    const bool DIsBiggerMagnitude = CompareMagnitudes2Way(
        effExpD,
        effExpE,
        numActualDigits,
        numActualDigitsPlusGuard,
        shiftDLeftToGetMsb,
        shiftELeftToGetMsb,
        ext_D_2X,
        ext_E_B);

    if constexpr (SharkFloatParams::HostVerbose) {
        std::cout << "DIsBiggerMagnitude: " << DIsBiggerMagnitude << std::endl;
    }

    // --- Phase 1: D+E ---
    int32_t outExponent_DE = 0;
    Phase1_DE(
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
        extResult_D_E,
        debugStates);


    // then carry-propagate extResult_ABC into OutXY1->Digits, set OutXY1->Exponent/IsNegative

    // --- Phase 2: Propagation ---
    // Propagate carries (if addition) or borrows (if subtraction)
    // and store the corrected 32-bit digit into propagatedResult.

    const bool sameSignDE = (D_2X->IsNegative == E_B->IsNegative);

    uint32_t carry_ABC = 0;
    uint32_t carry_DE = 0;

    // Result after propagation
    std::vector<uint32_t> propagatedResult_ABC(numActualDigitsPlusGuard, 0);
    std::vector<uint32_t> propagatedResult_DE(numActualDigitsPlusGuard, 0);

    if constexpr (UseBellochPropagation) {
        CarryPropagationPP_DE<SharkFloatParams>(
            sameSignDE,
            numActualDigitsPlusGuard,
            extResult_D_E,
            carry_DE,
            propagatedResult_DE);

        // Need _ABC version?
        assert(false);
    } else {
        CarryPropagation_ABC<SharkFloatParams>(
            numActualDigitsPlusGuard,
            extResult_ABC,
            *reinterpret_cast<int32_t *>(&carry_ABC),
            propagatedResult_ABC);

        CarryPropagation_DE<SharkFloatParams>(
            sameSignDE,
            numActualDigitsPlusGuard,
            extResult_D_E,
            carry_DE,
            propagatedResult_DE);
    }

    // At this point, the propagatedResult_DE array holds the result of the borrow/carry propagation.
    // A subsequent normalization step would adjust these digits (and the exponent) so that the most-significant
    // bit is in the desired position. This normalization step is omitted here.

    if constexpr (SharkFloatParams::HostVerbose) {
        std::cout << "propagatedResult_ABC after arithmetic: " << VectorUintToHexString(propagatedResult_ABC) << std::endl;
        std::cout << "propagatedResult_ABC: " << VectorUintToHexString(propagatedResult_ABC) << std::endl;
        std::cout << "carry_ABC out: 0x" << std::hex << carry_ABC << std::endl;

        std::cout << "propagatedResult_DE after arithmetic: " << VectorUintToHexString(propagatedResult_DE) << std::endl;
        std::cout << "propagatedResult_DE: " << VectorUintToHexString(propagatedResult_DE) << std::endl;
        std::cout << "carry_DE out: 0x" << std::hex << carry_DE << std::endl;
    }

    if constexpr (SharkDebugChecksums) {
        const auto &debugResultState_ABC = GetCurrentDebugState<SharkFloatParams, DebugStatePurpose::FinalAdd1>(
            debugStates, propagatedResult_ABC.data(), numActualDigitsPlusGuard);

        const auto &debugResultState_DE = GetCurrentDebugState<SharkFloatParams, DebugStatePurpose::FinalAdd2>(
            debugStates, propagatedResult_DE.data(), numActualDigitsPlusGuard);

        if constexpr (SharkFloatParams::HostVerbose) {
            std::cout << "propagatedResult_ABC checksum: " << debugResultState_ABC.GetStr() << std::endl;
            std::cout << "propagatedResult_DE checksum: " << debugResultState_DE.GetStr() << std::endl;
        }
    }

    // captures: numActualDigitsPlusGuard
    auto handleFinalCarry = [&](
        int32_t &outExponent,
        uint32_t carry,
        std::vector<uint32_t> &propagatedResult
        ) {
            if (carry == 0) return;

            // we only ever see carry==1 or carry==2
            const int shift = (carry == 2 ? 2 : 1);
            outExponent += shift;

            // seed the bits that will be injected into the MS limb
            // (carry is already only low 1 or 2 bits)
            uint32_t highBits = carry;

            // now walk from MS limb downward
            for (int i = numActualDigitsPlusGuard - 1; i >= 0; --i) {
                uint32_t w = propagatedResult[i];
                // grab the low 'shift' bits to feed into the next iteration
                uint32_t lowMask = (1u << shift) - 1;
                uint32_t nextHighBits = w & lowMask;

                // shift right and OR-in the bits carried from above
                propagatedResult[i] = (w >> shift)
                    | (highBits << (32 - shift));

                highBits = nextHighBits;
            }
        };

    handleFinalCarry(outExponent_ABC, carry_ABC, propagatedResult_ABC);
    handleFinalCarry(outExponent_DE, carry_DE, propagatedResult_DE);

    // --- Final Normalization ---
    int32_t msdResult_DE = 0;
    int32_t msdResult_ABC = 0;

    auto findMSD = [&](const std::vector<uint32_t> &result, int32_t &msdResult) {
        for (int32_t i = numActualDigitsPlusGuard - 1; i >= 0; i--) {
            if (result[i] != 0) {
                msdResult = i;
                break;
            }
        }
        };

    findMSD(propagatedResult_ABC, msdResult_ABC);
    findMSD(propagatedResult_DE, msdResult_DE);

    auto finalResolution = [](
        const char *prefixOutStr,
        const int32_t msdResult,
        const int32_t actualDigits,
        const int32_t extDigits,
        int32_t &outExponent,
        const std::vector<uint32_t> &propagatedResult,
        HpSharkFloat<SharkFloatParams> *ResultOut) {
            const int32_t clzResult = CountLeadingZeros(propagatedResult[msdResult]);
            const int32_t currentOverall = msdResult * 32 + (31 - clzResult);
            const int32_t desiredOverall = (SharkFloatParams::GlobalNumUint32 - 1) * 32 + 31;

            if constexpr (SharkFloatParams::HostVerbose) {
                std::cout << "prefixOutStr: " << prefixOutStr << std::endl;
                std::cout << "Count leading zeros: " << clzResult << std::endl;
                std::cout << "Current MSB index: " << msdResult << std::endl;
                std::cout << "Current overall bit position: " << currentOverall << std::endl;
                std::cout << "Desired overall bit position: " << desiredOverall << std::endl;
            }

            const int32_t shiftNeeded = currentOverall - desiredOverall;
            if (shiftNeeded > 0) {
                if constexpr (SharkFloatParams::HostVerbose) {
                    std::cout << "Shift needed branch D_2X: " << shiftNeeded << std::endl;
                }

                const auto shiftedSz = SharkFloatParams::GlobalNumUint32;
                MultiWordRightShift_LittleEndian(propagatedResult.data(), extDigits, shiftNeeded, ResultOut->Digits, shiftedSz);
                outExponent += shiftNeeded;

                if constexpr (SharkFloatParams::HostVerbose) {
                    std::cout << "Final propagatedResult after right shift: " <<
                        VectorUintToHexString(ResultOut->Digits, shiftedSz) <<
                        std::endl;
                    std::cout << "ShiftNeeded after right shift: " << shiftNeeded << std::endl;
                    std::cout << "Final outExponent after right shift: " << outExponent << std::endl;
                }
            } else if (shiftNeeded < 0) {
                if constexpr (SharkFloatParams::HostVerbose) {
                    std::cout << "Shift needed branch E_B: " << shiftNeeded << std::endl;
                }

                const int32_t L = -shiftNeeded;
                const auto shiftedSz = static_cast<int32_t>(SharkFloatParams::GlobalNumUint32);
                MultiWordLeftShift_LittleEndian(
                    propagatedResult.data(),
                    actualDigits,
                    extDigits,
                    L,
                    ResultOut->Digits,
                    shiftedSz);
                outExponent -= L;

                if constexpr (SharkFloatParams::HostVerbose) {
                    std::cout << "Final propagatedResult after left shift: " <<
                        VectorUintToHexString(ResultOut->Digits, shiftedSz) <<
                        std::endl;
                    std::cout << "L after left shift: " << L << std::endl;
                    std::cout << "Final outExponent after left shift: " << outExponent << std::endl;
                }
            } else {
                if constexpr (SharkFloatParams::HostVerbose) {
                    std::cout << "No shift needed: " << shiftNeeded << std::endl;
                }
                // No shift needed, just copy the result.
                memcpy(ResultOut->Digits, propagatedResult.data(), SharkFloatParams::GlobalNumUint32 * sizeof(uint32_t));
            }
        };

    // --- Finalize the result ---
    finalResolution(
        "A - B + C: ",
        msdResult_ABC,
        numActualDigits,
        numActualDigitsPlusGuard,
        outExponent_ABC,
        propagatedResult_ABC,
        OutXY1);

    OutXY1->Exponent = outExponent_ABC;
    OutXY1->IsNegative = isNegative_ABC;

    finalResolution(
        "D + E: ",
        msdResult_DE,
        numActualDigits,
        numActualDigitsPlusGuard,
        outExponent_DE,
        propagatedResult_DE,
        OutXY2);

    OutXY2->Exponent = outExponent_DE;
    // Set the result sign.
    if (sameSignDE)
        OutXY2->IsNegative = D_2X->IsNegative;
    else
        OutXY2->IsNegative = DIsBiggerMagnitude ? D_2X->IsNegative : E_B->IsNegative;

    if constexpr (SharkFloatParams::HostVerbose) {
        std::cout << "Final Resolution completed" << std::endl;
        std::cout << "Formatted results: " << std::endl;
        std::cout << "OutXY1: " << OutXY2->ToHexString() << std::endl;
        std::cout << "OutXY2: " << OutXY2->ToHexString() << std::endl;
    }

    if constexpr (SharkDebugChecksums) {
        const auto &debugResultState_ABC = GetCurrentDebugState<SharkFloatParams, DebugStatePurpose::Result_Add1>(
            debugStates, OutXY1->Digits, SharkFloatParams::GlobalNumUint32);

        const auto &debugResultState_DE = GetCurrentDebugState<SharkFloatParams, DebugStatePurpose::Result_Add2>(
            debugStates, OutXY2->Digits, SharkFloatParams::GlobalNumUint32);

        if constexpr (SharkFloatParams::HostVerbose) {
            std::cout << "OutXY1->Digits checksum: " << debugResultState_ABC.GetStr() << std::endl;
            std::cout << "OutXY2->Digits checksum: " << debugResultState_DE.GetStr() << std::endl;
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
        const HpSharkFloat<SharkFloatParams> *, \
        const HpSharkFloat<SharkFloatParams> *, \
        const HpSharkFloat<SharkFloatParams> *, \
        HpSharkFloat<SharkFloatParams> *, \
        HpSharkFloat<SharkFloatParams> *, \
        std::vector<DebugStateHost<SharkFloatParams>> &debugStates);

ExplicitInstantiateAll();
