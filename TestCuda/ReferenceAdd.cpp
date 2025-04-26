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
// 'diffDE' is the additional right shift required for alignment.
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

// “Strict” ordering of three magnitudes (ignores exact ties – see note below)
enum class ThreeWayMagnitude {
    A_GT_B_GT_C,  // A > B > C
    A_GT_C_GT_B,  // A > C > B
    B_GT_A_GT_C,  // B > A > C
    B_GT_C_GT_A,  // B > C > A
    C_GT_A_GT_B,  // C > A > B
    C_GT_B_GT_A   // C > B > A
};

static ThreeWayMagnitude
CompareMagnitudes(
    int32_t effExpA,
    int32_t effExpB,
    int32_t effExpC,
    const int32_t actualDigits,
    const int32_t extDigits,
    const uint32_t *extA,
    const int32_t shiftA,
    const uint32_t *extB,
    const int32_t shiftB,
    const uint32_t *extC,
    const int32_t shiftC
) {
    // Helper: returns true if “first” is strictly bigger than “second”
    auto cmp = [&](const uint32_t *e1, int32_t s1, int32_t exp1,
        const uint32_t *e2, int32_t s2, int32_t exp2) {
            if (exp1 != exp2)
                return exp1 > exp2;
            // exponents equal → compare normalized digits high→low
            for (int32_t i = extDigits - 1; i >= 0; --i) {
                uint32_t d1 = GetNormalizedDigit(e1, actualDigits, extDigits, s1, i);
                uint32_t d2 = GetNormalizedDigit(e2, actualDigits, extDigits, s2, i);
                if (d1 != d2)
                    return d1 > d2;
            }
            return false;  // treat exact equality as “not greater”
        };

    // 1) Is A the strict max?
    if (cmp(extA, shiftA, effExpA, extB, shiftB, effExpB) &&
        cmp(extA, shiftA, effExpA, extC, shiftC, effExpC)) {
        // now order B vs C
        if (cmp(extB, shiftB, effExpB, extC, shiftC, effExpC))
            return ThreeWayMagnitude::A_GT_B_GT_C;
        else
            return ThreeWayMagnitude::A_GT_C_GT_B;
    }
    // 2) Is B the strict max?
    else if (cmp(extB, shiftB, effExpB, extA, shiftA, effExpA) &&
        cmp(extB, shiftB, effExpB, extC, shiftC, effExpC)) {
        // now order A vs C
        if (cmp(extA, shiftA, effExpA, extC, shiftC, effExpC))
            return ThreeWayMagnitude::B_GT_A_GT_C;
        else
            return ThreeWayMagnitude::B_GT_C_GT_A;
    }
    // 3) Otherwise C is the (strict) max
    else {
        // order A vs B
        if (cmp(extA, shiftA, effExpA, extB, shiftB, effExpB))
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

template<class SharkFloatParams>
void Phase1_DE (
    const bool sameSignDE,
    const int32_t &extDigits,
    const auto *ext_D_2x,
    const int32_t &actualDigits,
    const auto *ext_E_B,
    const int32_t &shiftD,
    const int32_t &shiftE,
    const bool DIsBiggerMagnitude,
    const int32_t &diffDE,
    std::vector<uint64_t> &extResult_D_E,
    std::vector<DebugStateHost<SharkFloatParams>> &debugStates)
{
    // --- Phase 1: Raw Extended Arithmetic ---
    // Compute the raw limb-wise result without propagation.
    if (sameSignDE) {
        // Addition branch.
        for (int32_t i = 0; i < extDigits; i++) {
            uint64_t alignedA = 0, alignedB = 0;
            GetCorrespondingLimbs<SharkFloatParams>(
                ext_D_2x, actualDigits, extDigits,
                ext_E_B, actualDigits, extDigits,
                shiftD, shiftE,
                DIsBiggerMagnitude, diffDE, i,
                alignedA, alignedB);
            extResult_D_E[i] = alignedA + alignedB;
        }
    } else {
        // Subtraction branch.
        std::vector<uint64_t> alignedADebug;
        std::vector<uint64_t> alignedBDebug;

        if (DIsBiggerMagnitude) {
            for (int32_t i = 0; i < extDigits; i++) {
                uint64_t alignedA = 0, alignedB = 0;
                GetCorrespondingLimbs<SharkFloatParams>(
                    ext_D_2x, actualDigits, extDigits,
                    ext_E_B, actualDigits, extDigits,
                    shiftD, shiftE,
                    DIsBiggerMagnitude, diffDE, i,
                    alignedA, alignedB);
                // Compute raw difference (which may be negative).
                int64_t rawDiff = (int64_t)alignedA - (int64_t)alignedB;
                extResult_D_E[i] = (uint64_t)rawDiff;

                alignedADebug.push_back(alignedA);
                alignedBDebug.push_back(alignedB);
            }
        } else {
            for (int32_t i = 0; i < extDigits; i++) {
                uint64_t alignedA = 0, alignedB = 0;
                GetCorrespondingLimbs<SharkFloatParams>(
                    ext_D_2x, actualDigits, extDigits,
                    ext_E_B, actualDigits, extDigits,
                    shiftD, shiftE,
                    DIsBiggerMagnitude, diffDE, i,
                    alignedA, alignedB);
                int64_t rawDiff = (int64_t)alignedB - (int64_t)alignedA;
                extResult_D_E[i] = (uint64_t)rawDiff;

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
            debugStates, extResult_D_E.data(), extDigits);

        if constexpr (SharkFloatParams::HostVerbose) {
            std::cout << "extResult_D_E checksum: " << debugResultState.GetStr() << std::endl;
            std::cout << "extResult_D_E after arithmetic: " << VectorUintToHexString(extResult_D_E) << std::endl;
        }
    }
}

// raw extended A - B + C, no carry/borrow propagation, stores into extResult_ABC
template<class SharkFloatParams>
void Phase1_ABC(
    const int32_t  extDigits,
    const int32_t  actualDigits,
    const uint32_t *extA,    // A_X2 -> Digits
    const int32_t  shiftA,
    const uint32_t *extB,    // B_Y2 -> Digits
    const int32_t  shiftB,
    const uint32_t *extC,    // C_A  -> Digits
    const int32_t  shiftC,
    const ThreeWayMagnitude threeWayMagnitude,
    const int32_t  effExpA,
    const int32_t  effExpB,
    const int32_t  effExpC,
    std::vector<uint64_t> &extResult_ABC,
    std::vector<DebugStateHost<SharkFloatParams>> &debugStates
) {
    // 1) figure out A vs B subtraction direction + exponent diff
    const bool AIsBiggerThanB =
        (threeWayMagnitude == ThreeWayMagnitude::A_GT_B_GT_C) ||
        (threeWayMagnitude == ThreeWayMagnitude::A_GT_C_GT_B) ||
        (threeWayMagnitude == ThreeWayMagnitude::C_GT_A_GT_B);
    const int32_t diffAB = AIsBiggerThanB
        ? (effExpA - effExpB)
        : (effExpB - effExpA);
    const int32_t expAB = AIsBiggerThanB ? effExpA : effExpB;

    // 2) figure out which exponent is larger: (A-B) vs C, so we can align C onto the diff
    const bool ABIsBiggerThanC = (expAB >= effExpC);
    const int32_t diffABC = ABIsBiggerThanC
        ? (expAB - effExpC)
        : (effExpC - expAB);

    if constexpr (SharkFloatParams::HostVerbose) {
        std::cout << "Phase1_ABC: A>B? " << AIsBiggerThanB
            << ", diffAB=" << diffAB
            << ", ABexp=" << expAB
            << ", ABvsC? " << ABIsBiggerThanC
            << ", diffABC=" << diffABC
            << std::endl;
    }

    // 3) Stage 1: compute raw A-B (or B-A) into an intermediate array
    std::vector<uint64_t> rawAB(extDigits);
    for (int32_t i = 0; i < extDigits; ++i) {
        uint64_t a_i, b_i;
        if (AIsBiggerThanB) {
            a_i = GetNormalizedDigit(extA, actualDigits, extDigits, shiftA, i);
            b_i = GetShiftedNormalizedDigit<SharkFloatParams>(
                extB, actualDigits, extDigits, shiftB, diffAB, i);
            rawAB[i] = a_i - b_i;
        } else {
            b_i = GetNormalizedDigit(extB, actualDigits, extDigits, shiftB, i);
            a_i = GetShiftedNormalizedDigit<SharkFloatParams>(
                extA, actualDigits, extDigits, shiftA, diffAB, i);
            rawAB[i] = b_i - a_i;
        }
    }

    if constexpr (SharkFloatParams::HostVerbose) {
        std::cout << "Phase1_ABC raw A–B: " << VectorUintToHexString(rawAB) << std::endl;
    }

    // 4) Stage 2: add C onto that difference
    for (int32_t i = 0; i < extDigits; ++i) {
        uint64_t c_i = ABIsBiggerThanC
            ? GetShiftedNormalizedDigit<SharkFloatParams>(
                extC, actualDigits, extDigits, shiftC, diffABC, i)
            : GetNormalizedDigit(extC, actualDigits, extDigits, shiftC, i);
        extResult_ABC[i] = rawAB[i] + c_i;
    }

    if constexpr (SharkFloatParams::HostVerbose) {
        std::cout << "Phase1_ABC final A–B+ C: " << VectorUintToHexString(extResult_ABC)
            << std::endl;
    }

    // 5) optional: snapshot debug state
    if constexpr (SharkDebugChecksums) {
        const auto &dbg = GetCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z2XY>(
            debugStates, extResult_ABC.data(), extDigits);
        if constexpr (SharkFloatParams::HostVerbose) {
            std::cout << "Phase1_ABC checksum: " << dbg.GetStr() << std::endl;
        }
    }
}


//
// Ternary operator that calculates:
// OutXY1 = A_X2 - B_Y2 + C_A
// OutXY2 = D_2X + E_B
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
    )
{
    if constexpr (SharkDebugChecksums) {
        constexpr auto NewDebugStateSize = static_cast<int>(DebugStatePurpose::NumPurposes);
        debugStates.resize(NewDebugStateSize);
    }

    // Make local copies.
    const auto *ext_A_X2 = A_X2->Digits;
    const auto *ext_B_Y2 = B_Y2->Digits;
    const auto *ext_C_A = C_A->Digits;
    const auto *ext_D_2x = D_2X->Digits;
    const auto *ext_E_B = E_B->Digits;

    // --- Set up extended working precision ---
    constexpr int32_t guard = 2;
    constexpr int32_t actualDigits = SharkFloatParams::GlobalNumUint32;
    constexpr int32_t extDigits = SharkFloatParams::GlobalNumUint32 + guard;
    // Create extended arrays (little-endian, index 0 is LSB).
    std::vector<uint64_t> extResult_ABC(extDigits, 0);
    std::vector<uint64_t> extResult_D_E(extDigits, 0);

    // The guard words (indices GlobalNumUint32 to extDigits-1) are left as zero.

    if constexpr (SharkFloatParams::HostVerbose) {
        std::cout << "ext_A_X2: " << VectorUintToHexString(ext_A_X2, actualDigits) << std::endl;
        std::cout << "ext_A_X2 exponent: " << A_X2->Exponent << std::endl;

        std::cout << "ext_B_Y2: " << VectorUintToHexString(ext_B_Y2, actualDigits) << std::endl;
        std::cout << "ext_B_Y2 exponent: " << B_Y2->Exponent << std::endl;

        std::cout << "ext_C_A: " << VectorUintToHexString(ext_C_A, actualDigits) << std::endl;
        std::cout << "ext_C_A exponent: " << C_A->Exponent << std::endl;

        std::cout << "ext_D_2x: " << VectorUintToHexString(ext_D_2x, actualDigits) << std::endl;
        std::cout << "ext_D_2x exponent: " << D_2X->Exponent << std::endl;

        std::cout << "ext_E_B: " << VectorUintToHexString(ext_E_B, actualDigits) << std::endl;
        std::cout << "ext_E_B exponent: " << E_B->Exponent << std::endl;
    }

    if constexpr (SharkDebugChecksums) {
        // Compute checksums for the extended arrays.
        // Note: we use the actual digits (not the extended size) for the checksum.

        const auto &debugState = GetCurrentDebugState<SharkFloatParams, DebugStatePurpose::ADigits>(
            debugStates, ext_A_X2, actualDigits);

        const auto &debugBState = GetCurrentDebugState<SharkFloatParams, DebugStatePurpose::BDigits>(
            debugStates, ext_B_Y2, actualDigits);

        const auto &debugCState = GetCurrentDebugState<SharkFloatParams, DebugStatePurpose::CDigits>(
            debugStates, ext_C_A, actualDigits);

        const auto &debugDState = GetCurrentDebugState<SharkFloatParams, DebugStatePurpose::DDigits>(
            debugStates, ext_D_2x, actualDigits);

        const auto &debugEState = GetCurrentDebugState<SharkFloatParams, DebugStatePurpose::EDigits>(
            debugStates, ext_E_B, actualDigits);

        if constexpr (SharkFloatParams::HostVerbose) {
            std::cout << "A_X2->Digits checksum: " << debugState.GetStr() << std::endl;
            std::cout << "B_Y2->Digits checksum: " << debugBState.GetStr() << std::endl;
            std::cout << "C_A->Digits checksum: " << debugCState.GetStr() << std::endl;
            std::cout << "D_2X->Digits checksum: " << debugDState.GetStr() << std::endl;
            std::cout << "E_B->Digits checksum: " << debugEState.GetStr() << std::endl;
        }
    }

    // --- Extended Normalization using shift indices ---
    const bool sameSignDE = (D_2X->IsNegative == E_B->IsNegative);

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
    const int32_t shiftA = ExtendedNormalizeShiftIndex(
        ext_A_X2,
        actualDigits,
        extDigits,
        newAExponent,
        normA_isZero);

    const int32_t shiftB = ExtendedNormalizeShiftIndex(
        ext_B_Y2,
        actualDigits,
        extDigits,
        newBExponent,
        normB_isZero);

    const int32_t shiftC = ExtendedNormalizeShiftIndex(
        ext_C_A,
        actualDigits,
        extDigits,
        newCExponent,
        normC_isZero);

    const int32_t shiftD = ExtendedNormalizeShiftIndex(
        ext_D_2x,
        actualDigits,
        extDigits,
        newDExponent,
        normD_isZero);

    const int32_t shiftE = ExtendedNormalizeShiftIndex(
        ext_E_B,
        actualDigits,
        extDigits,
        newEExponent,
        normE_isZero);

    if constexpr (SharkFloatParams::HostVerbose) {
        std::cout << "ext_A_X2 after normalization: " << VectorUintToHexString(ext_A_X2, actualDigits) << std::endl;
        std::cout << "shiftA: " << shiftA << std::endl;

        std::cout << "ext_B_Y2 after normalization: " << VectorUintToHexString(ext_B_Y2, actualDigits) << std::endl;
        std::cout << "shiftB: " << shiftB << std::endl;

        std::cout << "ext_C_A after normalization: " << VectorUintToHexString(ext_C_A, actualDigits) << std::endl;
        std::cout << "shiftC: " << shiftC << std::endl;

        std::cout << "ext_D_2x after normalization: " << VectorUintToHexString(ext_D_2x, actualDigits) << std::endl;
        std::cout << "shiftD: " << shiftD << std::endl;

        std::cout << "ext_E_B after normalization: " << VectorUintToHexString(ext_E_B, actualDigits) << std::endl;
        std::cout << "shiftE: " << shiftE << std::endl;
    }

    // --- Compute Effective Exponents ---
    const int32_t effExpA = normA_isZero ? -100'000'000 : newAExponent + (SharkFloatParams::GlobalNumUint32 * 32 - 32);
    const int32_t effExpB = normB_isZero ? -100'000'000 : newBExponent + (SharkFloatParams::GlobalNumUint32 * 32 - 32);
    const int32_t effExpC = normC_isZero ? -100'000'000 : newCExponent + (SharkFloatParams::GlobalNumUint32 * 32 - 32);
    const int32_t effExpD = normD_isZero ? -100'000'000 : newDExponent + (SharkFloatParams::GlobalNumUint32 * 32 - 32);
    const int32_t effExpE = normE_isZero ? -100'000'000 : newEExponent + (SharkFloatParams::GlobalNumUint32 * 32 - 32);

    if constexpr (SharkFloatParams::HostVerbose) {
        std::cout << "effExpA: " << effExpA << std::endl;
        std::cout << "effExpB: " << effExpB << std::endl;
        std::cout << "effExpC: " << effExpC << std::endl;
        std::cout << "effExpD: " << effExpD << std::endl;
        std::cout << "effExpE: " << effExpE << std::endl;
    }

    // --- Determine which operand has larger magnitude ---
    // If effective exponents differ, use them. If equal, compare normalized digits on the fly.
    const bool DIsBiggerMagnitude = CompareMagnitudes(
        effExpD,
        effExpE,
        actualDigits,
        extDigits,
        ext_D_2x,
        shiftD,
        ext_E_B,
        shiftE);

    const int32_t diffDE = DIsBiggerMagnitude ? (effExpD - effExpE) : (effExpE - effExpD);
    int32_t outExponent = DIsBiggerMagnitude ? newDExponent : newEExponent;

    if constexpr (SharkFloatParams::HostVerbose) {
        std::cout << "DIsBiggerMagnitude: " << DIsBiggerMagnitude << std::endl;
        std::cout << "diffDE: " << diffDE << std::endl;
        std::cout << "outExponent: " << outExponent << std::endl;
    }

    // --- Phase 1: D+E ---
    Phase1_DE(
        sameSignDE,
        extDigits,
        ext_D_2x,
        actualDigits,
        ext_E_B,
        shiftD,
        shiftE,
        DIsBiggerMagnitude,
        diffDE,
        extResult_D_E,
        debugStates);

    // Do a 3-way comparison of the other three operands.
    // We need to compare A_X2, B_Y2, and C_A.
    // The result is a 3-way ordering of the three operands.

    const auto threeWayMagnitude = CompareMagnitudes(
        effExpA,
        effExpB,
        effExpC,
        actualDigits,
        extDigits,
        ext_A_X2,
        shiftA,
        ext_B_Y2,
        shiftB,
        ext_C_A,
        shiftC);

    // --- Phase 1: A-B+C ---
    Phase1_ABC<SharkFloatParams>(
        extDigits,
        actualDigits,
        ext_A_X2,
        shiftA,
        ext_B_Y2,
        shiftB,
        ext_C_A,
        shiftC,
        threeWayMagnitude,
        effExpA,
        effExpB,
        effExpC,
        extResult_ABC,
        debugStates
    );

    // then carry‐propagate extResult_ABC into OutXY1->Digits, set OutXY1->Exponent/IsNegative

    // --- Phase 2: Propagation ---
    // Propagate carries (if addition) or borrows (if subtraction)
    // and store the corrected 32-bit digit into propagatedResult.

    uint32_t carry = 0;
    std::vector<uint32_t> propagatedResult(extDigits, 0); // Result after propagation
    if constexpr (UseBellochPropagation) {
        CarryPropagationPP<SharkFloatParams>(
            sameSignDE,
            extDigits,
            extResult_D_E,
            carry,
            propagatedResult);
    } else {
        CarryPropagation<SharkFloatParams>(
            sameSignDE,
            extDigits,
            extResult_D_E,
            carry,
            propagatedResult);
    }

    // At this point, the propagatedResult array holds the result of the borrow/carry propagation.
    // A subsequent normalization step would adjust these digits (and the exponent) so that the most-significant
    // bit is in the desired position. This normalization step is omitted here.

    if constexpr (SharkFloatParams::HostVerbose) {
        std::cout << "propagatedResult after arithmetic: " << VectorUintToHexString(propagatedResult) << std::endl;
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
            std::cout << "Shift needed branch D_2X: " << shiftNeeded << std::endl;
        }

        const auto shiftedSz = SharkFloatParams::GlobalNumUint32;
        MultiWordRightShift_LittleEndian(propagatedResult.data(), extDigits, shiftNeeded, OutXY2->Digits, shiftedSz);
        outExponent += shiftNeeded;

        if constexpr (SharkFloatParams::HostVerbose) {
            std::cout << "Final propagatedResult after right shift: " <<
                VectorUintToHexString(OutXY2->Digits, shiftedSz) <<
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
            OutXY2->Digits,
            shiftedSz);
        outExponent -= L;

        if constexpr (SharkFloatParams::HostVerbose) {
            std::cout << "Final propagatedResult after left shift: " <<
                VectorUintToHexString(OutXY2->Digits, shiftedSz) <<
                std::endl;
            std::cout << "L after left shift: " << L << std::endl;
            std::cout << "Final outExponent after left shift: " << outExponent << std::endl;
        }
    } else {
        if constexpr (SharkFloatParams::HostVerbose) {
            std::cout << "No shift needed: " << shiftNeeded << std::endl;
        }
        // No shift needed, just copy the result.
        memcpy(OutXY2->Digits, propagatedResult.data(), SharkFloatParams::GlobalNumUint32 * sizeof(uint32_t));
    }

    OutXY2->Exponent = outExponent;
    // Set the result sign.
    if (sameSignDE)
        OutXY2->IsNegative = D_2X->IsNegative;
    else
        OutXY2->IsNegative = DIsBiggerMagnitude ? D_2X->IsNegative : E_B->IsNegative;

    if constexpr (SharkDebugChecksums) {
        const auto &debugResultState = GetCurrentDebugState<SharkFloatParams, DebugStatePurpose::Result_offsetXY>(
            debugStates, OutXY2->Digits, SharkFloatParams::GlobalNumUint32);

        if constexpr (SharkFloatParams::HostVerbose) {
            std::cout << "OutXY2->Digits checksum: " << debugResultState.GetStr() << std::endl;
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
