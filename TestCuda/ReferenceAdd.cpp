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

//
// Helper functions to perform bit shifts on a fixed-width digit array.
// They mirror the CUDA device functions but work sequentially on the full array.
//

// ShiftRight: Shifts the number (given by its digit array) right by shiftBits.
// idx is the index of the digit to compute. The parameter numDigits prevents out-of-bounds access.
uint32_t ShiftRight (
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
uint32_t ShiftLeft (
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
int32_t CountLeadingZeros (
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
void MultiWordRightShift_LittleEndian (
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
void MultiWordLeftShift_LittleEndian(
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

uint32_t GetExtLimb(
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
int32_t ExtendedNormalizeShiftIndex (
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
uint32_t GetNormalizedDigit (
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
uint32_t GetShiftedNormalizedDigit (
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
void GetCorrespondingLimbs (
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

bool CompareMagnitudes (
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

//
// Extended arithmetic using little-endian representation.
// This version uses the new normalization approach, where the extended operands
// are not copied; instead, a shift index is returned and used later to compute
// normalized digits on the fly.
//
template<class SharkFloatParams>
void AddHelper(
    const HpSharkFloat<SharkFloatParams> *A,
    const HpSharkFloat<SharkFloatParams> *B,
    HpSharkFloat<SharkFloatParams> *OutXY,
    std::vector<DebugStateHost<SharkFloatParams>> &debugStates
) {
    // Make local copies.
    const auto *extA = A->Digits;
    const auto *extB = B->Digits;

    // --- Set up extended working precision ---
    constexpr int32_t guard = 2;
    constexpr int32_t actualDigits = SharkFloatParams::GlobalNumUint32;
    constexpr int32_t extDigits = SharkFloatParams::GlobalNumUint32 + guard;
    // Create extended arrays (little-endian, index 0 is LSB).
    std::vector<uint32_t> extResult(extDigits, 0);

    // The guard words (indices GlobalNumUint32 to extDigits-1) are left as zero.

    if constexpr (SharkFloatParams::HostVerbose) {
        std::cout << "extA: " << VectorUintToHexString(extA) << std::endl;
        std::cout << "extA exponent: " << A->Exponent << std::endl;
        std::cout << "extB: " << VectorUintToHexString(extB) << std::endl;
        std::cout << "extB exponent: " << B->Exponent << std::endl;
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

    // --- Extended Arithmetic ---
    if (sameSign) {
        // ---- Addition Branch ----
        uint64_t carry = 0;
        for (int32_t i = 0; i < extDigits; i++) {
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
            uint64_t sum = alignedA + alignedB + carry;
            extResult[i] = (uint32_t)(sum & 0xFFFFFFFF);
            carry = sum >> 32;
        }
        if (carry > 0) {
            outExponent += 1;
            // Use a single variable to hold the extra bit from the next word.
            uint32_t nextBit = (uint32_t)carry & 1; // carry is either 0 or 1
            // Process in reverse order so that we use the original value of each word.
            for (int32_t i = extDigits - 1; i >= 0; i--) {
                uint32_t current = extResult[i];
                // The new value is the current word right-shifted by 1, with the previous word's LSB (held in nextBit)
                // shifted into the high bit.
                extResult[i] = (current >> 1) | (nextBit << 31);
                // Extract the LSB of the current word (from the original value) to use for the next iteration.
                nextBit = current & 1;
            }
        }

        // The result sign remains the same.
    } else {
        // ---- Subtraction Branch ----
        uint64_t borrow = 0;
        if (AIsBiggerMagnitude) {
            for (int32_t i = 0; i < extDigits; i++) {
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
                int64_t diffVal = (int64_t)alignedA - (int64_t)alignedB - borrow;
                if (diffVal < 0) {
                    diffVal += ((uint64_t)1 << 32);
                    borrow = 1;
                } else {
                    borrow = 0;
                }
                extResult[i] = (uint32_t)(diffVal & 0xFFFFFFFF);
            }
        } else {
            for (int32_t i = 0; i < extDigits; i++) {
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
                int64_t diffVal = (int64_t)alignedB - (int64_t)alignedA - borrow;
                if (diffVal < 0) {
                    diffVal += ((uint64_t)1 << 32);
                    borrow = 1;
                } else {
                    borrow = 0;
                }
                extResult[i] = (uint32_t)(diffVal & 0xFFFFFFFF);
            }
        }
        assert(borrow == 0 && "Final borrow in subtraction should be zero");
    }

    if constexpr (SharkFloatParams::HostVerbose) {
        std::cout << "extResult after arithmetic: " << VectorUintToHexString(extResult) << std::endl;
        std::cout << "outExponent after arithmetic, before renormalization: " << outExponent << std::endl;
        std::cout << "extResult: " << VectorUintToHexString(extResult) << std::endl;
    }

    // --- Final Normalization ---
    int32_t msdResult = 0;
    for (int32_t i = extDigits - 1; i >= 0; i--) {
        if (extResult[i] != 0) {
            msdResult = i;
            break;
        }
    }

    const int32_t clzResult = CountLeadingZeros(extResult[msdResult]);
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
        MultiWordRightShift_LittleEndian(extResult.data(), extDigits, shiftNeeded, OutXY->Digits, shiftedSz);
        outExponent += shiftNeeded;

        if constexpr (SharkFloatParams::HostVerbose) {
            std::cout << "Final extResult after right shift: " <<
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
            extResult.data(),
            actualDigits,
            extDigits,
            L,
            OutXY->Digits,
            shiftedSz);
        outExponent -= L;

        if constexpr (SharkFloatParams::HostVerbose) {
            std::cout << "Final extResult after left shift: " <<
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
        memcpy(OutXY->Digits, extResult.data(), SharkFloatParams::GlobalNumUint32 * sizeof(uint32_t));
    }

    OutXY->Exponent = outExponent;
    // Set the result sign.
    if (sameSign)
        OutXY->IsNegative = A->IsNegative;
    else
        OutXY->IsNegative = AIsBiggerMagnitude ? A->IsNegative : B->IsNegative;

    DebugStateHost<SharkFloatParams> dbg;
    debugStates.push_back(dbg);
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
