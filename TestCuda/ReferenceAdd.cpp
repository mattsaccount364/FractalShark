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
inline uint32_t ShiftRight (const uint32_t *digits, int shiftBits, int idx, int numDigits)
{
    int shiftWords = shiftBits / 32;
    int shiftBitsMod = shiftBits % 32;
    uint32_t lower = (idx + shiftWords < numDigits) ? digits[idx + shiftWords] : 0;
    uint32_t upper = (idx + shiftWords + 1 < numDigits) ? digits[idx + shiftWords + 1] : 0;
    if (shiftBitsMod == 0) {
        return lower;
    } else {
        return (lower >> shiftBitsMod) | (upper << (32 - shiftBitsMod));
    }
}

// ShiftLeft: Shifts the number (given by its digit array) left by shiftBits.
// idx is the index of the digit to compute.
inline uint32_t ShiftLeft (const uint32_t *digits, int shiftBits, int idx)
{
    int shiftWords = shiftBits / 32;
    int shiftBitsMod = shiftBits % 32;
    int srcIdx = idx - shiftWords;
    uint32_t lower = (srcIdx >= 0) ? digits[srcIdx] : 0;
    uint32_t upper = (srcIdx - 1 >= 0) ? digits[srcIdx - 1] : 0;
    if (shiftBitsMod == 0) {
        return lower;
    } else {
        return (lower << shiftBitsMod) | (upper >> (32 - shiftBitsMod));
    }
}

// Portable helper: CountLeadingZeros for a 32-bit integer.
// On CUDA consider:
// __device__ inline int CountLeadingZerosCUDA(uint32_t x) {
// return __clz(x);
// }
inline int CountLeadingZeros (uint32_t x)
{
    int count = 0;
    for (int bit = 31; bit >= 0; --bit) {
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
inline void MultiWordRightShift_LittleEndian (const std::vector<uint32_t> &in, int L, std::vector<uint32_t> &out)
{
    const int n = static_cast<int>(in.size());
    out.resize(n, 0);
    for (int i = 0; i < n; i++) {
        out[i] = ShiftRight(in.data(), L, i, n);
    }
}

// MultiWordLeftShift_LittleEndian: shift an array 'in' (of length n) left by L bits,
// storing the result in 'out'.
inline void MultiWordLeftShift_LittleEndian (const std::vector<uint32_t> &in, int L, std::vector<uint32_t> &out)
{
    const int n = static_cast<int>(in.size());
    out.resize(n, 0);
    for (int i = 0; i < n; i++) {
        out[i] = ShiftLeft(in.data(), L, i);
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
int ExtendedNormalizeShiftIndex (const std::vector<uint32_t> &ext, int &storedExp, bool &isZero)
{
    const int n = static_cast<int>(ext.size());
    int msd = n - 1;
    while (msd >= 0 && ext[msd] == 0)
        msd--;
    if (msd < 0) {
        isZero = true;
        return 0;  // For zero, the shift offset is irrelevant.
    }
    isZero = false;
    int clz = CountLeadingZeros(ext[msd]);
    // In little-endian, the overall bit index of the MSB is:
    //    current_msb = msd * 32 + (31 - clz)
    int current_msb = msd * 32 + (31 - clz);
    int totalExtBits = n * 32;
    // Compute the left-shift needed so that the MSB moves to bit (totalExtBits - 1).
    int L = (totalExtBits - 1) - current_msb;
    // Adjust the exponent as if we had shifted the number left by L bits.
    storedExp -= L;
    return L;
}

//
// Helper to retrieve a normalized digit on the fly.
// Given the original extended array and a shift offset (obtained from ExtendedNormalizeShiftIndex),
// this returns the digit at index 'idx' as if the array had been left-shifted by shiftOffset bits.
//
inline uint32_t GetNormalizedDigit (const std::vector<uint32_t> &ext, int shiftOffset, int idx)
{
    return ShiftLeft(ext.data(), shiftOffset, idx);
}

//
// Final normalization: Convert the extended result (little-endian)
// into a nominal result with GlobalNumUint32 words by re-normalizing.
// This function remains mostly unchanged.
//
template <class SharkFloatParams>
void FinalNormalizeExtendedResult (
    const std::vector<uint32_t> &extResult,
    int &outExponent,
    std::vector<uint32_t> &finalResult)
{
    const int nominalWords = SharkFloatParams::GlobalNumUint32;
    const int extDigits = extResult.size();
    // Find the index of the highest nonzero word.
    int msd = 0;
    for (int i = extDigits - 1; i >= 0; i--) {
        if (extResult[i] != 0) { msd = i; break; }
    }
    int clz = CountLeadingZeros(extResult[msd]);
    int currentOverall = msd * 32 + (31 - clz);
    // Desired overall bit position for the nominal result's MSB:
    int desired = (nominalWords - 1) * 32 + 31;
    int shiftNeeded = currentOverall - desired;
    // Work on a temporary copy.
    std::vector<uint32_t> temp = extResult;
    if (shiftNeeded > 0) {
        std::vector<uint32_t> shifted;
        MultiWordRightShift_LittleEndian(temp, shiftNeeded, shifted);
        temp = shifted;
        outExponent += shiftNeeded;
    } else if (shiftNeeded < 0) {
        int L = -shiftNeeded;
        std::vector<uint32_t> shifted;
        MultiWordLeftShift_LittleEndian(temp, L, shifted);
        temp = shifted;
        outExponent -= L;
    }
    // Extract the nominal GlobalNumUint32 words.
    finalResult.resize(nominalWords);
    for (int i = 0; i < nominalWords; i++) {
        finalResult[i] = temp[i];
    }
}

// New helper: Computes the aligned digit for the normalized value on the fly.
// 'diff' is the additional right shift required for alignment.
template <class SharkFloatParams>
inline uint32_t GetShiftedNormalizedDigit (const std::vector<uint32_t> &ext, int shiftOffset, int diff, int idx)
{
    // const int n = SharkFloatParams::GlobalNumUint32; // normalized length
    const int n = static_cast<int>(ext.size());
    int wordShift = diff / 32;
    int bitShift = diff % 32;
    uint32_t lower = (idx + wordShift < n) ? GetNormalizedDigit(ext, shiftOffset, idx + wordShift) : 0;
    uint32_t upper = (idx + wordShift + 1 < n) ? GetNormalizedDigit(ext, shiftOffset, idx + wordShift + 1) : 0;
    if (bitShift == 0)
        return lower;
    else
        return (lower >> bitShift) | (upper << (32 - bitShift));
}

template<class SharkFloatParams>
inline void GetCorrespondingLimbs (
    const std::vector<uint32_t> &extA,
    const std::vector<uint32_t> &extB,
    const int shiftA,
    const int shiftB,
    const bool AIsBiggerMagnitude,
    const int diff,
    const int index,
    uint64_t &alignedA,
    uint64_t &alignedB)
{
    if (AIsBiggerMagnitude) {
        // A is larger: normalized A is used as is.
        // For B, we normalize and then shift right by 'diff'.
        alignedA = GetNormalizedDigit(extA, shiftA, index);
        alignedB = GetShiftedNormalizedDigit<SharkFloatParams>(extB, shiftB, diff, index);
    } else {
        // B is larger: normalized B is used as is.
        // For A, we normalize and shift right by 'diff'.
        alignedB = GetNormalizedDigit(extB, shiftB, index);
        alignedA = GetShiftedNormalizedDigit<SharkFloatParams>(extA, shiftA, diff, index);
    }
}

bool CompareMagnitudes (
    int effExpA,
    int effExpB,
    const int extDigits,
    const std::vector<uint32_t> &extA,
    const int32_t shiftA,
    const std::vector<uint32_t> &extB,
    const int32_t shiftB) {
    bool AIsBiggerMagnitude;

    if (effExpA > effExpB) {
        AIsBiggerMagnitude = true;
    } else if (effExpA < effExpB) {
        AIsBiggerMagnitude = false;
    } else {
        AIsBiggerMagnitude = false; // default if equal
        for (int i = extDigits - 1; i >= 0; i--) {
            uint32_t digitA = GetNormalizedDigit(extA, shiftA, i);
            uint32_t digitB = GetNormalizedDigit(extB, shiftB, i);
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
    const auto &normA = *A;
    const auto &normB = *B;

    // --- Set up extended working precision ---
    const int guard = 2;
    const int extDigits = SharkFloatParams::GlobalNumUint32 + guard;
    // Create extended arrays (little-endian, index 0 is LSB).
    std::vector<uint32_t> extA(extDigits, 0);
    std::vector<uint32_t> extB(extDigits, 0);
    for (int i = 0; i < SharkFloatParams::GlobalNumUint32; i++) {
        extA[i] = normA.Digits[i];
        extB[i] = normB.Digits[i];
    }
    // The guard words (indices GlobalNumUint32 to extDigits-1) are left as zero.

    if constexpr (SharkFloatParams::HostVerbose) {
        std::cout << "extA: " << VectorUintToHexString(extA) << std::endl;
        std::cout << "extA exponent: " << normA.Exponent << std::endl;
        std::cout << "extB: " << VectorUintToHexString(extB) << std::endl;
        std::cout << "extB exponent: " << normB.Exponent << std::endl;
    }

    // --- Extended Normalization using shift indices ---
    const bool sameSign = (normA.IsNegative == normB.IsNegative);
    bool normA_isZero = false, normB_isZero = false;
    int32_t newAExponent = normA.Exponent;
    int32_t newBExponent = normB.Exponent;
    const int32_t shiftA = ExtendedNormalizeShiftIndex(extA, newAExponent, normA_isZero);
    const int32_t shiftB = ExtendedNormalizeShiftIndex(extB, newBExponent, normB_isZero);

    // --- Compute Effective Exponents ---
    const int effExpA = normA_isZero ? -100000000 : newAExponent + (SharkFloatParams::GlobalNumUint32 * 32 - 32);
    const int effExpB = normB_isZero ? -100000000 : newBExponent + (SharkFloatParams::GlobalNumUint32 * 32 - 32);

    // --- Determine which operand has larger magnitude ---
    // If effective exponents differ, use them. If equal, compare normalized digits on the fly.
    const bool AIsBiggerMagnitude = CompareMagnitudes(effExpA, effExpB, extDigits, extA, shiftA, extB, shiftB);

    const int diff = AIsBiggerMagnitude ? (effExpA - effExpB) : (effExpB - effExpA);
    int32_t outExponent = AIsBiggerMagnitude ? newAExponent : newBExponent;

    // --- Extended Arithmetic ---
    std::vector<uint32_t> extResult(extDigits, 0);
    if (sameSign) {
        // ---- Addition Branch ----
        uint64_t carry = 0;
        for (int i = 0; i < extDigits; i++) {
            uint64_t alignedA = 0, alignedB = 0;
            GetCorrespondingLimbs<SharkFloatParams>(
                extA, extB, shiftA, shiftB, AIsBiggerMagnitude, diff, i, alignedA, alignedB);
            uint64_t sum = alignedA + alignedB + carry;
            extResult[i] = (uint32_t)(sum & 0xFFFFFFFF);
            carry = sum >> 32;
        }
        if (carry > 0) {
            // Handle overflow by a 1-bit right shift of the result.
            outExponent += 1;
            std::vector<uint32_t> temp(extDigits + 1, 0);
            for (int i = 0; i < extDigits; i++) {
                temp[i] = extResult[i];
            }
            temp[extDigits] = (uint32_t)carry;
            std::vector<uint32_t> shifted(temp.size(), 0);
            for (int i = 0; i < (int)temp.size(); i++) {
                const auto sz = temp.size();
                shifted[i] = ShiftRight(temp.data(), 1, i, static_cast<int>(sz));
            }
            for (int i = 0; i < extDigits; i++) {
                extResult[i] = shifted[i];
            }
        }
        // The result sign remains the same.
    } else {
        // ---- Subtraction Branch ----
        uint64_t borrow = 0;
        if (AIsBiggerMagnitude) {
            for (int i = 0; i < extDigits; i++) {
                uint64_t alignedA = 0, alignedB = 0;
                GetCorrespondingLimbs<SharkFloatParams>(
                    extA, extB, shiftA, shiftB, AIsBiggerMagnitude, diff, i, alignedA, alignedB);
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
            for (int i = 0; i < extDigits; i++) {
                uint64_t alignedA = 0, alignedB = 0;
                GetCorrespondingLimbs<SharkFloatParams>(
                    extA, extB, shiftA, shiftB, AIsBiggerMagnitude, diff, i, alignedA, alignedB);
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
    int msdResult = 0;
    for (int i = extDigits - 1; i >= 0; i--) {
        if (extResult[i] != 0) { msdResult = i; break; }
    }
    int clzResult = CountLeadingZeros(extResult[msdResult]);
    int currentOverall = msdResult * 32 + (31 - clzResult);
    int desiredOverall = (SharkFloatParams::GlobalNumUint32 - 1) * 32 + 31;

    if constexpr (SharkFloatParams::HostVerbose) {
        std::cout << "Count leading zeros: " << clzResult << std::endl;
        std::cout << "Current MSB index: " << msdResult << std::endl;
        std::cout << "Current overall bit position: " << currentOverall << std::endl;
        std::cout << "Desired overall bit position: " << desiredOverall << std::endl;
    }

    int shiftNeeded = currentOverall - desiredOverall;
    std::vector<uint32_t> finalExt = extResult;
    if (shiftNeeded > 0) {
        if constexpr (SharkFloatParams::HostVerbose) {
            std::cout << "Shift needed branch A: " << shiftNeeded << std::endl;
        }

        std::vector<uint32_t> shifted;
        MultiWordRightShift_LittleEndian(finalExt, shiftNeeded, shifted);
        finalExt = shifted;
        outExponent += shiftNeeded;

        if constexpr (SharkFloatParams::HostVerbose) {
            std::cout << "Final extResult after right shift: " << VectorUintToHexString(finalExt) << std::endl;
            std::cout << "ShiftNeeded after right shift: " << shiftNeeded << std::endl;
            std::cout << "Final outExponent after right shift: " << outExponent << std::endl;
        }
    } else if (shiftNeeded < 0) {
        if constexpr (SharkFloatParams::HostVerbose) {
            std::cout << "Shift needed branch B: " << shiftNeeded << std::endl;
        }

        int L = -shiftNeeded;
        std::vector<uint32_t> shifted;
        MultiWordLeftShift_LittleEndian(finalExt, L, shifted);
        finalExt = shifted;
        outExponent -= L;

        if constexpr (SharkFloatParams::HostVerbose) {
            std::cout << "Final extResult after left shift: " << VectorUintToHexString(finalExt) << std::endl;
            std::cout << "L after left shift: " << L << std::endl;
            std::cout << "Final outExponent after left shift: " << outExponent << std::endl;
        }
    }
    std::vector<uint32_t> finalResult(SharkFloatParams::GlobalNumUint32, 0);
    for (int i = 0; i < SharkFloatParams::GlobalNumUint32; i++) {
        finalResult[i] = finalExt[i];
    }
    OutXY->Exponent = outExponent;
    for (int i = 0; i < SharkFloatParams::GlobalNumUint32; i++) {
        OutXY->Digits[i] = finalResult[i];
    }
    // Set the result sign.
    if (sameSign)
        OutXY->IsNegative = normA.IsNegative;
    else
        OutXY->IsNegative = AIsBiggerMagnitude ? normA.IsNegative : normB.IsNegative;

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
