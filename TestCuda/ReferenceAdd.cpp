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

//
// Helper functions to perform bit shifts on a fixed-width digit array.
// They mirror the CUDA device functions but work sequentially on the full array.
//

// ShiftRight: Shifts the number (given by its digit array) right by shiftBits.
// idx is the index of the digit to compute. The parameter numDigits prevents out-of-bounds access.
inline uint32_t ShiftRight(const uint32_t *digits, int shiftBits, int idx, int numDigits) {
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
inline uint32_t ShiftLeft(const uint32_t *digits, int shiftBits, int idx, int /*numDigits*/) {
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
inline int CountLeadingZeros(uint32_t x) {
    int count = 0;
    for (int bit = 31; bit >= 0; --bit) {
        if (x & (1u << bit))
            break;
        ++count;

    }
    return count;
}

//---------------------------------------------------------------------
// Multi-word shifts for extended (little-endian) arrays.
// We'll implement multi-word right shift in little endian order
// by applying our ShiftRight() to each word.
// (We assume that our provided ShiftRight already works for little endian.)
//---------------------------------------------------------------------

// MultiWordRightShift_LittleEndian: shift an array 'in' of length 'n' right by L bits,
// storing the result in 'out'. (out and in may be distinct.)
inline void MultiWordRightShift_LittleEndian(const std::vector<uint32_t> &in, int L, std::vector<uint32_t> &out) {
    int n = in.size();
    out.resize(n, 0);
    for (int i = 0; i < n; i++) {
        out[i] = ShiftRight(in.data(), L, i, n);
    }
}

// MultiWordLeftShift_LittleEndian: shift an array 'in' of length 'n' left by L bits,
// storing the result in 'out'.
inline void MultiWordLeftShift_LittleEndian(const std::vector<uint32_t> &in, int L, std::vector<uint32_t> &out) {
    int n = in.size();
    out.resize(n, 0);
    for (int i = 0; i < n; i++) {
        out[i] = ShiftLeft(in.data(), L, i, n);
    }
}

//---------------------------------------------------------------------
// Extended Normalization in little-endian order.
// We assume the nominal representation has GlobalNumUint32 words (LSB at index 0)
// and we work in an extended array of length extDigits.
// We copy the nominal digits into the lower indices (0 .. GlobalNumUint32-1) and guard words go in higher indices.
// Then we shift the extended number left so that the most-significant set bit is moved to bit position: (totalExtBits - 1).
// Since the array is little-endian, the most-significant word is at index (extDigits - 1).
// The overall bit-index of a bit in word i is: i*32 + bitIndex, where bitIndex goes from 0 (LSB) to 31 (MSB).
//---------------------------------------------------------------------
void ExtendedNormalize(std::vector<uint32_t> &ext, int &storedExp) {
    const int n = ext.size();
    int msd = 0;
    // In little endian, the most-significant nonzero word is the highest index.
    for (int i = n - 1; i >= 0; i--) {
        if (ext[i] != 0) { msd = i; break; }
    }
    int clz = CountLeadingZeros(ext[msd]);
    // The most significant set bit is at bit position (31 - clz) in word msd.
    int current_msb = msd * 32 + (31 - clz);
    const int totalExtBits = n * 32;
    int L = (totalExtBits - 1) - current_msb; // how many bits to shift left
    if (L > 0) {
        std::vector<uint32_t> shifted;
        MultiWordLeftShift_LittleEndian(ext, L, shifted);
        ext = shifted;
        storedExp -= L;
    }
}

//---------------------------------------------------------------------
// Final normalization: convert the extended result (little endian)
// into a nominal result with GlobalNumUint32 words, by re-normalizing.
// In little endian, the nominal representation's most significant word is at index (GlobalNumUint32 - 1).
// We find the most significant set bit in the extended result and compute how many bits
// we need to shift right (if the number is "too wide") or left (if it's "too narrow")
// so that the most significant set bit ends up at bit position: (GlobalNumUint32*32 - 1).
// Then we adjust the exponent accordingly.
//---------------------------------------------------------------------
template <class SharkFloatParams>
void FinalNormalizeExtendedResult(const std::vector<uint32_t> &extResult, int &outExponent, std::vector<uint32_t> &finalResult) {
    const int nominalWords = SharkFloatParams::GlobalNumUint32;
    const int extDigits = extResult.size();
    // In little endian, find the index of the highest nonzero word.
    int msd = 0;
    for (int i = extDigits - 1; i >= 0; i--) {
        if (extResult[i] != 0) { msd = i; break; }
    }
    int clz = CountLeadingZeros(extResult[msd]);
    int currentOverall = msd * 32 + (31 - clz);
    // Desired overall bit position for the nominal result's MSB:
    int desired = (nominalWords - 1) * 32 + 31;
    int shiftNeeded = currentOverall - desired;
    // Copy extResult to a working vector.
    std::vector<uint32_t> temp = extResult;
    if (shiftNeeded > 0) {
        // Need to shift right by shiftNeeded bits.
        std::vector<uint32_t> shifted;
        MultiWordRightShift_LittleEndian(temp, shiftNeeded, shifted);
        temp = shifted;
        outExponent += shiftNeeded;
    } else if (shiftNeeded < 0) {
        // Need to shift left by -shiftNeeded bits.
        int L = -shiftNeeded;
        std::vector<uint32_t> shifted;
        MultiWordLeftShift_LittleEndian(temp, L, shifted);
        temp = shifted;
        outExponent -= L;
    }
    // Now, we want the nominal result to be in the lower nominalWords digits.
    // In little endian, that means indices 0..(nominalWords - 1).
    finalResult.resize(nominalWords);
    for (int i = 0; i < nominalWords; i++) {
        finalResult[i] = temp[i];
    }
}


//---------------------------------------------------------------------
// Extended arithmetic using little-endian representation.
// This function uses extended working precision (with guard words) to perform
// addition/subtraction with correct alignment and sign handling.
//---------------------------------------------------------------------
template<class SharkFloatParams>
void AddHelper(
    const HpSharkFloat<SharkFloatParams> *A,
    const HpSharkFloat<SharkFloatParams> *B,
    HpSharkFloat<SharkFloatParams> *OutXY,
    std::vector<DebugStateHost<SharkFloatParams>> &debugStates
) {
    // Make local copies.
    auto normA = *A;
    auto normB = *B;

    // --- Set up extended working precision ---
    const int guard = 2;
    const int extDigits = SharkFloatParams::GlobalNumUint32 + guard;
    const int totalExtBits = extDigits * 32;
    // Create extended arrays (little endian, index 0 is LSB).
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

    // --- Extended Normalization ---
    ExtendedNormalize(extA, normA.Exponent);
    ExtendedNormalize(extB, normB.Exponent);
    if constexpr (SharkFloatParams::HostVerbose) {
        std::cout << "extA after normalization: " << VectorUintToHexString(extA) << std::endl;
        std::cout << "normA exponent after normalization: " << normA.Exponent << std::endl;
        std::cout << "extB after normalization: " << VectorUintToHexString(extB) << std::endl;
        std::cout << "normB exponent after normalization: " << normB.Exponent << std::endl;
    }

    // --- Compute Effective Exponents ---
    // For a normalized little-endian number, assume the nominal representation has
    // GlobalNumUint32 words and its most significant word is at index (GlobalNumUint32 - 1).
    // Then if nonzero:
    //    effectiveExp = storedExponent + (GlobalNumUint32*32 - 32)
    // (If a number is zero we should assign it a very low effective exponent.)
    int effExpA, effExpB;
    auto isZeroNominal = [&](const HpSharkFloat<SharkFloatParams> &num) -> bool {
        for (int i = 0; i < SharkFloatParams::GlobalNumUint32; i++) {
            if (num.Digits[i] != 0)
                return false;
        }
        return true;
        };

    if (isZeroNominal(normA))
        effExpA = -100000000;
    else
        effExpA = normA.Exponent + (SharkFloatParams::GlobalNumUint32 * 32 - 32);

    if (isZeroNominal(normB))
        effExpB = -100000000;
    else
        effExpB = normB.Exponent + (SharkFloatParams::GlobalNumUint32 * 32 - 32);

    // --- Determine which operand has larger magnitude ---
    bool AIsBiggerMagnitude;
    if (effExpA > effExpB) {
        AIsBiggerMagnitude = true;
    } else if (effExpA < effExpB) {
        AIsBiggerMagnitude = false;
    } else {
        // Effective exponents are equal: compare extended normalized arrays.
        // extA and extB have been produced by ExtendedNormalize.
        AIsBiggerMagnitude = false; // default: they are equal
        for (int i = extDigits - 1; i >= 0; i--) {
            if (extA[i] > extB[i]) {
                AIsBiggerMagnitude = true;
                break;
            } else if (extA[i] < extB[i]) {
                AIsBiggerMagnitude = false;
                break;
            }
        }
    }
    int diff = AIsBiggerMagnitude ? (effExpA - effExpB) : (effExpB - effExpA);
    int32_t outExponent = AIsBiggerMagnitude ? normA.Exponent : normB.Exponent;


    // --- Determine operation: addition if same sign; otherwise subtraction.
    bool sameSign = (normA.IsNegative == normB.IsNegative);

    // --- Alignment ---
    // In little-endian representation, we align by shifting the smaller operand right by 'diff' bits.
    std::vector<uint32_t> alignedA = extA;
    std::vector<uint32_t> alignedB(extDigits, 0);
    if (AIsBiggerMagnitude) {
        for (int i = 0; i < extDigits; i++) {
            alignedB[i] = ShiftRight(extB.data(), diff, i, extDigits);
        }
    } else {
        for (int i = 0; i < extDigits; i++) {
            alignedA[i] = ShiftRight(extA.data(), diff, i, extDigits);
        }
        alignedB = extB;
    }
    if constexpr (SharkFloatParams::HostVerbose) {
        std::cout << "alignedA: " << VectorUintToHexString(alignedA) << std::endl;
        std::cout << "alignedB: " << VectorUintToHexString(alignedB) << std::endl;
    }

    // --- Extended Arithmetic ---
    std::vector<uint32_t> extResult(extDigits, 0);
    if (sameSign) {
        // ---- Addition Branch ----
        uint64_t carry = 0;
        // Addition: process from LSB (index 0) upward.
        for (int i = 0; i < extDigits; i++) {
            uint64_t sum = (uint64_t)alignedA[i] + (uint64_t)alignedB[i] + carry;
            extResult[i] = (uint32_t)(sum & 0xFFFFFFFF);
            carry = sum >> 32;
        }
        if (carry > 0) {
            // Instead of a simple right-shift loop, allocate a temporary array with one extra word.
            outExponent += 1;
            std::vector<uint32_t> temp(extDigits + 1, 0);
            // Copy extResult into temp (indices 0..extDigits-1).
            for (int i = 0; i < extDigits; i++) {
                temp[i] = extResult[i];
            }
            // Set the extra (most significant) word to the carry.
            temp[extDigits] = (uint32_t)carry;
            // Now perform a 1-bit right shift over the entire (extDigits+1)-word array.
            std::vector<uint32_t> shifted(temp.size(), 0);
            for (int i = 0; i < (int)temp.size(); i++) {
                shifted[i] = ShiftRight(temp.data(), 1, i, temp.size());
            }
            // Drop the extra word (i.e. take indices 0..extDigits-1) as the new extResult.
            for (int i = 0; i < extDigits; i++) {
                extResult[i] = shifted[i];
            }
        }
        // The result sign remains the same.
    } else {
        uint64_t borrow = 0;
        if (AIsBiggerMagnitude) {
            // Compute extResult = alignedA - alignedB.
            for (int i = 0; i < extDigits; i++) {
                int64_t diffVal = (int64_t)alignedA[i] - (int64_t)alignedB[i] - borrow;
                if (diffVal < 0) {
                    diffVal += ((uint64_t)1 << 32);
                    borrow = 1;
                } else {
                    borrow = 0;
                }
                extResult[i] = (uint32_t)(diffVal & 0xFFFFFFFF);
            }
        } else {
            // Compute extResult = alignedB - alignedA.
            for (int i = 0; i < extDigits; i++) {
                int64_t diffVal = (int64_t)alignedB[i] - (int64_t)alignedA[i] - borrow;
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
    }

    // --- Final Normalization: re-normalize the extended result into nominal precision.
    // In little-endian, the nominal representation consists of GlobalNumUint32 words,
    // with index 0 = LSB and index (GlobalNumUint32 - 1) = MSB.
    // Find the overall bit index of the most significant set bit in extResult.
    int msdResult = 0;
    for (int i = extDigits - 1; i >= 0; i--) {
        if (extResult[i] != 0) { msdResult = i; break; }
    }
    int clzResult = CountLeadingZeros(extResult[msdResult]);
    int currentOverall = msdResult * 32 + (31 - clzResult);
    int desiredOverall = (SharkFloatParams::GlobalNumUint32 - 1) * 32 + 31;
    int shiftNeeded = currentOverall - desiredOverall;
    std::vector<uint32_t> finalExt = extResult;
    if (shiftNeeded > 0) {
        // Shift right by shiftNeeded bits.
        std::vector<uint32_t> shifted;
        MultiWordRightShift_LittleEndian(finalExt, shiftNeeded, shifted);
        finalExt = shifted;
        outExponent += shiftNeeded;
    } else if (shiftNeeded < 0) {
        int L = -shiftNeeded;
        std::vector<uint32_t> shifted;
        MultiWordLeftShift_LittleEndian(finalExt, L, shifted);
        finalExt = shifted;
        outExponent -= L;
    }
    // Now, extract the nominal GlobalNumUint32 words.
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


#define ExplicitlyInstantiate(SharkFloatParams) \
    template void AddHelper<SharkFloatParams>( \
        const HpSharkFloat<SharkFloatParams> *, \
        const HpSharkFloat<SharkFloatParams> *, \
        HpSharkFloat<SharkFloatParams> *, \
        std::vector<DebugStateHost<SharkFloatParams>> &debugStates);

ExplicitInstantiateAll();