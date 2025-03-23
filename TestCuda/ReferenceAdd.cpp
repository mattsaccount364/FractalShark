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

//
// Sequential AddHelper: Performs high-precision addition (or subtraction) on A and B,
// writing the result into OutXY. The debugStates vector is used to record optional debug info.
//

template<class SharkFloatParams>
void AddHelper(
    const HpSharkFloat<SharkFloatParams> *A,
    const HpSharkFloat<SharkFloatParams> *B,
    HpSharkFloat<SharkFloatParams> *OutXY,
    std::vector<DebugStateHost<SharkFloatParams>> &debugStates
) {
    const int numDigits = SharkFloatParams::GlobalNumUint32;

    if constexpr (SharkFloatParams::HostVerbose) {
        std::cout << "A: " << A->ToString() << std::endl;
        std::cout << "B: " << B->ToString() << std::endl;
    }

    HpSharkFloat<SharkFloatParams> normA = *A;
    HpSharkFloat<SharkFloatParams> normB = *B;
    //Normalize(&normA);
    //Normalize(&normB);

    // ----- Phase 1: Zero Detection -----
    bool AIsZero = true, BIsZero = true;
    for (int i = 0; i < numDigits; i++) {
        if (normA.Digits[i] != 0) { AIsZero = false; break; }
    }
    for (int i = 0; i < numDigits; i++) {
        if (normB.Digits[i] != 0) { BIsZero = false; break; }
    }

    if constexpr (SharkFloatParams::HostVerbose) {
        // Print the normalized numbers
        std::cout << "Normalized A: " << normA.ToString() << std::endl;
        std::cout << "Normalized B: " << normB.ToString() << std::endl;
        std::cout << "AIsZero: " << AIsZero << ", BIsZero: " << BIsZero << std::endl;
    }

    if (AIsZero && BIsZero) {
        // Both numbers are zero.
        for (int i = 0; i < numDigits; i++) {
            OutXY->Digits[i] = 0;
        }
        OutXY->Exponent = 0;
        OutXY->IsNegative = false;
        return;
    } else if (AIsZero) {
        // A is zero; copy B.
        for (int i = 0; i < numDigits; i++) {
            OutXY->Digits[i] = normB.Digits[i];
        }
        OutXY->Exponent = normB.Exponent;
        OutXY->IsNegative = normB.IsNegative;
        return;
    } else if (BIsZero) {
        // B is zero; copy A.
        for (int i = 0; i < numDigits; i++) {
            OutXY->Digits[i] = normA.Digits[i];
        }
        OutXY->Exponent = normA.Exponent;
        OutXY->IsNegative = normA.IsNegative;
        return;
    }

    // ----- Phase 1: Exponent Alignment -----
    int32_t expDiff = normA.Exponent - normB.Exponent;
    bool AIsBiggerExponent = (expDiff >= 0);
    // When the sign bits match we perform addition; otherwise subtraction.
    bool isAddition = (normA.IsNegative == B->IsNegative);
    int32_t shiftBits = (expDiff < 0) ? -expDiff : expDiff;
    const int totalBits = numDigits * 32;
    int shiftAmount = 0;
    int32_t outExponent = 0;

    if (shiftBits >= totalBits) {
        // When the exponent difference exceeds the total available bits,
        // we shift the number with the larger exponent left.
        shiftAmount = shiftBits - totalBits + 1;
        outExponent = AIsBiggerExponent ? normB.Exponent : normA.Exponent;
    } else {
        // Normal case: shift the number with the smaller exponent right.
        shiftAmount = 0;
        outExponent = AIsBiggerExponent ? normA.Exponent : normB.Exponent;
    }

    if constexpr (SharkFloatParams::HostVerbose) {
        std::cout << "Exponent difference: " << std::dec << expDiff << std::endl;
        std::cout << "Shift bits: " << std::dec << shiftBits;
        std::cout << ", Shift amount: " << std::dec << shiftAmount << std::endl;
        std::cout << "Out exponent: " << std::dec << outExponent << std::endl;
        
        // Print true or false for isAddition
        std::cout << "isAddition: " << (isAddition ? "true" : "false") << std::endl;
    }

    // Prepare aligned digits for normA and normB.
    std::vector<uint32_t> alignedA(numDigits, 0);
    std::vector<uint32_t> alignedB(numDigits, 0);

    if (shiftAmount > 0) {
        // One operand must be shifted left.
        if (AIsBiggerExponent) {
            // Shift normA left.
            for (int i = 0; i < numDigits; i++) {
                alignedA[i] = ShiftLeft(normA.Digits, shiftAmount, i, numDigits);
                alignedB[i] = normB.Digits[i];
            }
        } else {
            // Shift normB left.
            for (int i = 0; i < numDigits; i++) {
                alignedA[i] = normA.Digits[i];
                alignedB[i] = ShiftLeft(normB.Digits, shiftAmount, i, numDigits);
            }
        }

        outExponent += shiftAmount;
    } else {
        // Normal case: shift the number with the smaller exponent right.
        if (AIsBiggerExponent) {
            // Shift normB right.
            for (int i = 0; i < numDigits; i++) {
                alignedA[i] = normA.Digits[i];
                alignedB[i] = ShiftRight(normB.Digits, shiftBits, i, numDigits);
            }
        } else {
            // Shift normA right.
            for (int i = 0; i < numDigits; i++) {
                alignedA[i] = ShiftRight(normA.Digits, shiftBits, i, numDigits);
                alignedB[i] = normB.Digits[i];
            }
        }
    }

    if constexpr (SharkFloatParams::HostVerbose) {
        // Print aligned arrays using hex digits
        std::cout << "Aligned normA: ";
        for (int i = 0; i < numDigits; i++) {
            std::cout << std::hex << "0x" << alignedA[i] << " ";
        }

        std::cout << "\nAligned normB: ";
        for (int i = 0; i < numDigits; i++) {
            std::cout << std::hex << "0x" << alignedB[i] << " ";
        }
    }

    // ----- Phase 2: Compare Magnitudes for Subtraction -----
    bool AIsBiggerMagnitude = true; // Default assumption.
    if (!isAddition) {
        for (int idx = numDigits - 1; idx >= 0; idx--) {
            if (alignedA[idx] > alignedB[idx]) {
                AIsBiggerMagnitude = true;
                break;
            } else if (alignedA[idx] < alignedB[idx]) {
                AIsBiggerMagnitude = false;
                break;
            }
        }
    }

    if constexpr (SharkFloatParams::HostVerbose) {
        std::cout << "\nA is bigger magnitude: " << AIsBiggerMagnitude << std::endl;
    }

    // ----- Phase 3: Perform Digit-wise Addition/Subtraction -----
    std::vector<uint32_t> resultDigits(numDigits, 0);

    if (isAddition) {
        // Perform addition with carry propagation.
        uint64_t carry = 0;
        for (int i = 0; i < numDigits; i++) {
            uint64_t sum = (uint64_t)alignedA[i] + (uint64_t)alignedB[i] + carry;
            resultDigits[i] = (uint32_t)(sum & 0xFFFFFFFF);
            carry = sum >> 32;
        }

        if constexpr (SharkFloatParams::HostVerbose) {
            std::cout << "\nResult digits after addition: ";
            for (int i = 0; i < numDigits; i++) {
                std::cout << std::hex << resultDigits[i] << " ";
            }
        }

        // ----- Phase 4: Handle Final Carry-Out (Digit Shifting) -----
        if (carry > 0) {

            if constexpr (SharkFloatParams::HostVerbose) {
                std::cout << "\nCarry out detected: " << carry << std::endl;
            }

            // If there is a carry out, increment the exponent and shift the result right by one bit.
            outExponent += 1;
            uint32_t newCarry = (uint32_t)carry; // Typically 1.
            for (int i = numDigits - 1; i >= 0; i--) {
                uint32_t lsb = resultDigits[i] & 1; // Save LSB to propagate.
                resultDigits[i] = (resultDigits[i] >> 1) | (newCarry << 31);
                newCarry = lsb;
            }
        }
    } else {
        // Perform subtraction. Subtract the smaller magnitude from the larger.
        uint64_t borrow = 0;
        if (AIsBiggerMagnitude) {
            for (int i = 0; i < numDigits; i++) {
                int64_t diff = (int64_t)alignedA[i] - (int64_t)alignedB[i] - borrow;
                if (diff < 0) {
                    diff += ((int64_t)1 << 32);
                    borrow = 1;
                } else {
                    borrow = 0;
                }
                resultDigits[i] = (uint32_t)(diff & 0xFFFFFFFF);
            }
        } else {
            for (int i = 0; i < numDigits; i++) {
                int64_t diff = (int64_t)alignedB[i] - (int64_t)alignedA[i] - borrow;
                if (diff < 0) {
                    diff += ((int64_t)1 << 32);
                    borrow = 1;
                } else {
                    borrow = 0;
                }
                resultDigits[i] = (uint32_t)(diff & 0xFFFFFFFF);
            }
        }

        if constexpr (SharkFloatParams::HostVerbose) {
            std::cout << "\nResult digits after subtraction: ";
            for (int i = 0; i < numDigits; i++) {
                std::cout << std::hex << resultDigits[i] << " ";
            }
        }

        // (In a well-formed subtraction the final borrow should be zero.)
        assert(borrow == 0 && "Final borrow in subtraction should be zero");
    }

    // ----- Finalize Output -----
    for (int i = 0; i < numDigits; i++) {
        OutXY->Digits[i] = resultDigits[i];
    }
    OutXY->Exponent = outExponent;
    if (isAddition) {
        OutXY->IsNegative = normA.IsNegative; // Both had the same sign.
    } else {
        // For subtraction, the sign is that of the number with the larger magnitude.
        OutXY->IsNegative = (AIsBiggerMagnitude ? normA.IsNegative : normB.IsNegative);
    }

    // Optionally, store some debugging state.
    DebugStateHost<SharkFloatParams> state;
    // (Fill state with any desired intermediate values for debugging.)
    debugStates.push_back(state);
}

#define ExplicitlyInstantiate(SharkFloatParams) \
    template void AddHelper<SharkFloatParams>( \
        const HpSharkFloat<SharkFloatParams> *, \
        const HpSharkFloat<SharkFloatParams> *, \
        HpSharkFloat<SharkFloatParams> *, \
        std::vector<DebugStateHost<SharkFloatParams>> &debugStates);

ExplicitInstantiateAll();