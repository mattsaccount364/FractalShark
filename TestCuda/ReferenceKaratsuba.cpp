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

void Add128 (
    uint64_t a_low, uint64_t a_high,
    uint64_t b_low, uint64_t b_high,
    uint64_t &result_low, uint64_t &result_high) {

    result_low = a_low + b_low;
    uint64_t carry = (result_low < a_low) ? 1 : 0;
    result_high = a_high + b_high + carry;
}

void Subtract128 (
    uint64_t a_low, uint64_t a_high,
    uint64_t b_low, uint64_t b_high,
    uint64_t &result_low, uint64_t &result_high) {

    uint64_t borrow = 0;

    // Subtract low parts
    result_low = a_low - b_low;
    borrow = (a_low < b_low) ? 1 : 0;
    //result_low += (borrow << 32);

    // Subtract high parts with borrow
    result_high = a_high - b_high - borrow;
}

static void NativeMultiply64 (
    int n,
    int total_k,
    const std::vector<uint32_t> &A,
    const std::vector<uint32_t> &B,
    std::vector<uint64_t> &Res) {

    assert(n == A.size());
    assert(n == B.size());

    // Number of partial sums = (2*n - 1).
    // Each partial sum is stored as two 64-bit words in Res.
    // int total_k = 2 * n - 1;
    // int total_k = 2 * A.size() - 1;

    // For each partial sum index k
    for (int k = 0; k < total_k; k++) {
        // We'll keep a full 128-bit partial sum in sum_128_{low,high}.
        uint64_t sum_128_low = 0ULL;
        uint64_t sum_128_high = 0ULL;

        // Determine which A[i] and B[k - i] to multiply
        int i_start = (k < n) ? 0 : (k - (n - 1));
        int i_end = (k < n) ? k : (n - 1);

        for (int i = i_start; i <= i_end; ++i) {
            uint64_t a = A[i];
            uint64_t b = B[k - i];

            // 64-bit product
            uint64_t product = a * b;

            // Add product to the 128-bit accumulator
            // sum_128 = sum_128 + product (treating product as 64-bit low, 0 high)
            uint64_t res_low, res_high;
            Add128(sum_128_low, sum_128_high, product, 0ULL, res_low, res_high);

            sum_128_low = res_low;
            sum_128_high = res_high;
        }

        uint64_t sum_low = sum_128_low;
        uint64_t sum_high = sum_128_high;

        // Store in Res
        // - Res[k*2]   gets the "low 64 bits" slot, but effectively only the low 32 bits are meaningful.
        // - Res[k*2+1] gets the upper portion, which can be up to 64 bits.
        const auto idx = k * 2;
        Res[idx] = sum_low;
        Res[idx + 1] = sum_high;
    }
}

// Helper to add two 64-bit values plus a carry-in, produce sum and carry-out
static inline void add64withCarry (uint64_t x, uint64_t y, uint64_t carry_in,
    uint64_t &sum, uint64_t &carry_out) {
    uint64_t low = x + carry_in;
    carry_out = (low < x) ? 1ULL : 0ULL;
    uint64_t old = low;
    low += y;
    if (low < old) carry_out += 1ULL;
    sum = low;
}

template<class SharkFloatParams>
int32_t
CompareDigits (const std::vector<uint32_t> &highArray, const std::vector<uint32_t> &lowArray) {
    // The biggest possible “digit index” is one less
    // than the max of the two sizes.
    int maxLen = static_cast<int>(std::max(highArray.size(), lowArray.size()));

    // Compare top-down, from maxLen-1 down to 0
    for (int i = maxLen - 1; i >= 0; --i) {
        // Treat out-of-range as zero
        uint32_t a_val = (i < static_cast<int>(highArray.size())) ? highArray[i] : 0u;
        uint32_t b_val = (i < static_cast<int>(lowArray.size())) ? lowArray[i] : 0u;

        if (a_val > b_val) {
            return 1;  // A is bigger
        } else if (a_val < b_val) {
            return -1; // B is bigger
        }
    }

    // They are exactly equal
    if constexpr (SharkFloatParams::HostVerbose) {
        std::cout << "CompareDigits: Result: 0" << std::endl;
    }
    return 0;
}

template<class SharkFloatParams>
uint32_t
SubtractDigitsSerial (
    const std::vector<uint32_t> &A_,
    const std::vector<uint32_t> &B_,
    std::vector<uint32_t> &Res) {

    const auto n1 = static_cast<int>(A_.size());
    const auto n2 = static_cast<int>(B_.size());
    const auto maxN = std::max(n1, n2);

    if constexpr (SharkFloatParams::HostVerbose) {
        std::cout << "SubtractDigitsSerial Input: A: " << VectorUintToHexString(A_) << std::endl;
        std::cout << "SubtractDigitsSerial Input: B: " << VectorUintToHexString(B_) << std::endl;
    }

    uint64_t borrow = 0;
    assert(maxN == Res.size());

    for (int i = 0; i < maxN; i++) {
        uint64_t a_val;
        if (i >= n1) {
            a_val = 0;
        } else {
            a_val = A_[i];
        }

        uint64_t b_val;
        
        if (i >= n2) {
            b_val = 0;
        } else {
            b_val = B_[i];
        }

        uint64_t diff = a_val - b_val - borrow;
        if (a_val < (b_val + borrow)) {
            diff += (1ULL << 32);
            borrow = 1;
        } else {
            borrow = 0;
        }
        Res[i] = ((uint32_t)diff);
    }
    // Assuming highArray >= lowArray, no final borrow remains.

    if constexpr (SharkFloatParams::HostVerbose) {
        std::cout << "SubtractDigitsSerial: Result: " << VectorUintToHexString(Res) << std::endl;
    }

    assert(borrow == 0);

    return (uint32_t)borrow;
}

template<
    class SharkFloatParams,
    int RecursionDepth,
    int CallIndex,
    DebugStatePurpose Purpose,
    typename ArrayType>
const DebugStateHost<SharkFloatParams> &
GetCurrentDebugState (
    std::vector<DebugStateHost<SharkFloatParams>> &debugStates,
    UseConvolution useConvolution,
    const ArrayType *arrayToChecksum,
    size_t arraySize) {

    constexpr auto maxPurposes = static_cast<int>(DebugStatePurpose::NumPurposes);
    constexpr auto curPurpose = static_cast<int>(Purpose);
    const auto index = CallIndex * maxPurposes + curPurpose;

    auto &retval = debugStates[index];
    retval.Reset(
        arrayToChecksum, arraySize, Purpose, RecursionDepth, CallIndex, useConvolution);
    return retval;
}

template<
    class SharkFloatParams,
    int NewNumBlocks,
    int RecursionDepth,
    int CallIndex>
void KaratsubaRecursiveDigits (
    const std::vector<uint32_t> &A_digits,  // pointer to A's digits
    const std::vector<uint32_t> &B_digits,
    std::vector<uint64_t> &final128XX,
    std::vector<uint64_t> &final128XY,
    std::vector<uint64_t> &final128YY,
    std::vector<DebugStateHost<SharkFloatParams>> &debugStates
)
{
    const int NewN = static_cast<int>(A_digits.size());

    {
        [[maybe_unused]] const int fullBDigits = static_cast<int>(B_digits.size());
        assert(NewN == fullBDigits);
    }

    const int NewN1 = (NewN + 1) / 2;
    const int NewN2 = NewN / 2;

    using DebugState = DebugStateHost<SharkFloatParams>;

    constexpr bool UseConvolutionBool =
        (NewNumBlocks <= std::max(SharkFloatParams::GlobalNumBlocks / SharkFloatParams::ConvolutionLimit, 1) ||
            (NewNumBlocks % 3 != 0));
    constexpr auto UseConvolutionHere = UseConvolutionBool ? UseConvolution::Yes : UseConvolution::No;

    constexpr auto NewSize = (CallIndex + 1) * static_cast<int>(DebugStatePurpose::NumPurposes);
    if (NewSize > debugStates.size()) {
        debugStates.resize(NewSize);
    }

    const auto &debugAState = GetCurrentDebugState<SharkFloatParams, RecursionDepth, CallIndex, DebugStatePurpose::ADigits>(
        debugStates, UseConvolutionHere, A_digits.data(), A_digits.size());
    const auto &debugBState = GetCurrentDebugState<SharkFloatParams, RecursionDepth, CallIndex, DebugStatePurpose::BDigits>(
        debugStates, UseConvolutionHere, B_digits.data(), B_digits.size());

    if constexpr (SharkFloatParams::HostVerbose) {
        std::cout << "KaratsubaRecursiveDigits (index: " << CallIndex << "):" << std::endl;
        std::cout << "NewN = " << NewN << std::endl;
        std::cout << "A->Digits checksum: " << debugAState.GetStr() << std::endl;
        std::cout << "B->Digits checksum: " << debugBState.GetStr() << std::endl;
        std::cout << VectorUintToHexString(A_digits) << " * ";
        std::cout << VectorUintToHexString(B_digits) << std::endl;
    }

    std::vector<uint32_t> A_low;
    for (size_t i = 0; i < NewN1; i++) {
        A_low.push_back(A_digits[i]);
    }

    assert(A_low.size() == NewN1);

    std::vector<uint32_t> A_high;
    for (size_t i = NewN1; i < NewN; i++) {
        A_high.push_back(A_digits[i]);
    }

    assert(A_high.size() == NewN2);

    std::vector<uint32_t> B_low;
    for (size_t i = 0; i < NewN1; i++) {
        B_low.push_back(B_digits[i]);
    }

    assert(B_low.size() == NewN1);

    std::vector<uint32_t> B_high;
    for (size_t i = NewN1; i < NewN; i++) {
        B_high.push_back(B_digits[i]);
    }

    assert(B_high.size() == NewN2);

    // Print lengths of A_low, A_high, B_low, B_high
    if constexpr (SharkFloatParams::HostVerbose) {
        std::cout << "A_low: " << A_low.size() << std::endl;
        std::cout << "A_high: " << A_high.size() << std::endl;
        std::cout << "B_low: " << B_low.size() << std::endl;
        std::cout << "B_high: " << B_high.size() << std::endl;
    }

    GetCurrentDebugState<SharkFloatParams, RecursionDepth, CallIndex, DebugStatePurpose::AHalfHigh>(
            debugStates, UseConvolutionHere, A_high.data(), A_high.size());
    GetCurrentDebugState<SharkFloatParams, RecursionDepth, CallIndex, DebugStatePurpose::AHalfLow>(
            debugStates, UseConvolutionHere, A_low.data(), A_low.size());
    GetCurrentDebugState<SharkFloatParams, RecursionDepth, CallIndex, DebugStatePurpose::BHalfHigh>(
            debugStates, UseConvolutionHere, B_high.data(), B_high.size());
    GetCurrentDebugState<SharkFloatParams, RecursionDepth, CallIndex, DebugStatePurpose::BHalfLow>(
            debugStates, UseConvolutionHere, B_low.data(), B_low.size());

    const auto initValue64 = 0xCDCDCDCDCDCDCDCDULL;
    const auto initValue32 = 0xCDCDCDCDU;
    //const auto initValue64 = 0;
    //const auto initValue32 = 0;

    int x_cmp = CompareDigits<SharkFloatParams>(A_high, A_low);
    bool x_diff_neg = false;
    std::vector<uint32_t> x_diff;
    x_diff.resize(NewN1, initValue32);

    if (x_cmp >= 0) {
        if constexpr (SharkFloatParams::HostVerbose) {
            std::cout << "A_high - A_low" << std::endl;
        }

        SubtractDigitsSerial<SharkFloatParams>(A_high, A_low, x_diff);
    } else {
        if constexpr (SharkFloatParams::HostVerbose) {
            std::cout << "A_low - A_high" << std::endl;
        }

        SubtractDigitsSerial<SharkFloatParams>(A_low, A_high, x_diff);
        x_diff_neg = true;
    }

    int y_cmp = CompareDigits<SharkFloatParams>(B_high, B_low);
    bool y_diff_neg = false;
    std::vector<uint32_t> y_diff;
    y_diff.resize(NewN1, initValue32);

    if (y_cmp >= 0) {
        if constexpr (SharkFloatParams::HostVerbose) {
            std::cout << "B_high - B_low" << std::endl;
        }

        SubtractDigitsSerial<SharkFloatParams>(B_high, B_low, y_diff);
    } else {
        if constexpr (SharkFloatParams::HostVerbose) {
            std::cout << "B_low - B_high" << std::endl;
        }

        SubtractDigitsSerial<SharkFloatParams>(B_low, B_high, y_diff);
        y_diff_neg = true;
    }

    const auto &xDiffChecksum =
        GetCurrentDebugState<SharkFloatParams, RecursionDepth, CallIndex, DebugStatePurpose::XDiff>(
            debugStates, UseConvolutionHere, x_diff.data(), x_diff.size());
    const auto &yDiffChecksum =
        GetCurrentDebugState<SharkFloatParams, RecursionDepth, CallIndex, DebugStatePurpose::YDiff>(
            debugStates, UseConvolutionHere, y_diff.data(), y_diff.size());

    assert(x_diff.size() == y_diff.size());
    assert(x_diff.size() == NewN1);
    assert(y_diff.size() == NewN1);
    const auto MaxHalfN = std::max(NewN2, NewN1);
    const auto total_k = 2 * MaxHalfN - 1;
    assert(x_diff.size() == MaxHalfN);
    assert(y_diff.size() == MaxHalfN);

    if constexpr (SharkFloatParams::HostVerbose) {
        std::cout << "x_diff: " << VectorUintToHexString(x_diff) << std::endl;
        std::cout << "x_diff checksum: " << xDiffChecksum.GetStr() << std::endl;
        std::cout << "y_diff: " << VectorUintToHexString(y_diff) << std::endl;
        std::cout << "y_diff checksum: " << yDiffChecksum.GetStr() << std::endl;
    }

    assert(A_low.size() == B_low.size());
    const auto FinalZ0Size =
        (UseConvolutionHere == UseConvolution::Yes) ?
        (total_k * 2) :  // (2 * A_low.size() - 1) * 2
        (2 * 2 * A_low.size());
    std::vector<uint64_t> Z0XX(FinalZ0Size, initValue64);
    std::vector<uint64_t> Z0XY(FinalZ0Size, initValue64);
    std::vector<uint64_t> Z0YY(FinalZ0Size, initValue64);

    assert(A_high.size() == B_high.size());
    const auto FinalZ2Size =
        (UseConvolutionHere == UseConvolution::Yes) ?
        total_k * 2 : // (2 * A_high.size() - 1) * 2
        (2 * 2 * A_high.size());
    std::vector<uint64_t> Z2XX(FinalZ2Size, initValue64);
    std::vector<uint64_t> Z2XY(FinalZ2Size, initValue64);
    std::vector<uint64_t> Z2YY(FinalZ2Size, initValue64);

    assert(x_diff.size() == y_diff.size());
    const auto FinalZ1TempSize = 
        (UseConvolutionHere == UseConvolution::Yes) ?
        total_k * 2 : // (2 * MaxHalfN - 1) * 2
        (2 * 2 * x_diff.size());
    std::vector<uint64_t> Z1_tempXX(FinalZ1TempSize, initValue64);
    std::vector<uint64_t> Z1_tempXY(FinalZ1TempSize, initValue64);
    std::vector<uint64_t> Z1_tempYY(FinalZ1TempSize, initValue64);

    if constexpr (UseConvolutionHere == UseConvolution::Yes) {

        assert(A_low.size() == B_low.size());

        NativeMultiply64(NewN1, total_k, A_low, A_low, Z0XX);
        NativeMultiply64(NewN1, total_k, A_low, B_low, Z0XY);
        NativeMultiply64(NewN1, total_k, B_low, B_low, Z0YY);


        auto ProcessOneZ0 = [&]<auto DebugPurpose>(
            [[maybe_unused]] const char *name,
            const std::vector<uint64_t> &Z0) {

            const auto &Z0Checksum =
                GetCurrentDebugState<SharkFloatParams, RecursionDepth, CallIndex, DebugPurpose>(
                    debugStates, UseConvolutionHere, Z0.data(), Z0.size());

            if constexpr (SharkFloatParams::HostVerbose) {
                std::cout << name << ": " << VectorUintToHexString(Z0) << std::endl;
                std::cout << name << " checksum: " << Z0Checksum.GetStr() << std::endl;
            }
            };

        ProcessOneZ0.template operator()<DebugStatePurpose::Z0XX>(
            "Z0XX", Z0XX);
        ProcessOneZ0.template operator()<DebugStatePurpose::Z0XY>(
            "Z0XY", Z0XY);
        ProcessOneZ0.template operator()<DebugStatePurpose::Z0YY>(
            "Z0YY", Z0YY);

        assert(A_high.size() == B_high.size());

        NativeMultiply64(NewN2, total_k, A_high, A_high, Z2XX);
        NativeMultiply64(NewN2, total_k, A_high, B_high, Z2XY);
        NativeMultiply64(NewN2, total_k, B_high, B_high, Z2YY);

        auto ProcessOneZ2 = [&]<auto DebugPurpose>(
            [[maybe_unused]] const char *name,
            const std::vector<uint64_t> &Z2) {

            const auto &Z2Checksum =
                GetCurrentDebugState<SharkFloatParams, RecursionDepth, CallIndex, DebugPurpose>(
                    debugStates, UseConvolutionHere, Z2.data(), Z2.size());
            if constexpr (SharkFloatParams::HostVerbose) {
                std::cout << name << ": " << VectorUintToHexString(Z2) << std::endl;
                std::cout << name << " checksum: " << Z2Checksum.GetStr() << std::endl;
            }
            };
        
        ProcessOneZ2.template operator()<DebugStatePurpose::Z2XX>(
            "Z2XX", Z2XX);
        ProcessOneZ2.template operator()<DebugStatePurpose::Z2XY>(
            "Z2XY", Z2XY);
        ProcessOneZ2.template operator()<DebugStatePurpose::Z2YY>(
            "Z2YY", Z2YY);

        assert(x_diff.size() == y_diff.size());

        NativeMultiply64(MaxHalfN, total_k, x_diff, x_diff, Z1_tempXX);
        NativeMultiply64(MaxHalfN, total_k, x_diff, y_diff, Z1_tempXY);
        NativeMultiply64(MaxHalfN, total_k, y_diff, y_diff, Z1_tempYY);

        auto ProcessOneZ1Temp = [&]<auto DebugPurpose>(
            [[maybe_unused]] const char *name,
            const std::vector<uint64_t> &Z1_temp) {

            const auto &Z1TempChecksum =
                GetCurrentDebugState<SharkFloatParams, RecursionDepth, CallIndex, DebugPurpose>(
                    debugStates, UseConvolutionHere, Z1_temp.data(), Z1_temp.size());
            if constexpr (SharkFloatParams::HostVerbose) {
                std::cout << name << ": " << VectorUintToHexString(Z1_temp) << std::endl;
                std::cout << name << " checksum: " << Z1TempChecksum.GetStr() << std::endl;
            }
            };

        ProcessOneZ1Temp.template operator()<DebugStatePurpose::Z1_offsetXX>(
            "Z1_tempXX", Z1_tempXX);
        ProcessOneZ1Temp.template operator()<DebugStatePurpose::Z1_offsetXY>(
            "Z1_tempXY", Z1_tempXY);
        ProcessOneZ1Temp.template operator()<DebugStatePurpose::Z1_offsetYY>(
            "Z1_tempYY", Z1_tempYY);
    } else {
        
        KaratsubaRecursiveDigits<
            SharkFloatParams,
            NewNumBlocks / 3,
            RecursionDepth + 1,
            CallIndex * 3 - 1>(
            A_low,
            B_low,
            Z0XX,
            Z0XY,
            Z0YY,
            debugStates);

        auto ProcessOneZ0 = [&]<auto DebugPurpose>(
            [[maybe_unused]] const char *name,
            std::vector<uint64_t> &Z0) {

            const auto &Z0Checksum =
                GetCurrentDebugState<SharkFloatParams, RecursionDepth, CallIndex, DebugPurpose>(
                    debugStates, UseConvolutionHere, Z0.data(), Z0.size());

            if constexpr (SharkFloatParams::HostVerbose) {
                std::cout << name << ": " << VectorUintToHexString(Z0) << std::endl;
                std::cout << name << " checksum: " << Z0Checksum.GetStr() << std::endl;
            }
        };

        ProcessOneZ0.template operator()<DebugStatePurpose::Z0XX>(
            "Z0XX", Z0XX);
        ProcessOneZ0.template operator()<DebugStatePurpose::Z0XY>(
            "Z0XY", Z0XY);
        ProcessOneZ0.template operator()<DebugStatePurpose::Z0YY>(
            "Z0YY", Z0YY);

        KaratsubaRecursiveDigits<
            SharkFloatParams,
            NewNumBlocks / 3,
            RecursionDepth + 1,
            CallIndex * 3>(
                A_high,
                B_high,
                Z2XX,
                Z2XY,
                Z2YY,
                debugStates);

        auto ProcessOneZ2 = [&]<auto DebugPurpose>(
            [[maybe_unused]] const char *name,
            std::vector<uint64_t> &Z2) {

            const auto &Z2Checksum =
                GetCurrentDebugState<SharkFloatParams, RecursionDepth, CallIndex, DebugPurpose>(
                    debugStates, UseConvolutionHere, Z2.data(), Z2.size());
            if constexpr (SharkFloatParams::HostVerbose) {
                std::cout << name << ": " << VectorUintToHexString(Z2) << std::endl;
                std::cout << name << " checksum: " << Z2Checksum.GetStr() << std::endl;
            }
            };

        ProcessOneZ2.template operator()<DebugStatePurpose::Z2XX>(
            "Z2XX", Z2XX);
        ProcessOneZ2.template operator()<DebugStatePurpose::Z2XY>(
            "Z2XY", Z2XY);
        ProcessOneZ2.template operator()<DebugStatePurpose::Z2YY>(
            "Z2YY", Z2YY);

        KaratsubaRecursiveDigits<
            SharkFloatParams,
            NewNumBlocks / 3,
            RecursionDepth + 1,
            CallIndex * 3 + 1>(
                x_diff,
                y_diff,
                Z1_tempXX,
                Z1_tempXY,
                Z1_tempYY,
                debugStates);

        auto ProcessOneZ1Temp = [&]<auto DebugPurpose>(
            [[maybe_unused]] const char *name,
            std::vector<uint64_t> &Z1_temp) {

            const auto &Z1TempChecksum =
                GetCurrentDebugState<SharkFloatParams, RecursionDepth, CallIndex, DebugPurpose>(
                    debugStates, UseConvolutionHere, Z1_temp.data(), Z1_temp.size());
            if constexpr (SharkFloatParams::HostVerbose) {
                std::cout << name << ": " << VectorUintToHexString(Z1_temp) << std::endl;
                std::cout << name << " checksum: " << Z1TempChecksum.GetStr() << std::endl;
            }
            };

        ProcessOneZ1Temp.template operator()<DebugStatePurpose::Z1_offsetXX>(
            "Z1_tempXX", Z1_tempXX);
        ProcessOneZ1Temp.template operator()<DebugStatePurpose::Z1_offsetXY>(
            "Z1_tempXY", Z1_tempXY);
        ProcessOneZ1Temp.template operator()<DebugStatePurpose::Z1_offsetYY>(
            "Z1_tempYY", Z1_tempYY);
    }

    const int z1_signXX = (x_diff_neg ^ x_diff_neg) ? 1 : 0;
    const int z1_signXY = (x_diff_neg ^ y_diff_neg) ? 1 : 0;
    const int z1_signYY = (y_diff_neg ^ y_diff_neg) ? 1 : 0;
    int total_k_full = 2 * NewN1 - 1;

    // Compute Z1=(Z2+Z0)±Z1_temp
    // First Z2+Z0
    std::vector<uint64_t> Z1XX(2 * total_k_full, 0ULL);
    std::vector<uint64_t> Z1XY(2 * total_k_full, 0ULL);
    std::vector<uint64_t> Z1YY(2 * total_k_full, 0ULL);

    auto ProcessOneZ1 = [&]<auto DebugPurpose>(
        [[maybe_unused]] const char *name,
        const int z1_sign,
        std::vector<uint64_t> &Z1,
        const std::vector<uint64_t> &Z0,
        const std::vector<uint64_t> &Z2,
        const std::vector<uint64_t> &Z1_temp) {

        // For each "digit" k, we compute Z1[k] = (Z2[k] + Z0[k]) ± Z1_temp[k]
        // Each "digit" is two 64-bit values: low and high
        for (int k = 0; k < total_k_full; ++k) {
            int idx_low = k * 2;
            int idx_high = idx_low + 1;

            uint64_t z0_low = Z0[idx_low];
            uint64_t z0_high = Z0[idx_high];
            uint64_t z2_low = idx_low < Z2.size() ? Z2[idx_low] : 0;
            uint64_t z2_high = idx_high < Z2.size() ? Z2[idx_high] : 0;
            uint64_t z1t_low = Z1_temp[idx_low];
            uint64_t z1t_high = Z1_temp[idx_high];

            // Compute (Z2 + Z0)
            uint64_t temp_low, temp_high;
            Add128(z2_low, z2_high, z0_low, z0_high, temp_low, temp_high);

            // Compute Z1 = (Z2+Z0) ± Z1_temp
            uint64_t z1_low, z1_high;
            if (z1_sign == 0) {
                // Z1 = (Z2 + Z0) - Z1_temp
                Subtract128(temp_low, temp_high, z1t_low, z1t_high, z1_low, z1_high);
            } else {
                // Z1 = (Z2 + Z0) + Z1_temp
                Add128(temp_low, temp_high, z1t_low, z1t_high, z1_low, z1_high);
            }

            if constexpr (SharkFloatParams::HostVerbose) {
                std::string addOrSubtract = (z1_sign == 0) ? "subtract" : "add";

                // Convert temp_low to hex string:
                std::string temp_low_hex = UintToHexString(temp_low);

                std::cout << addOrSubtract <<
                    " temp_low: " << UintToHexString(temp_low) <<
                    ", temp_high: " << UintToHexString(temp_high) <<
                    ", z1t_low: " << UintToHexString(z1t_low) <<
                    ", z1t_high: " << UintToHexString(z1t_high) <<
                    ", z1_low: " << UintToHexString(z1_low) <<
                    ", z1_high : " << UintToHexString(z1_high) <<
                    std::endl;
            }

            Z1[idx_low] = z1_low;
            Z1[idx_high] = z1_high;
        }

        const auto &Z1Checksum =
            GetCurrentDebugState<SharkFloatParams, RecursionDepth, CallIndex, DebugPurpose>(
                debugStates, UseConvolutionHere, Z1.data(), Z1.size());

        if constexpr (SharkFloatParams::HostVerbose) {
            std::cout << name << ": " << VectorUintToHexString(Z1) << std::endl;
            std::cout << name << " checksum: " << Z1Checksum.GetStr() << std::endl;
        }
        };

    ProcessOneZ1.template operator()<DebugStatePurpose::Z1XX>(
        "Z1XX", z1_signXX, Z1XX, Z0XX, Z2XX, Z1_tempXX);
    ProcessOneZ1.template operator()<DebugStatePurpose::Z1XY>(
        "Z1XY", z1_signXY, Z1XY, Z0XY, Z2XY, Z1_tempXY);
    ProcessOneZ1.template operator()<DebugStatePurpose::Z1YY>(
        "Z1YY", z1_signYY, Z1YY, Z0YY, Z2YY, Z1_tempYY);

    // Now Z1 matches the per-digit computation that CUDA performs.
    // The subsequent steps (carry propagation, final combination) should produce identical results to CUDA.

    // final=Z0+(Z1<<(32*n))+(Z2<<(64*n))
    // N even means (32*n) bits = n/2 64-bit words
    // (64*n) bits = n 64-bit words

    auto ProcessOneFinal128 = [&]<auto DebugPurpose>(
        [[maybe_unused]] const char *name,
        std::vector<uint64_t> &final128,
        const std::vector<uint64_t> &Z0,
        const std::vector<uint64_t> &Z1,
        const std::vector<uint64_t> &Z2) {

        const int total_result_digits = 2 * NewN;

        // If the input array is larger, then the high-order entries
        // remain zero.
        assert(final128.size() == total_result_digits * 2);

        for (int idx = 0; idx < total_result_digits; ++idx) {
            uint64_t sum_low = 0ULL;
            uint64_t sum_high = 0ULL;

            // Add Z0 if in range
            if (idx < 2 * NewN1 - 1) {
                int z0_idx = idx * 2;
                uint64_t z0_low = Z0[z0_idx];
                uint64_t z0_high = Z0[z0_idx + 1];

                Add128(sum_low, sum_high, z0_low, z0_high, sum_low, sum_high);
            }

            // Add Z1 << (32*n)
            // Shifting by 32*n means skipping n digits. If idx >= n, we add Z1 digit (idx-n)
            if (idx >= NewN1 && (idx - NewN1) < (2 * NewN1 - 1)) {
                int z1_idx = (idx - NewN1) * 2;
                uint64_t z1_low = Z1[z1_idx];
                uint64_t z1_high = Z1[z1_idx + 1];

                Add128(sum_low, sum_high, z1_low, z1_high, sum_low, sum_high);
            }

            // Add Z2 << (64*n)
            // Shifting by 64*n means skipping 2*n digits. If idx >= 2*n, we add Z2 digit (idx-2*n)
            if (idx >= 2 * NewN1 && (idx - 2 * NewN1) < (2 * NewN1 - 1)) {
                int z2_idx = (idx - 2 * NewN1) * 2;
                uint64_t z2_low = z2_idx < Z2.size() ? Z2[z2_idx] : 0;
                uint64_t z2_high = (z2_idx + 1 < Z2.size()) ? Z2[z2_idx + 1] : 0;

                Add128(sum_low, sum_high, z2_low, z2_high, sum_low, sum_high);
            }

            final128[idx * 2] = sum_low;
            final128[idx * 2 + 1] = sum_high;
        }

        // Checksum only assigned digits
        const auto &Final128Checksum =
            GetCurrentDebugState<SharkFloatParams, RecursionDepth, CallIndex, DebugPurpose>(
                debugStates, UseConvolutionHere, final128.data(), total_result_digits * 2);

        if constexpr (SharkFloatParams::HostVerbose) {
            std::cout << name << " after Z2: " << VectorUintToHexString(final128.data(), total_result_digits * 2) << std::endl;
            std::cout << name << " checksum: " << Final128Checksum.GetStr() << std::endl;
        }
    };

    ProcessOneFinal128.template operator()<DebugStatePurpose::Final128XX>(
        "final128XX", final128XX, Z0XX, Z1XX, Z2XX);
    ProcessOneFinal128.template operator()<DebugStatePurpose::Final128XY>(
        "final128XY", final128XY, Z0XY, Z1XY, Z2XY);
    ProcessOneFinal128.template operator()<DebugStatePurpose::Final128YY>(
        "final128YY", final128YY, Z0YY, Z1YY, Z2YY);
}

template<class SharkFloatParams>
void MultiplyHelperKaratsubaV2 (
    const HpSharkFloat<SharkFloatParams> *A,
    const HpSharkFloat<SharkFloatParams> *B,
    HpSharkFloat<SharkFloatParams> *OutXX,
    HpSharkFloat<SharkFloatParams> *OutXY,
    HpSharkFloat<SharkFloatParams> *OutYY,
    std::vector<DebugStateHost<SharkFloatParams>> &debugStates
) {
    constexpr int N = SharkFloatParams::GlobalNumUint32;

    if constexpr (SharkFloatParams::HostVerbose) {
        std::cout << std::endl;
        std::cout << "Will perform Karatsuba multiplication on host, running function MultiplyHelperKaratsubaV2." << std::endl;
    }

    // 3) Allocate array for up to 2*N digits result
    constexpr int total_result_digits = 2 * N;

    std::vector<uint64_t> final128XX;
    std::vector<uint64_t> final128XY;
    std::vector<uint64_t> final128YY;
    
    final128XX.resize(total_result_digits * 2);
    final128XY.resize(total_result_digits * 2);
    final128YY.resize(total_result_digits * 2);

    constexpr auto CallIndex = 0;
    constexpr auto RecursionDepth = 0;
    using DebugState = DebugStateHost<SharkFloatParams>;
    //DebugState debugAState{ A->Digits, SharkFloatParams::GlobalNumUint32, DebugStatePurpose::ADigits, CallIndex };
    //debugStates.push_back(debugAState);

    //DebugState debugBState{ B->Digits, SharkFloatParams::GlobalNumUint32, DebugStatePurpose::BDigits, CallIndex };
    //debugStates.push_back(debugBState);

    //debugStates.resize(SharkFloatParams::NumDebugStates);

    {
        // 1) Possibly do sign logic, exponent logic, etc.
        // 2) Convert A->Digits, B->Digits into pointers
        const uint32_t *A_ptr = A->Digits;
        const uint32_t *B_ptr = B->Digits;

        // Copy A_ptr into A_vector
        std::vector<uint32_t> A_vector;
        A_vector.reserve(N);
        for (int i = 0; i < N; i++) {
            A_vector.push_back(A_ptr[i]);
        }

        // Copy B_ptr into B_vector
        std::vector<uint32_t> B_vector;
        B_vector.reserve(N);
        for (int i = 0; i < N; i++) {
            B_vector.push_back(B_ptr[i]);
        }

        // 4) Call KaratsubaRecursiveDigits
        KaratsubaRecursiveDigits<
            SharkFloatParams,
            SharkFloatParams::GlobalNumBlocks,
            RecursionDepth + 1,
            CallIndex + 1>(
            A_vector,
            B_vector,
            final128XX,
            final128XY,
            final128YY,
            debugStates);
    }

    // Assume final128 is arranged as pairs: final128[0], final128[1] form the first pair (sum_low, sum_high),
    // final128[2], final128[3] the second pair, and so forth.
    // Each pair corresponds to one "digit position" before final normalization.

    std::vector<uint32_t> tempDigitsXX;
    std::vector<uint32_t> tempDigitsXY;
    std::vector<uint32_t> tempDigitsYY;

    auto ProcessOneTempDigits = [&] (
        std::vector<uint32_t> &tempDigits,
        const std::vector<uint64_t> &final128) {

        tempDigits.reserve(total_result_digits); // At least one digit per pair
        uint64_t local_carry = 0ULL;
        for (int idx = 0; idx < total_result_digits; ++idx) {
            uint64_t sum_low;
            uint64_t sum_high;

            if (idx >= final128.size() / 2) {
                sum_low = 0ULL;
                sum_high = 0ULL;
            } else {
                sum_low = final128[idx * 2];
                sum_high = final128[idx * 2 + 1];
            }

            // Add local carry to sum_low
            bool new_sum_low_negative = false;
            uint64_t new_sum_low = sum_low + local_carry;

            // Extract one 32-bit digit from new_sum_low
            auto digit = static_cast<uint32_t>(new_sum_low & 0xFFFFFFFFULL);
            tempDigits.push_back(digit);

            bool local_carry_negative = ((local_carry & (1ULL << 63)) != 0);
            local_carry = 0ULL;

            if (!local_carry_negative && new_sum_low < sum_low) {
                local_carry = 1ULL << 32;
            } else if (local_carry_negative && new_sum_low > sum_low) {
                new_sum_low_negative = (new_sum_low & 0x8000'0000'0000'0000) != 0;
            }

            // Update local_carry
            if (new_sum_low_negative) {
                // Shift sum_high by 32 bits and add carry_from_low
                uint64_t upper_new_sum_low = new_sum_low >> 32;
                upper_new_sum_low |= 0xFFFF'FFFF'0000'0000;
                local_carry += upper_new_sum_low;
                local_carry += sum_high << 32;
            } else {
                local_carry += new_sum_low >> 32;
                local_carry += sum_high << 32;
            }
        }
        };

    ProcessOneTempDigits(tempDigitsXX, final128XX);
    ProcessOneTempDigits(tempDigitsXY, final128XY);
    ProcessOneTempDigits(tempDigitsYY, final128YY);

    // Now we have a sequence of 32-bit digits in tempDigits exactly as the CUDA code would produce.
    // The next step is to normalize, find highest_nonzero_index, and adjust exponent and shift_digits
    // in the same manner as the CUDA version does.

    auto PrintOneTempDigits = [&]<auto DebugPurpose>(
        [[maybe_unused]] const std::vector<uint32_t> &tempDigits,
        [[maybe_unused]] const std::string &name) {

        [[maybe_unused]] const auto &resultDigitsChecksumXY =
            GetCurrentDebugState<SharkFloatParams, RecursionDepth, CallIndex, DebugPurpose>(
                debugStates, UseConvolution::No, tempDigits.data(), 2 * N);

        if constexpr (SharkFloatParams::HostVerbose) {
            std::cout << name << ": " << VectorUintToHexString(tempDigits) << std::endl;
            std::cout << name << " checksum: " << resultDigitsChecksumXY.GetStr() << std::endl;
        }
    };

    PrintOneTempDigits.template operator()<DebugStatePurpose::Result_offsetXX>(
        tempDigitsXX, "tempDigitsXX");
    PrintOneTempDigits.template operator()<DebugStatePurpose::Result_offsetXY>(
        tempDigitsXY, "tempDigitsXY");
    PrintOneTempDigits.template operator()<DebugStatePurpose::Result_offsetYY>(
        tempDigitsYY, "tempDigitsYY");

    // tempDigits now contains a properly carry-normalized array of 32-bit digits.
    // You can then adjust the exponent and choose exactly N digits for the final output.

    // Normalize result

    auto NormalizeOne = [&](
        HpSharkFloat<SharkFloatParams> &Out,
        const HpSharkFloat<SharkFloatParams> &A,
        const HpSharkFloat<SharkFloatParams> &B,
        const std::vector<uint32_t> &tempDigits) {

        int highest_nonzero_index = (int)tempDigits.size() - 1;
        while (highest_nonzero_index >= 0 && tempDigits[highest_nonzero_index] == 0) {
            highest_nonzero_index--;
        }

        if constexpr (SharkFloatParams::HostVerbose) {
            std::cout << "Highest nonzero index: " << highest_nonzero_index << std::endl;
        }

        int significant_digits = highest_nonzero_index + 1;
        if (significant_digits < 1) {
            // Product is zero
            significant_digits = 1;
            for (int i = 0; i < N; i++) {
                Out.Digits[i] = 0;
            }
            Out.Exponent = A.Exponent + B.Exponent;
            Out.SetNegative(false);
            return;
        }

        // Determine how many digits we need to shift to keep exactly N digits.
        int shift_digits = significant_digits - N;

        // Match CUDA behavior: If we have fewer than N significant digits, do not shift negatively
        if (shift_digits < 0) {
            shift_digits = 0;
            //assert(false);
        }

        if constexpr (SharkFloatParams::HostVerbose) {
            std::cout << "Shift digits: " << shift_digits << std::endl;
        }

        // Update the exponent accordingly
        Out.Exponent = A.Exponent + B.Exponent + shift_digits * 32;

        // Extract exactly N digits
        int src_idx = shift_digits;
        for (int i = 0; i < N; i++, src_idx++) {
            uint32_t val = 0;
            if (src_idx >= 0 && src_idx < significant_digits) {
                val = tempDigits[src_idx];
            }
            Out.Digits[i] = val;
        }

        // Print debugStates
        if constexpr (SharkFloatParams::HostVerbose) {
            for (const auto &state : debugStates) {
                std::cout << state.GetStr() << std::endl;
            }
        }
        };

    NormalizeOne(*OutXX, *A, *A, tempDigitsXX);
    OutXX->SetNegative(false);

    NormalizeOne(*OutXY, *A, *B, tempDigitsXY);
    OutXY->SetNegative(A->GetNegative() ^ B->GetNegative());

    NormalizeOne(*OutYY, *B, *B, tempDigitsYY);
    OutYY->SetNegative(false);
}


#define ExplicitlyInstantiate(SharkFloatParams) \
    template void MultiplyHelperKaratsubaV2<SharkFloatParams>( \
        const HpSharkFloat<SharkFloatParams> *, \
        const HpSharkFloat<SharkFloatParams> *, \
        HpSharkFloat<SharkFloatParams> *, \
        HpSharkFloat<SharkFloatParams> *, \
        HpSharkFloat<SharkFloatParams> *, \
        std::vector<DebugStateHost<SharkFloatParams>> &debugStates);

ExplicitInstantiateAll();