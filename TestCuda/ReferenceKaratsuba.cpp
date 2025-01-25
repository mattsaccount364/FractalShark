#include "ReferenceKaratsuba.h"
#include "HpSharkFloat.cuh"
#include "DebugChecksumHost.h"

#include <cstdint>
#include <algorithm>
#include <cstring> // for memset
#include <vector>
#include <iostream>
#include <assert.h>

// Helper: Add arrays of potentially different lengths
// A_len >= B_len; result length = A_len or A_len+1
static uint32_t AddArraysDifferentLengthV1(const uint32_t *A, int A_len, const uint32_t *B, int B_len, uint32_t *Res) {
    uint64_t carry = 0;
    int i = 0;
    for (; i < B_len; i++) {
        uint64_t sum = (uint64_t)A[i] + (uint64_t)B[i] + carry;
        Res[i] = (uint32_t)sum;
        carry = sum >> 32;
    }
    for (; i < A_len; i++) {
        uint64_t sum = (uint64_t)A[i] + carry;
        Res[i] = (uint32_t)sum;
        carry = sum >> 32;
    }
    return (uint32_t)carry;
}


// Helper: Naive O(N²) multiplication
// A and B length n, result length = 2*n
static void NaiveMultiplyV1(const uint32_t *A, const uint32_t *B, int n, uint32_t *Res) {
    for (int i = 0; i < 2 * n; i++) {
        Res[i] = 0;
    }

    for (int i = 0; i < n; i++) {
        uint64_t carry = 0;
        uint64_t a = A[i];
        for (int j = 0; j < n; j++) {
            uint64_t mul = (uint64_t)a * (uint64_t)B[j] + (uint64_t)Res[i + j] + carry;
            Res[i + j] = (uint32_t)(mul & 0xFFFFFFFFULL);
            carry = mul >> 32;
        }
        Res[i + n] = (uint32_t)carry;
    }
}


// Single-level Karatsuba-like function
// N = SharkFloatParams::GlobalNumUint32
// We split A and B into two halves: A_low, A_high, B_low, B_high
// Then compute Z0, Z2, and Z1 = (A_low+A_high)*(B_low+B_high)-Z0-Z2
template<class SharkFloatParams>
void MultiplyHelperKaratsubaV1(
    const HpSharkFloat<SharkFloatParams> *A,
    const HpSharkFloat<SharkFloatParams> *B,
    HpSharkFloat<SharkFloatParams> *Out
) {
    constexpr int N = SharkFloatParams::GlobalNumUint32;
    if constexpr (N == 1) {
        // Just multiply single 32-bit limbs
        uint64_t prod = (uint64_t)A->Digits[0] * (uint64_t)B->Digits[0];
        Out->Digits[0] = (uint32_t)(prod & 0xFFFFFFFFULL);
        // If we had more limbs, they'd go here, but we only have one.
        Out->Exponent = A->Exponent + B->Exponent;
        Out->IsNegative = (A->IsNegative ^ B->IsNegative);
        return;
    }

    // Split in half
    int half = N / 2; // If N is not even, we can still do this, one half is half, another half = N-half
    int high_len = N - half;

    const uint32_t *A_low = A->Digits;
    const uint32_t *A_high = A->Digits + half;
    const uint32_t *B_low = B->Digits;
    const uint32_t *B_high = B->Digits + half;

    // Compute Z0 = A_low * B_low (length 2*half)
    std::vector<uint32_t> Z0(2 * half, 0);
    NaiveMultiplyV1(A_low, B_low, half, Z0.data());

    // Compute Z2 = A_high * B_high (length 2*high_len)
    std::vector<uint32_t> Z2(2 * high_len, 0);
    NaiveMultiplyV1(A_high, B_high, high_len, Z2.data());

    // Compute (A_low + A_high) and (B_low + B_high)
    // The sum arrays can have length = max(half, high_len)+1 in worst case
    int sum_len = std::max(half, high_len) + 1;
    std::vector<uint32_t> A_sum(sum_len, 0);
    std::vector<uint32_t> B_sum(sum_len, 0);

    // Add A_low and A_high
    if (high_len >= half) {
        // Copy A_high
        for (int i = 0; i < high_len; i++) A_sum[i] = A_high[i];
        uint32_t carry = AddArraysDifferentLengthV1(A_sum.data(), high_len, A_low, half, A_sum.data());
        if (carry) A_sum[high_len] = carry;
    } else {
        // Copy A_low
        for (int i = 0; i < half; i++) A_sum[i] = A_low[i];
        uint32_t carry = AddArraysDifferentLengthV1(A_sum.data(), half, A_high, high_len, A_sum.data());
        if (carry) A_sum[half] = carry;
    }

    // Add B_low and B_high
    if (high_len >= half) {
        for (int i = 0; i < high_len; i++) B_sum[i] = B_high[i];
        uint32_t carry = AddArraysDifferentLengthV1(B_sum.data(), high_len, B_low, half, B_sum.data());
        if (carry) B_sum[high_len] = carry;
    } else {
        for (int i = 0; i < half; i++) B_sum[i] = B_low[i];
        uint32_t carry = AddArraysDifferentLengthV1(B_sum.data(), half, B_high, high_len, B_sum.data());
        if (carry) B_sum[half] = carry;
    }

    // Determine actual lengths of A_sum and B_sum
    int A_sum_len = sum_len; while (A_sum_len > 1 && A_sum[A_sum_len - 1] == 0) A_sum_len--;
    int B_sum_len = sum_len; while (B_sum_len > 1 && B_sum[B_sum_len - 1] == 0) B_sum_len--;

    int max_sum_len = std::max(A_sum_len, B_sum_len);
    // Z1_temp = (A_sum * B_sum)
    std::vector<uint32_t> Z1_temp(2 * max_sum_len, 0);
    NaiveMultiplyV1(A_sum.data(), B_sum.data(), max_sum_len, Z1_temp.data());

    // Z1 = Z1_temp
    std::vector<uint32_t> Z1 = Z1_temp;

    // Z1 = Z1 - Z0 (no shift)
    {
        uint32_t borrow = 0;
        int min_len = std::min((int)Z1.size(), (int)Z0.size());
        for (int i = 0; i < min_len; i++) {
            uint64_t z1_val = Z1[i];
            uint64_t z0_val = (uint64_t)Z0[i] + borrow;
            if (z1_val < z0_val) {
                Z1[i] = (uint32_t)((z1_val + ((uint64_t)1 << 32)) - z0_val);
                borrow = 1;
            } else {
                Z1[i] = (uint32_t)(z1_val - z0_val);
                borrow = 0;
            }
        }
        
        // After finishing the min_len loop:
        for (int i = min_len; borrow && i < (int)Z1.size(); i++) {
            uint64_t z1_val = Z1[i];
            if (z1_val < borrow) {
                Z1[i] = (uint32_t)((z1_val + ((uint64_t)1 << 32)) - borrow);
                borrow = 1; // still have borrow because we wrapped
            } else {
                Z1[i] = (uint32_t)(z1_val - borrow);
                borrow = 0; // borrow resolved
            }
        }

        // If borrow is still not zero here and we have no more digits in Z1, 
        // that would indicate Z1 went negative, which shouldn't happen in correct Karatsuba logic.
    }

    // Z1 = Z1 - Z2 (no shift)
    {
        uint32_t borrow = 0;
        int min_len = std::min((int)Z1.size(), (int)Z2.size());
        for (int i = 0; i < min_len; i++) {
            uint64_t z1_val = Z1[i];
            uint64_t z2_val = (uint64_t)Z2[i] + borrow;
            if (z1_val < z2_val) {
                Z1[i] = (uint32_t)((z1_val + ((uint64_t)1 << 32)) - z2_val);
                borrow = 1;
            } else {
                Z1[i] = (uint32_t)(z1_val - z2_val);
                borrow = 0;
            }
        }
        
        // After finishing the min_len loop:
        for (int i = min_len; borrow && i < (int)Z1.size(); i++) {
            uint64_t z1_val = Z1[i];
            if (z1_val < borrow) {
                Z1[i] = (uint32_t)((z1_val + ((uint64_t)1 << 32)) - borrow);
                borrow = 1; // still have borrow because we wrapped
            } else {
                Z1[i] = (uint32_t)(z1_val - borrow);
                borrow = 0; // borrow resolved
            }
        }

        // If borrow is still not zero here and we have no more digits in Z1, 
        // that would indicate Z1 went negative, which shouldn't happen in correct Karatsuba logic.
    }

    // Combine:
    // Result = Z0 + (Z1 << (32*half)) + (Z2 << (32*(2*half)))

    // Initialize result:
    std::vector<uint32_t> product(2 * N, 0);

    // Add Z0
    {
        int z0_len = (int)Z0.size(); // Typically 2*half
        int copy_len = std::min(z0_len, 2 * N);
        for (int i = 0; i < copy_len; i++) {
            product[i] = Z0[i];
        }
        // If z0_len < 2*N, the rest are already zero.
    }

    // Add Z1 at offset half
    {
        int z1_len = (int)Z1.size();
        int max_i = 2 * N - half; // maximum length we can add
        uint64_t carry = 0;
        for (int i = 0; i < max_i; i++) {
            uint64_t z1_val = (i < z1_len) ? (uint64_t)Z1[i] : 0ULL;
            uint64_t sum = (uint64_t)product[i + half] + z1_val + carry;
            product[i + half] = (uint32_t)(sum & 0xFFFFFFFFULL);
            carry = sum >> 32;
        }
        // If carry remains and we have no space, it just doesn't fit (which typically shouldn't happen in correct Karatsuba)
    }

    // Add Z2 at offset 2*half
    {
        int z2_len = (int)Z2.size(); // Typically 2*high_len
        int max_i = 2 * N - 2 * half;
        uint64_t carry = 0;
        for (int i = 0; i < max_i; i++) {
            uint64_t z2_val = (i < z2_len) ? (uint64_t)Z2[i] : 0ULL;
            uint64_t sum = (uint64_t)product[i + 2 * half] + z2_val + carry;
            product[i + 2 * half] = (uint32_t)(sum & 0xFFFFFFFFULL);
            carry = sum >> 32;
        }
    }

    // Normalize result
    int highest_nonzero_index = 2 * N - 1;
    while (highest_nonzero_index >= 0 && product[highest_nonzero_index] == 0) {
        highest_nonzero_index--;
    }

    int significant_digits = highest_nonzero_index + 1;
    if (significant_digits < 1) {
        // Product is zero
        significant_digits = 1;
        for (int i = 0; i < N; i++) {
            Out->Digits[i] = 0;
        }
        Out->Exponent = A->Exponent + B->Exponent;
        Out->IsNegative = false;
        return;
    }

    // Determine how many digits we need to shift to keep exactly N digits
    int shift_digits = significant_digits - N;
    //if (shift_digits < 0) shift_digits = 0;

    Out->Exponent = A->Exponent + B->Exponent + shift_digits * 32;

    int src_idx = shift_digits;
    for (int i = 0; i < N; i++, src_idx++) {
        uint32_t val = 0;
        if (src_idx >= 0 && src_idx < significant_digits) {
            val = product[src_idx];
        }
        Out->Digits[i] = val;
    }

    Out->IsNegative = (A->IsNegative ^ B->IsNegative);
}

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

static void NativeMultiply64(
    const std::vector<uint32_t> &A,
    const std::vector<uint32_t> &B,
    std::vector<uint64_t> &Res) {

    auto n = static_cast<int>(A.size()); // TODO

    assert(n == B.size());

    // Number of partial sums = (2*n - 1).
    // Each partial sum is stored as two 64-bit words in Res.
    int total_k = 2 * n - 1;

    // Initialize Res to zero
    for (int i = 0; i < total_k * 2; i++) {
        Res[i] = 0ULL;
    }

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
static inline void add64withCarry(uint64_t x, uint64_t y, uint64_t carry_in,
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
        Res.push_back((uint32_t)diff);
    }
    // Assuming highArray >= lowArray, no final borrow remains.

    if constexpr (SharkFloatParams::HostVerbose) {
        std::cout << "SubtractDigitsSerial: Result: " << VectorUintToHexString(Res) << std::endl;
    }

    assert(borrow == 0);

    return (uint32_t)borrow;
}

template<class SharkFloatParams, int NewNumBlocks, int CallIndex>
void KaratsubaRecursiveDigits(
    const std::vector<uint32_t> &A_digits,  // pointer to A's digits
    const std::vector<uint32_t> &B_digits,
    std::vector<uint64_t> &final128, // place where the product digits go
    std::vector<DebugStateHost<SharkFloatParams>> &debugStates
)
{
    const int fullADigits = static_cast<int>(A_digits.size());
    const int fullBDigits = static_cast<int>(B_digits.size());
    assert(fullADigits == fullBDigits);

    const int halfARoundedUp = (fullADigits + 1) / 2;
    const int halfARoundedDown = fullADigits / 2;
    const int halfBRoundedUp = (fullBDigits + 1) / 2;
    const int halfBRoundedDown = fullBDigits / 2;

    const int halfRoundedUp = halfARoundedUp;
    const int halfRoundedDown = halfARoundedDown;
    assert(halfRoundedUp == halfBRoundedUp);
    assert(halfRoundedDown == halfBRoundedDown);

    using DebugState = DebugStateHost<SharkFloatParams>;

    DebugState debugAState{ A_digits, DebugState::Purpose::ADigits, CallIndex };
    debugStates.push_back(debugAState);

    DebugState debugBState{ B_digits, DebugState::Purpose::BDigits, CallIndex };
    debugStates.push_back(debugBState);

    if constexpr (SharkFloatParams::HostVerbose) {
        std::cout << "KaratsubaRecursiveDigits (index: " << CallIndex << "):" << std::endl;
        std::cout << "fullADigits = " << fullADigits << std::endl;
        std::cout << "fullBDigits = " << fullBDigits << std::endl;
        std::cout << "A->Digits checksum: " << debugAState.GetStr() << std::endl;
        std::cout << "B->Digits checksum: " << debugBState.GetStr() << std::endl;
        std::cout << VectorUintToHexString(A_digits) << " * ";
        std::cout << VectorUintToHexString(B_digits) << std::endl;
    }

    std::vector<uint32_t> A_low;
    for (size_t i = 0; i < halfRoundedUp; i++) {
        A_low.push_back(A_digits[i]);
    }

    assert(A_low.size() == halfRoundedUp);

    std::vector<uint32_t> A_high;
    for (size_t i = halfRoundedUp; i < fullADigits; i++) {
        A_high.push_back(A_digits[i]);
    }

    assert(A_high.size() == halfRoundedDown);

    std::vector<uint32_t> B_low;
    for (size_t i = 0; i < halfRoundedUp; i++) {
        B_low.push_back(B_digits[i]);
    }

    assert(B_low.size() == halfRoundedUp);

    std::vector<uint32_t> B_high;
    for (size_t i = halfRoundedUp; i < fullBDigits; i++) {
        B_high.push_back(B_digits[i]);
    }

    assert(B_high.size() == halfRoundedDown);

    // Print lengths of A_low, A_high, B_low, B_high
    if constexpr (SharkFloatParams::HostVerbose) {
        std::cout << "A_low: " << A_low.size() << std::endl;
        std::cout << "A_high: " << A_high.size() << std::endl;
        std::cout << "B_low: " << B_low.size() << std::endl;
        std::cout << "B_high: " << B_high.size() << std::endl;
    }

    int x_cmp = CompareDigits<SharkFloatParams>(A_high, A_low);
    bool x_diff_neg = false;
    std::vector<uint32_t> x_diff;
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

    DebugState xDiffChecksum{ x_diff, DebugState::Purpose::XDiff, CallIndex };
    debugStates.push_back(xDiffChecksum);

    DebugState yDiffChecksum{ y_diff, DebugState::Purpose::YDiff, CallIndex };
    debugStates.push_back(yDiffChecksum);

    assert(x_diff.size() == y_diff.size());
    assert(x_diff.size() == halfRoundedUp);
    assert(y_diff.size() == halfRoundedUp);
    const auto maxHalfSize = std::max(halfRoundedDown, halfRoundedUp);
    assert(x_diff.size() == maxHalfSize);
    assert(y_diff.size() == maxHalfSize);

    if constexpr (SharkFloatParams::HostVerbose) {
        std::cout << "x_diff: " << VectorUintToHexString(x_diff) << std::endl;
        std::cout << "x_diff checksum: " << xDiffChecksum.GetStr() << std::endl;
        std::cout << "y_diff: " << VectorUintToHexString(y_diff) << std::endl;
        std::cout << "y_diff checksum: " << yDiffChecksum.GetStr() << std::endl;
    }

    const auto zLen = (2 * maxHalfSize - 1) * 2;
    std::vector<uint64_t> Z0(zLen, 0ULL);
    std::vector<uint64_t> Z2(zLen, 0ULL);
    std::vector<uint64_t> Z1_temp(zLen, 0ULL);

    constexpr bool UseConvolution =
        (NewNumBlocks <= std::max(SharkFloatParams::GlobalNumBlocks / SharkFloatParams::ConvolutionLimit, 1) ||
        (NewNumBlocks % 3 != 0));

    //const bool UseConvolution = true;

    if constexpr (UseConvolution) {

        assert(A_low.size() == B_low.size());
        NativeMultiply64(A_low, B_low, Z0);

        DebugState Z0Checksum{ Z0, DebugState::Purpose::Z0, CallIndex };
        debugStates.push_back(Z0Checksum);

        if constexpr (SharkFloatParams::HostVerbose) {
            std::cout << "Z0: " << VectorUintToHexString(Z0) << std::endl;
            std::cout << "Z0 checksum: " << Z0Checksum.GetStr() << std::endl;
        }

        assert(A_high.size() == B_high.size());
        NativeMultiply64(A_high, B_high, Z2);

        DebugState Z2Checksum{ Z2, DebugState::Purpose::Z2, CallIndex };
        debugStates.push_back(Z2Checksum);

        if constexpr (SharkFloatParams::HostVerbose) {
            std::cout << "Z2: " << VectorUintToHexString(Z2) << std::endl;
            std::cout << "Z2 checksum: " << Z2Checksum.GetStr() << std::endl;
        }

        assert(x_diff.size() == y_diff.size());
        NativeMultiply64(x_diff, y_diff, Z1_temp);

        DebugState Z1TempChecksum{ Z1_temp, DebugState::Purpose::Z1_temp_offset, CallIndex };
        debugStates.push_back(Z1TempChecksum);

        if constexpr (SharkFloatParams::HostVerbose) {
            std::cout << "Z1_temp: " << VectorUintToHexString(Z1_temp) << std::endl;
            std::cout << "Z1_temp checksum: " << Z1TempChecksum.GetStr() << std::endl;
        }
    } else {
        KaratsubaRecursiveDigits<SharkFloatParams, NewNumBlocks / 3, CallIndex * 3 - 1>(
            A_low,
            B_low,
            Z0,
            debugStates);

        DebugState Z0Checksum{ Z0, DebugState::Purpose::Z0, CallIndex };
        debugStates.push_back(Z0Checksum);

        if constexpr (SharkFloatParams::HostVerbose) {
            std::cout << "Z0: " << VectorUintToHexString(Z0) << std::endl;
            std::cout << "Z0 checksum: " << Z0Checksum.GetStr() << std::endl;
        }

        KaratsubaRecursiveDigits<SharkFloatParams, NewNumBlocks / 3, CallIndex * 3>(
            A_high,
            B_high,
            Z2,
            debugStates);

        DebugState Z2Checksum{ Z2, DebugState::Purpose::Z2, CallIndex };
        debugStates.push_back(Z2Checksum);

        if constexpr (SharkFloatParams::HostVerbose) {
            std::cout << "Z2: " << VectorUintToHexString(Z2) << std::endl;
            std::cout << "Z2 checksum: " << Z2Checksum.GetStr() << std::endl;
        }

        KaratsubaRecursiveDigits<SharkFloatParams, NewNumBlocks / 3, CallIndex * 3 + 1>(
            x_diff,
            y_diff,
            Z1_temp,
            debugStates);

        DebugState Z1TempChecksum{ Z1_temp, DebugState::Purpose::Z1_temp_offset, CallIndex };
        debugStates.push_back(Z1TempChecksum);

        if constexpr (SharkFloatParams::HostVerbose) {
            std::cout << "Z1_temp: " << VectorUintToHexString(Z1_temp) << std::endl;
            std::cout << "Z1_temp checksum: " << Z1TempChecksum.GetStr() << std::endl;
        }
    }

    int z1_sign = (x_diff_neg ^ y_diff_neg) ? 1 : 0;
    int total_k_full = 2 * halfRoundedUp - 1;

    // Compute Z1=(Z2+Z0)±Z1_temp
    // First Z2+Z0
    std::vector<uint64_t> Z1(2 * total_k_full, 0ULL);

    // For each "digit" k, we compute Z1[k] = (Z2[k] + Z0[k]) ± Z1_temp[k]
    // Each "digit" is two 64-bit values: low and high
    for (int k = 0; k < total_k_full; ++k) {
        int idx_low = k * 2;
        int idx_high = idx_low + 1;

        uint64_t z0_low = Z0[idx_low];
        uint64_t z0_high = Z0[idx_high];
        uint64_t z2_low = Z2[idx_low];
        uint64_t z2_high = Z2[idx_high];
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

    DebugState Z1Checksum{ Z1, DebugState::Purpose::Z1, CallIndex };
    debugStates.push_back(Z1Checksum);

    if constexpr (SharkFloatParams::HostVerbose) {
        std::cout << "Z1: " << VectorUintToHexString(Z1) << std::endl;
        std::cout << "Z1 checksum: " << Z1Checksum.GetStr() << std::endl;
    }

    // Now Z1 matches the per-digit computation that CUDA performs.
    // The subsequent steps (carry propagation, final combination) should produce identical results to CUDA.

    // final=Z0+(Z1<<(32*n))+(Z2<<(64*n))
    // N even means (32*n) bits = n/2 64-bit words
    // (64*n) bits = n 64-bit words

    const int total_result_digits = 2 * fullADigits - 1;

    // If the input array is larger, then the high-order entries
    // remain zero.
    if (final128.size() < total_result_digits * 2) {
        final128.resize(total_result_digits * 2, 0ULL);
    }

    for (int idx = 0; idx < total_result_digits; ++idx) {
        uint64_t sum_low = 0ULL;
        uint64_t sum_high = 0ULL;

        // Add Z0 if in range
        if (idx < 2 * halfRoundedUp - 1) {
            int z0_idx = idx * 2;
            uint64_t z0_low = Z0[z0_idx];
            uint64_t z0_high = Z0[z0_idx + 1];

            Add128(sum_low, sum_high, z0_low, z0_high, sum_low, sum_high);
        }

        // Add Z1 << (32*n)
        // Shifting by 32*n means skipping n digits. If idx >= n, we add Z1 digit (idx-n)
        if (idx >= halfRoundedUp && (idx - halfRoundedUp) < (2 * halfRoundedUp - 1)) {
            int z1_idx = (idx - halfRoundedUp) * 2;
            uint64_t z1_low = Z1[z1_idx];
            uint64_t z1_high = Z1[z1_idx + 1];

            Add128(sum_low, sum_high, z1_low, z1_high, sum_low, sum_high);
        }

        // Add Z2 << (64*n)
        // Shifting by 64*n means skipping 2*n digits. If idx >= 2*n, we add Z2 digit (idx-2*n)
        if (idx >= 2 * halfRoundedUp && (idx - 2 * halfRoundedUp) < (2 * halfRoundedUp - 1)) {
            int z2_idx = (idx - 2 * halfRoundedUp) * 2;
            uint64_t z2_low = Z2[z2_idx];
            uint64_t z2_high = Z2[z2_idx + 1];

            Add128(sum_low, sum_high, z2_low, z2_high, sum_low, sum_high);
        }

        final128[idx * 2] = sum_low;
        final128[idx * 2 + 1] = sum_high;
    }

    // Checksum only assigned digits
    DebugState Final128Checksum{ final128.data(), total_result_digits * 2llu, DebugState::Purpose::Final128, CallIndex};
    debugStates.push_back(Final128Checksum);

    if constexpr (SharkFloatParams::HostVerbose) {
        std::cout << "final128 after Z2: " << VectorUintToHexString(final128.data(), total_result_digits * 2) << std::endl;
        std::cout << "final128 checksum: " << Final128Checksum.GetStr() << std::endl;
    }
}

template<class SharkFloatParams>
void MultiplyHelperKaratsubaV2(
    const HpSharkFloat<SharkFloatParams> *A,
    const HpSharkFloat<SharkFloatParams> *B,
    HpSharkFloat<SharkFloatParams> *Out,
    std::vector<DebugStateHost<SharkFloatParams>> &debugStates
) {
    constexpr int N = SharkFloatParams::GlobalNumUint32;

    if constexpr (SharkFloatParams::HostVerbose) {
        std::cout << std::endl;
        std::cout << "Will perform Karatsuba multiplication on host, running function MultiplyHelperKaratsubaV2." << std::endl;
    }

    if constexpr (N == 1) {
        uint64_t prod = (uint64_t)A->Digits[0] * (uint64_t)B->Digits[0];
        Out->Digits[0] = (uint32_t)(prod & 0xFFFFFFFFULL);
        Out->Exponent = A->Exponent + B->Exponent;
        Out->IsNegative = (A->IsNegative ^ B->IsNegative);

        if constexpr (SharkFloatParams::HostVerbose) {
            std::cout << "Karatsuba N=1: A=" << A->ToHexString() << ", B=" << B->ToHexString() << ", Out=" << Out->ToHexString() << std::endl;
        }

        return;
    }

    // 3) Allocate array for up to 2*N digits result
    constexpr int total_result_digits = 2 * N;

    std::vector<uint64_t> final128;

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
        KaratsubaRecursiveDigits<SharkFloatParams, SharkFloatParams::GlobalNumBlocks, 1>(A_vector, B_vector, final128, debugStates);
    }

    // Assume final128 is arranged as pairs: final128[0], final128[1] form the first pair (sum_low, sum_high),
    // final128[2], final128[3] the second pair, and so forth.
    // Each pair corresponds to one "digit position" before final normalization.

    std::vector<uint32_t> tempDigits;
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

    // Now we have a sequence of 32-bit digits in tempDigits exactly as the CUDA code would produce.
    // The next step is to normalize, find highest_nonzero_index, and adjust exponent and shift_digits
    // in the same manner as the CUDA version does.

    if constexpr (SharkFloatParams::HostVerbose) {
        std::cout << "tempDigits: " << VectorUintToHexString(tempDigits) << std::endl;
    }

    // tempDigits now contains a properly carry-normalized array of 32-bit digits.
    // You can then adjust the exponent and choose exactly N digits for the final output.

    // Normalize result
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
            Out->Digits[i] = 0;
        }
        Out->Exponent = A->Exponent + B->Exponent;
        Out->IsNegative = false;
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
    Out->Exponent = A->Exponent + B->Exponent + shift_digits * 32;

    // Extract exactly N digits
    int src_idx = shift_digits;
    for (int i = 0; i < N; i++, src_idx++) {
        uint32_t val = 0;
        if (src_idx >= 0 && src_idx < significant_digits) {
            val = tempDigits[src_idx];
        }
        Out->Digits[i] = val;
    }

    // Set the sign
    Out->IsNegative = (A->IsNegative ^ B->IsNegative);

    // Print debugStates
    for (const auto &state : debugStates) {
        std::cout << state.GetStr() << std::endl;
    }
}


#define ExplicitlyInstantiate(SharkFloatParams) \
    template void MultiplyHelperKaratsubaV1<SharkFloatParams>( \
        const HpSharkFloat<SharkFloatParams> *, \
        const HpSharkFloat<SharkFloatParams> *, \
        HpSharkFloat<SharkFloatParams> *); \
    template void MultiplyHelperKaratsubaV2<SharkFloatParams>( \
        const HpSharkFloat<SharkFloatParams> *, \
        const HpSharkFloat<SharkFloatParams> *, \
        HpSharkFloat<SharkFloatParams> *, \
        std::vector<DebugStateHost<SharkFloatParams>> &debugStates);

ExplicitInstantiateAll();