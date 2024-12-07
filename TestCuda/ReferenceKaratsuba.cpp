#include "ReferenceKaratsuba.h"
#include "HpSharkFloat.cuh"

#include <cstdint>
#include <algorithm>
#include <cstring> // for memset
#include <vector>

// Helper: Add arrays of potentially different lengths
// A_len >= B_len; result length = A_len or A_len+1
static uint32_t AddArraysDifferentLength(const uint32_t *A, int A_len, const uint32_t *B, int B_len, uint32_t *Res) {
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
static void NaiveMultiply(const uint32_t *A, const uint32_t *B, int n, uint32_t *Res) {
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
// N = SharkFloatParams::NumUint32
// We split A and B into two halves: A_low, A_high, B_low, B_high
// Then compute Z0, Z2, and Z1 = (A_low+A_high)*(B_low+B_high)-Z0-Z2
template<class SharkFloatParams>
void MultiplyHelperKaratsubaV1(
    const HpSharkFloat<SharkFloatParams> *A,
    const HpSharkFloat<SharkFloatParams> *B,
    HpSharkFloat<SharkFloatParams> *Out,
    uint64_t * /*carryOuts_phase3*/, // Unused
    uint64_t * /*carryOuts_phase6*/, // Unused
    uint64_t * /*carryIns*/,         // Unused
    uint64_t * /*tempProducts*/      // Unused
) {
    constexpr int N = SharkFloatParams::NumUint32;
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
    NaiveMultiply(A_low, B_low, half, Z0.data());

    // Compute Z2 = A_high * B_high (length 2*high_len)
    std::vector<uint32_t> Z2(2 * high_len, 0);
    NaiveMultiply(A_high, B_high, high_len, Z2.data());

    // Compute (A_low + A_high) and (B_low + B_high)
    // The sum arrays can have length = max(half, high_len)+1 in worst case
    int sum_len = std::max(half, high_len) + 1;
    std::vector<uint32_t> A_sum(sum_len, 0);
    std::vector<uint32_t> B_sum(sum_len, 0);

    // Add A_low and A_high
    if (high_len >= half) {
        // Copy A_high
        for (int i = 0; i < high_len; i++) A_sum[i] = A_high[i];
        uint32_t carry = AddArraysDifferentLength(A_sum.data(), high_len, A_low, half, A_sum.data());
        if (carry) A_sum[high_len] = carry;
    } else {
        // Copy A_low
        for (int i = 0; i < half; i++) A_sum[i] = A_low[i];
        uint32_t carry = AddArraysDifferentLength(A_sum.data(), half, A_high, high_len, A_sum.data());
        if (carry) A_sum[half] = carry;
    }

    // Add B_low and B_high
    if (high_len >= half) {
        for (int i = 0; i < high_len; i++) B_sum[i] = B_high[i];
        uint32_t carry = AddArraysDifferentLength(B_sum.data(), high_len, B_low, half, B_sum.data());
        if (carry) B_sum[high_len] = carry;
    } else {
        for (int i = 0; i < half; i++) B_sum[i] = B_low[i];
        uint32_t carry = AddArraysDifferentLength(B_sum.data(), half, B_high, high_len, B_sum.data());
        if (carry) B_sum[half] = carry;
    }

    // Determine actual lengths of A_sum and B_sum
    int A_sum_len = sum_len; while (A_sum_len > 1 && A_sum[A_sum_len - 1] == 0) A_sum_len--;
    int B_sum_len = sum_len; while (B_sum_len > 1 && B_sum[B_sum_len - 1] == 0) B_sum_len--;

    int max_sum_len = std::max(A_sum_len, B_sum_len);
    // Z1_temp = (A_sum * B_sum)
    std::vector<uint32_t> Z1_temp(2 * max_sum_len, 0);
    NaiveMultiply(A_sum.data(), B_sum.data(), max_sum_len, Z1_temp.data());

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
    if (shift_digits < 0) shift_digits = 0;

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

// Compare two arrays A and B of length nA and nB (zero-extend shorter one).
// Returns:
//   1 if A > B
//   0 if A == B
//  -1 if A < B
static int CompareArrays(const uint32_t *A, int A_len, const uint32_t *B, int B_len) {
    int n = std::max(A_len, B_len);
    for (int i = n - 1; i >= 0; i--) {
        uint64_t a_val = (i < A_len) ? A[i] : 0;
        uint64_t b_val = (i < B_len) ? B[i] : 0;
        if (a_val > b_val) return 1;
        if (a_val < b_val) return -1;
    }
    return 0;
}

// Subtract B from A, both at least length n, assuming A >= B (after comparison).
// Result in Res (length n).
// Returns borrow out, should be 0 if A>=B.
// If arrays are of different length, treat missing elements as 0.
static uint32_t SubtractArrays(const uint32_t *A, int A_len, const uint32_t *B, int B_len, uint32_t *Res) {
    uint64_t borrow = 0;
    int n = std::max(A_len, B_len);
    for (int i = 0; i < n; i++) {
        uint64_t a_val = (i < A_len) ? A[i] : 0;
        uint64_t b_val = (i < B_len) ? B[i] : 0;
        uint64_t diff = a_val - b_val - borrow;
        if (a_val < b_val + borrow) {
            diff += ((uint64_t)1 << 32);
            borrow = 1;
        } else {
            borrow = 0;
        }
        Res[i] = (uint32_t)diff;
    }
    return (uint32_t)borrow;
}

template<class SharkFloatParams>
void MultiplyHelperKaratsubaV2(
    const HpSharkFloat<SharkFloatParams> *A,
    const HpSharkFloat<SharkFloatParams> *B,
    HpSharkFloat<SharkFloatParams> *Out,
    uint64_t * /*carryOuts_phase3*/, // Unused
    uint64_t * /*carryOuts_phase6*/, // Unused
    uint64_t * /*carryIns*/,         // Unused
    uint64_t * /*tempProducts*/      // Unused
) {
    constexpr int N = SharkFloatParams::NumUint32;
    if constexpr (N == 1) {
        // Single-limb multiplication
        uint64_t prod = (uint64_t)A->Digits[0] * (uint64_t)B->Digits[0];
        Out->Digits[0] = (uint32_t)(prod & 0xFFFFFFFFULL);
        Out->Exponent = A->Exponent + B->Exponent;
        Out->IsNegative = (A->IsNegative ^ B->IsNegative);
        return;
    }

    int half = N / 2;
    int high_len = N - half;

    const uint32_t *A_low = A->Digits;
    const uint32_t *A_high = A->Digits + half;
    const uint32_t *B_low = B->Digits;
    const uint32_t *B_high = B->Digits + half;

    // Compute Z0 = A_low * B_low
    std::vector<uint32_t> Z0(2 * half, 0);
    NaiveMultiply(A_low, B_low, half, Z0.data());

    // Compute Z2 = A_high * B_high
    std::vector<uint32_t> Z2(2 * high_len, 0);
    NaiveMultiply(A_high, B_high, high_len, Z2.data());

    // Compute differences: x_diff = A_1 - A_0, y_diff = B_1 - B_0
    int diff_len = std::max(half, high_len);
    std::vector<uint32_t> x_diff(diff_len, 0);
    std::vector<uint32_t> y_diff(diff_len, 0);

    int x_cmp = CompareArrays(A_high, high_len, A_low, half);
    bool x_diff_neg = false;
    if (x_cmp >= 0) {
        SubtractArrays(A_high, high_len, A_low, half, x_diff.data());
    } else {
        SubtractArrays(A_low, half, A_high, high_len, x_diff.data());
        x_diff_neg = true;
    }

    int y_cmp = CompareArrays(B_high, high_len, B_low, half);
    bool y_diff_neg = false;
    if (y_cmp >= 0) {
        SubtractArrays(B_high, high_len, B_low, half, y_diff.data());
    } else {
        SubtractArrays(B_low, half, B_high, high_len, y_diff.data());
        y_diff_neg = true;
    }

    // Trim trailing zeros
    auto trim = [](std::vector<uint32_t> &arr) {
        while (arr.size() > 1 && arr.back() == 0) arr.pop_back();
    };
    trim(x_diff);
    trim(y_diff);

    int x_len = (int)x_diff.size();
    int y_len = (int)y_diff.size();
    int mul_len = std::max(x_len, y_len);

    // Compute Z1_temp = |x_diff| * |y_diff|
    std::vector<uint32_t> Z1_temp(2 * mul_len, 0);
    NaiveMultiply(x_diff.data(), y_diff.data(), mul_len, Z1_temp.data());

    bool same_sign = (x_diff_neg == y_diff_neg); // If both neg or both pos, signs match

    // Compute Z1 = Z2 + Z0 ± Z1_temp depending on sign
    // Create Z1 array large enough
    std::vector<uint32_t> Z1(2 * N, 0);

    // Z1 = Z2 + Z0 first
    {
        // Add Z2 to Z1
        int z2_len = (int)Z2.size();
        for (int i = 0; i < z2_len && i < 2 * N; i++) {
            Z1[i] = Z2[i];
        }

        // Add Z0 to Z1
        int z0_len = (int)Z0.size();
        uint64_t carry = 0;
        for (int i = 0; i < 2 * N; i++) {
            uint64_t z1_val = Z1[i];
            uint64_t z0_val = (uint64_t)((i < z0_len) ? Z0[i] : 0);
            uint64_t sum = z1_val + z0_val + carry;
            Z1[i] = (uint32_t)(sum & 0xFFFFFFFFULL);
            carry = sum >> 32;
        }
        // carry beyond 2*N ignored, typically shouldn't happen
    }

    // Now add or subtract Z1_temp
    if (same_sign) {
        // Z1 = Z1 - Z1_temp
        uint64_t borrow = 0;
        int z1t_len = (int)Z1_temp.size();
        for (int i = 0; i < 2 * N; i++) {
            uint64_t z1_val = Z1[i];
            uint64_t z1t_val = (uint64_t)((i < z1t_len) ? Z1_temp[i] : 0) + borrow;
            if (z1_val < z1t_val) {
                Z1[i] = (uint32_t)((z1_val + ((uint64_t)1 << 32)) - z1t_val);
                borrow = 1;
            } else {
                Z1[i] = (uint32_t)(z1_val - z1t_val);
                borrow = 0;
            }
        }
        // If borrow remains, that would indicate a negative result, which should not occur with correct logic.
    } else {
        // Z1 = Z1 + Z1_temp
        uint64_t carry = 0;
        int z1t_len = (int)Z1_temp.size();
        for (int i = 0; i < 2 * N; i++) {
            uint64_t z1_val = Z1[i];
            uint64_t z1t_val = (uint64_t)((i < z1t_len) ? Z1_temp[i] : 0);
            uint64_t sum = z1_val + z1t_val + carry;
            Z1[i] = (uint32_t)(sum & 0xFFFFFFFFULL);
            carry = sum >> 32;
        }
        // carry beyond 2*N ignored
    }

    // Now we have Z0, Z1, Z2 computed correctly at the same scale (no shifts yet).
    // Construct the final product:
    // final = Z0 + (Z1 << (32*half)) + (Z2 << (64*half))
    // Wait, we must NOT add Z0 and Z2 again! We already formed Z1 using Z2 and Z0, but that was just for Z1.
    // Actually, we've ended with a pure Z1 that corresponds to the middle portion alone. The previous steps have given us a correct Z1.

    // IMPORTANT: Z1 now is just the middle portion alone (because we started Z1 from Z2+Z0 and then adjusted).
    // Actually, we must revert our logic slightly: The formula for final product is:
    // final = Z0 + (Z1 << (32*half)) + (Z2 << (64*half))
    // Where Z1 is as we computed. We must not have combined Z2 and Z0 into Z1 initially if we intend to follow the formula strictly.

    // Let's fix the logic: 
    // Step-by-step final combination:
    std::vector<uint32_t> product(2 * N, 0);

    // Put Z0 at offset 0
    {
        int z0_len = (int)Z0.size();
        for (int i = 0; i < z0_len && i < 2 * N; i++) {
            product[i] = Z0[i];
        }
    }

    // Add Z1 at offset half
    {
        uint64_t carry = 0;
        for (int i = 0; i < 2 * N - half; i++) {
            uint64_t sum = (uint64_t)product[i + half] + (uint64_t)Z1[i] + carry;
            product[i + half] = (uint32_t)(sum & 0xFFFFFFFFULL);
            carry = sum >> 32;
        }
    }

    // Add Z2 at offset 2*half
    {
        int z2_len = (int)Z2.size();
        uint64_t carry = 0;
        for (int i = 0; i < 2 * N - 2 * half; i++) {
            uint64_t z2_val = (uint64_t)((i < z2_len) ? Z2[i] : 0);
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

    int shift_digits = significant_digits - N;
    if (shift_digits < 0) shift_digits = 0;

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


#define ExplicitlyInstantiate(SharkFloatParams) \
    template void MultiplyHelperKaratsubaV1<SharkFloatParams>( \
        const HpSharkFloat<SharkFloatParams> *, \
        const HpSharkFloat<SharkFloatParams> *, \
        HpSharkFloat<SharkFloatParams> *, \
        uint64_t *, /*carryOuts_phase3*/ \
        uint64_t *, /*carryOuts_phase6*/ \
        uint64_t *, /*carryIns*/ \
        uint64_t * /*tempProducts*/); \
    template void MultiplyHelperKaratsubaV2<SharkFloatParams>( \
        const HpSharkFloat<SharkFloatParams> *, \
        const HpSharkFloat<SharkFloatParams> *, \
        HpSharkFloat<SharkFloatParams> *, \
        uint64_t *, /*carryOuts_phase3*/ \
        uint64_t *, /*carryOuts_phase6*/ \
        uint64_t *, /*carryIns*/ \
        uint64_t * /*tempProducts*/); \

ExplicitlyInstantiate(Test4x4SharkParams);
ExplicitlyInstantiate(Test4x2SharkParams);
ExplicitlyInstantiate(Test8x1SharkParams);
ExplicitlyInstantiate(Test128x64SharkParams);
