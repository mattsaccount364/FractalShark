#include "ReferenceKaratsuba.h"
#include "HpSharkFloat.cuh"

#include <cstdint>
#include <algorithm>
#include <cstring> // for memset
#include <vector>
#include <iostream>

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


static void NativeMultiply64(const uint32_t *A, const uint32_t *B, int n, uint64_t *Res) {
    // Number of partial sums
    int total_k = 2 * n - 1;
    // Initialize
    for (int k = 0; k < total_k * 2; k++) {
        Res[k] = 0ULL;
    }

    for (int k = 0; k < total_k; k++) {
        uint64_t sum_low = 0ULL;
        uint64_t sum_high = 0ULL;

        int i_start = (k < n) ? 0 : (k - (n - 1));
        int i_end = (k < n) ? k : (n - 1);

        for (int i = i_start; i <= i_end; ++i) {
            uint64_t a = A[i];
            uint64_t b = B[k - i];

            uint64_t product = a * b;

            uint64_t old_sum_low = sum_low;
            sum_low += product;
            if (sum_low < old_sum_low) {
                sum_high += 1ULL;
            }
        }

        Res[k * 2] = sum_low;
        Res[k * 2 + 1] = sum_high;
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

// NativeMultiply64 is unchanged, producing sum_low, sum_high pairs in a uint64_t array.

template<class SharkFloatParams>
void MultiplyHelperKaratsubaV2(
    const HpSharkFloat<SharkFloatParams> *A,
    const HpSharkFloat<SharkFloatParams> *B,
    HpSharkFloat<SharkFloatParams> *Out,
    uint64_t * /*carryOuts_phase3*/,
    uint64_t * /*carryOuts_phase6*/,
    uint64_t * /*carryIns*/,
    uint64_t * /*tempProducts*/
) {
    constexpr int N = SharkFloatParams::NumUint32;
    constexpr int n = (N + 1) / 2;
    static_assert((N % 2) == 0, "N must be even for this simplified logic.");

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

    int half = N / 2;

    const uint32_t *A_low = A->Digits;
    const uint32_t *A_high = A->Digits + half;
    const uint32_t *B_low = B->Digits;
    const uint32_t *B_high = B->Digits + half;

    auto CompareArrays = [&](const uint32_t *A_, const uint32_t *B_, int length) {
        // Compare from the most significant limb downward
        int result = 0;

        for (int i = length - 1; i >= 0; i--) {
            uint64_t a_val = A_[i];
            uint64_t b_val = B_[i];
            if (a_val > b_val) {
                result = 1;
                break;
            }

            if (a_val < b_val) {
                result = -1;
                break;
            }
        }

        if constexpr (SharkFloatParams::HostVerbose) {
            std::cout << "CompareArrays: Result: " << result << std::endl;
        }

        return result;
    };

    auto SubtractArrays = [&](const uint32_t *A_, const uint32_t *B_, int length, uint32_t *Res) {
        uint64_t borrow = 0;
        for (int i = 0; i < length; i++) {
            uint64_t a_val = A_[i];
            uint64_t b_val = B_[i];
            uint64_t diff = a_val - b_val - borrow;
            if (a_val < (b_val + borrow)) {
                diff += (1ULL << 32);
                borrow = 1;
            } else {
                borrow = 0;
            }
            Res[i] = (uint32_t)diff;
        }
        // Assuming A_ >= B_, no final borrow remains.

        if constexpr (SharkFloatParams::HostVerbose) {
            std::cout << "SubtractArrays: Result: " << UintArrayToHexString(Res, length) << std::endl;
        }

        return (uint32_t)borrow;
    };

    int x_cmp = CompareArrays(A_high, A_low, half);
    bool x_diff_neg = false;
    std::vector<uint32_t> x_diff(half, 0);
    if (x_cmp >= 0) {
        SubtractArrays(A_high, A_low, half, x_diff.data());
    } else {
        SubtractArrays(A_low, A_high, half, x_diff.data());
        x_diff_neg = true;
    }

    int y_cmp = CompareArrays(B_high, B_low, half);
    bool y_diff_neg = false;
    std::vector<uint32_t> y_diff(half, 0);
    if (y_cmp >= 0) {
        SubtractArrays(B_high, B_low, half, y_diff.data());
    } else {
        SubtractArrays(B_low, B_high, half, y_diff.data());
        y_diff_neg = true;
    }

    if constexpr (SharkFloatParams::HostVerbose) {
        std::cout << "x_diff: " << VectorUintToHexString(x_diff) << std::endl;
        std::cout << "y_diff: " << VectorUintToHexString(y_diff) << std::endl;
    }

    int total_k_z0 = 2 * half - 1;
    int total_k_z2 = 2 * half - 1;
    int total_k_z1t = 2 * half - 1;
    int mul_len = half;

    std::vector<uint64_t> Z0(2 * total_k_z0, 0ULL);
    NativeMultiply64(A_low, B_low, half, Z0.data());

    if constexpr (SharkFloatParams::HostVerbose) {
        std::cout << "Z0: " << VectorUintToHexString(Z0) << std::endl;
    }

    std::vector<uint64_t> Z2(2 * total_k_z2, 0ULL);
    NativeMultiply64(A_high, B_high, half, Z2.data());

    if constexpr (SharkFloatParams::HostVerbose) {
        std::cout << "Z2: " << VectorUintToHexString(Z2) << std::endl;
    }

    std::vector<uint64_t> Z1_temp(2 * total_k_z1t, 0ULL);
    NativeMultiply64((uint32_t *)x_diff.data(), (uint32_t *)y_diff.data(), mul_len, Z1_temp.data());

    if constexpr (SharkFloatParams::HostVerbose) {
        std::cout << "Z1_temp: " << VectorUintToHexString(Z1_temp) << std::endl;
    }

    auto add128 = [](
        uint64_t a_low, uint64_t a_high,
        uint64_t b_low, uint64_t b_high,
        uint64_t &result_low, uint64_t &result_high) {

            result_low = a_low + b_low;
            uint64_t carry = (result_low < a_low) ? 1 : 0;
            result_high = a_high + b_high + carry;
        };

    auto subtract128 = [](
        uint64_t a_low, uint64_t a_high,
        uint64_t b_low, uint64_t b_high,
        uint64_t &result_low, uint64_t &result_high) {

            uint64_t borrow = 0;

            // Subtract low parts
            result_low = a_low - b_low;
            borrow = (a_low < b_low) ? 1 : 0;

            // Subtract high parts with borrow
            result_high = a_high - b_high - borrow;
        };

    int z1_sign = (x_diff_neg ^ y_diff_neg) ? 1 : 0;
    int total_k_full = 2 * n - 1;

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
        add128(z2_low, z2_high, z0_low, z0_high, temp_low, temp_high);

        // Compute Z1 = (Z2+Z0) ± Z1_temp
        uint64_t z1_low, z1_high;
        if (z1_sign == 0) {
            // Z1 = (Z2 + Z0) - Z1_temp
            subtract128(temp_low, temp_high, z1t_low, z1t_high, z1_low, z1_high);
        } else {
            // Z1 = (Z2 + Z0) + Z1_temp
            add128(temp_low, temp_high, z1t_low, z1t_high, z1_low, z1_high);
        }

        Z1[idx_low] = z1_low;
        Z1[idx_high] = z1_high;
    }

    // Now Z1 matches the per-digit computation that CUDA performs.
    // The subsequent steps (carry propagation, final combination) should produce identical results to CUDA.


    if constexpr (SharkFloatParams::HostVerbose) {
        std::cout << "Z1: " << VectorUintToHexString(Z1) << std::endl;
    }

    // final=Z0+(Z1<<(32*n))+(Z2<<(64*n))
    // N even means (32*n) bits = n/2 64-bit words
    // (64*n) bits = n 64-bit words
    int z1_shift = (n / 2); // shift in 64-bit words
    int z2_shift = n;

    constexpr int total_result_digits = 2 * N + 1;
    std::vector<uint64_t> final128(total_result_digits * 2, 0ULL);

    for (int idx = 0; idx < total_result_digits; ++idx) {
        uint64_t sum_low = 0ULL;
        uint64_t sum_high = 0ULL;

        // Add Z0 if in range
        if (idx < 2 * n - 1) {
            int z0_idx = idx * 2;
            uint64_t z0_low = Z0[z0_idx];
            uint64_t z0_high = Z0[z0_idx + 1];
            add128(sum_low, sum_high, z0_low, z0_high, sum_low, sum_high);
        }

        // Add Z1 << (32*n)
        // Shifting by 32*n means skipping n digits. If idx >= n, we add Z1 digit (idx-n)
        if (idx >= n && (idx - n) < (2 * n - 1)) {
            int z1_idx = (idx - n) * 2;
            uint64_t z1_low = Z1[z1_idx];
            uint64_t z1_high = Z1[z1_idx + 1];
            add128(sum_low, sum_high, z1_low, z1_high, sum_low, sum_high);
        }

        // Add Z2 << (64*n)
        // Shifting by 64*n means skipping 2*n digits. If idx >= 2*n, we add Z2 digit (idx-2*n)
        if (idx >= 2 * n && (idx - 2 * n) < (2 * n - 1)) {
            int z2_idx = (idx - 2 * n) * 2;
            uint64_t z2_low = Z2[z2_idx];
            uint64_t z2_high = Z2[z2_idx + 1];
            add128(sum_low, sum_high, z2_low, z2_high, sum_low, sum_high);
        }

        final128[idx * 2] = sum_low;
        final128[idx * 2 + 1] = sum_high;
    }

    if constexpr (SharkFloatParams::HostVerbose) {
        std::cout << "final128 after Z2: " << VectorUintToHexString(final128) << std::endl;
    }

    // Assume final128 is arranged as pairs: final128[0], final128[1] form the first pair (sum_low, sum_high),
    // final128[2], final128[3] the second pair, and so forth.
    // Each pair corresponds to one "digit position" before final normalization.

    std::vector<uint32_t> tempDigits;
    tempDigits.reserve(total_result_digits); // At least one digit per pair
    uint64_t local_carry = 0ULL;

    for (int idx = 0; idx < total_result_digits; ++idx) {
        uint64_t sum_low = final128[idx * 2];
        uint64_t sum_high = final128[idx * 2 + 1];

        // This matches the CUDA logic:
        // new_sum_low = sum_low + local_carry
        uint64_t new_sum_low = sum_low + local_carry;
        uint64_t carry_from_low = (new_sum_low < sum_low) ? 1ULL : 0ULL;

        // new_sum_high = (sum_high << 32) + carry_from_low
        // sum_high shifted by 32 bits means sum_high * 2^32
        uint64_t new_sum_high = (sum_high << 32) + carry_from_low;

        // Extract one 32-bit digit
        uint32_t digit = (uint32_t)(new_sum_low & 0xFFFFFFFFULL);
        tempDigits.push_back(digit);

        // Compute new local_carry:
        // local_carry = (new_sum_low >> 32) + new_sum_high
        uint64_t upper_new_sum_low = new_sum_low >> 32;
        local_carry = new_sum_high + upper_new_sum_low;
    }

    // If local_carry remains, extract its digits
    while (local_carry > 0ULL) {
        uint32_t digit = (uint32_t)(local_carry & 0xFFFFFFFFULL);
        tempDigits.push_back(digit);
        local_carry >>= 32;
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
    //if (shift_digits < 0) {
    //    shift_digits = 0;
    //}

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
