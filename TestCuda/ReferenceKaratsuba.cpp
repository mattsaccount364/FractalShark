#include "ReferenceKaratsuba.h"
#include "HpSharkFloat.cuh"

#include <cstdint>
#include <algorithm>
#include <cstring> // for memset
#include <vector>

// Helper: Add two arrays (length n) of 32-bit limbs, result in Res
// Returns carry (0 or 1)
static uint32_t AddArrays(const uint32_t *A, const uint32_t *B, int n, uint32_t *Res) {
    uint64_t carry = 0;
    for (int i = 0; i < n; i++) {
        uint64_t sum = (uint64_t)A[i] + (uint64_t)B[i] + carry;
        Res[i] = (uint32_t)(sum & 0xFFFFFFFFULL);
        carry = sum >> 32;
    }
    return (uint32_t)carry;
}

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

// Helper: Subtract B from A, both length n, A >= B
// Returns borrow
static uint32_t SubArrays(const uint32_t *A, const uint32_t *B, int n, uint32_t *Res) {
    uint64_t borrow = 0;
    for (int i = 0; i < n; i++) {
        uint64_t a = A[i];
        uint64_t b = (uint64_t)B[i] + borrow;
        if (a < b) {
            Res[i] = (uint32_t)((a + ((uint64_t)1 << 32)) - b);
            borrow = 1;
        } else {
            Res[i] = (uint32_t)(a - b);
            borrow = 0;
        }
    }
    return (uint32_t)borrow;
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
void MultiplyHelperKaratsuba(
    const HpSharkFloat<SharkFloatParams> *A,
    const HpSharkFloat<SharkFloatParams> *B,
    HpSharkFloat<SharkFloatParams> *Out,
    uint64_t * /*carryOuts_phase3*/, // Unused
    uint64_t * /*carryOuts_phase6*/, // Unused
    uint64_t * /*carryIns*/,         // Unused
    uint64_t * /*tempProducts*/      // Unused
) {
    constexpr int N = SharkFloatParams::NumUint32;
    if (N == 1) {
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

    // Z1 = Z1_temp - Z0 - Z2 shifted by half
    // We'll construct Z1 in a vector large enough to hold shifts
    // final result length = 2*N
    std::vector<uint32_t> Z1(2 * N, 0);

    // Start Z1 = Z1_temp
    for (int i = 0; i < 2 * max_sum_len; i++) {
        Z1[i] = Z1_temp[i];
    }

    // Z1 = Z1 - Z0
    {
        int min_len = std::min((int)Z1.size(), (int)Z0.size());
        uint32_t borrow = 0;
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
        // Propagate borrow if needed
        for (int i = min_len; borrow && i < (int)Z1.size(); i++) {
            uint64_t z1_val = Z1[i];
            if (z1_val < borrow) {
                Z1[i] = (uint32_t)((z1_val + ((uint64_t)1 << 32)) - borrow);
                borrow = 1;
            } else {
                Z1[i] = (uint32_t)(z1_val - borrow);
                borrow = 0;
            }
        }
        // If borrow still 1, result negative (shouldn't happen in correct Karatsuba logic)
    }

    // Z1 = Z1 - Z2 shifted by half
    // Z2 should be subtracted starting at index half (since that corresponds to the "middle" portion)
    {
        uint32_t borrow = 0;
        for (int i = 0; i < (int)Z2.size(); i++) {
            int idx = i + half;
            uint64_t z1_val = Z1[idx];
            uint64_t z2_val = (uint64_t)Z2[i] + borrow;
            if (z1_val < z2_val) {
                Z1[idx] = (uint32_t)((z1_val + ((uint64_t)1 << 32)) - z2_val);
                borrow = 1;
            } else {
                Z1[idx] = (uint32_t)(z1_val - z2_val);
                borrow = 0;
            }
        }
        for (int idx = half + (int)Z2.size(); borrow && idx < (int)Z1.size(); idx++) {
            uint64_t z1_val = Z1[idx];
            if (z1_val < borrow) {
                Z1[idx] = (uint32_t)((z1_val + ((uint64_t)1 << 32)) - borrow);
                borrow = 1;
            } else {
                Z1[idx] = (uint32_t)(z1_val - borrow);
                borrow = 0;
            }
        }
    }

    // Combine:
    // Result = Z0 + (Z1 << (32*half)) + (Z2 << (32*(2*half)))
    // Initialize result:
    std::vector<uint32_t> product(2 * N, 0);

    // Add Z0
    for (int i = 0; i < 2 * half; i++) {
        product[i] = Z0[i];
    }

    // Add Z1 at offset half
    {
        uint64_t carry = 0;
        for (int i = 0; i < 2 * N - half; i++) {
            uint64_t sum = (uint64_t)product[i + half] + (uint64_t)Z1[i] + carry;
            product[i + half] = (uint32_t)(sum & 0xFFFFFFFFULL);
            carry = sum >> 32;
        }
        // If carry remains, we ignore since result length fixed at 2*N
    }

    // Add Z2 at offset 2*half
    {
        uint64_t carry = 0;
        for (int i = 0; i < 2 * high_len; i++) {
            uint64_t sum = (uint64_t)product[i + 2 * half] + (uint64_t)Z2[i] + carry;
            product[i + 2 * half] = (uint32_t)(sum & 0xFFFFFFFFULL);
            carry = sum >> 32;
        }
        // If carry remains, ignore as above.
    }

    // Normalize result
    int highest_nonzero_index = 2 * N - 1;
    while (highest_nonzero_index >= 0 && product[highest_nonzero_index] == 0) highest_nonzero_index--;
    int significant_digits = highest_nonzero_index + 1;
    if (significant_digits < 1) significant_digits = 1; // at least one digit

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
