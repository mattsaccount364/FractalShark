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

static void NativeMultiply64(const uint32_t *A, const uint32_t *B, int n, uint32_t *Res) {
    // Res will have length 2*n (at least).
    // We'll mimic the CUDA approach: accumulate into 64-bit and store two 64-bit parts (low,high)
    // per partial sum index.

    // Instead of producing final 32-bit words immediately, let's produce a 64-bit structure:
    // We'll store the results as pairs of 64-bit (low, high) just like CUDA does.
    // Each pair corresponds to sum_low and sum_high in the CUDA code.

    // Temporary buffer for 64-bit accumulations:
    std::vector<uint64_t> accum(2 * n, 0);
    // accum[i] will hold sum_low for partial sum i
    // We also need a place to hold sum_high. We'll store that in the same array by packing:
    // Actually, for clarity, we'll use separate arrays for sum_low and sum_high:
    std::vector<uint64_t> accum_low(2 * n, 0);
    std::vector<uint64_t> accum_high(2 * n, 0);

    for (int k = 0; k < 2 * n - 1; k++) {
        uint64_t sum_low = 0;
        uint64_t sum_high = 0;

        int i_start = std::max(0, k - (n - 1));
        int i_end = std::min(k, n - 1);

        for (int i = i_start; i <= i_end; ++i) {
            uint64_t a = A[i];
            uint64_t b = B[k - i];

            uint64_t product = a * b;

            // Accumulate the product
            uint64_t old_sum_low = sum_low;
            sum_low += product;
            if (sum_low < old_sum_low) {
                // overflow occurred in sum_low, increment sum_high
                sum_high += 1;
            }
        }

        // Store sum_low and sum_high.
        // Here we mimic what CUDA does: 
        // In CUDA code: tempProducts[idx] = sum_low; tempProducts[idx+1] = sum_high;
        // For CPU, if we just want to replicate the logic, we can store them in a vector similarly.

        // Let's store them in Res as pairs:
        // idx = k * 2 because each partial sum k stores two 64-bit words.
        // But our Res is currently expecting 32-bit results. We'll store 64-bit results into a temporary array first.
        accum_low[k] = sum_low;
        accum_high[k] = sum_high;
    }

    // Now we have each partial sum in accum_low/high as 64-bit values.
    // If we need final 32-bit output (like NaiveMultiply originally produced),
    // we must combine sum_low and sum_high back into standard 32-bit digits.

    // Each "sum_low" can be more than 32 bits, so we do final carry propagation:
    // sum_high << 32 means sum_high represent the upper 32-bits going beyond sum_low capacity.

    // Combine sum_high into sum_low:
    // Actually, sum_high here is just a count of how many times we overflowed 2^64 in sum_low.
    // In the CUDA code, sum_high was tracking overflows from sum_low (which can happen if you do a lot of additions).
    // If we followed the CUDA logic literally, sum_high increments each time sum_low overflows 64-bit.
    // But sum_low + sum_high * 2^64 can produce up to 128-bit number.

    // However, in the original CUDA code snippet, sum_high was treated as a simple increment, not large additions.
    // Let's follow CUDA logic literally:
    // The CUDA snippet:
    //   sum_low += product;
    //   if (sum_low < product) sum_high += 1;
    // sum_high increments each time 64-bit sum_low wraps around.
    // So sum_low is effectively the lower 64 bits, sum_high counts how many times we've wrapped 2^64.
    // This can represent up to 2^(64+some margin) bits total. For large n, sum_high might be large.

    // For simplicity, let's assume we must reduce them to 32-bit limbs now:
    // We'll have a large number: final_value = sum_low + (sum_high << 64).
    // But sum_high << 64 is beyond 64 bits. We must do a final carry propagation into multiple 32-bit words.

    // Actually, let's store them exactly as CUDA does:
    // Just like the CUDA snippet:
    //     tempProducts[idx] = sum_low;
    //     tempProducts[idx+1] = sum_high;
    // We'll store them similarly into Res, interpreting Res as a uint64_t array.

    for (int k = 0; k < 2 * n - 1; k++) {
        // Store sum_low and sum_high as two 32-bit words each in Res.
        // sum_low is a 64-bit value, sum_high is 64-bit as well.
        // For a fair comparison, let's do what CUDA does: store 64-bit sum_low and sum_high "as is" in Res.
        // That means Res must be large enough and used as a 64-bit storage.
        // If we must store as 32-bit words, we must split sum_low and sum_high into two 32-bit parts each.
        // sum_low: lower 32 bits go to Res[k*2], upper 32 bits (sum_low >> 32) must be added as carry...
        // This gets complicated.

        // Instead, let's mimic exactly the CUDA approach: no final splitting yet.
        // We'll store the two 64-bit words "as is" into a temporary array of 64-bit integers, then the caller can handle final steps.

        // If we must stick to Res as a 32-bit array, we can do:
        uint64_t sl = accum_low[k];
        uint64_t sh = accum_high[k];

        // store low 64 bits of sl into Res
        Res[k * 4 + 0] = (uint32_t)(sl & 0xFFFFFFFFULL);
        Res[k * 4 + 1] = (uint32_t)(sl >> 32);
        // store sh similarly
        Res[k * 4 + 2] = (uint32_t)(sh & 0xFFFFFFFFULL);
        Res[k * 4 + 3] = (uint32_t)(sh >> 32);
    }

    // Note: This converts NaiveMultiply into a function that produces a similar output format to the CUDA snippet: 
    // four 32-bit words per partial sum (two for sum_low and two for sum_high).

    // The rest of the code can then follow the CUDA logic for final combination steps.
}

static void add128(
    uint64_t a_low, uint64_t a_high,
    uint64_t b_low, uint64_t b_high,
    uint64_t &result_low, uint64_t &result_high) {

    // Add the low 64 bits
    uint64_t sum_low = a_low + b_low;
    uint64_t carry = (sum_low < a_low) ? 1ULL : 0ULL;

    // Add the high 64 bits plus the carry
    uint64_t sum_high = a_high + b_high + carry;

    result_low = sum_low;
    result_high = sum_high;
}

static void subtract128(
    uint64_t a_low, uint64_t a_high,
    uint64_t b_low, uint64_t b_high,
    uint64_t &result_low, uint64_t &result_high) {

    uint64_t borrow = 0ULL;

    // Subtract low parts
    uint64_t temp_low = a_low - b_low;
    if (a_low < b_low) {
        borrow = 1ULL;
    }

    // Subtract high parts with borrow
    uint64_t temp_high = a_high - b_high - borrow;

    result_low = temp_low;
    result_high = temp_high;
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
    constexpr int n = (N + 1) / 2;
    if constexpr (N == 1) {
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

    // Compute Z0, Z2, and Z1_temp using NativeMultiply64
    int total_k_z0 = 2 * half - 1;
    std::vector<uint32_t> Z0(4 * total_k_z0, 0);
    NativeMultiply64(A_low, B_low, half, Z0.data());

    int total_k_z2 = 2 * high_len - 1;
    std::vector<uint32_t> Z2(4 * total_k_z2, 0);
    NativeMultiply64(A_high, B_high, high_len, Z2.data());

    // Compute x_diff and y_diff
    auto CompareArrays = [&](const uint32_t *A_, int A_len, const uint32_t *B_, int B_len) {
        int length = std::max(A_len, B_len);
        for (int i = length - 1; i >= 0; i--) {
            uint64_t a_val = (i < A_len) ? A_[i] : 0;
            uint64_t b_val = (i < B_len) ? B_[i] : 0;
            if (a_val > b_val)return 1;
            if (a_val < b_val)return -1;
        }
        return 0;
        };

    auto SubtractArrays = [&](const uint32_t *A_, int A_len, const uint32_t *B_, int B_len, uint32_t *Res) {
        uint64_t borrow = 0;
        int length = std::max(A_len, B_len);
        for (int i = 0; i < length; i++) {
            uint64_t a_val = (i < A_len) ? A_[i] : 0;
            uint64_t b_val = (i < B_len) ? B_[i] : 0;
            uint64_t diff = a_val - b_val - borrow;
            if (a_val < (b_val + borrow)) {
                diff += (1ULL << 32);
                borrow = 1;
            } else borrow = 0;
            Res[i] = (uint32_t)diff;
        }
        return (uint32_t)borrow;
        };

    int x_cmp = CompareArrays(A_high, high_len, A_low, half);
    bool x_diff_neg = false;
    std::vector<uint32_t> x_diff(std::max(half, high_len), 0);
    if (x_cmp >= 0) {
        SubtractArrays(A_high, high_len, A_low, half, x_diff.data());
    } else {
        SubtractArrays(A_low, half, A_high, high_len, x_diff.data());
        x_diff_neg = true;
    }

    int y_cmp = CompareArrays(B_high, high_len, B_low, half);
    bool y_diff_neg = false;
    std::vector<uint32_t> y_diff(std::max(half, high_len), 0);
    if (y_cmp >= 0) {
        SubtractArrays(B_high, high_len, B_low, half, y_diff.data());
    } else {
        SubtractArrays(B_low, half, B_high, high_len, y_diff.data());
        y_diff_neg = true;
    }

    auto trim = [&](std::vector<uint32_t> &arr) {
        while (arr.size() > 1 && arr.back() == 0)arr.pop_back();
        };
    trim(x_diff);
    trim(y_diff);

    int x_len = (int)x_diff.size();
    int y_len = (int)y_diff.size();
    int mul_len = std::max(x_len, y_len);

    int total_k_z1t = 2 * mul_len - 1;
    std::vector<uint32_t> Z1_temp(4 * total_k_z1t, 0);
    NativeMultiply64(x_diff.data(), y_diff.data(), mul_len, Z1_temp.data());

    int z1_sign = (x_diff_neg ^ y_diff_neg) ? 1 : 0;

    int total_k_full = 2 * n - 1;
    auto padTo = [&](std::vector<uint32_t> &arr, int old_k) {
        if (old_k < total_k_full) {
            arr.resize(4 * total_k_full, 0);
        }
        };
    padTo(Z0, total_k_z0);
    padTo(Z2, total_k_z2);
    padTo(Z1_temp, total_k_z1t);

    // Compute Z1=(Z2+Z0) ± Z1_temp
    std::vector<uint32_t> Z1(4 * total_k_full, 0);
    for (int k = 0; k < total_k_full; k++) {
        uint64_t z0_low = (uint64_t)Z0[k * 4 + 0] | ((uint64_t)Z0[k * 4 + 1] << 32);
        uint64_t z0_high = (uint64_t)Z0[k * 4 + 2] | ((uint64_t)Z0[k * 4 + 3] << 32);

        uint64_t z2_low = (uint64_t)Z2[k * 4 + 0] | ((uint64_t)Z2[k * 4 + 1] << 32);
        uint64_t z2_high = (uint64_t)Z2[k * 4 + 2] | ((uint64_t)Z2[k * 4 + 3] << 32);

        uint64_t z1t_low = (uint64_t)Z1_temp[k * 4 + 0] | ((uint64_t)Z1_temp[k * 4 + 1] << 32);
        uint64_t z1t_high = (uint64_t)Z1_temp[k * 4 + 2] | ((uint64_t)Z1_temp[k * 4 + 3] << 32);

        uint64_t temp_low, temp_high;
        add128(z2_low, z2_high, z0_low, z0_high, temp_low, temp_high);

        uint64_t z1_low, z1_high;
        if (z1_sign == 0) {
            // same sign: Z1=(Z2+Z0)-Z1_temp
            subtract128(temp_low, temp_high, z1t_low, z1t_high, z1_low, z1_high);
        } else {
            add128(temp_low, temp_high, z1t_low, z1t_high, z1_low, z1_high);
        }

        Z1[k * 4 + 0] = (uint32_t)(z1_low & 0xFFFFFFFFULL);
        Z1[k * 4 + 1] = (uint32_t)(z1_low >> 32);
        Z1[k * 4 + 2] = (uint32_t)(z1_high & 0xFFFFFFFFULL);
        Z1[k * 4 + 3] = (uint32_t)(z1_high >> 32);
    }

    int total_result_digits = 2 * N + 1;
    // We'll accumulate the final result in a large temp array of 32-bit words
    // Each idx corresponds to a partial sum. We must now incorporate sum_high properly.

    // Re-interpret final sums:
    // final(idx)=Z0[idx] + Z1[idx-n]*(if valid) + Z2[idx-2*n]*(if valid)
    // We have them in form of 64-bit pairs. Let's just trust we computed sum_low_arr, sum_high_arr steps incorrectly before.
    // Instead, do direct 128-bit addition as we intended:
    // Actually, we must replicate final combination from scratch:
    // final = Z0 + (Z1 << (32*n)) + (Z2 << (64*n))

    // We'll store final result in a big array of 32-bit words (tempDigits),
    // indexed by idx for partial sums. But now we must place each partial sum at correct offsets:
    // Actually, each partial sum index 'idx' we used before was a direct approach.
    // Let's keep the same indexing as before:
    std::vector<uint64_t> sum_low_arr(total_result_digits, 0);
    std::vector<uint64_t> sum_high_arr(total_result_digits, 0);

    // We'll do the same addition logic as before:
    for (int idx = 0; idx < total_result_digits; idx++) {
        uint64_t sum_low = 0, sum_high = 0;

        // Add Z0 if idx<2*n-1
        if (idx < 2 * n - 1) {
            uint64_t zl = (uint64_t)Z0[idx * 4 + 0] | ((uint64_t)Z0[idx * 4 + 1] << 32);
            uint64_t zh = (uint64_t)Z0[idx * 4 + 2] | ((uint64_t)Z0[idx * 4 + 3] << 32);
            add128(sum_low, sum_high, zl, zh, sum_low, sum_high);
        }

        // Add Z1 if idx>=n && (idx-n)<2*n-1
        if (idx >= n && (idx - n) < 2 * n - 1) {
            uint64_t zl = (uint64_t)Z1[(idx - n) * 4 + 0] | ((uint64_t)Z1[(idx - n) * 4 + 1] << 32);
            uint64_t zh = (uint64_t)Z1[(idx - n) * 4 + 2] | ((uint64_t)Z1[(idx - n) * 4 + 3] << 32);
            add128(sum_low, sum_high, zl, zh, sum_low, sum_high);
        }

        // Add Z2 if idx>=2*n && (idx-2*n)<2*n-1
        if (idx >= 2 * n && (idx - 2 * n) < 2 * n - 1) {
            uint64_t zl = (uint64_t)Z2[(idx - 2 * n) * 4 + 0] | ((uint64_t)Z2[(idx - 2 * n) * 4 + 1] << 32);
            uint64_t zh = (uint64_t)Z2[(idx - 2 * n) * 4 + 2] | ((uint64_t)Z2[(idx - 2 * n) * 4 + 3] << 32);
            add128(sum_low, sum_high, zl, zh, sum_low, sum_high);
        }

        sum_low_arr[idx] = sum_low;
        sum_high_arr[idx] = sum_high;
    }

    // After computing sum_low_arr and sum_high_arr for all partial sums:

    std::vector<uint32_t> tempDigits(2 * N + 4, 0);
    uint64_t carry = 0;

    for (int idx = 0; idx < 2 * N; idx++) {
        uint64_t sl = sum_low_arr[idx];
        uint64_t sh = sum_high_arr[idx];

        // Break down sl into two 32-bit words:
        uint64_t sl_lo = (uint64_t)(uint32_t)(sl & 0xFFFFFFFFULL);
        uint64_t sl_hi = (uint64_t)(uint32_t)(sl >> 32);

        // Break down sh into two 32-bit words:
        uint64_t sh_lo = (uint64_t)(uint32_t)(sh & 0xFFFFFFFFULL);
        uint64_t sh_hi = (uint64_t)(uint32_t)(sh >> 32);

        // Add word0 (sl_lo) to tempDigits[idx]
        {
            uint64_t val = (uint64_t)tempDigits[idx] + sl_lo;
            tempDigits[idx] = (uint32_t)(val & 0xFFFFFFFFULL);
            carry = val >> 32;
        }
        // Propagate carry immediately to next digit
        if (carry > 0) {
            uint64_t val = (uint64_t)tempDigits[idx + 1] + carry;
            tempDigits[idx + 1] = (uint32_t)(val & 0xFFFFFFFFULL);
            carry = val >> 32;
        }

        // Add word1 (sl_hi) to tempDigits[idx+1]
        {
            uint64_t val = (uint64_t)tempDigits[idx + 1] + sl_hi;
            tempDigits[idx + 1] = (uint32_t)(val & 0xFFFFFFFFULL);
            uint64_t c = val >> 32;
            // Add carry from previous step if any
            if (c > 0) {
                uint64_t val2 = (uint64_t)tempDigits[idx + 2] + c;
                tempDigits[idx + 2] = (uint32_t)(val2 & 0xFFFFFFFFULL);
                c = val2 >> 32;
                if (c > 0) {
                    uint64_t val3 = (uint64_t)tempDigits[idx + 3] + c;
                    tempDigits[idx + 3] = (uint32_t)(val3 & 0xFFFFFFFFULL);
                }
            }
        }

        // Add word2 (sh_lo) to tempDigits[idx+2]
        {
            uint64_t val = (uint64_t)tempDigits[idx + 2] + sh_lo;
            tempDigits[idx + 2] = (uint32_t)(val & 0xFFFFFFFFULL);
            uint64_t c = val >> 32;
            if (c > 0) {
                uint64_t val2 = (uint64_t)tempDigits[idx + 3] + c;
                tempDigits[idx + 3] = (uint32_t)(val2 & 0xFFFFFFFFULL);
                c = val2 >> 32;
                if (c > 0 && (idx + 4) < (int)tempDigits.size()) {
                    uint64_t val3 = (uint64_t)tempDigits[idx + 4] + c;
                    tempDigits[idx + 4] = (uint32_t)(val3 & 0xFFFFFFFFULL);
                }
            }
        }

        // Add word3 (sh_hi) to tempDigits[idx+3]
        {
            uint64_t val = (uint64_t)tempDigits[idx + 3] + sh_hi;
            tempDigits[idx + 3] = (uint32_t)(val & 0xFFFFFFFFULL);
            uint64_t c = val >> 32;
            // If c>0 and idx+4<tempDigits.size(), propagate further if needed
            if (c > 0 && (idx + 4) < (int)tempDigits.size()) {
                uint64_t val2 = (uint64_t)tempDigits[idx + 4] + c;
                tempDigits[idx + 4] = (uint32_t)(val2 & 0xFFFFFFFFULL);
                // ... continue if more carry
            }
        }
    }

    // Now a global carry propagation pass over tempDigits:
    uint64_t c = 0;
    for (int i = 0; i < (int)tempDigits.size(); i++) {
        uint64_t val = (uint64_t)tempDigits[i] + c;
        tempDigits[i] = (uint32_t)(val & 0xFFFFFFFFULL);
        c = val >> 32;
    }
    // If c>0, handle final carry if needed (expand array or adjust exponent)

    // Find highest_nonzero_index:
    int highest_nonzero_index = (int)tempDigits.size() - 1;
    while (highest_nonzero_index >= 0 && tempDigits[highest_nonzero_index] == 0) highest_nonzero_index--;
    int significant_digits = highest_nonzero_index + 1;
    if (significant_digits < 1) significant_digits = 1;

    int shift_digits = significant_digits - N;
    if (shift_digits < 0) shift_digits = 0;
    Out->Exponent = A->Exponent + B->Exponent + shift_digits * 32;

    for (int i = 0; i < N; i++) {
        int src_idx = i + shift_digits;
        uint32_t val = (src_idx < significant_digits) ? tempDigits[src_idx] : 0;
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
