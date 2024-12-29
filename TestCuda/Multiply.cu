#include "Multiply.cuh"

#include <cuda_runtime.h>

#include "HpSharkFloat.cuh"
#include "BenchmarkTimer.h"

#include <iostream>
#include <vector>
#include <gmp.h>
#include <cstring>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <algorithm>

#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda/barrier>
namespace cg = cooperative_groups;

__device__ int compareDigits(const uint32_t *a, const uint32_t *b, int n) {
    for (int i = n - 1; i >= 0; --i) {
        if (a[i] > b[i]) return 1;
        if (a[i] < b[i]) return -1;
    }
    return 0;
}


// Device function to compute carry-in signals by shifting carry-out signals right by one
__device__ static void ComputeCarryInWarp(
    const uint32_t *carryOuts, // Input carry-out array
    uint32_t *carryIn,         // Output carry-in array
    int n) {                   // Number of digits

    int lane = threadIdx.x; // Thread index within the warp
    unsigned int mask = 0xFFFFFFFF; // Full mask for synchronization

    // Load the carry-out value for this digit
    uint32_t carry_out = (lane < n) ? carryOuts[lane] : 0;

    // Shift carry-out values up by one to compute carry-in
    uint32_t shifted_carry = __shfl_up_sync(mask, carry_out, 1);

    // The first digit has no carry-in
    if (lane == 0) {
        shifted_carry = 0;
    }

    // Store the carry-in value
    if (lane < n) {
        carryIn[lane] = shifted_carry;
    }
}

// Device function to compute carry-in signals across warps using statically allocated shared memory
__device__ static void ComputeCarryIn(
    cg::thread_block &block,
    const uint32_t *carryOuts, // Input carry-out array
    uint32_t *carryIn,         // Output carry-in array
    int n,                     // Number of digits
    uint32_t *sharedCarryOut   // Statically allocated sharedCarryOut array (size SharkFloatParams::ThreadsPerBlock)
) {
    int tid = threadIdx.x;

    // Step 1: Store carryOuts into shared memory
    if (tid < n) {
        sharedCarryOut[tid] = carryOuts[tid];
    }

    // Ensure all carryOuts are written before proceeding
    block.sync();

    // Step 2: Compute carryIn
    if (tid < n) {
        if (tid == 0) {
            // No carry-in for the first digit
            carryIn[tid] = 0;
        } else {
            // Carry-in is the carry-out from the previous digit
            carryIn[tid] = sharedCarryOut[tid - 1];
        }
    }

    // No need for further synchronization as carryIn is now computed
}

// Device function to compute carry-in signals across warps using statically allocated shared memory
template<class SharkFloatParams>
__device__ void ComputeCarryInDecider(
    cg::thread_block &block,
    const uint32_t *carryOuts, // Input carry-out array
    uint32_t *carryIn,         // Output carry-in array
    int n,                     // Number of digits
    uint32_t *sharedCarryOut   // Statically allocated sharedCarryOut array (size SharkFloatParams::ThreadsPerBlock)
) {
    if constexpr (SharkFloatParams::ThreadsPerBlock <= 32) {
        ComputeCarryInWarp(carryOuts, carryIn, n);
    } else {
        ComputeCarryIn(block, carryOuts, carryIn, n, sharedCarryOut);
    }
}

__device__ static void subtractDigits(const uint32_t *a, const uint32_t *b, uint32_t *result, int n) {
    uint64_t borrow = 0;
    for (int i = 0; i < n; ++i) {
        uint64_t ai = a[i];
        uint64_t bi = b[i];
        uint64_t temp = ai - bi - borrow;
        if (ai < bi + borrow) {
            borrow = 1;
            temp += ((uint64_t)1 << 32);
        } else {
            borrow = 0;
        }
        result[i] = (uint32_t)temp;
    }
}

#include <cooperative_groups.h>
namespace cg = cooperative_groups;

/**
 * Parallel subtraction (a - b), stored in result, using a multi-pass approach
 * to propagate borrows.
 *
 * The function attempts to subtract each digit of 'b' from 'a' in parallel,
 * then uses repeated passes (do/while) to handle newly introduced borrows
 * until no more remain or a maximum pass count is reached.
 */
template<class SharkFloatParams>
__device__ __forceinline__ void SubtractDigitsParallel(
    uint32_t *__restrict__ shared_data,
    const uint32_t *__restrict__ a,
    const uint32_t *__restrict__ b,
    uint32_t *__restrict__ subtractionBorrows,
    uint32_t *__restrict__ subtractionBorrows2,
    uint32_t *__restrict__ result,
    uint32_t *__restrict__ globalBorrowAny,
    cg::grid_group &grid,
    cg::thread_block &block
) {
    // Constants 
    constexpr int N = SharkFloatParams::NumUint32;
    constexpr int n = (N + 1) / 2;     // 'n' is how many digits we handle in half
    constexpr int MaxPasses = 10;      // maximum number of multi-pass sweeps
    constexpr int threadsPerBlock = SharkFloatParams::ThreadsPerBlock;
    constexpr int numBlocks = SharkFloatParams::NumBlocks;

    // Identify the block/thread
    const int tid = static_cast<int>(threadIdx.x);
    const int blockId = static_cast<int>(blockIdx.x);

    // Determine which slice of digits this block will handle
    // Example: we split 'n' digits evenly among numBlocks
    constexpr int digitsPerBlock = (n + numBlocks - 1) / numBlocks;
    const int startDigit = blockId * digitsPerBlock;
    const int endDigit = min(startDigit + digitsPerBlock, n);
    const int digitsInBlock = endDigit - startDigit;

    const int startIdx = startDigit + tid;
    const int endIdx = startDigit + digitsInBlock;


    // Pointers in shared memory
    // We'll store partial differences and borrow bits here
    //uint32_t *a_shared = shared_data;                       // [0 .. digitsInBlock-1]
    //uint32_t *b_shared = a_shared + digitsInBlock;          // [digitsInBlock .. 2*digitsInBlock-1]
    //uint32_t *partialDiff = b_shared + digitsInBlock;          // ...
    //uint32_t *borrowBits = partialDiff + digitsInBlock;
    // optionally you can do more arrays for second pass or prefix sums, etc.

    // 1) Load 'a' and 'b' digits into shared memory
    //for (int idx = tid; idx < digitsInBlock; idx += threadsPerBlock) {
    //    a_shared[idx] = a[startDigit + idx];
    //    b_shared[idx] = b[startDigit + idx];
    //}
    //block.sync();
    //grid.sync();

    // 2) First pass: compute naive partial difference (a[i] - b[i]), store in partialDiff
    //    and whether a[i]<b[i] => borrowBit=1 else 0
    for (int idx = startIdx; idx < endIdx; idx += threadsPerBlock) {

        uint32_t ai = a[idx];
        uint32_t bi = b[idx];

        // naive difference
        uint64_t temp = static_cast<uint64_t>(ai) - static_cast<uint64_t>(bi);
        uint32_t borrow = (ai < bi) ? 1u : 0u;

        result[idx] = static_cast<uint32_t>(temp & 0xFFFFFFFFu);
        subtractionBorrows[idx] = borrow;
    }

    // We'll do repeated passes to fix newly introduced borrows
    // We'll store in carryOuts[blockId] any leftover borrow from this block if needed
    // but let's keep it simple: we do an in-block multi-pass approach first

    int pass = 0;
    uint64_t local_borrow = 0;

    uint32_t *curBorrow = subtractionBorrows;
    uint32_t *newBorrow = subtractionBorrows2;

    do {
        // Synchronize all blocks
        grid.sync();

        // Zero out the global carry count for the current pass
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            *globalBorrowAny = 0;
        }

        grid.sync();

        // Each thread processes its assigned digits
        for (int idx = startIdx; idx < endIdx; idx += threadsPerBlock) {
            // Get carry-in from the previous digit
            local_borrow = 0;
            if (idx > 0) {
                local_borrow = curBorrow[idx - 1];
            }

            // Read the previously stored digit
            uint32_t digit = result[idx];

            // Add local_borrow to digit
            uint64_t sum = static_cast<uint64_t>(digit) - local_borrow;

            // Update digit
            digit = static_cast<uint32_t>(sum & 0xFFFFFFFFULL);
            result[idx] = digit;

            // Create new borrow
            if (sum & 0x8000'0000'0000'0000) {
                newBorrow[idx] = 1;
                atomicAdd(globalBorrowAny, 1);
            } else {
                newBorrow[idx] = 0;
            }
        }

        grid.sync();

        // If no carries remain, exit the loop
        if (*globalBorrowAny == 0) {
            break;
        }

        // Swap newBorrow and curBorrow
        std::swap(curBorrow, newBorrow);

        pass++;
    } while (pass < MaxPasses);
}



// Function to perform addition with carry
__device__ __forceinline__ static void add128(
    uint64_t a_low, uint64_t a_high,
    uint64_t b_low, uint64_t b_high,
    uint64_t &result_low, uint64_t &result_high) {

    result_low = a_low + b_low;
    uint64_t carry = (result_low < a_low) ? 1 : 0;
    result_high = a_high + b_high + carry;
}

__device__ __forceinline__ static void subtract128(
    uint64_t a_low, uint64_t a_high,
    uint64_t b_low, uint64_t b_high,
    uint64_t &result_low, uint64_t &result_high) {

    uint64_t borrow = 0;

    // Subtract low parts
    result_low = a_low - b_low;
    borrow = (a_low < b_low) ? 1 : 0;

    // Subtract high parts with borrow
    result_high = a_high - b_high - borrow;
}

template<class SharkFloatParams>
__device__ __forceinline__ static void CarryPropagation (
    uint64_t *__restrict__ shared_carries,
    cg::grid_group &grid,
    cg::thread_block &block,
    const uint3 &threadIdx,
    const uint3 &blockIdx,
    const uint3 &blockDim,
    const uint3 &gridDim,
    int thread_start_idx,
    int thread_end_idx,
    int Convolution_offset,
    int Result_offset,
    uint64_t * __restrict__ block_carry_outs,
    uint64_t * __restrict__ tempProducts,
    uint64_t * __restrict__ globalCarryCheck) {

    // First Pass: Process convolution results to compute initial digits and local carries
    // Initialize local carry
    uint64_t local_carry = 0;

    // Constants and offsets
    constexpr int MaxPasses = 10; // Maximum number of carry propagation passes
    constexpr int total_result_digits = 2 * SharkFloatParams::NumUint32;

    uint64_t *carries_remaining_global = globalCarryCheck;

    for (int idx = thread_start_idx; idx < thread_end_idx; ++idx) {
        int sum_low_idx = Convolution_offset + idx * 2;
        int sum_high_idx = sum_low_idx + 1;

        uint64_t sum_low = tempProducts[sum_low_idx];     // Lower 64 bits
        uint64_t sum_high = tempProducts[sum_high_idx];   // Higher 64 bits

        // Add local carry to sum_low
        bool new_sum_low_negative = false;
        uint64_t new_sum_low = sum_low + local_carry;

        // Extract one 32-bit digit from new_sum_low
        auto digit = static_cast<uint32_t>(new_sum_low & 0xFFFFFFFFULL);
        tempProducts[Result_offset + idx] = digit;

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

    if (threadIdx.x == SharkFloatParams::ThreadsPerBlock - 1) {
        block_carry_outs[blockIdx.x] = local_carry;
    } else {
        shared_carries[threadIdx.x] = local_carry;
    }

    // Inter-Block Carry Propagation
    int pass = 0;

    do {
        // Synchronize all blocks
        grid.sync();

        // Zero out the global carry count for the current pass
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            *carries_remaining_global = 0;
        }

        // Get carry-in from the previous block
        local_carry = 0;
        if (threadIdx.x == 0 && blockIdx.x > 0) {
            local_carry = block_carry_outs[blockIdx.x - 1];
        } else {
            if (threadIdx.x > 0) {
                local_carry = shared_carries[threadIdx.x - 1];
            }
        }

        // Each thread processes its assigned digits
        bool local_carry_negative = false;
        for (int idx = thread_start_idx; idx < thread_end_idx; ++idx) {
            // Read the previously stored digit
            uint32_t digit = tempProducts[Result_offset + idx];

            // Add local_carry to digit
            uint64_t sum = static_cast<uint64_t>(digit) + local_carry;
            if (local_carry_negative) {
                // Clear high order 32 bits of sum:
                sum &= 0x0000'0000'FFFF'FFFF;
            }

            // Update digit
            digit = static_cast<uint32_t>(sum & 0xFFFFFFFFULL);
            tempProducts[Result_offset + idx] = digit;

            // Compute new local_carry for next digit
            local_carry_negative = ((local_carry & (1ULL << 63)) != 0);
            local_carry = sum >> 32;
        }

        shared_carries[threadIdx.x] = local_carry;
        block.sync();

        // The block's carry-out is the carry from the last thread
        auto temp = shared_carries[threadIdx.x];
        if (threadIdx.x == SharkFloatParams::ThreadsPerBlock - 1) {
            block_carry_outs[blockIdx.x] = temp;
        }

        if (temp != 0) {
            atomicAdd(carries_remaining_global, 1);
        }

        // Synchronize all blocks before checking if carries remain
        grid.sync();

        // If no carries remain, exit the loop
        if (*carries_remaining_global == 0) {
            break;
        }

        pass++;
    } while (pass < MaxPasses);

    // ---- Handle Final Carry-Out ----

    // Handle final carry-out
    if (threadIdx.x == 0 && blockIdx.x == gridDim.x - 1) {
        uint64_t final_carry = block_carry_outs[blockIdx.x];
        if (final_carry > 0) {
            // Store the final carry as an additional digit
            tempProducts[Result_offset + total_result_digits] = static_cast<uint32_t>(final_carry & 0xFFFFFFFFULL);
            // Optionally, you may need to adjust total_result_digits
        }
    }

    // Synchronize all blocks before finalization
    // grid.sync();
}


//
// static constexpr int32_t SharkFloatParams::ThreadsPerBlock = /* power of 2 */;
// static constexpr int32_t SharkFloatParams::NumBlocks = /* power of 2 */;
// static constexpr int32_t SharkFloatParams::NumUint32 = SharkFloatParams::ThreadsPerBlock * SharkFloatParams::NumBlocks;
// 

// Assuming that SharkFloatParams::NumUint32 can be large and doesn't fit in shared memory
// We'll use the provided global memory buffers for large intermediates
template<class SharkFloatParams>
__device__ void MultiplyHelperKaratsubaV2(
    const HpSharkFloat<SharkFloatParams> *__restrict__ A,
    const HpSharkFloat<SharkFloatParams> *__restrict__ B,
    HpSharkFloat<SharkFloatParams> *__restrict__ Out,
    cg::grid_group grid,
    uint64_t *__restrict__ tempProducts) {

    cg::thread_block block = cg::this_thread_block();

    const int threadIdxGlobal = blockIdx.x * SharkFloatParams::ThreadsPerBlock + threadIdx.x;

    constexpr int total_threads = SharkFloatParams::ThreadsPerBlock * SharkFloatParams::NumBlocks;
    constexpr int N = SharkFloatParams::NumUint32;         // Total number of digits
    constexpr int n = (N + 1) / 2;              // Half of N

    // Constants for tempProducts offsets
    constexpr int Z0_offset = 0;
    constexpr int Z2_offset = Z0_offset + 4 * N;
    constexpr int Z1_temp_offset = Z2_offset + 4 * N;
    constexpr int Z1_offset = Z1_temp_offset + 4 * N;
    constexpr int Convolution_offset = Z1_offset + 4 * N;
    constexpr int Result_offset = Convolution_offset + 4 * N;
    constexpr int XDiff_offset = Result_offset + 2 * N;
    constexpr int YDiff_offset = XDiff_offset + 1 * N;
    constexpr int GlobalCarryOffset = YDiff_offset + 1 * N;
    constexpr int CarryInsOffset = GlobalCarryOffset + 1 * N;
    constexpr int CarryInsOffset2 = CarryInsOffset + 1 * N;
    constexpr int BorrowAnyOffset = CarryInsOffset2 + 1 * N;

    extern __shared__ uint32_t shared_data[];
    auto *__restrict__ a_shared = shared_data;
    auto *__restrict__ b_shared = a_shared + N;

    cg::memcpy_async(block, a_shared, A->Digits, sizeof(uint32_t) * N);
    cg::memcpy_async(block, b_shared, B->Digits, sizeof(uint32_t) * N);

    // Wait for the first batch of A to be loaded
    cg::wait(block);

    // Common variables for convolution loops
    constexpr int total_k = 2 * n - 1; // Total number of k values

    int k_start = (threadIdxGlobal * total_k) / total_threads;
    int k_end = ((threadIdxGlobal + 1) * total_k) / total_threads;

    // ---- Convolution for Z0 = A0 * B0 ----
    for (int k = k_start; k < k_end; ++k) {
        uint64_t sum_low = 0;
        uint64_t sum_high = 0;

        int i_start = max(0, k - (n - 1));
        int i_end = min(k, n - 1);

        for (int i = i_start; i <= i_end; ++i) {
            uint64_t a = a_shared[i]; //A_shared[i];         // A0[i]
            uint64_t b = b_shared[k - i]; //B_shared[k - i];     // B0[k - i]

            uint64_t product = a * b;

            // Add product to sum
            sum_low += product;
            if (sum_low < product) {
                sum_high += 1;
            }
        }

        // Store sum_low and sum_high in tempProducts
        int idx = Z0_offset + k * 2;
        tempProducts[idx] = sum_low;
        tempProducts[idx + 1] = sum_high;
    }

    // Synchronize before next convolution
    //block.sync();

    // ---- Convolution for Z2 = A1 * B1 ----
    for (int k = k_start; k < k_end; ++k) {
        uint64_t sum_low = 0;
        uint64_t sum_high = 0;

        int i_start = max(0, k - (n - 1));
        int i_end = min(k, n - 1);

        for (int i = i_start; i <= i_end; ++i) {
            uint64_t a = a_shared[i + n]; // A_shared[i];         // A1[i]
            uint64_t b = b_shared[k - i + n]; // B_shared[k - i];     // B1[k - i]

            uint64_t product = a * b;

            // Add product to sum
            sum_low += product;
            if (sum_low < product) {
                sum_high += 1;
            }
        }

        // Store sum_low and sum_high in tempProducts
        int idx = Z2_offset + k * 2;
        tempProducts[idx] = sum_low;
        tempProducts[idx + 1] = sum_high;
    }

    // No sync

    // ---- Compute Differences x_diff = A1 - A0 and y_diff = B1 - B0 ----

    constexpr int total_result_digits = 2 * N + 1;
    constexpr auto digits_per_block = SharkFloatParams::ThreadsPerBlock * 2;
    auto block_start_idx = blockIdx.x * digits_per_block;
    auto block_end_idx = min(block_start_idx + digits_per_block, total_result_digits);

    int digits_per_thread = (digits_per_block + blockDim.x - 1) / blockDim.x;

    int thread_start_idx = block_start_idx + threadIdx.x * digits_per_thread;
    int thread_end_idx = min(thread_start_idx + digits_per_thread, block_end_idx);

    // Arrays to hold the absolute differences (size n)
    auto * __restrict__ x_diff_abs = reinterpret_cast<uint32_t *>(&tempProducts[XDiff_offset]);
    auto * __restrict__ y_diff_abs = reinterpret_cast<uint32_t *>(&tempProducts[YDiff_offset]);
    int x_diff_sign = 0; // 0 if positive, 1 if negative
    int y_diff_sign = 0; // 0 if positive, 1 if negative

    // Compute x_diff_abs and x_diff_sign
    auto * __restrict__ subtractionBorrows = reinterpret_cast<uint32_t *>(&tempProducts[CarryInsOffset]);
    auto * __restrict__ subtractionBorrows2 = reinterpret_cast<uint32_t *>(&tempProducts[CarryInsOffset2]);

    constexpr bool useParallelSubtract = true;

    if constexpr (!SharkFloatParams::DisableSubtraction) {
        if constexpr (useParallelSubtract) {
            uint32_t *globalBorrowAny = reinterpret_cast<uint32_t *>(&tempProducts[BorrowAnyOffset]);
            int x_compare = compareDigits(a_shared + n, a_shared, n);

            if (x_compare >= 0) {
                x_diff_sign = 0;
                SubtractDigitsParallel<SharkFloatParams>(
                    shared_data,
                    a_shared + n,
                    a_shared,
                    subtractionBorrows,
                    subtractionBorrows2,
                    x_diff_abs,
                    globalBorrowAny,
                    grid,
                    block); // x_diff = A1 - A0
            } else {
                x_diff_sign = 1;
                SubtractDigitsParallel<SharkFloatParams>(
                    shared_data,
                    a_shared,
                    a_shared + n,
                    subtractionBorrows,
                    subtractionBorrows2,
                    x_diff_abs,
                    globalBorrowAny,
                    grid,
                    block); // x_diff = A0 - A1
            }

            // Compute y_diff_abs and y_diff_sign
            int y_compare = compareDigits(b_shared + n, b_shared, n);
            if (y_compare >= 0) {
                y_diff_sign = 0;
                SubtractDigitsParallel<SharkFloatParams>(
                    shared_data,
                    b_shared + n,
                    b_shared,
                    subtractionBorrows,
                    subtractionBorrows2,
                    y_diff_abs,
                    globalBorrowAny,
                    grid,
                    block); // y_diff = B1 - B0
            } else {
                y_diff_sign = 1;
                SubtractDigitsParallel<SharkFloatParams>(
                    shared_data,
                    b_shared,
                    b_shared + n,
                    subtractionBorrows,
                    subtractionBorrows2,
                    y_diff_abs,
                    globalBorrowAny,
                    grid,
                    block); // y_diff = B0 - B1
            }
        } else {
            int x_compare = compareDigits(a_shared + n, a_shared, n);

            if (x_compare >= 0) {
                x_diff_sign = 0;
                subtractDigits(a_shared + n, a_shared, x_diff_abs, n); // x_diff = A1 - A0
            } else {
                x_diff_sign = 1;
                subtractDigits(a_shared, a_shared + n, x_diff_abs, n); // x_diff = A0 - A1
            }

            // Compute y_diff_abs and y_diff_sign
            int y_compare = compareDigits(b_shared + n, b_shared, n);
            if (y_compare >= 0) {
                y_diff_sign = 0;
                subtractDigits(b_shared + n, b_shared, y_diff_abs, n); // y_diff = B1 - B0
            } else {
                y_diff_sign = 1;
                subtractDigits(b_shared, b_shared + n, y_diff_abs, n); // y_diff = B0 - B1
            }
        }
    }

    // Determine the sign of Z1_temp
    int z1_sign = x_diff_sign ^ y_diff_sign;


    // Synchronize before convolution
    grid.sync();

    // Replace A and B in shared memory with their absolute differences
    cg::memcpy_async(block, a_shared, x_diff_abs, sizeof(uint32_t) * n);
    cg::memcpy_async(block, b_shared, y_diff_abs, sizeof(uint32_t) * n);

    // Wait for the first batch of A to be loaded
    cg::wait(block);

    // ---- Convolution for Z1_temp = |x_diff| * |y_diff| ----
    // Update total_k for convolution of differences
    int total_k_diff = 2 * n - 1;

    int k_diff_start = (threadIdxGlobal * total_k_diff) / total_threads;
    int k_diff_end = ((threadIdxGlobal + 1) * total_k_diff) / total_threads;

    for (int k = k_diff_start; k < k_diff_end; ++k) {
        uint64_t sum_low = 0;
        uint64_t sum_high = 0;

        int i_start = max(0, k - (n - 1));
        int i_end = min(k, n - 1);

        for (int i = i_start; i <= i_end; ++i) {
            uint64_t a = a_shared[i];
            uint64_t b = b_shared[k - i];

            uint64_t product = a * b;

            // Accumulate the product
            sum_low += product;
            if (sum_low < product) {
                sum_high += 1;
            }
        }

        // Store sum_low and sum_high in tempProducts
        int idx = Z1_temp_offset + k * 2;
        tempProducts[idx] = sum_low;
        tempProducts[idx + 1] = sum_high;
    }

    if constexpr (!SharkFloatParams::DisableAllAdditions) {

        // Synchronize before combining results
        grid.sync();

        // After computing Z1_temp (Z1'), we now form Z1 directly:
        // If z1_sign == 0: Z1 = Z2 + Z0 - Z1_temp
        // If z1_sign == 1: Z1 = Z2 + Z0 + Z1_temp

        for (int k = k_start; k < k_end; ++k) {
            // Retrieve Z0
            int z0_idx = Z0_offset + k * 2;
            uint64_t z0_low = tempProducts[z0_idx];
            uint64_t z0_high = tempProducts[z0_idx + 1];

            // Retrieve Z2
            int z2_idx = Z2_offset + k * 2;
            uint64_t z2_low = tempProducts[z2_idx];
            uint64_t z2_high = tempProducts[z2_idx + 1];

            // Retrieve Z1_temp (Z1')
            int z1_temp_idx = Z1_temp_offset + k * 2;
            uint64_t z1_temp_low = tempProducts[z1_temp_idx];
            uint64_t z1_temp_high = tempProducts[z1_temp_idx + 1];

            // Combine Z2 + Z0 first
            uint64_t temp_low, temp_high;
            add128(z2_low, z2_high, z0_low, z0_high, temp_low, temp_high);

            uint64_t z1_low, z1_high;
            if (z1_sign == 0) {
                // same sign: Z1 = (Z2 + Z0) - Z1_temp
                subtract128(temp_low, temp_high, z1_temp_low, z1_temp_high, z1_low, z1_high);
            } else {
                // opposite signs: Z1 = (Z2 + Z0) + Z1_temp
                add128(temp_low, temp_high, z1_temp_low, z1_temp_high, z1_low, z1_high);
            }

            // Store fully formed Z1
            int z1_idx = Z1_offset + k * 2;
            tempProducts[z1_idx] = z1_low;
            tempProducts[z1_idx + 1] = z1_high;
        }

        // Synchronize before final combination
        grid.sync();

        // Now the final combination is just:
        // final = Z0 + (Z1 << (32*n)) + (Z2 << (64*n))
        int idx_start = (threadIdxGlobal * total_result_digits) / total_threads;
        int idx_end = ((threadIdxGlobal + 1) * total_result_digits) / total_threads;

        for (int idx = idx_start; idx < idx_end; ++idx) {
            uint64_t sum_low = 0;
            uint64_t sum_high = 0;

            // Add Z0
            if (idx < 2 * n - 1) {
                int z0_idx = Z0_offset + idx * 2;
                uint64_t z0_low = tempProducts[z0_idx];
                uint64_t z0_high = tempProducts[z0_idx + 1];
                add128(sum_low, sum_high, z0_low, z0_high, sum_low, sum_high);
            }

            // Add Z1 shifted by n
            if (idx >= n && (idx - n) < 2 * n - 1) {
                int z1_idx = Z1_offset + (idx - n) * 2;
                uint64_t z1_low = tempProducts[z1_idx];
                uint64_t z1_high = tempProducts[z1_idx + 1];
                add128(sum_low, sum_high, z1_low, z1_high, sum_low, sum_high);
            }

            // Add Z2 shifted by 2*n
            if (idx >= 2 * n && (idx - 2 * n) < 2 * n - 1) {
                int z2_idx = Z2_offset + (idx - 2 * n) * 2;
                uint64_t z2_low = tempProducts[z2_idx];
                uint64_t z2_high = tempProducts[z2_idx + 1];
                add128(sum_low, sum_high, z2_low, z2_high, sum_low, sum_high);
            }

            int result_idx = Convolution_offset + idx * 2;
            tempProducts[result_idx] = sum_low;
            tempProducts[result_idx + 1] = sum_high;
        }

        // Synchronize before carry propagation
        grid.sync();
    }

    // ---- Carry Propagation ----

    // Global memory for block carry-outs
    // Allocate space for gridDim.x block carry-outs after total_result_digits
    uint64_t *block_carry_outs = &tempProducts[CarryInsOffset];

    if constexpr (!SharkFloatParams::DisableCarryPropagation) {

        uint64_t *globalCarryCheck = &tempProducts[GlobalCarryOffset];

        // First Pass: Process convolution results to compute initial digits and local carries
        CarryPropagation<SharkFloatParams>(
            (uint64_t *)shared_data,
            grid,
            block,
            threadIdx,
            blockIdx,
            blockDim,
            gridDim,
            thread_start_idx,
            thread_end_idx,
            Convolution_offset,
            Result_offset,
            block_carry_outs,
            tempProducts,
            globalCarryCheck
        );
    } else {
        grid.sync();
    }

    // ---- Finalize the Result ----
    if constexpr (!SharkFloatParams::DisableFinalConstruction) {
        // uint64_t final_carry = carryOuts_phase6[SharkFloatParams::NumBlocks - 1];

        // Initial total_result_digits is 2 * N
        int total_result_digits = 2 * N;

        // Determine the highest non-zero digit index in the full result
        int highest_nonzero_index = total_result_digits - 1;

        while (highest_nonzero_index >= 0) {
            int result_idx = Result_offset + highest_nonzero_index;
            uint32_t digit = static_cast<uint32_t>(tempProducts[result_idx]);
            if (digit != 0) {
                break;
            }

            highest_nonzero_index--;
        }

        // Determine the number of significant digits
        int significant_digits = highest_nonzero_index + 1;
        // Calculate the number of digits to shift to keep the most significant N digits
        int shift_digits = significant_digits - N;
        if (shift_digits < 0) {
            shift_digits = 0;  // No need to shift if we have fewer than N significant digits
        }

        if (blockIdx.x == 0 && threadIdx.x == 0) {
            // Adjust the exponent based on the number of bits shifted
            Out->Exponent = A->Exponent + B->Exponent + shift_digits * 32;

            // Set the sign of the result
            Out->IsNegative = A->IsNegative ^ B->IsNegative;
        }

        int tid = threadIdx.x + blockIdx.x * blockDim.x;
        int stride = blockDim.x * gridDim.x;

        // src_idx is the starting index in tempProducts[] from which we copy
        int src_idx = Result_offset + shift_digits;
        int last_src = Result_offset + highest_nonzero_index; // The last valid index

        // We'll do a grid-stride loop over i in [0 .. N)
        for (int i = tid; i < N; i += stride) {
            // Corresponding source index for digit i
            int src = src_idx + i;

            if (src <= last_src) {
                // Copy from tempProducts
                Out->Digits[i] = tempProducts[src];
            } else {
                // Pad with zero if we've run out of digits
                Out->Digits[i] = 0;
            }
        }
    }
}

template<class SharkFloatParams>
__global__ void MultiplyKernelKaratsubaV2(
    const HpSharkFloat<SharkFloatParams> *A,
    const HpSharkFloat<SharkFloatParams> *B,
    HpSharkFloat<SharkFloatParams> *Out,
    uint64_t *tempProducts) {

    // Initialize cooperative grid group
    cg::grid_group grid = cg::this_grid();

    // Call the MultiplyHelper function
    //MultiplyHelper(A, B, Out, carryIns, grid, tempProducts);
    MultiplyHelperKaratsubaV2(A, B, Out, grid, tempProducts);
}

template<class SharkFloatParams>
__global__ void MultiplyKernelKaratsubaV2TestLoop(
    HpSharkFloat<SharkFloatParams> *A,
    HpSharkFloat<SharkFloatParams> *B,
    HpSharkFloat<SharkFloatParams> *Out,
    uint64_t *tempProducts) { // Array to store cumulative carries

    // Initialize cooperative grid group
    cg::grid_group grid = cg::this_grid();

    for (int i = 0; i < TestIterCount; ++i) {
        // MultiplyHelper(A, B, Out, carryIns, grid, tempProducts);
        if constexpr (!SharkFloatParams::ForceNoOp) {
            MultiplyHelperKaratsubaV2(A, B, Out, grid, tempProducts);
        } else {
            grid.sync();
        }
    }
}

template<class SharkFloatParams>
void PrintMaxActiveBlocks(int sharedAmountBytes) {
    std::cout << "Shared memory size: " << sharedAmountBytes << std::endl;

    int numBlocks;
    cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocks,
        MultiplyKernelKaratsubaV2<SharkFloatParams>,
        SharkFloatParams::ThreadsPerBlock,
        sharedAmountBytes
    );

    if (err != cudaSuccess) {
        std::cerr << "CUDA error in cudaOccupancyMaxActiveBlocksPerMultiprocessor: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    std::cout << "Max active blocks: " << numBlocks << std::endl;
}

template<class SharkFloatParams>
void ComputeMultiplyKaratsubaV2Gpu(void *kernelArgs[]) {

    cudaError_t err;

    constexpr int N = SharkFloatParams::NumUint32;
    constexpr auto n = (N + 1) / 2;              // Half of N
    const auto sharedAmountBytes = 5 * n * sizeof(uint32_t);

    if constexpr (UseCustomStream) {
        cudaFuncSetAttribute(
            MultiplyKernelKaratsubaV2<SharkFloatParams>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            sharedAmountBytes);

        PrintMaxActiveBlocks<SharkFloatParams>(sharedAmountBytes);
    }

    err = cudaLaunchCooperativeKernel(
        (void *)MultiplyKernelKaratsubaV2<SharkFloatParams>,
        dim3(SharkFloatParams::NumBlocks),
        dim3(SharkFloatParams::ThreadsPerBlock),
        kernelArgs,
        sharedAmountBytes, // Shared memory size
        0 // Stream
    );

    cudaDeviceSynchronize();

    if (err != cudaSuccess) {
        std::cerr << "CUDA error in MultiplyKernelKaratsubaV2: " << cudaGetErrorString(err) << std::endl;
    }
}

template<class SharkFloatParams>
void ComputeMultiplyKaratsubaV2GpuTestLoop(cudaStream_t &stream, void *kernelArgs[]) {

    constexpr int N = SharkFloatParams::NumUint32;
    constexpr auto n = (N + 1) / 2;              // Half of N
    constexpr auto sharedAmountBytes = UseSharedMemory ? (5 * n * sizeof(uint32_t)) : 0;

    if constexpr (UseCustomStream) {
        cudaFuncSetAttribute(
            MultiplyKernelKaratsubaV2TestLoop<SharkFloatParams>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            sharedAmountBytes);

        PrintMaxActiveBlocks<SharkFloatParams>(sharedAmountBytes);
    }

    cudaError_t err = cudaLaunchCooperativeKernel(
        (void *)MultiplyKernelKaratsubaV2TestLoop<SharkFloatParams>,
        dim3(SharkFloatParams::NumBlocks),
        dim3(SharkFloatParams::ThreadsPerBlock),
        kernelArgs,
        sharedAmountBytes, // Shared memory size
        stream // Stream
    );

    cudaDeviceSynchronize();

    if (err != cudaSuccess) {
        std::cerr << "CUDA error in MultiplyKernelKaratsubaTestLoop: " << cudaGetErrorString(err) << std::endl;
    }
}

#define ExplicitlyInstantiate(SharkFloatParams) \
    template void ComputeMultiplyKaratsubaV2Gpu<SharkFloatParams>(void *kernelArgs[]); \
    template void ComputeMultiplyKaratsubaV2GpuTestLoop<SharkFloatParams>(cudaStream_t &stream, void *kernelArgs[]);

ExplicitlyInstantiate(Test4x4SharkParams);
ExplicitlyInstantiate(Test4x2SharkParams);
ExplicitlyInstantiate(Test8x1SharkParams);
ExplicitlyInstantiate(Test8x8SharkParams);

ExplicitlyInstantiate(Test128x64SharkParams);
ExplicitlyInstantiate(Test64x64SharkParams);
ExplicitlyInstantiate(Test32x64SharkParams);
ExplicitlyInstantiate(Test16x64SharkParams);

ExplicitlyInstantiate(Test128x32SharkParams);
ExplicitlyInstantiate(Test128x16SharkParams);
ExplicitlyInstantiate(Test128x8SharkParams);
ExplicitlyInstantiate(Test128x4SharkParams);