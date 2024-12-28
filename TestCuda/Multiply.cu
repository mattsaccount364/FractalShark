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
 *   shared_data: pointer to shared memory for this block
 *   a, b       : pointer to global arrays of digits (size at least n)
 *   carryOuts  : array in global or shared memory to store block-level borrows
 *   cumulativeCarries : array in global memory to store final block-to-block borrows
 *   result     : pointer to global array where the final digits are written
 *   grid,block : cooperative_groups handles for synchronization
 *
 * The function attempts to subtract each digit of 'b' from 'a' in parallel,
 * then uses repeated passes (do/while) to handle newly introduced borrows
 * until no more remain or a maximum pass count is reached.
 */
template<class SharkFloatParams>
__device__ void subtractDigitsParallel(
    uint32_t *__restrict__ shared_data,
    const uint32_t *__restrict__ a,
    const uint32_t *__restrict__ b,
    uint32_t *carryOuts,        // Shared array to store block-level borrows
    uint32_t *cumulativeCarries,
    uint32_t *__restrict__ result,
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
    const int digitsPerBlock = (n + numBlocks - 1) / numBlocks;
    const int startDigit = blockId * digitsPerBlock;
    const int endDigit = min(startDigit + digitsPerBlock, n);
    const int digitsInBlock = endDigit - startDigit;

    // If this block has no digits to process, we can exit quickly
    if (digitsInBlock <= 0) return;

    // Pointers in shared memory
    // We'll store partial differences and borrow bits here
    uint32_t *a_shared = shared_data;                       // [0 .. digitsInBlock-1]
    uint32_t *b_shared = a_shared + digitsInBlock;          // [digitsInBlock .. 2*digitsInBlock-1]
    uint32_t *partialDiff = b_shared + digitsInBlock;          // ...
    uint32_t *borrowBits = partialDiff + digitsInBlock;
    // optionally you can do more arrays for second pass or prefix sums, etc.

    // 1) Load 'a' and 'b' digits into shared memory
    for (int i = tid; i < digitsInBlock; i += threadsPerBlock) {
        a_shared[i] = a[startDigit + i];
        b_shared[i] = b[startDigit + i];
    }
    block.sync();

    // 2) First pass: compute naive partial difference (a[i] - b[i]), store in partialDiff
    //    and whether a[i]<b[i] => borrowBit=1 else 0
    for (int i = tid; i < digitsInBlock; i += threadsPerBlock) {
        uint32_t ai = a_shared[i];
        uint32_t bi = b_shared[i];
        // naive difference
        uint64_t temp = static_cast<uint64_t>(ai) - static_cast<uint64_t>(bi);
        uint32_t borrow = (ai < bi) ? 1u : 0u;

        partialDiff[i] = static_cast<uint32_t>(temp & 0xFFFFFFFFu);
        borrowBits[i] = borrow;
    }
    block.sync();

    // We'll do repeated passes to fix newly introduced borrows
    // We'll store in carryOuts[blockId] any leftover borrow from this block if needed
    // but let's keep it simple: we do an in-block multi-pass approach first
    uint32_t changed_any = 1;  // track if any borrow changed in this pass
    int passCount = 0;

    do {
        // Zero 'changed_any' in shared memory or a local var
        if (tid == 0) {
            changed_any = 0;
        }
        block.sync();

        // Each thread checks whether borrowBits[i] => we must borrow from partialDiff[i+1]
        for (int i = tid; i < digitsInBlock - 1; i += threadsPerBlock) {
            // If borrowBits[i] == 1 => partialDiff[i+1] must subtract 1
            if (borrowBits[i] == 1) {
                // Sub the borrow from partialDiff[i+1]
                uint32_t nextVal = partialDiff[i + 1];
                uint32_t oldVal = nextVal;
                nextVal = nextVal - 1u;

                // If oldVal==0, we generate a new borrow in borrowBits[i+1]
                uint32_t newBorrow = (oldVal == 0u) ? 1u : 0u;

                // Update partialDiff[i+1] and borrowBits[i+1] if changed
                partialDiff[i + 1] = nextVal;
                if (newBorrow > borrowBits[i + 1]) {
                    borrowBits[i + 1] = newBorrow;
                    // Mark that we changed something in this pass
                    atomicMax(&changed_any, 1u);
                }

                // Clear the borrow bit for i => we've consumed it
                borrowBits[i] = 0u;
            }
        }

        block.sync();

        // If changed_any==0 => no new borrows were introduced => we're done
        // Let one thread read changed_any from shared memory or we do a reduce, etc.
        uint32_t changed_local = changed_any;
        block.sync();

        // We can do a single thread check:
        if (tid == 0) {
            // gather changed_any from all threads if needed
        }
        block.sync();

        passCount++;
        if (passCount >= MaxPasses) {
            // if we exceed pass count => break
            break;
        }

        // We'll rely on changed_any to be 0 => no new borrows
    } while (__syncthreads_and(changed_any != 0u));

    // Now partialDiff[] is corrected for in-block borrows
    // Write final partialDiff to global result
    for (int i = tid; i < digitsInBlock; i += threadsPerBlock) {
        result[startDigit + i] = partialDiff[i];
    }
    block.sync();

    // 3) If you want to pass a leftover borrow out of the block (the last digit), 
    //    you can check borrowBits[digitsInBlock-1]. If it is 1 => store that in carryOuts[blockId].
    if (tid == 0 && digitsInBlock > 0) {
        carryOuts[blockId] = borrowBits[digitsInBlock - 1];
    }
    block.sync();

    // 4) In a full multi-block scenario, you'd do an inter-block pass, 
    //    reading carryOuts[i-1] => applying it to the block i's first digit => 
    //    re-run the in-block pass. But that is typically done outside this function 
    //    or in repeated calls with a higher-level logic, similar to your addition multi-pass approach.

    // If you want a final pass to handle cumulativeCarries, you'd do that here, 
    // but it depends on your multi-block structure.

    // Done
}



// Function to perform addition with carry
__device__ static void add128(
    uint64_t a_low, uint64_t a_high,
    uint64_t b_low, uint64_t b_high,
    uint64_t &result_low, uint64_t &result_high) {

    result_low = a_low + b_low;
    uint64_t carry = (result_low < a_low) ? 1 : 0;
    result_high = a_high + b_high + carry;
}

__device__ static void subtract128(
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
__device__ static void CarryPropagation (
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
    uint64_t * __restrict__ carryOuts_phase3) {

    // First Pass: Process convolution results to compute initial digits and local carries
    // Initialize local carry
    uint64_t local_carry = 0;

    // Constants and offsets
    constexpr int MaxPasses = 10; // Maximum number of carry propagation passes
    constexpr int total_result_digits = 2 * SharkFloatParams::NumUint32;

    uint64_t *carries_remaining_global = carryOuts_phase3;

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

    // Synchronize all blocks
    grid.sync();

    // Inter-Block Carry Propagation
    int pass = 0;

    do {
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

        grid.sync();
        pass++;
    } while (pass < MaxPasses);

    // ---- Handle Final Carry-Out ----

    // Synchronize all blocks
    grid.sync();

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
    grid.sync();
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
    uint64_t *__restrict__ carryOuts_phase3,
    uint64_t *__restrict__ carryOuts_phase6,
    uint64_t *__restrict__ carryIns,
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

    //// Shared memory allocation
    //__shared__ uint64_t A_shared[n];
    //__shared__ uint64_t B_shared[n];

    //// Load segments of A and B into shared memory
    //for (int i = threadIdx.x; i < n; i += SharkFloatParams::ThreadsPerBlock) {
    //    A_shared[i] = (i < N) ? A->Digits[i] : 0;  // A0
    //    B_shared[i] = (i < N) ? B->Digits[i] : 0;  // B0
    //}

    // Synchronize before starting convolutions
    //block.sync();

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
            uint64_t a = A->Digits[i]; //A_shared[i];         // A0[i]
            uint64_t b = B->Digits[k - i]; //B_shared[k - i];     // B0[k - i]

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

    //// Load A1 and B1 into shared memory
    //for (int i = threadIdx.x; i < n; i += SharkFloatParams::ThreadsPerBlock) {
    //    int index = i + n;
    //    A_shared[i] = (index < N) ? A->Digits[index] : 0;    // A1
    //    B_shared[i] = (index < N) ? B->Digits[index] : 0;    // B1
    //}

    //block.sync();

    // ---- Convolution for Z2 = A1 * B1 ----
    for (int k = k_start; k < k_end; ++k) {
        uint64_t sum_low = 0;
        uint64_t sum_high = 0;

        int i_start = max(0, k - (n - 1));
        int i_end = min(k, n - 1);

        for (int i = i_start; i <= i_end; ++i) {
            uint64_t a = A->Digits[i + n]; // A_shared[i];         // A1[i]
            uint64_t b = B->Digits[k - i + n]; // B_shared[k - i];     // B1[k - i]

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

    //// Synchronize before next convolution
    block.sync();

    //// Compute (A0 + A1) and (B0 + B1) and store in shared memory
    //for (int i = threadIdx.x; i < n; i += SharkFloatParams::ThreadsPerBlock) {
    //    uint64_t A0 = (i < N) ? A->Digits[i] : 0;
    //    uint64_t A1 = (i + n < N) ? A->Digits[i + n] : 0;
    //    A_shared[i] = A0 + A1;               // (A0 + A1)

    //    uint64_t B0 = (i < N) ? B->Digits[i] : 0;
    //    uint64_t B1 = (i + n < N) ? B->Digits[i + n] : 0;
    //    B_shared[i] = B0 + B1;               // (B0 + B1)
    //}

    //block.sync();

    // ---- Compute Differences x_diff = A1 - A0 and y_diff = B1 - B0 ----

// Arrays to hold the absolute differences (size n)
    auto *x_diff_abs = reinterpret_cast<uint32_t*>(carryOuts_phase6);
    auto *y_diff_abs = reinterpret_cast<uint32_t *>(carryOuts_phase6) + n;
    int x_diff_sign = 0; // 0 if positive, 1 if negative
    int y_diff_sign = 0; // 0 if positive, 1 if negative

    // Compute x_diff_abs and x_diff_sign
    extern __shared__ uint32_t shared_data[];

    auto *subtractionCarries = reinterpret_cast<uint32_t *>(carryIns);
    auto *cumulativeCarries = subtractionCarries + SharkFloatParams::NumBlocks;

    constexpr bool useParallelSubtract = true;

    if constexpr (useParallelSubtract) {
        int x_compare = compareDigits(A->Digits + n, A->Digits, n);

        if (x_compare >= 0) {
            x_diff_sign = 0;
            subtractDigitsParallel<SharkFloatParams>(
                shared_data,
                A->Digits + n,
                A->Digits,
                subtractionCarries,
                cumulativeCarries,
                x_diff_abs,
                grid,
                block); // x_diff = A1 - A0
        } else {
            x_diff_sign = 1;
            subtractDigitsParallel<SharkFloatParams>(
                shared_data,
                A->Digits,
                A->Digits + n,
                subtractionCarries,
                cumulativeCarries,
                x_diff_abs,
                grid,
                block); // x_diff = A0 - A1
        }

        // Compute y_diff_abs and y_diff_sign
        int y_compare = compareDigits(B->Digits + n, B->Digits, n);
        if (y_compare >= 0) {
            y_diff_sign = 0;
            subtractDigitsParallel<SharkFloatParams>(
                shared_data,
                B->Digits + n,
                B->Digits,
                subtractionCarries,
                cumulativeCarries,
                y_diff_abs,
                grid,
                block); // y_diff = B1 - B0
        } else {
            y_diff_sign = 1;
            subtractDigitsParallel<SharkFloatParams>(
                shared_data,
                B->Digits,
                B->Digits + n,
                subtractionCarries,
                cumulativeCarries,
                y_diff_abs,
                grid,
                block); // y_diff = B0 - B1
        }
    } else {
        int x_compare = compareDigits(A->Digits + n, A->Digits, n);

        if (x_compare >= 0) {
            x_diff_sign = 0;
            subtractDigits(A->Digits + n, A->Digits, x_diff_abs, n); // x_diff = A1 - A0
        } else {
            x_diff_sign = 1;
            subtractDigits(A->Digits, A->Digits + n, x_diff_abs, n); // x_diff = A0 - A1
        }

        // Compute y_diff_abs and y_diff_sign
        int y_compare = compareDigits(B->Digits + n, B->Digits, n);
        if (y_compare >= 0) {
            y_diff_sign = 0;
            subtractDigits(B->Digits + n, B->Digits, y_diff_abs, n); // y_diff = B1 - B0
        } else {
            y_diff_sign = 1;
            subtractDigits(B->Digits, B->Digits + n, y_diff_abs, n); // y_diff = B0 - B1
        }
    }

    // Determine the sign of Z1_temp
    int z1_sign = x_diff_sign ^ y_diff_sign;


    // Synchronize before convolution
    grid.sync();

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
            uint64_t a = x_diff_abs[i];
            uint64_t b = y_diff_abs[k - i];

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

    // Synchronize before combining results
    grid.sync();

    // After computing Z1_temp (Z1'), we now form Z1 directly:
    // If z1_sign == 0: Z1 = Z2 + Z0 - Z1_temp
    // If z1_sign == 1: Z1 = Z2 + Z0 + Z1_temp

    grid.sync(); // Ensure all computations up to Z1_temp are done

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
    constexpr int total_result_digits = 2 * N + 1;
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

    // ---- Carry Propagation ----

    // Global memory for block carry-outs
    // Allocate space for gridDim.x block carry-outs after total_result_digits
    uint64_t *block_carry_outs = carryIns;
    //uint64_t *blocks_need_to_continue = carryIns + SharkFloatParams::NumBlocks;

    constexpr auto digits_per_block = SharkFloatParams::ThreadsPerBlock * 2;
    auto block_start_idx = blockIdx.x * digits_per_block;
    auto block_end_idx = min(block_start_idx + digits_per_block, total_result_digits);

    int digits_per_thread = (digits_per_block + blockDim.x - 1) / blockDim.x;

    int thread_start_idx = block_start_idx + threadIdx.x * digits_per_thread;
    int thread_end_idx = min(thread_start_idx + digits_per_thread, block_end_idx);

    //So the idea is we process the chunk of digits that has digits interleaved with carries.
    //    This logic should be similar to the global carry propagation but done in parallel on
    //    each block.
    //After that step is done, we should have reduced the number of digits we care about
    //    because weve propagated all the intermediate junk produced above during convolution.
    //    so we end up with 2 * SharkFloatParams::NumBlocks * SharkFloatParams::ThreadsPerBlock digits.
    //At that point we do inter-block carry propagation, which is iterative.
    

    if constexpr (!SharkFloatParams::DisableCarryPropagation) {

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
            carryOuts_phase3
        );
    } else {
        grid.sync();
    }

    // ---- Finalize the Result ----

    // ---- Handle Any Remaining Final Carry ----

    // Only one thread handles the final carry propagation
    if constexpr (!SharkFloatParams::DisableFinalConstruction) {
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            // uint64_t final_carry = carryOuts_phase6[SharkFloatParams::NumBlocks - 1];

            // Initial total_result_digits is 2 * N
            int total_result_digits = 2 * N;

            // Handle the final carry-out from the most significant digit
            //if (final_carry > 0) {
            //    // Append the final carry as a new digit at the end (most significant digit)
            //    tempProducts[Result_offset + total_result_digits] = static_cast<uint32_t>(final_carry & 0xFFFFFFFFULL);
            //    total_result_digits += 1;
            //}

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

            // Adjust the exponent based on the number of bits shifted
            Out->Exponent = A->Exponent + B->Exponent + shift_digits * 32;

            // Copy the least significant N digits to Out->Digits
            int src_idx = Result_offset + shift_digits;
            for (int i = 0; i < N; ++i, ++src_idx) {
                if (src_idx <= Result_offset + highest_nonzero_index) {
                    Out->Digits[i] = tempProducts[src_idx];
                } else {
                    // If we've run out of digits, pad with zeros
                    Out->Digits[i] = 0;
                }
            }

            // Set the sign of the result
            Out->IsNegative = A->IsNegative ^ B->IsNegative;
        }
    }
}

template<class SharkFloatParams>
__global__ void MultiplyKernelKaratsubaV2(
    const HpSharkFloat<SharkFloatParams> *A,
    const HpSharkFloat<SharkFloatParams> *B,
    HpSharkFloat<SharkFloatParams> *Out,
    uint64_t *carryOuts_phase3,
    uint64_t *carryOuts_phase6,
    uint64_t *carryIns,
    uint64_t *tempProducts) {

    // Initialize cooperative grid group
    cg::grid_group grid = cg::this_grid();

    // Call the MultiplyHelper function
    //MultiplyHelper(A, B, Out, carryOuts_phase3, carryOuts_phase6, carryIns, grid, tempProducts);
    MultiplyHelperKaratsubaV2(A, B, Out, carryOuts_phase3, carryOuts_phase6, carryIns, grid, tempProducts);
}

template<class SharkFloatParams>
__global__ void MultiplyKernelKaratsubaV2TestLoop(
    HpSharkFloat<SharkFloatParams> *A,
    HpSharkFloat<SharkFloatParams> *B,
    HpSharkFloat<SharkFloatParams> *Out,
    uint64_t *carryOuts_phase3,
    uint64_t *carryOuts_phase6,
    uint64_t *carryIns,
    uint64_t *tempProducts) { // Array to store cumulative carries

    // Initialize cooperative grid group
    cg::grid_group grid = cg::this_grid();

    for (int i = 0; i < TestIterCount; ++i) {
        // MultiplyHelper(A, B, Out, carryOuts_phase3, carryOuts_phase6, carryIns, grid, tempProducts);
        MultiplyHelperKaratsubaV2(A, B, Out, carryOuts_phase3, carryOuts_phase6, carryIns, grid, tempProducts);
    }
}

template<class SharkFloatParams>
void ComputeMultiplyKaratsubaV2Gpu(void *kernelArgs[]) {

    int numBlocks;
    cudaError_t err;

    constexpr int N = SharkFloatParams::NumUint32;
    constexpr auto n = (N + 1) / 2;              // Half of N

    const auto sharedAmountBytes = 10 * n * sizeof(uint32_t);

    std::cout << "Shared memory size: " << sharedAmountBytes << std::endl;

    err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocks,
        MultiplyKernelKaratsubaV2<SharkFloatParams>,
        SharkFloatParams::ThreadsPerBlock,
        sharedAmountBytes
    );

    if (err != cudaSuccess) {
        std::cerr << "CUDA error in cudaOccupancyMaxActiveBlocksPerMultiprocessor: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    // Print the number of blocks
    std::cout << "Number of blocks: " << numBlocks << std::endl;

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
void ComputeMultiplyKaratsubaV2GpuTestLoop(void *kernelArgs[]) {

    cudaError_t err = cudaLaunchCooperativeKernel(
        (void *)MultiplyKernelKaratsubaV2TestLoop<SharkFloatParams>,
        dim3(SharkFloatParams::NumBlocks),
        dim3(SharkFloatParams::ThreadsPerBlock),
        kernelArgs,
        0, // Shared memory size
        0 // Stream
    );

    cudaDeviceSynchronize();

    if (err != cudaSuccess) {
        std::cerr << "CUDA error in MultiplyKernelKaratsubaTestLoop: " << cudaGetErrorString(err) << std::endl;
    }
}

#define ExplicitlyInstantiate(SharkFloatParams) \
    template void ComputeMultiplyKaratsubaV2Gpu<SharkFloatParams>(void *kernelArgs[]); \
    template void ComputeMultiplyKaratsubaV2GpuTestLoop<SharkFloatParams>(void *kernelArgs[]);

ExplicitlyInstantiate(Test4x4SharkParams);
ExplicitlyInstantiate(Test4x2SharkParams);
ExplicitlyInstantiate(Test8x1SharkParams);
ExplicitlyInstantiate(Test128x64SharkParams);