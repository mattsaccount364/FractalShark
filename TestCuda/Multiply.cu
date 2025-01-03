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
    uint32_t *sharedCarryOut   // Statically allocated sharedCarryOut array (size SharkFloatParams::GlobalThreadsPerBlock)
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
    uint32_t *sharedCarryOut   // Statically allocated sharedCarryOut array (size SharkFloatParams::GlobalThreadsPerBlock)
) {
    if constexpr (SharkFloatParams::GlobalThreadsPerBlock <= 32) {
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
template<class SharkFloatParams, int N>
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
    constexpr int n = (N + 1) / 2;     // 'n' is how many digits
    constexpr int MaxPasses = 10;      // maximum number of multi-pass sweeps

    // We'll define a grid–stride range covering [0..n) for each pass
    // 1) global thread id
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // 2) stride
    int stride = blockDim.x * gridDim.x;

    // (1) First pass: naive partial difference (a[i] - b[i]) and set borrowBit
    // Instead of dividing digits among blocks, each thread does a grid–stride loop:
    for (int idx = tid; idx < n; idx += stride) {
        uint32_t ai = a[idx];
        uint32_t bi = b[idx];

        // naive difference
        uint64_t diff = (uint64_t)ai - (uint64_t)bi;
        uint32_t borrow = (ai < bi) ? 1u : 0u;

        result[idx] = static_cast<uint32_t>(diff & 0xFFFFFFFFu);
        subtractionBorrows[idx] = borrow;
    }

    // sync the entire grid before multi-pass fixes
    grid.sync();

    // We'll do repeated passes to fix newly introduced borrows
    uint32_t *curBorrow = subtractionBorrows;
    uint32_t *newBorrow = subtractionBorrows2;
    int pass = 0;

    do {
        // Zero out the borrow count
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            *globalBorrowAny = 0;
        }
        grid.sync();

        // (2) For each digit, apply the borrow from the previous digit
        for (int idx = tid; idx < n; idx += stride) {
            uint64_t borrow_in = 0ULL;
            if (idx > 0) {   // borrow_in is from digit (idx-1)
                borrow_in = (uint64_t)(curBorrow[idx - 1]);
            }

            uint32_t digit = result[idx];
            // subtract the borrow
            uint64_t sum = (uint64_t)digit - borrow_in;

            // store updated digit
            result[idx] = static_cast<uint32_t>(sum & 0xFFFFFFFFULL);

            // If sum is negative => top bit is 1 => new borrow
            if (sum & 0x8000'0000'0000'0000ULL) {
                newBorrow[idx] = 1;
                atomicAdd(globalBorrowAny, 1);
            } else {
                newBorrow[idx] = 0;
            }
        }

        // sync before checking if any new borrows remain
        grid.sync();

        if (*globalBorrowAny == 0) {
            break;  // no new borrows => done
        }

        grid.sync();

        // swap curBorrow, newBorrow
        uint32_t *tmp = curBorrow;
        curBorrow = newBorrow;
        newBorrow = tmp;

        pass++;
    } while (pass < MaxPasses);
}



// Function to perform addition with carry
__device__ __forceinline__ static void Add128(
    uint64_t a_low, uint64_t a_high,
    uint64_t b_low, uint64_t b_high,
    uint64_t &result_low, uint64_t &result_high) {

    result_low = a_low + b_low;
    uint64_t carry = (result_low < a_low) ? 1 : 0;
    result_high = a_high + b_high + carry;
}

__device__ __forceinline__ static void Subtract128(
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
__device__ __forceinline__ static void SerialCarryPropagation(
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
    uint64_t *__restrict__ block_carry_outs,
    uint64_t *__restrict__ tempProducts,
    uint64_t *__restrict__ globalCarryCheck) {

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        uint64_t local_carry = 0;

        for (int idx = 0; idx < SharkFloatParams::GlobalNumUint32 * 2 + 1; ++idx) {
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
    }
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
    constexpr int total_result_digits = 2 * SharkFloatParams::GlobalNumUint32;

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

    if (threadIdx.x == SharkFloatParams::GlobalThreadsPerBlock - 1) {
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
            // We'll check if local_carry is negative *before* or after we add— 
            // it depends on how your code indicates negativity. Typically:
            bool local_carry_negative = ((local_carry & (1ULL << 63)) != 0ULL);

            uint64_t sum = static_cast<uint64_t>(digit) + local_carry;

            // Update digit
            digit = static_cast<uint32_t>(sum & 0xFFFFFFFFULL);
            tempProducts[Result_offset + idx] = digit;

            local_carry = 0;

            // Check negativity of the 64-bit sum
            // If "sum" is negative, its top bit is set. 
            bool sum_is_negative = ((sum & (1ULL << 63)) != 0ULL);

            if (sum_is_negative) {
                // sign-extend the top 32 bits
                uint64_t upper_bits = (sum >> 32);
                upper_bits |= 0xFFFF'FFFF'0000'0000ULL;  // set top 32 bits to 1
                local_carry += upper_bits;               // incorporate sign-extended bits
            } else {
                // normal path: just add top 32 bits
                local_carry += (sum >> 32);
            }
        }

        shared_carries[threadIdx.x] = local_carry;
        grid.sync();

        // The block's carry-out is the carry from the last thread
        auto temp = shared_carries[threadIdx.x];
        if (threadIdx.x == SharkFloatParams::GlobalThreadsPerBlock - 1) {
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

#define DefineTempProductsOffsets(TempBase) \
    constexpr int n = (NewN + 1) / 2; \
    const int threadIdxGlobal = blockIdx.x * SharkFloatParams::GlobalThreadsPerBlock + threadIdx.x; \
    constexpr int Z0_offset = TempBase; \
    constexpr int Z2_offset = Z0_offset + 4 * NewN; \
    constexpr int Z1_temp_offset = Z2_offset + 4 * NewN; \
    constexpr int Z1_offset = Z1_temp_offset + 4 * NewN; \
    constexpr int Convolution_offset = Z1_offset + 4 * NewN; \
    constexpr int Result_offset = Convolution_offset + 4 * NewN; \
    constexpr int XDiff_offset = Result_offset + 2 * NewN; \
    constexpr int YDiff_offset = XDiff_offset + 1 * NewN; \
    constexpr int GlobalCarryOffset = YDiff_offset + 1 * NewN; \
    constexpr int SubtractionOffset1 = GlobalCarryOffset + 1 * NewN; \
    constexpr int SubtractionOffset2 = SubtractionOffset1 + 1 * NewN; \
    constexpr int BorrowAnyOffset = SubtractionOffset2 + 1 * NewN; \
    /* Note, overlaps: */ \
    constexpr int CarryInsOffset = TempBase; \
    constexpr int CarryInsOffset2 = CarryInsOffset + 2 * NewN;


#define DefineExtraDefinitions() \
    constexpr int total_result_digits = 2 * NewN + 1; \
    constexpr auto digits_per_block = NewN * 2 / SharkFloatParams::GlobalNumBlocks; \
    auto block_start_idx = blockIdx.x * digits_per_block; \
    auto block_end_idx = min(block_start_idx + digits_per_block, total_result_digits); \
    int digits_per_thread = (digits_per_block + blockDim.x - 1) / blockDim.x; \
    int thread_start_idx = block_start_idx + threadIdx.x * digits_per_thread; \
    int thread_end_idx = min(thread_start_idx + digits_per_thread, block_end_idx);

#define DefineCarryDefinitions() \
    constexpr int total_result_digits = 2 * NewN + 1; \
    constexpr auto digits_per_block = SharkFloatParams::GlobalThreadsPerBlock * 2; \
    auto block_start_idx = blockIdx.x * digits_per_block; \
    auto block_end_idx = min(block_start_idx + digits_per_block, total_result_digits); \
    int digits_per_thread = (digits_per_block + blockDim.x - 1) / blockDim.x; \
    int thread_start_idx = block_start_idx + threadIdx.x * digits_per_thread; \
    int thread_end_idx = min(thread_start_idx + digits_per_thread, block_end_idx);


// Assuming that SharkFloatParams::GlobalNumUint32 can be large and doesn't fit in shared memory
// We'll use the provided global memory buffers for large intermediates
#define SharkRestrict __restrict__
// #define SharkRestrict

template<class SharkFloatParams, int NewN, int NewNumBlocks, int TempBase>
__device__ void MultiplyDigitsOnly(
    uint32_t *SharkRestrict shared_data,
    const uint32_t *SharkRestrict aDigits,
    const uint32_t *SharkRestrict bDigits,
    uint32_t *SharkRestrict x_diff_abs,
    uint32_t *SharkRestrict y_diff_abs,
    uint64_t *SharkRestrict final128,
    cg::grid_group &grid,
    cg::thread_block &block,
    uint64_t *SharkRestrict tempProducts) {

    DefineTempProductsOffsets(TempBase);

    const auto *a_shared = aDigits;
    const auto *b_shared = bDigits;

    // Common variables for convolution loops
    constexpr int total_k = 2 * n - 1; // Total number of k values

    auto *Z0_OutDigits = &tempProducts[Z0_offset];
    auto *Z2_OutDigits = &tempProducts[Z2_offset];

    constexpr bool UseConvolution =
        (NewNumBlocks <= std::max(SharkFloatParams::GlobalNumBlocks / 4, 1));

    //constexpr bool UseConvolution = true;

    // Arrays to hold the absolute differences (size n)
    //if (x_diff_abs == nullptr || y_diff_abs == nullptr) {
        auto *global_x_diff_abs = reinterpret_cast<uint32_t *>(&tempProducts[XDiff_offset]);
        auto *global_y_diff_abs = reinterpret_cast<uint32_t *>(&tempProducts[YDiff_offset]);
    //}


    if constexpr (UseConvolution) {

        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;

        // A single loop that covers 2*total_k elements
        for (int idx = tid; idx < 2 * total_k; idx += stride) {
            // Check if idx < total_k => handle Z0, else handle Z2
            if (idx < total_k) {
                // Z0 partial sums
                int k = idx;
                uint64_t sum_low = 0ULL, sum_high = 0ULL;

                int i_start = max(0, k - (n - 1));
                int i_end = min(k, n - 1);
                for (int i = i_start; i <= i_end; i++) {
                    uint64_t a = a_shared[i]; //A_shared[i];         // A0[i]
                    uint64_t b = b_shared[k - i]; //B_shared[k - i];     // B0[k - i]

                    uint64_t product = a * b;

                    // Add product to sum
                    sum_low += product;
                    if (sum_low < product) {
                        sum_high += 1;
                    }
                }

                // store sum_low, sum_high in Z0_OutDigits
                int out_idx = k * 2;
                Z0_OutDigits[out_idx] = sum_low;
                Z0_OutDigits[out_idx + 1] = sum_high;
            } else {
                // Z2 partial sums
                int k = idx - total_k; // shift to [0..total_k-1]
                uint64_t sum_low = 0ULL, sum_high = 0ULL;

                int i_start = max(0, k - (n - 1));
                int i_end = min(k, n - 1);
                for (int i = i_start; i <= i_end; i++) {
                    uint64_t a = a_shared[i + n]; // A_shared[i];         // A1[i]
                    uint64_t b = b_shared[k - i + n]; // B_shared[k - i];     // B1[k - i]

                    uint64_t product = a * b;

                    // Add product to sum
                    sum_low += product;
                    if (sum_low < product) {
                        sum_high += 1;
                    }
                }

                // store sum_low, sum_high in Z2_OutDigits
                int out_idx = k * 2;
                Z2_OutDigits[out_idx] = sum_low;
                Z2_OutDigits[out_idx + 1] = sum_high;
            }
        }
    } else {
        constexpr auto NewTempBase = TempBase + 32 * SharkFloatParams::GlobalNumUint32;

        MultiplyDigitsOnly<SharkFloatParams, NewN / 2, NewNumBlocks / 2, NewTempBase>(
            shared_data,
            a_shared,
            b_shared,
            x_diff_abs,
            y_diff_abs,
            Z0_OutDigits,
            grid,
            block,
            tempProducts);

        grid.sync();

        MultiplyDigitsOnly<SharkFloatParams, NewN / 2, NewNumBlocks / 2, NewTempBase>(
            shared_data,
            a_shared + n,
            b_shared + n,
            x_diff_abs,
            y_diff_abs,
            Z2_OutDigits,
            grid,
            block,
            tempProducts);

        grid.sync();
    }

    // No sync

    // ---- Compute Differences x_diff = A1 - A0 and y_diff = B1 - B0 ----

    DefineExtraDefinitions();

    int x_diff_sign = 0; // 0 if positive, 1 if negative
    int y_diff_sign = 0; // 0 if positive, 1 if negative

    // Compute x_diff_abs and x_diff_sign
    auto *SharkRestrict subtractionBorrows = reinterpret_cast<uint32_t *>(&tempProducts[SubtractionOffset1]);
    auto *SharkRestrict subtractionBorrows2 = reinterpret_cast<uint32_t *>(&tempProducts[SubtractionOffset2]);
    auto *SharkRestrict globalBorrowAny = reinterpret_cast<uint32_t *>(&tempProducts[BorrowAnyOffset]);

    constexpr bool useParallelSubtract = true;

    if constexpr (!SharkFloatParams::DisableSubtraction) {
        if constexpr (useParallelSubtract) {
            int x_compare = compareDigits(a_shared + n, a_shared, n);

            if (x_compare >= 0) {
                x_diff_sign = 0;
                SubtractDigitsParallel<SharkFloatParams, NewN>(
                    shared_data,
                    a_shared + n,
                    a_shared,
                    subtractionBorrows,
                    subtractionBorrows2,
                    global_x_diff_abs,
                    globalBorrowAny,
                    grid,
                    block); // x_diff = A1 - A0
            } else {
                x_diff_sign = 1;
                SubtractDigitsParallel<SharkFloatParams, NewN>(
                    shared_data,
                    a_shared,
                    a_shared + n,
                    subtractionBorrows,
                    subtractionBorrows2,
                    global_x_diff_abs,
                    globalBorrowAny,
                    grid,
                    block); // x_diff = A0 - A1
            }

            // Compute y_diff_abs and y_diff_sign
            int y_compare = compareDigits(b_shared + n, b_shared, n);
            if (y_compare >= 0) {
                y_diff_sign = 0;
                SubtractDigitsParallel<SharkFloatParams, NewN>(
                    shared_data,
                    b_shared + n,
                    b_shared,
                    subtractionBorrows,
                    subtractionBorrows2,
                    global_y_diff_abs,
                    globalBorrowAny,
                    grid,
                    block); // y_diff = B1 - B0
            } else {
                y_diff_sign = 1;
                SubtractDigitsParallel<SharkFloatParams, NewN>(
                    shared_data,
                    b_shared,
                    b_shared + n,
                    subtractionBorrows,
                    subtractionBorrows2,
                    global_y_diff_abs,
                    globalBorrowAny,
                    grid,
                    block); // y_diff = B0 - B1
            }
        } else {
            if (threadIdxGlobal < NewN) {
                int x_compare = compareDigits(a_shared + n, a_shared, n);

                if (x_compare >= 0) {
                    x_diff_sign = 0;
                    subtractDigits(a_shared + n, a_shared, global_x_diff_abs, n); // x_diff = A1 - A0
                } else {
                    x_diff_sign = 1;
                    subtractDigits(a_shared, a_shared + n, global_x_diff_abs, n); // x_diff = A0 - A1
                }

                // Compute y_diff_abs and y_diff_sign
                int y_compare = compareDigits(b_shared + n, b_shared, n);
                if (y_compare >= 0) {
                    y_diff_sign = 0;
                    subtractDigits(b_shared + n, b_shared, global_y_diff_abs, n); // y_diff = B1 - B0
                } else {
                    y_diff_sign = 1;
                    subtractDigits(b_shared, b_shared + n, global_y_diff_abs, n); // y_diff = B0 - B1
                }
            }
        }
    }

    // Determine the sign of Z1_temp
    int z1_sign = x_diff_sign ^ y_diff_sign;


    // Synchronize before convolution
    grid.sync();

    // ---- Convolution for Z1_temp = |x_diff| * |y_diff| ----
    // Update total_k for convolution of differences
    auto *Z1_temp_digits = &tempProducts[Z1_temp_offset];

    if constexpr (true) {
        // Replace A and B in shared memory with their absolute differences
        cg::memcpy_async(block, const_cast<uint32_t*>(x_diff_abs), global_x_diff_abs, sizeof(uint32_t) * n);
        cg::memcpy_async(block, const_cast<uint32_t*>(y_diff_abs), global_y_diff_abs, sizeof(uint32_t) * n);

        // Wait for the first batch of A to be loaded
        cg::wait(block);

        int tid = threadIdxGlobal;
        int stride = blockDim.x * SharkFloatParams::GlobalNumBlocks;
        int total_k_diff = 2 * n - 1;
        for (int k = tid; k < total_k_diff; k += stride) {
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
            int idx = k * 2;
            Z1_temp_digits[idx] = sum_low;
            Z1_temp_digits[idx + 1] = sum_high;
        }

        grid.sync();
    } else {
        constexpr auto NewTempBase = TempBase + 32 * SharkFloatParams::GlobalNumUint32;
        MultiplyDigitsOnly<SharkFloatParams, NewN / 2, NewNumBlocks / 2, NewTempBase>(
            shared_data,
            x_diff_abs,
            y_diff_abs,
            nullptr,
            nullptr,
            Z1_temp_digits,
            grid,
            block,
            tempProducts);
        grid.sync();
    }

    auto *Z1_digits = &tempProducts[Z1_offset];

    if constexpr (!SharkFloatParams::DisableAllAdditions) {

        // Synchronize before combining results
        grid.sync();

        // After computing Z1_temp (Z1'), we now form Z1 directly:
        // If z1_sign == 0: Z1 = Z2 + Z0 - Z1_temp
        // If z1_sign == 1: Z1 = Z2 + Z0 + Z1_temp

        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;

        for (int i = tid; i < total_k; i += stride) {
            // Retrieve Z0
            int z0_idx = Z0_offset + i * 2;
            uint64_t z0_low = tempProducts[z0_idx];
            uint64_t z0_high = tempProducts[z0_idx + 1];

            // Retrieve Z2
            int z2_idx = Z2_offset + i * 2;
            uint64_t z2_low = tempProducts[z2_idx];
            uint64_t z2_high = tempProducts[z2_idx + 1];

            // Retrieve Z1_temp (Z1')
            int z1_temp_idx = Z1_temp_offset + i * 2;
            uint64_t z1_temp_low = tempProducts[z1_temp_idx];
            uint64_t z1_temp_high = tempProducts[z1_temp_idx + 1];

            // Combine Z2 + Z0 first
            uint64_t temp_low, temp_high;
            Add128(z2_low, z2_high, z0_low, z0_high, temp_low, temp_high);

            uint64_t z1_low, z1_high;
            if (z1_sign == 0) {
                // same sign: Z1 = (Z2 + Z0) - Z1_temp
                Subtract128(temp_low, temp_high, z1_temp_low, z1_temp_high, z1_low, z1_high);
            } else {
                // opposite signs: Z1 = (Z2 + Z0) + Z1_temp
                Add128(temp_low, temp_high, z1_temp_low, z1_temp_high, z1_low, z1_high);
            }

            // Store fully formed Z1
            int z1_idx = i * 2;
            Z1_digits[z1_idx] = z1_low;
            Z1_digits[z1_idx + 1] = z1_high;
        }

        // Synchronize before final combination
        grid.sync();

        // Now the final combination is just:
        // final = Z0 + (Z1 << (32*n)) + (Z2 << (64*n))
        for (int i = tid; i < total_result_digits; i += stride) {
            uint64_t sum_low = 0;
            uint64_t sum_high = 0;

            // Add Z0
            if (i < 2 * n - 1) {
                int z0_idx = Z0_offset + i * 2;
                uint64_t z0_low = tempProducts[z0_idx];
                uint64_t z0_high = tempProducts[z0_idx + 1];
                Add128(sum_low, sum_high, z0_low, z0_high, sum_low, sum_high);
            }

            // Add Z1 shifted by n
            if (i >= n && (i - n) < 2 * n - 1) {
                int z1_idx = Z1_offset + (i - n) * 2;
                uint64_t z1_low = tempProducts[z1_idx];
                uint64_t z1_high = tempProducts[z1_idx + 1];
                Add128(sum_low, sum_high, z1_low, z1_high, sum_low, sum_high);
            }

            // Add Z2 shifted by 2*n
            if (i >= 2 * n && (i - 2 * n) < 2 * n - 1) {
                int z2_idx = Z2_offset + (i - 2 * n) * 2;
                uint64_t z2_low = tempProducts[z2_idx];
                uint64_t z2_high = tempProducts[z2_idx + 1];
                Add128(sum_low, sum_high, z2_low, z2_high, sum_low, sum_high);
            }

            //int result_idx = Convolution_offset + idx * 2;
            //tempProducts[result_idx] = sum_low;
            //tempProducts[result_idx + 1] = sum_high;

            int result_idx = i * 2;
            final128[result_idx] = sum_low;
            final128[result_idx + 1] = sum_high;
        }

        // Synchronize before carry propagation
        grid.sync();
    }
}

//
// static constexpr int32_t SharkFloatParams::GlobalThreadsPerBlock = /* power of 2 */;
// static constexpr int32_t SharkFloatParams::GlobalNumBlocks = /* power of 2 */;
// static constexpr int32_t SharkFloatParams::GlobalNumUint32 = SharkFloatParams::GlobalThreadsPerBlock * SharkFloatParams::GlobalNumBlocks;
// 

// Assuming that SharkFloatParams::GlobalNumUint32 can be large and doesn't fit in shared memory
// We'll use the provided global memory buffers for large intermediates
template<class SharkFloatParams>
__device__ void MultiplyHelperKaratsubaV2(
    const HpSharkFloat<SharkFloatParams> *__restrict__ A,
    const HpSharkFloat<SharkFloatParams> *__restrict__ B,
    HpSharkFloat<SharkFloatParams> *__restrict__ Out,
    cg::grid_group &grid,
    cg::thread_block &block,
    uint64_t *__restrict__ tempProducts) {

    constexpr int N = SharkFloatParams::GlobalNumUint32;         // Total number of digits
    constexpr int NewN = N;
    constexpr int NewNumBlocks = SharkFloatParams::GlobalNumBlocks;
    extern __shared__ uint32_t shared_data[];

    constexpr auto TempBase = 0;
    DefineTempProductsOffsets(TempBase);

    auto *SharkRestrict a_shared = shared_data;
    auto *SharkRestrict b_shared = a_shared + NewN;
    auto *SharkRestrict x_diff_abs = b_shared + NewN;
    auto *SharkRestrict y_diff_abs = x_diff_abs + NewN / 2;

    cg::memcpy_async(block, a_shared, A->Digits, sizeof(uint32_t) * NewN);
    cg::memcpy_async(block, b_shared, B->Digits, sizeof(uint32_t) * NewN);

    // Wait for the first batch of A to be loaded
    cg::wait(block);

    auto *final128 = &tempProducts[Convolution_offset];
    MultiplyDigitsOnly<SharkFloatParams, N, SharkFloatParams::GlobalNumBlocks, TempBase>(
        shared_data,
        a_shared,
        b_shared,
        x_diff_abs,
        y_diff_abs,
        final128,
        grid,
        block,
        tempProducts);

    // ---- Carry Propagation ----

    // Global memory for block carry-outs
    // Allocate space for gridDim.x block carry-outs after total_result_digits
    uint64_t *block_carry_outs = &tempProducts[CarryInsOffset];
    auto *resultDigits = &tempProducts[Result_offset];

    if constexpr (!SharkFloatParams::DisableCarryPropagation) {

        DefineCarryDefinitions();

        constexpr bool UseParallelCarry = true;
        uint64_t *globalCarryCheck = &tempProducts[GlobalCarryOffset];

        if constexpr (UseParallelCarry) {

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
            SerialCarryPropagation<SharkFloatParams>(
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

            grid.sync();
        }
    } else {
        grid.sync();
    }

    // ---- Finalize the Result ----
    if constexpr (!SharkFloatParams::DisableFinalConstruction) {
        // uint64_t final_carry = carryOuts_phase6[SharkFloatParams::GlobalNumBlocks - 1];

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
    cg::thread_block block = cg::this_thread_block();

    // Call the MultiplyHelper function
    //MultiplyHelper(A, B, Out, carryIns, grid, tempProducts);
    MultiplyHelperKaratsubaV2(A, B, Out, grid, block, tempProducts);
}

template<class SharkFloatParams>
__global__ void MultiplyKernelKaratsubaV2TestLoop(
    HpSharkFloat<SharkFloatParams> *A,
    HpSharkFloat<SharkFloatParams> *B,
    HpSharkFloat<SharkFloatParams> *Out,
    uint64_t *tempProducts) { // Array to store cumulative carries

    // Initialize cooperative grid group
    cg::grid_group grid = cg::this_grid();
    cg::thread_block block = cg::this_thread_block();

    for (int i = 0; i < TestIterCount; ++i) {
        // MultiplyHelper(A, B, Out, carryIns, grid, tempProducts);
        if constexpr (!SharkFloatParams::ForceNoOp) {
            MultiplyHelperKaratsubaV2(A, B, Out, grid, block, tempProducts);
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
        SharkFloatParams::GlobalThreadsPerBlock,
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

    constexpr int N = SharkFloatParams::GlobalNumUint32;
    constexpr auto n = (N + 1) / 2;              // Half of N
    constexpr auto sharedAmountBytes = UseSharedMemory ? (2 * N + 2 * n) * sizeof(uint32_t) : 0;

    if constexpr (UseCustomStream) {
        cudaFuncSetAttribute(
            MultiplyKernelKaratsubaV2<SharkFloatParams>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            sharedAmountBytes);

        PrintMaxActiveBlocks<SharkFloatParams>(sharedAmountBytes);
    }

    err = cudaLaunchCooperativeKernel(
        (void *)MultiplyKernelKaratsubaV2<SharkFloatParams>,
        dim3(SharkFloatParams::GlobalNumBlocks),
        dim3(SharkFloatParams::GlobalThreadsPerBlock),
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

    constexpr int N = SharkFloatParams::GlobalNumUint32;
    constexpr auto n = (N + 1) / 2;              // Half of N
    constexpr auto sharedAmountBytes = UseSharedMemory ? (2 * N + 2 * n) * sizeof(uint32_t) : 0;

    if constexpr (UseCustomStream) {
        cudaFuncSetAttribute(
            MultiplyKernelKaratsubaV2TestLoop<SharkFloatParams>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            sharedAmountBytes);

        PrintMaxActiveBlocks<SharkFloatParams>(sharedAmountBytes);
    }

    cudaError_t err = cudaLaunchCooperativeKernel(
        (void *)MultiplyKernelKaratsubaV2TestLoop<SharkFloatParams>,
        dim3(SharkFloatParams::GlobalNumBlocks),
        dim3(SharkFloatParams::GlobalThreadsPerBlock),
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
ExplicitlyInstantiate(Test16x4SharkParams);

ExplicitlyInstantiate(Test128x64SharkParams);
ExplicitlyInstantiate(Test64x64SharkParams);
ExplicitlyInstantiate(Test32x64SharkParams);
ExplicitlyInstantiate(Test16x64SharkParams);

ExplicitlyInstantiate(Test128x32SharkParams);
ExplicitlyInstantiate(Test128x16SharkParams);
ExplicitlyInstantiate(Test128x8SharkParams);
ExplicitlyInstantiate(Test128x4SharkParams);