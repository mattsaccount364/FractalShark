#include "Multiply.cuh"

#include <cuda_runtime.h>

#include "HpSharkFloat.cuh"
#include "BenchmarkTimer.h"
#include "DebugChecksum.cuh"

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

#ifdef _DEBUG
#define SharkForceInlineReleaseOnly
#else
// #define SharkForceInlineReleaseOnly __forceinline__
#define SharkForceInlineReleaseOnly
#endif

template<int n1, int n2>
__device__ int CompareDigits(const uint32_t *highArray, const uint32_t *lowArray) {
    // The biggest possible “digit index” is one less
    // than the max of the two sizes.
    int maxLen = std::max(n1, n2);

    // Compare top-down, from maxLen-1 down to 0
    for (int i = maxLen - 1; i >= 0; --i) {
        // Treat out-of-range as zero
        uint32_t a_val = (i < n1) ? highArray[i] : 0u;
        uint32_t b_val = (i < n2) ? lowArray[i] : 0u;

        if (a_val > b_val) {
            return 1;  // A is bigger
        } else if (a_val < b_val) {
            return -1; // B is bigger
        }
    }
    return 0;
}

template<int n1, int n2>
__device__ static void SubtractDigitsSerial(const uint32_t *a, const uint32_t *b, uint32_t *result) {
    uint64_t borrow = 0;
    for (int i = 0; i < n1; ++i) {
        uint64_t ai;
        uint64_t bi;

        ai = a[i];

        if (i >= n2) {
            bi = 0;
        } else {
            bi = b[i];
        }

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
 * Parallel subtraction (a1 - b1) and (a2 - b2), stored in global_(x|y)_diff_abs,
 * using a multi-pass approach to propagate borrows.
 *
 * The function attempts to subtract each digit of 'b' from 'a' in parallel,
 * then uses repeated passes (do/while) to handle newly introduced borrows
 * until no more remain or a maximum pass count is reached.
 * 
 * Corrupts x_diff_abs shared memory intentionally
 */
template<
    class SharkFloatParams,
    int a1n,
    int b1n,
    int a2n,
    int b2n,
    int ExecutionBlockBase,
    int ExecutionNumBlocks>
__device__ SharkForceInlineReleaseOnly void SubtractDigitsParallel(
    uint32_t *__restrict__ x_diff_abs,
    uint32_t *__restrict__ y_diff_abs,
    const uint32_t *__restrict__ a1,
    const uint32_t *__restrict__ b1,
    const uint32_t *__restrict__ a2,
    const uint32_t *__restrict__ b2,
    uint32_t *__restrict__ subtractionBorrows1a,
    uint32_t *__restrict__ subtractionBorrows1b,
    uint32_t *__restrict__ subtractionBorrows2a,
    uint32_t *__restrict__ subtractionBorrows2b,
    uint32_t *__restrict__ global_x_diff_abs,
    uint32_t *__restrict__ global_y_diff_abs,
    uint32_t *__restrict__ globalBorrowAny,
    cg::grid_group &grid,
    cg::thread_block &block
) {
    // Note: stops on this.
    auto *sharedBorrowAny = x_diff_abs;

    // Note: not ExecutionBlockBase
    if (block.group_index().x == 0 && block.thread_index().x == 0) {
        *globalBorrowAny = 0;
    }

    if (block.thread_index().x == 0) {
        *sharedBorrowAny = 0;
    }

    // Constants 
    constexpr int MaxPasses = 5000;     // maximum number of multi-pass sweeps

    // We'll define a grid–stride range covering [0..n) for each pass
    // 1) global thread id
    int tid = (block.group_index().x - ExecutionBlockBase) * block.dim_threads().x + block.thread_index().x;
    // 2) stride
    int stride = block.dim_threads().x * ExecutionNumBlocks;

    constexpr auto n1max = std::max(a1n, b1n);
    constexpr auto n2max = std::max(a2n, b2n);
    constexpr auto nmax = std::max(n1max, n2max);

    // (1) First pass: naive partial difference (a[i] - b[i]) and set borrowBit
    // Instead of dividing digits among blocks, each thread does a grid–stride loop:
    for (int idx = tid; idx < nmax; idx += stride) {
        uint32_t ai1;
        uint32_t bi1;
        uint32_t ai2;
        uint32_t bi2;

        // Fill in with 0s if idx is out of bounds
        if (idx < a1n) {
            ai1 = a1[idx];
        } else {
            ai1 = 0;
        }

        if (idx < a2n) {
            ai2 = a2[idx];
        } else {
            ai2 = 0;
        }

        if (idx < b1n) {
            bi1 = b1[idx];
        } else {
            bi1 = 0;
        }

        if (idx < b2n) {
            bi2 = b2[idx];
        } else {
            bi2 = 0;
        }

        // naive difference
        uint64_t diff1 = (uint64_t)ai1 - (uint64_t)bi1;
        uint64_t diff2 = (uint64_t)ai2 - (uint64_t)bi2;

        uint32_t borrow1 = (ai1 < bi1) ? 1u : 0u;
        uint32_t borrow2 = (ai2 < bi2) ? 1u : 0u;

        global_x_diff_abs[idx] = static_cast<uint32_t>(diff1 & 0xFFFFFFFFu);
        subtractionBorrows1a[idx] = borrow1;

        global_y_diff_abs[idx] = static_cast<uint32_t>(diff2 & 0xFFFFFFFFu);
        subtractionBorrows2a[idx] = borrow2;
    }

    // sync the entire grid before multi-pass fixes
    //grid.sync();

    // We'll do repeated passes to fix newly introduced borrows
    uint32_t *curBorrow1 = subtractionBorrows1a;
    uint32_t *newBorrow1 = subtractionBorrows1b;
    uint32_t *curBorrow2 = subtractionBorrows2a;
    uint32_t *newBorrow2 = subtractionBorrows2b;
    int pass = 0;
    uint32_t initialBorrowAny = 0;

    grid.sync();

    do {
        // (2) For each digit, apply the borrow from the previous digit
        for (int idx = tid; idx < nmax; idx += stride) {
            uint64_t borrow_in1 = 0ULL;
            uint64_t borrow_in2 = 0ULL;
            if (idx > 0) {   // borrow_in is from digit (idx-1)
                borrow_in1 = (uint64_t)(curBorrow1[idx - 1]);
                borrow_in2 = (uint64_t)(curBorrow2[idx - 1]);
            }

            uint32_t digit1 = global_x_diff_abs[idx];
            uint32_t digit2 = global_y_diff_abs[idx];

            // subtract the borrow
            uint64_t sum1 = (uint64_t)digit1 - borrow_in1;
            uint64_t sum2 = (uint64_t)digit2 - borrow_in2;

            // store updated digit
            global_x_diff_abs[idx] = static_cast<uint32_t>(sum1 & 0xFFFFFFFFULL);
            global_y_diff_abs[idx] = static_cast<uint32_t>(sum2 & 0xFFFFFFFFULL);

            // If sum is negative => top bit is 1 => new borrow
            if (sum1 & 0x8000'0000'0000'0000ULL) {
                newBorrow1[idx] = 1;
                atomicAdd(sharedBorrowAny, 1);
            } else {
                newBorrow1[idx] = 0;
            }

            if (sum2 & 0x8000'0000'0000'0000ULL) {
                newBorrow2[idx] = 1;
                atomicAdd(sharedBorrowAny, 1);
            } else {
                newBorrow2[idx] = 0;
            }
        }

        // (a) Block-level synchronization (so all threads see final sharedBorrowAny)
        block.sync();

        // The block's thread 0 aggregates once into globalBorrowAny
        if (block.thread_index().x == 0) {
            // Add sharedBorrowAny to the global counter
            atomicAdd(globalBorrowAny, *sharedBorrowAny);

            // Reset local aggregator for the next pass
            *sharedBorrowAny = 0;
        }

        // sync before checking if any new borrows remain
        grid.sync();

        auto tempCopyGlobalBorrowAny = *globalBorrowAny;
        if (tempCopyGlobalBorrowAny == initialBorrowAny) {
            break;  // no new borrows => done
        }

        grid.sync();
        initialBorrowAny = tempCopyGlobalBorrowAny;

        // swap curBorrow, newBorrow
        uint32_t *tmp = curBorrow1;
        curBorrow1 = newBorrow1;
        newBorrow1 = tmp;

        tmp = curBorrow2;
        curBorrow2 = newBorrow2;
        newBorrow2 = tmp;

        pass++;
    } while (pass < MaxPasses);

    if constexpr (SharkDebug) {
        if (pass == MaxPasses && block.group_index().x == 0) {
            // This will deadlock the kernel because this problem is hard to diagnose
            grid.sync();
        }
    }
}



// Function to perform addition with carry
__device__ SharkForceInlineReleaseOnly static void Add128(
    uint64_t a_low, uint64_t a_high,
    uint64_t b_low, uint64_t b_high,
    uint64_t &result_low, uint64_t &result_high) {

    result_low = a_low + b_low;
    uint64_t carry = (result_low < a_low) ? 1 : 0;
    result_high = a_high + b_high + carry;
}

__device__ SharkForceInlineReleaseOnly static void Subtract128(
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
__device__ SharkForceInlineReleaseOnly static void SerialCarryPropagation(
    uint64_t *__restrict__ shared_carries,
    cg::grid_group &grid,
    cg::thread_block &block,
    const uint3 &threadIdx,
    const uint3 &blockIdx,
    int thread_start_idx,
    int thread_end_idx,
    int Convolution_offset,
    int Result_offset,
    uint64_t *__restrict__ block_carry_outs,
    uint64_t *__restrict__ tempProducts,
    uint64_t *__restrict__ globalCarryCheck) {

    if (block.thread_index().x == 0 && block.group_index().x == 0) {
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
__device__ SharkForceInlineReleaseOnly static void CarryPropagation (
    uint64_t *__restrict__ shared_carries,
    cg::grid_group &grid,
    cg::thread_block &block,
    const uint3 &threadIdx,
    const uint3 &blockIdx,
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
    constexpr int MaxPasses = 150; // Maximum number of carry propagation passes
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

    if (block.thread_index().x == SharkFloatParams::GlobalThreadsPerBlock - 1) {
        block_carry_outs[block.group_index().x] = local_carry;
    } else {
        shared_carries[block.thread_index().x] = local_carry;
    }

    // Inter-Block Carry Propagation
    int pass = 0;

    do {
        // Synchronize all blocks
        grid.sync();

        // Zero out the global carry count for the current pass
        if (block.group_index().x == 0 && block.thread_index().x == 0) {
            *carries_remaining_global = 0;
        }

        // Get carry-in from the previous block
        local_carry = 0;
        if (block.thread_index().x == 0 && block.group_index().x > 0) {
            local_carry = block_carry_outs[block.group_index().x - 1];
        } else {
            if (block.thread_index().x > 0) {
                local_carry = shared_carries[block.thread_index().x - 1];
            }
        }

        // Each thread processes its assigned digits
        for (int idx = thread_start_idx; idx < thread_end_idx; ++idx) {
            // Read the previously stored digit
            uint32_t digit = tempProducts[Result_offset + idx];

            // Add local_carry to digit
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

        shared_carries[block.thread_index().x] = local_carry;
        block.sync();

        // The block's carry-out is the carry from the last thread
        auto temp = shared_carries[block.thread_index().x];
        if (block.thread_index().x == SharkFloatParams::GlobalThreadsPerBlock - 1) {
            block_carry_outs[block.group_index().x] = temp;
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
    if (block.thread_index().x == 0 && block.group_index().x == grid.dim_blocks().x - 1) {
        uint64_t final_carry = block_carry_outs[block.group_index().x];
        if (final_carry > 0) {
            // Store the final carry as an additional digit
            tempProducts[Result_offset + total_result_digits] = static_cast<uint32_t>(final_carry & 0xFFFFFFFFULL);
            // Optionally, you may need to adjust total_result_digits
        }
    }

    // Synchronize all blocks before finalization
    // grid.sync();
}

// Look for CalculateFrameSize and ScratchMemoryArrays
// and make sure the number of NewN arrays we're using here fits within that limit.
// The list here should go up to ScratchMemoryArrays.
static_assert(AdditionalUInt64PerFrame == 256, "See below");
#define DefineTempProductsOffsets(TempBase, CallIndex) \
    const int threadIdxGlobal = block.group_index().x * SharkFloatParams::GlobalThreadsPerBlock + block.thread_index().x; \
    constexpr int TestMultiplier = 1; \
    constexpr auto CallOffset = CallIndex * CalculateFrameSize<SharkFloatParams>(); \
    constexpr auto TempBaseOffset = TempBase + CallOffset; \
    constexpr auto BorrowGlobalOffset = 0; \
    constexpr auto Checksum_offset = AdditionalGlobalSyncSpace; \
    auto *debugTrackerArray = reinterpret_cast<DebugState<SharkFloatParams>*>(&tempProducts[Checksum_offset]); \
    constexpr auto Z0_offset = TempBaseOffset + AdditionalUInt64PerFrame; \
    constexpr auto Z2_offset = Z0_offset + 4 * NewN * TestMultiplier; \
    constexpr auto Z1_temp_offset = Z2_offset + 4 * NewN * TestMultiplier; \
    constexpr auto Z1_offset = Z1_temp_offset + 4 * NewN * TestMultiplier; \
    constexpr auto Convolution_offset = Z1_offset + 4 * NewN * TestMultiplier;       /* 17 */ \
    constexpr auto Result_offset = Convolution_offset + 4 * NewN * TestMultiplier;   /* 21 */ \
    constexpr auto XDiff_offset = Result_offset + 2 * NewN * TestMultiplier;         /* 23 */ \
    constexpr auto YDiff_offset = XDiff_offset + 1 * NewN * TestMultiplier;          /* 24 */ \
    constexpr auto GlobalCarryOffset = YDiff_offset + 1 * NewN * TestMultiplier;     /* 25 */ \
    constexpr auto SubtractionOffset1 = GlobalCarryOffset + 1 * NewN * TestMultiplier;   /* 26 */ \
    constexpr auto SubtractionOffset2 = SubtractionOffset1 + 1 * NewN * TestMultiplier;  /* 27 */ \
    constexpr auto SubtractionOffset3 = SubtractionOffset2 + 1 * NewN * TestMultiplier;  /* 28 */ \
    constexpr auto SubtractionOffset4 = SubtractionOffset3 + 1 * NewN * TestMultiplier;  /* 29 */


#define DefineExtraDefinitions() \
    const auto RelativeBlockIndex = block.group_index().x - ExecutionBlockBase; \
    constexpr int total_result_digits = 2 * NewN; \
    constexpr auto digits_per_block = NewN * 2 / ExecutionNumBlocks; \
    auto block_start_idx = block.group_index().x * digits_per_block; \
    auto block_end_idx = min(block_start_idx + digits_per_block, total_result_digits); \
    int digits_per_thread = (digits_per_block + block.dim_threads().x - 1) / block.dim_threads().x; \
    int thread_start_idx = block_start_idx + block.thread_index().x * digits_per_thread; \
    int thread_end_idx = min(thread_start_idx + digits_per_thread, block_end_idx);

#define DefineCarryDefinitions() \
    constexpr int total_result_digits = 2 * NewN; \
    constexpr auto digits_per_block = SharkFloatParams::GlobalThreadsPerBlock * 2; \
    auto block_start_idx = block.group_index().x * digits_per_block; \
    auto block_end_idx = min(block_start_idx + digits_per_block, total_result_digits); \
    int digits_per_thread = (digits_per_block + block.dim_threads().x - 1) / block.dim_threads().x; \
    int thread_start_idx = block_start_idx + block.thread_index().x * digits_per_thread; \
    int thread_end_idx = min(thread_start_idx + digits_per_thread, block_end_idx);

template<
    class SharkFloatParams,
    int CallIndex,
    DebugStatePurpose Purpose>
__device__ SharkForceInlineReleaseOnly void
EraseCurrentDebugState(
    bool record,
    DebugState<SharkFloatParams> *debugTrackerArray,
    cooperative_groups::grid_group &grid,
    cooperative_groups::thread_block &block) {

    constexpr auto maxPurposes = static_cast<int>(DebugStatePurpose::NumPurposes);
    constexpr auto curPurpose = static_cast<int>(Purpose);
    debugTrackerArray[CallIndex * maxPurposes + curPurpose].Erase(
        record, grid, block, Purpose, CallIndex);
}

template<
    class SharkFloatParams,
    int CallIndex,
    DebugStatePurpose Purpose,
    typename ArrayType>
__device__ SharkForceInlineReleaseOnly void
StoreCurrentDebugState (
    bool record,
    DebugState<SharkFloatParams> *debugTrackerArray,
    cooperative_groups::grid_group &grid,
    cooperative_groups::thread_block &block,
    const ArrayType *arrayToChecksum,
    size_t arraySize)
{
    constexpr auto maxPurposes = static_cast<int>(DebugStatePurpose::NumPurposes);
    constexpr auto curPurpose = static_cast<int>(Purpose);
    debugTrackerArray[CallIndex * maxPurposes + curPurpose].Reset(
        record, grid, block, arrayToChecksum, arraySize, Purpose, CallIndex);
}

// Assuming that SharkFloatParams::GlobalNumUint32 can be large and doesn't fit in shared memory
// We'll use the provided global memory buffers for large intermediates
// #define SharkRestrict __restrict__
#define SharkRestrict

template<
    class SharkFloatParams,
    int RecursionDepth,
    int CallIndex,
    int NewN,
    int n1,
    int n2,
    int ExecutionBlockBase,
    int ExecutionNumBlocks,
    int NewNumBlocks,
    int TempBase>
__device__ SharkForceInlineReleaseOnly void MultiplyDigitsOnly(
    uint32_t *SharkRestrict shared_data,
    const HpSharkFloat<SharkFloatParams> *SharkRestrict A,
    const HpSharkFloat<SharkFloatParams> *SharkRestrict B,
    const uint32_t *SharkRestrict aDigits,
    const uint32_t *SharkRestrict bDigits,
    uint32_t *SharkRestrict x_diff_abs,
    uint32_t *SharkRestrict y_diff_abs,
    uint64_t *SharkRestrict final128,
    cg::grid_group &grid,
    cg::thread_block &block,
    uint64_t *SharkRestrict tempProducts) {

    if ((ExecutionBlockBase > 0 && block.group_index().x < ExecutionBlockBase) ||
        block.group_index().x >= ExecutionBlockBase + ExecutionNumBlocks) {

        return;
    }

    DefineTempProductsOffsets(TempBase, CallIndex);
    constexpr auto MaxHalfN = std::max(n1, n2);
    constexpr int total_k = MaxHalfN * 2 - 1; // Total number of k values
    constexpr bool UseConvolution =
        (NewNumBlocks <= std::max(SharkFloatParams::GlobalNumBlocks / SharkFloatParams::ConvolutionLimit, 1) ||
        (NewNumBlocks % 3 != 0));
    constexpr bool EnableSharedDiff = true; // TODO
    constexpr bool UseParallelSubtract = true;

    using DebugState = DebugState<SharkFloatParams>;

    const bool record = block.thread_index().x == 0 && block.group_index().x == ExecutionBlockBase;

    if constexpr (DebugChecksums) {
        grid.sync();

        EraseCurrentDebugState<SharkFloatParams, CallIndex, DebugStatePurpose::Invalid>(
            record, debugTrackerArray, grid, block);
        StoreCurrentDebugState<SharkFloatParams, CallIndex, DebugStatePurpose::ADigits, uint32_t>(
            record, debugTrackerArray, grid, block, aDigits, NewN);
        StoreCurrentDebugState<SharkFloatParams, CallIndex, DebugStatePurpose::BDigits, uint32_t>(
            record, debugTrackerArray, grid, block, bDigits, NewN);

        grid.sync();
    }

    auto *Z0_OutDigits = &tempProducts[Z0_offset];
    auto *Z1_temp_digits = &tempProducts[Z1_temp_offset];
    auto *Z2_OutDigits = &tempProducts[Z2_offset];

    // Arrays to hold the absolute differences (size n)
    auto *global_x_diff_abs = reinterpret_cast<uint32_t *>(&tempProducts[XDiff_offset]);
    auto *global_y_diff_abs = reinterpret_cast<uint32_t *>(&tempProducts[YDiff_offset]);

    // ---- Compute Differences x_diff = A1 - A0 and y_diff = B1 - B0 ----

    DefineExtraDefinitions();

    int x_diff_sign = 0; // 0 if positive, 1 if negative
    int y_diff_sign = 0; // 0 if positive, 1 if negative

    // Compute x_diff_abs and x_diff_sign
    auto *SharkRestrict subtractionBorrows = reinterpret_cast<uint32_t *>(&tempProducts[SubtractionOffset1]);
    auto *SharkRestrict subtractionBorrows2 = reinterpret_cast<uint32_t *>(&tempProducts[SubtractionOffset2]);
    auto *SharkRestrict subtractionBorrows3 = reinterpret_cast<uint32_t *>(&tempProducts[SubtractionOffset3]);
    auto *SharkRestrict subtractionBorrows4 = reinterpret_cast<uint32_t *>(&tempProducts[SubtractionOffset4]);
    auto *SharkRestrict globalBorrowAny = reinterpret_cast<uint32_t *>(&tempProducts[BorrowGlobalOffset]);

    const auto SharkRestrict *a_high = aDigits + n1;
    const auto SharkRestrict *b_high = bDigits + n1;
    const auto SharkRestrict *a_low = aDigits;
    const auto SharkRestrict *b_low = bDigits;

    if constexpr (DebugChecksums) {
        grid.sync();

        StoreCurrentDebugState<SharkFloatParams, CallIndex, DebugStatePurpose::AHalfHigh>(
            record, debugTrackerArray, grid, block, a_high, n2);
        StoreCurrentDebugState<SharkFloatParams, CallIndex, DebugStatePurpose::AHalfLow>(
            record, debugTrackerArray, grid, block, a_low, n1);
        StoreCurrentDebugState<SharkFloatParams, CallIndex, DebugStatePurpose::BHalfHigh>(
            record, debugTrackerArray, grid, block, b_high, n2);
        StoreCurrentDebugState<SharkFloatParams, CallIndex, DebugStatePurpose::BHalfLow>(
            record, debugTrackerArray, grid, block, b_low, n1);

        grid.sync();
    }

    if constexpr (!SharkFloatParams::DisableSubtraction) {
        if constexpr (UseParallelSubtract) {
            int x_compare = CompareDigits<n2, n1>(a_high, a_low);
            int y_compare = CompareDigits<n2, n1>(b_high, b_low);

            if (x_compare >= 0 && y_compare >= 0) {
                x_diff_sign = 0;
                y_diff_sign = 0;
                SubtractDigitsParallel<
                    SharkFloatParams,
                    n2,
                    n1,
                    n2,
                    n1,
                    ExecutionBlockBase,
                    ExecutionNumBlocks>(
                        x_diff_abs,
                        y_diff_abs,
                        a_high,
                        a_low,
                        b_high,
                        b_low,
                        subtractionBorrows,
                        subtractionBorrows2,
                        subtractionBorrows3,
                        subtractionBorrows4,
                        global_x_diff_abs,
                        global_y_diff_abs,
                        globalBorrowAny,
                        grid,
                        block);
            } else if (x_compare < 0 && y_compare < 0) {
                x_diff_sign = 1;
                y_diff_sign = 1;
                SubtractDigitsParallel<
                    SharkFloatParams,
                    n1,
                    n2,
                    n1,
                    n2,
                    ExecutionBlockBase,
                    ExecutionNumBlocks>(
                        x_diff_abs,
                        y_diff_abs,
                        a_low,
                        a_high,
                        b_low,
                        b_high,
                        subtractionBorrows,
                        subtractionBorrows2,
                        subtractionBorrows3,
                        subtractionBorrows4,
                        global_x_diff_abs,
                        global_y_diff_abs,
                        globalBorrowAny,
                        grid,
                        block);
            } else if (x_compare >= 0 && y_compare < 0) {
                x_diff_sign = 0;
                y_diff_sign = 1;
                SubtractDigitsParallel<
                    SharkFloatParams,
                    n2,
                    n1,
                    n1,
                    n2,
                    ExecutionBlockBase,
                    ExecutionNumBlocks>(
                        x_diff_abs,
                        y_diff_abs,
                        a_high,
                        a_low,
                        b_low,
                        b_high,
                        subtractionBorrows,
                        subtractionBorrows2,
                        subtractionBorrows3,
                        subtractionBorrows4,
                        global_x_diff_abs,
                        global_y_diff_abs,
                        globalBorrowAny,
                        grid,
                        block);
            } else {
                x_diff_sign = 1;
                y_diff_sign = 0;
                SubtractDigitsParallel<
                    SharkFloatParams,
                    n1,
                    n2,
                    n2,
                    n1,
                    ExecutionBlockBase,
                    ExecutionNumBlocks>(
                        x_diff_abs,
                        y_diff_abs,
                        a_low,
                        a_high,
                        b_high,
                        b_low,
                        subtractionBorrows,
                        subtractionBorrows2,
                        subtractionBorrows3,
                        subtractionBorrows4,
                        global_x_diff_abs,
                        global_y_diff_abs,
                        globalBorrowAny,
                        grid,
                        block);
            }
        } else {
            if (block.thread_index().x == 0 && block.group_index().x == ExecutionBlockBase) {
                int x_compare = CompareDigits<n1, n2>(a_high, a_low);

                if (x_compare >= 0) {
                    x_diff_sign = 0;
                    SubtractDigitsSerial<n2, n1>(a_high, a_low, global_x_diff_abs); // x_diff = A1 - A0
                } else {
                    x_diff_sign = 1;
                    SubtractDigitsSerial<n1, n2>(a_low, a_high, global_x_diff_abs); // x_diff = A0 - A1
                }

                // Compute y_diff_abs and y_diff_sign
                int y_compare = CompareDigits<n1, n2>(b_high, b_low);
                if (y_compare >= 0) {
                    y_diff_sign = 0;
                    SubtractDigitsSerial<n2, n1>(b_high, b_low, global_y_diff_abs); // y_diff = B1 - B0
                } else {
                    y_diff_sign = 1;
                    SubtractDigitsSerial<n1, n2>(b_low, b_high, global_y_diff_abs); // y_diff = B0 - B1
                }
            }
        }
    }


    if constexpr (DebugChecksums) {
        grid.sync();
    
        StoreCurrentDebugState<SharkFloatParams, CallIndex, DebugStatePurpose::XDiff>(
            record, debugTrackerArray, grid, block, global_x_diff_abs, MaxHalfN);
        StoreCurrentDebugState<SharkFloatParams, CallIndex, DebugStatePurpose::YDiff>(
            record, debugTrackerArray, grid, block, global_y_diff_abs, MaxHalfN);

        grid.sync();
    }

    constexpr auto NumBlocksRatio =
        SharkFloatParams::ConvolutionLimitPow *
        SharkFloatParams::GlobalNumBlocks /
        NewNumBlocks;

    constexpr auto SubNewNRoundUp = (NewN + 1) / 2;
    constexpr auto SubNewN2a = SubNewNRoundUp / 2;
    constexpr auto SubNewN1a = SubNewNRoundUp - SubNewN2a;   /* n1 is larger or same */

    constexpr auto SubRemainingNewN = NewN - SubNewNRoundUp;
    constexpr auto SubNewN2b = SubRemainingNewN / 2;
    constexpr auto SubNewN1b = SubRemainingNewN - SubNewN2b;   /* n1 is larger or same */

    // Determine the sign of Z1_temp
    int z1_sign = x_diff_sign ^ y_diff_sign;

    if constexpr (UseConvolution) {
        // Replace A and B in shared memory with their absolute differences
        if constexpr (EnableSharedDiff) {
            cg::memcpy_async(block, const_cast<uint32_t *>(x_diff_abs), global_x_diff_abs, sizeof(uint32_t) * MaxHalfN);
            cg::memcpy_async(block, const_cast<uint32_t *>(y_diff_abs), global_y_diff_abs, sizeof(uint32_t) * MaxHalfN);
        }

        const int tid = RelativeBlockIndex * block.dim_threads().x + block.thread_index().x;
        const int stride = block.dim_threads().x * ExecutionNumBlocks;

        if constexpr (EnableSharedDiff) {
            // Wait for the first batch of A to be loaded
            cg::wait(block);
        }

        // A single loop that covers 2*total_k elements
        for (int idx = tid; idx < 3 * total_k; idx += stride) {
            
            // Check if idx < total_k => handle Z0, else handle Z2
            if (idx < total_k) {
                // Z0 partial sums
                int k = idx;
                uint64_t sum_low = 0ULL, sum_high = 0ULL;

                int i_start = (k < n1) ? 0 : (k - (n1 - 1));
                int i_end = (k < n1) ? k : (n1 - 1);

                for (int i = i_start; i <= i_end; i++) {
                    uint64_t a;
                    uint64_t b;

                    a = aDigits[i]; // A_shared[i];         // A0[i]
                    b = bDigits[k - i]; // B_shared[k - i];     // B0[k - i]

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
            } else if (idx < 2 * total_k) {
                // Z2 partial sums
                int k = idx - total_k; // shift to [0..total_k-1]
                uint64_t sum_low = 0ULL, sum_high = 0ULL;

                int i_start = (k < n2) ? 0 : (k - (n2 - 1));
                int i_end = (k < n2) ? k : (n2 - 1);

                for (int i = i_start; i <= i_end; i++) {
                    uint64_t a;
                    uint64_t b;

                    a = aDigits[i + n1]; // A_shared[i];         // A1[i]
                    b = bDigits[k - i + n1]; // B_shared[k - i];     // B1[k - i]

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
            } else {
                int k = idx - 2 * total_k; // shift to [0..total_k-1]
                uint64_t sum_low = 0;
                uint64_t sum_high = 0;

                int i_start = (k < MaxHalfN) ? 0 : (k - (MaxHalfN - 1));
                int i_end = (k < MaxHalfN) ? k : (MaxHalfN - 1);

                for (int i = i_start; i <= i_end; ++i) {
                    uint64_t a;
                    uint64_t b;

                    a = EnableSharedDiff ? x_diff_abs[i] : global_x_diff_abs[i];
                    b = EnableSharedDiff ? y_diff_abs[k - i] : global_y_diff_abs[k - i];

                    uint64_t product = a * b;

                    // Accumulate the product
                    sum_low += product;
                    if (sum_low < product) {
                        sum_high += 1;
                    }
                }

                // Store sum_low and sum_high in tempProducts
                int out_idx = k * 2;
                Z1_temp_digits[out_idx] = sum_low;
                Z1_temp_digits[out_idx + 1] = sum_high;
            }
        }

        if constexpr (DebugChecksums) {
            grid.sync();

            StoreCurrentDebugState<SharkFloatParams, CallIndex, DebugStatePurpose::Z0>(
                record, debugTrackerArray, grid, block, Z0_OutDigits, total_k * 2);
            StoreCurrentDebugState<SharkFloatParams, CallIndex, DebugStatePurpose::Z2>(
                record, debugTrackerArray, grid, block, Z2_OutDigits, total_k * 2);
            StoreCurrentDebugState<SharkFloatParams, CallIndex, DebugStatePurpose::Z1_offset>(
                record, debugTrackerArray, grid, block, Z1_temp_digits, total_k * 2);

            grid.sync();
        }

    } else {
        static_assert(RecursionDepth <= 5, "Unexpected recursion depth");

        MultiplyDigitsOnly<
            SharkFloatParams,
            RecursionDepth + 1,
            CallIndex * 3 - 1,
            SubNewNRoundUp,
            SubNewN1a,
            SubNewN2a,
            ExecutionBlockBase,
            ExecutionNumBlocks / 3,
            NewNumBlocks / 3,
            TempBase>(
            shared_data,
            A,
            B,
            aDigits,
            bDigits,
            x_diff_abs,
            y_diff_abs,
            Z0_OutDigits,
            grid,
            block,
            tempProducts);

        MultiplyDigitsOnly<
            SharkFloatParams,
            RecursionDepth + 1,
            CallIndex * 3,
            SubRemainingNewN,
            SubNewN1b,
            SubNewN2b,
            ExecutionBlockBase + ExecutionNumBlocks / 3,
            ExecutionNumBlocks / 3,
            NewNumBlocks / 3,
            TempBase>(
            shared_data,
            A,
            B,
            aDigits + n1,
            bDigits + n1,
            x_diff_abs,
            y_diff_abs,
            Z2_OutDigits,
            grid,
            block,
            tempProducts);

        //grid.sync();

        {
            constexpr auto NewExecutionBlockBase = ExecutionBlockBase + 2 * ExecutionNumBlocks / 3;
            constexpr auto NewExecutionNumBlocks = ExecutionNumBlocks / 3;

            const bool ExecuteAtAll =
                !((NewExecutionBlockBase > 0 && block.group_index().x < NewExecutionBlockBase) ||
                    block.group_index().x >= NewExecutionBlockBase + NewExecutionNumBlocks);
            constexpr auto MaxSubNewN = std::max(SubNewN1a, SubNewN2a);

            if (ExecuteAtAll) {
                // Replace A and B in shared memory with their absolute differences
                if constexpr (EnableSharedDiff) {
                    cg::memcpy_async(block,
                        const_cast<uint32_t *>(aDigits),
                        global_x_diff_abs,
                        sizeof(uint32_t) * MaxHalfN);
                    cg::memcpy_async(block,
                        const_cast<uint32_t *>(bDigits),
                        global_y_diff_abs,
                        sizeof(uint32_t) * MaxHalfN);
                    cg::wait(block);
                }

                MultiplyDigitsOnly<
                    SharkFloatParams,
                    RecursionDepth + 1,
                    CallIndex * 3 + 1,
                    SubNewNRoundUp,
                    SubNewN1a,
                    SubNewN2a,
                    NewExecutionBlockBase,
                    NewExecutionNumBlocks,
                    NewNumBlocks / 3,
                    TempBase>(
                        shared_data,
                        A,
                        B,
                        EnableSharedDiff ? aDigits : global_x_diff_abs,
                        EnableSharedDiff ? bDigits : global_y_diff_abs,
                        x_diff_abs,
                        y_diff_abs,
                        Z1_temp_digits,
                        grid,
                        block,
                        tempProducts);

                if constexpr (EnableSharedDiff) {
                    cg::memcpy_async(block,
                        const_cast<uint32_t *>(aDigits),
                        A->Digits,
                        sizeof(uint32_t) * SharkFloatParams::GlobalNumUint32);
                    cg::memcpy_async(block,
                        const_cast<uint32_t *>(bDigits),
                        B->Digits,
                        sizeof(uint32_t) * SharkFloatParams::GlobalNumUint32);
                    cg::wait(block);
                }
            }

            if constexpr (DebugChecksums) {
                grid.sync();

                StoreCurrentDebugState<SharkFloatParams, CallIndex, DebugStatePurpose::Z0>(
                    record, debugTrackerArray, grid, block, Z0_OutDigits, SubNewNRoundUp * 2 * 2);
                StoreCurrentDebugState<SharkFloatParams, CallIndex, DebugStatePurpose::Z2>(
                    record, debugTrackerArray, grid, block, Z2_OutDigits, SubRemainingNewN * 2 * 2);
                StoreCurrentDebugState<SharkFloatParams, CallIndex, DebugStatePurpose::Z1_offset>(
                    record, debugTrackerArray, grid, block, Z1_temp_digits, SubNewNRoundUp * 2 * 2);

                grid.sync();
            }
        }
    }

    grid.sync();

    auto *Z1_digits = &tempProducts[Z1_offset];

    if constexpr (!SharkFloatParams::DisableAllAdditions) {

        // After computing Z1_temp (Z1'), we now form Z1 directly:
        // If z1_sign == 0: Z1 = Z2 + Z0 - Z1_temp
        // If z1_sign == 1: Z1 = Z2 + Z0 + Z1_temp

        const int tid = RelativeBlockIndex * block.dim_threads().x + block.thread_index().x;
        const int stride = block.dim_threads().x * ExecutionNumBlocks;

        for (int i = tid; i < total_k; i += stride) {
            // Retrieve Z0
            int z0_idx = i * 2;
            uint64_t z0_low = Z0_OutDigits[z0_idx];
            uint64_t z0_high = Z0_OutDigits[z0_idx + 1];

            // Retrieve Z2
            int z2_idx = i * 2;
            uint64_t z2_low = Z2_OutDigits[z2_idx];
            uint64_t z2_high = Z2_OutDigits[z2_idx + 1];

            // Retrieve Z1_temp (Z1')
            int z1_temp_idx = i * 2;
            uint64_t z1_temp_low = Z1_temp_digits[z1_temp_idx];
            uint64_t z1_temp_high = Z1_temp_digits[z1_temp_idx + 1];

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

        if constexpr (DebugChecksums) {
            grid.sync();

            StoreCurrentDebugState<SharkFloatParams, CallIndex, DebugStatePurpose::Z1>(
                record, debugTrackerArray, grid, block, Z1_digits, total_k * 2);
        }

        // Synchronize before final combination
        grid.sync();

        // Now the final combination is just:
        // final = Z0 + (Z1 << (32*n)) + (Z2 << (64*n))
        for (int i = tid; i < total_result_digits; i += stride) {
            uint64_t sum_low = 0;
            uint64_t sum_high = 0;

            // Add Z0
            if (i < 2 * n1 - 1) {
                int z0_idx = i * 2;
                uint64_t z0_low = Z0_OutDigits[z0_idx];
                uint64_t z0_high = Z0_OutDigits[z0_idx + 1];
                Add128(sum_low, sum_high, z0_low, z0_high, sum_low, sum_high);
            }

            // Add Z1 shifted by n
            if (i >= n1 && (i - n1) < 2 * n1 - 1) {
                int z1_idx = (i - n1) * 2;
                uint64_t z1_low = Z1_digits[z1_idx];
                uint64_t z1_high = Z1_digits[z1_idx + 1];
                Add128(sum_low, sum_high, z1_low, z1_high, sum_low, sum_high);
            }

            // Add Z2 shifted by 2*n
            if (i >= 2 * n1 && (i - 2 * n1) < 2 * n1 - 1) {
                int z2_idx = (i - 2 * n1) * 2;
                uint64_t z2_low = Z2_OutDigits[z2_idx];
                uint64_t z2_high = Z2_OutDigits[z2_idx + 1];
                Add128(sum_low, sum_high, z2_low, z2_high, sum_low, sum_high);
            }

            int result_idx = i * 2;
            final128[result_idx] = sum_low;
            final128[result_idx + 1] = sum_high;
        }

        if constexpr (DebugChecksums) {
            grid.sync();

            StoreCurrentDebugState<SharkFloatParams, CallIndex, DebugStatePurpose::Final128>(
                record, debugTrackerArray, grid, block, final128, total_result_digits * 2);
            EraseCurrentDebugState<SharkFloatParams, CallIndex, DebugStatePurpose::Result_offset>(
                record, debugTrackerArray, grid, block);
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
__device__ void MultiplyHelperKaratsubaV2 (
    const HpSharkFloat<SharkFloatParams> *__restrict__ A,
    const HpSharkFloat<SharkFloatParams> *__restrict__ B,
    HpSharkFloat<SharkFloatParams> *__restrict__ Out,
    cg::grid_group &grid,
    cg::thread_block &block,
    uint64_t *__restrict__ tempProducts) {

    extern __shared__ uint32_t shared_data[];

    constexpr auto NewN = SharkFloatParams::GlobalNumUint32;         // Total number of digits
    constexpr auto NewN1 = (NewN + 1) / 2;
    constexpr auto NewN2 = NewN - NewN1;   /* n1 is larger or same */
    constexpr auto TempBase = AdditionalUInt64Global;
    constexpr auto CallIndex = 0;
    constexpr auto CarryInsOffset = TempBase;
    constexpr auto ExecutionBlockBase = 0;
    constexpr auto ExecutionNumBlocks = SharkFloatParams::GlobalNumBlocks;
    constexpr auto RecursionDepth = 1;
    DefineTempProductsOffsets(TempBase, CallIndex);

    auto *SharkRestrict aDigits = shared_data;
    auto *SharkRestrict bDigits = aDigits + NewN;
    auto *SharkRestrict x_diff_abs = bDigits + NewN;
    auto *SharkRestrict y_diff_abs = x_diff_abs + (NewN + 1) / 2;

    cg::memcpy_async(block, aDigits, A->Digits, sizeof(uint32_t) * NewN);
    cg::memcpy_async(block, bDigits, B->Digits, sizeof(uint32_t) * NewN);

    if constexpr (DebugChecksums) {
        const bool record = block.thread_index().x == 0 && block.group_index().x == ExecutionBlockBase;
        EraseCurrentDebugState<SharkFloatParams, CallIndex, DebugStatePurpose::Invalid>(record, debugTrackerArray, grid, block);
        EraseCurrentDebugState<SharkFloatParams, CallIndex, DebugStatePurpose::ADigits>(record, debugTrackerArray, grid, block);
        EraseCurrentDebugState<SharkFloatParams, CallIndex, DebugStatePurpose::BDigits>(record, debugTrackerArray, grid, block);
        EraseCurrentDebugState<SharkFloatParams, CallIndex, DebugStatePurpose::AHalfHigh>(record, debugTrackerArray, grid, block);
        EraseCurrentDebugState<SharkFloatParams, CallIndex, DebugStatePurpose::AHalfLow>(record, debugTrackerArray, grid, block);
        EraseCurrentDebugState<SharkFloatParams, CallIndex, DebugStatePurpose::BHalfHigh>(record, debugTrackerArray, grid, block);
        EraseCurrentDebugState<SharkFloatParams, CallIndex, DebugStatePurpose::BHalfLow>(record, debugTrackerArray, grid, block);
        EraseCurrentDebugState<SharkFloatParams, CallIndex, DebugStatePurpose::XDiff>(record, debugTrackerArray, grid, block);
        EraseCurrentDebugState<SharkFloatParams, CallIndex, DebugStatePurpose::YDiff>(record, debugTrackerArray, grid, block);
        EraseCurrentDebugState<SharkFloatParams, CallIndex, DebugStatePurpose::Z0>(record, debugTrackerArray, grid, block);
        EraseCurrentDebugState<SharkFloatParams, CallIndex, DebugStatePurpose::Z1>(record, debugTrackerArray, grid, block);
        EraseCurrentDebugState<SharkFloatParams, CallIndex, DebugStatePurpose::Z2>(record, debugTrackerArray, grid, block);
        EraseCurrentDebugState<SharkFloatParams, CallIndex, DebugStatePurpose::Z1_offset>(record, debugTrackerArray, grid, block);
        EraseCurrentDebugState<SharkFloatParams, CallIndex, DebugStatePurpose::Final128>(record, debugTrackerArray, grid, block);
        EraseCurrentDebugState<SharkFloatParams, CallIndex, DebugStatePurpose::Result_offset>(record, debugTrackerArray, grid, block);
        static_assert(static_cast<int>(DebugStatePurpose::NumPurposes) == 15, "Unexpected number of purposes");
    }

    // Wait for the first batch of A to be loaded
    cg::wait(block);

    auto *final128 = &tempProducts[Convolution_offset];
    MultiplyDigitsOnly<
        SharkFloatParams,
        RecursionDepth,
        CallIndex + 1,
        NewN,
        NewN1,
        NewN2,
        ExecutionBlockBase,
        ExecutionNumBlocks,
        SharkFloatParams::GlobalNumBlocks,
        TempBase>(
        shared_data,
        A,
        B,
        aDigits,
        bDigits,
        x_diff_abs,
        y_diff_abs,
        final128,
        grid,
        block,
        tempProducts);

    grid.sync();

    // ---- Carry Propagation ----

    // Global memory for block carry-outs
    // Allocate space for grid.dim_blocks().x block carry-outs after total_result_digits
    // Note, overlaps:
    uint64_t *block_carry_outs = &tempProducts[CarryInsOffset];

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
                block.thread_index(),
                block.group_index(),
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
                block.thread_index(),
                block.group_index(),
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

    using DebugState = DebugState<SharkFloatParams>;
    const uint64_t *resultEntries = &tempProducts[Result_offset];
    const bool record = block.thread_index().x == 0 && block.group_index().x == ExecutionBlockBase;

    if constexpr (DebugChecksums) {
        grid.sync();

        StoreCurrentDebugState<SharkFloatParams, CallIndex, DebugStatePurpose::Result_offset>(
            record, debugTrackerArray, grid, block, resultEntries, 2 * NewN);

        grid.sync();
    }

    // ---- Finalize the Result ----
    if constexpr (!SharkFloatParams::DisableFinalConstruction) {
        // uint64_t final_carry = carryOuts_phase6[SharkFloatParams::GlobalNumBlocks - 1];

        // Initial total_result_digits is 2 * NewN
        int total_result_digits = 2 * NewN;

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
        // Calculate the number of digits to shift to keep the most significant NewN digits
        int shift_digits = significant_digits - NewN;
        if (shift_digits < 0) {
            shift_digits = 0;  // No need to shift if we have fewer than NewN significant digits
        }

        if (block.group_index().x == 0 && block.thread_index().x == 0) {
            // Adjust the exponent based on the number of bits shifted
            Out->Exponent = A->Exponent + B->Exponent + shift_digits * 32;

            // Set the sign of the result
            Out->IsNegative = A->IsNegative ^ B->IsNegative;
        }

        int tid = block.thread_index().x + block.group_index().x * block.dim_threads().x;
        int stride = block.dim_threads().x * grid.dim_blocks().x;

        // src_idx is the starting index in tempProducts[] from which we copy
        int src_idx = Result_offset + shift_digits;
        int last_src = Result_offset + highest_nonzero_index; // The last valid index

        // We'll do a grid-stride loop over i in [0 .. NewN)
        for (int i = tid; i < NewN; i += stride) {
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

    for (int i = 0; i < SharkTestIterCount; ++i) {
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

    constexpr int NewN = SharkFloatParams::GlobalNumUint32;
    constexpr auto n = (NewN + 1) / 2;              // Half of NewN
    constexpr auto sharedAmountBytes = UseSharedMemory ? (2 * NewN + 2 * n) * sizeof(uint32_t) : 0;

    if constexpr (SharkCustomStream) {
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

    constexpr int NewN = SharkFloatParams::GlobalNumUint32;
    constexpr auto n = (NewN + 1) / 2;              // Half of NewN
    constexpr auto sharedAmountBytes = UseSharedMemory ? (2 * NewN + 2 * n) * sizeof(uint32_t) : 0;

    if constexpr (SharkCustomStream) {
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

ExplicitInstantiateAll();