#include "Multiply.cuh"

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda/barrier>

#include "HpSharkFloat.cuh"

#include <iostream>

namespace cg = cooperative_groups;

//
// This implementation is busted in several key ways:
// - The per-digit multiplication is not implemented correctly because
//   the sum may overflow.  Use 128-bit addition and keep track of the carries during
//   the first phase.
// - The inter-block carry propagation is incorrect
//

template<class SharkFloatParams>
__device__ void MultiplyHelperN2(
    const HpSharkFloat<SharkFloatParams> *__restrict__ A,
    const HpSharkFloat<SharkFloatParams> *__restrict__ B,
    HpSharkFloat<SharkFloatParams> *__restrict__ Out,
    uint64_t *__restrict__ carryOuts_phase3, // Array to store carry-out from Phase 3
    uint64_t *__restrict__ carryOuts_phase6, // Array to store carry-out from Phase 6
    uint64_t *__restrict__ carryIns,          // Array to store carry-in for each block
    cg::grid_group grid,
    uint64_t *__restrict__ tempProducts      // Temporary buffer to store intermediate products
) {
    // Calculate the thread's unique index
    const int threadIdxGlobal = blockIdx.x * SharkFloatParams::ThreadsPerBlock + threadIdx.x;

    const int threadIdxGlobalMin = blockIdx.x * SharkFloatParams::ThreadsPerBlock;
    const int threadIdxGlobalMax = threadIdxGlobalMin + SharkFloatParams::ThreadsPerBlock - 1;

    const int lowDigitIdxMin = threadIdxGlobalMin * 2;
    const int lowDigitIdxMax = threadIdxGlobalMax * 2;

    const int highDigitIdxMin = lowDigitIdxMin + 1;
    const int highDigitIdxMax = lowDigitIdxMax + 1;

    // Each thread handles two digits: low and high
    const int lowDigitIdx = threadIdxGlobal * 2;
    const int highDigitIdx = lowDigitIdx + 1;

    // Ensure indices do not exceed the temporary buffer size
    if (lowDigitIdx >= 2 * SharkFloatParams::NumUint32) return;

    // Initialize temporary products to zero
    tempProducts[lowDigitIdx] = 0;
    if (highDigitIdx < 2 * SharkFloatParams::NumUint32) {
        tempProducts[highDigitIdx] = 0;
    }

    static constexpr int32_t BATCH_SIZE_A = BatchSize;
    static constexpr int32_t BATCH_SIZE_B = BatchSize;

    // Compute k_min and k_max
    const int k_min = 2 * blockIdx.x * SharkFloatParams::ThreadsPerBlock;
    const int k_max = min(2 * (blockIdx.x + 1) * SharkFloatParams::ThreadsPerBlock - 1, 2 * SharkFloatParams::NumUint32 - 1);

    // Compute j_min_block and j_max_block
    const int j_min_block = max(0, k_min - (SharkFloatParams::NumUint32 - 1));
    const int j_max_block = min(k_max, SharkFloatParams::NumUint32 - 1);

    const int a_shared_size_required = j_max_block - j_min_block + 1;

    // Shared memory for A and B with double buffering
    __shared__ __align__(16) uint32_t A_shared[2][BATCH_SIZE_A];
    __shared__ __align__(16) uint32_t B_shared[2][BATCH_SIZE_B];

    const int numBatches_A = (a_shared_size_required + BATCH_SIZE_A - 1) / BATCH_SIZE_A;
    const int numBatches_B = (SharkFloatParams::NumUint32 + BATCH_SIZE_B - 1) / BATCH_SIZE_B;

    uint32_t *__restrict__ tempBufferA = nullptr;
    uint32_t *__restrict__ currentBufferA = A_shared[0];
    uint32_t *__restrict__ nextBufferA = A_shared[1];

    uint32_t *__restrict__ tempBufferB = nullptr;
    uint32_t *__restrict__ currentBufferB = B_shared[0];
    uint32_t *__restrict__ nextBufferB = B_shared[1];

    cg::thread_block block = cg::this_thread_block();

    // Start loading the first batch of A asynchronously
    const int batchStartA = j_min_block;
    const int elementsToCopyA = min(BATCH_SIZE_A, a_shared_size_required);

    cg::memcpy_async(block, &currentBufferA[0], &A->Digits[batchStartA], sizeof(uint32_t) * elementsToCopyA);

    // Wait for the first batch of A to be loaded
    cg::wait(block);

    uint64_t lowDigitIdxSum = 0;
    uint64_t highDigitIdxSum = 0;

    // Loop over batches of A
    for (int32_t batchA = 0; batchA < numBatches_A; ++batchA) {
        block.sync();

        const int batchStartA = j_min_block + batchA * BATCH_SIZE_A;
        const int batchEndA = batchStartA + elementsToCopyA - 1;

        // Start loading the next batch of A asynchronously if not the last batch
        if (batchA + 1 < numBatches_A) {
            const int nextBatchStartA = j_min_block + (batchA + 1) * BATCH_SIZE_A;
            const int nextElementsToCopyA = min(BATCH_SIZE_A, a_shared_size_required - (batchA + 1) * BATCH_SIZE_A);

            cg::memcpy_async(block, &nextBufferA[0], &A->Digits[nextBatchStartA], sizeof(uint32_t) * nextElementsToCopyA);
        }

        const int bIndex_min_low = lowDigitIdxMin - batchEndA;
        const int bIndex_max_low = lowDigitIdxMax - batchStartA;

        const int bIndex_min_high = highDigitIdxMin - batchEndA;
        const int bIndex_max_high = highDigitIdxMax - batchStartA;

        const int bIndex_min = max(0, min(bIndex_min_low, bIndex_min_high));
        const int bIndex_max = min(SharkFloatParams::NumUint32 - 1, max(bIndex_max_low, bIndex_max_high));

        const int batchB_start = bIndex_min / BATCH_SIZE_B;
        // const int batchB_end = bIndex_max / BATCH_SIZE_B;

        int batchStartB = batchB_start * BATCH_SIZE_B;
        {
            const int elementsToCopyB = min(BATCH_SIZE_B, SharkFloatParams::NumUint32 - batchStartB);
            cg::memcpy_async(block, &currentBufferB[0], &B->Digits[batchStartB], sizeof(uint32_t) * elementsToCopyB);
        }

        // Loop over batches of B
        for (int batchB = batchB_start; batchB < numBatches_B; ++batchB) {
            //block.sync();

            const int elementsToCopyB = min(BATCH_SIZE_B, SharkFloatParams::NumUint32 - batchStartB);
            const int batchEndB = batchStartB + elementsToCopyB - 1;

            // Start loading the next batch of B asynchronously if not the last batch
            if (batchB + 1 < numBatches_B) {
                int nextBatchStartB = (batchB + 1) * BATCH_SIZE_B;
                int nextElementsToCopyB = min(BATCH_SIZE_B, SharkFloatParams::NumUint32 - nextBatchStartB);

                cg::memcpy_async(block, &nextBufferB[0], &B->Digits[nextBatchStartB], sizeof(uint32_t) * nextElementsToCopyB);
                cg::wait_prior<1>(block);
            } else {
                cg::wait(block);
            }

            // Compute partial products for lowDigitIdx
            {
                uint64_t sumLow = 0;
                uint64_t sumHigh = 0;

                // Calculate the valid ranges of j for lowDigitIdx and highDigitIdx
                int j_min_low = max(batchStartA, max(j_min_block, lowDigitIdx - batchEndB));
                int j_max_low = min(batchEndA, min(j_max_block, lowDigitIdx - batchStartB));

                int j_min_high = max(batchStartA, max(j_min_block, highDigitIdx - batchEndB));
                int j_max_high = min(batchEndA, min(j_max_block, highDigitIdx - batchStartB));

                // Combined range
                int j_min = min(j_min_low, j_min_high);
                int j_max = max(j_max_low, j_max_high);

                // Iterate over the combined range
                for (int j = j_min; j <= j_max; ++j) {
                    int aSharedIndex = j - batchStartA;
                    uint32_t aValue = currentBufferA[aSharedIndex];

                    // Compute for lowDigitIdx
                    if (j >= j_min_low && j <= j_max_low) {
                        int bIndexLow = lowDigitIdx - j;
                        int bSharedIndexLow = bIndexLow - batchStartB;
                        uint32_t bValueLow = currentBufferB[bSharedIndexLow];

                        sumLow += static_cast<uint64_t>(aValue) * static_cast<uint64_t>(bValueLow);
                    }

                    // Compute for highDigitIdx
                    if (highDigitIdx < 2 * SharkFloatParams::NumUint32 && j >= j_min_high && j <= j_max_high) {
                        int bIndexHigh = highDigitIdx - j;
                        int bSharedIndexHigh = bIndexHigh - batchStartB;
                        uint32_t bValueHigh = currentBufferB[bSharedIndexHigh];

                        sumHigh += static_cast<uint64_t>(aValue) * static_cast<uint64_t>(bValueHigh);
                    }
                }
                lowDigitIdxSum += sumLow;
                highDigitIdxSum += sumHigh;
            }

            // Switch buffers for double buffering of B
            tempBufferB = currentBufferB;
            currentBufferB = nextBufferB;
            nextBufferB = tempBufferB;

            batchStartB += BATCH_SIZE_B;
        }

        // Switch buffers for double buffering of A
        tempBufferA = currentBufferA;
        currentBufferA = nextBufferA;
        nextBufferA = tempBufferA;

        // Wait for the next batch of A to be loaded
        if (batchA + 1 < numBatches_A) {
            cg::wait(block);
        }
        //block.sync();
    }

    tempProducts[lowDigitIdx] = lowDigitIdxSum;
    tempProducts[highDigitIdx] = highDigitIdxSum;

    // Shared memory to store per-thread digits and carries
    __shared__ uint64_t digitLowShared[SharkFloatParams::ThreadsPerBlock];
    __shared__ uint64_t digitHighShared[SharkFloatParams::ThreadsPerBlock];

    digitLowShared[threadIdx.x] = lowDigitIdxSum;
    digitHighShared[threadIdx.x] = highDigitIdxSum;

    grid.sync();

    if (threadIdx.x == 0) {
        uint64_t carry = 0;

        // Process the digits sequentially
        for (int i = 0; i < SharkFloatParams::ThreadsPerBlock; ++i) {
            // Get digits and carries from shared memory
            uint64_t digitLow = digitLowShared[i];
            uint64_t digitHigh = digitHighShared[i];

            // Process low digit with carry and carryLow
            uint64_t sumLow = static_cast<uint64_t>(digitLow) + carry;
            digitLowShared[i] = static_cast<uint32_t>(sumLow & 0xFFFFFFFF);
            carry = sumLow >> 32; // Update carry

            // Process high digit with carry and carryHigh
            uint64_t sumHigh = static_cast<uint64_t>(digitHigh) + carry;
            digitHighShared[i] = static_cast<uint32_t>(sumHigh & 0xFFFFFFFF);
            carry = sumHigh >> 32; // Update carry
        }

        // Store the final carry-out from the block
        carryOuts_phase6[blockIdx.x] = carry;
    }

    grid.sync(); // Ensure all blocks have access to carryIns

    // Compute carry-ins for each block
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        // Block 0 has carry-in of zero
        carryIns[0] = 0;

        // Propagate carry-outs to carry-ins for subsequent blocks
        for (int i = 1; i < SharkFloatParams::NumBlocks; ++i) {
            carryIns[i] = carryOuts_phase6[i - 1];
        }
    }

    grid.sync(); // Ensure all blocks have computed their carryOuts_phase6

    // Each block uses carryIns[blockIdx.x] as its carry-in for inter-block carry propagation
    uint64_t carry = carryIns[blockIdx.x];

    // Now, perform inter-block carry propagation within each block
    if (threadIdx.x == 0) {
        // Process the digits sequentially
        for (int i = 0; i < SharkFloatParams::ThreadsPerBlock; ++i) {
            // Get digits from shared memory
            uint64_t digitLow = static_cast<uint32_t>(digitLowShared[i]);
            uint64_t digitHigh = static_cast<uint32_t>(digitHighShared[i]);

            // Process low digit
            uint64_t sumLow = static_cast<uint64_t>(digitLow) + carry;
            digitLowShared[i] = static_cast<uint32_t>(sumLow & 0xFFFFFFFF);
            carry = sumLow >> 32;

            // Process high digit
            uint64_t sumHigh = static_cast<uint64_t>(digitHigh) + carry;
            digitHighShared[i] = static_cast<uint32_t>(sumHigh & 0xFFFFFFFF);
            carry = sumHigh >> 32;
        }

        // Store the final carry-out from the block
        carryOuts_phase6[blockIdx.x] = carry;
    }

    // Synchronize to ensure all blocks have completed inter-block carry propagation
    grid.sync();

    // Handle final carry-out from the last block
    uint64_t finalCarry = 0;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        finalCarry = carryOuts_phase6[SharkFloatParams::NumBlocks - 1];
        carryIns[0] = finalCarry; // Store final carry for later use
    }
    grid.sync();

    // Retrieve finalCarry
    finalCarry = carryIns[0];

    // After sequential carry propagation and grid synchronization

    // Each thread determines its highest non-zero index using updated digits
    int localHighestIndex = -1;
    if (digitHighShared[threadIdx.x] != 0) {
        localHighestIndex = highDigitIdx;
    } else if (digitLowShared[threadIdx.x] != 0) {
        localHighestIndex = lowDigitIdx;
    }

    // Perform reduction to find the block's highest non-zero index
    __shared__ int sharedHighestIndices[SharkFloatParams::ThreadsPerBlock];
    sharedHighestIndices[threadIdx.x] = localHighestIndex;
    block.sync();

    // Reduction within block to find blockHighestIndex
    for (int offset = SharkFloatParams::ThreadsPerBlock / 2; offset > 0; offset /= 2) {
        if (threadIdx.x < offset) {
            int other = sharedHighestIndices[threadIdx.x + offset];
            if (other > sharedHighestIndices[threadIdx.x]) {
                sharedHighestIndices[threadIdx.x] = other;
            }
        }
        block.sync();
    }

    int blockHighestIndex = sharedHighestIndices[0];

    if (threadIdx.x == 0) {
        // Store block highest index to global array
        carryOuts_phase3[blockIdx.x] = blockHighestIndex;
    }

    grid.sync();

    // Block 0 finds the global highest index
    int highestIndex = -1;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        for (int i = 0; i < SharkFloatParams::NumBlocks; ++i) {
            int idx = static_cast<int>(carryOuts_phase3[i]);
            if (idx > highestIndex) highestIndex = idx;
        }
        carryIns[0] = highestIndex; // Reuse carryIns[0] to store highest index
    }

    grid.sync();

    highestIndex = carryIns[0];

    // Calculate the total number of digits in the result
    int totalResultDigits = highestIndex + 1;

    // Calculate the initial number of shifts needed
    int shifts = totalResultDigits - SharkFloatParams::NumUint32;
    if (shifts < 0) {
        shifts = 0;
    }

    // Calculate the required shift to ensure maxHighDigitIdx - shifts <= SharkFloatParams::NumUint32 - 1
    int requiredShift = highestIndex - (SharkFloatParams::NumUint32 - 1);
    if (shifts < requiredShift) {
        shifts = requiredShift;
    }

    // Adjust the exponent accordingly
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        Out->Exponent = A->Exponent + B->Exponent + shifts * 32;
        Out->IsNegative = A->IsNegative ^ B->IsNegative;
    }

    // Each thread copies its digits from shared memory to Out->Digits, applying the shift
    if (lowDigitIdx >= shifts && (lowDigitIdx - shifts) < SharkFloatParams::NumUint32) {
        Out->Digits[lowDigitIdx - shifts] = static_cast<uint32_t>(digitLowShared[threadIdx.x]);
    }
    if (highDigitIdx >= shifts && (highDigitIdx - shifts) < SharkFloatParams::NumUint32) {
        Out->Digits[highDigitIdx - shifts] = static_cast<uint32_t>(digitHighShared[threadIdx.x]);
    }

    // Handle the final carry digits if any
    if (finalCarry != 0 && blockIdx.x == 0 && threadIdx.x == 0) {
        // Determine the number of digits needed for finalCarry
        int finalCarryBits = 0;
        uint64_t tempCarry = finalCarry;
        while (tempCarry > 0) {
            tempCarry >>= 1;
            finalCarryBits += 1;
        }
        int carryDigits = (finalCarryBits + 31) / 32;

        // Shift existing digits to make room for finalCarry digits
        for (int i = SharkFloatParams::NumUint32 - 1; i >= carryDigits; --i) {
            Out->Digits[i] = Out->Digits[i - carryDigits];
        }

        // Insert the finalCarry digits into the highest positions
        tempCarry = finalCarry;
        for (int i = 0; i < carryDigits; ++i) {
            Out->Digits[i] = static_cast<uint32_t>(tempCarry & 0xFFFFFFFF);
            tempCarry >>= 32;
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////

// MultiplyHelperKaratsuba


__device__ static void multiply_uint64(
    uint64_t a, uint64_t b,
    uint64_t &low, uint64_t &high) {
    low = a * b;
    high = __umul64hi(a, b);
}

// Function to perform addition with carry
__device__ static void addWithCarry(
    uint64_t a_low, uint64_t a_high,
    uint64_t b_low, uint64_t b_high,
    uint64_t &result_low, uint64_t &result_high) {

    result_low = a_low + b_low;
    uint64_t carry = (result_low < a_low) ? 1 : 0;
    result_high = a_high + b_high + carry;
}

// Function to perform subtraction with borrow
__device__ static void subtractWithBorrow(
    uint64_t a_low, uint64_t a_high,
    uint64_t b_low, uint64_t b_high,
    uint64_t &result_low, uint64_t &result_high) {

    bool borrow_low = a_low < b_low;
    result_low = a_low - b_low;
    uint64_t borrow_high = borrow_low ? 1 : 0;
    result_high = a_high - b_high - borrow_high;
}

template<class SharkFloatParams>
__device__ static void CarryPropagation(
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
    uint64_t *__restrict__ carryOuts_phase3) {

    // First Pass: Process convolution results to compute initial digits and local carries
    __shared__ uint64_t shared_carries[SharkFloatParams::ThreadsPerBlock];

    // Initialize local carry
    uint64_t local_carry = 0;

    // Constants and offsets
    constexpr int MaxPasses = 10; // Maximum number of carry propagation passes
    constexpr int total_result_digits = 2 * SharkFloatParams::NumUint32;

    // Each thread processes its assigned digits
    for (int idx = thread_start_idx; idx < thread_end_idx; ++idx) {
        int sum_low_idx = Convolution_offset + idx * 2;
        int sum_high_idx = sum_low_idx + 1;

        // Read sum_low and sum_high from global memory
        uint64_t sum_low = tempProducts[sum_low_idx];     // Lower 64 bits
        uint64_t sum_high = tempProducts[sum_high_idx];   // Higher 64 bits

        // Add local carry to sum_low
        uint64_t new_sum_low = sum_low + local_carry;
        uint64_t carry_from_low = (new_sum_low < sum_low) ? 1 : 0;

        // Combine sum_high and carry_from_low
        uint64_t new_sum_high = (sum_high << 32) + carry_from_low;

        // Extract digit
        uint32_t digit = static_cast<uint32_t>(new_sum_low & 0xFFFFFFFFULL);

        // Compute local carry for next digit
        local_carry = new_sum_high + (new_sum_low >> 32);

        // Store the partial digit
        tempProducts[Result_offset + idx] = digit;

        // Continue to next digit without synchronization since carries are local
    }

    if (threadIdx.x == SharkFloatParams::ThreadsPerBlock - 1) {
        block_carry_outs[blockIdx.x] = local_carry;
    } else {
        shared_carries[threadIdx.x] = local_carry;
    }

    // Synchronize all blocks
    grid.sync();

    uint64_t *carries_remaining_global = carryOuts_phase3;

    // Inter-Block Carry Propagation
    int pass = 0;

    do {
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
        for (int idx = thread_start_idx; idx < thread_end_idx; ++idx) {
            // Read the previously stored digit
            uint32_t digit = tempProducts[Result_offset + idx];

            // Add local_carry to digit
            uint64_t sum = static_cast<uint64_t>(digit) + local_carry;

            // Update digit
            digit = static_cast<uint32_t>(sum & 0xFFFFFFFFULL);
            tempProducts[Result_offset + idx] = digit;

            // Compute new local_carry for next digit
            local_carry = sum >> 32;
        }

        // Store the final local_carry of each thread into shared memory
        shared_carries[threadIdx.x] = local_carry;
        block.sync();

        // The block's carry-out is the carry from the last thread
        if (threadIdx.x == SharkFloatParams::ThreadsPerBlock - 1) {
            auto temp = shared_carries[threadIdx.x];
            block_carry_outs[blockIdx.x] = temp;
            if (temp > 0) {
                atomicAdd(carries_remaining_global, 1);
            }
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

// Assuming that SharkFloatParams::NumUint32 can be large and doesn't fit in shared memory
// We'll use the provided global memory buffers for large intermediates
template<class SharkFloatParams>
__device__ void MultiplyHelperKaratsubaV1(
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
    //block.sync();

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

    // ---- Convolution for Z1_temp = (A0 + A1) * (B0 + B1) ----
    for (int k = k_start; k < k_end; ++k) {
        uint64_t sum_low = 0;
        uint64_t sum_high = 0;

        int i_start = max(0, k - (n - 1));
        int i_end = min(k, n - 1);

        for (int i = i_start; i <= i_end; ++i) {
            // uint64_t a = A_shared[i];         // (A0 + A1)[i]
            // uint64_t b = B_shared[k - i];     // (B0 + B1)[k - i]

            uint64_t A0 = A->Digits[i];
            uint64_t A1 = A->Digits[i + n];
            uint64_t B0 = B->Digits[k - i];
            uint64_t B1 = B->Digits[k - i + n];
            auto a = A0 + A1;
            auto b = B0 + B1;

            // Compute full 128-bit product
            uint64_t prod_low, prod_high;
            multiply_uint64(a, b, prod_low, prod_high);

            // Accumulate the product
            addWithCarry(sum_low, sum_high, prod_low, prod_high, sum_low, sum_high);
        }

        // Store sum_low and sum_high in tempProducts
        int idx = Z1_temp_offset + k * 2;
        tempProducts[idx] = sum_low;
        tempProducts[idx + 1] = sum_high;
    }

    // Synchronize before subtraction
    grid.sync();

    // ---- Compute Z1 = Z1_temp - Z0 - Z2 ----
    for (int k = k_start; k < k_end; ++k) {
        // Retrieve Z1_temp
        int z1_temp_idx = Z1_temp_offset + k * 2;
        uint64_t z1_temp_low = tempProducts[z1_temp_idx];
        uint64_t z1_temp_high = tempProducts[z1_temp_idx + 1];

        // Retrieve Z0
        int z0_idx = Z0_offset + k * 2;
        uint64_t z0_low = tempProducts[z0_idx];
        uint64_t z0_high = tempProducts[z0_idx + 1];

        // Retrieve Z2
        int z2_idx = Z2_offset + k * 2;
        uint64_t z2_low = tempProducts[z2_idx];
        uint64_t z2_high = tempProducts[z2_idx + 1];

        // Compute z0 + z2
        uint64_t z0z2_low, z0z2_high;
        addWithCarry(z0_low, z0_high, z2_low, z2_high, z0z2_low, z0z2_high);

        // Compute Z1 = Z1_temp - (Z0 + Z2)
        uint64_t z1_low, z1_high;
        subtractWithBorrow(z1_temp_low, z1_temp_high, z0z2_low, z0z2_high, z1_low, z1_high);

        // Store z1_low and z1_high in tempProducts
        int z1_idx = Z1_offset + k * 2;
        tempProducts[z1_idx] = z1_low;
        tempProducts[z1_idx + 1] = z1_high;
    }

    // Synchronize before combining results
    grid.sync();

    // ---- Combine Z0, Z1, Z2 into the final result ----
    constexpr int total_result_digits = 2 * N;
    int idx_start = (threadIdxGlobal * total_result_digits) / total_threads;
    int idx_end = ((threadIdxGlobal + 1) * total_result_digits) / total_threads;

    for (int idx = idx_start; idx < idx_end; ++idx) {
        uint64_t sum_low = 0;
        uint64_t sum_high = 0;

        // Add Z0 component
        if (idx < 2 * n - 1) {
            int z0_idx = Z0_offset + idx * 2;
            uint64_t z0_low = tempProducts[z0_idx];
            uint64_t z0_high = tempProducts[z0_idx + 1];
            addWithCarry(sum_low, sum_high, z0_low, z0_high, sum_low, sum_high);
        }

        // Add Z1 component shifted by n digits
        if (idx >= n && (idx - n) < 2 * n - 1) {
            int z1_idx = Z1_offset + (idx - n) * 2;
            uint64_t z1_low = tempProducts[z1_idx];
            uint64_t z1_high = tempProducts[z1_idx + 1];
            addWithCarry(sum_low, sum_high, z1_low, z1_high, sum_low, sum_high);
        }

        // Add Z2 component shifted by 2n digits
        if (idx >= 2 * n && (idx - 2 * n) < 2 * n - 1) {
            int z2_idx = Z2_offset + (idx - 2 * n) * 2;
            uint64_t z2_low = tempProducts[z2_idx];
            uint64_t z2_high = tempProducts[z2_idx + 1];
            addWithCarry(sum_low, sum_high, z2_low, z2_high, sum_low, sum_high);
        }

        // Store sum_low and sum_high in tempProducts
        int result_idx = Convolution_offset + idx * 2;
        tempProducts[result_idx] = sum_low;
        tempProducts[result_idx + 1] = sum_high;
    }

    // Synchronize before carry propagation
    block.sync();

    // ---- Carry Propagation ----

    //if (blockIdx.x == 0 && threadIdx.x == 0) {
    //    // Only one thread performs the carry propagation
    //    uint64_t carry = 0;
    //    int total_result_digits = 2 * N;

    //    for (int idx = 0; idx < total_result_digits; ++idx) {
    //        int result_idx = Convolution_offset + idx * 2;
    //        uint64_t sum_low = tempProducts[result_idx];        // Lower 64 bits
    //        uint64_t sum_high = tempProducts[result_idx + 1];   // Higher 64 bits

    //        // Add carry to sum_low
    //        uint64_t new_sum_low = sum_low + carry;
    //        uint64_t carry_from_low = (new_sum_low < sum_low) ? 1 : 0;

    //        // Add carry_from_low to sum_high
    //        uint64_t new_sum_high = (sum_high << 32) + carry_from_low;

    //        // Extract digit (lower 32 bits of new_sum_low)
    //        uint32_t digit = static_cast<uint32_t>(new_sum_low & 0xFFFFFFFFULL);

    //        // Compute carry for the next digit
    //        carry = new_sum_high + (new_sum_low >> 32);

    //        // Store the digit
    //        tempProducts[Result_offset + idx] = digit;
    //    }

    //    // Handle final carry
    //    if (carry > 0) {
    //        tempProducts[Result_offset + total_result_digits] = static_cast<uint32_t>(carry & 0xFFFFFFFFULL);
    //        total_result_digits += 1;
    //    }
    //}

    // Initialize variables

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

        CarryPropagation<SharkFloatParams>(
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

#define ExplicitlyInstantiate(SharkFloatParams) \
    template __device__ void MultiplyHelperKaratsubaV1<SharkFloatParams>( \
        const HpSharkFloat<SharkFloatParams> *__restrict__ A, \
        const HpSharkFloat<SharkFloatParams> *__restrict__ B, \
        HpSharkFloat<SharkFloatParams> *__restrict__ Out, \
        uint64_t *__restrict__ carryOuts_phase3, \
        uint64_t *__restrict__ carryOuts_phase6, \
        uint64_t *__restrict__ carryIns, \
        cg::grid_group grid, \
        uint64_t *__restrict__ tempProducts); \
    template __device__ void MultiplyHelperN2<SharkFloatParams>( \
        const HpSharkFloat<SharkFloatParams> *__restrict__ A, \
        const HpSharkFloat<SharkFloatParams> *__restrict__ B, \
        HpSharkFloat<SharkFloatParams> *__restrict__ Out, \
        uint64_t *__restrict__ carryOuts_phase3, \
        uint64_t *__restrict__ carryOuts_phase6, \
        uint64_t *__restrict__ carryIns, \
        cg::grid_group grid, \
        uint64_t *__restrict__ tempProducts);

ExplicitlyInstantiate(Test4x4SharkParams);
ExplicitlyInstantiate(Test4x2SharkParams);
ExplicitlyInstantiate(Test8x1SharkParams);
ExplicitlyInstantiate(Test128x64SharkParams);
