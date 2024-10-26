#include "Multiply.cuh"

#include <cuda_runtime.h>

#include "HpGpu.cuh"
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


// Structs for carry handling (similar to addition)
struct GlobalMulBlockData {
    // Define any necessary global data for multiplication
    // Placeholder: can be expanded as needed
};

struct PartialSum {
    uint64_t sum;
};

#include <cooperative_groups.h>
namespace cg = cooperative_groups;



//
// static constexpr int32_t ThreadsPerBlock = /* power of 2 */;
// static constexpr int32_t NumBlocks = /* power of 2 */;
// static constexpr int32_t HpGpu::NumUint32 = ThreadsPerBlock * NumBlocks;
// 

// Device function to perform high-precision multiplication
__device__ void MultiplyHelper(
    const HpGpu * __restrict__ A,
    const HpGpu * __restrict__ B,
    HpGpu *__restrict__ Out,
    uint64_t * __restrict__ carryOuts_phase3, // Array to store carry-out from Phase 3
    uint64_t * __restrict__ carryOuts_phase6, // Array to store carry-out from Phase 6
    uint64_t * __restrict__ carryIns,          // Array to store carry-in for each block
    cg::grid_group grid,
    uint64_t * __restrict__ tempProducts      // Temporary buffer to store intermediate products
) {
    // Calculate the thread's unique index
    int threadIdxGlobal = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread handles two digits: low and high
    int lowDigitIdx = threadIdxGlobal * 2;
    int highDigitIdx = lowDigitIdx + 1;

    // Ensure indices do not exceed the temporary buffer size
    if (lowDigitIdx >= 2 * HpGpu::NumUint32) return;

    // Initialize temporary products to zero
    tempProducts[lowDigitIdx] = 0;
    if (highDigitIdx < 2 * HpGpu::NumUint32) {
        tempProducts[highDigitIdx] = 0;
    }
    __syncthreads();

    static constexpr int32_t BATCH_SIZE_A = ThreadsPerBlock;
    static constexpr int32_t BATCH_SIZE_B = ThreadsPerBlock;

    // Compute k_min and k_max
    int k_min = 2 * blockIdx.x * blockDim.x;
    int k_max = min(2 * (blockIdx.x + 1) * blockDim.x - 1, 2 * HpGpu::NumUint32 - 1);

    // Compute j_min_block and j_max_block
    int j_min_block = max(0, k_min - (HpGpu::NumUint32 - 1));
    int j_max_block = min(k_max, HpGpu::NumUint32 - 1);

    int a_shared_size_required = j_max_block - j_min_block + 1;

    // Shared memory for A and B with double buffering
    __shared__ __align__(16) uint32_t A_shared[2][BATCH_SIZE_A];
    __shared__ __align__(16) uint32_t B_shared[2][BATCH_SIZE_B];

    int numBatches_A = (a_shared_size_required + BATCH_SIZE_A - 1) / BATCH_SIZE_A;
    int numBatches_B = (HpGpu::NumUint32 + BATCH_SIZE_B - 1) / BATCH_SIZE_B;

    uint32_t * __restrict__ tempBufferA = nullptr;
    uint32_t * __restrict__ currentBufferA = A_shared[0];
    uint32_t * __restrict__ nextBufferA = A_shared[1];

    uint32_t * __restrict__ tempBufferB = nullptr;
    uint32_t * __restrict__ currentBufferB = B_shared[0];
    uint32_t * __restrict__ nextBufferB = B_shared[1];

    cg::thread_block block = cg::this_thread_block();

    // Start loading the first batch of A asynchronously
    int batchA = 0;
    int batchStartA = j_min_block + batchA * BATCH_SIZE_A;
    int elementsToCopyA = min(BATCH_SIZE_A, a_shared_size_required - batchA * BATCH_SIZE_A);

    cg::memcpy_async(block, &currentBufferA[0], &A->Digits[batchStartA], sizeof(uint32_t) * elementsToCopyA);

    // Wait for the first batch of A to be loaded
    cg::wait(block);

    uint64_t lowDigitIdxSum = 0;
    uint64_t highDigitIdxSum = 0;

    bool waitForFirstBatch;

    // Loop over batches of A
    for (batchA = 0; batchA < numBatches_A; ++batchA) {
        // Start loading the first batch of B asynchronously
        int batchB = 0;
        int batchStartB = batchB * BATCH_SIZE_B;
        int elementsToCopyB = min(BATCH_SIZE_B, HpGpu::NumUint32 - batchStartB);

        cg::memcpy_async(block, &currentBufferB[0], &B->Digits[batchStartB], sizeof(uint32_t) * elementsToCopyB);

        block.sync();

        int batchStartA = j_min_block + batchA * BATCH_SIZE_A;
        int batchEndA = batchStartA + elementsToCopyA - 1;

        waitForFirstBatch = false;

        // Start loading the next batch of A asynchronously if not the last batch
        if (batchA + 1 < numBatches_A) {
            int nextBatchStartA = j_min_block + (batchA + 1) * BATCH_SIZE_A;
            int nextElementsToCopyA = min(BATCH_SIZE_A, a_shared_size_required - (batchA + 1) * BATCH_SIZE_A);

            cg::memcpy_async(block, &nextBufferA[0], &A->Digits[nextBatchStartA], sizeof(uint32_t) * nextElementsToCopyA);

            // Wait for the first batch of B to be loaded
            cg::wait_prior<1>(block);
        } else {
            // Wait for the first batch of B to be loaded
            waitForFirstBatch = true;
        }

        // Loop over batches of B
        for (batchB = 0; batchB < numBatches_B; ++batchB) {

            // Start loading the next batch of B asynchronously if not the last batch
            if (batchB + 1 < numBatches_B) {
                int nextBatchStartB = (batchB + 1) * BATCH_SIZE_B;
                int nextElementsToCopyB = min(BATCH_SIZE_B, HpGpu::NumUint32 - nextBatchStartB);

                cg::memcpy_async(block, &nextBufferB[0], &B->Digits[nextBatchStartB], sizeof(uint32_t) * nextElementsToCopyB);
            }

            if (waitForFirstBatch) {
                cg::wait_prior<1>(block);
                waitForFirstBatch = false;
            }

            //block.sync();

            int batchStartB = batchB * BATCH_SIZE_B;
            int batchEndB = batchStartB + elementsToCopyB - 1;

            // Compute partial products for lowDigitIdx
            {
                uint64_t sumLow = 0;

                // Calculate the valid range of j for lowDigitIdx
                int j_min_low = max(batchStartA, max(j_min_block, lowDigitIdx - batchEndB));
                int j_max_low = min(batchEndA, min(j_max_block, lowDigitIdx - batchStartB));

                for (int j = j_min_low; j <= j_max_low; ++j) {
                    int aSharedIndex = j - batchStartA;
                    int bIndex = lowDigitIdx - j;
                    int bSharedIndex = bIndex - batchStartB;

                    uint32_t aValue = currentBufferA[aSharedIndex];
                    uint32_t bValue = currentBufferB[bSharedIndex];

                    sumLow += static_cast<uint64_t>(aValue) * static_cast<uint64_t>(bValue);
                }
                lowDigitIdxSum += sumLow;
            }

            // Compute partial products for highDigitIdx
            {
                uint64_t sumHigh = 0;

                // Calculate the valid range of j for highDigitIdx
                int j_min_high = max(batchStartA, max(j_min_block, highDigitIdx - batchEndB));
                int j_max_high = min(batchEndA, min(j_max_block, highDigitIdx - batchStartB));

                for (int j = j_min_high; j <= j_max_high; ++j) {
                    int aSharedIndex = j - batchStartA;
                    int bIndex = highDigitIdx - j;
                    int bSharedIndex = bIndex - batchStartB;

                    uint32_t aValue = currentBufferA[aSharedIndex];
                    uint32_t bValue = currentBufferB[bSharedIndex];

                    sumHigh += static_cast<uint64_t>(aValue) * static_cast<uint64_t>(bValue);
                }
                highDigitIdxSum += sumHigh;
            }

            // Switch buffers for double buffering of B
            tempBufferB = currentBufferB;
            currentBufferB = nextBufferB;
            nextBufferB = tempBufferB;

            // Wait for the next batch of B to be loaded
            if (batchB + 1 < numBatches_B) {
                cg::wait(block);
            }
            block.sync();
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

    // Phase 2: Perform initial carry propagation for two digits per thread
    uint64_t lowValue = lowDigitIdxSum;
    uint32_t digitLow = static_cast<uint32_t>(lowValue & 0xFFFFFFFF);
    uint32_t carryLow = static_cast<uint32_t>(lowValue >> 32);

    uint64_t highValue = highDigitIdxSum;
    uint32_t digitHigh = static_cast<uint32_t>(highValue & 0xFFFFFFFF);
    uint32_t carryHigh = static_cast<uint32_t>(highValue >> 32);

    // Apply carry from low to high digit
    uint64_t highSum = static_cast<uint64_t>(digitHigh) + carryLow;
    digitHigh = static_cast<uint32_t>(highSum & 0xFFFFFFFF);
    carryHigh += static_cast<uint32_t>(highSum >> 32);

    // Store the digits back into tempProducts
    tempProducts[lowDigitIdx] = digitLow;
    tempProducts[highDigitIdx] = static_cast<uint64_t>(digitHigh);

    // Phase 3: Sequentially propagate carries within the block
    __shared__ uint32_t carryOutsShared[ThreadsPerBlock];
    carryOutsShared[threadIdx.x] = carryHigh; // Each thread's carryOut from high digit
    grid.sync();

    // Sequential carry propagation within the block
    for (int i = 1; i < ThreadsPerBlock; ++i) {
        if (threadIdx.x == i) {
            uint32_t carryIn = carryOutsShared[i - 1];
            carryOutsShared[i - 1] = 0; // Reset carryOut for this thread
            if (carryIn) {
                // Add carryIn to this thread's low digit
                uint64_t currentLow = tempProducts[lowDigitIdx];
                uint64_t newLowSum = currentLow + carryIn;
                tempProducts[lowDigitIdx] = newLowSum;
                uint32_t newCarry = static_cast<uint32_t>(newLowSum >> 32);

                // Add carryIn to this thread's high digit
                uint64_t currentHigh = tempProducts[highDigitIdx];
                uint64_t newHighSum = currentHigh + newCarry;
                tempProducts[highDigitIdx] = newHighSum;
                newCarry = static_cast<uint32_t>(newHighSum >> 32);
                carryOutsShared[i] += newCarry;
            }
        }
        __syncthreads();
    }

    // After propagation, the last thread's carryOut is the block's carryOut
    if (threadIdx.x == ThreadsPerBlock - 1) {
        carryOuts_phase3[blockIdx.x] = carryOutsShared[ThreadsPerBlock - 1];
    }
    grid.sync(); // Ensure all blocks have computed carryIns

    // Phase 4: Compute carry-ins using prefix sum on carryOuts_phase3
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        carryIns[0] = 0; // First block has no carry-in
        for (int i = 1; i < NumBlocks; ++i) {
            carryIns[i] = carryIns[i - 1] + carryOuts_phase3[i - 1];
        }
    }
    __syncthreads();
    grid.sync(); // Ensure all blocks have computed carryIns


    // Phase 5: Apply carry-ins to each block's output digits
    if (lowDigitIdx < HpGpu::NumUint32) {
        uint64_t sumLow = static_cast<uint64_t>(tempProducts[lowDigitIdx] & 0xFFFFFFFF) + carryOuts_phase3[blockIdx.x];
        Out->Digits[lowDigitIdx] = static_cast<uint32_t>(sumLow & 0xFFFFFFFF);
        uint32_t newCarryLow = static_cast<uint32_t>(sumLow >> 32);
        carryOutsShared[threadIdx.x * 2] = newCarryLow;
    } else {
        carryOutsShared[threadIdx.x * 2] = 0;
    }

    if (highDigitIdx < HpGpu::NumUint32) {
        uint64_t sumHigh = static_cast<uint64_t>(tempProducts[highDigitIdx] & 0xFFFFFFFF) + carryOutsShared[threadIdx.x * 2];
        Out->Digits[highDigitIdx] = static_cast<uint32_t>(sumHigh & 0xFFFFFFFF);
        uint32_t newCarryHigh = static_cast<uint32_t>(sumHigh >> 32);
        carryOutsShared[threadIdx.x * 2 + 1] = newCarryHigh;
    } else {
        carryOutsShared[threadIdx.x * 2 + 1] = 0;
    }
    __syncthreads();

    // Phase 6: Record any new carry-outs generated by carry-ins
    if (threadIdx.x == 0) {
        uint64_t blockCarryOut = 0;
        for (int i = 0; i < ThreadsPerBlock * 2; ++i) {
            blockCarryOut += carryOutsShared[i];
        }
        carryOuts_phase6[blockIdx.x] = blockCarryOut;
    }
    __syncthreads();

    // FIX ME: carryOuts_phase6 is unused.

    // Synchronize all blocks before handling final carry-outs
    grid.sync();

    // Corrected Phase 7: Handle significant digits beyond the mantissa
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        // Step 1: Find the highest non-zero index in tempProducts
        int highestNonZeroIndex = -1;
        for (int i = 2 * HpGpu::NumUint32 - 1; i >= 0; --i) {
            if (tempProducts[i] != 0) {
                highestNonZeroIndex = i;
                break;
            }
        }

        // Step 2: Calculate the total number of significant digits
        int totalResultDigits = highestNonZeroIndex + 1;

        // Step 3: Determine the number of shifts needed
        int shifts = totalResultDigits - HpGpu::NumUint32;
        if (shifts < 0) {
            shifts = 0; // No shift needed if result fits within the mantissa
        }

        // Step 4: Shift the mantissa to the right by 'shifts' digits
        for (int i = 0; i < HpGpu::NumUint32; ++i) {
            int srcIndex = i + shifts;
            if (srcIndex < 2 * HpGpu::NumUint32) {
                Out->Digits[i] = static_cast<uint32_t>(tempProducts[srcIndex] & 0xFFFFFFFF);
            } else {
                Out->Digits[i] = 0; // Pad with zeros if beyond tempProducts
            }
        }

        // Step 5: Adjust the exponent accordingly
        Out->Exponent += shifts * 32;

        // Step 6: Perform carry propagation within the mantissa
        uint64_t carry = 0;
        for (int i = 0; i < HpGpu::NumUint32; ++i) {
            uint64_t sum = static_cast<uint64_t>(Out->Digits[i]) + carry;
            Out->Digits[i] = static_cast<uint32_t>(sum & 0xFFFFFFFF);
            carry = sum >> 32;
        }

        // Step 7: Handle any remaining carry
        if (carry > 0) {
            // Shift the mantissa to the right by one more digit
            for (int i = 0; i < HpGpu::NumUint32 - 1; ++i) {
                Out->Digits[i] = Out->Digits[i + 1];
            }
            Out->Digits[HpGpu::NumUint32 - 1] = static_cast<uint32_t>(carry & 0xFFFFFFFF);
            // Adjust the exponent
            Out->Exponent += 32;
        }

        // Step 8: Set the sign and adjust the exponent
        Out->IsNegative = A->IsNegative != B->IsNegative;
        Out->Exponent += A->Exponent + B->Exponent;
    }
    __syncthreads();

    //// Phase 8: Initialize result properties (only block 0's thread 0 does this)
    //if (blockIdx.x == 0 && threadIdx.x == 0) {
    //    // Determine the sign of the result
    //    Out->IsNegative = A->IsNegative != B->IsNegative;
    //    // Calculate the initial exponent of the result
    //    Out->Exponent = A->Exponent + B->Exponent;
    //    // Note: Any additional exponent adjustments have been handled in Phase 7
    //}
}

__global__ void MultiplyKernel(
    const HpGpu *A,
    const HpGpu *B,
    HpGpu *Out,
    uint64_t *carryOuts_phase3,
    uint64_t *carryOuts_phase6,
    uint64_t *carryIns,
    uint64_t *tempProducts) {

    // Initialize cooperative grid group
    cg::grid_group grid = cg::this_grid();

    // Call the MultiplyHelper function
    MultiplyHelper(A, B, Out, carryOuts_phase3, carryOuts_phase6, carryIns, grid, tempProducts);
}

__global__ void MultiplyKernelTestLoop(
    HpGpu *A,
    HpGpu *B,
    HpGpu *Out,
    uint64_t *carryOuts_phase3,
    uint64_t *carryOuts_phase6,
    uint64_t *carryIns,
    uint64_t *tempProducts) { // Array to store cumulative carries

    // Initialize cooperative grid group
    cg::grid_group grid = cg::this_grid();

    for (int i = 0; i < NUM_ITER; ++i) {
        MultiplyHelper(A, B, Out, carryOuts_phase3, carryOuts_phase6, carryIns, grid, tempProducts);
    }
}


void ComputeMultiplyGpu(void *kernelArgs[]) {

    cudaError_t err = cudaLaunchCooperativeKernel(
        (void *)MultiplyKernel,
        dim3(NumBlocks),
        dim3(ThreadsPerBlock),
        kernelArgs,
        0, // Shared memory size
        0 // Stream
    );

    cudaDeviceSynchronize();
}

void ComputeMultiplyGpuTestLoop(void *kernelArgs[]) {

    cudaError_t err = cudaLaunchCooperativeKernel(
        (void *)MultiplyKernelTestLoop,
        dim3(NumBlocks),
        dim3(ThreadsPerBlock),
        kernelArgs,
        0, // Shared memory size
        0 // Stream
    );

    cudaDeviceSynchronize();
}

