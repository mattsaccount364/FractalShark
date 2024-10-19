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


// Device function to perform high-precision multiplication
__device__ void MultiplyHelper(
    const HpGpu *A,
    const HpGpu *B,
    HpGpu *Out,
    uint64_t *carryOuts_phase3, // Array to store carry-out from Phase 3
    uint64_t *carryOuts_phase6, // Array to store carry-out from Phase 6
    uint64_t *carryIns,          // Array to store carry-in for each block
    cg::grid_group grid,
    uint64_t *tempProducts      // Temporary buffer to store intermediate products
) {
    // Calculate the thread's unique index
    auto block = cooperative_groups::this_thread_block();
    int threadIdxGlobal = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread handles two digits: low and high
    int lowDigitIdx = threadIdxGlobal * 2;
    int highDigitIdx = lowDigitIdx + 1;

    // Ensure indices do not exceed the temporary buffer size
    if (lowDigitIdx >= 2 * HpGpu::NumUint32) return;
    if (highDigitIdx >= 2 * HpGpu::NumUint32) {
        // Handle boundary condition for odd number of digits if necessary
        // For simplicity, assume NumUint32 is even
    }

    // Shared memory for A and B in double buffering
    constexpr int BATCH_SIZE = 4;
    __shared__ uint32_t A_shared[2][BATCH_SIZE];
    __shared__ uint32_t B_shared[2][BATCH_SIZE + 1];

    // Flags to indicate which buffer is currently being used
    int currentBuffer = 0;
    int nextBuffer = 1;

    // Calculate total number of batches
    int totalBatches = (HpGpu::NumUint32 + BATCH_SIZE - 1) / BATCH_SIZE + 1;

    // Initialize temporary products to zero
    if (lowDigitIdx < 2 * HpGpu::NumUint32) {
        tempProducts[lowDigitIdx] = 0;
    }
    if (highDigitIdx < 2 * HpGpu::NumUint32) {
        tempProducts[highDigitIdx] = 0;
    }

    // Calculate the starting index for this batch
    int nextBatchStart = 1 * BATCH_SIZE;

    // Copy A->Digits
    cg::memcpy_async(block, &A_shared[currentBuffer][0], &A->Digits[0], sizeof(uint32_t) * BATCH_SIZE);
    // Copy B->Digits
    cg::memcpy_async(block, &B_shared[currentBuffer][0], &B->Digits[HpGpu::NumUint32 - nextBatchStart], sizeof(uint32_t) * BATCH_SIZE);
    cg::wait(block);

    // Step 1: Compute partial products without atomics
    // Loop over batches
    int origJ = 0;
    for (int batch = 1; batch < totalBatches; ++batch) {
        block.sync();

        // Phase 1: Asynchronously load a batch of A and B into shared memory
        // Only the first thread in the block initiates the copy.
        // The last iteration does not start a new copy.
        if (threadIdx.x == 0 && batch < totalBatches - 1) {
            nextBatchStart = (batch + 1) * BATCH_SIZE;

            // Copy A->Digits
            cg::memcpy_async(block, &A_shared[nextBuffer][0], &A->Digits[nextBatchStart - BATCH_SIZE], sizeof(uint32_t) * BATCH_SIZE);
            // Copy B->Digits
            cg::memcpy_async(block, &B_shared[nextBuffer][0], &B->Digits[HpGpu::NumUint32 - nextBatchStart], sizeof(uint32_t) * BATCH_SIZE);
        }

        {
            uint64_t sum = 0;
            for (int j = 0; j < BATCH_SIZE; ++j) {
                int origBIndex = lowDigitIdx - origJ;
                int bIndex = lowDigitIdx - (batch - 1) * BATCH_SIZE - j;
                if (origBIndex >= 0 && origBIndex < HpGpu::NumUint32) {
                    sum += static_cast<uint64_t>(A_shared[currentBuffer][j]) * static_cast<uint64_t>(B_shared[currentBuffer][bIndex]);
                }

                origJ++;
            }
            tempProducts[lowDigitIdx] += sum;
        }

        origJ -= BATCH_SIZE;

        {
            uint64_t sum = 0;
            for (int j = 0; j < BATCH_SIZE; ++j) {
                int origBIndex = highDigitIdx - origJ;
                int bIndex = highDigitIdx - (batch - 1) * BATCH_SIZE - j;
                if (origBIndex >= 0 && origBIndex < HpGpu::NumUint32) {
                    sum += static_cast<uint64_t>(A_shared[currentBuffer][j]) * static_cast<uint64_t>(B_shared[currentBuffer][bIndex]);
                }

                origJ++;
            }
            tempProducts[highDigitIdx] += sum;
        }

        // Phase 2: Wait for the previous copy to complete before using the data
        if (batch > 0) {
            // Wait for the previous copy to complete
            cg::wait(block);
        }

        // Phase 4: Switch buffers for double buffering
        currentBuffer = 1 - currentBuffer;
        nextBuffer = 1 - nextBuffer;
    }
    
    grid.sync();

    // Phase 2: Perform initial carry propagation for two digits per thread
    uint64_t lowValue = tempProducts[lowDigitIdx];
    uint32_t digitLow = static_cast<uint32_t>(lowValue & 0xFFFFFFFF);
    uint32_t carryLow = static_cast<uint32_t>(lowValue >> 32);

    uint64_t highValue = tempProducts[highDigitIdx];
    uint32_t digitHigh = static_cast<uint32_t>(highValue & 0xFFFFFFFF);
    uint32_t carryHigh = static_cast<uint32_t>(highValue >> 32);

    // Apply carry from low to high digit
    uint64_t highSum = static_cast<uint64_t>(digitHigh) + carryLow;
    digitHigh = static_cast<uint32_t>(highSum & 0xFFFFFFFF);
    carryHigh += static_cast<uint32_t>(highSum >> 32);

    // Store the digits back into tempProducts
    tempProducts[lowDigitIdx] = digitLow;
    tempProducts[highDigitIdx] = static_cast<uint64_t>(digitHigh);
    __syncthreads();

    // Phase 3: Sequentially propagate carries within the block
    __shared__ uint32_t carryOutsShared[ThreadsPerBlock];
    carryOutsShared[threadIdx.x] = carryHigh; // Each thread's carryOut from high digit
    __syncthreads();

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

