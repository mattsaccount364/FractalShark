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
    cg::grid_group grid,
    uint64_t *__restrict__ tempProducts      // Temporary buffer to store intermediate products
) {
    // Calculate the thread's unique index
    const int threadIdxGlobal = blockIdx.x * SharkFloatParams::GlobalThreadsPerBlock + threadIdx.x;

    const int threadIdxGlobalMin = blockIdx.x * SharkFloatParams::GlobalThreadsPerBlock;
    const int threadIdxGlobalMax = threadIdxGlobalMin + SharkFloatParams::GlobalThreadsPerBlock - 1;

    const int lowDigitIdxMin = threadIdxGlobalMin * 2;
    const int lowDigitIdxMax = threadIdxGlobalMax * 2;

    const int highDigitIdxMin = lowDigitIdxMin + 1;
    const int highDigitIdxMax = lowDigitIdxMax + 1;

    // Each thread handles two digits: low and high
    const int lowDigitIdx = threadIdxGlobal * 2;
    const int highDigitIdx = lowDigitIdx + 1;

    // Ensure indices do not exceed the temporary buffer size
    if (lowDigitIdx >= 2 * SharkFloatParams::GlobalNumUint32) return;

    // Initialize temporary products to zero
    tempProducts[lowDigitIdx] = 0;
    if (highDigitIdx < 2 * SharkFloatParams::GlobalNumUint32) {
        tempProducts[highDigitIdx] = 0;
    }

    uint64_t *carryOuts_phase3 = tempProducts + 2 * SharkFloatParams::GlobalNumUint32;
    uint64_t *carryOuts_phase6 = carryOuts_phase3 + 2 * SharkFloatParams::GlobalNumUint32;
    uint64_t *carryIns = carryOuts_phase6 + 2 * SharkFloatParams::GlobalNumUint32;

    static constexpr int32_t BATCH_SIZE_A = BatchSize;
    static constexpr int32_t BATCH_SIZE_B = BatchSize;

    // Compute k_min and k_max
    const int k_min = 2 * blockIdx.x * SharkFloatParams::GlobalThreadsPerBlock;
    const int k_max = min(2 * (blockIdx.x + 1) * SharkFloatParams::GlobalThreadsPerBlock - 1, 2 * SharkFloatParams::GlobalNumUint32 - 1);

    // Compute j_min_block and j_max_block
    const int j_min_block = max(0, k_min - (SharkFloatParams::GlobalNumUint32 - 1));
    const int j_max_block = min(k_max, SharkFloatParams::GlobalNumUint32 - 1);

    const int a_shared_size_required = j_max_block - j_min_block + 1;

    // Shared memory for A and B with double buffering
    __shared__ __align__(16) uint32_t A_shared[2][BATCH_SIZE_A];
    __shared__ __align__(16) uint32_t B_shared[2][BATCH_SIZE_B];

    const int numBatches_A = (a_shared_size_required + BATCH_SIZE_A - 1) / BATCH_SIZE_A;
    const int numBatches_B = (SharkFloatParams::GlobalNumUint32 + BATCH_SIZE_B - 1) / BATCH_SIZE_B;

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
        const int bIndex_max = min(SharkFloatParams::GlobalNumUint32 - 1, max(bIndex_max_low, bIndex_max_high));

        const int batchB_start = bIndex_min / BATCH_SIZE_B;
        // const int batchB_end = bIndex_max / BATCH_SIZE_B;

        int batchStartB = batchB_start * BATCH_SIZE_B;
        {
            const int elementsToCopyB = min(BATCH_SIZE_B, SharkFloatParams::GlobalNumUint32 - batchStartB);
            cg::memcpy_async(block, &currentBufferB[0], &B->Digits[batchStartB], sizeof(uint32_t) * elementsToCopyB);
        }

        // Loop over batches of B
        for (int batchB = batchB_start; batchB < numBatches_B; ++batchB) {
            //block.sync();

            const int elementsToCopyB = min(BATCH_SIZE_B, SharkFloatParams::GlobalNumUint32 - batchStartB);
            const int batchEndB = batchStartB + elementsToCopyB - 1;

            // Start loading the next batch of B asynchronously if not the last batch
            if (batchB + 1 < numBatches_B) {
                int nextBatchStartB = (batchB + 1) * BATCH_SIZE_B;
                int nextElementsToCopyB = min(BATCH_SIZE_B, SharkFloatParams::GlobalNumUint32 - nextBatchStartB);

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
                    if (highDigitIdx < 2 * SharkFloatParams::GlobalNumUint32 && j >= j_min_high && j <= j_max_high) {
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
    __shared__ uint64_t digitLowShared[SharkFloatParams::GlobalThreadsPerBlock];
    __shared__ uint64_t digitHighShared[SharkFloatParams::GlobalThreadsPerBlock];

    digitLowShared[threadIdx.x] = lowDigitIdxSum;
    digitHighShared[threadIdx.x] = highDigitIdxSum;

    grid.sync();

    if (threadIdx.x == 0) {
        uint64_t carry = 0;

        // Process the digits sequentially
        for (int i = 0; i < SharkFloatParams::GlobalThreadsPerBlock; ++i) {
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
        for (int i = 1; i < SharkFloatParams::GlobalNumBlocks; ++i) {
            carryIns[i] = carryOuts_phase6[i - 1];
        }
    }

    grid.sync(); // Ensure all blocks have computed their carryOuts_phase6

    // Each block uses carryIns[blockIdx.x] as its carry-in for inter-block carry propagation
    uint64_t carry = carryIns[blockIdx.x];

    // Now, perform inter-block carry propagation within each block
    if (threadIdx.x == 0) {
        // Process the digits sequentially
        for (int i = 0; i < SharkFloatParams::GlobalThreadsPerBlock; ++i) {
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
        finalCarry = carryOuts_phase6[SharkFloatParams::GlobalNumBlocks - 1];
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
    __shared__ int sharedHighestIndices[SharkFloatParams::GlobalThreadsPerBlock];
    sharedHighestIndices[threadIdx.x] = localHighestIndex;
    block.sync();

    // Reduction within block to find blockHighestIndex
    for (int offset = SharkFloatParams::GlobalThreadsPerBlock / 2; offset > 0; offset /= 2) {
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
        for (int i = 0; i < SharkFloatParams::GlobalNumBlocks; ++i) {
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
    int shifts = totalResultDigits - SharkFloatParams::GlobalNumUint32;
    if (shifts < 0) {
        shifts = 0;
    }

    // Calculate the required shift to ensure maxHighDigitIdx - shifts <= SharkFloatParams::GlobalNumUint32 - 1
    int requiredShift = highestIndex - (SharkFloatParams::GlobalNumUint32 - 1);
    if (shifts < requiredShift) {
        shifts = requiredShift;
    }

    // Adjust the exponent accordingly
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        Out->Exponent = A->Exponent + B->Exponent + shifts * 32;
        Out->IsNegative = A->IsNegative ^ B->IsNegative;
    }

    // Each thread copies its digits from shared memory to Out->Digits, applying the shift
    if (lowDigitIdx >= shifts && (lowDigitIdx - shifts) < SharkFloatParams::GlobalNumUint32) {
        Out->Digits[lowDigitIdx - shifts] = static_cast<uint32_t>(digitLowShared[threadIdx.x]);
    }
    if (highDigitIdx >= shifts && (highDigitIdx - shifts) < SharkFloatParams::GlobalNumUint32) {
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
        for (int i = SharkFloatParams::GlobalNumUint32 - 1; i >= carryDigits; --i) {
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




template<class SharkFloatParams>
__global__ void MultiplyKernelN2(
    const HpSharkFloat<SharkFloatParams> *A,
    const HpSharkFloat<SharkFloatParams> *B,
    HpSharkFloat<SharkFloatParams> *Out,
    uint64_t *tempProducts) {

    // Initialize cooperative grid group
    cg::grid_group grid = cg::this_grid();

    // Call the MultiplyHelper function
    //MultiplyHelper(A, B, Out, grid, tempProducts);
    MultiplyHelperN2(A, B, Out, grid, tempProducts);
}

template<class SharkFloatParams>
__global__ void MultiplyKernelN2TestLoop(
    HpSharkFloat<SharkFloatParams> *A,
    HpSharkFloat<SharkFloatParams> *B,
    HpSharkFloat<SharkFloatParams> *Out,
    uint64_t *tempProducts) { // Array to store cumulative carries

    // Initialize cooperative grid group
    cg::grid_group grid = cg::this_grid();

    for (int i = 0; i < TestIterCount; ++i) {
        // MultiplyHelper(A, B, Out, grid, tempProducts);
        MultiplyHelperN2(A, B, Out, grid, tempProducts);
    }
}

template<class SharkFloatParams>
void ComputeMultiplyN2Gpu(void *kernelArgs[]) {

    cudaError_t err = cudaLaunchCooperativeKernel(
        (void *)MultiplyKernelN2<SharkFloatParams>,
        dim3(SharkFloatParams::GlobalNumBlocks),
        dim3(SharkFloatParams::GlobalThreadsPerBlock),
        kernelArgs,
        0, // Shared memory size
        0 // Stream
    );

    cudaDeviceSynchronize();

    if (err != cudaSuccess) {
        std::cerr << "CUDA error in MultiplyKernelN2: " << cudaGetErrorString(err) << std::endl;
    }
}

template<class SharkFloatParams>
void ComputeMultiplyN2GpuTestLoop(cudaStream_t &stream, void *kernelArgs[]) {

    cudaError_t err = cudaLaunchCooperativeKernel(
        (void *)MultiplyKernelN2TestLoop<SharkFloatParams>,
        dim3(SharkFloatParams::GlobalNumBlocks),
        dim3(SharkFloatParams::GlobalThreadsPerBlock),
        kernelArgs,
        0, // Shared memory size
        stream // Stream
    );

    cudaDeviceSynchronize();

    if (err != cudaSuccess) {
        std::cerr << "CUDA error in MultiplyKernelKaratsubaTestLoop: " << cudaGetErrorString(err) << std::endl;
    }
}

#define ExplicitlyInstantiate(SharkFloatParams) \
    template void ComputeMultiplyN2Gpu<SharkFloatParams>(void *kernelArgs[]); \
    template void ComputeMultiplyN2GpuTestLoop<SharkFloatParams>(cudaStream_t &stream, void *kernelArgs[]);

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