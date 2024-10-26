#include <cuda_runtime.h>

#include "HpGpu.cuh"
#include "BenchmarkTimer.h"
#include "TestTracker.h"
#include "Tests.cuh"

#include <iostream>
#include <vector>
#include <gmp.h>
#include <cstring>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <assert.h>

#include <cooperative_groups.h>


namespace cg = cooperative_groups;

__device__ void BlellochScanWarp1(uint32_t *carryBits, uint32_t *scanResults, int n) {
    const uint32_t laneId = threadIdx.x % warpSize;
    const uint32_t warpMask = __activemask();

    // Each thread loads its carry-out bit
    uint32_t carryContinue = (laneId < n) ? carryBits[laneId] : 1; // Initialize to 1 if out of bounds

    // Compute the inclusive scan with AND operator
    uint32_t cumulativeCarryContinue = carryContinue;

    // Perform the inclusive scan
#pragma unroll
    for (int offset = 1; offset < warpSize; offset *= 2) {
        uint32_t neighborCarry = __shfl_up_sync(warpMask, cumulativeCarryContinue, offset);
        if (laneId >= offset) {
            cumulativeCarryContinue &= neighborCarry;
        }
    }

    // All threads call __shfl_up_sync unconditionally
    uint32_t temp = __shfl_up_sync(warpMask, cumulativeCarryContinue, 1);
    uint32_t carryIn;

    // Determine carry-in based on laneId
    if (laneId > 0) {
        carryIn = temp;
    } else {
        carryIn = 0; // First thread has carry-in zero
    }

    // Write carry-in to scanResults
    if (laneId < n) {
        scanResults[laneId] = carryIn;
    }
}

// input = [0, 1, 0, 0]
// The output[0, 0, 1, 1] from the BlellochScanWarp function is correct for an exclusive prefix sum of your input[0, 1, 0, 0].
// If your application requires the output[0, 1, 1, 1], you should use an inclusive prefix sum.
__device__ void BlellochScanWarp(uint32_t * input, uint32_t * output, int n) {
    uint32_t val = 0;
    int lane = threadIdx.x; // lane index within the warp (since ThreadsPerBlock == warpSize)
    uint32_t warp_mask = __ballot_sync(0xFFFFFFFF, lane < n); // Active threads in the warp

    // Load the input value for each thread
    val = (lane < n) ? input[lane] : 0;

    // Perform inclusive scan using warp-level shuffles
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        uint32_t n = __shfl_up_sync(warp_mask, val, offset);
        if (lane >= offset) {
            val += n;
        }
    }

    // Convert inclusive scan result to exclusive scan result
    uint32_t scan_res = __shfl_up_sync(warp_mask, val, 1);
    if (lane == 0) {
        scan_res = 0;
    }

    // Write the result to the output array
    if (lane < n) {
        output[lane] = scan_res;
    }
}

// Device function to compute carry-in signals by shifting carry-out signals right by one
__device__ void ComputeCarryInWarp(
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
__device__ void ComputeCarryIn(
    const uint32_t *carryOuts, // Input carry-out array
    uint32_t *carryIn,         // Output carry-in array
    int n,                     // Number of digits
    uint32_t *sharedCarryOut   // Statically allocated sharedCarryOut array (size ThreadsPerBlock)
) {
    int tid = threadIdx.x;

    // Step 1: Store carryOuts into shared memory
    if (tid < n) {
        sharedCarryOut[tid] = carryOuts[tid];
    }

    // Ensure all carryOuts are written before proceeding
    __syncthreads();

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
__device__ void ComputeCarryInDecider(
    const uint32_t *carryOuts, // Input carry-out array
    uint32_t *carryIn,         // Output carry-in array
    int n,                     // Number of digits
    uint32_t *sharedCarryOut   // Statically allocated sharedCarryOut array (size ThreadsPerBlock)
) {
    if constexpr (ThreadsPerBlock <= 32) {
        ComputeCarryInWarp(carryOuts, carryIn, n);
    } else {
        ComputeCarryIn(carryOuts, carryIn, n, sharedCarryOut);
    }
}

__device__ uint32_t ShiftRight(uint32_t *digits, int shiftBits, int idx) {
    int shiftWords = shiftBits / 32;
    int shiftBitsMod = shiftBits % 32;
    uint32_t lower = 0, upper = 0;
    int srcIdx = idx + shiftWords;
    const int numDigits = HpGpu::NumUint32;
    if (srcIdx < numDigits) {
        lower = digits[srcIdx];
    }
    if (srcIdx + 1 < numDigits) {
        upper = digits[srcIdx + 1];
    }
    if (shiftBitsMod == 0) {
        return lower;
    } else {
        return (lower >> shiftBitsMod) | (upper << (32 - shiftBitsMod));
    }
}

// Device function to perform addition/subtraction with carry handling using Blelloch scan and static shared memory
__device__ void AddHelper(
    HpGpu *A,
    HpGpu *B,
    HpGpu *Out,
    GlobalAddBlockData *globalData,
    CarryInfo *carryOuts,        // Array to store carry-out for each block
    uint32_t *cumulativeCarries, // Array to store cumulative carries
    cg::grid_group grid,
    int numBlocks) {

    const int numDigits = HpGpu::NumUint32;
    const int tid = threadIdx.x;
    const int blockId = blockIdx.x;

    // Determine the range of digits this block will handle
    const int digitsPerBlock = (numDigits + numBlocks - 1) / numBlocks;
    const int startDigit = blockId * digitsPerBlock;
    const int endDigit = min(startDigit + digitsPerBlock, numDigits);
    const int digitsInBlock = endDigit - startDigit;

    // Static shared memory allocation based on predefined maximums
    __shared__ uint32_t alignedA_static[ThreadsPerBlock];
    __shared__ uint32_t alignedB_static[ThreadsPerBlock];
    __shared__ uint32_t tempSum_static[ThreadsPerBlock];
    __shared__ uint32_t carryBits_static[ThreadsPerBlock];
    __shared__ uint32_t scanResults_static[ThreadsPerBlock];
    __shared__ uint32_t sharedCarryOut[ThreadsPerBlock];        // Shared carry-out for ComputeCarryIn

    // Additional shared variables
    __shared__ bool isAddition;
    __shared__ bool AIsBiggerExponent;
    __shared__ bool AIsBiggerMagnitude;
    __shared__ int32_t outExponent;
    __shared__ bool resultIsZero;

    // Phase 1: Exponent Alignment
    if (tid == 0) {
        int32_t expDiff = A->Exponent - B->Exponent;
        AIsBiggerExponent = (expDiff >= 0);
        outExponent = AIsBiggerExponent ? A->Exponent : B->Exponent;
        isAddition = (A->IsNegative == B->IsNegative);
        resultIsZero = false; // Initialize resultIsZero
        AIsBiggerMagnitude = true; // Default assumption
    }
    __syncthreads();  // Synchronize all threads in the grid

    // Initialize aligned digits to zero
    for (int i = tid; i < digitsInBlock; i += ThreadsPerBlock) {
        alignedA_static[i] = 0;
        alignedB_static[i] = 0;
    }
    __syncthreads();  // Synchronize within the block after initialization

    // Perform shifting based on exponent difference
    int shiftBits = abs(A->Exponent - B->Exponent);

    for (int i = tid; i < digitsInBlock; i += ThreadsPerBlock) {
        int globalIdx = startDigit + i;
        if (globalIdx < numDigits) {
            // Shift the number with the smaller exponent
            if (AIsBiggerExponent) {
                // Shift B right
                alignedA_static[i] = A->Digits[globalIdx];
                alignedB_static[i] = ShiftRight(B->Digits, shiftBits, globalIdx);
            } else {
                // Shift A right
                alignedA_static[i] = ShiftRight(A->Digits, shiftBits, globalIdx);
                alignedB_static[i] = B->Digits[globalIdx];
            }
        } else {
            // Zero-fill for indices beyond numDigits
            alignedA_static[i] = 0;
            alignedB_static[i] = 0;
        }
    }
    __syncthreads();  // Synchronize within the block after alignment

    // Phase 2: Compare Magnitudes if necessary
    if (!isAddition) {
        if (tid == 0) { // Let only thread 0 perform the magnitude comparison
            bool magnitudeDetermined = false;
            for (int idx = numDigits - 1; idx >= 0 && !magnitudeDetermined; --idx) {
                uint32_t a_digit = 0, b_digit = 0;

                if (AIsBiggerExponent) {
                    // Shift B right
                    a_digit = A->Digits[idx];
                    b_digit = ShiftRight(B->Digits, shiftBits, idx);
                } else {
                    // Shift A right
                    a_digit = ShiftRight(A->Digits, shiftBits, idx);
                    b_digit = B->Digits[idx];
                }

                if (a_digit > b_digit) {
                    AIsBiggerMagnitude = true;
                    magnitudeDetermined = true;
                } else if (a_digit < b_digit) {
                    AIsBiggerMagnitude = false;
                    magnitudeDetermined = true;
                }
                // If equal, continue to next digit
            }

            // If all digits are equal, set resultIsZero
            if (!magnitudeDetermined) {
                resultIsZero = true;
            }
        }
    }
    __syncthreads();  // Synchronize to ensure magnitude comparison is complete

    // Phase 3: Perform Addition or Subtraction within the block
    if (!resultIsZero) {
        for (int i = tid; i < digitsInBlock; i += ThreadsPerBlock) {
            if (i >= digitsInBlock) continue;
            if (isAddition) {
                // Perform addition
                uint64_t fullSum = (uint64_t)alignedA_static[i] + (uint64_t)alignedB_static[i];
                tempSum_static[i] = (uint32_t)(fullSum & 0xFFFFFFFF);
                carryBits_static[i] = (uint32_t)(fullSum >> 32);
            } else {
                // Perform subtraction: larger magnitude minus smaller magnitude
                uint32_t operandLarge = AIsBiggerMagnitude ? alignedA_static[i] : alignedB_static[i];
                uint32_t operandSmall = AIsBiggerMagnitude ? alignedB_static[i] : alignedA_static[i];

                if (operandLarge >= operandSmall) {
                    uint64_t fullDiff = (uint64_t)operandLarge - (uint64_t)operandSmall;
                    tempSum_static[i] = (uint32_t)(fullDiff & 0xFFFFFFFF);
                    carryBits_static[i] = 0;
                } else {
                    // Borrow occurs
                    uint64_t fullDiff = ((uint64_t)1 << 32) + (uint64_t)operandLarge - (uint64_t)operandSmall;
                    tempSum_static[i] = (uint32_t)(fullDiff & 0xFFFFFFFF);
                    carryBits_static[i] = 1; // Indicate borrow
                }
            }
        }
    }
    __syncthreads();  // Synchronize after addition/subtraction

    // Phase 4: Parallel Blelloch Scan on carryBits to compute carry-ins
    uint32_t initialCarryOutLastDigit = 0;
    if (!resultIsZero) {
        // Save the initial carry-out from the last digit before overwriting
        initialCarryOutLastDigit = carryBits_static[digitsInBlock - 1];

        // Perform Blelloch scan on carryBits_static
        ComputeCarryInDecider(carryBits_static, scanResults_static, digitsInBlock, sharedCarryOut);

        // After scan, scanResults_static contains the carry-in for each digit
        // Apply the carry-ins to tempSum_static to get the final output digits
        for (int i = tid; i < digitsInBlock; i += ThreadsPerBlock) {
            if (i >= digitsInBlock) continue;
            if (isAddition) {
                // Add the carry-in to the sum
                uint64_t finalSum = (uint64_t)tempSum_static[i] + (uint64_t)scanResults_static[i];
                Out->Digits[startDigit + i] = (uint32_t)(finalSum & 0xFFFFFFFF);
                // Record carry-out for the next phase
                carryBits_static[i] = (uint32_t)(finalSum >> 32);
            } else {
                // Subtraction already handled borrow in carryBits_static
                // Apply borrow-in from scanResults_static
                uint32_t borrowIn = scanResults_static[i];
                uint64_t finalDiff = (uint64_t)tempSum_static[i] - (uint64_t)borrowIn;
                Out->Digits[startDigit + i] = (uint32_t)(finalDiff & 0xFFFFFFFF);
                // If borrow occurs, set carryBits_static[i] to 1
                carryBits_static[i] = (finalDiff >> 63) & 1;
            }
        }
    }
    __syncthreads();  // Synchronize after applying carry-ins

    // Phase 5 & 6: Record carry-outs and compute cumulative carries
    if (tid == 0) {
        if (!resultIsZero) {
            carryOuts[blockId].carryOut = initialCarryOutLastDigit + carryBits_static[digitsInBlock - 1];
        } else {
            carryOuts[blockId].carryOut = 0;
        }

        // Only block 0 computes cumulativeCarries after all carryOuts are recorded
        if (blockId == 0) {
            cumulativeCarries[0] = 0; // Initial carry-in is zero
            for (int i = 1; i <= numBlocks; ++i) {
                cumulativeCarries[i] = carryOuts[i - 1].carryOut;
            }
        }
    }
    grid.sync();  // Single synchronization after both operations

    // Phase 7: Apply Cumulative Carries to Each Block's Output
    if (!resultIsZero) {
        uint32_t blockCarryIn = cumulativeCarries[blockId];
        if (digitsInBlock > 0) {
            if (tid == 0) {
                if (isAddition) {
                    uint64_t finalSum = (uint64_t)Out->Digits[startDigit] + (uint64_t)blockCarryIn;
                    Out->Digits[startDigit] = (uint32_t)(finalSum & 0xFFFFFFFF);
                    // Update carryBits_static[0] with any new carry-out from this addition
                    carryBits_static[0] = (uint32_t)(finalSum >> 32);
                } else {
                    uint64_t finalDiff = (uint64_t)Out->Digits[startDigit] - (uint64_t)blockCarryIn;
                    Out->Digits[startDigit] = (uint32_t)(finalDiff & 0xFFFFFFFF);
                    // Update carryBits_static[0] with any new borrow-out
                    carryBits_static[0] = (finalDiff >> 63) & 1;
                }
            }
            // Do not modify carryBits_static[tid] for tid != 0
        }
    }

    grid.sync();  // Synchronize after applying cumulative carry-ins

    // Phase 7.1: Propagate New Carry-Out Within the Block if Necessary
    if (!resultIsZero && digitsInBlock > 1) {
        // Perform Blelloch scan on carryBits_static
        ComputeCarryInDecider(carryBits_static, scanResults_static, digitsInBlock, sharedCarryOut);

        // Apply the new carry-ins to the output digits
        for (int i = tid; i < digitsInBlock; i += ThreadsPerBlock) {
            if (i > 0) { // Skip the first digit
                if (isAddition) {
                    uint64_t finalSum = (uint64_t)Out->Digits[startDigit + i] + (uint64_t)scanResults_static[i];
                    Out->Digits[startDigit + i] = (uint32_t)(finalSum & 0xFFFFFFFF);
                    // Update carryBits_static[i] with any new carry-out
                    carryBits_static[i] = (uint32_t)(finalSum >> 32);
                } else {
                    uint64_t finalDiff = (uint64_t)Out->Digits[startDigit + i] - (uint64_t)scanResults_static[i];
                    Out->Digits[startDigit + i] = (uint32_t)(finalDiff & 0xFFFFFFFF);
                    // Update carryBits_static[i] with any new borrow-out
                    carryBits_static[i] = (finalDiff >> 63) & 1;
                }
            }
        }
    }
    __syncthreads();  // Synchronize after propagating new carry-outs

    // Corrected shifting code in Phase 8
    if (blockId == numBlocks - 1 && tid == 0) {
        if (isAddition) {
            // Retrieve the total carry-out from all blocks
            uint32_t finalCarryOut = cumulativeCarries[numBlocks];

            if (finalCarryOut > 0) {
                // Increment the exponent to account for the carry-out
                outExponent += 1;

                // Shift digits right by one bit, starting from the most significant digit
                uint32_t carry = finalCarryOut; // Initialize carry with the final carry-out
                for (int i = numDigits - 1; i >= 0; --i) {
                    uint32_t newCarry = Out->Digits[i] & 1; // Save LSB before shift
                    Out->Digits[i] = (Out->Digits[i] >> 1) | (carry << 31); // Shift right and insert carry
                    carry = newCarry; // Update carry for next digit
                }
                // No need to set the MSB separately
            }
        } else {
            // Subtraction: Handle borrow-out if necessary
            // (Similar logic applies for subtraction if needed)
        }

        Out->Exponent = outExponent;
        if (isAddition) {
            Out->IsNegative = A->IsNegative;
        } else {
            // The sign is determined by the operand with the larger magnitude
            Out->IsNegative = AIsBiggerMagnitude ? A->IsNegative : B->IsNegative;
        }
    }

    //grid.sync();  // Synchronize after adjusting exponent and digits
}

__global__ void AddKernel(
    HpGpu *A,
    HpGpu *B,
    HpGpu *Out,
    GlobalAddBlockData *globalBlockData,
    CarryInfo *carryOuts,        // Array to store carry-out for each block
    uint32_t *cumulativeCarries) { // Array to store cumulative carries

    // Initialize cooperative grid group
    cg::grid_group grid = cg::this_grid();

    // Total number of blocks launched
    int numBlocks = gridDim.x;

    // Call the AddHelper function
    AddHelper(A, B, Out, globalBlockData, carryOuts, cumulativeCarries, grid, numBlocks);
}



__global__ void AddKernelTestLoop(
    HpGpu *A,
    HpGpu *B,
    HpGpu *Out,
    GlobalAddBlockData *globalBlockData,
    CarryInfo *carryOuts,        // Array to store carry-out for each block
    uint32_t *cumulativeCarries) { // Array to store cumulative carries

    // Initialize cooperative grid group
    cg::grid_group grid = cg::this_grid();

    // Total number of blocks launched
    int numBlocks = gridDim.x;

    for (int i = 0; i < NUM_ITER; ++i) {
        AddHelper(A, B, Out, globalBlockData, carryOuts, cumulativeCarries, grid, numBlocks);
    }
}

//__global__ void multiply_high_precision_kernel(
//    HpGpu *A,
//    HpGpu *B,
//    HpGpu *Out) {
//    const int idx = threadIdx.x;
//    __shared__ uint32_t tempProduct[HpGpu::NumUint32 * 2];
//
//    // Initialize tempProduct to zero
//    if (idx < HpGpu::NumUint32 * 2) {
//        tempProduct[idx] = 0;
//    }
//
//    __syncthreads();
//
//    // Each thread computes partial products
//    for (int i = 0; i < HpGpu::NumUint32; i++) {
//        if (idx < HpGpu::NumUint32) {
//            uint64_t product = (uint64_t)A->Digits[idx] * (uint64_t)B->Digits[i];
//            int pos = idx + i;
//
//            // Atomic addition to handle concurrent writes
//            atomicAdd(&tempProduct[pos], (uint32_t)(product & 0xFFFFFFFF));
//            atomicAdd(&tempProduct[pos + 1], (uint32_t)(product >> 32));
//        }
//    }
//
//    __syncthreads();
//
//    // Perform carry propagation
//    uint32_t carry = 0;
//    if (idx < HpGpu::NumUint32 * 2) {
//        uint64_t sum = (uint64_t)tempProduct[idx] + carry;
//        tempProduct[idx] = (uint32_t)(sum & 0xFFFFFFFF);
//        carry = (uint32_t)(sum >> 32);
//    }
//
//    __syncthreads();
//
//    // Write the result to Out
//    if (idx < HpGpu::NumUint32) {
//        Out->Digits[idx] = tempProduct[idx];
//    }
//
//    // Adjust exponent and sign
//    if (idx == 0) {
//        Out->Exponent = A->Exponent + B->Exponent;
//        Out->IsNegative = A->IsNegative ^ B->IsNegative;
//    }
//}

void ComputeAddGpu(void *kernelArgs[]) {

    cudaError_t err = cudaLaunchCooperativeKernel(
        (void *)AddKernel,
        dim3(NumBlocks),
        dim3(ThreadsPerBlock),
        kernelArgs,
        0, // Shared memory size
        0 // Stream
    );

    cudaDeviceSynchronize();
}

void ComputeAddGpuTestLoop(void *kernelArgs[]) {

    cudaError_t err = cudaLaunchCooperativeKernel(
        (void *)AddKernelTestLoop,
        dim3(NumBlocks),
        dim3(ThreadsPerBlock),
        kernelArgs,
        0, // Shared memory size
        0 // Stream
    );

    cudaDeviceSynchronize();
}
