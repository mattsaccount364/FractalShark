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

#include <cooperative_groups.h>
namespace cg = cooperative_groups;

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


//
// static constexpr int32_t ThreadsPerBlock = /* power of 2 */;
// static constexpr int32_t NumBlocks = /* power of 2 */;
// static constexpr int32_t HpGpu::NumUint32 = ThreadsPerBlock * NumBlocks;
// 

// Assuming that HpGpu::NumUint32 can be large and doesn't fit in shared memory
// We'll use the provided global memory buffers for large intermediates

__device__ void MultiplyHelperKaratsuba(
    const HpGpu *__restrict__ A,
    const HpGpu *__restrict__ B,
    HpGpu *__restrict__ Out,
    uint64_t *__restrict__ carryOuts_phase3,
    uint64_t *__restrict__ carryOuts_phase6,
    uint64_t *__restrict__ carryIns,
    cg::grid_group grid,
    uint64_t *__restrict__ tempProducts) {

    cg::thread_block block = cg::this_thread_block();

    const int threadIdxGlobal = blockIdx.x * ThreadsPerBlock + threadIdx.x;

    constexpr int total_threads = ThreadsPerBlock * NumBlocks;
    constexpr int N = HpGpu::NumUint32;         // Total number of digits
    constexpr int n = (N + 1) / 2;              // Half of N

    // Constants for tempProducts offsets
    constexpr int Z0_offset = 0;
    constexpr int Z2_offset = Z0_offset + 4 * N;
    constexpr int Z1_temp_offset = Z2_offset + 4 * N;
    constexpr int Z1_offset = Z1_temp_offset + 4 * N;
    constexpr int Convolution_offset = Z1_offset + 4 * N;
    constexpr int Result_offset = Convolution_offset + 4 * N;

    // Shared memory allocation
    __shared__ uint32_t A_shared[n];
    __shared__ uint32_t B_shared[n];

    // Shared memory arrays
    __shared__ uint64_t per_thread_carry_out[ThreadsPerBlock];
    __shared__ uint64_t per_thread_carry_in[ThreadsPerBlock];

    // Load segments of A and B into shared memory
    for (int i = threadIdx.x; i < n; i += ThreadsPerBlock) {
        A_shared[i] = (i < N) ? A->Digits[i] : 0;  // A0
        B_shared[i] = (i < N) ? B->Digits[i] : 0;  // B0
    }

    // Synchronize before starting convolutions
    __syncthreads();

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
            uint64_t a = A_shared[i];         // A0[i]
            uint64_t b = B_shared[k - i];     // B0[k - i]

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
    __syncthreads();

    // Load A1 and B1 into shared memory
    for (int i = threadIdx.x; i < n; i += ThreadsPerBlock) {
        int index = i + n;
        A_shared[i] = (index < N) ? A->Digits[index] : 0;    // A1
        B_shared[i] = (index < N) ? B->Digits[index] : 0;    // B1
    }

    __syncthreads();

    // ---- Convolution for Z2 = A1 * B1 ----
    for (int k = k_start; k < k_end; ++k) {
        uint64_t sum_low = 0;
        uint64_t sum_high = 0;

        int i_start = max(0, k - (n - 1));
        int i_end = min(k, n - 1);

        for (int i = i_start; i <= i_end; ++i) {
            uint64_t a = A_shared[i];         // A1[i]
            uint64_t b = B_shared[k - i];     // B1[k - i]

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

    // Synchronize before next convolution
    __syncthreads();

    // Compute (A0 + A1) and (B0 + B1) and store in shared memory
    for (int i = threadIdx.x; i < n; i += ThreadsPerBlock) {
        uint64_t A0 = (i < N) ? A->Digits[i] : 0;
        uint64_t A1 = (i + n < N) ? A->Digits[i + n] : 0;
        A_shared[i] = A0 + A1;               // (A0 + A1)

        uint64_t B0 = (i < N) ? B->Digits[i] : 0;
        uint64_t B1 = (i + n < N) ? B->Digits[i + n] : 0;
        B_shared[i] = B0 + B1;               // (B0 + B1)
    }

    __syncthreads();

    // ---- Convolution for Z1_temp = (A0 + A1) * (B0 + B1) ----
    for (int k = k_start; k < k_end; ++k) {
        uint64_t sum_low = 0;
        uint64_t sum_high = 0;

        int i_start = max(0, k - (n - 1));
        int i_end = min(k, n - 1);

        for (int i = i_start; i <= i_end; ++i) {
            uint64_t a = A_shared[i];         // (A0 + A1)[i]
            uint64_t b = B_shared[k - i];     // (B0 + B1)[k - i]

            uint64_t product = a * b;

            // Add product to sum
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
    __syncthreads();

    // ---- Carry Propagation ----

    uint64_t thread_carry = 0;

    for (int idx = idx_start; idx < idx_end; ++idx) {
        int result_idx = Convolution_offset + idx * 2;

        uint64_t sum_low = tempProducts[result_idx];        // Lower 64 bits
        uint64_t sum_high = tempProducts[result_idx + 1];   // Higher 64 bits

        // Add thread_carry to sum_low
        uint64_t new_sum_low = sum_low + thread_carry;
        uint64_t carry_from_low = (new_sum_low < sum_low) ? 1 : 0;

        // Add carry_from_low to sum_high
        uint64_t new_sum_high = sum_high + carry_from_low;

        // Extract digit (lower 32 bits of new_sum_low)
        uint32_t digit = static_cast<uint32_t>(new_sum_low & 0xFFFFFFFFULL);

        // Compute carry for the next digit
        thread_carry = new_sum_high + (new_sum_low >> 32);

        // Overwrite tempProducts with the final digit
        result_idx = Result_offset + idx;
        tempProducts[result_idx] = digit;
    }

    // Store per-thread carry-out
    per_thread_carry_out[threadIdx.x] = thread_carry;

    // Synchronize threads
    __syncthreads();

    // Perform inter-thread carry propagation serially using a single thread
    if (threadIdx.x == 0) {
        uint64_t carry = 0;
        for (int i = 0; i < ThreadsPerBlock; ++i) {
            per_thread_carry_in[i] = carry;
            carry = per_thread_carry_out[i] + carry;
        }
        // Optionally store the final carry-out for the block if needed
        carryOuts_phase6[blockIdx.x] = carry;
    }
    __syncthreads();

    // ---- Second Pass: Adjust Digits with Inter-Thread Carries ----

    thread_carry = per_thread_carry_in[threadIdx.x];

    for (int idx = idx_start; idx < idx_end; ++idx) {
        int result_idx = Result_offset + idx;

        uint32_t digit = tempProducts[result_idx];

        // Add carry to digit
        uint64_t sum = static_cast<uint64_t>(digit) + thread_carry;

        // Update digit and carry
        digit = static_cast<uint32_t>(sum & 0xFFFFFFFFULL);
        thread_carry = sum >> 32;

        // Store updated digit
        tempProducts[result_idx] = digit;
    }

    // Update per-thread carry-out
    per_thread_carry_out[threadIdx.x] = thread_carry;


    // Synchronize threads
    __syncthreads();

    // The last thread in the block stores the final carry
    if (threadIdx.x == ThreadsPerBlock - 1) {
        carryOuts_phase6[blockIdx.x] = per_thread_carry_out[threadIdx.x];
    }

    //grid.sync();
    // TODO block-level carry is busted. Need to fix this.

    // Perform block-level carry propagation using a single thread
    //if (blockIdx.x == 0 && threadIdx.x == 0) {
    //    uint64_t carry = 0;
    //    for (int i = 0; i < gridDim.x; ++i) {
    //        uint64_t block_carry_in = carry;
    //        carry = carryOuts_phase6[i] + carry;
    //        per_block_carry_in[i] = block_carry_in;
    //    }
    //}

    // Synchronize before adjusting exponent and sign
    grid.sync();

    // ---- Handle Any Remaining Final Carry ----

    // Only one thread handles the final carry propagation
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        uint64_t final_carry = carryOuts_phase6[NumBlocks - 1];

        // Initial total_result_digits is 2 * N
        int total_result_digits = 2 * N;

        // Handle the final carry-out from the most significant digit
        if (final_carry > 0) {
            // Append the final carry as a new digit at the end (most significant digit)
            tempProducts[Result_offset + total_result_digits] = static_cast<uint32_t>(final_carry & 0xFFFFFFFFULL);
            total_result_digits += 1;
        }
        
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

__global__ void MultiplyKernelKaratsuba(
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
    //MultiplyHelper(A, B, Out, carryOuts_phase3, carryOuts_phase6, carryIns, grid, tempProducts);
    MultiplyHelperKaratsuba(A, B, Out, carryOuts_phase3, carryOuts_phase6, carryIns, grid, tempProducts);
}

__global__ void MultiplyKernelKaratsubaTestLoop(
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
        // MultiplyHelper(A, B, Out, carryOuts_phase3, carryOuts_phase6, carryIns, grid, tempProducts);
        MultiplyHelperKaratsuba(A, B, Out, carryOuts_phase3, carryOuts_phase6, carryIns, grid, tempProducts);
    }
}


void ComputeMultiplyGpu(void *kernelArgs[]) {

    cudaError_t err = cudaLaunchCooperativeKernel(
        (void *)MultiplyKernelKaratsuba,
        dim3(NumBlocks),
        dim3(ThreadsPerBlock),
        kernelArgs,
        0, // Shared memory size
        0 // Stream
    );

    cudaDeviceSynchronize();

    if (err != cudaSuccess) {
        std::cerr << "CUDA error in MultiplyKernelKaratsuba: " << cudaGetErrorString(err) << std::endl;
    }
}

void ComputeMultiplyGpuTestLoop(void *kernelArgs[]) {

    cudaError_t err = cudaLaunchCooperativeKernel(
        (void *)MultiplyKernelKaratsubaTestLoop,
        dim3(NumBlocks),
        dim3(ThreadsPerBlock),
        kernelArgs,
        0, // Shared memory size
        0 // Stream
    );

    cudaDeviceSynchronize();

    if (err != cudaSuccess) {
        std::cerr << "CUDA error in MultiplyKernelKaratsubaTestLoop: " << cudaGetErrorString(err) << std::endl;
    }
}

