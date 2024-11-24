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

__device__ void multiply_uint64(
    uint64_t a, uint64_t b,
    uint64_t &low, uint64_t &high) {
    // Split the inputs into 32-bit halves
    uint64_t a_low = (uint32_t)(a & 0xFFFFFFFFULL);
    uint64_t a_high = a >> 32;
    uint64_t b_low = (uint32_t)(b & 0xFFFFFFFFULL);
    uint64_t b_high = b >> 32;

    // Compute partial products
    uint64_t p0 = a_low * b_low;
    uint64_t p1 = a_low * b_high;
    uint64_t p2 = a_high * b_low;
    uint64_t p3 = a_high * b_high;

    // Compute the lower 64 bits of the product
    uint64_t temp = (p1 + p2);
    uint64_t carry = (temp < p1) ? 1 : 0;  // Check for overflow in temp

    // Combine the partial products
    uint64_t low_part = p0 + (temp << 32);
    if (low_part < p0) {
        carry += 1;  // Carry from lower 64 bits
    }

    // Compute the higher 64 bits of the product
    uint64_t high_part = p3 + (temp >> 32) + carry;

    // Assign the results
    low = low_part;
    high = high_part;
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
    __shared__ uint64_t A_shared[n];
    __shared__ uint64_t B_shared[n];

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
    __syncthreads();

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

    // Constants and offsets
    constexpr int MaxPasses = 10; // Maximum number of carry propagation passes

    // Initialize variables
    int pass = 0;

    // Global memory for block carry-outs
    // Allocate space for gridDim.x block carry-outs after total_result_digits in carryOuts_phase6
    uint64_t *block_carry_outs = tempProducts + Result_offset + total_result_digits;
    constexpr auto digits_per_block = ThreadsPerBlock * 2;
    auto block_start_idx = blockIdx.x * digits_per_block;
    auto block_end_idx = min(block_start_idx + digits_per_block, total_result_digits);

    // First Pass: Process convolution results to compute initial digits and local carries
    {
        // Calculate the number of digits per thread
        int digits_per_thread = (digits_per_block + blockDim.x - 1) / blockDim.x;

        // Calculate the start and end indices for this thread
        int thread_start_idx = block_start_idx + threadIdx.x * digits_per_thread;
        int thread_end_idx = min(thread_start_idx + digits_per_thread, block_end_idx);

        // Shared memory for per-thread carries
        __shared__ uint64_t shared_carries[ThreadsPerBlock + 1];

        // Initialize local carry
        uint64_t local_carry = 0;

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
            uint64_t new_sum_high = sum_high + carry_from_low;

            // Extract partial_digit
            uint32_t partial_digit = static_cast<uint32_t>(new_sum_low & 0xFFFFFFFFULL);

            // Compute local carry for next digit
            local_carry = (new_sum_low >> 32) + (new_sum_high << 32);

            // Store the partial digit
            tempProducts[Result_offset + idx] = partial_digit;

            // Continue to next digit without synchronization since carries are local
        }

        // Store the final local_carry of each thread into shared memory
        shared_carries[threadIdx.x] = local_carry;
        __syncthreads();

        // Perform an exclusive scan on shared_carries to compute cumulative carries
        uint64_t cumulative_carry = 0;
        for (int offset = 1; offset < blockDim.x; offset <<= 1) {
            uint64_t val = 0;
            if (threadIdx.x >= offset) {
                val = shared_carries[threadIdx.x - offset];
            }
            __syncthreads();
            uint64_t temp = shared_carries[threadIdx.x];
            shared_carries[threadIdx.x] = temp + val;
            __syncthreads();
        }

        // Get the cumulative carry for this thread
        cumulative_carry = (threadIdx.x == 0) ? 0 : shared_carries[threadIdx.x - 1];

        // Each thread adds the cumulative carry to the digits it processed
        local_carry = cumulative_carry;
        for (int idx = thread_start_idx; idx < thread_end_idx; ++idx) {
            // Read the previously stored partial_digit
            uint32_t partial_digit = tempProducts[Result_offset + idx];

            // Add local_carry to partial_digit
            uint64_t sum = static_cast<uint64_t>(partial_digit) + local_carry;

            // Update partial_digit
            partial_digit = static_cast<uint32_t>(sum & 0xFFFFFFFFULL);
            tempProducts[Result_offset + idx] = partial_digit;

            // Compute new local_carry for next digit
            local_carry = sum >> 32;
        }

        // Store the final local_carry of each thread into shared memory
        shared_carries[threadIdx.x] = local_carry;
        __syncthreads();

        // The block's carry-out is the carry from the last thread
        if (threadIdx.x == blockDim.x - 1) {
            block_carry_outs[blockIdx.x] = local_carry;
        }
    }

    // Synchronize all blocks
    grid.sync();

    // Inter-Block Carry Propagation
    bool carries_remaining;
    do {
        carries_remaining = false;

        // Get carry-in from the previous block
        uint64_t block_carry_in = 0;
        if (blockIdx.x > 0) {
            block_carry_in = block_carry_outs[blockIdx.x - 1];
        }

        // If there is a carry-in, process the digits again
        if (block_carry_in > 0) {
            carries_remaining = true;

            // Reset local carry
            uint64_t local_carry = block_carry_in;

            // Each thread processes its assigned digits
            for (int idx = block_start_idx + threadIdx.x; idx < block_end_idx; idx += blockDim.x) {
                // Read the previously stored digit
                uint32_t partial_digit = tempProducts[Result_offset + idx];

                // Add local carry to partial_digit
                uint64_t sum = static_cast<uint64_t>(partial_digit) + local_carry;

                // Update partial_digit
                partial_digit = static_cast<uint32_t>(sum & 0xFFFFFFFFULL);
                tempProducts[Result_offset + idx] = partial_digit;

                // Compute new local carry
                local_carry = sum >> 32;

                // Synchronize threads before processing the next digit
                __syncthreads();
            }

            // Update block's carry-out
            if (threadIdx.x == 0) {
                block_carry_outs[blockIdx.x] = local_carry;
            }
        }

        // Synchronize all blocks before checking if carries remain
        grid.sync();

        // Use shared memory to check if any block has carries remaining
        __shared__ bool any_block_carries_remaining;
        if (threadIdx.x == 0) {
            any_block_carries_remaining = carries_remaining;
        }
        __syncthreads();

        // Determine if any block has carries remaining
        bool carries_remaining_global = any_block_carries_remaining;
        __syncthreads();

        // Synchronize all blocks before next iteration
        grid.sync();

        // If no carries remain, exit the loop
        if (!carries_remaining_global) {
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


    // ---- Finalize the Result ----

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

