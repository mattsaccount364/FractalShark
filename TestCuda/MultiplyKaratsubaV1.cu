#include "Multiply.cuh"

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda/barrier>

#include "HpSharkFloat.cuh"

#include <iostream>

namespace cg = cooperative_groups;

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

    // First Pass: Process convolution results to compute initial digits and local carries
    // Initialize local carry
    uint64_t local_carry = 0;

    // Constants and offsets
    constexpr int MaxPasses = 10; // Maximum number of carry propagation passes
    constexpr int total_result_digits = 2 * SharkFloatParams::NumUint32;

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

// Assuming that SharkFloatParams::NumUint32 can be large and doesn't fit in shared memory
// We'll use the provided global memory buffers for large intermediates
template<class SharkFloatParams>
__device__ void MultiplyHelperKaratsubaV1(
    const HpSharkFloat<SharkFloatParams> *__restrict__ A,
    const HpSharkFloat<SharkFloatParams> *__restrict__ B,
    HpSharkFloat<SharkFloatParams> *__restrict__ Out,
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
    constexpr int GlobalCarryOffset = Result_offset + 1 * N;
    constexpr int CarryInsOffset = GlobalCarryOffset + 1 * N;

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
    uint64_t *block_carry_outs = &tempProducts[CarryInsOffset];

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

    __shared__ uint64_t shared_carries[SharkFloatParams::ThreadsPerBlock];


    if constexpr (!SharkFloatParams::DisableCarryPropagation) {

        uint64_t *globalCarryCheck = &tempProducts[GlobalCarryOffset];

        CarryPropagation<SharkFloatParams>(
            shared_carries,
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
__global__ void MultiplyKernelKaratsubaV1(
    const HpSharkFloat<SharkFloatParams> *A,
    const HpSharkFloat<SharkFloatParams> *B,
    HpSharkFloat<SharkFloatParams> *Out,
    uint64_t *tempProducts) {

    // Initialize cooperative grid group
    cg::grid_group grid = cg::this_grid();

    // Call the MultiplyHelper function
    //MultiplyHelper(A, B, Out, grid, tempProducts);
    MultiplyHelperKaratsubaV1(A, B, Out, grid, tempProducts);
}

template<class SharkFloatParams>
__global__ void MultiplyKernelKaratsubaV1TestLoop(
    HpSharkFloat<SharkFloatParams> *A,
    HpSharkFloat<SharkFloatParams> *B,
    HpSharkFloat<SharkFloatParams> *Out,
    uint64_t *tempProducts) { // Array to store cumulative carries

    // Initialize cooperative grid group
    cg::grid_group grid = cg::this_grid();

    for (int i = 0; i < TestIterCount; ++i) {
        // MultiplyHelper(A, B, Out, grid, tempProducts);
        MultiplyHelperKaratsubaV1(A, B, Out, grid, tempProducts);
    }
}

template<class SharkFloatParams>
void ComputeMultiplyKaratsubaV1Gpu(void *kernelArgs[]) {

    cudaError_t err = cudaLaunchCooperativeKernel(
        (void *)MultiplyKernelKaratsubaV1<SharkFloatParams>,
        dim3(SharkFloatParams::NumBlocks),
        dim3(SharkFloatParams::ThreadsPerBlock),
        kernelArgs,
        0, // Shared memory size
        0 // Stream
    );

    cudaDeviceSynchronize();

    if (err != cudaSuccess) {
        std::cerr << "CUDA error in MultiplyKernelKaratsubaV1: " << cudaGetErrorString(err) << std::endl;
    }
}

template<class SharkFloatParams>
void ComputeMultiplyKaratsubaV1GpuTestLoop(cudaStream_t &stream, void *kernelArgs[]) {

    cudaError_t err = cudaLaunchCooperativeKernel(
        (void *)MultiplyKernelKaratsubaV1TestLoop<SharkFloatParams>,
        dim3(SharkFloatParams::NumBlocks),
        dim3(SharkFloatParams::ThreadsPerBlock),
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
    template void ComputeMultiplyKaratsubaV1Gpu<SharkFloatParams>(void *kernelArgs[]); \
    template void ComputeMultiplyKaratsubaV1GpuTestLoop<SharkFloatParams>(cudaStream_t &stream, void *kernelArgs[]);

ExplicitlyInstantiate(Test4x4SharkParams);
ExplicitlyInstantiate(Test4x2SharkParams);
ExplicitlyInstantiate(Test8x1SharkParams);
ExplicitlyInstantiate(Test8x8SharkParams);

ExplicitlyInstantiate(Test128x64SharkParams);
ExplicitlyInstantiate(Test64x64SharkParams);
ExplicitlyInstantiate(Test32x64SharkParams);
ExplicitlyInstantiate(Test16x64SharkParams);

ExplicitlyInstantiate(Test128x32SharkParams);
ExplicitlyInstantiate(Test128x16SharkParams);
ExplicitlyInstantiate(Test128x8SharkParams);
ExplicitlyInstantiate(Test128x4SharkParams);