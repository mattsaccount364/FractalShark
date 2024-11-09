#include "Multiply.cuh"

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda/barrier>


#include "HpGpu.cuh"

namespace cg = cooperative_groups;

#if 0

//
// Completely busted toom-3 implementation.  Just wanted to check what it looks like.
// 
// It's complex and this one has many problems.  It's fundamentally broken.
//

__device__ void add192(
    uint64_t a_low, uint64_t a_mid, uint64_t a_high,
    uint64_t b_low, uint64_t b_mid, uint64_t b_high,
    uint64_t &result_low, uint64_t &result_mid, uint64_t &result_high) {

    result_low = a_low + b_low;
    uint64_t carry_low = (result_low < a_low) ? 1 : 0;

    result_mid = a_mid + b_mid + carry_low;
    uint64_t carry_mid = (result_mid < a_mid + carry_low) ? 1 : 0;

    result_high = a_high + b_high + carry_mid;
}

__device__ void subtract192(
    uint64_t a_low, uint64_t a_mid, uint64_t a_high,
    uint64_t b_low, uint64_t b_mid, uint64_t b_high,
    uint64_t &result_low, uint64_t &result_mid, uint64_t &result_high) {

    bool borrow_low = a_low < b_low;
    result_low = a_low - b_low;

    uint64_t temp_mid = a_mid - b_mid - borrow_low;
    bool borrow_mid = (a_mid < b_mid + borrow_low);
    result_mid = temp_mid;

    result_high = a_high - b_high - borrow_mid;
}


__device__ void multiply64to128(
    uint64_t a,
    uint64_t b,
    uint64_t &result_low,
    uint64_t &result_high) {

    uint64_t a_low = static_cast<uint32_t>(a);
    uint64_t a_high = a >> 32;
    uint64_t b_low = static_cast<uint32_t>(b);
    uint64_t b_high = b >> 32;

    // Compute partial products
    uint64_t low_low = a_low * b_low;
    uint64_t low_high = a_low * b_high;
    uint64_t high_low = a_high * b_low;
    uint64_t high_high = a_high * b_high;

    // Compute cross terms and carry
    uint64_t carry = ((low_low >> 32) + (low_high & 0xFFFFFFFFULL) + (high_low & 0xFFFFFFFFULL)) >> 32;

    uint64_t mid = (low_high << 32) + (high_low << 32) + (low_low & 0xFFFFFFFF00000000ULL);

    // Final result
    result_low = (low_low & 0xFFFFFFFFULL) + (mid & 0xFFFFFFFFFFFFFFFFULL);
    result_high = high_high + (low_high >> 32) + (high_low >> 32) + carry + (mid >> 64);
}

__device__ void multiply128By64(
    uint64_t a_low, uint64_t a_high,
    uint64_t b,
    uint64_t &result_low, uint64_t &result_mid, uint64_t &result_high) {

    // Multiply lower 64 bits
    uint64_t low_low, low_high;
    multiply64to128(a_low, b, low_low, low_high);

    // Multiply higher 64 bits
    uint64_t high_low, high_high;
    multiply64to128(a_high, b, high_low, high_high);

    // Add the cross terms
    uint64_t mid_low = low_high + high_low;

    // Handle carry
    uint64_t carry = (mid_low < low_high) ? 1 : 0;

    result_low = low_low;
    result_mid = mid_low;
    result_high = high_high + carry;
}



// Function signature remains the same
__device__ void MultiplyHelperToomCook(
    const HpGpu *__restrict__ A,
    const HpGpu *__restrict__ B,
    HpGpu *__restrict__ Out,
    cg::grid_group grid,
    uint64_t *__restrict__ tempProducts) {

    // Initialize cooperative groups
    cg::thread_block block = cg::this_thread_block();

    constexpr int N = HpGpu::NumUint32;       // Total number of digits
    constexpr int NumParts = 3;               // Toom-3 multiplication
    constexpr int N_part = (N + NumParts - 1) / NumParts; // Digits per part

    // Shared memory allocation
    extern __shared__ uint32_t shared_mem[]; // Dynamically allocated shared memory
    uint32_t *A_shared = &shared_mem[0];
    uint32_t *B_shared = &shared_mem[N_part * NumParts];

    // Each thread processes two digits
    int idx = threadIdx.x * 2;

    // **Step 1: Split A and B into parts and load into shared memory**

    // Load A parts into shared memory
    for (int part = 0; part < NumParts; ++part) {
        int global_idx = part * N_part + idx;
        if (global_idx < N) {
            A_shared[part * N_part + idx] = A->Digits[global_idx];
        } else {
            A_shared[part * N_part + idx] = 0;
        }
        if (global_idx + 1 < N) {
            A_shared[part * N_part + idx + 1] = A->Digits[global_idx + 1];
        } else {
            A_shared[part * N_part + idx + 1] = 0;
        }
    }

    // Load B parts into shared memory
    for (int part = 0; part < NumParts; ++part) {
        int global_idx = part * N_part + idx;
        if (global_idx < N) {
            B_shared[part * N_part + idx] = B->Digits[global_idx];
        } else {
            B_shared[part * N_part + idx] = 0;
        }
        if (global_idx + 1 < N) {
            B_shared[part * N_part + idx + 1] = B->Digits[global_idx + 1];
        } else {
            B_shared[part * N_part + idx + 1] = 0;
        }
    }

    // Synchronize after loading
    block.sync();

    // **Step 2: Evaluate at points 0, 1, -1, 8**

    // We need to compute:
    // EVALUATIONS:
    // A(0), A(1), A(-1), A(8)
    // B(0), B(1), B(-1), B(8)

    // We'll store evaluations in shared memory due to size constraints
    // Each evaluation is stored as an array of uint32_t digits

    uint32_t *A_eval[4];
    uint32_t *B_eval[4];

    // Allocate memory for evaluations in shared memory
    size_t eval_offset = 2 * N_part * NumParts; // After A_shared and B_shared
    for (int i = 0; i < 4; ++i) {
        A_eval[i] = &shared_mem[eval_offset + i * N_part];
        B_eval[i] = &shared_mem[eval_offset + (i + 4) * N_part];
    }

    // Evaluate A and B at the points
    // We will perform per-digit addition for evaluation

    // **Evaluate A and B at x=0 (A0, B0)**
    // A(0) = a0
    // B(0) = b0

    if (idx < N_part) {
        A_eval[0][idx] = A_shared[idx];
        B_eval[0][idx] = B_shared[idx];
    }

    // **Evaluate A and B at x=8 (A8, B8)**
    // A(8) = a2
    // B(8) = b2

    if (idx < N_part) {
        A_eval[3][idx] = A_shared[2 * N_part + idx];
        B_eval[3][idx] = B_shared[2 * N_part + idx];
    }

    // **Evaluate A and B at x=1 (A1, B1)**
    // A(1) = a0 + a1 + a2
    // B(1) = b0 + b1 + b2

    uint64_t A_digit = 0;
    uint64_t B_digit = 0;

    if (idx < N_part) {
        A_digit = static_cast<uint64_t>(A_shared[idx]) +
            static_cast<uint64_t>(A_shared[N_part + idx]) +
            static_cast<uint64_t>(A_shared[2 * N_part + idx]);
        B_digit = static_cast<uint64_t>(B_shared[idx]) +
            static_cast<uint64_t>(B_shared[N_part + idx]) +
            static_cast<uint64_t>(B_shared[2 * N_part + idx]);

        A_eval[1][idx] = static_cast<uint32_t>(A_digit & 0xFFFFFFFFULL);
        B_eval[1][idx] = static_cast<uint32_t>(B_digit & 0xFFFFFFFFULL);
    }

    // **Evaluate A and B at x=-1 (Am1, Bm1)**
    // A(-1) = a0 - a1 + a2
    // B(-1) = b0 - b1 + b2

    if (idx < N_part) {
        uint64_t a0 = A_shared[idx];
        uint64_t a1 = A_shared[N_part + idx];
        uint64_t a2 = A_shared[2 * N_part + idx];
        uint64_t b0 = B_shared[idx];
        uint64_t b1 = B_shared[N_part + idx];
        uint64_t b2 = B_shared[2 * N_part + idx];

        uint64_t A_temp = a0 + a2; // a0 + a2
        uint64_t B_temp = b0 + b2; // b0 + b2

        // Subtract a1 and b1
        uint64_t A_result_low, A_result_high;
        uint64_t B_result_low, B_result_high;
        subtract128(A_temp, 0, a1, 0, A_result_low, A_result_high);
        subtract128(B_temp, 0, b1, 0, B_result_low, B_result_high);

        A_eval[2][idx] = static_cast<uint32_t>(A_result_low & 0xFFFFFFFFULL);
        B_eval[2][idx] = static_cast<uint32_t>(B_result_low & 0xFFFFFFFFULL);
    }

    // Synchronize after evaluations
    block.sync();

    // **Step 3: Pointwise Multiplication**

    // Multiply the evaluations pointwise
    // C(x) = A(x) * B(x)

    uint32_t *C_eval[4]; // Arrays to store C(x) evaluations
    size_t C_eval_offset = eval_offset + 8 * N_part; // After A_eval and B_eval
    for (int i = 0; i < 4; ++i) {
        C_eval[i] = &shared_mem[C_eval_offset + i * 2 * N_part]; // Each product may be up to 64 bits per digit
    }

    for (int eval = 0; eval < 4; ++eval) {
        if (idx < N_part) {
            uint64_t A_val = static_cast<uint64_t>(A_eval[eval][idx]);
            uint64_t B_val = static_cast<uint64_t>(B_eval[eval][idx]);

            // Multiply A_val and B_val using 128-bit multiplication
            uint64_t product_low, product_high;
            multiply64(A_val, B_val, product_low, product_high);

            // Store the result (product_low, product_high)
            C_eval[eval][2 * idx] = static_cast<uint32_t>(product_low & 0xFFFFFFFFULL);
            C_eval[eval][2 * idx + 1] = static_cast<uint32_t>((product_low >> 32) & 0xFFFFFFFFULL);
            // We may need to handle product_high if necessary
            // For simplicity, we'll assume product_high is negligible here
        }
    }

    // Synchronize after pointwise multiplication
    block.sync();

    // **Step 4: Interpolation**

    // Use the evaluation results to compute the coefficients c0, c1, c2

    // We'll store the coefficients in shared memory
    uint32_t *C_coeffs[4]; // c0, c1, c2, c3
    size_t C_coeffs_offset = C_eval_offset + 8 * N_part; // After C_eval
    for (int i = 0; i < 4; ++i) {
        C_coeffs[i] = &shared_mem[C_coeffs_offset + i * 2 * N_part];
    }

    // Compute coefficients
    if (idx < N_part) {
        // c0 = C(0)
        C_coeffs[0][2 * idx] = C_eval[0][2 * idx];
        C_coeffs[0][2 * idx + 1] = C_eval[0][2 * idx + 1];

        // c3 = C(8)
        C_coeffs[3][2 * idx] = C_eval[3][2 * idx];
        C_coeffs[3][2 * idx + 1] = C_eval[3][2 * idx + 1];

        // Compute c1 = (C(1) - C(-1)) / 2
        uint64_t c1_low, c1_high;
        uint64_t c1_temp_low, c1_temp_high;
        subtract128(
            C_eval[1][2 * idx], C_eval[1][2 * idx + 1],
            C_eval[2][2 * idx], C_eval[2][2 * idx + 1],
            c1_temp_low, c1_temp_high);
        // Divide by 2 (shift right by 1)
        c1_high = (c1_temp_high >> 1);
        c1_low = (c1_temp_low >> 1) | (c1_temp_high << 63);

        // Store c1_low and c1_high into C_coeffs[1]
        C_coeffs[1][2 * idx] = static_cast<uint32_t>(c1_low & 0xFFFFFFFFULL);
        C_coeffs[1][2 * idx + 1] = static_cast<uint32_t>((c1_low >> 32) & 0xFFFFFFFFULL);
        C_coeffs[1][2 * idx + 2] = static_cast<uint32_t>(c1_high & 0xFFFFFFFFULL);
        C_coeffs[1][2 * idx + 3] = static_cast<uint32_t>((c1_high >> 32) & 0xFFFFFFFFULL);

        // Multiply C(8) by 2
        uint64_t two_Cinf_low, two_Cinf_mid, two_Cinf_high;
        multiply128By64(
            C_eval[3][2 * idx], C_eval[3][2 * idx + 1],
            2ULL,
            two_Cinf_low, two_Cinf_mid, two_Cinf_high);

        // Compute sum = C(1) + C(-1)
        uint64_t sum_low, sum_high;
        add128(
            C_eval[1][2 * idx], C_eval[1][2 * idx + 1],
            C_eval[2][2 * idx], C_eval[2][2 * idx + 1],
            sum_low, sum_high);

        // Extend sum to 192 bits
        uint64_t sum_mid = 0;

        // Subtract 2 * C(8)
        uint64_t temp_low, temp_mid, temp_high;
        subtract192(
            sum_low, sum_mid, sum_high,
            two_Cinf_low, two_Cinf_mid, two_Cinf_high,
            temp_low, temp_mid, temp_high);

        // Divide by 2 (shift right by 1)
        uint64_t c2_low = (temp_low >> 1) | (temp_mid << 63);
        uint64_t c2_mid = (temp_mid >> 1) | (temp_high << 63);
        uint64_t c2_high = temp_high >> 1;

        // Store c2 into C_coeffs[2]
        C_coeffs[2][2 * idx] = static_cast<uint32_t>(c2_low & 0xFFFFFFFFULL);
        C_coeffs[2][2 * idx + 1] = static_cast<uint32_t>((c2_low >> 32) & 0xFFFFFFFFULL);
        C_coeffs[2][2 * idx + 2] = static_cast<uint32_t>(c2_mid & 0xFFFFFFFFULL);
        C_coeffs[2][2 * idx + 3] = static_cast<uint32_t>((c2_mid >> 32) & 0xFFFFFFFFULL);
        C_coeffs[2][2 * idx + 4] = static_cast<uint32_t>(c2_high & 0xFFFFFFFFULL);
        C_coeffs[2][2 * idx + 5] = static_cast<uint32_t>((c2_high >> 32) & 0xFFFFFFFFULL);
    }

    // Synchronize after interpolation
    block.sync();

    // **Step 5: Recomposition**

    // Compute the final result by recombining the coefficients

    // Each coefficient represents a term in the polynomial:
    // Result = c0 + c1 * x^(N_part) + c2 * x^(2 * N_part) + c3 * x^(3 * N_part)

    // We'll write the result to tempProducts
    // Each thread handles its own portion

    // Initialize tempProducts
    for (int i = idx; i < 4 * N_part; i += ThreadsPerBlock * 2) {
        tempProducts[i] = 0;
    }

    // Synchronize before recomposition
    block.sync();

    // Recompose the result
    for (int coeff = 0; coeff < 4; ++coeff) {
        int result_idx = idx + coeff * N_part;

        if (result_idx < 4 * N_part) {
            // Add the coefficient to the result at the correct position
            uint64_t sum_low = tempProducts[result_idx];
            uint64_t sum_high = 0;
            uint64_t coeff_low = C_coeffs[coeff][2 * idx];
            uint64_t coeff_high = C_coeffs[coeff][2 * idx + 1];

            // Perform 128-bit addition
            add128(sum_low, sum_high, coeff_low, coeff_high, sum_low, sum_high);

            // Store back to tempProducts
            tempProducts[result_idx] = sum_low;
            // We need to handle carry propagation later
        }
    }

    // Synchronize after recomposition
    block.sync();

    // **Step 6: Carry Propagation**

    // We need to perform carry propagation across tempProducts

    // Each thread processes its assigned digits
    uint64_t carry = 0;

    for (int i = idx; i < 4 * N_part; i += ThreadsPerBlock * 2) {
        uint64_t sum = tempProducts[i] + carry;
        tempProducts[i] = static_cast<uint32_t>(sum & 0xFFFFFFFFULL);
        carry = sum >> 32;
    }

    // **Intra-block carry propagation**

    // Store per-thread carry
    __shared__ uint64_t per_thread_carry[ThreadsPerBlock];
    per_thread_carry[threadIdx.x] = carry;

    // Synchronize threads
    block.sync();

    // Perform exclusive scan over per-thread carries
    uint64_t thread_carry_in = 0;
    for (int offset = 1; offset < ThreadsPerBlock; offset <<= 1) {
        uint64_t temp = 0;
        if (threadIdx.x >= offset) {
            temp = per_thread_carry[threadIdx.x - offset];
        }
        block.sync();
        if (threadIdx.x >= offset) {
            per_thread_carry[threadIdx.x] += temp;
        }
        block.sync();
    }

    if (threadIdx.x > 0) {
        thread_carry_in = per_thread_carry[threadIdx.x - 1];
    } else {
        thread_carry_in = 0;
    }

    // Adjust digits with carry_in
    carry = thread_carry_in;
    for (int i = idx; i < 4 * N_part; i += ThreadsPerBlock * 2) {
        uint64_t sum = tempProducts[i] + carry;
        tempProducts[i] = static_cast<uint32_t>(sum & 0xFFFFFFFFULL);
        carry = sum >> 32;
    }

    // Synchronize threads
    block.sync();

    // **Inter-block carry propagation**

    // Store block carry
    if (threadIdx.x == ThreadsPerBlock - 1) {
        // Last thread in block
        tempProducts[4 * N_part + blockIdx.x] = carry;
    }

    // Synchronize blocks
    grid.sync();

    // Inter-block carry propagation
    if (blockIdx.x == 0) {
        uint64_t total_carry = 0;
        for (int b = 0; b < gridDim.x; ++b) {
            uint64_t block_carry = tempProducts[4 * N_part + b];
            uint64_t sum = tempProducts[4 * N_part * NumBlocks + b] + total_carry + block_carry;
            tempProducts[4 * N_part * NumBlocks + b] = static_cast<uint32_t>(sum & 0xFFFFFFFFULL);
            total_carry = sum >> 32;
        }
    }

    // Synchronize blocks
    grid.sync();

    // **Step 7: Adjust Exponent and Shift Result**

    // Since multiplying two N-digit numbers results in up to 2N digits,
    // we may need to adjust the exponent and shift the result.

    if (blockIdx.x == 0 && threadIdx.x == 0) {
        // Determine if we need to shift the result
        int totalResultDigits = 4 * N_part;
        int shifts = totalResultDigits - N;
        if (shifts < 0) {
            shifts = 0;
        }

        // Adjust exponent
        Out->Exponent = A->Exponent + B->Exponent + shifts * 32; // Each digit is 32 bits

        // Copy the most significant N digits to Out->Digits
        for (int i = 0; i < N; ++i) {
            int srcIdx = i + shifts;
            if (srcIdx < totalResultDigits) {
                Out->Digits[i] = static_cast<uint32_t>(tempProducts[srcIdx]);
            } else {
                Out->Digits[i] = 0;
            }
        }

        // Set the sign
        Out->IsNegative = A->IsNegative ^ B->IsNegative;
    }
}

__global__ void MultiplyKernelToomCook(
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
    MultiplyHelperToomCook(A, B, Out, grid, tempProducts);
}

__global__ void MultiplyKernelToomCookTestLoop(
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
        MultiplyHelperToomCook(A, B, Out, grid, tempProducts);
    }
}

#endif

//
// This implementation is busted in several key ways:
// - The per-digit multiplication is not implemented correctly because
//   the sum may overflow.  Use 128-bit addition and keep track of the carries during
//   the first phase.
// - The inter-block carry propagation is incorrect
//

__device__ void MultiplyHelperN2(
    const HpGpu *__restrict__ A,
    const HpGpu *__restrict__ B,
    HpGpu *__restrict__ Out,
    uint64_t *__restrict__ carryOuts_phase3, // Array to store carry-out from Phase 3
    uint64_t *__restrict__ carryOuts_phase6, // Array to store carry-out from Phase 6
    uint64_t *__restrict__ carryIns,          // Array to store carry-in for each block
    cg::grid_group grid,
    uint64_t *__restrict__ tempProducts      // Temporary buffer to store intermediate products
) {
    // Calculate the thread's unique index
    const int threadIdxGlobal = blockIdx.x * ThreadsPerBlock + threadIdx.x;

    const int threadIdxGlobalMin = blockIdx.x * ThreadsPerBlock;
    const int threadIdxGlobalMax = threadIdxGlobalMin + ThreadsPerBlock - 1;

    const int lowDigitIdxMin = threadIdxGlobalMin * 2;
    const int lowDigitIdxMax = threadIdxGlobalMax * 2;

    const int highDigitIdxMin = lowDigitIdxMin + 1;
    const int highDigitIdxMax = lowDigitIdxMax + 1;

    // Each thread handles two digits: low and high
    const int lowDigitIdx = threadIdxGlobal * 2;
    const int highDigitIdx = lowDigitIdx + 1;

    // Ensure indices do not exceed the temporary buffer size
    if (lowDigitIdx >= 2 * HpGpu::NumUint32) return;

    // Initialize temporary products to zero
    tempProducts[lowDigitIdx] = 0;
    if (highDigitIdx < 2 * HpGpu::NumUint32) {
        tempProducts[highDigitIdx] = 0;
    }

    static constexpr int32_t BATCH_SIZE_A = BatchSize;
    static constexpr int32_t BATCH_SIZE_B = BatchSize;

    // Compute k_min and k_max
    const int k_min = 2 * blockIdx.x * ThreadsPerBlock;
    const int k_max = min(2 * (blockIdx.x + 1) * ThreadsPerBlock - 1, 2 * HpGpu::NumUint32 - 1);

    // Compute j_min_block and j_max_block
    const int j_min_block = max(0, k_min - (HpGpu::NumUint32 - 1));
    const int j_max_block = min(k_max, HpGpu::NumUint32 - 1);

    const int a_shared_size_required = j_max_block - j_min_block + 1;

    // Shared memory for A and B with double buffering
    __shared__ __align__(16) uint32_t A_shared[2][BATCH_SIZE_A];
    __shared__ __align__(16) uint32_t B_shared[2][BATCH_SIZE_B];

    const int numBatches_A = (a_shared_size_required + BATCH_SIZE_A - 1) / BATCH_SIZE_A;
    const int numBatches_B = (HpGpu::NumUint32 + BATCH_SIZE_B - 1) / BATCH_SIZE_B;

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
        const int bIndex_max = min(HpGpu::NumUint32 - 1, max(bIndex_max_low, bIndex_max_high));

        const int batchB_start = bIndex_min / BATCH_SIZE_B;
        // const int batchB_end = bIndex_max / BATCH_SIZE_B;

        int batchStartB = batchB_start * BATCH_SIZE_B;
        {
            const int elementsToCopyB = min(BATCH_SIZE_B, HpGpu::NumUint32 - batchStartB);
            cg::memcpy_async(block, &currentBufferB[0], &B->Digits[batchStartB], sizeof(uint32_t) * elementsToCopyB);
        }

        // Loop over batches of B
        for (int batchB = batchB_start; batchB < numBatches_B; ++batchB) {
            //block.sync();

            const int elementsToCopyB = min(BATCH_SIZE_B, HpGpu::NumUint32 - batchStartB);
            const int batchEndB = batchStartB + elementsToCopyB - 1;

            // Start loading the next batch of B asynchronously if not the last batch
            if (batchB + 1 < numBatches_B) {
                int nextBatchStartB = (batchB + 1) * BATCH_SIZE_B;
                int nextElementsToCopyB = min(BATCH_SIZE_B, HpGpu::NumUint32 - nextBatchStartB);

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
                    if (highDigitIdx < 2 * HpGpu::NumUint32 && j >= j_min_high && j <= j_max_high) {
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
    __shared__ uint64_t digitLowShared[ThreadsPerBlock];
    __shared__ uint64_t digitHighShared[ThreadsPerBlock];

    digitLowShared[threadIdx.x] = lowDigitIdxSum;
    digitHighShared[threadIdx.x] = highDigitIdxSum;

    grid.sync();

    if (threadIdx.x == 0) {
        uint64_t carry = 0;

        // Process the digits sequentially
        for (int i = 0; i < ThreadsPerBlock; ++i) {
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
        for (int i = 1; i < NumBlocks; ++i) {
            carryIns[i] = carryOuts_phase6[i - 1];
        }
    }

    grid.sync(); // Ensure all blocks have computed their carryOuts_phase6

    // Each block uses carryIns[blockIdx.x] as its carry-in for inter-block carry propagation
    uint64_t carry = carryIns[blockIdx.x];

    // Now, perform inter-block carry propagation within each block
    if (threadIdx.x == 0) {
        // Process the digits sequentially
        for (int i = 0; i < ThreadsPerBlock; ++i) {
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
        finalCarry = carryOuts_phase6[NumBlocks - 1];
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
    __shared__ int sharedHighestIndices[ThreadsPerBlock];
    sharedHighestIndices[threadIdx.x] = localHighestIndex;
    block.sync();

    // Reduction within block to find blockHighestIndex
    for (int offset = ThreadsPerBlock / 2; offset > 0; offset /= 2) {
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
        for (int i = 0; i < NumBlocks; ++i) {
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
    int shifts = totalResultDigits - HpGpu::NumUint32;
    if (shifts < 0) {
        shifts = 0;
    }

    // Calculate the required shift to ensure maxHighDigitIdx - shifts <= HpGpu::NumUint32 - 1
    int requiredShift = highestIndex - (HpGpu::NumUint32 - 1);
    if (shifts < requiredShift) {
        shifts = requiredShift;
    }

    // Adjust the exponent accordingly
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        Out->Exponent = A->Exponent + B->Exponent + shifts * 32;
        Out->IsNegative = A->IsNegative ^ B->IsNegative;
    }

    // Each thread copies its digits from shared memory to Out->Digits, applying the shift
    if (lowDigitIdx >= shifts && (lowDigitIdx - shifts) < HpGpu::NumUint32) {
        Out->Digits[lowDigitIdx - shifts] = static_cast<uint32_t>(digitLowShared[threadIdx.x]);
    }
    if (highDigitIdx >= shifts && (highDigitIdx - shifts) < HpGpu::NumUint32) {
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
        for (int i = HpGpu::NumUint32 - 1; i >= carryDigits; --i) {
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
