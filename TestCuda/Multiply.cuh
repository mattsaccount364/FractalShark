#pragma once

struct HpGpu;

#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda/barrier>

void TestMultiplyTwoNumbers(int testNum, const char *num1, const char *num2);
bool CheckAllTestsPassed();

void ComputeMultiplyGpu(void *kernelArgs[]);
void ComputeMultiplyGpuTestLoop(void *kernelArgs[]);

__device__ void MultiplyHelperKaratsuba(
    const HpGpu *__restrict__ A,
    const HpGpu *__restrict__ B,
    HpGpu *__restrict__ Out,
    uint64_t *__restrict__ carryOuts_phase3,
    uint64_t *__restrict__ carryOuts_phase6,
    uint64_t *__restrict__ carryIns,
    cooperative_groups::grid_group grid,
    uint64_t *__restrict__ tempProducts);

__device__ void MultiplyHelperN2(
    const HpGpu *__restrict__ A,
    const HpGpu *__restrict__ B,
    HpGpu *__restrict__ Out,
    uint64_t *__restrict__ carryOuts_phase3, // Array to store carry-out from Phase 3
    uint64_t *__restrict__ carryOuts_phase6, // Array to store carry-out from Phase 6
    uint64_t *__restrict__ carryIns,          // Array to store carry-in for each block
    cooperative_groups::grid_group grid,
    uint64_t *__restrict__ tempProducts);      // Temporary buffer to store intermediate products

__global__ void MultiplyKernelKaratsuba(
    const HpGpu *A,
    const HpGpu *B,
    HpGpu *Out,
    uint64_t *carryOuts_phase3,
    uint64_t *carryOuts_phase6,
    uint64_t *carryIns,
    uint64_t *tempProducts);

__global__ void MultiplyKernelKaratsubaTestLoop(
    HpGpu *A,
    HpGpu *B,
    HpGpu *Out,
    uint64_t *carryOuts_phase3,
    uint64_t *carryOuts_phase6,
    uint64_t *carryIns,
    uint64_t *tempProducts);