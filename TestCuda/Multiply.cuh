#pragma once

template<class SharkFloatParams>
struct HpSharkFloat;

#include "Tests.cuh"

#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda/barrier>

void TestMultiplyTwoNumbers(int testNum, const char *num1, const char *num2);
bool CheckAllTestsPassed();

////////////////////////////////////////

template<class SharkFloatParams>
void ComputeMultiplyN2Gpu(void *kernelArgs[]);

template<class SharkFloatParams>
void ComputeMultiplyN2GpuTestLoop(void *kernelArgs[]);

template<class SharkFloatParams>
__device__ void MultiplyHelperN2(
    const HpSharkFloat<SharkFloatParams> *__restrict__ A,
    const HpSharkFloat<SharkFloatParams> *__restrict__ B,
    HpSharkFloat<SharkFloatParams> *__restrict__ Out,
    uint64_t *__restrict__ carryOuts_phase3,
    uint64_t *__restrict__ carryOuts_phase6,
    uint64_t *__restrict__ carryIns,
    cooperative_groups::grid_group grid,
    uint64_t *__restrict__ tempProducts);

template<class SharkFloatParams>
__global__ void MultiplyKernelN2(
    const HpSharkFloat<SharkFloatParams> *A,
    const HpSharkFloat<SharkFloatParams> *B,
    HpSharkFloat<SharkFloatParams> *Out,
    uint64_t *carryOuts_phase3,
    uint64_t *carryOuts_phase6,
    uint64_t *carryIns,
    uint64_t *tempProducts);

template<class SharkFloatParams>
__global__ void MultiplyKernelN2TestLoop(
    HpSharkFloat<SharkFloatParams> *A,
    HpSharkFloat<SharkFloatParams> *B,
    HpSharkFloat<SharkFloatParams> *Out,
    uint64_t *carryOuts_phase3,
    uint64_t *carryOuts_phase6,
    uint64_t *carryIns,
    uint64_t *tempProducts);


////////////////////////////////////////

template<class SharkFloatParams>
void ComputeMultiplyKaratsubaV1Gpu(void *kernelArgs[]);

template<class SharkFloatParams>
void ComputeMultiplyKaratsubaV1GpuTestLoop(void *kernelArgs[]);

template<class SharkFloatParams>
__device__ void MultiplyHelperKaratsubaV1(
    const HpSharkFloat<SharkFloatParams> *__restrict__ A,
    const HpSharkFloat<SharkFloatParams> *__restrict__ B,
    HpSharkFloat<SharkFloatParams> *__restrict__ Out,
    cooperative_groups::grid_group grid,
    uint64_t *__restrict__ tempProducts);

template<class SharkFloatParams>
__global__ void MultiplyKernelKaratsubaV1(
    const HpSharkFloat<SharkFloatParams> *A,
    const HpSharkFloat<SharkFloatParams> *B,
    HpSharkFloat<SharkFloatParams> *Out,
    uint64_t *tempProducts);

template<class SharkFloatParams>
__global__ void MultiplyKernelKaratsubaV1TestLoop(
    HpSharkFloat<SharkFloatParams> *A,
    HpSharkFloat<SharkFloatParams> *B,
    HpSharkFloat<SharkFloatParams> *Out,
    uint64_t *tempProducts);

////////////////////////////////////////

template<class SharkFloatParams>
void ComputeMultiplyKaratsubaV2Gpu(void *kernelArgs[]);

template<class SharkFloatParams>
void ComputeMultiplyKaratsubaV2GpuTestLoop(void *kernelArgs[]);

template<class SharkFloatParams>
__device__ void MultiplyHelperKaratsubaV2(
    const HpSharkFloat<SharkFloatParams> *__restrict__ A,
    const HpSharkFloat<SharkFloatParams> *__restrict__ B,
    HpSharkFloat<SharkFloatParams> *__restrict__ Out,
    cooperative_groups::grid_group grid,
    uint64_t *__restrict__ tempProducts);

template<class SharkFloatParams>
__device__ void MultiplyHelperN2(
    const HpSharkFloat<SharkFloatParams> *__restrict__ A,
    const HpSharkFloat<SharkFloatParams> *__restrict__ B,
    HpSharkFloat<SharkFloatParams> *__restrict__ Out,
    uint64_t *__restrict__ carryOuts_phase3, // Array to store carry-out from Phase 3
    uint64_t *__restrict__ carryOuts_phase6, // Array to store carry-out from Phase 6
    uint64_t *__restrict__ carryIns,          // Array to store carry-in for each block
    cooperative_groups::grid_group grid,
    uint64_t *__restrict__ tempProducts);      // Temporary buffer to store intermediate products

template<class SharkFloatParams>
__global__ void MultiplyKernelKaratsubaV2(
    const HpSharkFloat<SharkFloatParams> *A,
    const HpSharkFloat<SharkFloatParams> *B,
    HpSharkFloat<SharkFloatParams> *Out,
    uint64_t *tempProducts);

template<class SharkFloatParams>
__global__ void MultiplyKernelKaratsubaV2TestLoop(
    HpSharkFloat<SharkFloatParams> *A,
    HpSharkFloat<SharkFloatParams> *B,
    HpSharkFloat<SharkFloatParams> *Out,
    uint64_t *tempProducts);