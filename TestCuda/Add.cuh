#pragma once

template<class SharkFloatParams>
struct HpSharkFloat;

#include "Tests.cuh"

#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda/barrier>

bool CheckAllTestsPassed();

template<class SharkFloatParams>
void ComputeAddGpuTestLoop(void *kernelArgs[]);

template<class SharkFloatParams>
void ComputeAddGpu(void *kernelArgs[]);

template<class SharkFloatParams>
__device__ void AddHelper(
    HpSharkFloat<SharkFloatParams> *A,
    HpSharkFloat<SharkFloatParams> *B,
    HpSharkFloat<SharkFloatParams> *Out,
    GlobalAddBlockData *globalData,
    CarryInfo *carryOuts,        // Array to store carry-out for each block
    uint32_t *cumulativeCarries, // Array to store cumulative carries
    cooperative_groups::grid_group grid,
    int numBlocks);

template<class SharkFloatParams>
__global__ void AddKernel(
    HpSharkFloat<SharkFloatParams> *A,
    HpSharkFloat<SharkFloatParams> *B,
    HpSharkFloat<SharkFloatParams> *Out,
    GlobalAddBlockData *globalBlockData,
    CarryInfo *carryOuts,        // Array to store carry-out for each block
    uint32_t *cumulativeCarries); // Array to store cumulative carries

template<class SharkFloatParams>
__global__ void AddKernelTestLoop(
    HpSharkFloat<SharkFloatParams> *A,
    HpSharkFloat<SharkFloatParams> *B,
    HpSharkFloat<SharkFloatParams> *Out,
    GlobalAddBlockData *globalBlockData,
    CarryInfo *carryOuts,        // Array to store carry-out for each block
    uint32_t *cumulativeCarries); // Array to store cumulative carries

