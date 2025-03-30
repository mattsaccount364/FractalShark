#pragma once

#include "CudaCrap.h"
#include <stdint.h>

template<class SharkFloatParams>
struct HpSharkFloat;

#include "KernelInvoke.cuh"
#include "Tests.h"

bool CheckAllTestsPassed();

template<class SharkFloatParams>
void ComputeAddGpuTestLoop(void *kernelArgs[]);

template<class SharkFloatParams>
void ComputeAddGpu(void *kernelArgs[]);

template<class SharkFloatParams>
CUDA_GLOBAL void AddKernel(
    HpSharkFloat<SharkFloatParams> *A,
    HpSharkFloat<SharkFloatParams> *B,
    HpSharkFloat<SharkFloatParams> *Out,
    GlobalAddBlockData *globalBlockData,
    CarryInfo *carryOuts,        // Array to store carry-out for each block
    uint32_t *cumulativeCarries); // Array to store cumulative carries

template<class SharkFloatParams>
CUDA_GLOBAL void AddKernelTestLoop(
    HpSharkFloat<SharkFloatParams> *A,
    HpSharkFloat<SharkFloatParams> *B,
    HpSharkFloat<SharkFloatParams> *Out,
    uint64_t numIters,
    GlobalAddBlockData *globalBlockData,
    CarryInfo *carryOuts,        // Array to store carry-out for each block
    uint32_t *cumulativeCarries); // Array to store cumulative carries

