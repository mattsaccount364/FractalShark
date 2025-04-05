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
    HpSharkAddComboResults<SharkFloatParams> *combo,
    uint32_t *tempData);

template<class SharkFloatParams>
CUDA_GLOBAL void AddKernelTestLoop(
    HpSharkAddComboResults<SharkFloatParams> *combo,
    uint64_t numIters,
    uint32_t *tempData);

