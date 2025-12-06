#pragma once

#include "CudaCrap.h"
#include "HpSharkFloat.cuh"
#include <stdint.h>

template<class SharkFloatParams>
struct HpSharkFloat;

bool CheckAllTestsPassed();

template<class SharkFloatParams>
void ComputeAddGpuTestLoop(const HpShark::LaunchParams &launchParams, void *kernelArgs[]);

template<class SharkFloatParams>
void ComputeAddGpu(const HpShark::LaunchParams &launchParams, void *kernelArgs[]);

template<class SharkFloatParams>
CUDA_GLOBAL void AddKernel(
    HpSharkAddComboResults<SharkFloatParams> *combo,
    uint64_t *tempData);

template<class SharkFloatParams>
CUDA_GLOBAL void AddKernelTestLoop(
    HpSharkAddComboResults<SharkFloatParams> *combo,
    uint64_t numIters,
    uint64_t *tempData);

