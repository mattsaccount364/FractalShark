#pragma once

#include "CudaCrap.h"
#include <stdint.h>

template<class SharkFloatParams>
struct HpSharkFloat;

template<class SharkFloatParams>
struct HpSharkComboResults;

#include "KernelInvoke.cuh"
#include "Tests.h"

void TestMultiplyTwoNumbers(int testNum, const char *num1, const char *num2);
bool CheckAllTestsPassed();

template<class SharkFloatParams>
void ComputeMultiplyKaratsubaV2Gpu(void *kernelArgs[]);

template<class SharkFloatParams>
void ComputeMultiplyKaratsubaV2GpuTestLoop(
    cudaStream_t &stream,
    void *kernelArgs[]);

template<class SharkFloatParams>
CUDA_GLOBAL void MultiplyKernelKaratsubaV2(
    HpSharkComboResults<SharkFloatParams> *combo,
    uint64_t *tempProducts);

template<class SharkFloatParams>
CUDA_GLOBAL void MultiplyKernelKaratsubaV2TestLoop(
    HpSharkComboResults<SharkFloatParams> *combo,
    uint64_t numIters,
    uint64_t *tempProducts);
