#pragma once

#include "CudaCrap.h"
#include <stdint.h>

template <class SharkFloatParams> struct HpSharkFloat;

template <class SharkFloatParams> struct HpSharkComboResults;

#include "KernelInvoke.cuh"
#include "Tests.h"

template <class SharkFloatParams> void ComputeMultiplyNTTGpu(void* kernelArgs[]);

template <class SharkFloatParams>
void ComputeMultiplyNTTGpuTestLoop(cudaStream_t& stream, void* kernelArgs[]);

template <class SharkFloatParams>
CUDA_GLOBAL void MultiplyKernelNTT(HpSharkComboResults<SharkFloatParams>* combo,
                                    uint64_t* tempProducts);

template <class SharkFloatParams>
CUDA_GLOBAL void MultiplyKernelNTTTestLoop(HpSharkComboResults<SharkFloatParams>* combo,
                                                   uint64_t numIters,
                                                   uint64_t* tempProducts);
