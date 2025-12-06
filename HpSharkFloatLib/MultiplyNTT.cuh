#pragma once

#include "CudaCrap.h"
#include <cuda_runtime.h>
#include <stdint.h>

namespace HpShark {
struct LaunchParams;
}

template <class SharkFloatParams> struct HpSharkFloat;

template <class SharkFloatParams> struct HpSharkComboResults;

//#include "KernelInvoke.cuh"

template <class SharkFloatParams>
void ComputeMultiplyNTTGpu(const HpShark::LaunchParams &launchParams, void *kernelArgs[]);

template <class SharkFloatParams>
void ComputeMultiplyNTTGpuTestLoop(const HpShark::LaunchParams &launchParams,
                                   cudaStream_t &stream,
                                   void *kernelArgs[]);

template <class SharkFloatParams>
CUDA_GLOBAL void MultiplyKernelNTT(HpSharkComboResults<SharkFloatParams> *combo,
                                    uint64_t* tempProducts);

template <class SharkFloatParams>
CUDA_GLOBAL void MultiplyKernelNTTTestLoop(HpSharkComboResults<SharkFloatParams> *combo,
                                                   uint64_t numIters,
                                                   uint64_t* tempProducts);
