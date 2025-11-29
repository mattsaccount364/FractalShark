#pragma once

#include "CudaCrap.h"
#include <cuda_runtime.h>
#include <stdint.h>

struct SharkLaunchParams;

template <class SharkFloatParams> struct HpSharkFloat;

template <class SharkFloatParams> struct HpSharkComboResults;

//#include "KernelInvoke.cuh"

template <class SharkFloatParams>
void ComputeMultiplyNTTGpu(const SharkLaunchParams &launchParams, void *kernelArgs[]);

template <class SharkFloatParams>
void ComputeMultiplyNTTGpuTestLoop(const SharkLaunchParams &launchParams,
                                   cudaStream_t &stream,
                                   void *kernelArgs[]);

template <class SharkFloatParams>
CUDA_GLOBAL void MultiplyKernelNTT(HpSharkComboResults<SharkFloatParams> *combo,
                                    uint64_t* tempProducts);

template <class SharkFloatParams>
CUDA_GLOBAL void MultiplyKernelNTTTestLoop(HpSharkComboResults<SharkFloatParams> *combo,
                                                   uint64_t numIters,
                                                   uint64_t* tempProducts);
