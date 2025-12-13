#pragma once

#include "CudaCrap.h"
#include <stdint.h>

template <class SharkFloatParams> struct HpSharkFloat;

bool CheckAllTestsPassed();

template <class SharkFloatParams>
void ComputeHpSharkReferenceGpuLoop(const HpShark::LaunchParams &launchParams,
                                    cudaStream_t &stream,
                                    void *kernelArgs[]);

template <class SharkFloatParams>
CUDA_GLOBAL void HpSharkReferenceGpuKernel(HpSharkReferenceResults<SharkFloatParams> *combo,
                                           uint64_t *tempData);

template <class SharkFloatParams>
CUDA_GLOBAL void HpSharkReferenceGpuLoop(HpSharkReferenceResults<SharkFloatParams> *combo,
                                         uint64_t *tempData);
