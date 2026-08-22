#pragma once

#include "CudaCrap.h"
#include "HpSharkFloat.h"
#include "LaunchParams.h"
#include <stdint.h>

template <class SharkFloatParams>
void ComputeHpSharkReferenceGpuLoop(const HpShark::LaunchParams &launchParams,
                                    cudaStream_t &stream,
                                    void *kernelArgs[]);

template <class SharkFloatParams>
void ComputeHpSharkReferenceSetup(const HpShark::LaunchParams &launchParams,
                                  cudaStream_t &stream,
                                  void *kernelArgs[]);

template <class SharkFloatParams>
CUDA_GLOBAL void HpSharkReferenceGpuLoop(HpSharkReferenceResults<SharkFloatParams> *combo,
                                         uint64_t *tempData);

template <class SharkFloatParams>
CUDA_GLOBAL void HpSharkReferenceSetupKernel(HpSharkReferenceWorkspace<SharkFloatParams> *workspace,
                                             uint64_t *tempData);
