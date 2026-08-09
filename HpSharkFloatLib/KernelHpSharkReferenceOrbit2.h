#pragma once

#include "CudaCrap.h"
#include "HpSharkFloat.h"
#include "LaunchParams.h"
#include <stdint.h>

template <class SharkFloatParams>
void ComputeHpSharkReference2GpuLoop(const HpShark::LaunchParams &launchParams,
                                     cudaStream_t &stream,
                                     void *kernelArgs[]);

template <class SharkFloatParams>
void ComputeHpSharkReference2Setup(const HpShark::LaunchParams &launchParams,
                                   cudaStream_t &stream,
                                   void *kernelArgs[]);

template <class SharkFloatParams>
CUDA_GLOBAL void HpSharkReference2GpuLoop(HpSharkReferenceResults<SharkFloatParams> *combo,
                                          uint64_t *tempData);

template <class SharkFloatParams>
CUDA_GLOBAL void HpSharkReference2SetupKernel(HpSharkReference2Workspace<SharkFloatParams> *workspace,
                                              uint64_t *tempData);
