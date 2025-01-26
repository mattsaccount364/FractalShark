#pragma once

#include "CudaCrap.h"

#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda/barrier>

#include "HpSharkFloat.cuh"

template<class SharkFloatParams>
CUDA_CRAP void MultiplyHelperKaratsubaV2(
    const HpSharkFloat<SharkFloatParams> *__restrict__ A,
    const HpSharkFloat<SharkFloatParams> *__restrict__ B,
    HpSharkFloat<SharkFloatParams> *__restrict__ Out,
    cooperative_groups::grid_group grid,
    uint64_t *__restrict__ tempProducts);
