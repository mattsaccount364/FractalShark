#pragma once

#include "CudaCrap.h"

#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda/barrier>

#include "HpSharkFloat.cuh"

template<class SharkFloatParams>
CUDA_CRAP void MultiplyHelperKaratsubaV2(
    HpSharkComboResults<SharkFloatParams> *SharkRestrict combo,
    cooperative_groups::grid_group grid,
    cooperative_groups::thread_block &block,
    uint64_t *SharkRestrict tempProducts);
