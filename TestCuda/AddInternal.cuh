#pragma once


#include "CudaCrap.h"
#include <stdint.h>

#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda/barrier>

#include "HpSharkFloat.cuh"

template<class SharkFloatParams>
CUDA_CRAP void AddHelper(
    HpSharkFloat<SharkFloatParams> *A,
    HpSharkFloat<SharkFloatParams> *B,
    HpSharkFloat<SharkFloatParams> *Out,
    GlobalAddBlockData *globalData,
    CarryInfo *carryOuts,        // Array to store carry-out for each block
    uint32_t *cumulativeCarries, // Array to store cumulative carries
    cooperative_groups::grid_group grid,
    int numBlocks);