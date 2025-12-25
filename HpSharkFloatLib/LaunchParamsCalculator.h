#pragma once
#include <cstdint>
#include <cuda_runtime.h>

#include "LaunchParams.h"

namespace HpShark {

struct CudaLaunchConfig {
    enum class GridPolicy : uint8_t {
        FillMachine,   // blocks = blocksPerSM * SMs (good for grid-stride)
        CoverNExactly, // blocks = ceil(N / threadsPerBlock) (non-grid-stride mapping)
        MaxOfBoth      // blocks = max(FillMachine, CoverNExactly)
    };

    // ---------- Inputs ----------
    uint64_t N = 0;         // total logical work items (only used by CoverNExactly / MaxOfBoth)
    int device = -1;        // -1 = current device
    int blockSizeLimit = 0; // 0 = no limit
    int maxBlocksPerSM = 0; // 0 = no limit

    GridPolicy gridPolicy = GridPolicy::FillMachine;

    int preferredThreadsPerBlock =
        0; // 0 = let occupancy pick; else force (must be multiple of warp recommended)

    // Optional global clamps (0 = no clamp)
    int minBlocks = 0;
    int maxBlocks = 0;

    // ---------- Outputs ----------
    int threadsPerBlock = 0;
    int blocks = 0;
    int blocksPerSM = 0;
    int smCount = 0;
    int minGridSize = 0;
    size_t dynamicSmemBytes = 0;

    cudaError_t status = cudaSuccess;

    CudaLaunchConfig();

    bool ok() const;

    // Fixed dynamic shared memory (bytes per block)
    cudaError_t compute(const void *kernelFunc, size_t dynSmemBytes, LaunchParams &outLaunchParams);
    cudaError_t compute(const void *kernelFunc, size_t dynSmemBytes);
};

} // namespace HpShark