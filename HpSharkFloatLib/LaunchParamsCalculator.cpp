#include "LaunchParamsCalculator.h"

#include <algorithm>
#include <type_traits>

// ---------------- utility ----------------

namespace HpShark {

CudaLaunchConfig::CudaLaunchConfig()
    : threadsPerBlock{}, blocks{}, blocksPerSM{}, smCount{}, minGridSize{}, dynamicSmemBytes{},
      status{cudaSuccess}
{
}

static inline int
clamp_nonzero(int v)
{
    return (v <= 0) ? 1 : v;
}

static inline int
clamp_int(int v, int lo, int hi)
{
    if (lo > 0)
        v = std::max(v, lo);
    if (hi > 0)
        v = std::min(v, hi);
    return v;
}

static inline int
ceil_div_u64_to_int(uint64_t a, uint64_t b)
{
    if (b == 0)
        return INT_MAX;
    uint64_t q = (a + b - 1ull) / b;
    return (q > (uint64_t)INT_MAX) ? INT_MAX : (int)q;
}

bool
CudaLaunchConfig::ok() const
{
    return status == cudaSuccess;
}

cudaError_t
CudaLaunchConfig::compute(const void *kernelFunc, size_t dynSmemBytes, LaunchParams &outLaunchParams)
{
    cudaError_t e = compute(kernelFunc, dynSmemBytes);
    if (e != cudaSuccess)
        return e;
    outLaunchParams = LaunchParams{blocks, threadsPerBlock};
    return cudaSuccess;
}

cudaError_t
CudaLaunchConfig::compute(const void *kernelFunc, size_t dynSmemBytes)
{
    auto fail = [&](cudaError_t e) {
        status = e;
        __debugbreak();
        return e;
    };
    if (!kernelFunc)
        return fail(cudaErrorInvalidDeviceFunction);

    // ----- device -----
    if (device >= 0) {
        if (cudaError_t e = cudaSetDevice(device); e != cudaSuccess)
            return fail(e);
    }

    int curDev = 0;
    if (cudaError_t e = cudaGetDevice(&curDev); e != cudaSuccess)
        return fail(e);

    cudaDeviceProp prop{};
    if (cudaError_t e = cudaGetDeviceProperties(&prop, curDev); e != cudaSuccess)
        return fail(e);

    smCount = prop.multiProcessorCount;

    // ----- choose block size -----
    int blockSize = 0;

    if (preferredThreadsPerBlock > 0) {
        blockSize = preferredThreadsPerBlock;
        // still fill minGridSize for visibility/diagnostics:
        int tmpMinGrid = 0, tmpBlock = 0;
        cudaError_t e = cudaOccupancyMaxPotentialBlockSize(
            &tmpMinGrid, &tmpBlock, kernelFunc, (int)dynSmemBytes, blockSizeLimit ? blockSizeLimit : 0);
        if (e != cudaSuccess)
            return fail(e);
        minGridSize = tmpMinGrid;
    } else {
        cudaError_t e = cudaOccupancyMaxPotentialBlockSize(&minGridSize,
                                                           &blockSize,
                                                           kernelFunc,
                                                           (int)dynSmemBytes,
                                                           blockSizeLimit ? blockSizeLimit : 0);
        if (e != cudaSuccess)
            return fail(e);
    }

    threadsPerBlock = blockSize;
    dynamicSmemBytes = dynSmemBytes;

    // ----- blocks per SM -----
    cudaError_t e =
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernelFunc, blockSize, dynSmemBytes);
    if (e != cudaSuccess)
        return fail(e);

    if (maxBlocksPerSM > 0)
        blocksPerSM = std::min(blocksPerSM, maxBlocksPerSM);
    blocksPerSM = clamp_nonzero(blocksPerSM);

    const int fillMachineBlocks = clamp_nonzero(blocksPerSM * smCount);

    // ----- N coverage blocks (only meaningful for non-grid-stride mapping) -----
    int coverNBlocks = 1;
    if (N > 0) {
        coverNBlocks = clamp_nonzero(ceil_div_u64_to_int(N, (uint64_t)blockSize));
    }

    // ----- final blocks selection policy -----
    int chosenBlocks = 1;
    switch (gridPolicy) {
        case GridPolicy::FillMachine:
            chosenBlocks = fillMachineBlocks;
            break;
        case GridPolicy::CoverNExactly:
            chosenBlocks = coverNBlocks;
            break;
        case GridPolicy::MaxOfBoth:
            chosenBlocks = std::max(fillMachineBlocks, coverNBlocks);
            break;
        default:
            chosenBlocks = fillMachineBlocks;
            break;
    }

    chosenBlocks = clamp_int(chosenBlocks, minBlocks, maxBlocks);
    chosenBlocks = clamp_nonzero(chosenBlocks);

    blocks = chosenBlocks;
    return status = cudaSuccess;
}

} // namespace HpShark