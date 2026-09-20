#include "LaunchParamsCalculator.h"
#include "ConsoleLog.h"
#include "Environment.h"

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
ClampNonzero(int value)
{
    return (value <= 0) ? 1 : value;
}

static inline int
ClampInt(int value, int lowerBound, int upperBound)
{
    if (lowerBound > 0)
        value = std::max(value, lowerBound);
    if (upperBound > 0)
        value = std::min(value, upperBound);
    return value;
}

static inline int
CeilDivU64ToInt(uint64_t numerator, uint64_t denominator)
{
    if (denominator == 0)
        return INT_MAX;
    uint64_t quotient = (numerator + denominator - 1ull) / denominator;
    return (quotient > (uint64_t)INT_MAX) ? INT_MAX : (int)quotient;
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
        Environment::DebugBreakpoint();
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
            &tmpMinGrid, &tmpBlock, kernelFunc, (int)dynSmemBytes, blockSizeLimit);
        if (e != cudaSuccess)
            return fail(e);
        minGridSize = tmpMinGrid;
    } else {
        cudaError_t e = cudaOccupancyMaxPotentialBlockSize(
            &minGridSize, &blockSize, kernelFunc, (int)dynSmemBytes, blockSizeLimit);
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
    blocksPerSM = ClampNonzero(blocksPerSM);

    const int fillMachineBlocks = ClampNonzero(blocksPerSM * smCount);

    // ----- N coverage blocks (only meaningful for non-grid-stride mapping) -----
    int coverNBlocks = 1;
    if (N > 0) {
        coverNBlocks = ClampNonzero(CeilDivU64ToInt(N, (uint64_t)blockSize));
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

    chosenBlocks = ClampInt(chosenBlocks, minBlocks, maxBlocks);
    chosenBlocks = ClampNonzero(chosenBlocks);

    blocks = chosenBlocks;
    status = cudaSuccess;

    FractalSharkLog::LogLine(__FILE__, __LINE__)
        << "CudaLaunchConfig: blocks=" << blocks << " threadsPerBlock=" << threadsPerBlock
        << " blocksPerSM=" << blocksPerSM << " SMs=" << smCount
        << " sharedMemBytes=" << dynamicSmemBytes;

    return status;
}

} // namespace HpShark
