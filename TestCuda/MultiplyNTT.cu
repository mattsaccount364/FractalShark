#include "MultiplyNTT.cuh"

#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#include "BenchmarkTimer.h"
#include "DebugChecksum.cuh"
#include "HpSharkFloat.cuh"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <gmp.h>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda/barrier>

namespace cg = cooperative_groups;


template <class SharkFloatParams, DebugStatePurpose Purpose>
__device__ SharkForceInlineReleaseOnly static void
EraseCurrentDebugState(RecordIt record,
                       DebugState<SharkFloatParams>* debugStates,
                       cooperative_groups::grid_group& grid,
                       cooperative_groups::thread_block& block)
{
    constexpr auto RecursionDepth = 0;
    constexpr auto CallIndex = 0;
    constexpr auto maxPurposes = static_cast<int>(DebugStatePurpose::NumPurposes);
    constexpr auto curPurpose = static_cast<int>(Purpose);
    debugStates[curPurpose].Erase(
        record, grid, block, Purpose, RecursionDepth, CallIndex);
}

template <class SharkFloatParams>
static __device__ void
MultiplyHelperNTTV2Separates(const HpSharkFloat<SharkFloatParams>* SharkRestrict A,
                                   const HpSharkFloat<SharkFloatParams>* SharkRestrict B,
                                   HpSharkFloat<SharkFloatParams>* SharkRestrict OutXX,
                                   HpSharkFloat<SharkFloatParams>* SharkRestrict OutXY,
                                   HpSharkFloat<SharkFloatParams>* SharkRestrict OutYY,
                                   cg::grid_group& grid,
                                   cg::thread_block& block,
                                   uint64_t* SharkRestrict tempProducts)
{

    extern __shared__ uint32_t shared_data[];

    constexpr auto ExecutionBlockBase = 0;
    constexpr auto ExecutionNumBlocks = SharkFloatParams::GlobalNumBlocks;

    // TODO: indexes
    auto* SharkRestrict debugMultiplyCounts =
        reinterpret_cast<DebugMultiplyCount<SharkFloatParams>*>(&tempProducts[0]);
    auto* SharkRestrict debugStates =
        reinterpret_cast<DebugState<SharkFloatParams>*>(&tempProducts[0]);

    if constexpr (SharkPrintMultiplyCounts) {
        const auto CurBlock = block.group_index().x;
        const auto CurThread = block.thread_index().x;
        debugMultiplyCounts[CurBlock * SharkFloatParams::GlobalThreadsPerBlock + CurThread]
            .DebugMultiplyErase();
    }

    if constexpr (SharkDebugChecksums) {
        const auto CurBlock = block.group_index().x;
        const auto CurThread = block.thread_index().x;
        debugMultiplyCounts[CurBlock * SharkFloatParams::GlobalThreadsPerBlock + CurThread]
            .DebugMultiplyErase();

        const RecordIt record =
            (block.thread_index().x == 0 && block.group_index().x == ExecutionBlockBase) ? RecordIt::Yes
                                                                                         : RecordIt::No;
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Invalid>(
            record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::ADigits>(
            record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::BDigits>(
            record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::CDigits>(
            record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::DDigits>(
            record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::EDigits>(
            record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams,
                               DebugStatePurpose::AHalfHigh>(record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::AHalfLow>(
            record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams,
                               DebugStatePurpose::BHalfHigh>(record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::BHalfLow>(
            record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::XDiff>(
            record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::YDiff>(
            record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z0XX>(
            record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z0XY>(
            record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z0YY>(
            record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z1XX>(
            record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z1XY>(
            record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z1YY>(
            record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z2XX>(
            record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z2XY>(
            record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z2YY>(
            record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z2_Perm1>(
            record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z2_Perm2>(
            record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z2_Perm3>(
            record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z2_Perm4>(
            record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z2_Perm5>(
            record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z2_Perm6>(
            record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams,
                               DebugStatePurpose::Z1_offsetXX>(record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams,
                               DebugStatePurpose::Z1_offsetXY>(record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams,
                               DebugStatePurpose::Z1_offsetYY>(record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams,
                               DebugStatePurpose::Final128XX>(record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams,
                               DebugStatePurpose::Final128XY>(record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams,
                               DebugStatePurpose::Final128YY>(record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams,
                               DebugStatePurpose::FinalAdd1>(record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams,
                               DebugStatePurpose::FinalAdd2>(record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams,
                               DebugStatePurpose::FinalAdd3>(record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams,
                               DebugStatePurpose::Result_offsetXX>(record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams,
                               DebugStatePurpose::Result_offsetXY>(record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams,
                               DebugStatePurpose::Result_offsetYY>(record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams,
                               DebugStatePurpose::Result_Add1>(record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams,
                               DebugStatePurpose::Result_Add2>(record, debugStates, grid, block);
        static_assert(static_cast<int32_t>(DebugStatePurpose::NumPurposes) == 41,
                      "Unexpected number of purposes");
    }
}

template <class SharkFloatParams>
void
PrintMaxActiveBlocks(void* kernelFn, int sharedAmountBytes)
{
    std::cout << "Shared memory size: " << sharedAmountBytes << std::endl;

    int numBlocks;

    {
        // Check the maximum number of active blocks per multiprocessor
        // with the given shared memory size
        // This is useful to determine if we can fit more blocks
        // in the shared memory

        const auto err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &numBlocks, kernelFn, SharkFloatParams::GlobalThreadsPerBlock, sharedAmountBytes);

        if (err != cudaSuccess) {
            std::cerr << "CUDA error in cudaOccupancyMaxActiveBlocksPerMultiprocessor: "
                      << cudaGetErrorString(err) << std::endl;
            return;
        }

        std::cout << "Max active blocks per multiprocessor: " << numBlocks << std::endl;
    }

    {
        size_t availableSharedMemory = 0;
        const auto err = cudaOccupancyAvailableDynamicSMemPerBlock(
            &availableSharedMemory, kernelFn, numBlocks, SharkFloatParams::GlobalThreadsPerBlock);

        if (err != cudaSuccess) {
            std::cerr << "CUDA error in cudaOccupancyAvailableDynamicSMemPerBlock: "
                      << cudaGetErrorString(err) << std::endl;
            return;
        }

        std::cout << "Available shared memory per block: " << availableSharedMemory << std::endl;
    }

    // Check the number of multiprocessors on the device
    int numSM;

    {
        const auto err = cudaDeviceGetAttribute(&numSM, cudaDevAttrMultiProcessorCount, 0);

        if (err != cudaSuccess) {
            std::cerr << "CUDA error in cudaDeviceGetAttribute: " << cudaGetErrorString(err)
                      << std::endl;
            return;
        }

        std::cout << "Number of multiprocessors: " << numSM << std::endl;
    }

    int maxConcurrentBlocks = numSM * numBlocks;

    std::cout << "Max concurrent blocks: " << maxConcurrentBlocks << std::endl;
    if (maxConcurrentBlocks < SharkFloatParams::GlobalNumBlocks) {
        std::cout << "Warning: Max concurrent blocks exceeds the number of blocks requested."
                  << std::endl;
    }

    {
        // Check the maximum number of threads per block
        int maxThreadsPerBlock;
        const auto err = cudaDeviceGetAttribute(&maxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, 0);

        if (err != cudaSuccess) {
            std::cerr << "CUDA error in cudaDeviceGetAttribute: " << cudaGetErrorString(err)
                      << std::endl;
            return;
        }

        std::cout << "Max threads per block: " << maxThreadsPerBlock << std::endl;
    }

    {
        // Check the maximum number of threads per multiprocessor
        int maxThreadsPerMultiprocessor;
        const auto err = cudaDeviceGetAttribute(
            &maxThreadsPerMultiprocessor, cudaDevAttrMaxThreadsPerMultiProcessor, 0);
        if (err != cudaSuccess) {
            std::cerr << "CUDA error in cudaDeviceGetAttribute: " << cudaGetErrorString(err)
                      << std::endl;
            return;
        }
        std::cout << "Max threads per multiprocessor: " << maxThreadsPerMultiprocessor << std::endl;
    }

    // Check if this device supports cooperative launches
    int cooperativeLaunch;

    {
        const auto err = cudaDeviceGetAttribute(&cooperativeLaunch, cudaDevAttrCooperativeLaunch, 0);

        if (err != cudaSuccess) {
            std::cerr << "CUDA error in cudaDeviceGetAttribute: " << cudaGetErrorString(err)
                      << std::endl;
            return;
        }

        if (cooperativeLaunch) {
            std::cout << "This device supports cooperative launches." << std::endl;
        } else {
            std::cout << "This device does not support cooperative launches." << std::endl;
        }
    }
}

// Assuming that SharkFloatParams::GlobalNumUint32 can be large and doesn't fit in shared memory
// We'll use the provided global memory buffers for large intermediates
template <class SharkFloatParams>
static __device__ void
MultiplyHelperNTT(HpSharkComboResults<SharkFloatParams>* SharkRestrict combo,
                          cg::grid_group& grid,
                          cg::thread_block& block,
                          uint64_t* SharkRestrict tempProducts)
{

    MultiplyHelperNTTV2Separates<SharkFloatParams>(&combo->A,
                                                         &combo->B,
                                                         &combo->ResultX2,
                                                         &combo->Result2XY,
                                                         &combo->ResultY2,
                                                         grid,
                                                         block,
                                                         tempProducts);
}
