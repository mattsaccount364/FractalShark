#include "LaunchParamsCalculator.h"
#include "MultiplyNTT.cu"
#include "TestVerbose.h"
#include <sstream>
#include <stdexcept>

template <class SharkFloatParams>
__maxnreg__(HpShark::RegisterLimit) __global__
    void MultiplyKernelNTT(HpSharkComboResults<SharkFloatParams> *combo, uint64_t *tempProducts)
{

    // Initialize cooperative grid group
    cg::grid_group grid = cg::this_grid();
    cg::thread_block block = cg::this_thread_block();

    // Call the MultiplyHelper function
    // MultiplyHelper(A, B, Out, carryIns, grid, tempProducts);
    if constexpr (!SharkFloatParams::ForceNoOp) {
        MultiplyHelperNTT(combo, grid, block, tempProducts);
    } else {
        grid.sync();
    }
}

template <class SharkFloatParams>
__global__ void
__maxnreg__(HpShark::RegisterLimit)
    MultiplyKernelNTTTestLoop(HpSharkComboResults<SharkFloatParams> *combo,
                              uint64_t numIters,
                              uint64_t *tempProducts)
{ // Array to store cumulative carries

    // Initialize cooperative grid group
    cg::grid_group grid = cg::this_grid();
    cg::thread_block block = cg::this_thread_block();

    for (int i = 0; i < numIters; ++i) {
        // MultiplyHelper(A, B, Out, carryIns, grid, tempProducts);
        if constexpr (!SharkFloatParams::ForceNoOp) {
            MultiplyHelperNTT(combo, grid, block, tempProducts);
        } else {
            grid.sync();
        }
    }
}

template <class SharkFloatParams>
void
ComputeMultiplyNTTGpu(const HpShark::LaunchParams &launchParams, void *kernelArgs[])
{
    constexpr auto SharedMemSize = HpShark::CalculateNTTSharedMemorySize<SharkFloatParams>();

    if constexpr (HpShark::CustomStream) {
        cudaError_t err = cudaFuncSetAttribute(MultiplyKernelNTT<SharkFloatParams>,
                                               cudaFuncAttributeMaxDynamicSharedMemorySize,
                                               SharedMemSize);
        if (err != cudaSuccess) {
            std::ostringstream oss;
            oss << "cudaFuncSetAttribute(MultiplyKernelNTT, MaxDynamicSharedMemorySize) failed: "
                << cudaGetErrorString(err) << " (code " << static_cast<int>(err) << ")"
                << " | requested shmem=" << SharedMemSize;
            throw std::runtime_error(oss.str());
        }

        if (SharkVerbose == VerboseMode::Debug) {
            PrintMaxActiveBlocks<SharkFloatParams>(
                launchParams, MultiplyKernelNTT<SharkFloatParams>, SharedMemSize);
        }
    }

    HpShark::LaunchParams newLaunchParams{launchParams};
    if (newLaunchParams.NumBlocks == 0) {
        HpShark::CudaLaunchConfig launchConfig;
        launchConfig.compute(MultiplyKernelNTT<SharkFloatParams>, SharedMemSize, newLaunchParams);
    }

    cudaError_t err = cudaLaunchCooperativeKernel((void *)MultiplyKernelNTT<SharkFloatParams>,
                                                  dim3(newLaunchParams.NumBlocks),
                                                  dim3(newLaunchParams.ThreadsPerBlock),
                                                  kernelArgs,
                                                  SharedMemSize,
                                                  0);
    if (err != cudaSuccess) {
        std::ostringstream oss;
        oss << "cudaLaunchCooperativeKernel(MultiplyKernelNTT) failed: " << cudaGetErrorString(err)
            << " (code " << static_cast<int>(err) << ")"
            << " | blocks=" << newLaunchParams.NumBlocks
            << " threads=" << newLaunchParams.ThreadsPerBlock << " shmem=" << SharedMemSize;
        throw std::runtime_error(oss.str());
    }

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::ostringstream oss;
        oss << "cudaGetLastError() after MultiplyKernelNTT launch failed: " << cudaGetErrorString(err)
            << " (code " << static_cast<int>(err) << ")";
        throw std::runtime_error(oss.str());
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::ostringstream oss;
        oss << "cudaDeviceSynchronize() after MultiplyKernelNTT failed: " << cudaGetErrorString(err)
            << " (code " << static_cast<int>(err) << ")";
        throw std::runtime_error(oss.str());
    }
}

template <class SharkFloatParams>
void
ComputeMultiplyNTTGpuTestLoop(const HpShark::LaunchParams &launchParams,
                              cudaStream_t &stream,
                              void *kernelArgs[])
{
    constexpr auto SharedMemSize = HpShark::CalculateNTTSharedMemorySize<SharkFloatParams>();

    if constexpr (HpShark::CustomStream) {
        cudaError_t err = cudaFuncSetAttribute(MultiplyKernelNTTTestLoop<SharkFloatParams>,
                                               cudaFuncAttributeMaxDynamicSharedMemorySize,
                                               SharedMemSize);
        if (err != cudaSuccess) {
            std::ostringstream oss;
            oss << "cudaFuncSetAttribute(MultiplyKernelNTTTestLoop, MaxDynamicSharedMemorySize) failed: "
                << cudaGetErrorString(err) << " (code " << static_cast<int>(err) << ")"
                << " | requested shmem=" << SharedMemSize;
            throw std::runtime_error(oss.str());
        }

        if (SharkVerbose == VerboseMode::Debug) {
            PrintMaxActiveBlocks<SharkFloatParams>(
                launchParams, MultiplyKernelNTTTestLoop<SharkFloatParams>, SharedMemSize);
        }
    }

    HpShark::LaunchParams newLaunchParams{launchParams};
    if (newLaunchParams.NumBlocks == 0) {
        HpShark::CudaLaunchConfig launchConfig;
        launchConfig.compute(
            MultiplyKernelNTTTestLoop<SharkFloatParams>, SharedMemSize, newLaunchParams);
    }

    cudaError_t err = cudaLaunchCooperativeKernel((void *)MultiplyKernelNTTTestLoop<SharkFloatParams>,
                                                  dim3(newLaunchParams.NumBlocks),
                                                  dim3(newLaunchParams.ThreadsPerBlock),
                                                  kernelArgs,
                                                  SharedMemSize,
                                                  stream);
    if (err != cudaSuccess) {
        std::ostringstream oss;
        oss << "cudaLaunchCooperativeKernel(MultiplyKernelNTTTestLoop) failed: "
            << cudaGetErrorString(err) << " (code " << static_cast<int>(err) << ")"
            << " | blocks=" << newLaunchParams.NumBlocks
            << " threads=" << newLaunchParams.ThreadsPerBlock << " shmem=" << SharedMemSize;
        throw std::runtime_error(oss.str());
    }

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::ostringstream oss;
        oss << "cudaGetLastError() after MultiplyKernelNTTTestLoop launch failed: "
            << cudaGetErrorString(err) << " (code " << static_cast<int>(err) << ")";
        throw std::runtime_error(oss.str());
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::ostringstream oss;
        oss << "cudaDeviceSynchronize() after MultiplyKernelNTTTestLoop failed: "
            << cudaGetErrorString(err) << " (code " << static_cast<int>(err) << ")";
        throw std::runtime_error(oss.str());
    }
}

//#define ExplicitlyInstantiate(SharkFloatParams)                                                         \
//    template void ComputeMultiplyNTTGpu<SharkFloatParams>(const HpShark::LaunchParams &launchParams,    \
//                                                          void *kernelArgs[]);                          \
//    template void ComputeMultiplyNTTGpuTestLoop<SharkFloatParams>(                                      \
//        const HpShark::LaunchParams &launchParams, cudaStream_t &stream, void *kernelArgs[]);
//
// ExplicitInstantiateAll();
