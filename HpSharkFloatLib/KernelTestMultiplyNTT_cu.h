#include "MultiplyNTT.cu"
#include "TestVerbose.h"

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

    cudaError_t err;

    constexpr auto SharedMemSize = HpShark::CalculateNTTSharedMemorySize<SharkFloatParams>();

    if constexpr (HpShark::CustomStream) {
        cudaFuncSetAttribute(MultiplyKernelNTT<SharkFloatParams>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             SharedMemSize);

        if (SharkVerbose == VerboseMode::Debug) {
            PrintMaxActiveBlocks<SharkFloatParams>(
                launchParams, MultiplyKernelNTT<SharkFloatParams>, SharedMemSize);
        }
    }

    err = cudaLaunchCooperativeKernel((void *)MultiplyKernelNTT<SharkFloatParams>,
                                      dim3(launchParams.NumBlocks),
                                      dim3(launchParams.ThreadsPerBlock),
                                      kernelArgs,
                                      SharedMemSize, // Shared memory size
                                      0                  // Stream
    );

    auto err2 = cudaGetLastError();
    if (err != cudaSuccess || err2 != cudaSuccess) {
        std::cerr << "CUDA error in cudaLaunchCooperativeKernel: " << cudaGetErrorString(err2)
                  << "err: " << err << std::endl;
    }

    cudaDeviceSynchronize();

    if (err != cudaSuccess) {
        std::cerr << "CUDA error in MultiplyKernelKaratsubaV2: " << cudaGetErrorString(err) << std::endl;
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
        cudaFuncSetAttribute(MultiplyKernelNTTTestLoop<SharkFloatParams>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             SharedMemSize);

        if (SharkVerbose == VerboseMode::Debug) {
            PrintMaxActiveBlocks<SharkFloatParams>(
                launchParams, MultiplyKernelNTTTestLoop<SharkFloatParams>, SharedMemSize);
        }
    }

    cudaError_t err = cudaLaunchCooperativeKernel((void *)MultiplyKernelNTTTestLoop<SharkFloatParams>,
                                                  dim3(launchParams.NumBlocks),
                                                  dim3(launchParams.ThreadsPerBlock),
                                                  kernelArgs,
                                                  SharedMemSize, // Shared memory size
                                                  stream             // Stream
    );

    cudaDeviceSynchronize();

    if (err != cudaSuccess) {
        std::cerr << "CUDA error in MultiplyKernelNTTTestLoop: " << cudaGetErrorString(err) << std::endl;
    }
}

//#define ExplicitlyInstantiate(SharkFloatParams)                                                         \
//    template void ComputeMultiplyNTTGpu<SharkFloatParams>(const HpShark::LaunchParams &launchParams,    \
//                                                          void *kernelArgs[]);                          \
//    template void ComputeMultiplyNTTGpuTestLoop<SharkFloatParams>(                                      \
//        const HpShark::LaunchParams &launchParams, cudaStream_t &stream, void *kernelArgs[]);
//
//ExplicitInstantiateAll();
