#include "MultiplyNTT.cu"

template <class SharkFloatParams>
__maxnreg__(SharkRegisterLimit) __global__
    void MultiplyKernelNTT(HpSharkComboResults<SharkFloatParams>* combo, uint64_t* tempProducts)
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
__maxnreg__(SharkRegisterLimit)
    MultiplyKernelNTTTestLoop(HpSharkComboResults<SharkFloatParams>* combo,
                                      uint64_t numIters,
                                      uint64_t* tempProducts)
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
ComputeMultiplyNTTGpu(void* kernelArgs[])
{

    cudaError_t err;

    constexpr auto sharedAmountBytes = CalculateNTTSharedMemorySize<SharkFloatParams>();

    if constexpr (SharkCustomStream) {
        cudaFuncSetAttribute(MultiplyKernelNTT<SharkFloatParams>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             sharedAmountBytes);

        PrintMaxActiveBlocks<SharkFloatParams>(MultiplyKernelNTT<SharkFloatParams>,
                                               sharedAmountBytes);
    }

    err = cudaLaunchCooperativeKernel((void*)MultiplyKernelNTT<SharkFloatParams>,
                                      dim3(SharkFloatParams::GlobalNumBlocks),
                                      dim3(SharkFloatParams::GlobalThreadsPerBlock),
                                      kernelArgs,
                                      sharedAmountBytes, // Shared memory size
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
ComputeMultiplyNTTGpuTestLoop(cudaStream_t& stream, void* kernelArgs[])
{

    constexpr auto sharedAmountBytes = CalculateNTTSharedMemorySize<SharkFloatParams>();

    if constexpr (SharkCustomStream) {
        cudaFuncSetAttribute(MultiplyKernelNTTTestLoop<SharkFloatParams>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             sharedAmountBytes);

        PrintMaxActiveBlocks<SharkFloatParams>(MultiplyKernelNTTTestLoop<SharkFloatParams>,
                                               sharedAmountBytes);
    }

    cudaError_t err = cudaLaunchCooperativeKernel((void*)MultiplyKernelNTTTestLoop<SharkFloatParams>,
                                                  dim3(SharkFloatParams::GlobalNumBlocks),
                                                  dim3(SharkFloatParams::GlobalThreadsPerBlock),
                                                  kernelArgs,
                                                  sharedAmountBytes, // Shared memory size
                                                  stream             // Stream
    );

    cudaDeviceSynchronize();

    if (err != cudaSuccess) {
        std::cerr << "CUDA error in MultiplyKernelNTTTestLoop: " << cudaGetErrorString(err) << std::endl;
    }
}

#define ExplicitlyInstantiate(SharkFloatParams)                                                         \
    template void ComputeMultiplyNTTGpu<SharkFloatParams>(void* kernelArgs[]);                          \
    template void ComputeMultiplyNTTGpuTestLoop<SharkFloatParams>(cudaStream_t & stream,                \
                                                                  void* kernelArgs[]);

#if defined(ENABLE_MULTIPLY_NTT_KERNEL)
ExplicitInstantiateAll();
#endif
