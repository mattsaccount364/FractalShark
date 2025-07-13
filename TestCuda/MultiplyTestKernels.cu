#include "Multiply.cu"

template<class SharkFloatParams>
__maxnreg__(SharkRegisterLimit)
__global__ void MultiplyKernelKaratsubaV2(
    HpSharkComboResults<SharkFloatParams> *combo,
    uint64_t *tempProducts) {

    // Initialize cooperative grid group
    cg::grid_group grid = cg::this_grid();
    cg::thread_block block = cg::this_thread_block();

    // Call the MultiplyHelper function
    //MultiplyHelper(A, B, Out, carryIns, grid, tempProducts);
    if constexpr (!SharkFloatParams::ForceNoOp) {
        MultiplyHelperKaratsubaV2(combo, grid, block, tempProducts);
    } else {
        grid.sync();
    }
}

template<class SharkFloatParams>
__global__ void
__maxnreg__(SharkRegisterLimit)
MultiplyKernelKaratsubaV2TestLoop(
    HpSharkComboResults<SharkFloatParams> *combo,
    uint64_t numIters,
    uint64_t *tempProducts) { // Array to store cumulative carries

    // Initialize cooperative grid group
    cg::grid_group grid = cg::this_grid();
    cg::thread_block block = cg::this_thread_block();

    for (int i = 0; i < numIters; ++i) {
        // MultiplyHelper(A, B, Out, carryIns, grid, tempProducts);
        if constexpr (!SharkFloatParams::ForceNoOp) {
            MultiplyHelperKaratsubaV2(combo, grid, block, tempProducts);
        } else {
            grid.sync();
        }
    }
}

template<class SharkFloatParams>
void PrintMaxActiveBlocks(void *kernelFn, int sharedAmountBytes) {
    std::cout << "Shared memory size: " << sharedAmountBytes << std::endl;

    int numBlocks;

    {
        // Check the maximum number of active blocks per multiprocessor
        // with the given shared memory size
        // This is useful to determine if we can fit more blocks
        // in the shared memory

        const auto err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &numBlocks,
            kernelFn,
            SharkFloatParams::GlobalThreadsPerBlock,
            sharedAmountBytes
        );

        if (err != cudaSuccess) {
            std::cerr << "CUDA error in cudaOccupancyMaxActiveBlocksPerMultiprocessor: " << cudaGetErrorString(err) << std::endl;
            return;
        }

        std::cout << "Max active blocks per multiprocessor: " << numBlocks << std::endl;
    }

    {
        size_t availableSharedMemory = 0;
        const auto err = cudaOccupancyAvailableDynamicSMemPerBlock(
            &availableSharedMemory,
            kernelFn,
            numBlocks,
            SharkFloatParams::GlobalThreadsPerBlock
        );

        if (err != cudaSuccess) {
            std::cerr << "CUDA error in cudaOccupancyAvailableDynamicSMemPerBlock: " << cudaGetErrorString(err) << std::endl;
            return;
        }

        std::cout << "Available shared memory per block: " << availableSharedMemory << std::endl;
    }

    // Check the number of multiprocessors on the device
    int numSM;

    {
        const auto err = cudaDeviceGetAttribute(
            &numSM,
            cudaDevAttrMultiProcessorCount,
            0
        );

        if (err != cudaSuccess) {
            std::cerr << "CUDA error in cudaDeviceGetAttribute: " << cudaGetErrorString(err) << std::endl;
            return;
        }

        std::cout << "Number of multiprocessors: " << numSM << std::endl;
    }

    int maxConcurrentBlocks = numSM * numBlocks;

    std::cout << "Max concurrent blocks: " << maxConcurrentBlocks << std::endl;
    if (maxConcurrentBlocks < SharkFloatParams::GlobalNumBlocks) {
        std::cout << "Warning: Max concurrent blocks exceeds the number of blocks requested." << std::endl;
    }

    {
        // Check the maximum number of threads per block
        int maxThreadsPerBlock;
        const auto err = cudaDeviceGetAttribute(
            &maxThreadsPerBlock,
            cudaDevAttrMaxThreadsPerBlock,
            0
        );

        if (err != cudaSuccess) {
            std::cerr << "CUDA error in cudaDeviceGetAttribute: " << cudaGetErrorString(err) << std::endl;
            return;
        }

        std::cout << "Max threads per block: " << maxThreadsPerBlock << std::endl;
    }

    {
        // Check the maximum number of threads per multiprocessor
        int maxThreadsPerMultiprocessor;
        const auto err = cudaDeviceGetAttribute(
            &maxThreadsPerMultiprocessor,
            cudaDevAttrMaxThreadsPerMultiProcessor,
            0
        );
        if (err != cudaSuccess) {
            std::cerr << "CUDA error in cudaDeviceGetAttribute: " << cudaGetErrorString(err) << std::endl;
            return;
        }
        std::cout << "Max threads per multiprocessor: " << maxThreadsPerMultiprocessor << std::endl;
    }

    // Check if this device supports cooperative launches
    int cooperativeLaunch;

    {
        const auto err = cudaDeviceGetAttribute(
            &cooperativeLaunch,
            cudaDevAttrCooperativeLaunch,
            0
        );

        if (err != cudaSuccess) {
            std::cerr << "CUDA error in cudaDeviceGetAttribute: " << cudaGetErrorString(err) << std::endl;
            return;
        }

        if (cooperativeLaunch) {
            std::cout << "This device supports cooperative launches." << std::endl;
        } else {
            std::cout << "This device does not support cooperative launches." << std::endl;
        }
    }
}

template<class SharkFloatParams>
void ComputeMultiplyKaratsubaV2Gpu(void *kernelArgs[]) {

    cudaError_t err;

    constexpr int NewN = SharkFloatParams::GlobalNumUint32;
    constexpr auto n = (NewN + 1) / 2;              // Half of NewN
    constexpr auto sharedAmountBytes =
        SharkUseSharedMemory ?
        (2 * NewN + 2 * n) * sizeof(uint32_t) :
        SharkConstantSharedRequiredBytes;

    if constexpr (SharkCustomStream) {
        cudaFuncSetAttribute(
            MultiplyKernelKaratsubaV2<SharkFloatParams>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            sharedAmountBytes);

        PrintMaxActiveBlocks<SharkFloatParams>(
            MultiplyKernelKaratsubaV2<SharkFloatParams>,
            sharedAmountBytes);
    }

    err = cudaLaunchCooperativeKernel(
        (void *)MultiplyKernelKaratsubaV2<SharkFloatParams>,
        dim3(SharkFloatParams::GlobalNumBlocks),
        dim3(SharkFloatParams::GlobalThreadsPerBlock),
        kernelArgs,
        sharedAmountBytes, // Shared memory size
        0 // Stream
    );

    auto err2 = cudaGetLastError();
    if (err != cudaSuccess || err2 != cudaSuccess) {
        std::cerr << "CUDA error in cudaLaunchCooperativeKernel: " << cudaGetErrorString(err2) <<
            "err: " << err << std::endl;
    }

    cudaDeviceSynchronize();

    if (err != cudaSuccess) {
        std::cerr << "CUDA error in MultiplyKernelKaratsubaV2: " << cudaGetErrorString(err) << std::endl;
    }
}

template<class SharkFloatParams>
void ComputeMultiplyKaratsubaV2GpuTestLoop(cudaStream_t &stream, void *kernelArgs[]) {

    constexpr int NewN = SharkFloatParams::GlobalNumUint32;
    constexpr auto n = (NewN + 1) / 2;              // Half of NewN
    constexpr auto sharedAmountBytes =
        SharkUseSharedMemory ?
        (2 * NewN + 2 * n) * sizeof(uint32_t) :
        SharkConstantSharedRequiredBytes;

    if constexpr (SharkCustomStream) {
        cudaFuncSetAttribute(
            MultiplyKernelKaratsubaV2TestLoop<SharkFloatParams>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            sharedAmountBytes);

        PrintMaxActiveBlocks<SharkFloatParams>(
            MultiplyKernelKaratsubaV2TestLoop<SharkFloatParams>,
            sharedAmountBytes);
    }

    cudaError_t err = cudaLaunchCooperativeKernel(
        (void *)MultiplyKernelKaratsubaV2TestLoop<SharkFloatParams>,
        dim3(SharkFloatParams::GlobalNumBlocks),
        dim3(SharkFloatParams::GlobalThreadsPerBlock),
        kernelArgs,
        sharedAmountBytes, // Shared memory size
        stream // Stream
    );

    cudaDeviceSynchronize();

    if (err != cudaSuccess) {
        std::cerr << "CUDA error in MultiplyKernelKaratsubaTestLoop: " << cudaGetErrorString(err) << std::endl;
    }
}

#define ExplicitlyInstantiate(SharkFloatParams) \
    template void ComputeMultiplyKaratsubaV2Gpu<SharkFloatParams>(void *kernelArgs[]); \
    template void ComputeMultiplyKaratsubaV2GpuTestLoop<SharkFloatParams>(cudaStream_t &stream, void *kernelArgs[]);

#ifdef SHARK_INCLUDE_KERNELS
ExplicitInstantiateAll();
#endif