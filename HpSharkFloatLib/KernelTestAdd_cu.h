#include "Add.cu"
#include "LaunchParamsCalculator.h"

template <class SharkFloatParams>
__global__ void
__maxnreg__(HpShark::RegisterLimit)
    AddKernel(HpSharkAddComboResults<SharkFloatParams> *SharkRestrict combo, uint64_t *tempData)
{

    // Initialize cooperative grid group
    cg::grid_group grid = cg::this_grid();
    cg::thread_block block = cg::this_thread_block();

    // Call the AddHelper function
    AddHelper(grid, block, combo, tempData);
}

template <class SharkFloatParams>
__global__ void
__maxnreg__(HpShark::RegisterLimit)
    AddKernelTestLoop(HpSharkAddComboResults<SharkFloatParams> *SharkRestrict combo,
                      uint64_t numIters,
                      uint64_t *tempData)
{

    // Initialize cooperative grid group
    cg::grid_group grid = cg::this_grid();
    cg::thread_block block = cg::this_thread_block();

    for (int32_t i = 0; i < numIters; ++i) {
        AddHelper(grid, block, combo, tempData);
    }
}

template <class SharkFloatParams>
void
ComputeAddGpu(const HpShark::LaunchParams &launchParams, void *kernelArgs[])
{

    constexpr auto ExpandedNumDigits = SharkFloatParams::GlobalNumUint32;
    constexpr size_t SharedMemSize = HpShark::CalculateNTTSharedMemorySize<SharkFloatParams>();

    HpShark::LaunchParams newLaunchParams{launchParams};
    if (newLaunchParams.NumBlocks == 0) {
        HpShark::CudaLaunchConfig launchConfig;
        launchConfig.compute(AddKernel<SharkFloatParams>, SharedMemSize, newLaunchParams);
    }

    cudaError_t err = cudaLaunchCooperativeKernel((void *)AddKernel<SharkFloatParams>,
                                                  dim3(newLaunchParams.NumBlocks),
                                                  dim3(newLaunchParams.ThreadsPerBlock),
                                                  kernelArgs,
                                                  SharedMemSize, // Shared memory size
                                                  0              // Stream
    );

    cudaDeviceSynchronize();

    if (err != cudaSuccess) {
        std::cerr << "CUDA error in ComputeAddGpu: " << cudaGetErrorString(err) << std::endl;
        assert(false);
    }
}

template <class SharkFloatParams>
void
ComputeAddGpuTestLoop(const HpShark::LaunchParams &launchParams, void *kernelArgs[])
{

    constexpr auto ExpandedNumDigits = SharkFloatParams::GlobalNumUint32;
    constexpr size_t SharedMemSize = HpShark::CalculateNTTSharedMemorySize<SharkFloatParams>();

    HpShark::LaunchParams newLaunchParams{launchParams};
    if (newLaunchParams.NumBlocks == 0) {
        HpShark::CudaLaunchConfig launchConfig;
        launchConfig.compute(AddKernelTestLoop<SharkFloatParams>, SharedMemSize, newLaunchParams);
    }

    cudaError_t err = cudaLaunchCooperativeKernel((void *)AddKernelTestLoop<SharkFloatParams>,
                                                  dim3(newLaunchParams.NumBlocks),
                                                  dim3(newLaunchParams.ThreadsPerBlock),
                                                  kernelArgs,
                                                  SharedMemSize, // Shared memory size
                                                  0              // Stream
    );

    cudaDeviceSynchronize();

    if (err != cudaSuccess) {
        std::cerr << "CUDA error in ComputeAddGpuTestLoop: " << cudaGetErrorString(err) << std::endl;
        assert(false);
    }
}

//#define ExplicitlyInstantiate(SharkFloatParams)                                                         \
//    template void ComputeAddGpu<SharkFloatParams>(const HpShark::LaunchParams &launchParams,            \
//                                                  void *kernelArgs[]);                                  \
//    template void ComputeAddGpuTestLoop<SharkFloatParams>(const HpShark::LaunchParams &launchParams,    \
//                                                          void *kernelArgs[]);
//
//ExplicitInstantiateAll();
