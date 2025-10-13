#include "Add.cu"
#include "MultiplyNTT.cu"
#include "PeriodicityChecker.cuh"

//
// Returns true if we should continue iterating, false if we should stop (period found).
//

template <class SharkFloatParams>
__device__ [[nodiscard]] bool
ReferenceHelper(cg::grid_group &grid,
                cg::thread_block &block,
                uint64_t currentIteration,
                typename SharkFloatParams::Float *SharkRestrict cx_cast,
                typename SharkFloatParams::Float *SharkRestrict cy_cast,
                typename SharkFloatParams::Float *SharkRestrict dzdcX,
                typename SharkFloatParams::Float *SharkRestrict dzdcY,
                HpSharkReferenceResults<SharkFloatParams> *SharkRestrict reference,
                uint64_t *tempData,
                typename SharkFloatParams::ReferenceIterT *gpuReferenceIters)
{
    //
    // All threads do periodicity checking and update the period if found.
    //

    if constexpr (SharkFloatParams::Periodicity) {
        const auto shouldContinue = PeriodicityChecker(grid,
                                                       block,
                                                       currentIteration,
                                                       cx_cast,
                                                       cy_cast,
                                                       dzdcX,
                                                       dzdcY,
                                                       reference,
                                                       gpuReferenceIters);

        if (!shouldContinue) {
            return false;
        }
    }

    // Note: no synchronization needed here because periodicity checker
    // does not rely on any output before the next grid.sync inside multiply.

    //
    // Note: the multiply doesn't depend on the constants.
    // A = Z_real
    // B = Z_imaginary
    //

    MultiplyHelperNTTV2Separates<SharkFloatParams>(reference->Multiply.Roots,
                                                   &reference->Multiply.A,
                                                   &reference->Multiply.B,
                                                   &reference->Multiply.ResultX2,
                                                   &reference->Multiply.Result2XY,
                                                   &reference->Multiply.ResultY2,
                                                   grid,
                                                   block,
                                                   tempData);

    // At this point, we've calculated the intermediate results:
    // ResultX2 = Add.C_A * Add.C_A = Z_real^2
    // ResultY2 = Add.E_B * Add.E_B = Z_imaginary^2
    // ResultXY = Add.C_A * Add.E_B = Z_real * Z_imaginary
    // And just above we've multiplied ResultXY by 2.

    AddHelperSeparates<SharkFloatParams>(
        grid,
        block,
        &reference->Multiply.ResultX2,  // x^2 = Z_real^2
        &reference->Multiply.ResultY2,  // y^2 = Z_imaginary^2
        &reference->Add.C_A,            // constant C_real
        &reference->Multiply.Result2XY, // 2*x*y = 2 * Z_real * Z_imaginary
        &reference->Add.E_B,            // constant C_imaginary
        &reference->Multiply.A,         // Real result = Z_real
        &reference->Multiply.B,         // Imaginary result = Z_imaginary
        tempData);

    return true;
}

template <class SharkFloatParams>
__global__ void
HpSharkReferenceGpuKernel(HpSharkReferenceResults<SharkFloatParams> *SharkRestrict combo,
                          uint64_t *tempData)
{

    // Initialize cooperative grid group
    cg::grid_group grid = cg::this_grid();
    cg::thread_block block = cg::this_thread_block();

    // Call the AddHelper function
    constexpr auto currentIteration = 0;

    if constexpr (SharkFloatParams::Periodicity) {
        // This path is not supported: running one iteration with periodicity checking is pointless.
        // Correctness checking of all this should take place via the integrated loop version just below.
        return;
    } else {
        const auto [[maybe_unused]] shouldContinue = ReferenceHelper<SharkFloatParams>(
            grid, block, currentIteration, nullptr, nullptr, nullptr, nullptr, combo, tempData, nullptr);
    }
}

template <class SharkFloatParams>
__global__ void
HpSharkReferenceGpuLoop(HpSharkReferenceResults<SharkFloatParams> *SharkRestrict combo,
                        uint64_t numIters,
                        uint64_t *tempData,
                        typename SharkFloatParams::ReferenceIterT *gpuReferenceIters)
{

    // Initialize cooperative grid group
    cg::grid_group grid = cg::this_grid();
    cg::thread_block block = cg::this_thread_block();

    typename SharkFloatParams::Float dzdcX{1};
    typename SharkFloatParams::Float dzdcY{0};

    typename SharkFloatParams::Float cx_cast = combo->Add.C_A.ToHDRFloat<SharkFloatParams::SubType>(0);
    typename SharkFloatParams::Float cy_cast = combo->Add.E_B.ToHDRFloat<SharkFloatParams::SubType>(0);

    // if constexpr (SharkFloatParams::Periodicity) {
    // 
    //     gpuReferenceIters[0].x = {};
    //     gpuReferenceIters[0].y = {};
    // }

    for (uint64_t i = 0; i < numIters; ++i) {
        const auto shouldContinue = ReferenceHelper(
            grid, block, i, &cx_cast, &cy_cast, &dzdcX, &dzdcY, combo, tempData, gpuReferenceIters);
        if (!shouldContinue) {
            break;
        }
    }
}

template <class SharkFloatParams>
void
ComputeHpSharkReferenceGpu(void *kernelArgs[])
{

    constexpr auto sharedAmountBytes = CalculateNTTSharedMemorySize<SharkFloatParams>();

    if constexpr (SharkCustomStream) {
        cudaFuncSetAttribute(HpSharkReferenceGpuLoop<SharkFloatParams>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             sharedAmountBytes);

        PrintMaxActiveBlocks<SharkFloatParams>(HpSharkReferenceGpuLoop<SharkFloatParams>,
                                               sharedAmountBytes);
    }

    cudaError_t err = cudaLaunchCooperativeKernel((void *)HpSharkReferenceGpuKernel<SharkFloatParams>,
                                                  dim3(SharkFloatParams::GlobalNumBlocks),
                                                  dim3(SharkFloatParams::GlobalThreadsPerBlock),
                                                  kernelArgs,
                                                  sharedAmountBytes,
                                                  0 // Stream
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
ComputeHpSharkReferenceGpuLoop(cudaStream_t &stream, void *kernelArgs[])
{

    constexpr auto sharedAmountBytes = CalculateNTTSharedMemorySize<SharkFloatParams>();

    if constexpr (SharkCustomStream) {
        cudaFuncSetAttribute(HpSharkReferenceGpuLoop<SharkFloatParams>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             sharedAmountBytes);

        PrintMaxActiveBlocks<SharkFloatParams>(HpSharkReferenceGpuLoop<SharkFloatParams>,
                                               sharedAmountBytes);
    }

    cudaError_t err = cudaLaunchCooperativeKernel((void *)HpSharkReferenceGpuLoop<SharkFloatParams>,
                                                  dim3(SharkFloatParams::GlobalNumBlocks),
                                                  dim3(SharkFloatParams::GlobalThreadsPerBlock),
                                                  kernelArgs,
                                                  sharedAmountBytes,
                                                  stream // Stream
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

#define ExplicitlyInstantiate(SharkFloatParams)                                                         \
    template void ComputeHpSharkReferenceGpu<SharkFloatParams>(void *kernelArgs[]);                     \
    template void ComputeHpSharkReferenceGpuLoop<SharkFloatParams>(cudaStream_t & stream,               \
                                                                   void *kernelArgs[]);

#ifdef ENABLE_REFERENCE_KERNEL
ExplicitInstantiateAll();
#endif
