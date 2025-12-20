#include "Add.cu"
#include "MultiplyNTT.cu"
#include "PeriodicityChecker.h"
#include "TestVerbose.h"

//
// Returns true if we should continue iterating, false if we should stop (period found).
//

template <class SharkFloatParams>
__device__ [[nodiscard]] PeriodicityResult
ReferenceHelper(cg::grid_group &grid,
                cg::thread_block &block,
                uint64_t currentLocalIteration,
                typename SharkFloatParams::Float *SharkRestrict cx_cast,
                typename SharkFloatParams::Float *SharkRestrict cy_cast,
                typename SharkFloatParams::Float *dzdcX,
                typename SharkFloatParams::Float *dzdcY,
                HpSharkReferenceResults<SharkFloatParams> *SharkRestrict reference,
                uint64_t *tempData)
{
    //
    // All threads do periodicity checking and update the period if found.
    //

    if constexpr (SharkFloatParams::EnablePeriodicity) {
        if (block.group_index().x == 0 && block.thread_index().x == 0) {
            PeriodicityChecker(
                grid, block, currentLocalIteration, cx_cast, cy_cast, dzdcX, dzdcY, reference);
        }
    
        //
        // Note: we can get rid of this if we move this check into multiply after the first sync of if we
        // do the same calculation on every thread.  But if we do that then we need to reconcile the fact
        // that there's only one copy of dzdcX/Y currently.
        //

        grid.sync();

        if (reference->PeriodicityStatus !=
                PeriodicityResult::Continue) {
            return reference->PeriodicityStatus;
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

    return PeriodicityResult::Continue;
}

template <class SharkFloatParams>
__global__ void
__maxnreg__(HpShark::RegisterLimit)
    HpSharkReferenceGpuKernel(HpSharkReferenceResults<SharkFloatParams> *SharkRestrict combo,
                              uint64_t *tempData)
{

    // Initialize cooperative grid group
    cg::grid_group grid = cg::this_grid();
    cg::thread_block block = cg::this_thread_block();

    // Call the AddHelper function
    constexpr auto currentIteration = 0;

    if constexpr (SharkFloatParams::EnablePeriodicity) {
        // This path is not supported: running one iteration with periodicity checking is pointless.
        // Correctness checking of all this should take place via the integrated loop version just below.
        return;
    } else {
        const auto [[maybe_unused]] shouldContinue = ReferenceHelper<SharkFloatParams>(
            grid, block, currentIteration, nullptr, nullptr, nullptr, nullptr, combo, tempData);
    }
}

template <class SharkFloatParams>
__global__ void
__maxnreg__(HpShark::RegisterLimit)
    HpSharkReferenceGpuLoop(HpSharkReferenceResults<SharkFloatParams> *combo,
                            uint64_t *tempData)
{

    // Initialize cooperative grid group
    cg::grid_group grid = cg::this_grid();
    cg::thread_block block = cg::this_thread_block();

    //typename SharkFloatParams::Float dzdcX{1};
    //typename SharkFloatParams::Float dzdcY{0};
    auto *dzdcX = &combo->dzdcX;
    auto *dzdcY = &combo->dzdcY;

    typename SharkFloatParams::Float cx_cast = combo->Add.C_A.ToHDRFloat<SharkFloatParams::SubType>(0);
    typename SharkFloatParams::Float cy_cast = combo->Add.E_B.ToHDRFloat<SharkFloatParams::SubType>(0);

    // Erase this global debug state if needed.
    if constexpr (HpShark::DebugGlobalState) {
        constexpr auto DebugGlobals_offset = HpShark::AdditionalGlobalSyncSpace;
        // constexpr auto DebugChecksum_offset = DebugGlobals_offset +
        // AdditionalGlobalDebugPerThread;

        // auto *SharkRestrict debugStates =
        //     reinterpret_cast<DebugState<SharkFloatParams> *>(&tempData[DebugChecksum_offset]);
        auto *SharkRestrict debugGlobalState =
            reinterpret_cast<DebugGlobalCount<SharkFloatParams> *>(&tempData[DebugGlobals_offset]);

        const auto CurBlock = block.group_index().x;
        const auto CurThread = block.thread_index().x;
        debugGlobalState[CurBlock * block.dim_threads().x + CurThread].DebugMultiplyErase();
    }

    // MaxRuntimeIters had better be <= HpSharkReferenceResults<SharkFloatParams>::MaxOutputIters
    for (uint64_t i = 0; i < combo->MaxRuntimeIters; ++i) {
        const auto shouldContinue =
            ReferenceHelper(grid, block, i, &cx_cast, &cy_cast, dzdcX, dzdcY, combo, tempData);

        if (shouldContinue != PeriodicityResult::Continue) {
            break;
        }
    }
}

template <class SharkFloatParams>
void
ComputeHpSharkReferenceGpuLoop(const HpShark::LaunchParams &launchParams,
                               cudaStream_t &stream,
                               void *kernelArgs[])
{

    constexpr auto SharedMemSize = HpShark::CalculateNTTSharedMemorySize<SharkFloatParams>();

    if constexpr (HpShark::CustomStream) {
        cudaFuncSetAttribute(HpSharkReferenceGpuLoop<SharkFloatParams>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             SharedMemSize);

        if (SharkVerbose == VerboseMode::Debug) {
            PrintMaxActiveBlocks<SharkFloatParams>(
                launchParams, HpSharkReferenceGpuLoop<SharkFloatParams>, SharedMemSize);
        }
    }

    cudaError_t err = cudaLaunchCooperativeKernel((void *)HpSharkReferenceGpuLoop<SharkFloatParams>,
                                                  dim3(launchParams.NumBlocks),
                                                  dim3(launchParams.ThreadsPerBlock),
                                                  kernelArgs,
                                                  SharedMemSize,
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
    template void ComputeHpSharkReferenceGpuLoop<SharkFloatParams>(                                     \
        const HpShark::LaunchParams &launchParams, cudaStream_t &stream, void *kernelArgs[]);

ExplicitInstantiateAll();
