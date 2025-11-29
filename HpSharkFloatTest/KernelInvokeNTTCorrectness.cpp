#include "DbgHeap.h"
#include "KernelInvoke.cuh"
#include "KernelInvokeInternal.cuh"

template <class SharkFloatParams>
void
InvokeMultiplyNTTKernelCorrectness(const SharkLaunchParams &launchParams,
                                   BenchmarkTimer &timer,
                                   HpSharkComboResults<SharkFloatParams> &combo,
                                   DebugGpuCombo *debugCombo)
{

    // Prepare kernel arguments
    // Allocate memory for carryOuts and cumulativeCarries
    uint64_t *d_tempProducts;
    constexpr auto BytesToAllocate =
        (AdditionalUInt64Global + CalculateNTTFrameSize<SharkFloatParams>()) * sizeof(uint64_t);
    std::cout << " Allocating " << BytesToAllocate << " bytes for d_tempProducts " << std::endl;
    cudaMalloc(&d_tempProducts, BytesToAllocate);

    if constexpr (!HpShark::TestInitCudaMemory) {
        cudaMemset(d_tempProducts, 0, BytesToAllocate);
    } else {
        cudaMemset(d_tempProducts, 0xCD, BytesToAllocate);
    }

    HpSharkComboResults<SharkFloatParams> *comboGpu;
    cudaMalloc(&comboGpu, sizeof(HpSharkComboResults<SharkFloatParams>));
    cudaMemcpy(comboGpu, &combo, sizeof(HpSharkComboResults<SharkFloatParams>), cudaMemcpyHostToDevice);

    {
        SharkNTT::RootTables NTTRoots;
        SharkNTT::BuildRoots<SharkFloatParams>(
            SharkFloatParams::NTTPlan.N, SharkFloatParams::NTTPlan.stages, NTTRoots);

        CopyRootsToCuda<SharkFloatParams>(comboGpu->Roots, NTTRoots);
        SharkNTT::DestroyRoots<SharkFloatParams>(false, NTTRoots);
    }

    if constexpr (!HpShark::TestInitCudaMemory) {
        cudaMemset(&comboGpu->ResultX2, 0, sizeof(HpSharkFloat<SharkFloatParams>));
        cudaMemset(&comboGpu->Result2XY, 0, sizeof(HpSharkFloat<SharkFloatParams>));
        cudaMemset(&comboGpu->ResultY2, 0, sizeof(HpSharkFloat<SharkFloatParams>));
    } else {
        cudaMemset(&comboGpu->ResultX2, 0xCD, sizeof(HpSharkFloat<SharkFloatParams>));
        cudaMemset(&comboGpu->Result2XY, 0xCD, sizeof(HpSharkFloat<SharkFloatParams>));
        cudaMemset(&comboGpu->ResultY2, 0xCD, sizeof(HpSharkFloat<SharkFloatParams>));
    }

    void *kernelArgs[] = {(void *)&comboGpu, (void *)&d_tempProducts};

    {
        ScopedBenchmarkStopper stopper{timer};
        ComputeMultiplyNTTGpu<SharkFloatParams>(launchParams, kernelArgs);
    }

    cudaMemcpy(&combo, comboGpu, sizeof(HpSharkComboResults<SharkFloatParams>), cudaMemcpyDeviceToHost);

    if (debugCombo != nullptr) {
        if constexpr (HpShark::DebugChecksums) {
            debugCombo->States.resize(SharkFloatParams::NumDebugStates);
            cudaMemcpy(debugCombo->States.data(),
                       &d_tempProducts[AdditionalChecksumsOffset],
                       SharkFloatParams::NumDebugStates * sizeof(DebugStateRaw),
                       cudaMemcpyDeviceToHost);
        }

        if constexpr (HpShark::DebugGlobalState) {
            debugCombo->MultiplyCounts.resize(SharkFloatParams::NumDebugMultiplyCounts);
            cudaMemcpy(debugCombo->MultiplyCounts.data(),
                       &d_tempProducts[AdditionalMultipliesOffset],
                       SharkFloatParams::NumDebugMultiplyCounts * sizeof(DebugGlobalCountRaw),
                       cudaMemcpyDeviceToHost);
        }
    }

    SharkNTT::DestroyRoots<SharkFloatParams>(true, comboGpu->Roots);

    cudaFree(comboGpu);
    cudaFree(d_tempProducts);
}

#ifdef ENABLE_MULTIPLY_NTT_KERNEL
#define ExplicitlyInstantiateMultiplyNTT(SharkFloatParams)                                              \
    template void InvokeMultiplyNTTKernelCorrectness<SharkFloatParams>(                                 \
        const SharkLaunchParams &launchParams,                                                         \
        BenchmarkTimer & timer,                                                                         \
        HpSharkComboResults<SharkFloatParams> & combo,                                                  \
        DebugGpuCombo * debugCombo);
#else
#define ExplicitlyInstantiateMultiplyNTT(SharkFloatParams) ;
#endif

#define ExplicitlyInstantiate(SharkFloatParams) ExplicitlyInstantiateMultiplyNTT(SharkFloatParams)

ExplicitInstantiateAll();
