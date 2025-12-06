#include "DbgHeap.h"
#include "KernelInvoke.cuh"
#include "KernelInvokeInternal.cuh"

//
// Note: This test ignores the period because it executes only one iteration.
//

template <class SharkFloatParams>
void
InvokeHpSharkReferenceKernelCorrectness(const HpShark::LaunchParams &launchParams,
                                        BenchmarkTimer &timer,
                                        HpSharkReferenceResults<SharkFloatParams> &combo,
                                        DebugGpuCombo *debugCombo)
{

    // Prepare kernel arguments
    // Allocate memory for carryOuts and cumulativeCarries

    // TODO max of add/multiply frame size
    // TODO checksum handled
    uint64_t *d_tempProducts;
    constexpr auto BytesToAllocate =
        (HpShark::AdditionalUInt64Global +
         HpShark::ScratchMemoryCopies * HpShark::CalculateKaratsubaFrameSize<SharkFloatParams>()) *
        sizeof(uint64_t);
    cudaMalloc(&d_tempProducts, BytesToAllocate);

    if constexpr (!HpShark::TestInitCudaMemory) {
        cudaMemset(d_tempProducts, 0, BytesToAllocate);
    } else {
        cudaMemset(d_tempProducts, 0xCD, BytesToAllocate);
    }

    HpSharkReferenceResults<SharkFloatParams> *comboGpu;
    cudaMalloc(&comboGpu, sizeof(HpSharkReferenceResults<SharkFloatParams>));

    uint8_t byteToSet = HpShark::TestInitCudaMemory ? 0xCD : 0;

    cudaMemcpy(
        comboGpu, &combo, sizeof(HpSharkReferenceResults<SharkFloatParams>), cudaMemcpyHostToDevice);

    cudaMemcpy(&comboGpu->RadiusY,
               &combo.RadiusY,
               sizeof(HDRFloat<typename SharkFloatParams::SubType>),
               cudaMemcpyHostToDevice);

    cudaMemset(&comboGpu->Add.A_X2, byteToSet, sizeof(HpSharkFloat<SharkFloatParams>));
    cudaMemset(&comboGpu->Add.B_Y2, byteToSet, sizeof(HpSharkFloat<SharkFloatParams>));
    cudaMemset(&comboGpu->Add.D_2X, byteToSet, sizeof(HpSharkFloat<SharkFloatParams>));
    cudaMemset(&comboGpu->Add.Result1_A_B_C, byteToSet, sizeof(HpSharkFloat<SharkFloatParams>));
    cudaMemset(&comboGpu->Add.Result2_D_E, byteToSet, sizeof(HpSharkFloat<SharkFloatParams>));
    cudaMemset(&comboGpu->Multiply.ResultX2, byteToSet, sizeof(HpSharkFloat<SharkFloatParams>));
    cudaMemset(&comboGpu->Multiply.Result2XY, byteToSet, sizeof(HpSharkFloat<SharkFloatParams>));
    cudaMemset(&comboGpu->Multiply.ResultY2, byteToSet, sizeof(HpSharkFloat<SharkFloatParams>));
    cudaMemset(&comboGpu->Period, byteToSet, sizeof(uint64_t));
    cudaMemset(&comboGpu->EscapedIteration, byteToSet, sizeof(uint64_t));

    // Build NTT plan + roots exactly like correctness path
    {
        SharkNTT::RootTables NTTRoots;
        SharkNTT::BuildRoots<SharkFloatParams>(
            SharkFloatParams::NTTPlan.N, SharkFloatParams::NTTPlan.stages, NTTRoots);

        CopyRootsToCuda<SharkFloatParams>(comboGpu->Multiply.Roots, NTTRoots);
        SharkNTT::DestroyRoots<SharkFloatParams>(false, NTTRoots);
    }

    void *kernelArgs[] = {(void *)&comboGpu, (void *)&d_tempProducts};

    {
        ScopedBenchmarkStopper stopper{timer};
        ComputeHpSharkReferenceGpu<SharkFloatParams>(launchParams, kernelArgs);
    }

    cudaMemcpy(
        &combo, comboGpu, sizeof(HpSharkReferenceResults<SharkFloatParams>), cudaMemcpyDeviceToHost);

    if (debugCombo != nullptr) {
        if constexpr (HpShark::DebugChecksums) {
            debugCombo->States.resize(SharkFloatParams::NumDebugStates);
            cudaMemcpy(debugCombo->States.data(),
                       &d_tempProducts[HpShark::AdditionalChecksumsOffset],
                       SharkFloatParams::NumDebugStates * sizeof(DebugStateRaw),
                       cudaMemcpyDeviceToHost);
        }

        if constexpr (HpShark::DebugGlobalState) {
            debugCombo->MultiplyCounts.resize(SharkFloatParams::NumDebugMultiplyCounts);
            cudaMemcpy(debugCombo->MultiplyCounts.data(),
                       &d_tempProducts[HpShark::AdditionalMultipliesOffset],
                       SharkFloatParams::NumDebugMultiplyCounts * sizeof(DebugGlobalCountRaw),
                       cudaMemcpyDeviceToHost);
        }
    }

    // Roots were device-allocated in CopyRootsToCuda; destroy like correctness does
    SharkNTT::DestroyRoots<SharkFloatParams>(true, comboGpu->Multiply.Roots);

    cudaFree(comboGpu);
    cudaFree(d_tempProducts);
}

#if defined(ENABLE_REFERENCE_KERNEL) || defined(ENABLE_FULL_KERNEL)
#define ExplicitlyInstantiateHpSharkReference(SharkFloatParams)                                         \
    template void InvokeHpSharkReferenceKernelCorrectness<SharkFloatParams>(                            \
        const HpShark::LaunchParams &launchParams,                                                        \
        BenchmarkTimer & timer,                                                                         \
        HpSharkReferenceResults<SharkFloatParams> & combo,                                              \
        DebugGpuCombo * debugCombo);
#else
#define ExplicitlyInstantiateHpSharkReference(SharkFloatParams) ;
#endif

#define ExplicitlyInstantiate(SharkFloatParams) ExplicitlyInstantiateHpSharkReference(SharkFloatParams)

ExplicitInstantiateAll();
