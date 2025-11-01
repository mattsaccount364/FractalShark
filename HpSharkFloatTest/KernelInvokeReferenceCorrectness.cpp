#include "DbgHeap.h"
#include "KernelInvoke.cuh"
#include "KernelInvokeInternal.cuh"

//
// Note: This test ignores the period because it executes only one iteration.
//

template <class SharkFloatParams>
void
InvokeHpSharkReferenceKernelCorrectness(BenchmarkTimer &timer,
                                        HpSharkReferenceResults<SharkFloatParams> &combo,
                                        DebugGpuCombo *debugCombo)
{

    // Prepare kernel arguments
    // Allocate memory for carryOuts and cumulativeCarries

    // TODO max of add/multiply frame size
    // TODO checksum handled
    uint64_t *d_tempProducts;
    constexpr auto BytesToAllocate =
        (AdditionalUInt64Global +
         ScratchMemoryCopies * CalculateKaratsubaFrameSize<SharkFloatParams>()) *
        sizeof(uint64_t);
    cudaMalloc(&d_tempProducts, BytesToAllocate);

    if constexpr (!SharkTestInitCudaMemory) {
        cudaMemset(d_tempProducts, 0, BytesToAllocate);
    } else {
        cudaMemset(d_tempProducts, 0xCD, BytesToAllocate);
    }

    HpSharkReferenceResults<SharkFloatParams> *comboGpu;
    cudaMalloc(&comboGpu, sizeof(HpSharkReferenceResults<SharkFloatParams>));

    uint8_t byteToSet = SharkTestInitCudaMemory ? 0xCD : 0;

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
        ComputeHpSharkReferenceGpu<SharkFloatParams>(kernelArgs);
    }

    cudaMemcpy(
        &combo, comboGpu, sizeof(HpSharkReferenceResults<SharkFloatParams>), cudaMemcpyDeviceToHost);

    if (debugCombo != nullptr) {
        if constexpr (SharkDebugChecksums) {
            debugCombo->States.resize(SharkFloatParams::NumDebugStates);
            cudaMemcpy(debugCombo->States.data(),
                       &d_tempProducts[AdditionalChecksumsOffset],
                       SharkFloatParams::NumDebugStates * sizeof(DebugStateRaw),
                       cudaMemcpyDeviceToHost);
        }

        if constexpr (SharkPrintMultiplyCounts) {
            debugCombo->MultiplyCounts.resize(SharkFloatParams::NumDebugMultiplyCounts);
            cudaMemcpy(debugCombo->MultiplyCounts.data(),
                       &d_tempProducts[AdditionalMultipliesOffset],
                       SharkFloatParams::NumDebugMultiplyCounts * sizeof(DebugMultiplyCountRaw),
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
        BenchmarkTimer & timer,                                                                         \
        HpSharkReferenceResults<SharkFloatParams> & combo,                                              \
        DebugGpuCombo * debugCombo);
#else
#define ExplicitlyInstantiateHpSharkReference(SharkFloatParams) ;
#endif

#define ExplicitlyInstantiate(SharkFloatParams) ExplicitlyInstantiateHpSharkReference(SharkFloatParams)

ExplicitInstantiateAll();
