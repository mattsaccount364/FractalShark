#include "DbgHeap.h"
#include "KernelInvoke.h"
#include "KernelInvokeInternal.h"

namespace HpShark {

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
    // Match TestPerf-style invocation, but correctness assumes exactly one iteration.
    constexpr uint64_t kNumIters = 1;

    // ---------------------------------------------------------------------
    // Allocate temp scratch (TestPerf-style sizing: NTT frame).
    // ---------------------------------------------------------------------
    constexpr size_t BytesToAllocate =
        (HpShark::AdditionalUInt64Global + HpShark::CalculateNTTFrameSize<SharkFloatParams>()) *
        sizeof(uint64_t);

    cudaMalloc(&combo.d_tempProducts, BytesToAllocate);

    if constexpr (!HpShark::TestInitCudaMemory) {
        cudaMemset(combo.d_tempProducts, 0, BytesToAllocate);
    } else {
        cudaMemset(combo.d_tempProducts, 0xCD, BytesToAllocate);
    }

    // ---------------------------------------------------------------------
    // Allocate + shallow-copy combo to device (TestPerf style).
    // ---------------------------------------------------------------------
    cudaMalloc(&combo.comboGpu, sizeof(HpSharkReferenceResults<SharkFloatParams>));

    // Note: shallow copy; we will memset specific members below (same idea as TestPerf).
    cudaMemcpy(combo.comboGpu,
               &combo,
               sizeof(HpSharkReferenceResults<SharkFloatParams>),
               cudaMemcpyHostToDevice);

    // Host-only kernel arg staging (same convention as TestPerf).
    combo.kernelArgs[0] = (void *)&combo.comboGpu;
    combo.kernelArgs[1] = (void *)&combo.d_tempProducts;

    // Correctness path doesn't need a custom stream; keep stream = 0 like default.
    combo.stream = 0;
    static_assert(sizeof(cudaStream_t) == sizeof(combo.stream),
                  "cudaStream_t size mismatch with combo.stream");

    uint8_t byteToSet = HpShark::TestInitCudaMemory ? 0xCD : 0;

    // Clear result fields (keep behavior consistent with existing correctness code).
    cudaMemset(&combo.comboGpu->Add.A_X2, byteToSet, sizeof(HpSharkFloat<SharkFloatParams>));
    cudaMemset(&combo.comboGpu->Add.B_Y2, byteToSet, sizeof(HpSharkFloat<SharkFloatParams>));
    cudaMemset(&combo.comboGpu->Add.D_2X, byteToSet, sizeof(HpSharkFloat<SharkFloatParams>));
    cudaMemset(&combo.comboGpu->Add.Result1_A_B_C, byteToSet, sizeof(HpSharkFloat<SharkFloatParams>));
    cudaMemset(&combo.comboGpu->Add.Result2_D_E, byteToSet, sizeof(HpSharkFloat<SharkFloatParams>));
    cudaMemset(&combo.comboGpu->Multiply.ResultX2, byteToSet, sizeof(HpSharkFloat<SharkFloatParams>));
    cudaMemset(&combo.comboGpu->Multiply.Result2XY, byteToSet, sizeof(HpSharkFloat<SharkFloatParams>));
    cudaMemset(&combo.comboGpu->Multiply.ResultY2, byteToSet, sizeof(HpSharkFloat<SharkFloatParams>));

    // For correctness, the iter counter should start at 0 deterministically.
    {
        const uint64_t zero = 0;
        cudaMemcpy(&combo.comboGpu->OutputIterCount, &zero, sizeof(uint64_t), cudaMemcpyHostToDevice);
    }

    // ---------------------------------------------------------------------
    // Roots: build on host, copy to device, destroy host roots (same lifecycle as today).
    // ---------------------------------------------------------------------
    {
        SharkNTT::RootTables NTTRoots;
        SharkNTT::BuildRoots<SharkFloatParams>(
            SharkFloatParams::NTTPlan.N, SharkFloatParams::NTTPlan.stages, NTTRoots);

        CopyRootsToCuda<SharkFloatParams>(combo.comboGpu->Multiply.Roots, NTTRoots);
        SharkNTT::DestroyRoots<SharkFloatParams>(false, NTTRoots);
    }

    // ---------------------------------------------------------------------
    // One-iteration loop-style launch (TestPerf-style kernel entry point).
    // ---------------------------------------------------------------------
    cudaMemcpy(&combo.comboGpu->MaxRuntimeIters, &kNumIters, sizeof(uint64_t), cudaMemcpyHostToDevice);

    void *kernelArgs[] = {(void *)&combo.comboGpu, (void *)&combo.d_tempProducts};

    {
        ScopedBenchmarkStopper stopper{timer};
        ComputeHpSharkReferenceGpuLoop<SharkFloatParams>(
            launchParams, *reinterpret_cast<cudaStream_t *>(&combo.stream), kernelArgs);
    }

    // Copy everything back (device pointer -> host struct).
    cudaMemcpy(&combo,
               combo.comboGpu,
               sizeof(HpSharkReferenceResults<SharkFloatParams>),
               cudaMemcpyDeviceToHost);

    // ---------------------------------------------------------------------
    // Optional debug readback (keep the correctness behavior).
    // ---------------------------------------------------------------------
    if (debugCombo != nullptr) {
        if constexpr (HpShark::DebugChecksums) {
            debugCombo->States.resize(SharkFloatParams::NumDebugStates);
            cudaMemcpy(debugCombo->States.data(),
                       &combo.d_tempProducts[HpShark::AdditionalChecksumsOffset],
                       SharkFloatParams::NumDebugStates * sizeof(DebugStateRaw),
                       cudaMemcpyDeviceToHost);
        }

        if constexpr (HpShark::DebugGlobalState) {
            debugCombo->MultiplyCounts.resize(SharkFloatParams::NumDebugMultiplyCounts);
            cudaMemcpy(debugCombo->MultiplyCounts.data(),
                       &combo.d_tempProducts[HpShark::AdditionalMultipliesOffset],
                       SharkFloatParams::NumDebugMultiplyCounts * sizeof(DebugGlobalCountRaw),
                       cudaMemcpyDeviceToHost);
        }
    }

    // Roots were device-allocated in CopyRootsToCuda; destroy them like correctness does.
    SharkNTT::DestroyRoots<SharkFloatParams>(true, combo.comboGpu->Multiply.Roots);

    cudaFree(combo.comboGpu);
    cudaFree(combo.d_tempProducts);

    combo.comboGpu = nullptr;
    combo.d_tempProducts = nullptr;
}


#if defined(ENABLE_FULL_KERNEL)
#define ExplicitlyInstantiateHpSharkReference(SharkFloatParams)                                         \
    template void InvokeHpSharkReferenceKernelCorrectness<SharkFloatParams>(                            \
        const HpShark::LaunchParams &launchParams,                                                      \
        BenchmarkTimer &timer,                                                                          \
        HpSharkReferenceResults<SharkFloatParams> &combo,                                               \
        DebugGpuCombo *debugCombo);
#else
#define ExplicitlyInstantiateHpSharkReference(SharkFloatParams) ;
#endif

#define ExplicitlyInstantiate(SharkFloatParams) ExplicitlyInstantiateHpSharkReference(SharkFloatParams)

ExplicitInstantiateAll();

} // namespace HpShark