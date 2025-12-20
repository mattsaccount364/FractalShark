#include "DbgHeap.h"
#include "KernelInvoke.h"
#include "KernelInvokeInternal.h"

namespace HpShark {

template <class SharkFloatParams>
void
InvokeAddKernelCorrectness(const HpShark::LaunchParams &launchParams,
                           BenchmarkTimer &timer,
                           HpSharkAddComboResults<SharkFloatParams> &combo,
                           DebugGpuCombo *debugCombo)
{

    // Perform the calculation on the GPU
    HpSharkAddComboResults<SharkFloatParams> *comboResults;
    cudaMalloc(&comboResults, sizeof(HpSharkAddComboResults<SharkFloatParams>));
    cudaMemcpy(
        comboResults, &combo, sizeof(HpSharkAddComboResults<SharkFloatParams>), cudaMemcpyHostToDevice);

    if constexpr (!HpShark::TestInitCudaMemory) {
        cudaMemset(&comboResults->Result1_A_B_C, 0, sizeof(HpSharkFloat<SharkFloatParams>));
        cudaMemset(&comboResults->Result2_D_E, 0, sizeof(HpSharkFloat<SharkFloatParams>));
    } else {
        cudaMemset(&comboResults->Result1_A_B_C, 0xCD, sizeof(HpSharkFloat<SharkFloatParams>));
        cudaMemset(&comboResults->Result2_D_E, 0xCD, sizeof(HpSharkFloat<SharkFloatParams>));
    }

    constexpr auto BytesToAllocate =
        (HpShark::AdditionalUInt64Global + HpShark::CalculateAddFrameSize<SharkFloatParams>()) *
        sizeof(uint64_t);
    uint64_t *g_extResult;
    cudaMalloc(&g_extResult, BytesToAllocate);

    // Prepare kernel arguments
    void *kernelArgs[] = {(void *)&comboResults, (void *)&g_extResult};

    {
        ScopedBenchmarkStopper stopper{timer};
        ComputeAddGpu<SharkFloatParams>(launchParams, kernelArgs);
    }

    cudaMemcpy(
        &combo, comboResults, sizeof(HpSharkAddComboResults<SharkFloatParams>), cudaMemcpyDeviceToHost);

    if (debugCombo != nullptr) {
        if constexpr (HpShark::DebugChecksums) {
            debugCombo->States.resize(SharkFloatParams::NumDebugStates);
            cudaMemcpy(debugCombo->States.data(),
                       &g_extResult[HpShark::AdditionalChecksumsOffset],
                       SharkFloatParams::NumDebugStates * sizeof(DebugStateRaw),
                       cudaMemcpyDeviceToHost);
        }

        if constexpr (HpShark::DebugGlobalState) {
            debugCombo->MultiplyCounts.resize(SharkFloatParams::NumDebugMultiplyCounts);
            cudaMemcpy(debugCombo->MultiplyCounts.data(),
                       &g_extResult[HpShark::AdditionalMultipliesOffset],
                       SharkFloatParams::NumDebugMultiplyCounts * sizeof(DebugGlobalCountRaw),
                       cudaMemcpyDeviceToHost);
        }
    }

    cudaFree(g_extResult);
    cudaFree(comboResults);
}

#define ExplicitlyInstantiateAdd(SharkFloatParams)                                                      \
    template void InvokeAddKernelCorrectness<SharkFloatParams>(                                         \
        const HpShark::LaunchParams &launchParams,                                                      \
        BenchmarkTimer &timer,                                                                          \
        HpSharkAddComboResults<SharkFloatParams> &combo,                                                \
        DebugGpuCombo *debugCombo);

#define ExplicitlyInstantiate(SharkFloatParams) ExplicitlyInstantiateAdd(SharkFloatParams)

ExplicitInstantiateAll();

} // namespace HpShark