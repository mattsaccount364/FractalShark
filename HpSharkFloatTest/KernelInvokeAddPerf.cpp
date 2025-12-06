#include "DbgHeap.h"
#include "KernelInvoke.h"
#include "KernelInvokeInternal.h"

template <class SharkFloatParams>
void
InvokeAddKernelPerf(const HpShark::LaunchParams &launchParams,
                    BenchmarkTimer &timer,
                    HpSharkAddComboResults<SharkFloatParams> &combo,
                    uint64_t numIters)
{

    // Perform the calculation on the GPU
    HpSharkAddComboResults<SharkFloatParams> *comboResults;
    cudaMalloc(&comboResults, sizeof(HpSharkAddComboResults<SharkFloatParams>));
    cudaMemcpy(
        comboResults, &combo, sizeof(HpSharkAddComboResults<SharkFloatParams>), cudaMemcpyHostToDevice);

    constexpr auto BytesToAllocate =
        (HpShark::AdditionalUInt64Global + HpShark::CalculateAddFrameSize<SharkFloatParams>()) *
        sizeof(uint64_t);
    uint64_t *g_extResult;
    cudaMalloc(&g_extResult, BytesToAllocate);

    // Prepare kernel arguments
    void *kernelArgs[] = {(void *)&comboResults, (void *)&numIters, (void *)&g_extResult};

    // Launch the cooperative kernel
    {
        ScopedBenchmarkStopper stopper{timer};
        ComputeAddGpuTestLoop<SharkFloatParams>(launchParams, kernelArgs);
    }

    cudaMemcpy(
        &combo, comboResults, sizeof(HpSharkAddComboResults<SharkFloatParams>), cudaMemcpyDeviceToHost);

    // Free memory
    cudaFree(g_extResult);
    cudaFree(comboResults);
}

#ifdef ENABLE_ADD_KERNEL
#define ExplicitlyInstantiateAdd(SharkFloatParams)                                                      \
    template void InvokeAddKernelPerf<SharkFloatParams>(                                                \
        const HpShark::LaunchParams &launchParams,                                                      \
        BenchmarkTimer &timer,                                                                          \
        HpSharkAddComboResults<SharkFloatParams> &combo,                                                \
        uint64_t numIters);
#else
#define ExplicitlyInstantiateAdd(SharkFloatParams) ;
#endif

#define ExplicitlyInstantiate(SharkFloatParams) ExplicitlyInstantiateAdd(SharkFloatParams)

ExplicitInstantiateAll();
