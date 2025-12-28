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
    HpSharkAddComboResults<SharkFloatParams> *comboResults = nullptr;
    cudaError_t err = cudaMalloc(reinterpret_cast<void **>(&comboResults),
                                 sizeof(HpSharkAddComboResults<SharkFloatParams>));
    if (err != cudaSuccess) {
        std::ostringstream oss;
        oss << "cudaMalloc(comboResults) failed: " << cudaGetErrorString(err) << " (code "
            << static_cast<int>(err) << ")";
        throw std::runtime_error(oss.str());
    }

    err = cudaMemcpy(
        comboResults, &combo, sizeof(HpSharkAddComboResults<SharkFloatParams>), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(comboResults);
        std::ostringstream oss;
        oss << "cudaMemcpy(combo -> comboResults) failed: " << cudaGetErrorString(err) << " (code "
            << static_cast<int>(err) << ")";
        throw std::runtime_error(oss.str());
    }

    const int initByte = HpShark::TestInitCudaMemory ? 0xCD : 0x00;

    err = cudaMemset(&comboResults->Result1_A_B_C, initByte, sizeof(HpSharkFloat<SharkFloatParams>));
    if (err != cudaSuccess) {
        cudaFree(comboResults);
        std::ostringstream oss;
        oss << "cudaMemset(Result1_A_B_C) failed: " << cudaGetErrorString(err) << " (code "
            << static_cast<int>(err) << ")";
        throw std::runtime_error(oss.str());
    }

    err = cudaMemset(&comboResults->Result2_D_E, initByte, sizeof(HpSharkFloat<SharkFloatParams>));
    if (err != cudaSuccess) {
        cudaFree(comboResults);
        std::ostringstream oss;
        oss << "cudaMemset(Result2_D_E) failed: " << cudaGetErrorString(err) << " (code "
            << static_cast<int>(err) << ")";
        throw std::runtime_error(oss.str());
    }

    constexpr auto BytesToAllocate =
        (HpShark::AdditionalUInt64Global + HpShark::CalculateAddFrameSize<SharkFloatParams>()) *
        sizeof(uint64_t);

    uint64_t *g_extResult = nullptr;
    err = cudaMalloc(reinterpret_cast<void **>(&g_extResult), BytesToAllocate);
    if (err != cudaSuccess) {
        cudaFree(comboResults);
        std::ostringstream oss;
        oss << "cudaMalloc(g_extResult) failed: " << cudaGetErrorString(err) << " (code "
            << static_cast<int>(err) << ")";
        throw std::runtime_error(oss.str());
    }

    // Prepare kernel arguments
    void *kernelArgs[] = {(void *)&comboResults, (void *)&g_extResult};

    {
        ScopedBenchmarkStopper stopper{timer};
        ComputeAddGpu<SharkFloatParams>(launchParams, kernelArgs);
    }

    err = cudaMemcpy(
        &combo, comboResults, sizeof(HpSharkAddComboResults<SharkFloatParams>), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        cudaFree(g_extResult);
        cudaFree(comboResults);
        std::ostringstream oss;
        oss << "cudaMemcpy(comboResults -> combo) failed: " << cudaGetErrorString(err) << " (code "
            << static_cast<int>(err) << ")";
        throw std::runtime_error(oss.str());
    }

    if (debugCombo != nullptr) {
        if constexpr (HpShark::DebugChecksums) {
            debugCombo->States.resize(SharkFloatParams::NumDebugStates);
            err = cudaMemcpy(debugCombo->States.data(),
                             &g_extResult[HpShark::AdditionalChecksumsOffset],
                             SharkFloatParams::NumDebugStates * sizeof(DebugStateRaw),
                             cudaMemcpyDeviceToHost);
            if (err != cudaSuccess) {
                cudaFree(g_extResult);
                cudaFree(comboResults);
                std::ostringstream oss;
                oss << "cudaMemcpy(DebugChecksums) failed: " << cudaGetErrorString(err) << " (code "
                    << static_cast<int>(err) << ")";
                throw std::runtime_error(oss.str());
            }
        }

        if constexpr (HpShark::DebugGlobalState) {
            debugCombo->MultiplyCounts.resize(SharkFloatParams::NumDebugMultiplyCounts);
            err = cudaMemcpy(debugCombo->MultiplyCounts.data(),
                             &g_extResult[HpShark::AdditionalMultipliesOffset],
                             SharkFloatParams::NumDebugMultiplyCounts * sizeof(DebugGlobalCountRaw),
                             cudaMemcpyDeviceToHost);
            if (err != cudaSuccess) {
                cudaFree(g_extResult);
                cudaFree(comboResults);
                std::ostringstream oss;
                oss << "cudaMemcpy(DebugGlobalState) failed: " << cudaGetErrorString(err) << " (code "
                    << static_cast<int>(err) << ")";
                throw std::runtime_error(oss.str());
            }
        }
    }

    err = cudaFree(g_extResult);
    if (err != cudaSuccess) {
        cudaFree(comboResults);
        std::ostringstream oss;
        oss << "cudaFree(g_extResult) failed: " << cudaGetErrorString(err) << " (code "
            << static_cast<int>(err) << ")";
        throw std::runtime_error(oss.str());
    }

    err = cudaFree(comboResults);
    if (err != cudaSuccess) {
        std::ostringstream oss;
        oss << "cudaFree(comboResults) failed: " << cudaGetErrorString(err) << " (code "
            << static_cast<int>(err) << ")";
        throw std::runtime_error(oss.str());
    }
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