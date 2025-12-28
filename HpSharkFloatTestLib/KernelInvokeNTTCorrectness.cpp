#include "DbgHeap.h"
#include "KernelInvoke.h"
#include "KernelInvokeInternal.h"
#include "TestVerbose.h"
#include <sstream>
#include <stdexcept>

namespace HpShark {

template <class SharkFloatParams>
void
InvokeMultiplyNTTKernelCorrectness(const HpShark::LaunchParams &launchParams,
                                   BenchmarkTimer &timer,
                                   HpSharkComboResults<SharkFloatParams> &combo,
                                   DebugGpuCombo *debugCombo)
{
    // Prepare kernel arguments
    // Allocate memory for carryOuts and cumulativeCarries
    uint64_t *d_tempProducts;
    constexpr auto BytesToAllocate =
        (HpShark::AdditionalUInt64Global + HpShark::CalculateNTTFrameSize<SharkFloatParams>()) *
        sizeof(uint64_t);

    if (SharkVerbose == VerboseMode::Debug) {
        std::cout << " Allocating " << BytesToAllocate << " bytes for d_tempProducts " << std::endl;
    }

    cudaError_t err = cudaSuccess;
    err = cudaMalloc(&d_tempProducts, BytesToAllocate);
    if (err != cudaSuccess) {
        std::ostringstream oss;
        oss << "cudaMalloc(&d_tempProducts, BytesToAllocate) failed: " << cudaGetErrorString(err)
            << " (code " << static_cast<int>(err) << ")";
        throw std::runtime_error(oss.str());
    }

    if constexpr (!HpShark::TestInitCudaMemory) {
        cudaError_t err = cudaSuccess;
        err = cudaMemset(d_tempProducts, 0, BytesToAllocate);
        if (err != cudaSuccess) {
            std::ostringstream oss;
            oss << "cudaMemset(d_tempProducts, 0, BytesToAllocate) failed: " << cudaGetErrorString(err)
                << " (code " << static_cast<int>(err) << ")";
            throw std::runtime_error(oss.str());
        }
    } else {
        err = cudaMemset(d_tempProducts, 0xCD, BytesToAllocate);
        if (err != cudaSuccess) {
            std::ostringstream oss;
            oss << "cudaMemset(d_tempProducts, 0xCD, BytesToAllocate) failed: "
                << cudaGetErrorString(err) << " (code " << static_cast<int>(err) << ")";
            throw std::runtime_error(oss.str());
        }
    }

    HpSharkComboResults<SharkFloatParams> *comboGpu;
    err = cudaMalloc(&comboGpu, sizeof(HpSharkComboResults<SharkFloatParams>));
    if (err != cudaSuccess) {
        std::ostringstream oss;
        oss << "cudaMalloc(&comboGpu, sizeof(HpSharkComboResults<SharkFloatParams>)) failed: "
            << cudaGetErrorString(err) << " (code " << static_cast<int>(err) << ")";
        throw std::runtime_error(oss.str());
    }
    err = cudaMemcpy(
        comboGpu, &combo, sizeof(HpSharkComboResults<SharkFloatParams>), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::ostringstream oss;
        oss << "err = cudaMemcpy(comboGpu, &combo, sizeof(HpSharkComboResults<SharkFloatParams>), "
               "cudaMemcpyHostToDevice) failed: "
            << cudaGetErrorString(err) << " (code " << static_cast<int>(err) << ")";
        throw std::runtime_error(oss.str());
        if (err != cudaSuccess) {
            std::ostringstream oss;
            oss << "cudaMemcpy failed: " << cudaGetErrorString(err) << " (code " << static_cast<int>(err)
                << ")";
            throw std::runtime_error(oss.str());
        }
    }

    {
        SharkNTT::RootTables NTTRoots;
        SharkNTT::BuildRoots<SharkFloatParams>(
            SharkFloatParams::NTTPlan.N, SharkFloatParams::NTTPlan.stages, NTTRoots);

        CopyRootsToCuda<SharkFloatParams>(comboGpu->Roots, NTTRoots);
        SharkNTT::DestroyRoots<SharkFloatParams>(false, NTTRoots);
    }

    if constexpr (!HpShark::TestInitCudaMemory) {
        err = cudaMemset(&comboGpu->ResultX2, 0, sizeof(HpSharkFloat<SharkFloatParams>));
        if (err != cudaSuccess) {
            std::ostringstream oss;
            oss << "cudaMemset(&comboGpu->ResultX2, 0, sizeof(HpSharkFloat<SharkFloatParams>)) failed: "
                << cudaGetErrorString(err) << " (code " << static_cast<int>(err) << ")";
            throw std::runtime_error(oss.str());
        }
        err = cudaMemset(&comboGpu->Result2XY, 0, sizeof(HpSharkFloat<SharkFloatParams>));
        if (err != cudaSuccess) {
            std::ostringstream oss;
            oss << "cudaMemset(&comboGpu->Result2XY, 0, sizeof(HpSharkFloat<SharkFloatParams>)) failed: "
                << cudaGetErrorString(err) << " (code " << static_cast<int>(err) << ")";
            throw std::runtime_error(oss.str());
        }
        err = cudaMemset(&comboGpu->ResultY2, 0, sizeof(HpSharkFloat<SharkFloatParams>));
        if (err != cudaSuccess) {
            std::ostringstream oss;
            oss << "cudaMemset(&comboGpu->ResultY2, 0, sizeof(HpSharkFloat<SharkFloatParams>)) failed: "
                << cudaGetErrorString(err) << " (code " << static_cast<int>(err) << ")";
            throw std::runtime_error(oss.str());
        }
    } else {
        err = cudaMemset(&comboGpu->ResultX2, 0xCD, sizeof(HpSharkFloat<SharkFloatParams>));
        if (err != cudaSuccess) {
            std::ostringstream oss;
            oss << "cudaMemset(&comboGpu->ResultX2, 0xCD, sizeof(HpSharkFloat<SharkFloatParams>)) "
                   "failed: "
                << cudaGetErrorString(err) << " (code " << static_cast<int>(err) << ")";
            throw std::runtime_error(oss.str());
        }
        err = cudaMemset(&comboGpu->Result2XY, 0xCD, sizeof(HpSharkFloat<SharkFloatParams>));
        if (err != cudaSuccess) {
            std::ostringstream oss;
            oss << "cudaMemset(&comboGpu->Result2XY, 0xCD, sizeof(HpSharkFloat<SharkFloatParams>)) "
                   "failed: "
                << cudaGetErrorString(err) << " (code " << static_cast<int>(err) << ")";
            throw std::runtime_error(oss.str());
        }
        err = cudaMemset(&comboGpu->ResultY2, 0xCD, sizeof(HpSharkFloat<SharkFloatParams>));
        if (err != cudaSuccess) {
            std::ostringstream oss;
            oss << "cudaMemset(&comboGpu->ResultY2, 0xCD, sizeof(HpSharkFloat<SharkFloatParams>)) "
                   "failed: "
                << cudaGetErrorString(err) << " (code " << static_cast<int>(err) << ")";
            throw std::runtime_error(oss.str());
        }
    }

    void *kernelArgs[] = {(void *)&comboGpu, (void *)&d_tempProducts};

    {
        ScopedBenchmarkStopper stopper{timer};
        ComputeMultiplyNTTGpu<SharkFloatParams>(launchParams, kernelArgs);
    }

    err = cudaMemcpy(
        &combo, comboGpu, sizeof(HpSharkComboResults<SharkFloatParams>), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::ostringstream oss;
        oss << "err = cudaMemcpy(&combo, comboGpu, sizeof(HpSharkComboResults<SharkFloatParams>), "
               "cudaMemcpyDeviceToHost) failed: "
            << cudaGetErrorString(err) << " (code " << static_cast<int>(err) << ")";
        throw std::runtime_error(oss.str());
        if (err != cudaSuccess) {
            std::ostringstream oss;
            oss << "cudaMemcpy failed: " << cudaGetErrorString(err) << " (code " << static_cast<int>(err)
                << ")";
            throw std::runtime_error(oss.str());
        }
    }

    if (debugCombo != nullptr) {
        if constexpr (HpShark::DebugChecksums) {
            debugCombo->States.resize(SharkFloatParams::NumDebugStates);
            cudaError_t err = cudaSuccess;
            err = cudaMemcpy(debugCombo->States.data(),
                             &d_tempProducts[HpShark::AdditionalChecksumsOffset],
                             SharkFloatParams::NumDebugStates * sizeof(DebugStateRaw),
                             cudaMemcpyDeviceToHost);
            if (err != cudaSuccess) {
                std::ostringstream oss;
                oss << "cudaMemcpy failed: " << cudaGetErrorString(err) << " (code "
                    << static_cast<int>(err) << ")";
                throw std::runtime_error(oss.str());
            }
        }

        if constexpr (HpShark::DebugGlobalState) {
            debugCombo->MultiplyCounts.resize(SharkFloatParams::NumDebugMultiplyCounts);
            err = cudaMemcpy(debugCombo->MultiplyCounts.data(),
                             &d_tempProducts[HpShark::AdditionalMultipliesOffset],
                             SharkFloatParams::NumDebugMultiplyCounts * sizeof(DebugGlobalCountRaw),
                             cudaMemcpyDeviceToHost);
            if (err != cudaSuccess) {
                std::ostringstream oss;
                oss << "cudaMemcpy failed: " << cudaGetErrorString(err) << " (code "
                    << static_cast<int>(err) << ")";
                throw std::runtime_error(oss.str());
            }
        }
    }

    SharkNTT::DestroyRoots<SharkFloatParams>(true, comboGpu->Roots);

    err = cudaFree(comboGpu);
    if (err != cudaSuccess) {
        std::ostringstream oss;
        oss << "cudaFree(comboGpu) failed: " << cudaGetErrorString(err) << " (code "
            << static_cast<int>(err) << ")";
        throw std::runtime_error(oss.str());
    }
    err = cudaFree(d_tempProducts);
    if (err != cudaSuccess) {
        std::ostringstream oss;
        oss << "cudaFree(d_tempProducts) failed: " << cudaGetErrorString(err) << " (code "
            << static_cast<int>(err) << ")";
        throw std::runtime_error(oss.str());
    }
}

#define ExplicitlyInstantiateMultiplyNTT(SharkFloatParams)                                              \
    template void InvokeMultiplyNTTKernelCorrectness<SharkFloatParams>(                                 \
        const HpShark::LaunchParams &launchParams,                                                      \
        BenchmarkTimer &timer,                                                                          \
        HpSharkComboResults<SharkFloatParams> &combo,                                                   \
        DebugGpuCombo *debugCombo);

#define ExplicitlyInstantiate(SharkFloatParams) ExplicitlyInstantiateMultiplyNTT(SharkFloatParams)

ExplicitInstantiateAll();

} // namespace HpShark