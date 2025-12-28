#include "DbgHeap.h"
#include "KernelInvoke.h"
#include "KernelInvokeInternal.h"
#include <sstream>
#include <stdexcept>

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

    cudaError_t err = cudaSuccess;
    err = cudaMalloc(&combo.d_tempProducts, BytesToAllocate);
    if (err != cudaSuccess) {
        std::ostringstream oss;
        oss << "cudaMalloc(&combo.d_tempProducts, BytesToAllocate) failed: " << cudaGetErrorString(err)
            << " (code " << static_cast<int>(err) << ")";
        throw std::runtime_error(oss.str());
    }

    if constexpr (!HpShark::TestInitCudaMemory) {
        cudaError_t err = cudaSuccess;
        err = cudaMemset(combo.d_tempProducts, 0, BytesToAllocate);
        if (err != cudaSuccess) {
            std::ostringstream oss;
            oss << "cudaMemset(combo.d_tempProducts, 0, BytesToAllocate) failed: "
                << cudaGetErrorString(err) << " (code " << static_cast<int>(err) << ")";
            throw std::runtime_error(oss.str());
        }
    } else {
        err = cudaMemset(combo.d_tempProducts, 0xCD, BytesToAllocate);
        if (err != cudaSuccess) {
            std::ostringstream oss;
            oss << "cudaMemset(combo.d_tempProducts, 0xCD, BytesToAllocate) failed: "
                << cudaGetErrorString(err) << " (code " << static_cast<int>(err) << ")";
            throw std::runtime_error(oss.str());
        }
    }

    // ---------------------------------------------------------------------
    // Allocate + shallow-copy combo to device (TestPerf style).
    // ---------------------------------------------------------------------
    err = cudaMalloc(&combo.comboGpu, sizeof(HpSharkReferenceResults<SharkFloatParams>));
    if (err != cudaSuccess) {
        std::ostringstream oss;
        oss << "cudaMalloc(&combo.comboGpu, sizeof(HpSharkReferenceResults<SharkFloatParams>)) failed: "
            << cudaGetErrorString(err) << " (code " << static_cast<int>(err) << ")";
        throw std::runtime_error(oss.str());
    }

    // Note: shallow copy; we will memset specific members below (same idea as TestPerf).
    err = cudaMemcpy(combo.comboGpu,
                     &combo,
                     sizeof(HpSharkReferenceResults<SharkFloatParams>),
                     cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::ostringstream oss;
        oss << "cudaMemcpy failed: " << cudaGetErrorString(err) << " (code " << static_cast<int>(err)
            << ")";
        throw std::runtime_error(oss.str());
    }

    // Host-only kernel arg staging (same convention as TestPerf).
    combo.kernelArgs[0] = (void *)&combo.comboGpu;
    combo.kernelArgs[1] = (void *)&combo.d_tempProducts;

    // Correctness path doesn't need a custom stream; keep stream = 0 like default.
    combo.stream = 0;
    static_assert(sizeof(cudaStream_t) == sizeof(combo.stream),
                  "cudaStream_t size mismatch with combo.stream");

    uint8_t byteToSet = HpShark::TestInitCudaMemory ? 0xCD : 0;

    // Clear result fields (keep behavior consistent with existing correctness code).
    err = cudaMemset(&combo.comboGpu->Add.A_X2, byteToSet, sizeof(HpSharkFloat<SharkFloatParams>));
    if (err != cudaSuccess) {
        std::ostringstream oss;
        oss << "cudaMemset(&combo.comboGpu->Add.A_X2, byteToSet, "
               "sizeof(HpSharkFloat<SharkFloatParams>)) failed: "
            << cudaGetErrorString(err) << " (code " << static_cast<int>(err) << ")";
        throw std::runtime_error(oss.str());
    }
    err = cudaMemset(&combo.comboGpu->Add.B_Y2, byteToSet, sizeof(HpSharkFloat<SharkFloatParams>));
    if (err != cudaSuccess) {
        std::ostringstream oss;
        oss << "cudaMemset(&combo.comboGpu->Add.B_Y2, byteToSet, "
               "sizeof(HpSharkFloat<SharkFloatParams>)) failed: "
            << cudaGetErrorString(err) << " (code " << static_cast<int>(err) << ")";
        throw std::runtime_error(oss.str());
    }
    err = cudaMemset(&combo.comboGpu->Add.D_2X, byteToSet, sizeof(HpSharkFloat<SharkFloatParams>));
    if (err != cudaSuccess) {
        std::ostringstream oss;
        oss << "cudaMemset(&combo.comboGpu->Add.D_2X, byteToSet, "
               "sizeof(HpSharkFloat<SharkFloatParams>)) failed: "
            << cudaGetErrorString(err) << " (code " << static_cast<int>(err) << ")";
        throw std::runtime_error(oss.str());
    }
    err = cudaMemset(
        &combo.comboGpu->Add.Result1_A_B_C, byteToSet, sizeof(HpSharkFloat<SharkFloatParams>));
    if (err != cudaSuccess) {
        std::ostringstream oss;
        oss << "cudaMemset(&combo.comboGpu->Add.Result1_A_B_C, byteToSet, "
               "sizeof(HpSharkFloat<SharkFloatParams>)) failed: "
            << cudaGetErrorString(err) << " (code " << static_cast<int>(err) << ")";
        throw std::runtime_error(oss.str());
    }
    err =
        cudaMemset(&combo.comboGpu->Add.Result2_D_E, byteToSet, sizeof(HpSharkFloat<SharkFloatParams>));
    if (err != cudaSuccess) {
        std::ostringstream oss;
        oss << "cudaMemset(&combo.comboGpu->Add.Result2_D_E, byteToSet, "
               "sizeof(HpSharkFloat<SharkFloatParams>)) failed: "
            << cudaGetErrorString(err) << " (code " << static_cast<int>(err) << ")";
        throw std::runtime_error(oss.str());
    }
    err = cudaMemset(
        &combo.comboGpu->Multiply.ResultX2, byteToSet, sizeof(HpSharkFloat<SharkFloatParams>));
    if (err != cudaSuccess) {
        std::ostringstream oss;
        oss << "cudaMemset(&combo.comboGpu->Multiply.ResultX2, byteToSet, "
               "sizeof(HpSharkFloat<SharkFloatParams>)) failed: "
            << cudaGetErrorString(err) << " (code " << static_cast<int>(err) << ")";
        throw std::runtime_error(oss.str());
    }
    err = cudaMemset(
        &combo.comboGpu->Multiply.Result2XY, byteToSet, sizeof(HpSharkFloat<SharkFloatParams>));
    if (err != cudaSuccess) {
        std::ostringstream oss;
        oss << "cudaMemset(&combo.comboGpu->Multiply.Result2XY, byteToSet, "
               "sizeof(HpSharkFloat<SharkFloatParams>)) failed: "
            << cudaGetErrorString(err) << " (code " << static_cast<int>(err) << ")";
        throw std::runtime_error(oss.str());
    }
    err = cudaMemset(
        &combo.comboGpu->Multiply.ResultY2, byteToSet, sizeof(HpSharkFloat<SharkFloatParams>));
    if (err != cudaSuccess) {
        std::ostringstream oss;
        oss << "cudaMemset(&combo.comboGpu->Multiply.ResultY2, byteToSet, "
               "sizeof(HpSharkFloat<SharkFloatParams>)) failed: "
            << cudaGetErrorString(err) << " (code " << static_cast<int>(err) << ")";
        throw std::runtime_error(oss.str());
    }

    // For correctness, the iter counter should start at 0 deterministically.
    {
        const uint64_t zero = 0;
        err = cudaMemcpy(
            &combo.comboGpu->OutputIterCount, &zero, sizeof(uint64_t), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            std::ostringstream oss;
            cudaError_t err = cudaSuccess;
            oss << "err = cudaMemcpy(&combo.comboGpu->OutputIterCount, &zero, sizeof(uint64_t), "
                   "cudaMemcpyHostToDevice) failed: "
                << cudaGetErrorString(err) << " (code " << static_cast<int>(err) << ")";
            throw std::runtime_error(oss.str());
            if (err != cudaSuccess) {
                std::ostringstream oss;
                oss << "cudaMemcpy failed: " << cudaGetErrorString(err) << " (code "
                    << static_cast<int>(err) << ")";
                throw std::runtime_error(oss.str());
            }
        }
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
    err = cudaMemcpy(
        &combo.comboGpu->MaxRuntimeIters, &kNumIters, sizeof(uint64_t), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::ostringstream oss;
        oss << "err = cudaMemcpy(&combo.comboGpu->MaxRuntimeIters, &kNumIters, sizeof(uint64_t), "
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

    void *kernelArgs[] = {(void *)&combo.comboGpu, (void *)&combo.d_tempProducts};

    {
        ScopedBenchmarkStopper stopper{timer};
        ComputeHpSharkReferenceGpuLoop<SharkFloatParams>(
            launchParams, *reinterpret_cast<cudaStream_t *>(&combo.stream), kernelArgs);
    }

    // Copy everything back (device pointer -> host struct).
    err = cudaMemcpy(&combo,
                     combo.comboGpu,
                     sizeof(HpSharkReferenceResults<SharkFloatParams>),
                     cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::ostringstream oss;
        oss << "cudaMemcpy failed: " << cudaGetErrorString(err) << " (code " << static_cast<int>(err)
            << ")";
        throw std::runtime_error(oss.str());
    }

    // ---------------------------------------------------------------------
    // Optional debug readback (keep the correctness behavior).
    // ---------------------------------------------------------------------
    if (debugCombo != nullptr) {
        if constexpr (HpShark::DebugChecksums) {
            debugCombo->States.resize(SharkFloatParams::NumDebugStates);
            err = cudaMemcpy(debugCombo->States.data(),
                             &combo.d_tempProducts[HpShark::AdditionalChecksumsOffset],
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
                             &combo.d_tempProducts[HpShark::AdditionalMultipliesOffset],
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

    // Roots were device-allocated in CopyRootsToCuda; destroy them like correctness does.
    SharkNTT::DestroyRoots<SharkFloatParams>(true, combo.comboGpu->Multiply.Roots);

    err = cudaFree(combo.comboGpu);
    if (err != cudaSuccess) {
        std::ostringstream oss;
        oss << "cudaFree(combo.comboGpu) failed: " << cudaGetErrorString(err) << " (code "
            << static_cast<int>(err) << ")";
        throw std::runtime_error(oss.str());
    }
    err = cudaFree(combo.d_tempProducts);
    if (err != cudaSuccess) {
        std::ostringstream oss;
        oss << "cudaFree(combo.d_tempProducts) failed: " << cudaGetErrorString(err) << " (code "
            << static_cast<int>(err) << ")";
        throw std::runtime_error(oss.str());
    }

    combo.comboGpu = nullptr;
    combo.d_tempProducts = nullptr;
}

#define ExplicitlyInstantiateHpSharkReference(SharkFloatParams)                                         \
    template void InvokeHpSharkReferenceKernelCorrectness<SharkFloatParams>(                            \
        const HpShark::LaunchParams &launchParams,                                                      \
        BenchmarkTimer &timer,                                                                          \
        HpSharkReferenceResults<SharkFloatParams> &combo,                                               \
        DebugGpuCombo *debugCombo);

#define ExplicitlyInstantiate(SharkFloatParams) ExplicitlyInstantiateHpSharkReference(SharkFloatParams)

ExplicitInstantiateAll();

} // namespace HpShark