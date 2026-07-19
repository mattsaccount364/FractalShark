#include "DbgHeap.h"
#include "Exceptions.h"
#include "KernelInvoke.h"
#include "KernelInvokeInternal.h"

#include <sstream>
#include <utility>

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
        throw FractalSharkSeriousException(oss.str());
    }

    if constexpr (!HpShark::TestInitCudaMemory) {
        cudaError_t err = cudaSuccess;
        err = cudaMemset(combo.d_tempProducts, 0, BytesToAllocate);
        if (err != cudaSuccess) {
            std::ostringstream oss;
            oss << "cudaMemset(combo.d_tempProducts, 0, BytesToAllocate) failed: "
                << cudaGetErrorString(err) << " (code " << static_cast<int>(err) << ")";
            throw FractalSharkSeriousException(oss.str());
        }
    } else {
        err = cudaMemset(combo.d_tempProducts, 0xCD, BytesToAllocate);
        if (err != cudaSuccess) {
            std::ostringstream oss;
            oss << "cudaMemset(combo.d_tempProducts, 0xCD, BytesToAllocate) failed: "
                << cudaGetErrorString(err) << " (code " << static_cast<int>(err) << ")";
            throw FractalSharkSeriousException(oss.str());
        }
    }

    // Zero the debug checksum region so stale data from previous kernels
    // doesn't cause false mismatches.
    if constexpr (HpShark::DebugChecksums) {
        err = cudaMemset(&combo.d_tempProducts[HpShark::AdditionalChecksumsOffset],
                         0,
                         SharkFloatParams::NumDebugStates * sizeof(DebugStateRaw));
        if (err != cudaSuccess) {
            std::ostringstream oss;
            oss << "cudaMemset(debug checksum region) failed: " << cudaGetErrorString(err) << " (code "
                << static_cast<int>(err) << ")";
            throw FractalSharkSeriousException(oss.str());
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
        throw FractalSharkSeriousException(oss.str());
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
        throw FractalSharkSeriousException(oss.str());
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
        throw FractalSharkSeriousException(oss.str());
    }
    err = cudaMemset(&combo.comboGpu->Add.B_Y2, byteToSet, sizeof(HpSharkFloat<SharkFloatParams>));
    if (err != cudaSuccess) {
        std::ostringstream oss;
        oss << "cudaMemset(&combo.comboGpu->Add.B_Y2, byteToSet, "
               "sizeof(HpSharkFloat<SharkFloatParams>)) failed: "
            << cudaGetErrorString(err) << " (code " << static_cast<int>(err) << ")";
        throw FractalSharkSeriousException(oss.str());
    }
    err = cudaMemset(&combo.comboGpu->Add.D_2X, byteToSet, sizeof(HpSharkFloat<SharkFloatParams>));
    if (err != cudaSuccess) {
        std::ostringstream oss;
        oss << "cudaMemset(&combo.comboGpu->Add.D_2X, byteToSet, "
               "sizeof(HpSharkFloat<SharkFloatParams>)) failed: "
            << cudaGetErrorString(err) << " (code " << static_cast<int>(err) << ")";
        throw FractalSharkSeriousException(oss.str());
    }
    err = cudaMemset(
        &combo.comboGpu->Add.Result1_A_B_C, byteToSet, sizeof(HpSharkFloat<SharkFloatParams>));
    if (err != cudaSuccess) {
        std::ostringstream oss;
        oss << "cudaMemset(&combo.comboGpu->Add.Result1_A_B_C, byteToSet, "
               "sizeof(HpSharkFloat<SharkFloatParams>)) failed: "
            << cudaGetErrorString(err) << " (code " << static_cast<int>(err) << ")";
        throw FractalSharkSeriousException(oss.str());
    }
    err =
        cudaMemset(&combo.comboGpu->Add.Result2_D_E, byteToSet, sizeof(HpSharkFloat<SharkFloatParams>));
    if (err != cudaSuccess) {
        std::ostringstream oss;
        oss << "cudaMemset(&combo.comboGpu->Add.Result2_D_E, byteToSet, "
               "sizeof(HpSharkFloat<SharkFloatParams>)) failed: "
            << cudaGetErrorString(err) << " (code " << static_cast<int>(err) << ")";
        throw FractalSharkSeriousException(oss.str());
    }
    err = cudaMemset(
        &combo.comboGpu->Multiply.ResultX2, byteToSet, sizeof(HpSharkFloat<SharkFloatParams>));
    if (err != cudaSuccess) {
        std::ostringstream oss;
        oss << "cudaMemset(&combo.comboGpu->Multiply.ResultX2, byteToSet, "
               "sizeof(HpSharkFloat<SharkFloatParams>)) failed: "
            << cudaGetErrorString(err) << " (code " << static_cast<int>(err) << ")";
        throw FractalSharkSeriousException(oss.str());
    }
    err = cudaMemset(
        &combo.comboGpu->Multiply.Result2XY, byteToSet, sizeof(HpSharkFloat<SharkFloatParams>));
    if (err != cudaSuccess) {
        std::ostringstream oss;
        oss << "cudaMemset(&combo.comboGpu->Multiply.Result2XY, byteToSet, "
               "sizeof(HpSharkFloat<SharkFloatParams>)) failed: "
            << cudaGetErrorString(err) << " (code " << static_cast<int>(err) << ")";
        throw FractalSharkSeriousException(oss.str());
    }
    err = cudaMemset(
        &combo.comboGpu->Multiply.ResultY2, byteToSet, sizeof(HpSharkFloat<SharkFloatParams>));
    if (err != cudaSuccess) {
        std::ostringstream oss;
        oss << "cudaMemset(&combo.comboGpu->Multiply.ResultY2, byteToSet, "
               "sizeof(HpSharkFloat<SharkFloatParams>)) failed: "
            << cudaGetErrorString(err) << " (code " << static_cast<int>(err) << ")";
        throw FractalSharkSeriousException(oss.str());
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
            throw FractalSharkSeriousException(oss.str());
            if (err != cudaSuccess) {
                std::ostringstream oss;
                oss << "cudaMemcpy failed: " << cudaGetErrorString(err) << " (code "
                    << static_cast<int>(err) << ")";
                throw FractalSharkSeriousException(oss.str());
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
        throw FractalSharkSeriousException(oss.str());
        if (err != cudaSuccess) {
            std::ostringstream oss;
            oss << "cudaMemcpy failed: " << cudaGetErrorString(err) << " (code " << static_cast<int>(err)
                << ")";
            throw FractalSharkSeriousException(oss.str());
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
        throw FractalSharkSeriousException(oss.str());
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
                throw FractalSharkSeriousException(oss.str());
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
                throw FractalSharkSeriousException(oss.str());
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
        throw FractalSharkSeriousException(oss.str());
    }
    err = cudaFree(combo.d_tempProducts);
    if (err != cudaSuccess) {
        std::ostringstream oss;
        oss << "cudaFree(combo.d_tempProducts) failed: " << cudaGetErrorString(err) << " (code "
            << static_cast<int>(err) << ")";
        throw FractalSharkSeriousException(oss.str());
    }

    combo.comboGpu = nullptr;
    combo.d_tempProducts = nullptr;
}

template <class SharkFloatParams>
void
InvokeHpSharkReference2KernelCorrectness(const HpShark::LaunchParams &launchParams,
                                         BenchmarkTimer &timer,
                                         HpSharkReferenceResults<SharkFloatParams> &combo,
                                         DebugGpuCombo *debugCombo)
{
    // Match TestPerf-style invocation, but correctness assumes exactly one iteration.
    constexpr uint64_t kNumIters = 1;

    combo.Reference2Workspace = nullptr;
    combo.d_reference2WorkspaceStorage = nullptr;
    combo.reference2WorkspaceStorageBytes = 0;

    // ---------------------------------------------------------------------
    // Allocate the globally shared debug/checksum scratch used by Ref2.
    // ---------------------------------------------------------------------
    constexpr size_t BytesToAllocate = HpShark::AdditionalUInt64Global * sizeof(uint64_t);

    cudaError_t err = cudaSuccess;
    err = cudaMalloc(&combo.d_tempProducts, BytesToAllocate);
    if (err != cudaSuccess) {
        std::ostringstream oss;
        oss << "cudaMalloc(&combo.d_tempProducts, BytesToAllocate) failed: " << cudaGetErrorString(err)
            << " (code " << static_cast<int>(err) << ")";
        throw FractalSharkSeriousException(oss.str());
    }

    if constexpr (!HpShark::TestInitCudaMemory) {
        cudaError_t err = cudaSuccess;
        err = cudaMemset(combo.d_tempProducts, 0, BytesToAllocate);
        if (err != cudaSuccess) {
            std::ostringstream oss;
            oss << "cudaMemset(combo.d_tempProducts, 0, BytesToAllocate) failed: "
                << cudaGetErrorString(err) << " (code " << static_cast<int>(err) << ")";
            throw FractalSharkSeriousException(oss.str());
        }
    } else {
        err = cudaMemset(combo.d_tempProducts, 0xCD, BytesToAllocate);
        if (err != cudaSuccess) {
            std::ostringstream oss;
            oss << "cudaMemset(combo.d_tempProducts, 0xCD, BytesToAllocate) failed: "
                << cudaGetErrorString(err) << " (code " << static_cast<int>(err) << ")";
            throw FractalSharkSeriousException(oss.str());
        }
    }

    // Zero the debug checksum region so stale data from previous kernels
    // doesn't cause false mismatches.
    if constexpr (HpShark::DebugChecksums) {
        err = cudaMemset(&combo.d_tempProducts[HpShark::AdditionalChecksumsOffset],
                         0,
                         SharkFloatParams::NumDebugStates * sizeof(DebugStateRaw));
        if (err != cudaSuccess) {
            std::ostringstream oss;
            oss << "cudaMemset(debug checksum region) failed: " << cudaGetErrorString(err) << " (code "
                << static_cast<int>(err) << ")";
            throw FractalSharkSeriousException(oss.str());
        }
    }

    // Ref2 keeps its fused NTT intermediates in one fixed-capacity workspace.
    using Workspace = HpSharkReference2Workspace<SharkFloatParams>;
    constexpr size_t spectrumCount = 7u + (SharkFloatParams::EnableNewtonRaphson ? 5u : 0u);
    constexpr size_t limbCount = SharkFloatParams::EnableNewtonRaphson ? 4u : 2u;
    auto alignWorkspace = [](size_t value, size_t alignment) {
        return (value + alignment - 1) & ~(alignment - 1);
    };

    size_t workspaceBytes = 0;
    workspaceBytes = alignWorkspace(workspaceBytes, alignof(uint64_t));
    workspaceBytes += spectrumCount * static_cast<size_t>(Workspace::MaxFusedN) * sizeof(uint64_t);
    workspaceBytes = alignWorkspace(workspaceBytes, alignof(int64_t));
    workspaceBytes += limbCount * static_cast<size_t>(Workspace::MaxFusedLimbs) * sizeof(int64_t);
    workspaceBytes = alignWorkspace(workspaceBytes, alignof(uint32_t));
    workspaceBytes += 2u * static_cast<size_t>(Workspace::MaxFusedLimbs) * sizeof(uint32_t);
    workspaceBytes = alignWorkspace(workspaceBytes, alignof(uint64_t));
    workspaceBytes += static_cast<size_t>(Workspace::MaxFusedLimbs) * sizeof(uint64_t);
    workspaceBytes = alignWorkspace(workspaceBytes, alignof(HpSharkReference2CarryPrefixDescriptor));
    workspaceBytes += static_cast<size_t>(Workspace::MaxCarryPrefixParts) *
                      sizeof(HpSharkReference2CarryPrefixDescriptor);
    workspaceBytes = alignWorkspace(workspaceBytes, alignof(uint32_t));
    workspaceBytes += 4u * sizeof(uint32_t);
    workspaceBytes = alignWorkspace(workspaceBytes, alignof(uint64_t));
    workspaceBytes += 4u * static_cast<size_t>(Workspace::MaxFusedN) * sizeof(uint64_t);
    workspaceBytes += 2u * static_cast<size_t>(Workspace::MaxFusedStages) * sizeof(uint64_t);

    void *workspaceStorage = nullptr;
    err = cudaMalloc(&workspaceStorage, workspaceBytes);
    if (err != cudaSuccess) {
        std::ostringstream oss;
        oss << "cudaMalloc(Reference2 workspace storage) failed: " << cudaGetErrorString(err)
            << " (code " << static_cast<int>(err) << ")";
        throw FractalSharkSeriousException(oss.str());
    }
    err = cudaMemset(workspaceStorage, 0, workspaceBytes);
    if (err != cudaSuccess) {
        std::ostringstream oss;
        oss << "cudaMemset(Reference2 workspace storage) failed: " << cudaGetErrorString(err)
            << " (code " << static_cast<int>(err) << ")";
        throw FractalSharkSeriousException(oss.str());
    }

    auto *workspaceBase = static_cast<uint8_t *>(workspaceStorage);
    size_t workspaceOffset = 0;
    auto allocateWorkspace = [&](size_t count, size_t elementSize, size_t alignment) {
        workspaceOffset = alignWorkspace(workspaceOffset, alignment);
        void *result = workspaceBase + workspaceOffset;
        workspaceOffset += count * elementSize;
        return result;
    };
    auto allocateSpectrum = [&] {
        return static_cast<uint64_t *>(
            allocateWorkspace(Workspace::MaxFusedN, sizeof(uint64_t), alignof(uint64_t)));
    };
    auto allocateLimbs = [&] {
        return static_cast<int64_t *>(
            allocateWorkspace(Workspace::MaxFusedLimbs, sizeof(int64_t), alignof(int64_t)));
    };

    Workspace workspace{};
    workspace.ZReal = allocateSpectrum();
    workspace.ZImag = allocateSpectrum();
    workspace.CReal = allocateSpectrum();
    workspace.CImag = allocateSpectrum();
    if constexpr (SharkFloatParams::EnableNewtonRaphson) {
        workspace.DzdcReal = allocateSpectrum();
        workspace.DzdcImag = allocateSpectrum();
        workspace.One = allocateSpectrum();
    }
    workspace.RealOutput = allocateSpectrum();
    workspace.ImagOutput = allocateSpectrum();
    if constexpr (SharkFloatParams::EnableNewtonRaphson) {
        workspace.DzdcRealOutput = allocateSpectrum();
        workspace.DzdcImagOutput = allocateSpectrum();
    }
    workspace.Product = allocateSpectrum();
    workspace.RealLimbs = allocateLimbs();
    workspace.ImagLimbs = allocateLimbs();
    if constexpr (SharkFloatParams::EnableNewtonRaphson) {
        workspace.DzdcRealLimbs = allocateLimbs();
        workspace.DzdcImagLimbs = allocateLimbs();
    }
    workspace.MagnitudeDigits = static_cast<uint32_t *>(
        allocateWorkspace(Workspace::MaxFusedLimbs, sizeof(uint32_t), alignof(uint32_t)));
    workspace.Magnitude = static_cast<uint32_t *>(
        allocateWorkspace(Workspace::MaxFusedLimbs, sizeof(uint32_t), alignof(uint32_t)));
    workspace.CarryPrefixTransforms = static_cast<uint64_t *>(
        allocateWorkspace(Workspace::MaxFusedLimbs, sizeof(uint64_t), alignof(uint64_t)));
    workspace.CarryPrefixDescriptors = static_cast<HpSharkReference2CarryPrefixDescriptor *>(
        allocateWorkspace(Workspace::MaxCarryPrefixParts,
                          sizeof(HpSharkReference2CarryPrefixDescriptor),
                          alignof(HpSharkReference2CarryPrefixDescriptor)));
    workspace.CarryPrefixControl =
        static_cast<uint32_t *>(allocateWorkspace(4u, sizeof(uint32_t), alignof(uint32_t)));
    workspace.Roots.stage_omegas = static_cast<uint64_t *>(
        allocateWorkspace(Workspace::MaxFusedStages, sizeof(uint64_t), alignof(uint64_t)));
    workspace.Roots.stage_omegas_inv = static_cast<uint64_t *>(
        allocateWorkspace(Workspace::MaxFusedStages, sizeof(uint64_t), alignof(uint64_t)));
    workspace.Roots.psi_pows = allocateSpectrum();
    workspace.Roots.psi_inv_pows = allocateSpectrum();
    workspace.Roots.stage_twiddles_fwd = allocateSpectrum();
    workspace.Roots.stage_twiddles_inv = allocateSpectrum();

    if (workspaceOffset != workspaceBytes)
        throw FractalSharkSeriousException("Reference2 workspace size does not match its layout");

    Workspace *workspaceGpu = nullptr;
    err = cudaMalloc(&workspaceGpu, sizeof(Workspace));
    if (err != cudaSuccess) {
        std::ostringstream oss;
        oss << "cudaMalloc(Reference2 workspace descriptor) failed: " << cudaGetErrorString(err)
            << " (code " << static_cast<int>(err) << ")";
        throw FractalSharkSeriousException(oss.str());
    }
    err = cudaMemcpy(workspaceGpu, &workspace, sizeof(Workspace), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::ostringstream oss;
        oss << "cudaMemcpy(Reference2 workspace descriptor H2D) failed: " << cudaGetErrorString(err)
            << " (code " << static_cast<int>(err) << ")";
        throw FractalSharkSeriousException(oss.str());
    }

    combo.Reference2Workspace = workspaceGpu;
    combo.d_reference2WorkspaceStorage = workspaceStorage;
    combo.reference2WorkspaceStorageBytes = workspaceBytes;

    // ---------------------------------------------------------------------
    // Allocate + shallow-copy combo to device (TestPerf style).
    // ---------------------------------------------------------------------
    err = cudaMalloc(&combo.comboGpu, sizeof(HpSharkReferenceResults<SharkFloatParams>));
    if (err != cudaSuccess) {
        std::ostringstream oss;
        oss << "cudaMalloc(&combo.comboGpu, sizeof(HpSharkReferenceResults<SharkFloatParams>)) failed: "
            << cudaGetErrorString(err) << " (code " << static_cast<int>(err) << ")";
        throw FractalSharkSeriousException(oss.str());
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
        throw FractalSharkSeriousException(oss.str());
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
        throw FractalSharkSeriousException(oss.str());
    }
    err = cudaMemset(&combo.comboGpu->Add.B_Y2, byteToSet, sizeof(HpSharkFloat<SharkFloatParams>));
    if (err != cudaSuccess) {
        std::ostringstream oss;
        oss << "cudaMemset(&combo.comboGpu->Add.B_Y2, byteToSet, "
               "sizeof(HpSharkFloat<SharkFloatParams>)) failed: "
            << cudaGetErrorString(err) << " (code " << static_cast<int>(err) << ")";
        throw FractalSharkSeriousException(oss.str());
    }
    err = cudaMemset(&combo.comboGpu->Add.D_2X, byteToSet, sizeof(HpSharkFloat<SharkFloatParams>));
    if (err != cudaSuccess) {
        std::ostringstream oss;
        oss << "cudaMemset(&combo.comboGpu->Add.D_2X, byteToSet, "
               "sizeof(HpSharkFloat<SharkFloatParams>)) failed: "
            << cudaGetErrorString(err) << " (code " << static_cast<int>(err) << ")";
        throw FractalSharkSeriousException(oss.str());
    }
    err = cudaMemset(
        &combo.comboGpu->Add.Result1_A_B_C, byteToSet, sizeof(HpSharkFloat<SharkFloatParams>));
    if (err != cudaSuccess) {
        std::ostringstream oss;
        oss << "cudaMemset(&combo.comboGpu->Add.Result1_A_B_C, byteToSet, "
               "sizeof(HpSharkFloat<SharkFloatParams>)) failed: "
            << cudaGetErrorString(err) << " (code " << static_cast<int>(err) << ")";
        throw FractalSharkSeriousException(oss.str());
    }
    err =
        cudaMemset(&combo.comboGpu->Add.Result2_D_E, byteToSet, sizeof(HpSharkFloat<SharkFloatParams>));
    if (err != cudaSuccess) {
        std::ostringstream oss;
        oss << "cudaMemset(&combo.comboGpu->Add.Result2_D_E, byteToSet, "
               "sizeof(HpSharkFloat<SharkFloatParams>)) failed: "
            << cudaGetErrorString(err) << " (code " << static_cast<int>(err) << ")";
        throw FractalSharkSeriousException(oss.str());
    }
    err = cudaMemset(
        &combo.comboGpu->Multiply.ResultX2, byteToSet, sizeof(HpSharkFloat<SharkFloatParams>));
    if (err != cudaSuccess) {
        std::ostringstream oss;
        oss << "cudaMemset(&combo.comboGpu->Multiply.ResultX2, byteToSet, "
               "sizeof(HpSharkFloat<SharkFloatParams>)) failed: "
            << cudaGetErrorString(err) << " (code " << static_cast<int>(err) << ")";
        throw FractalSharkSeriousException(oss.str());
    }
    err = cudaMemset(
        &combo.comboGpu->Multiply.Result2XY, byteToSet, sizeof(HpSharkFloat<SharkFloatParams>));
    if (err != cudaSuccess) {
        std::ostringstream oss;
        oss << "cudaMemset(&combo.comboGpu->Multiply.Result2XY, byteToSet, "
               "sizeof(HpSharkFloat<SharkFloatParams>)) failed: "
            << cudaGetErrorString(err) << " (code " << static_cast<int>(err) << ")";
        throw FractalSharkSeriousException(oss.str());
    }
    err = cudaMemset(
        &combo.comboGpu->Multiply.ResultY2, byteToSet, sizeof(HpSharkFloat<SharkFloatParams>));
    if (err != cudaSuccess) {
        std::ostringstream oss;
        oss << "cudaMemset(&combo.comboGpu->Multiply.ResultY2, byteToSet, "
               "sizeof(HpSharkFloat<SharkFloatParams>)) failed: "
            << cudaGetErrorString(err) << " (code " << static_cast<int>(err) << ")";
        throw FractalSharkSeriousException(oss.str());
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
            throw FractalSharkSeriousException(oss.str());
            if (err != cudaSuccess) {
                std::ostringstream oss;
                oss << "cudaMemcpy failed: " << cudaGetErrorString(err) << " (code "
                    << static_cast<int>(err) << ")";
                throw FractalSharkSeriousException(oss.str());
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
        throw FractalSharkSeriousException(oss.str());
        if (err != cudaSuccess) {
            std::ostringstream oss;
            oss << "cudaMemcpy failed: " << cudaGetErrorString(err) << " (code " << static_cast<int>(err)
                << ")";
            throw FractalSharkSeriousException(oss.str());
        }
    }

    void *kernelArgs[] = {(void *)&combo.comboGpu, (void *)&combo.d_tempProducts};

    {
        ScopedBenchmarkStopper stopper{timer};
        ComputeHpSharkReference2GpuLoop<SharkFloatParams>(
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
        throw FractalSharkSeriousException(oss.str());
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
                throw FractalSharkSeriousException(oss.str());
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
                throw FractalSharkSeriousException(oss.str());
            }
        }
    }

    // Roots were device-allocated in CopyRootsToCuda; destroy them like correctness does.
    SharkNTT::DestroyRoots<SharkFloatParams>(true, combo.comboGpu->Multiply.Roots);

    err = cudaFree(combo.Reference2Workspace);
    if (err != cudaSuccess) {
        std::ostringstream oss;
        oss << "cudaFree(combo.Reference2Workspace) failed: " << cudaGetErrorString(err) << " (code "
            << static_cast<int>(err) << ")";
        throw FractalSharkSeriousException(oss.str());
    }
    err = cudaFree(combo.d_reference2WorkspaceStorage);
    if (err != cudaSuccess) {
        std::ostringstream oss;
        oss << "cudaFree(combo.d_reference2WorkspaceStorage) failed: " << cudaGetErrorString(err)
            << " (code " << static_cast<int>(err) << ")";
        throw FractalSharkSeriousException(oss.str());
    }
    err = cudaFree(combo.comboGpu);
    if (err != cudaSuccess) {
        std::ostringstream oss;
        oss << "cudaFree(combo.comboGpu) failed: " << cudaGetErrorString(err) << " (code "
            << static_cast<int>(err) << ")";
        throw FractalSharkSeriousException(oss.str());
    }
    err = cudaFree(combo.d_tempProducts);
    if (err != cudaSuccess) {
        std::ostringstream oss;
        oss << "cudaFree(combo.d_tempProducts) failed: " << cudaGetErrorString(err) << " (code "
            << static_cast<int>(err) << ")";
        throw FractalSharkSeriousException(oss.str());
    }

    combo.Reference2Workspace = nullptr;
    combo.d_reference2WorkspaceStorage = nullptr;
    combo.reference2WorkspaceStorageBytes = 0;
    combo.comboGpu = nullptr;
    combo.d_tempProducts = nullptr;
}

#define ExplicitlyInstantiateHpSharkReference(SharkFloatParams)                                         \
    template void InvokeHpSharkReferenceKernelCorrectness<SharkFloatParams>(                            \
        const HpShark::LaunchParams &launchParams,                                                      \
        BenchmarkTimer &timer,                                                                          \
        HpSharkReferenceResults<SharkFloatParams> &combo,                                               \
        DebugGpuCombo *debugCombo);                                                                     \
    template void InvokeHpSharkReference2KernelCorrectness<SharkFloatParams>(                           \
        const HpShark::LaunchParams &launchParams,                                                      \
        BenchmarkTimer &timer,                                                                          \
        HpSharkReferenceResults<SharkFloatParams> &combo,                                               \
        DebugGpuCombo *debugCombo);

#define ExplicitlyInstantiate(SharkFloatParams) ExplicitlyInstantiateHpSharkReference(SharkFloatParams)

ExplicitInstantiateAll();

} // namespace HpShark
