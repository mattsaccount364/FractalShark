#pragma once

#include "Exceptions.h"
#include "HpSharkFloat.h"
#include "KernelInvoke.h"
#include "KernelInvokeInternal.h"
#include "LaunchParams.h"

#include <algorithm>
#include <chrono>
#include <memory>
#include <sstream>

namespace HpShark {

template <class SharkFloatParams>
std::unique_ptr<HpSharkReferenceResults<SharkFloatParams>>
InitHpSharkReference2Kernel(const HpShark::LaunchParams &launchParams,
                            const typename SharkFloatParams::Float hdrRadiusY,
                            const mpf_t srcX,
                            const mpf_t srcY)
{
    auto inputX = std::make_unique<HpSharkFloat<SharkFloatParams>>();
    auto inputY = std::make_unique<HpSharkFloat<SharkFloatParams>>();

    // Convert srcX and srcY to HpSharkFloat
    inputX->MpfToHpGpu(
        srcX, HpSharkFloat<SharkFloatParams>::DefaultMpirBits, InjectNoiseInLowOrder::Enable);
    inputY->MpfToHpGpu(
        srcY, HpSharkFloat<SharkFloatParams>::DefaultMpirBits, InjectNoiseInLowOrder::Enable);

    return InitHpSharkReference2Kernel<SharkFloatParams>(launchParams, hdrRadiusY, *inputX, *inputY);
}

template <class SharkFloatParams>
std::unique_ptr<HpSharkReferenceResults<SharkFloatParams>>
InitHpSharkReference2Kernel(const HpShark::LaunchParams &launchParams,
                            const typename SharkFloatParams::Float hdrRadiusY,
                            const HpSharkFloat<SharkFloatParams> &xNum,
                            const HpSharkFloat<SharkFloatParams> &yNum)
{
    auto combo = std::make_unique<HpSharkReferenceResults<SharkFloatParams>>();

    combo->RadiusY = hdrRadiusY;
    combo->Add.C_A = xNum;
    combo->Add.E_B = yNum;
    combo->Multiply.A = xNum;
    combo->Multiply.B = yNum;
    combo->PeriodicityStatus = PeriodicityResult::Unknown;
    combo->dzdcX = typename SharkFloatParams::Float{1};
    combo->dzdcY = typename SharkFloatParams::Float{0};
    combo->OutputIterCount = 0;
    combo->MaxRuntimeIters = 0; // Set below
    combo->Reference2Workspace = nullptr;
    combo->d_reference2WorkspaceStorage = nullptr;
    combo->reference2WorkspaceStorageBytes = 0;

    // NR state initialization
    if constexpr (SharkFloatParams::EnableNewtonRaphson) {
        // DzdcReal/DzdcImag default-constructed to zero (HpSharkFloat default ctor)
        // d2 initialized to zero
        combo->d2Real = typename SharkFloatParams::Float{};
        combo->d2Imag = typename SharkFloatParams::Float{};

        // Construct One constant for derivative add (+1)
        combo->Add.One.template FromHDRFloat<typename SharkFloatParams::SubType>(
            HDRFloat<typename SharkFloatParams::SubType>{typename SharkFloatParams::SubType(1.0)});
    }

    // Allocate the globally shared debug/checksum scratch used by Ref2.
    constexpr size_t BytesToAllocate = HpShark::AdditionalUInt64Global * sizeof(uint64_t);
    {
        cudaError_t cudaErr = cudaMalloc(&combo->d_tempProducts, BytesToAllocate);
        if (cudaErr != cudaSuccess) {
            std::ostringstream oss;
            oss << "cudaMalloc failed: " << cudaGetErrorString(cudaErr) << " (code "
                << static_cast<int>(cudaErr) << ")";
            throw FractalSharkSeriousException(oss.str());
        }
    }

    if constexpr (!HpShark::TestInitCudaMemory) {
        {
            cudaError_t cudaErr = cudaMemset(combo->d_tempProducts, 0, BytesToAllocate);
            if (cudaErr != cudaSuccess) {
                std::ostringstream oss;
                oss << "cudaMemset failed: " << cudaGetErrorString(cudaErr) << " (code "
                    << static_cast<int>(cudaErr) << ")";
                throw FractalSharkSeriousException(oss.str());
            }
        }
    } else {
        {
            cudaError_t cudaErr = cudaMemset(combo->d_tempProducts, 0xCD, BytesToAllocate);
            if (cudaErr != cudaSuccess) {
                std::ostringstream oss;
                oss << "cudaMemset failed: " << cudaGetErrorString(cudaErr) << " (code "
                    << static_cast<int>(cudaErr) << ")";
                throw FractalSharkSeriousException(oss.str());
            }
        }
    }

    // Ref2 keeps its fused NTT intermediates in one fixed-capacity workspace.
    using Workspace = HpSharkReference2Workspace<SharkFloatParams>;
    constexpr size_t spectrumCount = 6u + (SharkFloatParams::EnableNewtonRaphson ? 5u : 0u);
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
    workspaceBytes += static_cast<size_t>(Workspace::CarryPrefixControlCount) * sizeof(uint32_t);
    workspaceBytes = alignWorkspace(workspaceBytes, alignof(uint64_t));
    workspaceBytes += 4u * static_cast<size_t>(Workspace::MaxFusedN) * sizeof(uint64_t);
    workspaceBytes += 2u * static_cast<size_t>(Workspace::MaxFusedStages) * sizeof(uint64_t);

    void *workspaceStorage = nullptr;
    {
        cudaError_t cudaErr = cudaMalloc(&workspaceStorage, workspaceBytes);
        if (cudaErr != cudaSuccess) {
            std::ostringstream oss;
            oss << "cudaMalloc(Reference2 workspace storage) failed: " << cudaGetErrorString(cudaErr)
                << " (code " << static_cast<int>(cudaErr) << ")";
            throw FractalSharkSeriousException(oss.str());
        }
    }
    {
        cudaError_t cudaErr = cudaMemset(workspaceStorage, 0, workspaceBytes);
        if (cudaErr != cudaSuccess) {
            std::ostringstream oss;
            oss << "cudaMemset(Reference2 workspace storage) failed: " << cudaGetErrorString(cudaErr)
                << " (code " << static_cast<int>(cudaErr) << ")";
            throw FractalSharkSeriousException(oss.str());
        }
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
    workspace.CarryPrefixControl = static_cast<uint32_t *>(
        allocateWorkspace(Workspace::CarryPrefixControlCount, sizeof(uint32_t), alignof(uint32_t)));
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
    {
        cudaError_t cudaErr = cudaMalloc(&workspaceGpu, sizeof(Workspace));
        if (cudaErr != cudaSuccess) {
            std::ostringstream oss;
            oss << "cudaMalloc(Reference2 workspace descriptor) failed: " << cudaGetErrorString(cudaErr)
                << " (code " << static_cast<int>(cudaErr) << ")";
            throw FractalSharkSeriousException(oss.str());
        }
    }
    {
        cudaError_t cudaErr =
            cudaMemcpy(workspaceGpu, &workspace, sizeof(Workspace), cudaMemcpyHostToDevice);
        if (cudaErr != cudaSuccess) {
            std::ostringstream oss;
            oss << "cudaMemcpy(Reference2 workspace descriptor H2D) failed: "
                << cudaGetErrorString(cudaErr) << " (code " << static_cast<int>(cudaErr) << ")";
            throw FractalSharkSeriousException(oss.str());
        }
    }

    combo->Reference2Workspace = workspaceGpu;
    combo->d_reference2WorkspaceStorage = workspaceStorage;
    combo->reference2WorkspaceStorageBytes = workspaceBytes;

    // Host only
    combo->kernelArgs[0] = (void *)&combo->comboGpu;
    combo->kernelArgs[1] = (void *)&combo->d_tempProducts;
    combo->stream = 0;

    static_assert(sizeof(cudaStream_t) == sizeof(combo->stream),
                  "cudaStream_t size mismatch with combo->stream");

    if constexpr (HpShark::CustomStream) {
        auto &stream = *reinterpret_cast<cudaStream_t *>(&combo->stream);
        auto res = cudaStreamCreate(&stream); // Create a stream

        if (res != cudaSuccess) {
            std::cerr << "CUDA error in creating stream: " << cudaGetErrorString(res) << std::endl;
        }
    }

    {
        cudaError_t cudaErr =
            cudaMalloc(&combo->comboGpu, sizeof(HpSharkReferenceResults<SharkFloatParams>));
        if (cudaErr != cudaSuccess) {
            std::ostringstream oss;
            oss << "cudaMalloc failed: " << cudaGetErrorString(cudaErr) << " (code "
                << static_cast<int>(cudaErr) << ")";
            throw FractalSharkSeriousException(oss.str());
        }
    }

    // Note; shallow copy; we will memset specific members below
    cudaMemcpy(combo->comboGpu,
               combo.get(),
               sizeof(HpSharkReferenceResults<SharkFloatParams>),
               cudaMemcpyHostToDevice);

    uint8_t byteToSet = HpShark::TestInitCudaMemory ? 0xCD : 0;

    // Note: we're clearing a specific set of members here, not the whole struct.
    {
        cudaError_t cudaErr =
            cudaMemset(&combo->comboGpu->Add.A_X2, byteToSet, sizeof(HpSharkFloat<SharkFloatParams>));
        if (cudaErr != cudaSuccess) {
            std::ostringstream oss;
            oss << "cudaMemset failed: " << cudaGetErrorString(cudaErr) << " (code "
                << static_cast<int>(cudaErr) << ")";
            throw FractalSharkSeriousException(oss.str());
        }
    }
    {
        cudaError_t cudaErr =
            cudaMemset(&combo->comboGpu->Add.B_Y2, byteToSet, sizeof(HpSharkFloat<SharkFloatParams>));
        if (cudaErr != cudaSuccess) {
            std::ostringstream oss;
            oss << "cudaMemset failed: " << cudaGetErrorString(cudaErr) << " (code "
                << static_cast<int>(cudaErr) << ")";
            throw FractalSharkSeriousException(oss.str());
        }
    }
    {
        cudaError_t cudaErr =
            cudaMemset(&combo->comboGpu->Add.D_2X, byteToSet, sizeof(HpSharkFloat<SharkFloatParams>));
        if (cudaErr != cudaSuccess) {
            std::ostringstream oss;
            oss << "cudaMemset failed: " << cudaGetErrorString(cudaErr) << " (code "
                << static_cast<int>(cudaErr) << ")";
            throw FractalSharkSeriousException(oss.str());
        }
    }
    {
        cudaError_t cudaErr = cudaMemset(
            &combo->comboGpu->Add.Result1_A_B_C, byteToSet, sizeof(HpSharkFloat<SharkFloatParams>));
        if (cudaErr != cudaSuccess) {
            std::ostringstream oss;
            oss << "cudaMemset failed: " << cudaGetErrorString(cudaErr) << " (code "
                << static_cast<int>(cudaErr) << ")";
            throw FractalSharkSeriousException(oss.str());
        }
    }
    {
        cudaError_t cudaErr = cudaMemset(
            &combo->comboGpu->Add.Result2_D_E, byteToSet, sizeof(HpSharkFloat<SharkFloatParams>));
        if (cudaErr != cudaSuccess) {
            std::ostringstream oss;
            oss << "cudaMemset failed: " << cudaGetErrorString(cudaErr) << " (code "
                << static_cast<int>(cudaErr) << ")";
            throw FractalSharkSeriousException(oss.str());
        }
    }
    {
        cudaError_t cudaErr = cudaMemset(
            &combo->comboGpu->Multiply.ResultX2, byteToSet, sizeof(HpSharkFloat<SharkFloatParams>));
        if (cudaErr != cudaSuccess) {
            std::ostringstream oss;
            oss << "cudaMemset failed: " << cudaGetErrorString(cudaErr) << " (code "
                << static_cast<int>(cudaErr) << ")";
            throw FractalSharkSeriousException(oss.str());
        }
    }
    {
        cudaError_t cudaErr = cudaMemset(
            &combo->comboGpu->Multiply.Result2XY, byteToSet, sizeof(HpSharkFloat<SharkFloatParams>));
        if (cudaErr != cudaSuccess) {
            std::ostringstream oss;
            oss << "cudaMemset failed: " << cudaGetErrorString(cudaErr) << " (code "
                << static_cast<int>(cudaErr) << ")";
            throw FractalSharkSeriousException(oss.str());
        }
    }
    {
        cudaError_t cudaErr = cudaMemset(
            &combo->comboGpu->Multiply.ResultY2, byteToSet, sizeof(HpSharkFloat<SharkFloatParams>));
        if (cudaErr != cudaSuccess) {
            std::ostringstream oss;
            oss << "cudaMemset failed: " << cudaGetErrorString(cudaErr) << " (code "
                << static_cast<int>(cudaErr) << ")";
            throw FractalSharkSeriousException(oss.str());
        }
    }

    // Build NTT plan + roots exactly like correctness path
    {
        SharkNTT::RootTables NTTRoots;
        SharkNTT::BuildRoots<SharkFloatParams>(
            SharkFloatParams::NTTPlan.N, SharkFloatParams::NTTPlan.stages, NTTRoots);

        CopyRootsToCuda<SharkFloatParams>(combo->comboGpu->Multiply.Roots, NTTRoots);
        SharkNTT::DestroyRoots<SharkFloatParams>(false, NTTRoots);
    }

    cudaDeviceProp prop;
    int deviceId = 0;

    if constexpr (HpShark::CustomStream) {
        {
            cudaError_t cudaErr = cudaGetDeviceProperties(&prop, deviceId);
            if (cudaErr != cudaSuccess) {
                std::ostringstream oss;
                oss << "cudaGetDeviceProperties failed: " << cudaGetErrorString(cudaErr) << " (code "
                    << static_cast<int>(cudaErr) << ")";
                throw FractalSharkSeriousException(oss.str());
            }
        }
        // Set L2 persisting cache to cover our actual working set, not the full L2.
        const size_t workingSetBytes =
            sizeof(HpSharkReferenceResults<SharkFloatParams>) + BytesToAllocate;
        const size_t persistingSize =
            (workingSetBytes < static_cast<size_t>(prop.persistingL2CacheMaxSize))
                ? workingSetBytes
                : static_cast<size_t>(prop.persistingL2CacheMaxSize);
        cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, persistingSize);

        auto setAccess = [&](void *ptr, size_t numBytes) {
            cudaStreamAttrValue streamAttribute; // Stream level attributes data structure
            streamAttribute.accessPolicyWindow.base_ptr =
                reinterpret_cast<void *>(ptr); // Global Memory data pointer
            streamAttribute.accessPolicyWindow.num_bytes =
                numBytes; // Number of bytes for persisting accesses.
            // (Must be less than cudaDeviceProp::accessPolicyMaxWindowSize)
            streamAttribute.accessPolicyWindow.hitRatio =
                1.0; // Hint for L2 cache hit ratio for persisting accesses in the numBytes region
            streamAttribute.accessPolicyWindow.hitProp =
                cudaAccessPropertyPersisting; // Type of access property on cache hit
            streamAttribute.accessPolicyWindow.missProp =
                cudaAccessPropertyStreaming; // Type of access property on cache miss.

            // Set the attributes to a CUDA stream of type cudaStream_t
            auto &stream = *reinterpret_cast<cudaStream_t *>(&combo->stream);
            cudaError_t err =
                cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &streamAttribute);
            if (err != cudaSuccess) {
                std::ostringstream oss;
                oss << "cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow) failed: "
                    << cudaGetErrorString(err) << " (code " << static_cast<int>(err) << ")";
                throw FractalSharkSeriousException(oss.str());
            }
        };

        setAccess(combo->comboGpu, sizeof(HpSharkReferenceResults<SharkFloatParams>));
        setAccess(combo->d_tempProducts, BytesToAllocate);
    }

    return combo;
}

template <class SharkFloatParams>
void
InvokeHpSharkReference2Kernel(const HpShark::LaunchParams &launchParams,
                              HpSharkReferenceResults<SharkFloatParams> &combo,
                              uint64_t numIters)
{
    auto *comboGpu = combo.comboGpu;
    {
        cudaError_t res =
            cudaMemcpy(&comboGpu->MaxRuntimeIters, &numIters, sizeof(uint64_t), cudaMemcpyHostToDevice);
        if (res != cudaSuccess) {
            std::ostringstream oss;
            oss << "cudaMemcpy(MaxRuntimeIters H2D) failed: " << cudaGetErrorString(res) << " (code "
                << static_cast<int>(res) << ")";
            throw FractalSharkSeriousException(oss.str());
        }
    }
    ComputeHpSharkReference2GpuLoop<SharkFloatParams>(
        launchParams, *reinterpret_cast<cudaStream_t *>(&combo.stream), combo.kernelArgs);

    // Note: comboGpu is device pointer
    // Note: we copy everything back, even host-only stuff
    {
        cudaError_t res = cudaMemcpy(
            &combo, comboGpu, sizeof(HpSharkReferenceResults<SharkFloatParams>), cudaMemcpyDeviceToHost);
        if (res != cudaSuccess) {
            std::ostringstream oss;
            oss << "cudaMemcpy(comboGpu D2H) failed: " << cudaGetErrorString(res) << " (code "
                << static_cast<int>(res) << ")";
            throw FractalSharkSeriousException(oss.str());
        }
    }
}

template <class SharkFloatParams>
void
ShutdownHpSharkReference2Kernel(const HpShark::LaunchParams &launchParams,
                                HpSharkReferenceResults<SharkFloatParams> &combo,
                                DebugGpuCombo *debugCombo)
{
    if (debugCombo != nullptr) {
        if constexpr (HpShark::DebugGlobalState) {
            debugCombo->MultiplyCounts.resize(SharkFloatParams::NumDebugMultiplyCounts);
            cudaMemcpy(debugCombo->MultiplyCounts.data(),
                       &combo.d_tempProducts[HpShark::AdditionalMultipliesOffset],
                       SharkFloatParams::NumDebugMultiplyCounts * sizeof(DebugGlobalCountRaw),
                       cudaMemcpyDeviceToHost);
        }
    }

    // Roots were device-allocated in CopyRootsToCuda; destroy like correctness does
    SharkNTT::DestroyRoots<SharkFloatParams>(true, combo.comboGpu->Multiply.Roots);

    {
        cudaError_t cudaErr = cudaFree(combo.Reference2Workspace);
        if (cudaErr != cudaSuccess) {
            std::ostringstream oss;
            oss << "cudaFree(Reference2 workspace descriptor) failed: " << cudaGetErrorString(cudaErr)
                << " (code " << static_cast<int>(cudaErr) << ")";
            throw FractalSharkSeriousException(oss.str());
        }
        combo.Reference2Workspace = nullptr;
    }
    {
        cudaError_t cudaErr = cudaFree(combo.d_reference2WorkspaceStorage);
        if (cudaErr != cudaSuccess) {
            std::ostringstream oss;
            oss << "cudaFree(Reference2 workspace storage) failed: " << cudaGetErrorString(cudaErr)
                << " (code " << static_cast<int>(cudaErr) << ")";
            throw FractalSharkSeriousException(oss.str());
        }
        combo.d_reference2WorkspaceStorage = nullptr;
        combo.reference2WorkspaceStorageBytes = 0;
    }

    // Ref2 uses the same session/destruction contract as Ref1, with an
    // additional fixed-capacity workspace that is allocated on first Ref2
    // invocation and must be released before the combo storage disappears.
    if (combo.Reference2Workspace != nullptr) {
        cudaError_t cudaErr = cudaFree(combo.Reference2Workspace);
        if (cudaErr != cudaSuccess) {
            std::ostringstream oss;
            oss << "cudaFree(Reference2 descriptor) failed: " << cudaGetErrorString(cudaErr) << " (code "
                << static_cast<int>(cudaErr) << ")";
            throw FractalSharkSeriousException(oss.str());
        }
        combo.Reference2Workspace = nullptr;
    }
    if (combo.d_reference2WorkspaceStorage != nullptr) {
        cudaError_t cudaErr = cudaFree(combo.d_reference2WorkspaceStorage);
        if (cudaErr != cudaSuccess) {
            std::ostringstream oss;
            oss << "cudaFree(Reference2 workspace storage) failed: " << cudaGetErrorString(cudaErr)
                << " (code " << static_cast<int>(cudaErr) << ")";
            throw FractalSharkSeriousException(oss.str());
        }
        combo.d_reference2WorkspaceStorage = nullptr;
        combo.reference2WorkspaceStorageBytes = 0;
    }

    {
        cudaError_t cudaErr = cudaFree(combo.comboGpu);
        if (cudaErr != cudaSuccess) {
            std::ostringstream oss;
            oss << "cudaFree failed: " << cudaGetErrorString(cudaErr) << " (code "
                << static_cast<int>(cudaErr) << ")";
            throw FractalSharkSeriousException(oss.str());
        }
    }
    {
        cudaError_t cudaErr = cudaFree(combo.d_tempProducts);
        if (cudaErr != cudaSuccess) {
            std::ostringstream oss;
            oss << "cudaFree failed: " << cudaGetErrorString(cudaErr) << " (code "
                << static_cast<int>(cudaErr) << ")";
            throw FractalSharkSeriousException(oss.str());
        }
    }

    if constexpr (HpShark::CustomStream) {
        auto &stream = *reinterpret_cast<cudaStream_t *>(&combo.stream);
        auto res = cudaStreamDestroy(stream); // Destroy the stream

        if (res != cudaSuccess) {
            std::cerr << "CUDA error in destroying stream: " << cudaGetErrorString(res) << std::endl;
        }
    }
}

template <class SharkFloatParams>
uint64_t
EvaluateCriticalOrbitAndDerivs2_GPU(const mpf_t cReal,
                                    const mpf_t cImag,
                                    uint64_t period,
                                    mpf_t outZReal,
                                    mpf_t outZImag,
                                    mpf_t outDzdcReal,
                                    mpf_t outDzdcImag,
                                    HDRFloat<double> &outD2Real,
                                    HDRFloat<double> &outD2Imag,
                                    const HpShark::LaunchParams &externalLaunchParams,
                                    uint64_t startIter,
                                    bool (*shouldAbort)(),
                                    void (*onProgress)(uint64_t, void *),
                                    void *progressContext,
                                    uint64_t progressInterval)
{
    if constexpr (!SharkFloatParams::EnableNewtonRaphson) {
        return 0;
    }

    if (startIter > period)
        return startIter;

    constexpr int PrecBits = HpSharkFloat<SharkFloatParams>::DefaultPrecBits;
    typename SharkFloatParams::Float radiusY{1.0f};
    auto hpCR = std::make_unique<HpSharkFloat<SharkFloatParams>>();
    auto hpCI = std::make_unique<HpSharkFloat<SharkFloatParams>>();
    hpCR->MpfToHpGpu(
        *reinterpret_cast<const mpf_t *>(&cReal[0]), PrecBits, InjectNoiseInLowOrder::Disable);
    hpCI->MpfToHpGpu(
        *reinterpret_cast<const mpf_t *>(&cImag[0]), PrecBits, InjectNoiseInLowOrder::Disable);

    GpuOrbitSession2<SharkFloatParams> session(externalLaunchParams, radiusY, *hpCR, *hpCI);
    auto &combo = session.GetCombo();
    if (startIter == 0) {
        combo.Multiply.A = HpSharkFloat<SharkFloatParams>{};
        combo.Multiply.B = HpSharkFloat<SharkFloatParams>{};
        combo.Multiply.DzdcReal = HpSharkFloat<SharkFloatParams>{};
        combo.Multiply.DzdcImag = HpSharkFloat<SharkFloatParams>{};
        combo.d2Real = typename SharkFloatParams::Float{};
        combo.d2Imag = typename SharkFloatParams::Float{};
    } else {
        combo.Multiply.A.MpfToHpGpu(
            *reinterpret_cast<const mpf_t *>(&outZReal[0]), PrecBits, InjectNoiseInLowOrder::Disable);
        combo.Multiply.B.MpfToHpGpu(
            *reinterpret_cast<const mpf_t *>(&outZImag[0]), PrecBits, InjectNoiseInLowOrder::Disable);
        combo.Multiply.DzdcReal.MpfToHpGpu(
            *reinterpret_cast<const mpf_t *>(&outDzdcReal[0]), PrecBits, InjectNoiseInLowOrder::Disable);
        combo.Multiply.DzdcImag.MpfToHpGpu(
            *reinterpret_cast<const mpf_t *>(&outDzdcImag[0]), PrecBits, InjectNoiseInLowOrder::Disable);
        combo.d2Real = typename SharkFloatParams::Float{outD2Real};
        combo.d2Imag = typename SharkFloatParams::Float{outD2Imag};
    }
    combo.dzdcX = typename SharkFloatParams::Float{};
    combo.dzdcY = typename SharkFloatParams::Float{};
    combo.PeriodicityStatus = PeriodicityResult::Continue;

    SharkNTT::RootTables savedRoots;
    {
        cudaError_t cudaErr = cudaMemcpy(
            &savedRoots, &combo.comboGpu->Multiply.Roots, sizeof(savedRoots), cudaMemcpyDeviceToHost);
        if (cudaErr != cudaSuccess) {
            std::ostringstream oss;
            oss << "cudaMemcpy(Reference2 roots D2H) failed: " << cudaGetErrorString(cudaErr)
                << " (code " << static_cast<int>(cudaErr) << ")";
            throw FractalSharkSeriousException(oss.str());
        }
    }
    {
        cudaError_t cudaErr = cudaMemcpy(combo.comboGpu, &combo, sizeof(combo), cudaMemcpyHostToDevice);
        if (cudaErr != cudaSuccess) {
            std::ostringstream oss;
            oss << "cudaMemcpy(Reference2 initial state H2D) failed: " << cudaGetErrorString(cudaErr)
                << " (code " << static_cast<int>(cudaErr) << ")";
            throw FractalSharkSeriousException(oss.str());
        }
    }
    {
        cudaError_t cudaErr = cudaMemcpy(
            &combo.comboGpu->Multiply.Roots, &savedRoots, sizeof(savedRoots), cudaMemcpyHostToDevice);
        if (cudaErr != cudaSuccess) {
            std::ostringstream oss;
            oss << "cudaMemcpy(Reference2 roots H2D) failed: " << cudaGetErrorString(cudaErr)
                << " (code " << static_cast<int>(cudaErr) << ")";
            throw FractalSharkSeriousException(oss.str());
        }
    }

    constexpr uint64_t ChunkSize = HpSharkReferenceResults<SharkFloatParams>::MaxOutputIters;
    uint64_t done = startIter;
    uint64_t chunks = 0;
    while (done < period) {
        const uint64_t chunk = std::min(ChunkSize, period - done);
        InvokeHpSharkReference2Kernel(externalLaunchParams, combo, chunk);
        done += combo.OutputIterCount;
        ++chunks;

        combo.Multiply.A.HpGpuToMpf(*reinterpret_cast<mpf_t *>(&outZReal[0]));
        combo.Multiply.B.HpGpuToMpf(*reinterpret_cast<mpf_t *>(&outZImag[0]));
        combo.Multiply.DzdcReal.HpGpuToMpf(*reinterpret_cast<mpf_t *>(&outDzdcReal[0]));
        combo.Multiply.DzdcImag.HpGpuToMpf(*reinterpret_cast<mpf_t *>(&outDzdcImag[0]));
        outD2Real = HDRFloat<double>(combo.d2Real);
        outD2Imag = HDRFloat<double>(combo.d2Imag);

        if (onProgress && (progressInterval == 0 || chunks % progressInterval == 0))
            onProgress(done, progressContext);
        if ((shouldAbort && shouldAbort()) || combo.OutputIterCount == 0)
            break;
    }
    return done;
}

} // namespace HpShark
