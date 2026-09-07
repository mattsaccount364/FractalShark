#pragma once

#include "Exceptions.h"
#include "HpSharkFloat.h"
#include "KernelInvoke.h"
#include "KernelInvokeInternal.h"
#include "KernelInvokeReferenceSetup.h"
#include "LaunchParams.h"

#include <algorithm>
#include <chrono>
#include <memory>
#include <sstream>

namespace HpShark {

template <class SharkFloatParams>
std::unique_ptr<HpSharkReferenceResults<SharkFloatParams>>
InitHpSharkReferenceKernel(const HpShark::LaunchParams &launchParams,
                           const typename SharkFloatParams::Float hdrRadiusY,
                           const mpf_t srcX,
                           const mpf_t srcY,
                           uint32_t actualPrecisionLimbs)
{
    auto inputX = std::make_unique<HpSharkFloat<SharkFloatParams>>();
    auto inputY = std::make_unique<HpSharkFloat<SharkFloatParams>>();

    // Convert srcX and srcY to HpSharkFloat
    inputX->MpfToHpGpu(
        srcX, HpSharkFloat<SharkFloatParams>::DefaultMpirBits, InjectNoiseInLowOrder::Enable);
    inputY->MpfToHpGpu(
        srcY, HpSharkFloat<SharkFloatParams>::DefaultMpirBits, InjectNoiseInLowOrder::Enable);

    return InitHpSharkReferenceKernel<SharkFloatParams>(
        launchParams, hdrRadiusY, *inputX, *inputY, actualPrecisionLimbs);
}

template <class SharkFloatParams>
std::unique_ptr<HpSharkReferenceResults<SharkFloatParams>>
InitHpSharkReferenceKernel(const HpShark::LaunchParams &launchParams,
                           const typename SharkFloatParams::Float hdrRadiusY,
                           const HpSharkFloat<SharkFloatParams> &xNum,
                           const HpSharkFloat<SharkFloatParams> &yNum,
                           uint32_t actualPrecisionLimbs)
{
    auto prepared =
        PrepareHpSharkReferenceTables<SharkFloatParams>(launchParams, xNum, yNum, actualPrecisionLimbs);
    const size_t storageBytes = prepared->GetStorageBytes();
    auto combo =
        InitHpSharkReferenceKernel<SharkFloatParams>(launchParams, hdrRadiusY, xNum, yNum, *prepared);
    combo->Workspace = prepared->ReleaseDescriptor();
    combo->OwnedWorkspaceStorage = prepared->ReleaseStorage();
    combo->OwnedWorkspaceStorageBytes = storageBytes;
    return combo;
}

template <class SharkFloatParams>
std::unique_ptr<HpSharkReferenceResults<SharkFloatParams>>
InitHpSharkReferenceKernel(const HpShark::LaunchParams &launchParams,
                           const typename SharkFloatParams::Float hdrRadiusY,
                           const HpSharkFloat<SharkFloatParams> &xNum,
                           const HpSharkFloat<SharkFloatParams> &yNum,
                           ReferencePreparedTables<SharkFloatParams> &preparedTables)
{
    auto combo = std::make_unique<HpSharkReferenceResults<SharkFloatParams>>();

    combo->RadiusY = hdrRadiusY;
    combo->CReal = xNum;
    combo->CImag = yNum;
    combo->ZReal = xNum;
    combo->ZImag = yNum;
    combo->PeriodicityStatus = PeriodicityResult::Unknown;
    combo->DzdcX = typename SharkFloatParams::Float{1};
    combo->DzdcY = typename SharkFloatParams::Float{0};
    combo->OutputIterCount = 0;
    combo->MaxRuntimeIters = 0; // Set below
    combo->Workspace = preparedTables.GetDeviceDescriptor();
    combo->OwnedWorkspaceStorage = nullptr;
    combo->OwnedWorkspaceStorageBytes = 0;

    // NR state initialization
    if constexpr (SharkFloatParams::EnableNewtonRaphson) {
        // DzdcReal/DzdcImag default-constructed to zero (HpSharkFloat default ctor)
        // d2 initialized to zero
        combo->D2Real = typename SharkFloatParams::Float{};
        combo->D2Imag = typename SharkFloatParams::Float{};

        // Construct One constant for derivative add (+1)
        combo->One.template FromHDRFloat<typename SharkFloatParams::SubType>(
            HDRFloat<typename SharkFloatParams::SubType>{typename SharkFloatParams::SubType(1.0)});
    }

    // Allocate the globally shared debug/checksum scratch used by the reference kernel.
    constexpr size_t BytesToAllocate = HpShark::AdditionalUInt64Global * sizeof(uint64_t);
    {
        cudaError_t cudaErr = cudaMalloc(&combo->DeviceDebugStorage, BytesToAllocate);
        if (cudaErr != cudaSuccess) {
            std::ostringstream oss;
            oss << "cudaMalloc failed: " << cudaGetErrorString(cudaErr) << " (code "
                << static_cast<int>(cudaErr) << ")";
            throw FractalSharkSeriousException(oss.str());
        }
    }

    if constexpr (!HpShark::TestInitCudaMemory) {
        {
            cudaError_t cudaErr = cudaMemset(combo->DeviceDebugStorage, 0, BytesToAllocate);
            if (cudaErr != cudaSuccess) {
                std::ostringstream oss;
                oss << "cudaMemset failed: " << cudaGetErrorString(cudaErr) << " (code "
                    << static_cast<int>(cudaErr) << ")";
                throw FractalSharkSeriousException(oss.str());
            }
        }
    } else {
        {
            cudaError_t cudaErr = cudaMemset(combo->DeviceDebugStorage, 0xCD, BytesToAllocate);
            if (cudaErr != cudaSuccess) {
                std::ostringstream oss;
                oss << "cudaMemset failed: " << cudaGetErrorString(cudaErr) << " (code "
                    << static_cast<int>(cudaErr) << ")";
                throw FractalSharkSeriousException(oss.str());
            }
        }
    }

    // Host only
    combo->KernelArgs[0] = (void *)&combo->DeviceResults;
    combo->KernelArgs[1] = (void *)&combo->DeviceDebugStorage;
    combo->Stream = 0;

    static_assert(sizeof(cudaStream_t) == sizeof(combo->Stream),
                  "cudaStream_t size mismatch with combo->Stream");

    if constexpr (HpShark::CustomStream) {
        auto &stream = *reinterpret_cast<cudaStream_t *>(&combo->Stream);
        auto res = cudaStreamCreate(&stream); // Create a stream

        if (res != cudaSuccess) {
            std::cerr << "CUDA error in creating stream: " << cudaGetErrorString(res) << std::endl;
        }
    }

    {
        cudaError_t cudaErr =
            cudaMalloc(&combo->DeviceResults, sizeof(HpSharkReferenceResults<SharkFloatParams>));
        if (cudaErr != cudaSuccess) {
            std::ostringstream oss;
            oss << "cudaMalloc failed: " << cudaGetErrorString(cudaErr) << " (code "
                << static_cast<int>(cudaErr) << ")";
            throw FractalSharkSeriousException(oss.str());
        }
    }

    // Note; shallow copy; we will memset specific members below
    cudaMemcpy(combo->DeviceResults,
               combo.get(),
               sizeof(HpSharkReferenceResults<SharkFloatParams>),
               cudaMemcpyHostToDevice);

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
            auto &stream = *reinterpret_cast<cudaStream_t *>(&combo->Stream);
            cudaError_t err =
                cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &streamAttribute);
            if (err != cudaSuccess) {
                std::ostringstream oss;
                oss << "cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow) failed: "
                    << cudaGetErrorString(err) << " (code " << static_cast<int>(err) << ")";
                throw FractalSharkSeriousException(oss.str());
            }
        };

        setAccess(combo->DeviceResults, sizeof(HpSharkReferenceResults<SharkFloatParams>));
        setAccess(combo->DeviceDebugStorage, BytesToAllocate);
    }

    return combo;
}

template <class SharkFloatParams>
void
InvokeHpSharkReferenceKernel(const HpShark::LaunchParams &launchParams,
                             HpSharkReferenceResults<SharkFloatParams> &combo,
                             uint64_t numIters)
{
    auto *deviceResults = combo.DeviceResults;
    auto *referenceWorkspace = combo.Workspace;
    void *referenceWorkspaceStorage = combo.OwnedWorkspaceStorage;
    const size_t ownedWorkspaceStorageBytes = combo.OwnedWorkspaceStorageBytes;
    {
        cudaError_t res = cudaMemcpy(
            &deviceResults->MaxRuntimeIters, &numIters, sizeof(uint64_t), cudaMemcpyHostToDevice);
        if (res != cudaSuccess) {
            std::ostringstream oss;
            oss << "cudaMemcpy(MaxRuntimeIters H2D) failed: " << cudaGetErrorString(res) << " (code "
                << static_cast<int>(res) << ")";
            throw FractalSharkSeriousException(oss.str());
        }
    }
    ComputeHpSharkReferenceGpuLoop<SharkFloatParams>(
        launchParams, *reinterpret_cast<cudaStream_t *>(&combo.Stream), combo.KernelArgs);

    // Note: deviceResults is a device pointer.
    // Note: we copy everything back, even host-only stuff
    {
        cudaError_t res = cudaMemcpy(&combo,
                                     deviceResults,
                                     sizeof(HpSharkReferenceResults<SharkFloatParams>),
                                     cudaMemcpyDeviceToHost);
        if (res != cudaSuccess) {
            std::ostringstream oss;
            oss << "cudaMemcpy(deviceResults D2H) failed: " << cudaGetErrorString(res) << " (code "
                << static_cast<int>(res) << ")";
            throw FractalSharkSeriousException(oss.str());
        }
    }
    // The device copy borrows the workspace. Keep ownership exclusively in the host session.
    combo.Workspace = referenceWorkspace;
    combo.OwnedWorkspaceStorage = referenceWorkspaceStorage;
    combo.OwnedWorkspaceStorageBytes = ownedWorkspaceStorageBytes;
}

template <class SharkFloatParams>
void
ShutdownHpSharkReferenceKernel(const HpShark::LaunchParams &launchParams,
                               HpSharkReferenceResults<SharkFloatParams> &combo,
                               DebugGpuCombo *debugCombo)
{
    if (debugCombo != nullptr) {
        if constexpr (HpShark::DebugChecksums) {
            constexpr size_t debugStateCount = static_cast<size_t>(DebugStatePurpose::NumPurposes);
            debugCombo->States.resize(debugStateCount);
            const cudaError_t copyResult =
                cudaMemcpy(debugCombo->States.data(),
                           &combo.DeviceDebugStorage[HpShark::AdditionalChecksumsOffset],
                           debugStateCount * sizeof(DebugStateRaw),
                           cudaMemcpyDeviceToHost);
            if (copyResult != cudaSuccess) {
                std::ostringstream message;
                message << "cudaMemcpy(reference checksum states D2H) failed: "
                        << cudaGetErrorString(copyResult) << " (code " << static_cast<int>(copyResult)
                        << ')';
                throw FractalSharkSeriousException(message.str());
            }
        }

        if constexpr (HpShark::DebugGlobalState) {
            debugCombo->MultiplyCounts.resize(SharkFloatParams::NumDebugMultiplyCounts);
            cudaMemcpy(debugCombo->MultiplyCounts.data(),
                       &combo.DeviceDebugStorage[HpShark::AdditionalDebugCountsOffset],
                       SharkFloatParams::NumDebugMultiplyCounts * sizeof(DebugGlobalCountRaw),
                       cudaMemcpyDeviceToHost);
        }
    }

    if (combo.OwnedWorkspaceStorage != nullptr) {
        cudaError_t cudaErr = cudaFree(combo.Workspace);
        if (cudaErr != cudaSuccess) {
            std::ostringstream oss;
            oss << "cudaFree(Reference workspace descriptor) failed: " << cudaGetErrorString(cudaErr)
                << " (code " << static_cast<int>(cudaErr) << ")";
            throw FractalSharkSeriousException(oss.str());
        }
        combo.Workspace = nullptr;
        cudaErr = cudaFree(combo.OwnedWorkspaceStorage);
        if (cudaErr != cudaSuccess) {
            std::ostringstream oss;
            oss << "cudaFree(Reference workspace storage) failed: " << cudaGetErrorString(cudaErr)
                << " (code " << static_cast<int>(cudaErr) << ")";
            throw FractalSharkSeriousException(oss.str());
        }
        combo.OwnedWorkspaceStorage = nullptr;
        combo.OwnedWorkspaceStorageBytes = 0;
    } else {
        combo.Workspace = nullptr;
    }

    {
        cudaError_t cudaErr = cudaFree(combo.DeviceResults);
        if (cudaErr != cudaSuccess) {
            std::ostringstream oss;
            oss << "cudaFree failed: " << cudaGetErrorString(cudaErr) << " (code "
                << static_cast<int>(cudaErr) << ")";
            throw FractalSharkSeriousException(oss.str());
        }
    }
    {
        cudaError_t cudaErr = cudaFree(combo.DeviceDebugStorage);
        if (cudaErr != cudaSuccess) {
            std::ostringstream oss;
            oss << "cudaFree failed: " << cudaGetErrorString(cudaErr) << " (code "
                << static_cast<int>(cudaErr) << ")";
            throw FractalSharkSeriousException(oss.str());
        }
    }

    if constexpr (HpShark::CustomStream) {
        auto &stream = *reinterpret_cast<cudaStream_t *>(&combo.Stream);
        auto res = cudaStreamDestroy(stream); // Destroy the stream

        if (res != cudaSuccess) {
            std::cerr << "CUDA error in destroying stream: " << cudaGetErrorString(res) << std::endl;
        }
    }
}

template <class SharkFloatParams>
uint64_t
EvaluateCriticalOrbitAndDerivs_GPU(const mpf_t cReal,
                                   const mpf_t cImag,
                                   uint64_t period,
                                   mpf_t outZReal,
                                   mpf_t outZImag,
                                   mpf_t outDzdcReal,
                                   mpf_t outDzdcImag,
                                   HDRFloat<double> &outD2Real,
                                   HDRFloat<double> &outD2Imag,
                                   const HpShark::LaunchParams &externalLaunchParams,
                                   ReferencePreparedTables<SharkFloatParams> *preparedTables,
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

    std::unique_ptr<ReferencePreparedTables<SharkFloatParams>> ownedPreparedTables;
    if (preparedTables == nullptr) {
        ownedPreparedTables = PrepareHpSharkReferenceTables<SharkFloatParams>(
            externalLaunchParams, *hpCR, *hpCI, SharkFloatParams::GlobalNumUint32);
        preparedTables = ownedPreparedTables.get();
    }
    GpuOrbitSession<SharkFloatParams> session(
        externalLaunchParams, radiusY, *hpCR, *hpCI, *preparedTables, nullptr);
    auto &combo = session.GetResults();
    if (startIter == 0) {
        combo.ZReal = HpSharkFloat<SharkFloatParams>{};
        combo.ZImag = HpSharkFloat<SharkFloatParams>{};
        combo.DzdcReal = HpSharkFloat<SharkFloatParams>{};
        combo.DzdcImag = HpSharkFloat<SharkFloatParams>{};
        combo.D2Real = typename SharkFloatParams::Float{};
        combo.D2Imag = typename SharkFloatParams::Float{};
    } else {
        combo.ZReal.MpfToHpGpu(
            *reinterpret_cast<const mpf_t *>(&outZReal[0]), PrecBits, InjectNoiseInLowOrder::Disable);
        combo.ZImag.MpfToHpGpu(
            *reinterpret_cast<const mpf_t *>(&outZImag[0]), PrecBits, InjectNoiseInLowOrder::Disable);
        combo.DzdcReal.MpfToHpGpu(
            *reinterpret_cast<const mpf_t *>(&outDzdcReal[0]), PrecBits, InjectNoiseInLowOrder::Disable);
        combo.DzdcImag.MpfToHpGpu(
            *reinterpret_cast<const mpf_t *>(&outDzdcImag[0]), PrecBits, InjectNoiseInLowOrder::Disable);
        combo.D2Real = typename SharkFloatParams::Float{outD2Real};
        combo.D2Imag = typename SharkFloatParams::Float{outD2Imag};
    }
    combo.DzdcX = typename SharkFloatParams::Float{};
    combo.DzdcY = typename SharkFloatParams::Float{};
    combo.PeriodicityStatus = PeriodicityResult::Continue;

    {
        cudaError_t cudaErr =
            cudaMemcpy(combo.DeviceResults, &combo, sizeof(combo), cudaMemcpyHostToDevice);
        if (cudaErr != cudaSuccess) {
            std::ostringstream oss;
            oss << "cudaMemcpy(Reference initial state H2D) failed: " << cudaGetErrorString(cudaErr)
                << " (code " << static_cast<int>(cudaErr) << ")";
            throw FractalSharkSeriousException(oss.str());
        }
    }
    constexpr uint64_t ChunkSize = HpSharkReferenceResults<SharkFloatParams>::MaxOutputIters;
    uint64_t done = startIter;
    uint64_t chunks = 0;
    while (done < period) {
        const uint64_t chunk = std::min(ChunkSize, period - done);
        InvokeHpSharkReferenceKernel(externalLaunchParams, combo, chunk);
        done += combo.OutputIterCount;
        ++chunks;

        combo.ZReal.HpGpuToMpf(*reinterpret_cast<mpf_t *>(&outZReal[0]));
        combo.ZImag.HpGpuToMpf(*reinterpret_cast<mpf_t *>(&outZImag[0]));
        combo.DzdcReal.HpGpuToMpf(*reinterpret_cast<mpf_t *>(&outDzdcReal[0]));
        combo.DzdcImag.HpGpuToMpf(*reinterpret_cast<mpf_t *>(&outDzdcImag[0]));
        outD2Real = HDRFloat<double>(combo.D2Real);
        outD2Imag = HDRFloat<double>(combo.D2Imag);

        if (onProgress && (progressInterval == 0 || chunks % progressInterval == 0))
            onProgress(done, progressContext);
        if ((shouldAbort && shouldAbort()) || combo.OutputIterCount == 0)
            break;
    }
    return done;
}

template <class SharkFloatParams>
uint64_t
EvaluateCriticalOrbitAndDerivs_GPU(const mpf_t cReal,
                                   const mpf_t cImag,
                                   uint64_t period,
                                   mpf_t outZReal,
                                   mpf_t outZImag,
                                   mpf_t outDzdcReal,
                                   mpf_t outDzdcImag,
                                   HDRFloat<double> &outD2Real,
                                   HDRFloat<double> &outD2Imag,
                                   const HpShark::LaunchParams &externalLaunchParams,
                                   uint32_t actualPrecisionLimbs,
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

    auto preparedTables = PrepareHpSharkReferenceTables<SharkFloatParams>(
        externalLaunchParams, cReal, cImag, actualPrecisionLimbs);
    return EvaluateCriticalOrbitAndDerivs_GPU<SharkFloatParams>(cReal,
                                                                cImag,
                                                                period,
                                                                outZReal,
                                                                outZImag,
                                                                outDzdcReal,
                                                                outDzdcImag,
                                                                outD2Real,
                                                                outD2Imag,
                                                                externalLaunchParams,
                                                                preparedTables.get(),
                                                                startIter,
                                                                shouldAbort,
                                                                onProgress,
                                                                progressContext,
                                                                progressInterval);
}

} // namespace HpShark
