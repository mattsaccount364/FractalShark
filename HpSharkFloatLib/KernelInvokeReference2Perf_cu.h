#pragma once

#include "Exceptions.h"
#include "HpSharkFloat.h"
#include "KernelInvoke.h"
#include "KernelInvokeInternal.h"
#include "KernelInvokeReference2Setup.h"
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

    return InitHpSharkReference2Kernel<SharkFloatParams>(
        launchParams, hdrRadiusY, *inputX, *inputY, actualPrecisionLimbs);
}

template <class SharkFloatParams>
std::unique_ptr<HpSharkReferenceResults<SharkFloatParams>>
InitHpSharkReference2Kernel(const HpShark::LaunchParams &launchParams,
                            const typename SharkFloatParams::Float hdrRadiusY,
                            const HpSharkFloat<SharkFloatParams> &xNum,
                            const HpSharkFloat<SharkFloatParams> &yNum,
                            uint32_t actualPrecisionLimbs)
{
    auto prepared =
        PrepareHpSharkReference2Tables<SharkFloatParams>(launchParams, xNum, yNum, actualPrecisionLimbs);
    const size_t storageBytes = prepared->GetStorageBytes();
    auto combo =
        InitHpSharkReference2Kernel<SharkFloatParams>(launchParams, hdrRadiusY, xNum, yNum, *prepared);
    combo->Reference2Workspace = prepared->ReleaseDescriptor();
    combo->d_reference2WorkspaceStorage = prepared->ReleaseStorage();
    combo->reference2WorkspaceStorageBytes = storageBytes;
    return combo;
}

template <class SharkFloatParams>
std::unique_ptr<HpSharkReferenceResults<SharkFloatParams>>
InitHpSharkReference2Kernel(const HpShark::LaunchParams &launchParams,
                            const typename SharkFloatParams::Float hdrRadiusY,
                            const HpSharkFloat<SharkFloatParams> &xNum,
                            const HpSharkFloat<SharkFloatParams> &yNum,
                            Reference2PreparedTables<SharkFloatParams> &preparedTables)
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
    combo->Reference2Workspace = preparedTables.GetDeviceDescriptor();
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
    auto *reference2Workspace = combo.Reference2Workspace;
    void *reference2WorkspaceStorage = combo.d_reference2WorkspaceStorage;
    const size_t reference2WorkspaceStorageBytes = combo.reference2WorkspaceStorageBytes;
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
    // The device copy borrows the workspace. Keep ownership exclusively in the host session.
    combo.Reference2Workspace = reference2Workspace;
    combo.d_reference2WorkspaceStorage = reference2WorkspaceStorage;
    combo.reference2WorkspaceStorageBytes = reference2WorkspaceStorageBytes;
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

    if (combo.d_reference2WorkspaceStorage != nullptr) {
        cudaError_t cudaErr = cudaFree(combo.Reference2Workspace);
        if (cudaErr != cudaSuccess) {
            std::ostringstream oss;
            oss << "cudaFree(Reference2 workspace descriptor) failed: " << cudaGetErrorString(cudaErr)
                << " (code " << static_cast<int>(cudaErr) << ")";
            throw FractalSharkSeriousException(oss.str());
        }
        combo.Reference2Workspace = nullptr;
        cudaErr = cudaFree(combo.d_reference2WorkspaceStorage);
        if (cudaErr != cudaSuccess) {
            std::ostringstream oss;
            oss << "cudaFree(Reference2 workspace storage) failed: " << cudaGetErrorString(cudaErr)
                << " (code " << static_cast<int>(cudaErr) << ")";
            throw FractalSharkSeriousException(oss.str());
        }
        combo.d_reference2WorkspaceStorage = nullptr;
        combo.reference2WorkspaceStorageBytes = 0;
    } else {
        combo.Reference2Workspace = nullptr;
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
                                    Reference2PreparedTables<SharkFloatParams> *preparedTables,
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

    std::unique_ptr<Reference2PreparedTables<SharkFloatParams>> ownedPreparedTables;
    if (preparedTables == nullptr) {
        ownedPreparedTables = PrepareHpSharkReference2Tables<SharkFloatParams>(
            externalLaunchParams, *hpCR, *hpCI, SharkFloatParams::GlobalNumUint32);
        preparedTables = ownedPreparedTables.get();
    }
    GpuOrbitSession2<SharkFloatParams> session(
        externalLaunchParams, radiusY, *hpCR, *hpCI, *preparedTables);
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

    {
        cudaError_t cudaErr = cudaMemcpy(combo.comboGpu, &combo, sizeof(combo), cudaMemcpyHostToDevice);
        if (cudaErr != cudaSuccess) {
            std::ostringstream oss;
            oss << "cudaMemcpy(Reference2 initial state H2D) failed: " << cudaGetErrorString(cudaErr)
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
