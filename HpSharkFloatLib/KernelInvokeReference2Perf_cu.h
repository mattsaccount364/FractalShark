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

namespace Detail {

inline void
CheckReference2Cuda(cudaError_t error, const char *operation)
{
    if (error == cudaSuccess)
        return;
    std::ostringstream message;
    message << operation << " failed: " << cudaGetErrorString(error) << " (code "
            << static_cast<int>(error) << ")";
    throw FractalSharkSeriousException(message.str());
}

inline size_t
AlignReference2(size_t value, size_t alignment)
{
    return (value + alignment - 1) & ~(alignment - 1);
}

template <class SharkFloatParams>
size_t
Reference2WorkspaceStorageBytes()
{
    using Workspace = HpSharkReference2Workspace<SharkFloatParams>;
    // Four normal inputs, two normal outputs, and one product scratch. Newton-Raphson adds
    // three inputs and two outputs. The four root tables are accounted for separately below.
    constexpr size_t spectrumCount = 7u + (SharkFloatParams::EnableNewtonRaphson ? 5u : 0u);
    constexpr size_t limbCount = SharkFloatParams::EnableNewtonRaphson ? 4u : 2u;
    size_t bytes = 0;
    bytes = AlignReference2(bytes, alignof(uint64_t));
    bytes += spectrumCount * static_cast<size_t>(Workspace::MaxFusedN) * sizeof(uint64_t);
    bytes = AlignReference2(bytes, alignof(int64_t));
    bytes += limbCount * static_cast<size_t>(Workspace::MaxFusedLimbs) * sizeof(int64_t);
    bytes = AlignReference2(bytes, alignof(uint32_t));
    bytes += 2u * static_cast<size_t>(Workspace::MaxFusedLimbs) * sizeof(uint32_t);
    bytes = AlignReference2(bytes, alignof(uint64_t));
    bytes += static_cast<size_t>(Workspace::MaxFusedLimbs) * sizeof(uint64_t);
    bytes = AlignReference2(bytes, alignof(HpSharkReference2CarryPrefixDescriptor));
    bytes += static_cast<size_t>(Workspace::MaxCarryPrefixParts) *
             sizeof(HpSharkReference2CarryPrefixDescriptor);
    bytes = AlignReference2(bytes, alignof(uint32_t));
    bytes += 4u * sizeof(uint32_t);
    bytes = AlignReference2(bytes, alignof(uint64_t));
    bytes += 4u * static_cast<size_t>(Workspace::MaxFusedN) * sizeof(uint64_t);
    bytes += 2u * static_cast<size_t>(Workspace::MaxFusedStages) * sizeof(uint64_t);
    return bytes;
}

template <class SharkFloatParams>
void
InitializeReference2Workspace(HpSharkReferenceResults<SharkFloatParams> &combo)
{
    if (combo.Reference2Workspace != nullptr)
        return;

    using Workspace = HpSharkReference2Workspace<SharkFloatParams>;
    const size_t bytes = Reference2WorkspaceStorageBytes<SharkFloatParams>();
    void *storage = nullptr;
    Workspace *deviceDescriptor = nullptr;
    try {
        CheckReference2Cuda(cudaMalloc(&storage, bytes), "cudaMalloc(Reference2 workspace storage)");
        CheckReference2Cuda(cudaMemset(storage, 0, bytes), "cudaMemset(Reference2 workspace storage)");
        auto *base = static_cast<uint8_t *>(storage);
        size_t offset = 0;
        auto allocate = [&](size_t count, size_t elementSize, size_t alignment) {
            offset = AlignReference2(offset, alignment);
            void *result = base + offset;
            offset += count * elementSize;
            return result;
        };
        auto spectrum = [&] {
            return static_cast<uint64_t *>(
                allocate(Workspace::MaxFusedN, sizeof(uint64_t), alignof(uint64_t)));
        };
        auto limbs = [&] {
            return static_cast<int64_t *>(
                allocate(Workspace::MaxFusedLimbs, sizeof(int64_t), alignof(int64_t)));
        };

        Workspace descriptor{};
        descriptor.ZReal = spectrum();
        descriptor.ZImag = spectrum();
        descriptor.CReal = spectrum();
        descriptor.CImag = spectrum();
        if constexpr (SharkFloatParams::EnableNewtonRaphson) {
            descriptor.DzdcReal = spectrum();
            descriptor.DzdcImag = spectrum();
            descriptor.One = spectrum();
        }
        descriptor.RealOutput = spectrum();
        descriptor.ImagOutput = spectrum();
        if constexpr (SharkFloatParams::EnableNewtonRaphson) {
            descriptor.DzdcRealOutput = spectrum();
            descriptor.DzdcImagOutput = spectrum();
        }
        descriptor.Product = spectrum();
        descriptor.RealLimbs = limbs();
        descriptor.ImagLimbs = limbs();
        if constexpr (SharkFloatParams::EnableNewtonRaphson) {
            descriptor.DzdcRealLimbs = limbs();
            descriptor.DzdcImagLimbs = limbs();
        }
        descriptor.MagnitudeDigits = static_cast<uint32_t *>(
            allocate(Workspace::MaxFusedLimbs, sizeof(uint32_t), alignof(uint32_t)));
        descriptor.Magnitude = static_cast<uint32_t *>(
            allocate(Workspace::MaxFusedLimbs, sizeof(uint32_t), alignof(uint32_t)));
        descriptor.CarryPrefixTransforms = static_cast<uint64_t *>(
            allocate(Workspace::MaxFusedLimbs, sizeof(uint64_t), alignof(uint64_t)));
        descriptor.CarryPrefixDescriptors = static_cast<HpSharkReference2CarryPrefixDescriptor *>(
            allocate(Workspace::MaxCarryPrefixParts,
                     sizeof(HpSharkReference2CarryPrefixDescriptor),
                     alignof(HpSharkReference2CarryPrefixDescriptor)));
        descriptor.CarryPrefixControl =
            static_cast<uint32_t *>(allocate(4u, sizeof(uint32_t), alignof(uint32_t)));
        descriptor.Roots.stage_omegas = static_cast<uint64_t *>(
            allocate(Workspace::MaxFusedStages, sizeof(uint64_t), alignof(uint64_t)));
        descriptor.Roots.stage_omegas_inv = static_cast<uint64_t *>(
            allocate(Workspace::MaxFusedStages, sizeof(uint64_t), alignof(uint64_t)));
        descriptor.Roots.psi_pows = spectrum();
        descriptor.Roots.psi_inv_pows = spectrum();
        descriptor.Roots.stage_twiddles_fwd = spectrum();
        descriptor.Roots.stage_twiddles_inv = spectrum();

        if (offset != bytes)
            throw FractalSharkSeriousException("Reference2 workspace size does not match its layout");

        CheckReference2Cuda(cudaMalloc(&deviceDescriptor, sizeof(Workspace)),
                            "cudaMalloc(Reference2 descriptor)");
        CheckReference2Cuda(
            cudaMemcpy(deviceDescriptor, &descriptor, sizeof(Workspace), cudaMemcpyHostToDevice),
            "cudaMemcpy(Reference2 descriptor H2D)");
        CheckReference2Cuda(cudaMemcpy(&combo.comboGpu->Reference2Workspace,
                                       &deviceDescriptor,
                                       sizeof(deviceDescriptor),
                                       cudaMemcpyHostToDevice),
                            "cudaMemcpy(Reference2 descriptor pointer H2D)");

        combo.Reference2Workspace = deviceDescriptor;
        combo.d_reference2WorkspaceStorage = storage;
        combo.reference2WorkspaceStorageBytes = bytes;
    } catch (...) {
        if (deviceDescriptor != nullptr)
            cudaFree(deviceDescriptor);
        if (storage != nullptr)
            cudaFree(storage);
        throw;
    }
}

} // namespace Detail

template <class SharkFloatParams>
std::unique_ptr<HpSharkReferenceResults<SharkFloatParams>>
InitHpSharkReference2Kernel(const HpShark::LaunchParams &launchParams,
                            const typename SharkFloatParams::Float hdrRadiusY,
                            const mpf_t srcX,
                            const mpf_t srcY)
{
    auto inputX = std::make_unique<HpSharkFloat<SharkFloatParams>>();
    auto inputY = std::make_unique<HpSharkFloat<SharkFloatParams>>();
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
    // Keep Ref2's setup separate from Ref1's public lifecycle.  The common
    // combo/stream/root initialization intentionally remains the Ref1 setup,
    // then Ref2 adds its fixed-capacity workspace before the first launch.
    auto combo = InitHpSharkReferenceKernel<SharkFloatParams>(launchParams, hdrRadiusY, xNum, yNum);
    try {
        Detail::InitializeReference2Workspace(*combo);
    } catch (...) {
        ShutdownHpSharkReferenceKernel<SharkFloatParams>(launchParams, *combo, nullptr);
        throw;
    }
    return combo;
}

template <class SharkFloatParams>
void
InvokeHpSharkReference2Kernel(const HpShark::LaunchParams &launchParams,
                              HpSharkReferenceResults<SharkFloatParams> &combo,
                              uint64_t numIters)
{
    Detail::CheckReference2Cuda(
        cudaMemcpy(
            &combo.comboGpu->MaxRuntimeIters, &numIters, sizeof(numIters), cudaMemcpyHostToDevice),
        "cudaMemcpy(Reference2 MaxRuntimeIters H2D)");
    ComputeHpSharkReference2GpuLoop<SharkFloatParams>(
        launchParams, *reinterpret_cast<cudaStream_t *>(&combo.stream), combo.kernelArgs);
    auto *reference2Workspace = combo.Reference2Workspace;
    void *reference2WorkspaceStorage = combo.d_reference2WorkspaceStorage;
    const size_t reference2WorkspaceStorageBytes = combo.reference2WorkspaceStorageBytes;
    Detail::CheckReference2Cuda(cudaMemcpy(&combo,
                                           combo.comboGpu,
                                           sizeof(HpSharkReferenceResults<SharkFloatParams>),
                                           cudaMemcpyDeviceToHost),
                                "cudaMemcpy(Reference2 results D2H)");
    combo.Reference2Workspace = reference2Workspace;
    combo.d_reference2WorkspaceStorage = reference2WorkspaceStorage;
    combo.reference2WorkspaceStorageBytes = reference2WorkspaceStorageBytes;
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
    Detail::CheckReference2Cuda(
        cudaMemcpy(
            &savedRoots, &combo.comboGpu->Multiply.Roots, sizeof(savedRoots), cudaMemcpyDeviceToHost),
        "cudaMemcpy(Reference2 roots D2H)");
    Detail::CheckReference2Cuda(
        cudaMemcpy(combo.comboGpu, &combo, sizeof(combo), cudaMemcpyHostToDevice),
        "cudaMemcpy(Reference2 initial state H2D)");
    Detail::CheckReference2Cuda(
        cudaMemcpy(
            &combo.comboGpu->Multiply.Roots, &savedRoots, sizeof(savedRoots), cudaMemcpyHostToDevice),
        "cudaMemcpy(Reference2 roots H2D)");

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
