#pragma once

#include "Exceptions.h"
#include "HpSharkFloat.h"
#include "KernelInvoke.h"
#include "LaunchParams.h"

#include <cuda_runtime.h>
#include <memory>
#include <sstream>

namespace HpShark {

namespace Detail {

template <class SharkFloatParams>
__global__ void
HpSharkReference2GpuKernel(HpSharkReferenceResults<SharkFloatParams> *combo)
{
    combo->OutputIterCount = 0;
    combo->PeriodicityStatus = PeriodicityResult::Unknown;
}

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

} // namespace Detail

// Ref2's CUDA entry point deliberately remains incomplete.  It has the same
// invocation contract as the production reference-orbit kernel so test code
// validates its result normally and reports the missing implementation.
template <class SharkFloatParams>
void
InvokeHpSharkReference2Kernel(const HpShark::LaunchParams &launchParams,
                              HpSharkReferenceResults<SharkFloatParams> &combo,
                              uint64_t numIters)
{
    (void)launchParams;
    (void)numIters;

    const auto stream = *reinterpret_cast<cudaStream_t *>(&combo.stream);
    Detail::HpSharkReference2GpuKernel<SharkFloatParams><<<1, 1, 0, stream>>>(combo.comboGpu);
    Detail::CheckReference2Cuda(cudaGetLastError(), "HpSharkReference2GpuKernel launch");
    Detail::CheckReference2Cuda(cudaStreamSynchronize(stream),
                                "HpSharkReference2GpuKernel synchronization");
    Detail::CheckReference2Cuda(cudaMemcpy(&combo,
                                           combo.comboGpu,
                                           sizeof(HpSharkReferenceResults<SharkFloatParams>),
                                           cudaMemcpyDeviceToHost),
                                "HpSharkReference2GpuKernel result copy");
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

    (void)shouldAbort;
    (void)onProgress;
    (void)progressContext;
    (void)progressInterval;

    constexpr int precBits = HpSharkFloat<SharkFloatParams>::DefaultPrecBits;
    typename SharkFloatParams::Float hdrRadiusY{1.0f};

    auto hpCR = std::make_unique<HpSharkFloat<SharkFloatParams>>();
    auto hpCI = std::make_unique<HpSharkFloat<SharkFloatParams>>();
    hpCR->MpfToHpGpu(
        *reinterpret_cast<const mpf_t *>(&cReal[0]), precBits, InjectNoiseInLowOrder::Disable);
    hpCI->MpfToHpGpu(
        *reinterpret_cast<const mpf_t *>(&cImag[0]), precBits, InjectNoiseInLowOrder::Disable);

    GpuOrbitSession<SharkFloatParams> session(externalLaunchParams, hdrRadiusY, *hpCR, *hpCI);
    auto &combo = session.GetCombo();

    if (period > startIter) {
        InvokeHpSharkReference2Kernel(externalLaunchParams, combo, period - startIter);
    }

    combo.Multiply.A.HpGpuToMpf(*reinterpret_cast<mpf_t *>(&outZReal[0]));
    combo.Multiply.B.HpGpuToMpf(*reinterpret_cast<mpf_t *>(&outZImag[0]));
    combo.Multiply.DzdcReal.HpGpuToMpf(*reinterpret_cast<mpf_t *>(&outDzdcReal[0]));
    combo.Multiply.DzdcImag.HpGpuToMpf(*reinterpret_cast<mpf_t *>(&outDzdcImag[0]));
    outD2Real = HDRFloat<double>(combo.d2Real);
    outD2Imag = HDRFloat<double>(combo.d2Imag);

    return startIter;
}

} // namespace HpShark
