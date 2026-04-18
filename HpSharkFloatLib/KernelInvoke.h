#pragma once

#include <functional>
#include <memory>
#include <mpir.h>

#include "HDRFloat.h"
#include "LaunchParams.h"

template <class SharkFloatParams> struct HpSharkFloat;

template <class SharkFloatParams> struct HpSharkComboResults;

template <class SharkFloatParams> struct HpSharkAddComboResults;

template <class SharkFloatParams> struct HpSharkReferenceResults;

struct DebugStateRaw;

class BenchmarkTimer;

class DebugGpuCombo;

enum class Operator;

namespace HpShark {

template <class SharkFloatParams>
std::unique_ptr<HpSharkReferenceResults<SharkFloatParams>> InitHpSharkReferenceKernel(
    const HpShark::LaunchParams &launchParams,
    const typename SharkFloatParams::Float hdrRadiusY,
    const mpf_t srcX,
    const mpf_t srcY);

template <class SharkFloatParams>
std::unique_ptr<HpSharkReferenceResults<SharkFloatParams>> InitHpSharkReferenceKernel(
    const HpShark::LaunchParams &launchParams,
    const typename SharkFloatParams::Float hdrRadiusY,
    const HpSharkFloat<SharkFloatParams> &xNum,
    const HpSharkFloat<SharkFloatParams> &yNum);

template <class SharkFloatParams>
void InvokeHpSharkReferenceKernel(const HpShark::LaunchParams &launchParams,
                                  HpSharkReferenceResults<SharkFloatParams> &combo,
                                  uint64_t numIters);

template <class SharkFloatParams>
void InitHpSharkKernelProd(const HpShark::LaunchParams &launchParams,
                           HpSharkReferenceResults<SharkFloatParams> &combo,
                           mpf_t srcX,
                           mpf_t srcY,
                           uint64_t numIters,
                           DebugGpuCombo *debugCombo);

template <class SharkFloatParams>
void ShutdownHpSharkReferenceKernel(const HpShark::LaunchParams &launchParams,
                                    HpSharkReferenceResults<SharkFloatParams> &combo,
                                    DebugGpuCombo *debugCombo);

// RAII wrapper for the GPU reference orbit lifecycle (Init/Invoke/Shutdown).
// Ensures GPU resources (device memory, CUDA stream, NTT root tables) are
// always cleaned up, even if an exception is thrown during the chunk loop.
template <class SharkFloatParams> class GpuOrbitSession {
    std::unique_ptr<HpSharkReferenceResults<SharkFloatParams>> m_Combo;
    HpShark::LaunchParams m_LaunchParams;
    DebugGpuCombo *m_DebugCombo;

public:
    GpuOrbitSession(const HpShark::LaunchParams &launchParams,
                    typename SharkFloatParams::Float hdrRadiusY,
                    const mpf_t srcX,
                    const mpf_t srcY,
                    DebugGpuCombo *debugCombo = nullptr)
        : m_Combo{InitHpSharkReferenceKernel<SharkFloatParams>(launchParams, hdrRadiusY, srcX, srcY)},
          m_LaunchParams{launchParams}, m_DebugCombo{debugCombo}
    {
    }

    GpuOrbitSession(const HpShark::LaunchParams &launchParams,
                    typename SharkFloatParams::Float hdrRadiusY,
                    const HpSharkFloat<SharkFloatParams> &xNum,
                    const HpSharkFloat<SharkFloatParams> &yNum,
                    DebugGpuCombo *debugCombo = nullptr)
        : m_Combo{InitHpSharkReferenceKernel<SharkFloatParams>(launchParams, hdrRadiusY, xNum, yNum)},
          m_LaunchParams{launchParams}, m_DebugCombo{debugCombo}
    {
    }

    ~GpuOrbitSession()
    {
        ShutdownHpSharkReferenceKernel<SharkFloatParams>(m_LaunchParams, *m_Combo, m_DebugCombo);
    }

    GpuOrbitSession(const GpuOrbitSession &) = delete;
    GpuOrbitSession &operator=(const GpuOrbitSession &) = delete;
    GpuOrbitSession(GpuOrbitSession &&) = delete;
    GpuOrbitSession &operator=(GpuOrbitSession &&) = delete;

    void
    InvokeChunk(uint64_t numIters)
    {
        InvokeHpSharkReferenceKernel<SharkFloatParams>(m_LaunchParams, *m_Combo, numIters);
    }

    HpSharkReferenceResults<SharkFloatParams> &
    GetCombo()
    {
        return *m_Combo;
    }

    const HpSharkReferenceResults<SharkFloatParams> &
    GetCombo() const
    {
        return *m_Combo;
    }
};

template <class SharkFloatParams>
void InvokeMultiplyNTTKernelPerf(const HpShark::LaunchParams &launchParams,
                                 BenchmarkTimer &timer,
                                 HpSharkComboResults<SharkFloatParams> &combo,
                                 uint64_t numIters);

template <class SharkFloatParams>
void InvokeAddKernelPerf(const HpShark::LaunchParams &launchParams,
                         BenchmarkTimer &timer,
                         HpSharkAddComboResults<SharkFloatParams> &combo,
                         uint64_t numIters);

template <class SharkFloatParams>
void InvokeHpSharkReferenceKernelCorrectness(const HpShark::LaunchParams &launchParams,
                                             BenchmarkTimer &timer,
                                             HpSharkReferenceResults<SharkFloatParams> &combo,
                                             DebugGpuCombo *debugCombo);

template <class SharkFloatParams>
void InvokeMultiplyNTTKernelCorrectness(const HpShark::LaunchParams &launchParams,
                                        BenchmarkTimer &timer,
                                        HpSharkComboResults<SharkFloatParams> &combo,
                                        DebugGpuCombo *debugCombo);

template <class SharkFloatParams>
void InvokeAddKernelCorrectness(const HpShark::LaunchParams &launchParams,
                                BenchmarkTimer &timer,
                                HpSharkAddComboResults<SharkFloatParams> &combo,
                                DebugGpuCombo *debugCombo);

// GPU-accelerated drop-in replacement for EvaluateCriticalOrbitAndDerivs.
// When startIter > 0, reads initial z/dzdc/d2 from the out parameters (caller
// must populate them from checkpoint). Runs period - startIter iterations in chunks
// with host-side abort check between chunks.
// onProgress is called every progressInterval chunks with (itersCompleted, progressContext).
// Returns total iterations completed (== period if finished, < period if aborted).
template <class SharkFloatParams>
uint64_t EvaluateCriticalOrbitAndDerivs_GPU(const mpf_t cReal,
                                            const mpf_t cImag,
                                            uint64_t period,
                                            mpf_t outZReal,
                                            mpf_t outZImag,
                                            mpf_t outDzdcReal,
                                            mpf_t outDzdcImag,
                                            HDRFloat<double> &outD2Real,
                                            HDRFloat<double> &outD2Imag,
                                            const HpShark::LaunchParams &externalLaunchParams = {0, 0},
                                            uint64_t startIter = 0,
                                            bool (*shouldAbort)() = nullptr,
                                            void (*onProgress)(uint64_t, void *) = nullptr,
                                            void *progressContext = nullptr,
                                            uint64_t progressInterval = 64);

} // namespace HpShark
