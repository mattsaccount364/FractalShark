#pragma once

#include "Environment.h"
#include <functional>
#include <memory>

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

template <class SharkFloatParams> class Reference2PreparedTables;

template <class SharkFloatParams>
std::unique_ptr<Reference2PreparedTables<SharkFloatParams>> PrepareHpSharkReference2Tables(
    const HpShark::LaunchParams &launchParams,
    const HpSharkFloat<SharkFloatParams> &cReal,
    const HpSharkFloat<SharkFloatParams> &cImag,
    uint32_t actualPrecisionLimbs);

template <class SharkFloatParams>
std::unique_ptr<Reference2PreparedTables<SharkFloatParams>> PrepareHpSharkReference2Tables(
    const HpShark::LaunchParams &launchParams,
    const HpSharkFloat<SharkFloatParams> &cReal,
    const HpSharkFloat<SharkFloatParams> &cImag,
    uint32_t actualPrecisionLimbs,
    uint32_t minFusedStages,
    uint32_t maxFusedStages);

template <class SharkFloatParams>
std::unique_ptr<Reference2PreparedTables<SharkFloatParams>> PrepareHpSharkReference2Tables(
    const HpShark::LaunchParams &launchParams,
    const mpf_t cReal,
    const mpf_t cImag,
    uint32_t actualPrecisionLimbs);

template <class SharkFloatParams>
std::unique_ptr<Reference2PreparedTables<SharkFloatParams>> PrepareHpSharkReference2Tables(
    const HpShark::LaunchParams &launchParams,
    const mpf_t cReal,
    const mpf_t cImag,
    uint32_t actualPrecisionLimbs,
    uint32_t minFusedStages,
    uint32_t maxFusedStages);

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
std::unique_ptr<HpSharkReferenceResults<SharkFloatParams>> InitHpSharkReference2Kernel(
    const HpShark::LaunchParams &launchParams,
    const typename SharkFloatParams::Float hdrRadiusY,
    const mpf_t srcX,
    const mpf_t srcY,
    uint32_t actualPrecisionLimbs);

template <class SharkFloatParams>
std::unique_ptr<HpSharkReferenceResults<SharkFloatParams>> InitHpSharkReference2Kernel(
    const HpShark::LaunchParams &launchParams,
    const typename SharkFloatParams::Float hdrRadiusY,
    const HpSharkFloat<SharkFloatParams> &xNum,
    const HpSharkFloat<SharkFloatParams> &yNum,
    uint32_t actualPrecisionLimbs);

template <class SharkFloatParams>
std::unique_ptr<HpSharkReferenceResults<SharkFloatParams>> InitHpSharkReference2Kernel(
    const HpShark::LaunchParams &launchParams,
    const typename SharkFloatParams::Float hdrRadiusY,
    const HpSharkFloat<SharkFloatParams> &xNum,
    const HpSharkFloat<SharkFloatParams> &yNum,
    Reference2PreparedTables<SharkFloatParams> &preparedTables);

template <class SharkFloatParams>
void InvokeHpSharkReferenceKernel(const HpShark::LaunchParams &launchParams,
                                  HpSharkReferenceResults<SharkFloatParams> &combo,
                                  uint64_t numIters);

template <class SharkFloatParams>
void InvokeHpSharkReference2Kernel(const HpShark::LaunchParams &launchParams,
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

template <class SharkFloatParams>
void ShutdownHpSharkReference2Kernel(const HpShark::LaunchParams &launchParams,
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

// Ref2 uses an additional fixed-capacity fused-NTT workspace. A session can
// either own an internally prepared workspace or borrow one prepared by its caller.
template <class SharkFloatParams> class GpuOrbitSession2 {
    std::unique_ptr<HpSharkReferenceResults<SharkFloatParams>> m_Combo;
    HpShark::LaunchParams m_LaunchParams;
    DebugGpuCombo *m_DebugCombo;

public:
    GpuOrbitSession2(const HpShark::LaunchParams &launchParams,
                     typename SharkFloatParams::Float hdrRadiusY,
                     const mpf_t srcX,
                     const mpf_t srcY,
                     uint32_t actualPrecisionLimbs,
                     DebugGpuCombo *debugCombo = nullptr)
        : m_Combo{InitHpSharkReference2Kernel<SharkFloatParams>(
              launchParams, hdrRadiusY, srcX, srcY, actualPrecisionLimbs)},
          m_LaunchParams{launchParams}, m_DebugCombo{debugCombo}
    {
    }

    GpuOrbitSession2(const HpShark::LaunchParams &launchParams,
                     typename SharkFloatParams::Float hdrRadiusY,
                     const HpSharkFloat<SharkFloatParams> &xNum,
                     const HpSharkFloat<SharkFloatParams> &yNum,
                     Reference2PreparedTables<SharkFloatParams> &preparedTables,
                     DebugGpuCombo *debugCombo = nullptr)
        : m_Combo{InitHpSharkReference2Kernel<SharkFloatParams>(
              launchParams, hdrRadiusY, xNum, yNum, preparedTables)},
          m_LaunchParams{launchParams}, m_DebugCombo{debugCombo}
    {
    }

    GpuOrbitSession2(const HpShark::LaunchParams &launchParams,
                     typename SharkFloatParams::Float hdrRadiusY,
                     const HpSharkFloat<SharkFloatParams> &xNum,
                     const HpSharkFloat<SharkFloatParams> &yNum,
                     uint32_t actualPrecisionLimbs,
                     DebugGpuCombo *debugCombo = nullptr)
        : m_Combo{InitHpSharkReference2Kernel<SharkFloatParams>(
              launchParams, hdrRadiusY, xNum, yNum, actualPrecisionLimbs)},
          m_LaunchParams{launchParams}, m_DebugCombo{debugCombo}
    {
    }

    ~GpuOrbitSession2()
    {
        ShutdownHpSharkReference2Kernel<SharkFloatParams>(m_LaunchParams, *m_Combo, m_DebugCombo);
    }

    GpuOrbitSession2(const GpuOrbitSession2 &) = delete;
    GpuOrbitSession2 &operator=(const GpuOrbitSession2 &) = delete;
    GpuOrbitSession2(GpuOrbitSession2 &&) = delete;
    GpuOrbitSession2 &operator=(GpuOrbitSession2 &&) = delete;

    void
    InvokeChunk(uint64_t numIters)
    {
        InvokeHpSharkReference2Kernel<SharkFloatParams>(m_LaunchParams, *m_Combo, numIters);
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
void InvokeHpSharkReference2KernelCorrectness(const HpShark::LaunchParams &launchParams,
                                              BenchmarkTimer &timer,
                                              HpSharkReferenceResults<SharkFloatParams> &combo,
                                              DebugGpuCombo *debugCombo);

template <class SharkFloatParams>
void InvokeHpSharkReference2KernelCorrectness(
    const HpShark::LaunchParams &launchParams,
    BenchmarkTimer &timer,
    HpSharkReferenceResults<SharkFloatParams> &combo,
    DebugGpuCombo *debugCombo,
    Reference2PreparedTables<SharkFloatParams> &preparedTables);

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

// Serial CUDA Ref2 counterpart to the Ref1 Newton/derivative GPU interface.
// It preserves the Ref1 invocation/checkpoint contract while using Ref2's
// fixed-capacity fused NTT workspace internally.
template <class SharkFloatParams>
uint64_t EvaluateCriticalOrbitAndDerivs2_GPU(
    const mpf_t cReal,
    const mpf_t cImag,
    uint64_t period,
    mpf_t outZReal,
    mpf_t outZImag,
    mpf_t outDzdcReal,
    mpf_t outDzdcImag,
    HDRFloat<double> &outD2Real,
    HDRFloat<double> &outD2Imag,
    const HpShark::LaunchParams &externalLaunchParams = {0, 0},
    Reference2PreparedTables<SharkFloatParams> *preparedTables = nullptr,
    uint64_t startIter = 0,
    bool (*shouldAbort)() = nullptr,
    void (*onProgress)(uint64_t, void *) = nullptr,
    void *progressContext = nullptr,
    uint64_t progressInterval = 64);

// Ref2 overload that prepares a precision-window table internally. The
// caller supplies the effective precision in limbs after selecting a storage
// bucket; this keeps the prepared-table implementation inside the CUDA library
// while allowing host-side dispatch code to use the same precision logic.
template <class SharkFloatParams>
uint64_t EvaluateCriticalOrbitAndDerivs2_GPU(const mpf_t cReal,
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
                                             uint64_t progressInterval);

} // namespace HpShark
