#pragma once

#include "HDRFloat.h"
#include "LaunchParams.h"

#include <cstdint>
#include <functional>
#include <memory>

template <class SharkFloatParams> struct HpSharkFloat;
template <class SharkFloatParams> struct HpSharkReferenceResults;

class BenchmarkTimer;
class DebugGpuCombo;

namespace HpShark {

bool SupportsReferenceSharedOnlyMemory(uint32_t requestedBlocks);

template <class SharkFloatParams> class ReferencePreparedTables;

template <class SharkFloatParams>
std::unique_ptr<ReferencePreparedTables<SharkFloatParams>> PrepareHpSharkReferenceTables(
    const HpShark::LaunchParams &launchParams,
    const HpSharkFloat<SharkFloatParams> &cReal,
    const HpSharkFloat<SharkFloatParams> &cImag,
    uint32_t actualPrecisionLimbs);

template <class SharkFloatParams>
std::unique_ptr<ReferencePreparedTables<SharkFloatParams>> PrepareHpSharkReferenceTables(
    const HpShark::LaunchParams &launchParams,
    const HpSharkFloat<SharkFloatParams> &cReal,
    const HpSharkFloat<SharkFloatParams> &cImag,
    uint32_t actualPrecisionLimbs,
    uint32_t minFusedStages,
    uint32_t maxFusedStages);

template <class SharkFloatParams>
std::unique_ptr<ReferencePreparedTables<SharkFloatParams>> PrepareHpSharkReferenceTables(
    const HpShark::LaunchParams &launchParams,
    const mpf_t cReal,
    const mpf_t cImag,
    uint32_t actualPrecisionLimbs);

template <class SharkFloatParams>
std::unique_ptr<ReferencePreparedTables<SharkFloatParams>> PrepareHpSharkReferenceTables(
    const HpShark::LaunchParams &launchParams,
    const mpf_t cReal,
    const mpf_t cImag,
    uint32_t actualPrecisionLimbs,
    uint32_t minFusedStages,
    uint32_t maxFusedStages);

template <class SharkFloatParams>
std::unique_ptr<HpSharkReferenceResults<SharkFloatParams>> InitHpSharkReferenceKernel(
    const HpShark::LaunchParams &launchParams,
    typename SharkFloatParams::Float hdrRadiusY,
    const mpf_t srcX,
    const mpf_t srcY,
    uint32_t actualPrecisionLimbs);

template <class SharkFloatParams>
std::unique_ptr<HpSharkReferenceResults<SharkFloatParams>> InitHpSharkReferenceKernel(
    const HpShark::LaunchParams &launchParams,
    typename SharkFloatParams::Float hdrRadiusY,
    const HpSharkFloat<SharkFloatParams> &xNum,
    const HpSharkFloat<SharkFloatParams> &yNum,
    uint32_t actualPrecisionLimbs);

template <class SharkFloatParams>
std::unique_ptr<HpSharkReferenceResults<SharkFloatParams>> InitHpSharkReferenceKernel(
    const HpShark::LaunchParams &launchParams,
    typename SharkFloatParams::Float hdrRadiusY,
    const HpSharkFloat<SharkFloatParams> &xNum,
    const HpSharkFloat<SharkFloatParams> &yNum,
    ReferencePreparedTables<SharkFloatParams> &preparedTables);

template <class SharkFloatParams>
void InvokeHpSharkReferenceKernel(const HpShark::LaunchParams &launchParams,
                                  HpSharkReferenceResults<SharkFloatParams> &results,
                                  uint64_t numIters);

template <class SharkFloatParams>
void ShutdownHpSharkReferenceKernel(const HpShark::LaunchParams &launchParams,
                                    HpSharkReferenceResults<SharkFloatParams> &results,
                                    DebugGpuCombo *debugResults);

template <class SharkFloatParams> class GpuOrbitSession {
    std::unique_ptr<HpSharkReferenceResults<SharkFloatParams>> m_Results;
    HpShark::LaunchParams m_LaunchParams;
    DebugGpuCombo *m_DebugResults;

public:
    GpuOrbitSession(const HpShark::LaunchParams &launchParams,
                    typename SharkFloatParams::Float hdrRadiusY,
                    const mpf_t srcX,
                    const mpf_t srcY,
                    uint32_t actualPrecisionLimbs,
                    DebugGpuCombo *debugResults)
        : m_Results{InitHpSharkReferenceKernel<SharkFloatParams>(
              launchParams, hdrRadiusY, srcX, srcY, actualPrecisionLimbs)},
          m_LaunchParams{launchParams}, m_DebugResults{debugResults}
    {
    }

    GpuOrbitSession(const HpShark::LaunchParams &launchParams,
                    typename SharkFloatParams::Float hdrRadiusY,
                    const HpSharkFloat<SharkFloatParams> &xNum,
                    const HpSharkFloat<SharkFloatParams> &yNum,
                    ReferencePreparedTables<SharkFloatParams> &preparedTables,
                    DebugGpuCombo *debugResults)
        : m_Results{InitHpSharkReferenceKernel<SharkFloatParams>(
              launchParams, hdrRadiusY, xNum, yNum, preparedTables)},
          m_LaunchParams{launchParams}, m_DebugResults{debugResults}
    {
    }

    GpuOrbitSession(const HpShark::LaunchParams &launchParams,
                    typename SharkFloatParams::Float hdrRadiusY,
                    const HpSharkFloat<SharkFloatParams> &xNum,
                    const HpSharkFloat<SharkFloatParams> &yNum,
                    uint32_t actualPrecisionLimbs,
                    DebugGpuCombo *debugResults)
        : m_Results{InitHpSharkReferenceKernel<SharkFloatParams>(
              launchParams, hdrRadiusY, xNum, yNum, actualPrecisionLimbs)},
          m_LaunchParams{launchParams}, m_DebugResults{debugResults}
    {
    }

    ~GpuOrbitSession()
    {
        ShutdownHpSharkReferenceKernel<SharkFloatParams>(m_LaunchParams, *m_Results, m_DebugResults);
    }

    GpuOrbitSession(const GpuOrbitSession &) = delete;
    GpuOrbitSession &operator=(const GpuOrbitSession &) = delete;
    GpuOrbitSession(GpuOrbitSession &&) = delete;
    GpuOrbitSession &operator=(GpuOrbitSession &&) = delete;

    void
    InvokeChunk(uint64_t numIters)
    {
        InvokeHpSharkReferenceKernel<SharkFloatParams>(m_LaunchParams, *m_Results, numIters);
    }

    HpSharkReferenceResults<SharkFloatParams> &
    GetResults()
    {
        return *m_Results;
    }

    const HpSharkReferenceResults<SharkFloatParams> &
    GetResults() const
    {
        return *m_Results;
    }
};

template <class SharkFloatParams>
void InvokeHpSharkReferenceKernelCorrectness(const HpShark::LaunchParams &launchParams,
                                             BenchmarkTimer &timer,
                                             HpSharkReferenceResults<SharkFloatParams> &results,
                                             DebugGpuCombo *debugResults);

template <class SharkFloatParams>
void InvokeHpSharkReferenceKernelCorrectness(const HpShark::LaunchParams &launchParams,
                                             BenchmarkTimer &timer,
                                             HpSharkReferenceResults<SharkFloatParams> &results,
                                             DebugGpuCombo *debugResults,
                                             ReferencePreparedTables<SharkFloatParams> &preparedTables);

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
                                            const HpShark::LaunchParams &externalLaunchParams,
                                            ReferencePreparedTables<SharkFloatParams> *preparedTables,
                                            uint64_t startIter,
                                            bool (*shouldAbort)(),
                                            void (*onProgress)(uint64_t, void *),
                                            void *progressContext,
                                            uint64_t progressInterval);

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
                                            const HpShark::LaunchParams &externalLaunchParams,
                                            uint32_t actualPrecisionLimbs,
                                            uint64_t startIter,
                                            bool (*shouldAbort)(),
                                            void (*onProgress)(uint64_t, void *),
                                            void *progressContext,
                                            uint64_t progressInterval);

} // namespace HpShark
