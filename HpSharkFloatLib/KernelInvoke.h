#pragma once

#include <functional>
#include <memory>
#include <mpir.h>

namespace HpShark {
struct LaunchParams;
}

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

} // namespace HpShark
