#pragma once

#include <mpir.h>
#include <functional>

struct SharkLaunchParams;

template<class SharkFloatParams>
struct HpSharkFloat;

template<class SharkFloatParams>
struct HpSharkComboResults;

template<class SharkFloatParams>
struct HpSharkAddComboResults;

template<class SharkFloatParams>
struct HpSharkReferenceResults;

struct DebugStateRaw;

class BenchmarkTimer;

class DebugGpuCombo;

enum class Operator;

template<class SharkFloatParams>
void InvokeHpSharkReferenceKernelPerf(const SharkLaunchParams &launchParams,
    BenchmarkTimer *timer,
    HpSharkReferenceResults<SharkFloatParams> &combo,
    uint64_t numIters,
    DebugGpuCombo *debugCombo);

template <class SharkFloatParams>
void InvokeHpSharkReferenceKernelProd(const SharkLaunchParams &launchParams,
                                      HpSharkReferenceResults<SharkFloatParams> &combo,
                                      mpf_t srcX,
                                      mpf_t srcY,
                                      uint64_t numIters);

template <class SharkFloatParams>
void InvokeMultiplyNTTKernelPerf(const SharkLaunchParams &launchParams,
                                 BenchmarkTimer &timer,
                              HpSharkComboResults<SharkFloatParams>& combo,
                              uint64_t numIters);


template<class SharkFloatParams>
void InvokeAddKernelPerf(const SharkLaunchParams &launchParams,
                         BenchmarkTimer &timer,
    HpSharkAddComboResults<SharkFloatParams> &combo,
    uint64_t numIters);

template<class SharkFloatParams>
void InvokeHpSharkReferenceKernelCorrectness(const SharkLaunchParams &launchParams,
    BenchmarkTimer &timer,
    HpSharkReferenceResults<SharkFloatParams> &combo,
    DebugGpuCombo *debugCombo);


template <class SharkFloatParams>
void InvokeMultiplyNTTKernelCorrectness(const SharkLaunchParams &launchParams,
                                        BenchmarkTimer &timer,
                                              HpSharkComboResults<SharkFloatParams>& combo,
                                              DebugGpuCombo* debugCombo);

template<class SharkFloatParams>
void InvokeAddKernelCorrectness(const SharkLaunchParams &launchParams,
                                BenchmarkTimer &timer,
    HpSharkAddComboResults<SharkFloatParams> &combo,
    DebugGpuCombo *debugCombo);
