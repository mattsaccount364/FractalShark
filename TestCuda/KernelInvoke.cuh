#pragma once

#include <functional>

template<class SharkFloatParams>
struct HpSharkFloat;

template<class SharkFloatParams>
struct HpSharkComboResults;

template<class SharkFloatParams>
struct HpSharkAddComboResults;

template<class SharkFloatParams>
struct HpSharkReferenceResults;

struct DebugStateRaw;

struct CUstream_st;
using cudaStream_t = CUstream_st *;

class BenchmarkTimer;

class DebugGpuCombo;

enum class Operator;

template<class SharkFloatParams>
void InvokeHpSharkReferenceKernelPerf(
    BenchmarkTimer &timer,
    HpSharkReferenceResults<SharkFloatParams> &combo,
    uint64_t numIters);

template<class SharkFloatParams>
void InvokeMultiplyKernelPerf(
    BenchmarkTimer &timer,
    HpSharkComboResults<SharkFloatParams> &combo,
    uint64_t numIters);

template<class SharkFloatParams>
void InvokeAddKernelPerf(
    BenchmarkTimer &timer,
    HpSharkAddComboResults<SharkFloatParams> &combo,
    uint64_t numIters);

template<class SharkFloatParams>
void InvokeHpSharkReferenceKernelCorrectness(
    BenchmarkTimer &timer,
    HpSharkReferenceResults<SharkFloatParams> &combo,
    DebugGpuCombo *debugCombo);

template<class SharkFloatParams>
void InvokeMultiplyKernelCorrectness(
    BenchmarkTimer &timer,
    HpSharkComboResults<SharkFloatParams> &combo,
    DebugGpuCombo *debugCombo);

template<class SharkFloatParams>
void InvokeAddKernelCorrectness(
    BenchmarkTimer &timer,
    HpSharkAddComboResults<SharkFloatParams> &combo,
    DebugGpuCombo *debugCombo);
