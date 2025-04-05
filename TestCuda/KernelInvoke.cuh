#pragma once

#include <functional>

template<class SharkFloatParams>
struct HpSharkFloat;

template<class SharkFloatParams>
struct HpSharkComboResults;

template<class SharkFloatParams>
struct HpSharkAddComboResults;

struct DebugStateRaw;

struct CUstream_st;
using cudaStream_t = CUstream_st *;

class BenchmarkTimer;

enum class Operator;

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

template<class SharkFloatParams, Operator sharkOperator>
void InvokeMultiplyKernelCorrectness(
    BenchmarkTimer &timer,
    HpSharkComboResults<SharkFloatParams> &combo,
    std::vector<DebugStateRaw> *debugResults);

template<class SharkFloatParams, Operator sharkOperator>
void InvokeAddKernelCorrectness(
    BenchmarkTimer &timer,
    HpSharkAddComboResults<SharkFloatParams> &combo,
    std::vector<DebugStateRaw> *debugResults);
