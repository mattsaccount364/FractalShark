#pragma once

#include <functional>

template<class SharkFloatParams>
struct HpSharkFloat;

template<class SharkFloatParams>
struct HpSharkComboResults;

struct DebugStateRaw;

struct CUstream_st;
using cudaStream_t = CUstream_st *;

class BenchmarkTimer;

enum class Operator;

// Structure to hold carry information for each block
struct CarryInfo {
    uint32_t carryOut;    // Carry-out from the block's computation
};

struct GlobalAddBlockData {
    int32_t AIsBiggerMagnitude;
};

template<class SharkFloatParams>
void InvokeMultiplyKernel(
    BenchmarkTimer &timer,
    std::function<void(cudaStream_t &, void *[])> kernel,
    HpSharkComboResults<SharkFloatParams> &combo,
    uint64_t numIters);

template<class SharkFloatParams>
void InvokeAddKernel(
    BenchmarkTimer &timer,
    std::function<void(void *[])> kernel,
    const HpSharkFloat<SharkFloatParams> &xNum,
    const HpSharkFloat<SharkFloatParams> &yNum,
    HpSharkFloat<SharkFloatParams> &gpuResult2,
    uint64_t numIters);

template<class SharkFloatParams, Operator sharkOperator>
void InvokeMultiplyKernelCorrectness(
    BenchmarkTimer &timer,
    std::function<void(void *[])> kernel,
    HpSharkComboResults<SharkFloatParams> &combo,
    std::vector<DebugStateRaw> *debugResults);

template<class SharkFloatParams, Operator sharkOperator>
void InvokeAddKernelCorrectness(
    BenchmarkTimer &timer,
    std::function<void(void *[])> kernel,
    const HpSharkFloat<SharkFloatParams> &xNum,
    const HpSharkFloat<SharkFloatParams> &yNum,
    HpSharkFloat<SharkFloatParams> &gpuResult);
