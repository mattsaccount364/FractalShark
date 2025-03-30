#include <cuda_runtime.h>

#include "BenchmarkTimer.h"
#include "TestTracker.h"

#include "Tests.h"
#include "HpSharkFloat.cuh"
#include "Add.cuh"
#include "Multiply.cuh"
#include "ReferenceKaratsuba.h"
#include "DebugChecksumHost.h"

#include <iostream>
#include <vector>
#include <gmp.h>
#include <cstring>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <assert.h>


template<class SharkFloatParams>
void InvokeAddKernelPerf(
    BenchmarkTimer &timer,
    std::function<void(cudaStream_t &, void *[])> kernel,
    HpSharkComboResults<SharkFloatParams> &combo,
    uint64_t numIters) {

    // Prepare kernel arguments
    // Allocate memory for carryOuts and cumulativeCarries
    uint64_t *d_tempProducts;
    constexpr auto BytesToAllocate =
        (AdditionalUInt64Global + ScratchMemoryCopies * CalculateFrameSize<SharkFloatParams>()) * sizeof(uint64_t);
    cudaMalloc(&d_tempProducts, BytesToAllocate);

    HpSharkComboResults<SharkFloatParams> *comboGpu;
    cudaMalloc(&comboGpu, sizeof(HpSharkComboResults<SharkFloatParams>));
    cudaMemcpy(comboGpu, &combo, sizeof(HpSharkComboResults<SharkFloatParams>), cudaMemcpyHostToDevice);

    void *kernelArgs[] = {
        (void *)&comboGpu,
        (void *)&numIters,
        (void *)&d_tempProducts
    };

    cudaStream_t stream = nullptr;

    if constexpr (SharkCustomStream) {
        cudaStreamCreate(&stream); // Create a stream
    }

    cudaDeviceProp prop;
    int device_id = 0;

    if constexpr (SharkCustomStream) {
        cudaGetDeviceProperties(&prop, device_id);
        cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, prop.persistingL2CacheMaxSize); /* Set aside max possible size of L2 cache for persisting accesses */

        auto setAccess = [&](void *ptr, size_t num_bytes) {
            cudaStreamAttrValue stream_attribute;                                         // Stream level attributes data structure
            stream_attribute.accessPolicyWindow.base_ptr = reinterpret_cast<void *>(ptr); // Global Memory data pointer
            stream_attribute.accessPolicyWindow.num_bytes = num_bytes;                    // Number of bytes for persisting accesses.
            // (Must be less than cudaDeviceProp::accessPolicyMaxWindowSize)
            stream_attribute.accessPolicyWindow.hitRatio = 1.0;                          // Hint for L2 cache hit ratio for persisting accesses in the num_bytes region
            stream_attribute.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting; // Type of access property on cache hit
            stream_attribute.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;  // Type of access property on cache miss.

            //Set the attributes to a CUDA stream of type cudaStream_t
            cudaError_t err = cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);
            if (err != cudaSuccess) {
                std::cerr << "CUDA error in setting stream attribute: " << cudaGetErrorString(err) << std::endl;
            }
            };

        setAccess(comboGpu, sizeof(HpSharkComboResults<SharkFloatParams>));
        setAccess(d_tempProducts, 32 * SharkFloatParams::GlobalNumUint32 * sizeof(uint64_t));
    }

    {
        ScopedBenchmarkStopper stopper{ timer };
        kernel(stream, kernelArgs);
    }

    cudaMemcpy(&combo, comboGpu, sizeof(HpSharkComboResults<SharkFloatParams>), cudaMemcpyDeviceToHost);

    if constexpr (SharkCustomStream) {
        cudaStreamDestroy(stream); // Destroy the stream
    }

    cudaFree(comboGpu);
    cudaFree(d_tempProducts);
}

template<class SharkFloatParams>
void InvokeAddKernelPerf(
    BenchmarkTimer &timer,
    std::function<void(void *[])> kernel,
    const HpSharkFloat<SharkFloatParams> &xNum,
    const HpSharkFloat<SharkFloatParams> &yNum,
    HpSharkFloat<SharkFloatParams> &gpuResult2,
    uint64_t numIters) {

    // Perform the calculation on the GPU
    HpSharkFloat<SharkFloatParams> *xGpu;
    cudaMalloc(&xGpu, sizeof(HpSharkFloat<SharkFloatParams>));
    cudaMemcpy(xGpu, &xNum, sizeof(HpSharkFloat<SharkFloatParams>), cudaMemcpyHostToDevice);

    HpSharkFloat<SharkFloatParams> *yGpu;
    cudaMalloc(&yGpu, sizeof(HpSharkFloat<SharkFloatParams>));
    cudaMemcpy(yGpu, &yNum, sizeof(HpSharkFloat<SharkFloatParams>), cudaMemcpyHostToDevice);

    HpSharkFloat<SharkFloatParams> *internalGpuResult2;
    cudaMalloc(&internalGpuResult2, sizeof(HpSharkFloat<SharkFloatParams>));
    cudaMemset(internalGpuResult2, 0, sizeof(HpSharkFloat<SharkFloatParams>));

    uint32_t *g_extResult;
    cudaMalloc(&g_extResult, (SharkFloatParams::GlobalNumUint32 + 2) * sizeof(uint32_t));

    // Prepare kernel arguments
    void *kernelArgs[] = {
        (void *)&xGpu,
        (void *)&yGpu,
        (void *)&internalGpuResult2,
        (void *)&numIters,
        (void *)&g_extResult
    };

    // Launch the cooperative kernel
    {
        ScopedBenchmarkStopper stopper{ timer };
        kernel(kernelArgs);
    }

    cudaMemcpy(&gpuResult2, internalGpuResult2, sizeof(HpSharkFloat<SharkFloatParams>), cudaMemcpyDeviceToHost);

    // Free memory
    cudaFree(g_extResult);
    cudaFree(internalGpuResult2);
    cudaFree(yGpu);
    cudaFree(xGpu);
}

template<class SharkFloatParams, Operator sharkOperator>
void InvokeMultiplyKernelCorrectness(
    BenchmarkTimer &timer,
    std::function<void(void *[])> kernel,
    HpSharkComboResults<SharkFloatParams> &combo,
    std::vector<DebugStateRaw> *debugResults) {

    static constexpr bool DebugInitCudaMemory = true;

    // Prepare kernel arguments
    // Allocate memory for carryOuts and cumulativeCarries
    uint64_t *d_tempProducts;
    constexpr auto BytesToAllocate =
        (AdditionalUInt64Global + ScratchMemoryCopies * CalculateFrameSize<SharkFloatParams>()) * sizeof(uint64_t);
    cudaMalloc(&d_tempProducts, BytesToAllocate);

    if constexpr (!DebugInitCudaMemory) {
        cudaMemset(d_tempProducts, 0, BytesToAllocate);
    } else {
        cudaMemset(d_tempProducts, 0xCD, BytesToAllocate);
    }

    HpSharkComboResults<SharkFloatParams> *comboGpu;
    cudaMalloc(&comboGpu, sizeof(HpSharkComboResults<SharkFloatParams>));
    cudaMemcpy(comboGpu, &combo, sizeof(HpSharkComboResults<SharkFloatParams>), cudaMemcpyHostToDevice);

    if constexpr (!DebugInitCudaMemory) {
        cudaMemset(&comboGpu->ResultX2, 0, sizeof(HpSharkFloat<SharkFloatParams>));
        cudaMemset(&comboGpu->ResultXY, 0, sizeof(HpSharkFloat<SharkFloatParams>));
        cudaMemset(&comboGpu->ResultY2, 0, sizeof(HpSharkFloat<SharkFloatParams>));
    } else {
        cudaMemset(&comboGpu->ResultX2, 0xCD, sizeof(HpSharkFloat<SharkFloatParams>));
        cudaMemset(&comboGpu->ResultXY, 0xCD, sizeof(HpSharkFloat<SharkFloatParams>));
        cudaMemset(&comboGpu->ResultY2, 0xCD, sizeof(HpSharkFloat<SharkFloatParams>));
    }

    void *kernelArgs[] = {
        (void *)&comboGpu,
        (void *)&d_tempProducts
    };

    {
        ScopedBenchmarkStopper stopper{ timer };
        kernel(kernelArgs);
    }

    cudaMemcpy(&combo, comboGpu, sizeof(HpSharkComboResults<SharkFloatParams>), cudaMemcpyDeviceToHost);

    if (debugResults != nullptr) {
        if constexpr (SharkDebugChecksums) {
            debugResults->resize(SharkFloatParams::NumDebugStates);
            cudaMemcpy(
                debugResults->data(),
                &d_tempProducts[AdditionalGlobalSyncSpace],
                SharkFloatParams::NumDebugStates * sizeof(DebugStateRaw),
                cudaMemcpyDeviceToHost);
        }
    }

    cudaFree(comboGpu);
    cudaFree(d_tempProducts);
}

template<class SharkFloatParams, Operator sharkOperator>
void InvokeAddKernelCorrectness(
    BenchmarkTimer &timer,
    std::function<void(void *[])> kernel,
    const HpSharkFloat<SharkFloatParams> &xNum,
    const HpSharkFloat<SharkFloatParams> &yNum,
    HpSharkFloat<SharkFloatParams> &gpuResult,
    std::vector<DebugStateRaw> *debugResults) {

    // Perform the calculation on the GPU
    HpSharkFloat<SharkFloatParams> *xGpu;
    cudaMalloc(&xGpu, sizeof(HpSharkFloat<SharkFloatParams>));
    cudaMemcpy(xGpu, &xNum, sizeof(HpSharkFloat<SharkFloatParams>), cudaMemcpyHostToDevice);

    HpSharkFloat<SharkFloatParams> *yGpu;
    cudaMalloc(&yGpu, sizeof(HpSharkFloat<SharkFloatParams>));
    cudaMemcpy(yGpu, &yNum, sizeof(HpSharkFloat<SharkFloatParams>), cudaMemcpyHostToDevice);

    HpSharkFloat<SharkFloatParams> *internalGpuResult;
    cudaMalloc(&internalGpuResult, sizeof(HpSharkFloat<SharkFloatParams>));
    cudaMemset(internalGpuResult, 0, sizeof(HpSharkFloat<SharkFloatParams>));

    constexpr auto BytesToAllocate =
        (AdditionalUInt64Global + SharkFloatParams::GlobalNumUint32 * 4) * sizeof(uint32_t);
    uint32_t *g_extResult;
    cudaMalloc(&g_extResult, BytesToAllocate);

    // Prepare kernel arguments
    void *kernelArgs[] = {
        (void *)&xGpu,
        (void *)&yGpu,
        (void *)&internalGpuResult,
        (void *)&g_extResult
    };

    {
        ScopedBenchmarkStopper stopper{ timer };
        ComputeAddGpu<SharkFloatParams>(kernelArgs);
    }

    cudaMemcpy(&gpuResult, internalGpuResult, sizeof(HpSharkFloat<SharkFloatParams>), cudaMemcpyDeviceToHost);

    if (debugResults != nullptr) {
        if constexpr (SharkDebugChecksums) {
            debugResults->resize(SharkFloatParams::NumDebugStates);
            cudaMemcpy(
                debugResults->data(),
                &g_extResult[AdditionalGlobalSyncSpace],
                SharkFloatParams::NumDebugStates * sizeof(DebugStateRaw),
                cudaMemcpyDeviceToHost);
        }
    }

    cudaFree(g_extResult);
    cudaFree(internalGpuResult);
    cudaFree(yGpu);
    cudaFree(xGpu);
}

#define ExplicitlyInstantiate(SharkFloatParams) \
    template void InvokeAddKernelPerf<SharkFloatParams>( \
        BenchmarkTimer &timer, \
        std::function<void(cudaStream_t &, void *[])> kernel, \
        HpSharkComboResults<SharkFloatParams> &combo, \
        uint64_t numIters); \
    template void InvokeAddKernelPerf<SharkFloatParams>( \
        BenchmarkTimer &timer, \
        std::function<void(void*[])> kernel, \
        const HpSharkFloat<SharkFloatParams> &xNum, \
        const HpSharkFloat<SharkFloatParams> &yNum, \
        HpSharkFloat<SharkFloatParams> &gpuResult2, \
        uint64_t numIters); \
    template void InvokeMultiplyKernelCorrectness<SharkFloatParams, Operator::MultiplyKaratsubaV2>( \
        BenchmarkTimer &timer, \
        std::function<void(void*[])> kernel, \
        HpSharkComboResults<SharkFloatParams> &combo, \
        std::vector<DebugStateRaw> *debugResults); \
    template void InvokeAddKernelCorrectness<SharkFloatParams, Operator::Add>( \
        BenchmarkTimer &timer, \
        std::function<void(void*[])> kernel, \
        const HpSharkFloat<SharkFloatParams> &xNum, \
        const HpSharkFloat<SharkFloatParams> &yNum, \
        HpSharkFloat<SharkFloatParams> &gpuResult, \
        std::vector<DebugStateRaw> *debugResults);

#ifdef SHARK_INCLUDE_KERNELS
ExplicitInstantiateAll();
#endif