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
void InvokeMultiplyKernel(
    BenchmarkTimer &timer,
    std::function<void(cudaStream_t &, void *[])> kernel,
    const HpSharkFloat<SharkFloatParams> &xNum,
    const HpSharkFloat<SharkFloatParams> &yNum,
    HpSharkFloat<SharkFloatParams> &gpuResult2) {

    // Prepare kernel arguments
    // Allocate memory for carryOuts and cumulativeCarries
    uint64_t *d_tempProducts;
    constexpr auto BytesToAllocate =
        (AdditionalUInt64Global + ScratchMemoryCopies * CalculateFrameSize<SharkFloatParams>()) * sizeof(uint64_t);
    cudaMalloc(&d_tempProducts, BytesToAllocate);

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

    void *kernelArgs[] = {
        (void *)&xGpu,
        (void *)&yGpu,
        (void *)&internalGpuResult2,
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

        setAccess(xGpu, sizeof(HpSharkFloat<SharkFloatParams>));
        setAccess(yGpu, sizeof(HpSharkFloat<SharkFloatParams>));
        setAccess(internalGpuResult2, sizeof(HpSharkFloat<SharkFloatParams>));
        setAccess(d_tempProducts, 32 * SharkFloatParams::GlobalNumUint32 * sizeof(uint64_t));
    }

    {
        ScopedBenchmarkStopper stopper{ timer };
        kernel(stream, kernelArgs);
    }

    cudaMemcpy(&gpuResult2, internalGpuResult2, sizeof(HpSharkFloat<SharkFloatParams>), cudaMemcpyDeviceToHost);

    if constexpr (SharkCustomStream) {
        cudaStreamDestroy(stream); // Destroy the stream
    }

    cudaFree(internalGpuResult2);
    cudaFree(yGpu);
    cudaFree(xGpu);
    cudaFree(d_tempProducts);
}

template<class SharkFloatParams>
void InvokeAddKernel(
    BenchmarkTimer &timer,
    std::function<void(void *[])> kernel,
    const HpSharkFloat<SharkFloatParams> &xNum,
    const HpSharkFloat<SharkFloatParams> &yNum,
    HpSharkFloat<SharkFloatParams> &gpuResult2) {

    // Allocate memory for carryOuts and cumulativeCarries
    GlobalAddBlockData *globalBlockData;
    CarryInfo *d_carryOuts;
    uint32_t *d_cumulativeCarries;
    cudaMalloc(&globalBlockData, sizeof(GlobalAddBlockData));
    cudaMalloc(&d_carryOuts, (SharkFloatParams::GlobalNumBlocks + 1) * sizeof(CarryInfo));
    cudaMalloc(&d_cumulativeCarries, (SharkFloatParams::GlobalNumBlocks + 1) * sizeof(uint32_t));

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

    // Prepare kernel arguments
    void *kernelArgs[] = {
        (void *)&xGpu,
        (void *)&yGpu,
        (void *)&internalGpuResult2,
        (void *)&globalBlockData,
        (void *)&d_carryOuts,
        (void *)&d_cumulativeCarries
    };

    {
        ScopedBenchmarkStopper stopper{ timer };
        kernel(kernelArgs);
    }

    // Launch the cooperative kernel

    cudaMemcpy(&gpuResult2, internalGpuResult2, sizeof(HpSharkFloat<SharkFloatParams>), cudaMemcpyDeviceToHost);

    cudaFree(internalGpuResult2);
    cudaFree(yGpu);
    cudaFree(xGpu);

    cudaFree(globalBlockData);
    cudaFree(d_carryOuts);
    cudaFree(d_cumulativeCarries);
}


template<class SharkFloatParams, Operator sharkOperator>
void InvokeMultiplyKernelCorrectness(
    BenchmarkTimer &timer,
    std::function<void(void *[])> kernel,
    const HpSharkFloat<SharkFloatParams> &xNum,
    const HpSharkFloat<SharkFloatParams> &yNum,
    HpSharkFloat<SharkFloatParams> &gpuResult,
    std::vector<DebugStateRaw> *debugResults) {

    // Prepare kernel arguments
    // Allocate memory for carryOuts and cumulativeCarries
    uint64_t *d_tempProducts;
    constexpr auto BytesToAllocate =
        (AdditionalUInt64Global + ScratchMemoryCopies * CalculateFrameSize<SharkFloatParams>()) * sizeof(uint64_t);
    cudaMalloc(&d_tempProducts, BytesToAllocate);

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

    void *kernelArgs[] = {
        (void *)&xGpu,
        (void *)&yGpu,
        (void *)&internalGpuResult2,
        (void *)&d_tempProducts
    };

    {
        ScopedBenchmarkStopper stopper{ timer };
        kernel(kernelArgs);
    }

    cudaMemcpy(&gpuResult, internalGpuResult2, sizeof(HpSharkFloat<SharkFloatParams>), cudaMemcpyDeviceToHost);

    if (debugResults != nullptr && SharkDebug) {
        debugResults->resize(SharkFloatParams::NumDebugStates);
        cudaMemcpy(
            debugResults->data(),
            &d_tempProducts[AdditionalGlobalSyncSpace],
            SharkFloatParams::NumDebugStates * sizeof(DebugStateRaw),
            cudaMemcpyDeviceToHost);
    }

    cudaFree(internalGpuResult2);
    cudaFree(yGpu);
    cudaFree(xGpu);
    cudaFree(d_tempProducts);
}

template<class SharkFloatParams, Operator sharkOperator>
void InvokeAddKernelCorrectness(
    BenchmarkTimer &timer,
    std::function<void(void *[])> kernel,
    const HpSharkFloat<SharkFloatParams> &xNum,
    const HpSharkFloat<SharkFloatParams> &yNum,
    HpSharkFloat<SharkFloatParams> &gpuResult) {

    // Perform the calculation on the GPU
    HpSharkFloat<SharkFloatParams> *xGpu;
    HpSharkFloat<SharkFloatParams> *yGpu;

    cudaMalloc(&xGpu, sizeof(HpSharkFloat<SharkFloatParams>));
    cudaMalloc(&yGpu, sizeof(HpSharkFloat<SharkFloatParams>));
    cudaMemcpy(xGpu, &xNum, sizeof(HpSharkFloat<SharkFloatParams>), cudaMemcpyHostToDevice);
    cudaMemcpy(yGpu, &yNum, sizeof(HpSharkFloat<SharkFloatParams>), cudaMemcpyHostToDevice);

    HpSharkFloat<SharkFloatParams> *internalGpuResult;
    cudaMalloc(&internalGpuResult, sizeof(HpSharkFloat<SharkFloatParams>));

    // Allocate memory for carryOuts and cumulativeCarries
    GlobalAddBlockData *globalBlockData;
    CarryInfo *d_carryOuts;
    uint32_t *d_cumulativeCarries;
    cudaMalloc(&globalBlockData, sizeof(GlobalAddBlockData));
    cudaMalloc(&d_carryOuts, (SharkFloatParams::GlobalNumBlocks + 1) * sizeof(CarryInfo));
    cudaMalloc(&d_cumulativeCarries, (SharkFloatParams::GlobalNumBlocks + 1) * sizeof(uint32_t));

    // Prepare kernel arguments
    void *kernelArgs[] = {
        (void *)&xGpu,
        (void *)&yGpu,
        (void *)&internalGpuResult,
        (void *)&globalBlockData,
        (void *)&d_carryOuts,
        (void *)&d_cumulativeCarries
    };

    {
        ScopedBenchmarkStopper stopper{ timer };
        ComputeAddGpu<SharkFloatParams>(kernelArgs);
    }

    cudaFree(globalBlockData);
    cudaFree(d_carryOuts);
    cudaFree(d_cumulativeCarries);

    cudaMemcpy(&gpuResult, internalGpuResult, sizeof(HpSharkFloat<SharkFloatParams>), cudaMemcpyDeviceToHost);
    cudaFree(internalGpuResult);

    cudaFree(yGpu);
    cudaFree(xGpu);
}

#define ExplicitlyInstantiate(SharkFloatParams) \
    template void InvokeMultiplyKernel<SharkFloatParams>( \
        BenchmarkTimer &timer, \
        std::function<void(cudaStream_t &, void *[])> kernel, \
        const HpSharkFloat<SharkFloatParams> &xNum, \
        const HpSharkFloat<SharkFloatParams> &yNum, \
        HpSharkFloat<SharkFloatParams> &gpuResult2); \
    template void InvokeAddKernel<SharkFloatParams>( \
        BenchmarkTimer &timer, \
        std::function<void(void*[])> kernel, \
        const HpSharkFloat<SharkFloatParams> &xNum, \
        const HpSharkFloat<SharkFloatParams> &yNum, \
        HpSharkFloat<SharkFloatParams> &gpuResult2); \
    template void InvokeMultiplyKernelCorrectness<SharkFloatParams, Operator::MultiplyKaratsubaV2>( \
        BenchmarkTimer &timer, \
        std::function<void(void*[])> kernel, \
        const HpSharkFloat<SharkFloatParams> &xNum, \
        const HpSharkFloat<SharkFloatParams> &yNum, \
        HpSharkFloat<SharkFloatParams> &gpuResult, \
        std::vector<DebugStateRaw> *debugResults); \
    template void InvokeAddKernelCorrectness<SharkFloatParams, Operator::Add>( \
        BenchmarkTimer &timer, \
        std::function<void(void*[])> kernel, \
        const HpSharkFloat<SharkFloatParams> &xNum, \
        const HpSharkFloat<SharkFloatParams> &yNum, \
        HpSharkFloat<SharkFloatParams> &gpuResult);

ExplicitInstantiateAll();