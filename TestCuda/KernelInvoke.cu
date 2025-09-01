#include <cuda_runtime.h>

#include "BenchmarkTimer.h"
#include "TestTracker.h"

#include "Tests.h"
#include "HpSharkFloat.cuh"
#include "Add.cuh"
#include "Multiply.cuh"
#include "MultiplyNTT.cuh"
#include "HpSharkReferenceOrbit.cuh"

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
void InvokeHpSharkReferenceKernelPerf(
    BenchmarkTimer &timer,
    HpSharkReferenceResults<SharkFloatParams> &combo,
    uint64_t numIters) {

    // Prepare kernel arguments
    // Allocate memory for carryOuts and cumulativeCarries
    uint64_t *d_tempProducts;
    constexpr auto BytesToAllocate =
        (AdditionalUInt64Global + ScratchMemoryCopies * CalculateMultiplyFrameSize<SharkFloatParams>()) * sizeof(uint64_t);
    cudaMalloc(&d_tempProducts, BytesToAllocate);

    if constexpr (!SharkTestInitCudaMemory) {
        cudaMemset(d_tempProducts, 0, BytesToAllocate);
    } else {
        cudaMemset(d_tempProducts, 0xCD, BytesToAllocate);
    }

    HpSharkReferenceResults<SharkFloatParams> *comboGpu;
    cudaMalloc(&comboGpu, sizeof(HpSharkReferenceResults<SharkFloatParams>));
    cudaMemcpy(comboGpu, &combo, sizeof(HpSharkReferenceResults<SharkFloatParams>), cudaMemcpyHostToDevice);

    uint8_t byteToSet = SharkTestInitCudaMemory ? 0xCD : 0;

    cudaMemset(&comboGpu->Add.A_X2, byteToSet, sizeof(HpSharkFloat<SharkFloatParams>));
    cudaMemset(&comboGpu->Add.B_Y2, byteToSet, sizeof(HpSharkFloat<SharkFloatParams>));
    cudaMemset(&comboGpu->Add.D_2X, byteToSet, sizeof(HpSharkFloat<SharkFloatParams>));
    cudaMemset(&comboGpu->Add.Result1_A_B_C, byteToSet, sizeof(HpSharkFloat<SharkFloatParams>));
    cudaMemset(&comboGpu->Add.Result2_D_E, byteToSet, sizeof(HpSharkFloat<SharkFloatParams>));
    cudaMemset(&comboGpu->Multiply.ResultX2, byteToSet, sizeof(HpSharkFloat<SharkFloatParams>));
    cudaMemset(&comboGpu->Multiply.Result2XY, byteToSet, sizeof(HpSharkFloat<SharkFloatParams>));
    cudaMemset(&comboGpu->Multiply.ResultY2, byteToSet, sizeof(HpSharkFloat<SharkFloatParams>));

    void *kernelArgs[] = {
        (void *)&comboGpu,
        (void *)&numIters,
        (void *)&d_tempProducts
    };

    cudaStream_t stream = nullptr;

    if constexpr (SharkCustomStream) {
        auto res = cudaStreamCreate(&stream); // Create a stream

        if (res != cudaSuccess) {
            std::cerr << "CUDA error in creating stream: " << cudaGetErrorString(res) << std::endl;
        }
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

        setAccess(comboGpu, sizeof(HpSharkReferenceResults<SharkFloatParams>));
        setAccess(d_tempProducts, 32 * SharkFloatParams::GlobalNumUint32 * sizeof(uint64_t));
    }

    {
        ScopedBenchmarkStopper stopper{ timer };
        ComputeHpSharkReferenceGpuLoop<SharkFloatParams>(stream, kernelArgs);
    }

    cudaMemcpy(&combo, comboGpu, sizeof(HpSharkReferenceResults<SharkFloatParams>), cudaMemcpyDeviceToHost);

    cudaFree(comboGpu);
    cudaFree(d_tempProducts);

    if constexpr (SharkCustomStream) {
        auto res = cudaStreamDestroy(stream); // Destroy the stream

        if (res != cudaSuccess) {
            std::cerr << "CUDA error in destroying stream: " << cudaGetErrorString(res) << std::endl;
        }
    }
}

template<class SharkFloatParams>
void InvokeMultiplyKernelPerf(
    BenchmarkTimer &timer,
    HpSharkComboResults<SharkFloatParams> &combo,
    uint64_t numIters) {

    // Prepare kernel arguments
    // Allocate memory for carryOuts and cumulativeCarries
    uint64_t *d_tempProducts;
    constexpr auto BytesToAllocate =
        (AdditionalUInt64Global + ScratchMemoryCopies * CalculateMultiplyFrameSize<SharkFloatParams>()) * sizeof(uint64_t);
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
        ComputeMultiplyKaratsubaV2GpuTestLoop<SharkFloatParams>(stream, kernelArgs);
    }

    cudaMemcpy(&combo, comboGpu, sizeof(HpSharkComboResults<SharkFloatParams>), cudaMemcpyDeviceToHost);

    if constexpr (SharkCustomStream) {
        cudaStreamDestroy(stream); // Destroy the stream
    }

    cudaFree(comboGpu);
    cudaFree(d_tempProducts);
}

template <class SharkFloatParams>
void
InvokeMultiplyNTTKernelPerf(BenchmarkTimer& timer,
                         HpSharkComboResults<SharkFloatParams>& combo,
                         uint64_t numIters)
{

    // Prepare kernel arguments
    // Allocate memory for carryOuts and cumulativeCarries
    uint64_t* d_tempProducts;
    constexpr auto BytesToAllocate =
        (AdditionalUInt64Global + ScratchMemoryCopies * CalculateMultiplyFrameSize<SharkFloatParams>()) *
        sizeof(uint64_t);
    cudaMalloc(&d_tempProducts, BytesToAllocate);

    HpSharkComboResults<SharkFloatParams>* comboGpu;
    cudaMalloc(&comboGpu, sizeof(HpSharkComboResults<SharkFloatParams>));
    cudaMemcpy(comboGpu, &combo, sizeof(HpSharkComboResults<SharkFloatParams>), cudaMemcpyHostToDevice);

    void* kernelArgs[] = {(void*)&comboGpu, (void*)&numIters, (void*)&d_tempProducts};

    cudaStream_t stream = nullptr;

    if constexpr (SharkCustomStream) {
        cudaStreamCreate(&stream); // Create a stream
    }

    cudaDeviceProp prop;
    int device_id = 0;

    if constexpr (SharkCustomStream) {
        cudaGetDeviceProperties(&prop, device_id);
        cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize,
                           prop.persistingL2CacheMaxSize); /* Set aside max possible size of L2 cache for
                                                              persisting accesses */

        auto setAccess = [&](void* ptr, size_t num_bytes) {
            cudaStreamAttrValue stream_attribute; // Stream level attributes data structure
            stream_attribute.accessPolicyWindow.base_ptr =
                reinterpret_cast<void*>(ptr); // Global Memory data pointer
            stream_attribute.accessPolicyWindow.num_bytes =
                num_bytes; // Number of bytes for persisting accesses.
            // (Must be less than cudaDeviceProp::accessPolicyMaxWindowSize)
            stream_attribute.accessPolicyWindow.hitRatio =
                1.0; // Hint for L2 cache hit ratio for persisting accesses in the num_bytes region
            stream_attribute.accessPolicyWindow.hitProp =
                cudaAccessPropertyPersisting; // Type of access property on cache hit
            stream_attribute.accessPolicyWindow.missProp =
                cudaAccessPropertyStreaming; // Type of access property on cache miss.

            // Set the attributes to a CUDA stream of type cudaStream_t
            cudaError_t err =
                cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);
            if (err != cudaSuccess) {
                std::cerr << "CUDA error in setting stream attribute: " << cudaGetErrorString(err)
                          << std::endl;
            }
        };

        setAccess(comboGpu, sizeof(HpSharkComboResults<SharkFloatParams>));
        setAccess(d_tempProducts, 32 * SharkFloatParams::GlobalNumUint32 * sizeof(uint64_t));
    }

    {
        ScopedBenchmarkStopper stopper{timer};
        ComputeMultiplyNTTGpuTestLoop<SharkFloatParams>(stream, kernelArgs);
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
    HpSharkAddComboResults<SharkFloatParams> &combo,
    uint64_t numIters) {

    // Perform the calculation on the GPU
    HpSharkAddComboResults<SharkFloatParams> *comboResults;
    cudaMalloc(&comboResults, sizeof(HpSharkAddComboResults<SharkFloatParams>));
    cudaMemcpy(comboResults, &combo, sizeof(HpSharkAddComboResults<SharkFloatParams>), cudaMemcpyHostToDevice);

    constexpr auto BytesToAllocate =
        (AdditionalUInt64Global + CalculateAddFrameSize<SharkFloatParams>()) * sizeof(uint64_t);
    uint64_t *g_extResult;
    cudaMalloc(&g_extResult, BytesToAllocate);

    // Prepare kernel arguments
    void *kernelArgs[] = {
        (void *)&comboResults,
        (void *)&numIters,
        (void *)&g_extResult
    };

    // Launch the cooperative kernel
    {
        ScopedBenchmarkStopper stopper{ timer };
        ComputeAddGpuTestLoop<SharkFloatParams>(kernelArgs);
    }

    cudaMemcpy(&combo, comboResults, sizeof(HpSharkAddComboResults<SharkFloatParams>), cudaMemcpyDeviceToHost);

    // Free memory
    cudaFree(g_extResult);
    cudaFree(comboResults);
}

template<class SharkFloatParams>
void InvokeHpSharkReferenceKernelCorrectness(
    BenchmarkTimer &timer,
    HpSharkReferenceResults<SharkFloatParams> &combo,
    DebugGpuCombo *debugCombo) {

    // Prepare kernel arguments
    // Allocate memory for carryOuts and cumulativeCarries

    // TODO max of add/multiply frame size
    // TODO checksum handled
    uint64_t *d_tempProducts;
    constexpr auto BytesToAllocate =
        (AdditionalUInt64Global + ScratchMemoryCopies * CalculateMultiplyFrameSize<SharkFloatParams>()) * sizeof(uint64_t);
    cudaMalloc(&d_tempProducts, BytesToAllocate);

    if constexpr (!SharkTestInitCudaMemory) {
        cudaMemset(d_tempProducts, 0, BytesToAllocate);
    } else {
        cudaMemset(d_tempProducts, 0xCD, BytesToAllocate);
    }

    HpSharkReferenceResults<SharkFloatParams> *comboGpu;
    cudaMalloc(&comboGpu, sizeof(HpSharkReferenceResults<SharkFloatParams>));
    cudaMemcpy(comboGpu, &combo, sizeof(HpSharkReferenceResults<SharkFloatParams>), cudaMemcpyHostToDevice);

    uint8_t byteToSet = SharkTestInitCudaMemory ? 0xCD : 0;

    cudaMemset(&comboGpu->Add.A_X2, byteToSet, sizeof(HpSharkFloat<SharkFloatParams>));
    cudaMemset(&comboGpu->Add.B_Y2, byteToSet, sizeof(HpSharkFloat<SharkFloatParams>));
    cudaMemset(&comboGpu->Add.D_2X, byteToSet, sizeof(HpSharkFloat<SharkFloatParams>));
    cudaMemset(&comboGpu->Add.Result1_A_B_C, byteToSet, sizeof(HpSharkFloat<SharkFloatParams>));
    cudaMemset(&comboGpu->Add.Result2_D_E, byteToSet, sizeof(HpSharkFloat<SharkFloatParams>));
    cudaMemset(&comboGpu->Multiply.ResultX2, byteToSet, sizeof(HpSharkFloat<SharkFloatParams>));
    cudaMemset(&comboGpu->Multiply.Result2XY, byteToSet, sizeof(HpSharkFloat<SharkFloatParams>));
    cudaMemset(&comboGpu->Multiply.ResultY2, byteToSet, sizeof(HpSharkFloat<SharkFloatParams>));

    void *kernelArgs[] = {
        (void *)&comboGpu,
        (void *)&d_tempProducts
    };

    {
        ScopedBenchmarkStopper stopper{ timer };
        ComputeHpSharkReferenceGpu<SharkFloatParams>(kernelArgs);
    }

    cudaMemcpy(&combo, comboGpu, sizeof(HpSharkReferenceResults<SharkFloatParams>), cudaMemcpyDeviceToHost);

    if (debugCombo != nullptr) {
        if constexpr (SharkDebugChecksums) {
            debugCombo->States.resize(SharkFloatParams::NumDebugStates);
            cudaMemcpy(
                debugCombo->States.data(),
                &d_tempProducts[AdditionalChecksumsOffset],
                SharkFloatParams::NumDebugStates * sizeof(DebugStateRaw),
                cudaMemcpyDeviceToHost);
        }

        if constexpr (SharkPrintMultiplyCounts) {
            debugCombo->MultiplyCounts.resize(SharkFloatParams::NumDebugMultiplyCounts);
            cudaMemcpy(
                debugCombo->MultiplyCounts.data(),
                &d_tempProducts[AdditionalMultipliesOffset],
                SharkFloatParams::NumDebugMultiplyCounts * sizeof(DebugMultiplyCountRaw),
                cudaMemcpyDeviceToHost);
        }
    }

    cudaFree(comboGpu);
    cudaFree(d_tempProducts);
}

template<class SharkFloatParams>
void InvokeMultiplyKaratsubaKernelCorrectness(
    BenchmarkTimer &timer,
    HpSharkComboResults<SharkFloatParams> &combo,
    DebugGpuCombo *debugCombo) {

    // Prepare kernel arguments
    // Allocate memory for carryOuts and cumulativeCarries
    uint64_t *d_tempProducts;
    constexpr auto BytesToAllocate =
        (AdditionalUInt64Global + ScratchMemoryCopies * CalculateMultiplyFrameSize<SharkFloatParams>()) * sizeof(uint64_t);
    cudaMalloc(&d_tempProducts, BytesToAllocate);

    if constexpr (!SharkTestInitCudaMemory) {
        cudaMemset(d_tempProducts, 0, BytesToAllocate);
    } else {
        cudaMemset(d_tempProducts, 0xCD, BytesToAllocate);
    }

    HpSharkComboResults<SharkFloatParams> *comboGpu;
    cudaMalloc(&comboGpu, sizeof(HpSharkComboResults<SharkFloatParams>));
    cudaMemcpy(comboGpu, &combo, sizeof(HpSharkComboResults<SharkFloatParams>), cudaMemcpyHostToDevice);

    if constexpr (!SharkTestInitCudaMemory) {
        cudaMemset(&comboGpu->ResultX2, 0, sizeof(HpSharkFloat<SharkFloatParams>));
        cudaMemset(&comboGpu->Result2XY, 0, sizeof(HpSharkFloat<SharkFloatParams>));
        cudaMemset(&comboGpu->ResultY2, 0, sizeof(HpSharkFloat<SharkFloatParams>));
    } else {
        cudaMemset(&comboGpu->ResultX2, 0xCD, sizeof(HpSharkFloat<SharkFloatParams>));
        cudaMemset(&comboGpu->Result2XY, 0xCD, sizeof(HpSharkFloat<SharkFloatParams>));
        cudaMemset(&comboGpu->ResultY2, 0xCD, sizeof(HpSharkFloat<SharkFloatParams>));
    }

    void *kernelArgs[] = {
        (void *)&comboGpu,
        (void *)&d_tempProducts
    };

    {
        ScopedBenchmarkStopper stopper{ timer };
        ComputeMultiplyKaratsubaV2Gpu<SharkFloatParams>(kernelArgs);
    }

    cudaMemcpy(&combo, comboGpu, sizeof(HpSharkComboResults<SharkFloatParams>), cudaMemcpyDeviceToHost);

    if (debugCombo != nullptr) {
        if constexpr (SharkDebugChecksums) {
            debugCombo->States.resize(SharkFloatParams::NumDebugStates);
            cudaMemcpy(
                debugCombo->States.data(),
                &d_tempProducts[AdditionalChecksumsOffset],
                SharkFloatParams::NumDebugStates * sizeof(DebugStateRaw),
                cudaMemcpyDeviceToHost);
        }

        if constexpr (SharkPrintMultiplyCounts) {
            debugCombo->MultiplyCounts.resize(SharkFloatParams::NumDebugMultiplyCounts);
            cudaMemcpy(
                debugCombo->MultiplyCounts.data(),
                &d_tempProducts[AdditionalMultipliesOffset],
                SharkFloatParams::NumDebugMultiplyCounts * sizeof(DebugMultiplyCountRaw),
                cudaMemcpyDeviceToHost);
        }
    }

    cudaFree(comboGpu);
    cudaFree(d_tempProducts);
}

template <class SharkFloatParams>
void
InvokeMultiplyNTTKernelCorrectness(BenchmarkTimer& timer,
                                         HpSharkComboResults<SharkFloatParams>& combo,
                                         DebugGpuCombo* debugCombo)
{

    // Prepare kernel arguments
    // Allocate memory for carryOuts and cumulativeCarries
    uint64_t* d_tempProducts;
    constexpr auto BytesToAllocate =
        (AdditionalUInt64Global + ScratchMemoryCopies * CalculateMultiplyFrameSize<SharkFloatParams>()) *
        sizeof(uint64_t);
    cudaMalloc(&d_tempProducts, BytesToAllocate);

    if constexpr (!SharkTestInitCudaMemory) {
        cudaMemset(d_tempProducts, 0, BytesToAllocate);
    } else {
        cudaMemset(d_tempProducts, 0xCD, BytesToAllocate);
    }

    HpSharkComboResults<SharkFloatParams>* comboGpu;
    cudaMalloc(&comboGpu, sizeof(HpSharkComboResults<SharkFloatParams>));
    cudaMemcpy(comboGpu, &combo, sizeof(HpSharkComboResults<SharkFloatParams>), cudaMemcpyHostToDevice);

    if constexpr (!SharkTestInitCudaMemory) {
        cudaMemset(&comboGpu->ResultX2, 0, sizeof(HpSharkFloat<SharkFloatParams>));
        cudaMemset(&comboGpu->Result2XY, 0, sizeof(HpSharkFloat<SharkFloatParams>));
        cudaMemset(&comboGpu->ResultY2, 0, sizeof(HpSharkFloat<SharkFloatParams>));
    } else {
        cudaMemset(&comboGpu->ResultX2, 0xCD, sizeof(HpSharkFloat<SharkFloatParams>));
        cudaMemset(&comboGpu->Result2XY, 0xCD, sizeof(HpSharkFloat<SharkFloatParams>));
        cudaMemset(&comboGpu->ResultY2, 0xCD, sizeof(HpSharkFloat<SharkFloatParams>));
    }

    void* kernelArgs[] = {(void*)&comboGpu, (void*)&d_tempProducts};

    {
        ScopedBenchmarkStopper stopper{timer};
        ComputeMultiplyNTTGpu<SharkFloatParams>(kernelArgs);
    }

    cudaMemcpy(&combo, comboGpu, sizeof(HpSharkComboResults<SharkFloatParams>), cudaMemcpyDeviceToHost);

    if (debugCombo != nullptr) {
        if constexpr (SharkDebugChecksums) {
            debugCombo->States.resize(SharkFloatParams::NumDebugStates);
            cudaMemcpy(debugCombo->States.data(),
                       &d_tempProducts[AdditionalChecksumsOffset],
                       SharkFloatParams::NumDebugStates * sizeof(DebugStateRaw),
                       cudaMemcpyDeviceToHost);
        }

        if constexpr (SharkPrintMultiplyCounts) {
            debugCombo->MultiplyCounts.resize(SharkFloatParams::NumDebugMultiplyCounts);
            cudaMemcpy(debugCombo->MultiplyCounts.data(),
                       &d_tempProducts[AdditionalMultipliesOffset],
                       SharkFloatParams::NumDebugMultiplyCounts * sizeof(DebugMultiplyCountRaw),
                       cudaMemcpyDeviceToHost);
        }
    }

    cudaFree(comboGpu);
    cudaFree(d_tempProducts);
}

template<class SharkFloatParams>
void InvokeAddKernelCorrectness(
    BenchmarkTimer &timer,
    HpSharkAddComboResults<SharkFloatParams> &combo,
    DebugGpuCombo *debugCombo) {

    // Perform the calculation on the GPU
    HpSharkAddComboResults<SharkFloatParams> *comboResults;
    cudaMalloc(&comboResults, sizeof(HpSharkAddComboResults<SharkFloatParams>));
    cudaMemcpy(comboResults, &combo, sizeof(HpSharkAddComboResults<SharkFloatParams>), cudaMemcpyHostToDevice);

    if constexpr (!SharkTestInitCudaMemory) {
        cudaMemset(&comboResults->Result1_A_B_C, 0, sizeof(HpSharkFloat<SharkFloatParams>));
        cudaMemset(&comboResults->Result2_D_E, 0, sizeof(HpSharkFloat<SharkFloatParams>));
    } else {
        cudaMemset(&comboResults->Result1_A_B_C, 0xCD, sizeof(HpSharkFloat<SharkFloatParams>));
        cudaMemset(&comboResults->Result2_D_E, 0xCD, sizeof(HpSharkFloat<SharkFloatParams>));
    }

    constexpr auto BytesToAllocate =
        (AdditionalUInt64Global + CalculateAddFrameSize<SharkFloatParams>()) * sizeof(uint64_t);
    uint64_t *g_extResult;
    cudaMalloc(&g_extResult, BytesToAllocate);

    // Prepare kernel arguments
    void *kernelArgs[] = {
        (void *)&comboResults,
        (void *)&g_extResult
    };

    {
        ScopedBenchmarkStopper stopper{ timer };
        ComputeAddGpu<SharkFloatParams>(kernelArgs);
    }

    cudaMemcpy(&combo, comboResults, sizeof(HpSharkAddComboResults<SharkFloatParams>), cudaMemcpyDeviceToHost);

    if (debugCombo != nullptr) {
        if constexpr (SharkDebugChecksums) {
            debugCombo->States.resize(SharkFloatParams::NumDebugStates);
            cudaMemcpy(
                debugCombo->States.data(),
                &g_extResult[AdditionalChecksumsOffset],
                SharkFloatParams::NumDebugStates * sizeof(DebugStateRaw),
                cudaMemcpyDeviceToHost);
        }

        if constexpr (SharkPrintMultiplyCounts) {
            debugCombo->MultiplyCounts.resize(SharkFloatParams::NumDebugMultiplyCounts);
            cudaMemcpy(
                debugCombo->MultiplyCounts.data(),
                &g_extResult[AdditionalMultipliesOffset],
                SharkFloatParams::NumDebugMultiplyCounts * sizeof(DebugMultiplyCountRaw),
                cudaMemcpyDeviceToHost);
        }
    }

    cudaFree(g_extResult);
    cudaFree(comboResults);
}

#ifdef ENABLE_ADD_KERNEL
#define ExplicitlyInstantiateAdd(SharkFloatParams) \
    template void InvokeAddKernelPerf<SharkFloatParams>( \
        BenchmarkTimer &timer, \
        HpSharkAddComboResults<SharkFloatParams> &combo, \
        uint64_t numIters); \
    template void InvokeAddKernelCorrectness<SharkFloatParams>( \
        BenchmarkTimer &timer, \
        HpSharkAddComboResults<SharkFloatParams> &combo, \
        DebugGpuCombo *debugCombo);
#else
#define ExplicitlyInstantiateAdd(SharkFloatParams) ;
#endif

#ifdef ENABLE_MULTIPLY_KARATSUBA_KERNEL
#define ExplicitlyInstantiateMultiply(SharkFloatParams) \
    template void InvokeMultiplyKernelPerf<SharkFloatParams>( \
        BenchmarkTimer &timer, \
        HpSharkComboResults<SharkFloatParams> &combo, \
        uint64_t numIters); \
    template void InvokeMultiplyKaratsubaKernelCorrectness<SharkFloatParams>( \
        BenchmarkTimer &timer, \
        HpSharkComboResults<SharkFloatParams> &combo, \
        DebugGpuCombo *debugCombo);
#else
#define ExplicitlyInstantiateMultiply(SharkFloatParams) ;
#endif

#ifdef ENABLE_MULTIPLY_FFT2_KERNEL
#define ExplicitlyInstantiateMultiplyNTT(SharkFloatParams)                                                 \
    template void InvokeMultiplyNTTKernelPerf<SharkFloatParams>(                                           \
        BenchmarkTimer & timer, HpSharkComboResults<SharkFloatParams> & combo, uint64_t numIters);      \
    template void InvokeMultiplyNTTKernelCorrectness<SharkFloatParams>(                           \
        BenchmarkTimer & timer,                                                                         \
        HpSharkComboResults<SharkFloatParams> & combo,                                                  \
        DebugGpuCombo * debugCombo);
#else
#define ExplicitlyInstantiateMultiplyNTT(SharkFloatParams) ;
#endif


#ifdef ENABLE_REFERENCE_KERNEL
#define ExplicitlyInstantiateHpSharkReference(SharkFloatParams) \
    template void InvokeHpSharkReferenceKernelPerf<SharkFloatParams>(\
        BenchmarkTimer &timer, \
        HpSharkReferenceResults<SharkFloatParams> &combo, \
        uint64_t numIters); \
    template void InvokeHpSharkReferenceKernelCorrectness<SharkFloatParams>( \
        BenchmarkTimer &timer, \
        HpSharkReferenceResults<SharkFloatParams> &combo, \
        DebugGpuCombo *debugCombo);
#else
#define ExplicitlyInstantiateHpSharkReference(SharkFloatParams) ;
#endif

#define ExplicitlyInstantiate(SharkFloatParams) \
    ExplicitlyInstantiateAdd(SharkFloatParams) \
    ExplicitlyInstantiateMultiply(SharkFloatParams) \
    ExplicitlyInstantiateMultiplyNTT(SharkFloatParams) \
    ExplicitlyInstantiateHpSharkReference(SharkFloatParams)

ExplicitInstantiateAll();
