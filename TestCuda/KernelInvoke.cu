#include <cuda_runtime.h>

#include "BenchmarkTimer.h"
#include "TestTracker.h"

#include "Tests.h"
#include "HpSharkFloat.cuh"
#include "Add.cuh"
#include "Multiply.cuh"
#include "HpSharkReferenceOrbit.cuh"
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
void InvokeHpSharkReferenceKernelPerf(
    BenchmarkTimer &timer,
    HpSharkReferenceResults<SharkFloatParams> &combo,
    uint64_t numIters) {

    // Prepare kernel arguments
    // Allocate memory for carryOuts and cumulativeCarries
    uint64_t *d_tempProducts;
    constexpr auto BytesToAllocate =
        (AdditionalUInt64Global + ScratchMemoryCopies * CalculateMultiplyFrameSize<SharkFloatParams>()) * sizeof(uint64_t);
    cudaError_t err = cudaMalloc(&d_tempProducts, BytesToAllocate);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error in cudaMalloc (d_tempProducts): " << cudaGetErrorString(err) << std::endl;
        return;
    }

    if constexpr (!SharkTestInitCudaMemory) {
        err = cudaMemset(d_tempProducts, 0, BytesToAllocate);
        if (err != cudaSuccess) {
            std::cerr << "CUDA error in cudaMemset (d_tempProducts): " << cudaGetErrorString(err) << std::endl;
            cudaFree(d_tempProducts);
            return;
        }
    } else {
        err = cudaMemset(d_tempProducts, 0xCD, BytesToAllocate);
        if (err != cudaSuccess) {
            std::cerr << "CUDA error in cudaMemset (d_tempProducts): " << cudaGetErrorString(err) << std::endl;
            cudaFree(d_tempProducts);
            return;
        }
    }

    HpSharkReferenceResults<SharkFloatParams> *comboGpu;
    err = cudaMalloc(&comboGpu, sizeof(HpSharkReferenceResults<SharkFloatParams>));
    if (err != cudaSuccess) {
        std::cerr << "CUDA error in cudaMalloc (comboGpu): " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_tempProducts);
        return;
    }
    err = cudaMemcpy(comboGpu, &combo, sizeof(HpSharkReferenceResults<SharkFloatParams>), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error in cudaMemcpy (comboGpu): " << cudaGetErrorString(err) << std::endl;
        cudaFree(comboGpu);
        cudaFree(d_tempProducts);
        return;
    }

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

    err = cudaMemcpy(&combo, comboGpu, sizeof(HpSharkReferenceResults<SharkFloatParams>), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error in cudaMemcpy (device to host): " << cudaGetErrorString(err) << std::endl;
    }

    err = cudaFree(comboGpu);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error in cudaFree (comboGpu): " << cudaGetErrorString(err) << std::endl;
    }
    err = cudaFree(d_tempProducts);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error in cudaFree (d_tempProducts): " << cudaGetErrorString(err) << std::endl;
    }

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
    cudaError_t err = cudaMalloc(&d_tempProducts, BytesToAllocate);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error in cudaMalloc (d_tempProducts): " << cudaGetErrorString(err) << std::endl;
        return;
    }

    HpSharkComboResults<SharkFloatParams> *comboGpu;
    err = cudaMalloc(&comboGpu, sizeof(HpSharkComboResults<SharkFloatParams>));
    if (err != cudaSuccess) {
        std::cerr << "CUDA error in cudaMalloc (comboGpu): " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_tempProducts);
        return;
    }
    err = cudaMemcpy(comboGpu, &combo, sizeof(HpSharkComboResults<SharkFloatParams>), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error in cudaMemcpy (comboGpu): " << cudaGetErrorString(err) << std::endl;
        cudaFree(comboGpu);
        cudaFree(d_tempProducts);
        return;
    }

    void *kernelArgs[] = {
        (void *)&comboGpu,
        (void *)&numIters,
        (void *)&d_tempProducts
    };

    cudaStream_t stream = nullptr;

    if constexpr (SharkCustomStream) {
        err = cudaStreamCreate(&stream); // Create a stream
        if (err != cudaSuccess) {
            std::cerr << "CUDA error in cudaStreamCreate: " << cudaGetErrorString(err) << std::endl;
            cudaFree(comboGpu);
            cudaFree(d_tempProducts);
            return;
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

        setAccess(comboGpu, sizeof(HpSharkComboResults<SharkFloatParams>));
        setAccess(d_tempProducts, 32 * SharkFloatParams::GlobalNumUint32 * sizeof(uint64_t));
    }

    {
        ScopedBenchmarkStopper stopper{ timer };
        ComputeMultiplyKaratsubaV2GpuTestLoop<SharkFloatParams>(stream, kernelArgs);
    }

    err = cudaMemcpy(&combo, comboGpu, sizeof(HpSharkComboResults<SharkFloatParams>), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error in cudaMemcpy (device to host): " << cudaGetErrorString(err) << std::endl;
    }

    if constexpr (SharkCustomStream) {
        err = cudaStreamDestroy(stream); // Destroy the stream
        if (err != cudaSuccess) {
            std::cerr << "CUDA error in cudaStreamDestroy: " << cudaGetErrorString(err) << std::endl;
        }
    }

    err = cudaFree(comboGpu);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error in cudaFree (comboGpu): " << cudaGetErrorString(err) << std::endl;
    }
    err = cudaFree(d_tempProducts);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error in cudaFree (d_tempProducts): " << cudaGetErrorString(err) << std::endl;
    }
}

template<class SharkFloatParams>
void InvokeAddKernelPerf(
    BenchmarkTimer &timer,
    HpSharkAddComboResults<SharkFloatParams> &combo,
    uint64_t numIters) {

    // Perform the calculation on the GPU
    HpSharkAddComboResults<SharkFloatParams> *comboResults;
    cudaError_t err = cudaMalloc(&comboResults, sizeof(HpSharkAddComboResults<SharkFloatParams>));
    if (err != cudaSuccess) {
        std::cerr << "CUDA error in cudaMalloc (comboResults): " << cudaGetErrorString(err) << std::endl;
        return;
    }
    err = cudaMemcpy(comboResults, &combo, sizeof(HpSharkAddComboResults<SharkFloatParams>), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error in cudaMemcpy (comboResults): " << cudaGetErrorString(err) << std::endl;
        cudaFree(comboResults);
        return;
    }

    constexpr auto BytesToAllocate =
        (AdditionalUInt64Global + CalculateAddFrameSize<SharkFloatParams>()) * sizeof(uint64_t);
    uint64_t *g_extResult;
    err = cudaMalloc(&g_extResult, BytesToAllocate);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error in cudaMalloc (g_extResult): " << cudaGetErrorString(err) << std::endl;
        cudaFree(comboResults);
        return;
    }

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

    err = cudaMemcpy(&combo, comboResults, sizeof(HpSharkAddComboResults<SharkFloatParams>), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error in cudaMemcpy (device to host): " << cudaGetErrorString(err) << std::endl;
    }

    // Free memory
    err = cudaFree(g_extResult);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error in cudaFree (g_extResult): " << cudaGetErrorString(err) << std::endl;
    }
    err = cudaFree(comboResults);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error in cudaFree (comboResults): " << cudaGetErrorString(err) << std::endl;
    }
}

template<class SharkFloatParams>
void InvokeHpSharkReferenceKernelCorrectness(
    BenchmarkTimer &timer,
    HpSharkReferenceResults<SharkFloatParams> &combo,
    std::vector<DebugStateRaw> *debugResults) {

    // Prepare kernel arguments
    // Allocate memory for carryOuts and cumulativeCarries

    // TODO max of add/multiply frame size
    // TODO checksum handled
    uint64_t *d_tempProducts;
    constexpr auto BytesToAllocate =
        (AdditionalUInt64Global + ScratchMemoryCopies * CalculateMultiplyFrameSize<SharkFloatParams>()) * sizeof(uint64_t);
    cudaError_t err = cudaMalloc(&d_tempProducts, BytesToAllocate);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error in cudaMalloc (d_tempProducts): " << cudaGetErrorString(err) << std::endl;
        return;
    }

    if constexpr (!SharkTestInitCudaMemory) {
        err = cudaMemset(d_tempProducts, 0, BytesToAllocate);
        if (err != cudaSuccess) {
            std::cerr << "CUDA error in cudaMemset (d_tempProducts): " << cudaGetErrorString(err) << std::endl;
            cudaFree(d_tempProducts);
            return;
        }
    } else {
        err = cudaMemset(d_tempProducts, 0xCD, BytesToAllocate);
        if (err != cudaSuccess) {
            std::cerr << "CUDA error in cudaMemset (d_tempProducts): " << cudaGetErrorString(err) << std::endl;
            cudaFree(d_tempProducts);
            return;
        }
    }

    HpSharkReferenceResults<SharkFloatParams> *comboGpu;
    err = cudaMalloc(&comboGpu, sizeof(HpSharkReferenceResults<SharkFloatParams>));
    if (err != cudaSuccess) {
        std::cerr << "CUDA error in cudaMalloc (comboGpu): " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_tempProducts);
        return;
    }
    err = cudaMemcpy(comboGpu, &combo, sizeof(HpSharkReferenceResults<SharkFloatParams>), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error in cudaMemcpy (comboGpu): " << cudaGetErrorString(err) << std::endl;
        cudaFree(comboGpu);
        cudaFree(d_tempProducts);
        return;
    }

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

    err = cudaMemcpy(&combo, comboGpu, sizeof(HpSharkReferenceResults<SharkFloatParams>), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error in cudaMemcpy (device to host): " << cudaGetErrorString(err) << std::endl;
    }

    if (debugResults != nullptr) {
        if constexpr (SharkDebugChecksums) {
            debugResults->resize(SharkFloatParams::NumDebugStates);
            err = cudaMemcpy(
                debugResults->data(),
                &d_tempProducts[AdditionalGlobalSyncSpace],
                SharkFloatParams::NumDebugStates * sizeof(DebugStateRaw),
                cudaMemcpyDeviceToHost);
            if (err != cudaSuccess) {
                std::cerr << "CUDA error in cudaMemcpy (debug results): " << cudaGetErrorString(err) << std::endl;
            }
        }
    }

    err = cudaFree(comboGpu);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error in cudaFree (comboGpu): " << cudaGetErrorString(err) << std::endl;
    }
    err = cudaFree(d_tempProducts);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error in cudaFree (d_tempProducts): " << cudaGetErrorString(err) << std::endl;
    }
}

template<class SharkFloatParams>
void InvokeMultiplyKernelCorrectness(
    BenchmarkTimer &timer,
    HpSharkComboResults<SharkFloatParams> &combo,
    std::vector<DebugStateRaw> *debugResults) {

    // Prepare kernel arguments
    // Allocate memory for carryOuts and cumulativeCarries
    uint64_t *d_tempProducts;
    constexpr auto BytesToAllocate =
        (AdditionalUInt64Global + ScratchMemoryCopies * CalculateMultiplyFrameSize<SharkFloatParams>()) * sizeof(uint64_t);
    cudaError_t err = cudaMalloc(&d_tempProducts, BytesToAllocate);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error in cudaMalloc (d_tempProducts): " << cudaGetErrorString(err) << std::endl;
        return;
    }

    if constexpr (!SharkTestInitCudaMemory) {
        err = cudaMemset(d_tempProducts, 0, BytesToAllocate);
        if (err != cudaSuccess) {
            std::cerr << "CUDA error in cudaMemset (d_tempProducts): " << cudaGetErrorString(err) << std::endl;
            cudaFree(d_tempProducts);
            return;
        }
    } else {
        err = cudaMemset(d_tempProducts, 0xCD, BytesToAllocate);
        if (err != cudaSuccess) {
            std::cerr << "CUDA error in cudaMemset (d_tempProducts): " << cudaGetErrorString(err) << std::endl;
            cudaFree(d_tempProducts);
            return;
        }
    }

    HpSharkComboResults<SharkFloatParams> *comboGpu;
    err = cudaMalloc(&comboGpu, sizeof(HpSharkComboResults<SharkFloatParams>));
    if (err != cudaSuccess) {
        std::cerr << "CUDA error in cudaMalloc (comboGpu): " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_tempProducts);
        return;
    }
    err = cudaMemcpy(comboGpu, &combo, sizeof(HpSharkComboResults<SharkFloatParams>), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error in cudaMemcpy (comboGpu): " << cudaGetErrorString(err) << std::endl;
        cudaFree(comboGpu);
        cudaFree(d_tempProducts);
        return;
    }

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

    err = cudaMemcpy(&combo, comboGpu, sizeof(HpSharkComboResults<SharkFloatParams>), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error in cudaMemcpy (device to host): " << cudaGetErrorString(err) << std::endl;
    }

    if (debugResults != nullptr) {
        if constexpr (SharkDebugChecksums) {
            debugResults->resize(SharkFloatParams::NumDebugStates);
            err = cudaMemcpy(
                debugResults->data(),
                &d_tempProducts[AdditionalGlobalSyncSpace],
                SharkFloatParams::NumDebugStates * sizeof(DebugStateRaw),
                cudaMemcpyDeviceToHost);
            if (err != cudaSuccess) {
                std::cerr << "CUDA error in cudaMemcpy (debug results): " << cudaGetErrorString(err) << std::endl;
            }
        }
    }

    err = cudaFree(comboGpu);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error in cudaFree (comboGpu): " << cudaGetErrorString(err) << std::endl;
    }
    err = cudaFree(d_tempProducts);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error in cudaFree (d_tempProducts): " << cudaGetErrorString(err) << std::endl;
    }
}

template<class SharkFloatParams>
void InvokeAddKernelCorrectness(
    BenchmarkTimer &timer,
    HpSharkAddComboResults<SharkFloatParams> &combo,
    std::vector<DebugStateRaw> *debugResults) {

    // Perform the calculation on the GPU
    HpSharkAddComboResults<SharkFloatParams> *comboResults;
    cudaError_t err = cudaMalloc(&comboResults, sizeof(HpSharkAddComboResults<SharkFloatParams>));
    if (err != cudaSuccess) {
        std::cerr << "CUDA error in cudaMalloc (comboResults): " << cudaGetErrorString(err) << std::endl;
        return;
    }
    err = cudaMemcpy(comboResults, &combo, sizeof(HpSharkAddComboResults<SharkFloatParams>), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error in cudaMemcpy (comboResults): " << cudaGetErrorString(err) << std::endl;
        cudaFree(comboResults);
        return;
    }

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
    err = cudaMalloc(&g_extResult, BytesToAllocate);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error in cudaMalloc (g_extResult): " << cudaGetErrorString(err) << std::endl;
        cudaFree(comboResults);
        return;
    }

    // Prepare kernel arguments
    void *kernelArgs[] = {
        (void *)&comboResults,
        (void *)&g_extResult
    };

    {
        ScopedBenchmarkStopper stopper{ timer };
        ComputeAddGpu<SharkFloatParams>(kernelArgs);
    }

    err = cudaMemcpy(&combo, comboResults, sizeof(HpSharkAddComboResults<SharkFloatParams>), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error in cudaMemcpy (device to host): " << cudaGetErrorString(err) << std::endl;
    }

    if (debugResults != nullptr) {
        if constexpr (SharkDebugChecksums) {
            debugResults->resize(SharkFloatParams::NumDebugStates);
            err = cudaMemcpy(
                debugResults->data(),
                &g_extResult[AdditionalGlobalSyncSpace],
                SharkFloatParams::NumDebugStates * sizeof(DebugStateRaw),
                cudaMemcpyDeviceToHost);
            if (err != cudaSuccess) {
                std::cerr << "CUDA error in cudaMemcpy (debug results): " << cudaGetErrorString(err) << std::endl;
            }
        }
    }

    err = cudaFree(g_extResult);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error in cudaFree (g_extResult): " << cudaGetErrorString(err) << std::endl;
    }
    err = cudaFree(comboResults);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error in cudaFree (comboResults): " << cudaGetErrorString(err) << std::endl;
    }
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
        std::vector<DebugStateRaw> *debugResults);
#else
#define ExplicitlyInstantiateAdd(SharkFloatParams) ;
#endif

#ifdef ENABLE_MULTIPLY_KERNEL
#define ExplicitlyInstantiateMultiply(SharkFloatParams) \
    template void InvokeMultiplyKernelPerf<SharkFloatParams>( \
        BenchmarkTimer &timer, \
        HpSharkComboResults<SharkFloatParams> &combo, \
        uint64_t numIters); \
    template void InvokeMultiplyKernelCorrectness<SharkFloatParams>( \
        BenchmarkTimer &timer, \
        HpSharkComboResults<SharkFloatParams> &combo, \
        std::vector<DebugStateRaw> *debugResults);
#else
#define ExplicitlyInstantiateMultiply(SharkFloatParams) ;
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
        std::vector<DebugStateRaw> *debugResults);
#else
#define ExplicitlyInstantiateHpSharkReference(SharkFloatParams) ;
#endif

#define ExplicitlyInstantiate(SharkFloatParams) \
    ExplicitlyInstantiateAdd(SharkFloatParams) \
    ExplicitlyInstantiateMultiply(SharkFloatParams) \
    ExplicitlyInstantiateHpSharkReference(SharkFloatParams)

ExplicitInstantiateAll();
