#include "DbgHeap.h"
#include "KernelInvoke.h"
#include "KernelInvokeInternal.h"
#include "TestVerbose.h"
#include <sstream>
#include <stdexcept>

namespace HpShark {

template <class SharkFloatParams>
void
InvokeMultiplyNTTKernelPerf(const HpShark::LaunchParams &launchParams,
                            BenchmarkTimer &timer,
                            HpSharkComboResults<SharkFloatParams> &combo,
                            uint64_t numIters)
{
    // --- 0) Scratch arena (global) ---------------------------------------------------------
    uint64_t *d_tempProducts = nullptr;
    constexpr size_t BytesToAllocate =
        (HpShark::AdditionalUInt64Global + HpShark::CalculateNTTFrameSize<SharkFloatParams>()) *
        sizeof(uint64_t);

    if (SharkVerbose == VerboseMode::Debug) {
        std::cout << " Allocating " << BytesToAllocate << " bytes for d_tempProducts " << std::endl;
    }

    cudaError_t err = cudaSuccess;
    err = cudaMalloc(&d_tempProducts, BytesToAllocate);
    if (err != cudaSuccess) {
        std::ostringstream oss;
        oss << "cudaMalloc(&d_tempProducts, BytesToAllocate) failed: " << cudaGetErrorString(err)
            << " (code " << static_cast<int>(err) << ")";
        throw std::runtime_error(oss.str());
    }

    if constexpr (!HpShark::TestInitCudaMemory) {
        cudaError_t err = cudaSuccess;
        err = cudaMemset(d_tempProducts, 0, BytesToAllocate);
        if (err != cudaSuccess) {
            std::ostringstream oss;
            oss << "cudaMemset(d_tempProducts, 0, BytesToAllocate) failed: " << cudaGetErrorString(err)
                << " (code " << static_cast<int>(err) << ")";
            throw std::runtime_error(oss.str());
        }
    } else {
        err = cudaMemset(d_tempProducts, 0xCD, BytesToAllocate);
        if (err != cudaSuccess) {
            std::ostringstream oss;
            oss << "cudaMemset(d_tempProducts, 0xCD, BytesToAllocate) failed: "
                << cudaGetErrorString(err) << " (code " << static_cast<int>(err) << ")";
            throw std::runtime_error(oss.str());
        }
    }

    // --- 1) Stage combo struct, plan and roots on device -----------------------------------
    HpSharkComboResults<SharkFloatParams> *comboGpu = nullptr;
    err = cudaMalloc(&comboGpu, sizeof(HpSharkComboResults<SharkFloatParams>));
    if (err != cudaSuccess) {
        std::ostringstream oss;
        oss << "cudaMalloc(&comboGpu, sizeof(HpSharkComboResults<SharkFloatParams>)) failed: "
            << cudaGetErrorString(err) << " (code " << static_cast<int>(err) << ")";
        throw std::runtime_error(oss.str());
    }
    err = cudaMemcpy(
        comboGpu, &combo, sizeof(HpSharkComboResults<SharkFloatParams>), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::ostringstream oss;
        oss << "err = cudaMemcpy(comboGpu, &combo, sizeof(HpSharkComboResults<SharkFloatParams>), "
               "cudaMemcpyHostToDevice) failed: "
            << cudaGetErrorString(err) << " (code " << static_cast<int>(err) << ")";
        throw std::runtime_error(oss.str());
        if (err != cudaSuccess) {
            std::ostringstream oss;
            oss << "cudaMemcpy failed: " << cudaGetErrorString(err) << " (code " << static_cast<int>(err)
                << ")";
            throw std::runtime_error(oss.str());
        }
    }

    // Build NTT plan + roots exactly like correctness path
    {
        SharkNTT::RootTables NTTRoots;
        SharkNTT::BuildRoots<SharkFloatParams>(
            SharkFloatParams::NTTPlan.N, SharkFloatParams::NTTPlan.stages, NTTRoots);

        CopyRootsToCuda<SharkFloatParams>(comboGpu->Roots, NTTRoots);
        SharkNTT::DestroyRoots<SharkFloatParams>(false, NTTRoots);
    }

    // Clear result slots (matches correctness init semantics)
    {
        const uint8_t pat = HpShark::TestInitCudaMemory ? 0xCD : 0x00;
        err = cudaMemset(&comboGpu->ResultX2, pat, sizeof(HpSharkFloat<SharkFloatParams>));
        if (err != cudaSuccess) {
            std::ostringstream oss;
            oss << "cudaMemset(&comboGpu->ResultX2, pat, sizeof(HpSharkFloat<SharkFloatParams>)) "
                   "failed: "
                << cudaGetErrorString(err) << " (code " << static_cast<int>(err) << ")";
            throw std::runtime_error(oss.str());
        }
        err = cudaMemset(&comboGpu->Result2XY, pat, sizeof(HpSharkFloat<SharkFloatParams>));
        if (err != cudaSuccess) {
            std::ostringstream oss;
            oss << "cudaMemset(&comboGpu->Result2XY, pat, sizeof(HpSharkFloat<SharkFloatParams>)) "
                   "failed: "
                << cudaGetErrorString(err) << " (code " << static_cast<int>(err) << ")";
            throw std::runtime_error(oss.str());
        }
        err = cudaMemset(&comboGpu->ResultY2, pat, sizeof(HpSharkFloat<SharkFloatParams>));
        if (err != cudaSuccess) {
            std::ostringstream oss;
            oss << "cudaMemset(&comboGpu->ResultY2, pat, sizeof(HpSharkFloat<SharkFloatParams>)) "
                   "failed: "
                << cudaGetErrorString(err) << " (code " << static_cast<int>(err) << ")";
            throw std::runtime_error(oss.str());
        }
    }

    // --- 2) Stream + persisting L2 window (identical policy to correctness) ----------------
    cudaStream_t stream = nullptr;

    if constexpr (HpShark::CustomStream) {
        auto res = cudaStreamCreate(&stream);
        if (res != cudaSuccess) {
            std::cerr << "CUDA error in creating stream: " << cudaGetErrorString(res) << std::endl;
        }

        cudaDeviceProp prop{};
        int device_id = 0;
        err = cudaGetDeviceProperties(&prop, device_id);
        if (err != cudaSuccess) {
            std::ostringstream oss;
            oss << "cudaGetDeviceProperties(&prop, device_id) failed: " << cudaGetErrorString(err)
                << " (code " << static_cast<int>(err) << ")";
            throw std::runtime_error(oss.str());
        }
        // Reserve as much L2 as driver allows for persisting window
        err = cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, prop.persistingL2CacheMaxSize);
        if (err != cudaSuccess) {
            std::ostringstream oss;
            oss << "cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, prop.persistingL2CacheMaxSize) "
                   "failed: "
                << cudaGetErrorString(err) << " (code " << static_cast<int>(err) << ")";
            throw std::runtime_error(oss.str());
        }

        auto setAccess = [&](void *ptr, size_t num_bytes) {
            cudaStreamAttrValue attr{};
            attr.accessPolicyWindow.base_ptr = ptr;
            attr.accessPolicyWindow.num_bytes = num_bytes; // must be <= accessPolicyMaxWindowSize
            attr.accessPolicyWindow.hitRatio = 1.0;        // hint
            attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
            attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;

            cudaError_t err = err =
                cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr);
            if (err != cudaSuccess) {
                std::ostringstream oss;
                oss << "cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr) "
                       "failed: "
                    << cudaGetErrorString(err) << " (code " << static_cast<int>(err) << ")";
                throw std::runtime_error(oss.str());
            }
            if (err != cudaSuccess) {
                std::cerr << "cudaStreamSetAttribute: " << cudaGetErrorString(err) << std::endl;
            }
        };

        // Keep the hot state resident
        setAccess(comboGpu, sizeof(HpSharkComboResults<SharkFloatParams>));
        // Big scratch window (enough to cover typical working set)
        setAccess(d_tempProducts, 32ull * SharkFloatParams::GlobalNumUint32 * sizeof(uint64_t));
    }

    // --- 3) Launch (mirror correctness: test-loop entry + same arg order) ------------------
    void *kernelArgs[] = {(void *)&comboGpu, (void *)&numIters, (void *)&d_tempProducts};

    {
        ScopedBenchmarkStopper stopper{timer};
        // Use the *looping* entry so numIters lives on device (same as correctness)
        ComputeMultiplyNTTGpuTestLoop<SharkFloatParams>(launchParams, stream, kernelArgs);
    }

    // --- 4) Copy results back, teardown -----------------------------------------------------
    err = cudaMemcpy(
        &combo, comboGpu, sizeof(HpSharkComboResults<SharkFloatParams>), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::ostringstream oss;
        oss << "err = cudaMemcpy(&combo, comboGpu, sizeof(HpSharkComboResults<SharkFloatParams>), "
               "cudaMemcpyDeviceToHost) failed: "
            << cudaGetErrorString(err) << " (code " << static_cast<int>(err) << ")";
        throw std::runtime_error(oss.str());
        if (err != cudaSuccess) {
            std::ostringstream oss;
            oss << "cudaMemcpy failed: " << cudaGetErrorString(err) << " (code " << static_cast<int>(err)
                << ")";
            throw std::runtime_error(oss.str());
        }
    }

    // Roots were device-allocated in CopyRootsToCuda; destroy like correctness does
    SharkNTT::DestroyRoots<SharkFloatParams>(true, comboGpu->Roots);

    if constexpr (HpShark::CustomStream) {
        err = cudaStreamDestroy(stream);
        if (err != cudaSuccess) {
            std::ostringstream oss;
            oss << "cudaStreamDestroy(stream) failed: " << cudaGetErrorString(err) << " (code "
                << static_cast<int>(err) << ")";
            throw std::runtime_error(oss.str());
        }
    }

    err = cudaFree(comboGpu);
    if (err != cudaSuccess) {
        std::ostringstream oss;
        oss << "cudaFree(comboGpu) failed: " << cudaGetErrorString(err) << " (code "
            << static_cast<int>(err) << ")";
        throw std::runtime_error(oss.str());
    }
    err = cudaFree(d_tempProducts);
    if (err != cudaSuccess) {
        std::ostringstream oss;
        oss << "cudaFree(d_tempProducts) failed: " << cudaGetErrorString(err) << " (code "
            << static_cast<int>(err) << ")";
        throw std::runtime_error(oss.str());
    }
}

#define ExplicitlyInstantiateMultiplyNTT(SharkFloatParams)                                              \
    template void InvokeMultiplyNTTKernelPerf<SharkFloatParams>(                                        \
        const HpShark::LaunchParams &launchParams,                                                      \
        BenchmarkTimer &timer,                                                                          \
        HpSharkComboResults<SharkFloatParams> &combo,                                                   \
        uint64_t numIters);

#define ExplicitlyInstantiate(SharkFloatParams) ExplicitlyInstantiateMultiplyNTT(SharkFloatParams)

ExplicitInstantiateAll();

} // namespace HpShark