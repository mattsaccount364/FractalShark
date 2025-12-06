#include "DbgHeap.h"
#include "KernelInvoke.h"
#include "KernelInvokeInternal.h"
#include "TestVerbose.h"

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

    cudaMalloc(&d_tempProducts, BytesToAllocate);

    if constexpr (!HpShark::TestInitCudaMemory) {
        cudaMemset(d_tempProducts, 0, BytesToAllocate);
    } else {
        cudaMemset(d_tempProducts, 0xCD, BytesToAllocate);
    }

    // --- 1) Stage combo struct, plan and roots on device -----------------------------------
    HpSharkComboResults<SharkFloatParams> *comboGpu = nullptr;
    cudaMalloc(&comboGpu, sizeof(HpSharkComboResults<SharkFloatParams>));
    cudaMemcpy(comboGpu, &combo, sizeof(HpSharkComboResults<SharkFloatParams>), cudaMemcpyHostToDevice);

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
        cudaMemset(&comboGpu->ResultX2, pat, sizeof(HpSharkFloat<SharkFloatParams>));
        cudaMemset(&comboGpu->Result2XY, pat, sizeof(HpSharkFloat<SharkFloatParams>));
        cudaMemset(&comboGpu->ResultY2, pat, sizeof(HpSharkFloat<SharkFloatParams>));
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
        cudaGetDeviceProperties(&prop, device_id);
        // Reserve as much L2 as driver allows for persisting window
        cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, prop.persistingL2CacheMaxSize);

        auto setAccess = [&](void *ptr, size_t num_bytes) {
            cudaStreamAttrValue attr{};
            attr.accessPolicyWindow.base_ptr = ptr;
            attr.accessPolicyWindow.num_bytes = num_bytes; // must be <= accessPolicyMaxWindowSize
            attr.accessPolicyWindow.hitRatio = 1.0;        // hint
            attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
            attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;

            cudaError_t err =
                cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr);
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
    cudaMemcpy(&combo, comboGpu, sizeof(HpSharkComboResults<SharkFloatParams>), cudaMemcpyDeviceToHost);

    // Roots were device-allocated in CopyRootsToCuda; destroy like correctness does
    SharkNTT::DestroyRoots<SharkFloatParams>(true, comboGpu->Roots);

    if constexpr (HpShark::CustomStream) {
        cudaStreamDestroy(stream);
    }

    cudaFree(comboGpu);
    cudaFree(d_tempProducts);
}

#ifdef ENABLE_MULTIPLY_NTT_KERNEL
#define ExplicitlyInstantiateMultiplyNTT(SharkFloatParams)                                              \
    template void InvokeMultiplyNTTKernelPerf<SharkFloatParams>(                                        \
        const HpShark::LaunchParams &launchParams,                                                      \
        BenchmarkTimer &timer,                                                                          \
        HpSharkComboResults<SharkFloatParams> &combo,                                                   \
        uint64_t numIters);
#else
#define ExplicitlyInstantiateMultiplyNTT(SharkFloatParams) ;
#endif

#define ExplicitlyInstantiate(SharkFloatParams) ExplicitlyInstantiateMultiplyNTT(SharkFloatParams)

ExplicitInstantiateAll();
