#include "GPU_ReferenceIter.h"
//#include "KernelInvoke.cuh"
#include "KernelInvokeInternal.cuh"

//
// The "production" path
//
// Assumes:
// combo.Add.C_A, combo.Add.E_B, combo.Multiply.A, combo.Multiply.B are set
// C_A == Multiply.A
// E_B == Multiply.B
// combo.RadiusY is set
// combo.OutputIters is nullptr
//
// On output:
// combo.Period and combo.EscapedIteration are set
// combo.OutputIters is allocated and filled in if periodicity checking is enabled.
//   -- Free via delete[]
//

template <class SharkFloatParams>
void
InvokeHpSharkReferenceKernelProd(const HpShark::LaunchParams &launchParams,
                                 HpSharkReferenceResults<SharkFloatParams> &combo,
                                 mpf_t srcX,
                                 mpf_t srcY,
                                 uint64_t numIters)
{
    auto inputX = std::make_unique<HpSharkFloat<SharkFloatParams>>();
    auto inputY = std::make_unique<HpSharkFloat<SharkFloatParams>>();

    // Convert srcX and srcY to HpSharkFloat
    inputX->MpfToHpGpu(srcX, HpSharkFloat<SharkFloatParams>::DefaultMpirBits, InjectNoiseInLowOrder::Enable);
    inputY->MpfToHpGpu(srcY, HpSharkFloat<SharkFloatParams>::DefaultMpirBits, InjectNoiseInLowOrder::Enable);

    combo.Add.C_A = *inputX;
    combo.Add.E_B = *inputY;
    combo.Multiply.A = *inputX;
    combo.Multiply.B = *inputY;
    combo.Period = 0;
    combo.EscapedIteration = 0;
    combo.OutputIters = nullptr;
    assert(combo.OutputIters == nullptr);
    assert(memcmp(&combo.Add.C_A, &combo.Multiply.A, sizeof(HpSharkFloat<SharkFloatParams>)) == 0);
    assert(memcmp(&combo.Add.E_B, &combo.Multiply.B, sizeof(HpSharkFloat<SharkFloatParams>)) == 0);
    assert(combo.RadiusY.mantissa != 0); // RadiusY must be set.  Does 0 have any useful meaning here?

    InvokeHpSharkReferenceKernelPerf<SharkFloatParams>(launchParams, nullptr, combo, numIters, nullptr);
}

//
// This test is also something of a correctness test because
// it keeps track of the period and checks it subsequently.
//
template <class SharkFloatParams>
void
InvokeHpSharkReferenceKernelPerf(const HpShark::LaunchParams &launchParams,
                                 BenchmarkTimer *timer,
                                 HpSharkReferenceResults<SharkFloatParams> &combo,
                                 uint64_t numIters,
                                 DebugGpuCombo *debugCombo)
{

    typename SharkFloatParams::ReferenceIterT *gpuReferenceIters;
    cudaMalloc(&gpuReferenceIters, sizeof(SharkFloatParams::ReferenceIterT) * numIters);
    if constexpr (HpShark::TestInitCudaMemory) {
        cudaMemset(gpuReferenceIters, 0xCD, sizeof(SharkFloatParams::ReferenceIterT) * numIters);
    } else {
        cudaMemset(gpuReferenceIters, 0, sizeof(SharkFloatParams::ReferenceIterT) * numIters);
    }

    // Prepare kernel arguments
    // Allocate memory for carryOuts and cumulativeCarries
    uint64_t *d_tempProducts;
    constexpr size_t BytesToAllocate =
        (AdditionalUInt64Global + CalculateNTTFrameSize<SharkFloatParams>()) * sizeof(uint64_t);
    cudaMalloc(&d_tempProducts, BytesToAllocate);

    if constexpr (!HpShark::TestInitCudaMemory) {
        cudaMemset(d_tempProducts, 0, BytesToAllocate);
    } else {
        cudaMemset(d_tempProducts, 0xCD, BytesToAllocate);
    }

    HpSharkReferenceResults<SharkFloatParams> *comboGpu;
    cudaMalloc(&comboGpu, sizeof(HpSharkReferenceResults<SharkFloatParams>));
    cudaMemcpy(
        comboGpu, &combo, sizeof(HpSharkReferenceResults<SharkFloatParams>), cudaMemcpyHostToDevice);
    assert(combo.OutputIters == nullptr); // Should not be set on input

    uint8_t byteToSet = HpShark::TestInitCudaMemory ? 0xCD : 0;

    // Note: we're clearing a specific set of members here, not the whole struct.
    cudaMemset(&comboGpu->Add.A_X2, byteToSet, sizeof(HpSharkFloat<SharkFloatParams>));
    cudaMemset(&comboGpu->Add.B_Y2, byteToSet, sizeof(HpSharkFloat<SharkFloatParams>));
    cudaMemset(&comboGpu->Add.D_2X, byteToSet, sizeof(HpSharkFloat<SharkFloatParams>));
    cudaMemset(&comboGpu->Add.Result1_A_B_C, byteToSet, sizeof(HpSharkFloat<SharkFloatParams>));
    cudaMemset(&comboGpu->Add.Result2_D_E, byteToSet, sizeof(HpSharkFloat<SharkFloatParams>));
    cudaMemset(&comboGpu->Multiply.ResultX2, byteToSet, sizeof(HpSharkFloat<SharkFloatParams>));
    cudaMemset(&comboGpu->Multiply.Result2XY, byteToSet, sizeof(HpSharkFloat<SharkFloatParams>));
    cudaMemset(&comboGpu->Multiply.ResultY2, byteToSet, sizeof(HpSharkFloat<SharkFloatParams>));

    // Build NTT plan + roots exactly like correctness path
    {
        SharkNTT::RootTables NTTRoots;
        SharkNTT::BuildRoots<SharkFloatParams>(
            SharkFloatParams::NTTPlan.N, SharkFloatParams::NTTPlan.stages, NTTRoots);

        CopyRootsToCuda<SharkFloatParams>(comboGpu->Multiply.Roots, NTTRoots);
        SharkNTT::DestroyRoots<SharkFloatParams>(false, NTTRoots);
    }

    void *kernelArgs[] = {
        (void *)&comboGpu, (void *)&numIters, (void *)&d_tempProducts, (void *)&gpuReferenceIters};

    cudaStream_t stream = nullptr;

    if constexpr (HpShark::CustomStream) {
        auto res = cudaStreamCreate(&stream); // Create a stream

        if (res != cudaSuccess) {
            std::cerr << "CUDA error in creating stream: " << cudaGetErrorString(res) << std::endl;
        }
    }

    cudaDeviceProp prop;
    int device_id = 0;

    if constexpr (HpShark::CustomStream) {
        cudaGetDeviceProperties(&prop, device_id);
        cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize,
                           prop.persistingL2CacheMaxSize); /* Set aside max possible size of L2 cache for
                                                              persisting accesses */

        auto setAccess = [&](void *ptr, size_t num_bytes) {
            cudaStreamAttrValue stream_attribute; // Stream level attributes data structure
            stream_attribute.accessPolicyWindow.base_ptr =
                reinterpret_cast<void *>(ptr); // Global Memory data pointer
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

        setAccess(comboGpu, sizeof(HpSharkReferenceResults<SharkFloatParams>));
        setAccess(d_tempProducts, 32 * SharkFloatParams::GlobalNumUint32 * sizeof(uint64_t));
    }

    {
        ScopedBenchmarkStopper stopper{timer};
        ComputeHpSharkReferenceGpuLoop<SharkFloatParams>(launchParams, stream, kernelArgs);
    }

    cudaMemcpy(
        &combo, comboGpu, sizeof(HpSharkReferenceResults<SharkFloatParams>), cudaMemcpyDeviceToHost);

    // TODO Costly double-buffer, this could be improved e.g. cuda host memory allocation?
    combo.OutputIters = new typename SharkFloatParams::ReferenceIterT[numIters];
    cudaMemcpy(combo.OutputIters,
               gpuReferenceIters,
               sizeof(SharkFloatParams::ReferenceIterT) * numIters,
               cudaMemcpyDeviceToHost);

    if (debugCombo != nullptr) {
        if constexpr (HpShark::DebugGlobalState) {
            debugCombo->MultiplyCounts.resize(SharkFloatParams::NumDebugMultiplyCounts);
            cudaMemcpy(debugCombo->MultiplyCounts.data(),
                       &d_tempProducts[AdditionalMultipliesOffset],
                       SharkFloatParams::NumDebugMultiplyCounts * sizeof(DebugGlobalCountRaw),
                       cudaMemcpyDeviceToHost);
        }
    }

    // Roots were device-allocated in CopyRootsToCuda; destroy like correctness does
    SharkNTT::DestroyRoots<SharkFloatParams>(true, comboGpu->Multiply.Roots);

    cudaFree(comboGpu);
    cudaFree(d_tempProducts);
    cudaFree(gpuReferenceIters);

    if constexpr (HpShark::CustomStream) {
        auto res = cudaStreamDestroy(stream); // Destroy the stream

        if (res != cudaSuccess) {
            std::cerr << "CUDA error in destroying stream: " << cudaGetErrorString(res) << std::endl;
        }
    }
}

#if defined(ENABLE_REFERENCE_KERNEL) || defined(ENABLE_FULL_KERNEL)
#define ExplicitlyInstantiateHpSharkReference(SharkFloatParams)                                         \
    template void InvokeHpSharkReferenceKernelProd<SharkFloatParams>(                                   \
        const HpShark::LaunchParams &launchParams,                                                          \
        HpSharkReferenceResults<SharkFloatParams> &,                                                    \
        mpf_t,                                                                                          \
        mpf_t,                                                                                          \
        uint64_t);                           \
    template void InvokeHpSharkReferenceKernelPerf<SharkFloatParams>(                                   \
        const HpShark::LaunchParams &launchParams,                                                          \
        BenchmarkTimer *timer,                                                                         \
        HpSharkReferenceResults<SharkFloatParams> & combo,                                              \
        uint64_t numIters,                                                                              \
        DebugGpuCombo *debugCombo);
#else
#define ExplicitlyInstantiateHpSharkReference(SharkFloatParams) ;
#endif

#define ExplicitlyInstantiate(SharkFloatParams) ExplicitlyInstantiateHpSharkReference(SharkFloatParams)

ExplicitInstantiateAll();
