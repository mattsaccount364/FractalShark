#include "GPU_ReferenceIter.h"
// #include "KernelInvoke.h"
#include "KernelInvokeInternal.h"
#include "Vectors.h"

namespace HpShark {

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
// combo.OutputIterCount is set
// combo.PeriodicityStatus is set
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
    inputX->MpfToHpGpu(
        srcX, HpSharkFloat<SharkFloatParams>::DefaultMpirBits, InjectNoiseInLowOrder::Enable);
    inputY->MpfToHpGpu(
        srcY, HpSharkFloat<SharkFloatParams>::DefaultMpirBits, InjectNoiseInLowOrder::Enable);

    combo.Add.C_A = *inputX;
    combo.Add.E_B = *inputY;
    combo.Multiply.A = *inputX;
    combo.Multiply.B = *inputY;
    // Note: leave combo.dzdcX and combo.dzdcY unchanged
    combo.OutputIterCount = 0;
    assert(memcmp(&combo.Add.C_A, &combo.Multiply.A, sizeof(HpSharkFloat<SharkFloatParams>)) == 0);
    assert(memcmp(&combo.Add.E_B, &combo.Multiply.B, sizeof(HpSharkFloat<SharkFloatParams>)) == 0);
    assert(combo.RadiusY.mantissa != 0); // RadiusY must be set.  Does 0 have any useful meaning here?
    assert(false);
    InvokeHpSharkReferenceKernelTestPerf<SharkFloatParams>(launchParams, combo, numIters, nullptr);
}

template <class SharkFloatParams>
void
InitHpSharkKernelTest(const HpShark::LaunchParams &launchParams,
                      HpSharkReferenceResults<SharkFloatParams> &combo,
                      DebugGpuCombo *debugCombo)
{
    // Prepare kernel arguments
    // Allocate memory for carryOuts and cumulativeCarries
    constexpr size_t BytesToAllocate =
        (HpShark::AdditionalUInt64Global + HpShark::CalculateNTTFrameSize<SharkFloatParams>()) *
        sizeof(uint64_t);
    cudaMalloc(&combo.d_tempProducts, BytesToAllocate);

    if constexpr (!HpShark::TestInitCudaMemory) {
        cudaMemset(combo.d_tempProducts, 0, BytesToAllocate);
    } else {
        cudaMemset(combo.d_tempProducts, 0xCD, BytesToAllocate);
    }

    // Host only
    combo.kernelArgs[0] = (void *)&combo.comboGpu;
    combo.kernelArgs[1] = (void *)&combo.d_tempProducts;
    combo.stream = 0;

    static_assert(sizeof(cudaStream_t) == sizeof(combo.stream),
                  "cudaStream_t size mismatch with combo.stream");

    if constexpr (HpShark::CustomStream) {
        auto &stream = *reinterpret_cast<cudaStream_t *>(&combo.stream);
        auto res = cudaStreamCreate(&stream); // Create a stream

        if (res != cudaSuccess) {
            std::cerr << "CUDA error in creating stream: " << cudaGetErrorString(res) << std::endl;
        }
    }

    cudaMalloc(&combo.comboGpu, sizeof(HpSharkReferenceResults<SharkFloatParams>));

    // Note; shallow copy; we will memset specific members below
    cudaMemcpy(combo.comboGpu,
               &combo,
               sizeof(HpSharkReferenceResults<SharkFloatParams>),
               cudaMemcpyHostToDevice);

    uint8_t byteToSet = HpShark::TestInitCudaMemory ? 0xCD : 0;

    // Note: we're clearing a specific set of members here, not the whole struct.
    cudaMemset(&combo.comboGpu->Add.A_X2, byteToSet, sizeof(HpSharkFloat<SharkFloatParams>));
    cudaMemset(&combo.comboGpu->Add.B_Y2, byteToSet, sizeof(HpSharkFloat<SharkFloatParams>));
    cudaMemset(&combo.comboGpu->Add.D_2X, byteToSet, sizeof(HpSharkFloat<SharkFloatParams>));
    cudaMemset(&combo.comboGpu->Add.Result1_A_B_C, byteToSet, sizeof(HpSharkFloat<SharkFloatParams>));
    cudaMemset(&combo.comboGpu->Add.Result2_D_E, byteToSet, sizeof(HpSharkFloat<SharkFloatParams>));
    cudaMemset(&combo.comboGpu->Multiply.ResultX2, byteToSet, sizeof(HpSharkFloat<SharkFloatParams>));
    cudaMemset(&combo.comboGpu->Multiply.Result2XY, byteToSet, sizeof(HpSharkFloat<SharkFloatParams>));
    cudaMemset(&combo.comboGpu->Multiply.ResultY2, byteToSet, sizeof(HpSharkFloat<SharkFloatParams>));

    // Build NTT plan + roots exactly like correctness path
    {
        SharkNTT::RootTables NTTRoots;
        SharkNTT::BuildRoots<SharkFloatParams>(
            SharkFloatParams::NTTPlan.N, SharkFloatParams::NTTPlan.stages, NTTRoots);

        CopyRootsToCuda<SharkFloatParams>(combo.comboGpu->Multiply.Roots, NTTRoots);
        SharkNTT::DestroyRoots<SharkFloatParams>(false, NTTRoots);
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
            auto &stream = *reinterpret_cast<cudaStream_t *>(&combo.stream);
            cudaError_t err =
                cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);
            if (err != cudaSuccess) {
                std::cerr << "CUDA error in setting stream attribute: " << cudaGetErrorString(err)
                          << std::endl;
            }
        };

        setAccess(combo.comboGpu, sizeof(HpSharkReferenceResults<SharkFloatParams>));
        setAccess(combo.d_tempProducts, 32 * SharkFloatParams::GlobalNumUint32 * sizeof(uint64_t));
    }
}

//
// This test is also something of a correctness test because
// it keeps track of the period and checks it subsequently.
//
template <class SharkFloatParams>
void
InvokeHpSharkReferenceKernelTestPerf(const HpShark::LaunchParams &launchParams,
                                     HpSharkReferenceResults<SharkFloatParams> &combo,
                                     uint64_t numIters,
                                     DebugGpuCombo *debugCombo)
{
    auto *comboGpu = combo.comboGpu;
    cudaMemcpy(&comboGpu->MaxRuntimeIters, &numIters, sizeof(uint64_t), cudaMemcpyHostToDevice);

    ComputeHpSharkReferenceGpuLoop<SharkFloatParams>(
        launchParams, *reinterpret_cast<cudaStream_t *>(&combo.stream), combo.kernelArgs);

    // Note: comboGpu is device pointer
    // Note: we copy everything back, even host-only stuff
    cudaMemcpy(
        &combo, comboGpu, sizeof(HpSharkReferenceResults<SharkFloatParams>), cudaMemcpyDeviceToHost);
}

template <class SharkFloatParams>
void
ShutdownHpSharkKernel(const HpShark::LaunchParams &launchParams,
                      HpSharkReferenceResults<SharkFloatParams> &combo,
                      DebugGpuCombo *debugCombo)
{
    if (debugCombo != nullptr) {
        if constexpr (HpShark::DebugGlobalState) {
            debugCombo->MultiplyCounts.resize(SharkFloatParams::NumDebugMultiplyCounts);
            cudaMemcpy(debugCombo->MultiplyCounts.data(),
                       &combo.d_tempProducts[HpShark::AdditionalMultipliesOffset],
                       SharkFloatParams::NumDebugMultiplyCounts * sizeof(DebugGlobalCountRaw),
                       cudaMemcpyDeviceToHost);
        }
    }

    // Roots were device-allocated in CopyRootsToCuda; destroy like correctness does
    SharkNTT::DestroyRoots<SharkFloatParams>(true, combo.comboGpu->Multiply.Roots);

    cudaFree(combo.comboGpu);
    cudaFree(combo.d_tempProducts);

    if constexpr (HpShark::CustomStream) {
        auto &stream = *reinterpret_cast<cudaStream_t *>(&combo.stream);
        auto res = cudaStreamDestroy(stream); // Destroy the stream

        if (res != cudaSuccess) {
            std::cerr << "CUDA error in destroying stream: " << cudaGetErrorString(res) << std::endl;
        }
    }
}

#if defined(ENABLE_FULL_KERNEL)
#define ExplicitlyInstantiateHpSharkReference(SharkFloatParams)                                         \
    template void InvokeHpSharkReferenceKernelProd<SharkFloatParams>(                                   \
        const HpShark::LaunchParams &launchParams,                                                      \
        HpSharkReferenceResults<SharkFloatParams> &,                                                    \
        mpf_t,                                                                                          \
        mpf_t,                                                                                          \
        uint64_t);                                                                                      \
    template void InvokeHpSharkReferenceKernelTestPerf<SharkFloatParams>(                               \
        const HpShark::LaunchParams &launchParams,                                                      \
        HpSharkReferenceResults<SharkFloatParams> &combo,                                               \
        uint64_t numIters,                                                                              \
        DebugGpuCombo *debugCombo);                                                                     \
    template void InitHpSharkKernelTest<SharkFloatParams>(                                              \
        const HpShark::LaunchParams &launchParams,                                                      \
        HpSharkReferenceResults<SharkFloatParams> &combo,                                               \
        DebugGpuCombo *debugCombo);                                                                     \
    template void ShutdownHpSharkKernel<SharkFloatParams>(                                              \
        const HpShark::LaunchParams &launchParams,                                                      \
        HpSharkReferenceResults<SharkFloatParams> &combo,                                               \
        DebugGpuCombo *debugCombo);
#else
#define ExplicitlyInstantiateHpSharkReference(SharkFloatParams) ;
#endif

#define ExplicitlyInstantiate(SharkFloatParams) ExplicitlyInstantiateHpSharkReference(SharkFloatParams)

ExplicitInstantiateAll();

} // namespace HpShark
