#include "KernelInvoke.cuh"
#include "KernelInvokeInternal.cuh"

//
// This test is also something of a correctness test because
// it keeps track of the period and checks it subsequently.
//

template <class SharkFloatParams>
void
InvokeHpSharkReferenceKernelPerf(BenchmarkTimer &timer,
                                 HpSharkReferenceResults<SharkFloatParams> &combo,
                                 uint64_t numIters)
{

    // Prepare kernel arguments
    // Allocate memory for carryOuts and cumulativeCarries
    uint64_t *d_tempProducts;
    constexpr size_t BytesToAllocate =
        (AdditionalUInt64Global + CalculateNTTFrameSize<SharkFloatParams>()) * sizeof(uint64_t);
    cudaMalloc(&d_tempProducts, BytesToAllocate);

    if constexpr (!SharkTestInitCudaMemory) {
        cudaMemset(d_tempProducts, 0, BytesToAllocate);
    } else {
        cudaMemset(d_tempProducts, 0xCD, BytesToAllocate);
    }

    HpSharkReferenceResults<SharkFloatParams> *comboGpu;
    cudaMalloc(&comboGpu, sizeof(HpSharkReferenceResults<SharkFloatParams>));
    cudaMemcpy(
        comboGpu, &combo, sizeof(HpSharkReferenceResults<SharkFloatParams>), cudaMemcpyHostToDevice);

    uint8_t byteToSet = SharkTestInitCudaMemory ? 0xCD : 0;

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
    }

    void *kernelArgs[] = {(void *)&comboGpu, (void *)&numIters, (void *)&d_tempProducts};

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
        ComputeHpSharkReferenceGpuLoop<SharkFloatParams>(stream, kernelArgs);
    }

    cudaMemcpy(
        &combo, comboGpu, sizeof(HpSharkReferenceResults<SharkFloatParams>), cudaMemcpyDeviceToHost);

    // Roots were device-allocated in CopyRootsToCuda; destroy like correctness does
    SharkNTT::DestroyRoots<SharkFloatParams>(true, comboGpu->Multiply.Roots);

    cudaFree(comboGpu);
    cudaFree(d_tempProducts);

    if constexpr (SharkCustomStream) {
        auto res = cudaStreamDestroy(stream); // Destroy the stream

        if (res != cudaSuccess) {
            std::cerr << "CUDA error in destroying stream: " << cudaGetErrorString(res) << std::endl;
        }
    }
}

#ifdef ENABLE_REFERENCE_KERNEL
#define ExplicitlyInstantiateHpSharkReference(SharkFloatParams)                                         \
    template void InvokeHpSharkReferenceKernelPerf<SharkFloatParams>(                                   \
        BenchmarkTimer & timer, HpSharkReferenceResults<SharkFloatParams> & combo, uint64_t numIters);
#else
#define ExplicitlyInstantiateHpSharkReference(SharkFloatParams) ;
#endif

#define ExplicitlyInstantiate(SharkFloatParams) ExplicitlyInstantiateHpSharkReference(SharkFloatParams)

ExplicitInstantiateAll();
