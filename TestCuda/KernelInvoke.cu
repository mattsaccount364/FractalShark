#include "KernelInvoke.cuh"
#include "KernelInvokeInternal.cuh"

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
    ExplicitlyInstantiateMultiplyNTT(SharkFloatParams) \
    ExplicitlyInstantiateHpSharkReference(SharkFloatParams)

ExplicitInstantiateAll();
