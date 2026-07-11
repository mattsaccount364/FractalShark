#include "ReferenceOrbit2GpuStub.h"

#include "Exceptions.h"
#include "ExplicitInstantiate.h"
#include "HpSharkFloat.h"

#include <cuda_runtime.h>
#include <sstream>

namespace {

template <class SharkFloatParams>
__global__ void
ReferenceOrbit2GpuStubKernel(HpSharkReferenceResults<SharkFloatParams> *combo)
{
    combo->OutputIterCount = 0;
    combo->PeriodicityStatus = PeriodicityResult::Unknown;
}

void
CheckCuda(cudaError_t error, const char *operation)
{
    if (error == cudaSuccess)
        return;

    std::ostringstream message;
    message << operation << " failed: " << cudaGetErrorString(error) << " (code "
            << static_cast<int>(error) << ")";
    throw FractalSharkSeriousException(message.str());
}

} // namespace

namespace HpShark {

template <class SharkFloatParams>
void
InvokeReferenceOrbit2GpuStub(HpSharkReferenceResults<SharkFloatParams> &combo)
{
    const auto stream = *reinterpret_cast<cudaStream_t *>(&combo.stream);
    ReferenceOrbit2GpuStubKernel<SharkFloatParams><<<1, 1, 0, stream>>>(combo.comboGpu);
    CheckCuda(cudaGetLastError(), "ReferenceOrbit2GpuStubKernel launch");
    CheckCuda(cudaStreamSynchronize(stream), "ReferenceOrbit2GpuStubKernel synchronization");
    CheckCuda(cudaMemcpy(&combo,
                         combo.comboGpu,
                         sizeof(HpSharkReferenceResults<SharkFloatParams>),
                         cudaMemcpyDeviceToHost),
              "ReferenceOrbit2GpuStub result copy");
}

#define ExplicitlyInstantiateReferenceOrbit2GpuStub(SharkFloatParams)                                   \
    template void InvokeReferenceOrbit2GpuStub<SharkFloatParams>(                                       \
        HpSharkReferenceResults<SharkFloatParams> & combo);

#define ExplicitlyInstantiate(SharkFloatParams)                                                         \
    ExplicitlyInstantiateReferenceOrbit2GpuStub(SharkFloatParams)

ExplicitInstantiateAll();

} // namespace HpShark
