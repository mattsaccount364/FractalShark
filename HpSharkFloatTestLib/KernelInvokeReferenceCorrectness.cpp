#include "BenchmarkTimer.h"
#include "Exceptions.h"
#include "HpSharkFloat.h"
#include "KernelInvoke.h"
#include "KernelInvokeReferenceSetup.h"

#include <cuda_runtime.h>
#include <sstream>

namespace HpShark {

template <class SharkFloatParams>
void
InvokeHpSharkReferenceKernelCorrectness(const HpShark::LaunchParams &launchParams,
                                        BenchmarkTimer &timer,
                                        HpSharkReferenceResults<SharkFloatParams> &results,
                                        DebugGpuCombo *debugResults)
{
    auto preparedTables = PrepareHpSharkReferenceTables<SharkFloatParams>(
        launchParams, results.CReal, results.CImag, SharkFloatParams::GlobalNumUint32);
    InvokeHpSharkReferenceKernelCorrectness(launchParams, timer, results, debugResults, *preparedTables);
}

template <class SharkFloatParams>
void
InvokeHpSharkReferenceKernelCorrectness(const HpShark::LaunchParams &launchParams,
                                        BenchmarkTimer &timer,
                                        HpSharkReferenceResults<SharkFloatParams> &results,
                                        DebugGpuCombo *debugResults,
                                        ReferencePreparedTables<SharkFloatParams> &preparedTables)
{
    GpuOrbitSession<SharkFloatParams> session(
        launchParams, results.RadiusY, results.CReal, results.CImag, preparedTables, debugResults);
    auto &sessionResults = session.GetResults();
    sessionResults.ZReal = results.ZReal;
    sessionResults.ZImag = results.ZImag;
    sessionResults.DzdcReal = results.DzdcReal;
    sessionResults.DzdcImag = results.DzdcImag;
    sessionResults.D2Real = results.D2Real;
    sessionResults.D2Imag = results.D2Imag;

    const cudaError_t copyResult = cudaMemcpy(
        sessionResults.DeviceResults, &sessionResults, sizeof(sessionResults), cudaMemcpyHostToDevice);
    if (copyResult != cudaSuccess) {
        std::ostringstream message;
        message << "cudaMemcpy(reference correctness input H2D) failed: "
                << cudaGetErrorString(copyResult) << " (code " << static_cast<int>(copyResult) << ')';
        throw FractalSharkSeriousException(message.str());
    }

    {
        ScopedBenchmarkStopper stopper{timer};
        session.InvokeChunk(1);
    }

    results.ZReal = sessionResults.ZReal;
    results.ZImag = sessionResults.ZImag;
    results.DzdcReal = sessionResults.DzdcReal;
    results.DzdcImag = sessionResults.DzdcImag;
    results.D2Real = sessionResults.D2Real;
    results.D2Imag = sessionResults.D2Imag;
    results.PeriodicityStatus = sessionResults.PeriodicityStatus;
    results.OutputIterCount = sessionResults.OutputIterCount;
}

#define ExplicitlyInstantiate(SharkFloatParams)                                                         \
    template void InvokeHpSharkReferenceKernelCorrectness<SharkFloatParams>(                            \
        const HpShark::LaunchParams &,                                                                  \
        BenchmarkTimer &,                                                                               \
        HpSharkReferenceResults<SharkFloatParams> &,                                                    \
        DebugGpuCombo *);                                                                               \
    template void InvokeHpSharkReferenceKernelCorrectness<SharkFloatParams>(                            \
        const HpShark::LaunchParams &,                                                                  \
        BenchmarkTimer &,                                                                               \
        HpSharkReferenceResults<SharkFloatParams> &,                                                    \
        DebugGpuCombo *,                                                                                \
        ReferencePreparedTables<SharkFloatParams> &);

ExplicitInstantiateAll();

#undef ExplicitlyInstantiate

} // namespace HpShark
