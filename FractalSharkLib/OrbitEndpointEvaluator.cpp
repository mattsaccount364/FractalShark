#include "stdafx.h"

#include "OrbitEndpointEvaluator.h"

#include "AbortMonitor.h"
#include "GpuPrecisionDispatch.h"
#include "MpirOrbitEval.h"

#include "KernelInvoke.h"

uint64_t
EvaluateCriticalOrbitAndDerivs(NRInnerLoopBackend backend,
                               const mpf_complex &c,
                               uint64_t period,
                               mpf_complex &z,
                               mpf_complex &dzdc,
                               HDRFloat<double> &d2r,
                               HDRFloat<double> &d2i,
                               mp_bitcnt_t coord_prec,
                               mp_bitcnt_t deriv_prec,
                               uint64_t startIter,
                               void (*onProgress)(uint64_t, void *),
                               void *progressCtx)
{
    uint64_t completed = 0;

    switch (backend) {
        case NRInnerLoopBackend::GPU: {
            auto runGpu = [&]<class NRParams>() {
                const uint64_t requestedPrecisionLimbs = (static_cast<uint64_t>(coord_prec) + 31u) / 32u;
                const uint32_t effectivePrecisionLimbs = GetReferenceEffectivePrecisionLimbs(
                    requestedPrecisionLimbs, NRParams::GlobalNumUint32);
                const HpShark::LaunchParams launchParams{0, 0};
                completed = HpShark::EvaluateCriticalOrbitAndDerivs_GPU<NRParams>(
                    c.re,
                    c.im,
                    period,
                    z.re,
                    z.im,
                    dzdc.re,
                    dzdc.im,
                    d2r,
                    d2i,
                    launchParams,
                    effectivePrecisionLimbs,
                    startIter,
                    AbortMonitor::GetStopCalculatingGlobal,
                    onProgress,
                    progressCtx,
                    64);
            };
            DispatchByLimbCount<SharkParamsNRFamily>(BitsToSupportedLimbCount(coord_prec), runGpu);
        } break;

        case NRInnerLoopBackend::CpuMT:
            completed = EvaluateCriticalOrbitAndDerivsMT(c,
                                                         period,
                                                         z,
                                                         dzdc,
                                                         d2r,
                                                         d2i,
                                                         deriv_prec,
                                                         coord_prec,
                                                         startIter,
                                                         onProgress,
                                                         progressCtx);
            break;

        case NRInnerLoopBackend::CpuST:
            completed = EvaluateCriticalOrbitAndDerivsST(c,
                                                         period,
                                                         z,
                                                         dzdc,
                                                         d2r,
                                                         d2i,
                                                         deriv_prec,
                                                         coord_prec,
                                                         startIter,
                                                         onProgress,
                                                         progressCtx);
            break;
    }

    return completed;
}
