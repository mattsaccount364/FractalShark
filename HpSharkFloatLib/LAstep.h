#pragma once
// package fractalzoomer.core.la;
//
// import fractalzoomer.core.Complex;
// import fractalzoomer.core.MantExpComplex;

#include "HDRFloatComplex.h"

template <typename IterType, class Float, class SubType, PerturbExtras PExtras> class LAInfoDeep;

template <typename IterType, class Float, class SubType> class GPU_LAInfoDeep;

template <typename IterType, class Float, class SubType, PerturbExtras PExtras> class LAstep {
public:
    CUDA_CRAP
    LAstep()
        : step{}, nextStageLAindex{}, LAjdeep{}, Refp1Deep{}, newDzDeep{}, unusable{true}
    {
    }

    IterType step;
    IterType nextStageLAindex;

public:
    using HDRFloatComplex = std::conditional<std::is_same<Float, HDRFloat<float>>::value ||
                                                 std::is_same<Float, HDRFloat<double>>::value ||
                                                 std::is_same<Float, HDRFloat<MattDblflt>>::value,
                                             ::HDRFloatComplex<SubType>,
                                             ::FloatComplex<SubType>>::type;

    LAInfoDeep<IterType, Float, SubType, PExtras> *LAjdeep;
    HDRFloatComplex Refp1Deep;
    HDRFloatComplex newDzDeep;

public:
    bool unusable;

    CUDA_CRAP HDRFloatComplex Evaluate(HDRFloatComplex &dz, HDRFloatComplex DeltaSub0);
    CUDA_CRAP HDRFloatComplex Evaluate(HDRFloatComplex DeltaSub0);
    CUDA_CRAP void EvaluateDzdz(HDRFloatComplex &dz,
                                HDRFloatComplex &dzdz,
                                const HDRFloatComplex &dc) const;
    CUDA_CRAP void EvaluateDzdcDeep(HDRFloatComplex &dz,
                                    HDRFloatComplex &dzdc,
                                    const Float &ScalingFactor) const;
    CUDA_CRAP void EvaluateDerivatives(HDRFloatComplex &dz,
                                       HDRFloatComplex &dzdz,
                                       HDRFloatComplex &dzdc,
                                       const Float &ScalingFactor) const;
    CUDA_CRAP HDRFloatComplex EvaluateDzdcDeep(HDRFloatComplex z, HDRFloatComplex dzdc);
    CUDA_CRAP HDRFloatComplex EvaluateDzdc2Deep(HDRFloatComplex z,
                                                HDRFloatComplex dzdc2,
                                                HDRFloatComplex dzdc);
    CUDA_CRAP HDRFloatComplex getZ(HDRFloatComplex DeltaSubN);
};

template <typename IterType, class Float, class SubType, PerturbExtras PExtras>
CUDA_CRAP void
LAstep<IterType, Float, SubType, PExtras>::EvaluateDzdz(HDRFloatComplex &dz,
                                                        HDRFloatComplex &dzdz,
                                                        const HDRFloatComplex &dc) const
{
    LAjdeep->EvaluateDzdz(dz, dzdz, dc);
}

template <typename IterType, class Float, class SubType, PerturbExtras PExtras>
CUDA_CRAP void LAstep<IterType, Float, SubType, PExtras>::EvaluateDzdcDeep(HDRFloatComplex &dz,
                                                                        HDRFloatComplex &dzdc,
                                                                        const Float &ScalingFactor) const
{
    LAjdeep->EvaluateDzdc(dz, dzdc, ScalingFactor);
}

template <typename IterType, class Float, class SubType, PerturbExtras PExtras>
CUDA_CRAP void
LAstep<IterType, Float, SubType, PExtras>::EvaluateDerivatives(HDRFloatComplex &dz,
                                                               HDRFloatComplex &dzdz,
                                                               HDRFloatComplex &dzdc,
                                                               const Float &ScalingFactor) const
{
    LAjdeep->EvaluateDerivatives(dz, dzdz, dzdc, ScalingFactor);
}

template <typename IterType, class Float, class SubType, PerturbExtras PExtras>
CUDA_CRAP LAstep<IterType, Float, SubType, PExtras>::HDRFloatComplex
LAstep<IterType, Float, SubType, PExtras>::Evaluate(HDRFloatComplex DeltaSub0)
{
    return LAjdeep->Evaluate(newDzDeep, DeltaSub0);
}

template <typename IterType, class Float, class SubType, PerturbExtras PExtras>
CUDA_CRAP LAstep<IterType, Float, SubType, PExtras>::HDRFloatComplex
LAstep<IterType, Float, SubType, PExtras>::Evaluate(HDRFloatComplex &dz, HDRFloatComplex DeltaSub0)
{
    return LAjdeep->Evaluate(dz, DeltaSub0);
}


template <typename IterType, class Float, class SubType, PerturbExtras PExtras>
CUDA_CRAP LAstep<IterType, Float, SubType, PExtras>::HDRFloatComplex
LAstep<IterType, Float, SubType, PExtras>::EvaluateDzdcDeep(HDRFloatComplex z, HDRFloatComplex dzdc)
{
    return LAjdeep->EvaluateDzdc(z, dzdc);
}

template <typename IterType, class Float, class SubType, PerturbExtras PExtras>
CUDA_CRAP LAstep<IterType, Float, SubType, PExtras>::HDRFloatComplex
LAstep<IterType, Float, SubType, PExtras>::EvaluateDzdc2Deep(HDRFloatComplex z,
                                                             HDRFloatComplex dzdc2,
                                                             HDRFloatComplex dzdc)
{
    return LAjdeep->EvaluateDzdc2(z, dzdc2, dzdc);
}

template <typename IterType, class Float, class SubType, PerturbExtras PExtras>
CUDA_CRAP LAstep<IterType, Float, SubType, PExtras>::HDRFloatComplex
LAstep<IterType, Float, SubType, PExtras>::getZ(HDRFloatComplex DeltaSubN)
{
    return Refp1Deep + DeltaSubN;
}

////////////////////////////////////////////////

template <typename IterType, class Float, class SubType> class GPU_LAstep {
public:
    CUDA_CRAP
    GPU_LAstep()
        : step{}, nextStageLAindex{}, LAjdeep{}, Refp1Deep{}, newDzDeep{}, unusable{true}
    {
    }

    IterType step;
    IterType nextStageLAindex;

    using HDRFloatComplex =
        std::conditional<std::is_same<Float, HDRFloat<float>>::value ||
                             std::is_same<Float, HDRFloat<double>>::value ||
                             std::is_same<Float, HDRFloat<CudaDblflt<MattDblflt>>>::value,
                         ::HDRFloatComplex<SubType>,
                         ::FloatComplex<SubType>>::type;

    const GPU_LAInfoDeep<IterType, Float, SubType> *LAjdeep;
    HDRFloatComplex Refp1Deep;
    HDRFloatComplex newDzDeep;

public:
    bool unusable;

    CUDA_CRAP HDRFloatComplex Evaluate(HDRFloatComplex DeltaSub0) const;
    CUDA_CRAP HDRFloatComplex EvaluateDzdcDeep(HDRFloatComplex z, HDRFloatComplex dzdc) const;
    CUDA_CRAP HDRFloatComplex EvaluateDzdc2Deep(HDRFloatComplex z,
                                                HDRFloatComplex dzdc2,
                                                HDRFloatComplex dzdc) const;
    CUDA_CRAP HDRFloatComplex getZ(HDRFloatComplex DeltaSubN) const;
};

template <typename IterType, class Float, class SubType>
CUDA_CRAP GPU_LAstep<IterType, Float, SubType>::HDRFloatComplex
GPU_LAstep<IterType, Float, SubType>::Evaluate(HDRFloatComplex DeltaSub0) const
{
    return LAjdeep->Evaluate(newDzDeep, DeltaSub0);
}

template <typename IterType, class Float, class SubType>
CUDA_CRAP GPU_LAstep<IterType, Float, SubType>::HDRFloatComplex
GPU_LAstep<IterType, Float, SubType>::EvaluateDzdcDeep(HDRFloatComplex z, HDRFloatComplex dzdc) const
{
    return LAjdeep->EvaluateDzdc(z, dzdc);
}

template <typename IterType, class Float, class SubType>
CUDA_CRAP GPU_LAstep<IterType, Float, SubType>::HDRFloatComplex
GPU_LAstep<IterType, Float, SubType>::EvaluateDzdc2Deep(HDRFloatComplex z,
                                                        HDRFloatComplex dzdc2,
                                                        HDRFloatComplex dzdc) const
{
    return LAjdeep->EvaluateDzdc2(z, dzdc2, dzdc);
}

template <typename IterType, class Float, class SubType>
CUDA_CRAP GPU_LAstep<IterType, Float, SubType>::HDRFloatComplex
GPU_LAstep<IterType, Float, SubType>::getZ(HDRFloatComplex DeltaSubN) const
{
    return Refp1Deep + DeltaSubN;
}
