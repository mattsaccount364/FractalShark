#pragma once
//package fractalzoomer.core.la;
//
//import fractalzoomer.core.Complex;
//import fractalzoomer.core.MantExpComplex;

#include "HDRFloatComplex.h"

template<typename IterType, class T>
class LAInfoDeep;

template<typename IterType, class T>
class GPU_LAInfoDeep;

template<typename IterType, class SubType>
class LAstep {
public:
    CUDA_CRAP LAstep() :
        step{},
        nextStageLAindex{},
        LAjdeep{},
        Refp1Deep{},
        newDzDeep{},
        unusable{true} {
    }

    IterType step;
    IterType nextStageLAindex;

public:
    using HDRFloatComplex = HDRFloatComplex<SubType>;
    LAInfoDeep<IterType, SubType> *LAjdeep;
    HDRFloatComplex Refp1Deep;
    HDRFloatComplex newDzDeep;

public:
    bool unusable;

    CUDA_CRAP HDRFloatComplex Evaluate(HDRFloatComplex DeltaSub0);
    CUDA_CRAP HDRFloatComplex EvaluateDzdcDeep(HDRFloatComplex z, HDRFloatComplex dzdc);
    CUDA_CRAP HDRFloatComplex EvaluateDzdc2Deep(HDRFloatComplex z, HDRFloatComplex dzdc2, HDRFloatComplex dzdc);
    CUDA_CRAP HDRFloatComplex getZ(HDRFloatComplex DeltaSubN);
};

template<typename IterType, class SubType>
CUDA_CRAP
LAstep<IterType, SubType>::HDRFloatComplex LAstep<IterType, SubType>::Evaluate(HDRFloatComplex DeltaSub0) {
    return LAjdeep->Evaluate(newDzDeep, DeltaSub0);
}

template<typename IterType, class SubType>
CUDA_CRAP
LAstep<IterType, SubType>::HDRFloatComplex LAstep<IterType, SubType>::EvaluateDzdcDeep(HDRFloatComplex z, HDRFloatComplex dzdc) {
    return LAjdeep->EvaluateDzdc(z, dzdc);
}

template<typename IterType, class SubType>
CUDA_CRAP
LAstep<IterType, SubType>::HDRFloatComplex LAstep<IterType, SubType>::EvaluateDzdc2Deep(HDRFloatComplex z, HDRFloatComplex dzdc2, HDRFloatComplex dzdc) {
    return LAjdeep->EvaluateDzdc2(z, dzdc2, dzdc);
}

template<typename IterType, class SubType>
CUDA_CRAP
LAstep<IterType, SubType>::HDRFloatComplex LAstep<IterType, SubType>::getZ(HDRFloatComplex DeltaSubN) {
    return Refp1Deep + DeltaSubN;
}



////////////////////////////////////////////////


template<typename IterType, class SubType>
class GPU_LAstep {
public:
    CUDA_CRAP GPU_LAstep() :
        step{},
        nextStageLAindex{},
        LAjdeep{},
        Refp1Deep{},
        newDzDeep{},
        unusable{ true } {
    }

    IterType step;
    IterType nextStageLAindex;

    using HDRFloatComplex = HDRFloatComplex<SubType>;
    const GPU_LAInfoDeep<IterType, SubType>* LAjdeep;
    HDRFloatComplex Refp1Deep;
    HDRFloatComplex newDzDeep;

public:
    bool unusable;

    CUDA_CRAP HDRFloatComplex Evaluate(HDRFloatComplex DeltaSub0) const;
    CUDA_CRAP HDRFloatComplex EvaluateDzdcDeep(HDRFloatComplex z, HDRFloatComplex dzdc) const;
    CUDA_CRAP HDRFloatComplex EvaluateDzdc2Deep(HDRFloatComplex z, HDRFloatComplex dzdc2, HDRFloatComplex dzdc) const;
    CUDA_CRAP HDRFloatComplex getZ(HDRFloatComplex DeltaSubN) const;
};

template<typename IterType, class SubType>
CUDA_CRAP
GPU_LAstep<IterType, SubType>::HDRFloatComplex GPU_LAstep<IterType, SubType>::Evaluate(HDRFloatComplex DeltaSub0) const {
    return LAjdeep->Evaluate(newDzDeep, DeltaSub0);
}

template<typename IterType, class SubType>
CUDA_CRAP
GPU_LAstep<IterType, SubType>::HDRFloatComplex GPU_LAstep<IterType, SubType>::EvaluateDzdcDeep(HDRFloatComplex z, HDRFloatComplex dzdc) const {
    return LAjdeep->EvaluateDzdc(z, dzdc);
}

template<typename IterType, class SubType>
CUDA_CRAP
GPU_LAstep<IterType, SubType>::HDRFloatComplex GPU_LAstep<IterType, SubType>::EvaluateDzdc2Deep(HDRFloatComplex z, HDRFloatComplex dzdc2, HDRFloatComplex dzdc) const {
    return LAjdeep->EvaluateDzdc2(z, dzdc2, dzdc);
}

template<typename IterType, class SubType>
CUDA_CRAP
GPU_LAstep<IterType, SubType>::HDRFloatComplex GPU_LAstep<IterType, SubType>::getZ(HDRFloatComplex DeltaSubN) const {
    return Refp1Deep + DeltaSubN;
}
