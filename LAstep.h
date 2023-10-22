#pragma once
//package fractalzoomer.core.la;
//
//import fractalzoomer.core.Complex;
//import fractalzoomer.core.MantExpComplex;

#include "HDRFloatComplex.h"

template<class T>
class LAInfoDeep;

template<class T>
class GPU_LAInfoDeep;

template<class SubType>
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
    LAInfoDeep<SubType> *LAjdeep;
    HDRFloatComplex Refp1Deep;
    HDRFloatComplex newDzDeep;

public:
    bool unusable;

    CUDA_CRAP HDRFloatComplex Evaluate(HDRFloatComplex DeltaSub0);
    CUDA_CRAP HDRFloatComplex EvaluateDzdcDeep(HDRFloatComplex z, HDRFloatComplex dzdc);
    CUDA_CRAP HDRFloatComplex EvaluateDzdc2Deep(HDRFloatComplex z, HDRFloatComplex dzdc2, HDRFloatComplex dzdc);
    CUDA_CRAP HDRFloatComplex getZ(HDRFloatComplex DeltaSubN);
};

template<class SubType>
CUDA_CRAP
LAstep<SubType>::HDRFloatComplex LAstep<SubType>::Evaluate(HDRFloatComplex DeltaSub0) {
    return LAjdeep->Evaluate(newDzDeep, DeltaSub0);
}

template<class SubType>
CUDA_CRAP
LAstep<SubType>::HDRFloatComplex LAstep<SubType>::EvaluateDzdcDeep(HDRFloatComplex z, HDRFloatComplex dzdc) {
    return LAjdeep->EvaluateDzdc(z, dzdc);
}

template<class SubType>
CUDA_CRAP
LAstep<SubType>::HDRFloatComplex LAstep<SubType>::EvaluateDzdc2Deep(HDRFloatComplex z, HDRFloatComplex dzdc2, HDRFloatComplex dzdc) {
    return LAjdeep->EvaluateDzdc2(z, dzdc2, dzdc);
}

template<class SubType>
CUDA_CRAP
LAstep<SubType>::HDRFloatComplex LAstep<SubType>::getZ(HDRFloatComplex DeltaSubN) {
    return Refp1Deep + DeltaSubN;
}



////////////////////////////////////////////////


template<class SubType>
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
    const GPU_LAInfoDeep<SubType>* LAjdeep;
    HDRFloatComplex Refp1Deep;
    HDRFloatComplex newDzDeep;

public:
    bool unusable;

    CUDA_CRAP HDRFloatComplex Evaluate(HDRFloatComplex DeltaSub0) const;
    CUDA_CRAP HDRFloatComplex EvaluateDzdcDeep(HDRFloatComplex z, HDRFloatComplex dzdc) const;
    CUDA_CRAP HDRFloatComplex EvaluateDzdc2Deep(HDRFloatComplex z, HDRFloatComplex dzdc2, HDRFloatComplex dzdc) const;
    CUDA_CRAP HDRFloatComplex getZ(HDRFloatComplex DeltaSubN) const;
};

template<class SubType>
CUDA_CRAP
GPU_LAstep<SubType>::HDRFloatComplex GPU_LAstep<SubType>::Evaluate(HDRFloatComplex DeltaSub0) const {
    return LAjdeep->Evaluate(newDzDeep, DeltaSub0);
}

template<class SubType>
CUDA_CRAP
GPU_LAstep<SubType>::HDRFloatComplex GPU_LAstep<SubType>::EvaluateDzdcDeep(HDRFloatComplex z, HDRFloatComplex dzdc) const {
    return LAjdeep->EvaluateDzdc(z, dzdc);
}

template<class SubType>
CUDA_CRAP
GPU_LAstep<SubType>::HDRFloatComplex GPU_LAstep<SubType>::EvaluateDzdc2Deep(HDRFloatComplex z, HDRFloatComplex dzdc2, HDRFloatComplex dzdc) const {
    return LAjdeep->EvaluateDzdc2(z, dzdc2, dzdc);
}

template<class SubType>
CUDA_CRAP
GPU_LAstep<SubType>::HDRFloatComplex GPU_LAstep<SubType>::getZ(HDRFloatComplex DeltaSubN) const {
    return Refp1Deep + DeltaSubN;
}
