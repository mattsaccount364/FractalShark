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

template<class T>
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

    size_t step;
    size_t nextStageLAindex;

public:
    using HDRFloatComplex = T;
    LAInfoDeep<float> *LAjdeep;
    HDRFloatComplex Refp1Deep;
    HDRFloatComplex newDzDeep;

public:
    bool unusable;

    CUDA_CRAP HDRFloatComplex Evaluate(HDRFloatComplex DeltaSub0);
    CUDA_CRAP HDRFloatComplex EvaluateDzdcDeep(HDRFloatComplex z, HDRFloatComplex dzdc);
    CUDA_CRAP HDRFloatComplex EvaluateDzdc2Deep(HDRFloatComplex z, HDRFloatComplex dzdc2, HDRFloatComplex dzdc);
    CUDA_CRAP HDRFloatComplex getZ(HDRFloatComplex DeltaSubN);
};

template<class T>
CUDA_CRAP
LAstep<T>::HDRFloatComplex LAstep<T>::Evaluate(HDRFloatComplex DeltaSub0) {
    return LAjdeep->Evaluate(newDzDeep, DeltaSub0);
}

template<class T>
CUDA_CRAP
LAstep<T>::HDRFloatComplex LAstep<T>::EvaluateDzdcDeep(HDRFloatComplex z, HDRFloatComplex dzdc) {
    return LAjdeep->EvaluateDzdc(z, dzdc);
}

template<class T>
CUDA_CRAP
LAstep<T>::HDRFloatComplex LAstep<T>::EvaluateDzdc2Deep(HDRFloatComplex z, HDRFloatComplex dzdc2, HDRFloatComplex dzdc) {
    return LAjdeep->EvaluateDzdc2(z, dzdc2, dzdc);
}

template<class T>
CUDA_CRAP
LAstep<T>::HDRFloatComplex LAstep<T>::getZ(HDRFloatComplex DeltaSubN) {
    return Refp1Deep.plus(DeltaSubN);
}



////////////////////////////////////////////////


template<class T>
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

    int32_t step;
    int32_t nextStageLAindex;

public:
    using HDRFloatComplex = T;
    const GPU_LAInfoDeep<float>* LAjdeep;
    HDRFloatComplex Refp1Deep;
    HDRFloatComplex newDzDeep;

public:
    bool unusable;

    CUDA_CRAP HDRFloatComplex Evaluate(HDRFloatComplex DeltaSub0) const;
    CUDA_CRAP HDRFloatComplex EvaluateDzdcDeep(HDRFloatComplex z, HDRFloatComplex dzdc) const;
    CUDA_CRAP HDRFloatComplex EvaluateDzdc2Deep(HDRFloatComplex z, HDRFloatComplex dzdc2, HDRFloatComplex dzdc) const;
    CUDA_CRAP HDRFloatComplex getZ(HDRFloatComplex DeltaSubN) const;
};

template<class T>
CUDA_CRAP
GPU_LAstep<T>::HDRFloatComplex GPU_LAstep<T>::Evaluate(HDRFloatComplex DeltaSub0) const {
    return LAjdeep->Evaluate(newDzDeep, DeltaSub0);
}

template<class T>
CUDA_CRAP
GPU_LAstep<T>::HDRFloatComplex GPU_LAstep<T>::EvaluateDzdcDeep(HDRFloatComplex z, HDRFloatComplex dzdc) const {
    return LAjdeep->EvaluateDzdc(z, dzdc);
}

template<class T>
CUDA_CRAP
GPU_LAstep<T>::HDRFloatComplex GPU_LAstep<T>::EvaluateDzdc2Deep(HDRFloatComplex z, HDRFloatComplex dzdc2, HDRFloatComplex dzdc) const {
    return LAjdeep->EvaluateDzdc2(z, dzdc2, dzdc);
}

template<class T>
CUDA_CRAP
GPU_LAstep<T>::HDRFloatComplex GPU_LAstep<T>::getZ(HDRFloatComplex DeltaSubN) const {
    return Refp1Deep.plus(DeltaSubN);
}
