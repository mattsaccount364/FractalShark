#pragma once

#include "HDRFloat.h"

template<class T>
class ScaledBLA;

template<class T>
class BLA {
protected:
    T r2;
    T Ax;
    T Ay;
    T Bx;
    T By;
    int l;

    friend class ScaledBLA<float>;
    friend class ScaledBLA<double>;

public:

    BLA();

    CUDA_CRAP constexpr BLA(T r2,
                            T RealA,
                            T ImagA,
                            T RealB,
                            T ImagB,
                            int l);

    CUDA_CRAP void getValue(T &RealDeltaSubN,
                            T &ImagDeltaSubN,
                            const T &RealDeltaSub0,
                            const T &ImagDeltaSub0) const;

    CUDA_CRAP T hypotA() const;
    CUDA_CRAP T hypotB() const;
    CUDA_CRAP static BLA getGenericStep(T r2,
                                                  T RealA,
                                                  T ImagA,
                                                  T RealB,
                                                  T ImagB,
                                                  int l);

    // A = y.A * x.A
    CUDA_CRAP static void getNewA(const BLA &x,
                                            const BLA &y,
                                            T &RealValue,
                                            T &ImagValue);

    // B = y.A * x.B + y.B
    CUDA_CRAP static void getNewB(const BLA &x,
                                            const BLA &y,
                                            T& RealValue,
                                            T& ImagValue);

    CUDA_CRAP int getL() const;
    CUDA_CRAP T getR2() const;
    CUDA_CRAP const T *getR2Addr() const;
};

