#pragma once

#ifndef __CUDACC__ 
#define __host__
#define __device__
#endif

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

    BLA() = default;

    __host__ __device__ BLA(T r2,
                            T RealA,
                            T ImagA,
                            T RealB,
                            T ImagB,
                            int l);

    __host__ __device__ void getValue(T &RealDeltaSubN,
                                      T &ImagDeltaSubN,
                                      T RealDeltaSub0,
                                      T ImagDeltaSub0);

    __host__ __device__ T hypotA() const;
    __host__ __device__ T hypotB() const;
    __host__ __device__ static BLA getGenericStep(T r2,
                                                  T RealA,
                                                  T ImagA,
                                                  T RealB,
                                                  T ImagB,
                                                  int l);

    // A = y.A * x.A
    __host__ __device__ static void getNewA(const BLA &x,
                                            const BLA &y,
                                            T &RealValue,
                                            T &ImagValue);

    // B = y.A * x.B + y.B
    __host__ __device__ static void getNewB(const BLA &x,
                                            const BLA &y,
                                            T& RealValue,
                                            T& ImagValue);

    __host__ __device__ int getL() const;
    __host__ __device__ T getR2() const;
};

template<class T>
class ScaledBLA {
private:
    BLA<float> bla;

    float as, bs, r2s;

public:

    ScaledBLA() = default;
    ScaledBLA(const BLA<double>&);
    ScaledBLA& operator=(const ScaledBLA<T>& other);

    __host__ __device__ ScaledBLA(
        T r2,
        T RealA,
        T ImagA,
        T RealB,
        T ImagB,
        int l);

    __host__ __device__ void getValue(
        T& RealDeltaSubN,
        T& ImagDeltaSubN,
        T RealDeltaSub0,
        T ImagDeltaSub0,
        T &s);

    __host__ __device__ int getL() const;
    __host__ __device__ void getR2(T &outR2, T &outR2s) const;
};
