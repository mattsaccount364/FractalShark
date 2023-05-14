#pragma once

#ifndef __CUDACC__ 
#define __host__
#define __device__
#endif

class Complex {
public:
    double Real;
    double Imag;

    __host__ __device__ Complex() : Real(), Imag() {}
    __host__ __device__ Complex(double r, double i) : Real(r), Imag(i) {}
};

class BLA {
private:
    double r2;
    double Ax;
    double Ay;
    double Bx;
    double By;
    int l;

public:

    BLA() = default;

    __host__ __device__ BLA(double r2, Complex A, Complex B, int l);
    __host__ __device__ Complex getValue(Complex DeltaSubN, Complex DeltaSub0);

    __host__ __device__ double hypotA();
    __host__ __device__ double hypotB();
    __host__ __device__ static BLA getGenericStep(double r2, Complex A, Complex B, int l);

    // A = y.A * x.A
    __host__ __device__ static Complex getNewA(BLA x, BLA y);

    // B = y.A * x.B + y.B
    __host__ __device__ static Complex getNewB(BLA x, BLA y);
    __host__ __device__ int getL() const;
    __host__ __device__ double getR2() const;
};
