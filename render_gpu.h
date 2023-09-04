#pragma once
//
// Created by dany on 22/05/19.
//

#ifndef GPGPU_RENDER_GPU_HPP
#define GPGPU_RENDER_GPU_HPP

#include "BLA.h"
#include "BLAS.h"
#include "LAstep.h"
class LAReference;

enum class RenderAlgorithm {
    // CPU algorithms
    CpuHigh,
    CpuHDR32,
    CpuHDR64,
    Cpu64,
    Cpu64PerturbedBLA,
    Cpu32PerturbedBLAHDR,
    Cpu32PerturbedBLAV2HDR,
    Cpu64PerturbedBLAHDR,

    // GPU:
    Gpu1x64,
    Gpu1x64Perturbed,
    Gpu1x64PerturbedBLA,
    GpuHDRx64PerturbedBLA,
    Gpu2x64,
    Gpu4x64,
    Gpu1x32,
    Gpu1x32Perturbed,
    Gpu1x32PerturbedPeriodic,
    GpuHDRx32PerturbedBLA,
    GpuHDRx32PerturbedLAv2,
    GpuHDRx32PerturbedScaled,
    Gpu1x32PerturbedScaled,
    Gpu1x32PerturbedScaledBLA,
    Gpu2x32,
    Gpu2x32Perturbed,
    Gpu2x32PerturbedScaled,
    Gpu4x32,
};

#pragma pack(push, 16)
template<typename Type>
struct alignas(16) MattReferenceSingleIter {
    Type x;
    Type y;
    uint32_t bad;
    uint32_t padding;
};
#pragma pack(pop)

#pragma pack(push, 8)
struct MattDblflt {
    float x; // head
    float y; // tail
};
#pragma pack(pop)


#ifndef __CUDACC__ 
using float2 = MattDblflt;
#endif

struct MattQFltflt {
    float v1; // MSB
    float v2;
    float v3;
    float v4; // LSB
};

struct MattDbldbl {
    double head;
    double tail;
};

struct MattQDbldbl {
    double v1; // MSB
    double v2;
    double v3;
    double v4; // LSB
};

struct MattCoords {
    float floatOnly;
    double doubleOnly;
    MattDblflt flt;
    MattDbldbl dbl;
    MattQDbldbl qdbl;
    MattQFltflt qflt;
    HDRFloat<float> hdrflt;
    HDRFloat<double> hdrdbl;
};

template<class T>
struct MattPerturbResults {
    MattReferenceSingleIter<T> *iters;
    size_t size;
    size_t PeriodMaybeZero;

    MattPerturbResults(size_t in_size,
                       T *in_x,
                       T *in_y,
                       uint8_t *in_bad,
                       size_t in_bad_size,
                       size_t PeriodMaybeZero) :
        iters(new MattReferenceSingleIter<T>[in_size]),
        size(in_size),
        PeriodMaybeZero(PeriodMaybeZero) {

        //char(*__kaboom1)[sizeof(MattReferenceSingleIter<float>)] = 1;
        //char(*__kaboom2)[sizeof(MattReferenceSingleIter<double>)] = 1;
        //char(*__kaboom3)[sizeof(MattReferenceSingleIter<float2>)] = 1;

        static_assert(sizeof(MattReferenceSingleIter<float>) == 16, "Float");
        static_assert(sizeof(MattReferenceSingleIter<double>) == 32, "Double");
        static_assert(sizeof(MattReferenceSingleIter<float2>) == 32, "float2");
        //static_assert(sizeof(MattReferenceSingleIter<HDRFloat<float>>) == 12 * 4, "float2");
        static_assert(sizeof(float2) == 8, "float2 type");

        if (in_bad_size == in_size) {
            for (size_t i = 0; i < size; i++) {
                iters[i].x = in_x[i];
                iters[i].y = in_y[i];
                iters[i].bad = in_bad[i];
            }
        }
        else {
            for (size_t i = 0; i < size; i++) {
                iters[i].x = in_x[i];
                iters[i].y = in_y[i];
                iters[i].bad = 0;
            }
        }
    }

    T* GetComplex(size_t index) {
        return HDRFloatComplex<T>(iters[index].x, iters[index].y);
    }

    ~MattPerturbResults() {
        delete[] iters;
    }
};

class GPURenderer {
public:
    GPURenderer();
    ~GPURenderer();

    void ResetMemory();

    uint32_t Render(
        RenderAlgorithm algorithm,
        uint32_t* buffer,
        MattCoords cx,
        MattCoords cy,
        MattCoords dx,
        MattCoords dy,
        uint32_t n_iterations,
        int iteration_precision);

    uint32_t RenderPerturbBLA(
        RenderAlgorithm algorithm,
        uint32_t* buffer,
        MattPerturbResults<float>* results,
        BLAS<float>* blas,
        MattCoords cx,
        MattCoords cy,
        MattCoords dx,
        MattCoords dy,
        MattCoords centerX,
        MattCoords centerY,
        uint32_t n_iterations,
        int iteration_precision);

    uint32_t RenderPerturbBLA(
        RenderAlgorithm algorithm,
        uint32_t* buffer,
        MattPerturbResults<double>* results,
        BLAS<double> *blas,
        MattCoords cx,
        MattCoords cy,
        MattCoords dx,
        MattCoords dy,
        MattCoords centerX,
        MattCoords centerY,
        uint32_t n_iterations,
        int iteration_precision);

    template<class T>
    uint32_t RenderPerturbBLA(
        RenderAlgorithm algorithm,
        uint32_t* buffer,
        MattPerturbResults<T>* double_perturb,
        MattPerturbResults<float>* float_perturb,
        BLAS<T>* blas,
        MattCoords cx,
        MattCoords cy,
        MattCoords dx,
        MattCoords dy,
        MattCoords centerX,
        MattCoords centerY,
        uint32_t n_iterations,
        int iteration_precision);

    uint32_t RenderPerturbBLA(
        RenderAlgorithm algorithm,
        uint32_t* buffer,
        MattPerturbResults<float2>* results,
        BLAS<float2>* blas,
        MattCoords cx,
        MattCoords cy,
        MattCoords dx,
        MattCoords dy,
        MattCoords centerX,
        MattCoords centerY,
        uint32_t n_iterations,
        int iteration_precision);

    uint32_t RenderPerturbBLA(
        RenderAlgorithm algorithm,
        uint32_t* buffer,
        MattPerturbResults<HDRFloat<float>>* results,
        BLAS<HDRFloat<float>>* blas,
        MattCoords cx,
        MattCoords cy,
        MattCoords dx,
        MattCoords dy,
        MattCoords centerX,
        MattCoords centerY,
        uint32_t n_iterations,
        int iteration_precision);

    uint32_t RenderPerturbLAv2(
        RenderAlgorithm algorithm,
        uint32_t* buffer,
        MattPerturbResults<HDRFloat<float>>* float_perturb,
        const LAReference &LaReference,
        MattCoords cx,
        MattCoords cy,
        MattCoords dx,
        MattCoords dy,
        MattCoords centerX,
        MattCoords centerY,
        uint32_t n_iterations);

    uint32_t RenderPerturbBLA(
        RenderAlgorithm algorithm,
        uint32_t* buffer,
        MattPerturbResults<HDRFloat<double>>* results,
        BLAS<HDRFloat<double>>* blas,
        MattCoords cx,
        MattCoords cy,
        MattCoords dx,
        MattCoords dy,
        MattCoords centerX,
        MattCoords centerY,
        uint32_t n_iterations,
        int iteration_precision);

    // Side effect is this initializes CUDA the first time it's run
    uint32_t InitializeMemory(
        size_t w, // width
        size_t h); // height

    void ClearMemory();

    static const char* ConvertErrorToString(uint32_t err);

    // Match in Fractal.cpp
    static const int32_t NB_THREADS_W = 16;  // W=16, H=8 previously seemed OK
    static const int32_t NB_THREADS_H = 8;

private:
    void ClearLocals();
    uint32_t ExtractIters(uint32_t* buffer);

    uint32_t* iter_matrix_cu;

    uint32_t width;
    uint32_t height;
    uint32_t local_width;
    uint32_t local_height;
    uint32_t w_block;
    uint32_t h_block;
    size_t array_width;
    size_t N_cu;
};


#endif //GPGPU_RENDER_GPU_HPP