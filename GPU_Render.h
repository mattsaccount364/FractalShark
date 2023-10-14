#pragma once
//
// Created by dany on 22/05/19.
//

#ifndef GPGPU_RENDER_GPU_HPP
#define GPGPU_RENDER_GPU_HPP

#include "BLA.h"
#include "BLAS.h"
#include "LAstep.h"

template<class SubType>
class LAReference;

enum class LAv2Mode {
    Full,
    PO,
    LAO,
};

// TODO reorder these in some sane way that matches the menu etc
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
    Cpu64PerturbedBLAV2HDR,

    // GPU:
    Gpu1x64,
    Gpu1x64Perturbed,
    Gpu1x64PerturbedBLA,
    GpuHDRx64PerturbedBLA,

    GpuHDRx64PerturbedLAv2,
    GpuHDRx64PerturbedLAv2PO,
    GpuHDRx64PerturbedLAv2LAO,

    Gpu2x64,
    Gpu4x64,
    Gpu1x32,
    Gpu1x32Perturbed,
    Gpu1x32PerturbedPeriodic,
    GpuHDRx32PerturbedBLA,
    
    GpuHDRx32PerturbedLAv2,
    GpuHDRx32PerturbedLAv2PO,
    GpuHDRx32PerturbedLAv2LAO,

    GpuHDRx32PerturbedScaled,
    GpuHDRx32Perturbed,
    Gpu1x32PerturbedScaled,
    Gpu1x32PerturbedScaledBLA,
    Gpu2x32,
    Gpu2x32Perturbed,
    Gpu2x32PerturbedScaled,
    Gpu4x32,
};

#pragma pack(push, 8)
template<typename Type>
struct alignas(8) MattReferenceSingleIter {
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
    float x; // MSB
    float y;
    float z;
    float w; // LSB
};

struct MattDbldbl {
    double head;
    double tail;
};

struct MattQDbldbl {
    double x; // MSB
    double y;
    double z;
    double w; // LSB
};

template<class T>
struct MattCoords {
    T val;

    //float floatOnly;
    //double doubleOnly;
    //MattDblflt flt;
    //MattDbldbl dbl;
    //MattQDbldbl qdbl;
    //MattQFltflt qflt;
    //HDRFloat<float> hdrflt;
    //HDRFloat<double> hdrdbl;
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
        static_assert(sizeof(MattReferenceSingleIter<double>) == 24, "Double");
        static_assert(sizeof(MattReferenceSingleIter<float2>) == 24, "float2");
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

    ~MattPerturbResults() {
        delete[] iters;
    }
};

class GPURenderer {
public:
    GPURenderer();
    ~GPURenderer();

    void ResetMemory();

    template<class T>
    uint32_t Render(
        RenderAlgorithm algorithm,
        uint32_t* buffer,
        T cx,
        T cy,
        T dx,
        T dy,
        uint32_t n_iterations,
        int iteration_precision);

    template<class T>
    uint32_t RenderPerturb(
        RenderAlgorithm algorithm,
        uint32_t* buffer,
        MattPerturbResults<T>* results,
        T cx,
        T cy,
        T dx,
        T dy,
        T centerX,
        T centerY,
        uint32_t n_iterations,
        int iteration_precision);

    uint32_t RenderPerturbBLA(
        RenderAlgorithm algorithm,
        uint32_t* buffer,
        MattPerturbResults<double>* results,
        BLAS<double> *blas,
        double cx,
        double cy,
        double dx,
        double dy,
        double centerX,
        double centerY,
        uint32_t n_iterations,
        int iteration_precision);

    template<class T>
    uint32_t RenderPerturbBLA(
        RenderAlgorithm algorithm,
        uint32_t* buffer,
        MattPerturbResults<T>* double_perturb,
        MattPerturbResults<float>* float_perturb,
        BLAS<T>* blas,
        T cx,
        T cy,
        T dx,
        T dy,
        T centerX,
        T centerY,
        uint32_t n_iterations,
        int iteration_precision);

    uint32_t RenderPerturbBLA(
        RenderAlgorithm algorithm,
        uint32_t* buffer,
        MattPerturbResults<float2>* results,
        BLAS<float2>* blas,
        float2 cx,
        float2 cy,
        float2 dx,
        float2 dy,
        float2 centerX,
        float2 centerY,
        uint32_t n_iterations,
        int iteration_precision);

    template<class T>
    uint32_t RenderPerturbBLA(
        RenderAlgorithm algorithm,
        uint32_t* buffer,
        MattPerturbResults<T>* results,
        BLAS<T>* blas,
        T cx,
        T cy,
        T dx,
        T dy,
        T centerX,
        T centerY,
        uint32_t n_iterations,
        int iteration_precision);

    template<class T, class SubType, LAv2Mode Mode>
    uint32_t RenderPerturbLAv2(
        RenderAlgorithm algorithm,
        uint32_t* buffer,
        MattPerturbResults<T>* float_perturb,
        const LAReference<SubType> &LaReference,
        T cx,
        T cy,
        T dx,
        T dy,
        T centerX,
        T centerY,
        uint32_t n_iterations);

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