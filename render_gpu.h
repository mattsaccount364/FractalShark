#pragma once
//
// Created by dany on 22/05/19.
//

#ifndef GPGPU_RENDER_GPU_HPP
#define GPGPU_RENDER_GPU_HPP

#include "BLA.h"
#include "BLAS.h"

// Render algorithm: 'h' = high res CPU, 'l' = low res CPU
// 'f' = 1x64-bit double
// 'd' = 2x64-bit double (128-bit)
// 'F' = 1x32-bit float
// 'D' = 2x32-bit float
// 'B' = blend
enum class RenderAlgorithm {
    CpuHigh,
    Cpu64,
    Cpu64PerturbedBLA,
    Gpu1x64,
    Gpu1x64Perturbed,
    Gpu1x64PerturbedBLA,
    Gpu2x64,
    Gpu4x64,
    Gpu1x32,
    Gpu1x32Perturbed,
    Gpu1x32PerturbedScaled,
    Gpu2x32,
    Gpu2x32Perturbed,
    Gpu2x32PerturbedScaled,
    Gpu4x32,
};

#pragma pack(push, 8)
template<typename Type>
struct MattReferenceSingleIter {
    Type x;
    Type x2;
    Type y;
    Type y2;
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
};

struct MattPerturbResults {
    MattReferenceSingleIter<float> *float_iters;
    MattReferenceSingleIter<double> *double_iters;
    MattReferenceSingleIter<float2> *dblflt_iters;
    uint32_t* bad_counts;
    size_t bad_counts_size;
    size_t size;

    MattPerturbResults(size_t in_size,
                       size_t in_bad_counts_size,
                       double *in_x,
                       double *in_x2,
                       double *in_y,
                       double *in_y2,
                       uint8_t *in_bad,
                       uint32_t *in_bad_counts) :
        float_iters(new MattReferenceSingleIter<float>[in_size]),
        double_iters(new MattReferenceSingleIter<double>[in_size]),
        dblflt_iters(new MattReferenceSingleIter<float2>[in_size]),
        bad_counts(new uint32_t[in_bad_counts_size]),
        bad_counts_size(in_bad_counts_size),
        size(in_size) {

        //char(*__kaboom1)[sizeof(MattReferenceSingleIter<float>)] = 1;
        //char(*__kaboom2)[sizeof(MattReferenceSingleIter<double>)] = 1;
        //char(*__kaboom3)[sizeof(MattReferenceSingleIter<float2>)] = 1;

        static_assert(sizeof(MattReferenceSingleIter<float>) == 24, "Float");
        static_assert(sizeof(MattReferenceSingleIter<double>) == 40, "Double");
        static_assert(sizeof(MattReferenceSingleIter<float2>) == 40, "float2");
        static_assert(sizeof(float2) == 8, "float2 type");

        for (size_t i = 0; i < size; i++) {
            float_iters[i].x = (float)in_x[i];
            float_iters[i].x2 = (float)in_x2[i];
            float_iters[i].y = (float)in_y[i];
            float_iters[i].y2 = (float)in_y2[i];
            float_iters[i].bad = in_bad[i];

            double_iters[i].x = in_x[i];
            double_iters[i].x2 = in_x2[i];
            double_iters[i].y = in_y[i];
            double_iters[i].y2 = in_y2[i];
            double_iters[i].bad = in_bad[i];

            dblflt_iters[i].x.y = (float)in_x[i];
            dblflt_iters[i].x.x = (float)(in_x[i] - (double)((float)dblflt_iters[i].x.y));
            dblflt_iters[i].x2.y = (float)in_x2[i];
            dblflt_iters[i].x2.x = (float)(in_x2[i] - (double)((float)dblflt_iters[i].x2.y));
            dblflt_iters[i].y.y = (float)in_y[i];
            dblflt_iters[i].y.x = (float)(in_y[i] - (double)((float)dblflt_iters[i].y.y));
            dblflt_iters[i].y2.y = (float)in_y2[i];
            dblflt_iters[i].y2.x = (float)(in_y2[i] - (double)((float)dblflt_iters[i].y2.y));
            dblflt_iters[i].bad = in_bad[i];
        }

        memcpy(bad_counts, in_bad_counts, bad_counts_size);
    }

    ~MattPerturbResults() {
        delete[] float_iters;
        delete[] double_iters;
        delete[] dblflt_iters;
        delete[] bad_counts;
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
        MattPerturbResults* results,
        BLAS *blas,
        MattCoords cx,
        MattCoords cy,
        MattCoords dx,
        MattCoords dy,
        MattCoords centerX,
        MattCoords centerY,
        uint32_t n_iterations,
        int iteration_precision);

    uint32_t InitializeMemory(
        size_t w, // width
        size_t h, // height
        uint32_t aa, // antialiasing
        size_t MaxFractSize);

    void ClearMemory();

private:
    void ClearLocals();
    uint32_t ExtractIters(uint32_t* buffer);

    uint32_t* iter_matrix_cu;

    uint32_t width;
    uint32_t height;
    uint32_t antialiasing;
    uint32_t local_width;
    uint32_t local_height;
    uint32_t w_block;
    uint32_t h_block;
    size_t array_width;
    size_t N_cu;
};


#endif //GPGPU_RENDER_GPU_HPP