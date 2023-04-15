#pragma once
//
// Created by dany on 22/05/19.
//

#ifndef GPGPU_RENDER_GPU_HPP
#define GPGPU_RENDER_GPU_HPP

// Render algorithm: 'h' = high res CPU, 'l' = low res CPU
// 'f' = 1x64-bit double
// 'd' = 2x64-bit double (128-bit)
// 'F' = 1x32-bit float
// 'D' = 2x32-bit float
// 'B' = blend
enum class RenderAlgorithm {
    CpuHigh,
    Cpu64,
    Cpu64PerturbedGlitchy,
    Cpu64PerturbedBLA,
    Gpu1x64,
    Gpu1x64PerturbedGlitchy,
    Gpu1x64PerturbedBLA,
    Gpu2x64,
    Gpu4x64,
    Gpu1x32,
    Gpu1x32PerturbedBLA,
    Gpu2x32,
    Gpu2x32PerturbedGlitchy,
    Gpu2x32PerturbedBLA,
    Gpu4x32,
    Blend
};

struct MattDblflt {
    float head;
    float tail;
};

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

struct MattCoordsArray {
    float *floatOnly;
    double *doubleOnly;
    MattDblflt *flt;
    MattDbldbl *dbl;
    MattQFltflt *qflt;
    MattQDbldbl* qdbl;
    size_t size;

    MattCoordsArray(size_t size) {
        floatOnly = new float[size];
        doubleOnly = new double[size];
        flt = new MattDblflt[size];
        dbl = new MattDbldbl[size];
        qflt = new MattQFltflt[size];
        qdbl = new MattQDbldbl[size];
    }

    ~MattCoordsArray() {
        delete[]floatOnly;
        delete[]doubleOnly;
        delete[]flt;
        delete[]dbl;
        delete[]qflt;
        delete[]qdbl;
    }
};

struct MattPerturbResults {
    MattCoordsArray x;
    MattCoordsArray x2;
    MattCoordsArray y;
    MattCoordsArray y2;
    MattCoordsArray tolerancy;
    size_t size;

    MattPerturbResults(size_t size) :
        x(size),
        x2(size),
        y(size),
        y2(size),
        tolerancy(size) {

    }
};

class GPURenderer {
public:
    GPURenderer();
    ~GPURenderer();

    void ResetRatioMemory();
    void SetRatioMemory(uint8_t* ratioMemory, size_t MaxFractalSize);

    void Render(
        RenderAlgorithm algorithm,
        uint32_t* buffer,
        MattCoords cx,
        MattCoords cy,
        MattCoords dx,
        MattCoords dy,
        int n_iterations,
        int iteration_precision);

    void RenderPerturbGlitchy(
        RenderAlgorithm algorithm,
        uint32_t* buffer,
        MattPerturbResults *results,
        MattCoords cx,
        MattCoords cy,
        MattCoords dx,
        MattCoords dy,
        MattCoords centerX,
        MattCoords centerY,
        int n_iterations,
        int iteration_precision);

    void RenderPerturbBLA(
        RenderAlgorithm algorithm,
        uint32_t* buffer,
        MattPerturbResults* results,
        MattCoords cx,
        MattCoords cy,
        MattCoords dx,
        MattCoords dy,
        MattCoords centerX,
        MattCoords centerY,
        int n_iterations,
        int iteration_precision);

    void InitializeMemory(
        size_t w, // width
        size_t h, // height
        uint32_t aa, // antialiasing
        size_t MaxFractSize);

    void ClearMemory();

private:
    void ClearLocals();
    void ExtractIters(uint32_t* buffer);

    uint8_t* ratioMemory_cu = nullptr;
    int* iter_matrix_cu;

    uint32_t width;
    uint32_t height;
    uint32_t antialiasing;
    uint32_t local_width;
    uint32_t local_height;
    uint32_t w_block;
    uint32_t h_block;
    size_t MaxFractalSize;
    size_t N_cu;
};


#endif //GPGPU_RENDER_GPU_HPP