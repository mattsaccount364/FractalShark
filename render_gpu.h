#pragma once
//
// Created by dany on 22/05/19.
//

#ifndef GPGPU_RENDER_GPU_HPP
#define GPGPU_RENDER_GPU_HPP

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

class GPURenderer {
public:
    GPURenderer() {};
    ~GPURenderer();

    void ResetRatioMemory();
    void SetRatioMemory(uint8_t* ratioMemory, size_t MaxFractalSize);

    void render_gpu2(
        uint32_t algorithm,
        uint32_t* buffer,
        size_t MaxFractalSize,
        size_t width,
        size_t height,
        MattCoords cx,
        MattCoords cy,
        MattCoords dx,
        MattCoords dy,
        int n_iterations,
        int antialiasing,
        int iteration_precision);

private:
    uint8_t* ratioMemory_cu = nullptr;
};


#endif //GPGPU_RENDER_GPU_HPP