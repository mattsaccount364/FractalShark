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

struct MattDbldbl {
    double head;
    double tail;
};

struct MattCoords {
    float floatOnly;
    double doubleOnly;
    MattDblflt flt;
    MattDbldbl dbl;
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