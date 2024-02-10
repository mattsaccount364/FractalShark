#pragma once
//
// Created by dany on 22/05/19.
//

#ifndef GPGPU_RENDER_GPU_HPP
#define GPGPU_RENDER_GPU_HPP

#include "BLA.h"
#include "BLAS.h"
#include "LAstep.h"

#include "GPU_Types.h"

// This is the main class that does the rendering on the GPU
class GPURenderer {
public:
    GPURenderer();
    ~GPURenderer();

    template<typename IterType, class T>
    uint32_t Render(
        RenderAlgorithm algorithm,
        T cx,
        T cy,
        T dx,
        T dy,
        IterType n_iterations,
        int iteration_precision);

    template<typename IterType, class T>
    uint32_t RenderPerturbBLAScaled(
        RenderAlgorithm algorithm,
        const GPUPerturbResults<IterType, T, PerturbExtras::Bad>* double_perturb,
        const GPUPerturbResults<IterType, float, PerturbExtras::Bad>* float_perturb,
        T cx,
        T cy,
        T dx,
        T dy,
        T centerX,
        T centerY,
        IterType n_iterations,
        int iteration_precision);

    template<typename IterType>
    uint32_t RenderPerturbBLA(
        RenderAlgorithm algorithm,
        const GPUPerturbResults<IterType, MattDblflt, PerturbExtras::Disable>* results,
        BLAS<IterType, MattDblflt>* blas,
        MattDblflt cx,
        MattDblflt cy,
        MattDblflt dx,
        MattDblflt dy,
        MattDblflt centerX,
        MattDblflt centerY,
        IterType n_iterations,
        int iteration_precision);

    template<typename IterType, class T>
    uint32_t RenderPerturbBLA(
        RenderAlgorithm algorithm,
        const GPUPerturbResults<IterType, T, PerturbExtras::Disable>* results,
        BLAS<IterType, T>* blas,
        T cx,
        T cy,
        T dx,
        T dy,
        T centerX,
        T centerY,
        IterType n_iterations,
        int iteration_precision);

    template<typename IterType, class T, class SubType, LAv2Mode Mode, PerturbExtras PExtras>
    uint32_t RenderPerturbLAv2(
        RenderAlgorithm algorithm,
        T cx,
        T cy,
        T dx,
        T dy,
        T centerX,
        T centerY,
        IterType n_iterations);

    // Side effect is this initializes CUDA the first time it's run
    template<typename IterType>
    uint32_t InitializeMemory(
        uint32_t w, // original width * antialiasing
        uint32_t h, // original height * antialiasing
        uint32_t antialiasing, // w and h are ech scaled up by this amt
        const uint16_t *palR,
        const uint16_t *palG,
        const uint16_t *palB,
        uint32_t palIters,
        uint32_t paletteAuxDepth);

    template<typename IterType, class T1, class SubType, PerturbExtras PExtras, class T2>
    uint32_t InitializePerturb(
        size_t GenerationNumber1,
        const GPUPerturbResults<IterType, T1, PExtras>* Perturb1,
        size_t GenerationNumber2,
        const GPUPerturbResults<IterType, T2, PExtras>* Perturb2,
        const LAReference<IterType, T1, SubType, PExtras>* LaReferenceHost);

    template<typename IterType>
    void ClearMemory();

    static const char* ConvertErrorToString(uint32_t err);

    // Match in Fractal.cpp
    static const int32_t NB_THREADS_W = 16;  // W=16, H=8 previously seemed OK
    static const int32_t NB_THREADS_H = 8;

    static const int32_t NB_THREADS_W_AA = 16;  // W=16, H=8 previously seemed OK
    static const int32_t NB_THREADS_H_AA = 8;

public:
    template<typename IterType>
    uint32_t RenderCurrent(
        IterType n_iterations,
        IterType* iter_buffer,
        Color16* color_buffer,
        ReductionResults *reduction_results);

    uint32_t SyncStream(bool altStream);

private:
    bool MemoryInitialized() const;
    void ResetPalettesOnly();

    enum class ResetLocals {
        Yes,
        No
    };
    enum class ResetPalettes {
        Yes,
        No
    };

    enum class ResetPerturb {
        Yes,
        No
    };

    void ResetMemory(
        ResetLocals locals,
        ResetPalettes palettes,
        ResetPerturb perturb);
    void ClearLocals();

    template<typename IterType>
    uint32_t RunAntialiasing(IterType n_iterations, cudaStream_t *stream);

    template<typename IterType, bool Async>
    uint32_t ExtractItersAndColors(
        IterType* iter_buffer,
        Color16* color_buffer,
        ReductionResults *reduction_results);

    void* OutputIterMatrix;
    ReductionResults *OutputReductionResults;
    AntialiasedColors OutputColorMatrix;

    Palette Pals;

    uint32_t m_Width;
    uint32_t m_Height;
    uint32_t local_color_width;
    uint32_t local_color_height;
    uint32_t m_Antialiasing;
    uint32_t m_IterTypeSize;
    uint32_t w_block;
    uint32_t h_block;
    uint32_t w_color_block;
    uint32_t h_color_block;
    size_t N_cu;
    size_t N_color_cu;

    bool m_Stream1Initialized;
    cudaStream_t m_Stream1;
    int m_StreamPriorityLow;
    int m_StreamPriorityHigh;

    PerturbResultsCollection m_PerturbResults;
};


#endif //GPGPU_RENDER_GPU_HPP