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

class GPURenderer {
public:
    GPURenderer();
    ~GPURenderer();

    template<typename IterType, class T>
    uint32_t Render(
        RenderAlgorithm algorithm,
        IterType* iter_buffer,
        Color16 *color_buffer,
        T cx,
        T cy,
        T dx,
        T dy,
        IterType n_iterations,
        int iteration_precision);

    template<typename IterType, class T>
    uint32_t RenderPerturb(
        RenderAlgorithm algorithm,
        IterType* iter_buffer,
        Color16* color_buffer,
        MattPerturbResults<IterType, T>* results,
        T cx,
        T cy,
        T dx,
        T dy,
        T centerX,
        T centerY,
        IterType n_iterations,
        int iteration_precision);

    //template<typename IterType>
    //uint32_t RenderPerturbBLA(
    //    RenderAlgorithm algorithm,
    //    IterType* iter_buffer,
    //    Color16* color_buffer,
    //    MattPerturbResults<double>* results,
    //    BLAS<IterType, double> *blas,
    //    double cx,
    //    double cy,
    //    double dx,
    //    double dy,
    //    double centerX,
    //    double centerY,
    //    IterType n_iterations,
    //    int iteration_precision);

    template<typename IterType, class T>
    uint32_t RenderPerturbBLAScaled(
        RenderAlgorithm algorithm,
        IterType* iter_buffer,
        Color16* color_buffer,
        MattPerturbResults<IterType, T, CalcBad::Enable>* double_perturb,
        MattPerturbResults<IterType, float, CalcBad::Enable>* float_perturb,
        BLAS<IterType, T, CalcBad::Enable>* blas,
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
        IterType* iter_buffer,
        Color16* color_buffer,
        MattPerturbResults<IterType, MattDblflt>* results,
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
        IterType* iter_buffer,
        Color16* color_buffer,
        MattPerturbResults<IterType, T>* results,
        BLAS<IterType, T>* blas,
        T cx,
        T cy,
        T dx,
        T dy,
        T centerX,
        T centerY,
        IterType n_iterations,
        int iteration_precision);

    template<typename IterType, class T, class SubType, LAv2Mode Mode>
    uint32_t RenderPerturbLAv2(
        RenderAlgorithm algorithm,
        IterType* iter_buffer,
        Color16* color_buffer,
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

    template<typename IterType, class T1, class SubType, CalcBad Bad, class T2>
    uint32_t InitializePerturb(
        const void *OrigResults1,
        MattPerturbResults<IterType, T1, Bad>* Perturb1,
        const void* OrigResults2,
        MattPerturbResults<IterType, T2, Bad>* Perturb2,
        const LAReference<IterType, SubType>* LaReferenceHost);

    template<typename IterType>
    void ClearMemory();

    template<typename IterType>
    uint32_t OnlyAA(
        Color16* color_buffer,
        IterType n_iterations);

    static const char* ConvertErrorToString(uint32_t err);

    // Match in Fractal.cpp
    static const int32_t NB_THREADS_W = 16;  // W=16, H=8 previously seemed OK
    static const int32_t NB_THREADS_H = 8;

    static const int32_t NB_THREADS_W_AA = 16;  // W=16, H=8 previously seemed OK
    static const int32_t NB_THREADS_H_AA = 8;

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
    uint32_t RunAntialiasing(IterType n_iterations);

    template<typename IterType>
    uint32_t ExtractItersAndColors(IterType* iter_buffer, Color16* color_buffer);

    void* OutputIterMatrix;
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

    PerturbResultsCollection m_PerturbResults;
};


#endif //GPGPU_RENDER_GPU_HPP