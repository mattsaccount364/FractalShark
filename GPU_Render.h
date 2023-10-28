#pragma once
//
// Created by dany on 22/05/19.
//

#ifndef GPGPU_RENDER_GPU_HPP
#define GPGPU_RENDER_GPU_HPP

#include "BLA.h"
#include "BLAS.h"
#include "LAstep.h"

template<typename IterType, class SubType>
class LAReference;

enum class LAv2Mode {
    Full,
    PO,
    LAO,
};

struct Color32 {
    uint32_t r, g, b;
};

struct Color16 {
    uint16_t r, g, b, a;
};

struct AntialiasedColors {
#ifdef __CUDACC__ 
    Color16* __restrict__ aa_colors;
#else
    Color16* aa_colors;
#endif
};

struct Palette {
    Color16* local_pal;
    IterTypeFull local_palIters;

    const uint16_t* cached_hostPalR;
    const uint16_t* cached_hostPalG;
    const uint16_t* cached_hostPalB;
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

struct Empty {
};

struct BadField {
    uint32_t bad;
    uint32_t padding;
};

#pragma pack(push, 8)
template<typename Type, CalcBad Bad = CalcBad::Disable>
struct alignas(8) MattReferenceSingleIter : public std::conditional_t<Bad == CalcBad::Enable, BadField, Empty> {
    MattReferenceSingleIter()
        : x{ Type(0) },
        y{ Type(0) } {
    }

    MattReferenceSingleIter(Type x, Type y)
        : x{ x },
        y{ y } {
    }

    MattReferenceSingleIter(Type x, Type y, bool bad)
        : x{ x },
        y{ y } {
        if constexpr (Bad == CalcBad::Enable) {
            this->bad = bad;
        }
    }

    MattReferenceSingleIter(const MattReferenceSingleIter& other) = default;
    MattReferenceSingleIter &operator=(const MattReferenceSingleIter& other) = default;

    Type x;
    Type y;
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

template<class T, CalcBad Bad = CalcBad::Disable>
struct MattPerturbResults {
    MattReferenceSingleIter<T, Bad> *iters;
    IterType size;
    IterType PeriodMaybeZero;

    MattPerturbResults(IterType in_size,
                       MattReferenceSingleIter<T, Bad> *in_orb,
                       IterType PeriodMaybeZero) :
        iters(in_orb),
        size(in_size),
        PeriodMaybeZero(PeriodMaybeZero) {

        //char(*__kaboom1)[sizeof(MattReferenceSingleIter<float>)] = 1;
        //char(*__kaboom2)[sizeof(MattReferenceSingleIter<double>)] = 1;
        //char(*__kaboom3)[sizeof(MattReferenceSingleIter<float2>)] = 1;

        if constexpr (Bad == CalcBad::Enable) {
            static_assert(sizeof(MattReferenceSingleIter<float, Bad>) == 16, "Float");
            static_assert(sizeof(MattReferenceSingleIter<double, Bad>) == 24, "Double");
            static_assert(sizeof(MattReferenceSingleIter<float2, Bad>) == 24, "float2");
        }
        else {
            static_assert(sizeof(MattReferenceSingleIter<float, Bad>) == 8, "Float");
            static_assert(sizeof(MattReferenceSingleIter<double, Bad>) == 16, "Double");
            static_assert(sizeof(MattReferenceSingleIter<float2, Bad>) == 16, "float2");
        }

        //static_assert(sizeof(MattReferenceSingleIter<HDRFloat<float>>) == 12 * 4, "float2");
        static_assert(sizeof(float2) == 8, "float2 type");
        

        // TODO use a template, remove "bad" completely when it's not used.
        // TODO - better though, remove this class and copy from original results
        // to MattPerturbSingleResults directly
    }

    ~MattPerturbResults() {
    }
};

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
        MattPerturbResults<T>* results,
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
    //    BLAS<double> *blas,
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
        MattPerturbResults<T, CalcBad::Enable>* double_perturb,
        MattPerturbResults<float, CalcBad::Enable>* float_perturb,
        BLAS<T, CalcBad::Enable>* blas,
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
        MattPerturbResults<float2>* results,
        BLAS<float2>* blas,
        float2 cx,
        float2 cy,
        float2 dx,
        float2 dy,
        float2 centerX,
        float2 centerY,
        IterType n_iterations,
        int iteration_precision);

    template<typename IterType, class T>
    uint32_t RenderPerturbBLA(
        RenderAlgorithm algorithm,
        IterType* iter_buffer,
        Color16* color_buffer,
        MattPerturbResults<T>* results,
        BLAS<T>* blas,
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
        MattPerturbResults<T>* float_perturb,
        const LAReference<IterType, SubType> &LaReference,
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
        IterType palIters);

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

    void ResetMemory(ResetLocals locals, ResetPalettes palettes);
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
    uint32_t w_block;
    uint32_t h_block;
    uint32_t w_color_block;
    uint32_t h_color_block;
    size_t N_cu;
    size_t N_color_cu;
};


#endif //GPGPU_RENDER_GPU_HPP