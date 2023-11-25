#pragma once

#include "dblflt.h"

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
    Palette() :
        local_pal(nullptr),
        local_palIters(0),
        palette_aux_depth(0),
        cached_hostPalR(nullptr),
        cached_hostPalG(nullptr),
        cached_hostPalB(nullptr) {
    }

    Palette(
        Color16* local_pal,
        uint32_t local_palIters,
        uint32_t palette_aux_depth,
        const uint16_t* cached_hostPalR,
        const uint16_t* cached_hostPalG,
        const uint16_t* cached_hostPalB) :
        local_pal(local_pal),
        local_palIters(local_palIters),
        palette_aux_depth(palette_aux_depth),
        cached_hostPalR(cached_hostPalR),
        cached_hostPalG(cached_hostPalG),
        cached_hostPalB(cached_hostPalB) {
    }

    Color16* local_pal;
    uint32_t local_palIters;
    uint32_t palette_aux_depth;

    const uint16_t* cached_hostPalR;
    const uint16_t* cached_hostPalG;
    const uint16_t* cached_hostPalB;
};

// These should match the UI menu for sanity's sake
enum class RenderAlgorithm {
    // CPU algorithms
    CpuHigh,
    Cpu64,
    CpuHDR32,
    CpuHDR64,

    Cpu64PerturbedBLA,
    Cpu32PerturbedBLAHDR,
    Cpu64PerturbedBLAHDR,

    Cpu32PerturbedBLAV2HDR,
    Cpu64PerturbedBLAV2HDR,

    // GPU - low zoom depth:
    Gpu1x32,
    Gpu2x32,
    Gpu4x32,
    Gpu1x64,
    Gpu2x64,
    Gpu4x64,
    GpuHDRx32,

    // GPU
    Gpu1x32Perturbed,
    Gpu1x32PerturbedPeriodic,
    Gpu1x32PerturbedScaled,
    Gpu2x32Perturbed,
    Gpu2x32PerturbedScaled,
    Gpu1x64Perturbed,
    GpuHDRx32Perturbed,
    GpuHDRx32PerturbedScaled,

    Gpu1x32PerturbedScaledBLA,
    Gpu1x64PerturbedBLA,
    GpuHDRx32PerturbedBLA,
    GpuHDRx64PerturbedBLA,

    GpuHDRx32PerturbedLAv2,
    GpuHDRx32PerturbedLAv2PO,
    GpuHDRx32PerturbedLAv2LAO,
    GpuHDRx2x32PerturbedLAv2,
    GpuHDRx2x32PerturbedLAv2PO,
    GpuHDRx2x32PerturbedLAv2LAO,
    GpuHDRx64PerturbedLAv2,
    GpuHDRx64PerturbedLAv2PO,
    GpuHDRx64PerturbedLAv2LAO,

    MAX
};

// A list of all the algorithms that are supported by the GPU in string form.
// The algorithms are listed above in the same order as they are listed here.
static const char* RenderAlgorithmStr[(size_t)RenderAlgorithm::MAX] =
{
    "CpuHigh",
    "CpuHDR32",
    "CpuHDR64",
    "Cpu64",
    "Cpu64PerturbedBLA",
    "Cpu32PerturbedBLAHDR",
    "Cpu32PerturbedBLAV2HDR",
    "Cpu64PerturbedBLAHDR",
    "Cpu64PerturbedBLAV2HDR",

    "Gpu1x64",
    "Gpu1x64Perturbed",
    "Gpu1x64PerturbedBLA",
    "GpuHDRx64PerturbedBLA",

    "GpuHDRx64PerturbedLAv2",
    "GpuHDRx64PerturbedLAv2PO",
    "GpuHDRx64PerturbedLAv2LAO",

    "Gpu2x64",
    "Gpu4x64",
    "Gpu1x32",
    "GpuHDRx32",
    "Gpu1x32Perturbed",
    "Gpu1x32PerturbedPeriodic",
    "GpuHDRx32PerturbedBLA",

    "GpuHDRx32PerturbedLAv2",
    "GpuHDRx32PerturbedLAv2PO",
    "GpuHDRx32PerturbedLAv2LAO",

    "GpuHDRx2x32PerturbedLAv2",
    "GpuHDRx2x32PerturbedLAv2PO",
    "GpuHDRx2x32PerturbedLAv2LAO",

    "GpuHDRx32PerturbedScaled",
    "GpuHDRx32Perturbed",
    "Gpu1x32PerturbedScaled",
    "Gpu1x32PerturbedScaledBLA",
    "Gpu2x32",
    "Gpu2x32Perturbed",
    "Gpu2x32PerturbedScaled",
    "Gpu4x32",
};

struct Empty {
};

struct BadField {
    uint32_t bad;
    uint32_t padding;
};

#pragma pack(push, 8)
template<typename Type, CalcBad Bad = CalcBad::Disable>
struct /*alignas(8)*/ MattReferenceSingleIter :
    public std::conditional_t<Bad == CalcBad::Enable, BadField, Empty> {
    MattReferenceSingleIter()
        : x{ Type(0.0f) },
        y{ Type(0.0f) } {
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
    MattReferenceSingleIter& operator=(const MattReferenceSingleIter& other) = default;

    // Example of how to pull the SubType out for HdrFloat, or keep the primitive float/double
    using SubType = typename SubTypeChooser<
        std::is_fundamental<Type>::value || std::is_same<Type, MattDblflt>::value,
        Type>::type;

    static constexpr bool TypeCond =
        std::is_same<Type, HDRFloat<float, HDROrder::Left, int32_t>>::value ||
        std::is_same<Type, HDRFloat<double, HDROrder::Left, int32_t>>::value ||
        std::is_same<Type, HDRFloat<CudaDblflt<MattDblflt>, HDROrder::Left, int32_t>>::value ||
        std::is_same<Type, HDRFloat<CudaDblflt<dblflt>, HDROrder::Left, int32_t>>::value;
    std::conditional<TypeCond, Type, Type>::type x;
    std::conditional<TypeCond, HDRFloat<SubType, HDROrder::Right, int32_t>, Type>::type y;
};
#pragma pack(pop)

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

template<typename IterType, class T, CalcBad Bad = CalcBad::Disable>
struct MattPerturbResults {
    MattReferenceSingleIter<T, Bad>* iters;
    IterType size;
    IterType PeriodMaybeZero;

    MattPerturbResults(IterType in_size,
        MattReferenceSingleIter<T, Bad>* in_orb,
        IterType PeriodMaybeZero) :
        iters(in_orb),
        size(in_size),
        PeriodMaybeZero(PeriodMaybeZero) {

        //char(*__kaboom1)[sizeof(MattReferenceSingleIter<float>)] = 1;
        //char(*__kaboom2)[sizeof(MattReferenceSingleIter<double>)] = 1;
        //char(*__kaboom3)[sizeof(MattReferenceSingleIter<MattDblflt>)] = 1;

        if constexpr (Bad == CalcBad::Enable) {
            static_assert(sizeof(MattReferenceSingleIter<float, Bad>) == 16, "Float");
            static_assert(sizeof(MattReferenceSingleIter<double, Bad>) == 24, "Double");
            static_assert(sizeof(MattReferenceSingleIter<MattDblflt, Bad>) == 24, "MattDblflt");
        }
        else {
            static_assert(sizeof(MattReferenceSingleIter<float, Bad>) == 8, "Float");
            static_assert(sizeof(MattReferenceSingleIter<double, Bad>) == 16, "Double");
            static_assert(sizeof(MattReferenceSingleIter<MattDblflt, Bad>) == 16, "MattDblflt");
        }

        //static_assert(sizeof(MattReferenceSingleIter<HDRFloat<MattDblflt>>) == 12 * 4, "MattDblflt");
        static_assert(sizeof(MattDblflt) == 8, "MattDblflt type");


        // TODO use a template, remove "bad" completely when it's not used.
        // TODO - better though, remove this class and copy from original results
        // to MattPerturbSingleResults directly
    }

    ~MattPerturbResults() {
    }
};