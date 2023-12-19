#pragma once

#include "dblflt.h"

template<typename IterType, class T, class SubType>
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

#ifndef __CUDACC__
using cudaStream_t = void*;
#endif

struct ReductionResults {
    ReductionResults() :
        Max{},
        Min{},
        Sum{} {
    }

    uint64_t Min;
    uint64_t Max;
    uint64_t Sum;
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
    Gpu1x32PerturbedScaled,
    Gpu2x32PerturbedScaled,
    GpuHDRx32PerturbedScaled,

    Gpu1x32PerturbedScaledBLA,
    Gpu1x64PerturbedBLA,
    GpuHDRx32PerturbedBLA,
    GpuHDRx64PerturbedBLA,

    Gpu1x32PerturbedLAv2,
    Gpu1x32PerturbedLAv2PO,
    Gpu1x32PerturbedLAv2LAO,
    Gpu2x32PerturbedLAv2,
    Gpu2x32PerturbedLAv2PO,
    Gpu2x32PerturbedLAv2LAO,
    Gpu1x64PerturbedLAv2,
    Gpu1x64PerturbedLAv2PO,
    Gpu1x64PerturbedLAv2LAO,
    GpuHDRx32PerturbedLAv2,
    GpuHDRx32PerturbedLAv2PO,
    GpuHDRx32PerturbedLAv2LAO,
    GpuHDRx2x32PerturbedLAv2,
    GpuHDRx2x32PerturbedLAv2PO,
    GpuHDRx2x32PerturbedLAv2LAO,
    GpuHDRx64PerturbedLAv2,
    GpuHDRx64PerturbedLAv2PO,
    GpuHDRx64PerturbedLAv2LAO,

    AUTO,
    MAX
};

// A list of all the algorithms that are supported by the GPU in string form.
// The algorithms are listed above in the same order as they are listed here.
static const char* RenderAlgorithmStr[(size_t)RenderAlgorithm::MAX + 1] =
{
    "CpuHigh",
    "Cpu64",
    "CpuHDR32",
    "CpuHDR64",

    "Cpu64PerturbedBLA",
    "Cpu32PerturbedBLAHDR",
    "Cpu64PerturbedBLAHDR",

    "Cpu32PerturbedBLAV2HDR",
    "Cpu64PerturbedBLAV2HDR",

    "Gpu1x32",
    "Gpu2x32",
    "Gpu4x32",
    "Gpu1x64",
    "Gpu2x64",
    "Gpu4x64",
    "GpuHDRx32",

    "Gpu1x32PerturbedScaled",
    "Gpu2x32PerturbedScaled",
    "GpuHDRx32PerturbedScaled",

    "Gpu1x32PerturbedScaledBLA",
    "Gpu1x64PerturbedBLA",
    "GpuHDRx32PerturbedBLA",
    "GpuHDRx64PerturbedBLA",

    "Gpu1x32PerturbedLAv2",
    "Gpu1x32PerturbedLAv2PO",
    "Gpu1x32PerturbedLAv2LAO",
    "Gpu2x32PerturbedLAv2",
    "Gpu2x32PerturbedLAv2PO",
    "Gpu2x32PerturbedLAv2LAO",
    "Gpu1x64PerturbedLAv2",
    "Gpu1x64PerturbedLAv2PO",
    "Gpu1x64PerturbedLAv2LAO",
    "GpuHDRx32PerturbedLAv2",
    "GpuHDRx32PerturbedLAv2PO",
    "GpuHDRx32PerturbedLAv2LAO",
    "GpuHDRx2x32PerturbedLAv2",
    "GpuHDRx2x32PerturbedLAv2PO",
    "GpuHDRx2x32PerturbedLAv2LAO",
    "GpuHDRx64PerturbedLAv2",
    "GpuHDRx64PerturbedLAv2PO",
    "GpuHDRx64PerturbedLAv2LAO",

    "AutoSelect",
    "MAX"
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
    const MattReferenceSingleIter<T, Bad>* iters;
    IterType size;
    IterType PeriodMaybeZero;

    MattPerturbResults(IterType in_size,
        const MattReferenceSingleIter<T, Bad>* in_orb,
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

template<typename IterType, typename Type, CalcBad Bad = CalcBad::Disable>
struct MattPerturbSingleResults;


template<typename IterType, class T, class SubType>
class GPU_LAReference;

struct PerturbResultsCollection {
private:
    struct InternalResults {
        void Init() {
            m_Results32FloatDisable = nullptr;
            m_Results32FloatEnable = nullptr;
            m_Results32DoubleDisable = nullptr;
            m_Results32DoubleEnable = nullptr;
            m_Results32CudaDblfltDisable = nullptr;
            m_Results32CudaDblfltEnable = nullptr;
            m_Results32HdrFloatDisable = nullptr;
            m_Results32HdrFloatEnable = nullptr;
            m_Results32HdrDoubleDisable = nullptr;
            m_Results32HdrDoubleEnable = nullptr;
            m_Results32HdrCudaMattDblfltDisable = nullptr;
            m_Results32HdrCudaMattDblfltEnable = nullptr;

            m_Results64FloatDisable = nullptr;
            m_Results64FloatEnable = nullptr;
            m_Results64DoubleDisable = nullptr;
            m_Results64DoubleEnable = nullptr;
            m_Results64CudaDblfltDisable = nullptr;
            m_Results64CudaDblfltEnable = nullptr;
            m_Results64HdrFloatDisable = nullptr;
            m_Results64HdrFloatEnable = nullptr;
            m_Results64HdrDoubleDisable = nullptr;
            m_Results64HdrDoubleEnable = nullptr;
            m_Results64HdrCudaMattDblfltDisable = nullptr;
            m_Results64HdrCudaMattDblfltEnable = nullptr;

            m_GenerationNumber = 0;

            m_LaReference32Float = nullptr;
            m_LaReference32Double = nullptr;
            m_LaReference32CudaDblflt = nullptr;
            m_LaReference32HdrFloat = nullptr;
            m_LaReference32HdrDouble = nullptr;
            m_LaReference32HdrCudaMattDblflt = nullptr;

            m_LaReference64Float = nullptr;
            m_LaReference64Double = nullptr;
            m_LaReference64CudaDblflt = nullptr;
            m_LaReference64HdrFloat = nullptr;
            m_LaReference64HdrDouble = nullptr;
            m_LaReference64HdrCudaMattDblflt = nullptr;

            m_LaGenerationNumber = 0;
        }

        MattPerturbSingleResults<uint32_t, float, CalcBad::Disable>* m_Results32FloatDisable;
        MattPerturbSingleResults<uint32_t, float, CalcBad::Enable>* m_Results32FloatEnable;
        MattPerturbSingleResults<uint32_t, double, CalcBad::Disable>* m_Results32DoubleDisable;
        MattPerturbSingleResults<uint32_t, double, CalcBad::Enable>* m_Results32DoubleEnable;
        MattPerturbSingleResults<uint32_t, CudaDblflt<dblflt>, CalcBad::Disable>* m_Results32CudaDblfltDisable;
        MattPerturbSingleResults<uint32_t, CudaDblflt<dblflt>, CalcBad::Enable>* m_Results32CudaDblfltEnable;
        MattPerturbSingleResults<uint32_t, HDRFloat<float>, CalcBad::Disable>* m_Results32HdrFloatDisable;
        MattPerturbSingleResults<uint32_t, HDRFloat<float>, CalcBad::Enable>* m_Results32HdrFloatEnable;
        MattPerturbSingleResults<uint32_t, HDRFloat<double>, CalcBad::Disable>* m_Results32HdrDoubleDisable;
        MattPerturbSingleResults<uint32_t, HDRFloat<double>, CalcBad::Enable>* m_Results32HdrDoubleEnable;
        MattPerturbSingleResults<uint32_t, HDRFloat<CudaDblflt<MattDblflt>>, CalcBad::Disable>* m_Results32HdrCudaMattDblfltDisable;
        MattPerturbSingleResults<uint32_t, HDRFloat<CudaDblflt<MattDblflt>>, CalcBad::Enable>* m_Results32HdrCudaMattDblfltEnable;

        MattPerturbSingleResults<uint64_t, float, CalcBad::Disable>* m_Results64FloatDisable;
        MattPerturbSingleResults<uint64_t, float, CalcBad::Enable>* m_Results64FloatEnable;
        MattPerturbSingleResults<uint64_t, double, CalcBad::Disable>* m_Results64DoubleDisable;
        MattPerturbSingleResults<uint64_t, double, CalcBad::Enable>* m_Results64DoubleEnable;
        MattPerturbSingleResults<uint64_t, CudaDblflt<dblflt>, CalcBad::Disable>* m_Results64CudaDblfltDisable;
        MattPerturbSingleResults<uint64_t, CudaDblflt<dblflt>, CalcBad::Enable>* m_Results64CudaDblfltEnable;
        MattPerturbSingleResults<uint64_t, HDRFloat<float>, CalcBad::Disable>* m_Results64HdrFloatDisable;
        MattPerturbSingleResults<uint64_t, HDRFloat<float>, CalcBad::Enable>* m_Results64HdrFloatEnable;
        MattPerturbSingleResults<uint64_t, HDRFloat<double>, CalcBad::Disable>* m_Results64HdrDoubleDisable;
        MattPerturbSingleResults<uint64_t, HDRFloat<double>, CalcBad::Enable>* m_Results64HdrDoubleEnable;
        MattPerturbSingleResults<uint64_t, HDRFloat<CudaDblflt<MattDblflt>>, CalcBad::Disable>* m_Results64HdrCudaMattDblfltDisable;
        MattPerturbSingleResults<uint64_t, HDRFloat<CudaDblflt<MattDblflt>>, CalcBad::Enable>* m_Results64HdrCudaMattDblfltEnable;

        size_t m_GenerationNumber;

        GPU_LAReference<uint32_t, float, float> *m_LaReference32Float;
        GPU_LAReference<uint32_t, double, double>* m_LaReference32Double;
        GPU_LAReference<uint32_t, CudaDblflt<dblflt>, CudaDblflt<dblflt>>* m_LaReference32CudaDblflt;
        GPU_LAReference<uint32_t, HDRFloat<float>, float>* m_LaReference32HdrFloat;
        GPU_LAReference<uint32_t, HDRFloat<double>, double>* m_LaReference32HdrDouble;
        GPU_LAReference<uint32_t, HDRFloat<CudaDblflt<MattDblflt>>, CudaDblflt<MattDblflt>>* m_LaReference32HdrCudaMattDblflt;

        GPU_LAReference<uint64_t, float, float>* m_LaReference64Float;
        GPU_LAReference<uint64_t, double, double>* m_LaReference64Double;
        GPU_LAReference<uint64_t, CudaDblflt<dblflt>, CudaDblflt<dblflt>>* m_LaReference64CudaDblflt;
        GPU_LAReference<uint64_t, HDRFloat<float>, float>* m_LaReference64HdrFloat;
        GPU_LAReference<uint64_t, HDRFloat<double>, double>* m_LaReference64HdrDouble;
        GPU_LAReference<uint64_t, HDRFloat<CudaDblflt<MattDblflt>>, CudaDblflt<MattDblflt>>* m_LaReference64HdrCudaMattDblflt;

        size_t m_LaGenerationNumber;
    };

public:
    PerturbResultsCollection();
    ~PerturbResultsCollection();

private:
    template<typename Type, CalcBad Bad = CalcBad::Disable>
    void SetPtr32(
        size_t GenerationNumber,
        InternalResults& Results,
        MattPerturbSingleResults<uint32_t, Type, Bad>* ptr);

    template<typename Type, CalcBad Bad = CalcBad::Disable>
    void SetPtr64(
        size_t GenerationNumber,
        InternalResults& Results,
        MattPerturbSingleResults<uint64_t, Type, Bad>* ptr);

    template<typename Type, typename SubType>
    void SetLaReferenceInternal32(
        size_t LaGenerationNumber,
        InternalResults& Results,
        GPU_LAReference<uint32_t, Type, SubType>* LaReference);

    template<typename Type, typename SubType>
    void SetLaReferenceInternal64(
        size_t LaGenerationNumber,
        InternalResults& Results,
        GPU_LAReference<uint64_t, Type, SubType>* LaReference);

    template<typename Type, CalcBad Bad = CalcBad::Disable>
    MattPerturbSingleResults<uint32_t, Type, Bad>* GetPtrInternal32(InternalResults& Results);

    template<typename Type, CalcBad Bad = CalcBad::Disable>
    MattPerturbSingleResults<uint64_t, Type, Bad>* GetPtrInternal64(InternalResults& Results);

    template<typename IterType, typename Type, CalcBad Bad = CalcBad::Disable>
    MattPerturbSingleResults<IterType, Type, Bad>* GetPtrInternal(InternalResults& Results);

    template<typename Type, typename SubType>
    GPU_LAReference<uint32_t, Type, SubType>*
        GetLaReferenceInternal32(InternalResults& Results);

    template<typename Type, typename SubType>
    GPU_LAReference<uint64_t, Type, SubType>*
        GetLaReferenceInternal64(InternalResults& Results);

    template<typename IterType, typename Type, typename SubType>
    GPU_LAReference<IterType, Type, SubType>*
        GetLaReferenceInternal(InternalResults& Results);

    template<typename IterType, typename Type, typename SubType>
    void SetLaReferenceInternal(
        size_t LaGenerationNumber,
        InternalResults& Results,
        GPU_LAReference<IterType, Type, SubType>* LaReference);

    void DeleteAllInternal(InternalResults& Results);

public:
    template<typename IterType, typename Type, CalcBad Bad = CalcBad::Disable>
    void SetPtr1(size_t GenerationNumber, MattPerturbSingleResults<IterType, Type, Bad>* ptr);

    template<typename IterType, typename Type, CalcBad Bad = CalcBad::Disable>
    void SetPtr2(size_t GenerationNumber, MattPerturbSingleResults<IterType, Type, Bad>* ptr);

    template<typename IterType, typename Type, typename SubType>
    void SetLaReference1(
        size_t LaGenerationNumber,
        GPU_LAReference<IterType, Type, SubType>* LaReference);

    template<typename IterType, typename Type, CalcBad Bad = CalcBad::Disable>
    MattPerturbSingleResults<IterType, Type, Bad>* GetPtr1();

    template<typename IterType, typename Type, CalcBad Bad = CalcBad::Disable>
    MattPerturbSingleResults<IterType, Type, Bad>* GetPtr2();

    size_t GetHostGenerationNumber1() const;
    size_t GetHostGenerationNumber2() const;
    size_t GetHostLaGenerationNumber1() const;
    size_t GetHostLaGenerationNumber2() const;

    template<typename IterType, typename Type, typename SubType>
    GPU_LAReference<IterType, Type, SubType>* GetLaReference1();

    template<typename IterType, typename Type, typename SubType>
    GPU_LAReference<IterType, Type, SubType>* GetLaReference2();

    void DeleteAll();

private:
    InternalResults m_Results1;
    InternalResults m_Results2;
};