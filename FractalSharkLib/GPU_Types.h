#pragma once

#include "dblflt.h"
#include "PerturbationResultsHelpers.h"
#include "RenderAlgorithm.h"

template<typename IterType, class T, class SubType, PerturbExtras PExtras>
class LAReference;

struct Color32 {
    uint32_t r, g, b;
};

struct Color16 {
    uint16_t r, g, b, a;
};

struct AntialiasedColors {
#ifdef __CUDACC__ 
    Color16 *__restrict__ aa_colors;
#else
    Color16 *aa_colors;
#endif
};

#ifndef __CUDACC__
using cudaStream_t = void *;
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
        Color16 *local_pal,
        uint32_t local_palIters,
        uint32_t palette_aux_depth,
        const uint16_t *cached_hostPalR,
        const uint16_t *cached_hostPalG,
        const uint16_t *cached_hostPalB) :
        local_pal(local_pal),
        local_palIters(local_palIters),
        palette_aux_depth(palette_aux_depth),
        cached_hostPalR(cached_hostPalR),
        cached_hostPalG(cached_hostPalG),
        cached_hostPalB(cached_hostPalB) {
    }

    Color16 *local_pal;
    uint32_t local_palIters;
    uint32_t palette_aux_depth;

    const uint16_t *cached_hostPalR;
    const uint16_t *cached_hostPalG;
    const uint16_t *cached_hostPalB;
};

struct Empty {
};

struct BadField {
    BadField() : bad{ false }, padding{} {
    }

    BadField(bool bad) : bad{ bad }, padding{} {
    }

    [[msvc::no_unique_address]] uint32_t bad;
    [[msvc::no_unique_address]] uint32_t padding;
};

// TODO if we template this on IterType, update m_CompressionHelper
// (GPU + CPU versions) to avoid static_cast all over
struct CompressionIndexField {
    CompressionIndexField() : u{ } {
    }

    CompressionIndexField(IterTypeFull initIndex) {
        u.f.CompressionIndex = initIndex;
        u.f.Rebase = 0;
    }

    CompressionIndexField(IterTypeFull initIndex, IterTypeFull rebase) {
        u.f.CompressionIndex = initIndex;
        u.f.Rebase = rebase;
    }

    union U {
        struct F {
            IterTypeFull CompressionIndex : 63;
            IterTypeFull Rebase : 1;
        } f;

        IterTypeFull Raw;
    } u;
};

template<PerturbExtras PExtras>
class PerturbExtrasHack {
public:
    static constexpr PerturbExtras Val = PExtras;
};

#pragma pack(push, 8)
template<typename Type, PerturbExtras PExtras>
class /*alignas(8)*/ GPUReferenceIter :
    public std::conditional_t<
    PExtras == PerturbExtras::Bad,
    BadField,
    std::conditional_t<PExtras == PerturbExtras::SimpleCompression || PExtras == PerturbExtras::MaxCompression,
    CompressionIndexField,
    Empty>> {

public:

    using BaseClass = std::conditional_t<
        PExtras == PerturbExtras::Bad,
        BadField,
        std::conditional_t<PExtras == PerturbExtras::SimpleCompression || PExtras == PerturbExtras::MaxCompression,
        CompressionIndexField,
        Empty>>;

    static constexpr IterTypeFull BadCompressionIndex = 0xFFFF'FFFF'FFFF'FFFFull;

    struct Enabler {};

    GPUReferenceIter() : BaseClass{}, x{}, y{} {
    }

    template<
        typename U = PerturbExtrasHack<PExtras>,
        std::enable_if_t<U::Val == PerturbExtras::Disable, int> = 0>
    GPUReferenceIter(Type init_x, Type init_y) : x{ init_x }, y{ init_y } {
    }

    template<
        typename U = PerturbExtrasHack<PExtras>,
        std::enable_if_t<U::Val == PerturbExtras::SimpleCompression || U::Val == PerturbExtras::MaxCompression, int> = 0>
    GPUReferenceIter(Type init_x, Type init_y, IterTypeFull init_compression_index, IterTypeFull rebase)
        : CompressionIndexField(init_compression_index, rebase),
        x{ init_x },
        y{ init_y } {
    }

    template<
        typename U = PerturbExtrasHack<PExtras>,
        std::enable_if_t<U::Val == PerturbExtras::SimpleCompression || U::Val == PerturbExtras::MaxCompression, int> = 0>
    GPUReferenceIter(Type init_x, Type init_y, IterTypeFull init_compression_index)
        : CompressionIndexField(init_compression_index),
        x{ init_x },
        y{ init_y } {
    }

    template<
        typename U = PerturbExtrasHack<PExtras>,
        std::enable_if_t<U::Val == PerturbExtras::Bad, int> = 0>
    GPUReferenceIter(Type init_x, Type init_y, bool init_bad)
        : BadField(init_bad),
        x{ init_x },
        y{ init_y } {
    }

    GPUReferenceIter(const GPUReferenceIter &other) = default;
    GPUReferenceIter &operator=(const GPUReferenceIter &other) = default;

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

template<typename IterType, class T, PerturbExtras PExtras>
class GPUPerturbResults {
public:
    GPUPerturbResults(
        IterType compressed_size,
        IterType uncompressed_size,
        T OrbitXLow,
        T OrbitYLow,
        const GPUReferenceIter<T, PExtras> *in_orb,
        IterType PeriodMaybeZero) :
        FullOrbit(in_orb),
        OrbitSize(compressed_size),
        UncompressedSize(uncompressed_size),
        PeriodMaybeZero(PeriodMaybeZero),
        OrbitXLow(OrbitXLow),
        OrbitYLow(OrbitYLow) {

        //char(*__kaboom1)[sizeof(GPUReferenceIter<float>)] = 1;
        //char(*__kaboom2)[sizeof(GPUReferenceIter<double>)] = 1;
        //char(*__kaboom3)[sizeof(GPUReferenceIter<MattDblflt>)] = 1;

        if constexpr (PExtras == PerturbExtras::Disable) {
            static_assert(sizeof(GPUReferenceIter<float, PExtras>) == 8, "Float");
            static_assert(sizeof(GPUReferenceIter<double, PExtras>) == 16, "Double");
            static_assert(sizeof(GPUReferenceIter<MattDblflt, PExtras>) == 16, "MattDblflt");
        } else if constexpr (PExtras == PerturbExtras::Bad) {
            static_assert(sizeof(GPUReferenceIter<float, PExtras>) == 16, "Float");
            static_assert(sizeof(GPUReferenceIter<double, PExtras>) == 24, "Double");
            static_assert(sizeof(GPUReferenceIter<MattDblflt, PExtras>) == 24, "MattDblflt");
        } else {
            static_assert(sizeof(GPUReferenceIter<float, PExtras>) == 16, "Float");
            static_assert(sizeof(GPUReferenceIter<double, PExtras>) == 24, "Double");
            static_assert(sizeof(GPUReferenceIter<MattDblflt, PExtras>) == 24, "MattDblflt");
        }

        //static_assert(sizeof(GPUReferenceIter<HDRFloat<MattDblflt>>) == 12 * 4, "MattDblflt");
        static_assert(sizeof(MattDblflt) == 8, "MattDblflt type");


        // TODO use a template, remove "bad" completely when it's not used.
        // TODO - better though, remove this class and copy from original results
        // to GPUPerturbSingleResults directly
    }

    ~GPUPerturbResults() {
    }

    const GPUReferenceIter<T, PExtras> *GetFullOrbit() const {
        return FullOrbit;
    }

    IterType GetCompressedSize() const {
        return OrbitSize;
    }

    IterType GetUncompressedSize() const {
        return UncompressedSize;
    }

    IterType GetPeriodMaybeZero() const {
        return PeriodMaybeZero;
    }

    T GetOrbitXLow() const {
        return OrbitXLow;
    }

    T GetOrbitYLow() const {
        return OrbitYLow;
    }

private:
    // May be compressed
    const GPUReferenceIter<T, PExtras> *FullOrbit;

    // May be either compressed or uncompressed count
    IterType OrbitSize;

    IterType UncompressedSize;

    // Actual period
    IterType PeriodMaybeZero;

    T OrbitXLow;
    T OrbitYLow;
};

template<typename IterType, typename Type, PerturbExtras PExtras>
class GPUPerturbSingleResults;


template<typename IterType, class T, class SubType>
class GPU_LAReference;

struct PerturbResultsCollection {
private:
    struct InternalResults {
        void Init() {
            m_Results32FloatDisable = nullptr;
            m_Results32FloatEnable = nullptr;
            m_Results32FloatRC = nullptr;
            m_Results32DoubleDisable = nullptr;
            m_Results32DoubleEnable = nullptr;
            m_Results32DoubleRC = nullptr;
            m_Results32CudaDblfltDisable = nullptr;
            m_Results32CudaDblfltEnable = nullptr;
            m_Results32CudaDblfltRC = nullptr;
            m_Results32HdrFloatDisable = nullptr;
            m_Results32HdrFloatEnable = nullptr;
            m_Results32HdrFloatRC = nullptr;
            m_Results32HdrDoubleDisable = nullptr;
            m_Results32HdrDoubleEnable = nullptr;
            m_Results32HdrDoubleRC = nullptr;
            m_Results32HdrCudaMattDblfltDisable = nullptr;
            m_Results32HdrCudaMattDblfltEnable = nullptr;
            m_Results32HdrCudaMattDblfltRC = nullptr;

            m_Results64FloatDisable = nullptr;
            m_Results64FloatEnable = nullptr;
            m_Results64FloatRC = nullptr;
            m_Results64DoubleDisable = nullptr;
            m_Results64DoubleEnable = nullptr;
            m_Results64DoubleRC = nullptr;
            m_Results64CudaDblfltDisable = nullptr;
            m_Results64CudaDblfltEnable = nullptr;
            m_Results64CudaDblfltRC = nullptr;
            m_Results64HdrFloatDisable = nullptr;
            m_Results64HdrFloatEnable = nullptr;
            m_Results64HdrFloatRC = nullptr;
            m_Results64HdrDoubleDisable = nullptr;
            m_Results64HdrDoubleEnable = nullptr;
            m_Results64HdrDoubleRC = nullptr;
            m_Results64HdrCudaMattDblfltDisable = nullptr;
            m_Results64HdrCudaMattDblfltEnable = nullptr;
            m_Results64HdrCudaMattDblfltRC = nullptr;

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

        // THIS IS AWESOME.   HOW STUPID IS THIS?  I can't believe I did this.

        GPUPerturbSingleResults<uint32_t, float, PerturbExtras::Disable> *m_Results32FloatDisable;
        GPUPerturbSingleResults<uint32_t, float, PerturbExtras::Bad> *m_Results32FloatEnable;
        GPUPerturbSingleResults<uint32_t, float, PerturbExtras::SimpleCompression> *m_Results32FloatRC;
        GPUPerturbSingleResults<uint32_t, double, PerturbExtras::Disable> *m_Results32DoubleDisable;
        GPUPerturbSingleResults<uint32_t, double, PerturbExtras::Bad> *m_Results32DoubleEnable;
        GPUPerturbSingleResults<uint32_t, double, PerturbExtras::SimpleCompression> *m_Results32DoubleRC;
        GPUPerturbSingleResults<uint32_t, CudaDblflt<dblflt>, PerturbExtras::Disable> *m_Results32CudaDblfltDisable;
        GPUPerturbSingleResults<uint32_t, CudaDblflt<dblflt>, PerturbExtras::Bad> *m_Results32CudaDblfltEnable;
        GPUPerturbSingleResults<uint32_t, CudaDblflt<dblflt>, PerturbExtras::SimpleCompression> *m_Results32CudaDblfltRC;
        GPUPerturbSingleResults<uint32_t, HDRFloat<float>, PerturbExtras::Disable> *m_Results32HdrFloatDisable;
        GPUPerturbSingleResults<uint32_t, HDRFloat<float>, PerturbExtras::Bad> *m_Results32HdrFloatEnable;
        GPUPerturbSingleResults<uint32_t, HDRFloat<float>, PerturbExtras::SimpleCompression> *m_Results32HdrFloatRC;
        GPUPerturbSingleResults<uint32_t, HDRFloat<double>, PerturbExtras::Disable> *m_Results32HdrDoubleDisable;
        GPUPerturbSingleResults<uint32_t, HDRFloat<double>, PerturbExtras::Bad> *m_Results32HdrDoubleEnable;
        GPUPerturbSingleResults<uint32_t, HDRFloat<double>, PerturbExtras::SimpleCompression> *m_Results32HdrDoubleRC;
        GPUPerturbSingleResults<uint32_t, HDRFloat<CudaDblflt<MattDblflt>>, PerturbExtras::Disable> *m_Results32HdrCudaMattDblfltDisable;
        GPUPerturbSingleResults<uint32_t, HDRFloat<CudaDblflt<MattDblflt>>, PerturbExtras::Bad> *m_Results32HdrCudaMattDblfltEnable;
        GPUPerturbSingleResults<uint32_t, HDRFloat<CudaDblflt<MattDblflt>>, PerturbExtras::SimpleCompression> *m_Results32HdrCudaMattDblfltRC;

        GPUPerturbSingleResults<uint64_t, float, PerturbExtras::Disable> *m_Results64FloatDisable;
        GPUPerturbSingleResults<uint64_t, float, PerturbExtras::Bad> *m_Results64FloatEnable;
        GPUPerturbSingleResults<uint64_t, float, PerturbExtras::SimpleCompression> *m_Results64FloatRC;
        GPUPerturbSingleResults<uint64_t, double, PerturbExtras::Disable> *m_Results64DoubleDisable;
        GPUPerturbSingleResults<uint64_t, double, PerturbExtras::Bad> *m_Results64DoubleEnable;
        GPUPerturbSingleResults<uint64_t, double, PerturbExtras::SimpleCompression> *m_Results64DoubleRC;
        GPUPerturbSingleResults<uint64_t, CudaDblflt<dblflt>, PerturbExtras::Disable> *m_Results64CudaDblfltDisable;
        GPUPerturbSingleResults<uint64_t, CudaDblflt<dblflt>, PerturbExtras::Bad> *m_Results64CudaDblfltEnable;
        GPUPerturbSingleResults<uint64_t, CudaDblflt<dblflt>, PerturbExtras::SimpleCompression> *m_Results64CudaDblfltRC;
        GPUPerturbSingleResults<uint64_t, HDRFloat<float>, PerturbExtras::Disable> *m_Results64HdrFloatDisable;
        GPUPerturbSingleResults<uint64_t, HDRFloat<float>, PerturbExtras::Bad> *m_Results64HdrFloatEnable;
        GPUPerturbSingleResults<uint64_t, HDRFloat<float>, PerturbExtras::SimpleCompression> *m_Results64HdrFloatRC;
        GPUPerturbSingleResults<uint64_t, HDRFloat<double>, PerturbExtras::Disable> *m_Results64HdrDoubleDisable;
        GPUPerturbSingleResults<uint64_t, HDRFloat<double>, PerturbExtras::Bad> *m_Results64HdrDoubleEnable;
        GPUPerturbSingleResults<uint64_t, HDRFloat<double>, PerturbExtras::SimpleCompression> *m_Results64HdrDoubleRC;
        GPUPerturbSingleResults<uint64_t, HDRFloat<CudaDblflt<MattDblflt>>, PerturbExtras::Disable> *m_Results64HdrCudaMattDblfltDisable;
        GPUPerturbSingleResults<uint64_t, HDRFloat<CudaDblflt<MattDblflt>>, PerturbExtras::Bad> *m_Results64HdrCudaMattDblfltEnable;
        GPUPerturbSingleResults<uint64_t, HDRFloat<CudaDblflt<MattDblflt>>, PerturbExtras::SimpleCompression> *m_Results64HdrCudaMattDblfltRC;

        size_t m_GenerationNumber;

        GPU_LAReference<uint32_t, float, float> *m_LaReference32Float;
        GPU_LAReference<uint32_t, double, double> *m_LaReference32Double;
        GPU_LAReference<uint32_t, CudaDblflt<dblflt>, CudaDblflt<dblflt>> *m_LaReference32CudaDblflt;
        GPU_LAReference<uint32_t, HDRFloat<float>, float> *m_LaReference32HdrFloat;
        GPU_LAReference<uint32_t, HDRFloat<double>, double> *m_LaReference32HdrDouble;
        GPU_LAReference<uint32_t, HDRFloat<CudaDblflt<MattDblflt>>, CudaDblflt<MattDblflt>> *m_LaReference32HdrCudaMattDblflt;

        GPU_LAReference<uint64_t, float, float> *m_LaReference64Float;
        GPU_LAReference<uint64_t, double, double> *m_LaReference64Double;
        GPU_LAReference<uint64_t, CudaDblflt<dblflt>, CudaDblflt<dblflt>> *m_LaReference64CudaDblflt;
        GPU_LAReference<uint64_t, HDRFloat<float>, float> *m_LaReference64HdrFloat;
        GPU_LAReference<uint64_t, HDRFloat<double>, double> *m_LaReference64HdrDouble;
        GPU_LAReference<uint64_t, HDRFloat<CudaDblflt<MattDblflt>>, CudaDblflt<MattDblflt>> *m_LaReference64HdrCudaMattDblflt;

        size_t m_LaGenerationNumber;
    };

public:
    PerturbResultsCollection();
    ~PerturbResultsCollection();

private:
    template<typename Type, PerturbExtras PExtras>
    void SetPtr32(
        size_t GenerationNumber,
        InternalResults &Results,
        GPUPerturbSingleResults<uint32_t, Type, PExtras> *ptr);

    template<typename Type, PerturbExtras PExtras>
    void SetPtr64(
        size_t GenerationNumber,
        InternalResults &Results,
        GPUPerturbSingleResults<uint64_t, Type, PExtras> *ptr);

    template<typename Type, typename SubType>
    void SetLaReferenceInternal32(
        size_t LaGenerationNumber,
        InternalResults &Results,
        GPU_LAReference<uint32_t, Type, SubType> *LaReference);

    template<typename Type, typename SubType>
    void SetLaReferenceInternal64(
        size_t LaGenerationNumber,
        InternalResults &Results,
        GPU_LAReference<uint64_t, Type, SubType> *LaReference);

    template<typename Type, PerturbExtras PExtras>
    GPUPerturbSingleResults<uint32_t, Type, PExtras> *GetPtrInternal32(InternalResults &Results);

    template<typename Type, PerturbExtras PExtras>
    GPUPerturbSingleResults<uint64_t, Type, PExtras> *GetPtrInternal64(InternalResults &Results);

    template<typename IterType, typename Type, PerturbExtras PExtras>
    GPUPerturbSingleResults<IterType, Type, PExtras> *GetPtrInternal(InternalResults &Results);

    template<typename Type, typename SubType>
    GPU_LAReference<uint32_t, Type, SubType> *
        GetLaReferenceInternal32(InternalResults &Results);

    template<typename Type, typename SubType>
    GPU_LAReference<uint64_t, Type, SubType> *
        GetLaReferenceInternal64(InternalResults &Results);

    template<typename IterType, typename Type, typename SubType>
    GPU_LAReference<IterType, Type, SubType> *
        GetLaReferenceInternal(InternalResults &Results);

    template<typename IterType, typename Type, typename SubType>
    void SetLaReferenceInternal(
        size_t LaGenerationNumber,
        InternalResults &Results,
        GPU_LAReference<IterType, Type, SubType> *LaReference);

    void DeleteAllInternal(InternalResults &Results);

public:
    template<typename IterType, typename Type, PerturbExtras PExtras>
    void SetPtr1(size_t GenerationNumber, GPUPerturbSingleResults<IterType, Type, PExtras> *ptr);

    template<typename IterType, typename Type, PerturbExtras PExtras>
    void SetPtr2(size_t GenerationNumber, GPUPerturbSingleResults<IterType, Type, PExtras> *ptr);

    template<typename IterType, typename Type, typename SubType>
    void SetLaReference1(
        size_t LaGenerationNumber,
        GPU_LAReference<IterType, Type, SubType> *LaReference);

    template<typename IterType, typename Type, PerturbExtras PExtras>
    GPUPerturbSingleResults<IterType, Type, PExtras> *GetPtr1();

    template<typename IterType, typename Type, PerturbExtras PExtras>
    GPUPerturbSingleResults<IterType, Type, PExtras> *GetPtr2();

    size_t GetHostGenerationNumber1() const;
    size_t GetHostGenerationNumber2() const;
    size_t GetHostLaGenerationNumber1() const;
    size_t GetHostLaGenerationNumber2() const;

    template<typename IterType, typename Type, typename SubType>
    GPU_LAReference<IterType, Type, SubType> *GetLaReference1();

    template<typename IterType, typename Type, typename SubType>
    GPU_LAReference<IterType, Type, SubType> *GetLaReference2();

    void DeleteAll();

private:
    InternalResults m_Results1;
    InternalResults m_Results2;
};