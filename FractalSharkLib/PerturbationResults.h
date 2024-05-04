#pragma once

#include <vector>
#include <stdint.h>
#include <fstream>
#include <type_traits>

#include "HighPrecision.h"
#include "HDRFloatComplex.h"
#include "RefOrbitCalc.h"
#include "LAReference.h"

#include "GPU_Render.h"

#include "PerturbationResultsHelpers.h"
#include "Vectors.h"
#include "Utilities.h"

#include "ScopedMpir.h"
#include "PrecisionCalculator.h"

#include <thread>

namespace Introspection {
    template<typename IterType, typename Float, PerturbExtras PExtras>
    struct PerturbTypeParams {
        using IterType_ = IterType;
        using Float_ = Float;
        static constexpr auto PExtras_ = PExtras;
    };

    template<
        template <typename, typename, PerturbExtras> typename PerturbType,
        typename IterType,
        typename Float,
        PerturbExtras PExtras>
    constexpr auto Extract(const PerturbType<IterType, Float, PExtras> &) -> PerturbTypeParams<IterType, Float, PExtras>;

    template<typename PerturbType>
    constexpr auto Extract_PExtras = decltype(Extract(std::declval<PerturbType>()))::PExtras_;

    template<typename PerturbType>
    using Extract_Float = typename decltype(Extract(std::declval<PerturbType>()))::Float_;

    template<typename PerturbType, PerturbExtras PExtrasOther>
    static constexpr bool PerturbTypeHasPExtras() {
        return Extract_PExtras<PerturbType> == PExtrasOther;
    }

    template<typename PerturbType>
    static constexpr bool IsDblFlt() {
        return
            std::is_same<Extract_Float<PerturbType>, CudaDblflt<MattDblflt>>::value ||
            std::is_same<Extract_Float<PerturbType>, HDRFloat<CudaDblflt<MattDblflt>>>::value;
    }

    template<typename T>
    static constexpr bool IsTDblFlt() {
        return
            std::is_same<T, CudaDblflt<MattDblflt>>::value ||
            std::is_same<T, HDRFloat<CudaDblflt<MattDblflt>>>::value;
    }
} // namespace PerturbationResultsIntrospection

std::wstring GetTimeAsString(size_t generation_number);

template<typename IterType, class T, PerturbExtras PExtras>
class RefOrbitCompressor;

template<typename IterType, class T, PerturbExtras PExtras>
class SimpleIntermediateOrbitCompressor;

template<typename IterType, class T, PerturbExtras PExtras>
class MaxIntermediateOrbitCompressor;


template<typename IterType, class T, PerturbExtras PExtras>
class PerturbationResults : public TemplateHelpers<IterType, T, PExtras> {

public:
    template<typename IterType, class T, PerturbExtras PExtras> friend class PerturbationResults;

    using TemplateHelpers = TemplateHelpers<IterType, T, PExtras>;
    using SubType = TemplateHelpers::SubType;

    template<class LocalSubType>
    using HDRFloatComplex = TemplateHelpers::template HDRFloatComplex<LocalSubType>;

    friend class RuntimeDecompressor<IterType, T, PExtras>;
    friend class IntermediateRuntimeDecompressor<IterType, T, PExtras>;
    friend class IntermediateMaxRuntimeDecompressor<IterType, T, PExtras>;
    friend class RefOrbitCompressor<IterType, T, PExtras>;
    friend class SimpleIntermediateOrbitCompressor<IterType, T, PExtras>;
    friend class MaxIntermediateOrbitCompressor<IterType, T, PExtras>;

    static constexpr char Version[] = "0.44";

    static constexpr bool Is2X32 = (
        std::is_same<T, HDRFloat<CudaDblflt<MattDblflt>>>::value ||
        std::is_same<T, CudaDblflt<MattDblflt>>::value);

    // Generates a filename "base" for all the orbit files.
    std::wstring GenBaseFilename(size_t generation_number) const;

    std::wstring GenBaseFilename(
        AddPointOptions add_point_options,
        size_t generation_number) const;

    // Note: This function relies on m_BaseFilename, so order
    // of initialization is important.
    // If add_additional_suffix is true, add a number to the end
    // to ensure the filename is unique.
    std::wstring GenFilename(
        GrowableVectorTypes Type,
        std::wstring optional_suffix = L"",
        bool add_additional_suffix = false) const;

    // Parameters:
    //  add_point_options - whether to save the orbit or not
    //  Generation - the generation number of the orbit
    //  base_filename - the base filename of the orbit, used
    //    when opening existing
    PerturbationResults(
        std::wstring base_filename,
        AddPointOptions add_point_options,
        size_t Generation);

    PerturbationResults(
        AddPointOptions add_point_options,
        size_t Generation);

    ~PerturbationResults();

    PerturbationResults(PerturbationResults &&other) = delete;

    PerturbationResults(const PerturbationResults &other) = delete;
    PerturbationResults &operator=(const PerturbationResults &other) = delete;

    size_t GetGenerationNumber() const;

    void ClearLaReference();

    void SetLaReference(
        std::unique_ptr<LAReference<IterType, T, SubType, PExtras>> laReference);

    LAReference<IterType, T, SubType, PExtras> *GetLaReference() const;

    std::unique_ptr<PerturbationResults<IterType, T, PExtras>>
        CopyPerturbationResults(AddPointOptions add_point_options,
            size_t new_generation_number);

    template<bool IncludeLA, class Other, PerturbExtras PExtrasOther = PerturbExtras::Disable>
    void CopyFullOrbitVector(
        const PerturbationResults<IterType, Other, PExtrasOther> &other);

    template<bool IncludeLA, class Other, PerturbExtras PExtrasOther = PerturbExtras::Disable>
    void CopyPerturbationResults(
        const PerturbationResults<IterType, Other, PExtrasOther> &other);

    template<PerturbExtras OtherBad>
    void CopySettingsWithoutOrbit(const PerturbationResults<IterType, T, OtherBad> &other);

    void WriteMetadata() const;

    void MaybeOpenMetaFileForDelete() const;

    // This function uses CreateFileMapping and MapViewOfFile to map
    // the file into memory.  Then it loads the meta file using ifstream
    // to get the other data.
    //
    // TODO Loading a 2x32 (with or without HDR) does not work with LA enabled.
    // The code below works, but 
    // The LA table generation doesn't currently work with 2x32 and only supports 1x64
    // (which it then converts to 2x32).  This is a bug.
    bool ReadMetadata();

    template<class U = SubType>
    typename HDRFloatComplex<U> GetComplex(
        RuntimeDecompressor<IterType, T, PExtras> &PerThreadCompressionHelper,
        size_t uncompressed_index) const {

        if constexpr (PExtras == PerturbExtras::Disable || PExtras == PerturbExtras::Bad) {
            return {
                m_FullOrbit[uncompressed_index].x,
                m_FullOrbit[uncompressed_index].y };
        } else {
            return PerThreadCompressionHelper.GetCompressedComplex<U>(uncompressed_index);
        }
    }

    void InitReused();

    void InitResults(
        RefOrbitCalc::ReuseMode Reuse,
        const HighPrecision &cx,
        const HighPrecision &cy,
        const HighPrecision &minX,
        const HighPrecision &minY,
        const HighPrecision &maxX,
        const HighPrecision &maxY,
        IterType NumIterations,
        size_t GuessReserveSize);

    uint64_t GetBenchmarkOrbit() const;

    template<RefOrbitCalc::ReuseMode Reuse>
    void CompleteResults(std::unique_ptr<ThreadMemory> allocatorIfAny);

    size_t GetCompressedOrbitSize() const;

    IterType GetCountOrbitEntries() const;

    // PExtras == PerturbExtras::Disable || PExtras == PerturbExtras::Bad
    void AddUncompressedIteration(GPUReferenceIter<T, PExtras> result)
        requires (PExtras == PerturbExtras::Disable || PExtras == PerturbExtras::Bad);

    const GPUReferenceIter<T, PExtras> *GetOrbitData() const;

    // Location of orbit
    const HighPrecision &GetHiX() const;

    // Location of orbit
    const HighPrecision &GetHiY() const;

    const std::string &GetHiXStr() const;

    const std::string &GetHiYStr() const;

    T GetOrbitXLow() const;

    T GetOrbitYLow() const;

    // Radius used for periodicity checking
    T GetMaxRadius() const;

    HighPrecision GetMaxRadiusHigh() const;

    // Used only with scaled kernels
    void SetBad(bool bad) requires (PExtras == PerturbExtras::Bad);

    uint64_t GetAuthoritativePrecisionInBits() const;

    IterType GetPeriodMaybeZero() const;

    void SetPeriodMaybeZero(IterType period);

    int32_t GetCompressionErrorExp() const;

    int32_t GetIntermediateCompressionErrorExp() const;

    AddPointOptions GetRefOrbitOptions() const;

    size_t GetReuseSize() const;

    void AddUncompressedReusedEntry(
        HighPrecision x,
        HighPrecision y,
        IterTypeFull index);

    // Take references to pointers to avoid copying.
    // Set the pointers to point at the specified index.
    void GetCompressedReuseEntries(
        IntermediateRuntimeDecompressor<IterType, T, PExtras> &PerThreadCompressionHelper,
        size_t uncompressed_index,
        const mpf_t *&x,
        const mpf_t *&y) const;

    void GetMaxCompressedReuseEntries(
        IntermediateMaxRuntimeDecompressor<IterType, T, PExtras> &PerThreadCompressionHelper,
        size_t uncompressed_index,
        const mpf_t *&x,
        const mpf_t *&y) const;

    void GetUncompressedReuseEntries(
        size_t uncompressed_index,
        const mpf_t *&x,
        const mpf_t *&y) const;

    IterType GetMaxIterations() const;

    // For reference:
    // Used:
    //   https://code.mathr.co.uk/fractal-bits/tree/HEAD:/mandelbrot-reference-compression
    //   https://fractalforums.org/fractal-mathematics-and-new-theories/28/reference-compression/5142
    // as a reference for the compression algorithm.
    std::unique_ptr<PerturbationResults<IterType, T, PerturbExtras::SimpleCompression>>
        Compress(
            int32_t compression_error_exp_param,
            size_t new_generation_number)
        requires (PExtras == PerturbExtras::Disable && !Introspection::IsTDblFlt<T>());

    std::unique_ptr<PerturbationResults<IterType, T, PerturbExtras::Disable>>
        Decompress(size_t NewGenerationNumber)
        requires (PExtras == PerturbExtras::SimpleCompression && !Introspection::IsTDblFlt<T>());

    std::unique_ptr<PerturbationResults<IterType, T, PerturbExtras::SimpleCompression>>
        CompressMax(
            int32_t compression_error_exp_param,
            size_t new_generation_number)
        requires (PExtras == PerturbExtras::Disable && !Introspection::IsTDblFlt<T>());

    std::unique_ptr<PerturbationResults<IterType, T, PerturbExtras::Disable>>
        DecompressMax(size_t NewGenerationNumber)
        requires (PExtras == PerturbExtras::SimpleCompression && !Introspection::IsTDblFlt<T>());

    void SaveOrbitAsText() const;

    // For information purposes only, not used for anything
    // other than reporting.
    void GetIntermediatePrecision(
        int64_t &deltaPrecision,
        int64_t &extraPrecision) const;

    // For information purposes only, not used for anything
    // other than reporting.
    void SetIntermediateCachedPrecision(
        int64_t deltaPrecision,
        int64_t extraPrecision);

private:
    void CloseMetaFileIfOpen() const;

    void MapExistingFiles();

    HighPrecision m_OrbitX;
    HighPrecision m_OrbitY;
    std::string m_OrbitXStr;
    std::string m_OrbitYStr;
    T m_OrbitXLow;
    T m_OrbitYLow;
    T m_MaxRadius;
    HighPrecision m_MaxRadiusHigh; // Just for convenience
    IterType m_MaxIterations;
    IterType m_PeriodMaybeZero;  // Zero if not worked out
    int32_t m_CompressionErrorExp;
    int32_t m_IntermediateCompressionErrorExp;

    AddPointOptions m_RefOrbitOptions;
    std::wstring m_BaseFilename;
    mutable HANDLE m_MetaFileHandle;

    GrowableVector<GPUReferenceIter<T, PExtras>> m_FullOrbit;
    std::vector<IterTypeFull> m_Rebases;
    IterType m_UncompressedItersInOrbit;
    const size_t m_GenerationNumber;

    std::unique_ptr<LAReference<IterType, T, SubType, PExtras>> m_LaReference;

    uint64_t m_AuthoritativePrecisionInBits;
    std::vector<HighPrecisionT<HPDestructor::False>> m_ReuseX;
    std::vector<HighPrecisionT<HPDestructor::False>> m_ReuseY;
    std::vector<IterTypeFull> m_ReuseIndices;
    std::unique_ptr<ThreadMemory> m_ReuseAllocations;

    BenchmarkData m_BenchmarkOrbit;

    int64_t m_DeltaPrecisionCached;
    int64_t m_ExtraPrecisionCached;
};

template<typename IterType, class T, PerturbExtras PExtras>
class RefOrbitCompressor : public TemplateHelpers<IterType, T, PExtras> {
    template<typename IterType, class T, PerturbExtras PExtras> friend class PerturbationResults;

    using TemplateHelpers = TemplateHelpers<IterType, T, PExtras>;
    using SubType = TemplateHelpers::SubType;

    template<class LocalSubType>
    using HDRFloatComplex = TemplateHelpers::template HDRFloatComplex<LocalSubType>;

    friend class RuntimeDecompressor<IterType, T, PExtras>;

    PerturbationResults<IterType, T, PExtras> &results;
    T zx;
    T zy;
    T Two;
    T CompressionError;
    int32_t CompressionErrorExp;
    IterTypeFull CurCompressedIndex;

public:
    RefOrbitCompressor(
        PerturbationResults<IterType, T, PExtras> &results,
        int32_t CompressionErrorExp);

    void MaybeAddCompressedIteration(GPUReferenceIter<T, PExtras> iter)
        requires (PExtras == PerturbExtras::SimpleCompression && !Introspection::IsTDblFlt<T>());
};


template<typename IterType, class T, PerturbExtras PExtras>
class SimpleIntermediateOrbitCompressor : public TemplateHelpers<IterType, T, PExtras> {
    template<typename IterType, class T, PerturbExtras PExtras> friend class PerturbationResults;

    using TemplateHelpers = TemplateHelpers<IterType, T, PExtras>;
    using SubType = TemplateHelpers::SubType;

    template<class LocalSubType>
    using HDRFloatComplex = TemplateHelpers::template HDRFloatComplex<LocalSubType>;

    friend class RuntimeDecompressor<IterType, T, PExtras>;

    PerturbationResults<IterType, T, PExtras> &results;
    mpf_t zx;
    mpf_t zy;
    mpf_t cx;
    mpf_t cy;
    mpf_t Two;
    mpf_t CompressionError;
    mpf_t ReducedZx;
    mpf_t ReducedZy;
    mpf_t Temp[6];
    int32_t IntermediateCompressionErrorExp;
    IterTypeFull CurCompressedIndex;

public:
    SimpleIntermediateOrbitCompressor(
        PerturbationResults<IterType, T, PExtras> &results,
        int32_t CompressionErrorExp);

    ~SimpleIntermediateOrbitCompressor();

    void MaybeAddCompressedIteration(
        mpf_t incomingZx,
        mpf_t incomingZy,
        IterTypeFull index);
};

template<typename IterType, class T, PerturbExtras PExtras>
class MaxIntermediateOrbitCompressor : public TemplateHelpers<IterType, T, PExtras> {
    template<typename IterType, class T, PerturbExtras PExtras> friend class PerturbationResults;

    using TemplateHelpers = TemplateHelpers<IterType, T, PExtras>;
    using SubType = TemplateHelpers::SubType;

    template<class LocalSubType>
    using HDRFloatComplex = TemplateHelpers::template HDRFloatComplex<LocalSubType>;

    friend class RuntimeDecompressor<IterType, T, PExtras>;

    PerturbationResults<IterType, T, PExtras> &results;
    mpf_t zx;
    mpf_t zy;
    mpf_t cx;
    mpf_t cy;
    mpf_t Two;
    mpf_t CompressionError;
    mpf_t ReducedZx;
    mpf_t ReducedZy;
    mpf_t Temp[6];
    int32_t IntermediateCompressionErrorExp;
    IterTypeFull CurCompressedIndex;

public:
    MaxIntermediateOrbitCompressor(
        PerturbationResults<IterType, T, PExtras> &results,
        int32_t CompressionErrorExp);

    ~MaxIntermediateOrbitCompressor();

    void MaybeAddCompressedIteration(
        mpf_t incomingZx,
        mpf_t incomingZy,
        IterTypeFull index);
};
