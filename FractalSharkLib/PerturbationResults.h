#pragma once

#include "HighPrecision.h"
#include "RefOrbitCalc.h"
#include "PerturbationResultsHelpers.h"
#include "Vectors.h"
#include "ScopedMpir.h"
#include "Introspection.h"
#include "GPU_Types.h"

#include <vector>
#include <type_traits>
#include <array>

template<typename Type, PerturbExtras PExtras>
class GPUReferenceIter;

template<typename IterType, class T, class SubType, PerturbExtras PExtras>
class LAReference;

std::wstring GetTimeAsString(size_t generation_number);

template<typename IterType, class T, PerturbExtras PExtras>
class RefOrbitCompressor;

template<typename IterType, class T, PerturbExtras PExtras>
class SimpleIntermediateOrbitCompressor;

template<typename IterType, class T, PerturbExtras PExtras>
class MaxIntermediateOrbitCompressor;

class PerturbationResultsBase {
public:
    const HighPrecision &GetHiX() const;
    const HighPrecision &GetHiY() const;
    const HighPrecision &GetHiZoomFactor() const;
    IterTypeFull GetMaxIterations() const;

protected:
    PerturbationResultsBase() :
        m_OrbitX{},
        m_OrbitY{},
        m_ZoomFactor{},
        m_FullMaxIterations{} {
    }

    PerturbationResultsBase(
        const HighPrecision &orbitX,
        const HighPrecision &orbitY,
        const HighPrecision &zoomFactor,
        IterTypeFull maxIterations)
        : m_OrbitX(orbitX),
        m_OrbitY(orbitY),
        m_ZoomFactor(zoomFactor),
        m_FullMaxIterations(maxIterations) {
    }

    template<typename IterType>
    void SetMaxIterations(IterType numIterations);

    template<typename IterType>
    void SetMaxIterationsSaturate(IterTypeFull numIterations);

    HighPrecision m_OrbitX;
    HighPrecision m_OrbitY;
    HighPrecision m_ZoomFactor;

private:
    IterTypeFull m_FullMaxIterations;
};

template<typename IterType, class T, PerturbExtras PExtras>
class PerturbationResults : public PerturbationResultsBase, public TemplateHelpers<IterType, T, PExtras> {

public:
    template<typename IterType, class T, PerturbExtras PExtras> friend class PerturbationResults;

    using TemplateHelpers = TemplateHelpers<IterType, T, PExtras>;
    using SubType = TemplateHelpers::SubType;

    template<class LocalSubType>
    using HDRFloatComplex = TemplateHelpers::template HDRFloatComplex<LocalSubType>;

    using LocalIterType = IterType;
    using LocalT = T;
    static constexpr PerturbExtras LocalPExtras = PExtras;

    friend class RuntimeDecompressor<IterType, T, PExtras>;
    friend class IntermediateRuntimeDecompressor<IterType, T, PExtras>;
    friend class IntermediateMaxRuntimeDecompressor<IterType, T, PExtras>;
    friend class RefOrbitCompressor<IterType, T, PExtras>;
    friend class SimpleIntermediateOrbitCompressor<IterType, T, PExtras>;
    friend class MaxIntermediateOrbitCompressor<IterType, T, PExtras>;

    static constexpr char Version[] = "0.46";

    static constexpr bool Is2X32 = (
        std::is_same<T, HDRFloat<CudaDblflt<MattDblflt>>>::value ||
        std::is_same<T, CudaDblflt<MattDblflt>>::value);

    // Generates a filename "base" for all the orbit files.
    std::wstring GenBaseFilename(size_t generation_number) const;

    std::wstring GenBaseFilename(
        AddPointOptions addPointOptions,
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
    //  addPointOptions - whether to save the orbit or not
    //  Generation - the generation number of the orbit
    //  base_filename - the base filename of the orbit, used
    //    when opening existing
    PerturbationResults(
        std::wstring base_filename,
        AddPointOptions addPointOptions,
        size_t Generation);

    PerturbationResults(
        AddPointOptions addPointOptions,
        size_t Generation);

    ~PerturbationResults();

    PerturbationResults(PerturbationResults &&other) = delete;

    PerturbationResults(const PerturbationResults &other) = delete;
    PerturbationResults &operator=(const PerturbationResults &other) = delete;

    size_t GetGenerationNumber() const;

    void ClearLaReference();

    void SetLaReference(
        std::unique_ptr<LAReference<IterType, T, SubType, PExtras>> laReference)
        requires Introspection::TestPExtras<PExtras>::value;

    LAReference<IterType, T, SubType, PExtras> *GetLaReference()
        const
        requires Introspection::TestPExtras<PExtras>::value;

    std::unique_ptr<PerturbationResults<IterType, T, PExtras>>
        CopyPerturbationResults(AddPointOptions addPointOptions,
            size_t new_generation_number)
        requires Introspection::TestPExtras<PExtras>::value;

    template<bool IncludeLA, class Other, PerturbExtras PExtrasOther = PerturbExtras::Disable>
    void CopyFullOrbitVector(
        const PerturbationResults<IterType, Other, PExtrasOther> &other);

    template<bool IncludeLA, class Other, PerturbExtras PExtrasOther = PerturbExtras::Disable>
    void CopyPerturbationResults(
        const PerturbationResults<IterType, Other, PExtrasOther> &other)
        requires Introspection::TestPExtras<PExtras>::value;

    template<PerturbExtras OtherBad>
    void CopySettingsWithoutOrbit(const PerturbationResults<IterType, T, OtherBad> &other);

    void WriteMetadata() const
        requires (PExtras != PerturbExtras::MaxCompression);

    void MaybeOpenMetaFileForDelete() const;

    // This function uses CreateFileMapping and MapViewOfFile to map
    // the file into memory.  Then it loads the meta file using ifstream
    // to get the other data.
    //
    // TODO Loading a 2x32 (with or without HDR) does not work with LA enabled.
    // The code below works, but 
    // The LA table generation doesn't currently work with 2x32 and only supports 1x64
    // (which it then converts to 2x32).  This is a bug.
    bool ReadMetadata()
        requires Introspection::TestPExtras<PExtras>::value;

    void GetComplex(
        RuntimeDecompressor<IterType, T, PExtras> &PerThreadCompressionHelper,
        size_t uncompressed_index,
        T &outX,
        T &outY) const
        requires (!Introspection::IsTDblFlt<T>()) {

        if constexpr (PExtras == PerturbExtras::Disable || PExtras == PerturbExtras::Bad) {
            outX = m_FullOrbit[uncompressed_index].x;
            outY = m_FullOrbit[uncompressed_index].y;
        } else {
            PerThreadCompressionHelper.GetCompressedComplex(uncompressed_index, outX, outY);
        }
    }

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

    void InitResults(
        RefOrbitCalc::ReuseMode Reuse,
        const HighPrecision &cx,
        const HighPrecision &cy,
        const T &radY,
        IterType NumIterations,
        size_t GuessReserveSize);

    uint64_t GetBenchmarkOrbit() const;

    template<RefOrbitCalc::ReuseMode Reuse>
    void CompleteResults(std::unique_ptr<ThreadMemory> allocatorIfAny);

    size_t GetCompressedOrUncompressedOrbitSize() const;
    size_t GetCompressedOrbitSize() const
        requires (PExtras == PerturbExtras::SimpleCompression || PExtras == PerturbExtras::MaxCompression);
    IterType GetCountOrbitEntries() const;

    // PExtras == PerturbExtras::Disable || PExtras == PerturbExtras::Bad
    void AddUncompressedIteration(GPUReferenceIter<T, PExtras> result)
        requires (PExtras == PerturbExtras::Disable || PExtras == PerturbExtras::Bad);

    const GPUReferenceIter<T, PExtras> *GetOrbitData() const;

    const std::string &GetHiXStr() const;
    const std::string &GetHiYStr() const;
    const std::string &GetZoomFactorStr() const;

    T GetOrbitXLow() const;
    T GetOrbitYLow() const;
    T GetZoomFactorLow() const;

    // Radius used for periodicity checking
    T GetMaxRadius() const;

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

    void AddUncompressedReusedEntry(
        const mpf_t x,
        const mpf_t y,
        IterTypeFull index);

    void AddUncompressedRebase(IterTypeFull i, IterTypeFull index);

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

    // For reference:
    // Used:
    //   https://code.mathr.co.uk/fractal-bits/tree/HEAD:/mandelbrot-reference-compression
    //   https://fractalforums.org/fractal-mathematics-and-new-theories/28/reference-compression/5142
    // as a reference for the compression algorithm.
    std::unique_ptr<PerturbationResults<IterType, T, PerturbExtras::SimpleCompression>>
        Compress(
            int32_t compressionErrorExpParam,
            size_t new_generation_number)
        const
        requires (PExtras == PerturbExtras::Disable && !Introspection::IsTDblFlt<T>());

    std::unique_ptr<PerturbationResults<IterType, T, PerturbExtras::Disable>>
        Decompress(size_t NewGenerationNumber)
        const
        requires (PExtras == PerturbExtras::SimpleCompression && !Introspection::IsTDblFlt<T>());

    std::unique_ptr<PerturbationResults<IterType, T, PerturbExtras::SimpleCompression>>
        CompressMax(
            int32_t compressionErrorExpParam,
            size_t new_generation_number,
            bool includeDummy)
        const
        requires (!Introspection::IsTDblFlt<T>());

    template<PerturbExtras PExtrasDest>
    std::unique_ptr<PerturbationResults<IterType, T, PExtrasDest>>
        DecompressMax(int32_t compressionErrorExpParam, size_t NewGenerationNumber)
        const
        requires (PExtras == PerturbExtras::MaxCompression && !Introspection::IsTDblFlt<T>());

    void SaveOrbit(std::wstring filename) const;
    void SaveOrbitBin(std::ofstream &file) const
        requires (PExtras == PerturbExtras::SimpleCompression && !Introspection::IsTDblFlt<T>());
    void SaveOrbitLocation(std::ofstream &file) const;
    void LoadOrbitBin(
        HighPrecision orbitX,
        HighPrecision orbitY,
        IterType fileProvidedIters,
        const Imagina::HRReal &halfH,
        std::ifstream &file)
        requires(PExtras == PerturbExtras::MaxCompression); // std::is_same_v<T, HDRFloat<double>> && 

    void DiffOrbit(
        const PerturbationResults<IterType, T, PExtras> &other,
        std::wstring outFile) const
        requires (!Introspection::IsTDblFlt<T>());

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
    void RecreateFullOrbitVector(size_t overrideViewSize);

    void CloseMetaFileIfOpen() const;

    void MapExistingFiles()
        requires Introspection::TestPExtras<PExtras>::value;

    std::string m_OrbitXStr;
    std::string m_OrbitYStr;
    std::string m_ZoomFactorStr;
    T m_OrbitXLow;
    T m_OrbitYLow;
    T m_ZoomFactorLow;

    T m_MaxRadius;
    IterType m_PeriodMaybeZero;  // Zero if not worked out
    mutable int32_t m_CompressionErrorExp;
    int32_t m_IntermediateCompressionErrorExp;

    AddPointOptions m_RefOrbitOptions;
    std::wstring m_BaseFilename;
    mutable HANDLE m_MetaFileHandle;

    GrowableVector<GPUReferenceIter<T, PExtras>> m_FullOrbit;
    std::vector<IterTypeFull> m_Rebases;
    IterType m_UncompressedItersInOrbit;
    const size_t m_GenerationNumber;

    using LAReferenceConstrainedPtr =
        std::conditional<
            Introspection::TestPExtras<PExtras>::value,
            std::unique_ptr<LAReference<IterType, T, SubType, PExtras>>,
            void *>::type;
    LAReferenceConstrainedPtr m_LaReference;

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
        const mpf_t incomingZx,
        const mpf_t incomingZy,
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
    mpf_t Err;

    mpf_t Constant1;
    mpf_t Constant2;
    mpf_t Threshold2;

    // State maintained between calls
    IterTypeFull I;
    IterTypeFull J;
    IterTypeFull PrevWayPointIteration;
    IterTypeFull ItersSinceLastWrite;
    size_t PhaseDone;
    mpf_t Zx;
    mpf_t Zy;
    mpf_t Dzx;
    mpf_t Dzy;
    mpf_t DzxOld;

    constexpr static size_t TempCount = 13;
    std::array<mpf_t, TempCount> Temp;

    constexpr static size_t NormInternalTempCount = 3;
    std::array<mpf_t, NormInternalTempCount> NormInternalTemp;

    constexpr static size_t NormTempCount = 3;
    std::array<mpf_t, NormTempCount> NormTemp;

    int32_t IntermediateCompressionErrorExp;
    IterTypeFull CurCompressedIndex;

    static constexpr IterTypeFull MaxItersSinceLastWrite = 1000;

public:
    MaxIntermediateOrbitCompressor(
        PerturbationResults<IterType, T, PExtras> &results,
        int32_t CompressionErrorExp);

    ~MaxIntermediateOrbitCompressor();

    void MaybeAddCompressedIteration(
        const mpf_t incomingZx,
        const mpf_t incomingZy,
        IterTypeFull index);

    void CompleteResults();
    void WriteResultsForTesting();
};
