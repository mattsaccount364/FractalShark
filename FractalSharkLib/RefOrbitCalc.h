#pragma once

#include "HDRFloat.h"
#include "Vectors.h"
#include <variant>
#include <vector>
#include <type_traits>

class Fractal;
struct RefOrbitDetails;

class PerturbationResultsBase;
template<typename IterType, class T, PerturbExtras PExtras>
class PerturbationResults;

struct RecommendedSettings;
struct OrbitParameterPack;

class RenderAlgorithm;

class MPIRBoundedAllocator;
class MPIRBumpAllocator;

class RefOrbitCalc {
public:
    using AwesomeVariant = std::variant <
        const PerturbationResults<uint32_t, double, PerturbExtras::Disable> *,
        const PerturbationResults<uint32_t, float, PerturbExtras::Disable> *,
        const PerturbationResults<uint32_t, CudaDblflt<MattDblflt>, PerturbExtras::Disable> *,
        const PerturbationResults<uint32_t, HDRFloat<double>, PerturbExtras::Disable> *,
        const PerturbationResults<uint32_t, HDRFloat<float>, PerturbExtras::Disable> *,
        const PerturbationResults<uint32_t, HDRFloat<CudaDblflt<MattDblflt>>, PerturbExtras::Disable> *,

        const PerturbationResults<uint32_t, double, PerturbExtras::Bad> *,
        const PerturbationResults<uint32_t, float, PerturbExtras::Bad> *,
        const PerturbationResults<uint32_t, CudaDblflt<MattDblflt>, PerturbExtras::Bad> *,
        const PerturbationResults<uint32_t, HDRFloat<double>, PerturbExtras::Bad> *,
        const PerturbationResults<uint32_t, HDRFloat<float>, PerturbExtras::Bad> *,
        const PerturbationResults<uint32_t, HDRFloat<CudaDblflt<MattDblflt>>, PerturbExtras::Bad> *,

        const PerturbationResults<uint32_t, double, PerturbExtras::SimpleCompression> *,
        const PerturbationResults<uint32_t, float, PerturbExtras::SimpleCompression> *,
        const PerturbationResults<uint32_t, CudaDblflt<MattDblflt>, PerturbExtras::SimpleCompression> *,
        const PerturbationResults<uint32_t, HDRFloat<double>, PerturbExtras::SimpleCompression> *,
        const PerturbationResults<uint32_t, HDRFloat<float>, PerturbExtras::SimpleCompression> *,
        const PerturbationResults<uint32_t, HDRFloat<CudaDblflt<MattDblflt>>, PerturbExtras::SimpleCompression> *,

        const PerturbationResults<uint64_t, double, PerturbExtras::Disable> *,
        const PerturbationResults<uint64_t, float, PerturbExtras::Disable> *,
        const PerturbationResults<uint64_t, CudaDblflt<MattDblflt>, PerturbExtras::Disable> *,
        const PerturbationResults<uint64_t, HDRFloat<double>, PerturbExtras::Disable> *,
        const PerturbationResults<uint64_t, HDRFloat<float>, PerturbExtras::Disable> *,
        const PerturbationResults<uint64_t, HDRFloat<CudaDblflt<MattDblflt>>, PerturbExtras::Disable> *,

        const PerturbationResults<uint64_t, double, PerturbExtras::Bad> *,
        const PerturbationResults<uint64_t, float, PerturbExtras::Bad> *,
        const PerturbationResults<uint64_t, CudaDblflt<MattDblflt>, PerturbExtras::Bad> *,
        const PerturbationResults<uint64_t, HDRFloat<double>, PerturbExtras::Bad> *,
        const PerturbationResults<uint64_t, HDRFloat<float>, PerturbExtras::Bad> *,
        const PerturbationResults<uint64_t, HDRFloat<CudaDblflt<MattDblflt>>, PerturbExtras::Bad> *,

        const PerturbationResults<uint64_t, double, PerturbExtras::SimpleCompression> *,
        const PerturbationResults<uint64_t, float, PerturbExtras::SimpleCompression> *,
        const PerturbationResults<uint64_t, CudaDblflt<MattDblflt>, PerturbExtras::SimpleCompression> *,
        const PerturbationResults<uint64_t, HDRFloat<double>, PerturbExtras::SimpleCompression> *,
        const PerturbationResults<uint64_t, HDRFloat<float>, PerturbExtras::SimpleCompression> *,
        const PerturbationResults<uint64_t, HDRFloat<CudaDblflt<MattDblflt>>, PerturbExtras::SimpleCompression> *
    > ;

    using AwesomeVariantUniquePtr = std::variant <
        std::unique_ptr<PerturbationResults<uint32_t, double, PerturbExtras::Disable>>,
        std::unique_ptr<PerturbationResults<uint32_t, float, PerturbExtras::Disable>>,
        std::unique_ptr<PerturbationResults<uint32_t, CudaDblflt<MattDblflt>, PerturbExtras::Disable>>,
        std::unique_ptr<PerturbationResults<uint32_t, HDRFloat<double>, PerturbExtras::Disable>>,
        std::unique_ptr<PerturbationResults<uint32_t, HDRFloat<float>, PerturbExtras::Disable>>,
        std::unique_ptr<PerturbationResults<uint32_t, HDRFloat<CudaDblflt<MattDblflt>>, PerturbExtras::Disable>>,

        std::unique_ptr<PerturbationResults<uint32_t, double, PerturbExtras::Bad>>,
        std::unique_ptr<PerturbationResults<uint32_t, float, PerturbExtras::Bad>>,
        std::unique_ptr<PerturbationResults<uint32_t, CudaDblflt<MattDblflt>, PerturbExtras::Bad>>,
        std::unique_ptr<PerturbationResults<uint32_t, HDRFloat<double>, PerturbExtras::Bad>>,
        std::unique_ptr<PerturbationResults<uint32_t, HDRFloat<float>, PerturbExtras::Bad>>,
        std::unique_ptr<PerturbationResults<uint32_t, HDRFloat<CudaDblflt<MattDblflt>>, PerturbExtras::Bad>>,

        std::unique_ptr<PerturbationResults<uint32_t, double, PerturbExtras::SimpleCompression>>,
        std::unique_ptr<PerturbationResults<uint32_t, float, PerturbExtras::SimpleCompression>>,
        std::unique_ptr<PerturbationResults<uint32_t, CudaDblflt<MattDblflt>, PerturbExtras::SimpleCompression>>,
        std::unique_ptr<PerturbationResults<uint32_t, HDRFloat<double>, PerturbExtras::SimpleCompression>>,
        std::unique_ptr<PerturbationResults<uint32_t, HDRFloat<float>, PerturbExtras::SimpleCompression>>,
        std::unique_ptr<PerturbationResults<uint32_t, HDRFloat<CudaDblflt<MattDblflt>>, PerturbExtras::SimpleCompression>>,

        std::unique_ptr<PerturbationResults<uint64_t, double, PerturbExtras::Disable>>,
        std::unique_ptr<PerturbationResults<uint64_t, float, PerturbExtras::Disable>>,
        std::unique_ptr<PerturbationResults<uint64_t, CudaDblflt<MattDblflt>, PerturbExtras::Disable>>,
        std::unique_ptr<PerturbationResults<uint64_t, HDRFloat<double>, PerturbExtras::Disable>>,
        std::unique_ptr<PerturbationResults<uint64_t, HDRFloat<float>, PerturbExtras::Disable>>,
        std::unique_ptr<PerturbationResults<uint64_t, HDRFloat<CudaDblflt<MattDblflt>>, PerturbExtras::Disable>>,

        std::unique_ptr<PerturbationResults<uint64_t, double, PerturbExtras::Bad>>,
        std::unique_ptr<PerturbationResults<uint64_t, float, PerturbExtras::Bad>>,
        std::unique_ptr<PerturbationResults<uint64_t, CudaDblflt<MattDblflt>, PerturbExtras::Bad>>,
        std::unique_ptr<PerturbationResults<uint64_t, HDRFloat<double>, PerturbExtras::Bad>>,
        std::unique_ptr<PerturbationResults<uint64_t, HDRFloat<float>, PerturbExtras::Bad>>,
        std::unique_ptr<PerturbationResults<uint64_t, HDRFloat<CudaDblflt<MattDblflt>>, PerturbExtras::Bad>>,

        std::unique_ptr<PerturbationResults<uint64_t, double, PerturbExtras::SimpleCompression>>,
        std::unique_ptr<PerturbationResults<uint64_t, float, PerturbExtras::SimpleCompression>>,
        std::unique_ptr<PerturbationResults<uint64_t, CudaDblflt<MattDblflt>, PerturbExtras::SimpleCompression>>,
        std::unique_ptr<PerturbationResults<uint64_t, HDRFloat<double>, PerturbExtras::SimpleCompression>>,
        std::unique_ptr<PerturbationResults<uint64_t, HDRFloat<float>, PerturbExtras::SimpleCompression>>,
        std::unique_ptr<PerturbationResults<uint64_t, HDRFloat<CudaDblflt<MattDblflt>>, PerturbExtras::SimpleCompression>>
    > ;

    enum class ReuseMode {
        DontSaveForReuse,
        SaveForReuse1, // 3 thread, no compression
        SaveForReuse2, // 4 thread, no compression
        SaveForReuse3, // 4 thread, compression
        SaveForReuse4, // 4 thread, max compression
    };

    enum class BenchmarkMode {
        Disable,
        Enable
    };

    enum class PerturbationAlg {
        ST,
        MT,
        STPeriodicity,
        MTPeriodicity3,
        MTPeriodicity3PerturbMTHighSTMed,
        MTPeriodicity3PerturbMTHighMTMed1,
        MTPeriodicity3PerturbMTHighMTMed2,
        MTPeriodicity3PerturbMTHighMTMed3,
        MTPeriodicity3PerturbMTHighMTMed4,
        MTPeriodicity5,
        GPU,
        Auto
    };

    enum class PerturbationResultType {
        None,
        LAOnly,
        MediumRes,
        HighRes,
        All
    };

    RefOrbitCalc(
        const Fractal &fractal,
        uint64_t commitLimitInBytes);

    bool RequiresReuse() const;

    template<typename IterType, class T, PerturbExtras PExtras>
    void OptimizeMemory();

    void SetPerturbationAlg(PerturbationAlg alg);
    PerturbationAlg GetPerturbationAlg() const;
    std::string GetPerturbationAlgStr() const;

    template<typename IterType, class T, PerturbExtras PExtras>
    PerturbationResults<IterType, T, PExtras> *GetLast();

    template<typename IterType, class T, PerturbExtras PExtras>
    const PerturbationResults<IterType, T, PExtras> *GetLastConst() const;

    template<typename IterType, class T, PerturbExtras PExtras>
    PerturbationResults<IterType, T, PExtras> *GetElt(size_t i);

    template<typename IterType, class T, PerturbExtras PExtras>
    const PerturbationResults<IterType, T, PExtras> *GetEltConst(size_t i) const;

    void SetOptions(AddPointOptions options);

    template<
        typename IterType,
        class T,
        class SubType,
        PerturbExtras PExtras,
        BenchmarkMode BenchmarkState>
    void AddPerturbationReferencePoint();

    template<typename IterType, class T, bool Authoritative, PerturbExtras PExtras>
    bool IsPerturbationResultUsefulHere(size_t i) const;

    bool RequiresReferencePoints() const;

    enum class Extras {
        None,
        IncludeLAv2
    };

    template<
        typename IterType,
        class T,
        class SubType,
        PerturbExtras PExtras,
        RefOrbitCalc::Extras Ex,
        class ConvertTType = T>
    const PerturbationResults<IterType, ConvertTType, PExtras> *
        GetUsefulPerturbationResults() const;

    template<
        typename IterType,
        class T,
        class SubType,
        PerturbExtras PExtras,
        RefOrbitCalc::Extras Ex,
        class ConvertTType = T>
    PerturbationResults<IterType, ConvertTType, PExtras> *
        GetAndCreateUsefulPerturbationResults();

    template<typename IterType, class SrcT, PerturbExtras SrcEnableBad, class DestT, PerturbExtras DestEnableBad>
    PerturbationResults<IterType, DestT, DestEnableBad> *CopyUsefulPerturbationResults(
        PerturbationResults<IterType, SrcT, SrcEnableBad> &src_array)
        requires ((SrcEnableBad == PerturbExtras::Bad && DestEnableBad == PerturbExtras::Bad) ||
                  (SrcEnableBad == PerturbExtras::Disable && DestEnableBad == PerturbExtras::Disable));

    void ClearPerturbationResults(PerturbationResultType type);
    void ResetGuess(HighPrecision x = HighPrecision(0), HighPrecision y = HighPrecision(0));
    void ResetLastUsedOrbit();

    void SaveAllOrbits();
    void LoadAllOrbits();

    void GetSomeDetails(RefOrbitDetails &details) const;
    void SaveOrbit(CompressToDisk compression, std::wstring filename) const;
    void DiffOrbit(
        CompressToDisk compression,
        std::wstring outFile,
        std::wstring filename1,
        std::wstring filename2) const;

    template<typename IterType, class T, PerturbExtras PExtras>
    void SaveOrbitResults(const PerturbationResults<IterType, T, PExtras> &results, std::wstring imagFilename) const;
    void SaveOrbitResults(std::wstring imagFilename) const;

    const PerturbationResultsBase *LoadOrbit(
        ImaginaSettings imaginaSettings,
        CompressToDisk compression,
        RenderAlgorithm renderAlg,
        std::wstring imagFilename,
        RecommendedSettings *recommendedSettings);

    RefOrbitCalc::AwesomeVariantUniquePtr LoadOrbitConst(
        CompressToDisk compression,
        std::wstring imagFilename,
        RecommendedSettings *recommendedSettings) const;

    void DrawPerturbationResults();

private:
    static constexpr size_t MaxStoredOrbits = 64;

    template<typename DestIterType, class DestT, PerturbExtras DestPExtras>
    RefOrbitCalc::AwesomeVariantUniquePtr LoadOrbitConvert(
        CompressToDisk compression,
        std::wstring imagFilename,
        RecommendedSettings *recommendedSettings);

    void LoadOrbitConstInternal(
        OrbitParameterPack &orbitParameterPack,
        CompressToDisk compression,
        std::wstring imagFilename,
        RecommendedSettings *recommendedSettings) const;

    bool RequiresCompression(RenderAlgorithm renderAlg) const;
    //bool IsThisPerturbationArrayUsed(void *check) const;

    template<typename IterType, class T, PerturbExtras PExtras>
    const PerturbationResults<IterType, T, PExtras> *GetPerturbationResults(size_t index) const;

    template<typename IterType, class T, class SubType, bool Periodicity, BenchmarkMode BenchmarkState, ReuseMode Reuse>
    bool AddPerturbationReferencePointSTReuse(HighPrecision initX, HighPrecision initY);

    template<typename IterType, class T, class SubType, bool Periodicity, BenchmarkMode BenchmarkState, ReuseMode Reuse>
    bool AddPerturbationReferencePointMT3Reuse(HighPrecision initX, HighPrecision initY);

    template<typename IterType, class T, class SubType, bool Periodicity, BenchmarkMode BenchmarkState, PerturbExtras PExtras, ReuseMode Reuse>
    void AddPerturbationReferencePointST(HighPrecision initX, HighPrecision initY);

    template<typename IterType, class T, class SubType, bool Periodicity, BenchmarkMode BenchmarkState, PerturbExtras PExtras, ReuseMode Reuse>
    void AddPerturbationReferencePointMT3(HighPrecision initX, HighPrecision initY);

    template<typename IterType, class T, class SubType, bool Periodicity, BenchmarkMode BenchmarkState, PerturbExtras PExtras, ReuseMode Reuse>
    void AddPerturbationReferencePointMT5(HighPrecision initX, HighPrecision initY);

    template <typename IterType,
              class T,
              bool Periodicity,
              RefOrbitCalc::BenchmarkMode BenchmarkState,
              PerturbExtras PExtras,
              RefOrbitCalc::ReuseMode Reuse>
    void AddPerturbationReferencePointGPU(HighPrecision cx, HighPrecision cy);

    template<typename IterType, class T, bool Authoritative, PerturbExtras PExtras>
    PerturbationResults<IterType, T, PExtras> *GetUsefulPerturbationResultsMutable();

    template<typename IterType, class T, bool Authoritative, PerturbExtras PExtras>
    const PerturbationResults<IterType, T, PExtras> *GetUsefulPerturbationResultsConst() const;

    template<typename IterType, class T, class PerturbationResultsType, PerturbExtras PExtras, ReuseMode Reuse>
    void InitResults(PerturbationResultsType &results, const HighPrecision &initX, const HighPrecision &initY);

    template<typename IterType, class T, PerturbExtras PExtras>
    void DrawPerturbationResultsHelper();

    template<RefOrbitCalc::ReuseMode Reuse>
    void InitAllocatorsIfNeeded(
        std::unique_ptr<MPIRBoundedAllocator> &boundedAllocator,
        std::unique_ptr<MPIRBumpAllocator> &bumpAllocator);

    template<RefOrbitCalc::ReuseMode Reuse>
    void ShutdownAllocatorsIfNeeded(
        std::unique_ptr<MPIRBoundedAllocator> &boundedAllocator,
        std::unique_ptr<MPIRBumpAllocator> &bumpAllocator);

    template<
        typename IterType,
        class T>
    bool GetReuseResults(
        const HighPrecision &cx,
        const HighPrecision &cy,
        const PerturbationResults<IterType, T, PerturbExtras::Disable> &existingAuthoritativeResults,
        PerturbationResults<IterType, T, PerturbExtras::Disable> *&outResults);

    void GetEstimatedPrecision(
        uint64_t authoritativePrecisionInBits,
        int64_t &deltaPrecision,
        int64_t &extraPrecision) const;

    class ScopedAffinity {
    public:
        ScopedAffinity(
            RefOrbitCalc &refOrbitCalc,
            HANDLE thread1,
            HANDLE thread2,
            HANDLE thread3,
            HANDLE thread4);
        ~ScopedAffinity();

        void SetCpuAffinityAsNeeded();

    private:
        RefOrbitCalc &m_RefOrbitCalc;
        HANDLE m_Thread1;
        HANDLE m_Thread2;
        HANDLE m_Thread3;
        HANDLE m_Thread4;
    };


    PerturbationAlg m_PerturbationAlg;
    const Fractal &m_Fractal;
    HighPrecision m_PerturbationGuessCalcX;
    HighPrecision m_PerturbationGuessCalcY;

    AddPointOptions m_RefOrbitOptions;

    size_t m_GuessReserveSize;

    size_t GetNextGenerationNumber() const;
    mutable size_t m_GenerationNumber;

    size_t m_NumCpuCores;
    bool m_HyperthreadingEnabled;

    void PushbackResults(auto results);
    std::vector<AwesomeVariantUniquePtr> m_C;

    mutable AwesomeVariant m_LastUsedRefOrbit;

    // Helper to strip const and reference qualifiers
    template <typename T>
    struct StripQualifiers {
        using Type = typename std::remove_const<typename std::remove_reference<T>::type>::type;
    };

    const uint64_t m_CommitLimitInBytes;
};