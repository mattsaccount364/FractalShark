#pragma once

#include "HDRFloat.h"
#include "HighPrecision.h"
#include "Vectors.h"
#include "BenchmarkData.h"
#include <variant>
#include <vector>

class Fractal;

template<typename IterType, class T, PerturbExtras PExtras>
class PerturbationResults;

class RefOrbitCalc {
public:
    enum class ReuseMode {
        DontSaveForReuse,
        SaveForReuse,
        RunWithReuse
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
        MTPeriodicity3PerturbMTHighMTMed,
        MTPeriodicity5
    };

    enum class PerturbationResultType {
        None,
        LAOnly,
        MediumRes,
        HighRes,
        All
    };

    RefOrbitCalc(Fractal &Fractal);

    bool RequiresReuse() const;

    void OptimizeMemory();
    void SetPerturbationAlg(PerturbationAlg alg);
    PerturbationAlg GetPerturbationAlg() const;
    std::string GetPerturbationAlgStr() const;

    template<typename IterType, class T, PerturbExtras PExtras>
    std::vector<std::unique_ptr<PerturbationResults<IterType, T, PExtras>>>& GetPerturbationResults();

    void SetOptions(AddPointOptions options);

    template<
        typename IterType,
        class T,
        class SubType,
        PerturbExtras PExtras,
        BenchmarkMode BenchmarkState>
    void AddPerturbationReferencePoint();

    template<typename IterType, class T, bool Authoritative, PerturbExtras PExtras>
    bool IsPerturbationResultUsefulHere(size_t i);

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
    PerturbationResults<IterType, ConvertTType, PExtras>*
        GetUsefulPerturbationResults();

    template<
        typename IterType,
        class T,
        class SubType,
        PerturbExtras PExtras,
        RefOrbitCalc::Extras Ex,
        class ConvertTType = T>
    PerturbationResults<IterType, ConvertTType, PExtras>*
        GetAndCreateUsefulPerturbationResults();

    template<typename IterType, class SrcT, PerturbExtras SrcEnableBad, class DestT, PerturbExtras DestEnableBad>
    PerturbationResults<IterType, DestT, DestEnableBad>* CopyUsefulPerturbationResults(PerturbationResults<IterType, SrcT, SrcEnableBad>& src_array);

    void ClearPerturbationResults(PerturbationResultType type);
    void ResetGuess(HighPrecision x = HighPrecision(0), HighPrecision y = HighPrecision(0));

    void SaveAllOrbits();
    void LoadAllOrbits();

    void GetSomeDetails(
        uint64_t &InternalPeriodMaybeZero,
        uint64_t &CompressedIters,
        uint64_t &UncompressedIters,
        int32_t &CompressionErrorExp,
        uint64_t &OrbitMilliseconds,
        uint64_t &LAMilliseconds,
        std::string &PerturbationAlg);
    void SaveOrbitAsText() const;

    void DrawPerturbationResults();

private:
    static constexpr size_t MaxStoredOrbits = 64;

    bool RequiresCompression() const;
    bool IsThisPerturbationArrayUsed(void* check) const;

    template<typename IterType, class T, PerturbExtras PExtras>
    PerturbationResults<IterType, T, PExtras>* AddPerturbationResults(
        std::unique_ptr<PerturbationResults<IterType, T, PExtras>> results);

    template<typename IterType, class T, PerturbExtras PExtras>
    PerturbationResults<IterType, T, PExtras>& GetPerturbationResults(size_t index);

    template<typename IterType, class T, class SubType, bool Periodicity, BenchmarkMode BenchmarkState>
    bool AddPerturbationReferencePointSTReuse(HighPrecision initX, HighPrecision initY);

    template<typename IterType, class T, class SubType, bool Periodicity, BenchmarkMode BenchmarkState>
    bool AddPerturbationReferencePointMT3Reuse(HighPrecision initX, HighPrecision initY);

    template<typename IterType, class T, class SubType, bool Periodicity, BenchmarkMode BenchmarkState, PerturbExtras PExtras, ReuseMode Reuse>
    void AddPerturbationReferencePointST(HighPrecision initX, HighPrecision initY);

    template<typename IterType, class T, class SubType, bool Periodicity, BenchmarkMode BenchmarkState, PerturbExtras PExtras, ReuseMode Reuse>
    void AddPerturbationReferencePointMT3(HighPrecision initX, HighPrecision initY);

    template<typename IterType, class T, class SubType, bool Periodicity, BenchmarkMode BenchmarkState, PerturbExtras PExtras, ReuseMode Reuse>
    void AddPerturbationReferencePointMT5(HighPrecision initX, HighPrecision initY);

    template<typename IterType, class T, bool Authoritative, PerturbExtras PExtras>
    PerturbationResults<IterType, T, PExtras>* GetUsefulPerturbationResults();

    template<typename IterType, class T, class PerturbationResultsType, PerturbExtras PExtras, ReuseMode Reuse>
    void InitResults(PerturbationResultsType&results, const HighPrecision &initX, const HighPrecision &initY);

    template<typename IterType, class T, PerturbExtras PExtras>
    void DrawPerturbationResultsHelper();

    template<RefOrbitCalc::ReuseMode Reuse>
    void InitAllocatorsIfNeeded(
        std::unique_ptr<MPIRBoundedAllocator>& boundedAllocator,
        std::unique_ptr<MPIRBumpAllocator>& bumpAllocator);

    template<RefOrbitCalc::ReuseMode Reuse>
    void ShutdownAllocatorsIfNeeded(
        std::unique_ptr<MPIRBoundedAllocator>& boundedAllocator,
        std::unique_ptr<MPIRBumpAllocator>& bumpAllocator);

    class ScopedAffinity {
    public:
        ScopedAffinity(
            RefOrbitCalc& refOrbitCalc,
            HANDLE thread1,
            HANDLE thread2,
            HANDLE thread3);
        ~ScopedAffinity();

        void SetCpuAffinityAsNeeded();

    private:
        RefOrbitCalc& m_RefOrbitCalc;
        HANDLE m_Thread1;
        HANDLE m_Thread2;
        HANDLE m_Thread3;
    };


    PerturbationAlg m_PerturbationAlg;
    Fractal& m_Fractal;

    HighPrecision m_PerturbationGuessCalcX;
    HighPrecision m_PerturbationGuessCalcY;

    AddPointOptions m_RefOrbitOptions;

    size_t m_GuessReserveSize;

    size_t GetNextGenerationNumber();
    size_t m_GenerationNumber;

    size_t m_NumCpuCores;
    bool m_HyperthreadingEnabled;

    template<typename IterType, PerturbExtras PExtras>
    struct Container {
        std::vector<std::unique_ptr<PerturbationResults<IterType, double, PExtras>>> m_PerturbationResultsDouble;
        std::vector<std::unique_ptr<PerturbationResults<IterType, float, PExtras>>> m_PerturbationResultsFloat;
        std::vector<std::unique_ptr<PerturbationResults<IterType, CudaDblflt<MattDblflt>, PExtras>>> m_PerturbationResults2xFloat;
        std::vector<std::unique_ptr<PerturbationResults<IterType, HDRFloat<double>, PExtras>>> m_PerturbationResultsHDRDouble;
        std::vector<std::unique_ptr<PerturbationResults<IterType, HDRFloat<float>, PExtras>>> m_PerturbationResultsHDRFloat;
        std::vector<std::unique_ptr<PerturbationResults<IterType, HDRFloat<CudaDblflt<MattDblflt>>, PExtras>>> m_PerturbationResultsHDR2xFloat;
    };

    Container<uint32_t, PerturbExtras::Disable> c32d;
    Container<uint32_t, PerturbExtras::Bad> c32e;
    Container<uint32_t, PerturbExtras::EnableCompression> c32c;
    Container<uint64_t, PerturbExtras::Disable> c64d;
    Container<uint64_t, PerturbExtras::Bad> c64e;
    Container<uint64_t, PerturbExtras::EnableCompression> c64c;

    template<typename IterType, PerturbExtras PExtras>
    Container<IterType, PExtras> &GetContainer();

    template<typename IterType, PerturbExtras PExtras>
    const Container<IterType, PExtras>& GetContainer() const;

    std::variant<
        PerturbationResults<uint32_t, double, PerturbExtras::Disable>*,
        PerturbationResults<uint32_t, float, PerturbExtras::Disable>*,
        PerturbationResults<uint32_t, CudaDblflt<MattDblflt>, PerturbExtras::Disable>*,
        PerturbationResults<uint32_t, HDRFloat<double>, PerturbExtras::Disable>*,
        PerturbationResults<uint32_t, HDRFloat<float>, PerturbExtras::Disable>*,
        PerturbationResults<uint32_t, HDRFloat<CudaDblflt<MattDblflt>>, PerturbExtras::Disable>*,

        PerturbationResults<uint32_t, double, PerturbExtras::Bad>*,
        PerturbationResults<uint32_t, float, PerturbExtras::Bad>*,
        PerturbationResults<uint32_t, CudaDblflt<MattDblflt>, PerturbExtras::Bad>*,
        PerturbationResults<uint32_t, HDRFloat<double>, PerturbExtras::Bad>*,
        PerturbationResults<uint32_t, HDRFloat<float>, PerturbExtras::Bad>*,
        PerturbationResults<uint32_t, HDRFloat<CudaDblflt<MattDblflt>>, PerturbExtras::Bad>*,

        PerturbationResults<uint32_t, double, PerturbExtras::EnableCompression>*,
        PerturbationResults<uint32_t, float, PerturbExtras::EnableCompression>*,
        PerturbationResults<uint32_t, CudaDblflt<MattDblflt>, PerturbExtras::EnableCompression>*,
        PerturbationResults<uint32_t, HDRFloat<double>, PerturbExtras::EnableCompression>*,
        PerturbationResults<uint32_t, HDRFloat<float>, PerturbExtras::EnableCompression>*,
        PerturbationResults<uint32_t, HDRFloat<CudaDblflt<MattDblflt>>, PerturbExtras::EnableCompression>*,

        PerturbationResults<uint64_t, double, PerturbExtras::Disable>*,
        PerturbationResults<uint64_t, float, PerturbExtras::Disable>*,
        PerturbationResults<uint64_t, CudaDblflt<MattDblflt>, PerturbExtras::Disable>*,
        PerturbationResults<uint64_t, HDRFloat<double>, PerturbExtras::Disable>*,
        PerturbationResults<uint64_t, HDRFloat<float>, PerturbExtras::Disable>*,
        PerturbationResults<uint64_t, HDRFloat<CudaDblflt<MattDblflt>>, PerturbExtras::Disable>*,

        PerturbationResults<uint64_t, double, PerturbExtras::Bad>*,
        PerturbationResults<uint64_t, float, PerturbExtras::Bad>*,
        PerturbationResults<uint64_t, CudaDblflt<MattDblflt>, PerturbExtras::Bad>*,
        PerturbationResults<uint64_t, HDRFloat<double>, PerturbExtras::Bad>*,
        PerturbationResults<uint64_t, HDRFloat<float>, PerturbExtras::Bad>*,
        PerturbationResults<uint64_t, HDRFloat<CudaDblflt<MattDblflt>>, PerturbExtras::Bad>*,

        PerturbationResults<uint64_t, double, PerturbExtras::EnableCompression>*,
        PerturbationResults<uint64_t, float, PerturbExtras::EnableCompression>*,
        PerturbationResults<uint64_t, CudaDblflt<MattDblflt>, PerturbExtras::EnableCompression>*,
        PerturbationResults<uint64_t, HDRFloat<double>, PerturbExtras::EnableCompression>*,
        PerturbationResults<uint64_t, HDRFloat<float>, PerturbExtras::EnableCompression>*,
        PerturbationResults<uint64_t, HDRFloat<CudaDblflt<MattDblflt>>, PerturbExtras::EnableCompression>*
    > m_LastUsedRefOrbit;
};