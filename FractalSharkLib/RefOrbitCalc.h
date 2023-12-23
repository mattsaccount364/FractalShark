#pragma once

#include "HDRFloat.h"
#include "HighPrecision.h"

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
        MediumRes,
        HighRes,
        All
    };

    RefOrbitCalc(Fractal &Fractal);

    bool RequiresReuse() const;

    void OptimizeMemory();
    void SetPerturbationAlg(PerturbationAlg alg) { m_PerturbationAlg = alg; }
    PerturbationAlg GetPerturbationAlg() const { return m_PerturbationAlg; }

    template<typename IterType, class T, PerturbExtras PExtras>
    std::vector<std::unique_ptr<PerturbationResults<IterType, T, PExtras>>>& GetPerturbationResults() ;

    template<typename IterType, class T, class SubType, BenchmarkMode BenchmarkState>
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
    PerturbationResults<IterType, ConvertTType, PExtras>* GetAndCreateUsefulPerturbationResults();

    template<typename IterType, class SrcT, PerturbExtras SrcEnableBad, class DestT, PerturbExtras DestEnableBad>
    PerturbationResults<IterType, DestT, DestEnableBad>* CopyUsefulPerturbationResults(PerturbationResults<IterType, SrcT, SrcEnableBad>& src_array);

    void ClearPerturbationResults(PerturbationResultType type);
    void ResetGuess(HighPrecision x = HighPrecision(0), HighPrecision y = HighPrecision(0));

    void SaveAllOrbits();
    void LoadAllOrbits();

private:
    bool RequiresBadCalc() const;
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

    PerturbationAlg m_PerturbationAlg;
    Fractal& m_Fractal;

    HighPrecision m_PerturbationGuessCalcX;
    HighPrecision m_PerturbationGuessCalcY;

    size_t m_GuessReserveSize;

    size_t GetNextGenerationNumber();
    size_t GetNextLaGenerationNumber();
    size_t m_GenerationNumber;
    size_t m_LaGenerationNumber;

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
    Container<uint64_t, PerturbExtras::Disable> c64d;
    Container<uint64_t, PerturbExtras::Bad> c64e;

    template<typename IterType, PerturbExtras PExtras>
    Container<IterType, PExtras> &GetContainer();

    template<typename IterType, PerturbExtras PExtras>
    const Container<IterType, PExtras>& GetContainer() const;
};