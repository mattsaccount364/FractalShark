#pragma once

#include "HDRFloat.h"
#include "HighPrecision.h"

class Fractal;

template<typename IterType, class T, CalcBad Bad>
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

    template<typename IterType, class T, CalcBad Bad>
    std::vector<std::unique_ptr<PerturbationResults<IterType, T, Bad>>>& GetPerturbationResults() ;

    template<typename IterType, class T, CalcBad Bad>
    PerturbationResults<IterType, T, Bad> *AddPerturbationResults(
        std::unique_ptr<PerturbationResults<IterType, T, Bad>> results);

    template<typename IterType, class T, class SubType, BenchmarkMode BenchmarkState>
    void AddPerturbationReferencePoint();

    template<typename IterType, class T, bool Authoritative, CalcBad Bad>
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
        CalcBad Bad,
        RefOrbitCalc::Extras Ex,
        class ConvertTType = T>
    PerturbationResults<IterType, ConvertTType, Bad>* GetAndCreateUsefulPerturbationResults();

    template<typename IterType, class SrcT, CalcBad SrcEnableBad, class DestT, CalcBad DestEnableBad>
    PerturbationResults<IterType, DestT, DestEnableBad>* CopyUsefulPerturbationResults(PerturbationResults<IterType, SrcT, SrcEnableBad>& src_array);

    void ClearPerturbationResults(PerturbationResultType type);
    void ResetGuess(HighPrecision x = HighPrecision(0), HighPrecision y = HighPrecision(0));

    void SaveAllOrbits();
    void LoadAllOrbits();

private:
private:
    bool RequiresBadCalc() const;
    bool IsThisPerturbationArrayUsed(void* check) const;

    template<typename IterType, class T, CalcBad Bad>
    PerturbationResults<IterType, T, Bad>& GetPerturbationResults(size_t index);

    template<typename IterType, class T, class SubType, bool Periodicity, BenchmarkMode BenchmarkState>
    bool AddPerturbationReferencePointSTReuse(HighPrecision initX, HighPrecision initY);

    template<typename IterType, class T, class SubType, bool Periodicity, BenchmarkMode BenchmarkState>
    bool AddPerturbationReferencePointMT3Reuse(HighPrecision initX, HighPrecision initY);

    template<typename IterType, class T, class SubType, bool Periodicity, BenchmarkMode BenchmarkState, CalcBad Bad, ReuseMode Reuse>
    void AddPerturbationReferencePointST(HighPrecision initX, HighPrecision initY);

    template<typename IterType, class T, class SubType, bool Periodicity, BenchmarkMode BenchmarkState, CalcBad Bad, ReuseMode Reuse>
    void AddPerturbationReferencePointMT3(HighPrecision initX, HighPrecision initY);

    template<typename IterType, class T, class SubType, bool Periodicity, BenchmarkMode BenchmarkState, CalcBad Bad, ReuseMode Reuse>
    void AddPerturbationReferencePointMT5(HighPrecision initX, HighPrecision initY);

    template<typename IterType, class T, bool Authoritative, CalcBad Bad>
    PerturbationResults<IterType, T, Bad>* GetUsefulPerturbationResults();

    template<typename IterType, class T, class PerturbationResultsType, CalcBad Bad, ReuseMode Reuse>
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

    template<typename IterType, CalcBad Bad>
    struct Container {
        std::vector<std::unique_ptr<PerturbationResults<IterType, double, Bad>>> m_PerturbationResultsDouble;
        std::vector<std::unique_ptr<PerturbationResults<IterType, float, Bad>>> m_PerturbationResultsFloat;
        std::vector<std::unique_ptr<PerturbationResults<IterType, CudaDblflt<MattDblflt>, Bad>>> m_PerturbationResults2xFloat;
        std::vector<std::unique_ptr<PerturbationResults<IterType, HDRFloat<double>, Bad>>> m_PerturbationResultsHDRDouble;
        std::vector<std::unique_ptr<PerturbationResults<IterType, HDRFloat<float>, Bad>>> m_PerturbationResultsHDRFloat;
        std::vector<std::unique_ptr<PerturbationResults<IterType, HDRFloat<CudaDblflt<MattDblflt>>, Bad>>> m_PerturbationResultsHDR2xFloat;
    };

    Container<uint32_t, CalcBad::Disable> c32d;
    Container<uint32_t, CalcBad::Enable> c32e;
    Container<uint64_t, CalcBad::Disable> c64d;
    Container<uint64_t, CalcBad::Enable> c64e;

    template<typename IterType, CalcBad Bad>
    Container<IterType, Bad> &GetContainer();

    template<typename IterType, CalcBad Bad>
    const Container<IterType, Bad>& GetContainer() const;
};