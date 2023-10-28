#pragma once

#include "HDRFloat.h"
#include "HighPrecision.h"

class Fractal;

template<class T, CalcBad Bad>
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

private:
    bool RequiresBadCalc() const;
    bool IsThisPerturbationArrayUsed(void* check) const;

public:
    void OptimizeMemory();
    void SetPerturbationAlg(PerturbationAlg alg) { m_PerturbationAlg = alg; }
    PerturbationAlg GetPerturbationAlg() const { return m_PerturbationAlg; }

    template<class T, CalcBad Bad>
    std::vector<std::unique_ptr<PerturbationResults<T, Bad>>>& GetPerturbationResults();

    template<class T, class SubType, BenchmarkMode BenchmarkState>
    void AddPerturbationReferencePoint();

private:
    template<class T, class SubType, bool Periodicity, BenchmarkMode BenchmarkState>
    bool AddPerturbationReferencePointSTReuse(HighPrecision initX, HighPrecision initY);

    template<class T, class SubType, bool Periodicity, BenchmarkMode BenchmarkState>
    bool AddPerturbationReferencePointMT3Reuse(HighPrecision initX, HighPrecision initY);

    template<class T, class SubType, bool Periodicity, BenchmarkMode BenchmarkState, CalcBad Bad, ReuseMode Reuse>
    void AddPerturbationReferencePointST(HighPrecision initX, HighPrecision initY);

    template<class T, class SubType, bool Periodicity, BenchmarkMode BenchmarkState, CalcBad Bad, ReuseMode Reuse>
    void AddPerturbationReferencePointMT3(HighPrecision initX, HighPrecision initY);

    template<class T, class SubType, bool Periodicity, BenchmarkMode BenchmarkState, CalcBad Bad, ReuseMode Reuse>
    void AddPerturbationReferencePointMT5(HighPrecision initX, HighPrecision initY);

public:
    template<class T, bool Authoritative>
    bool IsPerturbationResultUsefulHere(size_t i) const;

    bool RequiresReferencePoints() const;

    enum class Extras {
        None,
        IncludeLAv2
    };

    template<class T, class SubType, CalcBad Bad, RefOrbitCalc::Extras Ex>
    PerturbationResults<T, Bad>* GetAndCreateUsefulPerturbationResults();

private:
    template<class T, CalcBad Bad>
    std::vector<std::unique_ptr<PerturbationResults<T, Bad>>>* GetPerturbationArray();

    template<class T, class SubType, bool Authoritative, CalcBad Bad>
    PerturbationResults<T, Bad>* GetUsefulPerturbationResults();

public:
    template<class SrcT, CalcBad SrcEnableBad, class DestT, CalcBad DestEnableBad>
    PerturbationResults<DestT, DestEnableBad>* CopyUsefulPerturbationResults(PerturbationResults<SrcT, SrcEnableBad>& src_array);

    void ClearPerturbationResults(PerturbationResultType type);
    void ResetGuess(HighPrecision x = HighPrecision(0), HighPrecision y = HighPrecision(0));

private:
    template<class T, class PerturbationResultsType, CalcBad Bad, ReuseMode Reuse>
    void InitResults(PerturbationResultsType&results, const HighPrecision &initX, const HighPrecision &initY);

    PerturbationAlg m_PerturbationAlg;
    Fractal& m_Fractal;

    HighPrecision m_PerturbationGuessCalcX;
    HighPrecision m_PerturbationGuessCalcY;

    size_t m_GuessReserveSize;

    std::vector<std::unique_ptr<PerturbationResults<double, CalcBad::Disable>>> m_PerturbationResultsDouble;
    std::vector<std::unique_ptr<PerturbationResults<float, CalcBad::Disable>>> m_PerturbationResultsFloat;
    std::vector<std::unique_ptr<PerturbationResults<HDRFloat<double>, CalcBad::Disable>>> m_PerturbationResultsHDRDouble;
    std::vector<std::unique_ptr<PerturbationResults<HDRFloat<float>, CalcBad::Disable>>> m_PerturbationResultsHDRFloat;

    std::vector<std::unique_ptr<PerturbationResults<double, CalcBad::Enable>>> m_PerturbationResultsDoubleB;
    std::vector<std::unique_ptr<PerturbationResults<float, CalcBad::Enable>>> m_PerturbationResultsFloatB;
    std::vector<std::unique_ptr<PerturbationResults<HDRFloat<double>, CalcBad::Enable>>> m_PerturbationResultsHDRDoubleB;
    std::vector<std::unique_ptr<PerturbationResults<HDRFloat<float>, CalcBad::Enable>>> m_PerturbationResultsHDRFloatB;
};