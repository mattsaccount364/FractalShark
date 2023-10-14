#pragma once

#include "HDRFloat.h"
#include "HighPrecision.h"

class Fractal;

template<class T>
class PerturbationResults;

class RefOrbitCalc {
public:
    enum class ReuseMode {
        DontSaveForReuse,
        SaveForReuse,
        RunWithReuse
    };

    enum class CalcBad {
        Disable,
        Enable
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

    template<class T>
    std::vector<std::unique_ptr<PerturbationResults<T>>>& GetPerturbationResults();

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

    template<class T, class SubType>
    PerturbationResults<T>* GetAndCreateUsefulPerturbationResults(Extras extras);

private:
    template<class T>
    std::vector<std::unique_ptr<PerturbationResults<T>>>* GetPerturbationArray();

    template<class T, class SubType, bool Authoritative>
    PerturbationResults<T>* GetUsefulPerturbationResults();

public:
    template<class SrcT, class DestT>
    PerturbationResults<DestT>* CopyUsefulPerturbationResults(PerturbationResults<SrcT>& src_array);

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

    std::vector<std::unique_ptr<PerturbationResults<double>>> m_PerturbationResultsDouble;
    std::vector<std::unique_ptr<PerturbationResults<float>>> m_PerturbationResultsFloat;
    std::vector<std::unique_ptr<PerturbationResults<HDRFloat<double>>>> m_PerturbationResultsHDRDouble;
    std::vector<std::unique_ptr<PerturbationResults<HDRFloat<float>>>> m_PerturbationResultsHDRFloat;
};