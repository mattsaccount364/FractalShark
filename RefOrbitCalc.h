#pragma once

#include "PerturbationResults.h"
#include "HDRFloat.h"
#include "HighPrecision.h"

class Fractal;

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
        MTPeriodicity2,
        MTPeriodicity2Perturb,
        MTPeriodicity5
    };

    RefOrbitCalc(Fractal &Fractal);
    bool RequiresBadCalc() const;
    bool IsThisPerturbationArrayUsed(void* check) const;
    void OptimizeMemory();
    void SetPerturbationAlg(PerturbationAlg alg) { m_PerturbationAlg = alg; }

    template<class T>
    std::vector<std::unique_ptr<PerturbationResults<T>>>& GetPerturbationResults();

    template<class T, class SubType, BenchmarkMode BenchmarkState>
    void AddPerturbationReferencePoint();

    template<class T, class SubType, bool Periodicity, BenchmarkMode BenchmarkState, CalcBad Bad>
    bool AddPerturbationReferencePointSTReuse(HighPrecision initX, HighPrecision initY);

    template<class T, class SubType, bool Periodicity, BenchmarkMode BenchmarkState, CalcBad Bad, ReuseMode Reuse>
    void AddPerturbationReferencePointST(HighPrecision initX, HighPrecision initY);

    template<class T, class SubType, bool Periodicity, BenchmarkMode BenchmarkState, CalcBad Bad, ReuseMode Reuse>
    void AddPerturbationReferencePointMT2(HighPrecision initX, HighPrecision initY);

    template<class T, class SubType, bool Periodicity, BenchmarkMode BenchmarkState, CalcBad Bad, ReuseMode Reuse>
    void AddPerturbationReferencePointMT5(HighPrecision initX, HighPrecision initY);

    template<class T, bool Authoritative>
    bool IsPerturbationResultUsefulHere(size_t i) const;

    bool RequiresReferencePoints() const;
    bool IsReferencePerturbationEnabled() const;

    template<class T, class SubType>
    PerturbationResults<T>* GetAndCreateUsefulPerturbationResults();

    template<class T>
    std::vector<std::unique_ptr<PerturbationResults<T>>>* GetPerturbationArray();

    template<class T, class SubType, bool Authoritative>
    PerturbationResults<T>* GetUsefulPerturbationResults();

    template<class SrcT, class DestT>
    PerturbationResults<DestT>* CopyUsefulPerturbationResults(PerturbationResults<SrcT>& src_array);

    void ClearPerturbationResults();
    void ResetGuess(HighPrecision x = HighPrecision(0), HighPrecision y = HighPrecision(0));

private:
    PerturbationAlg m_PerturbationAlg;
    Fractal& m_Fractal;

    HighPrecision m_PerturbationGuessCalcX;
    HighPrecision m_PerturbationGuessCalcY;

    std::vector<std::unique_ptr<PerturbationResults<double>>> m_PerturbationResultsDouble;
    std::vector<std::unique_ptr<PerturbationResults<float>>> m_PerturbationResultsFloat;
    std::vector<std::unique_ptr<PerturbationResults<HDRFloat<double>>>> m_PerturbationResultsHDRDouble;
    std::vector<std::unique_ptr<PerturbationResults<HDRFloat<float>>>> m_PerturbationResultsHDRFloat;
};