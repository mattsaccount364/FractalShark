#pragma once

#include <cstdint>
#include <iosfwd>

#include "FloatComplex.h"

template <typename IterType, class T, PerturbExtras PExtras> class PerturbationResults;
template <typename IterType, class T, PerturbExtras PExtras> class RuntimeDecompressor;

class FeatureSummary;

template <class IterType, class T, PerturbExtras Extras> class FeatureFinder final {
public:
    using IterTypeFull = uint64_t;
    using C = FloatComplex<double>;

    struct Params {
        uint32_t MaxNewtonIters = 32;

        // Match original: Tolerance = 2^-40, compare squared norms
        double RelStepTol = 0x1p-40;  // 2^-40
        double RelStepTol2 = 0x1p-80; // 2^-80  (or compute RelStepTol*RelStepTol)

        // Optional: keep residual accept as a *secondary* early-out (can be 0 to disable)
        double Eps2Accept = 0.0;

        double DampMin = 0.1;
        double DampMax = 1.0;
        bool PrintResult = true;
    };


    explicit FeatureFinder(const Params &p = Params{}) : m_params(p) {}

    bool FindPeriodicPoint(IterType iters, FeatureSummary &feature) const;

    bool FindPeriodicPoint(const PerturbationResults<IterType, T, Extras> &results,
                           RuntimeDecompressor<IterType, T, Extras> &dec,
                           FeatureSummary &feature) const;

private:
    template <class FindPeriodFn, class EvalFn>
    bool FindPeriodicPoint_Common(IterType refIters,
                                  FeatureSummary &feature,
                                  FindPeriodFn &&findPeriod,
                                  EvalFn &&evalAtPeriod) const;
    struct EvalState {
        C z{};
    };

private:
    bool Evaluate_PeriodResidualAndDzdc_PT(const PerturbationResults<IterType, T, Extras> &results,
                                           RuntimeDecompressor<IterType, T, Extras> &dec,
                                           const C &cAbs, // absolute candidate c
                                           IterType period,
                                           C &outDiff,   // z_p(c)  (or delta vs z0 if you prefer)
                                           C &outDzdc,   // dz_p/dc
                                           C &outZcoeff, // product(2*z_k) in your convention
                                           double &outResidual2) const;

    bool Evaluate_FindPeriod_PT(const PerturbationResults<IterType, T, Extras> &results,
                                RuntimeDecompressor<IterType, T, Extras> &dec,
                                const C &cAbs,              // absolute candidate c
                                IterTypeFull maxItersToTry, // how far to search for a period
                                double R,                   // radius for near-linear trigger
                                IterType &outPeriod,
                                EvalState &st) const;

    bool Evaluate_FindPeriod_Direct(
        const C &c, IterTypeFull maxIters, double R, IterType &outPeriod, EvalState &st) const;

    bool Evaluate_PeriodResidualAndDzdc_Direct(
        const C &c, IterType period, C &outDiff, C &outDzdc, C &outZcoeff, double &outResidual2) const;

    static C Div(const C &a, const C &b); // a/b for FloatComplex<double>

    bool Evaluate_AtPeriod(const PerturbationResults<IterType, T, Extras> &results,
                           RuntimeDecompressor<IterType, T, Extras> &dec,
                           const C &c,
                           IterType period,
                           EvalState &st,
                           double &outResidual2) const;

    static double ToDouble(const HighPrecision &v);

private:
    Params m_params{};
};
