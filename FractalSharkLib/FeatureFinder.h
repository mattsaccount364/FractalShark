#pragma once

#include <cstdint>
#include <iosfwd>

#include "FloatComplex.h"
#include "PerturbationResultsHelpers.h"

template <typename IterType, class T, PerturbExtras PExtras> class PerturbationResults;
template <typename IterType, class T, PerturbExtras PExtras> class RuntimeDecompressor;

class FeatureSummary;

template <class IterType, class T, PerturbExtras PExtras>
class FeatureFinder final : public TemplateHelpers<IterType, T, PExtras> {
public:
    using TemplateHelpers = TemplateHelpers<IterType, T, PExtras>;
    using SubType = TemplateHelpers::SubType;
    using C = TemplateHelpers::template HDRFloatComplex<SubType>;

    using IterTypeFull = uint64_t;

    struct Params {
        Params()
            : MaxNewtonIters(32), RelStepTol{0x1p-40}, // 2^-40
              RelStepTol2{0x1p-80},                    // 2^-80
              Eps2Accept{}, DampMin{0.1}, DampMax{1.0}, PrintResult(true)
        {
            HdrReduce(RelStepTol);
            HdrReduce(RelStepTol2);
            HdrReduce(DampMin);
            HdrReduce(DampMax);
        }

        uint32_t MaxNewtonIters;

        // Match original: Tolerance = 2^-40, compare squared norms
        T RelStepTol;  // 2^-40
        T RelStepTol2; // 2^-80  (or compute RelStepTol*RelStepTol)

        // Optional: keep residual accept as a *secondary* early-out (can be 0 to disable)
        T Eps2Accept;

        T DampMin;
        T DampMax;
        bool PrintResult;
    };

    explicit FeatureFinder(const Params &p = Params{}) : m_params(p) {}

    bool FindPeriodicPoint(IterType iters, FeatureSummary &feature) const;

    bool FindPeriodicPoint(const PerturbationResults<IterType, T, PExtras> &results,
                           RuntimeDecompressor<IterType, T, PExtras> &dec,
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

    T ChebAbs(const C &a) const;

    bool Evaluate_PeriodResidualAndDzdc_PT(const PerturbationResults<IterType, T, PExtras> &results,
                                           RuntimeDecompressor<IterType, T, PExtras> &dec,
                                           const C &cAbs, // absolute candidate c
                                           IterType period,
                                           C &outDiff,   // z_p(c)  (or delta vs z0 if you prefer)
                                           C &outDzdc,   // dz_p/dc
                                           C &outZcoeff, // product(2*z_k) in your convention
                                           T &outResidual2) const;

    bool Evaluate_FindPeriod_PT(const PerturbationResults<IterType, T, PExtras> &results,
                                RuntimeDecompressor<IterType, T, PExtras> &dec,
                                const C &cAbs,              // absolute candidate c
                                IterTypeFull maxItersToTry, // how far to search for a period
                                T R,                   // radius for near-linear trigger
                                IterType &outPeriod,
                                EvalState &st) const;

    bool Evaluate_FindPeriod_Direct(
        const C &c, IterTypeFull maxIters, T R, IterType &outPeriod, EvalState &st) const;

    bool Evaluate_PeriodResidualAndDzdc_Direct(
        const C &c, IterType period, C &outDiff, C &outDzdc, C &outZcoeff, T &outResidual2) const;

    static C Div(const C &a, const C &b);

    bool Evaluate_AtPeriod(const PerturbationResults<IterType, T, PExtras> &results,
                           RuntimeDecompressor<IterType, T, PExtras> &dec,
                           const C &c,
                           IterType period,
                           EvalState &st,
                           T &outResidual2) const;

    static T ToDouble(const HighPrecision &v);

private:
    Params m_params{};
};
