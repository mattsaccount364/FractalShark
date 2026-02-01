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

    bool FindPeriodicPoint(IterType maxIters, FeatureSummary &feature) const;

    bool FindPeriodicPoint(IterType maxIters,
                           const PerturbationResults<IterType, T, PExtras> &results,
                           RuntimeDecompressor<IterType, T, PExtras> &dec,
                           FeatureSummary &feature) const;

private:
    struct EvalState {
        C z{};
    };

    // Evaluator policy for direct iteration (no perturbation)
    struct DirectEvaluator {
        const FeatureFinder *self;

        template <bool FindPeriod>
        bool Eval(const C &c,
                  const HighPrecision &cX_hp,
                  const HighPrecision &cY_hp,
                  T SqrRadius,
                  IterTypeFull maxIters,
                  IterType &ioPeriod,
                  C &outDiff,
                  C &outDzdc,
                  C &outZcoeff,
                  T &outResidual2) const;
    };

    // Evaluator policy for perturbation theory iteration
    struct PTEvaluator {
        const FeatureFinder *self;
        const PerturbationResults<IterType, T, PExtras> *results;
        RuntimeDecompressor<IterType, T, PExtras> *dec;

        template <bool FindPeriod>
        bool Eval(const C &c,
                  const HighPrecision &cX_hp,
                  const HighPrecision &cY_hp,
                  T SqrRadius,
                  IterTypeFull maxIters,
                  IterType &ioPeriod,
                  C &outDiff,
                  C &outDzdc,
                  C &outZcoeff,
                  T &outResidual2) const;
    };

    template <class EvalPolicy>
    bool FindPeriodicPoint_Common(IterType refIters,
                                  FeatureSummary &feature,
                                  EvalPolicy &&evaluator) const;

    T ChebAbs(const C &a) const;

    template <bool FindPeriod>
    bool Evaluate_PT(
        const PerturbationResults<IterType, T, PExtras> &results,
        RuntimeDecompressor<IterType, T, PExtras> &dec,
        const C &origC,
        const HighPrecision &origCX_hp, // High precision original X
        const HighPrecision &origCY_hp, // High precision original Y
        T R,
        IterTypeFull maxIters,
        IterType &ioPeriod,
        C &outDiff,
        C &outDzdc,
        C &outZcoeff,
        T &outResidual2) const;


    bool Evaluate_FindPeriod_Direct(const C &c,
                                    IterTypeFull maxIters,
                                    T R,
                                    IterType &outPeriod,
                                    C &outDiff,
                                    C &outDzdc,
                                    C &outZcoeff,
                                    T &outResidual2) const;

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