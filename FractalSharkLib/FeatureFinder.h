#pragma once

#include <cstdint>
#include <iosfwd>

#include "FeatureFinderMode.h"
#include "FloatComplex.h"
#include "PerturbationResultsHelpers.h"

template <typename IterType, class T, PerturbExtras PExtras> class PerturbationResults;
template <typename IterType, class T, PerturbExtras PExtras> class RuntimeDecompressor;
template <typename IterType, class Float, class SubType, PerturbExtras PExtras> class LAReference;

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
              Eps2Accept{}, PrintResult(true)
        {
            HdrReduce(RelStepTol);
            HdrReduce(RelStepTol2);
        }

        uint32_t MaxNewtonIters;

        T RelStepTol;  // 2^-40
        T RelStepTol2; // 2^-80

        // Optional: keep residual accept as a *secondary* early-out (can be 0 to disable)
        T Eps2Accept;

        bool PrintResult;
    };

    explicit FeatureFinder(const Params &p = Params{}) : m_params(p) {}

    bool FindPeriodicPoint(IterType maxIters, FeatureSummary &feature) const;

    bool FindPeriodicPoint(IterType maxIters,
                           const PerturbationResults<IterType, T, PExtras> &results,
                           RuntimeDecompressor<IterType, T, PExtras> &dec,
                           FeatureSummary &feature) const;

    bool FindPeriodicPoint(IterType maxIters,
                           const PerturbationResults<IterType, T, PExtras> &results,
                           RuntimeDecompressor<IterType, T, PExtras> &dec,
                           LAReference<IterType, T, SubType, PExtras> &laRef,
                           FeatureSummary &feature) const;

    bool RefinePeriodicPoint_HighPrecision(FeatureSummary &feature) const;

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

    // NEW: Evaluator policy for Linear Approximation iteration
    struct LAEvaluator {
        const FeatureFinder *self;
        const PerturbationResults<IterType, T, PExtras> *results;
        RuntimeDecompressor<IterType, T, PExtras> *dec;
        LAReference<IterType, T, SubType, PExtras> *laRef;

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

    // NEW: LA evaluation method
    template <bool FindPeriod>
    bool Evaluate_LA(const PerturbationResults<IterType, T, PExtras> &results,
                     LAReference<IterType, T, SubType, PExtras> &laRef,
                     const HighPrecision &cX_hp,
                     const HighPrecision &cY_hp,
                     T R,
                     IterTypeFull maxIters,
                     IterType &ioPeriod,
                     C &outDiff,
                     C &outDzdc,
                     C &outZcoeff,
                     T &outResidual2) const;

    template <bool FindPeriod>
    bool Evaluate_PT(const PerturbationResults<IterType, T, PExtras> &results,
                     RuntimeDecompressor<IterType, T, PExtras> &dec,
                     const HighPrecision &cX_hp, // <-- CURRENT Newton iterate
                     const HighPrecision &cY_hp, // <-- CURRENT Newton iterate
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

    HighPrecision ComputeIntrinsicRadius_HP(const C &zcoeff, const C &dzdc) const;

    static C Div(const C &a, const C &b);

    bool Evaluate_AtPeriod(const PerturbationResults<IterType, T, PExtras> &results,
                           RuntimeDecompressor<IterType, T, PExtras> &dec,
                           const C &c,
                           IterType period,
                           EvalState &st,
                           T &outResidual2) const;

    static T ToDouble(const HighPrecision &v);

    IterType RefinePeriodicPoint_WithMPF(HighPrecision &cX_hp,
                                         HighPrecision &cY_hp,
                                         IterType period,
                                         mp_bitcnt_t coord_prec,
                                         const T &sqrRadius_T,
                                         int scaleExp2_for_deriv) const;

private:
    Params m_params{};
};
