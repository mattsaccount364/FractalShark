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
        IterType MaxPeriodToTry = 4096;
        uint32_t MaxNewtonIters = 32;
        double Eps2Accept = 1e-24; // accept when |F|^2 <= eps^2
        double DampMin = 0.1;
        double DampMax = 1.0;
        bool PrintResult = true;
    };

    explicit FeatureFinder(const Params &p = Params{}) : m_params(p) {}

    bool FindPeriodicPoint(const PerturbationResults<IterType, T, Extras> &results,
                           RuntimeDecompressor<IterType, T, Extras> &dec,
                           FeatureSummary &feature) const;

private:
    struct EvalState {
        C z{};
    };

private:
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

    static C
    MakeC(double re, double im)
    {
        return C(re, im);
    }
    static double
    Re(const C &z)
    {
        return static_cast<double>(z.getRe());
    }
    static double
    Im(const C &z)
    {
        return static_cast<double>(z.getIm());
    }
    static double
    Norm2(const C &z)
    {
        return static_cast<double>(z.norm_squared());
    }

    static double ToDouble(const HighPrecision &v);

private:
    Params m_params{};
};
