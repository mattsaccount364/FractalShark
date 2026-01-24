// PeriodicPointFinder.h
#pragma once

#include <cstdint>
#include <iosfwd>

#include "FloatComplex.h" // your FloatComplex<SubType>
#include "PerturbationResults.h"

template <class IterType> struct PeriodicPointFeature {
    HighPrecision X;
    HighPrecision Y;
    IterType Period{};
    double Residual2{}; // squared residual at acceptance (double for quick logging/debug)
};


template <class IterType, class T, PerturbExtras Extras> class PeriodicPointFinder final {
public:
    using IterTypeFull = uint64_t;
    using C = FloatComplex<double>; // <--- ONE complex type everywhere (per your request)

    struct Params {
        IterType MaxPeriodToTry = 4096;
        uint32_t MaxNewtonIters = 32;
        double Eps2Accept = 1e-24; // accept when |F|^2 <= eps^2
        double DampMin = 0.1;
        double DampMax = 1.0;
        bool PrintResult = true;
    };

    explicit PeriodicPointFinder(const Params &p = Params{}) : m_params(p) {}

    // Finds a periodic point near (seedX, seedY). The meaning of radius is up to your caller;
    // commonly it bounds the search neighborhood or sets a numeric scale for steps.
    bool FindPeriodicPoint(const PerturbationResults<IterType, T, Extras> &results,
                           RuntimeDecompressor<IterType, T, Extras> &dec,
                           const HighPrecision &seedX,
                           const HighPrecision &seedY,
                           const HighPrecision &radius,
                           PeriodicPointFeature<IterType> &outFeature) const;

private:
    struct EvalState {
        // You can stash per-evaluation scratch here if you want to avoid re-allocations.
        // Keeping it explicit also makes it easy to checksum/debug.
        C z{};
    };

private:
    // Direct / absolute evaluation (no decompressor, no perturbation):
    //  - If FindPeriod==true: run up to maxIters and trigger the same kind of
    //    dzdc-based "near-linear" criterion to *suggest* a period candidate.
    //  - If FindPeriod==false: run exactly "period" iterations and return diff/dzdc/zcoeff.
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

    // Helpers
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

    // You almost certainly have a real conversion; wire these to your existing code.
    static double ToDouble(const HighPrecision &v);

private:
    Params m_params{};
};
