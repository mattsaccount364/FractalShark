// PeriodicPointFinder.cpp  (or keep in FeatureFinder.cpp if you prefer)
//
// NOTE: This is written to use *only* FloatComplex<double> everywhere.
// No std::complex, no DComplex, no HDR complex types.

#include "stdafx.h"
#include "CudaDblflt.h"
#include "dblflt.h"
#include "FloatComplex.h"
#include "HighPrecision.h"
#include "FeatureFinder.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>

using C = FloatComplex<double>; // <--- ONE complex type everywhere (per your request)

// Choose a cheap norm proxy (Chebyshev / max-abs) to avoid overflow in squares.
static inline double
ChebAbs(const C &a)
{
    return std::max(std::abs(a.getRe()), std::abs(a.getIm()));
}

// Renormalize dzdc/zcoeff so their magnitude stays ~ 2^targetExp.
// This preserves the invariant: true_dzdc = dzdc * 2^scaleExp
// and forces the recurrence to add scalingFactor = 2^-scaleExp instead of 1.
static inline void
RenormalizeDzdcZcoeff(C &dzdc, C &zcoeff, int &scaleExp)
{
    // Pick thresholds with lots of headroom before double overflow (~2^1024).
    constexpr int kHiExp = 700;     // if exponent exceeds this, renormalize
    constexpr int kTargetExp = 200; // bring it back near this exponent

    const double m = std::max(ChebAbs(dzdc), ChebAbs(zcoeff));
    if (!(m > 0.0) || !std::isfinite(m)) {
        return; // caller will fail on non-finite elsewhere
    }

    // ilogb gives floor(log2(|m|)) as an int, without computing log().
    const int e = std::ilogb(m);
    if (e <= kHiExp) {
        return;
    }

    const int shift = e - kTargetExp;            // shift > 0
    const double down = std::ldexp(1.0, -shift); // 2^-shift

    dzdc = dzdc * down;
    zcoeff = zcoeff * down;
    scaleExp += shift;
}

// Convert scaled values back to true values.
static inline C
Unscale(const C &a, int scaleExp)
{
    return a * std::ldexp(1.0, scaleExp); // a * 2^scaleExp
}


template <class IterType, class T, PerturbExtras Extras>
double
PeriodicPointFinder<IterType, T, Extras>::ToDouble(const HighPrecision &v)
{
    return v.operator double();
}

// PeriodicPointFinder.cpp  (revised / related logic)
// NOTE: FloatComplex<double> only.

template <class IterType, class T, PerturbExtras Extras>
typename PeriodicPointFinder<IterType, T, Extras>::C
PeriodicPointFinder<IterType, T, Extras>::Div(const C &a, const C &b)
{
    // a / b = a * conj(b) / |b|^2
    const double br = Re(b);
    const double bi = Im(b);
    const double denom = br * br + bi * bi;

    // If denom is 0, caller is hosed; keep it explicit.
    if (!(denom > 0.0)) {
        return C(0.0, 0.0);
    }

    // a * conj(b)
    const double ar = Re(a);
    const double ai = Im(a);
    const double rr = (ar * br + ai * bi) / denom;
    const double ii = (ai * br - ar * bi) / denom;
    return C(rr, ii);
}

template <class IterType, class T, PerturbExtras Extras>
bool
PeriodicPointFinder<IterType, T, Extras>::Evaluate_FindPeriod_Direct(
    const C &c, IterTypeFull maxIters, double R, IterType &outPeriod, EvalState &st) const
{
    if (!(R > 0.0))
        return false;
    const double R2 = R * R;

    C z(0.0, 0.0);
    C dzdc(0.0, 0.0);
    C zcoeff(0.0, 0.0);

    int scaleExp = 0; // true_dzdc = dzdc * 2^scaleExp

    for (IterTypeFull n = 0; n < maxIters; ++n) {
        const double scalingFactor = std::ldexp(1.0, -scaleExp); // 2^-scaleExp

        // zcoeff ordering matches your reference:
        if (n == 0)
            zcoeff = C(scalingFactor, 0.0);
        else
            zcoeff = zcoeff * (z * 2.0);

        // dzdc <- 2*z*dzdc + scalingFactor  (instead of +1)
        dzdc = dzdc * (z * 2.0) + C(scalingFactor, 0.0);

        // Keep dzdc/zcoeff from blowing up; updates scaleExp.
        RenormalizeDzdcZcoeff(dzdc, zcoeff, scaleExp);

        // Advance orbit
        z = (z * z) + c;

        if (Norm2(z) > 4096.0)
            break;

        // Period trigger uses TRUE dzdc (unscaled)
        const double magZ = Norm2(z);
        const C dzdcTrue = Unscale(dzdc, scaleExp);
        const double magD = Norm2(dzdcTrue);

        if (magZ < (R2 * magD)) {
            const IterTypeFull cand = n + 1;
            if (cand <= static_cast<IterTypeFull>(std::numeric_limits<IterType>::max())) {
                outPeriod = static_cast<IterType>(cand);
                st.z = z;
                return true;
            }
            return false;
        }

        // Optional hard fail on NaNs/Infs
        if (!std::isfinite(Re(z)) || !std::isfinite(Im(z)) || !std::isfinite(Re(dzdc)) ||
            !std::isfinite(Im(dzdc))) {
            return false;
        }
    }

    return false;
}


template <class IterType, class T, PerturbExtras Extras>
bool
PeriodicPointFinder<IterType, T, Extras>::Evaluate_PeriodResidualAndDzdc_Direct(
    const C &c, IterType period, C &outDiff, C &outDzdc, C &outZcoeff, double &outResidual2) const
{
    C z(0.0, 0.0);
    C dzdc(0.0, 0.0);
    C zcoeff(0.0, 0.0);

    int scaleExp = 0;

    for (IterType i = 0; i < period; ++i) {
        const double scalingFactor = std::ldexp(1.0, -scaleExp);

        if (i == 0)
            zcoeff = C(scalingFactor, 0.0);
        else
            zcoeff = zcoeff * (z * 2.0);

        dzdc = dzdc * (z * 2.0) + C(scalingFactor, 0.0);

        RenormalizeDzdcZcoeff(dzdc, zcoeff, scaleExp);

        z = (z * z) + c;

        if (Norm2(z) > 4096.0)
            return false;

        if (!std::isfinite(Re(z)) || !std::isfinite(Im(z)) || !std::isfinite(Re(dzdc)) ||
            !std::isfinite(Im(dzdc))) {
            return false;
        }
    }

    outDiff = z;
    outResidual2 = Norm2(outDiff);

    // Unscale derivatives back to true values
    outDzdc = Unscale(dzdc, scaleExp);
    outZcoeff = Unscale(zcoeff, scaleExp);

    return true;
}


template <class IterType, class T, PerturbExtras Extras>
bool
PeriodicPointFinder<IterType, T, Extras>::FindPeriodicPoint(
    const PerturbationResults<IterType, T, Extras> & /*results*/,
    RuntimeDecompressor<IterType, T, Extras> & /*dec*/,
    const HighPrecision &seedX,
    const HighPrecision &seedY,
    const HighPrecision &radius,
    PeriodicPointFeature<IterType> &outFeature) const
{
    // This is now aligned with your FeatureFinder approach:
    //  1) Find a period candidate using dzdc-based near-linear criterion
    //  2) Newton: c <- c - diff/dzdc until convergence
    //
    // Using FloatComplex<double> only.

    const double cx0 = ToDouble(seedX);
    const double cy0 = ToDouble(seedY);
    const double R = std::abs(ToDouble(radius)); // caller-supplied neighborhood scale

    C origC = MakeC(cx0, cy0);
    C c = origC;

    EvalState st{};
    IterType period = 0;

    if (!Evaluate_FindPeriod_Direct(
            c, static_cast<IterTypeFull>(m_params.MaxPeriodToTry), R, period, st)) {
        return false;
    }

    // Now do Newton iterations at the *fixed period*:
    // F(c) = z_p(c) (starting z0=0)
    // F'(c) = dz_p/dc
    //
    // step = F / F'
    // c <- c - step

    // Convergence controls:
    // - Absolute residual test: outResidual2 <= Eps2Accept (but note: your default is likely too strict)
    // - Relative step test: |step|^2 <= |c|^2 * StepRelTol2
    //
    // Use a sane default relative step tolerance equivalent to your old 2^-40.
    constexpr double kTol = 0x1p-40; // 2^-40
    constexpr double kTol2 = kTol * kTol;

    C diff, dzdc, zcoeff;
    double residual2 = 0.0;

    // First evaluation at seed (for logging / initial correction).
    if (!Evaluate_PeriodResidualAndDzdc_Direct(c, period, diff, dzdc, zcoeff, residual2)) {
        return false;
    }

    // If derivative is degenerate, bail.
    if (!(Norm2(dzdc) > 0.0)) {
        return false;
    }

    // Initial correction (matches your code: dc -= diff/dzdc)
    {
        const C step = Div(diff, dzdc);
        c = c - step;
    }

    // Newton loop
    for (uint32_t it = 0; it < m_params.MaxNewtonIters; ++it) {
        if (!Evaluate_PeriodResidualAndDzdc_Direct(c, period, diff, dzdc, zcoeff, residual2)) {
            return false;
        }

        if (residual2 <= m_params.Eps2Accept) {
            break;
        }

        const double dzdc2 = Norm2(dzdc);
        if (!(dzdc2 > 0.0)) {
            return false;
        }

        C step = Div(diff, dzdc);

        // Optional damping scaffold (same idea as your params):
        // c <- c - damp*step
        double damp = m_params.DampMax;
        damp = std::clamp(damp, m_params.DampMin, m_params.DampMax);
        step = step * damp;

        c = c - step;

        // Relative step stop like your FeatureFinder:
        // if |step|^2 < |c|^2 * tol^2 => done
        const double step2 = Norm2(step);
        const double c2 = Norm2(c);
        if (step2 <= c2 * kTol2) {
            break;
        }
    }

    // Final evaluate for outputs
    if (!Evaluate_PeriodResidualAndDzdc_Direct(c, period, diff, dzdc, zcoeff, residual2)) {
        return false;
    }

    outFeature.Period = period;
    outFeature.Residual2 = residual2;

    // If you want X/Y filled, do it however your HighPrecision is constructed in your project.
    // (You can also keep them unset until you wire the correct constructors.)
    // outFeature.X = HighPrecision(Re(c));
    // outFeature.Y = HighPrecision(Im(c));

    //if (m_params.PrintResult) {
        std::cout << "Periodic point (direct): "
                  << "orig cx=" << Re(origC) << " orig cy=" << Im(origC)
                  << "new cx=" << Re(c) << " new cy=" << Im(c) << " period=" << static_cast<uint64_t>(period)
                  << " residual2=" << residual2 << "\n";
    //}

    return (residual2 <= m_params.Eps2Accept);
}





template <class IterType, class T, PerturbExtras Extras>
bool
PeriodicPointFinder<IterType, T, Extras>::Evaluate_AtPeriod(
    const PerturbationResults<IterType, T, Extras> & /*results*/,
    RuntimeDecompressor<IterType, T, Extras> &dec,
    const C &c,
    IterType period,
    EvalState &st,
    double &outResidual2) const
{
    // Re-evaluate orbit up to "period" and measure how close we are to z0.
    C z(0.0, 0.0);
    const C z0 = z;

    for (IterType i = 0; i < period; ++i) {
        z = (z * z) + c;
        (void)dec;
    }

    const C delta = z - z0;
    outResidual2 = Norm2(delta);
    st.z = z;
    return true;
}

// ------------------------------
// Explicit instantiations (minimal set you enabled)
// ------------------------------
#define InstantiatePeriodicPointFinder(IterTypeT, TT, PExtrasT)                                         \
    template struct PeriodicPointFeature<IterTypeT>;                                                    \
    template class PeriodicPointFinder<IterTypeT, TT, PExtrasT>;

//// ---- Disable ----
InstantiatePeriodicPointFinder(uint32_t, double, PerturbExtras::Disable);
InstantiatePeriodicPointFinder(uint64_t, double, PerturbExtras::Disable);
//
//InstantiatePeriodicPointFinder(uint32_t, float, PerturbExtras::Disable);
//InstantiatePeriodicPointFinder(uint64_t, float, PerturbExtras::Disable);
//
//InstantiatePeriodicPointFinder(uint32_t, CudaDblflt<MattDblflt>, PerturbExtras::Disable);
//InstantiatePeriodicPointFinder(uint64_t, CudaDblflt<MattDblflt>, PerturbExtras::Disable);
//
//InstantiatePeriodicPointFinder(uint32_t, HDRFloat<double>, PerturbExtras::Disable);
//InstantiatePeriodicPointFinder(uint64_t, HDRFloat<double>, PerturbExtras::Disable);
//
//InstantiatePeriodicPointFinder(uint32_t, HDRFloat<float>, PerturbExtras::Disable);
//InstantiatePeriodicPointFinder(uint64_t, HDRFloat<float>, PerturbExtras::Disable);
//
//InstantiatePeriodicPointFinder(uint32_t, HDRFloat<CudaDblflt<MattDblflt>>, PerturbExtras::Disable);
//InstantiatePeriodicPointFinder(uint64_t, HDRFloat<CudaDblflt<MattDblflt>>, PerturbExtras::Disable);
//
//// ---- Bad ----
//InstantiatePeriodicPointFinder(uint32_t, double, PerturbExtras::Bad);
//InstantiatePeriodicPointFinder(uint64_t, double, PerturbExtras::Bad);
//
//InstantiatePeriodicPointFinder(uint32_t, float, PerturbExtras::Bad);
//InstantiatePeriodicPointFinder(uint64_t, float, PerturbExtras::Bad);
//
//InstantiatePeriodicPointFinder(uint32_t, CudaDblflt<MattDblflt>, PerturbExtras::Bad);
//InstantiatePeriodicPointFinder(uint64_t, CudaDblflt<MattDblflt>, PerturbExtras::Bad);
//
//InstantiatePeriodicPointFinder(uint32_t, HDRFloat<double>, PerturbExtras::Bad);
//InstantiatePeriodicPointFinder(uint64_t, HDRFloat<double>, PerturbExtras::Bad);
//
//InstantiatePeriodicPointFinder(uint32_t, HDRFloat<float>, PerturbExtras::Bad);
//InstantiatePeriodicPointFinder(uint64_t, HDRFloat<float>, PerturbExtras::Bad);
//
//InstantiatePeriodicPointFinder(uint32_t, HDRFloat<CudaDblflt<MattDblflt>>, PerturbExtras::Bad);
//InstantiatePeriodicPointFinder(uint64_t, HDRFloat<CudaDblflt<MattDblflt>>, PerturbExtras::Bad);
//
//// ---- SimpleCompression ----
//InstantiatePeriodicPointFinder(uint32_t, double, PerturbExtras::SimpleCompression);
//InstantiatePeriodicPointFinder(uint64_t, double, PerturbExtras::SimpleCompression);
//
//InstantiatePeriodicPointFinder(uint32_t, float, PerturbExtras::SimpleCompression);
//InstantiatePeriodicPointFinder(uint64_t, float, PerturbExtras::SimpleCompression);
//
//InstantiatePeriodicPointFinder(uint32_t, CudaDblflt<MattDblflt>, PerturbExtras::SimpleCompression);
//InstantiatePeriodicPointFinder(uint64_t, CudaDblflt<MattDblflt>, PerturbExtras::SimpleCompression);
//
//InstantiatePeriodicPointFinder(uint32_t, HDRFloat<double>, PerturbExtras::SimpleCompression);
//InstantiatePeriodicPointFinder(uint64_t, HDRFloat<double>, PerturbExtras::SimpleCompression);
//
//InstantiatePeriodicPointFinder(uint32_t, HDRFloat<float>, PerturbExtras::SimpleCompression);
//InstantiatePeriodicPointFinder(uint64_t, HDRFloat<float>, PerturbExtras::SimpleCompression);
//
//InstantiatePeriodicPointFinder(uint32_t,
//                               HDRFloat<CudaDblflt<MattDblflt>>,
//                               PerturbExtras::SimpleCompression);
//InstantiatePeriodicPointFinder(uint64_t,
//                               HDRFloat<CudaDblflt<MattDblflt>>,
//                               PerturbExtras::SimpleCompression);

#undef InstantiatePeriodicPointFinder