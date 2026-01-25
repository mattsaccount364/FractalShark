#include "stdafx.h"

#include "FeatureFinder.h"
#include "FeatureSummary.h"
#include "FloatComplex.h"
#include "HighPrecision.h"
#include "PerturbationResults.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>

using C = FloatComplex<double>;

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
FeatureFinder<IterType, T, Extras>::ToDouble(const HighPrecision &v)
{
    return v.operator double();
}

template <class IterType, class T, PerturbExtras Extras>
typename FeatureFinder<IterType, T, Extras>::C
FeatureFinder<IterType, T, Extras>::Div(const C &a, const C &b)
{
    // a / b = a * conj(b) / |b|^2
    const double br = b.getRe();
    const double bi = b.getIm();
    const double denom = br * br + bi * bi;

    // If denom is 0, caller is hosed; keep it explicit.
    if (!(denom > 0.0)) {
        return C(0.0, 0.0);
    }

    // a * conj(b)
    const double ar = a.getRe();
    const double ai = a.getIm();
    const double rr = (ar * br + ai * bi) / denom;
    const double ii = (ai * br - ar * bi) / denom;
    return C(rr, ii);
}

template <class IterType, class T, PerturbExtras Extras>
bool
FeatureFinder<IterType, T, Extras>::Evaluate_FindPeriod_Direct(
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

        if (z.norm_squared() > 4096.0)
            break;

        // Period trigger uses TRUE dzdc (unscaled)
        const double magZ = z.norm_squared();
        const C dzdcTrue = Unscale(dzdc, scaleExp);
        const double magD = dzdcTrue.norm_squared();

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
        if (!std::isfinite(z.getRe()) || !std::isfinite(z.getIm()) || !std::isfinite(dzdc.getRe()) ||
            !std::isfinite(dzdc.getIm())) {
            return false;
        }
    }

    return false;
}

template <class IterType, class T, PerturbExtras Extras>
bool
FeatureFinder<IterType, T, Extras>::Evaluate_PeriodResidualAndDzdc_Direct(
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

        if (z.norm_squared() > 4096.0)
            return false;

        if (!std::isfinite(z.getRe()) || !std::isfinite(z.getIm()) || !std::isfinite(dzdc.getRe()) ||
            !std::isfinite(dzdc.getIm())) {
            return false;
        }
    }

    outDiff = z;
    outResidual2 = outDiff.norm_squared();

    // Unscale derivatives back to true values
    outDzdc = Unscale(dzdc, scaleExp);
    outZcoeff = Unscale(zcoeff, scaleExp);

    return true;
}

// =====================================================================================
// PT evaluation using PerturbationResults + RuntimeDecompressor
// (FloatComplex<double> only, like your new direct implementation)
// =====================================================================================

// --- helpers -------------------------------------------------------------

template <class IterType, class T, PerturbExtras Extras>
static inline C
ToC_fromRefOrbitPoint(const PerturbationResults<IterType, T, Extras> &results,
                      RuntimeDecompressor<IterType, T, Extras> &dec,
                      size_t idx)
{
    // For double / float orbit types this will be exact.
    // For HDRFloat / CudaDblflt variants, pick a SubType (double) and cast.
    //
    // Your PerturbationResults::GetComplex returns HDRFloatComplex<SubType>.
    // That type in your codebase typically has .x/.y or .real()/.imag().
    // Adjust member names if needed.

    auto v = results.GetComplex(dec, idx);

    // Common patterns in your code:
    //   return {m_FullOrbit[i].x, m_FullOrbit[i].y};
    // so v likely has .x and .y (or .real/.imag).
    //
    // If it's .x/.y:
    const double xr = static_cast<double>(v.getRe());
    const double yi = static_cast<double>(v.getIm());

    return C(xr, yi);
}

template <class IterType, class T, PerturbExtras Extras>
static inline C
ToC_fromReferenceC(const PerturbationResults<IterType, T, Extras> &results)
{
    // reference parameter c_ref that the authoritative orbit was generated at
    // (low precision fields are fine for PT in double; you can swap to hi if desired).
    const double cx = static_cast<double>(results.GetOrbitXLow());
    const double cy = static_cast<double>(results.GetOrbitYLow());
    return C(cx, cy);
}

// =====================================================================================
// 1) Find period candidate using PT (reference orbit)
// =====================================================================================
template <class IterType, class T, PerturbExtras Extras>
bool
FeatureFinder<IterType, T, Extras>::Evaluate_FindPeriod_PT(
    const PerturbationResults<IterType, T, Extras> &results,
    RuntimeDecompressor<IterType, T, Extras> &dec,
    const C &cAbs,              // absolute candidate c
    IterTypeFull maxItersToTry, // how far to search for a period
    double R,                   // radius for near-linear trigger
    IterType &outPeriod,
    EvalState &st) const
{
    if (!(R > 0.0))
        return false;
    const double R2 = R * R;

    const C cRef = ToC_fromReferenceC(results);
    const C dc = cAbs - cRef; // dc in perturbation formula

    // Guard: need reference orbit long enough.
    const size_t refCount = static_cast<size_t>(results.GetCountOrbitEntries());
    if (refCount < 2)
        return false;

    C dz(0.0, 0.0); // perturbation delta (your old "dz")
    C z(0.0, 0.0);  // full z
    C dzdc(0.0, 0.0);
    C zcoeff(0.0, 0.0);

    int scaleExp = 0; // true_dzdc = dzdc * 2^scaleExp (and same for zcoeff)

    // We need z_ref[n] each step. Assume results stores z_ref[0]=0 and subsequent.
    // We iterate n from 0..N-1, using z_ref[n] to advance delta to n+1, and then form z at n+1.
    const IterTypeFull N =
        std::min<IterTypeFull>(maxItersToTry, static_cast<IterTypeFull>(refCount - 1));

    for (IterTypeFull n = 0; n < N; ++n) {
        const double scalingFactor = std::ldexp(1.0, -scaleExp);

        // Full z_n = z_ref[n] + dz_n
        const C zref = ToC_fromRefOrbitPoint(results, dec, static_cast<size_t>(n));
        z = zref + dz;

        // Derivative recurrences match your reference ordering:
        if (n == 0)
            zcoeff = C(scalingFactor, 0.0);
        else
            zcoeff = zcoeff * (z * 2.0);

        dzdc = dzdc * (z * 2.0) + C(scalingFactor, 0.0);

        RenormalizeDzdcZcoeff(dzdc, zcoeff, scaleExp);

        // Perturbation advance:
        // dz_{n+1} = dz_n * (z_ref[n] + z_n) + dc
        //          = dz_n * (2*z_ref[n] + dz_n) + dc
        dz = dz * (zref + z) + dc;

        // Full z_{n+1} for escape / periodic check:
        const C zrefNext = ToC_fromRefOrbitPoint(results, dec, static_cast<size_t>(n + 1));
        const C zNext = zrefNext + dz;

        if (zNext.norm_squared() > 4096.0)
            break;

        // Period trigger uses TRUE dzdc (unscaled)
        const double magZ = zNext.norm_squared();
        const C dzdcTrue = Unscale(dzdc, scaleExp);
        const double magD = dzdcTrue.norm_squared();

        if (magZ < (R2 * magD)) {
            const IterTypeFull cand = n + 1; // matches your direct: period = n+1
            if (cand <= static_cast<IterTypeFull>(std::numeric_limits<IterType>::max())) {
                outPeriod = static_cast<IterType>(cand);
                st.z = zNext;
                return true;
            }
            return false;
        }

        // Optional hard fail on NaNs/Infs
        if (!std::isfinite(zNext.getRe()) || !std::isfinite(zNext.getIm()) ||
            !std::isfinite(dz.getRe()) || !std::isfinite(dz.getIm()) || !std::isfinite(dzdc.getRe()) ||
            !std::isfinite(dzdc.getIm())) {
            return false;
        }
    }

    return false;
}

// =====================================================================================
// 2) Evaluate diff=z_p(c), dzdc, zcoeff at a fixed period using PT
// =====================================================================================
template <class IterType, class T, PerturbExtras Extras>
bool
FeatureFinder<IterType, T, Extras>::Evaluate_PeriodResidualAndDzdc_PT(
    const PerturbationResults<IterType, T, Extras> &results,
    RuntimeDecompressor<IterType, T, Extras> &dec,
    const C &cAbs, // absolute candidate c
    IterType period,
    C &outDiff,   // z_p(c)  (or delta vs z0 if you prefer)
    C &outDzdc,   // dz_p/dc
    C &outZcoeff, // product(2*z_k) in your convention
    double &outResidual2) const
{
    const C cRef = ToC_fromReferenceC(results);
    const C dc = cAbs - cRef;

    const size_t refCount = static_cast<size_t>(results.GetCountOrbitEntries());
    if (refCount < static_cast<size_t>(period) + 1)
        return false;

    C dz(0.0, 0.0);
    C z(0.0, 0.0);
    C dzdc(0.0, 0.0);
    C zcoeff(0.0, 0.0);

    int scaleExp = 0;

    for (IterType i = 0; i < period; ++i) {
        const double scalingFactor = std::ldexp(1.0, -scaleExp);

        const C zref = ToC_fromRefOrbitPoint(results, dec, static_cast<size_t>(i));
        z = zref + dz;

        if (i == 0)
            zcoeff = C(scalingFactor, 0.0);
        else
            zcoeff = zcoeff * (z * 2.0);

        dzdc = dzdc * (z * 2.0) + C(scalingFactor, 0.0);

        RenormalizeDzdcZcoeff(dzdc, zcoeff, scaleExp);

        // perturbation step
        dz = dz * (zref + z) + dc;

        // optional bail-outs
        const C zrefNext = ToC_fromRefOrbitPoint(results, dec, static_cast<size_t>(i + 1));
        const C zNext = zrefNext + dz;
        if (zNext.norm_squared() > 4096.0)
            return false;

        if (!std::isfinite(zNext.getRe()) || !std::isfinite(zNext.getIm()) ||
            !std::isfinite(dz.getRe()) || !std::isfinite(dz.getIm()) || !std::isfinite(dzdc.getRe()) ||
            !std::isfinite(dzdc.getIm())) {
            return false;
        }
    }

    // After "period" steps, full z_p = zref[period] + dz_period.
    const C zrefP = ToC_fromRefOrbitPoint(results, dec, static_cast<size_t>(period));
    const C zP = zrefP + dz;

    outDiff = zP; // matches your direct version (z_p starting from z0=0)
    outResidual2 = outDiff.norm_squared();

    // Unscale derivatives back to true values
    outDzdc = Unscale(dzdc, scaleExp);
    outZcoeff = Unscale(zcoeff, scaleExp);

    return true;
}

template <class IterType, class T, PerturbExtras Extras>
template <class FindPeriodFn, class EvalFn>
bool
FeatureFinder<IterType, T, Extras>::FindPeriodicPoint_Common(IterType refIters,
                                                             FeatureSummary &feature,
                                                             FindPeriodFn &&findPeriod,
                                                             EvalFn &&evalAtPeriod) const
{
    const double cx0 = ToDouble(feature.GetOrigX());
    const double cy0 = ToDouble(feature.GetOrigY());
    const double R = std::abs(ToDouble(feature.GetRadius()));

    const C origC{cx0, cy0};
    C c = origC;

    EvalState st{};
    IterType period = 0;

    // 1) Find candidate period (same role as original: Evaluate<true>)
    if (!findPeriod(origC, refIters, R, period, st)) {
        return false;
    }

    // 2) Newton at fixed period (match original acceptance: relative step size)
    // Original:
    //   Tolerance = 2^-40; SqrTolerance = 2^-80;
    //   if (norm(step) < norm(c) * SqrTolerance) break;
    const double relTol = m_params.RelStepTol; // e.g. 0x1p-40
    const double relTol2 = relTol * relTol;

    C diff, dzdc, zcoeff;
    double residual2 = 0.0;

    // Initial eval + correction (matches original "dc -= diff/dzdc" before loop)
    if (!evalAtPeriod(c, period, diff, dzdc, zcoeff, residual2)) {
        return false;
    }
    const double dzdc2_init = dzdc.norm_squared();
    if (!(dzdc2_init > 0.0)) {
        return false;
    }
    C step0 = Div(diff, dzdc);
    c = c - step0;

    bool convergedByStep = false;

    // Newton loop (original did up to 32 iters, break by relative step size)
    for (uint32_t it = 0; it < m_params.MaxNewtonIters; ++it) {
        if (!evalAtPeriod(c, period, diff, dzdc, zcoeff, residual2)) {
            return false;
        }

        const double dzdc2 = dzdc.norm_squared();
        if (!(dzdc2 > 0.0)) {
            return false;
        }

        C step = Div(diff, dzdc);

        // To match the original behavior, do NOT damp by default.
        // If you want damping, apply it here (but understand it changes behavior).
        // const double damp = std::clamp(m_params.Damp, m_params.DampMin, m_params.DampMax);
        // step = step * damp;

        c = c - step;

        const double step2 = step.norm_squared();
        const double c2 = c.norm_squared();

        // Match original: norm(step) < norm(c) * 2^-80  (squared form)
        if (step2 <= c2 * relTol2) {
            convergedByStep = true;
            break;
        }

        // Optional: keep residual-based early-out only if you explicitly want it.
        // (Original did NOT do this.)
        if (m_params.Eps2Accept > 0.0 && residual2 <= m_params.Eps2Accept) {
            break;
        }
    }

    // Original does one extra "final eval + step" after the loop.
    // Keep that to match behavior closely.
    if (!evalAtPeriod(c, period, diff, dzdc, zcoeff, residual2)) {
        return false;
    }
    const double dzdc2_final = dzdc.norm_squared();
    if (!(dzdc2_final > 0.0)) {
        return false;
    }
    {
        C step = Div(diff, dzdc);
        c = c - step;

        const double step2 = step.norm_squared();
        const double c2 = c.norm_squared();
        if (step2 <= c2 * relTol2) {
            convergedByStep = true;
        }
    }

    // Final eval (so residual2 reflects the returned/printed solution)
    if (!evalAtPeriod(c, period, diff, dzdc, zcoeff, residual2)) {
        return false;
    }

    feature.SetFound(HighPrecision(c.getRe()), HighPrecision(c.getIm()), period, residual2);

    if (m_params.PrintResult) {
        std::cout << "Periodic point: "
                  << "orig cx=" << origC.getRe() << " orig cy=" << origC.getIm()
                  << " new cx=" << c.getRe() << " new cy=" << c.getIm()
                  << " period=" << static_cast<uint64_t>(period) << " residual2=" << residual2 << "\n";
    }

    // MATCH ORIGINAL "success" semantics:
    // success is primarily about Newton converging by step size.
    // (Optionally also accept small residual if you enabled it.)
    if (convergedByStep) {
        return true;
    }
    if (m_params.Eps2Accept > 0.0 && residual2 <= m_params.Eps2Accept) {
        return true;
    }
    return false;
}


// -----------------------------------------------------------------------------
// HOW IT'S CALLED (Direct)
// -----------------------------------------------------------------------------
template <class IterType, class T, PerturbExtras Extras>
bool
FeatureFinder<IterType, T, Extras>::FindPeriodicPoint(IterType iters, FeatureSummary &feature) const
{
    return FindPeriodicPoint_Common(
        iters,
        feature,
        // findPeriod
        [this](const C &cAbs, IterTypeFull maxItersToTry, double R, IterType &outPeriod, EvalState &st) {
            return this->Evaluate_FindPeriod_Direct(cAbs, maxItersToTry, R, outPeriod, st);
        },
        // evalAtPeriod
        [this](const C &cAbs, IterType period, C &diff, C &dzdc, C &zcoeff, double &residual2) {
            return this->Evaluate_PeriodResidualAndDzdc_Direct(
                cAbs, period, diff, dzdc, zcoeff, residual2);
        });
}

// -----------------------------------------------------------------------------
// HOW IT'S CALLED (PT)
// -----------------------------------------------------------------------------
template <class IterType, class T, PerturbExtras Extras>
bool
FeatureFinder<IterType, T, Extras>::FindPeriodicPoint(
    const PerturbationResults<IterType, T, Extras> &results,
    RuntimeDecompressor<IterType, T, Extras> &dec,
    FeatureSummary &feature) const
{
    return FindPeriodicPoint_Common(
        results.GetCountOrbitEntries(),
        feature,
        // findPeriod
        [this, &results, &dec](
            const C &cAbs, IterTypeFull maxItersToTry, double R, IterType &outPeriod, EvalState &st) {
            return this->Evaluate_FindPeriod_PT(results, dec, cAbs, maxItersToTry, R, outPeriod, st);
        },
        // evalAtPeriod
        [this, &results, &dec](
            const C &cAbs, IterType period, C &diff, C &dzdc, C &zcoeff, double &residual2) {
            return this->Evaluate_PeriodResidualAndDzdc_PT(
                results, dec, cAbs, period, diff, dzdc, zcoeff, residual2);
        });
}

template <class IterType, class T, PerturbExtras Extras>
bool
FeatureFinder<IterType, T, Extras>::Evaluate_AtPeriod(
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
    outResidual2 = delta.norm_squared();
    st.z = z;
    return true;
}

// ------------------------------
// Explicit instantiations (minimal set you enabled)
// ------------------------------
#define InstantiatePeriodicPointFinder(IterTypeT, TT, PExtrasT)                                         \
    template class FeatureFinder<IterTypeT, TT, PExtrasT>;

//// ---- Disable ----
InstantiatePeriodicPointFinder(uint32_t, double, PerturbExtras::Disable);
InstantiatePeriodicPointFinder(uint64_t, double, PerturbExtras::Disable);
//
// InstantiatePeriodicPointFinder(uint32_t, float, PerturbExtras::Disable);
// InstantiatePeriodicPointFinder(uint64_t, float, PerturbExtras::Disable);
//
// InstantiatePeriodicPointFinder(uint32_t, CudaDblflt<MattDblflt>, PerturbExtras::Disable);
// InstantiatePeriodicPointFinder(uint64_t, CudaDblflt<MattDblflt>, PerturbExtras::Disable);
//
// InstantiatePeriodicPointFinder(uint32_t, HDRFloat<double>, PerturbExtras::Disable);
// InstantiatePeriodicPointFinder(uint64_t, HDRFloat<double>, PerturbExtras::Disable);
//
// InstantiatePeriodicPointFinder(uint32_t, HDRFloat<float>, PerturbExtras::Disable);
// InstantiatePeriodicPointFinder(uint64_t, HDRFloat<float>, PerturbExtras::Disable);
//
// InstantiatePeriodicPointFinder(uint32_t, HDRFloat<CudaDblflt<MattDblflt>>, PerturbExtras::Disable);
// InstantiatePeriodicPointFinder(uint64_t, HDRFloat<CudaDblflt<MattDblflt>>, PerturbExtras::Disable);
//
//// ---- Bad ----
// InstantiatePeriodicPointFinder(uint32_t, double, PerturbExtras::Bad);
// InstantiatePeriodicPointFinder(uint64_t, double, PerturbExtras::Bad);
//
// InstantiatePeriodicPointFinder(uint32_t, float, PerturbExtras::Bad);
// InstantiatePeriodicPointFinder(uint64_t, float, PerturbExtras::Bad);
//
// InstantiatePeriodicPointFinder(uint32_t, CudaDblflt<MattDblflt>, PerturbExtras::Bad);
// InstantiatePeriodicPointFinder(uint64_t, CudaDblflt<MattDblflt>, PerturbExtras::Bad);
//
// InstantiatePeriodicPointFinder(uint32_t, HDRFloat<double>, PerturbExtras::Bad);
// InstantiatePeriodicPointFinder(uint64_t, HDRFloat<double>, PerturbExtras::Bad);
//
// InstantiatePeriodicPointFinder(uint32_t, HDRFloat<float>, PerturbExtras::Bad);
// InstantiatePeriodicPointFinder(uint64_t, HDRFloat<float>, PerturbExtras::Bad);
//
// InstantiatePeriodicPointFinder(uint32_t, HDRFloat<CudaDblflt<MattDblflt>>, PerturbExtras::Bad);
// InstantiatePeriodicPointFinder(uint64_t, HDRFloat<CudaDblflt<MattDblflt>>, PerturbExtras::Bad);
//
//// ---- SimpleCompression ----
// InstantiatePeriodicPointFinder(uint32_t, double, PerturbExtras::SimpleCompression);
// InstantiatePeriodicPointFinder(uint64_t, double, PerturbExtras::SimpleCompression);
//
// InstantiatePeriodicPointFinder(uint32_t, float, PerturbExtras::SimpleCompression);
// InstantiatePeriodicPointFinder(uint64_t, float, PerturbExtras::SimpleCompression);
//
// InstantiatePeriodicPointFinder(uint32_t, CudaDblflt<MattDblflt>, PerturbExtras::SimpleCompression);
// InstantiatePeriodicPointFinder(uint64_t, CudaDblflt<MattDblflt>, PerturbExtras::SimpleCompression);
//
// InstantiatePeriodicPointFinder(uint32_t, HDRFloat<double>, PerturbExtras::SimpleCompression);
// InstantiatePeriodicPointFinder(uint64_t, HDRFloat<double>, PerturbExtras::SimpleCompression);
//
// InstantiatePeriodicPointFinder(uint32_t, HDRFloat<float>, PerturbExtras::SimpleCompression);
// InstantiatePeriodicPointFinder(uint64_t, HDRFloat<float>, PerturbExtras::SimpleCompression);
//
// InstantiatePeriodicPointFinder(uint32_t,
//                                HDRFloat<CudaDblflt<MattDblflt>>,
//                                PerturbExtras::SimpleCompression);
// InstantiatePeriodicPointFinder(uint64_t,
//                                HDRFloat<CudaDblflt<MattDblflt>>,
//                                PerturbExtras::SimpleCompression);

#undef InstantiatePeriodicPointFinder
