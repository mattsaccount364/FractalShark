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

template<class IterType, class T, PerturbExtras PExtras>
T
FeatureFinder<IterType, T, PExtras>::ChebAbs(const C &a) const
{
    return HdrMaxReduced(HdrAbs(a.getRe()), HdrAbs(a.getIm()));
}

// Renormalize dzdc/zcoeff so their magnitude stays ~ 2^targetExp.
// This preserves the invariant: true_dzdc = dzdc * 2^scaleExp
// and forces the recurrence to add scalingFactor = 2^-scaleExp instead of 1.
template <class IterType, class T, PerturbExtras PExtras>
void
FeatureFinder<IterType, T, PExtras>::RenormalizeDzdcZcoeff(C &dzdc, C &zcoeff, int &scaleExp) const
{
    // Pick thresholds with lots of headroom before double overflow (~2^1024).
    constexpr int kHiExp = 700;     // if exponent exceeds this, renormalize
    constexpr int kTargetExp = 200; // bring it back near this exponent

    auto m = HdrMaxReduced(ChebAbs(dzdc), ChebAbs(zcoeff));
    if (!(HdrCompareToBothPositiveReducedGT(m, T{}))) {
        return; // caller will fail on non-finite elsewhere
    }

    // ilogb gives floor(log2(|m|)) as an int, without computing log().
    const int e = HdrIlogb(m);
    if (e <= kHiExp) {
        return;
    }

    const int shift = e - kTargetExp;            // shift > 0
    const T down = HdrLdexp(T{1.0}, -shift); // 2^-shift

    dzdc = dzdc * down;
    zcoeff = zcoeff * down;
    scaleExp += shift;
}

// Convert scaled values back to true values.
template <class IterType, class T, PerturbExtras PExtras>
FeatureFinder<IterType, T, PExtras>::C
FeatureFinder<IterType, T, PExtras>::Unscale(const C &a, int scaleExp) const
{
    return a * HdrLdexp(T{1.0}, scaleExp); // a * 2^scaleExp
}

template <class IterType, class T, PerturbExtras PExtras>
T
FeatureFinder<IterType, T, PExtras>::ToDouble(const HighPrecision &v)
{
    return T{v};
}

template <class IterType, class T, PerturbExtras PExtras>
typename FeatureFinder<IterType, T, PExtras>::C
FeatureFinder<IterType, T, PExtras>::Div(const C &a, const C &b)
{
    // a / b = a * conj(b) / |b|^2
    const T br = b.getRe();
    const T bi = b.getIm();
    const T denom = br * br + bi * bi;

    // If denom is 0, caller is hosed; keep it explicit.
    if (HdrCompareToBothPositiveReducedLE(denom, T{})) {
        return C{};
    }

    // a * conj(b)
    const T ar = a.getRe();
    const T ai = a.getIm();
    const T rr = (ar * br + ai * bi) / denom;
    const T ii = (ai * br - ar * bi) / denom;
    return C(rr, ii);
}

template <class IterType, class T, PerturbExtras PExtras>
bool
FeatureFinder<IterType, T, PExtras>::Evaluate_FindPeriod_Direct(
    const C &c, IterTypeFull maxIters, T R, IterType &outPeriod, EvalState &st) const
{
    auto reduced0{HdrReduce(T{})};
    HdrReduce(R);
    if (HdrCompareToBothPositiveReducedLE(R, reduced0)) {
        std::cout << "FeatureFinder::Evaluate_FindPeriod_Direct: R must be positive.\n";
        return false;
    }

    const T R2 = R * R;

    C z{};
    C dzdc{};
    C zcoeff{};

    int scaleExp = 0; // true_dzdc = dzdc * 2^scaleExp

    for (IterTypeFull n = 0; n < maxIters; ++n) {
        const T scalingFactor = HdrLdexp(T{1.0}, -scaleExp); // 2^-scaleExp

        // zcoeff ordering matches your reference:
        if (n == 0)
            zcoeff = C(scalingFactor, T{});
        else
            zcoeff = zcoeff * (z * T{2.0});

        // dzdc <- 2*z*dzdc + scalingFactor  (instead of +1)
        dzdc = dzdc * (z * T{2.0}) + C(scalingFactor, T{});

        // Keep dzdc/zcoeff from blowing up; updates scaleExp.
        RenormalizeDzdcZcoeff(dzdc, zcoeff, scaleExp);

        // Advance orbit
        z = (z * z) + c;

        if (HdrCompareToBothPositiveReducedGT(z.norm_squared(), T{4096.0})) {
            std::cout << "FeatureFinder::Evaluate_FindPeriod_Direct: orbit escaped.\n";
            break;
        }

        // Period trigger uses TRUE dzdc (unscaled)
        const T magZ = z.norm_squared();
        const C dzdcTrue = Unscale(dzdc, scaleExp);
        const T magD = dzdcTrue.norm_squared();

        if (HdrCompareToBothPositiveReducedLT(magZ, T{R2 * magD})) {
            const IterTypeFull cand = n + 1;
            if (cand <= static_cast<IterTypeFull>(std::numeric_limits<IterType>::max())) {
                std::cout << "FeatureFinder::Evaluate_FindPeriod_Direct: found period candidate " << cand
                          << " at iter " << n << ".\n";
                outPeriod = static_cast<IterType>(cand);
                st.z = z;
                return true;
            }

            std::cout << "FeatureFinder::Evaluate_FindPeriod_Direct: period candidate " << cand
                      << " exceeds max IterType.\n";
            return false;
        }
    }

    std::cout << "FeatureFinder::Evaluate_FindPeriod_Direct: no period found up to maxIters.\n";
    return false;
}

template <class IterType, class T, PerturbExtras PExtras>
bool
FeatureFinder<IterType, T, PExtras>::Evaluate_PeriodResidualAndDzdc_Direct(
    const C &c, IterType period, C &outDiff, C &outDzdc, C &outZcoeff, T &outResidual2) const
{
    C z{};
    C dzdc{};
    C zcoeff{};

    int scaleExp = 0;

    for (IterType i = 0; i < period; ++i) {
        const T scalingFactor = HdrLdexp(T{1.0}, -scaleExp);

        if (i == 0)
            zcoeff = C(scalingFactor, T{});
        else
            zcoeff = zcoeff * (z * T{2.0});

        zcoeff.Reduce();

        dzdc = dzdc * (z * T{2.0}) + C(scalingFactor, T{});
        dzdc.Reduce();

        RenormalizeDzdcZcoeff(dzdc, zcoeff, scaleExp);

        z = (z * z) + c;
        z.Reduce();

        auto normSq = z.norm_squared();
        HdrReduce(normSq);

        if (HdrCompareToBothPositiveReducedGT(normSq, T{4096.0})) {
            std::cout << "FeatureFinder::Evaluate_PeriodResidualAndDzdc_Direct: orbit escaped.\n";
            return false;
        }
    }

    outDiff = z;
    outResidual2 = outDiff.norm_squared();

    // Unscale derivatives back to true values
    outDzdc = Unscale(dzdc, scaleExp);
    outZcoeff = Unscale(zcoeff, scaleExp);

    std::cout << "FeatureFinder::Evaluate_PeriodResidualAndDzdc_Direct: residual^2 = "
              << HdrToString<false>(outResidual2)
              << ".\n";
    return true;
}

// =====================================================================================
// PT evaluation using PerturbationResults + RuntimeDecompressor
// (FloatComplex<double> only, like your new direct implementation)
// =====================================================================================

// --- helpers -------------------------------------------------------------

template <class IterType, class T, PerturbExtras PExtras>
FeatureFinder<IterType, T, PExtras>::C
FeatureFinder<IterType, T, PExtras>::ToC_fromRefOrbitPoint(
    const PerturbationResults<IterType, T, PExtras> &results,
                      RuntimeDecompressor<IterType, T, PExtras> &dec,
    size_t idx) const
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
    const auto xr = static_cast<T>(v.getRe());
    const auto yi = static_cast<T>(v.getIm());

    return C(xr, yi);
}

template <class IterType, class T, PerturbExtras PExtras>
FeatureFinder<IterType, T, PExtras>::C
FeatureFinder<IterType, T, PExtras>::ToC_fromReferenceC(
    const PerturbationResults<IterType, T, PExtras> &results) const
{
    // reference parameter c_ref that the authoritative orbit was generated at
    // (low precision fields are fine for PT in double; you can swap to hi if desired).
    const auto cx = static_cast<T>(results.GetOrbitXLow());
    const auto cy = static_cast<T>(results.GetOrbitYLow());
    return C(cx, cy);
}

// =====================================================================================
// 1) Find period candidate using PT (reference orbit)
// =====================================================================================
template <class IterType, class T, PerturbExtras PExtras>
bool
FeatureFinder<IterType, T, PExtras>::Evaluate_FindPeriod_PT(
    const PerturbationResults<IterType, T, PExtras> &results,
    RuntimeDecompressor<IterType, T, PExtras> &dec,
    const C &cAbs,              // absolute candidate c
    IterTypeFull maxItersToTry, // how far to search for a period
    T R,                   // radius for near-linear trigger
    IterType &outPeriod,
    EvalState &st) const
{
    if (HdrCompareToBothPositiveReducedLE(R, T{}))
        return false;
    const T R2 = R * R;

    const C cRef = ToC_fromReferenceC(results);
    const C dc = cAbs - cRef; // dc in perturbation formula

    // Guard: need reference orbit long enough.
    const size_t refCount = static_cast<size_t>(results.GetCountOrbitEntries());
    if (refCount < 2)
        return false;

    C dz{}; // perturbation delta (your old "dz")
    C z{};  // full z
    C dzdc{};
    C zcoeff{};

    int scaleExp = 0; // true_dzdc = dzdc * 2^scaleExp (and same for zcoeff)

    // We need z_ref[n] each step. Assume results stores z_ref[0]=0 and subsequent.
    // We iterate n from 0..N-1, using z_ref[n] to advance delta to n+1, and then form z at n+1.
    const IterTypeFull N =
        std::min<IterTypeFull>(maxItersToTry, static_cast<IterTypeFull>(refCount - 1));

    for (IterTypeFull n = 0; n < N; ++n) {
        const T scalingFactor = HdrLdexp(T{1.0}, -scaleExp);

        // Full z_n = z_ref[n] + dz_n
        const C zref = ToC_fromRefOrbitPoint(results, dec, static_cast<size_t>(n));

        z = zref + dz;
        HdrReduce(z);

        // Derivative recurrences match your reference ordering:
        if (n == 0)
            zcoeff = C(scalingFactor, T{});
        else
            zcoeff = zcoeff * (z * T{2.0});

        dzdc = dzdc * (z * T{2.0}) + C(scalingFactor, T{});
        HdrReduce(dzdc);

        RenormalizeDzdcZcoeff(dzdc, zcoeff, scaleExp);

        // Perturbation advance:
        // dz_{n+1} = dz_n * (z_ref[n] + z_n) + dc
        //          = dz_n * (2*z_ref[n] + dz_n) + dc
        dz = dz * (zref + z) + dc;
        HdrReduce(dz);

        // Full z_{n+1} for escape / periodic check:
        const C zrefNext = ToC_fromRefOrbitPoint(results, dec, static_cast<size_t>(n + 1));
        const C zNext = zrefNext + dz;

        if (HdrCompareToBothPositiveReducedGT(zNext.norm_squared(), T{4096.0}))
            break;

        // Period trigger uses TRUE dzdc (unscaled)
        T magZ = zNext.norm_squared();
        HdrReduce(magZ);

        const C dzdcTrue = Unscale(dzdc, scaleExp);
        T magD = dzdcTrue.norm_squared();
        HdrReduce(magD);

        T prodMagDR2 = R2 * magD;
        HdrReduce(prodMagDR2);

        if (HdrCompareToBothPositiveReducedLT(magZ, prodMagDR2)) {
            const IterTypeFull cand = n + 1; // matches your direct: period = n+1
            if (cand <= static_cast<IterTypeFull>(std::numeric_limits<IterType>::max())) {
                outPeriod = static_cast<IterType>(cand);
                st.z = zNext;
                return true;
            }
            return false;
        }
    }

    return false;
}

// =====================================================================================
// 2) Evaluate diff=z_p(c), dzdc, zcoeff at a fixed period using PT
// =====================================================================================
template <class IterType, class T, PerturbExtras PExtras>
bool
FeatureFinder<IterType, T, PExtras>::Evaluate_PeriodResidualAndDzdc_PT(
    const PerturbationResults<IterType, T, PExtras> &results,
    RuntimeDecompressor<IterType, T, PExtras> &dec,
    const C &cAbs, // absolute candidate c
    IterType period,
    C &outDiff,   // z_p(c)  (or delta vs z0 if you prefer)
    C &outDzdc,   // dz_p/dc
    C &outZcoeff, // product(2*z_k) in your convention
    T &outResidual2) const
{
    const C cRef = ToC_fromReferenceC(results);
    const C dc = cAbs - cRef;

    const size_t refCount = static_cast<size_t>(results.GetCountOrbitEntries());
    if (refCount < static_cast<size_t>(period) + 1)
        return false;

    C dz{};
    C z{};
    C dzdc{};
    C zcoeff{};

    int scaleExp = 0;

    for (IterType i = 0; i < period; ++i) {
        const T scalingFactor = HdrLdexp(T{1.0}, -scaleExp);

        const C zref = ToC_fromRefOrbitPoint(results, dec, static_cast<size_t>(i));
        z = zref + dz;

        if (i == 0)
            zcoeff = C{scalingFactor, T{}};
        else
            zcoeff = zcoeff * (z * T{2.0});

        dzdc = dzdc * (z * T{2.0}) + C(scalingFactor, T{});

        RenormalizeDzdcZcoeff(dzdc, zcoeff, scaleExp);

        // perturbation step
        dz = dz * (zref + z) + dc;

        // optional bail-outs
        const C zrefNext = ToC_fromRefOrbitPoint(results, dec, static_cast<size_t>(i + 1));
        const C zNext = zrefNext + dz;
        if (HdrCompareToBothPositiveReducedGT(zNext.norm_squared(), T{4096.0}))
            return false;
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

template <class IterType, class T, PerturbExtras PExtras>
template <class FindPeriodFn, class EvalFn>
bool
FeatureFinder<IterType, T, PExtras>::FindPeriodicPoint_Common(IterType refIters,
                                                             FeatureSummary &feature,
                                                             FindPeriodFn &&findPeriod,
                                                             EvalFn &&evalAtPeriod) const
{
    const T cx0 {feature.GetOrigX()};
    const T cy0 {feature.GetOrigY()};
    T R {feature.GetRadius()};
    R = HdrAbs(R);

    const C origC{cx0, cy0};
    C c = origC;

    EvalState st{};
    IterType period = 0;

    // 1) Find candidate period (same role as original: Evaluate<true>)
    if (!findPeriod(origC, refIters, R, period, st)) {
        std::cout << "Rejected: findPeriod failed\n";
        return false;
    }

    // 2) Newton at fixed period (match original acceptance: relative step size)
    // Original:
    //   Tolerance = 2^-40; SqrTolerance = 2^-80;
    //   if (norm(step) < norm(c) * SqrTolerance) break;
    const T relTol = m_params.RelStepTol; // e.g. 0x1p-40
    const T relTol2 = relTol * relTol;

    C diff, dzdc, zcoeff;
    T residual2{};

    // Initial eval + correction (matches original "dc -= diff/dzdc" before loop)
    if (!evalAtPeriod(c, period, diff, dzdc, zcoeff, residual2)) {
        std::cout << "Rejected: initial evalAtPeriod failed\n";
        return false;
    }

    const T dzdc2_init = dzdc.norm_squared();
    if (HdrCompareToBothPositiveReducedLE(dzdc2_init, T{})) {
        std::cout << "Rejected: initial dzdc too small\n";
        return false;
    }

    C step0 = Div(diff, dzdc);
    c = c - step0;

    bool convergedByStep = false;

    // Newton loop (original did up to 32 iters, break by relative step size)
    for (uint32_t it = 0; it < m_params.MaxNewtonIters; ++it) {
        if (!evalAtPeriod(c, period, diff, dzdc, zcoeff, residual2)) {
            std::cout << "Rejected: evalAtPeriod failed in loop\n";
            return false;
        }

        T dzdc2 = dzdc.norm_squared();
        HdrReduce(dzdc2);

        if (HdrCompareToBothPositiveReducedLE(dzdc2, T{})) {
            std::cout << "Rejected: dzdc too small\n";
            return false;
        }

        C step = Div(diff, dzdc);
        step.Reduce();

        // To match the original behavior, do NOT damp by default.
        // If you want damping, apply it here (but understand it changes behavior).
        // const double damp = std::clamp(m_params.Damp, m_params.DampMin, m_params.DampMax);
        // step = step * damp;

        c = c - step;

        T step2 = step.norm_squared();
        HdrReduce(step2);

        const T c2 = c.norm_squared();

        auto c2Prod = c2 * relTol2;
        HdrReduce(c2Prod);

        // Match original: norm(step) < norm(c) * 2^-80  (squared form)
        if (HdrCompareToBothPositiveReducedLE(step2, c2Prod)) {
            convergedByStep = true;
            break;
        }

        // Optional: keep residual-based early-out only if you explicitly want it.
        // (Original did NOT do this.)
        if (HdrCompareToBothPositiveReducedGT(m_params.Eps2Accept, T{}) &&
            HdrCompareToBothPositiveReducedLE(residual2, m_params.Eps2Accept)) {
            break;
        }
    }

    // Original does one extra "final eval + step" after the loop.
    // Keep that to match behavior closely.
    if (!evalAtPeriod(c, period, diff, dzdc, zcoeff, residual2)) {
        std::cout << "Rejected: intermediate evalAtPeriod failed\n";
        return false;
    }

    const T dzdc2_final = dzdc.norm_squared();
    if (HdrCompareToBothPositiveReducedLE(dzdc2_final, T{})) {
        std::cout << "Rejected: final dzdc too small\n";
        return false;
    }

    {
        C step = Div(diff, dzdc);
        c = c - step;

        const T step2 = step.norm_squared();
        const T c2 = c.norm_squared();
        if (HdrCompareToBothPositiveReducedLE(step2, c2 * relTol2)) {
            convergedByStep = true;
        }
    }

    // Final eval (so residual2 reflects the returned/printed solution)
    if (!evalAtPeriod(c, period, diff, dzdc, zcoeff, residual2)) {
        std::cout << "Rejected: final eval failed\n";
        return false;
    }

    //T R2 = R * R;
    //HdrReduce(R2);
    //
    //auto normSq = (origC - c).norm_squared();
    //HdrReduce(normSq);
    //
    //if (HdrCompareToBothPositiveReducedGT(normSq, R2)) {
    //    std::cout << "Rejected: final c too far from original c\n";
    //    return false;
    //}


    feature.SetFound(HighPrecision(c.getRe()), HighPrecision(c.getIm()), period, HDRFloat<double>{residual2});

    if (m_params.PrintResult) {
        std::cout << "Periodic point: "
                  << "orig cx=" << HdrToString<false, T>(origC.getRe()) << " orig cy=" << HdrToString<false, T>(origC.getIm())
                  << " new cx=" << HdrToString<false, T>(c.getRe()) << " new cy=" << HdrToString<false, T>(c.getIm())
                  << " period=" << static_cast<uint64_t>(period) << " residual2=" << HdrToString<false, T>(residual2) << "\n";
    }

    // MATCH ORIGINAL "success" semantics:
    // success is primarily about Newton converging by step size.
    // (Optionally also accept small residual if you enabled it.)
    //if (convergedByStep) {
    //    return true;
    //}
    //if (HdrCompareToBothPositiveReducedGT(m_params.Eps2Accept, T{}) && HdrCompareToBothPositiveReducedLE(residual2, m_params.Eps2Accept)) {
    //    return true;
    //}
    return true;
}


// -----------------------------------------------------------------------------
// HOW IT'S CALLED (Direct)
// -----------------------------------------------------------------------------
template <class IterType, class T, PerturbExtras PExtras>
bool
FeatureFinder<IterType, T, PExtras>::FindPeriodicPoint(IterType iters, FeatureSummary &feature) const
{
    return FindPeriodicPoint_Common(
        iters,
        feature,
        // findPeriod
        [this](const C &cAbs, IterTypeFull maxItersToTry, T R, IterType &outPeriod, EvalState &st) {
            return this->Evaluate_FindPeriod_Direct(cAbs, maxItersToTry, R, outPeriod, st);
        },
        // evalAtPeriod
        [this](const C &cAbs, IterType period, C &diff, C &dzdc, C &zcoeff, T &residual2) {
            return this->Evaluate_PeriodResidualAndDzdc_Direct(
                cAbs, period, diff, dzdc, zcoeff, residual2);
        });
}

// -----------------------------------------------------------------------------
// HOW IT'S CALLED (PT)
// -----------------------------------------------------------------------------
template <class IterType, class T, PerturbExtras PExtras>
bool
FeatureFinder<IterType, T, PExtras>::FindPeriodicPoint(
    const PerturbationResults<IterType, T, PExtras> &results,
    RuntimeDecompressor<IterType, T, PExtras> &dec,
    FeatureSummary &feature) const
{
    return FindPeriodicPoint_Common(
        results.GetCountOrbitEntries(),
        feature,
        // findPeriod
        [this, &results, &dec](
            const C &cAbs, IterTypeFull maxItersToTry, T R, IterType &outPeriod, EvalState &st) {
            return this->Evaluate_FindPeriod_PT(results, dec, cAbs, maxItersToTry, R, outPeriod, st);
        },
        // evalAtPeriod
        [this, &results, &dec](
            const C &cAbs, IterType period, C &diff, C &dzdc, C &zcoeff, T &residual2) {
            return this->Evaluate_PeriodResidualAndDzdc_PT(
                results, dec, cAbs, period, diff, dzdc, zcoeff, residual2);
        });
}

template <class IterType, class T, PerturbExtras PExtras>
bool
FeatureFinder<IterType, T, PExtras>::Evaluate_AtPeriod(
    const PerturbationResults<IterType, T, PExtras> & /*results*/,
    RuntimeDecompressor<IterType, T, PExtras> &dec,
    const C &c,
    IterType period,
    EvalState &st,
    T &outResidual2) const
{
    // Re-evaluate orbit up to "period" and measure how close we are to z0.
    C z{};
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
InstantiatePeriodicPointFinder(uint32_t, HDRFloat<double>, PerturbExtras::Disable);
InstantiatePeriodicPointFinder(uint64_t, HDRFloat<double>, PerturbExtras::Disable);
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
