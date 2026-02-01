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
    // Require positive radius
    HdrReduce(R);
    const T zero = HdrReduce(T{});
    if (HdrCompareToBothPositiveReducedLE(R, zero)) {
        std::cout << "FeatureFinder::Evaluate_FindPeriod_Direct: R must be positive.\n";
        return false;
    }

    // Pre-reduce constants used in Reduced compares
    const T two = HdrReduce(T{2.0});
    const T one = HdrReduce(T{1.0});
    const T escape2 = HdrReduce(T{4096.0});

    // R2 (reduced)
    T R2 = R * R;
    HdrReduce(R2);

    C z{};      // z_0 = 0
    C dzdc{};   // dz/dc at z0 is 0
    C zcoeff{}; // matches your reference ordering (product-ish)

    for (IterTypeFull n = 0; n < maxIters; ++n) {
        // zcoeff ordering matches your double direct reference:
        // if (n==0) zcoeff = 1; else zcoeff *= (2*z)
        if (n == 0) {
            zcoeff = C(one, T{});
        } else {
            zcoeff = zcoeff * (z * two);
        }
        zcoeff.Reduce();

        // dzdc <- 2*z*dzdc + 1
        dzdc = dzdc * (z * two) + C(one, T{});
        dzdc.Reduce();

        // Advance orbit: z <- z^2 + c
        z = (z * z) + c;
        z.Reduce();

        // Escape check on |z|^2
        T z2 = z.norm_squared();
        HdrReduce(z2);
        if (HdrCompareToBothPositiveReducedGT(z2, escape2)) {
            // Non-periodic / escaped before finding a candidate period.
            break;
        }

        // Period trigger:
        // if |z|^2 < R^2 * |dzdc|^2  => candidate period = n+1
        T d2 = dzdc.norm_squared();
        HdrReduce(d2);

        T rhs = R2 * d2;
        HdrReduce(rhs);

        if (HdrCompareToBothPositiveReducedLT(z2, rhs)) {
            const IterTypeFull cand = n + 1;
            if (cand <= static_cast<IterTypeFull>(std::numeric_limits<IterType>::max())) {
                outPeriod = static_cast<IterType>(cand);
                st.z = z;
                return true;
            }

            std::cout
                << "FeatureFinder::Evaluate_FindPeriod_Direct: candidate period exceeds IterType.\n";
            return false;
        }

        // (Optional) extra guard if dzdc went degenerate
        // if (HdrCompareToBothPositiveReducedLE(d2, zero)) break;
    }

    return false;
}

template <class IterType, class T, PerturbExtras PExtras>
bool
FeatureFinder<IterType, T, PExtras>::Evaluate_PeriodResidualAndDzdc_Direct(
    const C &c, IterType period, C &outDiff, C &outDzdc, C &outZcoeff, T &outResidual2) const
{
    C z{};
    C dzdc{};

    C oneC{T{1.0}, T{}};
    C zcoeff{T{1.0}, T{}};

    oneC.Reduce();
    zcoeff.Reduce();

    for (IterType i = 0; i < period; ++i) {
        zcoeff = zcoeff * (z * T{2.0});
        zcoeff.Reduce();

        dzdc = dzdc * (z * T{2.0}) + oneC;
        dzdc.Reduce();

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

    outDzdc = dzdc;
    outZcoeff = zcoeff;

    std::cout << "FeatureFinder::Evaluate_PeriodResidualAndDzdc_Direct: residual^2 = "
              << HdrToString<false>(outResidual2)
              << ".\n";
    return true;
}

// =====================================================================================
// 1) Find period candidate using PT (reference orbit)
// =====================================================================================
template <class IterType, class T, PerturbExtras PExtras>
bool
FeatureFinder<IterType, T, PExtras>::Evaluate_FindPeriod_PT(
    const PerturbationResults<IterType, T, PExtras> &results,
    RuntimeDecompressor<IterType, T, PExtras> &dec,
    const C &cAbs,
    IterTypeFull maxItersToTry,
    T R,
    IterType &outPeriod,
    EvalState &st) const
{
    auto Zref = [&](size_t n) -> C {
        auto v = results.GetComplex(dec, n);
        C z(static_cast<T>(v.getRe()), static_cast<T>(v.getIm()));
        z.Reduce();
        return z;
    };

    const IterType pScreen = results.GetPeriodMaybeZero();
    if (pScreen != 0) {
        outPeriod = pScreen;
        st.z = C{};
        std::cout << "[PT FindPeriod] SUCCESS: using stored period=" << (uint64_t)pScreen << "\n";
        return true;
    }

    HdrReduce(R);
    const T zero = HdrReduce(T{});
    if (HdrCompareToBothPositiveReducedLE(R, zero)) {
        std::cout << "[PT FindPeriod] FAIL: R <= 0\n";
        return false;
    }

    T SqrNearLinearRadius = R * R;
    HdrReduce(SqrNearLinearRadius);

    const T near1 = HdrReduce(T{0.25});
    T SqrNearLinearRadiusScale = near1 * near1;
    HdrReduce(SqrNearLinearRadiusScale);

    const T one = HdrReduce(T{1.0});
    const T two = HdrReduce(T{2.0});
    const T escape2 = HdrReduce(T{4096.0});

    const C cRef(ToDouble(results.GetHiX()), ToDouble(results.GetHiY()));
    C dc = cAbs - cRef;
    dc.Reduce();

    const size_t storedCount = static_cast<size_t>(results.GetCountOrbitEntries());
    if (storedCount < 2) {
        std::cout << "[PT FindPeriod] FAIL: ref orbit too short\n";
        return false;
    }

    const IterTypeFull hardCap = static_cast<IterTypeFull>(storedCount - 1);
    const IterTypeFull cap = std::min<IterTypeFull>(maxItersToTry, hardCap);

    if (cap < 1) {
        std::cout << "[PT FindPeriod] FAIL: cap < 1\n";
        return false;
    }

    C dz{};
    C z{};
    C dzdc{};
    C zcoeff{};

    dz.Reduce();
    dzdc.Reduce();
    zcoeff.Reduce();

    // Track reference iteration separately (key difference from original)
    size_t refIteration = 0;
    const size_t refOrbitLength = storedCount;

    for (IterTypeFull n = 0; n < cap; ++n) {
        const C zref = Zref(refIteration);

        // full z_n = zref + dz
        z = zref + dz;
        z.Reduce();

        // derivative updates using full z
        if (n == 0)
            zcoeff = C(one, T{});
        else
            zcoeff = zcoeff * (z * two);
        zcoeff.Reduce();

        dzdc = dzdc * (z * two) + C(one, T{});
        dzdc.Reduce();

        // PT advance: dz_{n+1} = dz * (zref + z) + dc
        dz = dz * (zref + z) + dc;
        dz.Reduce();

        // Advance reference iteration
        refIteration++;

        // Clamp refIteration to valid range for Zref access
        size_t safeRefIt = (refIteration < refOrbitLength) ? refIteration : refOrbitLength - 1;

        // Get next reference point
        const C zrefNext = Zref(safeRefIt);
        C zNext = zrefNext + dz;
        zNext.Reduce();

        // **REBASING**: Critical check from original implementation
        // When reference orbit exhausted OR perturbation exceeds full value
        T dzNorm = dz.norm_squared();
        HdrReduce(dzNorm);
        T zNextNorm = zNext.norm_squared();
        HdrReduce(zNextNorm);

        if (refIteration >= refOrbitLength || HdrCompareToBothPositiveReducedLT(zNextNorm, dzNorm)) {
            // Rebase: set dz = z (full value) and restart reference
            dz = zNext;
            dz.Reduce();
            refIteration = 0;
        }

        // Escape check
        T Magnitude = zNext.norm_squared();
        HdrReduce(Magnitude);
        if (HdrCompareToBothPositiveReducedGT(Magnitude, escape2)) {
            std::cout << "[PT FindPeriod] FAIL: escape at n=" << (uint64_t)n << "\n";
            return false;
        }

        // Near-linear trigger
        T d2 = dzdc.norm_squared();
        HdrReduce(d2);

        T rhs = SqrNearLinearRadius * d2;
        HdrReduce(rhs);

        if (HdrCompareToBothPositiveReducedLT(Magnitude, rhs)) {
            const IterTypeFull cand = n + 1;
            if (cand <= static_cast<IterTypeFull>(std::numeric_limits<IterType>::max())) {
                outPeriod = static_cast<IterType>(cand);
                st.z = zNext;
                std::cout << "[PT FindPeriod] SUCCESS: found period=" << (uint64_t)cand << "\n";
                return true;
            }
            std::cout << "[PT FindPeriod] FAIL: period overflow\n";
            return false;
        }

        // Adaptive tightening
        if (HdrCompareToBothPositiveReducedGT(d2, zero)) {
            T lhsTighten = Magnitude * SqrNearLinearRadiusScale;
            HdrReduce(lhsTighten);

            if (HdrCompareToBothPositiveReducedLT(lhsTighten, rhs)) {
                T newSqr = lhsTighten / d2;
                HdrReduce(newSqr);
                if (HdrCompareToBothPositiveReducedGT(newSqr, zero)) {
                    SqrNearLinearRadius = newSqr;
                }
            }
        }
    }

    std::cout << "[PT FindPeriod] FAIL: no trigger found\n";
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
    C &outDiff,
    C &outDzdc,
    C &outZcoeff,
    T &outResidual2) const
{
    auto Zref = [&](size_t n) -> C {
        auto v = results.GetComplex(dec, n);
        C z(static_cast<T>(v.getRe()), static_cast<T>(v.getIm()));
        z.Reduce();
        return z;
    };

    const size_t storedCount = static_cast<size_t>(results.GetCountOrbitEntries());
    if (storedCount < 2) {
        std::cout << "[PT EvalPeriod] FAIL: orbit too short; need at least 2, have "
                  << (uint64_t)storedCount << "\n";
        return false;
    }

    const T one = HdrReduce(T{1.0});
    const T two = HdrReduce(T{2.0});
    const T escape2 = HdrReduce(T{4096.0});

    // Use hi-precision c_ref so dc is correct even for deep zoom.
    const C cRef(ToDouble(results.GetHiX()), ToDouble(results.GetHiY()));
    C dc = cAbs - cRef;
    dc.Reduce();

    C dz{};     // dz_0 = 0
    C z{};      // full z_i
    C dzdc{};   // dz/dc
    C zcoeff{}; // product-ish convention parity

    dz.Reduce();
    dzdc.Reduce();
    zcoeff.Reduce();

    // Track reference iteration separately (matches original implementation)
    size_t refIteration = 0;
    const size_t refOrbitLength = storedCount;

    for (IterType i = 0; i < period; ++i) {
        const C zref = Zref(refIteration);

        // full z_i = zref + dz
        z = zref + dz;
        z.Reduce();

        // zcoeff update (parity with direct path ordering)
        if (i == 0)
            zcoeff = C(one, T{});
        else
            zcoeff = zcoeff * (z * two);
        zcoeff.Reduce();

        // dzdc update
        dzdc = dzdc * (z * two) + C(one, T{});
        dzdc.Reduce();

        // PT advance: dz_{i+1} = dz * (zref + z) + dc
        dz = dz * (zref + z) + dc;
        dz.Reduce();

        // Advance reference iteration
        refIteration++;

        // Clamp refIteration to valid range for Zref access
        size_t safeRefIt = (refIteration < refOrbitLength) ? refIteration : refOrbitLength - 1;

        // Get next reference point
        const C zrefNext = Zref(safeRefIt);
        C zNext = zrefNext + dz;
        zNext.Reduce();

        // **REBASING**: Critical check from original implementation
        // When reference orbit exhausted OR perturbation exceeds full value
        T dzNorm = dz.norm_squared();
        HdrReduce(dzNorm);
        T zNextNorm = zNext.norm_squared();
        HdrReduce(zNextNorm);

        if (refIteration >= refOrbitLength || HdrCompareToBothPositiveReducedLT(zNextNorm, dzNorm)) {
            // Rebase: set dz = z (full value) and restart reference
            dz = zNext;
            dz.Reduce();
            refIteration = 0;
        }

        // Escape check on full z_{i+1}
        T z2 = zNext.norm_squared();
        HdrReduce(z2);
        if (HdrCompareToBothPositiveReducedGT(z2, escape2)) {
            std::cout << "[PT EvalPeriod] FAIL: escape at i=" << (uint64_t)i << "\n";
            return false;
        }
    }

    // Compute final z_p = zref[refIteration] + dz
    const C zrefP = Zref(refIteration);
    C zP = zrefP + dz;
    zP.Reduce();

    outDiff = zP;
    outResidual2 = outDiff.norm_squared();
    HdrReduce(outResidual2);

    outDzdc = dzdc;
    outDzdc.Reduce();

    outZcoeff = zcoeff;
    outZcoeff.Reduce();

    std::cout << "[PT EvalPeriod] SUCCESS: period=" << (uint64_t)period
              << " residual2=" << HdrToString<false, T>(outResidual2) << "\n";

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
        // findPeriod (now uses stored period if present)
        [this, &results, &dec](
            const C &cAbs, IterTypeFull maxItersToTry, T R, IterType &outPeriod, EvalState &st) {
            return this->Evaluate_FindPeriod_PT(results, dec, cAbs, maxItersToTry, R, outPeriod, st);
        },
        // evalAtPeriod with fallback
        [this, &results, &dec](
            const C &cAbs, IterType period, C &diff, C &dzdc, C &zcoeff, T &residual2) {
            if (this->Evaluate_PeriodResidualAndDzdc_PT(
                    results, dec, cAbs, period, diff, dzdc, zcoeff, residual2)) {
                return true;
            }

            std::cout << "[PT->Direct fallback] INFO: PT eval failed; trying DIRECT at same period="
                      << (uint64_t)period << "\n";

            // Direct fallback uses exact dynamics at the same period.
            // This is the closest analogue to the original mixed-mode evaluator.
            return this->Evaluate_PeriodResidualAndDzdc_Direct(
                cAbs, period, diff, dzdc, zcoeff, residual2);
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
