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
FeatureFinder<IterType, T, PExtras>::Evaluate_FindPeriod_Direct(const C &c,
                                                                IterTypeFull maxIters,
                                                                T R,
                                                                IterType &outPeriod,
                                                                C &outDiff,
                                                                C &outDzdc,
                                                                C &outZcoeff,
                                                                T &outResidual2) const
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
                outDiff = z;
                outDzdc = dzdc;
                outZcoeff = zcoeff;
                outResidual2 = z2;
                return true;
            }

            std::cout
                << "FeatureFinder::Evaluate_FindPeriod_Direct: candidate period exceeds IterType.\n";
            return false;
        }
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
// Unified PT evaluation - FindPeriod=true searches for period, false evaluates at fixed period
// =====================================================================================
template <class IterType, class T, PerturbExtras PExtras>
template <bool FindPeriod>
bool
FeatureFinder<IterType, T, PExtras>::Evaluate_PT(
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
    T &outResidual2) const
{
    const T zero = T{};
    const T one = HdrReduce(T{1.0});
    const T two = HdrReduce(T{2.0});
    const T escape2 = HdrReduce(T{4096.0});

    T SqrNearLinearRadius{};
    T SqrNearLinearRadiusScale{};
    if constexpr (FindPeriod) {
        HdrReduce(R);
        if (HdrCompareToBothPositiveReducedLE(R, zero)) {
            std::cout << "[PT Evaluate] FAIL: R <= 0\n";
            return false;
        }
        SqrNearLinearRadius = R * R;
        HdrReduce(SqrNearLinearRadius);

        const T near1 = HdrReduce(T{0.25});
        SqrNearLinearRadiusScale = near1 * near1;
        HdrReduce(SqrNearLinearRadiusScale);
    }

    // Compute dc at HIGH PRECISION, then convert to T
    const HighPrecision dcX_hp = origCX_hp - results.GetHiX();
    const HighPrecision dcY_hp = origCY_hp - results.GetHiY();
    C dc(ToDouble(dcX_hp), ToDouble(dcY_hp));
    dc.Reduce();

    const size_t refOrbitLength = static_cast<size_t>(results.GetCountOrbitEntries());
    if (refOrbitLength < 2) {
        std::cout << "[PT Evaluate] FAIL: ref orbit too short\n";
        return false;
    }

    IterTypeFull cap;
    if constexpr (FindPeriod) {
        cap = maxIters;
    } else {
        cap = static_cast<IterTypeFull>(ioPeriod);
    }

    if (cap < 1) {
        std::cout << "[PT Evaluate] FAIL: cap < 1\n";
        return false;
    }

    size_t refIteration = 0;

    C dz{};
    C z{};
    C dzdc{};
    C zcoeff{};

    dz.Reduce();
    z.Reduce();
    dzdc.Reduce();
    zcoeff.Reduce();

    const C cRef(ToDouble(results.GetHiX()), ToDouble(results.GetHiY()));
    for (IterTypeFull n = 0; n < cap; ++n) {
        if (n == 0)
            zcoeff = C(one, T{});
        else
            zcoeff = zcoeff * (z * two);
        zcoeff.Reduce();

        dzdc = dzdc * (z * two) + C(one, T{});
        dzdc.Reduce();

        const C zref = results.GetComplex(dec, refIteration);
        dz = dz * (zref + z) + dc;
        dz.Reduce();

        refIteration++;

        const C zrefNext = results.GetComplex(dec, refIteration);
        z = zrefNext + dz;
        z.Reduce();

        T dzNorm = dz.norm_squared();
        HdrReduce(dzNorm);
        T zNorm = z.norm_squared();
        HdrReduce(zNorm);

        if (refIteration >= refOrbitLength - 1 || HdrCompareToBothPositiveReducedLT(zNorm, dzNorm)) {
            dz = z;
            dz.Reduce();
            refIteration = 0;
        }

        T Magnitude = zNorm;
        if (HdrCompareToBothPositiveReducedGT(Magnitude, escape2)) {
            std::cout << "[PT Evaluate] FAIL: escape at n=" << (uint64_t)n << "\n";
            return false;
        }

        if constexpr (FindPeriod) {
            T d2 = dzdc.norm_squared();
            HdrReduce(d2);

            T rhs = SqrNearLinearRadius * d2;
            HdrReduce(rhs);

            if (HdrCompareToBothPositiveReducedLT(Magnitude, rhs)) {
                const IterTypeFull cand = n + 1;
                if (cand <= static_cast<IterTypeFull>(std::numeric_limits<IterType>::max())) {
                    ioPeriod = static_cast<IterType>(cand);
                    outDiff = z;
                    outDzdc = dzdc;
                    outZcoeff = zcoeff;
                    outResidual2 = zNorm;
                    std::cout << "[PT Evaluate] SUCCESS: found period=" << (uint64_t)cand << "\n";
                    return true;
                }
                std::cout << "[PT Evaluate] FAIL: period overflow\n";
                return false;
            }

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
    }

    if constexpr (FindPeriod) {
        std::cout << "[PT Evaluate] FAIL: no trigger found\n";
        return false;
    } else {
        outDiff = z;
        outResidual2 = z.norm_squared();
        HdrReduce(outResidual2);

        outDzdc = dzdc;
        outDzdc.Reduce();

        outZcoeff = zcoeff;
        outZcoeff.Reduce();

        std::cout << "[PT Evaluate] SUCCESS: period=" << (uint64_t)ioPeriod
                  << " residual2=" << HdrToString<false, T>(outResidual2) << "\n";
        return true;
    }
}

template <class IterType, class T, PerturbExtras PExtras>
template <class EvalPolicy>
bool
FeatureFinder<IterType, T, PExtras>::FindPeriodicPoint_Common(IterType refIters,
                                                              FeatureSummary &feature,
                                                              EvalPolicy &&evaluator) const
{
    // Get high-precision coordinates from feature
    const HighPrecision &origCX_hp = feature.GetOrigX();
    const HighPrecision &origCY_hp = feature.GetOrigY();

    const T cx0{origCX_hp};
    const T cy0{origCY_hp};
    T R{feature.GetRadius()};
    R = HdrAbs(R);
    T SqrRadius = R * R;
    HdrReduce(SqrRadius);

    const C origC{cx0, cy0};
    C c = origC;

    // Track high-precision c for Newton iteration
    HighPrecision cX_hp = origCX_hp;
    HighPrecision cY_hp = origCY_hp;

    IterType period = 0;
    C diff, dzdc, zcoeff;
    T residual2{};

    // 1) Find candidate period: Evaluate<true>
    if (!evaluator.template Eval<true>(
            origC, cX_hp, cY_hp, SqrRadius, refIters, period, diff, dzdc, zcoeff, residual2)) {
        std::cout << "Rejected: findPeriod failed\n";
        return false;
    }

    // Initial Newton correction
    const T dzdc2_init = dzdc.norm_squared();
    if (HdrCompareToBothPositiveReducedLE(dzdc2_init, T{})) {
        std::cout << "Rejected: initial dzdc too small\n";
        return false;
    }

    C step0 = Div(diff, dzdc);
    c = c - step0;

    // Update high-precision c
    cX_hp = cX_hp - HighPrecision{step0.getRe()};
    cY_hp = cY_hp - HighPrecision{step0.getIm()};

    const T relTol = m_params.RelStepTol;
    const T relTol2 = relTol * relTol;
    bool convergedByStep = false;

    // 2) Newton loop: Evaluate<false>
    for (uint32_t it = 0; it < m_params.MaxNewtonIters; ++it) {
        if (!evaluator.template Eval<false>(
                c, cX_hp, cY_hp, SqrRadius, period, period, diff, dzdc, zcoeff, residual2)) {
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
        c = c - step;

        // Update high-precision c
        cX_hp = cX_hp - HighPrecision{step.getRe()};
        cY_hp = cY_hp - HighPrecision{step.getIm()};

        T step2 = step.norm_squared();
        HdrReduce(step2);
        const T c2 = c.norm_squared();
        auto c2Prod = c2 * relTol2;
        HdrReduce(c2Prod);

        if (HdrCompareToBothPositiveReducedLE(step2, c2Prod)) {
            convergedByStep = true;
            break;
        }

        if (HdrCompareToBothPositiveReducedGT(m_params.Eps2Accept, T{}) &&
            HdrCompareToBothPositiveReducedLE(residual2, m_params.Eps2Accept)) {
            break;
        }
    }

    // Final correction
    if (!evaluator.template Eval<false>(
            c, cX_hp, cY_hp, SqrRadius, period, period, diff, dzdc, zcoeff, residual2)) {
        std::cout << "Rejected: final eval failed\n";
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

        // Update high-precision c
        cX_hp = cX_hp - HighPrecision{step.getRe()};
        cY_hp = cY_hp - HighPrecision{step.getIm()};
    }

    // Final evaluation for residual
    if (!evaluator.template Eval<false>(
            c, cX_hp, cY_hp, SqrRadius, period, period, diff, dzdc, zcoeff, residual2)) {
        std::cout << "Rejected: final eval failed\n";
        return false;
    }

    feature.SetFound(cX_hp, cY_hp, period, HDRFloat<double>{residual2});

    if (m_params.PrintResult) {
        std::cout << "Periodic point: "
                  << "orig cx=" << HdrToString<false, T>(origC.getRe())
                  << " orig cy=" << HdrToString<false, T>(origC.getIm())
                  << " new cx=" << HdrToString<false, T>(c.getRe())
                  << " new cy=" << HdrToString<false, T>(c.getIm())
                  << " period=" << static_cast<uint64_t>(period)
                  << " residual2=" << HdrToString<false, T>(residual2) << "\n";
    }

    return true;
}

// DirectEvaluator::Eval implementation
template <class IterType, class T, PerturbExtras PExtras>
template <bool FindPeriod>
bool
FeatureFinder<IterType, T, PExtras>::DirectEvaluator::Eval(const C &c,
                                                           const HighPrecision &cX_hp,
                                                           const HighPrecision &cY_hp,
                                                           T SqrRadius,
                                                           IterTypeFull maxIters,
                                                           IterType &ioPeriod,
                                                           C &outDiff,
                                                           C &outDzdc,
                                                           C &outZcoeff,
                                                           T &outResidual2) const
{
    T R = HdrSqrt(SqrRadius);
    if constexpr (FindPeriod) {
        return self->Evaluate_FindPeriod_Direct(
            c, maxIters, R, ioPeriod, outDiff, outDzdc, outZcoeff, outResidual2);
    } else {
        return self->Evaluate_PeriodResidualAndDzdc_Direct(
            c, ioPeriod, outDiff, outDzdc, outZcoeff, outResidual2);
    }
}

// PTEvaluator::Eval implementation
// PTEvaluator::Eval implementation
template <class IterType, class T, PerturbExtras PExtras>
template <bool FindPeriod>
bool
FeatureFinder<IterType, T, PExtras>::PTEvaluator::Eval(const C &c,
                                                       const HighPrecision &cX_hp, // ADD
                                                       const HighPrecision &cY_hp, // ADD
                                                       T SqrRadius,
                                                       IterTypeFull maxIters,
                                                       IterType &ioPeriod,
                                                       C &outDiff,
                                                       C &outDzdc,
                                                       C &outZcoeff,
                                                       T &outResidual2) const
{
    T R = HdrSqrt(SqrRadius);
    if (self->template Evaluate_PT<FindPeriod>(*results,
                                               *dec,
                                               c,
                                               cX_hp,
                                               cY_hp,
                                               R,
                                               maxIters,
                                               ioPeriod,
                                               outDiff,
                                               outDzdc,
                                               outZcoeff,
                                               outResidual2)) {
        return true;
    }

    if constexpr (!FindPeriod) {
        std::cout << "[PT->Direct fallback] INFO: PT eval failed; trying DIRECT at same period="
                  << (uint64_t)ioPeriod << "\n";
        return self->Evaluate_PeriodResidualAndDzdc_Direct(
            c, ioPeriod, outDiff, outDzdc, outZcoeff, outResidual2);
    }
    return false;
}

// Simplified FindPeriodicPoint (Direct)
template <class IterType, class T, PerturbExtras PExtras>
bool
FeatureFinder<IterType, T, PExtras>::FindPeriodicPoint(IterType maxIters, FeatureSummary &feature) const
{
    return FindPeriodicPoint_Common(maxIters, feature, DirectEvaluator{this});
}

// Simplified FindPeriodicPoint (PT)
template <class IterType, class T, PerturbExtras PExtras>
bool
FeatureFinder<IterType, T, PExtras>::FindPeriodicPoint(
    IterType maxIters,
    const PerturbationResults<IterType, T, PExtras> &results,
    RuntimeDecompressor<IterType, T, PExtras> &dec,
    FeatureSummary &feature) const
{
    return FindPeriodicPoint_Common(maxIters, feature, PTEvaluator{this, &results, &dec});
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
