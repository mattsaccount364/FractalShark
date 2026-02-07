//
// This feature finder logic is heavily based on the implementation in Imagina
// but likely screws up some of the details.
//

#include "stdafx.h"

#include "FeatureFinder.h"
#include "FeatureSummary.h"
#include "FloatComplex.h"
#include "HighPrecision.h"
#include "LAInfoDeep.h"
#include "LAReference.h"
#include "PerturbationResults.h"
#include "LAReference.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>

struct mpf_complex {
    mpf_t re, im;
};

static inline void
mpf_complex_init(mpf_complex &z, mp_bitcnt_t prec)
{
    mpf_init2(z.re, prec);
    mpf_init2(z.im, prec);
}

static inline void
mpf_complex_clear(mpf_complex &z)
{
    mpf_clear(z.re);
    mpf_clear(z.im);
}

static inline void
mpf_complex_set(mpf_complex &dst, const mpf_complex &src)
{
    mpf_set(dst.re, src.re);
    mpf_set(dst.im, src.im);
}

static inline void
mpf_complex_set_ui(mpf_complex &z, unsigned long re, unsigned long im)
{
    mpf_set_ui(z.re, re);
    mpf_set_ui(z.im, im);
}

// out = a*b (alias-safe)  -- provided by you
static inline void
mpf_complex_mul_safe(
    mpf_complex &out, const mpf_complex &a, const mpf_complex &b, mpf_t tr, mpf_t ti, mpf_t t1, mpf_t t2)
{
    mpf_mul(t1, a.re, b.re);
    mpf_mul(t2, a.im, b.im);
    mpf_sub(tr, t1, t2);

    mpf_mul(t1, a.re, b.im);
    mpf_mul(t2, a.im, b.re);
    mpf_add(ti, t1, t2);

    mpf_set(out.re, tr);
    mpf_set(out.im, ti);
}

// out = a^2 (alias-safe)  -- provided by you
static inline void
mpf_complex_sqr_safe(mpf_complex &out, const mpf_complex &a, mpf_t tr, mpf_t ti, mpf_t t1, mpf_t t2)
{
    mpf_mul(t1, a.re, a.re);
    mpf_mul(t2, a.im, a.im);
    mpf_sub(tr, t1, t2);

    mpf_mul(t1, a.re, a.im);
    mpf_mul_ui(ti, t1, 2);

    mpf_set(out.re, tr);
    mpf_set(out.im, ti);
}

static inline void
mpf_complex_norm(mpf_t out, const mpf_complex &z, mpf_t t1, mpf_t t2)
{
    mpf_mul(t1, z.re, z.re);
    mpf_mul(t2, z.im, z.im);
    mpf_add(out, t1, t2);
}

// Return floor(log2(x)) approx using mpf_get_d_2exp (same as your helper)
static inline int
approx_ilogb_mpf(const mpf_t x)
{
    if (mpf_cmp_ui(x, 0) == 0)
        return INT32_MIN;
    long exp2;
    (void)mpf_get_d_2exp(&exp2, x); // mantissa unused
    // mpf_get_d_2exp gives x = mant * 2^exp2, mant in [0.5,1)
    return int(exp2 - 1);
}

// ------------------------------------------------------------
// z is maintained at coord_prec.
// dzdc and d2zdc2 are maintained at deriv_prec.
// Each iter we make z_d = z rounded to deriv_prec, and use that for derivative updates.
// ------------------------------------------------------------
static inline void
EvaluateCriticalOrbitAndDerivs_mpf_mixedprec(const mpf_complex &c_coord, // coord_prec
                                             uint64_t period,
                                             mpf_complex &z_coord,      // coord_prec (output z_p)
                                             mpf_complex &dzdc_deriv,   // deriv_prec (output dzdc_p)
                                             mpf_complex &d2zdc2_deriv, // deriv_prec (output d2_p)
                                             mpf_complex &z_deriv,    // deriv_prec (scratch: z rounded)
                                             mpf_complex &tmpA_deriv, // deriv_prec
                                             mpf_complex &tmpB_deriv, // deriv_prec
                                             mpf_complex &tmpZ_coord, // coord_prec
                                             mpf_t tr_d,
                                             mpf_t ti_d,
                                             mpf_t t1_d,
                                             mpf_t t2_d, // deriv_prec scalars
                                             mpf_t tr_c,
                                             mpf_t ti_c,
                                             mpf_t t1_c,
                                             mpf_t t2_c // coord_prec scalars
)
{
    // z = 0 (coord)
    mpf_set_ui(z_coord.re, 0);
    mpf_set_ui(z_coord.im, 0);

    // dzdc = 0 (deriv)
    mpf_set_ui(dzdc_deriv.re, 0);
    mpf_set_ui(dzdc_deriv.im, 0);

    // d2 = 0 (deriv)
    mpf_set_ui(d2zdc2_deriv.re, 0);
    mpf_set_ui(d2zdc2_deriv.im, 0);

    for (uint64_t i = 0; i < period; ++i) {
        // z_deriv = z_coord rounded to deriv precision
        mpf_set(z_deriv.re, z_coord.re);
        mpf_set(z_deriv.im, z_coord.im);

        // ---- d2 <- 2*(dzdc^2 + z*d2)
        // tmpA = dzdc^2
        mpf_complex_sqr_safe(tmpA_deriv, dzdc_deriv, tr_d, ti_d, t1_d, t2_d);

        // tmpB = z * d2
        mpf_complex_mul_safe(tmpB_deriv, z_deriv, d2zdc2_deriv, tr_d, ti_d, t1_d, t2_d);

        // tmpA = tmpA + tmpB
        mpf_add(tmpA_deriv.re, tmpA_deriv.re, tmpB_deriv.re);
        mpf_add(tmpA_deriv.im, tmpA_deriv.im, tmpB_deriv.im);

        // d2 = 2 * tmpA
        mpf_mul_ui(d2zdc2_deriv.re, tmpA_deriv.re, 2);
        mpf_mul_ui(d2zdc2_deriv.im, tmpA_deriv.im, 2);

        // ---- dzdc <- 2*z*dzdc + 1
        // tmpB = 2*z
        mpf_mul_ui(tmpB_deriv.re, z_deriv.re, 2);
        mpf_mul_ui(tmpB_deriv.im, z_deriv.im, 2);

        // dzdc = dzdc * (2*z)
        mpf_complex_mul_safe(dzdc_deriv, dzdc_deriv, tmpB_deriv, tr_d, ti_d, t1_d, t2_d);

        // +1 (real)
        mpf_add_ui(dzdc_deriv.re, dzdc_deriv.re, 1);

        // ---- z <- z^2 + c   (coord precision)
        mpf_complex_sqr_safe(tmpZ_coord, z_coord, tr_c, ti_c, t1_c, t2_c);
        mpf_add(z_coord.re, tmpZ_coord.re, c_coord.re);
        mpf_add(z_coord.im, tmpZ_coord.im, c_coord.im);
    }
}

// ------------------------------------------------------------
// Complex step computation in COORD precision:
//
// step = z / dzdc  computed as  (z * conj(dzdc)) / (|dzdc|^2)
//
// dzdc is provided in deriv precision, so we promote to coord once.
// ------------------------------------------------------------
static inline bool
ComputeNewtonStep_mpf_coord_from_deriv(mpf_complex &step_coord,       // coord_prec (out)
                                       const mpf_complex &z_coord,    // coord_prec
                                       const mpf_complex &dzdc_deriv, // deriv_prec
                                       mpf_complex &dzdc_coord, // coord_prec scratch (promoted dzdc)
                                       mpf_t denom_c,
                                       mpf_t tr_c,
                                       mpf_t ti_c,
                                       mpf_t t1_c,
                                       mpf_t t2_c)
{
    // promote dzdc to coord precision
    mpf_set(dzdc_coord.re, dzdc_deriv.re);
    mpf_set(dzdc_coord.im, dzdc_deriv.im);

    // denom = br^2 + bi^2
    mpf_mul(t1_c, dzdc_coord.re, dzdc_coord.re);
    mpf_mul(t2_c, dzdc_coord.im, dzdc_coord.im);
    mpf_add(denom_c, t1_c, t2_c);
    if (mpf_cmp_ui(denom_c, 0) == 0)
        return false;

    // tr = ar*br + ai*bi
    mpf_mul(t1_c, z_coord.re, dzdc_coord.re);
    mpf_mul(t2_c, z_coord.im, dzdc_coord.im);
    mpf_add(tr_c, t1_c, t2_c);

    // ti = ai*br - ar*bi
    mpf_mul(t1_c, z_coord.im, dzdc_coord.re);
    mpf_mul(t2_c, z_coord.re, dzdc_coord.im);
    mpf_sub(ti_c, t1_c, t2_c);

    // step = (tr + i*ti) / denom
    mpf_div(step_coord.re, tr_c, denom_c);
    mpf_div(step_coord.im, ti_c, denom_c);
    return true;
}

// ------------------------------------------------------------
// Revised: Error-controlled MPF refine with mixed precision.
// Intended as a *polish* step, not a second full solver.
//
// - coord_prec: high precision for c/z/step
// - deriv_prec: smaller precision for dzdc/d2
//
// Returns number of NR iterations performed.
// ------------------------------------------------------------
static inline uint32_t
RefinePeriodicPoint_ErrorControlled_mpf_mixedprec(
    mpf_complex &c_coord, // coord_prec (in/out)
    uint64_t period,
    mp_bitcnt_t coord_prec,
    mp_bitcnt_t deriv_prec,
    uint32_t max_nr_iters
)
{
    // coord temporaries
    mpf_t denom_c, tr_c, ti_c, t1_c, t2_c;
    mpf_init2(denom_c, coord_prec);
    mpf_init2(tr_c, coord_prec);
    mpf_init2(ti_c, coord_prec);
    mpf_init2(t1_c, coord_prec);
    mpf_init2(t2_c, coord_prec);

    mpf_t normStep_c, normStep2_c, err_c, tmp_c;
    mpf_init2(normStep_c, coord_prec);
    mpf_init2(normStep2_c, coord_prec);
    mpf_init2(err_c, coord_prec);
    mpf_init2(tmp_c, coord_prec);

    // deriv temporaries
    mpf_t tr_d, ti_d, t1_d, t2_d;
    mpf_init2(tr_d, deriv_prec);
    mpf_init2(ti_d, deriv_prec);
    mpf_init2(t1_d, deriv_prec);
    mpf_init2(t2_d, deriv_prec);

    mpf_t n1_c, n2_c; // norms promoted to coord precision (for error estimator)
    mpf_init2(n1_c, coord_prec);
    mpf_init2(n2_c, coord_prec);

    // complex state
    mpf_complex z_coord, step_coord, dzdc_coord, tmpZ_coord;
    mpf_complex_init(z_coord, coord_prec);
    mpf_complex_init(step_coord, coord_prec);
    mpf_complex_init(dzdc_coord, coord_prec); // promoted dzdc for division
    mpf_complex_init(tmpZ_coord, coord_prec);

    mpf_complex dzdc_deriv, d2_deriv, z_deriv, tmpA_d, tmpB_d;
    mpf_complex_init(dzdc_deriv, deriv_prec);
    mpf_complex_init(d2_deriv, deriv_prec);
    mpf_complex_init(z_deriv, deriv_prec);
    mpf_complex_init(tmpA_d, deriv_prec);
    mpf_complex_init(tmpB_d, deriv_prec);

    const int targetExp = int(coord_prec) * 2; // similar spirit to your old target

    uint32_t it = 0;
    for (; it < max_nr_iters; ++it) {
        EvaluateCriticalOrbitAndDerivs_mpf_mixedprec(c_coord,
                                                     period,
                                                     z_coord,
                                                     dzdc_deriv,
                                                     d2_deriv,
                                                     z_deriv,
                                                     tmpA_d,
                                                     tmpB_d,
                                                     tmpZ_coord,
                                                     tr_d,
                                                     ti_d,
                                                     t1_d,
                                                     t2_d,
                                                     tr_c,
                                                     ti_c,
                                                     t1_c,
                                                     t2_c);

        // step = z / dzdc (coord), using dzdc computed at deriv precision
        if (!ComputeNewtonStep_mpf_coord_from_deriv(
                step_coord, z_coord, dzdc_deriv, dzdc_coord, denom_c, tr_c, ti_c, t1_c, t2_c)) {
            break; // singular derivative
        }

        // c <- c - step
        mpf_sub(c_coord.re, c_coord.re, step_coord.re);
        mpf_sub(c_coord.im, c_coord.im, step_coord.im);

        // --- Error estimator (optional):
        // err = ||step||^4 * ||d2|| / ||dzdc||
        //
        // Compute ||step||^2 at coord precision
        mpf_complex_norm(normStep_c, step_coord, t1_c, t2_c);
        mpf_mul(normStep2_c, normStep_c, normStep_c);

        // ||d2||, ||dzdc|| computed at deriv precision, promote via mpf_set into coord vars
        // n2_c = norm(d2)
        {
            mpf_t t1p, t2p;
            mpf_init2(t1p, coord_prec);
            mpf_init2(t2p, coord_prec);

            mpf_set(t1p, d2_deriv.re);
            mpf_set(t2p, d2_deriv.im);
            mpf_mul(t1_c, t1p, t1p);
            mpf_mul(t2_c, t2p, t2p);
            mpf_add(n2_c, t1_c, t2_c);

            mpf_set(t1p, dzdc_deriv.re);
            mpf_set(t2p, dzdc_deriv.im);
            mpf_mul(t1_c, t1p, t1p);
            mpf_mul(t2_c, t2p, t2p);
            mpf_add(n1_c, t1_c, t2_c);

            mpf_clear(t1p);
            mpf_clear(t2p);
        }

        if (mpf_cmp_ui(n1_c, 0) == 0)
            break;

        mpf_mul(tmp_c, normStep2_c, n2_c);
        mpf_div(err_c, tmp_c, n1_c);

        const int e = approx_ilogb_mpf(err_c);
        if (-e >= targetExp) {
            // One final polish (optional)
            EvaluateCriticalOrbitAndDerivs_mpf_mixedprec(c_coord,
                                                         period,
                                                         z_coord,
                                                         dzdc_deriv,
                                                         d2_deriv,
                                                         z_deriv,
                                                         tmpA_d,
                                                         tmpB_d,
                                                         tmpZ_coord,
                                                         tr_d,
                                                         ti_d,
                                                         t1_d,
                                                         t2_d,
                                                         tr_c,
                                                         ti_c,
                                                         t1_c,
                                                         t2_c);
            if (ComputeNewtonStep_mpf_coord_from_deriv(
                    step_coord, z_coord, dzdc_deriv, dzdc_coord, denom_c, tr_c, ti_c, t1_c, t2_c)) {
                mpf_sub(c_coord.re, c_coord.re, step_coord.re);
                mpf_sub(c_coord.im, c_coord.im, step_coord.im);
            }
            ++it;
            break;
        }
    }

    // cleanup
    mpf_clear(denom_c);
    mpf_clear(tr_c);
    mpf_clear(ti_c);
    mpf_clear(t1_c);
    mpf_clear(t2_c);
    mpf_clear(normStep_c);
    mpf_clear(normStep2_c);
    mpf_clear(err_c);
    mpf_clear(tmp_c);
    mpf_clear(tr_d);
    mpf_clear(ti_d);
    mpf_clear(t1_d);
    mpf_clear(t2_d);
    mpf_clear(n1_c);
    mpf_clear(n2_c);

    mpf_complex_clear(z_coord);
    mpf_complex_clear(step_coord);
    mpf_complex_clear(dzdc_coord);
    mpf_complex_clear(tmpZ_coord);

    mpf_complex_clear(dzdc_deriv);
    mpf_complex_clear(d2_deriv);
    mpf_complex_clear(z_deriv);
    mpf_complex_clear(tmpA_d);
    mpf_complex_clear(tmpB_d);

    return it;
}

// ------------------------------------------------------------
// - coord_prec is the real “final precision” you care about for c
// - deriv_prec is chosen smaller to save work (heuristic)
// - iterations are few: this is meant to be a polish after your HDR Newton loop
// ------------------------------------------------------------
template <class IterType, class T, PerturbExtras PExtras>
IterTypeFull
FeatureFinder<IterType, T, PExtras>::RefinePeriodicPoint_WithMPF(HighPrecision &cX_hp,
                                                                 HighPrecision &cY_hp,
                                                                 IterType period,
                                                                 mp_bitcnt_t coord_prec) const
{
    // Heuristic derivative precision (tune!):
    // Start around ~half precision + headroom, capped to coord_prec.
    mp_bitcnt_t deriv_prec = coord_prec;
    {
        const mp_bitcnt_t half = coord_prec / 2;
        const mp_bitcnt_t headroom = 128;
        deriv_prec = std::min(coord_prec, half + headroom);
        deriv_prec = std::max<mp_bitcnt_t>(deriv_prec, 256); // don’t go too low
    }

    mpf_complex c;
    mpf_complex_init(c, coord_prec);

    // Seed from HighPrecision backends
    mpf_set(c.re, (mpf_srcptr)cX_hp.backend());
    mpf_set(c.im, (mpf_srcptr)cY_hp.backend());

    const uint32_t max_polish = 32;

    const uint32_t iters = RefinePeriodicPoint_ErrorControlled_mpf_mixedprec(
        c, (uint64_t)period, coord_prec, deriv_prec, max_polish);

    // Write back
    cX_hp = HighPrecision{c.re};
    cY_hp = HighPrecision{c.im};

    mpf_complex_clear(c);
    return (IterTypeFull)iters;
}















// Periodicity state for periodic-point detection
// All quantities are *squared* magnitudes.
//
// Imagina meanings:
//   Magnitude = norm(z)
//   norm(dzdc) = norm(dzdc)
//   SqrNearLinearRadius starts at R^2 and tightens over time.
//   SqrNearLinearRadiusScale = (0.25)^2
template <class IterType, class T, class C> struct PeriodicityPP {
    T SqrNearLinearRadius{};      // dynamic tightened R^2
    T SqrNearLinearRadiusScale{}; // (0.25)^2

    CUDA_CRAP void
    Init(T R)
    {
        const T zero = T{};
        const T near1 = HdrReduce(T{0.25});

        HdrReduce(R);
        if (HdrCompareToBothPositiveReducedLE(R, zero)) {
            // caller handles failure
            SqrNearLinearRadius = T{};
            SqrNearLinearRadiusScale = T{};
            return;
        }

        SqrNearLinearRadius = R * R;
        HdrReduce(SqrNearLinearRadius);

        SqrNearLinearRadiusScale = near1 * near1;
        HdrReduce(SqrNearLinearRadiusScale);
    }

    // Returns true if a period is detected at "Iteration".
    // Updates tightening every call.
    CUDA_CRAP bool
    CheckPeriodicity(const T &Magnitude,
                     const T &DzdcNormSq,
                     IterType Iteration,
                     IterType &outPrePeriod,
                     IterType &outPeriod)
    {
        // Trigger: Magnitude < SqrNearLinearRadius * norm(dzdc)
        T rhs = SqrNearLinearRadius * DzdcNormSq;
        HdrReduce(rhs);

        if (HdrCompareToBothPositiveReducedLT(Magnitude, rhs)) {
            outPrePeriod = 0;
            outPeriod = Iteration;
            return true;
        }

        // Tighten: if Magnitude * scale < SqrNearLinearRadius * norm(dzdc)
        // then SqrNearLinearRadius = Magnitude * scale / norm(dzdc)
        const T zero = T{};
        if (HdrCompareToBothPositiveReducedGT(DzdcNormSq, zero)) {
            T lhsTight = Magnitude * SqrNearLinearRadiusScale;
            HdrReduce(lhsTight);

            if (HdrCompareToBothPositiveReducedLT(lhsTight, rhs)) {
                T newSqr = lhsTight / DzdcNormSq;
                HdrReduce(newSqr);

                if (HdrCompareToBothPositiveReducedGT(newSqr, zero)) {
                    SqrNearLinearRadius = newSqr;
                }
            }
        }

        return false;
    }
};


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
    T denom = br * br + bi * bi;
    HdrReduce(denom);

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
    const T zero = T{};
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

    const T one = HdrReduce(T{1.0});
    const T two = HdrReduce(T{2.0});
    const T escape2 = HdrReduce(T{4096.0});

    C oneC{one, T{}};
    C zcoeff{}; // will be set properly below

    oneC.Reduce();

    for (IterType i = 0; i < period; ++i) {
        // IMPORTANT: match Evaluate_FindPeriod_Direct ordering:
        // zcoeff = 1 at i==0, else zcoeff *= (2*z) using CURRENT z before update
        if (i == 0) {
            zcoeff = C(one, T{});
        } else {
            zcoeff = zcoeff * (z * two);
        }
        zcoeff.Reduce();

        dzdc = dzdc * (z * two) + oneC;
        dzdc.Reduce();

        z = (z * z) + c;
        z.Reduce();

        T normSq = z.norm_squared();
        HdrReduce(normSq);

        if (HdrCompareToBothPositiveReducedGT(normSq, escape2)) {
            std::cout << "FeatureFinder::Evaluate_PeriodResidualAndDzdc_Direct: orbit escaped.\n";
            return false;
        }
    }

    outDiff = z;
    outResidual2 = outDiff.norm_squared();
    HdrReduce(outResidual2);

    outDzdc = dzdc;
    outZcoeff = zcoeff;

    return true;
}

template <class IterType, class T, PerturbExtras PExtras>
HighPrecision
FeatureFinder<IterType, T, PExtras>::ComputeIntrinsicRadius_HP(const C &zcoeff, const C &dzdc) const
{
    // Imagina: Scale = 1 / |zcoeff * dzdc|; radius = Scale * 4
    // We compute: |w| = sqrt(norm_squared(w)) with w = zcoeff*dzdc
    const T one = HdrReduce(T{1.0});
    const T four = HdrReduce(T{4.0});

    C w = zcoeff * dzdc;
    w.Reduce();

    T w2 = w.norm_squared(); // |w|^2
    HdrReduce(w2);

    if (HdrCompareToBothPositiveReducedLE(w2, T{})) {
        // Degenerate; return 0 so caller can fall back to something else if desired
        return HighPrecision{0};
    }

    T absW = HdrSqrt(w2); // |w|
    HdrReduce(absW);

    T scale = one / absW; // 1/|w|
    HdrReduce(scale);

    T radT = scale * four; // 4/|w|
    HdrReduce(radT);

    // Stay in HighPrecision for storage/zoom
    return HighPrecision{radT};
}

// Build a complex scalar (s + 0i) reduced.
template <class IterType, class T, PerturbExtras PExtras>
static inline typename FeatureFinder<IterType, T, PExtras>::C
MakeRealC(const T &s)
{
    using C = typename FeatureFinder<IterType, T, PExtras>::C;
    C out(s, T{});
    out.Reduce();
    return out;
}


template <class IterType, class T, PerturbExtras PExtras>
template <bool FindPeriod>
bool
FeatureFinder<IterType, T, PExtras>::Evaluate_PT(
    const PerturbationResults<IterType, T, PExtras> &results,
    RuntimeDecompressor<IterType, T, PExtras> &dec,
    const HighPrecision &cX_hp,
    const HighPrecision &cY_hp,
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

    // dc = current c - reference center (HP -> T)
    const HighPrecision dcX_hp = cX_hp - results.GetHiX();
    const HighPrecision dcY_hp = cY_hp - results.GetHiY();
    C dc(ToDouble(dcX_hp), ToDouble(dcY_hp));
    dc.Reduce();

    const size_t refOrbitLength = (size_t)results.GetCountOrbitEntries();
    if (refOrbitLength < 2)
        return false;

    const IterTypeFull cap = FindPeriod ? maxIters : (IterTypeFull)ioPeriod;
    if (cap < 1)
        return false;

    int scaleExp = 0;
    const T ScalingFactor = HdrLdexp(one, -scaleExp);   // 2^-scaleExp
    const T InvScalingFactor = HdrLdexp(one, scaleExp); // 2^scaleExp

    const C ScalingFactorC(ScalingFactor, T{});
    const C InvScalingFactorC(InvScalingFactor, T{});

    // Precompute InvScalingFactor^2 for periodicity math (true dzdc norm)
    T InvScale2 = InvScalingFactor * InvScalingFactor;
    HdrReduce(InvScale2);

    PeriodicityPP<IterType, T, C> pp;
    IterType prePeriod = 0;
    IterType period = 0;

    if constexpr (FindPeriod) {
        HdrReduce(R);
        if (HdrCompareToBothPositiveReducedLE(R, zero))
            return false;
        pp.Init(R);
        // pp.Init already builds R^2, scale = (0.25)^2
    }

    size_t refIteration = 0;
    C dz{}, z{};
    dz.Reduce();
    z.Reduce();

    // Stored scaled dzdc/zcoeff:
    C dzdc{};   // stored = true * ScalingFactor
    C zcoeff{}; // stored = true * ScalingFactor
    dzdc.Reduce();
    zcoeff.Reduce();

    for (IterTypeFull n = 0; n < cap; ++n) {
        // if Iteration==0 zcoeff = ScalingFactor else zcoeff *= 2*z
        if (n == 0) {
            zcoeff = ScalingFactorC;
        } else {
            zcoeff = zcoeff * (z * two);
        }
        zcoeff.Reduce();

        // scaled dzdc: dzdc = dzdc*(2z) + ScalingFactor
        dzdc = dzdc * (z * two) + ScalingFactorC;
        dzdc.Reduce();

        // PT delta recurrence
        const C zref = results.GetComplex(dec, refIteration);
        dz = dz * (zref + z) + dc;
        dz.Reduce();

        refIteration++;
        const C zrefNext = results.GetComplex(dec, refIteration);
        z = zrefNext + dz;
        z.Reduce();

        // Rebasing check
        T dzNorm = dz.norm_squared();
        HdrReduce(dzNorm);
        T zNorm = z.norm_squared();
        HdrReduce(zNorm);

        if (refIteration >= refOrbitLength - 1 || HdrCompareToBothPositiveReducedLT(zNorm, dzNorm)) {
            dz = z;
            dz.Reduce();
            refIteration = 0;
        }

        // Escape check
        if (HdrCompareToBothPositiveReducedGT(zNorm, escape2)) {
            return false;
        }

        if constexpr (FindPeriod) {
            // Imagina uses:
            //   Magnitude = norm(z)
            //   norm(dzdc) in trigger is true dzdc norm
            //
            // dzdc is stored scaled, so convert norm to true:
            T dzdcNormStored = dzdc.norm_squared();
            HdrReduce(dzdcNormStored);

            T dzdcNormTrue = dzdcNormStored * InvScale2;
            HdrReduce(dzdcNormTrue);

            // Iteration increments then checks; your n is 0-based
            // At end of loop, we've computed z_{n+1}, dzdc at same time.
            const IterType Iteration = (IterType)(n + 1);

            if (pp.CheckPeriodicity(zNorm, dzdcNormTrue, Iteration, prePeriod, period)) {
                ioPeriod = period; // prePeriod is always 0 here

                // Unscale outputs (to match your direct conventions)
                outDiff = z;
                outDzdc = dzdc * InvScalingFactorC;
                outZcoeff = zcoeff * InvScalingFactorC;
                outDiff.Reduce();
                outDzdc.Reduce();
                outZcoeff.Reduce();

                outResidual2 = zNorm;
                return true;
            }
        }
    }

    if constexpr (FindPeriod) {
        return false;
    }

    // Fixed-period path: unscale outputs
    outDiff = z;
    outResidual2 = z.norm_squared();
    HdrReduce(outResidual2);

    outDzdc = dzdc * InvScalingFactorC;
    outZcoeff = zcoeff * InvScalingFactorC;
    outDiff.Reduce();
    outDzdc.Reduce();
    outZcoeff.Reduce();
    return true;
}


// =====================================================================================
// LA Evaluation - fixed to match Imagina semantics:
//
// Key fixes vs your previous version:
//  1) Apply the same ScalingFactor / InvScalingFactor contract as PT/direct.
//  2) Update zcoeff/dzdc from *dz* (the delta state) and do it in the Imagina order:
//        Prepare -> (zcoeff update) -> (dzdc update) -> dz update -> advance iteration/ref -> z update
//        -> periodicity
//  3) Periodicity uses TRUE ||dzdc|| (unscale via InvScale2) and uses the shared PeriodicityPP
//  tightening logic. 4) Outputs are unscaled (multiply by InvScalingFactor) to match your direct
//  conventions.
//
// NOTE: This assumes LAstep exposes methods analogous to Imagina:
//
//   - las.Prepare(dz, dc)  (or Prepare is done inside laRef.getLA; in that case you can ignore temp)
//   - las.EvaluateDzdzDeep(dz, zcoeff, dc, ScalingFactorC)   // updates zcoeff for this macro-step
//   - las.EvaluateDzdcDeep(dz, dzdc, dc, ScalingFactorC)     // updates dzdc for this macro-step
//   - las.Evaluate(dz, dc) OR las.Evaluate(dc) returning new dz
//
// If your current LAstep only has Evaluate(dc) and EvaluateDzdcDeep(z, x),
// you should *split* that API so zcoeff/dzdc updates take (dz, dc, ScalingFactor),
// not absolute z, and not the previous-step z.
// =====================================================================================
template <class IterType, class T, PerturbExtras PExtras>
template <bool FindPeriod>
bool
FeatureFinder<IterType, T, PExtras>::Evaluate_LA(
    const PerturbationResults<IterType, T, PExtras> &results,
    LAReference<IterType, T, SubType, PExtras> &laRef,
    const HighPrecision &cX_hp,
    const HighPrecision &cY_hp,
    T R,
    IterTypeFull maxIters,
    IterType &ioPeriod,
    C &outDiff,
    C &outDzdc,
    C &outZcoeff,
    T &outResidual2) const
{
    if (!laRef.IsValid()) {
        std::cout << "[LA Evaluate] FAIL: LAReference not valid\n";
        return false;
    }

    const T zero = T{};
    const T one = HdrReduce(T{1.0});
    const T two = HdrReduce(T{2.0});
    const T escape2 = HdrReduce(T{4096.0});

    // --------------------------
    // Radius / periodicity state
    // --------------------------
    PeriodicityPP<IterType, T, C> pp;
    IterType prePeriod = 0;
    IterType period = 0;

    if constexpr (FindPeriod) {
        HdrReduce(R);
        if (HdrCompareToBothPositiveReducedLE(R, zero)) {
            std::cout << "[LA Evaluate] FAIL: R <= 0\n";
            return false;
        }
        pp.Init(R);
    }

    // --------------------------
    // dc = c - referenceCenter
    // --------------------------
    const HighPrecision dcX_hp = cX_hp - results.GetHiX();
    const HighPrecision dcY_hp = cY_hp - results.GetHiY();
    C dc(ToDouble(dcX_hp), ToDouble(dcY_hp));
    dc.Reduce();

    // --------------------------
    // Cap
    // --------------------------
    IterTypeFull cap;
    if constexpr (FindPeriod) {
        cap = maxIters;
    } else {
        cap = static_cast<IterTypeFull>(ioPeriod);
    }

    if (cap < 1) {
        std::cout << "[LA Evaluate] FAIL: cap < 1\n";
        return false;
    }

    // --------------------------
    // ScalingFactor contract (same as PT)
    // --------------------------
    int scaleExp = 0;
    const T ScalingFactor = HdrLdexp(one, -scaleExp);   // 2^-scaleExp
    const T InvScalingFactor = HdrLdexp(one, scaleExp); // 2^scaleExp

    const C ScalingFactorC(ScalingFactor, T{});
    const C InvScalingFactorC(InvScalingFactor, T{});

    // For periodicity: TRUE ||dzdc||^2 = STORED ||dzdc||^2 * (InvScalingFactor^2)
    T InvScale2 = InvScalingFactor * InvScalingFactor;
    HdrReduce(InvScale2);

    // --------------------------
    // State (matches Imagina)
    // --------------------------
    C dz{};
    dz.Reduce(); // delta state
    C z{};
    z.Reduce(); // absolute z = ref + dz
    C dzdc{};
    dzdc.Reduce(); // STORED scaled: true * ScalingFactor
    C zcoeff{};
    zcoeff.Reduce(); // STORED scaled: true * ScalingFactor

    IterTypeFull iteration = 0;
    const IterType laStageCount = laRef.GetLAStageCount();

    // Process LA stages from highest to lowest (coarse to fine)
    for (IterType currentLAStage = laStageCount; currentLAStage > 0 && iteration < cap;) {
        --currentLAStage;

        const IterType laIndex = laRef.getLAIndex(currentLAStage);
        const IterType macroItCount = laRef.getMacroItCount(currentLAStage);

        // Stage validity check (Imagina: ChebyshevNorm(dc) >= LA[0].LAThresholdC => jump out)
        if (laRef.isLAStageInvalid(laIndex, dc)) {
            continue;
        }

        IterType refIteration = 0;

        while (iteration < cap) {
            // Get the LA step description for this (laIndex, refIteration, iteration)
            // NOTE: laRef.getLA takes dz and returns an LAstep which contains:
            //       - unusable
            //       - step (macro step length)
            //       - access to ZCoeff and the reference z
            auto las = laRef.getLA(
                laIndex, dz, refIteration, static_cast<IterType>(iteration), static_cast<IterType>(cap));

            if (las.unusable) {
                // In Imagina, unusable triggers a jump to a different LA index for the same stage.
                // If/when your LAInfoI exposes NextStageLAIndex, do it here.
                // For now, fall through to try the next (finer) stage (or PT fallback).
                break;
            }

            // ------------------------------------------------------------
            // Imagina order:
            //   Prepare(dz, dc) -> update zcoeff -> update dzdc -> update dz -> advance it/ref ->
            //   compute z -> periodicity
            //
            // IMPORTANT: these updates must depend on (dz, dc, ScalingFactor), NOT absolute z.
            // ------------------------------------------------------------

            // zcoeff (scaled)
            if (iteration == 0) {
                // Imagina: zcoeff = ScalingFactor * LA[RefIteration].ZCoeff;
                zcoeff = ScalingFactorC * las.LAjdeep->getZCoeff();
            } else {
                // Imagina: LA[RefIteration].EvaluateDzdz(temp, dz, zcoeff, dc, ScalingFactor)
                las.EvaluateDzdz(dz, zcoeff, dc);
            }
            zcoeff.Reduce();

            // dzdc (scaled)
            // Imagina: LA[RefIteration].EvaluateDzdc(temp, dz, dzdc, dc, ScalingFactor)
            las.EvaluateDzdcDeep(dz, dzdc, ScalingFactor);
            dzdc.Reduce();

            // dz update
            // Imagina: LA[RefIteration].Evaluate(temp, dz, dc)
            // If your LAstep currently only provides Evaluate(dc) returning a new dz, that’s fine:
            // just make sure it’s equivalent to Evaluate(temp, dz, dc).
            dz = las.Evaluate(dz, dc);
            dz.Reduce();

            // Advance iteration/refIteration (Imagina: Iteration += StepLength; RefIteration++; z = dz +
            // Ref[RefIteration])
            iteration += las.step;
            refIteration++;

            // Compute z (absolute) from reference + dz
            z = las.getZ(dz);
            z.Reduce();

            // Rebasing check (Imagina: if RefIteration == MacroItCount || ChebyshevNorm(dz) >
            // ChebyshevNorm(z))
            if (refIteration >= macroItCount ||
                HdrCompareToBothPositiveReducedGT(ChebAbs(dz), ChebAbs(z))) {
                dz = z;
                dz.Reduce();
                refIteration = 0;
            }

            // Escape check
            T zNorm = z.norm_squared();
            HdrReduce(zNorm);

            if (HdrCompareToBothPositiveReducedGT(zNorm, escape2)) {
                std::cout << "[LA Evaluate] FAIL: escape at iteration=" << iteration << "\n";
                return false;
            }

            // Period detection
            if constexpr (FindPeriod) {
                // Use TRUE dzdc norm for periodicity (Imagina does norm(dzdc) in true space)
                T dzdcNormStored = dzdc.norm_squared();
                HdrReduce(dzdcNormStored);

                T dzdcNormTrue = dzdcNormStored * InvScale2;
                HdrReduce(dzdcNormTrue);

                const IterType IterationIt = static_cast<IterType>(iteration);

                if (pp.CheckPeriodicity(zNorm, dzdcNormTrue, IterationIt, prePeriod, period)) {
                    ioPeriod = period; // (prePeriod always 0 for periodic-point path)

                    // Unscale outputs to match DIRECT conventions
                    outDiff = z;
                    outDzdc = dzdc * InvScalingFactorC;
                    outZcoeff = zcoeff * InvScalingFactorC;

                    outDiff.Reduce();
                    outDzdc.Reduce();
                    outZcoeff.Reduce();

                    outResidual2 = zNorm;
                    return true;
                }
            }
        } // while iteration < cap
    } // for stages

    // If LA didn't reach cap, let caller fall back (PT/direct)
    if (iteration < cap) {
        std::cout << "[LA Evaluate] INFO: LA incomplete at iteration=" << iteration
                  << ", needs fallback\n";
        return false;
    }

    if constexpr (FindPeriod) {
        std::cout << "[LA Evaluate] FAIL: no trigger found\n";
        return false;
    }

    // Fixed-period path: return final state (unscaled) at ioPeriod==cap
    outDiff = z;
    outResidual2 = z.norm_squared();
    HdrReduce(outResidual2);

    outDzdc = dzdc * InvScalingFactorC;
    outZcoeff = zcoeff * InvScalingFactorC;

    outDiff.Reduce();
    outDzdc.Reduce();
    outZcoeff.Reduce();

    std::cout << "[LA Evaluate] SUCCESS: period=" << ioPeriod
              << " residual2=" << HdrToString<false, T>(outResidual2) << "\n";
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

// Add after the PTEvaluator::Eval implementation (around line 542)

// LAEvaluator::Eval implementation
template <class IterType, class T, PerturbExtras PExtras>
template <bool FindPeriod>
bool
FeatureFinder<IterType, T, PExtras>::LAEvaluator::Eval(const C &c,
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

    // Try LA first if valid
    if (laRef->IsValid()) {
        if (self->template Evaluate_LA<FindPeriod>(*results,
                                                   *laRef,
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
    }

    // Fall back to PT
    if (self->template Evaluate_PT<FindPeriod>(*results,
                                               *dec,
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

    // Final fallback to direct (for non-FindPeriod case)
    if constexpr (!FindPeriod) {
        std::cout << "[LA->Direct fallback] INFO: LA/PT eval failed; trying DIRECT at period="
                  << (uint64_t)ioPeriod << "\n";
        return self->Evaluate_PeriodResidualAndDzdc_Direct(
            c, ioPeriod, outDiff, outDzdc, outZcoeff, outResidual2);
    }
    return false;
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

// FindPeriodicPoint with LA support
template <class IterType, class T, PerturbExtras PExtras>
bool
FeatureFinder<IterType, T, PExtras>::FindPeriodicPoint(
    IterType maxIters,
    const PerturbationResults<IterType, T, PExtras> &results,
    RuntimeDecompressor<IterType, T, PExtras> &dec,
    LAReference<IterType, T, SubType, PExtras> &laRef,
    FeatureSummary &feature) const
{
    return FindPeriodicPoint_Common(maxIters, feature, LAEvaluator{this, &results, &dec, &laRef});
}

template <class IterType, class T, PerturbExtras PExtras>
template <class EvalPolicy>
bool
FeatureFinder<IterType, T, PExtras>::FindPeriodicPoint_Common(IterType refIters,
                                                              FeatureSummary &feature,
                                                              EvalPolicy &&evaluator) const
{
    const HighPrecision &origCX_hp = feature.GetOrigX();
    const HighPrecision &origCY_hp = feature.GetOrigY();

    // Search radius (T-space), but we keep c updates in HP and regenerate T c each time.
    T R{feature.GetRadius()};
    R = HdrAbs(R);
    T SqrRadius = R * R;
    HdrReduce(SqrRadius);

    // Canonical parameter in HP (ONLY updated in HP to avoid drift)
    HighPrecision cX_hp = origCX_hp;
    HighPrecision cY_hp = origCY_hp;

    // T-space parameter used by evaluators (ALWAYS derived from HP)
    auto MakeCTFromHP = [&]() -> C {
        C out(ToDouble(cX_hp), ToDouble(cY_hp));
        out.Reduce();
        return out;
    };

    const C origC(ToDouble(origCX_hp), ToDouble(origCY_hp));
    C c = MakeCTFromHP();

    IterType period = 0;
    C diff{}, dzdc{}, zcoeff{};
    T residual2{};

    // -----------------------------------------------------------------------------
    // 1) Find candidate period
    // -----------------------------------------------------------------------------
    if (!evaluator.template Eval<true>(
            c, cX_hp, cY_hp, SqrRadius, refIters, period, diff, dzdc, zcoeff, residual2)) {
        std::cout << "Rejected: findPeriod failed\n";
        return false;
    }

    // -----------------------------------------------------------------------------
    // 2) Initial Newton correction: c <- c - diff/dzdc  (HP only)
    // -----------------------------------------------------------------------------
    {
        T dzdc2_init = dzdc.norm_squared();
        HdrReduce(dzdc2_init);
        if (HdrCompareToBothPositiveReducedLE(dzdc2_init, T{})) {
            std::cout << "Rejected: initial dzdc too small\n";
            return false;
        }

        C step0 = Div(diff, dzdc);
        step0.Reduce();

        cX_hp = cX_hp - HighPrecision{step0.getRe()};
        cY_hp = cY_hp - HighPrecision{step0.getIm()};

        c = MakeCTFromHP();
    }

    // -----------------------------------------------------------------------------
    // Tolerances
    // -----------------------------------------------------------------------------
    const T relTol = m_params.RelStepTol;
    T relTol2 = relTol * relTol;
    HdrReduce(relTol2);

    // Periodicity convergence tolerance:
    //   stop if |diff|^2 <= |c|^2 * tol^2
    const T diffTol = HdrReduce(T{0x1p-40});
    T diffTol2 = diffTol * diffTol;
    HdrReduce(diffTol2);

    // -----------------------------------------------------------------------------
    // 3) Newton loop
    // -----------------------------------------------------------------------------
    for (uint32_t it = 0; it < m_params.MaxNewtonIters; ++it) {
        // Always evaluate at c derived from HP
        c = MakeCTFromHP();

        if (!evaluator.template Eval<false>(
                c, cX_hp, cY_hp, SqrRadius, period, period, diff, dzdc, zcoeff, residual2)) {
            std::cout << "Rejected: evalAtPeriod failed in loop\n";
            return false;
        }

        {
            T diff2 = diff.norm_squared();
            HdrReduce(diff2);

            T c2 = c.norm_squared();
            HdrReduce(c2);

            T rhs = c2 * diffTol2;
            HdrReduce(rhs);

            if (HdrCompareToBothPositiveReducedLE(diff2, rhs)) {
                // Residual is small enough relative to current c => converged
                break;
            }
        }

        // dzdc must be non-degenerate
        T dzdc2 = dzdc.norm_squared();
        HdrReduce(dzdc2);
        if (HdrCompareToBothPositiveReducedLE(dzdc2, T{})) {
            std::cout << "Rejected: dzdc too small\n";
            return false;
        }

        // Newton step in T-space
        C step = Div(diff, dzdc);
        step.Reduce();

        // Update ONLY HP
        cX_hp = cX_hp - HighPrecision{step.getRe()};
        cY_hp = cY_hp - HighPrecision{step.getIm()};

        // Rebuild T-space c from HP
        c = MakeCTFromHP();

        // Optional: keep your original step-based stop as a secondary criterion
        {
            T step2 = step.norm_squared();
            HdrReduce(step2);

            T c2 = c.norm_squared();
            HdrReduce(c2);

            T rhs = c2 * relTol2;
            HdrReduce(rhs);

            if (HdrCompareToBothPositiveReducedLE(step2, rhs)) {
                break;
            }
        }

        // Optional absolute residual accept (if you already use this)
        if (HdrCompareToBothPositiveReducedGT(m_params.Eps2Accept, T{}) &&
            HdrCompareToBothPositiveReducedLE(residual2, m_params.Eps2Accept)) {
            break;
        }
    }

    // -----------------------------------------------------------------------------
    // Final correction pass (same idea)
    // -----------------------------------------------------------------------------
    c = MakeCTFromHP();

    if (!evaluator.template Eval<false>(
            c, cX_hp, cY_hp, SqrRadius, period, period, diff, dzdc, zcoeff, residual2)) {
        std::cout << "Rejected: final eval failed\n";
        return false;
    }

    {
        T dzdc2_final = dzdc.norm_squared();
        HdrReduce(dzdc2_final);
        if (HdrCompareToBothPositiveReducedLE(dzdc2_final, T{})) {
            std::cout << "Rejected: final dzdc too small\n";
            return false;
        }

        C step = Div(diff, dzdc);
        step.Reduce();

        cX_hp = cX_hp - HighPrecision{step.getRe()};
        cY_hp = cY_hp - HighPrecision{step.getIm()};

        c = MakeCTFromHP();
    }

    // Final eval for residual and scale
    if (!evaluator.template Eval<false>(
            c, cX_hp, cY_hp, SqrRadius, period, period, diff, dzdc, zcoeff, residual2)) {
        std::cout << "Rejected: final eval failed\n";
        return false;
    }

{
        // -------------------------------------------------------------------------
        // Imagina-style precision selection (without needing intrinsicRadius yet):
        //
        // Imagina uses Feature.Scale.Exponent where
        //   Scale = 1 / |zcoeff * dzdc|
        //
        // We can compute the same thing right here from the current zcoeff & dzdc
        // and choose mpf coordinate precision from its exponent.
        // -------------------------------------------------------------------------

        // Compute w = zcoeff * dzdc in T-space (already available here)
        C w = zcoeff * dzdc;
        w.Reduce();

        T w2 = w.norm_squared(); // |w|^2
        HdrReduce(w2);

        mp_bitcnt_t prec_bits = cX_hp.precisionInBits(); // baseline: never go below current HP

        if (HdrCompareToBothPositiveReducedGT(w2, T{})) {
            // |w| = sqrt(|w|^2)
            T absW = HdrSqrt(w2);
            HdrReduce(absW);

            // Scale = 1/|w|
            // We only need its exponent-ish to choose precision.
            T scaleT = HdrReduce(T{1.0}) / absW;
            HdrReduce(scaleT);

            // Convert to HighPrecision to access exponent (matches your HighPrecision conventions)
            HighPrecision scaleHP{scaleT}; // Scale ≈ mant * 2^exp

            // If Scale is tiny/huge, exponent tells us required bits.
            // Required coordinate bits ~ -Scale.Exponent + marginBits
            // but for coordinate precision we want the full "-exp" order.)
            long exp_long;
            double mant;
            scaleHP.frexp(mant, exp_long);
            int exp2 = static_cast<int>(exp_long);

            // If scale = 2^exp2 * mant, then resolving to ~scale needs about -exp2 bits.
            // (If exp2 is positive, scale is large => not many bits needed from scale alone.)
            const int bitsFromScale = std::max(0, -exp2);
            const int marginBits = 256;

            mp_bitcnt_t want = (mp_bitcnt_t)(bitsFromScale + marginBits);

            // Don't go below current HP precision
            want = std::max(want, cX_hp.precisionInBits());

            // Keep a sane floor for MPF (avoid tiny precisions)
            want = std::max<mp_bitcnt_t>(want, 512);

            // Optional: cap runaway precision if you want
            // want = std::min<mp_bitcnt_t>(want, 200000);

            prec_bits = want;
        } else {
            // Degenerate w2 (shouldn't happen for valid dzdc), fall back conservatively
            prec_bits = std::max<mp_bitcnt_t>(prec_bits, 2048);
        }

        auto refineIters = RefinePeriodicPoint_WithMPF(cX_hp, cY_hp, period, prec_bits);

        // IMPORTANT: after MPF modifies HP, rebuild c in T-space and refresh outputs
        c = MakeCTFromHP();

        std::cout << "MPF refinement iters: " << refineIters << ", prec_bits=" << (uint64_t)prec_bits
                  << ", new cX_hp=" << cX_hp.str() << ", new cY_hp=" << cY_hp.str() << "\n";

        if (!evaluator.template Eval<false>(
                c, cX_hp, cY_hp, SqrRadius, period, period, diff, dzdc, zcoeff, residual2)) {
            std::cout << "Rejected: eval failed after MPF refine\n";
            return false;
        }
    }

    const HighPrecision intrinsicRadius = ComputeIntrinsicRadius_HP(zcoeff, dzdc);

    feature.SetFound(cX_hp, cY_hp, period, HDRFloat<double>{residual2}, intrinsicRadius);

    if (m_params.PrintResult) {
        std::cout << "Periodic point: "
                  << " orig cx=" << HdrToString<false, T>(origC.getRe())
                  << " orig cy=" << HdrToString<false, T>(origC.getIm())
                  << " new cx=" << HdrToString<false, T>(c.getRe())
                  << " new cy=" << HdrToString<false, T>(c.getIm())
                  << " period=" << static_cast<uint64_t>(period)
                  << " residual2=" << HdrToString<false, T>(residual2)
                  << " intrinsicRadius=" << intrinsicRadius.str() << "\n";
    }

    return true;
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
//InstantiatePeriodicPointFinder(uint32_t, double, PerturbExtras::Disable);
//InstantiatePeriodicPointFinder(uint64_t, double, PerturbExtras::Disable);
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
