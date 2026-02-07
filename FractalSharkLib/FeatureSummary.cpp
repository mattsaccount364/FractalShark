#include "stdafx.h"

#include "FeatureFinderMode.h" // (or whatever header defines FeatureFinderMode)
#include "FeatureSummary.h"
#include "Fractal.h" // needed for EstablishScreenCoordinates()

FeatureSummary::FeatureSummary(const HighPrecision &origX,
                               const HighPrecision &origY,
                               const HighPrecision &radius,
                               FeatureFinderMode mode)
    : Radius{radius}, OrigX{origX}, OrigY{origY}, FoundX{origX}, FoundY{origY},
      Precision{origX.precisionInBits()}, IntrinsicRadius{radius}, Mode{mode}
{
}

void
FeatureSummary::SetFound(const HighPrecision &foundX,
                         const HighPrecision &foundY,
                         IterTypeFull period,
                         T residual2,
                         const HighPrecision &intrinsicRadius)
{
    FoundX = foundX;
    FoundY = foundY;
    Period = period;
    Residual2 = residual2;
    IntrinsicRadius = intrinsicRadius;

    // Once we have a final answer, we typically don't need the staged candidate anymore.
    // Keep it if you want; otherwise clear it.
    // m_candidate = nullptr;
}

void
FeatureSummary::ClearCandidate()
{
    m_candidate = nullptr;
}

bool
FeatureSummary::HasCandidate() const
{
    return (bool)m_candidate;
}

void
FeatureSummary::SetCandidate(std::unique_ptr<PeriodicPointCandidate> cand)
{
    m_candidate = std::move(cand);
}

void
FeatureSummary::SetCandidate(const HighPrecision &candidateX,
                             const HighPrecision &candidateY,
                             IterTypeFull period,
                             T residual2,
                             const HighPrecision &sqrRadius_hp,
                             int scaleExp2_for_mpf,
                             mp_bitcnt_t mpfPrecBits)
{
    auto cand = std::make_unique<PeriodicPointCandidate>();

    cand->cX_hp = candidateX;
    cand->cY_hp = candidateY;
    cand->period = period;
    cand->residual2 = residual2;
    cand->scaleExp2_for_mpf = scaleExp2_for_mpf;
    cand->mpfPrecBits = mpfPrecBits;
    cand->sqrRadius_hp = sqrRadius_hp;

    m_candidate = std::move(cand);
}

const PeriodicPointCandidate *
FeatureSummary::GetCandidate() const
{
    return m_candidate.get();
}

PeriodicPointCandidate *
FeatureSummary::GetCandidate()
{
    return m_candidate.get();
}

const HighPrecision &
FeatureSummary::GetIntrinsicRadius() const
{
    return IntrinsicRadius;
}

const HighPrecision &
FeatureSummary::GetRadius() const
{
    return Radius;
}

const HighPrecision &
FeatureSummary::GetOrigX() const
{
    return OrigX;
}

const HighPrecision &
FeatureSummary::GetOrigY() const
{
    return OrigY;
}

const HighPrecision &
FeatureSummary::GetFoundX() const
{
    return FoundX;
}

const HighPrecision &
FeatureSummary::GetFoundY() const
{
    return FoundY;
}

size_t
FeatureSummary::GetPrecision() const
{
    return Precision;
}

IterTypeFull
FeatureSummary::GetPeriod() const
{
    return Period;
}

FeatureSummary::T
FeatureSummary::GetResidual2() const
{
    return Residual2;
}

void
FeatureSummary::EstablishScreenCoordinates(const Fractal &fractal)
{
    const HighPrecision sxHP = fractal.XFromCalcToScreen(OrigX);
    const HighPrecision syHP = fractal.YFromCalcToScreen(OrigY);

    const HighPrecision fxHP = fractal.XFromCalcToScreen(FoundX);
    const HighPrecision fyHP = fractal.YFromCalcToScreen(FoundY);

    const int x0 = (int)((double)sxHP);
    const int y0 = (int)((double)syHP);
    const int x1 = (int)((double)fxHP);
    const int y1 = (int)((double)fyHP);

    auto flipY = [&](int y) -> int {
        // Convert top-left origin (UI) to bottom-left origin (GL)
        return (int)fractal.GetRenderHeight() - 1 - y;
    };

    const int y0_gl = flipY(y0);
    const int y1_gl = flipY(y1);

    screenXStart = x0;
    screenYStart = y0_gl;
    screenXEnd = x1;
    screenYEnd = y1_gl;
}

void
FeatureSummary::GetScreenCoordinates(int &outXStart, int &outYStart, int &outXEnd, int &outYEnd) const
{
    outXStart = screenXStart;
    outYStart = screenYStart;
    outXEnd = screenXEnd;
    outYEnd = screenYEnd;
}
