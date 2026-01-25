#include "stdafx.h"
#include "FeatureSummary.h"
#include "Fractal.h"

FeatureSummary::FeatureSummary(const HighPrecision &origX,
                               const HighPrecision &origY,
                               const HighPrecision &radius)
    : Radius{radius}, OrigX{origX}, OrigY{origY}, FoundX{}, FoundY{}, Period{}, Residual2{},
      screenXStart{}, screenYStart{}, screenXEnd{}, screenYEnd{}
{
}

void
FeatureSummary::SetFound(const HighPrecision &foundX,
                         const HighPrecision &foundY,
                         IterTypeFull period,
                         double residual2)
{
    FoundX = foundX;
    FoundY = foundY;
    Period = period;
    Residual2 = residual2;
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

IterTypeFull
FeatureSummary::GetPeriod() const
{
    return Period;
}

double
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
