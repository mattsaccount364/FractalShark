#pragma once

#include "HighPrecision.h"
#include "HDRFloat.h"

class Fractal;

class FeatureSummary {
public:
    FeatureSummary(const HighPrecision &origX, const HighPrecision &origY, const HighPrecision &radius);
    using T = HDRFloat<double>;

    void SetFound(const HighPrecision &foundX,
                  const HighPrecision &foundY,
                  IterTypeFull period,
                  T residual2);

    const HighPrecision &GetRadius() const;
    const HighPrecision &GetOrigX() const;
    const HighPrecision &GetOrigY() const;
    const HighPrecision &GetFoundX() const;
    const HighPrecision &GetFoundY() const;
    IterTypeFull GetPeriod() const;
    T GetResidual2() const;

    void EstablishScreenCoordinates(const Fractal &fractal);
    void GetScreenCoordinates(int &outXStart, int &outYStart, int &outXEnd, int &outYEnd) const;

private:
    HighPrecision Radius;
    HighPrecision OrigX;
    HighPrecision OrigY;
    HighPrecision FoundX;
    HighPrecision FoundY;
    IterTypeFull Period{};
    T Residual2{}; // squared residual at acceptance (double for quick logging/debug)

    int screenXStart;
    int screenYStart;
    int screenXEnd;
    int screenYEnd;
};