#pragma once

#include "HighPrecision.h"
#include "HDRFloat.h"

class Fractal;

class FeatureSummary {
public:
    FeatureSummary(const HighPrecision &origX, const HighPrecision &origY, const HighPrecision &radius);
    using T = HDRFloat<double>;

    // CHANGED: add intrinsicRadius (aka "feature radius" / "scale*4" like Imagina)
    void SetFound(const HighPrecision &foundX,
                  const HighPrecision &foundY,
                  IterTypeFull period,
                  T residual2,
                  const HighPrecision &intrinsicRadius);

    const HighPrecision &GetRadius() const; // search radius (what you passed in)
    const HighPrecision &GetOrigX() const;
    const HighPrecision &GetOrigY() const;
    const HighPrecision &GetFoundX() const;
    const HighPrecision &GetFoundY() const;
    size_t GetPrecision() const;

    IterTypeFull GetPeriod() const;
    T GetResidual2() const;

    // NEW: intrinsic feature radius (Imagina: Scale*4)
    const HighPrecision &GetIntrinsicRadius() const;

    void EstablishScreenCoordinates(const Fractal &fractal);
    void GetScreenCoordinates(int &outXStart, int &outYStart, int &outXEnd, int &outYEnd) const;

private:
    HighPrecision Radius; // search radius (existing)
    HighPrecision OrigX;
    HighPrecision OrigY;
    HighPrecision FoundX;
    HighPrecision FoundY;

    size_t Precision;

    // NEW: intrinsic radius derived from zcoeff/dzdc at solution
    HighPrecision IntrinsicRadius;

    IterTypeFull Period{};
    T Residual2{};

    int screenXStart;
    int screenYStart;
    int screenXEnd;
    int screenYEnd;
};