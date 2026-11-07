#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

#include "FeatureFinderMode.h"
#include "HDRFloat.h"
#include "HighPrecision.h"
#include "RenderOverlaySnapshot.h"

class Fractal;
class PointZoomBBConverter;

struct PeriodicPointCandidate {
    using ResidualT = HDRFloat<double>;

    // Where the candidate currently is (HP is canonical)
    HighPrecision cX_hp;
    HighPrecision cY_hp;
    HighPrecision sqrRadius_hp;

    IterTypeFull period{}; // candidate period
    ResidualT residual2{}; // candidate residual^2 (optional but useful)

    // Which path produced this candidate (Direct/PT/LA)
    FeatureFinderMode modeFoundBy{};

    // Cached MPF setup (computed from w=zcoeff*dzdc at candidate time)
    int scaleExp2_for_mpf{0};   // exponent-ish of Scale = 1/|zcoeff*dzdc|
    mp_bitcnt_t mpfPrecBits{0}; // chosen coord precision for MPF
};

class FeatureSummary {
public:
    FeatureSummary(const HighPrecision &origX,
                   const HighPrecision &origY,
                   const HighPrecision &radius,
                   FeatureFinderMode mode);

    // Final refined result
    void SetFound(const HighPrecision &foundX,
                  const HighPrecision &foundY,
                  IterTypeFull period,
                  HDRFloat<double> residual2,
                  const HighPrecision &intrinsicRadius);

    // Candidate management (nullable)
    void ClearCandidate();
    bool HasCandidate() const;
    void SetCandidate(std::unique_ptr<PeriodicPointCandidate> cand);

    void SetCandidate(const HighPrecision &candidateX,
                      const HighPrecision &candidateY,
                      IterTypeFull period,
                      HDRFloat<double> residual2,
                      const HighPrecision &sqrRadius_hp,
                      int scaleExp2_for_mpf,
                      mp_bitcnt_t mpfPrecBits);

    const PeriodicPointCandidate *GetCandidate() const;
    PeriodicPointCandidate *GetCandidate();

    // Accessors
    const HighPrecision &GetRadius() const;
    const HighPrecision &GetOrigX() const;
    const HighPrecision &GetOrigY() const;

    const HighPrecision &GetFoundX() const;
    const HighPrecision &GetFoundY() const;

    size_t GetPrecision() const;

    IterTypeFull GetPeriod() const;
    HDRFloat<double> GetResidual2() const;

    void SetNumIterationsAtFind(IterTypeFull numIters);
    IterTypeFull GetNumIterationsAtFind() const;

    const HighPrecision &GetIntrinsicRadius() const;

    // Called only while the worker owns the feature; the returned geometry is independent.
    FractalShark::RenderOverlayLine ComputeScreenCoordinates(const PointZoomBBConverter &view,
                                                             size_t width,
                                                             size_t height) const;
    static HighPrecision ComputeZoomFactorForRadius(const HighPrecision &radius);
    HighPrecision ComputeZoomFactor(const PointZoomBBConverter &ptz) const;
    void SetRefined();
    bool IsRefined() const;

private:
    HighPrecision Radius; // search radius
    HighPrecision OrigX;
    HighPrecision OrigY;
    FeatureFinderMode Mode;

    std::unique_ptr<PeriodicPointCandidate> m_candidate; // null => no candidate staged

    HighPrecision FoundX;
    HighPrecision FoundY;

    size_t Precision{0};

    HighPrecision IntrinsicRadius;

    IterTypeFull Period{};
    HDRFloat<double> Residual2{};
    IterTypeFull NumIterationsAtFind{};

    bool Refined{false};
};
