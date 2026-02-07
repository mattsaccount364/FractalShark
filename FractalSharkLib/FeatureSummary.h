#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

#include "FeatureFinderMode.h"
#include "HDRFloat.h"
#include "HighPrecision.h"

// wherever IterTypeFull lives; include that instead if different
#include "Vectors.h" // (or your real header that defines IterTypeFull)

class Fractal;

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

    // Optional: if you want Phase B independent of FeatureSummary radius:
    // HighPrecision sqrRadius_hp;
};

class FeatureSummary {
public:
    using T = HDRFloat<double>;

    FeatureSummary(const HighPrecision &origX,
                   const HighPrecision &origY,
                   const HighPrecision &radius,
                   FeatureFinderMode mode);

    // Final refined result
    void SetFound(const HighPrecision &foundX,
                  const HighPrecision &foundY,
                  IterTypeFull period,
                  T residual2,
                  const HighPrecision &intrinsicRadius);

    // Candidate management (nullable)
    void ClearCandidate();
    bool HasCandidate() const;
    void SetCandidate(std::unique_ptr<PeriodicPointCandidate> cand);

    void SetCandidate(const HighPrecision &candidateX,
                      const HighPrecision &candidateY,
                      IterTypeFull period,
                      T residual2,
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
    T GetResidual2() const;

    const HighPrecision &GetIntrinsicRadius() const;

    void EstablishScreenCoordinates(const Fractal &fractal);
    void GetScreenCoordinates(int &outXStart, int &outYStart, int &outXEnd, int &outYEnd) const;

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
    T Residual2{};

    int screenXStart{0};
    int screenYStart{0};
    int screenXEnd{0};
    int screenYEnd{0};
};
