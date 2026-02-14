#include "stdafx.h"

#include "FeatureFinderMode.h" // (or whatever header defines FeatureFinderMode)
#include "FeatureSummary.h"
#include "Fractal.h" // needed for EstablishScreenCoordinates()
#include "PointZoomBBConverter.h"

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
                         HDRFloat<double> residual2,
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
                             HDRFloat<double> residual2,
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

HDRFloat<double>
FeatureSummary::GetResidual2() const
{
    return Residual2;
}

void
FeatureSummary::SetNumIterationsAtFind(IterTypeFull numIters)
{
    NumIterationsAtFind = numIters;
}

IterTypeFull
FeatureSummary::GetNumIterationsAtFind() const
{
    return NumIterationsAtFind;
}

void
FeatureSummary::EstablishScreenCoordinates(const Fractal &fractal)
{
    HighPrecision sxHP = fractal.XFromCalcToScreen(OrigX);
    HighPrecision syHP = fractal.YFromCalcToScreen(OrigY);

    HighPrecision fxHP = fractal.XFromCalcToScreen(FoundX);
    HighPrecision fyHP = fractal.YFromCalcToScreen(FoundY);

    // Pick the higher precision of the two points for clipping
    uint64_t precForClipping = std::max({sxHP.precisionInBits(),
                                         syHP.precisionInBits(),
                                         fxHP.precisionInBits(),
                                         fyHP.precisionInBits()});

    sxHP.precisionInBits(precForClipping);
    syHP.precisionInBits(precForClipping);
    fxHP.precisionInBits(precForClipping);
    fyHP.precisionInBits(precForClipping);

    const int64_t W = static_cast<int64_t>(fractal.GetRenderWidth());
    const int64_t H = static_cast<int64_t>(fractal.GetRenderHeight());

    if (W <= 0 || H <= 0) {
        screenXStart = screenYStart = screenXEnd = screenYEnd = 0;
        return;
    }

    // Clip rectangle in UI space (top-left origin)
    const HighPrecision xmin{0};
    const HighPrecision ymin{0};
    const HighPrecision xmax{W - 1};
    const HighPrecision ymax{H - 1};

    auto clamp_i64 = [](int64_t v, int64_t lo, int64_t hi) -> int64_t {
        if (v < lo)
            return lo;
        if (v > hi)
            return hi;
        return v;
    };

    // Cohen-Sutherland outcodes
    constexpr int INSIDE = 0;
    constexpr int LEFT = 1 << 0;
    constexpr int RIGHT = 1 << 1;
    constexpr int BOTTOM = 1 << 2; // y < ymin  (above top edge)
    constexpr int TOP = 1 << 3;    // y > ymax  (below bottom edge)

    auto outcode = [&](const HighPrecision &x, const HighPrecision &y) -> int {
        int c = INSIDE;

        if (x < xmin)
            c |= LEFT;
        else if (x > xmax)
            c |= RIGHT;

        if (y < ymin)
            c |= BOTTOM;
        else if (y > ymax)
            c |= TOP;

        return c;
    };

    auto clipLineToRect =
        [&](HighPrecision &x0, HighPrecision &y0, HighPrecision &x1, HighPrecision &y1) -> bool {
        int c0 = outcode(x0, y0);
        int c1 = outcode(x1, y1);

        for (;;) {
            if ((c0 | c1) == 0) {
                // trivially accept
                return true;
            }
            if (c0 & c1) {
                // trivially reject
                return false;
            }

            const int cOut = c0 ? c0 : c1;

            const HighPrecision dx = x1 - x0;
            const HighPrecision dy = y1 - y0;

            HighPrecision x = x0;
            HighPrecision y = y0;

            // Intersect with the appropriate boundary.
            // Use the parametric form:
            //   x = x0 + t*dx
            //   y = y0 + t*dy
            // Solve for t using the chosen boundary.
            if (cOut & TOP) {
                // y = ymax
                if (dy == HighPrecision{0})
                    return false;
                const HighPrecision t = (ymax - y0) / dy;
                x = x0 + t * dx;
                y = ymax;
            } else if (cOut & BOTTOM) {
                // y = ymin
                if (dy == HighPrecision{0})
                    return false;
                const HighPrecision t = (ymin - y0) / dy;
                x = x0 + t * dx;
                y = ymin;
            } else if (cOut & RIGHT) {
                // x = xmax
                if (dx == HighPrecision{0})
                    return false;
                const HighPrecision t = (xmax - x0) / dx;
                y = y0 + t * dy;
                x = xmax;
            } else { // LEFT
                // x = xmin
                if (dx == HighPrecision{0})
                    return false;
                const HighPrecision t = (xmin - x0) / dx;
                y = y0 + t * dy;
                x = xmin;
            }

            // Replace outside endpoint
            if (cOut == c0) {
                x0 = x;
                y0 = y;
                c0 = outcode(x0, y0);
            } else {
                x1 = x;
                y1 = y;
                c1 = outcode(x1, y1);
            }
        }
    };

    // Work in HighPrecision for clipping
    HighPrecision x0 = sxHP, y0 = syHP;
    HighPrecision x1 = fxHP, y1 = fyHP;

    const bool ok = clipLineToRect(x0, y0, x1, y1);

    // Convert to integer pixels in UI space
    int64_t ix0 = static_cast<int64_t>(static_cast<double>(x0));
    int64_t iy0 = static_cast<int64_t>(static_cast<double>(y0));
    int64_t ix1 = static_cast<int64_t>(static_cast<double>(x1));
    int64_t iy1 = static_cast<int64_t>(static_cast<double>(y1));

    // If no intersection, keep safe (fallback); you can also choose to mark invalid instead.
    if (!ok) {
        // fall back to clamping original endpoints (or set both to same point)
        ix0 = clamp_i64(static_cast<int64_t>(static_cast<double>(sxHP)), 0, W - 1);
        iy0 = clamp_i64(static_cast<int64_t>(static_cast<double>(syHP)), 0, H - 1);
        ix1 = clamp_i64(static_cast<int64_t>(static_cast<double>(fxHP)), 0, W - 1);
        iy1 = clamp_i64(static_cast<int64_t>(static_cast<double>(fyHP)), 0, H - 1);
    } else {
        // Numerical safety clamp
        ix0 = clamp_i64(ix0, 0, W - 1);
        iy0 = clamp_i64(iy0, 0, H - 1);
        ix1 = clamp_i64(ix1, 0, W - 1);
        iy1 = clamp_i64(iy1, 0, H - 1);
    }

    auto flipY = [&](int64_t y_ui) -> int64_t {
        return (H - 1) - y_ui; // UI top-left -> GL bottom-left
    };

    screenXStart = static_cast<uint64_t>(ix0);
    screenYStart = static_cast<uint64_t>(flipY(iy0));
    screenXEnd = static_cast<uint64_t>(ix1);
    screenYEnd = static_cast<uint64_t>(flipY(iy1));
}



void
FeatureSummary::GetScreenCoordinates(int &outXStart, int &outYStart, int &outXEnd, int &outYEnd) const
{
    outXStart = static_cast<int>(screenXStart);
    outYStart = static_cast<int>(screenYStart);
    outXEnd = static_cast<int>(screenXEnd);
    outYEnd = static_cast<int>(screenYEnd);
}

HighPrecision
FeatureSummary::ComputeZoomFactor(const PointZoomBBConverter &ptz) const
{
    const HighPrecision curHalfH = (ptz.GetMaxY() - ptz.GetMinY()) / HighPrecision{2};
    const HighPrecision r = GetIntrinsicRadius();

    if (r == HighPrecision{0} || curHalfH == HighPrecision{0}) {
        return ptz.GetZoomFactor();
    }

    const HighPrecision k = HighPrecision{6};
    const HighPrecision targetHalfH = r * k;

    // zoomFactor in your PointZoomBBConverter is "magnification": larger -> smaller BB -> zoom in.
    // halfHeight = factor / zoomFactor  (since BB uses +/- factor/zoomFactor)
    // so: zoomTarget = factor / targetHalfH
    const HighPrecision zTarget = HighPrecision{PointZoomBBConverter::factor} / targetHalfH;

    // Don't zoom out: if zTarget is smaller than current zoom, keep current
    const HighPrecision zCur = ptz.GetZoomFactor();
    if (zTarget < zCur)
        return zCur;

    return zTarget;
}

void FeatureSummary::SetRefined()
{
    Refined = true;
}

bool FeatureSummary::IsRefined() const
{
    return Refined;
}