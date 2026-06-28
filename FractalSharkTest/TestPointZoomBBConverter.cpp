#include "TestFramework.h"
#include "FeatureFinder.h"
#include "FeatureSummary.h"
#include "PointZoomBBConverter.h"

using TestMode = PointZoomBBConverter::TestMode;

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

TEST(DefaultConstruction)
{
    PointZoomBBConverter pz(TestMode::Enabled);
    ASSERT_TRUE(pz.GetMinX() == HighPrecision{0});
    ASSERT_TRUE(pz.GetMinY() == HighPrecision{0});
    ASSERT_TRUE(pz.GetMaxX() == HighPrecision{0});
    ASSERT_TRUE(pz.GetMaxY() == HighPrecision{0});
    ASSERT_TRUE(pz.Degenerate());
}

TEST(PointZoomConstruction_Origin)
{
    // pt = (0,0), zoom = 1 → bounds = [-2,-2] to [2,2] (Factor = 2)
    PointZoomBBConverter pz(
        HighPrecision{0}, HighPrecision{0}, HighPrecision{1}, TestMode::Enabled);

    ASSERT_TRUE(pz.GetMinX() == HighPrecision{-2});
    ASSERT_TRUE(pz.GetMinY() == HighPrecision{-2});
    ASSERT_TRUE(pz.GetMaxX() == HighPrecision{2});
    ASSERT_TRUE(pz.GetMaxY() == HighPrecision{2});
    ASSERT_TRUE(pz.GetPtX() == HighPrecision{0});
    ASSERT_TRUE(pz.GetPtY() == HighPrecision{0});
    ASSERT_TRUE(pz.GetZoomFactor() == HighPrecision{1});
    ASSERT_FALSE(pz.Degenerate());
}

TEST(PointZoomConstruction_NonOrigin)
{
    // pt = (1,2), zoom = 4 → min = pt - 2/4 = pt - 0.5, max = pt + 0.5
    PointZoomBBConverter pz(
        HighPrecision{1}, HighPrecision{2}, HighPrecision{4}, TestMode::Enabled);

    ASSERT_TRUE(pz.GetMinX() == HighPrecision{0.5});
    ASSERT_TRUE(pz.GetMinY() == HighPrecision{1.5});
    ASSERT_TRUE(pz.GetMaxX() == HighPrecision{1.5});
    ASSERT_TRUE(pz.GetMaxY() == HighPrecision{2.5});
    ASSERT_TRUE(pz.GetPtX() == HighPrecision{1});
    ASSERT_TRUE(pz.GetPtY() == HighPrecision{2});
    ASSERT_TRUE(pz.GetZoomFactor() == HighPrecision{4});
}

TEST(FeatureZoomFactorForRadius)
{
    const HighPrecision zoom = FeatureSummary::ComputeZoomFactorForRadius(HighPrecision{0.5});
    ASSERT_NEAR(static_cast<double>(zoom), 2.0 / 3.0, 1e-12);
}

TEST(ResumedFeatureSummaryKeepsLinearAndSquaredRadiiSeparate)
{
    const HighPrecision candidateX{1.25};
    const HighPrecision candidateY{-0.75};
    const HighPrecision intrinsicRadius{0.5};
    const HighPrecision sqrRadius{0.25};
    const HDRFloat<double> residual2{0.001};
    constexpr IterTypeFull Period = 7;
    constexpr int ScaleExp2ForMpf = -3;
    constexpr mp_bitcnt_t MpfPrecBits = 256;

    FeatureSummary summary(candidateX, candidateY, intrinsicRadius, FeatureFinderMode::Direct);
    summary.SetCandidate(
        candidateX, candidateY, Period, residual2, sqrRadius, ScaleExp2ForMpf, MpfPrecBits);
    summary.SetFound(candidateX, candidateY, Period, residual2, intrinsicRadius);

    ASSERT_TRUE(summary.GetRadius() == intrinsicRadius);
    ASSERT_TRUE(summary.GetCandidate() != nullptr);
    ASSERT_TRUE(summary.GetCandidate()->sqrRadius_hp == sqrRadius);
    ASSERT_TRUE(summary.GetIntrinsicRadius() == intrinsicRadius);
}

TEST(NRCheckpointPreviewZoomUsesIntrinsicRadiusWhenStepIsSmaller)
{
    DiagnosticState diag;
    diag.valid = true;
    diag.step_norm = HDRFloat<double>{0.0625};
    diag.err = HDRFloat<double>{10000.0};

    const std::string previewZoom = ComputeNRCheckpointPreviewZoom(diag, HighPrecision{0.5});
    const std::string expected = FeatureSummary::ComputeZoomFactorForRadius(HighPrecision{0.5}).str();

    ASSERT_EQ(previewZoom, expected);
}

TEST(NRCheckpointPreviewZoomUsesStepDistanceWhenLarger)
{
    DiagnosticState diag;
    diag.valid = true;
    diag.step_norm = HDRFloat<double>{1.0};
    diag.err = HDRFloat<double>{0.0001};

    const std::string previewZoom = ComputeNRCheckpointPreviewZoom(diag, HighPrecision{0.25});
    const std::string expected = FeatureSummary::ComputeZoomFactorForRadius(HighPrecision{1.0}).str();

    ASSERT_EQ(previewZoom, expected);
}

TEST(NRCheckpointPreviewZoomUnavailableForInvalidInputs)
{
    DiagnosticState diag;
    diag.valid = true;
    diag.step_norm = HDRFloat<double>{0.25};

    ASSERT_EQ(ComputeNRCheckpointPreviewZoom(DiagnosticState{}, HighPrecision{0.5}), "unavailable");

    diag.step_norm = HDRFloat<double>{0.0};
    ASSERT_EQ(ComputeNRCheckpointPreviewZoom(diag, HighPrecision{0.5}), "unavailable");

    diag.step_norm = HDRFloat<double>{0.25};
    ASSERT_EQ(ComputeNRCheckpointPreviewZoom(diag, HighPrecision{0}), "unavailable");
}

TEST(BoundingBoxConstruction)
{
    // bb [-2,-2,2,2] → pt = (0,0), zoom = Factor*2/deltaY = 2*2/4 = 1
    PointZoomBBConverter pz(
        HighPrecision{-2}, HighPrecision{-2}, HighPrecision{2}, HighPrecision{2},
        TestMode::Enabled);

    ASSERT_TRUE(pz.GetPtX() == HighPrecision{0});
    ASSERT_TRUE(pz.GetPtY() == HighPrecision{0});
    ASSERT_TRUE(pz.GetZoomFactor() == HighPrecision{1});
    ASSERT_TRUE(pz.GetMinX() == HighPrecision{-2});
    ASSERT_TRUE(pz.GetMaxY() == HighPrecision{2});
    ASSERT_FALSE(pz.Degenerate());
}

TEST(BoundingBoxConstruction_ZeroDeltaY)
{
    // When minY == maxY, deltaY is zero → zoom factor falls back to 1
    PointZoomBBConverter pz(
        HighPrecision{-1}, HighPrecision{3}, HighPrecision{1}, HighPrecision{3},
        TestMode::Enabled);

    ASSERT_TRUE(pz.GetZoomFactor() == HighPrecision{1});
}

// ---------------------------------------------------------------------------
// Degenerate detection
// ---------------------------------------------------------------------------

TEST(Degenerate_ZeroWidth)
{
    PointZoomBBConverter pz(
        HighPrecision{0}, HighPrecision{-1}, HighPrecision{0}, HighPrecision{1},
        TestMode::Enabled);
    ASSERT_TRUE(pz.Degenerate());
}

TEST(Degenerate_ZeroHeight)
{
    PointZoomBBConverter pz(
        HighPrecision{-1}, HighPrecision{0}, HighPrecision{1}, HighPrecision{0},
        TestMode::Enabled);
    ASSERT_TRUE(pz.Degenerate());
}

TEST(Degenerate_Normal)
{
    PointZoomBBConverter pz(
        HighPrecision{-1}, HighPrecision{-1}, HighPrecision{1}, HighPrecision{1},
        TestMode::Enabled);
    ASSERT_FALSE(pz.Degenerate());
}

// ---------------------------------------------------------------------------
// Coordinate roundtrips
// ---------------------------------------------------------------------------

TEST(XScreenToCalcRoundtrip)
{
    // Bounds [-2,-2,2,2], screen 100×100
    PointZoomBBConverter pz(
        HighPrecision{-2}, HighPrecision{-2}, HighPrecision{2}, HighPrecision{2},
        TestMode::Enabled);

    // Pixel 0 → calc -2 → pixel 0
    {
        HighPrecision calc = pz.XFromScreenToCalc(HighPrecision{0}, 100, 1);
        ASSERT_TRUE(calc == HighPrecision{-2});
        HighPrecision back = pz.XFromCalcToScreen(calc, 100);
        ASSERT_NEAR(static_cast<double>(back), 0.0, 1e-10);
    }

    // Pixel 50 → calc 0 → pixel 50
    {
        HighPrecision calc = pz.XFromScreenToCalc(HighPrecision{50}, 100, 1);
        ASSERT_NEAR(static_cast<double>(calc), 0.0, 1e-10);
        HighPrecision back = pz.XFromCalcToScreen(calc, 100);
        ASSERT_NEAR(static_cast<double>(back), 50.0, 1e-10);
    }
}

TEST(YScreenToCalcRoundtrip)
{
    PointZoomBBConverter pz(
        HighPrecision{-2}, HighPrecision{-2}, HighPrecision{2}, HighPrecision{2},
        TestMode::Enabled);

    // Pixel 0 (top) → calc maxY (2) → pixel 0
    {
        HighPrecision calc = pz.YFromScreenToCalc(HighPrecision{0}, 100, 1);
        ASSERT_NEAR(static_cast<double>(calc), 2.0, 1e-10);
        HighPrecision back = pz.YFromCalcToScreen(calc, 100);
        ASSERT_NEAR(static_cast<double>(back), 0.0, 1e-10);
    }

    // Pixel 50 (middle) → calc 0 → pixel 50
    {
        HighPrecision calc = pz.YFromScreenToCalc(HighPrecision{50}, 100, 1);
        ASSERT_NEAR(static_cast<double>(calc), 0.0, 1e-10);
        HighPrecision back = pz.YFromCalcToScreen(calc, 100);
        ASSERT_NEAR(static_cast<double>(back), 50.0, 1e-10);
    }
}

TEST(CoordinateRoundtrip_WithAntialiasing)
{
    PointZoomBBConverter pz(
        HighPrecision{-2}, HighPrecision{-2}, HighPrecision{2}, HighPrecision{2},
        TestMode::Enabled);

    // With 2x antialiasing, the effective resolution is 200×200 for a 100×100 screen.
    // Pixel 100 (middle of the 200-wide supersampled space) → calc 0
    HighPrecision calc = pz.XFromScreenToCalc(HighPrecision{100}, 100, 2);
    ASSERT_NEAR(static_cast<double>(calc), 0.0, 1e-10);
}

// ---------------------------------------------------------------------------
// Delta calculations
// ---------------------------------------------------------------------------

TEST(GetDeltaXY)
{
    PointZoomBBConverter pz(
        HighPrecision{-2}, HighPrecision{-2}, HighPrecision{2}, HighPrecision{2},
        TestMode::Enabled);

    // deltaX = (maxX - minX) / (width * aa) = 4 / (100*1) = 0.04
    HighPrecision dx = pz.GetDeltaX(100, 1);
    ASSERT_NEAR(static_cast<double>(dx), 0.04, 1e-15);

    HighPrecision dy = pz.GetDeltaY(100, 1);
    ASSERT_NEAR(static_cast<double>(dy), 0.04, 1e-15);

    // With 2x antialiasing: 4 / (100*2) = 0.02
    HighPrecision dx2 = pz.GetDeltaX(100, 2);
    ASSERT_NEAR(static_cast<double>(dx2), 0.02, 1e-15);
}

// ---------------------------------------------------------------------------
// Zoom operations
// ---------------------------------------------------------------------------

TEST(ZoomedAtCenter)
{
    PointZoomBBConverter pz(
        HighPrecision{0}, HighPrecision{0}, HighPrecision{1}, TestMode::Enabled);
    // Bounds: [-2,-2,2,2]

    // Zoom in: scale = -0.3 → divisor = 1/(1+2*(-0.3)) = 1/0.4 = 2.5
    // New halfX = 2 / 2.5 = 0.8 → bounds [-0.8, -0.8, 0.8, 0.8]
    PointZoomBBConverter zoomed = pz.ZoomedAtCenter(-0.3);

    // Center should be preserved
    ASSERT_NEAR(static_cast<double>(zoomed.GetPtX()), 0.0, 1e-10);
    ASSERT_NEAR(static_cast<double>(zoomed.GetPtY()), 0.0, 1e-10);

    // Extents should shrink
    ASSERT_NEAR(static_cast<double>(zoomed.GetMinX()), -0.8, 1e-10);
    ASSERT_NEAR(static_cast<double>(zoomed.GetMaxX()), 0.8, 1e-10);
    ASSERT_NEAR(static_cast<double>(zoomed.GetMinY()), -0.8, 1e-10);
    ASSERT_NEAR(static_cast<double>(zoomed.GetMaxY()), 0.8, 1e-10);

    // Zoom factor should increase
    ASSERT_TRUE(zoomed.GetZoomFactor() > pz.GetZoomFactor());
}

TEST(ZoomInPlace)
{
    PointZoomBBConverter pz1(
        HighPrecision{0}, HighPrecision{0}, HighPrecision{1}, TestMode::Enabled);
    PointZoomBBConverter pz2 = pz1;

    PointZoomBBConverter zoomed = pz1.ZoomedAtCenter(-0.3);
    pz2.ZoomInPlace(-0.3);

    ASSERT_NEAR(static_cast<double>(pz2.GetMinX()), static_cast<double>(zoomed.GetMinX()), 1e-10);
    ASSERT_NEAR(static_cast<double>(pz2.GetMaxX()), static_cast<double>(zoomed.GetMaxX()), 1e-10);
    ASSERT_NEAR(static_cast<double>(pz2.GetMinY()), static_cast<double>(zoomed.GetMinY()), 1e-10);
    ASSERT_NEAR(static_cast<double>(pz2.GetMaxY()), static_cast<double>(zoomed.GetMaxY()), 1e-10);
}

TEST(ZoomedRecentered)
{
    PointZoomBBConverter pz(
        HighPrecision{-2}, HighPrecision{-2}, HighPrecision{2}, HighPrecision{2},
        TestMode::Enabled);

    // Recenter on (1,1) with no scale change (scale=0 → divisor=1 → no zoom)
    PointZoomBBConverter zoomed = pz.ZoomedRecentered(HighPrecision{1}, HighPrecision{1}, 0.0);

    // New center should be (1,1), extents should be same width/height = 4
    ASSERT_NEAR(static_cast<double>(zoomed.GetPtX()), 1.0, 1e-10);
    ASSERT_NEAR(static_cast<double>(zoomed.GetPtY()), 1.0, 1e-10);

    double width = static_cast<double>(zoomed.GetMaxX()) - static_cast<double>(zoomed.GetMinX());
    double height = static_cast<double>(zoomed.GetMaxY()) - static_cast<double>(zoomed.GetMinY());
    ASSERT_NEAR(width, 4.0, 1e-10);
    ASSERT_NEAR(height, 4.0, 1e-10);
}

TEST(ZoomedTowardPoint)
{
    PointZoomBBConverter pz(
        HighPrecision{-2}, HighPrecision{-2}, HighPrecision{2}, HighPrecision{2},
        TestMode::Enabled);

    // Zoom toward (1,0) with scale=0.5 (expand outward)
    // leftWeight = (1-(-2))/4 = 0.75, rightWeight = 0.25
    // topWeight = (0-(-2))/4 = 0.5, bottomWeight = 0.5
    // newMinX = -2 - 4*0.75*0.5 = -3.5
    // newMaxX = 2 + 4*0.25*0.5 = 2.5
    // newMinY = -2 - 4*0.5*0.5 = -3
    // newMaxY = 2 + 4*0.5*0.5 = 3
    PointZoomBBConverter zoomed =
        pz.ZoomedTowardPoint(HighPrecision{1}, HighPrecision{0}, 0.5);

    ASSERT_NEAR(static_cast<double>(zoomed.GetMinX()), -3.5, 1e-10);
    ASSERT_NEAR(static_cast<double>(zoomed.GetMaxX()), 2.5, 1e-10);
    ASSERT_NEAR(static_cast<double>(zoomed.GetMinY()), -3.0, 1e-10);
    ASSERT_NEAR(static_cast<double>(zoomed.GetMaxY()), 3.0, 1e-10);
}

// ---------------------------------------------------------------------------
// SquareAspectRatio
// ---------------------------------------------------------------------------

TEST(SquareAspectRatio_AlreadySquare)
{
    PointZoomBBConverter pz(
        HighPrecision{-2}, HighPrecision{-2}, HighPrecision{2}, HighPrecision{2},
        TestMode::Enabled);

    double origWidth = static_cast<double>(pz.GetMaxX()) - static_cast<double>(pz.GetMinX());
    double origHeight = static_cast<double>(pz.GetMaxY()) - static_cast<double>(pz.GetMinY());

    // 100x100 screen with square bounds → should be no change
    pz.SquareAspectRatio(100, 100);

    double newWidth = static_cast<double>(pz.GetMaxX()) - static_cast<double>(pz.GetMinX());
    double newHeight = static_cast<double>(pz.GetMaxY()) - static_cast<double>(pz.GetMinY());
    ASSERT_NEAR(newWidth, origWidth, 1e-10);
    ASSERT_NEAR(newHeight, origHeight, 1e-10);
}

TEST(SquareAspectRatio_Wide)
{
    // Square bounds on a wide screen → X bounds should expand
    PointZoomBBConverter pz(
        HighPrecision{-1}, HighPrecision{-1}, HighPrecision{1}, HighPrecision{1},
        TestMode::Enabled);

    pz.SquareAspectRatio(200, 100);

    // Height in complex plane stays ~2, width should expand to 4 (200/100 * 2)
    double height = static_cast<double>(pz.GetMaxY()) - static_cast<double>(pz.GetMinY());
    double width = static_cast<double>(pz.GetMaxX()) - static_cast<double>(pz.GetMinX());
    ASSERT_NEAR(height, 2.0, 1e-10);
    ASSERT_NEAR(width, 4.0, 1e-10);
}

TEST(SquareAspectRatio_Tall)
{
    // Square bounds on a tall screen → Y bounds should expand
    PointZoomBBConverter pz(
        HighPrecision{-1}, HighPrecision{-1}, HighPrecision{1}, HighPrecision{1},
        TestMode::Enabled);

    pz.SquareAspectRatio(100, 200);

    double width = static_cast<double>(pz.GetMaxX()) - static_cast<double>(pz.GetMinX());
    double height = static_cast<double>(pz.GetMaxY()) - static_cast<double>(pz.GetMinY());
    ASSERT_NEAR(width, 2.0, 1e-10);
    ASSERT_NEAR(height, 4.0, 1e-10);
}

TEST(SquareAspectRatio_ZeroDims)
{
    PointZoomBBConverter pz(
        HighPrecision{-1}, HighPrecision{-1}, HighPrecision{1}, HighPrecision{1},
        TestMode::Enabled);

    double origMinX = static_cast<double>(pz.GetMinX());

    // Zero dimensions → no-op
    pz.SquareAspectRatio(0, 100);
    ASSERT_NEAR(static_cast<double>(pz.GetMinX()), origMinX, 1e-15);

    pz.SquareAspectRatio(100, 0);
    ASSERT_NEAR(static_cast<double>(pz.GetMinX()), origMinX, 1e-15);
}

// ---------------------------------------------------------------------------
// Recentered
// ---------------------------------------------------------------------------

TEST(Recentered)
{
    PointZoomBBConverter pz(
        HighPrecision{-2}, HighPrecision{-2}, HighPrecision{2}, HighPrecision{2},
        TestMode::Enabled);

    PointZoomBBConverter rc = pz.Recentered(HighPrecision{5}, HighPrecision{3});

    // New center at (5,3), same extents (4x4)
    ASSERT_NEAR(static_cast<double>(rc.GetPtX()), 5.0, 1e-10);
    ASSERT_NEAR(static_cast<double>(rc.GetPtY()), 3.0, 1e-10);

    double width = static_cast<double>(rc.GetMaxX()) - static_cast<double>(rc.GetMinX());
    double height = static_cast<double>(rc.GetMaxY()) - static_cast<double>(rc.GetMinY());
    ASSERT_NEAR(width, 4.0, 1e-10);
    ASSERT_NEAR(height, 4.0, 1e-10);
}

// ---------------------------------------------------------------------------
// Consistency between constructors
// ---------------------------------------------------------------------------

TEST(PointZoomAndBBConsistency)
{
    // Build from point+zoom and from bounding box — should agree
    PointZoomBBConverter fromPt(
        HighPrecision{0}, HighPrecision{0}, HighPrecision{1}, TestMode::Enabled);

    PointZoomBBConverter fromBB(
        fromPt.GetMinX(), fromPt.GetMinY(), fromPt.GetMaxX(), fromPt.GetMaxY(),
        TestMode::Enabled);

    ASSERT_TRUE(fromBB.GetPtX() == fromPt.GetPtX());
    ASSERT_TRUE(fromBB.GetPtY() == fromPt.GetPtY());
    ASSERT_TRUE(fromBB.GetZoomFactor() == fromPt.GetZoomFactor());
    ASSERT_FALSE(fromBB.Degenerate());
}

// ---------------------------------------------------------------------------
// Radius
// ---------------------------------------------------------------------------

TEST(RadiusCalculation)
{
    // Point+zoom: radius = (maxY - minY) / 2 = (2-(-2))/2 = 2
    PointZoomBBConverter pz(
        HighPrecision{0}, HighPrecision{0}, HighPrecision{1}, TestMode::Enabled);
    ASSERT_TRUE(pz.GetRadius() == HighPrecision{2});

    // Non-origin: radius = (2.5-1.5)/2 = 0.5
    PointZoomBBConverter pz2(
        HighPrecision{1}, HighPrecision{2}, HighPrecision{4}, TestMode::Enabled);
    ASSERT_TRUE(pz2.GetRadius() == HighPrecision{0.5});
}

// ---------------------------------------------------------------------------
// SetPrecision
// ---------------------------------------------------------------------------

TEST(SetPrecision)
{
    PointZoomBBConverter pz(
        HighPrecision{0}, HighPrecision{0}, HighPrecision{1}, TestMode::Enabled);

    pz.SetPrecision(512);

    // All internal HighPrecision values should now have 512-bit precision.
    // Verify via the precision of a getter.
    ASSERT_TRUE(pz.GetMinX().precisionInBits() >= 512);
    ASSERT_TRUE(pz.GetPtX().precisionInBits() >= 512);
    ASSERT_TRUE(pz.GetZoomFactor().precisionInBits() >= 512);

    // Values should be unchanged
    ASSERT_NEAR(static_cast<double>(pz.GetMinX()), -2.0, 1e-10);
    ASSERT_NEAR(static_cast<double>(pz.GetMaxX()), 2.0, 1e-10);
}
