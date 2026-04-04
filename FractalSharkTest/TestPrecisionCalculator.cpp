#include "TestFramework.h"
#include "PrecisionCalculator.h"
#include "PointZoomBBConverter.h"
#include "HDRFloat.h"

#include <cmath>

using TestMode = PointZoomBBConverter::TestMode;

// The constants from HighPrecision.h:
// AuthoritativeMinExtraPrecisionInBits = 120
// AuthoritativeReuseExtraPrecisionInBits = 800

static constexpr uint64_t MinExtra = 120;
static constexpr uint64_t ReuseExtra = 800;

// ---------------------------------------------------------------------------
// PointZoomBBConverter overload
// ---------------------------------------------------------------------------

TEST(PC_FromConverter_DefaultZoom)
{
    // Bounds [-2,-2,2,2], zoom=1 → deltaX=deltaY=4
    // HighPrecision→HDRFloat uses GMP [0.5,1.0) convention: 4.0 = 0.5 × 2^3 → exp=3
    // max(|3|,|3|) = 3 → 3 + 120 = 123
    PointZoomBBConverter pz(
        HighPrecision{0}, HighPrecision{0}, HighPrecision{1}, TestMode::Enabled);

    uint64_t prec = PrecisionCalculator::GetPrecision(pz, false);
    ASSERT_EQ(prec, 3 + MinExtra);
}

TEST(PC_FromConverter_WithReuse)
{
    PointZoomBBConverter pz(
        HighPrecision{0}, HighPrecision{0}, HighPrecision{1}, TestMode::Enabled);

    uint64_t prec = PrecisionCalculator::GetPrecision(pz, true);
    ASSERT_EQ(prec, 3 + ReuseExtra);
}

TEST(PC_FromConverter_DeepZoom)
{
    // Zoom = 2^50 → delta = 4/2^50 = 2^(-48)
    // HDRFloat of delta → exp ≈ -48 → max(|-48|,|-48|) = 48 → 48+120=168
    HighPrecision zoom{1};
    zoom.precisionInBits(256);
    // Construct zoom = 2^50 via repeated doubling
    for (int i = 0; i < 50; ++i) {
        zoom *= HighPrecision{2};
    }

    PointZoomBBConverter pz(HighPrecision{0}, HighPrecision{0}, zoom, TestMode::Enabled);

    uint64_t prec = PrecisionCalculator::GetPrecision(pz, false);
    // delta = Factor/zoom * 2 = 4 / 2^50 = 2^(-48)
    // GMP convention: 0.5 × 2^(-47) → exp = -47
    // Result = |-47| + 120 = 167
    ASSERT_EQ(prec, 47 + MinExtra);
}

// ---------------------------------------------------------------------------
// Bounding box overload
// ---------------------------------------------------------------------------

TEST(PC_FromBoundingBox)
{
    uint64_t prec = PrecisionCalculator::GetPrecision(
        HighPrecision{-2}, HighPrecision{-2},
        HighPrecision{2}, HighPrecision{2},
        false);
    // Same as converter: delta=4 → GMP exp=3 → 3+120=123
    ASSERT_EQ(prec, 3 + MinExtra);
}

TEST(PC_FromBoundingBox_AsymmetricDeltas)
{
    // deltaX = |10 - 0| = 10 → GMP: 0.625 × 2^4 → exp=4
    // deltaY = |0.001 - 0| = 0.001 → GMP: ~0.512 × 2^(-9) → exp=-9
    // max(|4|, |-9|) = 9 → 9 + 120 = 129
    uint64_t prec = PrecisionCalculator::GetPrecision(
        HighPrecision{0.0}, HighPrecision{0.0},
        HighPrecision{10.0}, HighPrecision{0.001},
        false);
    ASSERT_EQ(prec, 9 + MinExtra);
}

// ---------------------------------------------------------------------------
// HighPrecision radius overload
// ---------------------------------------------------------------------------

TEST(PC_FromRadii_HighPrecision)
{
    // radiusX = 0.5 → GMP: 0.5 × 2^0 → exp=0
    // max(|0|, |0|) = 0 → 0 + 120 = 120
    uint64_t prec = PrecisionCalculator::GetPrecision(
        HighPrecision{0.5}, HighPrecision{0.5}, false);
    ASSERT_EQ(prec, 0 + MinExtra);
}

// ---------------------------------------------------------------------------
// HDRFloat<double> template overload
// ---------------------------------------------------------------------------

TEST(PC_FromHDRFloat_Double)
{
    HDRFloat<double> rx(0, 1.0);
    rx.setExp(-100);
    HDRFloat<double> ry(0, 1.0);
    ry.setExp(-50);

    uint64_t prec = PrecisionCalculator::GetPrecision(rx, ry, false);
    // max(|-100|, |-50|) = 100 → 100 + 120 = 220
    ASSERT_EQ(prec, 100 + MinExtra);
}

TEST(PC_FromHDRFloat_Double_Reuse)
{
    HDRFloat<double> rx(0, 1.0);
    rx.setExp(-100);
    HDRFloat<double> ry(0, 1.0);
    ry.setExp(-50);

    uint64_t prec = PrecisionCalculator::GetPrecision(rx, ry, true);
    ASSERT_EQ(prec, 100 + ReuseExtra);
}

// ---------------------------------------------------------------------------
// double template overload
// ---------------------------------------------------------------------------

TEST(PC_FromDouble)
{
    // frexp(0.25) → mantissa=0.5, exp=-1 → |exp| = 1
    // frexp(1024.0) → mantissa=0.5, exp=11 → |exp| = 11
    // max(1, 11) = 11 → 11 + 120 = 131
    uint64_t prec = PrecisionCalculator::GetPrecision(0.25, 1024.0, false);
    ASSERT_EQ(prec, 11 + MinExtra);
}

// ---------------------------------------------------------------------------
// float template overload
// ---------------------------------------------------------------------------

TEST(PC_FromFloat)
{
    // frexp(0.125f) → exp = -2, frexp(64.0f) → exp = 7
    // max(|-2|, |7|) = 7 → 7 + 120 = 127
    uint64_t prec = PrecisionCalculator::GetPrecision(0.125f, 64.0f, false);
    ASSERT_EQ(prec, 7 + MinExtra);
}

// ---------------------------------------------------------------------------
// RequiresReuse flag difference
// ---------------------------------------------------------------------------

TEST(PC_ReuseDifference)
{
    // Same input, different RequiresReuse → difference should be exactly 680
    HDRFloat<double> rx(1.0);
    HDRFloat<double> ry(1.0);

    uint64_t precNormal = PrecisionCalculator::GetPrecision(rx, ry, false);
    uint64_t precReuse = PrecisionCalculator::GetPrecision(rx, ry, true);

    ASSERT_EQ(precReuse - precNormal, ReuseExtra - MinExtra);
}
