#include "TestFramework.h"
#include "ATInfo.h"

#include <cmath>

// Non-HDR instantiation for simplest testing
using ATInfoD = ATInfo<uint64_t, double, double>;
using ATResultD = ATResult<uint64_t, double, double>;
using FC = FloatComplex<double>;

// Helper: build a minimal ATInfo with known coefficients for testing
static ATInfoD
MakeSimpleATInfo(double sqrEscapeRadius, double thresholdC, double stepLength,
                 FC refC, FC zCoeff, FC cCoeff)
{
    ATInfoD at;
    at.StepLength = static_cast<uint64_t>(stepLength);
    at.SqrEscapeRadius = sqrEscapeRadius;
    at.ThresholdC = thresholdC;
    at.RefC = refC;
    at.ZCoeff = zCoeff;
    at.CCoeff = cCoeff;

    // Precompute derived fields
    double zr = zCoeff.getRe(), zi = zCoeff.getIm();
    double denom = zr * zr + zi * zi;
    if (denom > 0) {
        at.InvZCoeff = FC(zr / denom, -zi / denom); // 1/ZCoeff
    } else {
        at.InvZCoeff = FC(0.0, 0.0);
    }
    at.CCoeffInvZCoeff = FC(
        cCoeff.getRe() * at.InvZCoeff.getRe() - cCoeff.getIm() * at.InvZCoeff.getIm(),
        cCoeff.getRe() * at.InvZCoeff.getIm() + cCoeff.getIm() * at.InvZCoeff.getRe());
    at.CCoeffNormSqr = cCoeff.norm_squared();
    at.RefCNormSqr = refC.norm_squared();
    at.factor = std::pow(2.0, 32);
    return at;
}

// ---------------------------------------------------------------------------
// isValid: |DeltaSub0| <= ThresholdC
// ---------------------------------------------------------------------------

TEST(ATInfo_IsValid_BelowThreshold)
{
    ATInfoD at = MakeSimpleATInfo(256.0, 1.0, 1, FC(0, 0), FC(1, 0), FC(1, 0));
    // Delta with chebychev norm < 1.0
    ASSERT_TRUE(at.isValid(FC(0.5, 0.3)));
}

TEST(ATInfo_IsValid_AboveThreshold)
{
    ATInfoD at = MakeSimpleATInfo(256.0, 1.0, 1, FC(0, 0), FC(1, 0), FC(1, 0));
    // Delta with chebychev norm > 1.0
    ASSERT_FALSE(at.isValid(FC(2.0, 0.0)));
}

// ---------------------------------------------------------------------------
// getC: dc * CCoeff + RefC
// ---------------------------------------------------------------------------

TEST(ATInfo_GetC)
{
    // CCoeff = (2,0), RefC = (1,1)
    // getC((0.5, 0)) = (0.5,0)*(2,0) + (1,1) = (1,0) + (1,1) = (2,1)
    ATInfoD at = MakeSimpleATInfo(256.0, 10.0, 1, FC(1, 1), FC(1, 0), FC(2, 0));
    FC result = at.getC(FC(0.5, 0.0));
    ASSERT_NEAR(result.getRe(), 2.0, 1e-10);
    ASSERT_NEAR(result.getIm(), 1.0, 1e-10);
}

// ---------------------------------------------------------------------------
// getDZ: z * InvZCoeff
// ---------------------------------------------------------------------------

TEST(ATInfo_GetDZ)
{
    // ZCoeff = (2,0) → InvZCoeff = (0.5, 0)
    // getDZ((4,6)) = (4,6)*(0.5,0) = (2,3)
    ATInfoD at = MakeSimpleATInfo(256.0, 10.0, 1, FC(0, 0), FC(2, 0), FC(1, 0));
    FC result = at.getDZ(FC(4.0, 6.0));
    ASSERT_NEAR(result.getRe(), 2.0, 1e-10);
    ASSERT_NEAR(result.getIm(), 3.0, 1e-10);
}

// ---------------------------------------------------------------------------
// PerformAT: self-contained Mandelbrot iteration
// ---------------------------------------------------------------------------

TEST(ATInfo_PerformAT_FixedPoint)
{
    // c = 0 → z stays at 0 forever (fixed point)
    // CCoeff = 1, RefC = 0 → getC(delta) = delta * 1 + 0 = delta
    // With delta = 0 → c = 0 → z stays 0
    ATInfoD at = MakeSimpleATInfo(256.0, 10.0, 1, FC(0, 0), FC(1, 0), FC(1, 0));
    ATResultD result;
    at.PerformAT(100, FC(0.0, 0.0), result);
    // Should iterate all 100 times without escaping
    ASSERT_EQ(result.bla_iterations, static_cast<uint64_t>(100));
    ASSERT_EQ(result.bla_steps, static_cast<uint64_t>(100));
}

TEST(ATInfo_PerformAT_ImmediateEscape)
{
    // Set up so c is large → z escapes immediately
    // CCoeff = 1, RefC = (100, 0) → c = delta + (100,0)
    // With delta = 0 → c = 100 → z_1 = 100, |z_1|² = 10000 > 256
    ATInfoD at = MakeSimpleATInfo(256.0, 1000.0, 1, FC(100, 0), FC(1, 0), FC(1, 0));
    ATResultD result;
    at.PerformAT(100, FC(0.0, 0.0), result);
    // Should escape after 1 iteration (z_1 = 100, |z_1|² = 10000 > 256)
    ASSERT_EQ(result.bla_steps, static_cast<uint64_t>(1));
}

TEST(ATInfo_PerformAT_Period2)
{
    // c = -1 → period-2 orbit: 0 → -1 → 0 → -1 ...
    // CCoeff = 1, RefC = (-1, 0) → c = delta + (-1,0) = -1 (for delta=0)
    ATInfoD at = MakeSimpleATInfo(256.0, 1000.0, 1, FC(-1, 0), FC(1, 0), FC(1, 0));
    ATResultD result;
    at.PerformAT(100, FC(0.0, 0.0), result);
    // |z| never exceeds 1 < 256, so all iterations complete
    ASSERT_EQ(result.bla_iterations, static_cast<uint64_t>(100));
}

TEST(ATInfo_PerformAT_KnownEscapeTime)
{
    // c = 1 → z: 0 → 1 → 2 → 5 → 26 → ...
    // |z_3|² = 25 < 256, |z_4|² = 676 > 256 → escapes at step 4
    ATInfoD at = MakeSimpleATInfo(256.0, 1000.0, 1, FC(1, 0), FC(1, 0), FC(1, 0));
    ATResultD result;
    at.PerformAT(100, FC(0.0, 0.0), result);
    ASSERT_EQ(result.bla_steps, static_cast<uint64_t>(4));
}

TEST(ATInfo_PerformAT_StepLength)
{
    // StepLength = 5 → bla_iterations = bla_steps * 5
    ATInfoD at = MakeSimpleATInfo(256.0, 1000.0, 5, FC(1, 0), FC(1, 0), FC(1, 0));
    ATResultD result;
    at.PerformAT(100, FC(0.0, 0.0), result);
    ASSERT_EQ(result.bla_iterations, result.bla_steps * 5);
}

// ---------------------------------------------------------------------------
// Default construction
// ---------------------------------------------------------------------------

TEST(ATInfo_DefaultConstruction)
{
    ATInfoD at;
    ASSERT_EQ(at.StepLength, static_cast<uint64_t>(0));
}
