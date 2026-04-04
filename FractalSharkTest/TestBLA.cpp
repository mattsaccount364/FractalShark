#include "TestFramework.h"
#include "BLA.h"

#include <cmath>

// BLA<double> stores complex coefficients A=(Ax,Ay), B=(Bx,By), threshold r2, level l.
// getValue: Δz' = A·Δz + B·Δc (complex multiply-add)
// getNewA: A_new = y.A * x.A (complex multiply)
// getNewB: B_new = y.A * x.B + y.B (complex multiply + add)

using BLAd = BLA<double>;

// ---------------------------------------------------------------------------
// Construction & accessors
// ---------------------------------------------------------------------------

TEST(BLA_DefaultConstruction)
{
    BLAd b;
    ASSERT_NEAR(b.getR2(), 0.0, 1e-15);
    ASSERT_EQ(b.getL(), 0);
}

TEST(BLA_ParameterizedConstruction)
{
    // r2=4.0, A=(1,0), B=(0,1), level=3
    BLAd b(4.0, 1.0, 0.0, 0.0, 1.0, 3);
    ASSERT_NEAR(b.getR2(), 4.0, 1e-15);
    ASSERT_EQ(b.getL(), 3);
}

TEST(BLA_GetGenericStep)
{
    BLAd b = BLAd::getGenericStep(16.0, 2.0, 3.0, 4.0, 5.0, 7);
    ASSERT_NEAR(b.getR2(), 16.0, 1e-15);
    ASSERT_EQ(b.getL(), 7);
}

// ---------------------------------------------------------------------------
// getValue: Δz' = A·Δz + B·Δc
// ---------------------------------------------------------------------------

TEST(BLA_GetValue_Identity)
{
    // A = (1,0) (identity), B = (0,0) → Δz' = Δz
    BLAd b(4.0, 1.0, 0.0, 0.0, 0.0, 1);
    double dzR = 3.0, dzI = 4.0;
    double dc0R = 10.0, dc0I = 20.0;
    b.getValue(dzR, dzI, dc0R, dc0I);
    ASSERT_NEAR(dzR, 3.0, 1e-14);
    ASSERT_NEAR(dzI, 4.0, 1e-14);
}

TEST(BLA_GetValue_BOnly)
{
    // A = (0,0), B = (1,0) → Δz' = Δc
    BLAd b(4.0, 0.0, 0.0, 1.0, 0.0, 1);
    double dzR = 99.0, dzI = 99.0;
    double dc0R = 5.0, dc0I = 7.0;
    b.getValue(dzR, dzI, dc0R, dc0I);
    ASSERT_NEAR(dzR, 5.0, 1e-14);
    ASSERT_NEAR(dzI, 7.0, 1e-14);
}

TEST(BLA_GetValue_ComplexMultiply)
{
    // A = (1,2), B = (3,4), Δz = (1,0), Δc = (0,0)
    // Result = A·Δz = (1+2i)·(1+0i) = (1, 2)
    BLAd b(4.0, 1.0, 2.0, 3.0, 4.0, 1);
    double dzR = 1.0, dzI = 0.0;
    double dc0R = 0.0, dc0I = 0.0;
    b.getValue(dzR, dzI, dc0R, dc0I);
    ASSERT_NEAR(dzR, 1.0, 1e-14);
    ASSERT_NEAR(dzI, 2.0, 1e-14);
}

TEST(BLA_GetValue_Full)
{
    // A = (2,0), B = (0,3), Δz = (1,1), Δc = (1,0)
    // Re = 2*1 - 0*1 + 0*1 - 3*0 = 2
    // Im = 2*1 + 0*1 + 0*0 + 3*1 = 5
    BLAd b(4.0, 2.0, 0.0, 0.0, 3.0, 1);
    double dzR = 1.0, dzI = 1.0;
    double dc0R = 1.0, dc0I = 0.0;
    b.getValue(dzR, dzI, dc0R, dc0I);
    ASSERT_NEAR(dzR, 2.0, 1e-14);
    ASSERT_NEAR(dzI, 5.0, 1e-14);
}

// ---------------------------------------------------------------------------
// getNewA: A_new = y.A * x.A (complex multiply)
// ---------------------------------------------------------------------------

TEST(BLA_GetNewA)
{
    // x.A = (1,2), y.A = (3,4)
    // A_new = (3+4i)*(1+2i) = (3-8) + (6+4)i = (-5, 10)
    BLAd x(1.0, 1.0, 2.0, 0.0, 0.0, 1);
    BLAd y(1.0, 3.0, 4.0, 0.0, 0.0, 1);
    double realA, imagA;
    BLAd::getNewA(x, y, realA, imagA);
    ASSERT_NEAR(realA, -5.0, 1e-14);
    ASSERT_NEAR(imagA, 10.0, 1e-14);
}

// ---------------------------------------------------------------------------
// getNewB: B_new = y.A * x.B + y.B
// ---------------------------------------------------------------------------

TEST(BLA_GetNewB)
{
    // x.B = (1,0), y.A = (2,0), y.B = (10,20)
    // B_new = (2+0i)*(1+0i) + (10+20i) = (2+10, 0+20) = (12, 20)
    BLAd x(1.0, 0.0, 0.0, 1.0, 0.0, 1);
    BLAd y(1.0, 2.0, 0.0, 10.0, 20.0, 1);
    double realB, imagB;
    BLAd::getNewB(x, y, realB, imagB);
    ASSERT_NEAR(realB, 12.0, 1e-14);
    ASSERT_NEAR(imagB, 20.0, 1e-14);
}

TEST(BLA_GetNewB_Complex)
{
    // x.B = (1,1), y.A = (0,1), y.B = (0,0)
    // B_new = (0+1i)*(1+1i) + (0+0i) = (0-1) + (1+0)i = (-1, 1)
    BLAd x(1.0, 0.0, 0.0, 1.0, 1.0, 1);
    BLAd y(1.0, 0.0, 1.0, 0.0, 0.0, 1);
    double realB, imagB;
    BLAd::getNewB(x, y, realB, imagB);
    ASSERT_NEAR(realB, -1.0, 1e-14);
    ASSERT_NEAR(imagB, 1.0, 1e-14);
}

// ---------------------------------------------------------------------------
// hypotA / hypotB
// ---------------------------------------------------------------------------

TEST(BLA_HypotA)
{
    // A = (3,4) → |A| = 5
    BLAd b(1.0, 3.0, 4.0, 0.0, 0.0, 1);
    ASSERT_NEAR(b.hypotA(), 5.0, 1e-10);
}

TEST(BLA_HypotB)
{
    // B = (5,12) → |B| = 13
    BLAd b(1.0, 0.0, 0.0, 5.0, 12.0, 1);
    ASSERT_NEAR(b.hypotB(), 13.0, 1e-10);
}
