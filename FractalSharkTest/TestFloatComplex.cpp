#include "TestFramework.h"
#include "FloatComplex.h"

#include <cmath>
#include <sstream>

using FCd = FloatComplex<double>;

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

TEST(FC_DefaultConstruction)
{
    FCd z;
    ASSERT_NEAR(z.getRe(), 0.0, 1e-15);
    ASSERT_NEAR(z.getIm(), 0.0, 1e-15);
}

TEST(FC_FromScalars)
{
    FCd z(3.0, 4.0);
    ASSERT_NEAR(z.getRe(), 3.0, 1e-15);
    ASSERT_NEAR(z.getIm(), 4.0, 1e-15);
}

TEST(FC_CopyConstruction)
{
    FCd a(1.5, -2.5);
    FCd b(a);
    ASSERT_TRUE(a == b);
}

TEST(FC_CrossTypeCopy)
{
    FloatComplex<float> zf(3.0f, 4.0f);
    FCd zd(zf);
    ASSERT_NEAR(zd.getRe(), 3.0, 1e-5);
    ASSERT_NEAR(zd.getIm(), 4.0, 1e-5);
}

// ---------------------------------------------------------------------------
// Arithmetic: addition
// ---------------------------------------------------------------------------

TEST(FC_Addition)
{
    FCd a(1.0, 2.0);
    FCd b(3.0, 4.0);
    FCd c = a + b;
    ASSERT_NEAR(c.getRe(), 4.0, 1e-15);
    ASSERT_NEAR(c.getIm(), 6.0, 1e-15);
}

TEST(FC_AddScalar)
{
    FCd a(1.0, 2.0);
    FCd c = a + 5.0;
    ASSERT_NEAR(c.getRe(), 6.0, 1e-15);
    ASSERT_NEAR(c.getIm(), 2.0, 1e-15);
}

TEST(FC_AddAssign)
{
    FCd a(1.0, 2.0);
    FCd b(3.0, 4.0);
    a += b;
    ASSERT_NEAR(a.getRe(), 4.0, 1e-15);
    ASSERT_NEAR(a.getIm(), 6.0, 1e-15);
}

// ---------------------------------------------------------------------------
// Arithmetic: subtraction
// ---------------------------------------------------------------------------

TEST(FC_Subtraction)
{
    FCd a(5.0, 7.0);
    FCd b(2.0, 3.0);
    FCd c = a - b;
    ASSERT_NEAR(c.getRe(), 3.0, 1e-15);
    ASSERT_NEAR(c.getIm(), 4.0, 1e-15);
}

TEST(FC_SubAssign)
{
    FCd a(5.0, 7.0);
    FCd b(2.0, 3.0);
    a -= b;
    ASSERT_NEAR(a.getRe(), 3.0, 1e-15);
    ASSERT_NEAR(a.getIm(), 4.0, 1e-15);
}

// ---------------------------------------------------------------------------
// Arithmetic: multiplication
// ---------------------------------------------------------------------------

TEST(FC_Multiplication)
{
    // (1+2i)(3+4i) = (3-8) + (4+6)i = -5+10i
    FCd a(1.0, 2.0);
    FCd b(3.0, 4.0);
    FCd c = a * b;
    ASSERT_NEAR(c.getRe(), -5.0, 1e-14);
    ASSERT_NEAR(c.getIm(), 10.0, 1e-14);
}

TEST(FC_MultiplyScalar)
{
    FCd a(3.0, 4.0);
    FCd c = a * 2.0;
    ASSERT_NEAR(c.getRe(), 6.0, 1e-15);
    ASSERT_NEAR(c.getIm(), 8.0, 1e-15);
}

TEST(FC_MultiplyAssign)
{
    FCd a(1.0, 2.0);
    FCd b(3.0, 4.0);
    a *= b;
    ASSERT_NEAR(a.getRe(), -5.0, 1e-14);
    ASSERT_NEAR(a.getIm(), 10.0, 1e-14);
}

// ---------------------------------------------------------------------------
// Note: operator/ is NOT tested.
// - Complex-by-complex division: divide_mutable(FloatComplex) is commented out.
// - Scalar division: divide_mutable(SubType) has infinite self-recursion.
// Both are dead code — operator/ on FloatComplex is never instantiated
// in the production codebase.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Comparison
// ---------------------------------------------------------------------------

TEST(FC_Equality)
{
    FCd a(1.0, 2.0);
    FCd b(1.0, 2.0);
    FCd c(1.0, 3.0);
    ASSERT_TRUE(a == b);
    ASSERT_FALSE(a == c);
    ASSERT_TRUE(a != c);
    ASSERT_FALSE(a != b);
}

// ---------------------------------------------------------------------------
// Norm / special operations
// ---------------------------------------------------------------------------

TEST(FC_NormSquared)
{
    // |3+4i|² = 9+16 = 25
    FCd z(3.0, 4.0);
    ASSERT_NEAR(z.norm_squared(), 25.0, 1e-14);
}

TEST(FC_Norm)
{
    // |3+4i| = 5
    FCd z(3.0, 4.0);
    ASSERT_NEAR(z.norm(), 5.0, 1e-14);
}

TEST(FC_Reciprocal)
{
    // 1/(1+0i) = 1
    FCd one(1.0, 0.0);
    FCd r = one.reciprocal();
    ASSERT_NEAR(r.getRe(), 1.0, 1e-14);
    ASSERT_NEAR(r.getIm(), 0.0, 1e-14);

    // 1/(0+1i) = 0-1i
    FCd i_unit(0.0, 1.0);
    FCd ri = i_unit.reciprocal();
    ASSERT_NEAR(ri.getRe(), 0.0, 1e-14);
    ASSERT_NEAR(ri.getIm(), -1.0, 1e-14);

    // 1/(3+4i) = (3-4i)/25 = (0.12, -0.16)
    FCd z(3.0, 4.0);
    FCd rz = z.reciprocal();
    ASSERT_NEAR(rz.getRe(), 0.12, 1e-14);
    ASSERT_NEAR(rz.getIm(), -0.16, 1e-14);
}

// ---------------------------------------------------------------------------
// Pure real / pure imaginary edge cases
// ---------------------------------------------------------------------------

TEST(FC_PureReal)
{
    FCd a(5.0, 0.0);
    FCd b(3.0, 0.0);
    FCd prod = a * b;
    ASSERT_NEAR(prod.getRe(), 15.0, 1e-14);
    ASSERT_NEAR(prod.getIm(), 0.0, 1e-14);
}

TEST(FC_PureImaginary)
{
    // i × i = -1
    FCd i_unit(0.0, 1.0);
    FCd isq = i_unit * i_unit;
    ASSERT_NEAR(isq.getRe(), -1.0, 1e-14);
    ASSERT_NEAR(isq.getIm(), 0.0, 1e-14);
}

// ---------------------------------------------------------------------------
// I/O round-trip
// ---------------------------------------------------------------------------

TEST(FC_IO_DecimalRoundtrip)
{
    FCd original(1.5, -2.5);

    std::string s = original.ToString<false>();
    std::istringstream iss(s);

    FCd restored;
    restored.FromIStream<false>(iss);

    ASSERT_NEAR(restored.getRe(), original.getRe(), 1e-10);
    ASSERT_NEAR(restored.getIm(), original.getIm(), 1e-10);
}

TEST(FC_IO_HexRoundtrip)
{
    FCd original(1.23456789, -9.87654321);

    std::string s = original.ToString<true>();
    std::istringstream iss(s);

    FCd restored;
    restored.FromIStream<true>(iss);

    ASSERT_TRUE(original == restored);
}
