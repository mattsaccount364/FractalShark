#include "TestFramework.h"
#include "HDRFloatComplex.h"
#include "HDRFloat.h"

#include <cmath>
#include <limits>
#include <sstream>

using HDRd = HDRFloat<double>;
using HDRCd = HDRFloatComplex<double>;

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

TEST(HDRFC_DefaultConstruction)
{
    HDRCd z;
    ASSERT_NEAR(static_cast<double>(z.getRe()), 0.0, 1e-15);
    ASSERT_NEAR(static_cast<double>(z.getIm()), 0.0, 1e-15);
}

TEST(HDRFC_FromScalars)
{
    HDRCd z(3.0, 4.0);
    ASSERT_NEAR(static_cast<double>(z.getRe()), 3.0, 1e-10);
    ASSERT_NEAR(static_cast<double>(z.getIm()), 4.0, 1e-10);
}

TEST(HDRFC_FromHDRFloats)
{
    HDRd re(3.0);
    HDRd im(4.0);
    HDRCd z(re, im);
    ASSERT_NEAR(static_cast<double>(z.getRe()), 3.0, 1e-10);
    ASSERT_NEAR(static_cast<double>(z.getIm()), 4.0, 1e-10);
}

TEST(HDRFC_CrossTypeCopy)
{
    HDRFloatComplex<float> zf(3.0f, 4.0f);
    HDRCd zd(zf);
    ASSERT_NEAR(static_cast<double>(zd.getRe()), 3.0, 1e-5);
    ASSERT_NEAR(static_cast<double>(zd.getIm()), 4.0, 1e-5);
}

// ---------------------------------------------------------------------------
// Arithmetic
// ---------------------------------------------------------------------------

TEST(HDRFC_Addition)
{
    HDRCd a(1.0, 2.0);
    HDRCd b(3.0, 4.0);
    HDRCd c = a + b;
    ASSERT_NEAR(static_cast<double>(c.getRe()), 4.0, 1e-10);
    ASSERT_NEAR(static_cast<double>(c.getIm()), 6.0, 1e-10);
}

TEST(HDRFC_Subtraction)
{
    HDRCd a(5.0, 7.0);
    HDRCd b(2.0, 3.0);
    HDRCd c = a - b;
    ASSERT_NEAR(static_cast<double>(c.getRe()), 3.0, 1e-10);
    ASSERT_NEAR(static_cast<double>(c.getIm()), 4.0, 1e-10);
}

TEST(HDRFC_Multiplication)
{
    // (1+2i)(3+4i) = (1*3 - 2*4) + (1*4 + 2*3)i = -5 + 10i
    HDRCd a(1.0, 2.0);
    HDRCd b(3.0, 4.0);
    HDRCd c = a * b;
    ASSERT_NEAR(static_cast<double>(c.getRe()), -5.0, 1e-10);
    ASSERT_NEAR(static_cast<double>(c.getIm()), 10.0, 1e-10);
}

TEST(HDRFC_Division)
{
    // (1+2i)/(1+0i) = 1+2i
    HDRCd a(1.0, 2.0);
    HDRCd b(1.0, 0.0);
    HDRCd c = a / b;
    ASSERT_NEAR(static_cast<double>(c.getRe()), 1.0, 1e-10);
    ASSERT_NEAR(static_cast<double>(c.getIm()), 2.0, 1e-10);

    // (3+4i)/(1+2i) = (3+4i)(1-2i)/((1+2i)(1-2i)) = (3+8+4i-6i)/5 = (11-2i)/5
    HDRCd d(3.0, 4.0);
    HDRCd e(1.0, 2.0);
    HDRCd f = d / e;
    ASSERT_NEAR(static_cast<double>(f.getRe()), 2.2, 1e-10);
    ASSERT_NEAR(static_cast<double>(f.getIm()), -0.4, 1e-10);
}

TEST(HDRFC_AddHDRFloat)
{
    // Adding HDRFloat adds to real part only
    HDRCd a(1.0, 2.0);
    HDRd s(5.0);
    HDRCd c = a + s;
    ASSERT_NEAR(static_cast<double>(c.getRe()), 6.0, 1e-10);
    ASSERT_NEAR(static_cast<double>(c.getIm()), 2.0, 1e-10);
}

TEST(HDRFC_MultiplyHDRFloat)
{
    // Scalar multiply scales both components
    HDRCd a(3.0, 4.0);
    HDRd s(2.0);
    HDRCd c = a * s;
    ASSERT_NEAR(static_cast<double>(c.getRe()), 6.0, 1e-10);
    ASSERT_NEAR(static_cast<double>(c.getIm()), 8.0, 1e-10);
}

// ---------------------------------------------------------------------------
// Norm / Special operations
// ---------------------------------------------------------------------------

TEST(HDRFC_Norm)
{
    // |3+4i| = 5
    HDRCd z(3.0, 4.0);
    HDRd n = z.norm();
    ASSERT_NEAR(static_cast<double>(n), 5.0, 1e-10);
}

TEST(HDRFC_NormSquared)
{
    // |3+4i|² = 25
    HDRCd z(3.0, 4.0);
    HDRd ns = z.norm_squared();
    ASSERT_NEAR(static_cast<double>(ns), 25.0, 1e-10);
}

TEST(HDRFC_ChebychevNorm)
{
    // max(|3|, |4|) = 4
    HDRCd z(3.0, 4.0);
    HDRd cn = z.chebychevNorm();
    ASSERT_NEAR(static_cast<double>(cn), 4.0, 1e-10);
}

TEST(HDRFC_Reciprocal)
{
    // 1/(1+0i) = 1+0i
    HDRCd one(1.0, 0.0);
    HDRCd r = one.reciprocal();
    ASSERT_NEAR(static_cast<double>(r.getRe()), 1.0, 1e-10);
    ASSERT_NEAR(static_cast<double>(r.getIm()), 0.0, 1e-10);

    // 1/(0+1i) = 0-1i
    HDRCd i_unit(0.0, 1.0);
    HDRCd ri = i_unit.reciprocal();
    ASSERT_NEAR(static_cast<double>(ri.getRe()), 0.0, 1e-10);
    ASSERT_NEAR(static_cast<double>(ri.getIm()), -1.0, 1e-10);
}

// ---------------------------------------------------------------------------
// Comparison & edge cases
// ---------------------------------------------------------------------------

TEST(HDRFC_Equality)
{
    HDRCd a(1.0, 2.0);
    HDRCd b(1.0, 2.0);
    HDRCd c(1.0, 3.0);
    ASSERT_TRUE(a == b);
    ASSERT_TRUE(a != c);
}

TEST(HDRFC_PureReal)
{
    // Operations on pure-real complex numbers: imaginary should stay zero
    HDRCd a(5.0, 0.0);
    HDRCd b(3.0, 0.0);
    HDRCd sum = a + b;
    ASSERT_NEAR(static_cast<double>(sum.getRe()), 8.0, 1e-10);
    ASSERT_NEAR(static_cast<double>(sum.getIm()), 0.0, 1e-10);

    HDRCd prod = a * b;
    ASSERT_NEAR(static_cast<double>(prod.getRe()), 15.0, 1e-10);
    ASSERT_NEAR(static_cast<double>(prod.getIm()), 0.0, 1e-10);
}

TEST(HDRFC_PureImaginary)
{
    // i × i = -1
    HDRCd i_unit(0.0, 1.0);
    HDRCd isq = i_unit * i_unit;
    ASSERT_NEAR(static_cast<double>(isq.getRe()), -1.0, 1e-10);
    ASSERT_NEAR(static_cast<double>(isq.getIm()), 0.0, 1e-10);
}

// ---------------------------------------------------------------------------
// I/O round-trip
// ---------------------------------------------------------------------------

TEST(HDRFC_IO_DecimalRoundtrip)
{
    HDRCd original(1.5, -2.5);

    std::string s = original.ToString<false>();
    std::istringstream iss(s);

    HDRCd restored;
    restored.FromIStream<false>(iss);

    ASSERT_NEAR(static_cast<double>(restored.getRe()),
                static_cast<double>(original.getRe()), 1e-10);
    ASSERT_NEAR(static_cast<double>(restored.getIm()),
                static_cast<double>(original.getIm()), 1e-10);
}

TEST(HDRFC_IO_HexRoundtrip)
{
    HDRCd original(1.23456789, -9.87654321);

    std::string s = original.ToString<true>();
    std::istringstream iss(s);

    HDRCd restored;
    restored.FromIStream<true>(iss);

    ASSERT_TRUE(original == restored);
}

// ---------------------------------------------------------------------------
// Exponent handling
// ---------------------------------------------------------------------------

TEST(HDRFC_ExponentDiffIgnored)
{
    // When exponent diff > 120, smaller operand is ignored
    HDRCd big(1.0, 1.0);
    HDRCd tiny(1.0, 1.0);

    // Manually set tiny to have a very small exponent
    // Construct via HDRFloats with different exponents
    HDRd bigR(0, 1.0);
    bigR.setExp(0);
    HDRd tinyR(0, 1.0);
    tinyR.setExp(-200);

    HDRCd bigC(bigR, HDRd(0.0));
    HDRCd tinyC(tinyR, HDRd(0.0));

    HDRCd sum = bigC + tinyC;
    // tiny should be ignored: result ≈ bigC
    ASSERT_NEAR(static_cast<double>(sum.getRe()),
                static_cast<double>(bigC.getRe()), 1e-10);
}

TEST(HDRFC_SharedExponentPrecision)
{
    // When real and imag have very different magnitudes,
    // the smaller component loses precision due to shared exponent
    HDRd largeRe(1e100);
    HDRd tinyIm(1e-100);

    HDRCd z(largeRe, tinyIm);

    // Real part should be preserved
    ASSERT_NEAR(static_cast<double>(z.getRe()), 1e100, 1e90);

    // Imaginary part will be flushed toward zero due to shared exponent
    // (1e-100 scaled by 2^(-big_exponent) → effectively 0)
    double imVal = static_cast<double>(z.getIm());
    ASSERT_TRUE(std::abs(imVal) < 1e-10);
}
