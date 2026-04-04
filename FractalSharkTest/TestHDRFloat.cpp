#include "TestFramework.h"
#include "HDRFloat.h"
#include "HighPrecision.h"

#include <cmath>
#include <limits>
#include <sstream>

// Alias for convenience
using HDRd = HDRFloat<double>;
using HDRf = HDRFloat<float>;
using HRReal = Imagina::HRReal; // HDRFloat<double, HDROrder::Left, int64_t>

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

TEST(HDR_DefaultConstruction)
{
    HDRd x;
    ASSERT_NEAR(x.getMantissa(), 0.0, 1e-15);
    ASSERT_EQ(x.getExp(), HDRd::MIN_BIG_EXPONENT());
}

TEST(HDR_FromMantissaOnly)
{
    // Constructing from a scalar auto-reduces.
    // 8.0 = 1.0 × 2^3 → mantissa ∈ [1,2), exp adjusted by +3
    HDRd x(8.0);
    ASSERT_NEAR(x.getMantissa(), 1.0, 1e-15);
    ASSERT_EQ(x.getExp(), 3);
}

TEST(HDR_FromDouble)
{
    HDRd x(1.5);
    // 1.5 is already in [1,2) → mantissa=1.5, exp=0
    ASSERT_NEAR(x.getMantissa(), 1.5, 1e-15);
    ASSERT_EQ(x.getExp(), 0);
}

TEST(HDR_FromFloat)
{
    HDRf x(4.0f);
    // 4.0 = 1.0 × 2^2 → mantissa=1.0, exp=2
    ASSERT_NEAR(x.getMantissa(), 1.0f, 1e-6f);
    ASSERT_EQ(x.getExp(), 2);
}

TEST(HDR_FromInt)
{
    HDRd x(16);
    // 16 = 1.0 × 2^4
    ASSERT_NEAR(x.getMantissa(), 1.0, 1e-10);
    ASSERT_EQ(x.getExp(), 4);
}

TEST(HDR_FromExponentAndMantissa)
{
    // Direct (exp, mantissa) constructor: no normalization
    HDRd x(10, 3.14);
    ASSERT_NEAR(x.getMantissa(), 3.14, 1e-15);
    ASSERT_EQ(x.getExp(), 10);
}

TEST(HDR_ZeroScalar)
{
    // HDRd(0.0) selects the HDRFloat(T mant) constructor (non-template, exact match),
    // which sets exp=0 and calls HdrReduce (no-op for zero mantissa).
    // The scalar template constructor HDRFloat(const U number) would set MIN_BIG_EXPONENT,
    // but it's a worse overload match when U==T.
    HDRd x(0.0);
    ASSERT_NEAR(x.getMantissa(), 0.0, 1e-15);
    ASSERT_EQ(x.getExp(), 0);

    // Default constructor DOES use MIN_BIG_EXPONENT
    HDRd def;
    ASSERT_EQ(def.getExp(), HDRd::MIN_BIG_EXPONENT());

    // Scalar constructor from int uses MIN_BIG_EXPONENT for zero
    HDRd fromInt(0);
    ASSERT_EQ(fromInt.getExp(), HDRd::MIN_BIG_EXPONENT());
}

TEST(HDR_CrossOrderCopy)
{
    using HDRdR = HDRFloat<double, HDROrder::Right>;
    HDRd left(1.5);
    HDRdR right(left);
    ASSERT_NEAR(right.getMantissa(), left.getMantissa(), 1e-15);
    ASSERT_EQ(right.getExp(), left.getExp());
}

TEST(HDR_FromHighPrecision)
{
    HighPrecision hp(256.0);
    HDRd x(hp);
    // The mpf_t constructor uses GMP's mantissa convention [0.5, 1.0),
    // so the (mantissa, exp) decomposition differs from the double constructor.
    // Verify the numeric value is correct regardless.
    ASSERT_NEAR(static_cast<double>(x), 256.0, 1e-10);
}

// ---------------------------------------------------------------------------
// Reduce / Normalization
// ---------------------------------------------------------------------------

TEST(HDR_Reduce_AlreadyNormalized)
{
    HDRd x(10, 1.5); // mantissa in [1,2), already normalized
    auto origExp = x.getExp();
    x.Reduce();
    ASSERT_NEAR(x.getMantissa(), 1.5, 1e-15);
    ASSERT_EQ(x.getExp(), origExp); // exp unchanged
}

TEST(HDR_Reduce_Denormalized)
{
    HDRd x(0, 8.0); // 8.0 = 1.0 × 2^3 → after Reduce: mantissa=1.0, exp=3
    x.Reduce();
    ASSERT_NEAR(x.getMantissa(), 1.0, 1e-15);
    ASSERT_EQ(x.getExp(), 3);
}

TEST(HDR_Reduce_Zero)
{
    HDRd x(42, 0.0); // Zero mantissa: Reduce is a no-op
    x.Reduce();
    ASSERT_NEAR(x.getMantissa(), 0.0, 1e-15);
    ASSERT_EQ(x.getExp(), 42); // exp unchanged
}

TEST(HDR_Reduce_GetExpAmt)
{
    HDRd x(5, 16.0); // 16.0 = 1.0 × 2^4
    int32_t extracted = 0;
    x.Reduce<true>(&extracted);
    ASSERT_EQ(extracted, 4);
    ASSERT_NEAR(x.getMantissa(), 1.0, 1e-15);
    ASSERT_EQ(x.getExp(), 9); // 5 + 4
}

TEST(HDR_Reduce_Idempotent)
{
    HDRd x(100.0);
    HDRd after1 = x;
    after1.Reduce();
    HDRd after2 = after1;
    after2.Reduce();
    ASSERT_NEAR(after1.getMantissa(), after2.getMantissa(), 1e-15);
    ASSERT_EQ(after1.getExp(), after2.getExp());
}

TEST(HDR_Reduce_SmallValue)
{
    // A very small double: 2^-100
    double small = std::ldexp(1.0, -100);
    HDRd x(small);
    // After construction (which calls Reduce): mantissa should be ~1.0, exp ~ -100
    ASSERT_NEAR(x.getMantissa(), 1.0, 1e-10);
    ASSERT_EQ(x.getExp(), -100);
}

// ---------------------------------------------------------------------------
// Arithmetic
// ---------------------------------------------------------------------------

TEST(HDR_Addition)
{
    HDRd a(1.0); // 1.0 × 2^0
    HDRd b(1.0); // 1.0 × 2^0
    HDRd c = a + b;
    // 1+1 = 2 = 1.0 × 2^1
    double result = static_cast<double>(c);
    ASSERT_NEAR(result, 2.0, 1e-10);
}

TEST(HDR_Subtraction)
{
    HDRd a(3.0);
    HDRd b(1.0);
    HDRd c = a - b;
    double result = static_cast<double>(c);
    ASSERT_NEAR(result, 2.0, 1e-10);
}

TEST(HDR_Multiplication)
{
    HDRd a(3.0);
    HDRd b(4.0);
    HDRd c = a * b;
    double result = static_cast<double>(c);
    ASSERT_NEAR(result, 12.0, 1e-10);
}

TEST(HDR_Division)
{
    HDRd a(12.0);
    HDRd b(4.0);
    HDRd c = a / b;
    double result = static_cast<double>(c);
    ASSERT_NEAR(result, 3.0, 1e-10);
}

TEST(HDR_Negation)
{
    HDRd a(5.0);
    HDRd neg = a.negate();
    ASSERT_NEAR(neg.getMantissa(), -a.getMantissa(), 1e-15);
    ASSERT_EQ(neg.getExp(), a.getExp());
    ASSERT_NEAR(static_cast<double>(neg), -5.0, 1e-10);
}

TEST(HDR_Square)
{
    HDRd a(3.0);
    HDRd sq = a.square();
    double result = static_cast<double>(sq);
    ASSERT_NEAR(result, 9.0, 1e-10);
}

TEST(HDR_Multiply2_Divide2)
{
    HDRd a(5.0); // 5.0 = 1.25 × 2^2

    HDRd m2 = a.multiply2();
    ASSERT_NEAR(static_cast<double>(m2), 10.0, 1e-10);
    ASSERT_EQ(m2.getExp(), a.getExp() + 1);

    HDRd d2 = a.divide2();
    ASSERT_NEAR(static_cast<double>(d2), 2.5, 1e-10);
    ASSERT_EQ(d2.getExp(), a.getExp() - 1);

    HDRd m4 = a.multiply4();
    ASSERT_NEAR(static_cast<double>(m4), 20.0, 1e-10);

    HDRd d4 = a.divide4();
    ASSERT_NEAR(static_cast<double>(d4), 1.25, 1e-10);

    // Mutable variants
    HDRd am = a;
    am.multiply2_mutable();
    ASSERT_NEAR(static_cast<double>(am), 10.0, 1e-10);

    HDRd ad = a;
    ad.divide2_mutable();
    ASSERT_NEAR(static_cast<double>(ad), 2.5, 1e-10);

    HDRd ad4 = a;
    ad4.divide4_mutable();
    ASSERT_NEAR(static_cast<double>(ad4), 1.25, 1e-10);
}

TEST(HDR_ExponentDiffIgnored)
{
    // When exponent difference > 120, the smaller value is ignored in addition
    HDRd big(0, 1.5);   // 1.5 × 2^0
    HDRd tiny(0, 1.0);
    tiny.setExp(-200);   // 1.0 × 2^-200 (negligible compared to big)

    HDRd sum = big + tiny;
    // tiny should be ignored because |0 - (-200)| = 200 > 120
    ASSERT_NEAR(static_cast<double>(sum), static_cast<double>(big), 1e-15);
}

// ---------------------------------------------------------------------------
// Comparison
// ---------------------------------------------------------------------------

TEST(HDR_CompareTo_Basic)
{
    HDRd a(10.0);
    HDRd b(5.0);
    HDRd c(10.0);

    ASSERT_TRUE(a.compareTo(b) > 0);  // 10 > 5
    ASSERT_TRUE(b.compareTo(a) < 0);  // 5 < 10
    ASSERT_EQ(a.compareTo(c), 0);     // 10 == 10
}

TEST(HDR_CompareTo_Negative)
{
    HDRd pos(3.0);
    HDRd neg(-3.0);
    HDRd zero(0.0);

    ASSERT_TRUE(pos.compareTo(neg) > 0);
    ASSERT_TRUE(neg.compareTo(pos) < 0);
    ASSERT_TRUE(pos.compareTo(zero) > 0);
    ASSERT_TRUE(neg.compareTo(zero) < 0);
}

TEST(HDR_UnnormalizedEquality)
{
    // Same numeric value (512), different (exp, mantissa) decomposition
    HDRd a(0, 1.0);
    a.setExp(9);  // 1.0 × 2^9 = 512

    HDRd b(0, 0.5);
    b.setExp(10); // 0.5 × 2^10 = 512

    // operator== compares raw fields → NOT equal
    ASSERT_FALSE(a == b);

    // After Reduce, both should be normalized to same representation
    a.Reduce();
    b.Reduce();
    ASSERT_TRUE(a == b);
}

TEST(HDR_MaxMin)
{
    HDRd a(3.0);
    HDRd b(7.0);

    HDRd mx = HDRd::HDRMax(a, b);
    HDRd mn = HDRd::HDRMin(a, b);

    ASSERT_NEAR(static_cast<double>(mx), 7.0, 1e-10);
    ASSERT_NEAR(static_cast<double>(mn), 3.0, 1e-10);
}

TEST(HDR_ZeroComparisons)
{
    HDRd zero(0.0);
    HDRd pos(1.0);
    HDRd neg(-1.0);

    ASSERT_TRUE(zero.compareTo(pos) < 0);
    ASSERT_TRUE(zero.compareTo(neg) > 0);
    ASSERT_EQ(zero.compareTo(HDRd(0.0)), 0);
}

// ---------------------------------------------------------------------------
// Conversion
// ---------------------------------------------------------------------------

TEST(HDR_ToDouble_Roundtrip)
{
    double original = 123.456;
    HDRd x(original);
    double back = static_cast<double>(x);
    ASSERT_NEAR(back, original, 1e-10);
}

TEST(HDR_ToDoubleSub)
{
    HDRd x(0, 1.5);
    x.setExp(10); // 1.5 × 2^10

    // toDoubleSub(3) = mantissa × 2^(exp - 3) = 1.5 × 2^7 = 192
    double result = x.toDoubleSub(3);
    ASSERT_NEAR(result, 192.0, 1e-10);
}

TEST(HDR_Conversion_Underflow)
{
    // Very large negative exponent → toDouble returns 0
    HDRd x(0, 1.5);
    x.setExp(-2000);
    double result = static_cast<double>(x);
    ASSERT_NEAR(result, 0.0, 1e-300);
}

TEST(HDR_Conversion_Overflow)
{
    // Very large positive exponent → toDouble returns inf (not max)
    // because getMultiplier returns max, and max * mantissa(>1) overflows to inf
    HDRd x(0, 1.5);
    x.setExp(2000);
    double result = static_cast<double>(x);
    ASSERT_TRUE(result == std::numeric_limits<double>::infinity() ||
                result == std::numeric_limits<double>::max());
}

TEST(HDR_GetHighPrecision_Roundtrip)
{
    HDRd original(42.5);
    HighPrecision hp;
    original.GetHighPrecision(hp);

    // Verify the HighPrecision value is correct
    ASSERT_NEAR(static_cast<double>(hp), 42.5, 1e-10);

    // Round-trip: the (mantissa, exp) decomposition may differ due to GMP's
    // [0.5, 1.0) mantissa convention vs HDRFloat's [1.0, 2.0), but the
    // numeric value should be preserved.
    HDRd back(hp);
    ASSERT_NEAR(static_cast<double>(back), 42.5, 1e-10);
}

// ---------------------------------------------------------------------------
// I/O Round-Trip
// ---------------------------------------------------------------------------

TEST(HDR_IO_DecimalRoundtrip)
{
    HDRd original(1.5);
    original.setExp(10);

    std::string s = original.ToString<false>();
    std::istringstream iss(s);

    HDRd restored;
    restored.FromIStream<false>(iss);

    ASSERT_NEAR(restored.getMantissa(), original.getMantissa(), 1e-10);
    ASSERT_EQ(restored.getExp(), original.getExp());
}

TEST(HDR_IO_HexRoundtrip)
{
    HDRd original(1.23456789012345);
    original.setExp(42);

    std::string s = original.ToString<true>();
    std::istringstream iss(s);

    HDRd restored;
    restored.FromIStream<true>(iss);

    // Hex format preserves exact bit pattern
    ASSERT_EQ(restored.getMantissa(), original.getMantissa());
    ASSERT_EQ(restored.getExp(), original.getExp());
}

// ---------------------------------------------------------------------------
// Static Helpers: getMultiplier
// ---------------------------------------------------------------------------

TEST(HDR_GetMultiplier_ValidRange)
{
    // getMultiplier(0) = 2^0 = 1.0
    ASSERT_NEAR(HDRd::getMultiplier(0), 1.0, 1e-15);

    // getMultiplier(10) = 2^10 = 1024.0
    ASSERT_NEAR(HDRd::getMultiplier(10), 1024.0, 1e-10);

    // getMultiplier(-3) = 2^-3 = 0.125
    ASSERT_NEAR(HDRd::getMultiplier(-3), 0.125, 1e-15);
}

TEST(HDR_GetMultiplier_Underflow)
{
    // For double: exp ≤ -1023 → returns 0
    double result = HDRd::getMultiplier(-1023);
    ASSERT_NEAR(result, 0.0, 1e-300);

    result = HDRd::getMultiplier(-2000);
    ASSERT_NEAR(result, 0.0, 1e-300);
}

TEST(HDR_GetMultiplier_Overflow)
{
    // For double: exp ≥ 1024 → returns max
    double result = HDRd::getMultiplier(1024);
    ASSERT_EQ(result, std::numeric_limits<double>::max());

    result = HDRd::getMultiplier(5000);
    ASSERT_EQ(result, std::numeric_limits<double>::max());
}

// ---------------------------------------------------------------------------
// Template Variations
// ---------------------------------------------------------------------------

TEST(HDR_FloatType)
{
    // Basic operations with HDRFloat<float>
    HDRf a(6.0f);
    HDRf b(2.0f);

    HDRf sum = a + b;
    ASSERT_NEAR(static_cast<float>(sum), 8.0f, 1e-5f);

    HDRf prod = a * b;
    ASSERT_NEAR(static_cast<float>(prod), 12.0f, 1e-5f);

    HDRf sq = a.square();
    ASSERT_NEAR(static_cast<float>(sq), 36.0f, 1e-4f);

    // getMultiplier underflow/overflow for float
    ASSERT_NEAR(HDRf::getMultiplier(-127), 0.0f, 1e-30f);
    ASSERT_EQ(HDRf::getMultiplier(128), std::numeric_limits<float>::max());
}

TEST(HDR_Int64Exponent)
{
    // Imagina::HRReal = HDRFloat<double, HDROrder::Left, int64_t>
    // Verify construction and 64-bit exponent handling
    HRReal x(1.5);
    ASSERT_NEAR(x.getMantissa(), 1.5, 1e-15);
    ASSERT_EQ(x.getExp(), static_cast<int64_t>(0));

    // Set exponent beyond int32_t range
    x.setExp(static_cast<int64_t>(3'000'000'000LL));
    ASSERT_EQ(x.getExp(), static_cast<int64_t>(3'000'000'000LL));

    // Reduce still works with 64-bit exponents
    HRReal y(0, 16.0); // Denormalized
    y.Reduce();
    ASSERT_NEAR(y.getMantissa(), 1.0, 1e-15);
    ASSERT_EQ(y.getExp(), static_cast<int64_t>(4));

    // Negation works
    HRReal neg = x.negate();
    ASSERT_NEAR(neg.getMantissa(), -1.5, 1e-15);
    ASSERT_EQ(neg.getExp(), static_cast<int64_t>(3'000'000'000LL));
}

// ---------------------------------------------------------------------------
// Reciprocal
// ---------------------------------------------------------------------------

TEST(HDR_Reciprocal)
{
    HDRd a(4.0);
    HDRd r = a.reciprocal();
    ASSERT_NEAR(static_cast<double>(r), 0.25, 1e-10);

    // reciprocal of reciprocal ≈ original
    HDRd rr = r.reciprocal();
    ASSERT_NEAR(static_cast<double>(rr), 4.0, 1e-10);

    // Non-power-of-2 value
    HDRd b(5.0);
    HDRd rb = b.reciprocal();
    ASSERT_NEAR(static_cast<double>(rb), 0.2, 1e-10);
}

// ---------------------------------------------------------------------------
// Arithmetic producing zero
// ---------------------------------------------------------------------------

TEST(HDR_ArithmeticZeroResult)
{
    HDRd a(5.0);
    HDRd b(5.0);
    HDRd zero = a - b;

    ASSERT_NEAR(zero.getMantissa(), 0.0, 1e-15);
    ASSERT_EQ(zero.getExp(), HDRd::MIN_BIG_EXPONENT());
}
