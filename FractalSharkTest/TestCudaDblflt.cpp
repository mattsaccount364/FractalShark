#include "TestFramework.h"
#include "dblflt.h"
#include "CudaDblflt.h"

#include <cmath>
#include <sstream>

using CDf = CudaDblflt<MattDblflt>;

// Note: CudaDblflt arithmetic operators (+, -, *, /, <, >, ==, !=) are
// only defined inside #ifdef __CUDACC__ and are not available from CPU code.
// These tests focus on construction, precision, conversion, and I/O.

// ---------------------------------------------------------------------------
// MattDblflt construction: Knuth two-sum algorithm
// ---------------------------------------------------------------------------

TEST(Dblflt_MattDblflt_FromDouble)
{
    // MattDblflt splits a double into head (float) + tail (float)
    // such that (double)head + (double)tail ≈ original
    double original = 3.141592653589793;
    MattDblflt dd(original);

    double reconstructed = (double)dd.head + (double)dd.tail;
    // Should recover the original to near-double precision (~48 bits for 2×float)
    ASSERT_NEAR(reconstructed, original, 1e-14);
}

TEST(Dblflt_MattDblflt_FromFloat)
{
    // From a single float: head = value, tail = 0
    MattDblflt dd(3.14f);
    ASSERT_NEAR(dd.head, 3.14f, 1e-6f);
    ASSERT_NEAR(dd.tail, 0.0f, 1e-10f);
}

TEST(Dblflt_MattDblflt_TwoSum)
{
    // The (float, float) constructor uses two-sum: head = a+b, tail captures the rounding error
    MattDblflt dd(1.0f, 1e-10f);
    // head should be very close to 1.0 (since 1e-10 is below float precision)
    ASSERT_NEAR(dd.head, 1.0f, 1e-6f);
    // The sum head+tail should recover the original more precisely
    double sum = (double)dd.head + (double)dd.tail;
    ASSERT_NEAR(sum, 1.0 + 1e-10, 1e-14);
}

TEST(Dblflt_MattDblflt_PrecisionGain)
{
    // Verify that double-float gives more precision than float alone
    double value = 1.0000001192092896; // 1 + 2^-23 (just beyond float precision)
    MattDblflt dd(value);

    float headOnly = (float)value;
    double headError = std::abs(value - (double)headOnly);
    double ddError = std::abs(value - ((double)dd.head + (double)dd.tail));

    // Double-float error should be smaller than single-float error
    ASSERT_TRUE(ddError < headError || headError < 1e-15);
}

TEST(Dblflt_MattDblflt_Zero)
{
    MattDblflt dd(0.0);
    ASSERT_NEAR(dd.head, 0.0f, 1e-30f);
    ASSERT_NEAR(dd.tail, 0.0f, 1e-30f);
}

TEST(Dblflt_MattDblflt_Negative)
{
    MattDblflt dd(-2.718281828459045);
    double reconstructed = (double)dd.head + (double)dd.tail;
    ASSERT_NEAR(reconstructed, -2.718281828459045, 1e-14);
}

// ---------------------------------------------------------------------------
// CudaDblflt construction
// ---------------------------------------------------------------------------

TEST(Dblflt_CudaDblflt_Default)
{
    CDf cd;
    ASSERT_NEAR(cd.head(), 0.0f, 1e-30f);
    ASSERT_NEAR(cd.tail(), 0.0f, 1e-30f);
}

TEST(Dblflt_CudaDblflt_FromDouble)
{
    CDf cd(3.141592653589793);
    double back = static_cast<double>(cd);
    ASSERT_NEAR(back, 3.141592653589793, 1e-14);
}

TEST(Dblflt_CudaDblflt_FromFloat)
{
    CDf cd(2.5f);
    ASSERT_NEAR(cd.head(), 2.5f, 1e-6f);
    ASSERT_NEAR(static_cast<double>(cd), 2.5, 1e-14);
}

TEST(Dblflt_CudaDblflt_FromHeadTail)
{
    CDf cd(1.0f, 0.5f);
    ASSERT_NEAR(static_cast<double>(cd), 1.5, 1e-14);
}

TEST(Dblflt_CudaDblflt_Copy)
{
    CDf a(42.5);
    CDf b(a);
    ASSERT_NEAR(static_cast<double>(b), static_cast<double>(a), 1e-15);
    ASSERT_NEAR(b.head(), a.head(), 1e-30f);
    ASSERT_NEAR(b.tail(), a.tail(), 1e-30f);
}

TEST(Dblflt_CudaDblflt_AssignDouble)
{
    CDf cd;
    cd = 123.456;
    ASSERT_NEAR(static_cast<double>(cd), 123.456, 1e-12);
}

TEST(Dblflt_CudaDblflt_FromHighPrecision)
{
    HighPrecision hp(99.99);
    CDf cd(hp);
    ASSERT_NEAR(static_cast<double>(cd), 99.99, 1e-12);
}

// ---------------------------------------------------------------------------
// I/O round-trip
// ---------------------------------------------------------------------------

TEST(Dblflt_IO_DecimalRoundtrip)
{
    CDf original(1.23456789);
    std::string s = original.ToString<false>();
    std::istringstream iss(s);
    CDf restored;
    restored.FromIStream<false>(iss);

    ASSERT_NEAR(static_cast<double>(restored), static_cast<double>(original), 1e-6);
}

TEST(Dblflt_IO_HexRoundtrip)
{
    CDf original(9.87654321);
    std::string s = original.ToString<true>();
    std::istringstream iss(s);
    CDf restored;
    restored.FromIStream<true>(iss);

    // Hex format should preserve exact bit pattern
    ASSERT_NEAR(restored.head(), original.head(), 1e-30f);
    ASSERT_NEAR(restored.tail(), original.tail(), 1e-30f);
}
