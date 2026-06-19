#include "HighPrecision.h"
#include "TestFramework.h"

#include <sstream>
#include <string>

// ---------------------------------------------------------------------------
// Hex64 round-trip tests
// ---------------------------------------------------------------------------

TEST(HPStr_Hex64Roundtrip_Simple)
{
    mpf_t original, restored;
    mpf_init2(original, 256);
    mpf_init2(restored, 256);

    mpf_set_d(original, 3.14159265358979);
    MpfNormalize(original);

    std::string hex = MpfToHex64StringInvertable(original);
    Hex64StringToMpf_Exact(hex, restored);

    ASSERT_EQ(mpf_cmp(original, restored), 0);

    // String identity: re-encoding produces identical string
    std::string hex2 = MpfToHex64StringInvertable(restored);
    ASSERT_TRUE(hex == hex2);

    mpf_clear(original);
    mpf_clear(restored);
}

TEST(HPStr_Hex64Roundtrip_Zero)
{
    mpf_t original, restored;
    mpf_init2(original, 256);

    mpf_set_d(original, 0.0);

    std::string hex = MpfToHex64StringInvertable(original);
    Hex64StringToMpf_Exact(hex, restored);

    ASSERT_EQ(mpf_cmp_d(restored, 0.0), 0);

    mpf_clear(original);
    mpf_clear(restored);
}

TEST(HPStr_Hex64Roundtrip_Negative)
{
    mpf_t original, restored;
    mpf_init2(original, 256);

    mpf_set_d(original, -42.5);
    MpfNormalize(original);

    std::string hex = MpfToHex64StringInvertable(original);
    // Verify negative sign in output
    ASSERT_TRUE(hex[0] == '-');

    Hex64StringToMpf_Exact(hex, restored);
    ASSERT_EQ(mpf_cmp(original, restored), 0);

    mpf_clear(original);
    mpf_clear(restored);
}

TEST(HPStr_Hex64Roundtrip_LargeExponent)
{
    mpf_t original, restored;
    mpf_init2(original, 256);

    // Set to a very large value: 2^500
    mpf_set_d(original, 1.0);
    mpf_mul_2exp(original, original, 500);

    std::string hex = MpfToHex64StringInvertable(original);
    Hex64StringToMpf_Exact(hex, restored);

    ASSERT_EQ(mpf_cmp(original, restored), 0);

    mpf_clear(original);
    mpf_clear(restored);
}

TEST(HPStr_Hex64Roundtrip_HighPrecision)
{
    // Test with higher precision (512 bits)
    mpf_t original, restored;
    mpf_init2(original, 512);

    mpf_set_d(original, 1.0);
    mpf_div_ui(original, original, 7); // 1/7 = 0.142857...
    MpfNormalize(original);

    std::string hex = MpfToHex64StringInvertable(original);
    Hex64StringToMpf_Exact(hex, restored);

    ASSERT_EQ(mpf_cmp(original, restored), 0);

    mpf_clear(original);
    mpf_clear(restored);
}

// ---------------------------------------------------------------------------
// Hex64 error handling
// ---------------------------------------------------------------------------

TEST(HPStr_Hex64Error_Empty)
{
    mpf_t out;
    mpf_init(out);
    ASSERT_THROWS(Hex64StringToMpf_Exact("", out), FractalSharkSeriousException);
    mpf_clear(out);
}

TEST(HPStr_Hex64Error_BadSign)
{
    mpf_t out;
    mpf_init(out);
    ASSERT_THROWS(Hex64StringToMpf_Exact("x limbs: 1 actualLimbsUsed: 1 0x0000000000000001 e 0", out),
                  FractalSharkSeriousException);
    mpf_clear(out);
}

TEST(HPStr_Hex64Error_MissingLimbs)
{
    mpf_t out;
    mpf_init(out);
    ASSERT_THROWS(Hex64StringToMpf_Exact("+ garbage data here", out), FractalSharkSeriousException);
    mpf_clear(out);
}

// ---------------------------------------------------------------------------
// MpfNormalize
// ---------------------------------------------------------------------------

TEST(HPStr_Normalize_AlreadyNormalized)
{
    mpf_t val;
    mpf_init2(val, 256);
    mpf_set_d(val, 3.14159);

    // Get value before normalize
    double before = mpf_get_d(val);
    MpfNormalize(val);
    double after = mpf_get_d(val);

    ASSERT_NEAR(before, after, 1e-15);
    mpf_clear(val);
}

TEST(HPStr_Normalize_PreservesValue)
{
    // Create a value via string parsing (which may leave un-normalized state),
    // then normalize and verify the numeric value is unchanged
    mpf_t val;
    mpf_init2(val, 256);
    mpf_set_str(val, "123456789.987654321", 10);

    double before = mpf_get_d(val);
    MpfNormalize(val);
    double after = mpf_get_d(val);

    ASSERT_NEAR(before, after, 1e-5);
    mpf_clear(val);
}

// ---------------------------------------------------------------------------
// MpfToHex32String (smoke test — not invertible)
// ---------------------------------------------------------------------------

TEST(HPStr_Hex32_SmokeTest)
{
    mpf_t val;
    mpf_init2(val, 128);
    mpf_set_d(val, 42.0);

    std::string hex32 = MpfToHex32String(val);
    ASSERT_FALSE(hex32.empty());
    // Should contain sign prefix
    ASSERT_TRUE(hex32[0] == '+' || hex32[0] == '-');
    // Should contain exponent marker
    ASSERT_TRUE(hex32.find("2^64^") != std::string::npos);

    mpf_clear(val);
}

// ---------------------------------------------------------------------------
// HighPrecision str() round-trip
// ---------------------------------------------------------------------------

TEST(HPStr_StrRoundtrip)
{
    HighPrecision original(3.14159);
    std::string s = original.str();

    HighPrecision restored(s);

    // Compare numerically (str() may lose some precision at extreme depths,
    // but for double-derived values the round-trip should be tight)
    double origD = static_cast<double>(original);
    double restD = static_cast<double>(restored);
    ASSERT_NEAR(origD, restD, 1e-10);
}
