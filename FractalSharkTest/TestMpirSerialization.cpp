#include "TestFramework.h"
#include "MpirSerialization.h"

#include <sstream>

// ---------------------------------------------------------------------------
// mpz round-trips
// ---------------------------------------------------------------------------

TEST(MpirSer_MpzRoundtrip_Simple)
{
    mpz_t original, restored;
    mpz_init_set_ui(original, 42);
    mpz_init(restored);

    std::stringstream ss;
    MpirSerialization::mpz_out_raw_stream(ss, original);
    MpirSerialization::mpz_inp_raw_stream(restored, ss);

    ASSERT_EQ(mpz_cmp(original, restored), 0);

    mpz_clear(original);
    mpz_clear(restored);
}

TEST(MpirSer_MpzRoundtrip_Negative)
{
    mpz_t original, restored;
    mpz_init_set_si(original, -123456789);
    mpz_init(restored);

    std::stringstream ss;
    MpirSerialization::mpz_out_raw_stream(ss, original);
    MpirSerialization::mpz_inp_raw_stream(restored, ss);

    ASSERT_EQ(mpz_cmp(original, restored), 0);

    mpz_clear(original);
    mpz_clear(restored);
}

TEST(MpirSer_MpzRoundtrip_Zero)
{
    mpz_t original, restored;
    mpz_init(original); // zero
    mpz_init(restored);

    std::stringstream ss;
    MpirSerialization::mpz_out_raw_stream(ss, original);
    MpirSerialization::mpz_inp_raw_stream(restored, ss);

    ASSERT_EQ(mpz_cmp_ui(restored, 0), 0);

    mpz_clear(original);
    mpz_clear(restored);
}

TEST(MpirSer_MpzRoundtrip_Large)
{
    mpz_t original, restored;
    mpz_init(original);
    mpz_init(restored);

    // 2^256 + 1
    mpz_ui_pow_ui(original, 2, 256);
    mpz_add_ui(original, original, 1);

    std::stringstream ss;
    MpirSerialization::mpz_out_raw_stream(ss, original);
    MpirSerialization::mpz_inp_raw_stream(restored, ss);

    ASSERT_EQ(mpz_cmp(original, restored), 0);

    mpz_clear(original);
    mpz_clear(restored);
}

// ---------------------------------------------------------------------------
// mpf round-trips
// ---------------------------------------------------------------------------

TEST(MpirSer_MpfRoundtrip_Simple)
{
    mpf_t original, restored;
    mpf_init2(original, 256);
    mpf_init2(restored, 256);

    mpf_set_d(original, 3.14159265358979);

    std::stringstream ss;
    MpirSerialization::mpf_out_raw_stream(ss, original);
    MpirSerialization::mpf_inp_raw_stream(ss, restored);

    ASSERT_EQ(mpf_cmp(original, restored), 0);

    mpf_clear(original);
    mpf_clear(restored);
}

TEST(MpirSer_MpfRoundtrip_Negative)
{
    mpf_t original, restored;
    mpf_init2(original, 256);
    mpf_init2(restored, 256);

    mpf_set_d(original, -2.718281828459045);

    std::stringstream ss;
    MpirSerialization::mpf_out_raw_stream(ss, original);
    MpirSerialization::mpf_inp_raw_stream(ss, restored);

    ASSERT_EQ(mpf_cmp(original, restored), 0);

    mpf_clear(original);
    mpf_clear(restored);
}

TEST(MpirSer_MpfRoundtrip_Zero)
{
    mpf_t original, restored;
    mpf_init2(original, 256);
    mpf_init2(restored, 256);

    mpf_set_d(original, 0.0);

    std::stringstream ss;
    MpirSerialization::mpf_out_raw_stream(ss, original);
    MpirSerialization::mpf_inp_raw_stream(ss, restored);

    ASSERT_EQ(mpf_cmp_d(restored, 0.0), 0);

    mpf_clear(original);
    mpf_clear(restored);
}

TEST(MpirSer_MpfRoundtrip_HighPrecision)
{
    // 512-bit precision with a value that needs many limbs: 1/7
    mpf_t original, restored;
    mpf_init2(original, 512);
    mpf_init2(restored, 512);

    mpf_set_d(original, 1.0);
    mpf_div_ui(original, original, 7);

    std::stringstream ss;
    MpirSerialization::mpf_out_raw_stream(ss, original);
    MpirSerialization::mpf_inp_raw_stream(ss, restored);

    ASSERT_EQ(mpf_cmp(original, restored), 0);

    mpf_clear(original);
    mpf_clear(restored);
}

TEST(MpirSer_MpfRoundtrip_LargeExponent)
{
    // Value with large exponent: 2^500
    mpf_t original, restored;
    mpf_init2(original, 256);
    mpf_init2(restored, 256);

    mpf_set_d(original, 1.0);
    mpf_mul_2exp(original, original, 500);

    std::stringstream ss;
    MpirSerialization::mpf_out_raw_stream(ss, original);
    MpirSerialization::mpf_inp_raw_stream(ss, restored);

    ASSERT_EQ(mpf_cmp(original, restored), 0);

    mpf_clear(original);
    mpf_clear(restored);
}

// ---------------------------------------------------------------------------
// Byte count consistency
// ---------------------------------------------------------------------------

TEST(MpirSer_ByteCountConsistency)
{
    // Verify that the number of bytes written matches what was read
    mpz_t z;
    mpz_init_set_ui(z, 999999);

    std::stringstream ss;
    size_t written = MpirSerialization::mpz_out_raw_stream(ss, z);
    ASSERT_TRUE(written > 0);

    // Stream should have exactly 'written' bytes
    ss.seekg(0, std::ios::end);
    size_t streamSize = static_cast<size_t>(ss.tellg());
    ASSERT_EQ(streamSize, written);

    mpz_clear(z);
}
