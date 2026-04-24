#include "MpirSerialization.h"
#include "TestFramework.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <random>
#include <sstream>
#include <string>
#include <vector>

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

// ---------------------------------------------------------------------------
// Byte-level wire format conformance
// ---------------------------------------------------------------------------
// These verify the exact byte layout of the raw mpz stream against hard-coded
// expected bytes. Matches MPIR mpz_out_raw_m: 4-byte big-endian signed int32
// header (sign * byte_count) followed by msb-first magnitude bytes.

namespace {

static void
CheckMpzBytes(mpz_srcptr value, const unsigned char *expected, size_t expected_len)
{
    std::stringstream ss;
    size_t written = MpirSerialization::mpz_out_raw_stream(ss, value);
    ASSERT_EQ(written, expected_len);

    std::string s = ss.str();
    ASSERT_EQ(s.size(), expected_len);
    for (size_t i = 0; i < expected_len; ++i) {
        unsigned char got = static_cast<unsigned char>(s[i]);
        ASSERT_EQ(static_cast<int>(got), static_cast<int>(expected[i]));
    }

    mpz_t restored;
    mpz_init(restored);
    MpirSerialization::mpz_inp_raw_stream(restored, ss);
    ASSERT_EQ(mpz_cmp(value, restored), 0);
    mpz_clear(restored);
}

} // namespace

TEST(MpirSer_WireFormat_Zero)
{
    mpz_t v;
    mpz_init_set_si(v, 0);
    const unsigned char expected[] = {0x00, 0x00, 0x00, 0x00};
    CheckMpzBytes(v, expected, sizeof(expected));
    mpz_clear(v);
}

TEST(MpirSer_WireFormat_One)
{
    mpz_t v;
    mpz_init_set_si(v, 1);
    const unsigned char expected[] = {0x00, 0x00, 0x00, 0x01, 0x01};
    CheckMpzBytes(v, expected, sizeof(expected));
    mpz_clear(v);
}

TEST(MpirSer_WireFormat_NegOne)
{
    mpz_t v;
    mpz_init_set_si(v, -1);
    const unsigned char expected[] = {0xFF, 0xFF, 0xFF, 0xFF, 0x01};
    CheckMpzBytes(v, expected, sizeof(expected));
    mpz_clear(v);
}

TEST(MpirSer_WireFormat_256)
{
    mpz_t v;
    mpz_init_set_si(v, 256);
    const unsigned char expected[] = {0x00, 0x00, 0x00, 0x02, 0x01, 0x00};
    CheckMpzBytes(v, expected, sizeof(expected));
    mpz_clear(v);
}

TEST(MpirSer_WireFormat_Neg256)
{
    mpz_t v;
    mpz_init_set_si(v, -256);
    const unsigned char expected[] = {0xFF, 0xFF, 0xFF, 0xFE, 0x01, 0x00};
    CheckMpzBytes(v, expected, sizeof(expected));
    mpz_clear(v);
}

TEST(MpirSer_WireFormat_65535)
{
    mpz_t v;
    mpz_init_set_ui(v, 65535u);
    const unsigned char expected[] = {0x00, 0x00, 0x00, 0x02, 0xFF, 0xFF};
    CheckMpzBytes(v, expected, sizeof(expected));
    mpz_clear(v);
}

TEST(MpirSer_WireFormat_Large64)
{
    mpz_t v;
    mpz_init(v);
    mpz_set_str(v, "1122334455667788", 16);
    const unsigned char expected[] = {
        0x00, 0x00, 0x00, 0x08, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88};
    CheckMpzBytes(v, expected, sizeof(expected));
    mpz_clear(v);
}

// ---------------------------------------------------------------------------
// Randomized round-trip
// ---------------------------------------------------------------------------
// Generates 100 pseudo-random mpz_t values of varied widths (including zero,
// negative, and multi-limb), serializes them all into a single stream, then
// reads them back and asserts equality. Deterministic seed so failures are
// reproducible.

TEST(MpirSer_MpzRoundtrip_100Random)
{
    std::mt19937_64 rng(0xBADC0FFEE123ULL);

    std::stringstream ss;
    std::vector<std::string> decimals;
    decimals.reserve(100);

    for (int i = 0; i < 100; ++i) {
        mpz_t z;
        mpz_init(z);

        int limbs = 1 + static_cast<int>(rng() % 32u);
        for (int k = 0; k < limbs; ++k) {
            mpz_mul_2exp(z, z, 64);
            mpz_add_ui(z, z, static_cast<unsigned long>(rng()));
        }
        if (rng() & 1ULL) {
            mpz_neg(z, z);
        }
        if ((rng() & 0xFULL) == 0) {
            mpz_set_ui(z, 0);
        }

        char *s = mpz_get_str(nullptr, 10, z);
        decimals.emplace_back(s);

        void (*freefn)(void *, size_t) = nullptr;
        mp_get_memory_functions(nullptr, nullptr, &freefn);
        freefn(s, std::strlen(s) + 1);

        MpirSerialization::mpz_out_raw_stream(ss, z);
        mpz_clear(z);
    }

    for (const std::string &dec : decimals) {
        mpz_t expected, restored;
        mpz_init_set_str(expected, dec.c_str(), 10);
        mpz_init(restored);

        MpirSerialization::mpz_inp_raw_stream(restored, ss);
        ASSERT_EQ(mpz_cmp(expected, restored), 0);

        mpz_clear(expected);
        mpz_clear(restored);
    }
}

// ---------------------------------------------------------------------------
// Cross-format (golden binary) test
// ---------------------------------------------------------------------------
// Verifies the raw stream format is identical across MPIR (Windows) and the
// GMP-compatible Linux reimplementation. The golden bytes are inlined below
// so the test has no filesystem dependency. Both read (deserialize and check
// decimal) and write (serialize and byte-compare) directions are exercised,
// catching drift in either direction.

namespace {

static constexpr std::array<std::uint8_t, 113> kGoldenMpirBytes = {
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0xFF, 0xFF, 0xFF, 0xFF, 0x01, 0x00, 0x00, 0x00,
    0x01, 0xFF, 0xFF, 0xFF, 0xFF, 0xFE, 0x01, 0x00, 0x00, 0x00, 0x00, 0x02, 0xFF, 0xFF, 0x00, 0x00, 0x00,
    0x08, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xF0, 0x80, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x15, 0x08,
    0x72, 0x7F, 0x63, 0x69, 0xAA, 0xF8, 0x3C, 0xA1, 0x50, 0x26, 0x74, 0x7A, 0xF8, 0xC7, 0xF1, 0x96, 0xCE,
    0x3F, 0x0A, 0xD2, 0xFF, 0xFF, 0xFF, 0xEB, 0x44, 0x6C, 0x3B, 0x15, 0xF9, 0x92, 0x66, 0x87, 0xD2, 0xC4,
    0x05, 0x34, 0xFD, 0xB5, 0x63, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
};

static constexpr const char *kGoldenMpirExpectedDecimal[10] = {
    "0",
    "1",
    "-1",
    "255",
    "-256",
    "65535",
    "18446744073709551615",
    "-170141183460469231731687303715884105729",
    "12345678901234567890123456789012345678901234567890",
    "-99999999999999999999999999999999999999999999999999",
};

} // namespace

TEST(MpirSer_GoldenBinary_CrossFormat)
{
    std::string bytes(reinterpret_cast<const char *>(kGoldenMpirBytes.data()), kGoldenMpirBytes.size());
    std::istringstream in(bytes, std::ios::binary);

    for (const char *expectedDecimal : kGoldenMpirExpectedDecimal) {
        mpz_t restored;
        mpz_init(restored);

        MpirSerialization::mpz_inp_raw_stream(restored, in);

        char *s = mpz_get_str(nullptr, 10, restored);
        std::string actualDecimal(s);
        void (*freefn)(void *, size_t) = nullptr;
        mp_get_memory_functions(nullptr, nullptr, &freefn);
        freefn(s, std::strlen(actualDecimal.c_str()) + 1);

        ASSERT_EQ(actualDecimal, std::string(expectedDecimal));

        mpz_clear(restored);
    }

    std::ostringstream out(std::ios::binary);
    for (const char *decimal : kGoldenMpirExpectedDecimal) {
        mpz_t z;
        mpz_init_set_str(z, decimal, 10);
        MpirSerialization::mpz_out_raw_stream(out, z);
        mpz_clear(z);
    }

    std::string written = out.str();
    ASSERT_EQ(written.size(), kGoldenMpirBytes.size());
    for (size_t i = 0; i < kGoldenMpirBytes.size(); ++i) {
        unsigned char got = static_cast<unsigned char>(written[i]);
        ASSERT_EQ(static_cast<int>(got), static_cast<int>(kGoldenMpirBytes[i]));
    }
}
