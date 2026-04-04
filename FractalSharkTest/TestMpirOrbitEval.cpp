#include "TestFramework.h"
#include "MpirOrbitEval.h"

#include <cmath>
#include <climits>

// RAII helper for mpf_complex lifecycle
struct ScopedMpfComplex {
    mpf_complex z;
    ScopedMpfComplex(mp_bitcnt_t prec) { mpf_complex_init(z, prec); }
    ~ScopedMpfComplex() { mpf_complex_clear(z); }
};

struct ScopedMpf {
    mpf_t val;
    ScopedMpf(mp_bitcnt_t prec) { mpf_init2(val, prec); }
    ~ScopedMpf() { mpf_clear(val); }
};

// ---------------------------------------------------------------------------
// mpf_complex helpers
// ---------------------------------------------------------------------------

TEST(MpirEval_ComplexSetUI)
{
    ScopedMpfComplex z(256);
    mpf_complex_set_ui(z.z, 3, 4);
    ASSERT_NEAR(mpf_get_d(z.z.re), 3.0, 1e-15);
    ASSERT_NEAR(mpf_get_d(z.z.im), 4.0, 1e-15);
}

TEST(MpirEval_ComplexAdd)
{
    ScopedMpfComplex a(256), b(256), out(256);
    mpf_set_d(a.z.re, 1.0);  mpf_set_d(a.z.im, 2.0);
    mpf_set_d(b.z.re, 3.0);  mpf_set_d(b.z.im, 4.0);

    mpf_complex_add(out.z, a.z, b.z);
    ASSERT_NEAR(mpf_get_d(out.z.re), 4.0, 1e-15);
    ASSERT_NEAR(mpf_get_d(out.z.im), 6.0, 1e-15);
}

TEST(MpirEval_ComplexSub)
{
    ScopedMpfComplex a(256), b(256), out(256);
    mpf_set_d(a.z.re, 5.0);  mpf_set_d(a.z.im, 7.0);
    mpf_set_d(b.z.re, 2.0);  mpf_set_d(b.z.im, 3.0);

    mpf_complex_sub(out.z, a.z, b.z);
    ASSERT_NEAR(mpf_get_d(out.z.re), 3.0, 1e-15);
    ASSERT_NEAR(mpf_get_d(out.z.im), 4.0, 1e-15);
}

TEST(MpirEval_ComplexNorm)
{
    // |3+4i|² = 9+16 = 25
    ScopedMpfComplex z(256);
    ScopedMpf out(256), t1(256), t2(256);
    mpf_set_d(z.z.re, 3.0);  mpf_set_d(z.z.im, 4.0);

    mpf_complex_norm(out.val, z.z, t1.val, t2.val);
    ASSERT_NEAR(mpf_get_d(out.val), 25.0, 1e-14);
}

TEST(MpirEval_ComplexSet)
{
    ScopedMpfComplex src(256), dst(256);
    mpf_set_d(src.z.re, 1.5);  mpf_set_d(src.z.im, -2.5);

    mpf_complex_set(dst.z, src.z);
    ASSERT_NEAR(mpf_get_d(dst.z.re), 1.5, 1e-15);
    ASSERT_NEAR(mpf_get_d(dst.z.im), -2.5, 1e-15);
}

// ---------------------------------------------------------------------------
// approx_ilogb_mpf
// ---------------------------------------------------------------------------

TEST(MpirEval_IlogbMpf_PowerOfTwo)
{
    ScopedMpf val(256);

    // 1.0 = 0.5 × 2^1 → mpf_get_d_2exp returns exp=1 → approx_ilogb = 0
    mpf_set_d(val.val, 1.0);
    ASSERT_EQ(approx_ilogb_mpf(val.val), 0);

    // 8.0 = 0.5 × 2^4 → exp=4 → ilogb = 3
    mpf_set_d(val.val, 8.0);
    ASSERT_EQ(approx_ilogb_mpf(val.val), 3);

    // 0.25 = 0.5 × 2^(-1) → exp=-1 → ilogb = -2
    mpf_set_d(val.val, 0.25);
    ASSERT_EQ(approx_ilogb_mpf(val.val), -2);
}

TEST(MpirEval_IlogbMpf_Zero)
{
    ScopedMpf val(256);
    mpf_set_d(val.val, 0.0);
    ASSERT_EQ(approx_ilogb_mpf(val.val), INT32_MIN);
}

TEST(MpirEval_IlogbAbs2)
{
    // |3+4i|² = 25 → ilogb(25) = 4 (since 25 = 0.78125 × 2^5 → exp=5 → 5-1=4)
    ScopedMpf t1(256), t2(256), outAbs2(256);
    mpf_t re, im;
    mpf_init2(re, 256);  mpf_init2(im, 256);
    mpf_set_d(re, 3.0);  mpf_set_d(im, 4.0);

    int result = approx_ilogb_mpf_abs2(re, im, t1.val, t2.val, outAbs2.val);
    ASSERT_EQ(result, 4);
    ASSERT_NEAR(mpf_get_d(outAbs2.val), 25.0, 1e-14);

    mpf_clear(re);  mpf_clear(im);
}

// ---------------------------------------------------------------------------
// ChooseDerivPrec_ImaginaStyle
// ---------------------------------------------------------------------------

TEST(MpirEval_ChooseDerivPrec_MinPrec)
{
    // scaleExp2 = 0, coord_prec = 1000, coordExp2_max_abs = 0
    // dp = (-0 + 32)/4 = 8 → clamped to minPrec=256
    auto prec = ChooseDerivPrec_ImaginaStyle(1000, 0, 0, 256);
    ASSERT_EQ(prec, static_cast<mp_bitcnt_t>(256));
}

TEST(MpirEval_ChooseDerivPrec_DeepZoom)
{
    // Deep zoom: scaleExp2 = -10000
    // dp = (10000 + 32)/4 = 2508
    // cap = coord_prec(4096) + coordExp2_max_abs(100) + 32 = 4228
    // dp(2508) < cap(4228) and dp(2508) < coord_prec(4096) → result = 2508
    auto prec = ChooseDerivPrec_ImaginaStyle(4096, -10000, 100, 256);
    ASSERT_EQ(prec, static_cast<mp_bitcnt_t>(2508));
}

TEST(MpirEval_ChooseDerivPrec_CapByCoordPrec)
{
    // Very deep zoom but low coord_prec
    // dp = (100000 + 32)/4 = 25008
    // coord_prec = 512 → dp > coord_prec → clamped to 512
    auto prec = ChooseDerivPrec_ImaginaStyle(512, -100000, 0, 256);
    ASSERT_EQ(prec, static_cast<mp_bitcnt_t>(512));
}

TEST(MpirEval_ChooseDerivPrec_CapByCap)
{
    // dp exceeds cap
    // dp = (10000 + 32)/4 = 2508
    // cap = coord_prec(2000) + coordExp2_max_abs(0) + 32 = 2032
    // dp(2508) > cap(2032), so dp = 2032
    // dp(2032) > coord_prec(2000), so dp = 2000
    auto prec = ChooseDerivPrec_ImaginaStyle(2000, -10000, 0, 256);
    ASSERT_EQ(prec, static_cast<mp_bitcnt_t>(2000));
}
