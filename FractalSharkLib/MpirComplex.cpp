#include "MpirComplex.h"

#include <climits>

// ============================================================
// mpf_complex helpers
// ============================================================

void
mpf_complex_init(mpf_complex &z, mp_bitcnt_t prec)
{
    mpf_init2(z.re, prec);
    mpf_init2(z.im, prec);
}

void
mpf_complex_clear(mpf_complex &z)
{
    mpf_clear(z.re);
    mpf_clear(z.im);
}

void
mpf_complex_set(mpf_complex &dst, const mpf_complex &src)
{
    mpf_set(dst.re, src.re);
    mpf_set(dst.im, src.im);
}

void
mpf_complex_set_ui(mpf_complex &z, unsigned long re, unsigned long im)
{
    mpf_set_ui(z.re, re);
    mpf_set_ui(z.im, im);
}

void
mpf_complex_sub(mpf_complex &out, const mpf_complex &a, const mpf_complex &b)
{
    mpf_sub(out.re, a.re, b.re);
    mpf_sub(out.im, a.im, b.im);
}

void
mpf_complex_add(mpf_complex &out, const mpf_complex &a, const mpf_complex &b)
{
    mpf_add(out.re, a.re, b.re);
    mpf_add(out.im, a.im, b.im);
}

void
mpf_complex_norm(mpf_t out, const mpf_complex &z, mpf_t t1, mpf_t t2)
{
    mpf_mul(t1, z.re, z.re);
    mpf_mul(t2, z.im, z.im);
    mpf_add(out, t1, t2);
}

int
approx_ilogb_mpf(const mpf_t x)
{
    if (mpf_cmp_ui(x, 0) == 0)
        return INT32_MIN;
    long exp2;
    (void)mpf_get_d_2exp(&exp2, x);
    return int(exp2 - 1);
}

int
approx_ilogb_mpf_abs2(const mpf_t re, const mpf_t im, mpf_t t1, mpf_t t2, mpf_t outAbs2)
{
    mpf_mul(t1, re, re);
    mpf_mul(t2, im, im);
    mpf_add(outAbs2, t1, t2);
    return approx_ilogb_mpf(outAbs2);
}

mp_bitcnt_t
ChooseDerivPrec_ImaginaStyle(mp_bitcnt_t coord_prec,
                             int scaleExp2,
                             int coordExp2_max_abs,
                             mp_bitcnt_t minPrec)
{
    long dp = (long)((-scaleExp2 + 32) / 4);
    if (dp < (long)minPrec)
        dp = (long)minPrec;
    long cap = (long)coord_prec + (long)coordExp2_max_abs + 32;
    if (dp > cap)
        dp = cap;
    if (dp > (long)coord_prec)
        dp = (long)coord_prec;
    return (mp_bitcnt_t)dp;
}
