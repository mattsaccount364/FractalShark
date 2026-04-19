#pragma once

// Compatibility header: use MPIR on Windows (MSVC), GMP on Linux/GCC.
// MPIR and GMP share the same public API (mpz_*, mpf_*, mp_set_memory_functions, etc.).

#ifdef _MSC_VER
#include <mpir.h>
#else
#include <gmp.h>

// MPIR provides mpf_get_2exp_d(double *d, mpf_t f) → long exp.
// GMP provides mpf_get_d_2exp(long *exp, mpf_t f) → double d.
// Provide the MPIR signature in terms of GMP:
inline long mpf_get_2exp_d(double *d, mpf_srcptr f)
{
    long exp;
    *d = mpf_get_d_2exp(&exp, f);
    return exp;
}
#endif
