#pragma once

// Compatibility header: use MPIR on Windows (MSVC), GMP on Linux/GCC.
// MPIR and GMP share the same public API (mpz_*, mpf_*, mp_set_memory_functions, etc.).

#ifdef _MSC_VER
#include <mpir.h>
#else
#include <gmp.h>
#endif
