#pragma once

#include "MpirGmp.h"
#include <istream>
#include <ostream>
#include <stdarg.h>

// gmp-impl.h and the MPIR-specific raw I/O functions (mpir_out_struct,
// mpz_inp_raw_p, mpz_out_raw_m) are only available with MPIR on Windows.
// A GMP-compatible rewrite using mpz_export/mpz_import is needed for Linux.
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4100)
#include <gmp-impl.h>
#pragma warning(pop)
#endif

namespace MpirSerialization {
size_t mpz_inp_raw_stream(mpz_ptr x, std::istream &fp);
size_t mpz_out_raw_stream(std::ostream &fp, mpz_srcptr x);
size_t mpf_out_raw_stream(std::ostream &f, mpf_srcptr X);
void mpf_inp_raw_stream(std::istream &f, mpf_ptr X);
} // namespace MpirSerialization
