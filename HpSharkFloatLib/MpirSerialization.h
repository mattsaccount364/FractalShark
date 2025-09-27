#pragma once

#include <mpir.h>
#include <istream>
#include <ostream>

// Disable warning C4100 using push/pop

#pragma warning(push)
#pragma warning(disable:4100)
#include <gmp-impl.h>
#pragma warning(pop)

namespace MpirSerialization {
	size_t mpz_inp_raw_stream(mpz_ptr x, std::istream &fp);
	size_t mpz_out_raw_stream(std::ostream &fp, mpz_srcptr x);
	size_t mpf_out_raw_stream(std::ostream &f, mpf_srcptr X);
	void mpf_inp_raw_stream(std::istream &f, mpf_ptr X);
}