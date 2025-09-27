#include "MpirSerialization.h"

namespace MpirSerialization {
	size_t mpz_inp_raw_stream(mpz_ptr x, std::istream &fp) {
		unsigned char  csize_bytes[4];
		mpir_out_struct out;

		/* 4 bytes for size */
		fp.read((char *)csize_bytes, sizeof(csize_bytes));

		mpz_inp_raw_p(x, csize_bytes, out);

		if (out->writtenSize != 0) {
			fp.read(out->written, out->writtenSize);

			mpz_inp_raw_m(x, out);
		}
		return out->writtenSize + 4;
	}
	size_t mpz_out_raw_stream(std::ostream &fp, mpz_srcptr x) {
		mpir_out_struct out;

		mpz_out_raw_m(out, x);

		fp.write(out->written, out->writtenSize);

		void (*gmp_free_func)(void *, size_t);

		mp_get_memory_functions(nullptr, nullptr, &gmp_free_func);
		(*gmp_free_func) (out->allocated, out->allocatedSize);
		return out->writtenSize;
	}
	size_t mpf_out_raw_stream(std::ostream &f, mpf_srcptr X) {
		long int expt; mpz_t Z; int nz;
		expt = X->_mp_exp;
		f.write((char *)&expt, sizeof(long int));
		nz = X->_mp_size;
		Z->_mp_alloc = std::abs(nz);
		Z->_mp_size = nz;
		Z->_mp_d = X->_mp_d;
		return mpz_out_raw_stream(f, Z) + sizeof(int);
	}
	void mpf_inp_raw_stream(std::istream &f, mpf_ptr X) {
		long int expt; mpz_t Z;
		mpz_init(Z);
		f.read((char *)&expt, sizeof(long int));
		mpz_inp_raw_stream(Z, f);
		mpf_set_z(X, Z);
		X->_mp_exp = expt;
		mpz_clear(Z);
	}
}