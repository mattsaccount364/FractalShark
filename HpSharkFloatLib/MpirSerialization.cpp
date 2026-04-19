#include "MpirSerialization.h"

#ifdef _MSC_VER

namespace MpirSerialization {
size_t
mpz_inp_raw_stream(mpz_ptr x, std::istream &fp)
{
    unsigned char csize_bytes[4];
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
size_t
mpz_out_raw_stream(std::ostream &fp, mpz_srcptr x)
{
    mpir_out_struct out;

    mpz_out_raw_m(out, x);

    fp.write(out->written, out->writtenSize);

    void (*gmp_free_func)(void *, size_t);

    mp_get_memory_functions(nullptr, nullptr, &gmp_free_func);
    (*gmp_free_func)(out->allocated, out->allocatedSize);
    return out->writtenSize;
}
size_t
mpf_out_raw_stream(std::ostream &f, mpf_srcptr X)
{
    long int expt;
    mpz_t Z;
    int nz;
    expt = X->_mp_exp;
    f.write((char *)&expt, sizeof(long int));
    nz = X->_mp_size;
    Z->_mp_alloc = std::abs(nz);
    Z->_mp_size = nz;
    Z->_mp_d = X->_mp_d;
    return mpz_out_raw_stream(f, Z) + sizeof(int);
}
void
mpf_inp_raw_stream(std::istream &f, mpf_ptr X)
{
    long int expt;
    mpz_t Z;
    mpz_init(Z);
    f.read((char *)&expt, sizeof(long int));
    mpz_inp_raw_stream(Z, f);
    mpf_set_z(X, Z);
    X->_mp_exp = expt;
    mpz_clear(Z);
}
} // namespace MpirSerialization

#else

// GMP-compatible reimplementation of the MPIR raw stream format used on
// Windows. The wire format matches mpz_out_raw_m / mpz_inp_raw_p:
//   - 4-byte big-endian signed int32 header: sign(x) * byte_count
//   - byte_count magnitude bytes, most-significant byte first
// This is byte-for-byte compatible with mpz_export(order=1, size=1, endian=1,
// nails=0, x) plus a big-endian signed length prefix.

#include <cstdint>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <vector>

// Guard against future GMP layout drift. mpf_t is a 1-element array of
// __mpf_struct, so sizeof(mpf_t) == sizeof(__mpf_struct).
static_assert(sizeof(mpf_t) == sizeof(__mpf_struct), "mpf_t layout drift");
static_assert(offsetof(__mpf_struct, _mp_prec) == 0, "_mp_prec offset drift");
static_assert(offsetof(__mpf_struct, _mp_size) == offsetof(__mpf_struct, _mp_prec) + sizeof(int),
              "_mp_size offset drift");
static_assert(offsetof(__mpf_struct, _mp_exp) > offsetof(__mpf_struct, _mp_size),
              "_mp_exp offset drift");
static_assert(offsetof(__mpf_struct, _mp_d) > offsetof(__mpf_struct, _mp_exp),
              "_mp_d offset drift");

namespace MpirSerialization {

size_t
mpz_out_raw_stream(std::ostream &fp, mpz_srcptr x)
{
    size_t byte_count = 0;
    int sign = mpz_sgn(x);
    if (sign != 0) {
        byte_count = (mpz_sizeinbase(x, 2) + 7) / 8;
    }

    int32_t header = (sign < 0 ? -1 : 1) * static_cast<int32_t>(byte_count);
    if (sign == 0) {
        header = 0;
    }

    uint32_t uhdr = static_cast<uint32_t>(header);
    unsigned char hdr[4];
    hdr[0] = static_cast<unsigned char>((uhdr >> 24) & 0xFF);
    hdr[1] = static_cast<unsigned char>((uhdr >> 16) & 0xFF);
    hdr[2] = static_cast<unsigned char>((uhdr >> 8) & 0xFF);
    hdr[3] = static_cast<unsigned char>(uhdr & 0xFF);
    fp.write(reinterpret_cast<const char *>(hdr), 4);

    if (byte_count > 0) {
        std::vector<unsigned char> buf(byte_count);
        size_t countp = 0;
        mpz_export(buf.data(), &countp, 1, 1, 1, 0, x);
        // countp should equal byte_count; pad leading zeros if GMP wrote fewer
        // (shouldn't happen, but defensive).
        if (countp < byte_count) {
            std::memmove(buf.data() + (byte_count - countp), buf.data(), countp);
            std::memset(buf.data(), 0, byte_count - countp);
        }
        fp.write(reinterpret_cast<const char *>(buf.data()), byte_count);
    }
    return byte_count + 4;
}

size_t
mpz_inp_raw_stream(mpz_ptr x, std::istream &fp)
{
    unsigned char hdr[4];
    fp.read(reinterpret_cast<char *>(hdr), 4);
    int32_t raw = static_cast<int32_t>((static_cast<uint32_t>(hdr[0]) << 24) |
                                       (static_cast<uint32_t>(hdr[1]) << 16) |
                                       (static_cast<uint32_t>(hdr[2]) << 8) |
                                       static_cast<uint32_t>(hdr[3]));
    if (raw == 0) {
        mpz_set_ui(x, 0);
        return 4;
    }

    size_t byte_count = static_cast<size_t>(std::abs(raw));
    std::vector<unsigned char> buf(byte_count);
    fp.read(reinterpret_cast<char *>(buf.data()), byte_count);
    mpz_import(x, byte_count, 1, 1, 1, 0, buf.data());
    if (raw < 0) {
        mpz_neg(x, x);
    }
    return byte_count + 4;
}

size_t
mpf_out_raw_stream(std::ostream &f, mpf_srcptr X)
{
    long int expt = X->_mp_exp;
    f.write(reinterpret_cast<const char *>(&expt), sizeof(long int));

    // Non-owning mpz view over X's limb array, matching the Windows impl.
    mpz_t Z;
    int nz = X->_mp_size;
    Z->_mp_alloc = std::abs(nz);
    Z->_mp_size = nz;
    Z->_mp_d = X->_mp_d;
    // Preserve the Windows return-value quirk (sizeof(int), not sizeof(long int)).
    return mpz_out_raw_stream(f, Z) + sizeof(int);
}

void
mpf_inp_raw_stream(std::istream &f, mpf_ptr X)
{
    long int expt;
    mpz_t Z;
    mpz_init(Z);
    f.read(reinterpret_cast<char *>(&expt), sizeof(long int));
    mpz_inp_raw_stream(Z, f);
    mpf_set_z(X, Z);
    X->_mp_exp = expt;
    mpz_clear(Z);
}

} // namespace MpirSerialization

#endif
