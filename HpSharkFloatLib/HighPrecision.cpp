#include "HighPrecision.h"

#include <string>
#include <sstream>

// typedef struct
// {
//   int _mp_prec;       // Max precision, in number of `mp_limb_t's.
//                       // Set by mpf_init and modified by
//                       // mpf_set_prec.  The area pointed to by the
//                       // _mp_d field contains `prec' + 1 limbs.
//   int _mp_size;       // abs(_mp_size) is the number of limbs the
//                       // last field points to.  If _mp_size is
//                       // negative this is a negative number.
//   mp_exp_t _mp_exp;   //  Exponent, in the base of `mp_limb_t'.
//   mp_limb_t *_mp_d;   //  Pointer to the limbs.
// } __mpf_struct;
//
// typedef __mpf_struct mpf_t[1];
std::string
MpfToHex32String(const mpf_t mpf_val)
{
    std::string result;
    mp_exp_t exponent;
    mp_limb_t *limbs = mpf_val[0]._mp_d;
    auto prec = mpf_val[0]._mp_prec;
    auto numLimbs = prec + 1;

    // First put a plus or minus depending on sign of _mp_size
    if (mpf_val[0]._mp_size < 0) {
        result += "-";
    } else {
        result += "+";
    }

    // Convert each limb to hex and append to result
    for (int i = 0; i < numLimbs; ++i) {
        char buffer[32];

        // Break lim into two 32-bit values and output individually, low then high
        uint32_t lowOrder = limbs[i] & 0xFFFFFFFF;
        uint32_t highOrder = limbs[i] >> 32;

        snprintf(buffer, sizeof(buffer), "0x%08X 0x%08X ", lowOrder, highOrder);
        result += buffer;
    }

    // Finally append exponent
    exponent = mpf_val[0]._mp_exp;
    result += "2^64^(0n" + std::to_string(exponent) + ")";

    return result;
}

std::string
MpfToHex64StringInvertable(const mpf_t mpf_val)
{
    std::string result;
    mp_exp_t exponent;
    mp_limb_t *limbs = mpf_val[0]._mp_d;
    const auto prec = mpf_val[0]._mp_prec;
    const auto actualLimbsUsed = mpf_val[0]._mp_size;
    auto numLimbs = prec + 1;

    // First put a plus or minus depending on sign of _mp_size
    if (mpf_val[0]._mp_size < 0) {
        result += "-";
    } else {
        result += "+";
    }

    result += " limbs: " + std::to_string(numLimbs) + " ";
    result += " actualLimbsUsed: " + std::to_string(actualLimbsUsed) + " ";

    // Convert each limb to hex and append to result
    for (int i = 0; i < numLimbs; ++i) {
        char buffer[32];

        // Break lim into two 32-bit values and output individually, low then high
        uint64_t curLimb = limbs[i];

        snprintf(buffer, sizeof(buffer), "0x%016llX ", curLimb);
        result += buffer;
    }

    // Finally append exponent
    exponent = mpf_val[0]._mp_exp;
    result += " e " + std::to_string(exponent);

    return result;
}

void
Hex64StringToMpf_Exact(const std::string &s, mpf_t out)
{
    static_assert(sizeof(mp_limb_t) == 8, "This function expects 64-bit GMP limbs.");

    // --- helpers ---
    auto fail = [](const char *msg) -> void { throw std::runtime_error(msg); };

    auto trim_left = [](std::string &t) {
        size_t i = 0;
        while (i < t.size() && std::isspace(static_cast<unsigned char>(t[i])))
            ++i;
        t.erase(0, i);
    };

    auto expect = [&](bool cond, const char *msg) {
        if (!cond)
            fail(msg);
    };

    auto parse_hex64 = [&](std::string tok) -> mp_limb_t {
        // Tolerate minor trailing punctuation (e.g., if caller appended separators)
        while (!tok.empty()) {
            unsigned char c = static_cast<unsigned char>(tok.back());
            if (std::isxdigit(c) || c == 'x' || c == 'X')
                break;
            tok.pop_back();
        }
        std::istringstream hs(tok);
        mp_limb_t v = 0;
        hs >> std::hex >> v;
        expect(!hs.fail(), "Failed to parse limb hex");
        return v;
    };

    // --- parse wire format ---
    // Format:
    //   <+|-> limbs: <numLimbs> actualLimbsUsed: <actualLimbsUsed> 0xLLLL... (numLimbs tokens) e <exp>
    std::string w = s;
    trim_left(w);
    expect(!w.empty(), "Empty input");

    const char leading_sign = w[0];
    expect(leading_sign == '+' || leading_sign == '-', "Missing leading sign (+/-)");
    w.erase(0, 1);
    trim_left(w);

    expect(w.compare(0, 6, "limbs:") == 0, "Missing 'limbs:' token");
    w.erase(0, 6);
    trim_left(w);

    std::istringstream iss(w);

    std::size_t numLimbs = 0;
    expect(static_cast<bool>(iss >> numLimbs), "Failed to read limb count");
    expect(numLimbs > 0, "Invalid limb count");

    // Expect "actualLimbsUsed:"
    {
        std::string tok;
        expect(static_cast<bool>(iss >> tok), "Missing 'actualLimbsUsed:' token");
        expect(tok == "actualLimbsUsed:", "Expected 'actualLimbsUsed:'");
    }

    long actualLimbsUsed = 0; // signed; may be negative
    expect(static_cast<bool>(iss >> actualLimbsUsed), "Failed to read actualLimbsUsed");

    // Collect exactly numLimbs limb tokens (as your dumper wrote all allocated limbs incl. guard).
    std::vector<mp_limb_t> limbs;
    limbs.reserve(numLimbs);
    for (std::size_t i = 0; i < numLimbs; ++i) {
        std::string tok;
        expect(static_cast<bool>(iss >> tok), "Not enough limb tokens");
        expect(tok.size() >= 3 && tok[0] == '0' && (tok[1] == 'x' || tok[1] == 'X'),
               "Limb token not in 0x... form");
        limbs.push_back(parse_hex64(tok));
    }

    // Expect 'e' then exponent (in limb units)
    std::string eTok;
    expect(static_cast<bool>(iss >> eTok), "Missing exponent marker");
    expect(eTok == "e", "Expected 'e' before exponent");

    long expLimbs = 0;
    expect(static_cast<bool>(iss >> expLimbs), "Failed to parse exponent");

    // Optional consistency check: sign in header vs sign(actualLimbsUsed)
    if ((leading_sign == '+' && actualLimbsUsed < 0) || (leading_sign == '-' && actualLimbsUsed > 0)) {
        fail("Leading sign and actualLimbsUsed sign disagree");
    }

    // --- rehydrate mpf internals (DANGEROUS/UNSUPPORTED, but byte-identical) ---
    // Your dumper used: numLimbs = _mp_prec + 1
    const mp_bitcnt_t bits = static_cast<mp_bitcnt_t>(numLimbs) * 64;
    mpf_init2(out, bits);

    __mpf_struct *p = out;
    p->_mp_prec = static_cast<int>(numLimbs) - 1;    // precision in limbs
    p->_mp_exp = static_cast<mp_exp_t>(expLimbs);    // exponent in limb units
    p->_mp_size = static_cast<int>(actualLimbsUsed); // signed, exact as dumped

    // Copy all allocated limbs exactly as printed (low -> high), including guard limb.
    for (std::size_t i = 0; i < numLimbs; ++i) {
        p->_mp_d[i] = limbs[i];
    }
}

// Canonicalize an mpf_t:
// - Remove MSW zeros from the used window d[0..|size|-1]
// - Adjust exponent so numeric value is preserved
// - Keep significant limbs packed at d[0..n-1]
void
MpfNormalize(mpf_t x)
{
    __mpf_struct *p = x;
    mp_limb_t *d = p->_mp_d;

    // Current used limb count (magnitude)
    mp_size_t s = (p->_mp_size >= 0) ? p->_mp_size : -p->_mp_size;
    mp_size_t n = s;

    // Trim zeros at the *high* end of the used window
    while (n > 0 && d[n - 1] == 0)
        --n;

    if (n == 0) {
        // Canonical zero
        p->_mp_size = 0;
        p->_mp_exp = 0;
        return;
    }

    // If we trimmed t = (s - n) high zero limbs, reduce exponent by t
    // to preserve value: value = (sum d[i] B^i) * B^(exp - used)
    // After trimming used->n, we need exp' - n == exp - s  =>  exp' = exp - (s - n)
    mp_size_t trimmed = s - n;
    if (trimmed)
        p->_mp_exp -= (mp_exp_t)trimmed;

    // Reapply sign with the new used length
    p->_mp_size = (p->_mp_size < 0) ? -n : n;

    // Optional hygiene: zero out slack (keeps guard/tail clean, not required)
    // const mp_size_t alloc = p->_mp_prec + 1; // total limbs incl. guard
    // for (mp_size_t i = n; i < alloc; ++i) d[i] = 0;
}
