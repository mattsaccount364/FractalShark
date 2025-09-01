#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <memory>
#include <span>
#include <unordered_map>

#include <array>
#include <type_traits>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

#include "DebugChecksumHost.h"
#include "HpSharkFloat.cuh" // your header (provides HpSharkFloat<>, DebugStateHost<>)
#include "ReferenceNTT2.h"
#include "TestVerbose.h"

namespace SharkNTT {

static inline uint32_t
ReverseBits32(uint32_t x, int bits)
{
    // Perform full 32-bit bit reversal manually since MSVC doesn't have __builtin_bitreverse32
    x = (x >> 16) | (x << 16);
    x = ((x & 0x00ff00ffu) << 8) | ((x & 0xff00ff00u) >> 8);
    x = ((x & 0x0f0f0f0fu) << 4) | ((x & 0xf0f0f0f0u) >> 4);
    x = ((x & 0x33333333u) << 2) | ((x & 0xccccccccu) >> 2);
    x = ((x & 0x55555555u) << 1) | ((x & 0xaaaaaaaau) >> 1);
    return x >> (32 - bits);
}

// Normalize directly from Final128 (lo64,hi64 per 32-bit digit position).
// Does not allocate; does two passes over Final128 to (1) find 'significant',
// then (2) write the window and set the Karatsuba-style exponent.
// NOTE: Does not set sign; do that at the call site.
template <class P>
static void
Normalize(
    HpSharkFloat<P>& Out,
    const HpSharkFloat<P>& A,
    const HpSharkFloat<P>& B,
    const uint64_t* final128,     // len = 2*Ddigits
    size_t Ddigits,               // number of 32-bit positions represented in Final128
    int32_t additionalFactorOfTwo // 0 for XX/YY; 1 for XY if you did NOT shift digits
)
{
    constexpr int N = P::GlobalNumUint32;

    auto pass_once = [&](bool write_window, size_t start, size_t needN) -> std::pair<int, int> {
        uint64_t carry_lo = 0, carry_hi = 0;
        size_t idx = 0;           // index in base-2^32 digits being produced
        int highest_nonzero = -1; // last non-zero digit index seen
        int out_written = 0;

        // Helper to emit one 32-bit digit (optionally writing into Out.Digits)
        auto emit_digit = [&](uint32_t dig) {
            if (dig != 0)
                highest_nonzero = (int)idx;
            if (write_window && idx >= start && out_written < (int)needN) {
                Out.Digits[out_written++] = dig;
            }
            ++idx;
        };

        // Main Ddigits loop
        for (size_t d = 0; d < Ddigits; ++d) {
            uint64_t lo = final128[2 * d + 0];
            uint64_t hi = final128[2 * d + 1];

            uint64_t s_lo = lo + carry_lo;
            uint64_t c0 = (s_lo < lo) ? 1ull : 0ull;
            uint64_t s_hi = hi + carry_hi + c0;

            emit_digit(static_cast<uint32_t>(s_lo & 0xffffffffull));

            carry_lo = (s_lo >> 32) | (s_hi << 32);
            carry_hi = (s_hi >> 32);
        }

        // Flush remaining carry into more digits
        while (carry_lo || carry_hi) {
            emit_digit(static_cast<uint32_t>(carry_lo & 0xffffffffull));
            uint64_t nlo = (carry_lo >> 32) | (carry_hi << 32);
            uint64_t nhi = (carry_hi >> 32);
            carry_lo = nlo;
            carry_hi = nhi;
        }

        // If we were asked to write a window and didn't fill all N yet, pad zeros
        if (write_window) {
            while (out_written < (int)needN) {
                Out.Digits[out_written++] = 0u;
            }
        }

        return {highest_nonzero, out_written};
    };

    // Pass 1: find 'significant' (last non-zero + 1)
    auto [highest_nonzero, _] = pass_once(/*write_window=*/false, /*start=*/0, /*needN=*/0);
    if (highest_nonzero < 0) {
        // Zero result
        std::memset(Out.Digits, 0, N * sizeof(uint32_t));
        Out.Exponent = A.Exponent + B.Exponent; // Karatsuba zero convention
        return;
    }
    const int significant = highest_nonzero + 1;
    int shift_digits = significant - N;
    if (shift_digits < 0)
        shift_digits = 0;

    // Karatsuba-style exponent
    Out.Exponent = A.Exponent + B.Exponent + 32 * shift_digits + additionalFactorOfTwo;

    // Pass 2: write exactly N digits starting at shift_digits
    (void)pass_once(/*write_window=*/true, /*start=*/(size_t)shift_digits, /*needN=*/(size_t)N);
}


struct u128_lohi {
    uint64_t lo, hi;
};

// Drop-in replacement: MultiplyHelperFFT2 — NTT over the 64-bit Goldilocks prime (MSVC-safe)
// -----------------------------------------------------------------------------
// This implements negacyclic convolution using a radix-2 NTT modulo
//   p = 0xFFFFFFFF00000001 = 2^64 - 2^32 + 1 ("Goldilocks").
// It mirrors the structure and PUBLIC SIGNATURE of the existing MultiplyHelperFFT:
//   template<class SharkFloatParams>
//   void MultiplyHelperFFT2(
//       const HpSharkFloat<SharkFloatParams> *A,
//       const HpSharkFloat<SharkFloatParams> *B,
//       HpSharkFloat<SharkFloatParams> *OutXX,
//       HpSharkFloat<SharkFloatParams> *OutXY,
//       HpSharkFloat<SharkFloatParams> *OutYY,
//       DebugHostCombo<SharkFloatParams> &debugCombo)
//
// Runtime self-checks integrated:
//   • Root orders for the chosen N (ψ^(2N)=1, ω^N=1)
//   • NTT round-trip on the twisted X
//   • Full coefficient-by-coefficient equality vs direct linear convolution (O(L^2))
//

// Goldilocks prime p = 2^64 - 2^32 + 1
static constexpr uint64_t P = 0xFFFF'FFFF'0000'0001ull;
static constexpr uint64_t NINV = 0xFFFF'FFFE'FFFF'FFFFull; // -p^{-1} mod 2^64
static constexpr uint64_t R2 = 0xFFFF'FFFE'0000'0001ull;   // (2^64)^2 mod p

constexpr u128_lohi
MulU64xU64to128(uint64_t a, uint64_t b)
{
    const uint64_t a0 = static_cast<uint32_t>(a);
    const uint64_t a1 = a >> 32;
    const uint64_t b0 = static_cast<uint32_t>(b);
    const uint64_t b1 = b >> 32;

    const uint64_t p00 = a0 * b0; // 32x32 -> 64
    const uint64_t p01 = a0 * b1; // 32x32 -> 64
    const uint64_t p10 = a1 * b0; // 32x32 -> 64
    const uint64_t p11 = a1 * b1; // 32x32 -> 64

    // Assemble:
    const uint64_t mid =
        p01 + p10; // may overflow 64? sum of two 64-bit <= ~2^65; but for 32x32 products it's safe in 64
    uint64_t lo = p00 + (mid << 32);
    uint64_t carry = (lo < p00); // carry from adding shifted mid
    uint64_t hi = p11 + (mid >> 32) + carry;
    return {lo, hi};
}

// ---------- Reduce (hi:lo) mod p, using 2^64 ≡ 2^32 - 1 (mod p) ----------
constexpr uint64_t
ReduceModP(uint64_t hi, uint64_t lo)
{
    // r = lo + (hi << 32) - hi  (mod 2^64), then correct into [0, p)
    uint64_t r = lo + (hi << 32);
    r -= hi;
    // Up to two corrections suffice for this p
    if (r >= P)
        r -= P;
    if (r >= P)
        r -= P;
    return r;
}

constexpr uint64_t
MulModP(uint64_t a, uint64_t b)
{
    const auto prod = MulU64xU64to128(a, b);
    return ReduceModP(prod.hi, prod.lo);
}

constexpr uint64_t
PowModP(uint64_t a, uint64_t e)
{
    uint64_t base = (a >= P ? a % P : a);
    uint64_t r = 1;
    while (e) {
        if (e & 1ull)
            r = MulModP(r, base);
        base = MulModP(base, base);
        e >>= 1;
    }
    return r;
}

// Prime factorization of phi = p-1 = 2^32 * (2^32 - 1), and 2^32 - 1 = 3*5*17*257*65537
constexpr std::array<uint64_t, 6> PHI_PRIME_FACTORS = {2ull, 3ull, 5ull, 17ull, 257ull, 65537ull};
constexpr uint64_t PHI = 0xFFFF'FFFF'0000'0000ull;

constexpr bool
IsPrimitiveRoot(uint64_t g)
{
    if (g <= 1 || g >= P)
        return false;
    for (uint64_t q : PHI_PRIME_FACTORS) {
        const uint64_t e = PHI / q;
        if (PowModP(g, e) == 1ull)
            return false;
    }
    return true;
}

// --------- Compile-time search (bounded) ----------
consteval uint64_t
FindGenerator()
{
    // Try small integers; Goldilocks has small generators (e.g. 7)
    for (uint64_t g = 7; g < 2000; ++g) {
        if (IsPrimitiveRoot(g))
            return g;
    }
    // If we ever get here, widen the bound or seed with a known generator.
    return 0; // signals failure at compile time
}

static constexpr uint64_t GoldilocksP = SharkNTT::P;
static constexpr uint64_t GoldilocksGenerator = SharkNTT::FindGenerator();

// Optional compile-time sanity checks:
static_assert(GoldilocksGenerator != 0, "Failed to find generator at compile time");
static_assert(SharkNTT::PowModP(GoldilocksGenerator, SharkNTT::PHI / 2) != 1,
              "Not primitive: factor 2");
static_assert(SharkNTT::PowModP(GoldilocksGenerator, SharkNTT::PHI / 3) != 1,
              "Not primitive: factor 3");

// =========================================================================================
//                           64×64→128 helpers (portable)
// =========================================================================================
static inline void
Mul64Wide(uint64_t a, uint64_t b, uint64_t& lo, uint64_t& hi)
{
#if defined(_MSC_VER) && defined(_M_X64)
    lo = _umul128(a, b, &hi);
#else
    unsigned __int128 t = (unsigned __int128)a * (unsigned __int128)b;
    lo = (uint64_t)t;
    hi = (uint64_t)(t >> 64);
#endif
}
static inline uint64_t
Add64WithCarry(uint64_t a, uint64_t b, uint64_t& carry)
{
    uint64_t s = a + b;
    uint64_t c = (s < a);
    uint64_t out = s + carry;
    carry = c | (out < s);
    return out;
}

// =========================================================================================
//                               Prime + Montgomery core
// =========================================================================================

static inline uint64_t
AddP(uint64_t a, uint64_t b)
{
    uint64_t s = a + b;
    if (s < a || s >= P)
        s -= P;
    return s;
}
static inline uint64_t
SubP(uint64_t a, uint64_t b)
{
    return (a >= b) ? (a - b) : (a + P - b);
}

template <class SharkFloatParams>
static inline uint64_t
MongomeryMultiply(DebugHostCombo<SharkFloatParams>& debugCombo, uint64_t a, uint64_t b)
{
    // We'll count 128-bit multiplications here as 3x64
    // So 3 + 3 + 1
    debugCombo.MultiplyCounts.DebugMultiplyIncrement(7);

    // t = a*b (128-bit)
    uint64_t t_lo, t_hi;
    Mul64Wide(a, b, t_lo, t_hi);

    // m = (t_lo * NINV) mod 2^64
    uint64_t m = t_lo * SharkNTT::NINV;

    // m*P (128-bit)
    uint64_t mp_lo, mp_hi;
    Mul64Wide(m, SharkNTT::P, mp_lo, mp_hi);

    // u = t + m*P
    // low 64 + carry0
    uint64_t carry0 = 0;
    (void)Add64WithCarry(t_lo, mp_lo, carry0); // updates carry0

    // high 64 + carry0  -> also track carry-out (carry1)
    uint64_t carry1 = carry0;
    uint64_t u_hi = Add64WithCarry(t_hi, mp_hi, carry1); // returns sum, updates carry1

    // r = u / 2^64; ensure r < P
    uint64_t r = u_hi;
    if (carry1 || r >= SharkNTT::P) // <-- include the high-limb carry-out
        r -= SharkNTT::P;

    return r;
}

template <class SharkFloatParams>
static inline uint64_t
ToMontgomery(DebugHostCombo<SharkFloatParams>& debugCombo, uint64_t x)
{
    return MongomeryMultiply(debugCombo, x, R2);
}
template <class SharkFloatParams>
static inline uint64_t
FromMontgomery(DebugHostCombo<SharkFloatParams>& debugCombo, uint64_t x)
{
    return MongomeryMultiply(debugCombo, x, 1);
}

template <class SharkFloatParams>
static inline uint64_t
MontgomeryPow(DebugHostCombo<SharkFloatParams>& debugCombo, uint64_t a_mont, uint64_t e)
{
    uint64_t x = ToMontgomery(debugCombo, 1);
    uint64_t y = a_mont;
    while (e) {
        if (e & 1)
            x = MongomeryMultiply(debugCombo, x, y);
        y = MongomeryMultiply(debugCombo, y, y);
        e >>= 1;
    }
    return x;
}

struct RootTables {
    std::vector<uint64_t> stage_omegas;     // [stages]
    std::vector<uint64_t> stage_omegas_inv; // [stages]
    std::vector<uint64_t> psi_pows;         // [N]
    std::vector<uint64_t> psi_inv_pows;     // [N]
    uint64_t Ninvm_mont{0};                 // N^{-1}
};

static inline bool
IsPow2(uint32_t x)
{
    return x && (0 == (x & (x - 1)));
}

template <class SharkFloatParams>
static uint64_t
FindGenerator(DebugHostCombo<SharkFloatParams>& debugCombo)
{
    using namespace SharkNTT;
    // p-1 = 2^32 * (2^32 - 1), with distinct prime factors:
    // {2, 3, 5, 17, 257, 65537}
    static const uint64_t factors[] = {2ull, 3ull, 5ull, 17ull, 257ull, 65537ull};

    for (uint64_t g = 3; g < 1'000; ++g) {
        uint64_t g_m = ToMontgomery(debugCombo, g);
        bool ok = true;
        for (uint64_t q : factors) {
            // if g^{(p-1)/q} == 1, then g is NOT a generator
            uint64_t t = MontgomeryPow(debugCombo, g_m, PHI / q);
            if (t == ToMontgomery(debugCombo, 1)) {
                ok = false;
                break;
            }
        }
        if (ok)
            return g; // found a primitive root
    }
    // Fallback (shouldn't happen): 7 is often fine, but order-checks will catch issues.
    return 7ull;
}

template <class SharkFloatParams>
static void
BuildRoots(DebugHostCombo<SharkFloatParams>& debugCombo, uint32_t N, uint32_t stages, RootTables& T)
{
    assert(IsPow2(N));
    T.stage_omegas.resize(stages);
    T.stage_omegas_inv.resize(stages);
    T.psi_pows.resize(N);
    T.psi_inv_pows.resize(N);

    assert((PHI % (2ull * (uint64_t)N)) == 0ull && "2N must divide p-1 for PHI to exist");

    if (SharkVerbose == VerboseMode::Debug) {
        std::cout << "=== Root Building Debug ===" << std::endl;
        std::cout << "N = " << N << ", stages = " << stages << std::endl;
        std::cout << "PHI = 0x" << std::hex << PHI << std::dec << std::endl;
        std::cout << "P = 0x" << std::hex << P << std::dec << std::endl;
        std::cout << "2N = " << (2 * N) << std::endl;
        std::cout << "PHI / (2N) = " << (PHI / (2ull * N)) << std::endl;
    }

    const uint64_t generator = SharkNTT::FindGenerator();
    const uint64_t g_m = ToMontgomery(debugCombo, generator);
    const uint64_t exponent = PHI / (2ull * N);
    const uint64_t psi_m = MontgomeryPow(debugCombo, g_m, exponent);

    if (SharkVerbose == VerboseMode::Debug) {
        std::cout << "Generator g = " << generator << std::endl;
        std::cout << "g in Montgomery = 0x" << std::hex << g_m << std::dec << std::endl;
        std::cout << "Exponent for psi = " << exponent << std::endl;
        std::cout << "psi in Montgomery = 0x" << std::hex << psi_m << std::dec << std::endl;
        std::cout << "psi (normal) = " << FromMontgomery(debugCombo, psi_m) << std::endl;
    }

    // Test psi^(2N) = 1 before proceeding
    uint64_t psi_test = MontgomeryPow(debugCombo, psi_m, 2ull * (uint64_t)N);
    uint64_t one_m = ToMontgomery(debugCombo, 1);

    if (SharkVerbose == VerboseMode::Debug) {
        std::cout << "Testing psi^(2N):" << std::endl;
        std::cout << "  psi^(2N) in Montgomery = 0x" << std::hex << psi_test << std::dec << std::endl;
        std::cout << "  psi^(2N) (normal) = " << std::hex << FromMontgomery(debugCombo, psi_test) << std::dec
                  << std::endl;
        std::cout << "  Expected 1 in Montgomery = 0x" << std::hex << one_m << std::dec << std::endl;
        std::cout << "  Expected 1 (normal) = " << std::hex << FromMontgomery(debugCombo, one_m) << std::dec
                  << std::endl;
    }

    if (psi_test != one_m) {
        if (SharkVerbose == VerboseMode::Debug) {
            std::cout << "ERROR: psi does not have order 2N!" << std::endl;

            // Try some other orders to see what we actually have
            for (uint64_t test_order = 1; test_order <= 4 * N; test_order *= 2) {
                uint64_t test_result = MontgomeryPow(debugCombo, psi_m, test_order);
                std::cout << "  psi^" << std::hex << test_order << " = "
                          << FromMontgomery(debugCombo, test_result) << std::dec << std::endl;
                if (test_result == one_m) {
                    std::cout << "  ^ This equals 1! Actual order divides " << test_order << std::endl;
                    break;
                }
            }
        }
        assert(false && "psi does not have order 2N");
    }

    const uint64_t psi_inv_m = MontgomeryPow(debugCombo, psi_m, PHI - 1ull);
    const uint64_t omega_m = MongomeryMultiply(debugCombo, psi_m, psi_m);
    const uint64_t omega_inv_m = MontgomeryPow(debugCombo, omega_m, PHI - 1ull);

    for (uint32_t s = 1; s <= stages; ++s) {
        uint32_t m = 1u << s;
        uint64_t e = (uint64_t)N / (uint64_t)m;
        T.stage_omegas[s - 1] = MontgomeryPow(debugCombo, omega_m, e);
        T.stage_omegas_inv[s - 1] = MontgomeryPow(debugCombo, omega_inv_m, e);
    }

    T.psi_pows[0] = ToMontgomery(debugCombo, 1);
    T.psi_inv_pows[0] = ToMontgomery(debugCombo, 1);
    for (uint32_t i = 1; i < N; ++i) {
        T.psi_pows[i] = MongomeryMultiply(debugCombo, T.psi_pows[i - 1], psi_m);
        T.psi_inv_pows[i] = MongomeryMultiply(debugCombo, T.psi_inv_pows[i - 1], psi_inv_m);
    }

    uint64_t inv2_m = ToMontgomery(debugCombo, (P + 1) >> 1); // (p+1)/2
    uint64_t Ninvm = ToMontgomery(debugCombo, 1);
    for (uint32_t i = 0; i < stages; ++i)
        Ninvm = MongomeryMultiply(debugCombo, Ninvm, inv2_m);
    T.Ninvm_mont = Ninvm;

    if (SharkVerbose == VerboseMode::Debug) {
        std::cout << "=== End Root Building Debug ===" << std::endl;
    }
}

struct RootCache {
    std::unordered_map<uint32_t, std::unique_ptr<RootTables>> map;

    template <class SharkFloatParams>
    const RootTables&
    get(DebugHostCombo<SharkFloatParams>& debugCombo, uint32_t N, uint32_t stages)
    {
        auto it = map.find(N);
        if (it != map.end())
            return *(it->second);
        auto ptr = std::make_unique<RootTables>();
        BuildRoots<SharkFloatParams>(debugCombo, N, stages, *ptr);
        auto* raw = ptr.get();
        map.emplace(N, std::move(ptr));
        return *raw;
    }
};

static void
BitReverseInplace64(uint64_t* A, uint32_t N, uint32_t stages)
{
    for (uint32_t i = 0; i < N; ++i) {
        uint32_t j = ReverseBits32(i, stages) & (N - 1);
        if (j > i)
            std::swap(A[i], A[j]);
    }
}

template <class SharkFloatParams>
static void
NTTRadix2(DebugHostCombo<SharkFloatParams>& debugCombo,
           uint64_t* A,
           uint32_t N,
           uint32_t stages,
           const std::vector<uint64_t>& stage_base)
{
    for (uint32_t s = 1; s <= stages; ++s) {
        uint32_t m = 1u << s;
        uint32_t half = m >> 1;
        uint64_t w_m = stage_base[s - 1];
        for (uint32_t k = 0; k < N; k += m) {
            uint64_t w = ToMontgomery(debugCombo, 1);
            for (uint32_t j = 0; j < half; ++j) {
                uint64_t U = A[k + j];
                uint64_t V = A[k + j + half];
                uint64_t t = MongomeryMultiply(debugCombo, V, w);
                A[k + j] = AddP(U, t);
                A[k + j + half] = SubP(U, t);
                w = MongomeryMultiply(debugCombo, w, w_m);
            }
        }
    }
}

} // namespace SharkNTT

// =========================================================================================
//                                    Planner (prime)
// =========================================================================================
struct PlanPrime {
    int n32 = 0;
    int b = 0;
    int L = 0;
    int N = 0;
    int stages = 0;
    bool ok = false;

    void
    Print()
    {
        if (SharkVerbose == VerboseMode::Debug) {
            std::cout << "PlanPrime: n32=" << n32 << " b=" << b << " L=" << L << " N=" << N
                      << " stages=" << stages << " ok=" << ok << std::endl;
        }
    }
};

static inline uint32_t
NextPow2U32(uint32_t x)
{
    if (x <= 1)
        return 1u;
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return x + 1;
}
static inline uint32_t
CeilDivU32(uint32_t a, uint32_t b)
{
    return (a + b - 1u) / b;
}
static inline int
CeilLog2U32(uint32_t x)
{
    int l = 0;
    uint32_t v = x - 1;
    while (v) {
        v >>= 1;
        ++l;
    }
    return l;
}

static PlanPrime
BuildPlanPrime(int n32, int b_hint = 26, int margin = 2)
{
    using namespace SharkNTT;
    PlanPrime pl{};
    pl.ok = false;
    pl.n32 = n32;
    const uint64_t totalBits = (uint64_t)n32 * 32ull;
    int b = b_hint;
    if (b < 16)
        b = 16;
    if (b > 30)
        b = 30;
    uint32_t L = CeilDivU32((uint32_t)totalBits, (uint32_t)b);
    uint32_t N = NextPow2U32(2u * L);
    int lgN = CeilLog2U32(N);
    int bmax = (64 - margin - lgN) / 2;
    if (b > bmax)
        b = bmax;
    if (b < 16)
        b = 16;
    L = CeilDivU32((uint32_t)totalBits, (uint32_t)b);
    N = NextPow2U32(2u * L);
    lgN = CeilLog2U32(N);

    if ((PHI % (2ull * (uint64_t)N)) != 0ull) {
        int b_up_max = std::min(bmax, 30);
        bool fixed = false;
        for (int b_try = b + 1; b_try <= b_up_max; ++b_try) {
            uint32_t L2 = CeilDivU32((uint32_t)totalBits, (uint32_t)b_try);
            uint32_t N2 = NextPow2U32(2u * L2);
            if ((PHI % (2ull * (uint64_t)N2)) == 0ull) {
                b = b_try;
                L = L2;
                N = N2;
                lgN = CeilLog2U32(N2);
                fixed = true;
                break;
            }
        }
        (void)fixed;
    }
    assert((PHI % (2ull * (uint64_t)N)) == 0ull && "2N must divide p-1 for PHI to exist");
    int need = 2 * b + lgN + margin;
    pl.ok = (need <= 64) && (N >= 2);
    pl.b = b;
    pl.L = (int)L;
    pl.N = (int)N;
    pl.stages = lgN;
    return pl;
}

// =========================================================================================
//                       Pack (base-2^b) and Unpack (to Final128)
// =========================================================================================

template <class P>
static uint64_t
ReadBitsSimple(const HpSharkFloat<P>* X, int64_t q, int b)
{
    const int B = P::GlobalNumUint32 * 32;
    if (q >= B || q < 0)
        return 0;
    uint64_t v = 0;
    int need = b;
    int outPos = 0;
    int64_t bit = q;
    while (need > 0 && bit < B) {
        int64_t w = bit / 32;
        int off = (int)(bit % 32);
        uint32_t limb = (w >= 0) ? X->Digits[(int)w] : 0u;
        uint32_t chunk = (off ? (limb >> off) : limb);
        int take = std::min(32 - off, need);
        v |= (uint64_t)(chunk & ((take == 32) ? 0xFFFFFFFFu : ((1u << take) - 1u))) << outPos;
        outPos += take;
        need -= take;
        bit += take;
    }
    return (b == 64) ? v : (v & ((1ull << b) - 1ull));
}

template <class SharkFloatParams>
static void
PackBase2bPrime(DebugHostCombo<SharkFloatParams>& debugCombo,
                  const HpSharkFloat<SharkFloatParams>* X,
                  const PlanPrime& pl,
                  std::vector<uint64_t>& Out,
                  std::vector<uint64_t>* RawOut = nullptr)
{
    Out.assign((size_t)pl.N, SharkNTT::ToMontgomery(debugCombo, 0));
    if (RawOut)
        RawOut->assign((size_t)pl.L, 0ull);

    for (int i = 0; i < pl.L; ++i) {
        uint64_t coeff = ReadBitsSimple(X, (int64_t)i * pl.b, pl.b);
        uint64_t cmod = coeff % SharkNTT::P;
        if (RawOut)
            (*RawOut)[(size_t)i] = cmod;
        Out[(size_t)i] = SharkNTT::ToMontgomery(debugCombo, cmod); // qualify helpers too
    }
}

static void
UnpackPrimeToFinal128(const uint64_t* A_norm, const PlanPrime& pl, uint64_t* Final128, size_t Ddigits)
{
    using namespace SharkNTT;
    std::memset(Final128, 0, sizeof(uint64_t) * 2 * Ddigits);
    const uint64_t HALF = (P - 1ull) >> 1;
    const int Imax = std::min(pl.N, 2 * pl.L - 1);
    for (int i = 0; i < Imax; ++i) {
        uint64_t v = A_norm[i];
        if (!v)
            continue;
        bool neg = (v > HALF);
        uint64_t mag64 = neg ? (P - v) : v;
        const uint64_t sBits = (uint64_t)i * (uint64_t)pl.b;
        const size_t q = (size_t)(sBits >> 5);
        const int r = (int)(sBits & 31);
        const uint64_t lo64 = (r ? (mag64 << r) : mag64);
        const uint64_t hi64 = (r ? (mag64 >> (64 - r)) : 0ull);
        uint32_t d0 = (uint32_t)(lo64 & 0xffffffffu), d1 = (uint32_t)((lo64 >> 32) & 0xffffffffu),
                 d2 = (uint32_t)(hi64 & 0xffffffffu), d3 = (uint32_t)((hi64 >> 32) & 0xffffffffu);
        auto add32 = [&](size_t j, uint32_t add) {
            if (!add)
                return;
            uint64_t& lo = Final128[2 * j + 0];
            uint64_t& hi = Final128[2 * j + 1];
            uint64_t old = lo;
            lo += (uint64_t)add;
            if (lo < old)
                hi += 1ull;
        };
        auto sub32 = [&](size_t j, uint32_t sub) {
            if (!sub)
                return;
            uint64_t& lo = Final128[2 * j + 0];
            uint64_t& hi = Final128[2 * j + 1];
            uint64_t old = lo;
            uint64_t dif = old - (uint64_t)sub;
            lo = dif;
            if (old < (uint64_t)sub)
                hi -= 1ull;
        };
        if (!neg) {
            add32(q + 0, d0);
            add32(q + 1, d1);
            add32(q + 2, d2);
            add32(q + 3, d3);
        } else {
            sub32(q + 0, d0);
            sub32(q + 1, d1);
            sub32(q + 2, d2);
            sub32(q + 3, d3);
        }
    }
}

template <typename SharkFloatParams>
bool
VerifyMontgomeryConstants(DebugHostCombo<SharkFloatParams>& debugCombo,
                          uint64_t P,
                          uint64_t NINV,
                          uint64_t R2)
{
    if (SharkVerbose == VerboseMode::Debug) {
        std::cout << "=== Enhanced Montgomery Verification ===\n";
        std::cout << "P = 0x" << std::hex << P << std::dec << "\n";
        std::cout << "NINV = 0x" << std::hex << NINV << std::dec << "\n";
        std::cout << "R2 = 0x" << std::hex << R2 << std::dec << "\n";

        // Test 1: Basic Montgomery property (P * NINV ≡ 2^64-1 mod 2^64)
        uint64_t test1 = P * NINV; // unsigned wrap is modulo 2^64
        std::cout << "P * NINV = 0x" << std::hex << test1 << std::dec;
        if (test1 == 0xFFFFFFFFFFFFFFFFull) {
            std::cout << " CORRECT\n";
        } else {
            std::cout << " INCORRECT (should be 0xFFFFFFFFFFFFFFFF)\n";
            return false;
        }

        // Test 2: Montgomery multiplication of 1
        uint64_t one_m = SharkNTT::ToMontgomery(debugCombo, 1);
        std::cout << "ToMontgomery(1) = 0x" << std::hex << one_m << std::dec << "\n";

        uint64_t one_squared = SharkNTT::MongomeryMultiply(debugCombo, one_m, one_m);
        std::cout << "MongomeryMultiply(ToMontgomery(1), ToMontgomery(1)) = 0x" << std::hex << one_squared << std::dec
                  << "\n";

        uint64_t back_to_one = SharkNTT::FromMontgomery(debugCombo, one_squared);
        std::cout << "FromMontgomery(result) = " << std::hex << back_to_one << std::dec;
        if (back_to_one == 1) {
            std::cout << " CORRECT\n";
        } else {
            std::cout << " INCORRECT\n";
            return false;
        }

        // Test 3: Verify R2 is correct (MongomeryMultiply(1, R2) == ToMontgomery(1))
        uint64_t mont_1 = SharkNTT::MongomeryMultiply(debugCombo, 1, R2);
        if (mont_1 == one_m) {
            std::cout << "R2 verification: CORRECT\n";
        } else {
            std::cout << "R2 verification: INCORRECT\n";
            std::cout << "  MongomeryMultiply(1, R2) = 0x" << std::hex << mont_1 << std::dec << "\n";
            std::cout << "  ToMontgomery(1) = 0x" << std::hex << one_m << std::dec << "\n";
            return false;
        }
    }

    return true;
}

template <class SharkFloatParams>
static void
VerifyNTTRoots(DebugHostCombo<SharkFloatParams>& debugCombo,
                 const SharkNTT::RootTables& T,
                 const PlanPrime& pl)
{
    if (SharkVerbose == VerboseMode::Debug) {
        std::cout << "  NINV = 0x" << std::hex << SharkNTT::NINV << std::dec << "\n";
        std::cout << "  R2   = 0x" << std::hex << SharkNTT::R2 << std::dec << "\n";
        std::cout << "  Generator g = 0x" << std::hex << SharkNTT::FindGenerator(debugCombo)
                  << std::dec << "\n";
        std::cout << "  psi = 0x" << std::hex << SharkNTT::FromMontgomery(debugCombo, T.psi_pows[1])
                  << std::dec << " (order " << (2 * pl.N) << ")\n";
        std::cout << "  omega = 0x" << std::hex
                  << SharkNTT::FromMontgomery(
                         debugCombo, SharkNTT::MongomeryMultiply(debugCombo, T.psi_pows[1], T.psi_pows[1]))
                  << std::dec << " (order " << pl.N << ")\n";
    }

    // Use the table's own 1 in Montgomery domain
    const uint64_t one_m = T.psi_pows[0];

    // ψ is T.psi_pows[1] (Montgomery). Check ψ^(2N) = 1
    uint64_t psi2N = SharkNTT::MontgomeryPow(debugCombo, T.psi_pows[1], 2ull * (uint64_t)pl.N);
    assert(psi2N == one_m && "psi^(2N) != 1 (check N and roots)");

    // ω = ψ^2. Check ω^N = 1
    uint64_t omega = SharkNTT::MongomeryMultiply(debugCombo, T.psi_pows[1], T.psi_pows[1]);
    uint64_t omegaN = SharkNTT::MontgomeryPow(debugCombo, omega, (uint64_t)pl.N);
    assert(omegaN == one_m && "omega^N != 1 (check N and roots)");
}

// =========================================================================================
//                          MultiplyHelperFFT2 (prime backend)
// =========================================================================================

template <class SharkFloatParams>
void
MultiplyHelperFFT2(const HpSharkFloat<SharkFloatParams>* A,
                   const HpSharkFloat<SharkFloatParams>* B,
                   HpSharkFloat<SharkFloatParams>* OutXX,
                   HpSharkFloat<SharkFloatParams>* OutXY,
                   HpSharkFloat<SharkFloatParams>* OutYY,
                   DebugHostCombo<SharkFloatParams>& debugCombo)
{

    using P = SharkFloatParams;
    using namespace SharkNTT;

    // x must be a positive constant expression

    // Verify power of 2
    static_assert(SharkFloatParams::GlobalNumUint32 > 0 &&
                      (SharkFloatParams::GlobalNumUint32 & (SharkFloatParams::GlobalNumUint32 - 1)) == 0,
                  "GlobalNumUint32 must be a power of 2");

    // --------- Plan and tables ---------
    PlanPrime pl = BuildPlanPrime(P::GlobalNumUint32, /*b_hint=*/26, /*margin=*/2);
    pl.Print();

    assert(pl.ok && "Prime plan build failed (check b/N headroom constraints)");
    assert(pl.N >= 2 * pl.L && "No-wrap condition violated: need N >= 2*L");
    assert((PHI % (2ull * (uint64_t)pl.N)) == 0ull);

    RootCache root_cache;
    const RootTables& T = root_cache.get(debugCombo, (uint32_t)pl.N, (uint32_t)pl.stages);

    // assuming you have `debugCombo` (your context) and constants already:
    const bool constants_ok =
        VerifyMontgomeryConstants(debugCombo, SharkNTT::P, SharkNTT::NINV, SharkNTT::R2);
    assert(constants_ok && "Montgomery constants verification failed");

    VerifyNTTRoots(debugCombo, T, pl);

    auto run_conv = [&](const HpSharkFloat<P>* XA,
                        const HpSharkFloat<P>* XB,
                        HpSharkFloat<P>* OUT,
                        const HpSharkFloat<P>& inA,
                        const HpSharkFloat<P>& inB,
                        int addFactorOfTwo,
                        bool isNegative) {
        // 1) Pack
        std::vector<uint64_t> X, Y;
        X.reserve(pl.N);
        Y.reserve(pl.N);
        std::vector<uint64_t> RawA, RawB;
        RawA.reserve(pl.L);
        RawB.reserve(pl.L);
        PackBase2bPrime(debugCombo, XA, pl, X, &RawA);
        PackBase2bPrime(debugCombo, XB, pl, Y, &RawB);
        // 2) Twist
        for (int i = 0; i < pl.N; ++i) {
            X[i] = MongomeryMultiply(debugCombo, X[i], T.psi_pows[(size_t)i]);
        }
        for (int i = 0; i < pl.N; ++i) {
            Y[i] = MongomeryMultiply(debugCombo, Y[i], T.psi_pows[(size_t)i]);
        }
        { // Roundtrip twisted X
            std::vector<uint64_t> Xchk = X, Xorig = X;
            BitReverseInplace64(Xchk.data(), (uint32_t)pl.N, (uint32_t)pl.stages);
            NTTRadix2(debugCombo, Xchk.data(), (uint32_t)pl.N, (uint32_t)pl.stages, T.stage_omegas);
            BitReverseInplace64(Xchk.data(), (uint32_t)pl.N, (uint32_t)pl.stages);
            NTTRadix2(debugCombo, Xchk.data(), (uint32_t)pl.N, (uint32_t)pl.stages, T.stage_omegas_inv);
            for (int i = 0; i < pl.N; ++i)
                Xchk[(size_t)i] = MongomeryMultiply(debugCombo, Xchk[(size_t)i], T.Ninvm_mont);
            for (int i = 0; i < pl.N; ++i)
                assert(Xchk[(size_t)i] == Xorig[(size_t)i]);
        }

        // 3) Forward NTT
        BitReverseInplace64(X.data(), (uint32_t)pl.N, (uint32_t)pl.stages);
        BitReverseInplace64(Y.data(), (uint32_t)pl.N, (uint32_t)pl.stages);
        NTTRadix2(debugCombo, X.data(), (uint32_t)pl.N, (uint32_t)pl.stages, T.stage_omegas);
        NTTRadix2(debugCombo, Y.data(), (uint32_t)pl.N, (uint32_t)pl.stages, T.stage_omegas);
        // 4) Pointwise
        for (int i = 0; i < pl.N; ++i)
            X[i] = MongomeryMultiply(debugCombo, X[i], Y[i]);
        // 5) Inverse NTT
        BitReverseInplace64(X.data(), (uint32_t)pl.N, (uint32_t)pl.stages);
        NTTRadix2(debugCombo, X.data(), (uint32_t)pl.N, (uint32_t)pl.stages, T.stage_omegas_inv);
        // 6) Untwist + scale
        for (int i = 0; i < pl.N; ++i) {
            uint64_t v = MongomeryMultiply(debugCombo, X[i], T.psi_inv_pows[(size_t)i]);
            X[i] = MongomeryMultiply(debugCombo, v, T.Ninvm_mont);
        }
        // 7) Convert out of Montgomery
        std::vector<uint64_t> X_norm((size_t)pl.N);
        for (int i = 0; i < pl.N; ++i)
            X_norm[(size_t)i] = FromMontgomery(debugCombo, X[i]);

        { // Full coefficient check vs direct convolution (mod p)
            const int Kmax = 2 * pl.L - 1;
            for (int k = 0; k < Kmax; ++k) {
                uint64_t cref_m = ToMontgomery(debugCombo, 0);
                int i0 = std::max(0, k - (pl.L - 1));
                int i1 = std::min(k, pl.L - 1);
                for (int i = i0; i <= i1; ++i) {
                    uint64_t ai_m = ToMontgomery(debugCombo, RawA[(size_t)i]);
                    uint64_t bj_m = ToMontgomery(debugCombo, RawB[(size_t)(k - i)]);
                    cref_m = AddP(cref_m, MongomeryMultiply(debugCombo, ai_m, bj_m));
                }
                uint64_t cref = FromMontgomery(debugCombo, cref_m);
                assert(X_norm[(size_t)k] == cref && "Convolution mismatch");
            }
        }

        // 8) Unpack to Final128 and normalize
        const size_t Ddigits = ((size_t)((2 * pl.L - 2) * (int64_t)pl.b + 64) + 31) / 32 + 2;
        std::vector<uint64_t> Final128(2ull * Ddigits, 0ull);
        UnpackPrimeToFinal128(X_norm.data(), pl, Final128.data(), Ddigits);
        OUT->SetNegative(isNegative);
        Normalize<P>(*OUT, inA, inB, Final128.data(), Ddigits, addFactorOfTwo);
    };

    // XX = A^2
    run_conv(A, A, OutXX, *A, *A, /*additionalFactorOfTwo=*/0, /*isNegative=*/false);
    // YY = B^2
    run_conv(B, B, OutYY, *B, *B, /*additionalFactorOfTwo=*/0, /*isNegative=*/false);
    // XY = 2*(A*B)
    run_conv(A,
             B,
             OutXY,
             *A,
             *B,
             /*additionalFactorOfTwo=*/1,
             /*isNegative=*/(A->GetNegative() ^ B->GetNegative()));
}


#define ExplicitlyInstantiate(SharkFloatParams) \
template void MultiplyHelperFFT2<SharkFloatParams>(const HpSharkFloat<SharkFloatParams>*, \
                                                   const HpSharkFloat<SharkFloatParams>*, \
                                                   HpSharkFloat<SharkFloatParams>*, \
                                                   HpSharkFloat<SharkFloatParams>*, \
                                                   HpSharkFloat<SharkFloatParams>*, \
                                                   DebugHostCombo<SharkFloatParams>&); \

ExplicitInstantiateAll();
