#include "MultiplyNTTCudaSetup.h"

#include <algorithm>
#include <array>
#include <iostream>

#include "DebugChecksumHost.h"
#include "HpSharkFloat.h"
#include "NTTConstexprGenerator.h"
#include "TestVerbose.h"

namespace SharkNTT {

void
PlanPrime::Print()
{
    if (SharkVerbose == VerboseMode::Debug) {
        std::cout << "PlanPrime: n32=" << n32 << " b=" << b << " L=" << L << " N=" << N
                  << " stages=" << stages << " ok=" << ok << std::endl;
    }
}

void
Mul64Wide(uint64_t a, uint64_t b, uint64_t &lo, uint64_t &hi)
{
#if defined(_MSC_VER) && defined(_M_X64)
    lo = _umul128(a, b, &hi);
#else
    unsigned __int128 t = (unsigned __int128)a * (unsigned __int128)b;
    lo = (uint64_t)t;
    hi = (uint64_t)(t >> 64);
#endif
}

uint64_t
Add64WithCarry(uint64_t a, uint64_t b, uint64_t &carry)
{
    uint64_t s = a + b;
    uint64_t c = (s < a);
    uint64_t out = s + carry;
    carry = c | (out < s);
    return out;
}

template <class SharkFloatParams>
uint64_t
MontgomeryMul(uint64_t a, uint64_t b)
{
    // t = a*b (128-bit)
    uint64_t t_lo, t_hi;
    Mul64Wide(a, b, t_lo, t_hi);

    // m = (t_lo * NINV) mod 2^64
    uint64_t m = t_lo * SharkNTT::MagicPrimeInv;

    // m*SharkNTT::MagicPrime (128-bit)
    uint64_t mp_lo, mp_hi;
    Mul64Wide(m, SharkNTT::MagicPrime, mp_lo, mp_hi);

    // u = t + m*SharkNTT::MagicPrime
    // low 64 + carry0
    uint64_t carry0 = 0;
    (void)Add64WithCarry(t_lo, mp_lo, carry0); // updates carry0

    // high 64 + carry0  -> also track carry-out (carry1)
    uint64_t carry1 = carry0;
    uint64_t u_hi = Add64WithCarry(t_hi, mp_hi, carry1); // returns sum, updates carry1

    // if (carry1) {
    //     for (;;)
    //         ;
    // }

    // r = u / 2^64; ensure r < SharkNTT::MagicPrime (include the high-limb carry-out)
    uint64_t r = u_hi;
    if (carry1 || r >= SharkNTT::MagicPrime)
        r -= SharkNTT::MagicPrime;

    return r;
}

template <class SharkFloatParams>
uint64_t
MontgomeryMul(DebugHostCombo<SharkFloatParams> &debugCombo, uint64_t a, uint64_t b)
{
    // We'll count 128-bit multiplications here as 3x64  (so 3 + 3 + 1)
    debugCombo.MultiplyCounts.DebugMultiplyIncrement(7);

    return MontgomeryMul<SharkFloatParams>(a, b);
}

template <class SharkFloatParams>
uint64_t
ToMontgomery(DebugHostCombo<SharkFloatParams> &debugCombo, uint64_t x)
{
    return MontgomeryMul<SharkFloatParams>(debugCombo, x, R2);
}

template <class SharkFloatParams>
uint64_t
ToMontgomery(uint64_t x)
{
    return MontgomeryMul<SharkFloatParams>(x, R2);
}

template <class SharkFloatParams>
uint64_t
FromMontgomery(DebugHostCombo<SharkFloatParams> &debugCombo, uint64_t x)
{
    return MontgomeryMul<SharkFloatParams>(debugCombo, x, 1);
}

template <class SharkFloatParams>
uint64_t
FromMontgomery(uint64_t x)
{
    return MontgomeryMul<SharkFloatParams>(x, 1);
}

template <class SharkFloatParams>
uint64_t
MontgomeryPow(DebugHostCombo<SharkFloatParams> &debugCombo, uint64_t a_mont, uint64_t e)
{
    uint64_t x = ToMontgomery(debugCombo, 1);
    uint64_t y = a_mont;
    while (e) {
        if (e & 1)
            x = MontgomeryMul(debugCombo, x, y);
        y = MontgomeryMul(debugCombo, y, y);
        e >>= 1;
    }
    return x;
}

template <class SharkFloatParams>
uint64_t
MontgomeryPow(uint64_t a_mont, uint64_t e)
{
    uint64_t x = ToMontgomery<SharkFloatParams>(1);
    uint64_t y = a_mont;
    while (e) {
        if (e & 1)
            x = MontgomeryMul<SharkFloatParams>(x, y);
        y = MontgomeryMul<SharkFloatParams>(y, y);
        e >>= 1;
    }
    return x;
}

template <class SharkFloatParams>
void
BuildRoots(uint32_t N, uint32_t stages, RootTables &roots)
{
    roots.stages = stages;
    roots.N = N;

    // Existing allocations
    roots.stage_omegas = new uint64_t[stages];
    roots.stage_omegas_inv = new uint64_t[stages];

    roots.psi_pows = new uint64_t[N];
    roots.psi_inv_pows = new uint64_t[N];

    // -----------------------------
    // 1) Generator and psi/omega in Montgomery domain
    // -----------------------------
    const uint64_t generator = SharkNTT::FindGeneratorConstexpr(); // normal-domain g
    const uint64_t g_m = ToMontgomery<SharkFloatParams>(generator);

    const uint64_t exponent = SharkNTT::PHI / (2ull * N);
    const uint64_t psi_m = MontgomeryPow<SharkFloatParams>(g_m, exponent);

    const uint64_t psi_inv_m = MontgomeryPow<SharkFloatParams>(psi_m, SharkNTT::PHI - 1ull);
    const uint64_t omega_m = MontgomeryMul<SharkFloatParams>(psi_m, psi_m); // ω = ψ^2 (Montgomery)
    const uint64_t omega_inv_m = MontgomeryPow<SharkFloatParams>(omega_m, SharkNTT::PHI - 1ull);

    // Per-stage ω_m and ω_m^{-1} (Montgomery)
    for (uint32_t s = 1; s <= stages; ++s) {
        uint32_t m = 1u << s;
        uint64_t e = static_cast<uint64_t>(N) / static_cast<uint64_t>(m);
        roots.stage_omegas[s - 1] = MontgomeryPow<SharkFloatParams>(omega_m, e);
        roots.stage_omegas_inv[s - 1] = MontgomeryPow<SharkFloatParams>(omega_inv_m, e);
    }

    // -----------------------------
    // 2) psi powers (forward / inverse) in Montgomery domain
    // -----------------------------
    roots.psi_pows[0] = ToMontgomery<SharkFloatParams>(1);
    roots.psi_inv_pows[0] = ToMontgomery<SharkFloatParams>(1);
    for (uint32_t i = 1; i < N; ++i) {
        roots.psi_pows[i] = MontgomeryMul<SharkFloatParams>(roots.psi_pows[i - 1], psi_m);
        roots.psi_inv_pows[i] = MontgomeryMul<SharkFloatParams>(roots.psi_inv_pows[i - 1], psi_inv_m);
    }

    // -----------------------------
    // 3) N^{-1} in Montgomery form: (1/2)^stages
    // -----------------------------
    uint64_t inv2_m = ToMontgomery<SharkFloatParams>((SharkNTT::MagicPrime + 1) >> 1); // (p+1)/2
    uint64_t Ninvm = ToMontgomery<SharkFloatParams>(1);
    for (uint32_t i = 0; i < stages; ++i)
        Ninvm = MontgomeryMul<SharkFloatParams>(Ninvm, inv2_m);
    roots.Ninvm_mont = Ninvm;

    // -----------------------------
    // 4) per-stage twiddle lookup table for GPU
    //    stage_twiddles_fwd:  w_{s,j} = (stage_omegas[s-1])^j in Montgomery domain
    // -----------------------------
    // RootTables is assumed to have:
    //   uint64_t *stage_twiddles_fwd;
    //   uint32_t *stage_twiddle_offset;
    //
    // Total twiddles: sum_{s=1..stages} 2^{s-1} = (1 << stages) - 1
    uint32_t total_twiddles = 0;
    for (uint32_t s = 1; s <= stages; ++s) {
        total_twiddles += (1u << (s - 1));
    }

    roots.stage_twiddles_fwd = new uint64_t[total_twiddles];
    roots.stage_twiddles_inv = new uint64_t[total_twiddles];

    const uint64_t one_mont = ToMontgomery<SharkFloatParams>(1);

    uint32_t offset = 0;

    for (uint32_t s = 1; s <= stages; ++s) {
        const uint32_t half = (1u << (s - 1));    // twiddles j = 0..half-1
        uint64_t w_m = roots.stage_omegas[s - 1]; // base for this stage (Montgomery)
        uint64_t w = one_mont;

        // j = 0
        roots.stage_twiddles_fwd[offset + 0] = w;

        // j = 1..half-1
        for (uint32_t j = 1; j < half; ++j) {
            w = MontgomeryMul<SharkFloatParams>(w, w_m);
            roots.stage_twiddles_fwd[offset + j] = w;
        }

        offset += half;
    }

    // Inverse table: same offsets, but use stage_omegas_inv
    offset = 0;
    for (uint32_t s = 1; s <= stages; ++s) {
        const uint32_t half = 1u << (s - 1);

        uint64_t w_m_inv = roots.stage_omegas_inv[s - 1]; // w^{-1} for this stage
        uint64_t w = one_mont;

        roots.stage_twiddles_inv[offset + 0] = w;
        for (uint32_t j = 1; j < half; ++j) {
            w = MontgomeryMul<SharkFloatParams>(w, w_m_inv);
            roots.stage_twiddles_inv[offset + j] = w; // (w^{-1})^j = w^{-j}
        }

        offset += half;
    }

    roots.total_twiddles = total_twiddles;

    // -----------------------------
    // 5) Debug output
    // -----------------------------
    if (SharkVerbose == VerboseMode::Debug) {
        std::cout << "Roots for N=" << N << " stages=" << stages << std::endl;
        for (uint32_t s = 1; s <= stages; ++s) {
            std::cout << "Stage " << s << " omega: " << roots.stage_omegas[s - 1]
                      << " omega_inv: " << roots.stage_omegas_inv[s - 1] << std::endl;
        }
        std::cout << "Total twiddles: " << total_twiddles << std::endl;
        std::cout << "Ninvm_mont = " << roots.Ninvm_mont << std::endl;
    }
}

template <class SharkFloatParams>
void
CopyRootsToCuda(RootTables &outT, const RootTables &inT)
{
    // 1) Shallow copy the struct (scalar fields OK, pointer fields still host-side)
    cudaMemcpy(&outT, &inT, sizeof(RootTables), cudaMemcpyHostToDevice);

    // 2) stage_omegas / stage_omegas_inv  [stages elements]
    {
        size_t stage_bytes = static_cast<size_t>(inT.stages) * sizeof(uint64_t);

        uint64_t *stage_omegas_ptr = nullptr;
        cudaMalloc(&stage_omegas_ptr, stage_bytes);
        cudaMemcpy(stage_omegas_ptr, inT.stage_omegas, stage_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(&outT.stage_omegas, &stage_omegas_ptr, sizeof(uint64_t *), cudaMemcpyHostToDevice);

        uint64_t *stage_omegas_inv_ptr = nullptr;
        cudaMalloc(&stage_omegas_inv_ptr, stage_bytes);
        cudaMemcpy(stage_omegas_inv_ptr, inT.stage_omegas_inv, stage_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(
            &outT.stage_omegas_inv, &stage_omegas_inv_ptr, sizeof(uint64_t *), cudaMemcpyHostToDevice);
    }

    // 3) psi_pows / psi_inv_pows  [N elements]
    {
        size_t N_bytes = static_cast<size_t>(inT.N) * sizeof(uint64_t);

        uint64_t *psi_pows_ptr = nullptr;
        cudaMalloc(&psi_pows_ptr, N_bytes);
        cudaMemcpy(psi_pows_ptr, inT.psi_pows, N_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(&outT.psi_pows, &psi_pows_ptr, sizeof(uint64_t *), cudaMemcpyHostToDevice);

        uint64_t *psi_inv_pows_ptr = nullptr;
        cudaMalloc(&psi_inv_pows_ptr, N_bytes);
        cudaMemcpy(psi_inv_pows_ptr, inT.psi_inv_pows, N_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(&outT.psi_inv_pows, &psi_inv_pows_ptr, sizeof(uint64_t *), cudaMemcpyHostToDevice);
    }

    // 4) stage_twiddles_fwd / stage_twiddles_inv  [total_twiddles elements]
    {
        size_t tw_bytes = static_cast<size_t>(inT.total_twiddles) * sizeof(uint64_t);

        uint64_t *stage_twiddles_fwd_ptr = nullptr;
        cudaMalloc(&stage_twiddles_fwd_ptr, tw_bytes);
        cudaMemcpy(stage_twiddles_fwd_ptr, inT.stage_twiddles_fwd, tw_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(&outT.stage_twiddles_fwd,
                   &stage_twiddles_fwd_ptr,
                   sizeof(uint64_t *),
                   cudaMemcpyHostToDevice);

        uint64_t *stage_twiddles_inv_ptr = nullptr;
        cudaMalloc(&stage_twiddles_inv_ptr, tw_bytes);
        cudaMemcpy(stage_twiddles_inv_ptr, inT.stage_twiddles_inv, tw_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(&outT.stage_twiddles_inv,
                   &stage_twiddles_inv_ptr,
                   sizeof(uint64_t *),
                   cudaMemcpyHostToDevice);
    }

    // 5) Scalars that might be handy to explicitly refresh (already copied in step 1,
    //    but this keeps your original pattern)
    cudaMemcpy(&outT.Ninvm_mont, &inT.Ninvm_mont, sizeof(uint64_t), cudaMemcpyHostToDevice);
    // N, stages, total_twiddles, etc. are already copied by the struct memcpy.
}

template <class SharkFloatParams>
void
DestroyRoots(bool cuda, RootTables &roots)
{
    if (cuda == false) {
        delete[] roots.stage_omegas;
        delete[] roots.stage_omegas_inv;
        delete[] roots.psi_pows;
        delete[] roots.psi_inv_pows;
        delete[] roots.stage_twiddles_fwd;
        delete[] roots.stage_twiddles_inv;

        roots.stage_omegas = nullptr;
        roots.stage_omegas_inv = nullptr;
        roots.psi_pows = nullptr;
        roots.psi_inv_pows = nullptr;
        roots.stage_twiddles_fwd = nullptr;
        roots.stage_twiddles_inv = nullptr;

        roots.stages = 0;
        roots.N = 0;
        roots.Ninvm_mont = 0;
        roots.total_twiddles = 0;
    } else {
        RootTables localRoots;
        cudaMemcpy(&localRoots, &roots, sizeof(RootTables), cudaMemcpyDeviceToHost);

        cudaFree(localRoots.stage_omegas);
        cudaFree(localRoots.stage_omegas_inv);
        cudaFree(localRoots.psi_pows);
        cudaFree(localRoots.psi_inv_pows);
        cudaFree(localRoots.stage_twiddles_fwd);
        cudaFree(localRoots.stage_twiddles_inv);

        cudaMemset(&roots, 0, sizeof(RootTables));
    }
}

} // namespace SharkNTT

#define ExplicitlyInstantiate(SharkFloatParams)                                                         \
    template void SharkNTT::BuildRoots<SharkFloatParams>(uint32_t, uint32_t, SharkNTT::RootTables &);   \
    template uint64_t SharkNTT::MontgomeryMul<SharkFloatParams>(uint64_t a, uint64_t b);                \
    template uint64_t SharkNTT::MontgomeryMul<SharkFloatParams>(                                        \
        DebugHostCombo<SharkFloatParams> & debugCombo, uint64_t a, uint64_t b);                         \
    template uint64_t SharkNTT::ToMontgomery<SharkFloatParams>(uint64_t x);                             \
    template uint64_t SharkNTT::ToMontgomery<SharkFloatParams>(                                         \
        DebugHostCombo<SharkFloatParams> & debugCombo, uint64_t x);                                     \
    template uint64_t SharkNTT::FromMontgomery<SharkFloatParams>(uint64_t x);                           \
    template uint64_t SharkNTT::FromMontgomery<SharkFloatParams>(                                       \
        DebugHostCombo<SharkFloatParams> & debugCombo, uint64_t x);                                     \
    template uint64_t SharkNTT::MontgomeryPow<SharkFloatParams>(uint64_t a_mont, uint64_t e);           \
    template uint64_t SharkNTT::MontgomeryPow<SharkFloatParams>(                                        \
        DebugHostCombo<SharkFloatParams> & debugCombo, uint64_t a_mont, uint64_t e);                    \
    template void SharkNTT::CopyRootsToCuda<SharkFloatParams>(SharkNTT::RootTables & outT,              \
                                                              const SharkNTT::RootTables &inT);         \
    template void SharkNTT::DestroyRoots<SharkFloatParams>(bool cuda, SharkNTT::RootTables &T);

ExplicitInstantiateAll();
