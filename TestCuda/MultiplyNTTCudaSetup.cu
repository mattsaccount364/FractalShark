#include "MultiplyNTTCudaSetup.h"

#include <algorithm>
#include <array>
#include <iostream>

#include "NTTConstexprGenerator.h"
#include "HpSharkFloat.cuh"
#include "TestVerbose.h"
#include "DebugChecksumHost.h"

namespace SharkNTT {

void
PlanPrime::Print()
{
    if (SharkVerbose == VerboseMode::Debug) {
        std::cout << "PlanPrime: n32=" << n32 << " b=" << b << " L=" << L << " N=" << N
                  << " stages=" << stages << " ok=" << ok << std::endl;
    }
}

uint32_t
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

uint32_t
CeilDivU32(uint32_t a, uint32_t b)
{
    return (a + b - 1u) / b;
}

int
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


void
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

uint64_t
Add64WithCarry(uint64_t a, uint64_t b, uint64_t& carry)
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

    // r = u / 2^64; ensure r < P (include the high-limb carry-out)
    uint64_t r = u_hi;
    if (carry1 || r >= SharkNTT::P)
        r -= SharkNTT::P;

    return r;
}

template <class SharkFloatParams>
uint64_t
MontgomeryMul(DebugHostCombo<SharkFloatParams>& debugCombo, uint64_t a, uint64_t b)
{
    // We'll count 128-bit multiplications here as 3x64  (so 3 + 3 + 1)
    debugCombo.MultiplyCounts.DebugMultiplyIncrement(7);

    return MontgomeryMul<SharkFloatParams>(a, b);
}


template <class SharkFloatParams>
uint64_t
ToMontgomery(DebugHostCombo<SharkFloatParams>& debugCombo, uint64_t x)
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
FromMontgomery(DebugHostCombo<SharkFloatParams>& debugCombo, uint64_t x)
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
MontgomeryPow(DebugHostCombo<SharkFloatParams>& debugCombo, uint64_t a_mont, uint64_t e)
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

PlanPrime
BuildPlanPrime(int n32, int b_hint, int margin)
{
    using namespace SharkNTT;
    PlanPrime plan{};
    plan.ok = false;
    plan.n32 = n32;

    const uint64_t totalBits = (uint64_t)n32 * 32ull;

    int b = b_hint;
    if (b < 16)
        b = 16;
    if (b > 30)
        b = 30;

    uint32_t L = CeilDivU32((uint32_t)totalBits, (uint32_t)b);
    uint32_t N = NextPow2U32(2u * L);
    int lgN = CeilLog2U32(N);

    // Enough headroom for negacyclic wrap-free convolution (b + b + lgN + margin <= 64)
    int bmax = (64 - margin - lgN) / 2;
    if (b > bmax)
        b = bmax;
    if (b < 16)
        b = 16;

    L = CeilDivU32((uint32_t)totalBits, (uint32_t)b);
    N = NextPow2U32(2u * L);
    lgN = CeilLog2U32(N);

    // Ensure 2N | (p-1) to admit ψ of order 2N
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

    int need = 2 * b + lgN + margin;
    plan.ok = (need <= 64) && (N >= 2);
    plan.b = b;
    plan.L = (int)L;
    plan.N = (int)N;
    plan.stages = lgN;
    return plan;
}

template <class SharkFloatParams>
void
BuildRoots(uint32_t N, uint32_t stages, RootTables& roots)
{
    roots.stages = stages;
    roots.N = N;

    roots.stage_omegas = new uint64_t[stages];
    roots.stage_omegas_inv = new uint64_t[stages];

    roots.psi_pows = new uint64_t[N];
    roots.psi_inv_pows = new uint64_t[N];

    const uint64_t generator = SharkNTT::FindGeneratorConstexpr();
    const uint64_t g_m = ToMontgomery<SharkFloatParams>(generator);
    const uint64_t exponent = PHI / (2ull * N);
    const uint64_t psi_m = MontgomeryPow<SharkFloatParams>(g_m, exponent);

    // Test ψ^(2N) = 1 before proceeding
    uint64_t psi_test = MontgomeryPow<SharkFloatParams>(psi_m, 2ull * (uint64_t)N);
    uint64_t one_m = ToMontgomery<SharkFloatParams>(1);

    const uint64_t psi_inv_m = MontgomeryPow<SharkFloatParams>(psi_m, PHI - 1ull);
    const uint64_t omega_m = MontgomeryMul<SharkFloatParams>(psi_m, psi_m); // ω = ψ^2 (Montgomery)
    const uint64_t omega_inv_m = MontgomeryPow<SharkFloatParams>(omega_m, PHI - 1ull);

    for (uint32_t s = 1; s <= stages; ++s) {
        uint32_t m = 1u << s;
        uint64_t e = (uint64_t)N / (uint64_t)m;
        roots.stage_omegas[s - 1] = MontgomeryPow<SharkFloatParams>(omega_m, e);
        roots.stage_omegas_inv[s - 1] = MontgomeryPow<SharkFloatParams>(omega_inv_m, e);
    }

    roots.psi_pows[0] = ToMontgomery<SharkFloatParams>(1);
    roots.psi_inv_pows[0] = ToMontgomery<SharkFloatParams>(1);
    for (uint32_t i = 1; i < N; ++i) {
        roots.psi_pows[i] = MontgomeryMul<SharkFloatParams>(roots.psi_pows[i - 1], psi_m);
        roots.psi_inv_pows[i] = MontgomeryMul<SharkFloatParams>(roots.psi_inv_pows[i - 1], psi_inv_m);
    }

    // N^{-1} in Montgomery form: (1/2)^stages
    uint64_t inv2_m = ToMontgomery<SharkFloatParams>((P + 1) >> 1); // (p+1)/2
    uint64_t Ninvm = ToMontgomery<SharkFloatParams>(1);
    for (uint32_t i = 0; i < stages; ++i)
        Ninvm = MontgomeryMul<SharkFloatParams>(Ninvm, inv2_m);
    roots.Ninvm_mont = Ninvm;
}

template <class SharkFloatParams>
void
CopyRootsToCuda(RootTables& outT, const RootTables& inT)
{
    // Shallow copy first, pointers will be wrong
    cudaMemcpy(&outT, &inT, sizeof(RootTables), cudaMemcpyHostToDevice);

    // Fix the pointers:
    size_t stage_bytes = (size_t)inT.stages * sizeof(uint64_t);
    uint64_t* stage_omegas_ptr;
    cudaMalloc(&stage_omegas_ptr, stage_bytes);
    cudaMemcpy(stage_omegas_ptr, inT.stage_omegas, stage_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(&outT.stage_omegas, &stage_omegas_ptr, sizeof(uint64_t*), cudaMemcpyHostToDevice);

    uint64_t* stage_omegas_inv_ptr;
    cudaMalloc(&stage_omegas_inv_ptr, stage_bytes);
    cudaMemcpy(stage_omegas_inv_ptr, inT.stage_omegas_inv, stage_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(&outT.stage_omegas_inv, &stage_omegas_inv_ptr, sizeof(uint64_t*), cudaMemcpyHostToDevice);

    uint64_t* psi_pows_ptr;
    size_t N_bytes = (size_t)inT.N * sizeof(uint64_t);
    cudaMalloc(&psi_pows_ptr, N_bytes);
    cudaMemcpy(psi_pows_ptr, inT.psi_pows, N_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(&outT.psi_pows, &psi_pows_ptr, sizeof(uint64_t*), cudaMemcpyHostToDevice);

    uint64_t* psi_inv_pows_ptr;
    cudaMalloc(&psi_inv_pows_ptr, N_bytes);
    cudaMemcpy(psi_inv_pows_ptr, inT.psi_inv_pows, N_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(&outT.psi_inv_pows, &psi_inv_pows_ptr, sizeof(uint64_t*), cudaMemcpyHostToDevice);

    cudaMemcpy(&outT.Ninvm_mont, &inT.Ninvm_mont, sizeof(uint64_t), cudaMemcpyHostToDevice);
}

template <class SharkFloatParams>
void
DestroyRoots(bool cuda, RootTables& roots)
{
    if (cuda == false) {
        delete[] roots.stage_omegas;
        delete[] roots.stage_omegas_inv;
        delete[] roots.psi_pows;
        delete[] roots.psi_inv_pows;

        roots.stage_omegas = nullptr;
        roots.stage_omegas_inv = nullptr;
        roots.psi_pows = nullptr;
        roots.psi_inv_pows = nullptr;

        roots.stages = 0;
        roots.N = 0;
        roots.Ninvm_mont = 0;
    } else {
        RootTables localRoots;
        cudaMemcpy(&localRoots, &roots, sizeof(RootTables), cudaMemcpyDeviceToHost);

        cudaFree(localRoots.stage_omegas);
        cudaFree(localRoots.stage_omegas_inv);
        cudaFree(localRoots.psi_pows);
        cudaFree(localRoots.psi_inv_pows);

        cudaMemset(&roots, 0, sizeof(RootTables));
    }
}

} // namespace SharkNTT

#define ExplicitlyInstantiate(SharkFloatParams)                                                         \
    template void SharkNTT::BuildRoots<SharkFloatParams>(uint32_t, uint32_t, SharkNTT::RootTables&);             \
    template uint64_t SharkNTT::MontgomeryMul<SharkFloatParams>(uint64_t a, uint64_t b); \
    template uint64_t SharkNTT::MontgomeryMul<SharkFloatParams>(                                        \
        DebugHostCombo<SharkFloatParams> & debugCombo, uint64_t a, uint64_t b); \
    template uint64_t SharkNTT::ToMontgomery<SharkFloatParams>(uint64_t x);   \
    template uint64_t SharkNTT::ToMontgomery<SharkFloatParams>(                                        \
        DebugHostCombo<SharkFloatParams> & debugCombo, uint64_t x); \
    template uint64_t SharkNTT::FromMontgomery<SharkFloatParams>(uint64_t x); \
    template uint64_t SharkNTT::FromMontgomery<SharkFloatParams>(                                      \
        DebugHostCombo<SharkFloatParams> & debugCombo, uint64_t x); \
    template uint64_t SharkNTT::MontgomeryPow<SharkFloatParams>(uint64_t a_mont, uint64_t e); \
    template uint64_t SharkNTT::MontgomeryPow<SharkFloatParams>(                                      \
        DebugHostCombo<SharkFloatParams> & debugCombo, uint64_t a_mont, uint64_t e); \
    template void SharkNTT::CopyRootsToCuda<SharkFloatParams>(SharkNTT::RootTables & outT,              \
                                                              const SharkNTT::RootTables& inT); \
    template void SharkNTT::DestroyRoots<SharkFloatParams>(bool cuda, SharkNTT::RootTables& T);

ExplicitInstantiateAll();
