#pragma once

#include "NTTConstexprGenerator.h"
#include <algorithm>
#include <stdint.h>

namespace SharkNTT {

struct PlanPrime {
    int n32 = 0;    // number of 32-bit digits in HpSharkFloat<P>
    int b = 0;      // base = 2^b (packing chunk size in bits)
    int L = 0;      // number of packed coefficients (ceil(totalBits / b))
    int N = 0;      // transform size
    int stages = 0; // log2(N)
    bool ok = false;
};

constexpr uint32_t
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

// headers (ensure these are constexpr)
constexpr uint32_t
CeilDivU32(uint32_t a, uint32_t b)
{
    return (a + b - 1u) / b;
}

constexpr uint32_t
CeilLog2U32(uint32_t x)
{
    uint32_t r = 0, v = x ? x - 1u : 0u;
    while (v) {
        v >>= 1u;
        ++r;
    }
    return r;
}

constexpr PlanPrime
BuildPlanPrime(int n32, int b_hint, int margin)
{
    PlanPrime plan{};
    plan.ok = false;
    plan.n32 = n32;

    const uint64_t totalBits = static_cast<uint64_t>(n32) * 32ull;

    // 1) clamp b to [16, 30]
    int b = b_hint;
    if (b < 16)
        b = 16;
    if (b > 30)
        b = 30;

    // 2) initial L, N=NextPow2(2*L), lgN
    uint32_t L = CeilDivU32(static_cast<uint32_t>(totalBits), static_cast<uint32_t>(b));
    uint32_t N = NextPow2U32(2u * L); // *** matches original ***
    uint32_t lgN = CeilLog2U32(N);

    // 3) headroom: bmax = floor((64 - margin - lgN)/2), cap b, keep >=16
    //    (guard if 64 - margin - lgN < 0)
    int headroom = 64 - margin - static_cast<int>(lgN);
    int bmax = (headroom > 0) ? (headroom / 2) : 0;
    if (b > bmax)
        b = bmax;
    if (b < 16)
        b = 16;

    // 4) recompute with possibly new b
    L = CeilDivU32(static_cast<uint32_t>(totalBits), static_cast<uint32_t>(b));
    N = NextPow2U32(2u * L);
    lgN = CeilLog2U32(N);

    // 5) ensure 2N | (p-1), else try b_up
    if ((PHI % (2ull * static_cast<uint64_t>(N))) != 0ull) {
        const int b_up_max = std::min(bmax, 30);
        for (int b_try = b + 1; b_try <= b_up_max; ++b_try) {
            const uint32_t L2 =
                CeilDivU32(static_cast<uint32_t>(totalBits), static_cast<uint32_t>(b_try));
            const uint32_t N2 = NextPow2U32(2u * L2);
            if ((PHI % (2ull * static_cast<uint64_t>(N2))) == 0ull) {
                b = b_try;
                L = L2;
                N = N2;
                lgN = CeilLog2U32(N2);
                break;
            }
        }
    }

    // 6) final accept/fields
    const int need = 2 * b + static_cast<int>(lgN) + margin;
    plan.ok = (need <= 64) && (N >= 2u);
    plan.b = b;
    plan.L = static_cast<int>(L);
    plan.N = static_cast<int>(N);
    plan.stages = static_cast<int>(lgN);
    return plan;
}

} // namespace SharkNTT
