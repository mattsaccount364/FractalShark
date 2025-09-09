#pragma once

#include <stdint.h>

template <class SharkFloatParams>
struct DebugMultiplyCount;

template <class SharkFloatParams> struct DebugHostCombo;

namespace SharkNTT {

struct PlanPrime {
    int n32 = 0;    // number of 32-bit digits in HpSharkFloat<P>
    int b = 0;      // base = 2^b (packing chunk size in bits)
    int L = 0;      // number of packed coefficients (ceil(totalBits / b))
    int N = 0;      // transform size
    int stages = 0; // log2(N)
    bool ok = false;

    void Print();
};

struct RootTables {
    int32_t stages;
    uint64_t* stage_omegas;     // [stages]
    uint64_t* stage_omegas_inv; // [stages]

    int32_t N;
    uint64_t* psi_pows;     // [N]
    uint64_t* psi_inv_pows; // [N]
    uint64_t Ninvm_mont;    // N^{-1} in Montgomery form
};

uint32_t NextPow2U32(uint32_t x);

uint32_t CeilDivU32(uint32_t a, uint32_t b);

int CeilLog2U32(uint32_t x);

void Mul64Wide(uint64_t a, uint64_t b, uint64_t& lo, uint64_t& hi);

uint64_t Add64WithCarry(uint64_t a, uint64_t b, uint64_t& carry);

template <class SharkFloatParams> uint64_t MontgomeryMul(uint64_t a, uint64_t b);

template <class SharkFloatParams>
uint64_t MontgomeryMul(DebugHostCombo<SharkFloatParams>& debugCombo, uint64_t a, uint64_t b);

template <class SharkFloatParams>
uint64_t ToMontgomery(DebugHostCombo<SharkFloatParams>& debugCombo, uint64_t x);

template <class SharkFloatParams> uint64_t ToMontgomery(uint64_t x);

template <class SharkFloatParams>
uint64_t FromMontgomery(DebugHostCombo<SharkFloatParams>& debugCombo, uint64_t x);

template <class SharkFloatParams> uint64_t FromMontgomery(uint64_t x);

template <class SharkFloatParams>
uint64_t MontgomeryPow(DebugHostCombo<SharkFloatParams>& debugCombo, uint64_t a_mont, uint64_t e);

template <class SharkFloatParams> uint64_t MontgomeryPow(uint64_t a_mont, uint64_t e);

PlanPrime BuildPlanPrime(int n32, int b_hint = 26, int margin = 2);

template <class SharkFloatParams>
void BuildRoots(uint32_t N,
                uint32_t stages,
                RootTables& T);

template <class SharkFloatParams> void CopyRootsToCuda(RootTables& outT, const RootTables& inT); 

template <class SharkFloatParams> void DestroyRoots(bool cuda, RootTables& T);


} // namespace SharkNTT