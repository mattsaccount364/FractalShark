#pragma once

#include <stdint.h>

#include "MultiplyNTTPlanBuilder.cuh"

template <class SharkFloatParams> struct DebugGlobalCount;

template <class SharkFloatParams> struct DebugHostCombo;

namespace SharkNTT {

struct RootTables {
    int32_t stages;
    uint64_t *stage_omegas;     // [stages], Montgomery domain
    uint64_t *stage_omegas_inv; // [stages], Montgomery domain

    int32_t N;
    uint64_t *psi_pows;     // [N],  psi^i in Montgomery domain
    uint64_t *psi_inv_pows; // [N], (psi^{-1})^i in Montgomery
    uint64_t Ninvm_mont;    // N^{-1} in Montgomery form

    // Flat table of per-stage twiddles in Montgomery domain:
    // For stage s (1-based), indices j = 0 .. (2^(s-1) - 1) live at:
    //   stage_twiddles[ stage_twiddle_offset[s] + j ]
    uint64_t *stage_twiddles_fwd; // built from stage_omegas
    uint64_t *stage_twiddles_inv; // built from stage_omegas_inv

    // Total number of twiddle entries, for convenience
    uint32_t total_twiddles;
};


template <class SharkFloatParams> uint64_t MontgomeryMul(uint64_t a, uint64_t b);

template <class SharkFloatParams>
uint64_t MontgomeryMul(DebugHostCombo<SharkFloatParams> &debugCombo, uint64_t a, uint64_t b);

template <class SharkFloatParams>
uint64_t ToMontgomery(DebugHostCombo<SharkFloatParams> &debugCombo, uint64_t x);

template <class SharkFloatParams> uint64_t ToMontgomery(uint64_t x);

template <class SharkFloatParams>
uint64_t FromMontgomery(DebugHostCombo<SharkFloatParams> &debugCombo, uint64_t x);

template <class SharkFloatParams> uint64_t FromMontgomery(uint64_t x);

template <class SharkFloatParams>
uint64_t MontgomeryPow(DebugHostCombo<SharkFloatParams> &debugCombo, uint64_t a_mont, uint64_t e);

template <class SharkFloatParams> uint64_t MontgomeryPow(uint64_t a_mont, uint64_t e);

template <class SharkFloatParams> void BuildRoots(uint32_t N, uint32_t stages, RootTables &T);

template <class SharkFloatParams> void CopyRootsToCuda(RootTables &outT, const RootTables &inT);

template <class SharkFloatParams> void DestroyRoots(bool cuda, RootTables &T);

} // namespace SharkNTT
