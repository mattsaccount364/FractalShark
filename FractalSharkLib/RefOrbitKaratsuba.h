#pragma once

#include "HighPrecision.h"

#include <memory>

namespace FractalShark {

// This threshold selects the backend in Auto mode; explicit selection is unrestricted.
inline constexpr uint64_t RefOrbitKaratsubaAutoPrecisionBits = 16'384;

// Computes three products at full destination precision. Outputs must be distinct, but may
// alias either input. Full stored input mantissas, including guard limbs, are snapshotted.
// The fixed precision applies to all outputs and bounds input sizes. The caller handles
// inputs shorter than two limbs. All scratch is sized once at construction and reused.
class RefOrbitKaratsubaProducts {
public:
    explicit RefOrbitKaratsubaProducts(mp_bitcnt_t precisionBits);
    ~RefOrbitKaratsubaProducts();
    void Compute(mpf_ptr xSquared, mpf_ptr ySquared, mpf_ptr cross, mpf_srcptr x, mpf_srcptr y);

private:
    class Impl;
    std::unique_ptr<Impl> m_Impl;
};

} // namespace FractalShark
