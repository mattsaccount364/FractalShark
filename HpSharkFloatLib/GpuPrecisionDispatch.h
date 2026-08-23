#pragma once

// Unified GPU precision dispatch for all SharkFloatParams families.
//
// DispatchByLimbCount maps indices 1–12 to the concrete SharkFloatParams types
// supplied by each caller and selects the correct specialization at runtime
// based on a power-of-2 limb count.

#include "Exceptions.h"
#include "HpSharkFloat.h"

#include <cstdint>

// Round a raw limb count up to the nearest supported power-of-2 in [256, 524288].
inline uint32_t
RoundToSupportedLimbCount(uint64_t rawLimbs)
{
    uint32_t p = 256;
    while (p < rawLimbs && p < 524288) {
        p <<= 1;
    }
    return p;
}

// Convert a precision in bits to a supported limb count.
inline uint32_t
BitsToSupportedLimbCount(uint64_t precBits)
{
    uint64_t rawLimbs = (precBits + 31) / 32;
    return RoundToSupportedLimbCount(rawLimbs);
}

inline bool
IsSupportedLimbCount(uint32_t limbCount)
{
    switch (limbCount) {
        case 256:
        case 512:
        case 1024:
        case 2048:
        case 4096:
        case 8192:
        case 16384:
        case 32768:
        case 65536:
        case 131072:
        case 262144:
        case 524288:
            return true;
        default:
            return false;
    }
}

// The GPU reference kernel consumes a precision window inside the selected storage bucket. Its
// setup requires that window to cover more than half of the bucket, while the
// input conversion cannot provide more limbs than the bucket stores.
inline uint32_t
GetReferenceEffectivePrecisionLimbs(uint64_t requestedLimbs, uint32_t storageLimbs)
{
    const uint64_t minimumLimbs = static_cast<uint64_t>(storageLimbs) / 2u + 1u;
    const uint64_t atLeastMinimum = requestedLimbs < minimumLimbs ? minimumLimbs : requestedLimbs;
    const uint64_t atMostStorage = atLeastMinimum > storageLimbs ? storageLimbs : atLeastMinimum;
    return static_cast<uint32_t>(atMostStorage);
}

// Dispatch a callback f.template operator()<ParamsType>() based on limb count.
// limbCount must be a power-of-2 in [256, 524288] (use RoundToSupportedLimbCount first).
template <class P1,
          class P2,
          class P3,
          class P4,
          class P5,
          class P6,
          class P7,
          class P8,
          class P9,
          class P10,
          class P11,
          class P12,
          class F>
void
DispatchByLimbCount(uint32_t limbCount, F &&f)
{
    switch (limbCount) {
        case 256:
            f.template operator()<P1>();
            break;
        case 512:
            f.template operator()<P2>();
            break;
        case 1024:
            f.template operator()<P3>();
            break;
        case 2048:
            f.template operator()<P4>();
            break;
        case 4096:
            f.template operator()<P5>();
            break;
        case 8192:
            f.template operator()<P6>();
            break;
        case 16384:
            f.template operator()<P7>();
            break;
        case 32768:
            f.template operator()<P8>();
            break;
        case 65536:
            f.template operator()<P9>();
            break;
        case 131072:
            f.template operator()<P10>();
            break;
        case 262144:
            f.template operator()<P11>();
            break;
        case 524288:
            f.template operator()<P12>();
            break;
        default:
            throw FractalSharkSeriousException("Unsupported limb count for GPU dispatch");
    }
}
