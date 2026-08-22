#pragma once

// Unified GPU precision dispatch for all SharkFloatParams families.
//
// Each family struct maps indices 1–12 to a concrete SharkFloatParams alias.
// DispatchByLimbCount selects the correct specialization at runtime based on
// a power-of-2 limb count.

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

// Production reference orbit family: periodicity enabled, SubType=float.
struct SharkParamsBaseFamily {
    using P1 = SharkParams1;
    using P2 = SharkParams2;
    using P3 = SharkParams3;
    using P4 = SharkParams4;
    using P5 = SharkParams5;
    using P6 = SharkParams6;
    using P7 = SharkParams7;
    using P8 = SharkParams8;
    using P9 = SharkParams9;
    using P10 = SharkParams10;
    using P11 = SharkParams11;
    using P12 = SharkParams12;
};

// Production reference orbit family: periodicity enabled, SubType=double.
struct SharkParamsDblFamily {
    using P1 = SharkParamsDbl1;
    using P2 = SharkParamsDbl2;
    using P3 = SharkParamsDbl3;
    using P4 = SharkParamsDbl4;
    using P5 = SharkParamsDbl5;
    using P6 = SharkParamsDbl6;
    using P7 = SharkParamsDbl7;
    using P8 = SharkParamsDbl8;
    using P9 = SharkParamsDbl9;
    using P10 = SharkParamsDbl10;
    using P11 = SharkParamsDbl11;
    using P12 = SharkParamsDbl12;
};

// Production reference orbit family: periodicity enabled, SubType=CudaDblflt<dblflt>.
struct SharkParamsDbfFamily {
    using P1 = SharkParamsDbf1;
    using P2 = SharkParamsDbf2;
    using P3 = SharkParamsDbf3;
    using P4 = SharkParamsDbf4;
    using P5 = SharkParamsDbf5;
    using P6 = SharkParamsDbf6;
    using P7 = SharkParamsDbf7;
    using P8 = SharkParamsDbf8;
    using P9 = SharkParamsDbf9;
    using P10 = SharkParamsDbf10;
    using P11 = SharkParamsDbf11;
    using P12 = SharkParamsDbf12;
};

// Newton-Raphson family: NR derivative tracking, no periodicity.
struct SharkParamsNRFamily {
    using P1 = SharkParamsNR1;
    using P2 = SharkParamsNR2;
    using P3 = SharkParamsNR3;
    using P4 = SharkParamsNR4;
    using P5 = SharkParamsNR5;
    using P6 = SharkParamsNR6;
    using P7 = SharkParamsNR7;
    using P8 = SharkParamsNR8;
    using P9 = SharkParamsNR9;
    using P10 = SharkParamsNR10;
    using P11 = SharkParamsNR11;
    using P12 = SharkParamsNR12;
};

// FractalShark's production dispatch uses the shared-only kernel for the three
// buckets whose low-limb working set fits the 96 KiB block budget.  The higher
// buckets deliberately retain the existing global-backed aliases.
struct SharkParamsFractalBaseFamily {
    using P1 = SharkParamsSharedOnly256;
    using P2 = SharkParamsSharedOnly512;
    using P3 = SharkParamsSharedOnly1024;
    using P4 = SharkParams4;
    using P5 = SharkParams5;
    using P6 = SharkParams6;
    using P7 = SharkParams7;
    using P8 = SharkParams8;
    using P9 = SharkParams9;
    using P10 = SharkParams10;
    using P11 = SharkParams11;
    using P12 = SharkParams12;
};

struct SharkParamsFractalNRFamily {
    using P1 = SharkParamsNRSharedOnly256;
    using P2 = SharkParamsNRSharedOnly512;
    using P3 = SharkParamsNRSharedOnly1024;
    using P4 = SharkParamsNR4;
    using P5 = SharkParamsNR5;
    using P6 = SharkParamsNR6;
    using P7 = SharkParamsNR7;
    using P8 = SharkParamsNR8;
    using P9 = SharkParamsNR9;
    using P10 = SharkParamsNR10;
    using P11 = SharkParamsNR11;
    using P12 = SharkParamsNR12;
};

struct SharkParamsFractalDblFamily {
    using P1 = SharkParamsDblSharedOnly256;
    using P2 = SharkParamsDblSharedOnly512;
    using P3 = SharkParamsDblSharedOnly1024;
    using P4 = SharkParamsDbl4;
    using P5 = SharkParamsDbl5;
    using P6 = SharkParamsDbl6;
    using P7 = SharkParamsDbl7;
    using P8 = SharkParamsDbl8;
    using P9 = SharkParamsDbl9;
    using P10 = SharkParamsDbl10;
    using P11 = SharkParamsDbl11;
    using P12 = SharkParamsDbl12;
};

struct SharkParamsFractalDbfFamily {
    using P1 = SharkParamsDbfSharedOnly256;
    using P2 = SharkParamsDbfSharedOnly512;
    using P3 = SharkParamsDbfSharedOnly1024;
    using P4 = SharkParamsDbf4;
    using P5 = SharkParamsDbf5;
    using P6 = SharkParamsDbf6;
    using P7 = SharkParamsDbf7;
    using P8 = SharkParamsDbf8;
    using P9 = SharkParamsDbf9;
    using P10 = SharkParamsDbf10;
    using P11 = SharkParamsDbf11;
    using P12 = SharkParamsDbf12;
};

// Dispatch a callback f.template operator()<ParamsType>() based on limb count.
// limbCount must be a power-of-2 in [256, 524288] (use RoundToSupportedLimbCount first).
template <class Family, class F>
void
DispatchByLimbCount(uint32_t limbCount, F &&f)
{
    switch (limbCount) {
        case 256:
            f.template operator()<typename Family::P1>();
            break;
        case 512:
            f.template operator()<typename Family::P2>();
            break;
        case 1024:
            f.template operator()<typename Family::P3>();
            break;
        case 2048:
            f.template operator()<typename Family::P4>();
            break;
        case 4096:
            f.template operator()<typename Family::P5>();
            break;
        case 8192:
            f.template operator()<typename Family::P6>();
            break;
        case 16384:
            f.template operator()<typename Family::P7>();
            break;
        case 32768:
            f.template operator()<typename Family::P8>();
            break;
        case 65536:
            f.template operator()<typename Family::P9>();
            break;
        case 131072:
            f.template operator()<typename Family::P10>();
            break;
        case 262144:
            f.template operator()<typename Family::P11>();
            break;
        case 524288:
            f.template operator()<typename Family::P12>();
            break;
        default:
            throw FractalSharkSeriousException("Unsupported limb count for GPU dispatch");
    }
}
