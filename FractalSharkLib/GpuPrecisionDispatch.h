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
