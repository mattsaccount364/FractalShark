#ifndef __GQF_COMMON_CU__
#define __GQF_COMMON_CU__

#include "../QuadDouble/gqd_type.h" //type definitions for gdd_real and gqf_real
#include "../QuadFloat/cuda_header.cuh"
#include "../QuadFloat/inline.cuh" //basic functions used by both gdd_real and gqf_real
#include <stdio.h>
#include <stdlib.h>

/* type definitions, defined in the type.h */
// defined in gqd_type.h

namespace GQF {

/* type construction */
__device__ __host__ gqf_real
make_qf(const float x, const float y, const float z, const float w)
{
    return make_float4(x, y, z, w);
}

__device__ __host__ gqf_real
make_qf(const float x)
{
    return make_qf(x, 0.0f, 0.0f, 0.0f);
}

} // namespace GQF

#endif /* __GQF_COMMON_CU__ */
