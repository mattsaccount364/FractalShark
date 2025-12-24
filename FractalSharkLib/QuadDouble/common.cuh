#ifndef __GQD_COMMON_CU__
#define __GQD_COMMON_CU__

#include <stdio.h>
#include <stdlib.h>
#include "gqd_type.h"		//type definitions for gdd_real and gqd_real
#include "cuda_header.cuh"
#include "inline.cuh" 		//basic functions used by both gdd_real and gqd_real


/* type definitions, defined in the type.h */
//defined in gqd_type.h

namespace GQD {

    __host__ __device__ __forceinline__ gqd_real
    make_qd(double x, double y = 0.0, double z = 0.0, double w = 0.0)
    {
        gqd_real r;
        r.x = x;
        r.y = y;
        r.z = z;
        r.w = w;
        return r;
    }

}

#endif /* __GQD_COMMON_CU__ */


