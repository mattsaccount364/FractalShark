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

    /* type construction */
    __device__ __host__
        gqd_real make_qd(const double x,
            const double y,
            const double z,
            const double w) {
        return make_double4(x, y, z, w);
    }

    __device__ __host__
        gqd_real make_qd(const double x) {
        return make_qd(x, 0.0, 0.0, 0.0);
    }

}

#endif /* __GQD_COMMON_CU__ */


