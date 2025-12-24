#ifndef __GDD_TYPE_H__
#define __GDD_TYPE_H__

#include <vector_types.h>


/* compiler switch */
/**
 * ALL_MATH will include advanced math functions, including
 * atan, acos, asin, sinh, cosh, tanh, asinh, acosh, atanh
 * WARNING: these functions take long time to compile, 
 * e.g., several hours
 * */
//#define ALL_MATH


/* type definition */

namespace GQD {
	typedef double4_16a gqd_real;
}

namespace GQF {
	typedef float4 gqf_real;
}


#endif /*__GDD_GQD_TYPE_H__*/
