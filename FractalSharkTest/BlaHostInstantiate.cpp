// Host-side compilation of BLA template definitions and explicit
// instantiations. On Windows these live in FractalSharkGpuLib (CUDA). The
// definitions in BLA.cuh compile cleanly as plain C++ because CUDA_CRAP
// expands to nothing on non-CUDA builds.

#include "BLA.h"

#define BLA_CUH_HOST_ONLY
#include "BLA.cuh"
#undef BLA_CUH_HOST_ONLY
