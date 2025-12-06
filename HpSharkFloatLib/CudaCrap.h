#pragma once

#ifdef __CUDA_ARCH__
#define CUDA_CRAP __device__
#define CUDA_CRAP_BOTH __host__ __device__
#define CUDA_GLOBAL __global__
#else
#define CUDA_CRAP
#define CUDA_CRAP_BOTH
#define CUDA_GLOBAL
#endif
