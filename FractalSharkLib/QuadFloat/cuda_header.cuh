
#ifndef _CUDA_HEADER_CU_
#define _CUDA_HEADER_CU_

#include <stdio.h>
#include <stdlib.h>
//#include <cutil_inline.h>


//#define CUT_CHECK_ERROR(x) x
//#define CUDA_SAFE_CALL(x) x
//
//#define cutilCheckMsg CUT_CHECK_ERROR
//#define cutilSafeCall CUDA_SAFE_CALL
//
///* kernel macros */
//#define NUM_TOTAL_THREAD (gridDim.x*blockDim.x)
//#define GLOBAL_THREAD_OFFSET (blockDim.x*blockIdx.x + threadIdx.x)
//
///** macro utility */
//#define GPUMALLOC(D_DATA, MEM_SIZE) cutilSafeCall(cudaMalloc(D_DATA, MEM_SIZE))
//#define TOGPU(D_DATA, H_DATA, MEM_SIZE) cutilSafeCall(cudaMemcpy(D_DATA, H_DATA, MEM_SIZE, cudaMemcpyHostToDevice))
//#define FROMGPU( H_DATA, D_DATA, MEM_SIZE ) cutilSafeCall(cudaMemcpy( H_DATA, D_DATA, MEM_SIZE, cudaMemcpyDeviceToHost))
//#define GPUTOGPU( DEST, SRC, MEM_SIZE ) cutilSafeCall(cudaMemcpy( DEST, SRC, MEM_SIZE, cudaMemcpyDeviceToDevice ))
//#define GPUFREE( MEM ) cutilSafeCall(cudaFree(MEM));
//


#endif

