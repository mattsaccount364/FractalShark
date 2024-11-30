#include "Add.cuh"
#include "Multiply.cuh"

#include "HpSharkFloat.cuh"

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda/barrier>

#include <iostream>

namespace cg = cooperative_groups;

template<class SharkFloatParams>
__global__ void IntegrationKernel(
    HpSharkFloat<SharkFloatParams> *A,
    HpSharkFloat<SharkFloatParams> *B,
    HpSharkFloat<SharkFloatParams> *Out,
    GlobalAddBlockData *globalBlockData,
    CarryInfo *carryOuts,        // Array to store carry-out for each block
    uint32_t *cumulativeCarries) { // Array to store cumulative carries

    // Initialize cooperative grid group
    cg::grid_group grid = cg::this_grid();

    // Total number of blocks launched
    int numBlocks = gridDim.x;

    // Call the AddHelper function
    AddHelper(A, B, Out, globalBlockData, carryOuts, cumulativeCarries, grid, numBlocks);
}


template<class SharkFloatParams>
__global__ void IntegrationKernelTestLoop(
    HpSharkFloat<SharkFloatParams> *A,
    HpSharkFloat<SharkFloatParams> *B,
    HpSharkFloat<SharkFloatParams> *Out,
    GlobalAddBlockData *globalBlockData,
    CarryInfo *carryOuts,        // Array to store carry-out for each block
    uint32_t *cumulativeCarries) { // Array to store cumulative carries

    // Initialize cooperative grid group
    cg::grid_group grid = cg::this_grid();

    // Total number of blocks launched
    int numBlocks = gridDim.x;

    for (int i = 0; i < TestIterCount; ++i) {
        AddHelper(A, B, Out, globalBlockData, carryOuts, cumulativeCarries, grid, numBlocks);
    }
}


template<class SharkFloatParams>
void ComputeIntegrationGpu(void *kernelArgs[]) {

    cudaError_t err = cudaLaunchCooperativeKernel(
        (void *)IntegrationKernel<SharkFloatParams>,
        dim3(SharkFloatParams::NumBlocks),
        dim3(SharkFloatParams::ThreadsPerBlock),
        kernelArgs,
        0, // Shared memory size
        0 // Stream
    );

    cudaDeviceSynchronize();

    if (err != cudaSuccess) {
        std::cerr << "CUDA error in ComputeIntegrationGpu: " << cudaGetErrorString(err) << std::endl;
    }
}

template<class SharkFloatParams>
void ComputeIntegrationTestLoop(void *kernelArgs[]) {

    cudaError_t err = cudaLaunchCooperativeKernel(
        (void *)IntegrationKernelTestLoop<SharkFloatParams>,
        dim3(SharkFloatParams::NumBlocks),
        dim3(SharkFloatParams::ThreadsPerBlock),
        kernelArgs,
        0, // Shared memory size
        0 // Stream
    );

    cudaDeviceSynchronize();

    if (err != cudaSuccess) {
        std::cerr << "CUDA error in ComputeIntegrationTestLoop: " << cudaGetErrorString(err) << std::endl;
    }
}
