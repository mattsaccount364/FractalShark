#include <cstdio>
#include <cuda_runtime.h>
#include "BenchmarkTimer.h"

#include <iostream>

__global__ void trivial_kernel(int *output) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    output[idx] = idx;
}

void TestNullKernel() {
    const int N = 256;
    const auto NumIterations = 100000;
    int *d_output;
    cudaError_t err = cudaMalloc(&d_output, N * sizeof(int));
    if (err != cudaSuccess) {
        std::cerr << "CUDA error in cudaMalloc: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    // Warm-up kernel to mitigate startup overhead
    trivial_kernel << <1, N >> > (d_output);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error in cudaDeviceSynchronize (warm-up): " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_output);
        return;
    }

    BenchmarkTimer timer;
    {
        ScopedBenchmarkStopper stopper(timer);

        for (int i = 0; i < NumIterations; ++i) {

            // Launch the trivial kernel
            trivial_kernel << <1, N >> > (d_output);

            // Ensure the kernel has completed
            err = cudaDeviceSynchronize();
            if (err != cudaSuccess) {
                std::cerr << "CUDA error in cudaDeviceSynchronize (iteration " << i << "): " << cudaGetErrorString(err) << std::endl;
                break;
            }
        }
    }

    uint64_t elapsed_ms = timer.GetDeltaInMs();
    std::cout << "Null kernel elapsed time: " << elapsed_ms << " ms" << std::endl;

    err = cudaFree(d_output);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error in cudaFree: " << cudaGetErrorString(err) << std::endl;
    }
}
