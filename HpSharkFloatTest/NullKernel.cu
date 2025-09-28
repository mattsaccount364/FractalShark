#include "BenchmarkTimer.h"
#include <cstdio>
#include <cuda_runtime.h>

#include <iostream>

__global__ void
trivial_kernel(int *output)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    output[idx] = idx;
}

void
TestNullKernel()
{
    const int N = 256;
    const auto NumIterations = 100000;
    int *d_output;
    cudaMalloc(&d_output, N * sizeof(int));

    // Warm-up kernel to mitigate startup overhead
    trivial_kernel<<<1, N>>>(d_output);
    cudaDeviceSynchronize();

    BenchmarkTimer timer;
    {
        ScopedBenchmarkStopper stopper(timer);

        for (int i = 0; i < NumIterations; ++i) {

            // Launch the trivial kernel
            trivial_kernel<<<1, N>>>(d_output);

            // Ensure the kernel has completed
            cudaDeviceSynchronize();
        }
    }

    uint64_t elapsed_ms = timer.GetDeltaInMs();
    std::cout << "Null kernel elapsed time: " << elapsed_ms << " ms" << std::endl;

    cudaFree(d_output);
}
