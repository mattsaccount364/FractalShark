#include "GpuTestRuntime.h"

#include <cuda_runtime_api.h>

bool
TestFramework::CheckGpuRuntime(std::string &description, std::string &error)
{
    int count = 0;
    cudaError_t result = cudaGetDeviceCount(&count);
    if (result == cudaSuccess && count == 0) {
        error = "no CUDA device available";
        return false;
    }
    if (result == cudaSuccess) {
        result = cudaSetDevice(0);
    }
    if (result == cudaSuccess) {
        result = cudaFree(nullptr);
    }
    cudaDeviceProp properties{};
    if (result == cudaSuccess) {
        result = cudaGetDeviceProperties(&properties, 0);
    }
    if (result != cudaSuccess) {
        error = cudaGetErrorString(result);
        return false;
    }
    description = std::string{properties.name} + " sm_" + std::to_string(properties.major) +
                  std::to_string(properties.minor) + " runtime=" + std::to_string(CUDART_VERSION);
    return true;
}
