#pragma once

__device__ uint64_t
AtomicMax(uint64_t *address, uint64_t val)
{
    unsigned long long *ull_address = reinterpret_cast<unsigned long long *>(address);
    unsigned long long old = *ull_address;
    unsigned long long assumed;
    while (val > old) {
        assumed = old;
        old = atomicCAS(ull_address, assumed, static_cast<unsigned long long>(val));
    }
    return old;
}

__device__ uint64_t
AtomicMin(uint64_t *address, uint64_t val)
{
    unsigned long long *ull_address = reinterpret_cast<unsigned long long *>(address);
    unsigned long long old = *ull_address;
    unsigned long long assumed;
    while (val < old) {
        assumed = old;
        old = atomicCAS(ull_address, assumed, static_cast<unsigned long long>(val));
    }
    return old;
}

__device__ uint64_t
AtomicSum(uint64_t *address, uint64_t val)
{
    unsigned long long *ull_address = reinterpret_cast<unsigned long long *>(address);
    unsigned long long old;
    unsigned long long assumed;
    for (;;) {
        old = *ull_address;
        assumed = old;
        auto temp = old + static_cast<unsigned long long>(val);
        old = atomicCAS(ull_address, assumed, temp);
        if (old == assumed) {
            break;
        }
    }
    return old;
}

template <typename IterType>
__global__ void
max_kernel(const IterType *__restrict__ OutputIterMatrix,
           uint32_t WidthWithAA,
           uint32_t HeightWithAA,
           ReductionResults *Output)
{

    auto GetIndex = [](size_t X, size_t Y, size_t OriginalWidth) -> size_t {
        auto RoundedBlocks =
            OriginalWidth / GPURenderer::NB_THREADS_W + (OriginalWidth % GPURenderer::NB_THREADS_W != 0);
        auto RoundedWidth = RoundedBlocks * GPURenderer::NB_THREADS_W;
        return Y * RoundedWidth + X;
    };

    __shared__ uint64_t MinShared[128];
    __shared__ uint64_t MaxShared[128];
    __shared__ uint64_t SumShared[128];

    int tid = blockDim.x * threadIdx.y + threadIdx.x;

    const int gidX = blockIdx.x * blockDim.x + threadIdx.x;
    const int gidY = blockIdx.y * blockDim.y + threadIdx.y;

    if constexpr (sizeof(IterType) == sizeof(uint64_t)) {
        MaxShared[tid] = 0;
        MinShared[tid] = std::numeric_limits<uint64_t>::max();
        SumShared[tid] = 0;
    } else {
        MaxShared[tid] = 0;
        MinShared[tid] = std::numeric_limits<uint32_t>::max();
        ;
        SumShared[tid] = 0;
    }

    __syncthreads();

    for (size_t inputX = gidX; inputX < WidthWithAA; inputX += gridDim.x * blockDim.x) {
        for (size_t inputY = gidY; inputY < HeightWithAA; inputY += gridDim.y * blockDim.y) {

            size_t idx = ConvertLocToIndex(inputX, inputY, WidthWithAA);
            uint64_t tempIters = OutputIterMatrix[idx];
            MaxShared[tid] = max(MaxShared[tid], tempIters);
            MinShared[tid] = min(MinShared[tid], tempIters);
            SumShared[tid] = SumShared[tid] + tempIters;
        }
    }

    __syncthreads();
    for (auto s = blockDim.x * blockDim.y / 2; s > 0; s >>= 1) {
        if (tid < s /* && gid < WidthWithAA * HeightWithAA*/) {
            MaxShared[tid] = max(MaxShared[tid], MaxShared[tid + s]);
            MinShared[tid] = min(MinShared[tid], MinShared[tid + s]);
            SumShared[tid] = SumShared[tid] + SumShared[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        AtomicMax(&Output->Max, MaxShared[0]);
        AtomicMin(&Output->Min, MinShared[0]);
        AtomicSum(&Output->Sum, SumShared[0]);
    }
}
