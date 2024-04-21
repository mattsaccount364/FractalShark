__device__ uint64_t AtomicMax(uint64_t *address, uint64_t val) {
    uint64_t old = *address;
    uint64_t assumed;
    while (val > old) {
        assumed = old;
        old = atomicCAS(address, assumed, val);
    }
    return old;
}

__device__ uint64_t AtomicMin(uint64_t *address, uint64_t val) {
    uint64_t old = *address;
    uint64_t assumed;
    while (val < old) {
        assumed = old;
        old = atomicCAS(address, assumed, val);
    }
    return old;
}

__device__ uint64_t AtomicSum(uint64_t *address, uint64_t val) {
    uint64_t old;
    uint64_t assumed;
    for (;;) {
        old = *address;
        assumed = old;
        auto temp = old + val;
        old = atomicCAS(address, assumed, temp);
        if (old == assumed) {
            break;
        }
    }
    return old;
}


#if 0
__global__ void max_reduce(const float *const d_array, float *d_max,
    const size_t elements) {
    extern __shared__ float shared[];

    int tid = threadIdx.x;
    int gid = (blockDim.x * blockIdx.x) + tid;
    shared[tid] = -FLOAT_MAX;

    while (gid < elements) {
        shared[tid] = max(shared[tid], d_array[gid]);
        gid += gridDim.x * blockDim.x;
    }
    __syncthreads();
    gid = (blockDim.x * blockIdx.x) + tid;  // 1
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && gid < elements)
            shared[tid] = max(shared[tid], shared[tid + s]);
        __syncthreads();
    }

    if (tid == 0)
        atomicMaxf(d_max, shared[0]);
}
#endif

template<typename IterType>
__global__
void
max_kernel(
    const IterType *__restrict__ OutputIterMatrix,
    uint32_t WidthWithAA,
    uint32_t HeightWithAA,
    ReductionResults *Output) {

    auto GetIndex = [](size_t X, size_t Y, size_t OriginalWidth) -> size_t {
        auto RoundedBlocks = OriginalWidth / GPURenderer::NB_THREADS_W + (OriginalWidth % GPURenderer::NB_THREADS_W != 0);
        auto RoundedWidth = RoundedBlocks * GPURenderer::NB_THREADS_W;
        return Y * RoundedWidth + X;
        };

    __shared__ uint64_t MinShared[128];
    __shared__ uint64_t MaxShared[128];
    __shared__ uint64_t SumShared[128];

    int tid = blockDim.x * threadIdx.y + threadIdx.x;

    const int gid_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int gid_y = blockIdx.y * blockDim.y + threadIdx.y;

    if constexpr (sizeof(IterType) == sizeof(uint64_t)) {
        MaxShared[tid] = 0;
        MinShared[tid] = std::numeric_limits<uint64_t>::max();
        SumShared[tid] = 0;
    } else {
        MaxShared[tid] = 0;
        MinShared[tid] = std::numeric_limits<uint32_t>::max();;
        SumShared[tid] = 0;
    }

    if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
        Output->Max = 0;
        Output->Min = MinShared[tid];
        Output->Sum = 0;
    }

    __syncthreads();

    for (size_t input_x = gid_x; input_x < WidthWithAA; input_x += gridDim.x * blockDim.x) {
        for (size_t input_y = gid_y; input_y < HeightWithAA; input_y += gridDim.y * blockDim.y) {

            size_t idx = ConvertLocToIndex(input_x, input_y, WidthWithAA);
            uint64_t tempIters = OutputIterMatrix[idx];
            MaxShared[tid] = max(MaxShared[tid], tempIters);
            MinShared[tid] = min(MinShared[tid], tempIters);
            SumShared[tid] = SumShared[tid] + tempIters;
        }
    }

    __syncthreads();
    for (auto s = blockDim.x * blockDim.y / 2; s > 0; s >>= 1) {
        if (tid < s/* && gid < WidthWithAA * HeightWithAA*/) {
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

